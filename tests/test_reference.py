"""The cadastral reference: parcel frontage, outward normals and the street each edge faces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely import affinity

from property_orientation.geometry import angular_difference, compass8, true_azimuth
from property_orientation.orientation import address_sites
from property_orientation.reference import (
    ReferenceParams,
    frontage_segments,
    name_segments,
    parcel_of_points,
    reference_orientation,
    unique_parcels,
)

REF_COLUMNS = ["ref_bearing_deg", "ref_frontage_m", "ref_strength", "ref_parcel", "ref_status"]


def segments_of(lot: shapely.Geometry, reserves: list[shapely.Geometry], **kwargs) -> pd.DataFrame:
    return frontage_segments(np.array([lot]), np.array(reserves), **kwargs)


def check_outward(segments: pd.DataFrame, lot: shapely.Geometry) -> None:
    """Each normal leaves the parcel: a step along it is outside, a step against it inside."""
    mx, my, nx, ny = (segments[c].to_numpy() for c in ("mx", "my", "nx", "ny"))
    assert not shapely.contains(lot, shapely.points(mx + 0.25 * nx, my + 0.25 * ny)).any()
    assert shapely.contains(lot, shapely.points(mx - 0.25 * nx, my - 0.25 * ny)).all()
    np.testing.assert_allclose(np.hypot(nx, ny), 1.0)


class TestFrontageSegments:
    def test_edge_along_the_reserve_with_a_normal_into_the_road(self, scene) -> None:
        reserve = scene.box(-100, 0, 100, 20)
        lot = scene.box(0, -40, 15, 0)
        segments = segments_of(lot, [reserve], tolerance_m=0.5)

        assert list(segments.columns) == ["parcel", "length", "mx", "my", "nx", "ny"]
        assert (segments["parcel"] == 0).all()
        frontage = segments[segments["length"] > 1.0]
        assert len(frontage) == 1
        edge = frontage.iloc[0]
        assert edge["length"] == pytest.approx(15.0)
        assert (edge["mx"], edge["my"]) == pytest.approx(scene.xy(7.5, 0))
        assert (edge["nx"], edge["ny"]) == pytest.approx((0.0, 1.0))
        # Besides the frontage edge, only the 0.5 m of each side edge inside the tolerance
        # band appears; the rear edge and the rest of the sides do not.
        assert segments["length"].sum() == pytest.approx(15.0 + 2 * 0.5)
        mids = shapely.points(segments["mx"], segments["my"])
        assert (shapely.distance(mids, reserve) <= 0.5 + 1e-9).all()
        check_outward(segments, lot)

    @pytest.mark.parametrize(
        ("reserve", "lot", "normal"),
        [
            pytest.param((-100, 0, 100, 20), (0, -40, 15, 0), (0, 1), id="south of the road"),
            pytest.param((-100, 0, 100, 20), (0, 20, 15, 60), (0, -1), id="north of the road"),
            pytest.param((0, -100, 20, 100), (-40, 0, 0, 15), (1, 0), id="west of the road"),
            pytest.param((0, -100, 20, 100), (20, 0, 60, 15), (-1, 0), id="east of the road"),
        ],
    )
    @pytest.mark.parametrize("clockwise", [False, True], ids=["ccw ring", "cw ring"])
    def test_normal_points_into_the_road(self, scene, reserve, lot, normal, clockwise) -> None:
        polygon = scene.box(*lot)
        if clockwise:
            polygon = shapely.Polygon(polygon.exterior.coords[::-1])
            assert not polygon.exterior.is_ccw
        segments = segments_of(polygon, [scene.box(*reserve)])
        edge = segments.loc[segments["length"].idxmax()]

        assert edge["length"] == pytest.approx(15.0)
        assert (edge["nx"], edge["ny"]) == pytest.approx(normal)
        check_outward(segments, polygon)

    @pytest.mark.parametrize("angle", [30.0, 135.0, 250.0])
    def test_normal_of_a_rotated_lot(self, scene, angle: float) -> None:
        origin = scene.point(0, 0)
        reserve = affinity.rotate(scene.box(-100, 0, 100, 20), angle, origin=origin)
        lot = affinity.rotate(scene.box(0, -40, 15, 0), angle, origin=origin)
        segments = segments_of(lot, [reserve])
        edge = segments.loc[segments["length"].idxmax()]

        theta = np.radians(angle)  # rotating north (0, 1) anticlockwise by theta
        assert (edge["nx"], edge["ny"]) == pytest.approx((-np.sin(theta), np.cos(theta)))
        assert edge["length"] == pytest.approx(15.0)
        check_outward(segments, lot)

    def test_lot_set_back_from_the_reserve(self, scene) -> None:
        reserve, lot = scene.box(-100, 0, 100, 20), scene.box(0, -40, 15, -1)
        assert segments_of(lot, [reserve]).empty
        wider = segments_of(lot, [reserve], tolerance_m=1.5)
        edge = wider.loc[wider["length"].idxmax()]
        assert edge["length"] == pytest.approx(15.0)
        assert (edge["nx"], edge["ny"]) == pytest.approx((0.0, 1.0))

    def test_parcel_column_indexes_the_input(self, block) -> None:
        segments = frontage_segments(
            block.parcels.geometry.to_numpy(), block.reserves.geometry.to_numpy()
        )
        with_frontage = set(block.parcels["lot"].iloc[np.unique(segments["parcel"])])
        assert with_frontage == {"corner", "inner", "front", "panhandle", "north", "east"}

    def test_no_parcels(self, scene) -> None:
        segments = frontage_segments(np.array([], dtype=object), np.array([scene.box(0, 0, 5, 5)]))
        assert segments.empty


class TestNameSegments:
    def test_corner_edges_take_the_street_they_face(self, block) -> None:
        corner = block.parcels.geometry.iloc[block.parcel_index("corner")]
        segments = segments_of(corner, list(block.reserves.geometry))
        keys = name_segments(
            segments,
            block.centrelines.geometry.to_numpy(),
            np.array(["SMITH STREET", "JONES AVENUE"], dtype=object),
        )
        named = segments.assign(street=keys)

        long_edges = {
            (round(row.nx), round(row.ny)): row.street
            for row in named.itertuples()
            if row.length > 1.0
        }
        assert long_edges == {(0, 1): "SMITH STREET", (1, 0): "JONES AVENUE"}
        # The stubs facing west and south cross no centreline within the ray length.
        assert named.loc[(named["nx"] < -0.5) | (named["ny"] < -0.5), "street"].isna().all()

    @pytest.fixture
    def edge(self, scene) -> pd.DataFrame:
        """One frontage segment at the origin, facing north."""
        return pd.DataFrame(
            {"parcel": [0], "length": [15.0], "mx": [scene.x0], "my": [scene.y0],
             "nx": [0.0], "ny": [1.0]}
        )  # fmt: skip

    def test_first_centreline_crossed(self, scene, edge) -> None:
        near, far = scene.line([(-50, 10), (50, 10)]), scene.line([(-50, 30), (50, 30)])
        keys = name_segments(edge, np.array([far, near]), np.array(["FAR ROAD", "NEAR ROAD"]))
        assert keys.tolist() == ["NEAR ROAD"]

    def test_unnamed_centreline_crossed_first(self, scene, edge) -> None:
        near, far = scene.line([(-50, 10), (50, 10)]), scene.line([(-50, 30), (50, 30)])
        keys = name_segments(edge, np.array([near, far]), np.array([None, "FAR ROAD"]))
        assert keys.tolist() == [None]

    def test_ray_length(self, scene, edge) -> None:
        road = np.array([scene.line([(-50, 50), (50, 50)])])
        assert name_segments(edge, road, np.array(["FAR ROAD"])).tolist() == [None]
        assert name_segments(edge, road, np.array(["FAR ROAD"]), ray_length_m=60.0).tolist() == [
            "FAR ROAD"
        ]

    def test_centreline_behind_the_edge_is_ignored(self, scene, edge) -> None:
        behind = np.array([scene.line([(-50, -10), (50, -10)])])
        assert name_segments(edge, behind, np.array(["BACK LANE"])).tolist() == [None]


def test_unique_parcels_drops_stacked_duplicates(scene) -> None:
    a, b = scene.box(0, 0, 10, 10), scene.box(10, 0, 20, 10)
    parcels = scene.frame([a, a, b, a], propid=[1, 2, 3, 4]).set_index(pd.Index([7, 8, 9, 10]))
    out = unique_parcels(parcels)
    assert out["propid"].tolist() == [1, 3]
    assert out.index.tolist() == [0, 1]


def test_parcel_of_points_picks_the_smallest_containing_parcel(scene) -> None:
    parcels = np.array(
        [scene.box(0, 0, 100, 100), scene.box(10, 10, 30, 30), scene.box(200, 0, 220, 20)]
    )
    points = np.array(
        [scene.point(20, 20), scene.point(50, 50), scene.point(500, 500), scene.point(210, 10)]
    )
    assert parcel_of_points(points, parcels).tolist() == [1, 0, -1, 2]


def block_sites(block, *rows: dict):
    sites, _ = address_sites(block.scene.addresses(*rows))
    return sites


def expected_bearing(block, lot: str, normal: tuple[float, float]) -> float:
    """True bearing of a grid direction at a lot."""
    x, y = block.scene.xy(*block.lots[lot])
    return float(true_azimuth(x, y, x + normal[0], y + normal[1], block.scene.crs))


class TestReferenceOrientation:
    @pytest.mark.parametrize(
        ("lot", "street", "normal", "label", "frontage"),
        [
            ("inner", ("SMITH", "STREET"), (0, 1), "N", 16.0),
            ("front", ("SMITH", "STREET"), (0, 1), "N", 14.0),
            ("north", ("SMITH", "STREET"), (0, -1), "S", 15.0),
            ("east", ("JONES", "AVENUE"), (-1, 0), "W", 40.0),
            ("corner", ("SMITH", "STREET"), (0, 1), "N", 15.0),
            ("corner", ("JONES", "AVENUE"), (1, 0), "E", 40.0),
        ],
    )
    def test_lot_faces_its_street(self, block, lot, street, normal, label, frontage) -> None:
        sites = block_sites(block, block.address(lot, *street))
        ref = reference_orientation(sites, block.parcels, block.reserves, block.centrelines)

        assert list(ref.columns) == REF_COLUMNS
        row = ref.iloc[0]
        assert row["ref_status"] == "ok"
        assert row["ref_parcel"] == block.parcel_index(lot)
        expected = expected_bearing(block, lot, normal)
        assert angular_difference(row["ref_bearing_deg"], expected) < 1e-3
        assert compass8(row["ref_bearing_deg"]).tolist() == [label]
        assert row["ref_strength"] == pytest.approx(1.0)
        # At the corner, the 0.5 m where both reserves' tolerance bands overlap counts twice.
        assert frontage <= row["ref_frontage_m"] <= frontage + 0.5 + 1e-9

    def test_statuses(self, block) -> None:
        rows = [
            block.address("inner", "SMITH", "STREET"),
            block.address("panhandle", "SMITH", "STREET"),  # 2 m access strip
            block.address("rear", "SMITH", "STREET"),  # no frontage at all
            block.address("north", "JONES", "AVENUE"),  # frontage on another street only
            {"at": (-100, 5), "street_name": "SMITH", "street_type": "STREET"},  # in the road
        ]
        sites = block_sites(block, *rows)
        ref = reference_orientation(sites, block.parcels, block.reserves, block.centrelines)

        assert ref["ref_status"].tolist() == [
            "ok", "short_frontage", "no_street_frontage", "no_street_frontage", "no_parcel"
        ]  # fmt: skip
        assert ref["ref_parcel"].tolist() == [
            block.parcel_index("inner"), block.parcel_index("panhandle"),
            block.parcel_index("rear"), block.parcel_index("north"), -1,
        ]  # fmt: skip
        assert ref["ref_bearing_deg"].notna().tolist() == [True, False, False, False, False]
        assert ref.loc[1, "ref_frontage_m"] == pytest.approx(2.0)
        assert ref.loc[[2, 3, 4], "ref_frontage_m"].isna().all()

    def test_minimum_frontage_is_a_parameter(self, block) -> None:
        sites = block_sites(block, block.address("panhandle", "SMITH", "STREET"))
        params = ReferenceParams(min_frontage_m=1.5)
        ref = reference_orientation(
            sites, block.parcels, block.reserves, block.centrelines, params=params
        )
        assert ref.loc[0, "ref_status"] == "ok"
        assert compass8(ref.loc[0, "ref_bearing_deg"]).tolist() == ["N"]

    @pytest.mark.parametrize(
        ("width", "depth", "label"),
        [
            pytest.param(30, 30, "NE", id="equal frontages"),
            pytest.param(10, 40, "E", id="longer frontage to the east"),
        ],
    )
    def test_bearing_is_the_length_weighted_mean_normal(self, scene, width, depth, label) -> None:
        # Smith Street wraps around the lot: its north edge (``width`` m) faces north onto
        # one arm and its east edge (``depth`` m) faces east onto the other. An unweighted
        # mean of the two normals would point NE whatever the lengths.
        lot = scene.box(0, -depth, width, 0)
        north, east = scene.box(-100, 0, 100, 20), scene.box(width, -100, width + 20, 0)
        arms = [
            scene.line([(-100, 10), (100, 10)]),
            scene.line([(width + 10, -100), (width + 10, 10)]),
        ]
        centrelines = scene.frame(arms, roadnamelabel=["SMITH ST", "SMITH ST"])
        centre = (width / 2, -depth / 2)
        sites, _ = address_sites(
            scene.addresses({"at": centre, "street_name": "SMITH", "street_type": "STREET"})
        )
        ref = reference_orientation(
            sites, scene.frame([lot]), scene.frame([north, east]), centrelines
        )
        row = ref.iloc[0]

        x, y = scene.xy(*centre)
        expected = true_azimuth(x, y, x + depth, y + width, scene.crs)  # sum of length x normal
        assert row["ref_status"] == "ok"
        assert angular_difference(row["ref_bearing_deg"], expected) < 1.0
        assert compass8(row["ref_bearing_deg"]).tolist() == [label]
        strength = np.hypot(width, depth) / (width + depth)
        assert row["ref_strength"] == pytest.approx(strength, abs=0.01)
        # Plus the 0.5 m stubs where each edge enters the other arm's tolerance band.
        assert width + depth <= row["ref_frontage_m"] <= width + depth + 1.0 + 1e-9

    def test_minimum_strength_is_a_parameter(self, scene) -> None:
        reserves = scene.frame([scene.box(-100, 0, 100, 20), scene.box(30, -100, 50, 0)])
        centrelines = scene.frame(
            [scene.line([(-100, 10), (100, 10)]), scene.line([(40, -100), (40, 10)])],
            roadnamelabel=["SMITH ST", "SMITH ST"],
        )
        sites, _ = address_sites(
            scene.addresses({"at": (15, -15), "street_name": "SMITH", "street_type": "STREET"})
        )
        lots = scene.frame([scene.box(0, -30, 30, 0)])
        strict = ReferenceParams(min_strength=0.75)
        ref = reference_orientation(sites, lots, reserves, centrelines, params=strict)

        assert ref.loc[0, "ref_status"] == "ambiguous_frontage"
        assert ref.loc[0, "ref_strength"] == pytest.approx(np.sqrt(0.5), abs=1e-6)
        assert np.isnan(ref.loc[0, "ref_bearing_deg"])

    def test_frontage_on_opposite_sides_is_ambiguous(self, scene) -> None:
        # A through lot between two carriageways of the same street: the normals cancel.
        reserves = scene.frame([scene.box(-100, 0, 100, 20), scene.box(-100, -60, 100, -40)])
        centrelines = scene.frame(
            [scene.line([(-100, 10), (100, 10)]), scene.line([(-100, -50), (100, -50)])],
            roadnamelabel=["SMITH ST", "SMITH ST"],
        )
        parcels = scene.frame([scene.box(0, -40, 15, 0)])
        sites, _ = address_sites(
            scene.addresses({"at": (7, -20), "street_name": "SMITH", "street_type": "STREET"})
        )
        ref = reference_orientation(sites, parcels, reserves, centrelines)

        assert ref.loc[0, "ref_status"] == "ambiguous_frontage"
        assert ref.loc[0, "ref_frontage_m"] == pytest.approx(30.0)
        assert ref.loc[0, "ref_strength"] == pytest.approx(0.0, abs=1e-9)
        assert np.isnan(ref.loc[0, "ref_bearing_deg"])

    def test_street_suffix_is_optional(self, block) -> None:
        sites = block_sites(block, block.address("inner", "SMITH", "STREET", street_suffix="WEST"))
        assert sites.loc[0, "street_key"] == "SMITH STREET WEST"
        ref = reference_orientation(sites, block.parcels, block.reserves, block.centrelines)
        assert ref.loc[0, "ref_status"] == "ok"

    def test_layers_in_any_crs_and_a_custom_name_column(self, block) -> None:
        sites = block_sites(
            block,
            block.address("corner", "JONES", "AVENUE"),
            block.address("north", "SMITH", "STREET"),
        )
        expected = reference_orientation(sites, block.parcels, block.reserves, block.centrelines)
        centrelines = block.centrelines.rename(columns={"roadnamelabel": "name"}).to_crs(4326)
        ref = reference_orientation(
            sites,
            block.parcels.to_crs(4326),
            block.reserves.to_crs(4326),
            centrelines,
            name_column="name",
        )
        assert ref["ref_status"].tolist() == expected["ref_status"].tolist() == ["ok", "ok"]
        np.testing.assert_allclose(ref["ref_bearing_deg"], expected["ref_bearing_deg"], atol=1e-6)
        np.testing.assert_allclose(ref["ref_frontage_m"], expected["ref_frontage_m"], atol=1e-3)

    def test_ref_parcel_indexes_the_deduplicated_parcels(self, block) -> None:
        # A stacked strata copy of the first lot shifts every later row of the input.
        stacked = pd.concat([block.parcels.iloc[[0]], block.parcels], ignore_index=True)
        sites = block_sites(block, block.address("east", "JONES", "AVENUE"))
        ref = reference_orientation(sites, stacked, block.reserves, block.centrelines)

        east = block.parcels.geometry.iloc[block.parcel_index("east")]
        k = ref.loc[0, "ref_parcel"]
        assert ref.loc[0, "ref_status"] == "ok"
        assert unique_parcels(stacked).geometry.iloc[k].equals(east)
        assert not stacked.geometry.iloc[k].equals(east)

    def test_aligned_with_the_sites_index(self, block) -> None:
        sites = block_sites(
            block,
            block.address("rear", "SMITH", "STREET"),
            block.address("east", "JONES", "AVENUE"),
        ).set_index(pd.Index([42, 7], name="site_id"))
        ref = reference_orientation(sites, block.parcels, block.reserves, block.centrelines)
        assert ref.index.tolist() == [42, 7]
        assert ref.loc[42, "ref_status"] == "no_street_frontage"
        assert ref.loc[7, "ref_status"] == "ok"
