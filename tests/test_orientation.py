"""Street-frontage orientation on synthetic streets (offsets in metres, EPSG:7856)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from property_orientation.geometry import compass8, true_azimuth
from property_orientation.orientation import (
    MATCH_LABELS,
    UNMATCHABLE_HIGHWAYS,
    OrientationParams,
    address_sites,
    assign_confidence,
    compute_orientation,
    facing,
    match_streets,
    nearest_road_baseline,
    prepare_roads,
)
from property_orientation.streets import gnaf_street_key

SMITH_STREET = {"highway": "residential", "name": "Smith Street", "line": [(-300, 0), (300, 0)]}
"""East-west street through the origin."""
JONES_AVENUE = {"highway": "residential", "name": "Jones Avenue", "line": [(0, -300), (0, 300)]}
"""North-south street through the origin."""

SITE_COLUMNS = [
    "x", "y", "street_key", "street_base", "geocode_type", "locality", "n_addresses",
    "geometry", "orientation", "bearing_deg", "matched_street", "match", "distance_m",
    "other_street_m", "confidence", "street_x", "street_y", "baseline_bearing_deg",
    "baseline_orientation", "baseline_distance_m", "baseline_highway",
]  # fmt: skip


def address(
    at: tuple[float, float],
    street_name: str,
    street_type: str | None = None,
    street_suffix: str | None = None,
    **fields: object,
) -> dict:
    """An address row for :meth:`Scene.addresses`."""
    parts = {"street_name": street_name, "street_type": street_type, "street_suffix": street_suffix}
    return {"at": at} | parts | fields


def smith(at: tuple[float, float], **fields: object) -> dict:
    return address(at, "SMITH", "STREET", **fields)


def jones(at: tuple[float, float], **fields: object) -> dict:
    return address(at, "JONES", "AVENUE", **fields)


def orient(scene, addresses: list[dict], roads: list[dict], **kwargs) -> pd.DataFrame:
    """Sites from :func:`compute_orientation` on synthetic rows."""
    result = compute_orientation(
        scene.addresses(*addresses), scene.roads(*roads), crs=scene.crs, **kwargs
    )
    return result.sites


def only(sites: pd.DataFrame) -> pd.Series:
    assert len(sites) == 1
    return sites.iloc[0]


def values(column: pd.Series) -> list:
    """Column values as a list, with every missing value as ``None``."""
    return [None if pd.isna(v) else v for v in column]


class TestFacing:
    @pytest.mark.parametrize(
        ("line", "at", "label"),
        [
            pytest.param([(-300, 0), (300, 0)], (0, -20), "N", id="south of E-W street faces N"),
            pytest.param([(-300, 0), (300, 0)], (0, 20), "S", id="north of E-W street faces S"),
            pytest.param([(0, -300), (0, 300)], (-25, 0), "E", id="west of N-S street faces E"),
            pytest.param([(0, -300), (0, 300)], (25, 0), "W", id="east of N-S street faces W"),
            pytest.param([(-200, 200), (200, -200)], (-15, -15), "NE", id="faces NE"),
            pytest.param([(-200, 200), (200, -200)], (15, 15), "SW", id="faces SW"),
            pytest.param([(-200, -200), (200, 200)], (15, -15), "NW", id="faces NW"),
            pytest.param([(-200, -200), (200, 200)], (-15, 15), "SE", id="faces SE"),
        ],
    )
    def test_faces_the_nearest_point_of_its_street(self, scene, line, at, label) -> None:
        road = {"highway": "residential", "name": "Smith Street", "line": line}
        site = only(orient(scene, [smith(at)], [road]))

        assert site["orientation"] == label
        assert (site["street_x"], site["street_y"]) == pytest.approx(scene.xy(0, 0))
        assert site["bearing_deg"] == pytest.approx(
            true_azimuth(*scene.xy(*at), *scene.xy(0, 0), scene.crs)
        )
        assert site["distance_m"] == pytest.approx(np.hypot(*at))
        assert site["matched_street"] == "Smith Street"
        assert site["match"] == "street_name"
        assert site["confidence"] == "high"

    def test_bearing_is_true_not_grid(self, scene) -> None:
        # Grid north in Chatswood is about 1° east of true north.
        site = only(orient(scene, [smith((0, -20))], [SMITH_STREET]))
        assert site["bearing_deg"] == pytest.approx(1.01, abs=0.01)
        site = only(orient(scene, [jones((-25, 0))], [JONES_AVENUE]))
        assert site["bearing_deg"] == pytest.approx(91.01, abs=0.01)


class TestIgnoresWaysAPropertyCannotFront:
    def test_footpath_between_house_and_street(self, scene) -> None:
        footpath = {"highway": "footway", "line": [(-300, -5), (300, -5)]}
        site = only(orient(scene, [smith((0, -20))], [SMITH_STREET, footpath]))

        assert site["matched_street"] == "Smith Street"
        assert site["distance_m"] == pytest.approx(20.0)
        assert site["orientation"] == "N"
        # The original method snaps to the footpath instead.
        assert site["baseline_highway"] == "footway"
        assert site["baseline_distance_m"] == pytest.approx(15.0)

    def test_footpath_beside_the_house_turns_only_the_baseline(self, scene) -> None:
        footpath = {"highway": "footway", "name": "Smith Walk", "line": [(4, 0), (4, -60)]}
        site = only(orient(scene, [smith((0, -20))], [SMITH_STREET, footpath]))

        assert site["orientation"] == "N"
        assert site["confidence"] == "high"
        assert site["baseline_highway"] == "footway"
        assert site["baseline_orientation"] == "E"

    def test_unnamed_driveway_closer_than_the_street(self, scene) -> None:
        driveway = {"highway": "service", "service": "driveway", "line": [(3, 0), (3, -40)]}
        site = only(orient(scene, [smith((0, -20))], [SMITH_STREET, driveway]))

        assert site["matched_street"] == "Smith Street"
        assert site["orientation"] == "N"
        assert np.isnan(site["other_street_m"])  # a driveway is not a competing street
        assert site["confidence"] == "high"
        assert site["baseline_highway"] == "service"
        assert site["baseline_orientation"] == "E"
        assert site["baseline_distance_m"] == pytest.approx(3.0)

    def test_unnamed_driveway_is_never_the_fallback_street(self, scene) -> None:
        roads = [
            {"highway": "service", "service": "driveway", "line": [(3, 0), (3, -40)]},
            {"highway": "footway", "line": [(-300, -5), (300, -5)]},
            SMITH_STREET | {"line": [(-300, 300), (300, 300)]},  # beyond the search radius
        ]
        site = only(orient(scene, [address((0, -20), "NOWHERE", "STREET")], roads))

        assert site["match"] == "none"
        assert site["confidence"] == "none"
        assert pd.isna(site["orientation"])
        assert site["baseline_highway"] == "service"  # the original method faces the driveway

    @pytest.mark.parametrize("highway", sorted(UNMATCHABLE_HIGHWAYS))
    def test_motorway_or_closed_way_with_the_addressed_name(self, scene, highway: str) -> None:
        # Motorways, ways under construction, platforms and the like are never the street.
        line = [(-300, -30), (300, -30)]
        freeway = {"highway": highway, "name": "Gore Hill Freeway", "line": line}
        gore_hill = address((0, -20), "GORE HILL", "FREEWAY")
        site = only(orient(scene, [gore_hill], [SMITH_STREET, freeway]))

        assert site["match"] == "nearest_street"
        assert site["matched_street"] == "Smith Street"
        assert site["orientation"] == "N"
        assert site["confidence"] == "low"


class TestCornerLots:
    @pytest.mark.parametrize(
        ("make", "label", "distance", "other", "confidence"),
        [
            pytest.param(smith, "N", 30.0, 15.0, "medium", id="farther street"),
            pytest.param(jones, "E", 15.0, 30.0, "high", id="nearer street"),
        ],
    )
    def test_faces_the_street_it_is_addressed_to(
        self, scene, make, label, distance, other, confidence
    ) -> None:
        site = only(orient(scene, [make((-15, -30))], [SMITH_STREET, JONES_AVENUE]))

        assert site["orientation"] == label
        assert site["distance_m"] == pytest.approx(distance)
        assert site["other_street_m"] == pytest.approx(other)
        assert site["confidence"] == confidence

    def test_one_point_on_two_streets_is_two_sites(self, scene) -> None:
        sites = orient(scene, [smith((-15, -30)), jones((-15, -30))], [SMITH_STREET, JONES_AVENUE])
        assert sites["street_key"].tolist() == ["SMITH STREET", "JONES AVENUE"]
        assert sites["orientation"].tolist() == ["N", "E"]

    def test_nearer_lane_with_another_name_lowers_confidence(self, scene) -> None:
        lane = {"highway": "service", "name": "Smith Lane", "line": [(-300, -30), (300, -30)]}
        site = only(orient(scene, [smith((0, -20))], [SMITH_STREET, lane]))

        assert site["orientation"] == "N"
        assert site["other_street_m"] == pytest.approx(10.0)
        assert site["confidence"] == "medium"


class TestNameMatching:
    def test_gnaf_suffix_missing_from_osm_is_at_most_low(self, scene) -> None:
        # MARKET STREET EAST is often a separate street that OSM lacks, not Market Street.
        road = SMITH_STREET | {"name": "Market Street"}
        site = only(orient(scene, [address((0, -20), "MARKET", "STREET", "EAST")], [road]))

        assert site["street_key"] == "MARKET STREET EAST"
        assert site["match"] == "street_name_without_suffix"
        assert site["matched_street"] == "Market Street"
        assert site["orientation"] == "N"
        assert site["confidence"] == "low"

    def test_suffix_missing_from_gnaf(self, scene) -> None:
        road = SMITH_STREET | {"name": "Mowbray Road West"}
        site = only(orient(scene, [address((0, -20), "MOWBRAY", "ROAD")], [road]))

        assert site["match"] == "street_name_without_suffix"
        assert site["matched_street"] == "Mowbray Road West"
        assert site["confidence"] == "high"

    @pytest.mark.parametrize(
        ("gnaf_suffix", "osm_name"),
        [("EAST", "Market Street West"), ("UPPER", "Market Street Lower")],
    )
    def test_different_suffixes_never_match(self, scene, gnaf_suffix, osm_name) -> None:
        road = SMITH_STREET | {"name": osm_name}
        site = only(orient(scene, [address((0, -20), "MARKET", "STREET", gnaf_suffix)], [road]))

        assert site["match"] == "nearest_street"
        assert site["confidence"] == "low"

    def test_base_name_without_road_type(self, scene) -> None:
        # G-NAF has HIGHLAND RIDGE + ROAD where OpenStreetMap has "Highland Ridge".
        road = SMITH_STREET | {"name": "Highland Ridge"}
        site = only(orient(scene, [address((0, -20), "HIGHLAND RIDGE", "ROAD")], [road]))

        assert site["street_key"] == "HIGHLAND RIDGE ROAD"
        assert site["street_base"] == "HIGHLAND RIDGE"
        assert site["match"] == "street_base_name"
        assert site["matched_street"] == "Highland Ridge"
        assert site["orientation"] == "N"
        assert site["confidence"] == "high"

    @pytest.mark.parametrize(
        ("parts", "near", "far", "match"),
        [
            (("MOWBRAY", "ROAD", "WEST"), "Mowbray Road", "Mowbray Road West", "street_name"),
            (("HIGHLAND RIDGE", "ROAD"), "Highland Ridge", "Highland Ridge Road", "street_name"),
            (("MARKET", "STREET", "EAST"), "Market", "Market Street", "street_name_without_suffix"),
        ],
        ids=["exact over suffix", "exact over base", "suffix over base"],
    )
    def test_better_tier_wins_over_a_nearer_way(self, scene, parts, near, far, match) -> None:
        roads = [
            {"highway": "residential", "name": near, "line": [(-300, 0), (300, 0)]},
            {"highway": "residential", "name": far, "line": [(-300, -50), (300, -50)]},
        ]
        site = only(orient(scene, [address((0, -20), *parts)], roads))

        assert site["matched_street"] == far
        assert site["match"] == match
        assert site["orientation"] == "S"
        assert site["distance_m"] == pytest.approx(30.0)

    @pytest.mark.parametrize("tag", ["alt_name", "official_name", "old_name"])
    def test_alternative_osm_names(self, scene, tag: str) -> None:
        road = SMITH_STREET | {"name": "Railway Parade", tag: "Rail Pde;Station St"}
        site = only(orient(scene, [address((0, -20), "STATION", "STREET")], [road]))

        assert site["match"] == "street_name"
        assert site["matched_street"] == "Railway Parade"
        assert site["confidence"] == "high"

    def test_street_that_exists_only_as_a_walkway(self, scene) -> None:
        roads = [
            {"highway": "footway", "name": "The Postern", "line": [(-300, -35), (300, -35)]},
            {"highway": "residential", "name": "Edinburgh Road", "line": [(-300, 20), (300, 20)]},
        ]
        site = only(orient(scene, [address((0, -20), "THE POSTERN")], roads))

        assert site["matched_street"] == "The Postern"
        assert site["match"] == "street_name"
        assert site["orientation"] == "S"

    def test_vehicular_way_preferred_over_a_same_name_footway(self, scene) -> None:
        side_path = {"highway": "footway", "name": "Smith Street", "line": [(5, 0), (5, -60)]}
        site = only(orient(scene, [smith((0, -20))], [side_path, SMITH_STREET]))

        assert site["distance_m"] == pytest.approx(20.0)
        assert site["orientation"] == "N"


class TestFallbacks:
    def test_no_name_match_uses_the_nearest_street(self, scene) -> None:
        roads = [
            SMITH_STREET,
            {"highway": "footway", "line": [(-300, -17), (300, -17)]},
            {"highway": "service", "line": [(4, 0), (4, -40)]},
            JONES_AVENUE | {"line": [(60, -300), (60, 300)]},
        ]
        site = only(orient(scene, [address((0, -20), "NOWHERE", "STREET")], roads))

        assert site["match"] == "nearest_street"
        assert site["matched_street"] == "Smith Street"
        assert site["distance_m"] == pytest.approx(20.0)
        assert site["orientation"] == "N"
        assert site["confidence"] == "low"

    def test_address_without_a_street_name_uses_the_nearest_street(self, scene) -> None:
        site = only(orient(scene, [{"at": (0, -20)}], [SMITH_STREET]))
        assert pd.isna(site["street_key"])
        assert site["match"] == "nearest_street"
        assert site["confidence"] == "low"

    def test_nothing_within_the_search_radius(self, scene) -> None:
        site = only(orient(scene, [smith((0, -400))], [SMITH_STREET]))

        assert site["match"] == "none"
        assert site["confidence"] == "none"
        assert pd.isna(site["orientation"])
        assert pd.isna(site["matched_street"])
        assert np.isnan([site["bearing_deg"], site["distance_m"], site["street_x"]]).all()
        # The baseline searches 500 m, so it still finds the street.
        assert site["baseline_orientation"] == "N"

    def test_search_radius_is_a_parameter(self, scene) -> None:
        params = OrientationParams(search_radius_m=500.0)
        site = only(orient(scene, [smith((0, -400))], [SMITH_STREET], params=params))

        assert site["match"] == "street_name"
        assert site["orientation"] == "N"
        assert site["confidence"] == "low"

    def test_baseline_radius_is_a_parameter(self, scene) -> None:
        params = OrientationParams(baseline_radius_m=10.0)
        site = only(orient(scene, [smith((0, -20))], [SMITH_STREET], params=params))

        assert site["orientation"] == "N"  # the street search radius is separate
        assert pd.isna(site["baseline_orientation"])
        assert np.isnan(site["baseline_distance_m"])

    def test_no_roads_at_all(self, scene) -> None:
        no_roads = scene.roads(SMITH_STREET).iloc[:0]
        result = compute_orientation(scene.addresses(smith((0, -20))), no_roads, crs=scene.crs)
        site = only(result.sites)

        assert site["match"] == "none"
        assert site["confidence"] == "none"
        assert pd.isna(site["orientation"])
        assert pd.isna(site["baseline_orientation"])

    @pytest.mark.parametrize("geocode", ["STREET LOCALITY", "LOCALITY"])
    def test_street_or_suburb_geocode_has_no_orientation(self, scene, geocode: str) -> None:
        site = only(orient(scene, [smith((0, -20), geocode_type=geocode)], [SMITH_STREET]))

        assert site["confidence"] == "none"
        assert pd.isna(site["orientation"])
        assert np.isnan(site["bearing_deg"])
        assert site["match"] == "street_name"  # the street is still identified


class TestConfidence:
    @pytest.mark.parametrize(
        ("distance", "confidence"),
        [
            (0.3, "none"),  # too close for a meaningful bearing
            (0.5, "high"),
            (30.0, "high"),
            (60.0, "high"),
            (60.5, "medium"),
            (120.0, "medium"),
            (120.5, "low"),
            (249.0, "low"),
        ],
    )
    def test_falls_with_distance_to_the_street(self, scene, distance, confidence) -> None:
        site = only(orient(scene, [smith((0, -distance))], [SMITH_STREET]))
        assert site["confidence"] == confidence
        assert pd.isna(site["orientation"]) == (confidence == "none")

    def test_building_centroid_is_at_most_medium(self, scene) -> None:
        building = smith((0, -20), geocode_type="BUILDING CENTROID")
        site = only(orient(scene, [building], [SMITH_STREET]))
        assert site["confidence"] == "medium"
        assert site["orientation"] == "N"

    def test_rules(self) -> None:
        cases = [
            # tier, distance, other street, geocode, expected
            (1, 10.0, np.nan, "PROPERTY CENTROID", "high"),
            (2, 10.0, 50.0, "PROPERTY CENTROID", "high"),
            (3, 10.0, np.nan, "GAP GEOCODE", "high"),
            (0, 10.0, np.nan, "PROPERTY CENTROID", "low"),
            (-1, np.nan, np.nan, "PROPERTY CENTROID", "none"),
            (1, 0.2, np.nan, "PROPERTY CENTROID", "none"),
            (1, 10.0, np.nan, "STREET LOCALITY", "none"),
            (1, 61.0, np.nan, "PROPERTY CENTROID", "medium"),
            (1, 121.0, np.nan, "PROPERTY CENTROID", "low"),
            (1, 10.0, 5.0, "PROPERTY CENTROID", "medium"),
            (1, 10.0, np.nan, "BUILDING CENTROID", "medium"),
            (1, 200.0, 5.0, "BUILDING CENTROID", "low"),
        ]
        tier, distance, other, geocode, expected = map(list, zip(*cases, strict=True))
        result = assign_confidence(
            np.array(tier),
            np.array(distance),
            np.array(other),
            np.array(geocode, dtype=object),
            OrientationParams(),
        )
        assert result.tolist() == expected

    def test_thresholds_are_parameters(self) -> None:
        params = OrientationParams(high_confidence_max_m=5.0, medium_confidence_max_m=8.0)
        geocode = np.array(["PROPERTY CENTROID"] * 3, dtype=object)
        distance = np.array([4.0, 6.0, 9.0])
        result = assign_confidence(np.ones(3), distance, np.full(3, np.nan), geocode, params)
        assert result.tolist() == ["high", "medium", "low"]


class TestSites:
    def test_units_at_one_point_on_one_street_are_one_site(self, scene) -> None:
        addresses = scene.addresses(
            smith((0, -20), primary_secondary="P"),
            *[smith((0, -20), primary_secondary="S") for _ in range(3)],
            jones((0, -20)),
            smith((0, -22)),
            smith((0.003, -20)),  # within the 1 cm grouping tolerance
        )
        sites, site_of_address = address_sites(addresses)

        assert site_of_address.tolist() == [0, 0, 0, 0, 1, 2, 0]
        assert sites["n_addresses"].tolist() == [5, 1, 1]
        assert sites["street_key"].tolist() == ["SMITH STREET", "JONES AVENUE", "SMITH STREET"]
        assert sites["street_base"].tolist() == ["SMITH", "JONES", "SMITH"]
        assert sites.index.tolist() == [0, 1, 2]
        assert sites.crs == addresses.crs

    @pytest.mark.parametrize("building_first", [True, False])
    def test_geocode_types_at_one_point_are_separate_sites(self, scene, building_first) -> None:
        # The confidence rule depends on the geocode type, so it must not depend on row order.
        rows = [
            smith((0, -20), geocode_type="PROPERTY CENTROID"),
            smith((0, -20), geocode_type="BUILDING CENTROID"),
        ]
        sites = orient(scene, rows[::-1] if building_first else rows, [SMITH_STREET])

        confidence = dict(zip(sites["geocode_type"], sites["confidence"], strict=True))
        assert confidence == {"PROPERTY CENTROID": "high", "BUILDING CENTROID": "medium"}
        bearing = sites["bearing_deg"]
        assert (bearing == bearing.iloc[0]).all()  # same point and street, so the same bearing

    def test_needs_a_projected_crs(self, scene) -> None:
        addresses = scene.addresses(smith((0, -20)))
        with pytest.raises(ValueError, match="projected"):
            address_sites(addresses.to_crs(4326))
        with pytest.raises(ValueError, match="projected"):
            address_sites(addresses.set_crs(None, allow_override=True))

    def test_compute_orientation_aligns_sites_with_addresses(self, scene) -> None:
        addresses = scene.addresses(
            smith((40, -20), primary_secondary="S"),
            jones((-15, -30)),
            smith((40, -20), primary_secondary="S"),
            address((25, 60), "NOWHERE", "LANE"),
            smith((-15, -30)),
            smith((-80, 30), geocode_type="STREET LOCALITY"),
            smith((40, -20), primary_secondary="P"),
            jones((-30, -900)),
        )
        roads = scene.roads(SMITH_STREET, JONES_AVENUE)
        # Inputs may come in any CRS; distances are computed in ``crs``.
        result = compute_orientation(addresses.to_crs(4326), roads.to_crs(4326), crs=scene.crs)
        sites, index = result.sites, result.site_of_address

        assert result.crs == scene.crs
        assert sites.crs.equals(addresses.crs)
        assert list(sites.columns) == SITE_COLUMNS
        assert sites.index.name == "site_id"
        assert sites.index.tolist() == list(range(len(sites)))
        assert len(sites) == 6
        assert len(index) == len(addresses)
        assert sorted(set(index)) == list(range(len(sites)))
        assert sites["n_addresses"].sum() == len(addresses)

        per_address = sites.iloc[index]
        parts = addresses[["street_name", "street_type", "street_suffix"]].itertuples(index=False)
        assert per_address["street_key"].tolist() == [gnaf_street_key(*p) for p in parts]
        np.testing.assert_allclose(per_address["x"], addresses.geometry.x, atol=0.01)
        np.testing.assert_allclose(per_address["y"], addresses.geometry.y, atol=0.01)
        assert values(per_address["orientation"]) == ["N", "E", "N", "W", "N", None, "N", None]
        assert set(sites["match"]) <= set(MATCH_LABELS.values())

        # The label is always the 8-point sector of the bearing.
        assert values(sites["orientation"]) == compass8(sites["bearing_deg"]).tolist()

        projected = compute_orientation(addresses, roads, crs=scene.crs)
        np.testing.assert_array_equal(projected.site_of_address, index)
        np.testing.assert_allclose(projected.sites["bearing_deg"], sites["bearing_deg"], atol=1e-6)

    def test_baseline_columns_are_optional(self, scene) -> None:
        sites = orient(scene, [smith((0, -20))], [SMITH_STREET], baseline=False)
        assert list(sites.columns) == SITE_COLUMNS[:-4]


class TestBuildingBlocks:
    def test_prepare_roads(self, scene) -> None:
        line = [(0, 0), (10, 0)]
        roads = scene.roads(
            {"highway": "residential", "name": "Smith St", "line": line},
            {"highway": "service", "service": "driveway", "line": line},
            {"highway": "service", "name": "Smith Lane", "service": "alley", "line": line},
            {"highway": "footway", "name": "Smith Walk", "line": line},
            {"highway": "pedestrian", "name": "The Concourse", "area": "yes", "line": line},
            {"highway": "pedestrian", "name": "Victoria Mall", "line": line},
            {"highway": "motorway", "name": "Gore Hill Fwy", "line": line},
            {"highway": "living_street", "name": "Mowbray Rd W", "line": line},
            {
                "highway": "residential",
                "name": "Railway Pde",
                "alt_name": "Rail St;Station St",
                "old_name": "Station Rd",
                "line": line,
            },
        )
        out = prepare_roads(roads.set_index(pd.Index(range(100, 109))))

        assert out.index.tolist() == list(range(9))
        assert values(out["street_key"])[:3] == ["SMITH STREET", None, "SMITH LANE"]
        assert out.loc[7, "street_key"] == "MOWBRAY ROAD WEST"
        assert out.loc[7, "street_key_nosuffix"] == "MOWBRAY ROAD"
        assert out.loc[8, "alt_keys"] == {"RAIL STREET", "STATION STREET", "STATION ROAD"}
        assert out.loc[0, "alt_keys"] == frozenset()
        frontage = [True, False, True, False, False, True, False, True, True]
        assert out["frontage"].tolist() == frontage

    def test_prepare_roads_adds_missing_tag_columns(self, scene) -> None:
        roads = scene.roads(SMITH_STREET)[["osm_id", "highway", "name", "geometry"]]
        out = prepare_roads(roads)

        assert {"alt_name", "official_name", "old_name", "service", "area"} <= set(out.columns)
        assert out["frontage"].tolist() == [True]
        assert "street_key" not in roads.columns  # the input is not modified

    def test_match_streets(self, scene) -> None:
        roads = prepare_roads(
            scene.roads(
                {"highway": "footway", "line": [(-300, -5), (300, -5)]},
                SMITH_STREET,
                JONES_AVENUE | {"line": [(10, -300), (10, 300)]},
            ).set_index(pd.Index([50, 60, 70]))
        )
        sites, _ = address_sites(
            scene.addresses(
                smith((0, -20)),
                address((0, -20), "NOWHERE", "STREET"),
                smith((0, -900)),
            )
        )
        sites.index = pd.Index([5, 6, 7], name="site_id")
        out = match_streets(sites, roads, radius_m=250.0)

        assert out.index.equals(sites.index)
        # Roads are numbered by position in ``roads``; -1 means none within the radius.
        assert out["road"].tolist() == [1, 2, -1]
        assert out["tier"].tolist() == [1, 0, -1]  # named match, fallback, nothing
        np.testing.assert_allclose(out["distance_m"], [20.0, 10.0, np.nan])
        # The nearest street with another name; the footway is not a street.
        np.testing.assert_allclose(out["other_street_m"], [10.0, 10.0, np.nan])

    def test_nearest_road_baseline(self, scene) -> None:
        steps = {"highway": "steps", "line": [(-6, -60), (-6, 60)]}
        roads = prepare_roads(scene.roads(SMITH_STREET, steps))
        sites, _ = address_sites(scene.addresses(smith((0, -20)), smith((0, -700))))
        out = nearest_road_baseline(sites, roads, radius_m=500.0)

        assert out.index.equals(sites.index)
        assert out.loc[0, "baseline_highway"] == "steps"
        assert out.loc[0, "baseline_orientation"] == "W"
        assert out.loc[0, "baseline_distance_m"] == pytest.approx(6.0)
        assert pd.isna(out.loc[1, "baseline_highway"])
        assert pd.isna(out.loc[1, "baseline_orientation"])
        assert out.loc[1, ["baseline_bearing_deg", "baseline_distance_m"]].isna().all()

    def test_facing(self, scene) -> None:
        points = np.array([scene.point(0, -20), scene.point(5, 5)], dtype=object)
        lines = np.array([scene.line([(-50, 0), (50, 0)]), None], dtype=object)
        azimuth, nx, ny = facing(points, lines, scene.crs)

        assert compass8(azimuth).tolist() == ["N", None]
        assert (nx[0], ny[0]) == pytest.approx(scene.xy(0, 0))
        assert np.isnan([nx[1], ny[1]]).all()
