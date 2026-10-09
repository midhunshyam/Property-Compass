"""Result tables: one row per address, facing lines, direction shares and suburb summaries."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
import shapely

from property_orientation.geometry import COMPASS_8
from property_orientation.orientation import compute_orientation
from property_orientation.outputs import (
    ADDRESS_TABLE_COLUMNS,
    address_table,
    facing_lines,
    orientation_shares,
    suburb_summary,
    write_geopackage,
)

SMITH_STREET = {"highway": "residential", "name": "Smith Street", "line": [(-300, 0), (300, 0)]}
JONES_AVENUE = {"highway": "residential", "name": "Jones Avenue", "line": [(0, -300), (0, 300)]}
COUNT_COLUMNS = [f"n_{d}" for d in COMPASS_8]
SHARE_COLUMNS = [f"share_{d}" for d in COMPASS_8]


def smith(at: tuple[float, float], **fields: object) -> dict:
    return {"at": at, "street_name": "SMITH", "street_type": "STREET"} | fields


@pytest.fixture
def oriented(scene):
    """Addresses (EPSG:4326, as G-NAF is stored) and their orientation result."""
    addresses = scene.addresses(
        smith((30, -20), address="UNIT 1, 5 SMITH STREET", primary_secondary="S"),
        smith((30, -20), address="5 SMITH STREET", primary_secondary="P"),
        smith((30, -20), address="UNIT 2, 5 SMITH STREET", primary_secondary="S"),
        {"at": (-25, -40), "street_name": "JONES", "street_type": "AVENUE",
         "address": "9 JONES AVENUE", "locality_name": "WILLOUGHBY", "postcode": "2068"},
        smith((-60, 25), address="12 SMITH STREET", geocode_type="STREET LOCALITY"),
    ).to_crs(4326)  # fmt: skip
    result = compute_orientation(addresses, scene.roads(SMITH_STREET, JONES_AVENUE), crs=scene.crs)
    return addresses, result


class TestAddressTable:
    def test_columns_order_and_rows(self, oriented) -> None:
        addresses, result = oriented
        table = address_table(addresses, result.sites, result.site_of_address)

        assert list(table.columns) == list(ADDRESS_TABLE_COLUMNS)
        assert table["gnaf_pid"].tolist() == addresses["gnaf_pid"].tolist()
        assert table["site_id"].tolist() == result.site_of_address.tolist() == [0, 0, 0, 1, 2]

    def test_rows_follow_the_input_order_whatever_its_index(self, oriented) -> None:
        addresses, result = oriented
        expected = address_table(addresses, result.sites, result.site_of_address)
        relabelled = addresses.set_index(pd.Index([90, 70, 50, 30, 10]))
        table = address_table(relabelled, result.sites, result.site_of_address)
        pd.testing.assert_frame_equal(table, expected)

    def test_address_kind(self, oriented) -> None:
        addresses, result = oriented
        table = address_table(addresses, result.sites, result.site_of_address)
        kinds = ["unit", "complex", "unit", "standalone", "standalone"]
        assert table["address_kind"].tolist() == kinds

    def test_values(self, oriented) -> None:
        addresses, result = oriented
        table = address_table(addresses, result.sites, result.site_of_address)

        assert table.loc[0, "address"] == "UNIT 1, 5 SMITH STREET, CHATSWOOD NSW 2067"
        assert table.loc[3, "address"] == "9 JONES AVENUE, WILLOUGHBY NSW 2068"
        assert table["locality"].tolist()[3:] == ["WILLOUGHBY", "CHATSWOOD"]
        np.testing.assert_allclose(table["longitude"], addresses.geometry.x, atol=1e-7)
        np.testing.assert_allclose(table["latitude"], addresses.geometry.y, atol=1e-7)
        assert ((table["longitude"] > 151) & (table["latitude"] < -33)).all()

        assert table["orientation"].iloc[:4].tolist() == ["N", "N", "N", "E"]
        assert pd.isna(table.loc[4, "orientation"])  # a street-locality geocode
        assert table["confidence"].tolist() == ["high"] * 4 + ["none"]
        assert table["match"].tolist() == ["street_name"] * 5
        assert table["street"].tolist()[:4] == ["Smith Street"] * 3 + ["Jones Avenue"]
        assert table["geocode_type"].iloc[4] == "STREET LOCALITY"
        assert table.loc[0, "distance_to_street_m"] == pytest.approx(20.0)
        assert table.loc[3, "distance_to_street_m"] == pytest.approx(25.0)
        # Bearings and distances are published to one decimal place.
        site_bearing = result.sites["bearing_deg"].iloc[result.site_of_address].to_numpy()
        np.testing.assert_allclose(table["bearing_deg"], np.round(site_bearing, 1))

    def test_published_bearings_stay_below_360(self, scene) -> None:
        addresses = scene.addresses(smith((0, -20))).to_crs(4326)
        sites = pd.DataFrame(
            {"orientation": ["N"], "bearing_deg": [359.97], "matched_street": ["Smith Street"],
             "distance_m": [20.0], "match": ["street_name"], "confidence": ["high"]}
        )  # fmt: skip
        table = address_table(addresses, sites, np.array([0]))
        assert 0.0 <= table.loc[0, "bearing_deg"] < 360.0


def test_facing_lines(oriented) -> None:
    _, result = oriented
    sites = result.sites
    lines = facing_lines(sites)

    assert lines.crs.to_epsg() == 4326
    assert lines.columns.tolist() == [
        "site_id", "street_key", "matched_street", "orientation", "bearing_deg", "distance_m",
        "confidence", "n_addresses", "geometry",
    ]  # fmt: skip
    assert lines["site_id"].tolist() == [0, 1]  # the street-locality site has no bearing
    assert lines["n_addresses"].tolist() == [3, 1]
    projected = lines.to_crs(sites.crs)
    for line, site_id in zip(projected.geometry, lines["site_id"], strict=True):
        site = sites.loc[site_id]
        start, end = shapely.get_coordinates(line)
        np.testing.assert_allclose(start, [site["x"], site["y"]], atol=1e-6)
        np.testing.assert_allclose(end, [site["street_x"], site["street_y"]], atol=1e-6)


class TestOrientationShares:
    @pytest.fixture
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "orientation": ["N", "N", "E", "S", "NW", None, "N", "NE"],
                "group": ["a", "a", "a", "b", "b", "b", "b", "a"],
            }
        )

    def test_overall(self, table: pd.DataFrame) -> None:
        shares = orientation_shares(table)

        columns = ["with_orientation", *COUNT_COLUMNS, *SHARE_COLUMNS, "share_northerly"]
        assert list(shares.columns) == columns
        assert len(shares) == 1
        row = shares.iloc[0]
        assert row["with_orientation"] == 7  # the address without an orientation is left out
        assert row[COUNT_COLUMNS].tolist() == [3, 1, 1, 0, 1, 0, 0, 1]
        assert row[SHARE_COLUMNS].sum() == pytest.approx(1.0, abs=8 * 5e-5)  # 4-decimal rounding
        assert row["share_N"] == pytest.approx(round(3 / 7, 4))
        assert row["share_northerly"] == pytest.approx(round(5 / 7, 4))  # NW + N + NE

    def test_by_group(self, table: pd.DataFrame) -> None:
        shares = orientation_shares(table, by="group")

        assert shares["group"].tolist() == ["a", "b"]
        assert shares["with_orientation"].tolist() == [4, 3]
        assert shares[COUNT_COLUMNS].sum(axis=1).tolist() == [4, 3]
        np.testing.assert_allclose(shares[SHARE_COLUMNS].sum(axis=1), 1.0, atol=8 * 5e-5)
        assert shares.loc[0, "n_SW"] == 0
        assert shares["share_northerly"].tolist() == pytest.approx([0.75, round(2 / 3, 4)])


def test_suburb_summary() -> None:
    table = pd.DataFrame(
        {
            "gnaf_pid": [f"GA{i}" for i in range(6)],
            "locality": ["CHATSWOOD"] * 3 + ["WILLOUGHBY"] * 3,
            "address_kind": ["standalone", "unit", "standalone", "standalone", "complex", "unit"],
            "confidence": ["high", "high", "low", "medium", "high", "high"],
            "orientation": ["N", "E", "S", "W", "N", "N"],
        }
    )
    summary = suburb_summary(table)

    assert summary["locality"].tolist() == ["CHATSWOOD", "WILLOUGHBY"]
    assert summary["addresses"].tolist() == [3, 3]
    assert summary["standalone"].tolist() == [2, 1]
    assert summary["high_confidence"].tolist() == pytest.approx([round(2 / 3, 4), round(2 / 3, 4)])
    # Direction shares describe standalone addresses (mostly houses) only.
    assert summary["with_orientation"].tolist() == [2, 1]
    assert summary["n_E"].tolist() == [0, 0]
    assert summary["share_W"].tolist() == [0.0, 1.0]


def test_write_geopackage_replaces_the_file(tmp_path, scene) -> None:
    path = tmp_path / "result.gpkg"
    write_geopackage(path, {"stale": scene.frame([scene.point(0, 0)], label=["old"])})
    write_geopackage(
        path,
        {
            "points": scene.frame([scene.point(1, 1)], label=["a"]),
            "lines": scene.frame([scene.line([(0, 0), (5, 5)])], label=["b"]),
        },
    )
    assert sorted(pyogrio.list_layers(path)[:, 0]) == ["lines", "points"]
    points = gpd.read_file(path, layer="points")
    assert points["label"].tolist() == ["a"]
    assert points.crs.to_epsg() == 7856
