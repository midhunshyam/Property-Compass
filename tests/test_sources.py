"""Downloaders, tested offline: HTTP and DuckDB are replaced by fakes, so nothing here
touches the network (the ``offline`` guard in ``conftest.py`` would fail the test)."""

from __future__ import annotations

import json
import re
import socket
import time
from collections.abc import Callable

import geopandas as gpd
import pandas as pd
import pytest
import requests
import shapely

from property_orientation.sources import USER_AGENT, gnaf, nsw_cadastre, osm
from property_orientation.sources.nsw_cadastre import esri_to_shapely
from property_orientation.sources.osm import _tiles

# Esri JSON polygons list rings flat: outer rings clockwise, holes anticlockwise.
OUTER = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
HOLE = [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]
SECOND_OUTER = [[20, 0], [20, 10], [30, 10], [30, 0], [20, 0]]


class FakeResponse:
    """Just enough of :class:`requests.Response` for the downloaders."""

    def __init__(self, payload: object = None, *, text: str = "", status: int = 200) -> None:
        self.payload, self.text, self.status_code = payload, text, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self) -> object:
        if self.payload is None:
            raise requests.JSONDecodeError("Expecting value", self.text, 0)
        return self.payload


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the downloaders' pauses instead of waiting."""
    pauses: list[float] = []
    monkeypatch.setattr(time, "sleep", pauses.append)
    return pauses


def test_the_suite_is_offline() -> None:
    with pytest.raises(pytest.fail.Exception, match="network access"):
        socket.create_connection(("overpass-api.de", 443), timeout=1)
    with pytest.raises(pytest.fail.Exception, match="network access"):
        requests.get("https://portal.spatial.nsw.gov.au", timeout=1)


def test_user_agent_avoids_the_word_property() -> None:
    # The Overpass API rejects user agents containing "property" (used by scrapers).
    assert "property" not in USER_AGENT.lower()


class TestEsriToShapely:
    def test_polygon_with_a_hole(self) -> None:
        geom = esri_to_shapely({"rings": [OUTER, HOLE]}, "esriGeometryPolygon")
        assert geom.geom_type == "Polygon"
        assert len(geom.interiors) == 1
        assert geom.area == pytest.approx(96.0)

    def test_each_outer_ring_starts_a_part(self) -> None:
        geom = esri_to_shapely({"rings": [OUTER, HOLE, SECOND_OUTER]}, "esriGeometryPolygon")
        assert geom.geom_type == "MultiPolygon"
        assert [len(part.interiors) for part in geom.geoms] == [1, 0]
        assert geom.area == pytest.approx(196.0)

    def test_degenerate_rings_are_skipped(self) -> None:
        sliver = [[0, 0], [1, 1], [0, 0]]
        geom = esri_to_shapely({"rings": [sliver, OUTER]}, "esriGeometryPolygon")
        assert geom.area == pytest.approx(100.0)

    def test_invalid_polygon_is_repaired(self) -> None:
        bowtie = [[0, 0], [0, 10], [10, 0], [10, 10], [0, 0]]
        geom = esri_to_shapely({"rings": [bowtie]}, "esriGeometryPolygon")
        assert geom.is_valid
        assert geom.area == pytest.approx(50.0)

    @pytest.mark.parametrize("geometry", [None, {}, {"rings": []}, {"rings": [[[0, 0], [1, 1]]]}])
    def test_empty_polygon_is_none(self, geometry: dict | None) -> None:
        assert esri_to_shapely(geometry, "esriGeometryPolygon") is None

    def test_polylines(self) -> None:
        one = esri_to_shapely({"paths": [[[0, 0], [5, 5]]]}, "esriGeometryPolyline")
        assert one.geom_type == "LineString"
        paths = [[[0, 0], [5, 5]], [[6, 6], [9, 9]], [[1, 1]]]
        two = esri_to_shapely({"paths": paths}, "esriGeometryPolyline")
        assert two.geom_type == "MultiLineString"
        assert len(two.geoms) == 2  # a one-point path is dropped


# --- NSW Spatial Services (ArcGIS REST) ------------------------------------------------

# A clockwise (Esri outer) ring in Chatswood, in longitude/latitude.
RING = [[151.18, -33.8], [151.18, -33.799], [151.181, -33.799], [151.181, -33.8], [151.18, -33.8]]


def feature(cadid: int, geometry: dict | None = None) -> dict:
    return {"attributes": {"cadid": cadid, "roadtype": "Public Road"}, "geometry": geometry}


class FakeSession:
    """A :class:`requests.Session` stand-in that replays query pages in order."""

    def __init__(self, *pages: object) -> None:
        self.headers: dict[str, str] = {}
        self.pages = list(pages)
        self.queries: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> object:
        if params == {"f": "json"}:
            return FakeResponse(
                {"objectIdField": "objectid", "geometryType": "esriGeometryPolygon"}
            )
        self.queries.append((url, dict(params or {})))
        page = self.pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return page if isinstance(page, FakeResponse) else FakeResponse(page)


class TestNswCadastre:
    def test_pages_until_the_transfer_limit_is_not_exceeded(self, sleeps) -> None:
        session = FakeSession(
            {"features": [feature(1, {"rings": [RING]}), feature(2, {"rings": [RING]})],
             "exceededTransferLimit": True},
            {"features": [feature(3, {"rings": [RING]}), feature(4)]},
        )  # fmt: skip
        bounds = (151.17, -33.81, 151.19, -33.79)
        frame = nsw_cadastre.fetch_layer("road_reserve", bounds, session=session)

        assert frame["cadid"].tolist() == [1, 2, 3]  # the feature without geometry is dropped
        assert frame.crs.to_epsg() == 4326
        assert frame.geom_type.eq("Polygon").all()
        assert session.headers["User-Agent"] == USER_AGENT
        urls = {url for url, _ in session.queries}
        assert urls == {f"{nsw_cadastre.SERVICE_URL}/{nsw_cadastre.LAYERS['road_reserve']}/query"}
        params = [p for _, p in session.queries]
        assert [p["resultOffset"] for p in params] == [0, 2]
        first = params[0]
        assert first["geometry"] == "151.17,-33.81,151.19,-33.79"
        assert (first["inSR"], first["outSR"]) == (4326, 4326)
        assert first["outFields"] == nsw_cadastre.FIELDS["road_reserve"]
        assert first["orderByFields"] == "objectid"  # stable paging order
        assert first["resultRecordCount"] == nsw_cadastre.PAGE_SIZE
        assert sleeps == [0.25]

    def test_an_empty_page_ends_paging(self, sleeps) -> None:
        session = FakeSession(
            {"features": [feature(1, {"rings": [RING]})], "exceededTransferLimit": True},
            {"features": [], "exceededTransferLimit": True},
        )
        frame = nsw_cadastre.fetch_layer("property", (0, 0, 1, 1), session=session)
        assert frame["cadid"].tolist() == [1]
        assert len(session.queries) == 2

    def test_no_features_in_the_bounds(self, sleeps) -> None:
        session = FakeSession({"features": []})
        frame = nsw_cadastre.fetch_layer("road_reserve", (0, 0, 1, 1), session=session)
        assert frame.empty
        assert frame.crs.to_epsg() == 4326

    def test_retries_errors_then_returns_the_page(self, sleeps) -> None:
        page = {"features": [feature(1, {"rings": [RING]})]}
        session = FakeSession(
            requests.ConnectionError("connection reset"),
            FakeResponse(status=503),
            {"error": {"code": 500, "message": "busy"}},  # an ArcGIS error payload
            FakeResponse(text="<html>gateway timeout</html>"),  # not JSON
            page,
        )
        assert nsw_cadastre._get_json(session, "https://example.invalid/query", {}) == page
        assert sleeps == [5, 10, 15, 20]

    def test_gives_up_after_the_last_attempt(self, sleeps) -> None:
        session = FakeSession(*[{"error": {"message": "busy"}}] * 3)
        with pytest.raises(RuntimeError, match="busy"):
            nsw_cadastre._get_json(session, "https://example.invalid/query", {}, attempts=3)
        assert sleeps == [5, 10]


# --- OpenStreetMap (Overpass API) --------------------------------------------------------


class TestOverpassTiles:
    def test_tiles_cover_the_bounds_without_overlap(self) -> None:
        bounds = (0.0, 0.0, 0.75, 0.5)
        tiles = list(_tiles(bounds, 0.25))
        assert len(tiles) == 6
        boxes = [shapely.box(*tile) for tile in tiles]
        assert sum(box.area for box in boxes) == pytest.approx(shapely.box(*bounds).area)
        assert shapely.union_all(boxes).equals(shapely.box(*bounds))

    def test_small_area_is_one_tile(self) -> None:
        bounds = (151.20, -33.80, 151.21, -33.79)
        assert list(_tiles(bounds, 0.03)) == [pytest.approx(bounds)]


def way(osm_id: int, coords: list[tuple[float, float]], **tags: str) -> dict:
    """An Overpass ``out tags geom`` way element."""
    nodes = [{"lon": lon, "lat": lat} for lon, lat in coords]
    return {"type": "way", "id": osm_id, "tags": tags, "geometry": nodes}


class TestOverpass:
    def test_query_fails_over_to_the_next_endpoint(self, monkeypatch, sleeps) -> None:
        calls: list[tuple[str, dict, dict]] = []

        def post(url: str, data: dict, headers: dict, timeout: float) -> FakeResponse:
            calls.append((url, data, headers))
            if len(calls) == 1:
                raise requests.ConnectionError("too busy")
            if len(calls) == 2:
                return FakeResponse(text="rate limited")  # an HTML error page, not JSON
            return FakeResponse({"elements": []})

        monkeypatch.setattr(osm.requests, "post", post)
        assert osm._query("[out:json];way(1);out;") == {"elements": []}
        assert [url for url, _, _ in calls] == list(osm.OVERPASS_ENDPOINTS)
        assert all(data == {"data": "[out:json];way(1);out;"} for _, data, _ in calls)
        assert all(headers["User-Agent"] == USER_AGENT for _, _, headers in calls)
        assert sleeps == []  # every endpoint is tried before pausing

    def test_query_reports_every_failure(self, monkeypatch, sleeps) -> None:
        def post(url: str, **kwargs: object) -> FakeResponse:
            return FakeResponse(status=504)

        monkeypatch.setattr(osm.requests, "post", post)
        with pytest.raises(RuntimeError, match="Overpass query failed") as failure:
            osm._query("[out:json];", rounds=2)
        lines = str(failure.value).splitlines()[1:]
        endpoints = [*osm.OVERPASS_ENDPOINTS, *osm.OVERPASS_ENDPOINTS]  # main server first, twice
        assert [line.split(": ")[0] for line in lines] == endpoints
        assert all("504" in line for line in lines)
        assert sleeps == [30]  # a pause between rounds, none after the last

    def test_query_retries_the_main_server_in_every_round(self, monkeypatch, sleeps) -> None:
        calls: list[str] = []

        def post(url: str, **kwargs: object) -> FakeResponse:
            calls.append(url)
            busy = len(calls) <= len(osm.OVERPASS_ENDPOINTS)  # the whole first round fails
            return FakeResponse(status=504) if busy else FakeResponse({"elements": []})

        monkeypatch.setattr(osm.requests, "post", post)
        assert osm._query("[out:json];") == {"elements": []}
        assert calls == [*osm.OVERPASS_ENDPOINTS, osm.OVERPASS_ENDPOINTS[0]]
        assert sleeps == [30]

    def test_fetch_highways_merges_tiles(self, monkeypatch, sleeps) -> None:
        shared = way(7, [(0.2, 0.1), (0.3, 0.1)], highway="residential", name="Smith Street")
        tiles = {
            "(0.0,0.0,0.25,0.25)": (
                "2026-10-09T05:44:36Z",
                [way(9, [(0.1, 0.1), (0.1, 0.2)], highway="footway"), shared],
            ),
            "(0.0,0.25,0.25,0.5)": (
                "2026-10-09T05:45:01Z",
                [
                    shared,  # crosses the tile edge, so both tiles return it
                    way(3, [(0.4, 0.1)], highway="residential"),  # a single node: no line
                    way(5, [(0.3, 0.2), (0.4, 0.2), (0.45, 0.24)], highway="service",
                        service="driveway", access="private", lanes="1"),
                ],
            ),
        }  # fmt: skip
        queries: list[str] = []

        def query(text: str) -> dict:
            queries.append(text)
            timestamp, elements = tiles[re.search(r"\([-\d.,]+\)", text).group(0)]
            return {"osm3s": {"timestamp_osm_base": timestamp}, "elements": elements}

        monkeypatch.setattr(osm, "_query", query)
        roads, timestamp = osm.fetch_highways((0.0, 0.0, 0.5, 0.25), tile_deg=0.25)

        assert len(queries) == 2
        assert all(q.startswith("[out:json]") and 'way["highway"]' in q for q in queries)
        assert timestamp == "2026-10-09T05:45:01Z"  # the newest tile
        assert roads["osm_id"].tolist() == [5, 7, 9]  # unique and sorted
        assert list(roads.columns) == ["osm_id", *osm.TAGS, "geometry"]  # "lanes" is not kept
        assert roads.crs.to_epsg() == 4326
        driveway = roads.iloc[0]
        assert (driveway["highway"], driveway["service"], driveway["access"]) == (
            "service", "driveway", "private"
        )  # fmt: skip
        assert pd.isna(driveway["name"])
        assert shapely.get_coordinates(driveway.geometry).tolist() == [
            [0.3, 0.2], [0.4, 0.2], [0.45, 0.24]
        ]  # fmt: skip
        assert sleeps == [2, 2]

    def test_fetch_highways_with_no_ways(self, monkeypatch, sleeps) -> None:
        empty = {"osm3s": {"timestamp_osm_base": "2026-10-09T05:44:36Z"}, "elements": []}
        monkeypatch.setattr(osm, "_query", lambda text: empty)
        roads, timestamp = osm.fetch_highways((0.0, 0.0, 0.01, 0.01))
        assert roads.empty
        assert timestamp == "2026-10-09T05:44:36Z"


# --- G-NAF and administrative boundaries (gnaf-loader GeoParquet on S3) -----------------


def listing(*keys: str, token: str | None = None) -> str:
    """An S3 ListObjectsV2 XML page."""
    contents = "".join(f"<Contents><Key>{key}</Key><Size>1</Size></Contents>" for key in keys)
    more = f"<IsTruncated>true</IsTruncated><NextContinuationToken>{token}</NextContinuationToken>"
    return f"<ListBucketResult>{contents}{more if token else ''}</ListBucketResult>"


class TestGnafListing:
    def test_dataset_parts_follows_continuation_tokens(self, monkeypatch) -> None:
        prefix = f"{gnaf.PREFIX}/address_principals"
        pages = {
            None: listing(
                f"{prefix}/part-0.parquet",
                f"{prefix}/._part-0.parquet",  # a hidden file
                f"{prefix}/_SUCCESS",
                token="page+2",
            ),
            "page+2": listing(f"{prefix}/part-1.parquet"),
        }
        calls: list[dict] = []

        def get(url: str, params: dict, headers: dict, timeout: float) -> FakeResponse:
            assert url == f"{gnaf.BUCKET_URL}/"
            assert headers["User-Agent"] == USER_AGENT
            calls.append(dict(params))
            return FakeResponse(text=pages[params.get("continuation-token")])

        monkeypatch.setattr(gnaf.requests, "get", get)
        parts = gnaf.dataset_parts("address_principals")

        assert parts == [
            f"{gnaf.BUCKET_URL}/{prefix}/part-0.parquet",
            f"{gnaf.BUCKET_URL}/{prefix}/part-1.parquet",
        ]
        assert [c.get("continuation-token") for c in calls] == [None, "page+2"]
        assert all(c["list-type"] == "2" and c["prefix"] == f"{prefix}/" for c in calls)

    def test_no_parts_is_an_error(self, monkeypatch) -> None:
        monkeypatch.setattr(gnaf.requests, "get", lambda *a, **k: FakeResponse(text=listing()))
        with pytest.raises(RuntimeError, match="no Parquet files found"):
            gnaf.dataset_parts("nowhere")

    def test_http_errors_are_raised(self, monkeypatch) -> None:
        monkeypatch.setattr(gnaf.requests, "get", lambda *a, **k: FakeResponse(status=403))
        with pytest.raises(requests.HTTPError):
            gnaf.dataset_parts("address_principals")


class FakeDuckDB:
    """Answers ``parquet_kv_metadata`` queries from a ``{url: geo metadata}`` mapping."""

    def __init__(self, geo: dict[str, dict | None]) -> None:
        self.geo = geo
        self.rows: list[tuple[str]] = []

    def execute(self, sql: str, params: object = None) -> FakeDuckDB:
        url = re.search(r"parquet_kv_metadata\('([^']+)'\)", sql).group(1)
        meta = self.geo[url]
        self.rows = [] if meta is None else [(json.dumps(meta),)]
        return self

    def fetchall(self) -> list[tuple[str]]:
        return self.rows


def geo_metadata(bbox: list[float] | None) -> dict:
    column = {"encoding": "WKB", "geometry_types": ["Point"]}
    return {"version": "1.0.0", "primary_column": "geom",
            "columns": {"geom": column | ({"bbox": bbox} if bbox else {})}}  # fmt: skip


class TestOverlappingParts:
    BOUNDS = (151.14, -33.83, 151.24, -33.77)

    def test_keeps_parts_whose_bbox_overlaps(self, monkeypatch) -> None:
        boxes = {
            "inside": [151.15, -33.82, 151.16, -33.81],
            "overlapping": [150.0, -35.0, 151.15, -33.0],
            "touching": [151.24, -34.0, 152.0, -33.0],
            "east": [151.25, -34.0, 152.0, -33.0],
            "west": [150.0, -34.0, 151.13, -33.0],
            "north": [150.0, -33.76, 152.0, -33.0],
            "south": [150.0, -35.0, 152.0, -33.84],
        }
        geo = {url: geo_metadata(bbox) for url, bbox in boxes.items()} | {"no metadata": None}
        monkeypatch.setattr(gnaf, "_connection", lambda: FakeDuckDB(geo))

        kept = gnaf._overlapping_parts(list(geo), self.BOUNDS)
        assert kept == ["inside", "overlapping", "touching", "no metadata"]

    def test_keeps_a_part_without_a_bbox(self, monkeypatch) -> None:
        geo = {"part-0": geo_metadata(None)}
        monkeypatch.setattr(gnaf, "_connection", lambda: FakeDuckDB(geo))
        assert gnaf._overlapping_parts(["part-0"], self.BOUNDS) == ["part-0"]


AREA = shapely.box(151.18, -33.81, 151.20, -33.79)
"""A study area in Chatswood (EPSG:4326)."""


def fake_query_geo(frame: gpd.GeoDataFrame, calls: list) -> Callable:
    def query_geo(sql: str, params: list | None = None) -> gpd.GeoDataFrame:
        calls.append((sql, params))
        return frame.copy()

    return query_geo


class TestGnafQueries:
    def test_fetch_lga_dissolves_the_parts(self, monkeypatch) -> None:
        calls: list = []
        parts = gpd.GeoDataFrame(
            {"lga_pid": ["lgacbffb11990f2", "lgacbffb11990f2"], "name": ["Willoughby"] * 2,
             "full_name": ["Willoughby City Council"] * 2, "state": ["NSW"] * 2},
            geometry=[shapely.box(151.18, -33.81, 151.19, -33.79),
                      shapely.box(151.19, -33.81, 151.20, -33.79)],
            crs=4326,
        )  # fmt: skip
        monkeypatch.setattr(
            gnaf, "dataset_parts", lambda dataset: [f"https://s3/{dataset}.parquet"]
        )
        monkeypatch.setattr(gnaf, "_query_geo", fake_query_geo(parts, calls))

        lga = gnaf.fetch_lga("Willoughby", "NSW")
        assert len(lga) == 1
        assert list(lga.columns) == ["lga_pid", "name", "full_name", "state", "geometry"]
        assert lga.geometry.iloc[0].equals(AREA)
        [(sql, params)] = calls
        assert "https://s3/local_government_areas.parquet" in sql
        assert params == ["Willoughby", "NSW"]

    def test_fetch_lga_unknown_name(self, monkeypatch) -> None:
        empty = gpd.GeoDataFrame(
            {"lga_pid": [], "name": [], "full_name": [], "state": []}, geometry=[], crs=4326
        )
        monkeypatch.setattr(gnaf, "dataset_parts", lambda dataset: ["https://s3/lga.parquet"])
        monkeypatch.setattr(gnaf, "_query_geo", fake_query_geo(empty, []))
        with pytest.raises(ValueError, match="no local government area named 'Nowhere' in NSW"):
            gnaf.fetch_lga("Nowhere", "NSW")

    def test_fetch_localities_share_in_area(self, monkeypatch) -> None:
        calls: list = []
        rows = gpd.GeoDataFrame(
            {"locality_pid": ["loc1", "loc2", "loc2", "loc3"],
             "locality_name": ["CHATSWOOD", "ARTARMON", "ARTARMON", "ROSEVILLE"],
             "postcode": ["2067", "2064", "2064", "2069"], "state": ["NSW"] * 4},
            geometry=[
                shapely.box(151.185, -33.805, 151.195, -33.795),  # inside
                shapely.box(151.19, -33.82, 151.20, -33.81),  # outside the area ...
                shapely.box(151.19, -33.81, 151.21, -33.80),  # ... but this part straddles it
                shapely.box(151.21, -33.81, 151.22, -33.80),  # bbox query hit, no overlap
            ],
            crs=4326,
        )  # fmt: skip
        monkeypatch.setattr(gnaf, "dataset_parts", lambda dataset: ["https://s3/loc.parquet"])
        monkeypatch.setattr(gnaf, "_query_geo", fake_query_geo(rows, calls))

        localities = gnaf.fetch_localities(AREA, "NSW").set_index("locality_name")
        assert sorted(localities.index) == ["ARTARMON", "CHATSWOOD"]
        assert localities.loc["CHATSWOOD", "share_in_area"] == 1.0
        # Artarmon's two parts are merged; the half of the larger part that lies inside the
        # area is a third of the whole.
        assert localities.loc["ARTARMON", "share_in_area"] == pytest.approx(1 / 3, abs=2e-3)
        assert localities.crs.to_epsg() == 4326
        [(sql, params)] = calls
        assert params == ["NSW"]
        assert "geom_bbox.xmax >= 151.18" in sql and "geom_bbox.ymin <= -33.79" in sql

    def test_fetch_addresses(self, monkeypatch) -> None:
        calls: list = []
        overlapping: list = []
        points = gpd.GeoDataFrame(
            {"gnaf_pid": ["GANSW3", "GANSW1", "GANSW2", "GANSW9"],
             "confidence": ["2", "", None, "1"], "reliability": ["2", "2", "3", "2"],
             "mb_2021_code": ["11205258900", "11205258900", None, "1"]},
            geometry=[shapely.Point(151.19, -33.80), shapely.Point(151.181234567891, -33.80),
                      shapely.Point(151.199, -33.809), shapely.Point(151.25, -33.80)],
            crs=4326,
        )  # fmt: skip

        def parts(urls: list[str], bounds: tuple) -> list[str]:
            overlapping.append((urls, bounds))
            return urls[1:]

        monkeypatch.setattr(gnaf, "dataset_parts", lambda dataset: [f"{dataset}-0", f"{dataset}-1"])
        monkeypatch.setattr(gnaf, "_overlapping_parts", parts)
        monkeypatch.setattr(gnaf, "_query_geo", fake_query_geo(points, calls))

        addresses = gnaf.fetch_addresses(AREA)
        assert overlapping == [(["address_principals-0", "address_principals-1"], AREA.bounds)]
        [(sql, _)] = calls
        assert "read_parquet(['address_principals-1'])" in sql
        assert all(column in sql for column in gnaf.ADDRESS_COLUMNS)

        assert addresses["gnaf_pid"].tolist() == ["GANSW1", "GANSW2", "GANSW3"]  # inside, sorted
        assert addresses.index.tolist() == [0, 1, 2]
        # Coordinates are copied from the geometry, rounded to 8 decimals (about 1 mm).
        longitude, latitude = [151.18123457, 151.199, 151.19], [-33.8, -33.809, -33.8]
        assert addresses["longitude"].tolist() == pytest.approx(longitude, abs=1e-10)
        assert addresses["latitude"].tolist() == pytest.approx(latitude, abs=1e-10)
        for column in ("confidence", "reliability", "mb_2021_code"):
            assert addresses[column].dtype == "Int64"
        assert addresses["confidence"].tolist() == [pd.NA, pd.NA, 2]
        assert addresses["mb_2021_code"].tolist() == [11205258900, pd.NA, 11205258900]
