"""G-NAF addresses and Geoscape administrative boundaries.

The data is read from the public GeoParquet export that the `gnaf-loader
<https://github.com/minus34/gnaf-loader>`_ project publishes on S3. DuckDB queries the
remote files over HTTP. Only files whose GeoParquet bounding box overlaps the study area
are scanned, so a single local government area downloads about 100 MB rather than the
full 1.3 GB.

Licence: G-NAF © Geoscape Australia, licensed by the Commonwealth of Australia under the
Open G-NAF End User Licence Agreement. Administrative Boundaries © Geoscape Australia,
licensed under CC BY 4.0.
"""

from __future__ import annotations

import json
import re
from functools import cache

import duckdb
import geopandas as gpd
import pandas as pd
import requests
import shapely
from shapely.geometry.base import BaseGeometry

from property_orientation.sources import USER_AGENT

RELEASE = "202608"
"""Geoscape release (YYYYMM) of G-NAF and the Administrative Boundaries."""

BUCKET_URL = "https://s3.ap-southeast-2.amazonaws.com/minus34.com"
PREFIX = f"opendata/geoscape-{RELEASE}/geoparquet"

ADDRESS_COLUMNS = (
    "gnaf_pid", "street_locality_pid", "locality_pid", "primary_secondary", "building_name",
    "lot_number", "flat_number", "level_number", "number_first", "number_last",
    "street_name", "street_type", "street_suffix", "address", "locality_name", "postcode",
    "state", "confidence", "legal_parcel_id", "mb_2021_code", "geocode_type", "reliability",
)  # fmt: skip


def dataset_parts(dataset: str) -> list[str]:
    """HTTPS URLs of the Parquet part files of a gnaf-loader dataset."""
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": f"{PREFIX}/{dataset}/"}
        if token:
            params["continuation-token"] = token
        response = requests.get(
            f"{BUCKET_URL}/", params=params, headers={"User-Agent": USER_AGENT}, timeout=60
        )
        response.raise_for_status()
        body = response.text
        keys += re.findall(r"<Key>([^<]+\.parquet)</Key>", body)
        match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not match:
            break
        token = match.group(1)
    parts = [f"{BUCKET_URL}/{k}" for k in keys if "/." not in k]
    if not parts:
        raise RuntimeError(f"no Parquet files found for dataset {dataset!r} ({PREFIX})")
    return parts


@cache
def _connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    return con


def _query_geo(sql: str, params: list | None = None) -> gpd.GeoDataFrame:
    """Run SQL that returns a ``wkb`` column, and return a GeoDataFrame in EPSG:4326."""
    frame = _connection().execute(sql, params or []).fetchdf()
    geometry = shapely.from_wkb(frame.pop("wkb").map(bytes).to_numpy())
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=4326)


def _overlapping_parts(parts: list[str], bounds: tuple[float, float, float, float]) -> list[str]:
    """Part files whose GeoParquet ``bbox`` metadata overlaps ``bounds``."""
    minx, miny, maxx, maxy = bounds
    keep = []
    for url in parts:
        rows = (
            _connection()
            .execute(
                f"SELECT decode(value) FROM parquet_kv_metadata('{url}') WHERE decode(key) = 'geo'"
            )
            .fetchall()
        )
        bbox = json.loads(rows[0][0])["columns"]["geom"].get("bbox") if rows else None
        if bbox is None or not (
            bbox[2] < minx or bbox[0] > maxx or bbox[3] < miny or bbox[1] > maxy
        ):
            keep.append(url)
    return keep


def fetch_lga(name: str, state: str) -> gpd.GeoDataFrame:
    """Boundary of a local government area, dissolved into one (multi)polygon."""
    parts = dataset_parts("local_government_areas")
    lga = _query_geo(
        f"""SELECT lga_pid, name, full_name, state, ST_AsWKB(geom) AS wkb
            FROM read_parquet({parts!r})
            WHERE upper(name) = upper(?) AND upper(state) = upper(?)""",
        [name, state],
    )
    if lga.empty:
        raise ValueError(f"no local government area named {name!r} in {state}")
    return lga.dissolve(by=["lga_pid", "name", "full_name", "state"]).reset_index()


def fetch_localities(area: BaseGeometry, state: str) -> gpd.GeoDataFrame:
    """Suburb (locality) boundaries intersecting ``area``, with the share inside it."""
    parts = dataset_parts("locality_bdys")
    minx, miny, maxx, maxy = area.bounds
    loc = _query_geo(
        f"""SELECT locality_pid, locality_name, postcode, state, ST_AsWKB(geom) AS wkb
            FROM read_parquet({parts!r})
            WHERE upper(state) = upper(?)
              AND geom_bbox.xmax >= {minx} AND geom_bbox.xmin <= {maxx}
              AND geom_bbox.ymax >= {miny} AND geom_bbox.ymin <= {maxy}""",
        [state],
    )
    loc = loc.dissolve(by=["locality_pid", "locality_name", "postcode", "state"]).reset_index()
    loc = loc[loc.intersects(area)].reset_index(drop=True)
    metric = loc.to_crs(loc.estimate_utm_crs())
    area_metric = gpd.GeoSeries([area], crs=4326).to_crs(metric.crs).iloc[0]
    loc["share_in_area"] = (metric.intersection(area_metric).area / metric.area).round(4)
    return loc


def fetch_addresses(area: BaseGeometry) -> gpd.GeoDataFrame:
    """Principal G-NAF addresses whose geocode falls inside ``area`` (EPSG:4326 points)."""
    parts = _overlapping_parts(dataset_parts("address_principals"), area.bounds)
    minx, miny, maxx, maxy = area.bounds
    columns = ", ".join(ADDRESS_COLUMNS)
    points = _query_geo(
        f"""SELECT {columns}, ST_AsWKB(geom) AS wkb
            FROM read_parquet({parts!r})
            WHERE geom_bbox.xmin BETWEEN {minx} AND {maxx}
              AND geom_bbox.ymin BETWEEN {miny} AND {maxy}"""
    )
    inside = points[points.within(area)].copy()
    inside["longitude"] = inside.geometry.x.round(8)
    inside["latitude"] = inside.geometry.y.round(8)
    for col in ("confidence", "reliability", "mb_2021_code"):
        inside[col] = pd.to_numeric(inside[col], errors="coerce").astype("Int64")
    return inside.sort_values("gnaf_pid").reset_index(drop=True)
