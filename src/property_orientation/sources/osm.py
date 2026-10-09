"""OpenStreetMap highways via the Overpass API.

Licence: © OpenStreetMap contributors, available under the Open Database Licence (ODbL).
"""

from __future__ import annotations

import logging
import math
import time

import geopandas as gpd
import requests
from shapely.geometry import LineString

from property_orientation.sources import USER_AGENT

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
log = logging.getLogger("property_orientation")
TAGS = ("highway", "name", "alt_name", "official_name", "old_name", "service", "access", "area")


def _query(query: str, rounds: int = 6) -> dict:
    """POST an Overpass query, trying every endpoint (main server first) in each round.

    Public servers answer "too busy" (HTTP 504) under load, which usually clears within a few
    minutes, so rounds are separated by a growing pause.
    """
    errors = []
    for attempt in range(rounds):
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                response = requests.post(
                    endpoint, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=300
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{endpoint}: {exc}")
                log.warning("Overpass server %s failed: %s", endpoint, exc)
        if attempt < rounds - 1:
            pause = 30 * (attempt + 1)
            log.warning(
                "all Overpass servers failed; retrying in %d s (%d/%d)", pause, attempt + 1, rounds
            )
            time.sleep(pause)
    raise RuntimeError("Overpass query failed:\n" + "\n".join(errors))


def _tiles(bounds: tuple[float, float, float, float], size_deg: float):
    minx, miny, maxx, maxy = bounds
    nx = max(1, math.ceil((maxx - minx) / size_deg))
    ny = max(1, math.ceil((maxy - miny) / size_deg))
    dx, dy = (maxx - minx) / nx, (maxy - miny) / ny
    for i in range(nx):
        for j in range(ny):
            yield minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy


def fetch_highways(
    bounds: tuple[float, float, float, float], tile_deg: float = 0.03
) -> tuple[gpd.GeoDataFrame, str]:
    """All ``highway=*`` ways intersecting ``bounds`` (minx, miny, maxx, maxy in EPSG:4326).

    The area is split into tiles of about ``tile_deg`` degrees and queried one tile at a
    time; ways that cross tile edges are de-duplicated by OSM id. Returns the ways as
    LineStrings, plus the timestamp of the OSM data that Overpass served.
    """
    ways: dict[int, dict] = {}
    timestamps = []
    for minx, miny, maxx, maxy in _tiles(bounds, tile_deg):
        payload = _query(
            f'[out:json][timeout:180];way["highway"]({miny},{minx},{maxy},{maxx});out tags geom;'
        )
        timestamps.append(payload["osm3s"]["timestamp_osm_base"])
        for element in payload["elements"]:
            coords = [(node["lon"], node["lat"]) for node in element.get("geometry", [])]
            if len(coords) < 2:
                continue
            tags = element.get("tags", {})
            ways[element["id"]] = (
                {"osm_id": element["id"]}
                | {tag: tags.get(tag) for tag in TAGS}
                | {"geometry": LineString(coords)}
            )
        time.sleep(2)  # stay well inside the public rate limit
    if not ways:
        columns = {name: [] for name in ("osm_id", *TAGS)}
        return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=4326)), max(timestamps)
    roads = gpd.GeoDataFrame(list(ways.values()), geometry="geometry", crs=4326)
    return roads.sort_values("osm_id").reset_index(drop=True), max(timestamps)
