"""NSW Digital Cadastral Database layers from NSW Spatial Services (ArcGIS REST).

Used only to build the independent validation reference: property parcels, road reserve
polygons and named road centrelines.

Licence: © Spatial Services (DCS), NSW Government, licensed under CC BY 4.0.
"""

from __future__ import annotations

import time

import geopandas as gpd
import requests
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from property_orientation.sources import USER_AGENT

SERVICE_URL = (
    "https://portal.spatial.nsw.gov.au/server/rest/services/"
    "NSW_Land_Parcel_Property_Theme/FeatureServer"
)
LAYERS = {"property": 12, "road_reserve": 6, "road_centreline": 1}
FIELDS = {
    "property": "propid,propertytype,address,housenumber,dissolveparcelcount",
    "road_reserve": "cadid,roadtype,classsubtype",
    "road_centreline": "roadnamelabel,urbanity",
}
PAGE_SIZE = 2000


def esri_to_shapely(geometry: dict | None, geometry_type: str):
    """Convert an Esri JSON polyline or polygon to a Shapely geometry.

    Esri polygons list their rings flat: outer rings run clockwise and holes run
    anticlockwise. Each hole belongs to the outer ring that precedes it.
    """
    if not geometry:
        return None
    if geometry_type == "esriGeometryPolyline":
        lines = [LineString(path) for path in geometry.get("paths", []) if len(path) >= 2]
        return lines[0] if len(lines) == 1 else MultiLineString(lines)
    shells: list[tuple[list, list]] = []
    for ring in geometry.get("rings", []):
        if len(ring) < 4:
            continue
        if not shapely.LinearRing(ring).is_ccw or not shells:
            shells.append((ring, []))
        else:
            shells[-1][1].append(ring)
    polygons = [Polygon(shell, holes) for shell, holes in shells]
    if not polygons:
        return None
    geom = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    return geom if geom.is_valid else shapely.make_valid(geom)


def fetch_layer(
    layer: str, bounds: tuple[float, float, float, float], session: requests.Session | None = None
) -> gpd.GeoDataFrame:
    """Every feature of ``layer`` intersecting ``bounds`` (EPSG:4326), paging 2,000 at a time."""
    session = session or requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    url = f"{SERVICE_URL}/{LAYERS[layer]}"
    meta = session.get(url, params={"f": "json"}, timeout=60).json()
    oid, geometry_type = meta["objectIdField"], meta["geometryType"]
    minx, miny, maxx, maxy = bounds
    records, offset = [], 0
    while True:
        params = {
            "where": "1=1",
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": FIELDS[layer],
            "returnGeometry": "true",
            "outSR": 4326,
            "orderByFields": oid,
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }
        page = _get_json(session, f"{url}/query", params)
        features = page["features"]
        for feature in features:
            records.append(
                feature["attributes"]
                | {"geometry": esri_to_shapely(feature.get("geometry"), geometry_type)}
            )
        if not features or not page.get("exceededTransferLimit"):
            break
        offset += len(features)
        time.sleep(0.25)  # be polite to a public service
    if not records:
        columns = {name: [] for name in FIELDS[layer].split(",")}
        return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=4326))
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
    return frame[frame.geometry.notna()].reset_index(drop=True)


def _get_json(session: requests.Session, url: str, params: dict, attempts: int = 5) -> dict:
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=180)
            response.raise_for_status()
            payload = response.json()
            if "features" in payload:
                return payload
            raise RuntimeError(str(payload)[:300])
        except (requests.RequestException, ValueError, RuntimeError):
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")
