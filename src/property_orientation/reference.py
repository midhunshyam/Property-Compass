"""An independent reference orientation from the cadastre.

The orientation method uses G-NAF points and OSM street centrelines. This module derives a
reference from different data, the NSW Digital Cadastral Database (DCDB):

1. The property parcel containing each address point is found.
2. Its *frontage* is the part of the parcel boundary shared with a road reserve polygon.
3. Each frontage segment gets its outward normal, the direction from the parcel into the road.
4. Each segment is named after the first NSW road centreline its normal crosses. That is the
   street the edge faces.
5. The reference bearing is the length-weighted mean normal of the frontage onto the
   addressed street.

The reference shares only the G-NAF address point and street name with the method. Its
geometry (parcel edges, road reserves and NSW centrelines) is independent of OpenStreetMap, so
it can expose errors in either the street match or the bearing.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from property_orientation.geometry import true_azimuth
from property_orientation.streets import normalise_street_name, strip_suffix


@dataclass(frozen=True)
class ReferenceParams:
    """Parameters of the cadastral reference."""

    tolerance_m: float = 0.5
    """Parcel edges within this distance of a road reserve count as frontage."""
    ray_length_m: float = 40.0
    """How far each frontage normal is traced to find the centreline it faces."""
    min_frontage_m: float = 3.0
    """Shortest frontage onto the addressed street accepted as a reference."""
    min_strength: float = 0.5
    """Smallest acceptable |mean normal| (1 = straight frontage, 0 = normals cancel out)."""


def unique_parcels(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop exact-duplicate polygons, such as strata units stacked on one building parcel."""
    duplicated = parcels.geometry.to_wkb().duplicated()
    return parcels.loc[~duplicated.to_numpy()].reset_index(drop=True)


def parcel_of_points(points: np.ndarray, parcels: np.ndarray) -> np.ndarray:
    """Index of the smallest parcel containing each point, or ``-1`` if there is none."""
    i, k = shapely.STRtree(parcels).query(points, predicate="within")
    order = pd.DataFrame({"i": i, "k": k, "area": shapely.area(parcels[k])})
    best = order.sort_values(["i", "area"]).drop_duplicates("i")
    out = np.full(len(points), -1)
    out[best["i"].to_numpy()] = best["k"].to_numpy()
    return out


def coregistration(points: np.ndarray, parcels: np.ndarray) -> dict[str, float]:
    """How closely address points sit on the parcels that contain them.

    Use property-centroid geocodes for ``points``. A datum mismatch between the two layers
    shows up as a systematic median offset: GDA94 against GDA2020 is about 1.5 m in Sydney.
    """
    parcel = parcel_of_points(points, parcels)
    inside = parcel >= 0
    centroid = shapely.centroid(parcels[parcel[inside]])
    dx = shapely.get_x(points[inside]) - shapely.get_x(centroid)
    dy = shapely.get_y(points[inside]) - shapely.get_y(centroid)

    def r(value: float, digits: int = 2) -> float:
        return round(float(value), digits) + 0.0  # + 0.0 turns -0.0 into 0.0

    return {
        "points": len(points),
        "share_inside_a_parcel": r(inside.mean(), 4),
        "median_distance_to_parcel_centroid_m": r(np.median(np.hypot(dx, dy))),
        "median_offset_east_m": r(np.median(dx)),
        "median_offset_north_m": r(np.median(dy)),
    }


def frontage_segments(
    parcels: np.ndarray, road_reserves: np.ndarray, tolerance_m: float = 0.5
) -> pd.DataFrame:
    """Parcel-boundary segments that lie along a road reserve.

    Parameters
    ----------
    parcels
        Parcel polygons in a projected CRS.
    road_reserves
        Road reserve polygons in the same CRS.

    Returns
    -------
    DataFrame with one row per straight segment: ``parcel`` (index into ``parcels``),
    ``length``, midpoint ``mx``/``my``, and outward unit normal ``nx``/``ny``.
    """
    buffered = shapely.buffer(road_reserves, tolerance_m)
    p, r = shapely.STRtree(buffered).query(parcels, predicate="intersects")
    pieces = shapely.intersection(shapely.boundary(parcels[p]), buffered[r])

    parts, part_of = shapely.get_parts(pieces, return_index=True)
    is_line = np.isin(shapely.get_type_id(parts), (1, 2))  # LineString, LinearRing
    parts, part_parcel = parts[is_line], p[part_of[is_line]]

    xy, line_of = shapely.get_coordinates(parts, return_index=True)
    same_line = line_of[:-1] == line_of[1:]
    start, end = xy[:-1][same_line], xy[1:][same_line]
    parcel = part_parcel[line_of[:-1][same_line]]
    dx, dy = (end - start).T
    length = np.hypot(dx, dy)
    keep = length > 1e-3
    start, end, parcel, dx, dy, length = (
        start[keep], end[keep], parcel[keep], dx[keep], dy[keep], length[keep]
    )  # fmt: skip
    mid = (start + end) / 2.0
    nx, ny = dy / length, -dx / length  # right-hand normal
    probe = shapely.points(mid[:, 0] + 0.25 * nx, mid[:, 1] + 0.25 * ny)
    inward = shapely.contains(parcels[parcel], probe)
    nx, ny = np.where(inward, -nx, nx), np.where(inward, -ny, ny)
    return pd.DataFrame(
        {"parcel": parcel, "length": length, "mx": mid[:, 0], "my": mid[:, 1], "nx": nx, "ny": ny}
    )


def name_segments(
    segments: pd.DataFrame,
    centrelines: np.ndarray,
    centreline_keys: np.ndarray,
    ray_length_m: float = 40.0,
) -> np.ndarray:
    """Name each frontage segment after the first centreline its outward normal crosses.

    Segments whose ray crosses no centreline within ``ray_length_m``, or crosses an unnamed
    one first, get ``None``.
    """
    origin = segments[["mx", "my"]].to_numpy()
    tip = origin + ray_length_m * segments[["nx", "ny"]].to_numpy()
    rays = shapely.linestrings(np.stack([origin, tip], axis=1))
    s, c = shapely.STRtree(centrelines).query(rays, predicate="intersects")
    hit = shapely.intersection(rays[s], centrelines[c])
    hits = pd.DataFrame({"s": s, "c": c, "d": shapely.distance(shapely.points(origin[s]), hit)})
    first = hits.sort_values(["s", "d"]).drop_duplicates("s")
    keys = np.full(len(segments), None, dtype=object)
    keys[first["s"].to_numpy()] = np.asarray(centreline_keys, dtype=object)[first["c"].to_numpy()]
    return keys


def reference_orientation(
    sites: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    road_reserves: gpd.GeoDataFrame,
    centrelines: gpd.GeoDataFrame,
    *,
    name_column: str = "roadnamelabel",
    params: ReferenceParams | None = None,
) -> pd.DataFrame:
    """Reference bearing for each site from cadastral frontage.

    ``sites`` comes from :func:`property_orientation.orientation.compute_orientation` and
    must have ``street_key`` and a projected CRS. The other layers are reprojected to match.

    Returns a frame aligned to ``sites`` with ``ref_bearing_deg``, ``ref_frontage_m``,
    ``ref_strength``, ``ref_parcel`` and ``ref_status``. ``ref_status`` is one of ``ok``,
    ``no_parcel``, ``no_street_frontage``, ``short_frontage`` or ``ambiguous_frontage``.
    """
    params = params or ReferenceParams()
    crs = sites.crs
    parcel_geoms = unique_parcels(parcels.to_crs(crs)).geometry.to_numpy()
    reserves = road_reserves.to_crs(crs).geometry.to_numpy()
    lines = centrelines.to_crs(crs)

    parcel = parcel_of_points(sites.geometry.to_numpy(), parcel_geoms)
    used = np.unique(parcel[parcel >= 0])
    segments = frontage_segments(parcel_geoms[used], reserves, params.tolerance_m)
    segments["parcel"] = used[segments["parcel"].to_numpy()]
    segments["street"] = name_segments(
        segments,
        lines.geometry.to_numpy(),
        lines[name_column].map(normalise_street_name).to_numpy(),
        params.ray_length_m,
    )
    segments["street_nosuffix"] = segments["street"].map(strip_suffix)

    site_table = pd.DataFrame(
        {
            "site": np.arange(len(sites)),
            "parcel": parcel,
            "street_key": sites["street_key"].to_numpy(),
        }
    )
    site_table["site_street_nosuffix"] = site_table["street_key"].map(strip_suffix)
    joined = site_table.merge(segments, on="parcel")
    onto_street = joined[
        joined["street"].notna()
        & (
            (joined["street"] == joined["street_key"])
            | (joined["street_nosuffix"] == joined["site_street_nosuffix"])
        )
    ]

    weighted = onto_street.assign(
        wx=onto_street["nx"] * onto_street["length"],
        wy=onto_street["ny"] * onto_street["length"],
        wmx=onto_street["mx"] * onto_street["length"],
        wmy=onto_street["my"] * onto_street["length"],
    )
    sums = weighted.groupby("site")[["wx", "wy", "wmx", "wmy", "length"]].sum()
    resultant = np.hypot(sums["wx"], sums["wy"])
    strength = resultant / sums["length"]
    anchor_x, anchor_y = sums["wmx"] / sums["length"], sums["wmy"] / sums["length"]
    bearing = true_azimuth(
        anchor_x,
        anchor_y,
        anchor_x + 10.0 * sums["wx"] / resultant,
        anchor_y + 10.0 * sums["wy"] / resultant,
        crs,
    )

    out = pd.DataFrame(
        {
            "ref_bearing_deg": np.nan,
            "ref_frontage_m": np.nan,
            "ref_strength": np.nan,
            "ref_parcel": parcel,
            "ref_status": np.where(parcel >= 0, "no_street_frontage", "no_parcel"),
        },
        index=pd.RangeIndex(len(sites)),
    )
    idx = sums.index.to_numpy()
    out.loc[idx, "ref_bearing_deg"] = bearing
    out.loc[idx, "ref_frontage_m"] = sums["length"].to_numpy()
    out.loc[idx, "ref_strength"] = strength.to_numpy()
    status = np.select(
        [
            out["ref_frontage_m"] < params.min_frontage_m,
            out["ref_strength"] < params.min_strength,
            out["ref_frontage_m"].notna(),
        ],
        ["short_frontage", "ambiguous_frontage", "ok"],
        default=out["ref_status"].to_numpy(),
    )
    out["ref_status"] = status
    out.loc[out["ref_status"] != "ok", "ref_bearing_deg"] = np.nan
    out.index = sites.index
    return out
