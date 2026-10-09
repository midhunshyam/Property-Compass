"""Street-frontage orientation of address points.

Method
------
1. Addresses that share a location, a street and a geocode type (for example, the units in
   one building) are collapsed into a *site*. Each site is computed once.
2. Each site is matched to the OpenStreetMap street it is addressed to, using normalised
   names, in three tiers: exact name, name with a suffix such as ``EAST`` dropped from one
   side, and the bare street name without a road type. Ways a property can front
   (:data:`FRONTAGE_HIGHWAYS`) are preferred; a footway is used only when the street exists
   in OSM solely as a walkway.
3. The site faces the nearest point on that street. Its orientation is the geodesic
   azimuth of that vector, binned into eight 45° compass sectors.
4. If no street of the right name lies within the search radius, the nearest street a
   property can front is used instead (no footpaths, cycleways, steps, motorways or
   unnamed driveways), and the result is marked low confidence.

:func:`nearest_road_baseline` reproduces the earlier approach (nearest OSM way of any
class) so that the two can be compared against the cadastral reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from property_orientation.geometry import compass8, true_azimuth
from property_orientation.streets import gnaf_street_key, normalise_street_name, strip_suffix

FRONTAGE_HIGHWAYS = frozenset(
    {
        "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
        "living_street", "pedestrian", "service", "road",
    }
)  # fmt: skip
"""OSM ``highway`` values a property can front. Unnamed ``service`` ways are excluded
separately because they are mostly driveways and car-park aisles."""

UNMATCHABLE_HIGHWAYS = frozenset(
    {"motorway", "motorway_link", "construction", "proposed", "abandoned", "platform", "elevator"}
)
"""OSM ``highway`` values never used as a property's street, even when the name matches."""

NON_PROPERTY_GEOCODES = frozenset({"STREET LOCALITY", "LOCALITY"})
"""G-NAF geocode types that place the point on a street or suburb centroid, not on the
property. No orientation is reported for them."""

MATCH_LABELS = {
    1: "street_name",
    2: "street_name_without_suffix",
    3: "street_base_name",
    0: "nearest_street",
    -1: "none",
}


@dataclass(frozen=True)
class OrientationParams:
    """Tuning parameters. The defaults were calibrated on Willoughby, NSW."""

    search_radius_m: float = 250.0
    """Furthest a site can be from its street (or from the fallback street)."""
    high_confidence_max_m: float = 60.0
    """Largest site-to-street distance that can still be ``high`` confidence."""
    medium_confidence_max_m: float = 120.0
    """Largest site-to-street distance that can still be ``medium`` confidence."""
    min_distance_m: float = 0.5
    """Sites closer than this to the street have no meaningful bearing."""
    baseline_radius_m: float = 500.0
    """Search radius of the nearest-road baseline (the original method used 500 m)."""


@dataclass
class OrientationResult:
    """Output of :func:`compute_orientation`."""

    sites: gpd.GeoDataFrame
    """One row per site (unique location, street and geocode type), in the projected CRS."""
    site_of_address: np.ndarray
    """For each input address (by position), the index of its site in :attr:`sites`."""
    crs: object
    """The projected CRS used for distances."""


def _take(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """``values[index]`` where ``index >= 0``, else ``None``; safe when ``values`` is empty."""
    out = np.full(len(index), None, dtype=object)
    ok = index >= 0
    out[ok] = values[index[ok]]
    return out


def address_sites(addresses: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Group addresses that share a location (to 1 cm), a street and a geocode type into sites.

    The geocode type is part of the key because it feeds the confidence rule: units geocoded
    to a building centroid and to a property centroid at the same point stay separate sites.
    ``addresses`` must be in a projected CRS and carry G-NAF's ``street_name``,
    ``street_type``, ``street_suffix`` and ``geocode_type`` columns.
    """
    if addresses.crs is None or not addresses.crs.is_projected:
        raise ValueError("addresses must be in a projected CRS (metres)")
    frame = pd.DataFrame(
        {
            "x": addresses.geometry.x.round(2).to_numpy(),
            "y": addresses.geometry.y.round(2).to_numpy(),
            "street_key": [
                gnaf_street_key(n, t, s)
                for n, t, s in zip(
                    addresses["street_name"],
                    addresses["street_type"],
                    addresses["street_suffix"],
                    strict=True,
                )
            ],
            "street_base": addresses["street_name"].map(normalise_street_name).to_numpy(),
            "geocode_type": addresses["geocode_type"].to_numpy(),
            "locality": (
                addresses["locality_name"].to_numpy() if "locality_name" in addresses else None
            ),
        }
    )
    key = ["x", "y", "street_key", "geocode_type"]
    site_id = frame.groupby(key, dropna=False, sort=False).ngroup().to_numpy()
    sites = (
        frame.assign(site_id=site_id)
        .groupby("site_id", sort=True)
        .agg(
            x=("x", "first"),
            y=("y", "first"),
            street_key=("street_key", "first"),
            street_base=("street_base", "first"),
            geocode_type=("geocode_type", "first"),
            locality=("locality", "first"),
            n_addresses=("x", "size"),
        )
    )
    geometry = gpd.GeoSeries(shapely.points(sites["x"], sites["y"]), crs=addresses.crs)
    return gpd.GeoDataFrame(sites, geometry=geometry.to_numpy(), crs=addresses.crs), site_id


def prepare_roads(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add normalised name keys and a ``frontage`` eligibility flag to OSM highways."""
    out = roads.reset_index(drop=True).copy()
    for col in ("name", "alt_name", "official_name", "old_name", "service", "area"):
        if col not in out:
            out[col] = None
    out["street_key"] = out["name"].map(normalise_street_name)
    out["street_key_nosuffix"] = out["street_key"].map(strip_suffix)
    out["alt_keys"] = [
        frozenset(
            k
            for value in values
            if isinstance(value, str)
            for k in map(normalise_street_name, value.split(";"))
            if k
        )
        for values in zip(out["alt_name"], out["official_name"], out["old_name"], strict=True)
    ]
    out["frontage"] = (
        out["highway"].isin(FRONTAGE_HIGHWAYS)
        & ~(out["highway"].eq("service") & out["name"].isna())
        & ~out["area"].eq("yes")
    )
    return out


def match_streets(
    sites: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, radius_m: float
) -> pd.DataFrame:
    """Match each site to the street it is addressed to.

    ``roads`` must come from :func:`prepare_roads`. Returns one row per site with the
    chosen road (positional index, ``-1`` if none), the match tier (see
    :data:`MATCH_LABELS`), the distance to that road, the distance to the nearest eligible
    street with a *different* name, and whether the match dropped a suffix that G-NAF has.
    """
    n = len(sites)
    points = sites.geometry.to_numpy()
    lines = roads.geometry.to_numpy()
    i, j = shapely.STRtree(lines).query(points, predicate="dwithin", distance=radius_m)
    pairs = pd.DataFrame({"i": i, "j": j, "d": shapely.distance(points[i], lines[j])})

    site_key = sites["street_key"].to_numpy()[i]
    site_key_nosuffix = np.array([strip_suffix(k) for k in site_key], dtype=object)
    site_base = sites["street_base"].to_numpy()[i]
    road_key = roads["street_key"].to_numpy()[j]
    road_alt = roads["alt_keys"].to_numpy()[j]

    has_key = pd.notna(site_key)
    tier1 = has_key & (
        (road_key == site_key)
        | np.fromiter((k in alt for k, alt in zip(site_key, road_alt, strict=True)), bool, len(i))
    )
    road_key_nosuffix = roads["street_key_nosuffix"].to_numpy()[j]
    site_suffixed = has_key & (site_key != site_key_nosuffix)
    road_suffixed = pd.notna(road_key) & (road_key != road_key_nosuffix)
    # Never match two different suffixes (EAST vs WEST): exactly one side may carry one.
    tier2 = (
        has_key
        & ~tier1
        & (road_key_nosuffix == site_key_nosuffix)
        & (site_suffixed != road_suffixed)
    )
    tier3 = pd.notna(site_base) & ~tier1 & ~tier2 & (road_key == site_base)
    pairs["tier"] = np.select([tier1, tier2, tier3], [1, 2, 3], default=0)
    # G-NAF's street carries a suffix that the OSM way lacks: often a separate street missing
    # from OSM (MARKET STREET EAST is not Market Street), so the match is only a guess.
    pairs["suffix_dropped"] = tier2 & site_suffixed
    pairs["frontage"] = roads["frontage"].to_numpy()[j]
    pairs["walkway"] = ~pairs["frontage"]
    pairs["named"] = pd.notna(road_key)
    matchable = ~roads["highway"].isin(UNMATCHABLE_HIGHWAYS).to_numpy()[j]

    named_match = (
        pairs[(pairs["tier"] > 0) & matchable]
        .sort_values(["i", "tier", "walkway", "d"])
        .drop_duplicates("i")
    )
    eligible = pairs[pairs["frontage"]]
    nearest = eligible.sort_values(["i", "d"]).drop_duplicates("i")
    other = (
        eligible[(eligible["tier"] == 0) & eligible["named"]]
        .sort_values(["i", "d"])
        .drop_duplicates("i")
    )

    out = pd.DataFrame(
        {
            "road": -1,
            "tier": -1,
            "distance_m": np.nan,
            "other_street_m": np.nan,
            "suffix_dropped": False,
        },
        index=pd.RangeIndex(n),
    )
    fallback = nearest[~nearest["i"].isin(named_match["i"])]
    for chosen, tier in ((named_match, None), (fallback, 0)):
        idx = chosen["i"].to_numpy()
        out.loc[idx, "road"] = chosen["j"].to_numpy()
        out.loc[idx, "tier"] = chosen["tier"].to_numpy() if tier is None else tier
        out.loc[idx, "distance_m"] = chosen["d"].to_numpy()
    out.loc[named_match["i"].to_numpy(), "suffix_dropped"] = named_match[
        "suffix_dropped"
    ].to_numpy()
    out.loc[other["i"].to_numpy(), "other_street_m"] = other["d"].to_numpy()
    out.index = sites.index
    return out


def facing(
    points: np.ndarray, lines: np.ndarray, crs: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuth from each point to the nearest point on its line, plus that nearest point.

    Returns ``(azimuth_deg, nearest_x, nearest_y)``; entries are NaN where ``lines`` is
    ``None``.
    """
    azimuth = np.full(len(points), np.nan)
    nx = np.full(len(points), np.nan)
    ny = np.full(len(points), np.nan)
    ok = pd.notna(lines)
    if ok.any():
        segment = shapely.shortest_line(points[ok], lines[ok])
        xy = shapely.get_coordinates(segment).reshape(-1, 2, 2)
        azimuth[ok] = true_azimuth(xy[:, 0, 0], xy[:, 0, 1], xy[:, 1, 0], xy[:, 1, 1], crs)
        nx[ok], ny[ok] = xy[:, 1, 0], xy[:, 1, 1]
    return azimuth, nx, ny


def nearest_road_baseline(
    sites: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, radius_m: float = 500.0
) -> pd.DataFrame:
    """The original approach: face the nearest OSM highway of *any* class.

    Footpaths, driveways and cycleways are all eligible, which is why this approach often
    points at the footpath beside a property rather than at the street it is addressed to.
    """
    points = sites.geometry.to_numpy()
    lines = roads.geometry.to_numpy()
    i, j = shapely.STRtree(lines).query_nearest(
        points, max_distance=radius_m, return_distance=False, all_matches=False
    )
    road = np.full(len(sites), -1)
    road[i] = j
    chosen = _take(lines, road)
    azimuth, _, _ = facing(points, chosen, sites.crs)
    distance = shapely.distance(points, chosen)
    return pd.DataFrame(
        {
            "baseline_bearing_deg": azimuth,
            "baseline_orientation": compass8(azimuth),
            "baseline_distance_m": distance,
            "baseline_highway": _take(roads["highway"].to_numpy(), road),
        },
        index=sites.index,
    )


def assign_confidence(
    tier: np.ndarray,
    distance_m: np.ndarray,
    other_street_m: np.ndarray,
    geocode_type: np.ndarray,
    params: OrientationParams,
    suffix_dropped: np.ndarray | None = None,
) -> np.ndarray:
    """Confidence class per site: ``high``, ``medium``, ``low`` or ``none``.

    The thresholds come from agreement with the cadastral reference in Willoughby (see
    ``results/willoughby/validation``). Error rises with distance to the street, for
    building-centroid geocodes (large multi-building sites), and when a differently
    named street is closer than the addressed one (corner and rear-lane sites). A match
    that drops a suffix G-NAF has (``MARKET STREET EAST`` to OSM ``Market Street``) is at
    most ``low``: the suffixed street is often a separate street missing from OSM.
    """
    tier = np.asarray(tier)
    distance_m = np.asarray(distance_m, float)
    geocode_type = np.asarray(geocode_type, dtype=object)
    no_bearing = (
        (tier < 0)
        | np.isin(geocode_type, list(NON_PROPERTY_GEOCODES))
        | ~(distance_m >= params.min_distance_m)
    )
    dropped = np.zeros(len(tier), bool) if suffix_dropped is None else np.asarray(suffix_dropped)
    named = (tier >= 1) & ~dropped
    other_closer = np.nan_to_num(np.asarray(other_street_m, float), nan=np.inf) < distance_m
    high = (
        named
        & (distance_m <= params.high_confidence_max_m)
        & (geocode_type != "BUILDING CENTROID")
        & ~other_closer
    )
    medium = named & (distance_m <= params.medium_confidence_max_m)
    return np.select([no_bearing, high, medium], ["none", "high", "medium"], default="low")


def compute_orientation(
    addresses: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    *,
    crs: object,
    params: OrientationParams | None = None,
    baseline: bool = True,
) -> OrientationResult:
    """Orientation of every address.

    Parameters
    ----------
    addresses
        G-NAF address points with ``street_name``, ``street_type``, ``street_suffix`` and
        ``geocode_type`` columns, in any CRS.
    roads
        OpenStreetMap highways (LineStrings) with ``highway`` and ``name`` columns.
    crs
        Projected CRS for distances, for example ``mga2020_epsg(longitude)``.
    params
        Tuning parameters; defaults to :class:`OrientationParams`.
    baseline
        Also compute the nearest-road baseline, for comparison.
    """
    params = params or OrientationParams()
    sites, site_of_address = address_sites(addresses.to_crs(crs))
    prepared = prepare_roads(roads.to_crs(crs))

    match = match_streets(sites, prepared, params.search_radius_m)
    road = match["road"].to_numpy()
    lines = _take(prepared.geometry.to_numpy(), road)
    azimuth, street_x, street_y = facing(sites.geometry.to_numpy(), lines, crs)

    confidence = assign_confidence(
        match["tier"].to_numpy(),
        match["distance_m"].to_numpy(),
        match["other_street_m"].to_numpy(),
        sites["geocode_type"].to_numpy(),
        params,
        suffix_dropped=match["suffix_dropped"].to_numpy(bool),
    )
    azimuth = np.where(confidence == "none", np.nan, azimuth)

    sites = sites.assign(
        # The label comes from the bearing as published (0.1°), so the two never disagree.
        orientation=compass8(np.mod(np.round(azimuth, 1), 360.0)),
        bearing_deg=azimuth,
        matched_street=_take(prepared["name"].to_numpy(), road),
        match=[MATCH_LABELS[t] for t in match["tier"]],
        distance_m=match["distance_m"].to_numpy(),
        other_street_m=match["other_street_m"].to_numpy(),
        confidence=confidence,
        street_x=street_x,
        street_y=street_y,
    )
    if baseline:
        sites = sites.join(nearest_road_baseline(sites, prepared, params.baseline_radius_m))
    return OrientationResult(sites=sites, site_of_address=site_of_address, crs=crs)
