"""Result tables: one row per address, a GeoPackage for GIS use, and suburb summaries."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from property_orientation.geometry import COMPASS_8

ADDRESS_KIND = {"P": "complex", "S": "unit"}
"""G-NAF ``primary_secondary`` -> kind. Addresses with neither code are ``standalone``."""

NORTHERLY = ("NW", "N", "NE")

ADDRESS_TABLE_COLUMNS = (
    "gnaf_pid", "address", "locality", "postcode", "longitude", "latitude", "address_kind",
    "orientation", "bearing_deg", "street", "distance_to_street_m", "match", "confidence",
    "geocode_type", "site_id",
)  # fmt: skip


def rounded_bearing(bearing: pd.Series) -> pd.Series:
    """Bearings rounded to 0.1° and kept in [0, 360): 359.96° is published as 0.0°."""
    return np.mod(bearing.round(1), 360.0)


def address_table(
    addresses: gpd.GeoDataFrame, sites: pd.DataFrame, site_of_address: np.ndarray
) -> pd.DataFrame:
    """One row per G-NAF address with its site's orientation."""
    site = sites.iloc[site_of_address].reset_index(drop=True)
    a = addresses.reset_index(drop=True)
    label = a["address"].str.cat(
        a["locality_name"] + " " + a["state"] + " " + a["postcode"].astype(str), sep=", "
    )
    table = pd.DataFrame(
        {
            "gnaf_pid": a["gnaf_pid"],
            "address": label,
            "locality": a["locality_name"],
            "postcode": a["postcode"],
            "longitude": a.geometry.x.round(7),
            "latitude": a.geometry.y.round(7),
            "address_kind": a["primary_secondary"].map(ADDRESS_KIND).fillna("standalone"),
            "orientation": site["orientation"],
            "bearing_deg": rounded_bearing(site["bearing_deg"]),
            "street": site["matched_street"],
            "distance_to_street_m": site["distance_m"].round(1),
            "match": site["match"],
            "confidence": site["confidence"],
            "geocode_type": a["geocode_type"],
            "site_id": site_of_address,
        }
    )
    return table[list(ADDRESS_TABLE_COLUMNS)]


def facing_lines(sites: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Segments from each site to the point on its street that it faces (EPSG:4326)."""
    ok = sites["street_x"].notna() & sites["bearing_deg"].notna()
    s = sites.loc[ok]
    lines = shapely.linestrings(
        np.stack(
            [
                np.stack([s["x"], s["y"]], axis=1),
                np.stack([s["street_x"], s["street_y"]], axis=1),
            ],
            axis=1,
        )
    )
    columns = ["street_key", "matched_street", "orientation", "bearing_deg", "distance_m",
               "confidence", "n_addresses"]  # fmt: skip
    frame = s[columns].rename_axis("site_id").reset_index()
    frame["bearing_deg"] = rounded_bearing(frame["bearing_deg"])
    frame["distance_m"] = frame["distance_m"].round(1)
    return gpd.GeoDataFrame(frame, geometry=lines, crs=sites.crs).to_crs(4326)


def orientation_shares(table: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Count and share of each compass direction, optionally per group."""
    rows = table[table["orientation"].notna()]
    group = rows[by] if by else pd.Series("all", index=rows.index, name="scope")
    counts = pd.crosstab(group, rows["orientation"]).reindex(columns=list(COMPASS_8), fill_value=0)
    counts.columns.name = None
    total = counts.sum(axis=1)
    shares = counts.div(total, axis=0).round(4).add_prefix("share_")
    out = counts.add_prefix("n_").join(shares)
    out.insert(0, "with_orientation", total)
    out["share_northerly"] = counts[list(NORTHERLY)].sum(axis=1).div(total).round(4)
    return out.reset_index() if by else out.reset_index(drop=True)


def suburb_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Orientation mix of standalone addresses (mostly detached houses) per suburb."""
    standalone = table[table["address_kind"].eq("standalone")]
    counts = table.groupby("locality").agg(
        addresses=("gnaf_pid", "size"),
        standalone=("address_kind", lambda s: int(s.eq("standalone").sum())),
        high_confidence=("confidence", lambda s: round(float(s.eq("high").mean()), 4)),
    )
    return counts.join(
        orientation_shares(standalone, by="locality").set_index("locality")
    ).reset_index()


GEOPACKAGE_NOTICE = (
    "Derived data. Incorporates or developed using G-NAF © Geoscape Australia, licensed by the "
    "Commonwealth of Australia under the Open G-NAF End User Licence Agreement; Administrative "
    "Boundaries © Geoscape Australia (CC BY 4.0); © OpenStreetMap contributors (ODbL 1.0, "
    "https://opendatacommons.org/licenses/odbl/1-0/). See results/README.md."
)
"""Written into each GeoPackage layer's description, so the licences travel with the file."""


def write_geopackage(
    path: Path, layers: dict[str, gpd.GeoDataFrame], *, timestamp: str | None = None
) -> None:
    """Write several layers to one GeoPackage, replacing any existing file.

    ``timestamp`` (ISO 8601, for example ``2026-10-09T00:00:00.000Z``) fixes the GeoPackage's
    last-change dates, so re-running on the same inputs gives a byte-identical file. Each
    layer's description carries :data:`GEOPACKAGE_NOTICE`.
    """
    import pyogrio

    path.unlink(missing_ok=True)
    if timestamp:
        pyogrio.set_gdal_config_options({"OGR_CURRENT_DATE": timestamp})
    try:
        for name, frame in layers.items():
            frame.to_file(
                path, layer=name, driver="GPKG", layer_options={"DESCRIPTION": GEOPACKAGE_NOTICE}
            )
    finally:
        if timestamp:
            pyogrio.set_gdal_config_options({"OGR_CURRENT_DATE": None})
