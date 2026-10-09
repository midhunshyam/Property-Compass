"""The two pipeline stages: ``fetch`` (download open data) and ``run`` (compute and validate)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import geopandas as gpd
import pandas as pd

from property_orientation import outputs, validation
from property_orientation.config import Paths, StudyArea
from property_orientation.geometry import mga2020_epsg
from property_orientation.orientation import OrientationParams, compute_orientation
from property_orientation.reference import (
    ReferenceParams,
    coregistration,
    reference_orientation,
    unique_parcels,
)

log = logging.getLogger("property_orientation")

LICENCES = {
    "gnaf": "G-NAF © Geoscape Australia, licensed by the Commonwealth of Australia under the "
    "Open G-NAF End User Licence Agreement",
    "boundaries": "Administrative Boundaries © Geoscape Australia, CC BY 4.0",
    "osm": "© OpenStreetMap contributors, ODbL 1.0",
    "nsw_cadastre": "© Spatial Services (DCS), NSW Government, CC BY 4.0",
}


def _cached(path: Path, download: Callable[[], gpd.GeoDataFrame], force: bool) -> gpd.GeoDataFrame:
    if path.exists() and not force:
        log.info("using cached %s", path.name)
        return gpd.read_parquet(path)
    log.info("downloading %s ...", path.stem)
    frame = download()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    log.info("saved %s (%d features)", path.name, len(frame))
    return frame


def fetch(area: StudyArea, paths: Paths, *, force: bool = False) -> dict:
    """Download every input for ``area`` into ``data/raw/<area>`` and write a manifest."""
    from property_orientation.sources import gnaf, nsw_cadastre, osm

    manifest_path = paths.raw / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    boundary = _cached(
        paths.raw_file("boundary"), lambda: gnaf.fetch_lga(area.lga, area.state), force
    )
    polygon = boundary.geometry.union_all()
    crs = mga2020_epsg(polygon.centroid.x)
    buffered = gpd.GeoSeries([polygon], crs=4326).to_crs(crs).buffer(area.fetch_buffer_m)
    bounds = tuple(float(v) for v in buffered.to_crs(4326).total_bounds)

    localities = _cached(
        paths.raw_file("localities"), lambda: gnaf.fetch_localities(polygon, area.state), force
    )
    addresses = _cached(paths.raw_file("addresses"), lambda: gnaf.fetch_addresses(polygon), force)

    osm_timestamp = previous.get("sources", {}).get("osm", {}).get("timestamp")

    def download_roads() -> gpd.GeoDataFrame:
        nonlocal osm_timestamp
        roads, osm_timestamp = osm.fetch_highways(bounds)
        return roads

    roads = _cached(paths.raw_file("osm_highways"), download_roads, force)

    counts = {
        "localities": len(localities),
        "addresses": len(addresses),
        "osm_highways": len(roads),
    }
    sources = {
        "gnaf": {
            "release": gnaf.RELEASE,
            "url": f"{gnaf.BUCKET_URL}/{gnaf.PREFIX}/",
            "licence": LICENCES["gnaf"],
        },
        "boundaries": {"release": gnaf.RELEASE, "licence": LICENCES["boundaries"]},
        "osm": {
            "endpoint": osm.OVERPASS_ENDPOINTS[0],
            "timestamp": osm_timestamp,
            "licence": LICENCES["osm"],
        },
    }
    if area.state.upper() == "NSW":
        for layer in nsw_cadastre.LAYERS:
            frame = _cached(
                paths.raw_file(f"nsw_{layer}"),
                lambda layer=layer: nsw_cadastre.fetch_layer(layer, bounds),
                force,
            )
            counts[f"nsw_{layer}"] = len(frame)
        retrieved = previous.get("sources", {}).get("nsw_cadastre", {}).get("retrieved")
        sources["nsw_cadastre"] = {
            "service": nsw_cadastre.SERVICE_URL,
            "layers": nsw_cadastre.LAYERS,
            "retrieved": (None if force else retrieved) or _today(),
            "licence": LICENCES["nsw_cadastre"],
        }

    manifest = {
        "area": {"lga": area.lga, "state": area.state, "slug": area.slug},
        "crs": f"EPSG:{crs}",
        "fetch_bounds_epsg4326": bounds,
        "fetch_buffer_m": area.fetch_buffer_m,
        "retrieved": (None if force else previous.get("retrieved")) or _today(),
        "counts": counts,
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run(
    area: StudyArea,
    paths: Paths,
    *,
    params: OrientationParams | None = None,
    reference_params: ReferenceParams | None = None,
) -> dict:
    """Compute orientation for every address, validate against the cadastre, write results."""
    manifest_path = paths.raw / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No downloaded data in {paths.raw}. Run `make data` (or `property-orientation "
            f"--lga '{area.lga}' --state {area.state} fetch`) first."
        )
    manifest = json.loads(manifest_path.read_text())
    addresses = gpd.read_parquet(paths.raw_file("addresses"))
    roads = gpd.read_parquet(paths.raw_file("osm_highways"))
    localities = gpd.read_parquet(paths.raw_file("localities"))
    boundary = gpd.read_parquet(paths.raw_file("boundary"))
    crs = int(manifest["crs"].split(":")[1])

    log.info("computing orientation for %d addresses", len(addresses))
    result = compute_orientation(addresses, roads, crs=crs, params=params)
    sites = result.sites

    cadastre = {
        name: gpd.read_parquet(paths.raw_file(f"nsw_{name}"))
        for name in ("property", "road_reserve", "road_centreline")
        if paths.raw_file(f"nsw_{name}").exists()
    }
    has_cadastre = len(cadastre) == 3
    if has_cadastre:
        log.info("building the cadastral reference")
        reference = reference_orientation(
            sites,
            cadastre["property"],
            cadastre["road_reserve"],
            cadastre["road_centreline"],
            params=reference_params,
        )
        sites = sites.join(reference)

    table = outputs.address_table(addresses, sites, result.site_of_address)

    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)
    sites.rename_axis("site_id").reset_index().to_parquet(paths.processed / "sites.parquet")
    retrieved = manifest.get("retrieved")  # fixes GeoPackage timestamps, so reruns are identical
    timestamp = f"{retrieved}T00:00:00.000Z" if retrieved else None
    standalone = _write_tables(paths, table, sites, localities, boundary, timestamp)

    summary = {
        "area": manifest["area"],
        "generated": _today(),
        "package_version": _package_version(),
        "crs": manifest["crs"],
        "inputs": manifest["counts"],
        "sources": manifest["sources"],
        "addresses": len(table),
        "sites": len(sites),
        "confidence_addresses": table["confidence"].value_counts().to_dict(),
        "confidence_sites": sites["confidence"].value_counts().to_dict(),
        "match_sites": sites["match"].value_counts().to_dict(),
        "orientation_share_standalone": standalone.iloc[0].filter(like="share_").to_dict(),
    }
    if has_cadastre:
        summary["coregistration"] = _coregistration(addresses, cadastre["property"], crs)
        summary["validation"] = _write_validation(paths, sites)
    summary["software"] = _software_versions()
    (paths.results / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n"
    )
    log.info("wrote results to %s", paths.results)
    return summary


def _write_tables(
    paths: Paths,
    table: pd.DataFrame,
    sites: gpd.GeoDataFrame,
    localities: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    timestamp: str | None,
) -> pd.DataFrame:
    """Write the address table, summaries and GeoPackage; return the standalone summary row."""
    table.to_csv(paths.results / "property_orientation.csv", index=False)
    suburbs = outputs.suburb_summary(table)
    suburbs.to_csv(paths.results / "summary_by_suburb.csv", index=False)
    overall = outputs.orientation_shares(table.assign(scope="all addresses"), by="scope")
    standalone_rows = table[table["address_kind"].eq("standalone")]
    standalone = outputs.orientation_shares(
        standalone_rows.assign(scope="standalone addresses"), by="scope"
    )
    pd.concat([overall, standalone]).to_csv(paths.results / "summary_orientation.csv", index=False)

    points = gpd.GeoDataFrame(
        table, geometry=gpd.points_from_xy(table["longitude"], table["latitude"]), crs=4326
    )
    suburb_layer = localities[["locality_name", "postcode", "geometry"]].merge(
        suburbs, left_on="locality_name", right_on="locality", how="inner"
    )
    outputs.write_geopackage(
        paths.results / "property_orientation.gpkg",
        {
            "addresses": points,
            "facing_lines": outputs.facing_lines(sites),
            "suburbs": suburb_layer.drop(columns="locality"),
            "study_area": boundary,
        },
        timestamp=timestamp,
    )
    return standalone


def _write_validation(paths: Paths, sites: pd.DataFrame) -> dict:
    """Write the validation tables and return their headline numbers."""
    paths.validation.mkdir(parents=True, exist_ok=True)
    methods = validation.compare_methods(sites)
    by_confidence = validation.by_group(
        sites, "confidence", order=["high", "medium", "low", "none"]
    )
    by_suburb = validation.by_group(sites, "locality")
    coverage = (
        sites["ref_status"].value_counts().rename_axis("ref_status").reset_index(name="sites")
    )
    methods.to_csv(paths.validation / "method_comparison.csv", index=False)
    by_confidence.to_csv(paths.validation / "by_confidence.csv", index=False)
    by_suburb.to_csv(paths.validation / "by_suburb.csv", index=False)
    validation.confusion_matrix(sites).to_csv(paths.validation / "confusion_matrix.csv")
    coverage.to_csv(paths.validation / "reference_coverage.csv", index=False)
    return {
        "reference_status": dict(zip(coverage["ref_status"], coverage["sites"], strict=True)),
        "methods": methods.to_dict(orient="records"),
        "by_confidence": by_confidence.to_dict(orient="records"),
    }


def _coregistration(addresses: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame, crs: int) -> dict:
    """Offset of standalone property-centroid geocodes from their cadastral parcel centroids."""
    standalone = addresses[
        addresses["geocode_type"].eq("PROPERTY CENTROID") & addresses["primary_secondary"].isna()
    ]
    return coregistration(
        standalone.to_crs(crs).geometry.to_numpy(),
        unique_parcels(parcels.to_crs(crs)).geometry.to_numpy(),
    )


def _software_versions() -> dict[str, str]:
    """Versions of the libraries (and GEOS/PROJ/GDAL) that produced the results."""
    import platform

    import duckdb
    import numpy
    import pyogrio
    import pyproj
    import shapely

    return {
        "python": platform.python_version(),
        "property-orientation": _package_version(),
        "geopandas": gpd.__version__,
        "shapely": shapely.__version__,
        "geos": shapely.geos_version_string,
        "pyproj": pyproj.__version__,
        "proj": pyproj.proj_version_str,
        "pyogrio": pyogrio.__version__,
        "gdal": pyogrio.__gdal_version_string__,
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "duckdb": duckdb.__version__,
    }


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _package_version() -> str:
    try:
        return version("property-orientation")
    except PackageNotFoundError:
        return "unknown"


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")
