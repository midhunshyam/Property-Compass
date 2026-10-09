"""Cut the small real-data fixtures in ``tests/fixtures/`` out of the full Willoughby inputs.

Run from the repository root after ``property-orientation fetch``::

    python tests/fixtures/make_fixture.py

Each fixture is a 350 m square window (EPSG:7856) holding every input layer of the
pipeline, with the same file names and columns as ``data/raw/willoughby``, so
``property-orientation run`` works on it unchanged. Layers are selected so that the
results for the addresses inside the window are the same as in the full run:

=======================  ===============================================================
``addresses``            G-NAF addresses inside the window
``osm_highways``         whole OSM ways within 250 m of the window (the street search radius)
``nsw_property``         whole parcels that intersect the window
``nsw_road_reserve``     whole road reserves within 1 m of those parcels
``nsw_road_centreline``  whole centrelines within 45 m of those parcels (frontage rays are 40 m)
``localities``           suburb polygons, clipped to the window plus 250 m
``boundary``             the LGA polygon, clipped to the window plus 250 m
``manifest.json``        the source manifest, with this fixture's window and feature counts
=======================  ===============================================================

Rows keep their order in the source files, so the output is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import shapely

from property_orientation.config import Paths, StudyArea

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CRS = 7856  # GDA2020 / MGA zone 56
SIZE_M = 350.0
MARGIN_M = 250.0

WINDOWS: dict[str, tuple[float, float]] = {
    # Walter Burley Griffin's 1920s estate: streets follow the contours and public
    # walkways run between the houses, so the nearest-way baseline often faces a path.
    "castlecrag": (335250.0, 6258325.0),
    # Apartments on Eastern Valley Way (unit and building-centroid addresses), with the
    # houses, cul-de-sacs and cycleway behind them.
    "northbridge": (333950.0, 6257575.0),
}
"""South-west corner of each window, in EPSG:7856 metres."""

LAYERS = (
    "addresses",
    "osm_highways",
    "nsw_property",
    "nsw_road_reserve",
    "nsw_road_centreline",
    "localities",
    "boundary",
)


def _read(source: Paths, layer: str) -> tuple[gpd.GeoDataFrame, gpd.GeoSeries]:
    """A raw layer as stored (EPSG:4326) and its geometry in the projected CRS."""
    frame = gpd.read_parquet(source.raw_file(layer))
    return frame, frame.geometry.to_crs(CRS)


def _clip(frame: gpd.GeoDataFrame, projected: gpd.GeoSeries, extent: shapely.Polygon):
    """Polygons clipped to ``extent``, back in the frame's own CRS."""
    clipped = projected.intersection(extent)
    keep = ~clipped.is_empty.to_numpy()
    return frame.loc[keep].set_geometry(clipped.loc[keep].to_crs(frame.crs).to_numpy())


def subset(source: Paths, corner: tuple[float, float]) -> dict[str, gpd.GeoDataFrame]:
    """Every input layer cut down to the window with south-west corner ``corner``."""
    x, y = corner
    window = shapely.box(x, y, x + SIZE_M, y + SIZE_M)
    extent = window.buffer(MARGIN_M, join_style="mitre")

    addresses, points = _read(source, "addresses")
    roads, lines = _read(source, "osm_highways")
    parcels, parcel_geoms = _read(source, "nsw_property")
    reserves, reserve_geoms = _read(source, "nsw_road_reserve")
    centrelines, centreline_geoms = _read(source, "nsw_road_centreline")

    in_window = parcel_geoms.intersects(window).to_numpy()
    lots = parcel_geoms.loc[in_window].union_all()
    layers = {
        "addresses": addresses.loc[points.within(window).to_numpy()],
        "osm_highways": roads.loc[lines.intersects(extent).to_numpy()],
        "nsw_property": parcels.loc[in_window],
        "nsw_road_reserve": reserves.loc[reserve_geoms.intersects(lots.buffer(1.0)).to_numpy()],
        "nsw_road_centreline": centrelines.loc[
            centreline_geoms.intersects(lots.buffer(45.0)).to_numpy()
        ],
    }
    for layer in ("localities", "boundary"):
        layers[layer] = _clip(*_read(source, layer), extent)
    return {name: frame.reset_index(drop=True) for name, frame in layers.items()}


def write(name: str, corner: tuple[float, float], source: Paths) -> Path:
    """Write one fixture directory and its manifest; return the directory."""
    layers = subset(source, corner)
    out = HERE / name
    out.mkdir(exist_ok=True)
    for layer in LAYERS:
        layers[layer].to_parquet(out / f"{layer}.parquet", compression="zstd")

    x, y = corner
    extent = gpd.GeoSeries([shapely.box(x, y, x + SIZE_M, y + SIZE_M)], crs=CRS)
    bounds = extent.buffer(MARGIN_M, join_style="mitre").to_crs(4326).total_bounds
    manifest = json.loads((source.raw / "manifest.json").read_text())
    manifest["fetch_bounds_epsg4326"] = [round(float(v), 7) for v in bounds]
    manifest["counts"] = {layer: len(layers[layer]) for layer in LAYERS if layer != "boundary"}
    manifest["fixture"] = {
        "name": name,
        "window_epsg7856": [x, y, x + SIZE_M, y + SIZE_M],
        "subset_of": f"data/raw/{source.area.slug}",
        "script": "tests/fixtures/make_fixture.py",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    source = Paths(root=ROOT, area=StudyArea())
    total = 0
    for name, corner in WINDOWS.items():
        out = write(name, corner, source)
        sizes = {path.name: path.stat().st_size for path in sorted(out.iterdir())}
        total += sum(sizes.values())
        counts = json.loads((out / "manifest.json").read_text())["counts"]
        print(f"{name}: {sum(sizes.values()) / 1024:.0f} KB, {counts}")
    print(f"total: {total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
