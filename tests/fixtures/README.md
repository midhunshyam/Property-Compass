# Test fixtures

Two small extracts of the real Willoughby inputs let the integration tests in
[`test_pipeline.py`](../test_pipeline.py) run the whole pipeline offline. Each one is a
350 m × 350 m window (GDA2020 / MGA zone 56, EPSG:7856) holding every input layer, with the
same file names and columns as `data/raw/willoughby/`, so `property-orientation run` works
on it unchanged.

| Fixture | South-west corner (EPSG:7856) | Why this window | Addresses | OSM ways | Parcels |
|---|---|---|---:|---:|---:|
| `castlecrag/` | 335250, 6258325 | Walter Burley Griffin's 1920s estate: streets follow the contours and public walkways run between the houses, so the nearest-way baseline often faces a path | 116 | 96 | 129 |
| `northbridge/` | 333950, 6257575 | Apartments on Eastern Valley Way (unit and building-centroid addresses), with the houses, cul-de-sacs and cycleway behind them | 227 | 345 | 227 |

Together the two folders and `make_fixture.py` take about 340 KB. The test suite fails if
`tests/fixtures/` grows past 600 KB.

## Provenance

[`make_fixture.py`](make_fixture.py) cuts the fixtures out of the full downloads in
`data/raw/willoughby/`. To rebuild them, from the repository root:

```bash
make data                           # or: property-orientation fetch
.venv/bin/python tests/fixtures/make_fixture.py   # after `make setup` and `make data`
```

The script is deterministic: rows keep their order in the source files. It selects whole
features, so the results for the addresses inside a window are the same as in the full run:

| File | Content |
|---|---|
| `addresses.parquet` | G-NAF principal addresses inside the window |
| `osm_highways.parquet` | whole OpenStreetMap `highway=*` ways within 250 m of the window (the street search radius) |
| `nsw_property.parquet` | whole NSW property parcels that intersect the window |
| `nsw_road_reserve.parquet` | whole road reserve polygons within 1 m of those parcels |
| `nsw_road_centreline.parquet` | whole road centrelines within 45 m of those parcels (frontage rays are 40 m long) |
| `localities.parquet` | suburb polygons, clipped to the window plus 250 m |
| `boundary.parquet` | the Willoughby LGA polygon, clipped to the window plus 250 m |
| `manifest.json` | the source manifest of the full download, with this window (`fixture`) and its feature counts |

All layers are stored in EPSG:4326, as downloaded. The current fixtures come from one
snapshot, recorded in each `manifest.json`:

| Source | Snapshot |
|---|---|
| G-NAF and Geoscape Administrative Boundaries | release 202608 (August 2026), gnaf-loader GeoParquet export |
| OpenStreetMap | Overpass data timestamp 2026-10-09T05:44:36Z |
| NSW Digital Cadastral Database | retrieved 2026-10-09 from NSW Spatial Services |

Two tests in `test_pipeline.py` check this provenance. Each one skips when its data is
missing or comes from a different download than the fixtures:

* `test_fixture_is_cut_from_the_raw_data_by_make_fixture` re-cuts each window from
  `data/raw/willoughby/` and compares every layer with the stored file. `data/raw/` is not
  committed, so this test runs only after `make data`.
* `test_committed_results_match_a_run_on_the_fixture` runs the pipeline on each fixture and
  compares its address table with the matching rows of the committed
  `results/willoughby/property_orientation.csv`.

## Licences

The fixtures are extracts of open data and keep the licences of their sources. The MIT
licence of this repository covers `make_fixture.py` and the tests, not the data.

| Files | Source and attribution | Licence |
|---|---|---|
| `addresses.parquet` | G-NAF © Geoscape Australia, licensed by the Commonwealth of Australia under the Open G-NAF End User Licence Agreement | [Open G-NAF End User Licence Agreement](https://data.gov.au/dataset/geocoded-national-address-file-g-naf) |
| `localities.parquet`, `boundary.parquet` | Administrative Boundaries © Geoscape Australia | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `osm_highways.parquet` | © OpenStreetMap contributors. This extract of the OpenStreetMap database is available under the same licence. | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) ([copyright](https://www.openstreetmap.org/copyright)) |
| `nsw_property.parquet`, `nsw_road_reserve.parquet`, `nsw_road_centreline.parquet` | © Spatial Services (DCS), NSW Government | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Each `manifest.json` repeats these attributions in `sources.*.licence`.
