# Data

Nothing in this folder is committed except this file. `make data` (or
`property-orientation fetch`) downloads the inputs, and `make run` writes the intermediates.

```
data/
├── raw/<area>/                     downloaded inputs (GeoParquet, EPSG:4326)
│   ├── boundary.parquet            local government area (Geoscape Administrative Boundaries)
│   ├── localities.parquet          suburbs intersecting the area
│   ├── addresses.parquet           G-NAF principal addresses inside the area
│   ├── osm_highways.parquet        every OpenStreetMap highway=* way (area + 500 m)
│   ├── nsw_property.parquet        NSW property parcels (area + 500 m)
│   ├── nsw_road_reserve.parquet    NSW road reserve polygons
│   ├── nsw_road_centreline.parquet NSW road centrelines (about 94% named)
│   └── manifest.json               sources, releases, retrieval dates and feature counts
└── processed/<area>/
    └── sites.parquet               one row per site with estimate, baseline and reference
```

| Source | Access | Licence |
|---|---|---|
| G-NAF and Administrative Boundaries (Geoscape, release 2026-08) | public GeoParquet export of [gnaf-loader](https://github.com/minus34/gnaf-loader), queried over HTTPS with DuckDB; for addresses, only the part files whose bounding box overlaps the area are read | G-NAF: Open G-NAF EULA; boundaries: CC BY 4.0 |
| OpenStreetMap | [Overpass API](https://overpass-api.de), queried in tiles | ODbL 1.0 |
| NSW Digital Cadastral Database | [NSW Spatial Services](https://portal.spatial.nsw.gov.au) ArcGIS REST, paged 2,000 features per request | CC BY 4.0 |

The cadastre layers are downloaded only for NSW areas, because they are used only for
validation. Elsewhere in Australia the method runs on G-NAF and OpenStreetMap alone.
