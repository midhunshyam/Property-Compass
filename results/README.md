# Results

Outputs for each study area live in `results/<area>/`. The pipeline writes all of them
(`make run`, or `property-orientation run`). Re-running on the same downloaded data reproduces
them byte for byte, apart from the `generated` date in `run_summary.json`. A fresh download picks
up later OpenStreetMap edits, so the numbers can shift a little (`run_summary.json` records the
OSM snapshot and library versions used).

```
results/willoughby/
├── property_orientation.csv     one row per principal G-NAF address: the main deliverable
├── property_orientation.gpkg    the same plus facing lines and suburbs, for QGIS/ArcGIS
├── summary_orientation.csv      direction mix, all addresses vs standalone addresses
├── summary_by_suburb.csv        direction mix of standalone addresses per suburb
├── run_summary.json             provenance (sources, releases, dates, software versions),
│                                headline counts and a G-NAF/cadastre co-registration check
└── validation/                  agreement with the independent cadastral reference
    ├── method_comparison.csv    this method vs the nearest-road baseline
    ├── by_confidence.csv        accuracy per confidence class
    ├── by_suburb.csv            accuracy per suburb
    ├── confusion_matrix.csv     reference direction (rows) × estimated direction (columns)
    └── reference_coverage.csv   how many sites have a usable reference, and why not
```

## `property_orientation.csv`

| Column | Description |
|---|---|
| `gnaf_pid` | G-NAF persistent identifier of the address (for example `GANSW703924475`). |
| `address` | Full address: G-NAF address string, suburb, state and postcode. |
| `locality` | Suburb (G-NAF locality name). |
| `postcode` | Postcode. |
| `longitude`, `latitude` | G-NAF geocode in degrees, EPSG:4326 as published. Standalone property-centroid geocodes sit a median of 0.15 m from the centroid of their NSW cadastral parcel (see `coregistration` below). |
| `address_kind` | `standalone` (an address with no sub-addresses, mostly a house), `complex` (the parent address of a multi-unit site) or `unit` (a unit, flat or shop within a complex). From G-NAF's primary/secondary flag. |
| `orientation` | 8-point compass direction the property faces: `N`, `NE`, `E`, `SE`, `S`, `SW`, `W` or `NW`. Each sector is 45° wide, centred on its direction, and is assigned from the published `bearing_deg`. Empty when `confidence` is `none`. |
| `bearing_deg` | Facing bearing in degrees clockwise from **true** north, rounded to 0.1° and in [0, 360). It is the geodesic azimuth from the address point to the nearest point on the faced street. |
| `street` | The OpenStreetMap street the property faces. |
| `distance_to_street_m` | Distance from the address point to that street's centreline, in metres. |
| `match` | How the street was identified: `street_name` (exact normalised name), `street_name_without_suffix` (a street suffix on one side only, for example G-NAF `JOUBERT STREET` = OSM `Joubert Street North`; when the dropped suffix is G-NAF's, confidence is at most `low`), `street_base_name` (OSM name without a road type), `nearest_street` (no name match within 250 m), or `none`. |
| `confidence` | `high`, `medium`, `low` or `none`; see below. |
| `geocode_type` | G-NAF geocode type (`PROPERTY CENTROID`, `BUILDING CENTROID`, `GAP GEOCODE`, `STREET LOCALITY`, ...). |
| `site_id` | Addresses sharing a location, street and geocode type (for example the units of one building) share a site. Links to the `facing_lines` layer. |

### Confidence classes

| Class | Rule |
|---|---|
| `high` | Street matched by name (not by dropping a suffix G-NAF has), address point ≤ 60 m from it, not a building-centroid geocode, and no differently named street is closer. |
| `medium` | Street matched by name (not by dropping a suffix G-NAF has) and ≤ 120 m away, but not `high`. |
| `low` | Fallback to the nearest street (no name match), a match that dropped a suffix G-NAF has (`MARKET STREET EAST` to OSM `Market Street`), or more than 120 m from the street. |
| `none` | No bearing reported: a street-locality or locality geocode (the point is not on the property), no street within 250 m, or a point within 0.5 m of the street centreline. |

The thresholds were set from the agreement with the cadastral reference: error rises with
distance to the street, for building centroids on large multi-building sites, and at corner
and rear-lane sites. Accuracy per class is in `validation/by_confidence.csv`.

## `property_orientation.gpkg`

| Layer | Geometry | Content |
|---|---|---|
| `addresses` | Point | Same columns as the CSV. |
| `facing_lines` | LineString | One line per site that has a bearing, from the address point to the faced point on the street, with `site_id`, `street_key`, `matched_street`, `orientation`, `bearing_deg`, `distance_m`, `confidence`, `n_addresses`. |
| `suburbs` | MultiPolygon | Whole suburb boundaries (some extend beyond the council area) with the `summary_by_suburb.csv` columns. |
| `study_area` | Polygon | The local government area boundary. |

All layers use EPSG:4326. Distances and bearings were computed in GDA2020 / MGA zone 56
(EPSG:7856), and bearings were corrected to true north.

## Summaries

* `summary_orientation.csv`: `scope` (`all addresses` or `standalone addresses`),
  `with_orientation` (addresses that have a bearing), `n_<dir>` counts, `share_<dir>` fractions,
  and `share_northerly` (N + NE + NW).
* `summary_by_suburb.csv`: per suburb, `addresses`, `standalone`, `high_confidence` (the share of
  addresses in the `high` class), then the same `with_orientation`, `n_<dir>`, `share_<dir>` and
  `share_northerly` columns **for standalone addresses only**. Apartment blocks would otherwise
  dominate the mix in suburbs such as Chatswood and St Leonards.

## Validation

All validation tables are computed on **sites** (unique location, street and geocode type), so
a 200-unit building counts once. A site is evaluated when it has a bearing and a usable cadastral
reference (`ref_status = ok`). Metrics:

* `sites`, `parcels`: evaluated sites, and the distinct cadastral parcels they sit on. A campus
  with many building-centroid geocodes is one parcel but many sites.
* `same_8_point_direction`: estimate and reference fall in the same 45° compass sector.
* `within_22_5_deg`, `within_45_deg`: angular error at most 22.5° or 45°.
* `median_error_deg`, `p90_error_deg`: median and 90th percentile of the absolute angular error.

`run_summary.json` labels its counts by unit: `confidence_addresses` counts addresses, while
`confidence_sites` and `match_sites` count sites. `software` lists the library versions (and
GEOS, PROJ and GDAL) that produced the results.

`run_summary.json` → `coregistration` measures how standalone `PROPERTY CENTROID` geocodes sit
in their cadastral parcels: the share inside a parcel and the median offset from the parcel
centroid. A datum mismatch, such as GDA94 against GDA2020, would show as a systematic offset of
about 1.5 m.

`same_8_point_direction` is stricter than it looks: a small error that straddles a sector
boundary counts as a miss. That is why `within_22_5_deg` is higher.

## Licences

The results are derived data, and each GeoPackage layer's description repeats this notice.

* Incorporates or developed using G-NAF © Geoscape Australia, licensed by the Commonwealth of
  Australia under the [Open G-NAF End User Licence Agreement](https://data.gov.au/dataset/geocoded-national-address-file-g-naf).
* Suburb and LGA boundaries © Geoscape Australia, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* Street names and geometry © OpenStreetMap contributors, available under the
  [Open Database Licence 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
  ([copyright](https://www.openstreetmap.org/copyright)). The street-derived columns and the
  `facing_lines` layer are a derivative database under the ODbL.
* The validation reference uses NSW cadastral data © Spatial Services (DCS), NSW Government,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
