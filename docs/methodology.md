# Methodology

This document specifies the orientation method, the independent validation reference, and
the design decisions behind them. The code is in
[`src/property_orientation/`](../src/property_orientation/). Apart from the out-of-sample
check in §5, all numbers are for Willoughby City Council and come from the files in
[`results/willoughby/`](../results/willoughby/); they depend on the OpenStreetMap snapshot
recorded in `run_summary.json`.

## 1. Definition

A property's **orientation** is the compass direction of its street frontage: the direction
from the property towards the street it is addressed to. It is reported as

* `bearing_deg`: a continuous bearing in degrees clockwise from true north, rounded to 0.1°, and
* `orientation`: one of eight 45° sectors (`N` = 337.5°–22.5°, `NE` = 22.5°–67.5°, ...),
  assigned from the published (rounded) bearing so the two always agree.

A property on the south side of an east–west street faces **north**.

## 2. Inputs

| Layer | Source | Use |
|---|---|---|
| Address points | G-NAF principal addresses, via the [gnaf-loader](https://github.com/minus34/gnaf-loader) GeoParquet export (Geoscape release 2026-08) | what to orient: one point per address, with street name, type and suffix |
| Street centrelines | OpenStreetMap `highway=*` ways via the Overpass API | what the properties face |
| Parcels, road reserves, road centrelines | NSW Digital Cadastral Database via NSW Spatial Services | the validation reference only |
| LGA and suburb boundaries | Geoscape Administrative Boundaries | study area and reporting |

Streets and cadastre are downloaded for the LGA plus a 500 m buffer, so addresses near the
boundary can match streets just outside it. All geometry is processed in GDA2020 / MGA
zone 56 (EPSG:7856), chosen automatically from the study area's longitude.

**Co-registration check.** Of 16,534 standalone G-NAF `PROPERTY CENTROID` geocodes, 99.96% fall
inside a NSW property parcel. They sit a median of 0.15 m from that parcel's centroid, with no
systematic shift: the median offsets are 0.00 m east and 0.00 m north. A GDA94/GDA2020 mix would
show about 1.5 m. So no datum transformation is applied between G-NAF and the NSW cadastre as
served (`run_summary.json` → `coregistration`).

## 3. Method

### 3.1 Sites

Addresses that share a location (to 1 cm), a street and a geocode type are collapsed into one
**site**. All units of an apartment block form a single site, so each geometric problem is
solved once: 43,551 addresses in Willoughby reduce to 19,436 sites. The geocode type is part of
the key because the confidence rule depends on it (§3.4). Where one building carries both
building-centroid and property-centroid geocodes, the two groups are separate sites with the
same bearing.

### 3.2 Street matching

G-NAF stores the street as separate name, type and suffix fields (`MOWBRAY` + `ROAD` +
`WEST`). OSM uses free text (`Mowbray Road West`), and NSW centrelines use abbreviations
(`MOWBRAY RD W`). `streets.normalise_street_name` maps all three to a canonical form. It
upper-cases, strips punctuation, expands AS/NZS 4819 road-type abbreviations and street
suffixes in their final position (including two-word suffixes such as `NORTH EAST`), and
writes "Saint" as `ST`.

For each site, every OSM way within 250 m is a candidate. Matches are ranked in tiers:

| Tier | Rule | Example |
|---|---|---|
| 1 | normalised names equal (including OSM `alt_name`, `official_name`, `old_name`) | `THE PARAPET` = `The Parapet` |
| 2 | equal once a street suffix is dropped from exactly one side | `JOUBERT STREET` = `Joubert Street North` |
| 3 | G-NAF name without its road type equals the OSM name | `HIGHLAND RIDGE` + `ROAD` = `Highland Ridge` |

A street suffix is any G-NAF suffix word: directions (`EAST`, `NORTH WEST`, ...) and qualifiers
such as `UPPER`, `LOWER`, `MALL` or `EXTENSION`. Two different suffixes never match, so
`MARKET STREET EAST` is never matched to `Market Street West`. When the dropped suffix is
G-NAF's (`MARKET STREET EAST` against an OSM `Market Street`), the match is kept but capped at
`low` confidence. In Willoughby, the 8 Market Street East addresses are a separate street that
OSM lacks, and the reference shows that they face elsewhere.

The best tier wins. Then ways a property can front (`FRONTAGE_HIGHWAYS`: streets, pedestrian
streets and named service lanes, but not link roads) are preferred over all other ways, and
finally the shorter distance decides. Footways can still win a name match, because a street
may exist in OSM only as a walkway; in the Willoughby snapshot no site needed this. Motorways,
motorway links and ways under construction are never matched.

If no candidate matches by name, the site falls back to the nearest way a property can front
(`trunk` … `residential`, `living_street`, `pedestrian`, named `service`), and is marked `low`
confidence. In Willoughby, 19,427 of 19,436 sites (99.95%) matched by name.

### 3.3 Bearing

The faced point is the nearest point on the matched way, from `shapely.shortest_line`. The
bearing is the **geodesic forward azimuth** on the GRS80 ellipsoid from the address point to
that point (`pyproj.Geod.inv`). This reports true north rather than MGA grid north; the two
differ by about 1° in Sydney (grid convergence).

The nearest-point vector is perpendicular to the street wherever the nearest point lies
inside a segment. It also handles cul-de-sac heads correctly, because a property at the end
of a cul-de-sac faces the end of the street.

### 3.4 Confidence

| Class | Rule |
|---|---|
| `high` | name match (not one that dropped a G-NAF suffix), ≤ 60 m, not a `BUILDING CENTROID` geocode, no differently named street closer |
| `medium` | name match (not one that dropped a G-NAF suffix), ≤ 120 m |
| `low` | nearest-street fallback, a match that dropped a G-NAF suffix, or > 120 m |
| `none` | `STREET LOCALITY`/`LOCALITY` geocode, no street within 250 m, or < 0.5 m from the centreline |

## 4. Validation reference

The reference shares only two inputs with the method: the G-NAF address point, used to find
the parcel, and the G-NAF street name, used to pick the frontage. Its geometry comes from the
NSW cadastre (parcel boundaries, road reserves and NSW road centrelines), so it is independent
of the OpenStreetMap centrelines the method relies on.

1. **Parcel.** Each site point is assigned the smallest NSW property parcel that contains it.
   Exact-duplicate polygons, such as stacked strata records, are removed first.
2. **Frontage.** The parcel boundary is intersected with the road-reserve polygons, buffered
   by 0.5 m. Each resulting straight segment is frontage.
3. **Outward normal.** For each segment, the normal that points *out of* the parcel is kept. It
   is tested with a point-in-polygon probe 0.25 m along the normal.
4. **Which street?** A 40 m ray is cast along each normal. The segment is named after the
   first NSW road centreline the ray crosses, which is the street that edge actually faces;
   a ray that first crosses an unnamed centreline leaves the segment unnamed. This handles
   corner lots and laneways: an earlier nearest-centreline rule mislabelled frontage onto
   unnamed lanes as the main street.
5. **Reference bearing.** This is the length-weighted mean of the outward normals of the
   frontage onto the *addressed* street. It is accepted when that frontage is at least 3 m long
   and the normals agree (mean resultant length ≥ 0.5).

A usable reference exists for 18,194 of 19,436 sites (93.6%). The others have no parcel
(118), no frontage onto the addressed street (830: rear lots, access-handle lots, or streets
named differently in the cadastre), less than 3 m of it (293), or frontage normals that cancel
out (1). See `results/willoughby/validation/reference_coverage.csv`.

## 5. Results

Agreement with the reference on the 18,193 sites (on 15,864 parcels) that have both a bearing
and a reference:

| Method | Same 8-point direction | Within 22.5° | Within 45° | Median error | 90th pct error |
|---|---|---|---|---|---|
| **Street-matched (this repository)** | **93.3%** | **96.8%** | **98.5%** | **0.5°** | **8.1°** |
| Nearest OSM way of any class (the v1 approach, re-run on the same data) | 64.9% | 67.2% | 69.4% | 1.5° | 121° |

The baseline's median error is small, because it is right when the nearest way is the
street. Its 90th percentile is large, because 30.6% of sites end up more than 45° off.

| Baseline snapped to | Share of sites | Wrong by > 45° | Share of baseline errors |
|---|---|---|---|
| footway (sidewalks, paths) | 41.5% | 22.2% | 30.1% |
| residential street | 37.4% | 14.6% | 17.8% |
| service way (driveways, lanes) | 15.9% | 79.3% | 41.2% |
| any other class (cycleway, path, steps, unclassified, tertiary, ...) | 5.1% | 64.7% | 10.9% |

A sidewalk runs parallel to its street, so snapping to it usually gives the right direction.
A driveway runs perpendicular to the street, so snapping to it usually does not. The
residential-street errors are almost all (96%) snaps to a differently named street that lies
closer than the addressed one, such as the side street at a corner or a rear street. Counting
a site as wrong when it is more than 45° off, street matching fixes 5,408 of the baseline's
errors and introduces 114 new ones.

Accuracy by confidence class:

| Confidence | Sites | Parcels | Same 8-point direction | Within 45° |
|---|---|---|---|---|
| high | 12,349 | 12,082 | 95.0% | 99.5% |
| medium | 5,683 | 3,909 | 91.0% | 97.9% |
| low | 161 | 41 | 37.9% | 46.6% |

The `low` class is small and clustered: 161 of its 199 sites have a reference, on 41
parcels. One large campus parcel on Herbert Street, Naremburn holds 73 evaluated sites. 63 of
them are `low` (62 building-centroid geocodes, 124–164 m from the street, all more than 45°
off) and make up 63 of the 86 low-class errors.

**Out-of-sample check.** The confidence thresholds were set on Willoughby. The same code on
Hunters Hill LGA (7,200 addresses, 3,828 evaluated sites), which was not used for any tuning,
gives 91.4% same 8-point direction and 97.2% within 45°. The nearest-road approach gives 68.3%
and 73.1% there. That run used G-NAF release 202608, the NSW cadastre as retrieved on
2026-10-09, and OpenStreetMap as of 2026-10-09T06:51:05Z. Reproduce with
`make data run LGA="Hunters Hill"`; the numbers will shift a little with later OSM edits.

## 6. Limitations

* **Frontage is not façade.** The method reports the direction of the street frontage. A house
  on a corner, or one deliberately turned to the view or the sun, can face elsewhere. For
  solar access the rear of the lot matters; it faces `bearing_deg + 180°`.
* **Large sites.** Schools, hospitals and campuses carry building-centroid geocodes far from
  the street, so they get `medium` or `low` confidence.
* **OSM completeness.** A street missing from OSM, or named differently there, falls back to
  the nearest street (`low` confidence). In Willoughby this affected 9 sites.
* **Units inherit their building's orientation.** Individual apartments can face any side of
  the building.
* **The reference is geometric, not ground truth.** It encodes the same frontage concept from
  independent data. It does not observe front doors.
