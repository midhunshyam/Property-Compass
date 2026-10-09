"""Interactive web map: every site, coloured by the direction it faces.

A site is the addresses that share one location, street and geocode type (the units of one
building, say); the pages count sites and addresses, and nothing else. :func:`build` writes one
self-contained page, ``docs/map/index.html`` for the Willoughby showcase (``docs/map/<area>/``
for any other area), and for the showcase only a small landing page, ``docs/index.html``, for
GitHub Pages. Leaflet comes from a CDN (pinned, with subresource integrity); the data is inline.
Sites are stored as compact column arrays and drawn by a subclass of Leaflet's canvas renderer,
so ~20,000 sites stay at about 2 MB and pan smoothly. Every number on either page comes from
the pipeline outputs (``results/`` and the processed sites), never typed in.
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import quote

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from property_orientation.config import Paths
from property_orientation.geometry import COMPASS_8, compass8
from property_orientation.orientation import NON_PROPERTY_GEOCODES, OrientationParams
from property_orientation.outputs import rounded_bearing
from property_orientation.viz import style
from property_orientation.viz.data import FigureData

log = logging.getLogger("property_orientation")

REPO_URL = "https://github.com/midhunshyam/property-orientation"
LEAFLET_URL = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4"
LEAFLET_SRI = {
    "leaflet.css": "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=",
    "leaflet.js": "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=",
}
SIZE_BUDGET_BYTES = 6_000_000
CONFIDENCE_LEVELS = ("high", "medium", "low", "none")
BASELINE_NAME = "nearest-road approach (v1)"
"""The approach this repository replaces: face the nearest OpenStreetMap way of any class."""

CREDITS = (
    "Addresses: G-NAF © Geoscape Australia (Open G-NAF EULA)",
    "Boundaries: © Geoscape Australia (CC BY 4.0)",
    "Streets: © OpenStreetMap contributors (ODbL)",
)
"""Data credits on both pages; the base maps credit themselves in the map's corner."""
CADASTRE_CREDIT = "Cadastre: © NSW Spatial Services (CC BY 4.0)"
"""Added when the area was validated against the NSW cadastre."""
LICENCE_URLS = {
    "Open G-NAF EULA": "https://data.gov.au/dataset/geocoded-national-address-file-g-naf",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "ODbL": "https://www.openstreetmap.org/copyright",
}
"""Where each licence named in the credits is published."""


def credits_html(credits: str) -> str:
    """The credits as HTML, with each licence name linked to its text."""
    out = html.escape(credits)
    for name, url in LICENCE_URLS.items():
        out = out.replace(f"({name})", f'(<a href="{url}">{name}</a>)')
    return out


BASEMAPS = (
    {
        "name": "Street map",
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "options": {
            "maxNativeZoom": 19,
            "maxZoom": 20,
            "className": "basemap-muted",
            "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">'
            "OpenStreetMap</a> contributors",
        },
    },
    {
        "name": "Aerial imagery",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"
        "/tile/{z}/{y}/{x}",
        "options": {
            "maxNativeZoom": 19,
            "maxZoom": 20,
            "attribution": 'Powered by <a href="https://www.esri.com/">Esri</a> &middot; '
            "Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community",
        },
    },
)
"""Keyless base layers; the first is the default. CARTO's Positron now answers keyless requests
with an "API key required" placeholder tile (checked October 2026), so the light street map is
OpenStreetMap's, shown in greyscale (the ``basemap-muted`` class) to keep the direction colours
the loudest thing on the page. Esri World Imagery is for checking against aerial photos (zoom
20 upscales zoom-19 tiles, which exist everywhere); its attribution is the service's own
copyright text. The NSW base map (lot boundaries) is not offered: Chrome rejects the
certificate of ``maps.six.nsw.gov.au`` (ERR_CERT_AUTHORITY_INVALID, October 2026)."""

_PARAMS = OrientationParams()
NOTES: dict[str, tuple[str, str]] = {
    "no_property_location": (
        "no property location in G-NAF",
        "G-NAF locates this address only to its street or suburb, not to the property, so it "
        "has no facing direction.",
    ),
    "on_street": (
        "on the street line",
        f"G-NAF places this site within {_PARAMS.min_distance_m:g} m of the street line, so it "
        "has no facing direction.",
    ),
    "no_street": (
        "no street nearby",
        f"No street a property can front was found within {_PARAMS.search_radius_m:g} m, so it "
        "has no facing direction.",
    ),
    "nearest_street": (
        "nearest street",
        f"No OpenStreetMap street with this name is within {_PARAMS.search_radius_m:g} m, so "
        "this faces the nearest street instead.",
    ),
}
"""Why a site has no direction, or faces a street other than its own: (short, full) text for
the tooltip and the popup. The thresholds come from :class:`OrientationParams`."""


@dataclass(frozen=True)
class PageFacts:
    """Numbers quoted on the pages, read from the pipeline outputs (never typed in)."""

    place: str
    addresses: int
    sites: int
    direction_counts: dict[str, int]
    """Sites facing each 8-point direction."""
    no_direction: int
    """Sites without a facing direction."""
    checked: int | None = None
    """Sites with both a facing bearing and a usable cadastral reference; ``None`` if the area
    was not validated (there is no cadastre outside NSW), and so for the fields below."""
    agreement: float | None = None
    """Share of checked sites with the same 8-point direction as the reference."""
    baseline_agreement: float | None = None
    """The same share for the nearest-road approach (v1)."""
    median_error_deg: float | None = None
    agreement_by_confidence: dict[str, float] = field(default_factory=dict)

    @property
    def validated(self) -> bool:
        return self.checked is not None

    @property
    def credits(self) -> str:
        """The data credits, as plain text."""
        return " · ".join((*CREDITS, CADASTRE_CREDIT) if self.validated else CREDITS)


def page_facts(data: FigureData, place: str) -> PageFacts:
    """Headline counts, and the validation results if the area has them."""
    counts = data.sites["orientation"].value_counts()
    facts = PageFacts(
        place=place,
        addresses=int(data.summary["addresses"]),
        sites=int(data.summary["sites"]),
        direction_counts={label: int(counts.get(label, 0)) for label in COMPASS_8},
        no_direction=int(data.sites["orientation"].isna().sum()),
    )
    if "validation" not in data.summary:
        return facts
    methods = data.validation("method_comparison").set_index("method")
    by_confidence = data.validation("by_confidence").set_index("confidence")
    return replace(
        facts,
        checked=int(methods.loc["street_matched", "sites"]),
        agreement=float(methods.loc["street_matched", "same_8_point_direction"]),
        baseline_agreement=float(methods.loc["nearest_road", "same_8_point_direction"]),
        median_error_deg=float(methods.loc["street_matched", "median_error_deg"]),
        agreement_by_confidence={
            str(level): float(share)
            for level, share in by_confidence["same_8_point_direction"].items()
        },
    )


_WORD_START = re.compile(r"\b[a-z]")
_NUMBER_SUFFIX = re.compile(r"(?<=\d)[a-z]\b")
_DIGITS = re.compile(r"\d+")


def title_case(text: str) -> str:
    """``12A O'CONNELL STREET`` -> ``12A O'Connell Street``, leaving ``1st`` alone."""
    text = _WORD_START.sub(lambda m: m[0].upper(), text.lower())
    return _NUMBER_SUFFIX.sub(lambda m: m[0].upper(), text)


def site_labels(addresses: pd.DataFrame) -> pd.Series:
    """One street address per site id, such as ``20-22 Cleland Road``.

    Unit, suite, shop and level prefixes are dropped, so a block of units is labelled by its
    building address. A site holding several building addresses takes the most common one.
    """
    building = addresses["address"].str.split(", ").str[-2]
    best = (
        pd.DataFrame({"site_id": addresses["site_id"], "label": building})
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["site_id", "count", "label"], ascending=[True, False, True])
        .drop_duplicates("site_id")
    )
    return pd.Series(best["label"].map(title_case).to_numpy(), index=best["site_id"].to_numpy())


def _address_key(label: str, locality: str) -> tuple[str, str, int, str]:
    """Sort by street, suburb, then house number (so 2 comes before 10)."""
    number, _, street = label.partition(" ")
    digits = _DIGITS.search(number)
    return street, locality, int(digits[0]) if digits else 0, number


def _floats(values: object, digits: int) -> list[float | None]:
    """Rounded floats for JSON, ``None`` for NaN. Built-in ``round`` keeps the repr short."""
    return [None if math.isnan(v) else round(v, digits) for v in np.asarray(values, float).tolist()]


def _codes(values: object, categories: list[str] | tuple[str, ...]) -> list[int]:
    """Position of each value in ``categories``; ``-1`` for missing values."""
    return pd.Categorical(values, categories=list(categories)).codes.tolist()


def published_directions(sites: pd.DataFrame) -> tuple[pd.Series, list[int]]:
    """Each site's bearing as published (0.1°, below 360°) and its 8-point label's index.

    The label is the site's ``orientation``, which the pipeline takes from the rounded bearing,
    so a site at 22.46° (published as 22.5°) is NE on the map as in ``results/``. Raises if
    ``sites`` comes from an older pipeline whose labels disagree with the published bearings.
    """
    bearing = rounded_bearing(sites["bearing_deg"])
    direction = _codes(sites["orientation"], COMPASS_8)
    if direction != _codes(compass8(bearing), COMPASS_8):
        raise ValueError(
            "the orientation labels in sites.parquet disagree with the published bearings: "
            "re-run the pipeline (`property-orientation run`)"
        )
    return bearing, direction


def site_notes(sites: pd.DataFrame) -> dict[str, int]:
    """Sparse ``{position: index into NOTES}`` for the few sites that need an explanation."""
    no_direction = sites["orientation"].isna().to_numpy()
    reason = np.select(
        [
            no_direction & sites["geocode_type"].isin(NON_PROPERTY_GEOCODES).to_numpy(),
            no_direction & (sites["match"] == "none").to_numpy(),
            no_direction,
            (sites["match"] == "nearest_street").to_numpy(),
        ],
        ["no_property_location", "no_street", "on_street", "nearest_street"],
        default="",
    )
    keys = list(NOTES)
    return {str(i): keys.index(r) for i, r in enumerate(reason.tolist()) if r}


def site_payload(data: FigureData) -> dict:
    """Every site as column arrays in WGS84, sorted by street, suburb and number.

    Coordinates are rounded to 6 decimals (about 0.1 m), bearings and distances to 0.1, and
    each site's direction (``dir``) is the 8-point label of its bearing as published (see
    :func:`published_directions`). Street and suburb names are stored once and referenced by
    index. ``face_lon``/``face_lat`` (the nearest point on the street) feed the optional
    facing-lines layer; ``note`` explains the sites without a direction and those that face a
    street other than their own.
    """
    sites = data.sites
    labels = site_labels(data.addresses).reindex(sites["site_id"]).fillna("").tolist()
    suburb = [title_case(v) for v in sites["locality"].fillna("")]
    order = sorted(range(len(sites)), key=lambda i: _address_key(labels[i], suburb[i]))

    s = sites.iloc[order]
    bearing, direction = published_directions(s)
    point = s.geometry.to_crs(4326)
    faced = gpd.GeoSeries(gpd.points_from_xy(s["street_x"], s["street_y"]), crs=s.crs).to_crs(4326)
    has_direction = s["orientation"].notna().to_numpy()
    confidence = _codes(s["confidence"], CONFIDENCE_LEVELS)
    if min(confidence) < 0:
        raise ValueError(
            f"unexpected confidence values: {set(s['confidence']) - {*CONFIDENCE_LEVELS}}"
        )
    streets = sorted(s["matched_street"].dropna().unique())
    localities = sorted(s["locality"].dropna().unique())
    return {
        "sites": {
            "lon": _floats(point.x, 6),
            "lat": _floats(point.y, 6),
            "dir": direction,
            "bearing": _floats(bearing, 1),
            "street": _codes(s["matched_street"], streets),
            "dist": _floats(s["distance_m"], 1),
            "conf": confidence,
            "n": s["n_addresses"].astype(int).tolist(),
            "addr": [labels[i] for i in order],
            "loc": _codes(s["locality"], localities),
            "face_lon": _floats(np.where(has_direction, faced.x, np.nan), 6),
            "face_lat": _floats(np.where(has_direction, faced.y, np.nan), 6),
        },
        "streets": streets,
        "localities": [title_case(v) for v in localities],
        "notes": [list(text) for text in NOTES.values()],
        "note": site_notes(s),
    }


def boundary_rings(boundary: gpd.GeoDataFrame, tolerance_m: float = 1.0) -> list:
    """Outline rings of the study area as ``[[lat, lon], ...]`` lists, simplified for the web."""
    shape = boundary.geometry.union_all().simplify(tolerance_m)
    shape = gpd.GeoSeries([shape], crs=boundary.crs).to_crs(4326).iloc[0]
    return [
        [[round(y, 6), round(x, 6)] for x, y in ring.coords]
        for polygon in shapely.get_parts(shape)
        for ring in (polygon.exterior, *polygon.interiors)
    ]


def compass_svg(counts: dict[str, int] | None = None) -> str:
    """The compass-rose legend of :func:`style.compass_legend` as inline SVG.

    With ``counts`` each wedge becomes a toggle button labelled with its number of sites.
    """
    outer, inner, label_radius = 40.0, 18.0, 52.0
    total = sum(counts.values()) if counts else 0

    def xy(radius: float, degrees: float) -> str:
        a = math.radians(degrees)
        return f"{round(radius * math.sin(a), 2) + 0.0:g} {round(-radius * math.cos(a), 2) + 0.0:g}"

    parts = []
    for k, label in enumerate(COMPASS_8):
        lo, hi, centre = 45 * k - 22.5, 45 * k + 22.5, 45 * k
        path = (
            f'<path d="M{xy(outer, lo)}A{outer:g} {outer:g} 0 0 1 {xy(outer, hi)}'
            f'L{xy(inner, hi)}A{inner:g} {inner:g} 0 0 0 {xy(inner, lo)}Z" '
            f'fill="{style.DIRECTION_COLORS[label]}"/>'
        )
        x, y = xy(label_radius, centre).split()
        kind = "cardinal" if len(label) == 1 else "intercardinal"
        text = f'<text x="{x}" y="{y}" class="{kind}">{label}</text>'
        if counts is None:
            parts.append(path + text)
            continue
        n = counts[label]
        desc = f"{style.DIRECTION_NAMES[label]}: {n:,} sites ({n / total:.0%})"
        parts.append(
            f'<g class="wedge" data-dir="{k + 1}" role="button" tabindex="0" '
            f'aria-pressed="false" aria-label="{desc}"><title>{desc}</title>{path}{text}</g>'
        )
    hidden = ' aria-hidden="true"' if counts is None else ' role="group" aria-label="Facing"'
    return f'<svg class="compass" viewBox="-64 -64 128 128"{hidden}>{"".join(parts)}</svg>'


def _favicon() -> str:
    """A compass needle: north yellow, south blue, as a data URI."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
        f"<circle cx='16' cy='16' r='14' fill='{style.SURFACE}' stroke='{style.INK}' "
        "stroke-width='2'/>"
        f"<path d='M16 4l5 12H11z' fill='{style.DIRECTION_COLORS['N']}'/>"
        f"<path d='M16 28l-5-12h10z' fill='{style.DIRECTION_COLORS['S']}'/></svg>"
    )
    return "data:image/svg+xml," + quote(svg)


def _root_css() -> str:
    """House-style tokens as CSS custom properties."""
    tokens = {
        "surface": style.SURFACE,
        "ink": style.INK,
        "ink-2": style.INK_SECONDARY,
        "ink-3": style.INK_MUTED,
        "grid": style.GRID,
        "baseline": style.BASELINE,
    }
    return ":root{" + "".join(f"--{k}:{v};" for k, v in tokens.items()) + "}"


def _fill(template: str, values: dict[str, str]) -> str:
    """Replace each ``{{NAME}}`` in one pass, so inserted text is never re-scanned."""
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values[m[1]], template)


def script_json(payload: dict) -> str:
    """Compact JSON that is safe inside ``<script>``: every ``<`` is escaped as ``\\u003c``,
    so neither ``</script>`` nor ``<!--`` can occur (``<`` only appears inside strings)."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def _validation_text(facts: PageFacts) -> str:
    """The map's one-sentence validation summary (HTML), or nothing for an unvalidated area."""
    if not facts.validated:
        return ""
    return (
        f"<br>Same 8-point direction as the NSW cadastre for {facts.agreement:.1%} of the "
        f"{facts.checked:,} sites checked, against {facts.baseline_agreement:.1%} for the "
        f"{BASELINE_NAME}."
    )


def render_map(payload: dict, facts: PageFacts) -> str:
    """The map page as one HTML string."""
    counts = facts.direction_counts
    validated = ", validated against the NSW cadastre" if facts.validated else ""
    return _fill(
        _MAP_HTML,
        {
            "TITLE": "Which way does every property face?",
            "PLACE": html.escape(facts.place),
            "VALIDATED": validated,
            "FAVICON": _favicon(),
            "LEAFLET": LEAFLET_URL,
            "CSS_SRI": LEAFLET_SRI["leaflet.css"],
            "JS_SRI": LEAFLET_SRI["leaflet.js"],
            "ROOT_CSS": _root_css(),
            "CSS": _MAP_CSS,
            "SITES": f"{facts.sites:,}",
            "ADDRESSES": f"{facts.addresses:,}",
            "VALIDATION": _validation_text(facts),
            "REPO": REPO_URL,
            "CREDITS": credits_html(facts.credits),
            "COMPASS": compass_svg(counts),
            "NO_DIRECTION": f"{facts.no_direction:,}",
            "DATA": script_json(payload),
            "JS": _MAP_JS,
        },
    )


def render_landing(facts: PageFacts) -> str:
    """A short landing page for GitHub Pages that leads to the map and the repository."""
    if not facts.validated:
        raise ValueError("the landing page quotes the validation against the NSW cadastre")
    return _fill(
        _LANDING_HTML,
        {
            "PLACE": html.escape(facts.place),
            "FAVICON": _favicon(),
            "ROOT_CSS": _root_css(),
            "COMPASS": compass_svg(),
            "ADDRESSES": f"{facts.addresses:,}",
            "SITES": f"{facts.sites:,}",
            "AGREEMENT": f"{facts.agreement:.1%}",
            "BASELINE": f"{facts.baseline_agreement:.1%}",
            "BASELINE_NAME": BASELINE_NAME,
            "CHECKED": f"{facts.checked:,}",
            "MEDIAN": f"{facts.median_error_deg:.1f}°",
            "REPO": REPO_URL,
            "CREDITS": credits_html(facts.credits),
        },
    )


def landing_page(paths: Paths) -> Path | None:
    """Where the landing page goes (``docs/index.html``): only the showcase area has one."""
    return paths.root / "docs" / "index.html" if paths.area.is_showcase else None


def build(paths: Paths) -> Path:
    """Write the map page of ``paths.area``, and for the showcase also the landing page.

    The map goes to ``paths.webmap`` (``docs/map/`` for Willoughby, ``docs/map/<area>/``
    otherwise), so another area never overwrites the showcase; the landing page,
    ``docs/index.html``, is written for the showcase only. The facing-lines layer is left out
    if the page would otherwise exceed :data:`SIZE_BUDGET_BYTES`. Returns the map page's path.
    """
    data = FigureData(paths)
    facts = page_facts(data, f"{paths.area.lga}, {paths.area.state}")
    payload = {
        "directions": list(COMPASS_8),
        "names": [style.DIRECTION_NAMES[label] for label in COMPASS_8],
        "colors": [style.DIRECTION_COLORS[label] for label in COMPASS_8],
        "theme": {
            "surface": style.SURFACE,
            "ink": style.INK,
            "ink2": style.INK_SECONDARY,
            "none": style.INK_MUTED,
        },
        "confidence": list(CONFIDENCE_LEVELS),
        "agreement": facts.agreement_by_confidence,
        "basemaps": list(BASEMAPS),
        "credits": credits_html(facts.credits),
        "boundary": boundary_rings(data.boundary),
        **site_payload(data),
    }

    page = render_map(payload, facts)
    if len(page.encode()) > SIZE_BUDGET_BYTES:
        log.warning("map page over budget: leaving out the facing-lines layer")
        for key in ("face_lon", "face_lat"):
            del payload["sites"][key]
        page = render_map(payload, facts)

    out = paths.webmap / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    landing = landing_page(paths)
    if landing is not None:
        landing.write_text(render_landing(facts), encoding="utf-8")
        log.info("wrote %s", landing)

    size = out.stat().st_size
    log.info("wrote %s (%.2f MB, %d sites)", out, size / 1e6, len(payload["sites"]["lon"]))
    if size > SIZE_BUDGET_BYTES:
        log.warning(
            "%s is %.1f MB, over the %.0f MB budget", out, size / 1e6, SIZE_BUDGET_BYTES / 1e6
        )
    return out


# --- Page templates ---------------------------------------------------------------------------
# Placeholders are {{NAME}}; everything inserted is escaped or generated above.

_MAP_HTML = """<!doctype html>
<html lang="en-AU">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>{{TITLE}} {{PLACE}}</title>
<meta name="description" content="The compass direction every property in {{PLACE}} faces, \
from G-NAF addresses and OpenStreetMap streets{{VALIDATED}}.">
<link rel="icon" href="{{FAVICON}}">
<link rel="stylesheet" href="{{LEAFLET}}/leaflet.css" integrity="{{CSS_SRI}}"
  crossorigin="anonymous" referrerpolicy="no-referrer">
<style>
{{ROOT_CSS}}
{{CSS}}
</style>
</head>
<body>
<div id="map" role="region" aria-label="Map of sites coloured by the direction they face"></div>
<noscript><p class="noscript">This map needs JavaScript. The results are also in
  <a href="{{REPO}}">the repository</a> as CSV.</p></noscript>

<section class="panel title-panel" id="title-panel" aria-labelledby="map-title">
  <h1 id="map-title">{{TITLE}}</h1>
  <p class="place">{{PLACE}}</p>
  <p class="lede">Each dot is a site (the addresses sharing one location, street and geocode
    type), coloured by the compass direction it faces: towards the nearest point on its own
    street.</p>
  <div class="find" role="search">
    <label class="sr-only" for="find">Find an address</label>
    <input id="find" type="search" placeholder="Find an address" autocomplete="off"
      spellcheck="false" role="combobox" aria-autocomplete="list" aria-expanded="false"
      aria-controls="find-list">
    <ul id="find-list" role="listbox" aria-label="Matching addresses" hidden></ul>
  </div>
  <p class="meta collapsible" id="about">{{ADDRESSES}} addresses at {{SITES}}
    sites{{VALIDATION}}</p>
  <p class="meta links"><a href="{{REPO}}">Method, code and data on GitHub</a>
    <button type="button" class="more-toggle" id="more-toggle" aria-expanded="false"
      aria-controls="about credits">About this map</button></p>
  <p class="credits collapsible" id="credits">{{CREDITS}}</p>
</section>

<section class="panel legend" id="legend" aria-labelledby="legend-title">
  <h2 id="legend-title">Facing</h2>
  {{COMPASS}}
  <button type="button" class="none" data-dir="0" aria-pressed="false"
    aria-label="No direction: {{NO_DIRECTION}} sites"
    title="No direction: {{NO_DIRECTION}} sites">
    <span class="dot" aria-hidden="true"></span>No direction</button>
  <p class="hint">Select a direction to show only it.</p>
  <p class="legend-status" id="legend-status" hidden><span id="legend-status-text"></span>
    <button type="button" id="show-all">Show all</button></p>
</section>

<script id="map-data" type="application/json">{{DATA}}</script>
<script src="{{LEAFLET}}/leaflet.js" integrity="{{JS_SRI}}" crossorigin="anonymous"
  referrerpolicy="no-referrer"></script>
<script>
{{JS}}
</script>
</body>
</html>
"""

_MAP_CSS = r"""
:root{--hover:#f0efea;--font:"Helvetica Neue",Helvetica,Arial,sans-serif;
  --shadow:0 1px 2px rgba(11,11,11,.06),0 3px 12px rgba(11,11,11,.10)}
*,*::before,*::after{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:var(--surface);color:var(--ink);font:13px/1.45 var(--font);
  -webkit-font-smoothing:antialiased}
#map{position:absolute;inset:0}
#map.map-error{padding:16px;font-size:14px}
.leaflet-container{background:var(--surface);font:13px/1.45 var(--font)}
.basemap-muted{filter:grayscale(1) contrast(.85) brightness(1.1)}
.leaflet-container a{color:var(--ink)}
:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);
  white-space:nowrap}
.noscript{position:absolute;top:16px;left:16px;right:16px;margin:0}

.panel{background:var(--surface);color:var(--ink);border:1px solid var(--grid);
  border-radius:10px;box-shadow:var(--shadow)}
.panel:not(.leaflet-control){display:none}

.title-panel{width:344px;max-width:calc(100vw - 80px);padding:14px 16px 12px}
.title-panel h1{margin:0;font-size:17px;line-height:1.25;font-weight:700;letter-spacing:-.01em;
  text-wrap:balance}
.title-panel p{margin:0}
.title-panel .place{margin-top:2px;font-size:14px;color:var(--ink-2)}
.title-panel .lede{margin-top:10px;color:var(--ink-2)}
.title-panel .meta{margin-top:10px;font-size:12px;color:var(--ink-3)}
.title-panel .meta+.meta{margin-top:4px}
.title-panel a{color:var(--ink);text-decoration:underline;text-decoration-color:var(--baseline);
  text-underline-offset:2px}
.title-panel a:hover{text-decoration-color:var(--ink)}
.title-panel .credits{margin-top:10px;padding-top:8px;border-top:1px solid var(--grid);
  font-size:11px;line-height:1.4;color:var(--ink-3)}
.title-panel .links{display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 12px}
.more-toggle{display:none;padding:0;font:inherit;color:var(--ink-2);background:none;border:0;
  text-decoration:underline;text-decoration-color:var(--baseline);text-underline-offset:2px;
  cursor:pointer}

.find{position:relative;margin-top:12px}
.find input{display:block;width:100%;height:34px;padding:0 10px;font:inherit;font-size:14px;
  color:var(--ink);background:#fff;border:1px solid var(--baseline);border-radius:8px;
  -webkit-appearance:none;appearance:none}
.find input::placeholder{color:var(--ink-3)}
.find input:focus-visible{outline-offset:1px}
.find ul{position:absolute;z-index:10;top:calc(100% + 4px);left:0;right:0;margin:0;padding:4px;
  list-style:none;background:var(--surface);border:1px solid var(--grid);border-radius:8px;
  box-shadow:var(--shadow);max-height:280px;overflow-y:auto}
.find li{padding:6px 8px;border-radius:6px;cursor:pointer}
.find li[aria-selected="true"]{background:var(--hover)}
.find li .loc{margin-left:6px;font-size:12px;color:var(--ink-3)}
.find li.empty{color:var(--ink-3);cursor:default}

.legend{width:156px;padding:10px 12px 10px}
.legend h2{margin:0;font-size:12px;font-weight:700;color:var(--ink-2)}
.legend p{margin:0}
.compass{display:block;width:128px;height:128px;margin:0 auto 2px;overflow:visible}
.compass text{fill:var(--ink-2);text-anchor:middle;dominant-baseline:central;
  font-family:var(--font)}
.compass .cardinal{font-size:12px;font-weight:700}
.compass .intercardinal{font-size:10px}
.compass path{stroke:var(--surface);stroke-width:1.5}
.wedge{cursor:pointer;outline:none}
.wedge:hover path,.wedge:focus-visible path{stroke:var(--ink);stroke-width:1.5}
.wedge.is-off path{fill:var(--grid)}
.wedge.is-off text{fill:var(--ink-3)}
.legend .none{display:flex;align-items:center;gap:7px;width:100%;padding:2px 0;font:inherit;
  font-size:12px;color:var(--ink-2);text-align:left;background:none;border:0;cursor:pointer}
.legend .none .dot{flex:none;width:10px;height:10px;border-radius:50%;
  background:var(--surface);border:2px solid var(--ink-3)}
.legend .none.is-off{color:var(--ink-3)}
.legend .none.is-off .dot{border-color:var(--grid)}
.legend .hint,.legend .legend-status{margin-top:4px;font-size:11px;line-height:1.35;
  color:var(--ink-3)}
.legend.is-isolated .hint{display:none}
.legend-status button{padding:0;font:inherit;color:var(--ink);background:none;border:0;
  text-decoration:underline;text-underline-offset:2px;cursor:pointer}

.leaflet-tooltip.site-tip{padding:6px 9px;font:12px/1.4 var(--font);color:var(--ink);
  background:var(--surface);border:1px solid var(--grid);border-radius:8px;box-shadow:var(--shadow)}
.leaflet-tooltip-top.site-tip::before{border-top-color:var(--surface)}
.tip-value{display:flex;align-items:center;gap:6px;font-size:13px;font-weight:700}
.tip-label{color:var(--ink-2)}
.tip-more{margin-top:2px;font-size:11px;color:var(--ink-3)}
.key{display:inline-block;flex:none;width:12px;height:3px;border-radius:2px}
.key.is-none{width:10px;height:10px;border-radius:50%;background:var(--surface);
  border:2px solid var(--ink-3)}

.site-pop .leaflet-popup-content-wrapper{color:var(--ink);background:var(--surface);
  border-radius:10px;box-shadow:var(--shadow)}
.site-pop .leaflet-popup-tip{background:var(--surface);box-shadow:none}
.site-pop .leaflet-popup-content{margin:12px 16px 14px 14px;font:13px/1.45 var(--font)}
.site-pop a.leaflet-popup-close-button{color:var(--ink-3)}
.site-pop .pop-site+.pop-site{margin-top:12px;padding-top:12px;border-top:1px solid var(--grid)}
.site-pop .pop-title{margin:0;padding-right:14px;font-size:14px;font-weight:700}
.site-pop .pop-sub,.site-pop .pop-note{margin:2px 0 0;font-size:12px;color:var(--ink-3)}
.site-pop .pop-note{margin-top:8px}
.site-pop dl{display:grid;grid-template-columns:max-content 1fr;gap:5px 12px;margin:10px 0 0}
.site-pop dt{padding-top:1px;font-size:12px;color:var(--ink-3)}
.site-pop dd{margin:0}
.site-pop dd .key{margin-right:6px;vertical-align:3px}
.site-pop dd .key.is-none{vertical-align:-1px}
.site-pop .note{display:block;font-size:12px;color:var(--ink-3)}

#map .leaflet-bar,#map .leaflet-control-layers{background:var(--surface);
  border:1px solid var(--grid);border-radius:10px;box-shadow:var(--shadow)}
#map .leaflet-bar a{color:var(--ink-2);background:var(--surface);border-bottom-color:var(--grid)}
#map .leaflet-bar a:hover{color:var(--ink);background:var(--hover)}
#map .leaflet-bar a:first-child{border-radius:9px 9px 0 0}
#map .leaflet-bar a:last-child{border-radius:0 0 9px 9px}
#map .leaflet-bar a.leaflet-disabled{color:var(--baseline)}
#map .leaflet-control-layers-expanded{padding:8px 12px 8px 10px;color:var(--ink-2)}
#map .leaflet-control-layers-separator{margin:6px -12px 6px -10px;border-top-color:var(--grid)}
#map .leaflet-control-layers label{margin:3px 0}
#map .leaflet-control-layers input{margin:0 6px 0 0;accent-color:var(--ink);vertical-align:-1px}
#map .leaflet-control-attribution{color:var(--ink-3);background:var(--surface);font-size:11px}
#map .leaflet-control-attribution a{color:var(--ink-2)}
#map .leaflet-control-scale-line{color:var(--ink-2);background:var(--surface);
  border-color:var(--ink-3);font-size:11px}
#map.is-pointing{cursor:pointer}

@media (max-width:640px){
  .title-panel{padding:12px 14px 10px}
  .title-panel h1{font-size:15px}
  .title-panel .place{font-size:13px}
  .find input{font-size:16px}
  .more-toggle{display:inline-block}
  .title-panel:not(.is-open) .collapsible{display:none}
  .title-panel:not(.is-open) .links{margin-top:10px}
  .legend{width:124px;padding:8px 10px}
  .compass{width:100px;height:100px}
  .legend .hint{display:none}
  #map .leaflet-bottom.leaflet-left .leaflet-control{margin-bottom:24px}
}
"""

_MAP_JS = r"""
"use strict";
(() => {
  const mapEl = document.getElementById("map");
  if (!window.L) {
    mapEl.classList.add("map-error");
    mapEl.textContent = "The map library could not be loaded. Please check your connection " +
                        "and reload the page.";
    return;
  }
  const D = JSON.parse(document.getElementById("map-data").textContent);
  const S = D.sites;
  const N = S.lon.length;
  const T = D.theme;
  const COLOR = [T.none].concat(D.colors); // by direction index + 1; 0 = no direction
  const RADIUS = [1.5, 1.75, 2.25, 2.75, 3.25, 4, 5, 6]; // zoom 11, 12, ... 18+
  const TAU = 2 * Math.PI;
  const shown = COLOR.map(() => true);
  const hasLines = Array.isArray(S.face_lon); // left out when the page is over its size budget
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Project every point once to zoom-0 pixels. At zoom z its layer point is p * 2^z - origin.
  const crs = L.CRS.EPSG3857;
  const px = new Float64Array(N), py = new Float64Array(N);
  const fx = new Float64Array(N), fy = new Float64Array(N);
  const ux = new Float64Array(N), uy = new Float64Array(N); // facing unit vector on screen
  const withLine = COLOR.map(() => []);
  const atPoint = new Map(); // "lat,lon" -> sites drawn there (corner lots on two streets)
  for (let i = 0; i < N; i++) {
    const p = crs.latLngToPoint(L.latLng(S.lat[i], S.lon[i]), 0);
    px[i] = p.x;
    py[i] = p.y;
    if (S.bearing[i] !== null) {
      const a = (S.bearing[i] * Math.PI) / 180; // Web Mercator is conformal: azimuths hold
      ux[i] = Math.sin(a);
      uy[i] = -Math.cos(a);
    }
    if (hasLines && S.dir[i] >= 0 && S.face_lat[i] !== null) {
      const q = crs.latLngToPoint(L.latLng(S.face_lat[i], S.face_lon[i]), 0);
      fx[i] = q.x;
      fy[i] = q.y;
      withLine[S.dir[i] + 1].push(i);
    }
    const key = `${S.lat[i]},${S.lon[i]}`;
    if (atPoint.has(key)) atPoint.get(key).push(i);
    else atPoint.set(key, [i]);
  }
  const radius = (z) => RADIUS[Math.min(RADIUS.length - 1, Math.max(0, Math.round(z) - 11))];
  const frame = (map) => {
    const o = map.getPixelOrigin();
    return { k: map.getZoomScale(map.getZoom(), 0), x: o.x, y: o.y };
  };

  // Leaflet's canvas renderer keeps panning, zoom animation and retina scaling; these
  // subclasses only replace what it draws, straight from the arrays above.
  const ArrayCanvas = L.Canvas.extend({
    redraw() {
      if (this._map) this._redraw();
      return this;
    },
  });

  const SitesCanvas = ArrayCanvas.extend({
    _draw() {
      const ctx = this._ctx, b = this._bounds, z = this._map.getZoom(), f = frame(this._map);
      const r = radius(z), ring = z >= 15 ? 1.5 : 0, tick = z >= 17 ? r + (z >= 18 ? 10 : 8) : 0;
      const m = r + ring + tick;
      ctx.save();
      ctx.lineCap = "round";
      for (let i = 0; i < N; i++) {
        const c = S.dir[i] + 1;
        if (!shown[c]) continue;
        const x = px[i] * f.k - f.x, y = py[i] * f.k - f.y;
        if (x < b.min.x - m || x > b.max.x + m || y < b.min.y - m || y > b.max.y + m) continue;
        if (tick && c) { // a short tick toward the street, so direction never relies on colour
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + ux[i] * tick, y + uy[i] * tick);
          ctx.strokeStyle = T.surface;
          ctx.lineWidth = 4;
          ctx.stroke();
          ctx.strokeStyle = COLOR[c];
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        if (ring) {
          ctx.beginPath();
          ctx.arc(x, y, r + ring, 0, TAU);
          ctx.fillStyle = T.surface;
          ctx.fill();
        }
        ctx.beginPath();
        if (c) {
          ctx.arc(x, y, r, 0, TAU);
          ctx.fillStyle = COLOR[c];
          ctx.fill();
        } else { // no direction: a hollow ring, never mistaken for a colour (or a tree)
          ctx.arc(x, y, Math.max(r - 0.75, 0.75), 0, TAU);
          ctx.fillStyle = T.surface;
          ctx.fill();
          ctx.strokeStyle = COLOR[0];
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
      ctx.restore();
    },
  });

  const LinesCanvas = ArrayCanvas.extend({
    _draw() {
      const ctx = this._ctx, b = this._bounds, f = frame(this._map);
      ctx.save();
      ctx.lineCap = "round";
      for (const casing of [true, false]) {
        for (let c = 1; c < COLOR.length; c++) {
          if (!shown[c]) continue;
          ctx.beginPath();
          for (const i of withLine[c]) {
            const x0 = px[i] * f.k - f.x, y0 = py[i] * f.k - f.y;
            const x1 = fx[i] * f.k - f.x, y1 = fy[i] * f.k - f.y;
            if (Math.max(x0, x1) < b.min.x || Math.min(x0, x1) > b.max.x ||
                Math.max(y0, y1) < b.min.y || Math.min(y0, y1) > b.max.y) continue;
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
          }
          ctx.strokeStyle = casing ? T.surface : COLOR[c];
          ctx.lineWidth = casing ? 3.5 : 1.5;
          ctx.stroke();
        }
      }
      ctx.restore();
    },
  });

  // --- Map, panes and layers -----------------------------------------------------------------
  const narrow = window.matchMedia("(max-width: 640px)").matches; // the CSS breakpoint
  const lga = L.latLngBounds(D.boundary.flat());
  // A quarter zoom step lets a phone open on the whole council area; see below.
  const map = L.map(mapEl, { zoomControl: false, minZoom: 11, maxZoom: 20,
                             zoomSnap: narrow ? 0.25 : 1, maxBounds: lga.pad(0.6) });
  for (const [name, z] of [["boundary", 380], ["facing", 390], ["sites", 410], ["focus", 420]]) {
    map.createPane(name).style.zIndex = String(z);
  }
  const bases = {};
  for (const b of D.basemaps) bases[b.name] = L.tileLayer(b.url, b.options);
  const outline = L.layerGroup([
    L.polyline(D.boundary, { pane: "boundary", interactive: false, color: T.surface,
                             weight: 4.5, opacity: 1 }),
    L.polyline(D.boundary, { pane: "boundary", interactive: false, color: T.ink2,
                             weight: 1.5, opacity: 1 }),
  ]);
  const sites = new SitesCanvas({ pane: "sites", padding: 0.25 });
  const lines = hasLines ? new LinesCanvas({ pane: "facing", padding: 0.25 }) : null;
  const focusRing = L.circleMarker([0, 0], { renderer: L.svg({ pane: "focus" }), pane: "focus",
                                             interactive: false, fill: false, color: T.ink,
                                             weight: 2, opacity: 1 });
  const tip = L.tooltip({ className: "site-tip", direction: "top", offset: [0, -10], opacity: 1 });
  const popup = L.popup({ className: "site-pop", minWidth: 230, maxWidth: 300,
                          autoPanPadding: [24, 24] });

  // --- Controls --------------------------------------------------------------------------------
  const Panel = L.Control.extend({
    onAdd() {
      L.DomEvent.disableClickPropagation(this.options.el);
      L.DomEvent.disableScrollPropagation(this.options.el);
      return this.options.el;
    },
  });
  const titleEl = document.getElementById("title-panel");
  const legendEl = document.getElementById("legend");
  const more = document.getElementById("more-toggle");
  function setAbout(open) {
    titleEl.classList.toggle("is-open", open);
    more.setAttribute("aria-expanded", String(open));
    more.textContent = open ? "Hide details" : "About this map";
  }
  new Panel({ position: "topleft", el: titleEl }).addTo(map);
  const overlays = {};
  if (lines) overlays["Facing lines <small>(to the street)</small>"] = lines;
  overlays["Council boundary"] = outline;
  L.control.layers(bases, overlays, { position: "topright", collapsed: narrow }).addTo(map);
  new Panel({ position: "bottomleft", el: legendEl }).addTo(map);
  L.control.scale({ position: "bottomright", imperial: false, maxWidth: 100 }).addTo(map);
  L.control.zoom({ position: "bottomright" }).addTo(map);

  // --- View: from the URL hash (#zoom/lat/lon) or the whole council area ----------------------
  const readHash = () => {
    const m = /^#(\d{1,2}(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)$/.exec(location.hash);
    const center = m && L.latLng(+m[2], +m[3]);
    return center && lga.pad(0.6).contains(center) ? { center, zoom: +m[1] } : null;
  };
  const start = readHash();
  if (start) {
    map.setView(start.center, start.zoom);
  } else {
    map.fitBounds(lga, narrow ? { padding: [12, 12] }
      : { paddingTopLeft: [titleEl.offsetWidth + 32, 24], paddingBottomRight: [24, 24] });
  }
  map.options.zoomSnap = 1; // only the opening view sits between zoom levels: tiles stay crisp
  bases[D.basemaps[0].name].addTo(map);
  map.attributionControl.addAttribution(D.credits); // data credits stay visible on every base map
  outline.addTo(map);
  sites.addTo(map);
  map.on("moveend", () => {
    const c = map.getCenter();
    const hash = `#${+map.getZoom().toFixed(2)}/${c.lat.toFixed(5)}/${c.lng.toFixed(5)}`;
    if (location.hash === hash) return;
    try {
      history.replaceState(null, "", hash);
    } catch (err) {
      // Some sandboxed frames refuse; the hash is only a convenience.
    }
  });
  window.addEventListener("hashchange", () => {
    const view = readHash();
    if (view) map.setView(view.center, view.zoom);
  });

  // --- Text for tooltips and popups (DOM nodes and textContent only) -------------------------
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const keyOf = (i) => {
    if (S.dir[i] < 0) return el("span", "key is-none");
    const key = el("span", "key");
    key.style.background = COLOR[S.dir[i] + 1];
    return key;
  };
  const latLngOf = (i) => L.latLng(S.lat[i], S.lon[i]);
  const streetOf = (i) => (S.street[i] >= 0 ? D.streets[S.street[i]] : "Unnamed street");
  const placeOf = (i) => [S.addr[i], D.localities[S.loc[i]]].filter(Boolean).join(", ");
  const facingOf = (i) => `${D.names[S.dir[i]]} · ${S.bearing[i].toFixed(1)}°`;
  const noteOf = (i) => (D.note[i] === undefined ? null : D.notes[D.note[i]]); // [short, full]
  // The site under the pointer first, then any others drawn at the same point.
  const groupOf = (i) => [i].concat(atPoint.get(`${S.lat[i]},${S.lon[i]}`)
    .filter((j) => j !== i && shown[S.dir[j] + 1]));

  function tipContent(i) {
    const value = el("div", "tip-value");
    value.append(keyOf(i), S.dir[i] >= 0 ? facingOf(i) : "No direction");
    const note = noteOf(i);
    const detail = S.dir[i] >= 0 ? `${D.confidence[S.conf[i]]} confidence`
      : note ? note[0] : "no bearing";
    const root = el("div");
    root.append(value, el("div", "tip-label", `${streetOf(i)} · ${detail}`));
    const others = groupOf(i).length - 1;
    if (others) {
      root.append(el("div", "tip-more",
                     `+${others} more ${others === 1 ? "site" : "sites"} at this point`));
    }
    return root;
  }

  function siteDetails(i) {
    const section = el("section", "pop-site");
    section.append(el("p", "pop-title", placeOf(i)));
    if (S.n[i] > 1) {
      section.append(el("p", "pop-sub",
                        `${S.n[i].toLocaleString("en-AU")} addresses at this site`));
    }
    const dl = el("dl");
    const row = (term, ...value) => {
      const dd = el("dd");
      dd.append(...value);
      dl.append(el("dt", undefined, term), dd);
    };
    const away = S.dist[i] === null ? "" : `, ${S.dist[i].toFixed(1)} m away`;
    if (S.dir[i] >= 0) {
      const d = S.dir[i];
      const level = D.confidence[S.conf[i]];
      row("Faces", keyOf(i), `${D.names[d]} (${D.directions[d]}), ` +
                             `${S.bearing[i].toFixed(1)}° from true north`);
      row("Street", streetOf(i) + away);
      const share = D.agreement[level];
      const check = share === undefined ? []
        : [el("span", "note", `${Math.round(share * 100)}% of the ${level}-confidence sites ` +
                              "checked have the same 8-point direction as the NSW cadastre")];
      row("Confidence", level[0].toUpperCase() + level.slice(1), ...check);
    } else {
      row("Faces", keyOf(i), "No direction");
      row("Street", streetOf(i) + away);
    }
    section.append(dl);
    const note = noteOf(i);
    if (note) section.append(el("p", "pop-note", note[1]));
    return section;
  }

  function popupContent(i) {
    const root = el("div", "pop");
    root.append(...groupOf(i).map(siteDetails));
    return root;
  }

  // --- Hover and click: the nearest visible point within a forgiving radius ------------------
  let hovered = -1, selected = -1, pointer = null, pending = 0;

  function nearest(point, slop) {
    const f = frame(map), lim = Math.max(12, radius(map.getZoom()) + slop);
    let best = -1, bestD2 = lim * lim;
    for (let i = 0; i < N; i++) {
      if (!shown[S.dir[i] + 1]) continue;
      const dx = px[i] * f.k - f.x - point.x;
      if (dx > lim || dx < -lim) continue;
      const dy = py[i] * f.k - f.y - point.y;
      if (dy > lim || dy < -lim) continue;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
        bestD2 = d2;
        best = i;
      }
    }
    return best;
  }

  function syncRing() {
    const i = hovered >= 0 ? hovered : selected;
    if (i < 0) {
      focusRing.remove();
      return;
    }
    focusRing.setLatLng(latLngOf(i)).setRadius(radius(map.getZoom()) + 3.5);
    if (!map.hasLayer(focusRing)) focusRing.addTo(map);
  }

  function setHover(i) {
    if (i === hovered) return;
    hovered = i;
    mapEl.classList.toggle("is-pointing", i >= 0);
    if (i >= 0 && i !== selected) {
      tip.setLatLng(latLngOf(i)).setContent(tipContent(i));
      if (!map.hasLayer(tip)) tip.addTo(map);
    } else {
      tip.remove();
    }
    syncRing();
  }

  function openSite(i) {
    selected = i;
    tip.remove();
    setAbout(false);
    // Pan the popup (at most ~330 px wide, centred on the point) clear of the title panel.
    const at = map.latLngToContainerPoint(latLngOf(i));
    const panel = titleEl.getBoundingClientRect(), box = mapEl.getBoundingClientRect();
    const clear = at.x - 170 > panel.right - box.left;
    popup.options.autoPanPaddingTopLeft = L.point(24, clear ? 24 : panel.bottom - box.top + 12);
    popup.setLatLng(latLngOf(i)).setContent(popupContent(i)).openOn(map);
    syncRing();
  }

  map.on("mousemove", (e) => {
    const target = e.originalEvent && e.originalEvent.target;
    if (map.dragging.moving() || (target && target.closest && target.closest(".leaflet-control"))) {
      setHover(-1);
      return;
    }
    pointer = e.layerPoint;
    if (!pending) {
      pending = requestAnimationFrame(() => {
        pending = 0;
        setHover(nearest(pointer, 8));
      });
    }
  });
  mapEl.addEventListener("mouseleave", () => setHover(-1));
  map.on("movestart zoomstart", () => setHover(-1));
  map.on("zoomend", syncRing);
  map.on("click", (e) => {
    const i = nearest(e.layerPoint, coarse ? 16 : 8);
    if (i >= 0) openSite(i);
  });
  map.on("popupclose", (e) => {
    if (e.popup !== popup) return;
    selected = -1;
    syncRing();
  });

  // --- Legend: select a direction to show only it ----------------------------------------------
  const toggles = Array.from(legendEl.querySelectorAll("[data-dir]"));
  const statusEl = document.getElementById("legend-status");
  const statusText = document.getElementById("legend-status-text");
  let isolated = -1;

  function isolate(c) {
    isolated = c === isolated ? -1 : c;
    shown.forEach((_, k) => { shown[k] = isolated < 0 || k === isolated; });
    for (const t of toggles) {
      const k = Number(t.dataset.dir);
      t.classList.toggle("is-off", !shown[k]);
      t.setAttribute("aria-pressed", String(k === isolated));
    }
    legendEl.classList.toggle("is-isolated", isolated >= 0);
    statusEl.hidden = isolated < 0;
    statusText.textContent = isolated < 0 ? ""
      : isolated === 0 ? "Showing sites with no direction only."
      : `Showing ${D.names[isolated - 1].toLowerCase()}-facing only.`;
    sites.redraw();
    if (lines) lines.redraw();
    setHover(-1);
    if (selected >= 0 && !shown[S.dir[selected] + 1]) map.closePopup(popup);
  }
  for (const t of toggles) {
    t.addEventListener("click", () => isolate(Number(t.dataset.dir)));
    if (t.tagName !== "BUTTON") {
      t.addEventListener("keydown", (e) => {
        if (e.key !== "Enter" && e.key !== " ") return;
        e.preventDefault();
        isolate(Number(t.dataset.dir));
      });
    }
  }
  document.getElementById("show-all").addEventListener("click", () => isolate(isolated));

  // --- Search: every query word must start a word of "number street suburb" ---------------------
  const input = document.getElementById("find");
  const list = document.getElementById("find-list");
  const norm = (text) => text.toLowerCase().replace(/[^a-z0-9-]+/g, " ").trim();
  let keys = null, hits = [], active = -1;

  function search(query) {
    const words = norm(query).split(" ").filter(Boolean).map((w) => " " + w);
    if (norm(query).length < 2) return null;
    if (!keys) keys = S.addr.map((a, i) => " " + norm(`${a} ${D.localities[S.loc[i]]}`));
    const found = [];
    for (let i = 0; i < N && found.length < 8; i++) { // sites are sorted by street and number
      if (words.every((w) => keys[i].includes(w))) found.push(i);
    }
    return found;
  }

  function mark() {
    Array.from(list.children).forEach((li, n) => {
      li.setAttribute("aria-selected", String(n === active));
    });
    if (active >= 0) input.setAttribute("aria-activedescendant", `find-${active}`);
    else input.removeAttribute("aria-activedescendant");
  }

  function showResults(found) {
    hits = found || [];
    active = hits.length ? 0 : -1;
    list.textContent = "";
    hits.forEach((i, n) => {
      const li = el("li", undefined, S.addr[i]);
      li.id = `find-${n}`;
      li.setAttribute("role", "option");
      li.append(el("span", "loc", D.localities[S.loc[i]]));
      li.addEventListener("mousedown", (e) => {
        e.preventDefault();
        choose(n);
      });
      list.append(li);
    });
    if (found && !hits.length) list.append(el("li", "empty", "No matching address"));
    list.hidden = !found;
    input.setAttribute("aria-expanded", String(Boolean(found)));
    mark();
  }

  function choose(n) {
    const i = hits[n];
    if (i === undefined) return;
    showResults(null);
    input.value = placeOf(i);
    if (!shown[S.dir[i] + 1]) isolate(isolated);
    map.once("moveend", () => openSite(i));
    map.flyTo(latLngOf(i), Math.max(map.getZoom(), 18), { animate: !still, duration: 0.8 });
  }

  input.addEventListener("input", () => showResults(search(input.value)));
  input.addEventListener("blur", () => showResults(null));
  input.addEventListener("keydown", (e) => {
    if ((e.key === "ArrowDown" || e.key === "ArrowUp") && hits.length) {
      e.preventDefault();
      active = (active + (e.key === "ArrowDown" ? 1 : hits.length - 1)) % hits.length;
      mark();
    } else if (e.key === "Enter" && hits.length) {
      e.preventDefault();
      choose(Math.max(active, 0));
    } else if (e.key === "Escape") {
      showResults(null);
    }
  });

  // --- Small screens: the figures and credits fold away behind "About this map" -------------
  more.addEventListener("click", () => setAbout(!titleEl.classList.contains("is-open")));
})();
"""

_LANDING_HTML = """<!doctype html>
<html lang="en-AU">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Which way does every property face? {{PLACE}}</title>
<meta name="description" content="Street-frontage orientation for every principal G-NAF \
address in {{PLACE}}, from G-NAF and OpenStreetMap.">
<link rel="icon" href="{{FAVICON}}">
<style>
{{ROOT_CSS}}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
  font:16px/1.55 "Helvetica Neue",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:720px;margin:0 auto;padding:64px 20px 48px}
header{display:flex;gap:28px;align-items:center}
.compass{flex:none;width:120px;height:120px;overflow:visible}
.compass path{stroke:var(--surface);stroke-width:1.5}
.compass text{fill:var(--ink-2);text-anchor:middle;dominant-baseline:central;
  font-family:inherit}
.compass .cardinal{font-size:12px;font-weight:700}
.compass .intercardinal{font-size:10px}
h1{margin:0;font-size:34px;line-height:1.12;letter-spacing:-.02em;text-wrap:balance}
.place{margin:8px 0 0;font-size:18px;color:var(--ink-2)}
.lede{margin:28px 0 0;color:var(--ink-2)}
.actions{display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center;margin:28px 0 0}
.button{display:inline-block;padding:11px 18px;border-radius:8px;background:var(--ink);
  color:var(--surface);font-weight:600;text-decoration:none}
.button:hover{background:var(--ink-2)}
a{color:var(--ink);text-decoration-color:var(--baseline);text-underline-offset:3px}
a:hover{text-decoration-color:var(--ink)}
:focus-visible{outline:2px solid var(--ink);outline-offset:3px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:44px 0 0;padding:22px 0;
  border-top:1px solid var(--grid);border-bottom:1px solid var(--grid)}
.stat-value{font-size:28px;font-weight:600;line-height:1.2}
.stat-label{margin-top:4px;font-size:13px;line-height:1.4;color:var(--ink-3)}
.compare{margin:18px 0 0;font-size:14px;color:var(--ink-2)}
footer{margin:40px 0 0;font-size:12px;color:var(--ink-3)}
@media (max-width:560px){
  main{padding-top:40px}
  header{flex-direction:column;align-items:flex-start;gap:20px}
  .compass{width:96px;height:96px}
  h1{font-size:28px}
  .stats{grid-template-columns:1fr;gap:16px}
}
</style>
</head>
<body>
<main>
  <header>
    {{COMPASS}}
    <div>
      <h1>Which way does every property face?</h1>
      <p class="place">{{PLACE}}</p>
    </div>
  </header>
  <p class="lede">The compass direction of every principal G-NAF address, measured towards
    the nearest point on the OpenStreetMap street it is addressed to, and checked against
    property boundaries in the NSW cadastre.</p>
  <p class="actions">
    <a class="button" href="map/">Open the interactive map</a>
    <a href="{{REPO}}">Method, code and data on GitHub</a>
  </p>
  <section class="stats" aria-label="Key results">
    <div><div class="stat-value">{{ADDRESSES}}</div>
      <div class="stat-label">addresses at {{SITES}} sites (sharing one location, street
        and geocode type)</div></div>
    <div><div class="stat-value">{{AGREEMENT}}</div>
      <div class="stat-label">same 8-point direction as the cadastre ({{CHECKED}} sites
        checked)</div></div>
    <div><div class="stat-value">{{MEDIAN}}</div>
      <div class="stat-label">median difference from the cadastral frontage</div></div>
  </section>
  <p class="compare">The {{BASELINE_NAME}}, which this replaces, faces the nearest
    OpenStreetMap way of any kind. It gives the same 8-point direction for only {{BASELINE}}
    of the same sites: it often snaps to a footpath or driveway.</p>
  <footer>{{CREDITS}}</footer>
</main>
</body>
</html>
"""
