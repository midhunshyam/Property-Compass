"""Static maps: the Castlecrag hero, the Willoughby overview and north-facing houses by suburb.

The base layers come from the project's own data rather than web tiles, so the maps are
reproducible offline: water is the locality that no parcel covers (Middle Harbour), land is
the NSW parcel fabric, and streets are the vehicular OpenStreetMap ways. Everything is drawn
in the study CRS (GDA2020 / MGA zone 56, metres), so the scale bars are exact.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib import patheffects
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
from PIL import Image, PngImagePlugin
from shapely import affinity
from shapely.ops import polylabel

from property_orientation.orientation import FRONTAGE_HIGHWAYS
from property_orientation.viz import style
from property_orientation.viz.data import FigureData

Bounds = tuple[float, float, float, float]
LabelSize = tuple[float, float, float, float]
"""Width, height, name height and value height of a name-over-value label, in map units."""

HERO_LOCALITY = "CASTLECRAG"
MIN_STANDALONE = 100
"""Suburbs with fewer standalone addresses than this get no north-facing share."""

ATTRIBUTION = f"{style.ATTRIBUTION} · Boundaries: © Geoscape Australia (CC BY 4.0)"
CHOROPLETH_ATTRIBUTION = " · ".join(
    part for part in ATTRIBUTION.split(" · ") if not part.startswith("Parcels")
)
"""The choropleth draws no parcels, so it does not credit them."""

_STREETS = FRONTAGE_HIGHWAYS | {
    "motorway", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}  # fmt: skip
_MAJOR_STREETS = frozenset({"motorway", "trunk", "primary", "secondary"})
_STREET_WIDTH_PT = {"motorway": 1.5, "trunk": 1.5, "primary": 1.3, "secondary": 1.2}
_MINOR_STREET_PT = 0.7
_WHITE = "#ffffff"
_HALO = [patheffects.withStroke(linewidth=2.6, foreground=style.SURFACE)]
_DOT_HALO = [patheffects.withStroke(linewidth=4.0, foreground=style.SURFACE)]
"""A wider halo for names over the overview's dots, so no dot touches a letter."""
_NAME_KW = {"fontsize": 8.5, "ha": "center", "va": "center", "linespacing": 1.05}
_VALUE_KW = {"fontsize": 10, "fontweight": "bold", "ha": "center", "va": "center"}
_WIDTH_IN, _HEADER_IN, _FOOTER_IN = 10.0, 1.52, 0.3
_COMPASS_IN = 1.45
"""Side of the compass inset; its hole (0.26 of the side) must clear the word "Facing"."""
_ELONGATED = 2.0
"""Suburbs whose minimum rotated rectangle is at least this elongated may take a label
rotated along their long axis."""
_TITLE_PT, _HERO_TITLE_PT, _SUBTITLE_PT = 15.0, 17.0, 10.5
"""Titles match the charts (15 pt); the README's opening image gets a larger one."""
_ARROW = {"length": 20.0, "shaft": 2.8, "head_width": 10.0, "head_length": 8.5}
"""Facing-arrow geometry on the hero map, in metres. A head 10 m wide leaves a gap between the
arrows of neighbouring 15 m lots."""


# ------------------------------------------------------------------------------ layers


def _is_water(data: FigureData, max_parcel_cover: float = 0.05) -> np.ndarray:
    """Which localities parcels (almost) do not cover: in Willoughby, Middle Harbour."""
    parcels = data.parcels.geometry.to_numpy()
    tree = shapely.STRtree(parcels)
    cover = []
    for geom in data.localities.geometry:
        hits = parcels[tree.query(geom, predicate="intersects")]
        cover.append(shapely.area(shapely.intersection(hits, geom)).sum() / geom.area)
    return np.asarray(cover) < max_parcel_cover


def _polygonal(geom: shapely.Geometry) -> shapely.Geometry:
    """The polygon parts of ``geom`` (clipping can leave stray lines and points)."""
    parts = shapely.get_parts(geom)
    return shapely.multipolygons(parts[np.isin(shapely.get_type_id(parts), (3, 6))])


def _suburbs(data: FigureData, min_area_m2: float = 50_000.0) -> gpd.GeoDataFrame:
    """Suburbs clipped to the LGA, without slivers, with a display ``name``."""
    lga = data.boundary.geometry.union_all()
    clipped = data.localities.assign(
        geometry=[_polygonal(g) for g in data.localities.intersection(lga)]
    )
    clipped = clipped[clipped.area >= min_area_m2].copy()
    clipped["name"] = clipped["locality_name"].str.title()
    return clipped.reset_index(drop=True)


def _streets(data: FigureData, bounds: Bounds, classes: Iterable[str]) -> gpd.GeoDataFrame:
    """OSM ways of the given ``highway`` classes inside ``bounds``, without unnamed driveways."""
    xmin, ymin, xmax, ymax = bounds
    roads = data.roads.cx[xmin:xmax, ymin:ymax]
    driveway = roads["highway"].eq("service") & roads["name"].isna()
    return roads[roads["highway"].isin(set(classes)) & ~driveway]


def _padded(bounds: Sequence[float], m: float) -> Bounds:
    """``bounds`` grown by ``m`` map units on every side (shrunk if ``m`` is negative)."""
    xmin, ymin, xmax, ymax = bounds
    return xmin - m, ymin - m, xmax + m, ymax + m


def _has_hero(data: FigureData) -> bool:
    """Whether the area's localities include the hero locality (Castlecrag, in Willoughby)."""
    return bool(data.localities["locality_name"].eq(HERO_LOCALITY).any())


def _hero_extent(data: FigureData, pad: float = 0.06) -> Bounds:
    """Bounds of the hero locality, padded by a share of its longer side."""
    if not _has_hero(data):
        raise ValueError(
            f"no {HERO_LOCALITY.title()} locality in {data.paths.area.lga}: the hero map is "
            "drawn for the Willoughby showcase"
        )
    loc = data.localities
    geom = loc.loc[loc["locality_name"].eq(HERO_LOCALITY), "geometry"].union_all()
    xmin, ymin, xmax, ymax = geom.bounds
    return _padded(geom.bounds, pad * max(xmax - xmin, ymax - ymin))


# ------------------------------------------------------------------------------ furniture


def _map_axes(fig: plt.Figure, rect: Sequence[float], bounds: Bounds) -> plt.Axes:
    """Axis-free map axes showing exactly ``bounds``."""
    ax = fig.add_axes(rect)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def _map_figure(bounds: Bounds) -> tuple[plt.Figure, plt.Axes]:
    """A figure holding a full-width map of ``bounds`` below a header band."""
    xmin, ymin, xmax, ymax = bounds
    map_h = _WIDTH_IN * (ymax - ymin) / (xmax - xmin)
    height = _HEADER_IN + map_h + _FOOTER_IN
    fig = plt.figure(figsize=(_WIDTH_IN, height))
    return fig, _map_axes(fig, (0, _FOOTER_IN / height, 1, map_h / height), bounds)


def _fitted(fig: plt.Figure, text: str, size: float, width_in: float, **kw: Any) -> float:
    """``size``, or the smaller size at which the widest line of ``text`` fits ``width_in``.

    The layout is tuned for Helvetica. With a wider fallback font (DejaVu Sans on most Linux
    machines) a line would otherwise run under the compass or past the edge of the figure.
    """
    artist = fig.text(0, 0, text, fontsize=size, **kw)
    width = artist.get_window_extent().width / fig.dpi
    artist.remove()
    return size if width <= width_in else size * width_in / width


def _header(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    title_size: float = _TITLE_PT,
    compass: bool = False,
) -> None:
    """Title and subtitle along the top, and optionally the compass legend at the top right."""
    w, h = fig.get_size_inches()
    room = w - (_COMPASS_IN + 0.25 if compass else 0.0)  # text stays clear of the compass
    title_size = _fitted(fig, title, title_size, room, fontweight="bold")
    subtitle_size = _fitted(fig, subtitle, _SUBTITLE_PT, room, linespacing=1.5)
    top = (h - 0.12) / h
    fig.text(
        0, top, title, ha="left", va="top", fontsize=title_size, fontweight="bold",
        color=style.INK,
    )  # fmt: skip
    fig.text(
        0, top - (title_size * 1.3 / 72 + 0.12) / h, subtitle, ha="left", va="top",
        fontsize=subtitle_size, color=style.INK_SECONDARY, linespacing=1.5,
    )  # fmt: skip
    if compass:
        side = _COMPASS_IN
        rect = ((w - side) / w, (h - side - 0.06) / h, side / w, side / h)
        style.compass_legend(fig.add_axes(rect))


def _scale_bar(ax: plt.Axes, length_m: float, x: float, y: float) -> None:
    """A scale bar whose left end is at axes fraction ``(x, y)``."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx, by = x0 + x * (x1 - x0), y0 + y * (y1 - y0)
    tick = 0.008 * (y1 - y0)
    kw = {"color": style.INK_SECONDARY, "solid_capstyle": "butt", "zorder": 20}
    ax.plot([bx, bx + length_m], [by, by], lw=1.6, path_effects=_HALO, **kw)
    for tx in (bx, bx + length_m / 2, bx + length_m):
        ax.plot([tx, tx], [by, by + tick], lw=0.8, **kw)
    unit = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    for tx, text in ((bx, "0"), (bx + length_m, unit)):
        ax.text(
            tx, by - tick, text, ha="center", va="top", fontsize=8.5, color=style.INK_SECONDARY,
            path_effects=_HALO, zorder=20,
        )  # fmt: skip


def _save(fig: plt.Figure, out: Path) -> Path:
    """Save in the house style, then re-encode as a 256-colour PNG.

    The maps are flat colour, so median-cut quantisation keeps the fills unchanged (only
    anti-aliased edges are merged) at about a third of the file size.
    """
    path = style.save(fig, out)
    with Image.open(path) as im:
        info = PngImagePlugin.PngInfo()
        for key, value in im.text.items():
            info.add_text(key, value)
        rgb = im.convert("RGB")
    palette = rgb.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette.save(path, optimize=True, pnginfo=info)
    return path


# ------------------------------------------------------------------------------ labels


def _luminance(colour: str) -> float:
    """WCAG relative luminance of a colour."""
    c = np.asarray(to_rgb(colour))
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(linear @ [0.2126, 0.7152, 0.0722])


def _contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two colours."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _ink_on(fill: str) -> str:
    """White or ink, whichever contrasts more with ``fill``."""
    return _WHITE if _contrast(_WHITE, fill) > _contrast(style.INK, fill) else style.INK


def _text_size(ax: plt.Axes, text: str, **kw: Any) -> tuple[float, float]:
    """Width and height of ``text`` in data units (for equal-aspect map axes)."""
    artist = ax.text(0, 0, text, **kw)
    box = artist.get_window_extent()
    artist.remove()
    per_pixel = (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.get_window_extent().width
    return box.width * per_pixel, box.height * per_pixel


def _box(centre: Sequence[float], w: float, h: float, angle: float = 0.0) -> shapely.Geometry:
    """A ``w`` by ``h`` rectangle centred on ``centre``, rotated by ``angle`` degrees."""
    box = affinity.rotate(shapely.box(-w / 2, -h / 2, w / 2, h / 2), angle, origin=(0, 0))
    return affinity.translate(box, *centre)


def _free(box: shapely.Geometry, taken: Sequence[shapely.Geometry]) -> bool:
    """Whether ``box`` overlaps none of the label footprints in ``taken``."""
    return not any(box.intersects(t) for t in taken)


def _interior_points(geom: shapely.Geometry, n: int = 40) -> np.ndarray:
    """Candidate label anchors inside ``geom``, deepest (furthest from the edge) first."""
    parts = shapely.get_parts(geom)
    poles = np.array([polylabel(part, tolerance=1.0) for part in parts])
    deepest = poles[np.argmax(shapely.distance(shapely.boundary(parts), poles))]
    centre = shapely.get_coordinates(deepest)
    xmin, ymin, xmax, ymax = geom.bounds
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
    gx, gy = gx.ravel(), gy.ravel()
    inside = shapely.contains_xy(geom, gx, gy)
    points = shapely.points(gx[inside], gy[inside])
    depth = shapely.distance(shapely.boundary(geom), points)
    return np.vstack([centre, shapely.get_coordinates(points[np.argsort(-depth, kind="stable")])])


def _first_fit(
    anchors: np.ndarray,
    size: tuple[float, float],
    accept: Callable[[shapely.Geometry], bool],
) -> tuple[float, float] | None:
    """The first anchor whose (unrotated) label box ``accept`` accepts."""
    for x, y in anchors:
        if accept(_box((x, y), *size)):
            return float(x), float(y)
    return None


def _place_name(
    ax: plt.Axes,
    geom: shapely.Geometry,
    name: str,
    taken: list[shapely.Geometry],
    *,
    spill: shapely.Geometry | None = None,
    avoid: shapely.STRtree | None = None,
    halo: Sequence[patheffects.AbstractPathEffect] = tuple(_HALO),
    **kw: Any,
) -> None:
    """Write ``name`` at the deepest free spot inside ``geom``, on one line or else on two.

    With ``avoid`` (a tree of data points), the spot in the deeper half of ``geom`` that
    covers the fewest points wins instead, so the label hides as little data as it can.
    A name that fits nowhere inside ``geom`` may spill out of it, as little as possible and
    only within ``spill`` (for example the LGA, so that no label crosses its outline);
    without ``spill`` it is left out.
    """
    anchors = _interior_points(geom)
    if avoid is not None:
        anchors = anchors[: max(1, len(anchors) // 2)]
    for region in (geom, spill):
        if region is None:
            continue
        for text in dict.fromkeys((name, _wrap(name))):  # one line first
            w, h = _text_size(ax, text, **kw)
            boxes = [_box(anchor, w * 1.08, h * 1.3) for anchor in anchors]
            covered = np.zeros(len(boxes))
            if avoid is not None:
                covered = np.array([len(avoid.query(b)) for b in boxes])
            spilled = np.zeros(len(boxes))
            if region is spill:
                spilled = np.array([round(b.difference(geom).area / b.area, 1) for b in boxes])
            order = np.lexsort((covered, spilled))  # least spilled, then fewest points covered
            for i in order:
                if region.contains(boxes[i]) and _free(boxes[i], taken):
                    ax.text(*anchors[i], text, path_effects=list(halo), zorder=9, **kw)
                    taken.append(_box(anchors[i], w, h))
                    return


def _wrap(name: str) -> str:
    """Break a multi-word name at the space nearest its middle."""
    spaces = [i for i, c in enumerate(name) if c == " "]
    if not spaces:
        return name
    i = min(spaces, key=lambda s: abs(s - len(name) / 2))
    return f"{name[:i]}\n{name[i + 1 :]}"


def _long_axis(geom: shapely.Geometry) -> tuple[float, float]:
    """Angle (degrees, -90 to 90) and elongation of the minimum rotated rectangle."""
    with np.errstate(invalid="ignore"):  # GEOS flags collinear edges, yet returns the rectangle
        rectangle = shapely.minimum_rotated_rectangle(geom)
    corners = shapely.get_coordinates(rectangle)[:3]
    edges = np.diff(corners, axis=0)
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    dx, dy = edges[np.argmax(lengths)]
    angle = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0 - 90.0
    return float(angle), float(lengths.max() / max(lengths.min(), 1e-9))


def _label_size(ax: plt.Axes, name: str, value: str) -> LabelSize:
    """Size of ``name`` over ``value`` (see ``LabelSize``), in map units."""
    nw, nh = _text_size(ax, name, **_NAME_KW)
    vw, vh = _text_size(ax, value, **_VALUE_KW) if value else (0.0, 0.0)
    return max(nw, vw), nh + 1.12 * vh, nh, vh


def _draw_label(
    ax: plt.Axes,
    centre: Sequence[float],
    angle: float,
    name: str,
    value: str,
    size: LabelSize,
    *,
    ink: str,
    halo: str,
) -> shapely.Geometry:
    """Draw ``name`` over ``value`` centred on ``centre``; return the label's footprint."""
    (x, y), (w, h, nh, vh) = centre, size
    nx, ny = -np.sin(np.radians(angle)), np.cos(np.radians(angle))
    # A halo in the colour underneath hides any suburb border that runs under the text.
    effects = [patheffects.withStroke(linewidth=2.4, foreground=halo)]
    lines = [(name, h / 2 - nh / 2, _NAME_KW), (value, vh / 2 - h / 2, _VALUE_KW)]
    for text, offset, kw in lines[: 2 if value else 1]:
        ax.text(
            x + offset * nx, y + offset * ny, text, rotation=angle, rotation_mode="anchor",
            color=ink, zorder=6, path_effects=effects, **kw,
        )  # fmt: skip
    return _box(centre, w, h, angle)


def _label_suburbs(
    ax: plt.Axes,
    suburbs: gpd.GeoDataFrame,
    values: Sequence[str],
    fills: Sequence[str],
    inks: Sequence[str],
    outside: shapely.Geometry,
    *,
    max_overflow: float = 0.3,
) -> None:
    """Label each suburb with its name over its value, without overlaps.

    A label goes where it fits inside its suburb (as is, wrapped, or along a long thin
    suburb); else where it overflows least onto fills it still contrasts with; else at the
    nearest free spot in ``outside``, with a leader line.
    """
    land = suburbs.geometry.union_all()
    regions = [*zip(suburbs.geometry, fills, strict=True), (None, style.SURFACE)]
    taken: list[shapely.Geometry] = []

    def legible(box: shapely.Geometry, ink: str) -> bool:
        for geom, fill in regions:
            part = box.difference(land) if geom is None else box.intersection(geom)
            if part.area > 0.03 * box.area and _contrast(ink, fill) < 3.0:
                return False
        return True

    for i in np.argsort(suburbs.area.to_numpy(), kind="stable"):  # tightest fits first
        geom, name = suburbs.geometry.iloc[i], suburbs["name"].iloc[i]
        value, ink, fill = values[i], inks[i], fills[i]
        inner = geom.buffer(-30.0)  # clear of the white borders
        angle, elongation = _long_axis(geom)
        rotated = [(name, angle)] if elongation >= _ELONGATED else []
        forms = [(name, 0.0), (_wrap(name), 0.0), *rotated]
        anchors = _interior_points(geom)[:400]
        best = None
        for rank, (text, a) in enumerate(forms):
            size = _label_size(ax, text, value)
            for x, y in anchors:
                box = _box((x, y), size[0] * 1.08, size[1] * 1.08, a)
                if not _free(box, taken):
                    continue
                overflow = round(box.difference(inner).area / box.area, 2)
                if overflow > max_overflow or (overflow > 0 and not legible(box, ink)):
                    continue
                if best is None or (overflow, rank) < best[0]:
                    best = (overflow, rank), ((x, y), a, text, size)
                if overflow == 0:
                    break
        if best is not None:
            centre, a, text, size = best[1]
            taken.append(_draw_label(ax, centre, a, text, value, size, ink=ink, halo=fill))
            continue

        # Callout: the free spot outside the LGA nearest the suburb, with a leader line.
        size = _label_size(ax, name, value)
        pad = (size[0] * 1.15, size[1] * 1.15)
        xmin, ymin, xmax, ymax = outside.bounds
        gx, gy = np.meshgrid(np.linspace(xmin, xmax, 80), np.linspace(ymin, ymax, 50))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        grid = grid[np.argsort(np.hypot(*(grid - anchors[0]).T), kind="stable")]
        spot = _first_fit(grid, pad, lambda b: outside.contains(b) and _free(b, taken))
        if spot is None:
            continue
        ink = style.INK if ink == _WHITE else ink
        taken.append(_draw_label(ax, spot, 0.0, name, value, size, ink=ink, halo=style.SURFACE))
        leader = shapely.shortest_line(shapely.Point(anchors[0]), _box(spot, *pad))
        (x0, y0), (x1, y1) = shapely.get_coordinates(leader)
        ax.plot([x0, x1], [y0, y1], color=style.INK_MUTED, lw=0.7, zorder=6)
        ax.plot(x0, y0, "o", ms=2.6, color=style.INK_SECONDARY, zorder=6)


# ------------------------------------------------------------------------------ hero map


def _parcel_orientation(sites: pd.DataFrame) -> pd.Series:
    """Orientation per parcel (indexed by parcel row): the one most of its addresses face.

    Ties go to the direction of the lowest ``site_id``, so the result is deterministic.
    """
    inside = sites[(sites["ref_parcel"] >= 0) & sites["orientation"].notna()]
    votes = (
        inside.groupby(["ref_parcel", "orientation"], as_index=False)
        .agg(addresses=("n_addresses", "sum"), first_site=("site_id", "min"))
        .sort_values(["ref_parcel", "addresses", "first_site"], ascending=[True, False, True])
        .drop_duplicates("ref_parcel")
    )
    return pd.Series(votes["orientation"].to_numpy(), index=votes["ref_parcel"].to_numpy())


def _arrow_vertices(
    sites: pd.DataFrame,
    *,
    length: float,
    shaft: float,
    head_width: float,
    head_length: float,
    max_share: float = 0.85,
) -> np.ndarray:
    """Vertices ``(n, 7, 2)`` of arrows from each site towards its street point (map units).

    Arrows are ``length`` long, or ``max_share`` of the way to the street when that is
    closer, so an arrow never reaches past the street. Sites on their street point get none.
    """
    sites = sites[np.hypot(sites["street_x"] - sites["x"], sites["street_y"] - sites["y"]) > 0]
    x, y = sites["x"].to_numpy(), sites["y"].to_numpy()
    dx, dy = sites["street_x"].to_numpy() - x, sites["street_y"].to_numpy() - y
    d = np.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    n = np.minimum(length, max_share * d)
    h = np.minimum(head_length, 0.6 * n)
    zero = np.zeros_like(n)
    along = np.stack([zero, n - h, n - h, n, n - h, n - h, zero], axis=1)
    across = np.array([-shaft, -shaft, -head_width, 0.0, head_width, shaft, shaft]) / 2
    vx = x[:, None] + along * ux[:, None] - across * uy[:, None]
    vy = y[:, None] + along * uy[:, None] + across * ux[:, None]
    return np.stack([vx, vy], axis=2)


def _draw_arrows(ax: plt.Axes, verts: np.ndarray, ring_pt: float = 1.0) -> None:
    """Ink arrows on a surface-coloured ring, so they read on every direction colour."""
    ring = PolyCollection(
        verts, facecolors=style.SURFACE, edgecolors=style.SURFACE, linewidths=ring_pt,
        joinstyle="round", zorder=6,
    )  # fmt: skip
    ax.add_collection(ring)
    ax.add_collection(PolyCollection(verts, facecolors=style.INK, linewidths=0, zorder=7))


def hero_map(data: FigureData, out: Path) -> Path:
    """Castlecrag: every parcel coloured by the direction its property faces, with arrows."""
    bounds = _hero_extent(data)
    xmin, ymin, xmax, ymax = bounds
    sites = data.sites.cx[xmin:xmax, ymin:ymax]
    focus = sites[sites["locality"].eq(HERO_LOCALITY)]
    oriented = focus[focus["orientation"].notna()]
    is_water = _is_water(data)
    fig, ax = _map_figure(bounds)

    water = data.localities[is_water].clip(bounds)
    if not water.empty:
        water.plot(ax=ax, color=style.WATER, linewidth=0, zorder=1)
    parcels = data.parcels.cx[xmin:xmax, ymin:ymax]
    # Largest first: a site's parcel is the smallest one containing it, and a near-duplicate
    # larger parcel drawn later would hide its colour.
    parcels = parcels.iloc[np.argsort(-parcels.area.to_numpy(), kind="stable")]
    fill = _parcel_orientation(focus).reindex(parcels.index).map(style.DIRECTION_COLORS)
    colours = fill.fillna(style.PARCEL_FILL).to_numpy()
    parcels.plot(ax=ax, color=colours, edgecolor=_WHITE, linewidth=0.35, zorder=2)
    streets = _streets(data, bounds, _STREETS)
    widths = streets["highway"].map(_STREET_WIDTH_PT).fillna(_MINOR_STREET_PT).to_numpy()
    streets.plot(ax=ax, color=style.ROAD, linewidth=widths, zorder=4)
    _draw_arrows(ax, _arrow_vertices(oriented, **_ARROW))

    # Muted names for the neighbouring suburbs and the harbour, where they fit.
    view = shapely.box(*bounds)
    taken: list[shapely.Geometry] = []
    for (_, row), water in zip(data.localities.iterrows(), is_water, strict=True):
        part = _polygonal(row.geometry.intersection(view)).buffer(-40.0)
        if row["locality_name"] == HERO_LOCALITY or part.area < 0.02 * view.area:
            continue
        _place_name(
            ax, part, row["locality_name"].title(), taken, fontsize=9,
            color=style.INK_MUTED, ha="center", va="center",
            fontstyle="italic" if water else "normal",
        )  # fmt: skip

    _scale_bar(ax, 200, 0.025, 0.04)
    # The counting unit is the site (addresses sharing a location, street and geocode type), as
    # on the overview map and the web map; the roses count standalone addresses instead.
    _header(
        fig,
        "Which way does every property face?",
        f"{HERO_LOCALITY.title()}, {data.summary['area']['state']} · "
        f"{int(oriented['n_addresses'].sum()):,} addresses at {len(oriented):,} sites (addresses "
        "sharing a location, street and geocode type).\nColour shows the direction each site "
        "faces; its arrow points to the street it is addressed to.\nWalter Burley Griffin's "
        "1920s streets follow "
        "the contours of a Middle Harbour peninsula, so houses face every way.",
        title_size=_HERO_TITLE_PT,
        compass=True,
    )
    style.attribution(fig, ATTRIBUTION, y=0)
    return _save(fig, out)


# ------------------------------------------------------------------------------ overview


def overview_map(data: FigureData, out: Path) -> Path:
    """Every property in the LGA as a dot coloured by the direction it faces."""
    lga = data.boundary
    bounds = _padded(lga.total_bounds, 200.0)
    xmin, ymin, xmax, ymax = bounds
    sites = data.sites
    oriented = sites[sites["orientation"].notna()].sort_values("site_id")
    cardinal = oriented["orientation"].isin(["N", "E", "S", "W"]).mean()
    area = data.summary["area"]
    fig, ax = _map_figure(bounds)

    # Context: the parcel fabric outside the LGA (its streets show as gaps), water, the LGA.
    outside = data.parcels.cx[xmin:xmax, ymin:ymax]
    outside.plot(ax=ax, color=style.PARCEL_FILL, linewidth=0, zorder=0)
    water = data.localities[_is_water(data)].clip(bounds)
    major = _streets(data, bounds, _MAJOR_STREETS).clip(lga.geometry.union_all())
    if not water.empty:  # a small or inland area may have no water, or no major road
        water.plot(ax=ax, color=style.WATER, linewidth=0, zorder=1)
    lga.plot(ax=ax, color=style.SURFACE, linewidth=0, zorder=1)
    if not major.empty:
        major.plot(ax=ax, color=style.ROAD, linewidth=0.8, zorder=2)
    suburbs = _suburbs(data)
    suburbs.boundary.plot(ax=ax, color=style.BASELINE, linewidth=0.6, zorder=3)
    colours = style.direction_colors(oriented["orientation"])
    ax.scatter(oriented["x"], oriented["y"], s=4.0, c=colours, linewidths=0, zorder=4)
    lga.boundary.plot(ax=ax, color=style.INK_SECONDARY, linewidth=0.9, zorder=5)

    taken: list[shapely.Geometry] = []
    if _has_hero(data):  # outline the extent of the hero (detail) map
        hx0, hy0, hx1, hy1 = _hero_extent(data)
        detail = Rectangle(
            (hx0, hy0), hx1 - hx0, hy1 - hy0, fill=False, ec=style.INK_SECONDARY, lw=1.0, zorder=9
        )
        ax.add_patch(detail)
        kw = {"fontsize": 8.5, "color": style.INK_SECONDARY, "path_effects": _HALO}
        ax.text(hx1, hy1 + 60, "Detail map", ha="right", va="bottom", zorder=9, **kw)
        w, h = _text_size(ax, "Detail map", **kw)
        taken += [
            shapely.box(hx0, hy0, hx1, hy1).exterior.buffer(30.0),
            shapely.box(hx1 - w, hy1 + 60, hx1, hy1 + 60 + h),
        ]
    dots = shapely.STRtree(shapely.points(oriented["x"], oriented["y"]))
    inside_lga = lga.geometry.union_all().buffer(-60.0)
    for i in np.argsort(-suburbs.area.to_numpy(), kind="stable"):
        _place_name(
            ax, suburbs.geometry.iloc[i], suburbs["name"].iloc[i], taken, spill=inside_lga,
            avoid=dots, halo=_DOT_HALO, fontsize=9.5, color=style.INK_SECONDARY, ha="center",
            va="center",
        )  # fmt: skip

    _scale_bar(ax, 1000, 0.025, 0.05)
    _header(
        fig,
        f"Street grids make {cardinal:.0%} of properties face north, east, south or west",
        f"{area['lga']} LGA, {area['state']} · {int(oriented['n_addresses'].sum()):,} addresses "
        f"at {len(oriented):,} sites, each site a dot coloured by the direction it faces.\n"
        "If facing directions were spread evenly, only half would face N, E, S or W.",
        compass=True,
    )
    style.attribution(fig, ATTRIBUTION, y=0)
    return _save(fig, out)


# ------------------------------------------------------------------------------ choropleth


def _class_breaks(values: np.ndarray, k: int = 4, step: float = 0.05) -> np.ndarray:
    """Up to ``k - 1`` breaks on multiples of ``step`` that minimise within-class variance.

    Values that no multiple of ``step`` separates (one value, or none) get no breaks.
    """
    v = np.sort(np.asarray(values, float))
    if not len(v):
        return np.array([])
    lo, hi = int(np.floor(v[0] / step)) + 1, int(np.ceil(v[-1] / step))
    candidates = np.round(np.arange(lo, hi) * step, 6)
    for n in range(min(k - 1, len(candidates)), 0, -1):
        best, best_cost = None, np.inf
        for breaks in itertools.combinations(candidates, n):
            cls = np.searchsorted(breaks, v, side="right")
            if len(np.unique(cls)) <= n:
                continue
            cost = sum(np.var(v[cls == c]) * np.sum(cls == c) for c in np.unique(cls))
            if cost < best_cost - 1e-12:
                best, best_cost = np.asarray(breaks), cost
        if best is not None:
            return best
    return np.array([])


def _class_labels(breaks: np.ndarray, values: np.ndarray) -> list[str]:
    """Legend labels in whole percent, for ``values`` rounded to whole percent.

    Without breaks there is one class, labelled with the range of ``values``.
    """
    if not len(breaks):
        lo, hi = (round(v * 100) for v in (np.min(values), np.max(values)))
        return [f"{lo}%" if lo == hi else f"{lo}–{hi}%"]
    pct = [round(b * 100) for b in breaks]
    middle = [f"{a}–{b - 1}%" for a, b in itertools.pairwise(pct)]
    return [f"under {pct[0]}%", *middle, f"{pct[-1]}% or more"]


def _ranked_bars(bx: plt.Axes, ranked: pd.DataFrame, overall: float) -> None:
    """Horizontal bars of ``share_northerly``, with the whole-LGA share as a reference line."""
    y = np.arange(len(ranked))
    bx.barh(y, ranked["share_northerly"], height=0.5, color=style.SEQUENTIAL[3], zorder=2)
    halo = [patheffects.withStroke(linewidth=3.0, foreground=style.SURFACE)]
    for yi, v in zip(y, ranked["share_northerly"], strict=True):
        bx.text(
            v + 0.008, yi, f"{v:.0%}", va="center", ha="left", fontsize=8.5, color=style.INK,
            path_effects=halo, zorder=3,
        )  # fmt: skip
    bx.set_yticks(y, ranked["name"], fontsize=9, color=style.INK_SECONDARY)
    bx.tick_params(axis="y", length=0, pad=6)
    bx.set_xticks([])
    bx.spines[["bottom", "top", "right"]].set_visible(False)
    # Behind the bars and the value labels' halos, so it never crosses a number.
    top = len(ranked) - 0.45
    bx.plot(
        [overall] * 2, [-0.6, top], color=style.INK_MUTED, lw=0.8, solid_capstyle="butt",
        zorder=1,
    )  # fmt: skip
    bx.text(
        overall, top + 0.15, f"Whole LGA {overall:.0%}", ha="center", va="bottom",
        fontsize=8, color=style.INK_SECONDARY,
    )  # fmt: skip
    bx.set_xlim(0, max(ranked["share_northerly"].max(), overall, 0.05) * 1.2)
    bx.set_ylim(-0.6, len(ranked) + 0.25)


def northerly_choropleth(data: FigureData, out: Path) -> Path:
    """Share of standalone addresses (mostly houses) facing N, NE or NW, by suburb."""
    summary = data.suburb_summary.set_index("locality")
    suburbs = _suburbs(data).join(summary, on="locality_name", how="inner").reset_index(drop=True)
    enough = (suburbs["standalone"] >= MIN_STANDALONE).to_numpy()
    if not enough.any():
        raise ValueError(
            f"no suburb of {data.paths.area.lga} has {MIN_STANDALONE} or more standalone "
            "addresses, so there is no share to map"
        )
    shown = suburbs["share_northerly"].round(2)  # classify the values the labels show

    breaks = _class_breaks(shown[enough].to_numpy())
    if len(breaks):
        steps = np.linspace(0, len(style.SEQUENTIAL) - 1, len(breaks) + 1)
    else:  # a single class (one suburb, or shares too close to split) takes the middle shade
        steps = np.array([(len(style.SEQUENTIAL) - 1) / 2])
    ramp = [style.SEQUENTIAL[round(s)] for s in steps]
    cls = np.searchsorted(breaks, shown.to_numpy(), side="right")
    fills = [ramp[c] if ok else style.PARCEL_FILL for c, ok in zip(cls, enough, strict=True)]

    lga = data.boundary
    bounds = _padded(lga.total_bounds, 150.0)
    xmin, ymin, xmax, ymax = bounds
    map_w, gap, names_w, bar_w = 6.4, 0.3, 1.25, 2.05
    map_h = map_w * (ymax - ymin) / (xmax - xmin)
    header, footer = 1.35, 0.95
    width, height = map_w + gap + names_w + bar_w, header + map_h + footer
    fig = plt.figure(figsize=(width, height))
    ax = _map_axes(fig, (0, footer / height, map_w / width, map_h / height), bounds)

    suburbs.plot(ax=ax, color=fills, edgecolor=style.SURFACE, linewidth=1.4, zorder=2)
    hatch = {"facecolor": style.PARCEL_FILL, "edgecolor": style.BASELINE, "hatch": "/////"}
    if not enough.all():
        suburbs[~enough].plot(ax=ax, linewidth=0, zorder=3, **hatch)

    values = [f"{v:.0%}" if ok else "" for v, ok in zip(shown, enough, strict=True)]
    inks = [_ink_on(f) if ok else style.INK_SECONDARY for f, ok in zip(fills, enough, strict=True)]
    margin = 0.015 * (xmax - xmin)
    lga_zone = lga.geometry.union_all().buffer(margin)
    outside = shapely.box(*_padded(bounds, -margin)).difference(lga_zone)
    _label_suburbs(ax, suburbs, values, fills, inks, outside)

    labels = _class_labels(breaks, shown[enough].to_numpy())
    handles = [Patch(facecolor=c, edgecolor="none") for c in ramp]
    if not enough.all():
        handles.append(Patch(linewidth=0, **hatch))
        labels.append(f"too few houses (under {MIN_STANDALONE})")
    legend = ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.0, -0.03), ncol=len(handles),
        frameon=False, fontsize=8.5, handlelength=1.5, handleheight=1.1, columnspacing=1.2,
        handletextpad=0.5, title="Share of houses facing N, NE or NW", title_fontsize=8.5,
        alignment="left", labelcolor=style.INK_SECONDARY,
    )  # fmt: skip
    legend.get_title().set_color(style.INK)

    ranked = suburbs[enough].sort_values("share_northerly", kind="stable")
    left = (map_w + gap + names_w) / width
    bx = fig.add_axes((left, footer / height, bar_w / width, map_h / height))
    _ranked_bars(bx, ranked, data.summary["orientation_share_standalone"]["share_northerly"])

    low, high = ranked.iloc[0], ranked.iloc[-1]
    title = (
        f"Houses facing N, NE or NW range from {low['share_northerly']:.0%} in {low['name']} "
        f"to {high['share_northerly']:.0%} in {high['name']}"
        if len(ranked) > 1
        else f"{high['share_northerly']:.0%} of houses in {high['name']} face N, NE or NW"
    )
    _header(
        fig,
        title,
        "Share of standalone addresses (mostly houses) whose street frontage faces north, "
        "north-east or north-west, by suburb.\nIn the southern hemisphere outside the tropics, "
        "the midday sun is always in the northern sky, so north-facing frontages get the most sun.",
    )
    style.attribution(fig, CHOROPLETH_ATTRIBUTION, y=0)
    return _save(fig, out)
