"""Explainer figure: the method, step by step, on one real block.

Four map panels share one extent: (a) the inputs, (b) the original approach, which faces
the nearest OpenStreetMap way of any class, (c) this repository's approach, which faces the
nearest point on the addressed street, and (d) the cadastral frontage both are checked
against.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.artist import Artist
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Transform
from numpy.typing import ArrayLike

from property_orientation.geometry import angular_difference, compass8, true_azimuth
from property_orientation.orientation import OrientationParams, prepare_roads
from property_orientation.reference import ReferenceParams, frontage_segments, name_segments
from property_orientation.streets import normalise_street_name, strip_suffix
from property_orientation.validation import evaluation_set
from property_orientation.viz import style
from property_orientation.viz.data import FigureData

# Oakville Road at Keary Street, Willoughby (EPSG:7856, metres). Found by scanning 160 m x
# 120 m windows around every site whose nearest-way baseline lands on a footpath or driveway
# and is more than 45° off the cadastral frontage while the street-matched bearing is within
# 22.5°. Windows were kept if they held 10-34 mostly single-address sites, all with a
# cadastral reference and no street-matched error over 22.5°. This one was picked by eye
# because OSM maps the sidewalks beside the street as separate ways, as well as driveways.
# The centre and size were then adjusted so that every arrow ends inside the frame.
WINDOW_CENTRE = (333516.0, 6258360.0)
WINDOW_SIZE = (192.0, 112.0)
"""Width and height of the window in metres."""

N_FEATURED = 4
"""Properties whose 8-point label is printed in panels b to d."""

STREET, PATH, DRIVEWAY = "street", "path", "driveway"
_CONTEXT_HIGHWAYS = frozenset(
    {"motorway", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
)

WAY_STYLES: dict[str, dict] = {
    STREET: {"color": style.ROAD, "linewidth": 3.4, "solid_capstyle": "round"},
    PATH: {"color": style.INK_SECONDARY, "linewidth": 1.1, "linestyle": (0, (0.1, 2.2)),
           "dash_capstyle": "round"},
    DRIVEWAY: {"color": style.INK_MUTED, "linewidth": 1.0},
}  # fmt: skip
WAY_LABELS = {
    STREET: "Street",
    PATH: "Footpath, cycleway or steps",
    DRIVEWAY: "Driveway (unnamed service way)",
}
REFERENCE_COLOR = style.INK_SECONDARY
RESERVE_FILL = "#ebeae4"
NORMAL_LENGTH_M = 9.0
ARROW_STYLE = "-|>,head_length=5,head_width=2.4"  # points, with mutation_scale=1
ARROW_WIDTH = 1.5

# Marks, in points.
DOT_SIZE, DOT_HALO = 3.6, 0.8
"""Diameter of an address point, and the width of the light ring that separates it."""
RING_SIZE, RING_WIDTH = 7.0, 1.4
RING_MERGE_M = 1.0
"""Baseline landings closer together than this share one ring, so rings do not stack."""
LABEL_SIZE = 9.5
LABEL_COLUMNS = ("baseline_orientation", "orientation", "ref_orientation")
"""The 8-point labels printed for featured sites in panels b, c and d."""
LABEL_GAP = 2.5
"""Space between a featured address point and its label."""
LABEL_CLEARANCE = 2.0
"""Space a label keeps from every line, arrow and street in panels b to d."""
LABEL_NEIGHBOUR = 2 * LABEL_GAP
"""Space a label keeps from every other address point and ring, so it reads as its own."""
HALO_WIDTH = 2.6
HALO = [pe.withStroke(linewidth=HALO_WIDTH, foreground=style.SURFACE)]

# Figure layout, in inches.
FIG_WIDTH = 10.6
MARGIN, COLUMN_GAP, ROW_GAP = 0.25, 0.4, 0.16
HEADER, HEADING, FOOTER = 1.23, 0.74, 0.34
"""``HEADER`` holds the title and three subtitle lines."""
LEGEND_ROWS = (3, 2)
"""Most legend rows under a map, in the top and bottom row of panels."""
LEGEND_ROW, LEGEND_PAD = 0.2, 0.095

TITLE = (
    "Nearest-way snapping lands on driveways and footpaths; "
    "street matching agrees with the frontage"
)


@dataclass
class Block:
    """Every layer of the explainer window, in the study CRS."""

    box: shapely.Polygon
    sites: pd.DataFrame
    """Sites in the window, plus the baseline landing point (``near_x``/``near_y``), the class
    of its way (``near_kind``) and its error against the reference (``err_old``), the
    reference label (``ref_orientation``), the frontage anchor (``anchor_x``/``anchor_y``) and
    outward unit normal (``normal_x``/``normal_y``), a ``featured`` flag and, for featured
    sites, the offset of their label in points (``label_dx``/``label_dy``)."""
    parcels: gpd.GeoSeries
    reserves: gpd.GeoSeries
    centrelines: gpd.GeoSeries
    ways: gpd.GeoDataFrame
    """OSM ways near the window, with a ``kind`` column (street, path or driveway)."""
    frontage: pd.DataFrame
    """Frontage segments onto each site's addressed street, as ``x0, y0, x1, y1``."""


def way_kind(roads: gpd.GeoDataFrame) -> np.ndarray:
    """Drawing class of each way from :func:`prepare_roads`: street, path or driveway."""
    street = roads["frontage"].to_numpy() | roads["highway"].isin(_CONTEXT_HIGHWAYS).to_numpy()
    driveway = ~street & roads["highway"].eq("service").to_numpy()
    return np.select([street, driveway], [STREET, DRIVEWAY], default=PATH)


def _window() -> shapely.Polygon:
    """The map extent shared by all four panels."""
    (cx, cy), (w, h) = WINDOW_CENTRE, WINDOW_SIZE
    return shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _map_size() -> tuple[float, float]:
    """Width and height of each map panel, in inches."""
    width = (FIG_WIDTH - 2 * MARGIN - COLUMN_GAP) / 2
    return width, width * WINDOW_SIZE[1] / WINDOW_SIZE[0]


def _points_per_metre() -> float:
    """Scale of every map panel."""
    return 72.0 * _map_size()[0] / WINDOW_SIZE[0]


def _baseline_targets(sites: pd.DataFrame, roads: gpd.GeoDataFrame, crs: object) -> pd.DataFrame:
    """Where the original method lands: the nearest point on the nearest way of any class.

    Mirrors :func:`property_orientation.orientation.nearest_road_baseline` and checks the
    bearings against the stored ``baseline_bearing_deg``.
    """
    points = shapely.points(sites["x"].to_numpy(), sites["y"].to_numpy())
    lines = roads.geometry.to_numpy()
    i, j = shapely.STRtree(lines).query_nearest(
        points,
        max_distance=OrientationParams().baseline_radius_m,
        return_distance=False,
        all_matches=False,
    )
    way = np.full(len(points), -1)
    way[i] = j
    if (way < 0).any():
        raise ValueError("every site in the explainer window needs a way within reach")
    xy = shapely.get_coordinates(shapely.shortest_line(points, lines[way])).reshape(-1, 2, 2)
    near_x, near_y = xy[:, 1, 0], xy[:, 1, 1]
    bearing = true_azimuth(sites["x"], sites["y"], near_x, near_y, crs)
    if np.nanmax(angular_difference(bearing, sites["baseline_bearing_deg"])) > 0.01:
        raise RuntimeError("baseline targets disagree with sites.parquet; re-run the pipeline")
    return pd.DataFrame(
        {"near_x": near_x, "near_y": near_y, "near_kind": way_kind(roads)[way]}, index=sites.index
    )


def _frontage(
    sites: pd.DataFrame, data: FigureData, centrelines: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frontage segments of each site's parcel onto its addressed street, and per-site normals.

    Repeats :func:`property_orientation.reference.reference_orientation` for the parcels in
    view, so the drawn normals are the reference bearings (checked against
    ``ref_bearing_deg``). Only sites with a usable reference (``ref_status`` ok) are kept.
    """
    params = ReferenceParams()
    used = np.unique(sites.loc[sites["ref_parcel"] >= 0, "ref_parcel"].to_numpy())
    parcels = data.parcels.geometry.to_numpy()[used]
    reach = shapely.box(*shapely.total_bounds(parcels)).buffer(params.ray_length_m + 10.0)
    reserves = data.road_reserves.geometry.to_numpy()

    segments = frontage_segments(
        parcels, reserves[shapely.intersects(reserves, reach)], params.tolerance_m
    )
    segments["ref_parcel"] = used[segments["parcel"].to_numpy()]
    lines = centrelines[centrelines.intersects(reach)]
    segments["street"] = name_segments(
        segments,
        lines.geometry.to_numpy(),
        lines["roadnamelabel"].map(normalise_street_name).to_numpy(),
        params.ray_length_m,
    )
    ok = sites[sites["ref_status"].eq("ok")]
    joined = ok[["site_id", "ref_parcel", "street_key"]].merge(segments, on="ref_parcel")
    seg = joined[
        joined["street"].notna()
        & (
            (joined["street"] == joined["street_key"])
            | (joined["street"].map(strip_suffix) == joined["street_key"].map(strip_suffix))
        )
    ].copy()

    half_x, half_y = -seg["ny"] * seg["length"] / 2, seg["nx"] * seg["length"] / 2
    seg["x0"], seg["x1"] = seg["mx"] - half_x, seg["mx"] + half_x
    seg["y0"], seg["y1"] = seg["my"] - half_y, seg["my"] + half_y

    weighted = seg.assign(
        wx=seg["nx"] * seg["length"],
        wy=seg["ny"] * seg["length"],
        wmx=seg["mx"] * seg["length"],
        wmy=seg["my"] * seg["length"],
    )
    sums = weighted.groupby("site_id")[["wx", "wy", "wmx", "wmy", "length"]].sum()
    resultant = np.hypot(sums["wx"], sums["wy"])
    normals = pd.DataFrame(
        {
            "anchor_x": sums["wmx"] / sums["length"],
            "anchor_y": sums["wmy"] / sums["length"],
            "normal_x": sums["wx"] / resultant,
            "normal_y": sums["wy"] / resultant,
        }
    )

    check = ok.join(normals, on="site_id", how="inner")
    bearing = true_azimuth(
        check["anchor_x"],
        check["anchor_y"],
        check["anchor_x"] + 10.0 * check["normal_x"],
        check["anchor_y"] + 10.0 * check["normal_y"],
        data.crs,
    )
    if len(check) < len(ok) or angular_difference(bearing, check["ref_bearing_deg"]).max() > 0.01:
        raise RuntimeError("frontage normals disagree with sites.parquet; re-run the pipeline")
    return seg[["site_id", "x0", "y0", "x1", "y1"]], normals


def _featured(sites: pd.DataFrame, n: int = N_FEATURED, spacing_m: float = 35.0) -> np.ndarray:
    """Well-spread properties where the baseline label is wrong, largest errors first.

    The worst case for each reference direction comes first, so the labels vary.
    """
    checked = sites[sites["ref_status"].eq("ok")]
    wrong = checked[checked["baseline_orientation"].ne(checked["ref_orientation"])]
    ranked = wrong.sort_values(["err_old", "site_id"], ascending=[False, True])
    per_direction = ranked.drop_duplicates("ref_orientation")
    chosen: dict[object, tuple[float, float]] = {}
    for idx, row in pd.concat([per_direction, ranked.drop(per_direction.index)]).iterrows():
        if all(np.hypot(row["x"] - x, row["y"] - y) >= spacing_m for x, y in chosen.values()):
            chosen[idx] = (row["x"], row["y"])
        if len(chosen) == n:
            break
    return sites.index.isin(list(chosen))


def _ink_extent(text: str) -> tuple[float, float, float, float]:
    """Ink bounds ``(x0, y0, x1, y1)`` of a featured label, in points from its baseline origin."""
    path = TextPath((0, 0), text, size=LABEL_SIZE, prop=FontProperties(weight="bold"))
    ext = path.get_extents()
    return ext.x0, ext.y0, ext.x1, ext.y1


def _segments(x0: ArrayLike, y0: ArrayLike, x1: ArrayLike, y1: ArrayLike) -> np.ndarray:
    """Two-point lines from arrays of start and end coordinates."""
    start = np.column_stack([x0, y0]).astype(float)
    end = np.column_stack([x1, y1]).astype(float)
    return shapely.linestrings(np.stack([start, end], axis=1))


def _label_positions(block: Block) -> pd.DataFrame:
    """One label spot per featured site, shared by panels b to d so they compare at a glance.

    Candidate spots ring the address point in 22.5° steps, starting behind the property
    (opposite its reference normal) and alternating outwards. The first spot that keeps
    ``LABEL_CLEARANCE`` from every line, arrow and street drawn in b to d, and lies clearly
    closer to its own point than to any other point or ring, wins; failing that, the clearest
    spot. Returns the offset of each label's ink centre from its point, in points.
    """
    sites, scale = block.sites, _points_per_metre()
    faced = sites[sites["bearing_deg"].notna()]
    normal = sites[sites["anchor_x"].notna()]
    landed = sites[sites["near_kind"].ne(STREET)]
    streets = block.ways.loc[block.ways["kind"].eq(STREET), "geometry"].to_numpy()
    f = block.frontage
    lines = np.concatenate(
        [
            _segments(sites["x"], sites["y"], sites["near_x"], sites["near_y"]),
            _segments(faced["x"], faced["y"], faced["street_x"], faced["street_y"]),
            _segments(
                normal["anchor_x"],
                normal["anchor_y"],
                normal["anchor_x"] + NORMAL_LENGTH_M * normal["normal_x"],
                normal["anchor_y"] + NORMAL_LENGTH_M * normal["normal_y"],
            ),
            _segments(f["x0"], f["y0"], f["x1"], f["y1"]),
            shapely.buffer(streets, WAY_STYLES[STREET]["linewidth"] / 2 / scale),
        ]
    )
    dot = DOT_SIZE / 2 + DOT_HALO
    dots = shapely.buffer(shapely.points(sites["x"], sites["y"]), dot / scale)
    rings = shapely.buffer(
        shapely.points(landed["near_x"], landed["near_y"]), RING_SIZE / 2 / scale
    )
    frame = block.box.buffer(-1.0)
    steps = [0, *(sign * k for k in range(1, 8) for sign in (1, -1)), 8]

    offsets = {}
    for pos in np.flatnonzero(sites["featured"].to_numpy()):
        site = sites.iloc[pos]
        extents = [_ink_extent(site[c]) for c in LABEL_COLUMNS if isinstance(site[c], str)]
        half_w = max(x1 - x0 for x0, _, x1, _ in extents) / 2 + HALO_WIDTH / 2
        half_h = max(y1 - y0 for _, y0, _, y1 in extents) / 2 + HALO_WIDTH / 2
        behind = np.degrees(np.arctan2(-site["normal_y"], -site["normal_x"]))
        behind = behind if np.isfinite(behind) else 90.0
        others = np.concatenate([np.delete(dots, pos), rings])
        best, best_score = (0.0, dot + LABEL_GAP + half_h), -np.inf
        for step in steps:
            theta = np.radians(behind + 22.5 * step)
            ux, uy = np.cos(theta), np.sin(theta)
            reach = dot + LABEL_GAP + abs(ux) * half_w + abs(uy) * half_h
            cx, cy = site["x"] + reach * ux / scale, site["y"] + reach * uy / scale
            w, h = half_w / scale, half_h / scale
            label = shapely.box(cx - w, cy - h, cx + w, cy + h)
            if not frame.contains(label):
                continue
            score = min(
                shapely.distance(label, lines).min() * scale - LABEL_CLEARANCE,
                shapely.distance(label, others).min() * scale - LABEL_NEIGHBOUR,
            )
            if score > best_score:
                best, best_score = (reach * ux, reach * uy), score
            if score >= 0:
                break
        offsets[sites.index[pos]] = best
    return pd.DataFrame.from_dict(offsets, orient="index", columns=["label_dx", "label_dy"])


def load_block(data: FigureData) -> Block:
    """Gather and derive every layer the explainer draws."""
    box = _window()
    view = box.buffer(60.0)

    inside = shapely.contains_xy(box, data.sites["x"].to_numpy(), data.sites["y"].to_numpy())
    if not inside.any():
        (x, y), (w, h) = WINDOW_CENTRE, WINDOW_SIZE
        raise ValueError(
            f"no {data.paths.area.lga} sites in the explainer window ({w:.0f} m x {h:.0f} m "
            f"around {x:.0f}, {y:.0f} in EPSG:7856): the figure is drawn for one block in "
            "Willoughby"
        )
    sites = pd.DataFrame(data.sites.loc[inside].drop(columns="geometry"))
    roads = prepare_roads(data.roads)
    sites = sites.join(_baseline_targets(sites, roads, data.crs))
    sites["err_old"] = angular_difference(sites["baseline_bearing_deg"], sites["ref_bearing_deg"])
    sites["ref_orientation"] = compass8(sites["ref_bearing_deg"])

    centrelines = gpd.read_parquet(data.paths.raw_file("nsw_road_centreline")).to_crs(data.crs)
    frontage, normals = _frontage(sites, data, centrelines)
    sites = sites.join(normals, on="site_id")
    sites["featured"] = _featured(sites)

    ways = roads[roads.intersects(view)].copy()
    ways["kind"] = way_kind(ways)
    parcels, reserves = data.parcels.geometry, data.road_reserves.geometry
    block = Block(
        box=box,
        sites=sites,
        parcels=parcels[parcels.intersects(view)],
        reserves=reserves[reserves.intersects(view)],
        centrelines=centrelines.geometry[centrelines.intersects(view)],
        ways=ways,
        frontage=frontage,
    )
    block.sites = sites.join(_label_positions(block))
    return block


# --- drawing helpers ------------------------------------------------------------------------


class _ArrowHandler(HandlerBase):
    """Legend key drawn as a short arrow, or with ``ring=True`` as a line ending in a ring."""

    def __init__(self, *, ring: bool = False) -> None:
        super().__init__()
        self.ring = ring

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Line2D,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> list[Artist]:
        y = height / 2 - ydescent
        x0, x1 = -xdescent, width - xdescent
        color, lw = orig_handle.get_color(), orig_handle.get_linewidth()
        if not self.ring:
            arrow = FancyArrowPatch(
                (x0, y), (x1, y), arrowstyle=ARROW_STYLE, mutation_scale=1, color=color,
                linewidth=lw, shrinkA=0, shrinkB=0, transform=trans,
            )  # fmt: skip
            return [arrow]
        centre = x1 - RING_SIZE / 2
        line = Line2D([x0, centre - RING_SIZE / 2], [y, y], color=color, linewidth=lw)
        ring = Line2D(
            [centre], [y], marker="o", ms=RING_SIZE, markerfacecolor="none",
            markeredgecolor=color, markeredgewidth=RING_WIDTH,
        )  # fmt: skip
        for artist in (line, ring):
            artist.set_transform(trans)
        return [line, ring]


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    lw: float = ARROW_WIDTH,
    head: bool = True,
    shrink_end: float = 0.0,
) -> None:
    """Arrow from ``start`` to ``end``; without a head it is a plain connector."""
    patch = FancyArrowPatch(
        start, end, arrowstyle=ARROW_STYLE if head else "-", mutation_scale=1, color=color,
        linewidth=lw, shrinkA=2.0, shrinkB=shrink_end, joinstyle="round", capstyle="round",
        zorder=6,
    )  # fmt: skip
    ax.add_patch(patch)


def _thin(points: np.ndarray, distance: float) -> np.ndarray:
    """Drop every point that lies within ``distance`` of a point kept before it."""
    kept: list[np.ndarray] = []
    for point in points:
        if all(np.hypot(*(point - other)) >= distance for other in kept):
            kept.append(point)
    return np.reshape(kept, (-1, 2))


def _lines(ax: plt.Axes, geoms: Iterable[shapely.Geometry], **kwargs: object) -> None:
    """Plot every part of every (multi)line in ``geoms``."""
    for geom in geoms:
        for part in shapely.get_parts(geom):
            xs, ys = shapely.get_coordinates(part).T
            ax.plot(xs, ys, **kwargs)


def _draw_parcels(ax: plt.Axes, block: Block) -> None:
    """NSW parcels as the light base layer of every panel."""
    block.parcels.plot(
        ax=ax, facecolor=style.PARCEL_FILL, edgecolor=style.PARCEL_EDGE, linewidth=0.6, zorder=1
    )


def _draw_ways(ax: plt.Axes, block: Block) -> None:
    """OSM ways, styled by class: street band, dotted footpaths, thin driveways."""
    for kind, zorder in ((STREET, 2), (DRIVEWAY, 3), (PATH, 3)):
        geoms = block.ways.loc[block.ways["kind"].eq(kind), "geometry"]
        _lines(ax, geoms, zorder=zorder, **WAY_STYLES[kind])


def _draw_points(ax: plt.Axes, block: Block) -> None:
    """G-NAF address points: all halos first, so near-coincident points do not cut each other."""
    x, y = block.sites["x"], block.sites["y"]
    dot = {"linestyle": "none", "marker": "o", "markeredgewidth": 0}
    ax.plot(x, y, ms=DOT_SIZE + 2 * DOT_HALO, color=style.SURFACE, zorder=8, **dot)
    ax.plot(x, y, ms=DOT_SIZE, color=style.INK, zorder=8.5, **dot)


def _label_streets(ax: plt.Axes, block: Block, min_length_m: float = 30.0) -> None:
    """Write each named street along its longest stretch in view, away from junctions.

    The label sits on the street in a box of the surface colour, which cleanly interrupts the
    street band and the driveways that meet it.
    """
    inner = block.box.buffer(-8.0)
    streets = block.ways[block.ways["kind"].eq(STREET)]
    for name, group in streets[streets["name"].notna()].groupby("name", sort=True):
        others = shapely.union_all(streets.loc[streets["name"].ne(name), "geometry"].to_numpy())
        merged = shapely.line_merge(
            shapely.intersection(shapely.union_all(group.geometry.to_numpy()), inner)
        )
        parts = [p for p in shapely.get_parts(merged) if p.length >= min_length_m]
        if not parts:
            continue
        line = max(parts, key=lambda p: p.length)
        at = max(
            np.linspace(0.25, 0.75, 11) * line.length,
            key=lambda d: shapely.distance(line.interpolate(d), others),
        )
        p, q, mid = line.interpolate(at - 4.0), line.interpolate(at + 4.0), line.interpolate(at)
        angle = (np.degrees(np.arctan2(q.y - p.y, q.x - p.x)) + 90.0) % 180.0 - 90.0  # upright
        ax.text(
            mid.x, mid.y, name, rotation=angle, rotation_mode="anchor", ha="center",
            va="center_baseline", fontsize=8.5, color=style.INK_SECONDARY, zorder=9,
            bbox={"boxstyle": "square,pad=0.25", "facecolor": style.SURFACE, "edgecolor": "none"},
        )  # fmt: skip


def _scale_bar(ax: plt.Axes, box: shapely.Polygon, length_m: float = 20.0) -> None:
    """Scale bar and grid-north arrow in the lower-left corner (true north is ~1° off)."""
    x0, y0, _, _ = box.bounds
    left, base = x0 + 5.0, y0 + 5.0
    ink = {"color": style.INK_SECONDARY, "zorder": 9}
    ax.plot([left, left + length_m], [base, base], linewidth=1.6, solid_capstyle="butt", **ink)
    ax.text(
        left + length_m / 2, base + 1.6, f"{length_m:.0f} m", ha="center", va="bottom",
        fontsize=8, path_effects=HALO, **ink,
    )  # fmt: skip
    north = left + length_m + 8.0
    ax.add_patch(
        FancyArrowPatch(
            (north, base - 1.0), (north, base + 9.0), arrowstyle=ARROW_STYLE, mutation_scale=1,
            linewidth=1.2, **ink,
        )
    )  # fmt: skip
    ax.text(
        north, base + 10.5, "N", ha="center", va="bottom", fontsize=8, fontweight="bold",
        path_effects=HALO, **ink,
    )  # fmt: skip


def _compass_labels(ax: plt.Axes, sites: pd.DataFrame, column: str) -> None:
    """Print each featured site's ``column`` label at its spot from :func:`_label_positions`."""
    for _, row in sites[sites["featured"] & sites[column].notna()].iterrows():
        _, y0, _, y1 = _ink_extent(row[column])
        ax.annotate(
            row[column], (row["x"], row["y"]),
            xytext=(row["label_dx"], row["label_dy"] - (y0 + y1) / 2), textcoords="offset points",
            ha="center", va="baseline", fontsize=LABEL_SIZE, fontweight="bold", color=style.INK,
            path_effects=HALO, zorder=10,
        )  # fmt: skip


def _legend(
    ax: plt.Axes,
    entries: list[tuple[Artist, str]],
    *,
    ncol: int = 1,
    arrows: tuple[Artist, ...] = (),
    rings: tuple[Artist, ...] = (),
) -> None:
    """Legend under the map; handles in ``arrows`` and ``rings`` are drawn like the map marks."""
    handles, labels = zip(*entries, strict=True)
    handler_map = {h: _ArrowHandler() for h in arrows}
    handler_map |= {h: _ArrowHandler(ring=True) for h in rings}
    ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=ncol, fontsize=8.5,
        handlelength=2.4, columnspacing=1.6, borderaxespad=0, handler_map=handler_map,
        labelcolor=style.INK_SECONDARY,
    )  # fmt: skip


def _finish_map(ax: plt.Axes, box: shapely.Polygon, letter: str, title: str, note: str) -> None:
    """Fix the extent, frame the map and write the panel letter, title and note above it."""
    x0, y0, x1, y1 = box.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")  # GeoPandas labels projected axes
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(style.GRID)
        spine.set_linewidth(0.8)
    top = HEADING * 72 - 4  # points above the map
    above = {"xycoords": "axes fraction", "textcoords": "offset points", "ha": "left",
             "va": "top"}  # fmt: skip
    ax.annotate(
        f"{letter}   {title}", (0, 1), xytext=(0, top), fontsize=11.5, fontweight="bold",
        color=style.INK, **above,
    )  # fmt: skip
    ax.annotate(
        note, (0, 1), xytext=(0, top - 19), fontsize=9, linespacing=1.4,
        color=style.INK_SECONDARY, **above,
    )  # fmt: skip


# --- panels -------------------------------------------------------------------------------


def _inputs(ax: plt.Axes, block: Block) -> None:
    """(a) Parcels, OSM ways by class, G-NAF points, street names, scale and north."""
    _draw_parcels(ax, block)
    _draw_ways(ax, block)
    _draw_points(ax, block)
    _label_streets(ax, block)
    _scale_bar(ax, block.box)
    entries = [(Line2D([], [], **WAY_STYLES[kind]), WAY_LABELS[kind]) for kind in WAY_LABELS]
    point = Line2D([], [], linestyle="none", marker="o", ms=DOT_SIZE, color=style.INK)
    parcel = Patch(facecolor=style.PARCEL_FILL, edgecolor=style.PARCEL_EDGE, linewidth=0.6)
    entries += [(point, "G-NAF address point"), (parcel, "NSW property parcel")]
    _legend(ax, entries, ncol=2)


def _baseline(ax: plt.Axes, block: Block) -> None:
    """(b) The original method: a line to the nearest way of any class, ringed off-street."""
    color = style.METHOD_COLORS["nearest_road"]
    sites = block.sites
    off_street = sites["near_kind"].ne(STREET)
    _draw_parcels(ax, block)
    _draw_ways(ax, block)
    # Landings on a footpath or driveway end in a ring rather than an arrowhead: many are only
    # a few metres from the address point, too short to carry a head.
    for (_, row), ring in zip(sites.iterrows(), off_street, strict=True):
        start, end = (row["x"], row["y"]), (row["near_x"], row["near_y"])
        _arrow(ax, start, end, color, head=not ring, shrink_end=RING_SIZE / 2 if ring else 0.0)
    rings = _thin(sites.loc[off_street, ["near_x", "near_y"]].to_numpy(), RING_MERGE_M)
    ax.plot(
        rings[:, 0], rings[:, 1], linestyle="none", marker="o", ms=RING_SIZE,
        markerfacecolor="none", markeredgecolor=color, markeredgewidth=RING_WIDTH, zorder=7,
    )  # fmt: skip
    _draw_points(ax, block)
    _compass_labels(ax, sites, "baseline_orientation")
    street = Line2D([], [], color=color, linewidth=ARROW_WIDTH)
    other = Line2D([], [], color=color, linewidth=ARROW_WIDTH)
    n, off = len(sites), int(off_street.sum())
    entries = [
        (street, f"Lands on a street ({n - off} of {n})"),
        (other, f"Lands on a footpath or driveway ({off} of {n})"),
    ]
    _legend(ax, entries, arrows=(street,), rings=(other,))


def _street_matched(ax: plt.Axes, block: Block) -> None:
    """(c) This repository: an arrow to the nearest point on the addressed street."""
    color = style.METHOD_COLORS["street_matched"]
    sites = block.sites[block.sites["bearing_deg"].notna()]
    _draw_parcels(ax, block)
    _draw_ways(ax, block)
    for _, row in sites.iterrows():
        _arrow(ax, (row["x"], row["y"]), (row["street_x"], row["street_y"]), color)
    _draw_points(ax, block)
    _compass_labels(ax, sites, "orientation")
    arrow = Line2D([], [], color=color, linewidth=ARROW_WIDTH)
    entries = [(arrow, "Faces the nearest point on the street named in its address")]
    _legend(ax, entries, arrows=(arrow,))


def _reference(ax: plt.Axes, block: Block) -> None:
    """(d) The cadastral reference: frontage onto the addressed street and its outward normal."""
    _draw_parcels(ax, block)
    block.reserves.plot(ax=ax, facecolor=RESERVE_FILL, edgecolor="none", zorder=0.5)
    _lines(ax, block.centrelines, color=style.INK_MUTED, linewidth=0.9, zorder=2)
    for _, seg in block.frontage.iterrows():
        ax.plot(
            [seg["x0"], seg["x1"]], [seg["y0"], seg["y1"]], color=REFERENCE_COLOR, linewidth=2.4,
            solid_capstyle="round", zorder=4,
        )  # fmt: skip
    sites = block.sites[block.sites["anchor_x"].notna()]
    for _, row in sites.iterrows():
        anchor = np.array([row["anchor_x"], row["anchor_y"]])
        tip = anchor + NORMAL_LENGTH_M * np.array([row["normal_x"], row["normal_y"]])
        _arrow(ax, tuple(anchor), tuple(tip), REFERENCE_COLOR, lw=1.4)
    _draw_points(ax, block)
    _compass_labels(ax, sites, "ref_orientation")
    normal = Line2D([], [], color=REFERENCE_COLOR, linewidth=1.4)
    entries = [
        (
            Line2D([], [], color=REFERENCE_COLOR, linewidth=2.4),
            "Frontage onto the addressed street",
        ),
        (Line2D([], [], color=style.INK_MUTED, linewidth=0.9), "NSW road centreline"),
        (normal, "Outward normal (reference bearing)"),
        (Patch(facecolor=RESERVE_FILL, edgecolor="none"), "Road reserve"),
    ]
    _legend(ax, entries, ncol=2, arrows=(normal,))


def method_explainer(data: FigureData, out: Path) -> Path:
    """Four maps of one block: inputs, the original method, this method, the cadastral check."""
    block = load_block(data)
    checked = evaluation_set(block.sites)  # the sites results/*/validation scores
    overall = data.validation("method_comparison").set_index("method")

    def agreement(method: str, column: str) -> str:
        same = int(checked[column].eq(checked["ref_orientation"]).sum())
        lga = overall.loc[method]
        return (
            f"Same 8-point direction as the cadastral frontage (d): {same} of {len(checked)}\n"
            f"Across the {data.paths.area.lga} council area: "
            f"{lga['same_8_point_direction']:.1%} of {int(lga['sites']):,} checked sites"
        )

    panels = [
        (
            _inputs,
            "Inputs",
            "G-NAF address points, OpenStreetMap ways by class, NSW parcels\n"
            "Both methods use only the address points and the OSM ways",
        ),
        (
            _baseline,
            "v1: nearest OSM way of any class",
            agreement("nearest_road", "baseline_orientation"),
        ),
        (
            _street_matched,
            "This repository: nearest point on the addressed street",
            agreement("street_matched", "orientation"),
        ),
        (
            _reference,
            "Validation reference: cadastral frontage",
            "Parcel edges on the road reserve, named after the NSW centreline they face\n"
            "Independent of OpenStreetMap: shares only the address point and street name",
        ),
    ]

    map_w, map_h = _map_size()
    legend_h = [LEGEND_PAD + LEGEND_ROW * rows for rows in LEGEND_ROWS]
    fig_h = HEADER + 2 * (HEADING + map_h) + sum(legend_h) + ROW_GAP + FOOTER
    fig = plt.figure(figsize=(FIG_WIDTH, fig_h))
    for k, (draw, title, note) in enumerate(panels):
        row, col = divmod(k, 2)
        left = MARGIN + col * (map_w + COLUMN_GAP)
        top = fig_h - HEADER - row * (HEADING + map_h + legend_h[0] + ROW_GAP)
        bottom = top - HEADING - map_h
        ax = fig.add_axes((left / FIG_WIDTH, bottom / fig_h, map_w / FIG_WIDTH, map_h / fig_h))
        draw(ax, block)
        _finish_map(ax, block.box, "abcd"[k], title, note)

    counts = block.sites["street_key"].value_counts()
    streets = sorted(counts.index, key=lambda s: (-counts[s], s))[:2]
    place = " at ".join(s.title() for s in streets)
    locality = block.sites["locality"].mode().iloc[0].title()
    fig.text(
        MARGIN / FIG_WIDTH, 1 - 0.1 / fig_h, TITLE, ha="left", va="top", fontsize=15,
        fontweight="bold", color=style.INK,
    )  # fmt: skip
    # How the block was chosen (see WINDOW_CENTRE): it shows how the methods differ, so the
    # subtitle says so and points to the council-wide rates in the notes of b and c.
    fig.text(
        MARGIN / FIG_WIDTH, 1 - 0.45 / fig_h,
        f"{place}, {locality}: chosen as a block where the original approach often fails and "
        "the street-matched method agrees with the reference.\nIt shows how the methods "
        "differ, not how often each is right: the council-wide rates are under b and c.\n"
        "In b and c, each line runs from a G-NAF address point to the point it is taken to "
        "face. Letters give the 8-point direction of the same four properties in b, c and d.",
        ha="left", va="top", fontsize=10, linespacing=1.5, color=style.INK_SECONDARY,
    )  # fmt: skip
    # Panel d also draws NSW road reserves and centrelines.
    credit = style.ATTRIBUTION.replace("Parcels:", "Parcels, road reserves and centrelines:")
    style.attribution(fig, credit)
    return style.save(fig, out)
