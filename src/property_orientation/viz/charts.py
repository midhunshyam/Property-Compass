"""Statistical charts for the README: validation, confusion matrix and orientation roses.

Every function takes ``(data, out)`` and returns the path it wrote. Numbers in titles and
labels are read from the result tables at render time, never typed in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

from property_orientation.geometry import COMPASS_8, angular_difference, compass8
from property_orientation.validation import METHODS, evaluation_set
from property_orientation.viz import style
from property_orientation.viz.data import FigureData

TITLE_SIZE = 15.0
SUBTITLE_SIZE = 10.5
PANEL_TITLE_SIZE = 11.5
LABEL_SIZE = 9.5
NOTE_SIZE = 8.5
MARGIN_IN = 0.25
KNOCKOUT = {"boxstyle": "square,pad=0.12", "facecolor": style.SURFACE, "edgecolor": "none"}
"""Surface-coloured text background, so gridlines break around in-plot labels."""
NBSP = "\u00a0"
"""Non-breaking space: keeps a phrase on one line when a title wraps."""

AGREEMENT_METRICS = {
    "same_8_point_direction": "Same 8-point direction",
    "within_22_5_deg": "Within 22.5°",
    "within_45_deg": "Within 45°",
}
CONFIDENCE_ORDER = ("high", "medium", "low")
WITHIN_45_COLOR = style.SEQUENTIAL[1]
"""Lighter step of the street-matched blue, for the looser 45° criterion."""

CELL_LABEL_MIN = 0.5
"""Smallest share of a row (%) that the confusion matrix prints in its cell."""

ROSE_BIN_DEG = 15.0
"""Bins of the drawn roses, centred on 0°, 15°, 30°, ... Their edges (7.5° + 15°k) include every
compass sector edge (22.5° + 45°k), so each bar lies inside one sector and takes its colour."""
ORDER_BIN_DEG = 10.0
"""Bins for the orientation order φ: Boeing's (2019) 36 bins of 10°, centred on 0°, 10°, ..."""
ROSE_MIN_ADDRESSES = 100
ROSE_RINGS = (0.10, 0.20, 0.30)
"""Shares of addresses marked by the rings on every rose."""
ROSE_CAPTION_PT = 5.0
"""Gap between a suburb rose and its caption."""
ROSE_BIG_CAPTION_PT = 20.0
"""Gap between the LGA rose and its caption, which clears the rose's "S" label."""


# ------------------------------------------------------------------------------ layout


@dataclass(frozen=True)
class _Header:
    """A takeaway title and muted subtitle, wrapped to the figure width."""

    title: str
    subtitle: str
    top_in: float = 0.18
    gap_in: float = 0.1

    @classmethod
    def fit(cls, fig: plt.Figure, title: str, subtitle: str) -> _Header:
        """Wrap both texts to the figure's width inside the margins (title lines balanced)."""
        width = fig.get_figwidth() - 2 * MARGIN_IN
        return cls(
            _wrap(fig, title, width, TITLE_SIZE, "bold", balance=True),
            _wrap(fig, subtitle, width, SUBTITLE_SIZE),
        )

    @property
    def _title_in(self) -> float:
        """Height of the wrapped title, in inches."""
        return (self.title.count("\n") + 1) * TITLE_SIZE * 1.25 / 72

    @property
    def height(self) -> float:
        """Inches from the top of the figure to the bottom of the subtitle."""
        subtitle_in = (self.subtitle.count("\n") + 1) * SUBTITLE_SIZE * 1.4 / 72
        return self.top_in + self._title_in + self.gap_in + subtitle_in

    def draw(self, fig: plt.Figure) -> None:
        """Place the text; call once the figure has its final height."""
        x, h = MARGIN_IN / fig.get_figwidth(), fig.get_figheight()
        fig.text(
            x,
            1 - self.top_in / h,
            self.title,
            ha="left",
            va="top",
            fontsize=TITLE_SIZE,
            fontweight="bold",
            color=style.INK,
            linespacing=1.25,
        )
        fig.text(
            x,
            1 - (self.top_in + self._title_in + self.gap_in) / h,
            self.subtitle,
            ha="left",
            va="top",
            fontsize=SUBTITLE_SIZE,
            color=style.INK_SECONDARY,
            linespacing=1.4,
        )


def _wrap(
    fig: plt.Figure,
    text: str,
    width_in: float,
    size: float,
    weight: str = "normal",
    *,
    balance: bool = False,
) -> str:
    """Word wrap of ``text`` to ``width_in`` inches, measured with the real font.

    Greedy by default. ``balance`` then narrows the measure as far as it goes without adding
    a line, so that a two-line title does not end in a short orphan. Lines break only at
    plain spaces, so a non-breaking space (``NBSP``) keeps a phrase together.
    """
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(family=plt.rcParams["font.sans-serif"], size=size, weight=weight)

    def greedy(limit_in: float) -> list[str]:
        lines: list[str] = []
        for paragraph in text.split("\n"):
            line = ""
            for word in filter(None, paragraph.split(" ")):
                trial = f"{line} {word}".strip()
                width_px, _, _ = renderer.get_text_width_height_descent(trial, prop, ismath=False)
                if line and width_px / fig.dpi > limit_in:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
        return lines

    lines = greedy(width_in)
    if balance and len(lines) > 1:
        narrow, wide = 0.0, width_in  # greedy(wide) always keeps the minimum line count
        for _ in range(12):
            mid = (narrow + wide) / 2
            narrow, wide = (narrow, mid) if len(greedy(mid)) <= len(lines) else (mid, wide)
        lines = greedy(wide)
    return "\n".join(lines)


def _axes_in(
    fig: plt.Figure, left: float, top: float, width: float, height: float, **kw: Any
) -> plt.Axes:
    """Axes placed in inches, measured from the figure's top-left corner."""
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    return fig.add_axes((left / fw, 1 - (top + height) / fh, width / fw, height / fh), **kw)


def _panel_title(ax: plt.Axes, text: str, *, pad_pt: float = 8.0) -> None:
    """Bold panel title, left-aligned ``pad_pt`` points above the axes."""
    ax.annotate(
        text,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, pad_pt),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        color=style.INK,
    )


def _legend(ax: plt.Axes, labels: Sequence[str], colors: Sequence[str], **kw: Any) -> None:
    """Swatch legend with ink-coloured text (identity comes from the swatch)."""
    handles = [
        Patch(facecolor=c, edgecolor="none", label=lab)
        for lab, c in zip(labels, colors, strict=True)
    ]
    options = {
        "fontsize": LABEL_SIZE,
        "handlelength": 0.9,
        "handleheight": 0.9,
        "handletextpad": 0.5,
        "borderaxespad": 0,
        "borderpad": 0,
        "columnspacing": 1.6,
    } | kw
    ax.legend(handles=handles, frameon=False, labelcolor=style.INK, **options)


def _pct(share: float) -> str:
    """A share (0-1) as a percentage with one decimal."""
    return f"{100 * share:.1f}%"


def _and_join(items: Sequence[str]) -> str:
    """``A``, ``A and B`` or ``A, B and C`` (and an empty string for no items)."""
    if not items:
        return ""
    *rest, last = items
    return f"{', '.join(rest)} and {last}" if rest else last


# ------------------------------------------------------------------------- bar panels


@dataclass(frozen=True)
class _Bars:
    """Vertical rhythm of a grouped horizontal bar panel, in inches."""

    label: float = 0.25
    bar: float = 0.15
    gap: float = 0.035
    group_gap: float = 0.24

    def height(self, groups: int, series: int) -> float:
        """Inches taken by ``groups`` labelled groups of ``series`` bars each."""
        per_group = self.label + series * self.bar + (series - 1) * self.gap
        return groups * per_group + (groups - 1) * self.group_gap


BARS = _Bars()


def _hbar(
    ax: plt.Axes, y: float, value: float, height: float, color: str, radius_pt: float = 3.5
) -> None:
    """Horizontal bar from zero with a rounded data end and a square baseline end.

    Call after the axes position and limits are final: the corner radius is converted
    from points to data units separately on each axis so that it stays circular.
    """
    if value <= 0:
        return
    fig = ax.get_figure()
    pos = ax.get_position()
    x_per_in = np.ptp(ax.get_xlim()) / (pos.width * fig.get_figwidth())
    y_per_in = abs(np.ptp(ax.get_ylim())) / (pos.height * fig.get_figheight())
    rx = min(radius_pt / 72 * x_per_in, value / 2)
    ry = min(radius_pt / 72 * y_per_in, height / 2)
    y0, y1, x1, k = y - height / 2, y + height / 2, value, 0.5523  # k: cubic quarter-arc
    vertices = [
        (0, y0), (x1 - rx, y0),
        (x1 - rx + k * rx, y0), (x1, y0 + ry - k * ry), (x1, y0 + ry),
        (x1, y1 - ry),
        (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1),
        (0, y1), (0, y0),
    ]  # fmt: skip
    curve, line = MplPath.CURVE4, MplPath.LINETO
    codes = [MplPath.MOVETO, line, *[curve] * 3, line, *[curve] * 3, line, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none"))


def _grouped_bars(
    ax: plt.Axes, groups: Sequence[tuple[str, str]], values: np.ndarray, colors: Sequence[str]
) -> None:
    """Groups of horizontal bars (0-100%), each headed by a label and valued at the tips.

    ``groups`` holds ``(label, note)`` pairs and ``values`` is a ``groups x series`` array of
    shares. The y axis runs in inches from the top of the axes, so the spacing is exact.
    """
    height_in = ax.get_position().height * ax.get_figure().get_figheight()
    ax.set_ylim(height_in, 0)
    ax.set_xlim(0, 113)
    ticks = np.arange(0, 101, 25)
    for x in ticks:
        ax.axvline(x, color=style.BASELINE if x == 0 else style.GRID, linewidth=0.8, zorder=0)
    ax.set_xticks(ticks, [f"{x}%" for x in ticks])
    ax.tick_params(axis="x", length=0, pad=4, labelsize=NOTE_SIZE, labelcolor=style.INK_MUTED)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    y = 0.0
    for (label, note), row in zip(groups, values, strict=True):
        text = ax.text(
            0,
            y + BARS.label - 0.075,
            label,
            ha="left",
            va="baseline",
            fontsize=LABEL_SIZE + 0.5,
            color=style.INK,
            bbox=KNOCKOUT,
        )
        if note:
            ax.annotate(
                note,
                xy=(1, 0),
                xycoords=text,
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=NOTE_SIZE,
                color=style.INK_MUTED,
                bbox=KNOCKOUT,
            )
        y += BARS.label
        for share, color in zip(row, colors, strict=True):
            centre = y + BARS.bar / 2
            _hbar(ax, centre, 100 * share, BARS.bar, color)
            ax.text(
                100 * share + 1.2,
                centre,
                _pct(share),
                ha="left",
                va="center",
                fontsize=LABEL_SIZE,
                color=style.INK,
                bbox=KNOCKOUT,
            )
            y += BARS.bar + BARS.gap
        y += BARS.group_gap - BARS.gap


# ------------------------------------------------------------------- validation chart


def _method_errors(sites: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sorted angular error of each method against the reference, on the evaluation set."""
    ev = evaluation_set(sites)
    errors: dict[str, np.ndarray] = {}
    for key, (column, _) in METHODS.items():
        ok = ev[column].notna()
        errors[key] = np.sort(angular_difference(ev.loc[ok, column], ev.loc[ok, "ref_bearing_deg"]))
    return errors


def validation_chart(data: FigureData, out: Path) -> Path:
    """Agreement of both methods with the cadastral frontage: rates, error CDF, confidence."""
    methods = data.validation("method_comparison").set_index("method")
    by_conf = data.validation("by_confidence").set_index("confidence")
    by_conf = by_conf.reindex([c for c in CONFIDENCE_ORDER if c in by_conf.index])
    keys = list(METHODS)
    colors = [style.METHOD_COLORS[k] for k in keys]
    new, old = methods.loc["street_matched"], methods.loc["nearest_road"]
    snapped = evaluation_set(data.sites)["baseline_highway"].value_counts(normalize=True)

    fig = plt.figure(figsize=(11.2, 8.0))
    header = _Header.fit(
        fig,
        f"Matching the addressed street gives the cadastral 8-point{NBSP}direction for "
        f"{new['same_8_point_direction']:.0%} of sites, vs "
        f"{old['same_8_point_direction']:.0%} for the nearest-road approach",
        f"Agreement with the outward normal of each parcel's frontage onto its addressed "
        f"street (NSW cadastre), on the same {int(new['sites']):,} sites. The nearest-road "
        f"approach faces the closest OpenStreetMap way of any class: a footpath for "
        f"{snapped.get('footway', 0):.0%} of these sites and a service way such as a "
        f"driveway for {snapped.get('service', 0):.0%}.",
    )
    # Layout in inches. Left column: method legend, panel A (agreement), panel C (confidence);
    # the right column holds panel B (error CDF) from the top of A to the bottom of C.
    left_w, gutter = 4.6, 0.85
    ticks_in, title_in, gap_in = 0.3, 0.28, 0.36  # tick labels, panel title, between panels
    legend_top = header.height + 0.28
    a_top = legend_top + 0.2 + 0.55  # the legend row, then room for panel A's title
    a_h = BARS.height(len(AGREEMENT_METRICS), len(keys)) + 0.04
    c_legend_in = 0.3  # room between panel C's title and its axes for the shade legend
    c_top = a_top + a_h + ticks_in + gap_in + title_in + c_legend_in
    c_h = BARS.height(len(by_conf), 2) + 0.04
    # The last term keeps the sources line clear of panel B's axis label, which sits lower than
    # panel C's tick labels.
    fig.set_figheight(c_top + c_h + ticks_in + 0.5)
    header.draw(fig)

    legend_ax = _axes_in(fig, MARGIN_IN, legend_top, 6.0, 0.2)
    legend_ax.axis("off")
    _legend(
        legend_ax, [style.METHOD_LABELS[k] for k in keys], colors, loc="center left", ncol=len(keys)
    )

    ax = _axes_in(fig, MARGIN_IN, a_top, left_w, a_h)
    values = np.array([[methods.loc[k, m] for k in keys] for m in AGREEMENT_METRICS])
    _grouped_bars(ax, [(label, "") for label in AGREEMENT_METRICS.values()], values, colors)
    _panel_title(ax, "Agreement with the cadastral frontage")

    ax = _axes_in(fig, MARGIN_IN, c_top, left_w, c_h)
    groups = [
        (f"{name.capitalize()} confidence", f"{int(row['sites']):,} sites")
        for name, row in by_conf.iterrows()
    ]
    shades = [style.METHOD_COLORS["street_matched"], WITHIN_45_COLOR]
    _grouped_bars(
        ax, groups, by_conf[["same_8_point_direction", "within_45_deg"]].to_numpy(), shades
    )
    _panel_title(ax, "Street-matched agreement by confidence class", pad_pt=c_legend_in * 72 + 6)
    _legend(
        ax,
        ["Same 8-point direction", "Within 45°"],
        shades,
        loc="lower left",
        bbox_to_anchor=(0, 1 + 0.09 / c_h),
        fontsize=NOTE_SIZE,
        ncol=2,
    )

    b_left = MARGIN_IN + left_w + gutter
    ax = _axes_in(fig, b_left, a_top, fig.get_figwidth() - b_left - 0.3, c_top + c_h - a_top)
    _error_cdf(ax, _method_errors(data.sites), keys)
    _panel_title(ax, "Cumulative share of sites by angular error")

    style.attribution(fig)
    return style.save(fig, out)


def _error_cdf(ax: plt.Axes, errors: dict[str, np.ndarray], keys: Sequence[str]) -> None:
    """Empirical CDF of angular error (0-180°) per method, with 22.5° and 45° guides."""
    grid = np.linspace(0, 180, 3601)
    for y in (25, 50, 75, 100):
        ax.axhline(y, color=style.GRID, linewidth=0.8, zorder=0)
    for x in (22.5, 45.0):
        ax.axvline(x, color=style.BASELINE, linewidth=0.8, zorder=1)
    cdf: dict[str, np.ndarray] = {}
    for rank, key in enumerate(keys):  # the first method (this repository's) draws on top
        cdf[key] = 100 * np.searchsorted(errors[key], grid, side="right") / len(errors[key])
        ax.plot(grid, cdf[key], color=style.METHOD_COLORS[key], linewidth=2, zorder=4 - rank)

    # Direct labels just under each curve, right of the 45° guide where the curves are apart.
    at = np.searchsorted(grid, 49.0)
    for key in keys:
        ax.text(
            grid[at],
            cdf[key][at] - 2.8,
            style.METHOD_LABELS[key],
            ha="left",
            va="top",
            fontsize=LABEL_SIZE,
            color=style.INK,
        )
    if "nearest_road" in cdf:
        # Measure the two steps in the nearest-road curve: ways beside and behind the house.
        error, curve = errors["nearest_road"], cdf["nearest_road"]
        steps = (
            ("sideways", 80.0, 100.0, 95.0, "left"),
            ("backwards", 170.0, 180.0, 178.0, "right"),
        )  # (direction, window low and high, label x, alignment)
        for way, lo, hi, x, ha in steps:
            share = np.mean((error > lo) & (error <= hi))
            i = np.searchsorted(grid, x)
            ax.text(
                grid[i],
                curve[i] - 4,
                f"{share:.0%} point {way}\n({lo:g}–{hi:g}° off)",
                ha=ha,
                va="top",
                fontsize=NOTE_SIZE,
                color=style.INK_SECONDARY,
                linespacing=1.3,
            )

    ax.set_xlim(0, 180)
    ax.set_ylim(0, 100)
    ticks = [0, 22.5, 45, 90, 135, 180]
    ax.set_xticks(ticks, [f"{t:g}°" for t in ticks])
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(length=0, pad=5, labelsize=NOTE_SIZE, labelcolor=style.INK_MUTED)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(style.BASELINE)
    ax.set_xlabel(
        "Angular difference from the cadastral frontage",
        fontsize=LABEL_SIZE,
        color=style.INK_SECONDARY,
        labelpad=8,
    )
    ax.set_ylabel(
        "Share of sites within that difference",
        fontsize=LABEL_SIZE,
        color=style.INK_SECONDARY,
        labelpad=8,
    )


# -------------------------------------------------------------------- confusion chart


def _sector_steps() -> np.ndarray:
    """Number of 45° sectors between each pair of compass labels (0-4, wrapping round)."""
    k = np.subtract.outer(np.arange(8), np.arange(8)) % 8
    return np.minimum(k, 8 - k)


def _relative_luminance(color: str | tuple[float, ...]) -> float:
    """WCAG relative luminance of a colour (0 for black, 1 for white)."""
    rgb = np.asarray(to_rgb(color))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def _ink_on(fill: str | tuple[float, ...]) -> str:
    """Surface (knock-out) or primary ink, whichever contrasts more with ``fill`` (WCAG)."""
    lum = _relative_luminance(fill)
    on_surface = (_relative_luminance(style.SURFACE) + 0.05) / (lum + 0.05)
    on_ink = (lum + 0.05) / (_relative_luminance(style.INK) + 0.05)
    return style.SURFACE if on_surface >= on_ink else style.INK


def confusion_chart(data: FigureData, out: Path) -> Path:
    """Cadastral reference direction against the street-matched estimate, row-normalised."""
    counts = (
        data.validation("confusion_matrix")
        .set_index("reference")
        .reindex(index=list(COMPASS_8), columns=list(COMPASS_8), fill_value=0)
        .to_numpy()
    )
    totals = counts.sum(axis=1)
    share = 100 * counts / np.where(totals > 0, totals, 1)[:, None]
    steps = _sector_steps()
    missed = counts[steps > 0].sum()
    one_off = counts[steps == 1].sum() / missed if missed else 0.0

    # How close the one-sector misses are, from the sites behind the table.
    ev = evaluation_set(data.sites)
    position = {label: i for i, label in enumerate(COMPASS_8)}
    ref = [position[label] for label in compass8(ev["ref_bearing_deg"])]
    est = [position[label] for label in compass8(ev["bearing_deg"])]
    error = angular_difference(ev["bearing_deg"], ev["ref_bearing_deg"])
    near_miss = np.mean(error[steps[ref, est] == 1] <= 22.5)

    # The README shows this figure at 70% width, so it is drawn small to keep text legible.
    fig = plt.figure(figsize=(7.2, 9.0))
    header = _Header.fit(
        fig,
        f"The estimate matches the cadastral 8-point{NBSP}direction for "
        f"{np.trace(counts) / counts.sum():.0%} of sites, and {one_off:.0%} of the misses are "
        f"one sector off",
        f"For each cadastral reference direction (row), the share of its sites that the "
        f"street-matched estimate puts in each direction (column); {int(counts.sum()):,} sites. "
        f"Sectors wrap round, so N and NW are neighbours. {near_miss:.0%}{NBSP}of one-sector "
        f"misses are within 22.5° of the reference: near-misses across a sector boundary.",
    )
    cell = 0.57
    left = 1.3  # room for the row labels and the axis title
    top = header.height + 0.78
    bar_top = top + 8 * cell + 0.42
    fig.set_figheight(bar_top + 0.1 + 0.62)
    header.draw(fig)

    ax = _axes_in(fig, left, top, 8 * cell, 8 * cell)
    cmap = LinearSegmentedColormap.from_list("sequential", [style.SURFACE, *style.SEQUENTIAL])
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=100)
    mesh = ax.pcolormesh(
        np.arange(9),
        np.arange(9),
        share,
        cmap=cmap,
        norm=norm,
        edgecolors=style.SURFACE,
        linewidth=2.0,
    )
    ax.set_xlim(0, 8)
    ax.set_ylim(8, 0)
    for i in range(8):
        for j in range(8):
            if share[i, j] >= CELL_LABEL_MIN:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{share[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=LABEL_SIZE,
                    color=_ink_on(cmap(norm(share[i, j]))),
                    fontweight="bold" if i == j else "normal",
                )
        ax.text(
            8.2,
            i + 0.5,
            f"{totals[i]:,}",
            ha="left",
            va="center",
            fontsize=NOTE_SIZE,
            color=style.INK_SECONDARY,
        )
    ax.annotate(
        "Sites",
        xy=(8.2, 1),
        xycoords=("data", "axes fraction"),
        xytext=(0, 7),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=NOTE_SIZE,
        color=style.INK_SECONDARY,
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    centres = np.arange(8) + 0.5
    ax.set_xticks(centres, COMPASS_8)
    ax.set_yticks(centres, COMPASS_8)
    ax.tick_params(length=0, pad=7, labelsize=LABEL_SIZE + 0.5, labelcolor=style.INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(
        "Estimated direction (street-matched)",
        fontsize=LABEL_SIZE,
        color=style.INK_SECONDARY,
        labelpad=10,
    )
    ax.set_ylabel(
        "Cadastral reference direction", fontsize=LABEL_SIZE, color=style.INK_SECONDARY, labelpad=10
    )

    cax = _axes_in(fig, left, bar_top, 4 * cell, 0.1)
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    ticks = [0, 1, 5, 10, 25, 50, 100]
    bar.set_ticks(ticks, labels=[str(t) for t in ticks])
    bar.outline.set_visible(False)
    cax.tick_params(length=0, pad=4, labelsize=NOTE_SIZE, labelcolor=style.INK_MUTED)
    cax.annotate(
        f"Share of the row (%), on a square-root scale.\n"
        f"Cells under {CELL_LABEL_MIN:g}% are left unlabelled.",
        xy=(1, 0.5),
        xycoords="axes fraction",
        xytext=(18, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=NOTE_SIZE,
        color=style.INK_SECONDARY,
        linespacing=1.4,
    )

    style.attribution(fig)
    return style.save(fig, out)


# ------------------------------------------------------------------- orientation roses


def _bin_shares(bearings: np.ndarray, bin_deg: float = ROSE_BIN_DEG) -> np.ndarray:
    """Share of bearings in each bin; bins are centred on 0°, ``bin_deg``, ... .

    Each bin is half-open, ``[centre - bin_deg / 2, centre + bin_deg / 2)``, like the compass
    sectors of :func:`compass8`, so a bearing on a shared edge falls on the same side of it.
    """
    bearings = np.asarray(bearings, float)
    if not len(bearings):
        raise ValueError("no bearings to bin")
    bins = round(360 / bin_deg)
    index = np.floor(np.mod(bearings + bin_deg / 2, 360) / bin_deg)
    return np.bincount(index.astype(int) % bins, minlength=bins) / len(bearings)


def orientation_order(shares: np.ndarray) -> float:
    """Orientation order φ (Boeing 2019): 1 for a perfect four-way grid, 0 for uniform.

    ``1 - ((H - H_grid) / (H_max - H_grid))²``, where ``H`` is the Shannon entropy of the
    binned bearings, ``H_grid = ln 4`` and ``H_max = ln(number of bins)``. The scale assumes
    ``H >= H_grid``: bearings more concentrated than a four-way grid (all houses on one
    street, say) score below 1 again. Every suburb rose here is above that bound.
    """
    p = shares[shares > 0]
    entropy = -np.sum(p * np.log(p))
    h_grid, h_max = np.log(4), np.log(len(shares))
    return float(1 - ((entropy - h_grid) / (h_max - h_grid)) ** 2)


def _rose(
    ax: plt.Axes, shares: np.ndarray, rmax: float, rings: Sequence[float], *, big: bool = False
) -> None:
    """Polar histogram of facing bearings, north up and clockwise; bar area ∝ share."""
    centres = np.arange(len(shares)) * 360 / len(shares)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # Wedges narrower than their bin leave a surface gap that closes towards the centre,
    # so the hub stays solid; equal widths keep bar area proportional to share. Every bin lies
    # inside one compass sector (see ROSE_BIN_DEG), so its centre gives its sector's colour.
    ax.bar(
        np.radians(centres),
        np.sqrt(shares),
        width=np.radians(360 / len(shares)) * 0.84,
        color=style.direction_colors(compass8(centres)),
        linewidth=0,
        zorder=3,
    )
    ax.set_ylim(0, rmax)
    ax.set_yticks(np.sqrt(rings), [])
    ax.grid(True, color=style.GRID, linewidth=0.7)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor(style.SURFACE)
    if not big:
        ax.set_xticks(np.radians([0, 90, 180, 270]), [])
        return
    ax.set_xticks(np.radians(np.arange(0, 360, 45)), COMPASS_8)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold" if len(label.get_text()) == 1 else "normal")
    ax.tick_params(axis="x", pad=1, labelsize=LABEL_SIZE + 0.5, labelcolor=style.INK_SECONDARY)
    # Ring labels run along a sector edge, midway between two spokes: the one with the
    # shortest bars on either side.
    edges = centres - 180 / len(shares)
    beside = np.maximum(shares, np.roll(shares, 1))  # the taller of the two bars at each edge
    sector_edge = np.isclose(np.mod(edges - 22.5, 45.0), 0.0)
    theta = np.radians(edges[sector_edge][np.argmin(beside[sector_edge])])
    for ring in rings:
        ax.text(
            theta,
            np.sqrt(ring),
            f"{ring:.0%}",
            ha="center",
            va="center",
            fontsize=NOTE_SIZE,
            color=style.INK_MUTED,
            bbox=KNOCKOUT,
            zorder=4,
        )


def _rose_caption(ax: plt.Axes, name: str, n: int, phi: float, *, big: bool = False) -> None:
    """Name, count and orientation order centred under a rose."""
    size = LABEL_SIZE + 2 if big else LABEL_SIZE + 0.5
    offset = -(ROSE_BIG_CAPTION_PT if big else ROSE_CAPTION_PT)
    ax.annotate(
        name,
        xy=(0.5, 0),
        xycoords="axes fraction",
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=size,
        fontweight="bold",
        color=style.INK,
    )
    ax.annotate(
        f"n = {n:,} · φ = {phi:.2f}",
        xy=(0.5, 0),
        xycoords="axes fraction",
        xytext=(0, offset - size * 1.4),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=NOTE_SIZE + 0.5,
        color=style.INK_SECONDARY,
    )


def rose_small_multiples(data: FigureData, out: Path) -> Path:
    """Facing bearings of standalone addresses, for the whole LGA and each larger suburb."""
    a = data.addresses
    standalone = a[a["address_kind"].eq("standalone") & a["bearing_deg"].notna()]
    if standalone.empty:
        raise ValueError("no standalone address has a facing bearing: there is no rose to draw")
    lga = standalone["bearing_deg"].to_numpy()
    groups = {
        locality: group["bearing_deg"].to_numpy()
        for locality, group in standalone.groupby("locality")
        if len(group) >= ROSE_MIN_ADDRESSES
    }
    suburbs = {locality: _bin_shares(bearings) for locality, bearings in groups.items()}
    phi = {
        locality: orientation_order(_bin_shares(bearings, ORDER_BIN_DEG))
        for locality, bearings in groups.items()
    }
    lga_shares = _bin_shares(lga)
    counts = standalone["locality"].value_counts()
    order = sorted(suburbs, key=phi.__getitem__, reverse=True)
    rmax = np.sqrt(max(s.max() for s in [lga_shares, *suburbs.values()])) * 1.03

    cols, cell_w, cell_h, small_d = 6, 1.86, 2.1, 1.42
    rows = max(2, int(np.ceil((len(order) + 4) / cols)))  # the LGA rose fills a 2 x 2 block
    fig = plt.figure(figsize=(2 * MARGIN_IN + cols * cell_w, 1.0))
    lga_name = data.summary["area"]["lga"]
    if len(order) > 3:  # the three least ordered suburbs, and grid suburbs to contrast them with
        organic = _and_join([s.title() for s in order[::-1][:3]])
        title = f"Houses in grid suburbs face four ways; in {organic} they face every way"
    else:
        title = f"Which way the houses of {lga_name} face"
    header = _Header.fit(
        fig,
        title,
        f"Facing direction of standalone addresses (mostly houses) in {ROSE_BIN_DEG:g}° bins, "
        f"{45 / ROSE_BIN_DEG:g} per compass sector, north up. Bar area is proportional to the "
        f"share of houses; rings mark {_and_join([f'{r:.0%}' for r in ROSE_RINGS])}. Suburbs "
        f"with at least {ROSE_MIN_ADDRESSES} such addresses, ordered by the orientation order φ "
        f"of their bearings in {ORDER_BIN_DEG:g}° bins (Boeing 2019): 1 for a four-way grid, 0 "
        f"for uniform.",
    )
    top = header.height + 0.4
    fig.set_figheight(top + rows * cell_h + 0.1)
    header.draw(fig)

    # The LGA rose starts just under its "N" label and is sized so that its caption lines up
    # with the captions of the second row of suburbs.
    big_top = top + 0.2
    big_d = top + cell_h + small_d + (ROSE_CAPTION_PT - ROSE_BIG_CAPTION_PT) / 72 - big_top
    ax = _axes_in(
        fig, MARGIN_IN + (2 * cell_w - big_d) / 2, big_top, big_d, big_d, projection="polar"
    )
    _rose(ax, lga_shares, rmax, ROSE_RINGS, big=True)
    _rose_caption(
        ax,
        f"{lga_name} LGA, all suburbs",
        len(standalone),
        orientation_order(_bin_shares(lga, ORDER_BIN_DEG)),
        big=True,
    )

    slots = [(r, c) for r in range(rows) for c in range(cols) if r >= 2 or c >= 2]
    for (r, c), suburb in zip(slots, order, strict=False):
        ax = _axes_in(
            fig,
            MARGIN_IN + c * cell_w + (cell_w - small_d) / 2,
            top + r * cell_h,
            small_d,
            small_d,
            projection="polar",
        )
        _rose(ax, suburbs[suburb], rmax, ROSE_RINGS)
        _rose_caption(ax, suburb.title(), int(counts[suburb]), phi[suburb])

    style.attribution(fig)
    return style.save(fig, out)
