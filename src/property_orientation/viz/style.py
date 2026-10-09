"""Shared visual language for every figure: palette, ink, type and helpers.

Direction palette
-----------------
The eight compass sectors use the documented categorical hues, assigned so that
*neighbouring* sectors (including NW-N) stay distinguishable under protanopia and
deuteranopia. The assignment was validated with the dataviz skill's palette checker:

* cyclic-adjacent CVD ΔE ≥ 15.2 and normal-vision ΔE ≥ 15.6 (targets 8 and 15)
* the four cardinal colours, which cover ~78% of properties, also pass all-pairs
  (CVD ΔE 13.0, normal-vision 16.3)

North is yellow, the sunny aspect in the southern hemisphere, and south is blue.
Yellow, magenta and aqua sit below 3:1 contrast on the light surface, so colour is
never the only channel: maps carry a facing arrow per property and a labelled compass
legend, and charts are direct-labelled.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

from property_orientation.geometry import COMPASS_8

DIRECTION_COLORS: dict[str, str] = {
    "N": "#eda100",  # yellow
    "NE": "#e87ba4",  # magenta
    "E": "#008300",  # green
    "SE": "#1baf7a",  # aqua
    "S": "#2a78d6",  # blue
    "SW": "#eb6834",  # orange
    "W": "#4a3aa7",  # violet
    "NW": "#e34948",  # red
}
DIRECTION_NAMES = {
    "N": "North", "NE": "North-east", "E": "East", "SE": "South-east",
    "S": "South", "SW": "South-west", "W": "West", "NW": "North-west",
}  # fmt: skip

# Chart chrome and ink (light surface). Text never takes a data colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots for comparing methods (slot 1, slot 2).
METHOD_COLORS = {"street_matched": "#2a78d6", "nearest_road": "#eb6834"}
METHOD_LABELS = {
    "street_matched": "Street-matched (this repo)",
    "nearest_road": "Nearest OSM way (v1)",
}

# Single-hue sequential ramp (blue, light -> dark) for magnitudes such as choropleths.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Map base layers.
PARCEL_EDGE = "#d9d8d2"
PARCEL_FILL = "#f3f2ee"
ROAD = "#b9b8b1"
WATER = "#dfe9f2"

ATTRIBUTION = (
    "Addresses: G-NAF © Geoscape Australia · Streets: © OpenStreetMap contributors · "
    "Parcels: © NSW Spatial Services (CC BY 4.0)"
)
ATTRIBUTION_PT = 8.5
"""Size of the source line: about 10 px high when a figure (2000-2300 px wide) is shown at the
README's ~900 px."""
ATTRIBUTION_X = 0.01
"""Left edge of the source line, as a fraction of the figure width (the right margin matches)."""

DPI = 200


def apply() -> None:
    """Set matplotlib defaults for the house style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 9.5,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.facecolor": SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 100,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def direction_colors(labels) -> list[str]:
    """Colour for each 8-point label; ``None`` maps to the muted ink."""
    return [DIRECTION_COLORS.get(label, INK_MUTED) for label in labels]


def compass_legend(ax: plt.Axes, *, title: str | None = "Facing", radius: float = 1.0) -> None:
    """Draw a labelled 8-wedge compass rose that doubles as the colour legend.

    ``ax`` should be a small, square, axis-off inset (for example from ``inset_axes``).
    """
    ax.set_aspect("equal")
    ax.set_xlim(-1.75 * radius, 1.75 * radius)
    ax.set_ylim(-1.75 * radius, 1.75 * radius)
    ax.axis("off")
    for k, label in enumerate(COMPASS_8):
        centre = 90 - 45 * k  # matplotlib angles are counter-clockwise from east
        ax.add_patch(
            Wedge((0, 0), radius, centre - 22.5, centre + 22.5, width=0.55 * radius,
                  facecolor=DIRECTION_COLORS[label], edgecolor=SURFACE, linewidth=1.5)
        )  # fmt: skip
        r = 1.38 * radius
        ax.text(
            r * np.cos(np.radians(centre)), r * np.sin(np.radians(centre)), label,
            ha="center", va="center", fontsize=8 if len(label) == 2 else 9,
            fontweight="bold" if len(label) == 1 else "normal", color=INK_SECONDARY,
        )  # fmt: skip
    if title:
        ax.text(0, 0, title, ha="center", va="center", fontsize=7.5, color=INK_MUTED)


def attribution(fig: plt.Figure, text: str = ATTRIBUTION, *, y: float = 0.012) -> plt.Text:
    """Muted data-source line along the bottom of a figure, always on one line.

    It is set at :data:`ATTRIBUTION_PT`. Every credit line fits its figure at that size in
    Helvetica; with a wider fallback font (DejaVu Sans on most Linux machines) the line is
    scaled down just enough to stay inside the figure.
    """
    artist = fig.text(
        ATTRIBUTION_X, y, text, fontsize=ATTRIBUTION_PT, color=INK_MUTED, ha="left", va="bottom"
    )
    room = fig.get_figwidth() * (1 - 2 * ATTRIBUTION_X)
    for _ in range(25):  # text width is only roughly proportional to its size (hinting)
        width = artist.get_window_extent().width / fig.dpi
        if width <= room:
            break
        artist.set_fontsize(artist.get_fontsize() * min(0.98, 0.99 * room / width))
    return artist


def save(fig: plt.Figure, path: Path, *, dpi: int = DPI) -> Path:
    """Save a figure as an optimised PNG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15,
                metadata={"Software": "property-orientation"})  # fmt: skip
    plt.close(fig)
    return path
