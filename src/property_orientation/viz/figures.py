"""Render every README figure into ``docs/images/``."""

from __future__ import annotations

import logging
from pathlib import Path

from property_orientation.config import Paths
from property_orientation.viz import charts, explainer, maps, style
from property_orientation.viz.data import FigureData

log = logging.getLogger("property_orientation")

FIGURES = {
    "hero_castlecrag.png": maps.hero_map,
    "overview_willoughby.png": maps.overview_map,
    "northerly_by_suburb.png": maps.northerly_choropleth,
    "method_explainer.png": explainer.method_explainer,
    "validation_accuracy.png": charts.validation_chart,
    "confusion_matrix.png": charts.confusion_chart,
    "orientation_roses.png": charts.rose_small_multiples,
}


def render_all(paths: Paths) -> list[Path]:
    """Render all figures; each function takes ``(data, out_path)``.

    The Agg backend is forced so that the PNGs do not depend on the platform's interactive
    backend (the macOS backend lays text out slightly differently).
    """
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    style.apply()
    data = FigureData(paths)
    written = []
    for name, render in FIGURES.items():
        out = paths.images / name
        log.info("rendering %s", name)
        render(data, out)
        written.append(out)
    return written
