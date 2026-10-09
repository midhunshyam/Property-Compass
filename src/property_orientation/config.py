"""Study area and file layout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SHOWCASE = "willoughby"
"""Slug of the area the README figures are drawn for (they include a Castlecrag hero map)."""


@dataclass(frozen=True)
class StudyArea:
    """A local government area (LGA) to process."""

    lga: str = "Willoughby"
    state: str = "NSW"
    fetch_buffer_m: float = 500.0
    """Roads and cadastre are fetched this far beyond the LGA boundary. Addresses near the
    edge can then still match streets just outside it."""

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.lga.lower()).strip("-")

    @property
    def is_showcase(self) -> bool:
        return self.slug == SHOWCASE


@dataclass(frozen=True)
class Paths:
    """Where each stage reads and writes, relative to the repository root."""

    root: Path
    area: StudyArea

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw" / self.area.slug

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed" / self.area.slug

    @property
    def results(self) -> Path:
        return self.root / "results" / self.area.slug

    @property
    def validation(self) -> Path:
        return self.results / "validation"

    @property
    def images(self) -> Path:
        return self.root / "docs" / "images"

    @property
    def webmap(self) -> Path:
        """``docs/map`` for the showcase area (served by GitHub Pages), else ``docs/map/<area>``."""
        base = self.root / "docs" / "map"
        return base if self.area.is_showcase else base / self.area.slug

    def raw_file(self, name: str) -> Path:
        return self.raw / f"{name}.parquet"
