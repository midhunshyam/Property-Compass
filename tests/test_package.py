"""Package metadata, the public API and the ``python -m property_orientation`` entry point."""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

import property_orientation

REPO = Path(__file__).resolve().parents[1]


def test_version_is_the_same_everywhere() -> None:
    pyproject = (REPO / "pyproject.toml").read_text()
    citation = (REPO / "CITATION.cff").read_text()
    version = property_orientation.__version__

    assert re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE).group(1) == version
    assert re.search(r"^version: (\S+)$", citation, re.MULTILINE).group(1) == version


def test_public_api() -> None:
    assert set(property_orientation.__all__) == {
        "COMPASS_8", "__version__", "angular_difference", "compass8", "compute_orientation",
        "gnaf_street_key", "normalise_street_name", "opposite", "true_azimuth",
    }  # fmt: skip
    for name in property_orientation.__all__:
        assert hasattr(property_orientation, name)


def test_module_entry_point(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["property-orientation", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("property_orientation", run_name="__main__")

    assert exit_info.value.code == 0
    usage = capsys.readouterr().out
    for command in ("fetch", "run", "figures", "webmap", "all"):
        assert command in usage
