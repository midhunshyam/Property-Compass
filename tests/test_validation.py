"""Agreement between estimated orientations and the cadastral reference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from property_orientation.geometry import COMPASS_8
from property_orientation.orientation import compute_orientation
from property_orientation.reference import reference_orientation
from property_orientation.validation import (
    METHODS,
    agreement,
    by_group,
    compare_methods,
    confusion_matrix,
    evaluation_set,
)

STATS = [
    "sites", "same_8_point_direction", "within_22_5_deg", "within_45_deg",
    "median_error_deg", "p90_error_deg",
]  # fmt: skip


def series(*values: float) -> pd.Series:
    return pd.Series(values, dtype=float)


class TestAgreement:
    def test_hand_made_bearings(self) -> None:
        # Errors 10, 10, 2, 90 and 1 degrees; only the fourth pair (S vs W) differs in sector.
        stats = agreement(series(0, 90, 359, 180, 45), series(10, 100, 1, 270, 44))

        assert list(stats) == STATS
        assert stats["sites"] == 5
        assert stats["same_8_point_direction"] == pytest.approx(0.8)
        assert stats["within_22_5_deg"] == pytest.approx(0.8)
        assert stats["within_45_deg"] == pytest.approx(0.8)
        assert stats["median_error_deg"] == pytest.approx(10.0)
        assert stats["p90_error_deg"] == pytest.approx(58.0)  # 10 + 0.6 * (90 - 10)

    def test_359_and_1_degree_agree(self) -> None:
        stats = agreement(series(359), series(1))
        assert stats["same_8_point_direction"] == 1.0
        assert stats["median_error_deg"] == pytest.approx(2.0)

    def test_sector_agreement_is_stricter_than_angular_error(self) -> None:
        # Two degrees apart, but on either side of the N/NE boundary at 22.5°.
        stats = agreement(series(21.5), series(23.5))
        assert stats["same_8_point_direction"] == 0.0
        assert stats["within_22_5_deg"] == 1.0

    def test_thresholds_are_inclusive(self) -> None:
        stats = agreement(series(0, 0, 0), series(22.5, 45, 45.5))
        assert stats["within_22_5_deg"] == pytest.approx(1 / 3)
        assert stats["within_45_deg"] == pytest.approx(2 / 3)

    def test_pairs_with_a_missing_bearing_are_dropped(self) -> None:
        stats = agreement(series(0, np.nan, 90), series(np.nan, 10, 95))
        assert stats["sites"] == 1
        assert stats["median_error_deg"] == pytest.approx(5.0)

    def test_no_pairs(self) -> None:
        assert agreement(series(np.nan), series(1)) == {"sites": 0}
        assert agreement(series(), series()) == {"sites": 0}


def make_sites() -> pd.DataFrame:
    """Hand-made sites: reference, street-matched estimate and nearest-way baseline."""
    return pd.DataFrame(
        {
            "ref_bearing_deg": [1.0, 3.0, 91.0, 181.0, 271.0, 46.0, np.nan, 10.0],
            "ref_status": ["ok"] * 6 + ["short_frontage", "ok"],
            "bearing_deg": [359.0, 30.0, 92.0, 180.0, 270.0, 44.0, 0.0, np.nan],
            "baseline_bearing_deg": [1.0, 3.0, 2.0, 270.0, np.nan, 44.0, 0.0, 10.0],
            "confidence": ["high", "high", "high", "medium", "low", "medium", "high", "none"],
        }
    )


def test_evaluation_set_needs_a_usable_reference_and_an_estimate() -> None:
    sites = make_sites()
    assert evaluation_set(sites).index.tolist() == [0, 1, 2, 3, 4, 5]


class TestCompareMethods:
    def test_one_row_per_method_on_the_same_sites(self) -> None:
        sites = make_sites()
        table = compare_methods(sites)

        assert list(table.columns) == ["method", "description", *STATS]
        assert table["method"].tolist() == list(METHODS)
        assert table["description"].tolist() == [label for _, label in METHODS.values()]
        ev = evaluation_set(sites)
        for row, (column, _) in zip(table.to_dict("records"), METHODS.values(), strict=True):
            expected = agreement(ev[column], ev["ref_bearing_deg"])
            assert {k: row[k] for k in STATS} == pytest.approx(expected)

        rows = table.set_index("method")
        assert rows.loc["street_matched", "sites"] == 6
        assert rows.loc["street_matched", "same_8_point_direction"] == pytest.approx(5 / 6)
        assert rows.loc["nearest_road", "sites"] == 5  # the site without a baseline drops out
        assert rows.loc["nearest_road", "same_8_point_direction"] == pytest.approx(3 / 5)

    def test_without_a_baseline(self) -> None:
        table = compare_methods(make_sites().drop(columns="baseline_bearing_deg"))
        assert table["method"].tolist() == ["street_matched"]


def test_by_group() -> None:
    sites = make_sites()
    table = by_group(sites, "confidence")

    assert table["confidence"].tolist() == ["high", "low", "medium"]
    assert table["sites"].tolist() == [3, 1, 2]
    assert table["same_8_point_direction"].tolist() == pytest.approx([2 / 3, 1.0, 1.0])
    baseline = by_group(sites, "confidence", column="baseline_bearing_deg")
    assert baseline["sites"].tolist() == [3, 0, 2]


class TestConfusionMatrix:
    def test_counts_reference_against_estimate(self) -> None:
        sites = make_sites()
        matrix = confusion_matrix(sites)

        assert matrix.shape == (8, 8)
        assert list(matrix.index) == list(COMPASS_8)
        assert list(matrix.columns) == list(COMPASS_8)
        assert (matrix.index.name, matrix.columns.name) == ("reference", "estimate")
        assert matrix.to_numpy().sum() == len(evaluation_set(sites))
        assert matrix.loc["N", "N"] == 1  # 1° vs 359°
        assert matrix.loc["N", "NE"] == 1  # 3° vs 30°
        assert matrix.loc["E", "E"] == matrix.loc["S", "S"] == matrix.loc["W", "W"] == 1
        assert matrix.loc["NE", "NE"] == 1

    def test_diagonal_share_is_the_same_direction_rate(self) -> None:
        sites = make_sites()
        matrix = confusion_matrix(sites).to_numpy()
        stats = compare_methods(sites).set_index("method").loc["street_matched"]
        assert np.trace(matrix) / matrix.sum() == pytest.approx(stats["same_8_point_direction"])

    def test_baseline_column(self) -> None:
        matrix = confusion_matrix(make_sites(), column="baseline_bearing_deg")
        assert matrix.loc["E", "N"] == 1  # 91° vs 2°
        assert matrix.loc["S", "W"] == 1
        assert matrix.shape == (8, 8)  # sites without a baseline bearing are not counted
        assert matrix.to_numpy().sum() == 5

    def test_empty_evaluation_set_still_has_every_direction(self) -> None:
        sites = make_sites().assign(ref_status="no_parcel")
        matrix = confusion_matrix(sites)
        assert matrix.shape == (8, 8)
        assert matrix.to_numpy().sum() == 0


def test_synthetic_block_end_to_end(block) -> None:
    """Estimate, baseline and cadastral reference on the synthetic block."""
    addresses = block.scene.addresses(
        block.address("corner", "SMITH", "STREET"),
        block.address("corner", "JONES", "AVENUE"),
        block.address("inner", "SMITH", "STREET"),
        block.address("front", "SMITH", "STREET"),
        block.address("panhandle", "SMITH", "STREET"),
        block.address("rear", "SMITH", "STREET"),
        block.address("north", "SMITH", "STREET"),
        block.address("east", "JONES", "AVENUE"),
    )
    result = compute_orientation(addresses, block.roads, crs=block.scene.crs)
    reference = reference_orientation(
        result.sites, block.parcels, block.reserves, block.centrelines
    )
    sites = result.sites.join(reference)

    assert sites["orientation"].tolist() == ["N", "E", "N", "N", "N", "N", "S", "W"]
    assert sites["ref_status"].tolist() == [
        "ok", "ok", "ok", "ok", "short_frontage", "no_street_frontage", "ok", "ok"
    ]  # fmt: skip
    # The corner lot on Smith Street snaps to Jones Avenue and the front lot to the driveway.
    assert sites["baseline_orientation"].iloc[[0, 3]].tolist() == ["E", "W"]
    assert sites["baseline_highway"].iloc[[0, 2, 3]].tolist() == [
        "residential", "footway", "service"
    ]  # fmt: skip

    table = compare_methods(sites).set_index("method")
    assert table.loc["street_matched", "sites"] == 6
    assert table.loc["street_matched", "same_8_point_direction"] == 1.0
    assert table.loc["nearest_road", "same_8_point_direction"] == pytest.approx(4 / 6)
    assert confusion_matrix(sites, "baseline_bearing_deg").loc["N", ["E", "W"]].tolist() == [1, 1]
