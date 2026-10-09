"""Agreement between estimated orientations and the cadastral reference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from property_orientation.geometry import COMPASS_8, angular_difference, compass8

METHODS = {
    "street_matched": ("bearing_deg", "Street-matched frontage (this repository)"),
    "nearest_road": ("baseline_bearing_deg", "Nearest OSM way of any class (v1 approach)"),
}


def evaluation_set(sites: pd.DataFrame) -> pd.DataFrame:
    """Sites that have both an estimated bearing and a usable cadastral reference."""
    ok = sites["ref_status"].eq("ok") & sites["bearing_deg"].notna()
    return sites.loc[ok]


def published_sector(bearing: ArrayLike) -> np.ndarray:
    """8-point label of a bearing as published: rounded to 0.1° first, like the result tables."""
    return compass8(np.mod(np.round(np.asarray(bearing, float), 1), 360.0))


def agreement(estimate: pd.Series, reference: pd.Series) -> dict[str, float]:
    """Agreement statistics for two aligned bearing series, in degrees.

    Sector agreement compares the published (0.1°-rounded) labels; angular errors use the
    unrounded bearings.
    """
    valid = estimate.notna() & reference.notna()
    est, ref = estimate[valid].to_numpy(), reference[valid].to_numpy()
    error = angular_difference(est, ref)
    if not len(error):
        return {"sites": 0}
    return {
        "sites": len(error),
        "same_8_point_direction": float(np.mean(published_sector(est) == published_sector(ref))),
        "within_22_5_deg": float(np.mean(error <= 22.5)),
        "within_45_deg": float(np.mean(error <= 45.0)),
        "median_error_deg": float(np.median(error)),
        "p90_error_deg": float(np.percentile(error, 90)),
    }


def scores(frame: pd.DataFrame, column: str) -> dict[str, float]:
    """:func:`agreement` for one estimate column, plus the number of distinct reference parcels.

    A large site with many address points (a campus, say) is one parcel but many sites, so the
    parcel count shows how concentrated the evaluated sites are.
    """
    stats = agreement(frame[column], frame["ref_bearing_deg"])
    if "ref_parcel" not in frame or not stats["sites"]:
        return stats
    valid = frame[column].notna() & frame["ref_bearing_deg"].notna()
    parcels = int(frame.loc[valid, "ref_parcel"].nunique())
    return {"sites": stats.pop("sites"), "parcels": parcels} | stats


def compare_methods(sites: pd.DataFrame) -> pd.DataFrame:
    """Agreement of each method with the reference, on the same set of sites."""
    ev = evaluation_set(sites)
    rows = [
        {"method": key, "description": label} | scores(ev, column)
        for key, (column, label) in METHODS.items()
        if column in ev
    ]
    return pd.DataFrame(rows)


def by_group(
    sites: pd.DataFrame,
    group: str,
    column: str = "bearing_deg",
    order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Agreement of one method with the reference, per value of ``group``.

    Rows are sorted by ``order`` when given (values missing from the data are skipped),
    otherwise alphabetically.
    """
    ev = evaluation_set(sites)
    groups = dict(tuple(ev.groupby(group, sort=True)))
    names = [name for name in order if name in groups] if order else list(groups)
    rows = [{group: name} | scores(groups[name], column) for name in names]
    return pd.DataFrame(rows)


def confusion_matrix(sites: pd.DataFrame, column: str = "bearing_deg") -> pd.DataFrame:
    """Counts of reference direction (rows) against estimated direction (columns)."""
    ev = evaluation_set(sites)
    ref = pd.Categorical(published_sector(ev["ref_bearing_deg"]), categories=COMPASS_8).codes
    est = pd.Categorical(published_sector(ev[column]), categories=COMPASS_8).codes
    ok = (ref >= 0) & (est >= 0)
    counts = np.zeros((len(COMPASS_8), len(COMPASS_8)), dtype=np.int64)
    np.add.at(counts, (ref[ok], est[ok]), 1)  # always 8 x 8, even with no sites
    return pd.DataFrame(
        counts,
        index=pd.Index(list(COMPASS_8), name="reference"),
        columns=pd.Index(list(COMPASS_8), name="estimate"),
    )
