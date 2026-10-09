"""Bearings, compass sectors and coordinate reference system helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS, Geod, Transformer

COMPASS_8: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
"""The eight compass sectors, clockwise from north. Each sector is 45° wide."""

_GEOD = Geod(ellps="GRS80")  # the ellipsoid of GDA94 and GDA2020


def mga2020_epsg(longitude: float) -> int:
    """EPSG code of the GDA2020 / MGA zone containing ``longitude``.

    MGA zones are 6° wide UTM zones on the GDA2020 datum (EPSG:7846–7859). Zone 56,
    which covers Sydney, is EPSG:7856.
    """
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude out of range: {longitude}")
    zone = int((longitude + 180.0) // 6.0) + 1
    if not 46 <= zone <= 59:
        raise ValueError(f"longitude {longitude} is outside the MGA2020 zones (46-59)")
    return 7800 + zone


def true_azimuth(
    x1: ArrayLike, y1: ArrayLike, x2: ArrayLike, y2: ArrayLike, crs: int | str | CRS
) -> NDArray[np.float64]:
    """Geodesic azimuth from point 1 to point 2, in degrees clockwise from true north.

    Coordinates are given in a projected CRS. They are converted to geographic coordinates
    and the azimuth is solved on the GRS80 ellipsoid. That removes grid convergence: grid
    north and true north differ by about 1° in Sydney.
    """
    to_geographic = Transformer.from_crs(crs, CRS(crs).geodetic_crs, always_xy=True)
    lon1, lat1 = to_geographic.transform(np.asarray(x1, float), np.asarray(y1, float))
    lon2, lat2 = to_geographic.transform(np.asarray(x2, float), np.asarray(y2, float))
    azimuth, _, _ = _GEOD.inv(lon1, lat1, lon2, lat2)
    return np.mod(np.asarray(azimuth, dtype=float), 360.0)


def compass8(azimuth: ArrayLike) -> NDArray[np.object_]:
    """Map azimuths in degrees to 8-point compass labels. NaN maps to ``None``.

    Sectors are centred on the cardinal and intercardinal directions, so ``N`` covers
    [337.5°, 22.5°) and ``NE`` covers [22.5°, 67.5°).
    """
    az = np.atleast_1d(np.asarray(azimuth, dtype=float))
    labels = np.full(az.shape, None, dtype=object)
    ok = np.isfinite(az)
    sector = np.floor((np.mod(az[ok], 360.0) + 22.5) / 45.0).astype(int) % 8
    labels[ok] = np.asarray(COMPASS_8, dtype=object)[sector]
    return labels


def angular_difference(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Smallest absolute difference between two bearings, in degrees (0–180)."""
    d = np.abs(np.mod(np.asarray(a, float) - np.asarray(b, float), 360.0))
    return np.minimum(d, 360.0 - d)


def opposite(azimuth: ArrayLike) -> NDArray[np.float64]:
    """The reverse bearing (for example, the direction a property's rear faces)."""
    return np.mod(np.asarray(azimuth, float) + 180.0, 360.0)
