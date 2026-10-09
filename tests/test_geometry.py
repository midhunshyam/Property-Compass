"""Bearings, compass sectors and coordinate reference system helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pyproj import CRS, Geod, Transformer

from property_orientation.geometry import (
    COMPASS_8,
    angular_difference,
    compass8,
    mga2020_epsg,
    opposite,
    true_azimuth,
)

GEOD = Geod(ellps="GRS80")
GDA2020 = 7844  # geographic CRS of every MGA2020 zone


class TestMga2020Epsg:
    @pytest.mark.parametrize(
        ("longitude", "epsg"),
        [
            (151.2, 7856),  # Sydney
            (115.9, 7850),  # Perth
            (144.96, 7855),  # Melbourne
            (153.03, 7856),  # Brisbane
            (138.6, 7854),  # Adelaide
            (130.84, 7852),  # Darwin
            (147.33, 7855),  # Hobart
            (90.0, 7846),  # western edge of zone 46
            (149.999, 7855),
            (150.0, 7856),  # a zone edge belongs to the zone east of it
            (173.999, 7859),
        ],
    )
    def test_zone_code(self, longitude: float, epsg: int) -> None:
        assert mga2020_epsg(longitude) == epsg

    @pytest.mark.parametrize("longitude", [96.8, 105.7, 115.86, 133.9, 151.2, 159.08, 167.95])
    def test_zone_covers_the_longitude(self, longitude: float) -> None:
        crs = CRS.from_epsg(mga2020_epsg(longitude))
        assert crs.name.startswith("GDA2020 / MGA zone ")
        assert crs.area_of_use.west <= longitude <= crs.area_of_use.east

    @pytest.mark.parametrize("longitude", [180.01, -181.0, math.nan])
    def test_invalid_longitude(self, longitude: float) -> None:
        with pytest.raises(ValueError, match="out of range"):
            mga2020_epsg(longitude)

    @pytest.mark.parametrize("longitude", [0.0, -151.2, 89.99, 174.0, 180.0])
    def test_longitude_outside_mga(self, longitude: float) -> None:
        with pytest.raises(ValueError, match="outside the MGA2020 zones"):
            mga2020_epsg(longitude)


class TestCompass8:
    @pytest.mark.parametrize(
        ("azimuth", "label"),
        [
            (0.0, "N"),
            (22.49, "N"),
            (22.5, "NE"),
            (45.0, "NE"),
            (67.49, "NE"),
            (67.5, "E"),
            (90.0, "E"),
            (112.5, "SE"),
            (157.5, "S"),
            (180.0, "S"),
            (202.5, "SW"),
            (247.5, "W"),
            (270.0, "W"),
            (292.5, "NW"),
            (337.49, "NW"),
            (337.5, "N"),
            (359.9, "N"),
        ],
    )
    def test_sector_boundaries(self, azimuth: float, label: str) -> None:
        assert compass8(azimuth).tolist() == [label]

    @pytest.mark.parametrize(
        ("azimuth", "label"),
        [
            (360.0, "N"),
            (-0.1, "N"),
            (-22.5, "N"),
            (-22.6, "NW"),
            (-90.0, "W"),
            (-315.0, "NE"),
            (405.0, "NE"),
            (720.0, "N"),
            (1000.0, "W"),  # 1000 - 720 = 280
        ],
    )
    def test_wraps_negative_and_large_angles(self, azimuth: float, label: str) -> None:
        assert compass8(azimuth).tolist() == [label]

    def test_each_sector_is_45_degrees_centred_on_its_direction(self) -> None:
        centres = np.arange(8) * 45.0
        for offset in (-22.5, -11.0, 0.0, 11.0, 22.49):
            assert compass8(centres + offset).tolist() == list(COMPASS_8)

    @pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf, None])
    def test_missing_azimuth_is_none(self, missing: float | None) -> None:
        assert compass8([missing, 90.0]).tolist() == [None, "E"]

    def test_keeps_shape_and_accepts_series(self) -> None:
        grid = compass8([[0.0, 90.0], [180.0, 270.0]])
        assert grid.shape == (2, 2)
        assert grid.tolist() == [["N", "E"], ["S", "W"]]
        assert compass8(pd.Series([45.0, np.nan])).tolist() == ["NE", None]
        assert compass8(10.0).shape == (1,)
        assert compass8([]).shape == (0,)


class TestAngularDifference:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (350.0, 10.0, 20.0),
            (10.0, 350.0, 20.0),
            (359.0, 1.0, 2.0),
            (0.0, 359.5, 0.5),
            (0.0, 180.0, 180.0),
            (45.0, 225.0, 180.0),
            (90.0, 90.0, 0.0),
            (-10.0, 10.0, 20.0),
            (720.0, 0.0, 0.0),
            (100.0, 300.0, 160.0),
        ],
    )
    def test_known_pairs(self, a: float, b: float, expected: float) -> None:
        assert angular_difference(a, b) == pytest.approx(expected)

    def test_symmetric_and_bounded(self) -> None:
        rng = np.random.default_rng(7856)
        a, b = rng.uniform(-720.0, 720.0, size=(2, 1000))
        d = angular_difference(a, b)
        np.testing.assert_allclose(d, angular_difference(b, a))
        assert ((d >= 0.0) & (d <= 180.0)).all()
        # Adding whole turns to either bearing changes nothing.
        np.testing.assert_allclose(d, angular_difference(a + 360.0, b - 720.0), atol=1e-9)

    def test_broadcasts_and_propagates_nan(self) -> None:
        d = angular_difference(np.array([0.0, 90.0, np.nan]), 45.0)
        np.testing.assert_allclose(d, [45.0, 45.0, np.nan])


class TestOpposite:
    @pytest.mark.parametrize(
        ("azimuth", "expected"),
        [(0.0, 180.0), (90.0, 270.0), (180.0, 0.0), (270.0, 90.0), (359.0, 179.0), (-90.0, 90.0)],
    )
    def test_reverse_bearing(self, azimuth: float, expected: float) -> None:
        assert opposite(azimuth) == pytest.approx(expected)

    def test_always_half_a_turn_away(self) -> None:
        azimuths = np.linspace(-360.0, 720.0, 97)
        np.testing.assert_allclose(angular_difference(azimuths, opposite(azimuths)), 180.0)
        assert ((opposite(azimuths) >= 0.0) & (opposite(azimuths) < 360.0)).all()


# (name, longitude, latitude): two points west of their zone's central meridian, where grid
# north lies east of true north, and two east of it, where it lies west of true north.
PLACES = [
    ("Sydney", 151.18, -33.80),
    ("Perth", 115.86, -31.95),
    ("Byron Bay", 153.61, -28.64),
    ("Darwin", 130.84, -12.46),
]


def _projected(longitude: float, latitude: float) -> tuple[int, float, float]:
    epsg = mga2020_epsg(longitude)
    x, y = Transformer.from_crs(GDA2020, epsg, always_xy=True).transform(longitude, latitude)
    return epsg, x, y


class TestTrueAzimuth:
    @pytest.mark.parametrize(("name", "longitude", "latitude"), PLACES)
    def test_grid_north_and_east_are_off_by_the_grid_convergence(
        self, name: str, longitude: float, latitude: float
    ) -> None:
        epsg, x, y = _projected(longitude, latitude)
        north = true_azimuth(x, y, x, y + 100.0, epsg)
        east = true_azimuth(x, y, x + 100.0, y, epsg)
        assert angular_difference(north, 0.0) < 1.1
        assert angular_difference(east, 90.0) < 1.1

        # The same geodesic, solved directly with pyproj from geographic coordinates.
        to_geo = Transformer.from_crs(epsg, GDA2020, always_xy=True)
        lon1, lat1 = to_geo.transform(x, y)
        lon2, lat2 = to_geo.transform(x, y + 100.0)
        expected, _, _ = GEOD.inv(lon1, lat1, lon2, lat2)
        assert angular_difference(north, expected) == pytest.approx(0.0, abs=1e-9)

        # Grid convergence of transverse Mercator: atan(tan(lon - central meridian) * sin(lat)).
        # West of the central meridian (southern hemisphere) it is positive, so grid north
        # has a true bearing just east of 0°; east of the meridian it is just west of 0°.
        params = {p.name: p.value for p in CRS.from_epsg(epsg).coordinate_operation.params}
        central_meridian = params["Longitude of natural origin"]
        convergence = math.degrees(
            math.atan(
                math.tan(math.radians(longitude - central_meridian))
                * math.sin(math.radians(latitude))
            )
        )
        assert ((north + 180.0) % 360.0) - 180.0 == pytest.approx(convergence, abs=1e-3)
        assert np.sign(expected) == np.sign(convergence) == np.sign(central_meridian - longitude)

    def test_sydney_grid_north_is_about_one_degree_east_of_true_north(self) -> None:
        epsg, x, y = _projected(151.18, -33.80)
        assert epsg == 7856
        assert true_azimuth(x, y, x, y + 50.0, epsg) == pytest.approx(1.01, abs=0.01)

    @pytest.mark.parametrize("azimuth", [0.0, 30.0, 90.0, 135.0, 180.0, 225.0, 270.0, 359.0])
    def test_recovers_a_geodesic_azimuth(self, azimuth: float) -> None:
        lon1, lat1 = 151.18, -33.80
        lon2, lat2, _ = GEOD.fwd(lon1, lat1, azimuth, 80.0)
        _, x1, y1 = _projected(lon1, lat1)
        _, x2, y2 = _projected(lon2, lat2)
        result = true_azimuth(x1, y1, x2, y2, 7856)
        assert angular_difference(result, azimuth) == pytest.approx(0.0, abs=1e-6)

    def test_vectorised_range_and_reverse(self) -> None:
        _, x, y = _projected(151.18, -33.80)
        angles = np.radians(np.arange(0.0, 360.0, 15.0))
        x2, y2 = x + 40.0 * np.sin(angles), y + 40.0 * np.cos(angles)
        forward = true_azimuth(np.full_like(x2, x), np.full_like(y2, y), x2, y2, 7856)
        assert forward.shape == angles.shape
        assert ((forward >= 0.0) & (forward < 360.0)).all()
        backward = true_azimuth(x2, y2, np.full_like(x2, x), np.full_like(y2, y), 7856)
        np.testing.assert_allclose(angular_difference(backward, opposite(forward)), 0.0, atol=0.01)

    def test_accepts_any_crs_form(self) -> None:
        _, x, y = _projected(151.18, -33.80)
        results = [
            true_azimuth(x, y, x + 30.0, y + 10.0, crs)
            for crs in (7856, "EPSG:7856", CRS.from_epsg(7856))
        ]
        assert results[1] == pytest.approx(results[0])
        assert results[2] == pytest.approx(results[0])

    def test_empty_input(self) -> None:
        assert true_azimuth([], [], [], [], 7856).shape == (0,)
