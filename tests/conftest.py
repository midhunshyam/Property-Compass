"""Shared fixtures: synthetic G-NAF, OpenStreetMap and NSW cadastre layers.

Synthetic layers are laid out in GDA2020 / MGA zone 56 (EPSG:7856) metres, as offsets
from an origin in Chatswood, NSW. Grid north there is about 1.0° east of true north, so a
property facing grid north has a true bearing of about 1°, well inside the ``N`` sector.

The suite runs offline: :func:`offline` fails any test that tries to use the network.
"""

from __future__ import annotations

import socket
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import geopandas as gpd
import pytest
import shapely

Offset = tuple[float, float]
"""(east, north) offset in metres from the scene origin."""

_LOOPBACK = (None, "localhost", "127.0.0.1", "::1")


@pytest.fixture(scope="session", autouse=True)
def offline() -> Iterator[None]:
    """Fail any test that looks up a remote host or opens an IPv4 or IPv6 connection.

    The downloaders retry on network errors, so the guard raises ``pytest.fail`` (a
    ``BaseException``), which their ``except`` clauses cannot swallow.
    """
    getaddrinfo, connect = socket.getaddrinfo, socket.socket.connect

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> list:
        if host not in _LOOPBACK:
            pytest.fail(f"network access in an offline test: lookup of {host!r}")
        return getaddrinfo(host, *args, **kwargs)

    def guarded_connect(self: socket.socket, address: object) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            pytest.fail(f"network access in an offline test: connection to {address!r}")
        connect(self, address)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
        patch.setattr(socket.socket, "connect", guarded_connect)
        yield


@dataclass(frozen=True)
class Scene:
    """Builds synthetic layers from offsets in metres around an origin in EPSG:7856."""

    x0: float = 331_500.0
    y0: float = 6_258_500.0
    crs: int = 7856

    def xy(self, dx: float, dy: float) -> tuple[float, float]:
        """Absolute EPSG:7856 coordinates of an offset."""
        return self.x0 + dx, self.y0 + dy

    def point(self, dx: float, dy: float) -> shapely.Point:
        return shapely.Point(self.xy(dx, dy))

    def line(self, coords: Sequence[Offset]) -> shapely.LineString:
        return shapely.LineString([self.xy(dx, dy) for dx, dy in coords])

    def box(self, xmin: float, ymin: float, xmax: float, ymax: float) -> shapely.Polygon:
        return shapely.box(self.x0 + xmin, self.y0 + ymin, self.x0 + xmax, self.y0 + ymax)

    def frame(
        self, geometries: Iterable[shapely.Geometry], **columns: Sequence
    ) -> gpd.GeoDataFrame:
        """A GeoDataFrame of absolute geometries in the scene CRS (``columns`` as keywords)."""
        return gpd.GeoDataFrame(columns, geometry=list(geometries), crs=self.crs)

    def addresses(self, *rows: dict) -> gpd.GeoDataFrame:
        """G-NAF-like address points.

        Each row gives ``at`` (an offset) and any G-NAF columns, usually ``street_name``
        and ``street_type``. Unspecified columns get G-NAF-like defaults.
        """
        records = []
        for k, row in enumerate(rows):
            fields = dict(row)
            at = fields.pop("at")
            name = " ".join(
                str(fields.get(c)) for c in ("street_name", "street_type") if fields.get(c)
            )
            defaults = {
                "gnaf_pid": f"GANSWTEST{k:05d}",
                "street_name": None,
                "street_type": None,
                "street_suffix": None,
                "geocode_type": "PROPERTY CENTROID",
                "primary_secondary": None,
                "address": f"{k + 1} {name}".strip(),
                "locality_name": "CHATSWOOD",
                "state": "NSW",
                "postcode": "2067",
            }
            records.append(defaults | fields | {"geometry": self.point(*at)})
        return gpd.GeoDataFrame(records, geometry="geometry", crs=self.crs)

    def roads(self, *rows: dict) -> gpd.GeoDataFrame:
        """OSM-like highway ways. Each row gives ``highway``, ``line`` (offsets) and tags."""
        records = []
        for k, row in enumerate(rows):
            fields = dict(row)
            line = fields.pop("line")
            defaults = dict.fromkeys(
                ("name", "alt_name", "official_name", "old_name", "service", "access", "area")
            )
            records.append({"osm_id": 1000 + k} | defaults | fields | {"geometry": self.line(line)})
        return gpd.GeoDataFrame(records, geometry="geometry", crs=self.crs)


@pytest.fixture
def scene() -> Scene:
    return Scene()


@dataclass(frozen=True)
class Block:
    """A street block with OSM ways and NSW-style cadastre, built on a :class:`Scene`.

    Smith Street runs east-west: road reserve y = 0..20 m, centreline y = 10. Jones Avenue
    runs north-south and ends at Smith Street: reserve x = 15..35, centreline x = 25. A
    footway runs along the south kerb of Smith Street (y = 2) and an unnamed driveway runs
    north-south through the panhandle lot's access strip (x = -31).

    =========  ============================================  ============================
    lot        extent (x, y offsets in m)                    frontage
    =========  ============================================  ============================
    corner     0..15, -40..0                                 Smith Street and Jones Avenue
    inner      -16..0, -40..0                                Smith Street
    front      -30..-16, -40..0                              Smith Street (driveway to W)
    panhandle  2 m access strip -32..-30, -40..0 plus body   2 m onto Smith Street
               -60..-30, -80..-40
    rear       -30..0, -80..-40                              none
    north      -15..0, 20..60                                Smith Street, from the north
    east       35..50, -80..-40                              Jones Avenue, from the east
    =========  ============================================  ============================
    """

    scene: Scene
    roads: gpd.GeoDataFrame
    parcels: gpd.GeoDataFrame
    reserves: gpd.GeoDataFrame
    centrelines: gpd.GeoDataFrame
    lots: dict[str, Offset]
    """An address point inside each lot."""

    def address(self, lot: str, street_name: str, street_type: str, **fields: object) -> dict:
        """An address row (for :meth:`Scene.addresses`) at the given lot."""
        at = self.lots[lot]
        return {"at": at, "street_name": street_name, "street_type": street_type} | fields

    def parcel_index(self, lot: str) -> int:
        """Row position of a lot in :attr:`parcels`."""
        return list(self.parcels["lot"]).index(lot)


@pytest.fixture
def block(scene: Scene) -> Block:
    s = scene
    roads = s.roads(
        {"highway": "residential", "name": "Smith Street", "line": [(-200, 10), (200, 10)]},
        {"highway": "residential", "name": "Jones Avenue", "line": [(25, -200), (25, 10)]},
        {"highway": "footway", "line": [(-200, 2), (200, 2)]},
        {"highway": "service", "service": "driveway", "line": [(-31, 10), (-31, -50)]},
    )
    lots = {
        "corner": s.box(0, -40, 15, 0),
        "inner": s.box(-16, -40, 0, 0),
        "front": s.box(-30, -40, -16, 0),
        "panhandle": shapely.union(s.box(-32, -40, -30, 0), s.box(-60, -80, -30, -40)),
        "rear": s.box(-30, -80, 0, -40),
        "north": s.box(-15, 20, 0, 60),
        "east": s.box(35, -80, 50, -40),
    }
    parcels = s.frame(lots.values(), lot=list(lots))
    reserves = s.frame([s.box(-200, 0, 200, 20), s.box(15, -200, 35, 0)])
    centrelines = s.frame(
        [s.line([(-200, 10), (200, 10)]), s.line([(25, -200), (25, 10)])],
        roadnamelabel=["SMITH ST", "JONES AVE"],
    )
    points = {
        "corner": (7.5, -20.0),
        "inner": (-8.0, -20.0),
        "front": (-23.0, -20.0),
        "panhandle": (-45.0, -60.0),
        "rear": (-15.0, -60.0),
        "north": (-7.5, 40.0),
        "east": (42.5, -60.0),
    }
    return Block(s, roads, parcels, reserves, centrelines, points)
