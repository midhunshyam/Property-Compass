"""Inputs for figures and the web map, loaded once and reprojected to the study CRS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
import pandas as pd

from property_orientation.config import Paths


@dataclass
class FigureData:
    """Lazy access to every layer a figure might need, in the projected study CRS.

    ``sites`` holds one row per unique location, street and geocode type, with the estimate
    (``orientation``, ``bearing_deg``, ``street_x``/``street_y``, ``confidence``), the
    baseline (``baseline_*``) and the cadastral reference (``ref_*``).
    ``addresses`` is the published one-row-per-address table.
    """

    paths: Paths

    @cached_property
    def summary(self) -> dict:
        return json.loads((self.paths.results / "run_summary.json").read_text())

    @cached_property
    def crs(self) -> str:
        return self.summary["crs"]

    @cached_property
    def sites(self) -> gpd.GeoDataFrame:
        return gpd.read_parquet(self.paths.processed / "sites.parquet").to_crs(self.crs)

    @cached_property
    def addresses(self) -> pd.DataFrame:
        return pd.read_csv(self.paths.results / "property_orientation.csv", dtype={"postcode": str})

    def _raw(self, name: str) -> gpd.GeoDataFrame:
        return gpd.read_parquet(self.paths.raw_file(name)).to_crs(self.crs)

    @cached_property
    def boundary(self) -> gpd.GeoDataFrame:
        return self._raw("boundary")

    @cached_property
    def localities(self) -> gpd.GeoDataFrame:
        return self._raw("localities")

    @cached_property
    def roads(self) -> gpd.GeoDataFrame:
        """All OSM highway ways (every class), as downloaded."""
        return self._raw("osm_highways")

    @cached_property
    def parcels(self) -> gpd.GeoDataFrame:
        """NSW property parcels with exact-duplicate (stacked strata) polygons removed."""
        from property_orientation.reference import unique_parcels

        return unique_parcels(self._raw("nsw_property"))

    @cached_property
    def road_reserves(self) -> gpd.GeoDataFrame:
        return self._raw("nsw_road_reserve")

    @cached_property
    def suburb_summary(self) -> pd.DataFrame:
        return pd.read_csv(self.paths.results / "summary_by_suburb.csv")

    def validation(self, name: str) -> pd.DataFrame:
        """A table from ``results/<area>/validation/`` (for example ``method_comparison``)."""
        return pd.read_csv(self.paths.validation / f"{name}.csv")
