"""Integration tests on small real-data extracts, offline (see ``fixtures/README.md``).

Each extract is a 350 m window of the Willoughby inputs (G-NAF, OpenStreetMap and the NSW
cadastre) with the same file layout as ``data/raw/<area>``, so the pipeline runs on it
unchanged. The tests do not need ``data/raw``; the two provenance checks at the end run only
when the local raw data or the committed results come from the fixtures' snapshot.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
import shapely
from geopandas.testing import assert_geodataframe_equal

from property_orientation import cli, pipeline, validation
from property_orientation.config import Paths, StudyArea
from property_orientation.geometry import COMPASS_8, compass8
from property_orientation.orientation import (
    FRONTAGE_HIGHWAYS,
    OrientationResult,
    compute_orientation,
)
from property_orientation.outputs import ADDRESS_TABLE_COLUMNS
from property_orientation.reference import reference_orientation

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
AREAS = ("castlecrag", "northbridge")
LAYERS = (
    "addresses",
    "osm_highways",
    "nsw_property",
    "nsw_road_reserve",
    "nsw_road_centreline",
    "localities",
    "boundary",
)
CADASTRE = ("nsw_property", "nsw_road_reserve", "nsw_road_centreline")
MIN_AGREEMENT = 0.85
"""Smallest acceptable share of sites facing the same 8-point direction as the cadastre."""
MAX_FIXTURE_BYTES = 600 * 1024


@dataclass(frozen=True)
class Extract:
    """One real-data fixture: its layers (EPSG:4326, as downloaded) and manifest."""

    name: str
    layers: dict[str, gpd.GeoDataFrame]
    manifest: dict

    @property
    def crs(self) -> int:
        return int(self.manifest["crs"].removeprefix("EPSG:"))

    def install(self, paths: Paths) -> None:
        """Copy the extract to where the pipeline expects downloaded inputs."""
        paths.raw.mkdir(parents=True, exist_ok=True)
        for name in (*(f"{layer}.parquet" for layer in LAYERS), "manifest.json"):
            shutil.copy(FIXTURES / self.name / name, paths.raw / name)


def load_extract(name: str) -> Extract:
    folder = FIXTURES / name
    layers = {layer: gpd.read_parquet(folder / f"{layer}.parquet") for layer in LAYERS}
    manifest = json.loads((folder / "manifest.json").read_text())
    return Extract(name, layers, manifest)


def read_table(path: Path) -> pd.DataFrame:
    """A published address table, with postcodes kept as text."""
    return pd.read_csv(path, dtype={"postcode": str})


@pytest.fixture(scope="module", params=AREAS)
def extract(request: pytest.FixtureRequest) -> Extract:
    return load_extract(request.param)


@pytest.fixture(scope="module")
def validated(extract: Extract) -> tuple[OrientationResult, pd.DataFrame]:
    """The orientation result, and its sites joined to the cadastral reference."""
    layers = extract.layers
    result = compute_orientation(layers["addresses"], layers["osm_highways"], crs=extract.crs)
    reference = reference_orientation(
        result.sites,
        layers["nsw_property"],
        layers["nsw_road_reserve"],
        layers["nsw_road_centreline"],
    )
    return result, result.sites.join(reference)


def test_extract_matches_its_manifest(extract: Extract) -> None:
    counts = {layer: len(frame) for layer, frame in extract.layers.items() if layer != "boundary"}
    assert extract.manifest["counts"] == counts
    assert extract.crs == 7856
    assert all(frame.crs.to_epsg() == 4326 for frame in extract.layers.values())

    window = shapely.box(*extract.manifest["fixture"]["window_epsg7856"])
    assert window.area == pytest.approx(350.0**2)
    points = extract.layers["addresses"].geometry.to_crs(extract.crs)
    assert points.within(window).all()


def test_street_matching_agrees_with_the_cadastre(validated) -> None:
    _, sites = validated
    table = validation.compare_methods(sites).set_index("method")
    street, nearest = table.loc["street_matched"], table.loc["nearest_road"]

    assert street["sites"] >= 80
    assert street["same_8_point_direction"] >= MIN_AGREEMENT
    assert street["same_8_point_direction"] > nearest["same_8_point_direction"] + 0.2
    assert street["within_45_deg"] > nearest["within_45_deg"]
    assert street["p90_error_deg"] < nearest["p90_error_deg"]


def test_baseline_errors_come_from_ways_a_property_cannot_front(validated) -> None:
    _, sites = validated
    ev = validation.evaluation_set(sites)
    wrong = ev[compass8(ev["baseline_bearing_deg"]) != compass8(ev["ref_bearing_deg"])]
    # Footways, paths, steps, cycleways and service ways (mostly driveways).
    not_a_street = ~wrong["baseline_highway"].isin(FRONTAGE_HIGHWAYS - {"service"})
    assert len(wrong) >= 20
    assert not_a_street.mean() >= 0.75


def test_every_address_belongs_to_one_site(extract: Extract, validated) -> None:
    result, sites = validated
    addresses = extract.layers["addresses"]

    assert len(result.site_of_address) == len(addresses)
    assert sites["n_addresses"].sum() == len(addresses)
    assert np.bincount(result.site_of_address).tolist() == sites["n_addresses"].tolist()
    labels = [None if pd.isna(v) else v for v in sites["orientation"]]
    assert labels == compass8(sites["bearing_deg"]).tolist()
    assert sites.loc[sites["confidence"].eq("none"), "orientation"].isna().all()
    assert sites.loc[sites["confidence"].ne("none"), "orientation"].notna().all()
    assert set(sites["ref_status"]) <= {
        "ok", "no_parcel", "no_street_frontage", "short_frontage", "ambiguous_frontage"
    }  # fmt: skip


@pytest.fixture(scope="module")
def run_paths(extract: Extract, tmp_path_factory: pytest.TempPathFactory) -> Paths:
    """Paths of a ``property-orientation run`` on a copy of the extract."""
    paths = Paths(root=tmp_path_factory.mktemp(extract.name), area=StudyArea(lga=extract.name))
    extract.install(paths)
    assert cli.main(["--root", str(paths.root), "--lga", extract.name, "run"]) == 0
    return paths


def test_run_writes_the_address_table(extract: Extract, run_paths: Paths) -> None:
    table = read_table(run_paths.results / "property_orientation.csv")
    addresses = extract.layers["addresses"]

    assert list(table.columns) == list(ADDRESS_TABLE_COLUMNS)
    assert table["gnaf_pid"].tolist() == addresses["gnaf_pid"].tolist()
    assert set(table["address_kind"]) <= {"standalone", "unit", "complex"}
    assert set(table["orientation"].dropna()) <= set(COMPASS_8)
    bearing = table["bearing_deg"].dropna()
    assert ((bearing >= 0.0) & (bearing < 360.0)).all()
    assert (table["orientation"].isna() == table["bearing_deg"].isna()).all()
    oriented = table.dropna(subset=["bearing_deg"])
    # The published label always matches the published (rounded) bearing.
    assert (compass8(oriented["bearing_deg"]) == oriented["orientation"]).all()

    shares = pd.read_csv(run_paths.results / "summary_orientation.csv")
    assert shares["scope"].tolist() == ["all addresses", "standalone addresses"]
    np.testing.assert_allclose(
        shares[[f"share_{d}" for d in COMPASS_8]].sum(axis=1), 1.0, atol=1e-3
    )

    suburbs = pd.read_csv(run_paths.results / "summary_by_suburb.csv")
    assert suburbs["addresses"].sum() == len(addresses)
    assert set(suburbs["locality"]) == set(addresses["locality_name"])


def test_run_writes_the_validation(validated, run_paths: Paths) -> None:
    _, sites = validated
    methods = pd.read_csv(run_paths.validation / "method_comparison.csv")
    expected = validation.compare_methods(sites)
    pd.testing.assert_frame_equal(methods, expected, check_dtype=False)

    matrix = pd.read_csv(run_paths.validation / "confusion_matrix.csv", index_col=0)
    assert matrix.index.tolist() == matrix.columns.tolist() == list(COMPASS_8)
    assert matrix.to_numpy().sum() == methods.loc[0, "sites"]

    coverage = pd.read_csv(run_paths.validation / "reference_coverage.csv")
    assert coverage["sites"].sum() == len(sites)
    by_confidence = pd.read_csv(run_paths.validation / "by_confidence.csv")
    assert by_confidence["sites"].sum() == methods.loc[0, "sites"]
    assert (run_paths.validation / "by_suburb.csv").exists()

    summary = json.loads((run_paths.results / "run_summary.json").read_text())
    assert summary["crs"] == "EPSG:7856"
    assert summary["sites"] == len(sites)
    assert summary["validation"]["methods"][0]["same_8_point_direction"] >= MIN_AGREEMENT


def test_run_writes_the_geopackage_and_sites(extract: Extract, run_paths: Paths) -> None:
    layers = dict(pyogrio.list_layers(run_paths.results / "property_orientation.gpkg"))
    assert list(layers) == ["addresses", "facing_lines", "suburbs", "study_area"]
    assert (layers["addresses"], layers["facing_lines"]) == ("Point", "LineString")
    assert {layers["suburbs"], layers["study_area"]} <= {"Polygon", "MultiPolygon"}
    points = gpd.read_file(run_paths.results / "property_orientation.gpkg", layer="addresses")
    assert len(points) == len(extract.layers["addresses"])
    assert points.crs.to_epsg() == 4326

    sites = gpd.read_parquet(run_paths.processed / "sites.parquet")
    assert sites["site_id"].tolist() == list(range(len(sites)))
    for column in ("orientation", "bearing_deg", "street_x", "baseline_highway", "ref_status"):
        assert column in sites.columns


def test_run_without_the_cadastre(extract: Extract, run_paths: Paths, tmp_path: Path) -> None:
    """Outside NSW there is no cadastre: the estimate is the same, only validation is skipped."""
    paths = Paths(root=tmp_path, area=StudyArea(lga=extract.name))
    extract.install(paths)
    for layer in CADASTRE:
        paths.raw_file(layer).unlink()
    summary = pipeline.run(paths.area, paths)

    assert "validation" not in summary
    assert not paths.validation.exists()
    sites = gpd.read_parquet(paths.processed / "sites.parquet")
    assert not [column for column in sites.columns if column.startswith("ref_")]
    pd.testing.assert_frame_equal(
        read_table(paths.results / "property_orientation.csv"),
        read_table(run_paths.results / "property_orientation.csv"),
    )
    layers = pyogrio.list_layers(paths.results / "property_orientation.gpkg")
    assert layers[:, 0].tolist() == ["addresses", "facing_lines", "suburbs", "study_area"]


def test_fetch_reuses_cached_inputs_offline(
    extract: Extract, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from property_orientation.sources import gnaf, nsw_cadastre, osm

    def offline(*args: object, **kwargs: object) -> None:
        raise AssertionError("fetch tried to download a layer that is already cached")

    for module, name in [
        (gnaf, "fetch_lga"),
        (gnaf, "fetch_localities"),
        (gnaf, "fetch_addresses"),
        (osm, "fetch_highways"),
        (nsw_cadastre, "fetch_layer"),
    ]:
        monkeypatch.setattr(module, name, offline)

    area = StudyArea(lga=extract.name)
    paths = Paths(root=tmp_path, area=area)
    extract.install(paths)
    manifest = pipeline.fetch(area, paths)

    assert manifest["crs"] == "EPSG:7856"
    assert manifest["counts"] == extract.manifest["counts"]
    assert manifest["retrieved"] == extract.manifest["retrieved"]
    assert manifest["sources"] == extract.manifest["sources"]
    assert manifest["area"]["slug"] == extract.name
    written = json.loads((paths.raw / "manifest.json").read_text())
    assert written == json.loads(json.dumps(manifest))


class FakeSources:
    """Stand-ins for the downloaders that serve the Castlecrag extract and log each call."""

    TIMESTAMP = "2026-01-02T03:04:05Z"

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from property_orientation.sources import gnaf, nsw_cadastre, osm

        self.layers = load_extract("castlecrag").layers
        self.downloads: list[str] = []
        self.bounds: dict[str, tuple[float, ...]] = {}
        monkeypatch.setattr(gnaf, "fetch_lga", lambda lga, state: self.serve("boundary"))
        monkeypatch.setattr(gnaf, "fetch_localities", lambda area, state: self.serve("localities"))
        monkeypatch.setattr(gnaf, "fetch_addresses", lambda area: self.serve("addresses"))
        monkeypatch.setattr(osm, "fetch_highways", self.fetch_highways)
        monkeypatch.setattr(nsw_cadastre, "fetch_layer", self.fetch_layer)

    def serve(self, name: str) -> gpd.GeoDataFrame:
        self.downloads.append(name)
        return self.layers[name]

    def fetch_highways(self, bounds: tuple[float, ...]) -> tuple[gpd.GeoDataFrame, str]:
        self.bounds["osm_highways"] = bounds
        return self.serve("osm_highways"), self.TIMESTAMP

    def fetch_layer(self, layer: str, bounds: tuple[float, ...]) -> gpd.GeoDataFrame:
        self.bounds[f"nsw_{layer}"] = bounds
        return self.serve(f"nsw_{layer}")


@pytest.mark.parametrize("state", ["NSW", "VIC"])
def test_fetch_downloads_each_layer_once(
    state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = FakeSources(monkeypatch)
    monkeypatch.setattr(pipeline, "_today", lambda: "2026-01-02")
    area = StudyArea(lga="Castlecrag", state=state)
    paths = Paths(root=tmp_path, area=area)
    manifest = pipeline.fetch(area, paths)

    around_the_area = ["osm_highways", *(CADASTRE if state == "NSW" else ())]  # NSW only
    expected = ["boundary", "localities", "addresses", *around_the_area]
    assert sources.downloads == expected
    for name in expected:
        assert len(gpd.read_parquet(paths.raw_file(name))) == len(sources.layers[name])
    counted = [name for name in expected if name != "boundary"]
    assert manifest["counts"] == {name: len(sources.layers[name]) for name in counted}
    assert manifest["area"] == {"lga": "Castlecrag", "state": state, "slug": "castlecrag"}
    assert manifest["crs"] == "EPSG:7856"
    assert manifest["retrieved"] == "2026-01-02"
    assert manifest["sources"]["osm"]["timestamp"] == FakeSources.TIMESTAMP
    assert ("nsw_cadastre" in manifest["sources"]) == (state == "NSW")
    assert json.loads((paths.raw / "manifest.json").read_text()) == json.loads(json.dumps(manifest))

    # Roads and cadastre are fetched for the area plus a 500 m buffer.
    bounds = manifest["fetch_bounds_epsg4326"]
    assert sources.bounds == dict.fromkeys(around_the_area, bounds)
    fetched = gpd.GeoSeries([shapely.box(*bounds)], crs=4326).to_crs(7856).iloc[0]
    boundary = sources.layers["boundary"].to_crs(7856).union_all()
    assert fetched.contains(boundary.buffer(area.fetch_buffer_m - 1.0))
    assert not fetched.contains(boundary.buffer(area.fetch_buffer_m + 100.0))

    # A second fetch reuses the cache and keeps the dates of the data it holds.
    monkeypatch.setattr(pipeline, "_today", lambda: "2026-03-04")
    sources.downloads.clear()
    cached = pipeline.fetch(area, paths)
    assert sources.downloads == []
    assert cached == manifest

    refreshed = pipeline.fetch(area, paths, force=True)
    assert sources.downloads == expected
    assert refreshed["retrieved"] == "2026-03-04"
    if state == "NSW":
        assert refreshed["sources"]["nsw_cadastre"]["retrieved"] == "2026-03-04"


@pytest.mark.parametrize(
    ("lga", "slug"),
    [
        ("Willoughby", "willoughby"),
        ("Hunters Hill", "hunters-hill"),
        ("Ku-ring-gai", "ku-ring-gai"),
    ],
)
def test_study_area_paths(lga: str, slug: str) -> None:
    paths = Paths(root=Path("/repo"), area=StudyArea(lga=lga))
    assert paths.area.slug == slug
    assert paths.raw_file("addresses") == Path(f"/repo/data/raw/{slug}/addresses.parquet")
    assert paths.processed == Path(f"/repo/data/processed/{slug}")
    assert paths.validation == Path(f"/repo/results/{slug}/validation")


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])
    args = cli.build_parser().parse_args(["--lga", "Hunters Hill", "fetch", "--force"])
    assert args.lga == "Hunters Hill"
    assert (args.state, args.command, args.force) == ("NSW", "fetch", True)


# --- The fixtures themselves ---------------------------------------------------------------


def test_fixtures_are_small_and_attributed() -> None:
    files = [p for p in FIXTURES.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert sum(p.stat().st_size for p in files) <= MAX_FIXTURE_BYTES

    readme = (FIXTURES / "README.md").read_text()
    for text in ("make_fixture.py", "Open G-NAF End User Licence Agreement", "ODbL", "CC BY 4.0"):
        assert text in readme
    for area in AREAS:
        manifest = json.loads((FIXTURES / area / "manifest.json").read_text())
        licences = {source: meta["licence"] for source, meta in manifest["sources"].items()}
        assert licences == pipeline.LICENCES


@cache
def make_fixture() -> ModuleType:
    """The fixture builder, ``tests/fixtures/make_fixture.py``."""
    spec = importlib.util.spec_from_file_location("make_fixture", FIXTURES / "make_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def skip_unless_same_snapshot(path: Path, extract: Extract) -> None:
    """Skip unless the manifest or run summary at ``path`` lists the fixture's sources."""
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} is not available")
    if json.loads(path.read_text()).get("sources") != extract.manifest["sources"]:
        pytest.skip(f"{path.relative_to(REPO)} is from another snapshot than the fixtures")


def test_fixture_is_cut_from_the_raw_data_by_make_fixture(extract: Extract) -> None:
    raw = Paths(root=REPO, area=StudyArea())
    skip_unless_same_snapshot(raw.raw / "manifest.json", extract)
    builder = make_fixture()
    corner = builder.WINDOWS[extract.name]
    assert extract.manifest["fixture"]["window_epsg7856"][:2] == list(corner)

    layers = builder.subset(raw, corner)
    for layer in LAYERS:
        # Clipping can order vertices differently across GEOS versions: compare normalised shapes.
        assert_geodataframe_equal(
            layers[layer], extract.layers[layer], normalize=True, check_less_precise=True
        )


@cache
def published_table() -> pd.DataFrame:
    return read_table(REPO / "results" / "willoughby" / "property_orientation.csv")


def test_committed_results_match_a_run_on_the_fixture(extract: Extract, run_paths: Paths) -> None:
    """The fixture holds every input its addresses need, so their published rows reproduce."""
    skip_unless_same_snapshot(REPO / "results" / "willoughby" / "run_summary.json", extract)
    ours = read_table(run_paths.results / "property_orientation.csv")
    published = published_table().set_index("gnaf_pid").loc[ours["gnaf_pid"]].reset_index()

    columns = [column for column in ADDRESS_TABLE_COLUMNS if column != "site_id"]
    pd.testing.assert_frame_equal(
        ours[columns], published[columns], check_dtype=False, check_exact=False, rtol=0, atol=1e-9
    )
