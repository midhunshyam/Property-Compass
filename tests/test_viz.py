"""Smoke tests for the figures and the web map, on the real-data fixtures (offline).

The pipeline runs on the 350 m extracts in ``tests/fixtures`` (as in ``test_pipeline.py``),
then the figure and map functions draw from its outputs into a temporary folder. The method
explainer is drawn for one hard-coded Willoughby block, so it is only rendered when the full
Willoughby data is present (``make data run``).
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from property_orientation import pipeline
from property_orientation.config import Paths, StudyArea
from property_orientation.geometry import COMPASS_8, compass8

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from property_orientation.viz import charts, explainer, maps, style, webmap  # noqa: E402
from property_orientation.viz.data import FigureData  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
EXTRACTS = ("castlecrag", "northbridge")
CADASTRE = ("nsw_property", "nsw_road_reserve", "nsw_road_centreline")
OTHER_AREA = "Hunters Hill"
"""Any area but the Willoughby showcase; the extracts are installed under its name."""
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DATA_SCRIPT = re.compile(r'(<script id="map-data" type="application/json">)(.*?)(</script>)', re.S)


def run_pipeline(
    root: Path, extract: str, lga: str, *, state: str = "NSW", cadastre: bool = True
) -> Paths:
    """Install a real-data extract as the downloaded inputs of ``lga`` and run the pipeline."""
    paths = Paths(root=root, area=StudyArea(lga=lga, state=state))
    paths.raw.mkdir(parents=True)
    for source in sorted((FIXTURES / extract).iterdir()):
        if source.is_file() and (cadastre or source.stem not in CADASTRE):
            shutil.copy(source, paths.raw / source.name)
    pipeline.run(paths.area, paths)
    return paths


@pytest.fixture(scope="module", params=EXTRACTS)
def run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Paths:
    """Pipeline outputs for one extract, run as an area other than the showcase."""
    return run_pipeline(tmp_path_factory.mktemp(request.param), request.param, OTHER_AREA)


@pytest.fixture(autouse=True)
def house_style() -> Iterator[None]:
    """The figures' matplotlib settings for one test, then the defaults again."""
    with mpl.rc_context():
        style.apply()
        yield
    plt.close("all")


def assert_png(path: Path, min_width: int = 1000) -> None:
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    with Image.open(path) as image:
        assert image.width >= min_width
        assert image.height >= 500


def split_page(page: str) -> tuple[str, dict]:
    """A map page's text with whitespace collapsed and its data left out, and the data."""
    match = DATA_SCRIPT.search(page)
    assert match is not None
    text = page[: match.start(2)] + page[match.end(2) :]
    return " ".join(text.split()), json.loads(match[2])


# --- The web map ----------------------------------------------------------------------------


def test_webmap_of_another_area_leaves_the_showcase_alone(run: Paths) -> None:
    showcase_map, landing = (
        run.root / "docs" / "map" / "index.html",
        run.root / "docs" / "index.html",
    )
    for page in (showcase_map, landing):
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("the showcase")

    out = webmap.build(run)

    assert out == run.root / "docs" / "map" / "hunters-hill" / "index.html"
    assert showcase_map.read_text() == landing.read_text() == "the showcase"
    assert sorted(p.relative_to(run.root).as_posix() for p in run.root.rglob("*.html")) == [
        "docs/index.html",
        "docs/map/hunters-hill/index.html",
        "docs/map/index.html",
    ]


def test_webmap_page_quotes_the_results(run: Paths) -> None:
    text, data = split_page(webmap.build(run).read_text(encoding="utf-8"))
    summary = json.loads((run.results / "run_summary.json").read_text())
    methods = pd.read_csv(run.validation / "method_comparison.csv").set_index("method")
    ours, v1 = methods.loc["street_matched"], methods.loc["nearest_road"]

    assert f"<title>Which way does every property face? {OTHER_AREA}, NSW</title>" in text
    assert "Willoughby" not in text  # the extracts come from Willoughby; the page does not
    assert f"{summary['addresses']:,} addresses at {summary['sites']:,} sites" in text
    assert (
        f"Same 8-point direction as the NSW cadastre for {ours['same_8_point_direction']:.1%} of "
        f"the {int(ours['sites']):,} sites checked, against {v1['same_8_point_direction']:.1%} "
        "for the nearest-road approach (v1)."
    ) in text
    assert "address point" not in text.lower()
    assert not re.search(r"\d points\b|points checked", text)
    for credit in (*webmap.CREDITS, webmap.CADASTRE_CREDIT):
        assert webmap.credits_html(credit) in text  # licence names link to their texts
    assert 'Boundaries: © Geoscape Australia (<a href="https://creativecommons.org' in text
    assert data["credits"] == webmap.credits_html(
        " · ".join((*webmap.CREDITS, webmap.CADASTRE_CREDIT))
    )  # the same credits ride in the map's attribution control
    assert "map.attributionControl.addAttribution(D.credits)" in text
    js = webmap.LEAFLET_SRI["leaflet.js"]
    assert f'<script src="{webmap.LEAFLET_URL}/leaflet.js" integrity="{js}"' in text
    assert "OpenStreetMap" in data["basemaps"][0]["options"]["attribution"]

    # Every site is on the map, and the legend counts them all.
    sites = data["sites"]
    assert len(sites["lon"]) == len(sites["dir"]) == len(sites["addr"]) == summary["sites"]
    assert sum(sites["n"]) == summary["addresses"]
    legend = re.findall(r'aria-label="[A-Za-z -]+: ([\d,]+) sites', text)  # 8 wedges and "none"
    assert len(legend) == 9
    assert sum(int(n.replace(",", "")) for n in legend) == summary["sites"]
    assert data["boundary"] and all(len(ring) >= 4 for ring in data["boundary"])


def test_webmap_colours_each_site_by_its_published_bearing(run: Paths) -> None:
    _, data = split_page(webmap.build(run).read_text(encoding="utf-8"))
    sites = data["sites"]
    bearing = np.array([np.nan if b is None else b for b in sites["bearing"]])
    direction = np.array(sites["dir"])

    assert ((direction >= 0) == np.isfinite(bearing)).all()
    assert ((bearing[direction >= 0] >= 0) & (bearing[direction >= 0] < 360)).all()
    assert (
        compass8(bearing[direction >= 0]) == np.array(COMPASS_8)[direction[direction >= 0]]
    ).all()

    # The same bearings and labels as the published table, site by site.
    table = pd.read_csv(run.results / "property_orientation.csv").drop_duplicates("site_id")
    published = sorted(
        zip(table["bearing_deg"].fillna(-1), table["orientation"].fillna(""), strict=True)
    )
    shown = sorted(
        (b if d >= 0 else -1, COMPASS_8[d] if d >= 0 else "")
        for b, d in zip(bearing, direction, strict=True)
    )
    assert shown == published


def test_map_direction_is_the_label_of_the_rounded_bearing() -> None:
    """22.46° is published as 22.5°, so the site is NE on the map, as in ``results/``."""
    azimuth = np.array([22.46, 22.44, 359.96, 67.46, np.nan])
    sites = pd.DataFrame(
        {"bearing_deg": azimuth, "orientation": compass8(np.mod(np.round(azimuth, 1), 360.0))}
    )
    bearing, direction = webmap.published_directions(sites)
    assert bearing.tolist()[:4] == [22.5, 22.4, 0.0, 67.5]
    assert direction == [1, 0, 0, 2, -1]  # NE, N, N, E and no direction

    stale = sites.assign(orientation=compass8(azimuth))  # labels of the unrounded bearings
    with pytest.raises(ValueError, match="re-run the pipeline"):
        webmap.published_directions(stale)


def test_webmap_of_the_showcase_writes_the_landing_page(tmp_path: Path) -> None:
    paths = run_pipeline(tmp_path, "castlecrag", "Willoughby")
    out = webmap.build(paths)
    assert out == tmp_path / "docs" / "map" / "index.html"

    text = " ".join((tmp_path / "docs" / "index.html").read_text(encoding="utf-8").split())
    summary = json.loads((paths.results / "run_summary.json").read_text())
    v1 = pd.read_csv(paths.validation / "method_comparison.csv").set_index("method")
    assert '<p class="place">Willoughby, NSW</p>' in text
    assert f"addresses at {summary['sites']:,} sites" in text
    assert (
        "The nearest-road approach (v1), which this replaces, faces the nearest OpenStreetMap way"
        in text
    )
    assert f"{v1.loc['nearest_road', 'same_8_point_direction']:.1%} of the same sites" in text
    assert webmap.credits_html(webmap.CREDITS[1]) in text
    assert 'href="map/"' in text
    assert "address point" not in text


def test_webmap_of_an_area_without_the_cadastre(tmp_path: Path) -> None:
    """Outside NSW there is no validation: the page claims none and credits no cadastre."""
    paths = run_pipeline(tmp_path, "castlecrag", "Ballarat", state="VIC", cadastre=False)
    text, data = split_page(webmap.build(paths).read_text(encoding="utf-8"))
    markup = re.sub(r"<script\b.*?</script>|<style>.*?</style>", "", text)

    assert "Which way does every property face? Ballarat, VIC" in markup
    assert "cadastre" not in markup.lower()
    assert data["agreement"] == {}  # so no popup quotes a validation rate either
    assert not (tmp_path / "docs" / "index.html").exists()


def test_webmap_is_deterministic(run: Paths) -> None:
    first = webmap.build(run).read_bytes()
    assert webmap.build(run).read_bytes() == first


# --- Charts ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "draw", [charts.validation_chart, charts.confusion_chart, charts.rose_small_multiples]
)
def test_charts_draw_from_the_results(run: Paths, draw, tmp_path: Path) -> None:
    out = tmp_path / "chart.png"
    assert draw(FigureData(run), out) == out
    assert_png(out)


def test_roses_need_bearings(run: Paths, tmp_path: Path) -> None:
    data = FigureData(run)
    data.addresses = data.addresses.assign(bearing_deg=np.nan)  # override the cached table
    with pytest.raises(ValueError, match="no standalone address"):
        charts.rose_small_multiples(data, tmp_path / "roses.png")


def test_every_rose_bar_lies_inside_one_compass_sector() -> None:
    """Each published bearing (0.1° steps) falls in a bar drawn in its own sector's colour."""
    bearings = np.arange(3600) / 10

    def bar_label(bin_deg: float) -> np.ndarray:
        bins = [int(np.argmax(charts._bin_shares([b], bin_deg))) for b in bearings]
        return compass8(np.array(bins) * bin_deg)

    assert (bar_label(charts.ROSE_BIN_DEG) == compass8(bearings)).all()
    assert (bar_label(10.0) != compass8(bearings)).any()  # 10° bins straddle sector edges


def test_orientation_order_runs_on_ten_degree_bins() -> None:
    def phi(bearings: np.ndarray) -> float:
        return charts.orientation_order(charts._bin_shares(bearings, charts.ORDER_BIN_DEG))

    assert charts.ORDER_BIN_DEG == 10.0
    assert phi(np.repeat([0.0, 90.0, 180.0, 270.0], 25)) == pytest.approx(1.0)
    assert phi(np.arange(360) + 0.5) == pytest.approx(0.0)


def test_chart_output_is_deterministic(run: Paths, tmp_path: Path) -> None:
    data = FigureData(run)
    first = charts.confusion_chart(data, tmp_path / "a.png").read_bytes()
    assert charts.confusion_chart(data, tmp_path / "b.png").read_bytes() == first


@pytest.mark.parametrize("width_in", [4.0, 7.2, 11.2])
def test_sources_line_stays_on_one_line_inside_the_figure(width_in: float) -> None:
    fig = plt.figure(figsize=(width_in, 2.0))
    text = style.attribution(fig, maps.ATTRIBUTION)
    room = width_in * (1 - 2 * style.ATTRIBUTION_X)
    assert "\n" not in text.get_text()
    assert text.get_window_extent().width / fig.dpi <= room * 1.001
    assert 0 < text.get_fontsize() <= style.ATTRIBUTION_PT
    assert style.attribution(plt.figure(figsize=(11.2, 2.0))).get_fontsize() == (
        style.ATTRIBUTION_PT
    )


# --- Maps -----------------------------------------------------------------------------------


def test_hero_map(run: Paths, tmp_path: Path) -> None:
    data = FigureData(run)
    if not maps._has_hero(data):
        pytest.skip(f"the {run.area.lga} extract has no {maps.HERO_LOCALITY.title()} locality")
    assert_png(maps.hero_map(data, tmp_path / "hero.png"))


def test_overview_map(run: Paths, tmp_path: Path) -> None:
    assert_png(maps.overview_map(FigureData(run), tmp_path / "overview.png"))


def test_northerly_choropleth(run: Paths, tmp_path: Path) -> None:
    data = FigureData(run)
    out = tmp_path / "northerly.png"
    if not (data.suburb_summary["standalone"] >= maps.MIN_STANDALONE).any():
        with pytest.raises(ValueError, match="standalone addresses"):
            maps.northerly_choropleth(data, out)
        pytest.skip(f"no suburb in this extract has {maps.MIN_STANDALONE} standalone addresses")
    assert_png(maps.northerly_choropleth(data, out))


def test_northerly_choropleth_of_small_suburbs(
    run: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Few suburbs give few (or no) class breaks; the map still draws."""
    monkeypatch.setattr(maps, "MIN_STANDALONE", 10)
    assert_png(maps.northerly_choropleth(FigureData(run), tmp_path / "northerly.png"))


@pytest.mark.parametrize(
    ("values", "labels"),
    [
        ([0.36], ["36%"]),
        ([0.31, 0.33], ["31–33%"]),
        (
            [0.40, 0.43, 0.36, 0.25, 0.45, 0.27, 0.37, 0.27, 0.25, 0.33, 0.34, 0.25, 0.16],
            ["under 20%", "20–29%", "30–39%", "40% or more"],
        ),
    ],
)
def test_choropleth_classes(values: list[float], labels: list[str]) -> None:
    shares = np.array(values)
    assert maps._class_labels(maps._class_breaks(shares), shares) == labels


def test_choropleth_classes_of_no_values() -> None:
    assert len(maps._class_breaks(np.array([]))) == 0


# --- The method explainer -------------------------------------------------------------------


def test_explainer_needs_its_willoughby_block(run: Paths, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explainer window"):
        explainer.method_explainer(FigureData(run), tmp_path / "explainer.png")


def test_explainer_draws_its_willoughby_block(tmp_path: Path) -> None:
    paths = Paths(root=REPO, area=StudyArea())
    manifest, summary = paths.raw / "manifest.json", paths.results / "run_summary.json"
    if not (manifest.exists() and (paths.processed / "sites.parquet").exists()):
        pytest.skip("needs data/raw/willoughby and data/processed/willoughby (make data run)")
    if json.loads(manifest.read_text())["sources"] != json.loads(summary.read_text())["sources"]:
        pytest.skip("data/raw/willoughby is from another download than results/willoughby")
    assert_png(explainer.method_explainer(FigureData(paths), tmp_path / "explainer.png"))
