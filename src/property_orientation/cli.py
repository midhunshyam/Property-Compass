"""Command line interface: ``property-orientation {fetch,run,figures,webmap,all}``."""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

from property_orientation.config import Paths, StudyArea


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="property-orientation",
        description="Which way does every property face? Street-frontage orientation for "
        "Australian addresses (G-NAF + OpenStreetMap), validated against the NSW cadastre.",
    )
    parser.add_argument(
        "--lga", default="Willoughby", help="local government area (default: Willoughby)"
    )
    parser.add_argument("--state", default="NSW", help="state or territory (default: NSW)")
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository root (default: .)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download G-NAF, OSM and (NSW only) cadastre data")
    fetch.add_argument("--force", action="store_true", help="re-download cached files")
    sub.add_parser("run", help="compute orientation and validation, and write results/")
    sub.add_parser("figures", help="render the README figures into docs/images/ (Willoughby only)")
    sub.add_parser(
        "webmap", help="build the interactive map in docs/map/ (other areas: a subfolder)"
    )
    lookup = sub.add_parser(
        "lookup", help="which way does an address face? (searches results/<area>/, no download)"
    )
    lookup.add_argument("address", nargs="+", help="for example: 22 The Parapet, Castlecrag")
    lookup.add_argument("--limit", type=int, default=5, help="most matches to show (default: 5)")
    lookup.add_argument("--json", action="store_true", help="print the matches as JSON")
    everything = sub.add_parser("all", help="fetch, run, figures and webmap")
    everything.add_argument("--force", action="store_true", help="re-download cached files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    area = StudyArea(lga=args.lga, state=args.state)
    paths = Paths(root=args.root.resolve(), area=area)

    if args.command == "lookup":
        return _lookup(paths, " ".join(args.address), args.limit, args.json)

    from property_orientation import pipeline

    if args.command in ("fetch", "all"):
        pipeline.fetch(area, paths, force=args.force)
    if args.command in ("run", "all"):
        pipeline.run(area, paths)
    if args.command in ("figures", "all"):
        if area.is_showcase:
            _viz("figures").render_all(paths)
        elif args.command == "figures":
            raise SystemExit(
                "The README figures are drawn for the Willoughby showcase only (they include a "
                "Castlecrag hero map). Use `webmap` for an interactive map of this area."
            )
        else:
            logging.getLogger("property_orientation").info(
                "skipping figures: they are drawn for the Willoughby showcase only"
            )
    if args.command in ("webmap", "all"):
        _viz("webmap").build(paths)
    return 0


def _lookup(paths: Paths, query: str, limit: int, as_json: bool) -> int:
    """Print which way the best-matching published addresses face."""
    from property_orientation.lookup import AddressBook, report

    table = paths.results / "property_orientation.csv"
    if not table.exists():
        raise SystemExit(
            f"No published results for {paths.area.lga} ({table}). Run "
            f'`make data run LGA="{paths.area.lga}" STATE={paths.area.state}` first.'
        )
    text, code = report(AddressBook.from_csv(table), query, limit, as_json)
    print(text)
    return code


def _viz(name: str):
    """Import a viz module, or explain how to install the optional dependency."""
    try:
        return importlib.import_module(f"property_orientation.viz.{name}")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"`{name}` needs the viz extra ({exc.name} is missing): pip install -e '.[viz]'"
        ) from exc
