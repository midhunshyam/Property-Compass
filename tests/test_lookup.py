"""Looking up which way an address faces, from the published one-row-per-address table."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import pandas as pd
import pytest

from property_orientation import cli
from property_orientation.lookup import AddressBook, report, tokens
from property_orientation.outputs import ADDRESS_TABLE_COLUMNS

REPO = Path(__file__).resolve().parents[1]


def row(address: str, orientation: str | None = "N", bearing: float | None = 2.0, **fields) -> dict:
    defaults = {
        "gnaf_pid": f"GANSW{zlib.crc32(address.encode()):010d}",
        "address": address,
        "locality": address.rsplit(", ", 1)[-1].rsplit(" NSW", 1)[0],
        "postcode": address[-4:],
        "longitude": 151.2,
        "latitude": -33.8,
        "address_kind": "unit"
        if "," in address.split(" NSW")[0].rsplit(", ", 1)[0]
        else "standalone",
        "orientation": orientation,
        "bearing_deg": bearing,
        "street": "Smith Street",
        "distance_to_street_m": 25.0,
        "match": "street_name",
        "confidence": "high" if orientation else "none",
        "geocode_type": "PROPERTY CENTROID",
        "site_id": 0,
    }
    return defaults | fields


@pytest.fixture
def book() -> AddressBook:
    rows = [
        row("UNIT 1, 10 SMITH STREET, CHATSWOOD NSW 2067"),
        row("10 SMITH STREET, CHATSWOOD NSW 2067", "E", 91.0),
        row("UNIT 3, 10 SMITH STREET, CHATSWOOD NSW 2067"),
        row("UNIT 10, 3 SMITH STREET, CHATSWOOD NSW 2067", "W", 270.0),
        row("10 SMITH STREET, ARTARMON NSW 2064", "S", 180.0),
        row("4 ST JOHNS AVENUE, ROSEVILLE NSW 2069", "NE", 44.0),
        row("2 MOWBRAY ROAD WEST, LANE COVE NORTH NSW 2066", "SW", 225.0),
        row("8 PACIFIC HIGHWAY NORTH EAST, CHATSWOOD NSW 2067", "NW", 315.0),
        row("5 EDGE STREET, CHATSWOOD NSW 2067", None, None, geocode_type="STREET LOCALITY"),
    ]
    return AddressBook(pd.DataFrame(rows, columns=list(ADDRESS_TABLE_COLUMNS)))


class TestTokens:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("10 Smith St", ["10", "SMITH", "ST"]),
            ("10 SMITH STREET", ["10", "SMITH", "ST"]),
            ("3/10 Smith St.", ["3", "10", "SMITH", "ST"]),
            ("Unit 3, 10 Smith Street", ["3", "10", "SMITH", "ST"]),
            ("Apt 3, 10 smith st", ["3", "10", "SMITH", "ST"]),
            ("Saint Johns Ave", ["ST", "JOHNS", "AV"]),
            ("Mowbray Rd W", ["MOWBRAY", "RD", "W"]),
            ("Mowbray Road West", ["MOWBRAY", "RD", "W"]),
            ("Pacific Hwy NE", ["PACIFIC", "HWY", "N", "E"]),
            ("Pacific Highway North East", ["PACIFIC", "HWY", "N", "E"]),
            ("", []),
        ],
    )
    def test_spellings_compare_equal(self, text: str, expected: list[str]) -> None:
        assert tokens(text) == expected


class TestSearch:
    def test_the_house_ranks_before_its_units(self, book: AddressBook) -> None:
        matches, total = book.search("10 Smith St Chatswood")
        assert matches[0].address == "10 SMITH STREET, CHATSWOOD NSW 2067"
        assert total == 4  # the house, two units of it, and UNIT 10, 3 SMITH STREET

    def test_unit_shorthand_keeps_the_number_order(self, book: AddressBook) -> None:
        matches, _ = book.search("3/10 Smith Street")
        assert matches[0].address == "UNIT 3, 10 SMITH STREET, CHATSWOOD NSW 2067"
        matches, _ = book.search("10/3 Smith Street")
        assert matches[0].address == "UNIT 10, 3 SMITH STREET, CHATSWOOD NSW 2067"

    def test_suburb_is_optional_and_narrows(self, book: AddressBook) -> None:
        everywhere, total = book.search("10 Smith Street", limit=10)
        assert total == 5
        assert {m.address for m in everywhere} >= {
            "10 SMITH STREET, CHATSWOOD NSW 2067",
            "10 SMITH STREET, ARTARMON NSW 2064",
        }
        artarmon, total = book.search("10 smith st artarmon")
        assert total == 1
        assert artarmon[0].row["orientation"] == "S"

    @pytest.mark.parametrize(
        ("query", "address"),
        [
            ("4 Saint Johns Ave Roseville", "4 ST JOHNS AVENUE, ROSEVILLE NSW 2069"),
            ("2 Mowbray Rd W", "2 MOWBRAY ROAD WEST, LANE COVE NORTH NSW 2066"),
            ("8 Pacific Hwy NE", "8 PACIFIC HIGHWAY NORTH EAST, CHATSWOOD NSW 2067"),
        ],
    )
    def test_abbreviations(self, book: AddressBook, query: str, address: str) -> None:
        matches, total = book.search(query)
        assert total == 1
        assert matches[0].address == address

    def test_limit(self, book: AddressBook) -> None:
        matches, total = book.search("Smith Street", limit=2)
        assert len(matches) == 2
        assert total == 5

    def test_no_match_suggests_close_addresses(self, book: AddressBook) -> None:
        matches, total = book.search("10 Smyth Street Chatswood")
        assert total == 0
        assert matches[0].address == "10 SMITH STREET, CHATSWOOD NSW 2067"

    def test_nothing_close(self, book: AddressBook) -> None:
        assert book.search("zzzz qqqq") == ([], 0)
        assert book.search("  ,, ") == ([], 0)


class TestReport:
    def test_describes_the_direction_and_rear(self, book: AddressBook) -> None:
        text, code = report(book, "10 smith st artarmon")
        assert code == 0
        assert text.splitlines() == [
            "10 SMITH STREET, ARTARMON NSW 2064",
            "  Faces       S (south), bearing 180.0° from true north",
            "  Street      Smith Street, 25.0 m from the address point",
            "  Rear faces  N (north)",
            "  Confidence  high",
        ]

    def test_explains_a_missing_direction(self, book: AddressBook) -> None:
        text, code = report(book, "5 Edge Street")
        assert code == 0
        assert "No direction: G-NAF places this address on its street" in text

    def test_says_how_many_more_match(self, book: AddressBook) -> None:
        text, _ = report(book, "Smith Street", limit=2)
        assert text.endswith("...and 3 more. Add the suburb or unit number to narrow it down.")

    def test_suggestions_exit_with_1(self, book: AddressBook) -> None:
        text, code = report(book, "10 Smyth Street Chatswood")
        assert code == 1
        assert text.startswith("No address matches '10 Smyth Street Chatswood'. Did you mean")
        assert "  10 SMITH STREET, CHATSWOOD NSW 2067" in text

    def test_json(self, book: AddressBook) -> None:
        text, code = report(book, "5 Edge Street", as_json=True)
        payload = json.loads(text)
        assert code == 0
        assert payload["matches"] == 1
        assert payload["suggestions"] is False
        assert payload["results"][0]["orientation"] is None  # NaN becomes null
        assert payload["results"][0]["geocode_type"] == "STREET LOCALITY"


class TestCommandLine:
    def test_lookup(self, book: AddressBook, tmp_path: Path, capsys) -> None:
        results = tmp_path / "results" / "willoughby"
        results.mkdir(parents=True)
        book.table.to_csv(results / "property_orientation.csv", index=False)

        code = cli.main(["--root", str(tmp_path), "lookup", "10", "Smith", "St,", "Artarmon"])
        assert code == 0
        assert "Faces       S (south)" in capsys.readouterr().out

        code = cli.main(["--root", str(tmp_path), "lookup", "--json", "nowhere at all"])
        assert code == 1
        assert json.loads(capsys.readouterr().out)["matches"] == 0

    def test_needs_published_results(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="No published results for Hunters Hill"):
            cli.main(["--root", str(tmp_path), "--lga", "Hunters Hill", "lookup", "1 Main St"])


def test_committed_willoughby_results() -> None:
    table = REPO / "results" / "willoughby" / "property_orientation.csv"
    if not table.exists():
        pytest.skip("results/willoughby is not present")
    matches, total = AddressBook.from_csv(table).search("22 The Parapet, Castlecrag")
    assert total == 1
    assert matches[0].row["orientation"] == "N"
    assert matches[0].row["street"] == "The Parapet"
