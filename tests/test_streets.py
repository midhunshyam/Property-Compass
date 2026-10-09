"""Street-name normalisation across G-NAF, OpenStreetMap and NSW spellings."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from property_orientation.streets import gnaf_street_key, normalise_street_name, strip_suffix


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Sailors Bay Rd", "SAILORS BAY ROAD"),
        ("PENSHURST ST", "PENSHURST STREET"),
        ("Victoria Ave", "VICTORIA AVENUE"),
        ("Victoria Av", "VICTORIA AVENUE"),
        ("TARAKAN CR", "TARAKAN CRESCENT"),
        ("Tarakan Cres", "TARAKAN CRESCENT"),
        ("Grand Pde", "GRAND PARADE"),
        ("PACIFIC HWY", "PACIFIC HIGHWAY"),
        ("GORE HILL FWY", "GORE HILL FREEWAY"),
        ("ROSANNA CL", "ROSANNA CLOSE"),
        ("HIGHLAND RIDGE RD", "HIGHLAND RIDGE ROAD"),
    ],
)
def test_expands_road_type_abbreviations(name: str, expected: str) -> None:
    assert normalise_street_name(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Eastern Valley Way", "EASTERN VALLEY WAY"),
        ("The Bulwark", "THE BULWARK"),
        ("Broadway", "BROADWAY"),
        ("Avenue Road", "AVENUE ROAD"),
        ("RD", "RD"),  # a lone word is a name, not a road type
        ("The Ave", "THE AVENUE"),
    ],
)
def test_expands_only_a_final_road_type(name: str, expected: str) -> None:
    assert normalise_street_name(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("MOWBRAY RD W", "MOWBRAY ROAD WEST"),
        ("Mowbray Road West", "MOWBRAY ROAD WEST"),
        ("MARKET ST E", "MARKET STREET EAST"),
        ("ANDERSON ST S", "ANDERSON STREET SOUTH"),
        ("Pacific Hwy N", "PACIFIC HIGHWAY NORTH"),
        ("SMITH RD NE", "SMITH ROAD NORTH EAST"),
        ("CLIFF RD UP", "CLIFF ROAD UPPER"),
        ("CLIFF RD LR", "CLIFF ROAD LOWER"),
    ],
)
def test_expands_suffixes(name: str, expected: str) -> None:
    assert normalise_street_name(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("North St", "NORTH STREET"),  # a leading direction is part of the name
        ("East Esplanade", "EAST ESPLANADE"),
        ("High St N", "HIGH STREET NORTH"),
    ],
)
def test_suffix_needs_a_name_and_type_before_it(name: str, expected: str) -> None:
    assert normalise_street_name(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Saint Johns Avenue", "ST JOHNS AVENUE"),
        ("St Johns Ave", "ST JOHNS AVENUE"),
        ("St. John's Ave", "ST JOHNS AVENUE"),
        ("ST LEONARDS RD", "ST LEONARDS ROAD"),
        ("Saint Ives", "ST IVES"),
    ],
)
def test_saint_is_written_st_and_a_leading_st_is_kept(name: str, expected: str) -> None:
    assert normalise_street_name(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("O'Connell Avenue", "OCONNELL AVENUE"),
        ("O’Connell Ave", "OCONNELL AVENUE"),  # typographic apostrophe
        ("O`Connell Ave", "OCONNELL AVENUE"),
        ("Ryrie's Road", "RYRIES ROAD"),
        ("Bay St.", "BAY STREET"),
        ("Smith, Lane", "SMITH LANE"),
        ("Smith-Jones Rd", "SMITH JONES ROAD"),
        ("KU-RING-GAI CHASE RD", "KU RING GAI CHASE ROAD"),
        ("Smith & Jones Lane", "SMITH AND JONES LANE"),
        ("Smith&Jones Lane", "SMITH AND JONES LANE"),
        ("  mowbray   rd  ", "MOWBRAY ROAD"),
    ],
)
def test_punctuation_hyphens_ampersands_and_spacing(name: str, expected: str) -> None:
    assert normalise_street_name(name) == expected


@pytest.mark.parametrize("blank", [None, "", "   ", "...", "'", math.nan, pd.NA, 42])
def test_blank_or_non_text_is_none(blank: object) -> None:
    assert normalise_street_name(blank) is None


@pytest.mark.parametrize(
    ("gnaf", "osm", "nsw"),
    [
        (("MOWBRAY", "ROAD", "WEST"), "Mowbray Road West", "MOWBRAY RD W"),
        (("ST JOHNS", "AVENUE", None), "Saint Johns Avenue", "ST JOHNS AVE"),
        (("TARAKAN", "CRESCENT", None), "Tarakan Crescent", "TARAKAN CR"),
        (("OCONNELL", "AVENUE", None), "O'Connell Avenue", "O'CONNELL AVE"),
        (("THE BULWARK", None, None), "The Bulwark", "THE BULWARK"),
        (("PACIFIC", "HIGHWAY", None), "Pacific Highway", "PACIFIC HWY"),
        (("PACIFIC", "HIGHWAY", "NORTH EAST"), "Pacific Highway North East", "PACIFIC HWY NE"),
        (("ST JOHNS", "AVENUE", "UPPER"), "Saint Johns Avenue Upper", "ST JOHNS AVE UP"),
    ],
)
def test_the_three_sources_agree(gnaf: tuple, osm: str, nsw: str) -> None:
    key = gnaf_street_key(*gnaf)
    assert key is not None
    assert key == normalise_street_name(osm) == normalise_street_name(nsw)


@pytest.mark.parametrize(
    "name",
    [
        "MOWBRAY RD W",
        "St Johns Ave",
        "The Ave",
        "Smith & Jones Lane",
        "O'Connell Avenue",
        "HIGH ST N",
        "SMITH RD NE",
        "Broadway",
    ],
)
def test_normalising_is_idempotent(name: str) -> None:
    once = normalise_street_name(name)
    assert normalise_street_name(once) == once


class TestGnafStreetKey:
    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            (("PACIFIC", "HIGHWAY", None), "PACIFIC HIGHWAY"),
            (("MOWBRAY", "ROAD", "WEST"), "MOWBRAY ROAD WEST"),
            (("MOWBRAY", "ROAD", "W"), "MOWBRAY ROAD WEST"),
            (("HIGHLAND RIDGE", "ROAD", None), "HIGHLAND RIDGE ROAD"),
            (("THE BULWARK", None, None), "THE BULWARK"),
            (("BROADWAY", math.nan, math.nan), "BROADWAY"),
            (("SMITH", "  ", None), "SMITH"),
            (("smith", "st", None), "SMITH STREET"),
        ],
    )
    def test_joins_name_type_and_suffix(self, parts: tuple, expected: str) -> None:
        assert gnaf_street_key(*parts) == expected

    def test_type_and_suffix_are_optional(self) -> None:
        assert gnaf_street_key("PACIFIC") == "PACIFIC"

    @pytest.mark.parametrize("parts", [(None, None, None), ("", " ", None), (math.nan,)])
    def test_no_name_is_none(self, parts: tuple) -> None:
        assert gnaf_street_key(*parts) is None


class TestStripSuffix:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("MARKET STREET EAST", "MARKET STREET"),
            ("MOWBRAY ROAD WEST", "MOWBRAY ROAD"),
            ("PACIFIC HIGHWAY UPPER", "PACIFIC HIGHWAY"),
            ("ST JOHNS AVENUE CENTRAL", "ST JOHNS AVENUE"),
            ("MOWBRAY ROAD", "MOWBRAY ROAD"),
            ("EAST ESPLANADE", "EAST ESPLANADE"),
            ("THE BULWARK", "THE BULWARK"),
            ("BROADWAY", "BROADWAY"),
        ],
    )
    def test_drops_a_trailing_suffix(self, key: str, expected: str) -> None:
        assert strip_suffix(key) == expected

    @pytest.mark.parametrize("key", [None, math.nan])
    def test_missing_is_none(self, key: object) -> None:
        assert strip_suffix(key) is None

    def test_matches_a_key_without_its_suffix(self) -> None:
        assert strip_suffix(gnaf_street_key("MARKET", "STREET", "EAST")) == normalise_street_name(
            "Market Street"
        )

    @pytest.mark.parametrize("suffix", ["NORTH EAST", "NORTH WEST", "SOUTH EAST", "SOUTH WEST"])
    def test_drops_a_two_word_suffix(self, suffix: str) -> None:
        key = gnaf_street_key("PACIFIC", "HIGHWAY", suffix)
        assert key == f"PACIFIC HIGHWAY {suffix}"
        assert strip_suffix(key) == "PACIFIC HIGHWAY"


def test_two_word_suffix_spellings_agree() -> None:
    assert normalise_street_name("Pacific Hwy North East") == normalise_street_name(
        "PACIFIC HWY NE"
    )
