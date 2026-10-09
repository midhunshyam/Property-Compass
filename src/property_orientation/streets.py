"""Street-name normalisation so G-NAF, OpenStreetMap and NSW names compare equal.

The three sources spell the same street differently:

=================  ===========================
G-NAF              ``MOWBRAY`` + ``ROAD`` + ``WEST``
OpenStreetMap      ``Mowbray Road West``
NSW centrelines    ``MOWBRAY RD W``
=================  ===========================

:func:`normalise_street_name` maps all of them to ``MOWBRAY ROAD WEST``.
"""

from __future__ import annotations

import re

# Road type abbreviations (AS/NZS 4819 and G-NAF authority codes) -> full G-NAF road type.
ROAD_TYPES: dict[str, str] = {
    "AL": "ALLEY", "APP": "APPROACH", "ARC": "ARCADE", "AV": "AVENUE", "AVE": "AVENUE",
    "BVD": "BOULEVARD", "BLVD": "BOULEVARD", "BR": "BRACE", "BRK": "BREAK", "BWLK": "BOARDWALK",
    "BYPA": "BYPASS", "CCT": "CIRCUIT", "CH": "CHASE", "CIR": "CIRCLE", "CL": "CLOSE",
    "CNR": "CORNER", "CON": "CONCOURSE", "CPS": "COPSE", "CR": "CRESCENT", "CRES": "CRESCENT",
    "CRST": "CREST", "CSAC": "CUL-DE-SAC", "CT": "COURT", "CTR": "CENTRE", "DE": "DEVIATION",
    "DR": "DRIVE", "DSTR": "DISTRIBUTOR", "ENT": "ENTRANCE", "ESP": "ESPLANADE",
    "EXP": "EXPRESSWAY", "EXTN": "EXTENSION", "FAWY": "FAIRWAY", "FWY": "FREEWAY",
    "GDNS": "GARDENS", "GLD": "GLADE", "GR": "GROVE", "GRA": "GRANGE", "GTE": "GATE",
    "HTS": "HEIGHTS", "HWY": "HIGHWAY", "JNC": "JUNCTION", "LA": "LANE", "LKT": "LOOKOUT",
    "LN": "LANE", "MNDR": "MEANDER", "MWY": "MOTORWAY", "OTLK": "OUTLOOK", "PDE": "PARADE",
    "PKWY": "PARKWAY", "PL": "PLACE", "PLZA": "PLAZA", "PNT": "POINT", "PROM": "PROMENADE",
    "PSGE": "PASSAGE", "PT": "POINT", "PWAY": "PATHWAY", "QY": "QUAY", "RCH": "REACH",
    "RD": "ROAD", "RDGE": "RIDGE", "RES": "RESERVE", "RMBL": "RAMBLE", "RSE": "RISE",
    "RTE": "ROUTE", "RTN": "RETURN", "RTT": "RETREAT", "RVR": "RIVER", "SQ": "SQUARE",
    "ST": "STREET", "STRA": "STRAIGHT", "SVWY": "SERVICEWAY", "TCE": "TERRACE", "TOLL": "TOLLWAY",
    "TRK": "TRACK", "TRL": "TRAIL", "TWR": "TOWER", "TWRS": "TOWERS", "VSTA": "VISTA",
    "VW": "VIEW", "WD": "WOOD", "WHF": "WHARF", "WKWY": "WALKWAY", "WY": "WAY",
}  # fmt: skip

# Street suffixes (directional and positional qualifiers that follow the road type), from the
# G-NAF street suffix authority table.
STREET_SUFFIXES: dict[str, str] = {
    "N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST",
    "NE": "NORTH EAST", "NW": "NORTH WEST", "SE": "SOUTH EAST", "SW": "SOUTH WEST",
    "CN": "CENTRAL", "DE": "DEVIATION", "EX": "EXTENSION", "IN": "INNER", "LR": "LOWER",
    "ML": "MALL", "OF": "OFF", "OT": "OUTER", "UP": "UPPER",
}  # fmt: skip
_SUFFIX_WORDS = frozenset(w for w in STREET_SUFFIXES.values() if " " not in w)
_TWO_WORD_SUFFIXES = frozenset(w for w in STREET_SUFFIXES.values() if " " in w)

_DROP = re.compile(r"[’'`.,]")


def normalise_street_name(name: object) -> str | None:
    """Canonical upper-case form of a street name, or ``None`` for blank input.

    The function removes punctuation, writes "Saint" as ``ST``, and expands an
    abbreviated road type or suffix in its final position (``RD W`` -> ``ROAD WEST``).
    A leading ``ST`` is left alone because it means "Saint" (``ST JOHNS AVE``).
    """
    if not isinstance(name, str):
        return None
    text = _DROP.sub("", name.upper()).replace("&", " AND ").replace("-", " ")
    tokens = text.split()
    if not tokens:
        return None
    if tokens[0] == "SAINT":
        tokens[0] = "ST"
    suffix: list[str] = []
    if len(tokens) >= 4 and " ".join(tokens[-2:]) in _TWO_WORD_SUFFIXES:
        suffix, tokens = tokens[-2:], tokens[:-2]
    elif len(tokens) >= 3 and (tokens[-1] in STREET_SUFFIXES or tokens[-1] in _SUFFIX_WORDS):
        suffix, tokens = STREET_SUFFIXES.get(tokens[-1], tokens[-1]).split(), tokens[:-1]
    if len(tokens) >= 2 and tokens[-1] in ROAD_TYPES:
        tokens[-1] = ROAD_TYPES[tokens[-1]]
    return " ".join(tokens + suffix)


def gnaf_street_key(name: object, road_type: object = None, suffix: object = None) -> str | None:
    """Normalised street name from G-NAF's separate name, type and suffix fields."""
    parts = [p for p in (name, road_type, suffix) if isinstance(p, str) and p.strip()]
    return normalise_street_name(" ".join(parts)) if parts else None


def strip_suffix(key: str | None) -> str | None:
    """Drop a trailing suffix: ``MARKET STREET EAST`` -> ``MARKET STREET``.

    Two-word suffixes are dropped whole: ``PACIFIC HIGHWAY NORTH EAST`` -> ``PACIFIC HIGHWAY``.
    """
    if not isinstance(key, str):
        return None
    tokens = key.split()
    if len(tokens) >= 4 and " ".join(tokens[-2:]) in _TWO_WORD_SUFFIXES:
        return " ".join(tokens[:-2])
    if len(tokens) >= 3 and tokens[-1] in _SUFFIX_WORDS:
        return " ".join(tokens[:-1])
    return key
