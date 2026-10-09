"""Which way does an address face? Look it up in a study area's published results.

    property-orientation lookup 22 The Parapet, Castlecrag

The search is forgiving. Case and punctuation are ignored, and road types match in either
form (``St`` or ``Street``). Unit shorthand works (``3/10 Smith St`` finds
``UNIT 3, 10 SMITH STREET``), and the suburb and postcode are optional. If nothing matches
word for word, the closest addresses are suggested instead.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from property_orientation.geometry import COMPASS_8
from property_orientation.streets import ROAD_TYPES, STREET_SUFFIXES

COMPASS_NAMES = {
    "N": "north", "NE": "north-east", "E": "east", "SE": "south-east",
    "S": "south", "SW": "south-west", "W": "west", "NW": "north-west",
}  # fmt: skip

NO_DIRECTION = {
    "STREET LOCALITY": "G-NAF places this address on its street, not on the property",
    "LOCALITY": "G-NAF places this address at the suburb's centre, not on the property",
}
"""Why an address has no direction, by geocode type. Otherwise its point is on the street."""

# Every spelling of a road type, suffix or "Saint" maps to one canonical token, so that the
# query and the published addresses compare equal however each is written.
_CANONICAL: dict[str, str] = {}
for _abbr, _full in ROAD_TYPES.items():
    _CANONICAL.setdefault(_full, _abbr)
    _CANONICAL[_abbr] = _CANONICAL[_full]
_CANONICAL["SAINT"] = _CANONICAL["STREET"]  # "ST" means both; both sides map the same way
for _abbr, _full in STREET_SUFFIXES.items():
    if " " not in _full:
        _CANONICAL[_full] = _abbr
_SPLIT = {abbr: full.split() for abbr, full in STREET_SUFFIXES.items() if " " in full}
_SPLIT = {abbr: [_CANONICAL[w] for w in words] for abbr, words in _SPLIT.items()}
_UNIT_WORDS = frozenset({"UNIT", "U", "APARTMENT", "APT", "FLAT", "SUITE"})
_WORD = re.compile(r"[A-Z0-9]+")


def tokens(text: str) -> list[str]:
    """Canonical search tokens: ``"3/10 Smith St"`` -> ``['3', '10', 'SMITH', 'ST']``."""
    out: list[str] = []
    for word in _WORD.findall(text.upper()):
        if word in _UNIT_WORDS:
            continue  # "Unit 3", "Apt 3" and "3/" all mean the same flat
        out.extend(_SPLIT.get(word, [_CANONICAL.get(word, word)]))
    return out


@dataclass(frozen=True)
class Match:
    """One published address and its orientation."""

    row: dict

    @property
    def address(self) -> str:
        return self.row["address"]

    def as_dict(self) -> dict:
        """JSON-friendly record (missing values become ``None``)."""
        return {k: (None if pd.isna(v) else v) for k, v in self.row.items()}

    def describe(self) -> str:
        """A few readable lines about which way the address faces."""
        r = self.row
        lines = [r["address"]]
        if pd.isna(r["orientation"]):
            reason = NO_DIRECTION.get(r["geocode_type"], "its point lies on the street itself")
            return "\n".join([*lines, f"  No direction: {reason}."])
        label = r["orientation"]
        rear = COMPASS_8[(COMPASS_8.index(label) + 4) % 8]
        lines += [
            f"  Faces       {label} ({COMPASS_NAMES[label]}), "
            f"bearing {r['bearing_deg']:.1f}° from true north",
            f"  Street      {r['street']}, "
            f"{r['distance_to_street_m']:.1f} m from the address point",
            f"  Rear faces  {rear} ({COMPASS_NAMES[rear]})",
            f"  Confidence  {r['confidence']}",
        ]
        return "\n".join(lines)


class AddressBook:
    """The published one-row-per-address table of a study area, ready to search."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table.reset_index(drop=True)

    @classmethod
    def from_csv(cls, path: Path) -> AddressBook:
        return cls(pd.read_csv(path, dtype={"postcode": str, "gnaf_pid": str}))

    @cached_property
    def _tokens(self) -> list[list[str]]:
        return [tokens(a) for a in self.table["address"]]

    def search(self, query: str, limit: int = 5) -> tuple[list[Match], int]:
        """Best matches for ``query`` and the total number of word-for-word matches.

        An address matches when it contains every word of the query. Matches whose numbers
        appear in the same order as in the query come first, then those with the fewest
        extra words: ``22 The Parapet`` ranks the house before ``UNIT 1, 22 THE PARAPET``.
        When nothing matches word for word, the most similar addresses are returned and the
        total is 0.
        """
        wanted = tokens(query)
        if not wanted:
            return [], 0
        need = set(wanted)
        numbers = [w for w in wanted if w.isdigit()]
        addresses = self.table["address"].to_numpy()
        ranked = []
        for i, have in enumerate(self._tokens):
            if need <= set(have):
                in_order = _is_subsequence(numbers, [w for w in have if w.isdigit()])
                ranked.append((not in_order, len(have) - len(need), addresses[i], i))
        if ranked:
            ranked.sort()
            return [self._match(i) for *_, i in ranked[:limit]], len(ranked)
        return self._suggest(" ".join(wanted), limit), 0

    def _suggest(self, query: str, limit: int) -> list[Match]:
        keys = [" ".join(t) for t in self._tokens]
        close = difflib.get_close_matches(query, keys, n=limit, cutoff=0.6)
        return [self._match(keys.index(k)) for k in close]

    def _match(self, i: int) -> Match:
        return Match(self.table.iloc[i].to_dict())


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(h == n for h in it) for n in needle)


def report(book: AddressBook, query: str, limit: int = 5, as_json: bool = False) -> tuple[str, int]:
    """The text (or JSON) answer for ``query`` and a process exit code (0 found, 1 not)."""
    matches, total = book.search(query, limit)
    if as_json:
        records = [m.as_dict() for m in matches]
        payload = {"query": query, "matches": total, "results": records, "suggestions": not total}
        return json.dumps(payload, indent=2, ensure_ascii=False), 0 if total else 1
    if total:
        body = "\n\n".join(m.describe() for m in matches)
        more = total - len(matches)
        if more > 0:
            body += f"\n\n...and {more:,} more. Add the suburb or unit number to narrow it down."
        return body, 0
    if matches:
        names = "\n".join(f"  {m.address}" for m in matches)
        return f"No address matches {query!r}. Did you mean one of these?\n{names}", 1
    return f"No address matches {query!r}.", 1
