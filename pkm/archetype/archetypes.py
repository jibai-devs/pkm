"""Loads staples.json, resolves staple card names to internal card_ids.

Mirrors the get_card_data() caching pattern in pkm/data/card_data.py.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pkm.archetype.aliases import ALIASES
from pkm.data.card_data import CardData, get_card_data

STAPLES_JSON_PATH = Path(__file__).resolve().parents[2] / "staples.json"


@dataclass
class StapleCard:
    name: str
    set: str
    number: str
    copies: int
    presence_pct: float
    tooltip: str
    card_id: int | None  # None if unresolved


@dataclass
class Archetype:
    id: str
    name: str
    url: str
    staples: list[StapleCard]


@dataclass
class ResolutionReport:
    total: int
    auto: int
    alias: int
    unresolved: list[tuple[str, str, str, str]]  # (archetype_name, staple_name, set, number)

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved)


def _parse_pct(pct: str) -> float:
    return float(pct.strip().rstrip("%")) / 100.0


_APOSTROPHE_VARIANTS = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})


def _normalize_name(name: str) -> str:
    """Collapse typographic apostrophe variants (engine card names use the
    curly U+2019 form, staples.json uses the straight ASCII form) so exact
    matching isn't defeated by encoding alone."""
    return name.translate(_APOSTROPHE_VARIANTS)


def _build_name_index(cards: dict[int, CardData]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for card_id, card in cards.items():
        index.setdefault(_normalize_name(card.name), []).append(card_id)
    return index


def _resolve_staple(
    name: str, set_: str, number: str, name_index: dict[str, list[int]]
) -> tuple[int | None, str]:
    """Returns (card_id or None, provenance in {"auto", "alias", "unresolved"})."""
    matches = name_index.get(_normalize_name(name), [])
    if len(matches) == 1:
        return matches[0], "auto"
    alias_hit = ALIASES.get((name, set_, number))
    if alias_hit is not None:
        return alias_hit, "alias"
    return None, "unresolved"


def load_archetypes_with_report() -> tuple[list[Archetype], ResolutionReport]:
    cards = get_card_data()
    name_index = _build_name_index(cards)

    raw = json.loads(STAPLES_JSON_PATH.read_text(encoding="utf-8"))

    archetypes: list[Archetype] = []
    total = auto = alias = 0
    unresolved: list[tuple[str, str, str, str]] = []

    for archetype_json in raw["archetypes"]:
        staples: list[StapleCard] = []
        for staple_json in archetype_json["staples"]:
            total += 1
            name = staple_json["name"]
            set_ = staple_json["set"]
            number = staple_json["number"]
            card_id, provenance = _resolve_staple(name, set_, number, name_index)
            if provenance == "auto":
                auto += 1
            elif provenance == "alias":
                alias += 1
            else:
                unresolved.append((archetype_json["name"], name, set_, number))
            staples.append(
                StapleCard(
                    name=name,
                    set=set_,
                    number=number,
                    copies=int(staple_json["copies"]),
                    presence_pct=_parse_pct(staple_json["presence_pct"]),
                    tooltip=staple_json["tooltip"],
                    card_id=card_id,
                )
            )
        archetypes.append(
            Archetype(
                id=archetype_json["id"],
                name=archetype_json["name"],
                url=archetype_json["url"],
                staples=staples,
            )
        )

    report = ResolutionReport(total=total, auto=auto, alias=alias, unresolved=unresolved)
    return archetypes, report


_ARCHETYPES: list[Archetype] | None = None


def get_archetypes() -> list[Archetype]:
    """Get all archetypes with staples resolved to card_ids (cached, loads once)."""
    global _ARCHETYPES
    if _ARCHETYPES is None:
        archetypes, _ = load_archetypes_with_report()
        _ARCHETYPES = archetypes
    return _ARCHETYPES
