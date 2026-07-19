"""Convert a real decklist (list of {name,set,number,count} entries) into a
legal deck/pool_<id>_<slug>.csv, resolving names to engine card_ids with the
same machinery pkm/archetype/archetypes.py uses for staples.json. Part of
Part 3a's per-archetype real-decklist sourcing (see
docs/opponent-archetype-classifier-plan.md Part 3a).

Usage:
    python -m pkm.archetype.build_pool_deck <entries.json> <archetype_id> <slug> [out_dir]

entries.json: [{"name": "...", "set": "...", "number": "...", "count": N}, ...]
"""

import json
import sys
from pathlib import Path

from pkm.archetype.aliases import ALIASES
from pkm.archetype.archetypes import _normalize_name
from pkm.data.card_data import get_card_data, get_pokemon_cards


def _build_name_index(cards: dict) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for card_id, card in cards.items():
        index.setdefault(_normalize_name(card.name), []).append(card_id)
    return index


def resolve_entry(
    name: str, set_: str, number: str, name_index: dict[str, list[int]]
) -> tuple[int | None, str | None]:
    """Returns (card_id or None, note or None). Unlike
    archetypes.py's _resolve_staple (which returns None on any ambiguity, for
    the classifier's resolution report), this picks a best-effort card_id on
    a multi-match rather than leaving it unresolved -- a real playable deck
    needs *some* card in every slot, and gameplay stats differing slightly
    between reprints of the same name is an acceptable approximation for a
    pool opponent bot."""
    matches = name_index.get(_normalize_name(name), [])
    if len(matches) == 1:
        return matches[0], None
    alias = ALIASES.get((name, set_, number))
    if alias is not None:
        return alias, None
    if len(matches) > 1:
        return (
            matches[0],
            f"multi-match ({len(matches)} candidates) for {name!r} {set_}/{number} "
            f"-- picked card_id {matches[0]} best-effort",
        )
    return None, f"NOT FOUND in engine: {name!r} {set_}/{number}"


def build_deck(entries: list[dict]) -> tuple[list[int], list[str]]:
    """Returns (card_ids of length exactly 60, human-readable notes)."""
    cards = get_card_data()
    name_index = _build_name_index(cards)
    notes: list[str] = []
    card_ids: list[int] = []
    type_votes: dict[int, int] = {}

    for entry in entries:
        card_id, note = resolve_entry(
            entry["name"], entry.get("set", ""), entry.get("number", ""), name_index
        )
        if note:
            notes.append(note)
        if card_id is None:
            continue  # unresolved -- padded below, not guessed at here
        card = cards.get(card_id)
        if card and card.card_type == 0 and 1 <= card.energy_type <= 8:
            type_votes[card.energy_type] = type_votes.get(card.energy_type, 0) + entry["count"]
        card_ids.extend([card_id] * entry["count"])

    shortfall = 60 - len(card_ids)
    if shortfall > 0:
        filler = max(type_votes, key=type_votes.get) if type_votes else 1
        card_ids.extend([filler] * shortfall)
        notes.append(f"padded {shortfall} slot(s) with basic energy card_id {filler} to reach 60")
    elif shortfall < 0:
        notes.append(f"trimmed {-shortfall} excess card(s) to reach 60")
        card_ids = card_ids[:60]

    if not any(cards[c].basic for c in card_ids if c in cards and cards[c].card_type == 0):
        basics = [c.card_id for c in get_pokemon_cards() if c.basic]
        if basics:
            card_ids[0] = basics[0]
            notes.append(f"forced a Basic Pokemon (card_id {basics[0]}) into slot 0 for legality")

    return card_ids, notes


def write_pool_deck(
    card_ids: list[int], notes: list[str], archetype_id: str, slug: str, out_dir: Path
) -> tuple[Path, list[str]]:
    """Appends >4-copy warnings and writes deck/pool_<id>_<slug>.csv. Returns
    (out_path, notes) -- shared by build_pool_deck's CLI and
    scrape_decklist's CLI so the >4-copy check and file-naming convention
    live in exactly one place."""
    assert len(card_ids) == 60, f"expected 60 cards, got {len(card_ids)}"
    cards = get_card_data()
    from collections import Counter

    for card_id, count in Counter(card_ids).items():
        card = cards.get(card_id)
        is_basic_energy = card and card.card_type in (5, 6) and card.name.startswith("Basic ")
        if not is_basic_energy and count > 4:
            notes.append(f"WARNING: card_id {card_id} has {count} copies (>4 limit)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pool_{archetype_id}_{slug}.csv"
    out_path.write_text("\n".join(str(c) for c in card_ids) + "\n", encoding="utf-8")
    return out_path, notes


def main() -> None:
    entries_path, archetype_id, slug = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("deck")
    entries = json.loads(Path(entries_path).read_text(encoding="utf-8"))

    card_ids, notes = build_deck(entries)
    out_path, notes = write_pool_deck(card_ids, notes, archetype_id, slug, out_dir)

    print(f"wrote {out_path} ({len(card_ids)} cards)")
    for note in notes:
        print(f"  NOTE: {note}")


if __name__ == "__main__":
    main()
