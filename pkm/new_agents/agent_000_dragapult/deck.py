"""Deck registry + closed card vocabulary for agent_000.

The agent supports **multiple fixed decks** (see :data:`DECKS`). Which 60-card
list you *play* is a runtime choice (the ``--deck`` flag threads through
train/eval/agent); which cards the network has *learned embedding rows* for is a
static, build-time property: the **superset vocabulary** over every registered
deck.

Two distinct concepts, kept separate on purpose:

* **A deck** — a specific 60-card list, ``(card_id, name, count)`` entries. This
  is what a seat plays. Changing/adding a deck does not change the network shape
  unless it introduces card IDs not already in the vocabulary.
* **The vocabulary** — the union of distinct card IDs across *all* registered
  decks (+ one ``UNK`` row for any card we do not own: opponent / unseen cards,
  which the model handles via attribute-based encoding instead of a learned
  per-ID row). The own-card embedding table (``encoder.CardEncoder.own_emb``) and
  the hand-histogram feature width are both sized by :data:`VOCAB_SIZE`, so the
  vocabulary is baked into every checkpoint's tensor shapes.

Because the vocabulary spans *all* decks, one trained network has real learned
rows for every registered deck's cards simultaneously. Adding a **new** deck
whose cards are not already covered grows :data:`VOCAB_SIZE` and therefore
requires re-fitting the embedding table (a retrain). Adding a deck that only
reuses already-known cards is free.

``scripts/gen_vocab.py`` writes an inspectable ``vocab.json`` snapshot of the
derived vocabulary; ``test_deck_vocab.py`` asserts the snapshot never drifts from
what this module computes.
"""

from __future__ import annotations

# --- Deck definitions ---------------------------------------------------------
# Each deck is a list of (card_id, name, count) — 60 cards total, max 4 of any
# non-basic-energy card. ``name`` is documentation only; the engine keys on id.

# The original Dragapult ex / Dusknoir control deck (agent_000's v1 deck).
DRAGAPULT: list[tuple[int, str, int]] = [
    (2, "Basic {R} Energy", 2),
    (5, "Basic {P} Energy", 2),
    (7, "Basic {D} Energy", 2),
    (10, "Neo Upper Energy", 1),
    (112, "Munkidori", 2),
    (119, "Dreepy", 4),
    (120, "Drakloak", 4),
    (121, "Dragapult ex", 3),
    (131, "Duskull", 2),
    (132, "Dusclops", 1),
    (133, "Dusknoir", 1),
    (140, "Fezandipiti ex", 1),
    (235, "Budew", 1),
    (1071, "Meowth ex", 2),
    (1079, "Rare Candy", 3),
    (1086, "Buddy-Buddy Poffin", 4),
    (1097, "Night Stretcher", 3),
    (1121, "Ultra Ball", 4),
    (1152, "Poké Pad", 4),
    (1182, "Boss’s Orders", 3),
    (1198, "Crispin", 1),
    (1213, "Judge", 2),
    (1225, "Hilda", 2),
    (1227, "Lillie's Determination", 4),
    (1246, "Jamming Tower", 1),
    (1260, "Risky Ruins", 1),
]

# Mega Alakazam / Dudunsparce psychic control (item-disruption toolbox).
# Card IDs resolved by name+text against the engine card DB (set/collector
# numbers in the source decklist do NOT map to engine IDs). Notable picks:
#   Alakazam 743  — "Psychic Draw" ability + "Powerful Hand" (hand-size payoff,
#                   pairs with 4x Hilda); the MEG printing.
#   Genesect 142  — "ACE Nullifier" (Tool → opponent can't play ACE SPEC); SFA #40.
#   Dudunsparce 66 — non-ex, "Run Away Draw" ability.
ALAKAZAM: list[tuple[int, str, int]] = [
    (5, "Basic {P} Energy", 1),
    (13, "Enriching Energy", 1),
    (19, "Telepath Psychic Energy", 4),
    (65, "Dunsparce", 1),
    (66, "Dudunsparce", 3),
    (140, "Fezandipiti ex", 1),
    (142, "Genesect", 1),
    (305, "Dunsparce", 2),
    (741, "Abra", 4),
    (742, "Kadabra", 4),
    (743, "Alakazam", 3),
    (858, "Psyduck", 1),
    (1013, "Shaymin", 1),
    (1038, "Dedenne", 1),
    (1079, "Rare Candy", 3),
    (1081, "Enhanced Hammer", 2),
    (1086, "Buddy-Buddy Poffin", 4),
    (1097, "Night Stretcher", 1),
    (1129, "Sacred Ash", 1),
    (1152, "Poké Pad", 4),
    (1161, "Handheld Fan", 3),
    (1182, "Boss’s Orders", 3),
    (1184, "Lana’s Aid", 1),
    (1225, "Hilda", 4),
    (1231, "Dawn", 3),
    (1264, "Battle Cage", 3),
]

# Registry: deck name -> definition. Insertion order is the vocabulary tie-break
# only indirectly (the vocab is sorted); it is otherwise just presentation order.
DECKS: dict[str, list[tuple[int, str, int]]] = {
    "dragapult": DRAGAPULT,
    "alakazam": ALAKAZAM,
}

# Default deck played when none is specified (backward compatible with v1, which
# only ever knew the Dragapult deck).
DEFAULT_DECK: str = "dragapult"


def deck_def(name: str = DEFAULT_DECK) -> list[tuple[int, str, int]]:
    """The ``(id, name, count)`` definition for a registered deck."""
    try:
        return DECKS[name]
    except KeyError:
        raise ValueError(
            f"unknown deck {name!r}; choose from {sorted(DECKS)}"
        ) from None


def deck_60(name: str = DEFAULT_DECK) -> list[int]:
    """A deck's 60-card ID list, in the format the deck-selection action wants."""
    return [cid for cid, _n, count in deck_def(name) for _ in range(count)]


# Backward-compatible aliases for the default deck (v1 call sites import these).
DECK: list[tuple[int, str, int]] = deck_def(DEFAULT_DECK)
DECK_60: list[int] = deck_60(DEFAULT_DECK)

# --- Vocabulary (superset over ALL registered decks) --------------------------
# Distinct card IDs we own across every deck, in a stable (sorted) order.
DISTINCT_IDS: list[int] = sorted(
    {cid for deck in DECKS.values() for cid, _n, _c in deck}
)

# Card ID -> embedding row. One extra row (== len(DISTINCT_IDS)) is reserved for
# UNK: any card we do not own (opponent / unseen).
ID_TO_ROW: dict[int, int] = {cid: i for i, cid in enumerate(DISTINCT_IDS)}
UNK_ROW: int = len(DISTINCT_IDS)
VOCAB_SIZE: int = len(DISTINCT_IDS) + 1  # +1 for UNK

# Name lookup across all decks (later registrations win on id collisions, but
# shared cards use the same name everywhere, so order is immaterial).
NAME_BY_ID: dict[int, str] = {
    cid: name for deck in DECKS.values() for cid, name, _c in deck
}


def row_of(card_id: int) -> int:
    """Embedding row for a card ID (UNK row if the card is in no registered deck)."""
    return ID_TO_ROW.get(card_id, UNK_ROW)


# --- Invariants ---------------------------------------------------------------
for _deck_name, _deck_def in DECKS.items():
    _count = sum(c for _, _, c in _deck_def)
    assert _count == 60, f"deck {_deck_name!r} must be 60 cards, got {_count}"
assert len(DECK_60) == 60, f"default deck must be 60 cards, got {len(DECK_60)}"
