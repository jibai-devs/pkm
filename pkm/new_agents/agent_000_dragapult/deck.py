"""The fixed decklist for agent_000_dragapult (hard-coded).

This deck defines the agent's *closed card vocabulary*. Our own card ID
embeddings are indexed over ``DISTINCT_IDS`` (+ one ``UNK`` row for every card
we do not own — i.e. opponent / unseen cards, which the model handles via
attribute-based encoding instead of a learned per-ID row).

Changing this decklist changes the vocabulary and therefore requires re-fitting
the own-card embedding table (a retrain). Everything downstream reads the deck
from here so there is a single source of truth.
"""

from __future__ import annotations

# (card_id, name, count) — 60 cards total, max 4 of any non-basic-energy card.
DECK: list[tuple[int, str, int]] = [
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

# 60-card ID list, in the format the engine expects for the deck-selection action.
DECK_60: list[int] = [cid for cid, _name, count in DECK for _ in range(count)]

# Closed vocabulary: distinct card IDs we own, in a stable (sorted) order.
DISTINCT_IDS: list[int] = sorted({cid for cid, _n, _c in DECK})

# Card ID -> embedding row. One extra row (== len(DISTINCT_IDS)) is reserved for
# UNK: any card we do not own (opponent / unseen).
ID_TO_ROW: dict[int, int] = {cid: i for i, cid in enumerate(DISTINCT_IDS)}
UNK_ROW: int = len(DISTINCT_IDS)
VOCAB_SIZE: int = len(DISTINCT_IDS) + 1  # +1 for UNK

NAME_BY_ID: dict[int, str] = {cid: name for cid, name, _c in DECK}


def row_of(card_id: int) -> int:
    """Embedding row for a card ID (UNK row if the card is not in our deck)."""
    return ID_TO_ROW.get(card_id, UNK_ROW)


assert len(DECK_60) == 60, f"deck must be 60 cards, got {len(DECK_60)}"
