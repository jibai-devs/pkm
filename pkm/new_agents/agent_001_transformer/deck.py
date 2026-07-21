"""Hard-coded deck registry for agent_001_transformer.

Unlike agent_000, this agent's network shape is **independent of which cards a
deck contains**: the encoder is a bag-of-features over a wide *shared* sparse
index space (``net.encoder_size`` / ``net.card_count``, keyed on raw engine card
IDs), not a learned per-deck vocabulary. So a "deck" here is purely a runtime
choice — a 60-card ID list handed to ``battle_start`` / ``mcts_agent`` — and
**adding a new deck never changes tensor shapes or forces a retrain**. It only
changes which 60 cards a seat plays (and, via self-play, what the net learns to
play *well*).

Each deck is a list of ``(card_id, name, count)`` — 60 cards total, max four of
any non-basic-energy card. ``name`` is documentation only; the engine keys on the
id. Card IDs were resolved against ``replay/cards.json`` (the live engine card
DB); the source decklists live in ``<repo>/deck/*.csv``.

The played deck is baked into every checkpoint (``train`` writes ``deck`` +
``deck_name`` into the blob), so a packed submission is self-contained: inference
submits whatever deck the checkpoint was trained on. ``--deck`` on ``cli.py``
train/eval/pack selects among the decks below.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Deck definitions
# --------------------------------------------------------------------------- #

# The reference notebook's deck (Mega Abomasnow ex / Kyogre). This is the
# transformer's original ``net.sample_deck`` and stays the DEFAULT so existing
# checkpoints (trained on it) keep matching what train/eval/pack use.
SAMPLE: list[tuple[int, str, int]] = [
    (721, "Kyogre", 2),
    (722, "Snover", 4),
    (723, "Mega Abomasnow ex", 4),
    (1092, "Secret Box", 1),
    (1121, "Ultra Ball", 2),
    (1145, "Mega Signal", 2),
    (1163, "Powerglass", 2),
    (1219, "Team Rocket's Petrel", 4),
    (1227, "Lillie's Determination", 4),
    (1262, "Surfing Beach", 2),
    (3, "Basic {W} Energy", 33),
]

# Dragapult ex / Dusknoir control (source: <repo>/deck/02_dragapult.csv).
DRAGAPULT: list[tuple[int, str, int]] = [
    (119, "Dreepy", 4),
    (120, "Drakloak", 4),
    (121, "Dragapult ex", 3),
    (131, "Duskull", 2),
    (132, "Dusclops", 1),
    (133, "Dusknoir", 1),
    (1071, "Meowth ex", 2),
    (112, "Munkidori", 2),
    (235, "Budew", 1),
    (140, "Fezandipiti ex", 1),
    (1227, "Lillie's Determination", 4),
    (1182, "Boss’s Orders", 3),
    (1213, "Judge", 2),
    (1225, "Hilda", 2),
    (1198, "Crispin", 1),
    (1086, "Buddy-Buddy Poffin", 4),
    (1152, "Poké Pad", 4),
    (1121, "Ultra Ball", 4),
    (1079, "Rare Candy", 3),
    (1097, "Night Stretcher", 3),
    (1246, "Jamming Tower", 1),
    (1260, "Risky Ruins", 1),
    (2, "Basic {R} Energy", 2),
    (5, "Basic {P} Energy", 2),
    (7, "Basic {D} Energy", 2),
    (10, "Neo Upper Energy", 1),
]

# Dragapult ex / Munkidori — NO Dusknoir, item-disruption toolbox
# (source: <repo>/deck/03_pult_munki.csv). Crushing Hammer / Xerosic's
# Machinations / Team Rocket's Watchtower are the disruption core.
PULT_MUNKI: list[tuple[int, str, int]] = [
    (119, "Dreepy", 4),
    (120, "Drakloak", 4),
    (121, "Dragapult ex", 3),
    (112, "Munkidori", 2),
    (235, "Budew", 2),
    (791, "Moltres", 1),
    (140, "Fezandipiti ex", 1),
    (1071, "Meowth ex", 1),
    (1227, "Lillie's Determination", 4),
    (1198, "Crispin", 3),
    (1182, "Boss’s Orders", 3),
    (1213, "Judge", 1),
    (1120, "Crushing Hammer", 4),
    (1086, "Buddy-Buddy Poffin", 4),
    (1152, "Poké Pad", 4),
    (1121, "Ultra Ball", 4),
    (1097, "Night Stretcher", 2),
    (1080, "Unfair Stamp", 1),
    (1260, "Risky Ruins", 1),
    (1256, "Team Rocket's Watchtower", 1),
    (1197, "Xerosic’s Machinations", 1),
    (2, "Basic {R} Energy", 4),
    (5, "Basic {P} Energy", 3),
    (7, "Basic {D} Energy", 2),
]

# Registry: deck name -> definition. Order is presentation order only.
DECKS: dict[str, list[tuple[int, str, int]]] = {
    "sample": SAMPLE,
    "dragapult": DRAGAPULT,
    "pult_munki": PULT_MUNKI,
}

# Default deck when none is specified. ``sample`` keeps backward compatibility:
# the transformer's pre-existing checkpoints were trained on this list.
DEFAULT_DECK: str = "sample"


def deck_names() -> list[str]:
    """Registered deck names, in registry order."""
    return list(DECKS)


def deck_def(name: str = DEFAULT_DECK) -> list[tuple[int, str, int]]:
    """The ``(id, name, count)`` definition for a registered deck."""
    try:
        return DECKS[name]
    except KeyError:
        raise ValueError(
            f"unknown deck {name!r}; choose from {deck_names()}"
        ) from None


def deck_60(name: str = DEFAULT_DECK) -> list[int]:
    """A deck's 60-card ID list, in the format the deck-selection action wants."""
    return [cid for cid, _n, count in deck_def(name) for _ in range(count)]


# Name lookup across all decks (shared cards use the same name everywhere).
NAME_BY_ID: dict[int, str] = {
    cid: name for deck in DECKS.values() for cid, name, _c in deck
}


def card_name(card_id: int) -> str:
    """Display name for a card ID, or ``#<id>`` if not in any registered deck."""
    return NAME_BY_ID.get(card_id, f"#{card_id}")


def resolve_deck(name_or_ids: str | list[int]) -> list[int]:
    """A 60-card list from a registry name, or a list passed through verbatim."""
    if isinstance(name_or_ids, str):
        return deck_60(name_or_ids)
    return list(name_or_ids)


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #
for _deck_name, _deck_def in DECKS.items():
    _count = sum(c for _, _, c in _deck_def)
    assert _count == 60, f"deck {_deck_name!r} must be 60 cards, got {_count}"
