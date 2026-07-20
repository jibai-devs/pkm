"""Deck-registry + vocabulary invariants and drift guard.

These lock the properties the network shape depends on: every registered deck is
legal (60 cards, ≤4 of any non-basic-energy card), the vocabulary is a correct
superset over all decks, and the committed ``vocab.json`` snapshot never drifts
from what ``deck.py`` computes.
"""

from __future__ import annotations

import json

from pkm.new_agents.agent_000_dragapult import deck
from pkm.new_agents.agent_000_dragapult.scripts.gen_vocab import (
    VOCAB_PATH,
    _serialise,
    build_vocab,
)

# Basic energies are exempt from the 4-copy limit. Engine IDs 1..12 are the
# twelve Basic {X} Energy cards.
_BASIC_ENERGY_IDS = set(range(1, 13))


def test_every_deck_is_a_legal_60() -> None:
    for name, definition in deck.DECKS.items():
        total = sum(count for _id, _n, count in definition)
        assert total == 60, f"{name}: expected 60 cards, got {total}"
        ids = [cid for cid, _n, _c in definition]
        assert len(ids) == len(set(ids)), f"{name}: duplicate id rows in definition"
        for cid, _n, count in definition:
            if cid not in _BASIC_ENERGY_IDS:
                assert count <= 4, f"{name}: {count}x of non-basic id {cid} (>4)"


def test_deck_60_expands_to_60_ids() -> None:
    for name in deck.DECKS:
        assert len(deck.deck_60(name)) == 60


def test_unknown_deck_raises() -> None:
    try:
        deck.deck_60("no-such-deck")
    except ValueError as e:
        assert "unknown deck" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown deck")


def test_vocab_is_superset_over_all_decks() -> None:
    union = {cid for d in deck.DECKS.values() for cid, _n, _c in d}
    assert set(deck.DISTINCT_IDS) == union
    assert deck.DISTINCT_IDS == sorted(union), "vocab order must be deterministic (sorted)"
    # Every card in every deck has a real (non-UNK) learned row.
    for name, definition in deck.DECKS.items():
        for cid, _n, _c in definition:
            assert deck.row_of(cid) != deck.UNK_ROW, f"{name}: id {cid} fell to UNK"


def test_vocab_size_and_unk_row() -> None:
    assert deck.UNK_ROW == len(deck.DISTINCT_IDS)
    assert deck.VOCAB_SIZE == len(deck.DISTINCT_IDS) + 1
    # UNK row is out of the learned-id range; an unowned id maps there.
    assert deck.row_of(999_999) == deck.UNK_ROW
    assert deck.ID_TO_ROW[deck.DISTINCT_IDS[0]] == 0


def test_committed_vocab_snapshot_matches_deckpy() -> None:
    """vocab.json must equal what gen_vocab computes from deck.py right now."""
    assert VOCAB_PATH.exists(), "run scripts/gen_vocab.py to create vocab.json"
    committed = VOCAB_PATH.read_text(encoding="utf-8")
    assert committed == _serialise(build_vocab()), (
        "vocab.json is stale — regenerate with scripts/gen_vocab.py"
    )
    snap = json.loads(committed)
    assert snap["vocab_size"] == deck.VOCAB_SIZE
    assert snap["unk_row"] == deck.UNK_ROW
    assert snap["distinct_ids"] == deck.DISTINCT_IDS
