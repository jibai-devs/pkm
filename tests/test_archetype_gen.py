"""Tests for pkm.archetype.gen: tooltip parsing and synthetic dataset generation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pkm.archetype.gen import (
    generate_dataset,
    parse_copy_distribution,
    sample_decklist,
    sample_mixed_archetype_decklist,
    sample_unknown_decklist,
)
from pkm.archetype.archetypes import get_archetypes
from pkm.data.card_data import get_card_data


def test_parse_single_line_tooltip():
    # Literal string from staples.json (Fezandipiti ex, Dragapult ex archetype).
    dist = parse_copy_distribution("97.96% of decklist include 1 of this card")
    assert dist[0] == pytest.approx(0.0204)
    assert dist[1] == pytest.approx(0.9796)
    assert dist.sum() == pytest.approx(1.0)


def test_parse_two_line_tooltip():
    # Literal string from staples.json (Dragapult ex staple, Dragapult ex archetype).
    tooltip = "86.50% of decklist include 3 of this card\n100% of decklist include at least 2 of this card"
    dist = parse_copy_distribution(tooltip)
    assert dist[2] == pytest.approx(0.135)
    assert dist[3] == pytest.approx(0.865)
    assert dist[0] == pytest.approx(0.0)
    assert dist.sum() == pytest.approx(1.0)


def test_parse_high_copy_energy_tooltip():
    # Literal string from staples.json (Metagross Metal Maker, Metal Energy) --
    # basic energy is exempt from the 4-copy limit, array must size past it.
    tooltip = (
        "14.29% of decklist include 18 of this card\n"
        "71.43% of decklist include at least 17 of this card\n"
        "85.71% of decklist include at least 16 of this card\n"
        "100% of decklist include at least 15 of this card"
    )
    dist = parse_copy_distribution(tooltip)
    assert len(dist) >= 19
    assert dist.sum() == pytest.approx(1.0)
    assert dist[:15].sum() == pytest.approx(0.0)


def test_parse_empty_tooltip_returns_zeros():
    dist = parse_copy_distribution("no matching text here")
    assert dist.sum() == 0.0


def test_sample_decklist_is_60_cards():
    archetypes = get_archetypes()
    rng = np.random.default_rng(0)
    for archetype in archetypes[:5]:
        deck = sample_decklist(archetype, rng)
        assert len(deck) == 60, archetype.name


def test_sample_decklist_respects_4_copy_limit_for_non_energy():
    cards = get_card_data()
    archetypes = get_archetypes()
    rng = np.random.default_rng(1)
    for archetype in archetypes:
        deck = sample_decklist(archetype, rng)
        from collections import Counter

        counts = Counter(deck)
        for card_id, n in counts.items():
            card = cards.get(card_id)
            if card is None:
                continue
            is_basic_energy = card.card_type in (5, 6) and card.name.startswith("Basic ")
            if not is_basic_energy:
                assert n <= 4, f"{archetype.name}: {card.name} has {n} copies"


def test_sample_unknown_decklist_is_60_cards_with_a_basic():
    cards = get_card_data()
    archetypes = get_archetypes()
    rng = np.random.default_rng(2)
    for _ in range(5):
        deck = sample_unknown_decklist(archetypes, rng)
        assert len(deck) == 60
        assert any(cards[cid].basic for cid in deck if cid in cards)


def test_sample_mixed_archetype_decklist_is_60_cards_with_a_basic():
    cards = get_card_data()
    archetypes = get_archetypes()
    rng = np.random.default_rng(3)
    for _ in range(5):
        deck = sample_mixed_archetype_decklist(archetypes, rng)
        assert len(deck) == 60
        assert any(cards[cid].basic for cid in deck if cid in cards)


def test_generate_dataset_shape_and_class_balance():
    examples = generate_dataset(n_per_class=3, seed=42)
    archetypes = get_archetypes()
    num_archetypes = len(archetypes)
    # n_per_class per archetype + round(n_per_class * unknown_frac=2.0) for "unknown".
    assert len(examples) == 3 * num_archetypes + 6
    labels = [e.label for e in examples]
    for label in range(num_archetypes):
        assert labels.count(label) == 3
    assert labels.count(num_archetypes) == 6
    for e in examples:
        assert 0.0 <= e.reveal_frac <= 0.5
        assert all(v > 0 for v in e.revealed.values())


def test_generate_dataset_is_deterministic_given_seed():
    a = generate_dataset(n_per_class=2, seed=7)
    b = generate_dataset(n_per_class=2, seed=7)
    assert [e.label for e in a] == [e.label for e in b]
    assert [e.revealed for e in a] == [e.revealed for e in b]
