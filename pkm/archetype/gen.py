"""Synthetic decklist + partial-reveal dataset generator for the
opponent-archetype classifier. Pure stdlib/numpy -- no torch dependency here,
so this module can run standalone (e.g. from the CLI) without pulling in a
training framework just to generate data.
"""

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from pkm.archetype.archetypes import Archetype, StapleCard, get_archetypes
from pkm.data.card_data import get_card_data, get_pokemon_cards

MAX_COPIES = 4
MIN_REVEAL_FRAC = 0.0
MAX_REVEAL_FRAC = 0.5

_TOOLTIP_LINE = re.compile(r"([\d.]+)%\s*of decklist include (?:at least )?(\d+) of this card")


def parse_copy_distribution(tooltip: str, min_size: int = MAX_COPIES + 1) -> np.ndarray:
    """Parse a staple's tooltip text into P(count=k) for k=0..max(N seen).

    Tooltip lines are cumulative thresholds sorted high-to-low by copy count,
    e.g. "88.55% of decklist include 4 of this card\\n100% of decklist
    include at least 3 of this card" means P(count>=4)=88.55%,
    P(count>=3)=100%. Every line -- whether or not it literally says "at
    least" -- is treated as a ">=" threshold; the highest N mentioned is
    treated as both floor and ceiling for that bucket, since no higher count
    was observed in the scraped meta data. Any implied "count < lowest
    threshold" probability mass is assigned to P(count=0) (card omitted
    entirely), since staples.json only reports presence thresholds, not an
    explicit zero-copies line.

    The array is sized to `max(highest N seen + 1, min_size)`, not a fixed
    4 -- basic/special energy cards are exempt from the normal 4-copy limit
    and staples.json reports real decks running 10-18+ copies of one.
    """
    thresholds: dict[int, float] = {}
    for line in tooltip.splitlines():
        match = _TOOLTIP_LINE.search(line)
        if not match:
            continue
        pct = float(match.group(1)) / 100.0
        n = int(match.group(2))
        thresholds[n] = pct

    if not thresholds:
        return np.zeros(min_size, dtype=np.float64)  # caller falls back to a point mass on `copies`

    ns = sorted(thresholds)
    size = max(ns[-1] + 1, min_size)
    dist = np.zeros(size, dtype=np.float64)
    for i in range(len(ns) - 1, -1, -1):
        n = ns[i]
        cum = thresholds[n]
        if i == len(ns) - 1:
            dist[n] = cum
        else:
            higher_n = ns[i + 1]
            dist[n] = cum - thresholds[higher_n]
    dist[0] += max(0.0, 1.0 - thresholds[ns[0]])

    total = dist.sum()
    if total > 0:
        dist /= total
    return dist


def _staple_distribution(staple: StapleCard) -> np.ndarray:
    dist = parse_copy_distribution(staple.tooltip, min_size=max(staple.copies + 1, MAX_COPIES + 1))
    if dist.sum() <= 0:
        # No parseable tooltip line: fall back to a point mass on the
        # reported modal `copies` count.
        dist = np.zeros(max(staple.copies + 1, MAX_COPIES + 1), dtype=np.float64)
        dist[staple.copies] = 1.0
    return dist


def _dominant_energy_type(archetype: Archetype, cards: dict) -> int | None:
    """Infer the archetype's primary basic-energy type from its resolved
    Pokemon staples' energy_type field, weighted by presence_pct. Basic
    energy card_ids 1-8 equal their energy_type code directly (verified
    against pkm.data.card_data.get_energy_cards())."""
    weights: dict[int, float] = {}
    for staple in archetype.staples:
        if staple.card_id is None or staple.card_id not in cards:
            continue
        card = cards[staple.card_id]
        if card.card_type != 0:  # Pokemon only
            continue
        if not (1 <= card.energy_type <= 8):
            continue
        weights[card.energy_type] = weights.get(card.energy_type, 0.0) + staple.presence_pct
    if not weights:
        return None
    return max(weights, key=weights.get)


def sample_decklist(archetype: Archetype, rng: np.random.Generator) -> list[int]:
    """Sample a synthetic 60-card decklist (one entry per physical card) for
    this archetype, using its staples' parsed copy-count distributions,
    padded with the archetype's dominant basic energy type to reach 60."""
    cards = get_card_data()
    deck: list[int] = []
    for staple in archetype.staples:
        if staple.card_id is None:
            continue  # unresolved card (see aliases.py TODO) -- skip, don't guess
        dist = _staple_distribution(staple)
        n = int(rng.choice(len(dist), p=dist))
        deck.extend([staple.card_id] * n)

    deck = deck[:60]
    pad_needed = 60 - len(deck)
    if pad_needed > 0:
        energy_type = _dominant_energy_type(archetype, cards) or 1  # default Grass
        deck.extend([energy_type] * pad_needed)
    return deck


def sample_uniform_random_decklist(rng: np.random.Generator) -> list[int]:
    """Sample a uniform-random legal-ish 60-card decklist not tied to any
    tracked archetype. One of two negative-generation strategies for the
    'unknown/off-meta' class -- see sample_unknown_decklist."""
    cards = get_card_data()
    pokemon_basics = [c.card_id for c in get_pokemon_cards() if c.basic]
    all_card_ids = list(cards.keys())

    deck: list[int] = []
    # Guarantee at least one Basic Pokemon, same legality floor as real decks.
    deck.extend([rng.choice(pokemon_basics)] * int(rng.integers(1, MAX_COPIES + 1)))

    while len(deck) < 60:
        card_id = int(rng.choice(all_card_ids))
        card = cards[card_id]
        is_basic_energy = card.card_type in (5, 6) and card.name.startswith("Basic ")
        current = sum(1 for c in deck if c == card_id)
        cap = 60 if is_basic_energy else MAX_COPIES
        if current >= cap:
            continue
        take = min(int(rng.integers(1, MAX_COPIES + 1)), cap - current, 60 - len(deck))
        deck.extend([card_id] * take)

    return deck[:60]


def sample_mixed_archetype_decklist(
    archetypes: list[Archetype], rng: np.random.Generator
) -> list[int]:
    """Sample a 'near-miss' negative: a random subset of staples drawn from
    2-4 different archetypes, mixed together. Unlike
    sample_uniform_random_decklist, this deck contains real per-archetype
    signal cards, just not any single archetype's overall composition --
    a harder negative that should punish the classifier for keying off "any
    single staple present" instead of the full distribution."""
    cards = get_card_data()
    k = min(int(rng.integers(2, 5)), len(archetypes))
    chosen_idx = rng.choice(len(archetypes), size=k, replace=False)

    deck: list[int] = []
    energy_votes: dict[int, int] = {}
    for idx in chosen_idx:
        archetype = archetypes[idx]
        resolved = [s for s in archetype.staples if s.card_id is not None]
        if not resolved:
            continue
        take_n = max(1, int(round(len(resolved) * rng.uniform(0.3, 0.7))))
        picked = rng.choice(len(resolved), size=min(take_n, len(resolved)), replace=False)
        for i in picked:
            staple = resolved[i]
            dist = _staple_distribution(staple)
            n = int(rng.choice(len(dist), p=dist))
            deck.extend([staple.card_id] * n)
        energy_type = _dominant_energy_type(archetype, cards)
        if energy_type is not None:
            energy_votes[energy_type] = energy_votes.get(energy_type, 0) + 1

    deck = deck[:60]
    pad_needed = 60 - len(deck)
    if pad_needed > 0:
        energy_type = max(energy_votes, key=energy_votes.get) if energy_votes else 1
        deck.extend([energy_type] * pad_needed)

    if not any(cards[c].basic for c in deck if c in cards and cards[c].card_type == 0):
        pokemon_basics = [c.card_id for c in get_pokemon_cards() if c.basic]
        if deck:
            deck[0] = int(rng.choice(pokemon_basics))

    return deck


def sample_unknown_decklist(archetypes: list[Archetype], rng: np.random.Generator) -> list[int]:
    """Sample a 60-card decklist for the explicit 'unknown/off-meta'
    negative class -- 50/50 uniform-random vs. mixed-archetype 'near-miss',
    covering both kinds of off-meta deck the classifier might face."""
    if rng.random() < 0.5:
        return sample_uniform_random_decklist(rng)
    return sample_mixed_archetype_decklist(archetypes, rng)


@dataclass
class Example:
    revealed: dict[int, int]
    label: int
    reveal_frac: float


def _reveal_subset(deck: list[int], reveal_frac: float, rng: np.random.Generator) -> dict[int, int]:
    n_reveal = int(round(reveal_frac * len(deck)))
    if n_reveal <= 0:
        return {}
    idx = rng.choice(len(deck), size=n_reveal, replace=False)
    revealed_cards = [deck[i] for i in idx]
    return dict(Counter(revealed_cards))


def generate_dataset(
    n_per_class: int,
    seed: int = 0,
    unknown_frac: float = 2.0,
) -> list[Example]:
    """Generate a synthetic partial-reveal dataset.

    n_per_class examples per tracked archetype, plus
    round(n_per_class * unknown_frac) examples for the "unknown" negative
    class -- weighted higher than 1x by default since "unknown" is a much
    more diverse distribution to characterize than any single archetype's.
    Each example's reveal fraction is sampled uniformly in [0, 0.5] -- the
    "how much of the opponent's deck have I actually seen at this point in
    the game" simplification documented in the plan.
    """
    rng = np.random.default_rng(seed)
    archetypes = get_archetypes()
    num_archetypes = len(archetypes)
    unknown_label = num_archetypes

    examples: list[Example] = []
    for idx, archetype in enumerate(archetypes):
        for _ in range(n_per_class):
            deck = sample_decklist(archetype, rng)
            reveal_frac = float(rng.uniform(MIN_REVEAL_FRAC, MAX_REVEAL_FRAC))
            revealed = _reveal_subset(deck, reveal_frac, rng)
            examples.append(Example(revealed=revealed, label=idx, reveal_frac=reveal_frac))

    for _ in range(round(n_per_class * unknown_frac)):
        deck = sample_unknown_decklist(archetypes, rng)
        reveal_frac = float(rng.uniform(MIN_REVEAL_FRAC, MAX_REVEAL_FRAC))
        revealed = _reveal_subset(deck, reveal_frac, rng)
        examples.append(Example(revealed=revealed, label=unknown_label, reveal_frac=reveal_frac))

    rng.shuffle(examples)  # type: ignore[arg-type]
    return examples
