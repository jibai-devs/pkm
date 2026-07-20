"""Refines the static `Attack.damage` card-data field for attacks whose real
damage is (partly or entirely) computed from card text at runtime rather
than being a flat number -- see
docs/superpowers/plans/2026-07-20-attack-damage-estimator.md for the full
investigation and design (Mega Abomasnow ex's Hammer-lanche is the
motivating example: declared `damage: 0`, real damage 0-600 depending on a
deck-mill effect).

Deliberately depends only on `pkm.types.obs`, `pkm.data.card_data`, and
`pkm.heuristics` (GameContext/DeckTracker) -- not on `pkm.rl.features` or
`pkm.rl.deterministic_features`, the same restriction
`deterministic_features.py` observes, since both of those get imported by
`pkm.rl.features` and importing back from here would cycle.

`estimate_attack_damage` returns `attack.damage + bonus`, where `bonus` is
the sum of every matched text pattern's contribution (0 if none match).
Summing (not replacing) handles two shapes uniformly: attacks whose ENTIRE
damage is text-computed (declared `damage: 0`, e.g. everything in Phase 1's
sizing table) just get `0 + bonus`; attacks with a real base plus a
secondary text effect (e.g. Dragapult ex's Phantom Dive: 200 direct + an
otherwise-unaccounted 60-damage bench spread) get both added together. A
single scalar can't represent a multi-target split precisely, but this
feature is already a coarse threat signal (see lethal_this_turn's role), not
a rules-accurate damage calculator.

`min_guaranteed_damage` is the same computation restricted to patterns that
are *certain* given the current game state -- it excludes the coin-flip
patterns (Phase 1) and the deck-mill pattern (Phase 2), all of which return
an expected value, not a guarantee. Use `estimate_attack_damage` for a
continuous magnitude/threat signal (e.g. the `attack_damage` PER_OPTION
feature); use `min_guaranteed_damage` wherever the caller needs to claim
certainty (e.g. `lethal_this_turn`'s "will this KO" boolean) -- treating an
expected value as guaranteed would be actively misleading, not just
imprecise.

Phase 1: closed-form/exact patterns computable from the observation alone,
no GameContext/DeckTracker involvement -- verified against every attack in
the live card database, not guessed.

Phase 2 (added 2026-07-20): the deck-mill pattern -- "discard the top N
cards of your deck, damage per matching card discarded that way" (Hammer-
lanche's own shape; searched the full card database for this text shape,
Hammer-lanche is currently the only real instance of it, though the pattern
is general). Real damage here depends on the *unrevealed order* of the
attacker's own remaining deck, which is never exactly knowable in advance --
but the attacker's own remaining deck *composition* is knowable via
`ctx.tracker` (GameContext/DeckTracker, already built for prize deduction),
so a hypergeometric expected value is a well-founded estimate. See
`_remaining_deck_energy_fraction`'s docstring for why this estimate stays
unbiased even before the deck/prize split is fully known (i.e. before any
full-deck reveal has happened in the game).
"""

import re
from re import Match

from pkm.data.card_data import Attack, get_card_by_id
from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import CardLocation
from pkm.types.obs import MAX_BENCH, EnergyType, Observation, board_pokemon

MAX_REASONABLE_DAMAGE = 600.0
"""Clamp ceiling for the final estimate -- matches Hammer-lanche's own
observed maximum (100 dmg x 6 possible energy discards) from the
investigation that motivated this module, comfortably above every pattern
below's realistic output."""

_ENERGY_CARD_TYPE = 5
"""CardData.card_type for every Basic {X} Energy card. Verified via
all_cards(): every cardType==5 entry in this card database is named
"Basic {X} Energy" -- no special energies exist in this pool, so no
name-based filtering is needed on top of the type check."""

_SYMBOL_TO_ENERGY_TYPE = {
    "G": EnergyType.GRASS,
    "R": EnergyType.FIRE,
    "W": EnergyType.WATER,
    "L": EnergyType.LIGHTNING,
    "P": EnergyType.PSYCHIC,
    "F": EnergyType.FIGHTING,
    "D": EnergyType.DARKNESS,
    "M": EnergyType.METAL,
    "N": EnergyType.DRAGON,
}


def _attacker(obs: Observation):
    state = obs.current
    assert state is not None
    return state.players[state.yourIndex].active_pokemon


def _energy_count_in_discard(obs: Observation, mine: bool, energy_type) -> int:
    """Exact, not estimated -- the discard pile is a public zone with full
    card identity for both players (obs_data_structure/OBSERVATION_SCHEMA.md),
    so this needs no GameContext/DeckTracker deduction."""
    state = obs.current
    assert state is not None
    you = state.yourIndex
    player = state.players[you] if mine else state.players[1 - you]
    count = 0
    for card in player.discard:
        if card is None:
            continue
        data = get_card_by_id(card.id)
        if data is None or data.card_type != _ENERGY_CARD_TYPE:
            continue
        if energy_type is None or data.energy_type == energy_type:
            count += 1
    return count


def _remaining_deck_energy_fraction(
    ctx: GameContext | None, energy_type
) -> tuple[int, int]:
    """(matching, total) among ctx.tracker's still-DECK-labeled cards.

    Before any full-deck reveal has happened in the game, DeckTracker can't
    yet tell the true remaining deck apart from the 6 unrevealed prize cards
    -- both are lumped under CardLocation.DECK ("still unobserved"). That
    doesn't bias the matching-*fraction*, though: the prizes were dealt as a
    uniformly random 6-card subset of the original 60-card deck, so
    conditional on everything already observed (hand/board/discard), the
    still-unobserved cards (true deck + hidden prizes) are exchangeable --
    the density of matching cards within that combined bucket equals the
    density within the true remaining deck alone, in expectation. So
    `matching / total` computed here is an unbiased estimate of the true
    remaining deck's matching-card density even pre-reveal, and becomes
    exact once prizes are known (`ctx.tracker.prizes_known`), since at that
    point the DECK bucket no longer includes any prize cards at all."""
    if ctx is None:
        return 0, 0
    cards = ctx.tracker.by_location(CardLocation.DECK)
    total = len(cards)
    matching = 0
    for c in cards:
        data = get_card_by_id(c.card_id)
        if data is None or data.card_type != _ENERGY_CARD_TYPE:
            continue
        if energy_type is None or data.energy_type == energy_type:
            matching += 1
    return matching, total


def _coin_n_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    n, dmg = int(m.group(1)), int(m.group(2))
    return n * dmg * 0.5


def _coin_one_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    return int(m.group(1)) * 0.5


def _coin_until_tails_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    # Geometric distribution, P(tails) = 0.5: E[heads before first tails] = 1.
    return int(m.group(1)) * 1.0


def _fixed_counters_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    return int(m.group(1)) * 10.0


def _energy_self_any_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    attacker = _attacker(obs)
    if attacker is None:
        return 0.0
    return int(m.group(1)) * len(attacker.energies)


def _energy_self_typed_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    attacker = _attacker(obs)
    if attacker is None:
        return 0.0
    energy_type = _SYMBOL_TO_ENERGY_TYPE.get(m.group(2))
    if energy_type is None:
        return 0.0
    count = sum(1 for e in attacker.energies if e == energy_type)
    return int(m.group(1)) * count


def _discard_damage_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    per_card = int(m.group(1))
    energy_type = _SYMBOL_TO_ENERGY_TYPE.get(m.group(2)) if m.group(2) else None
    mine = m.group(3) == "your"
    return per_card * _energy_count_in_discard(obs, mine=mine, energy_type=energy_type)


def _discard_counters_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    per_card = int(m.group(1)) * 10.0
    energy_type = _SYMBOL_TO_ENERGY_TYPE.get(m.group(2)) if m.group(2) else None
    mine = m.group(3) == "your"
    return per_card * _energy_count_in_discard(obs, mine=mine, energy_type=energy_type)


def _bench_count_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    state = obs.current
    assert state is not None
    me = state.players[state.yourIndex]
    count = sum(1 for p in me.bench if p is not None)
    return int(m.group(1)) * count


def _prize_count_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    state = obs.current
    assert state is not None
    you = state.yourIndex
    mine = m.group(2) == "you have taken"
    player = state.players[you] if mine else state.players[1 - you]
    prizes_remaining = sum(1 for p in player.prize if p is not None)
    taken = max(0, 6 - prizes_remaining)
    return int(m.group(1)) * taken


def _teammate_attack_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    per_mon = int(m.group(1))
    attack_name = m.group(2).strip()
    pokes = board_pokemon(obs)[: 1 + MAX_BENCH]  # my active + bench only
    count = 0
    for p in pokes:
        if p is None:
            continue
        card = get_card_by_id(p.id)
        if card is None:
            continue
        if any(a.name == attack_name for a in card.attacks):
            count += 1
    return per_mon * count


def _fixed_damage_to_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    """"This attack does N damage to ..." (1 of/2 of/each of your opponent's
    (Benched) Pokemon, the new Active Pokemon after a forced switch, etc.) --
    whatever the target-selection phrasing, N itself is always a fixed
    constant here, not scaled by anything. Distinct from the "for each ..."
    patterns above by construction: every attack in this card database uses
    either "damage to" (fixed target, fixed number) or "damage for each"
    (scaling), never both for the same clause."""
    return float(m.group(1))


def _hand_size_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    """Hand *contents* are hidden, but hand *size* (handCount) is always
    public -- no GameContext/DeckTracker deduction needed to count it."""
    per_card = int(m.group(1))
    opponent = obs.opponent
    return per_card * opponent.handCount


def _tool_count_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    per_tool = int(m.group(1))
    pokes = board_pokemon(obs)[: 1 + MAX_BENCH]  # my active + bench only
    count = sum(len(p.tools) for p in pokes if p is not None)
    return per_tool * count


def _deck_mill_energy_bonus(m: Match, attack: Attack, obs: Observation, ctx) -> float:
    """Hammer-lanche's own shape: "discard the top N cards of your deck,
    D damage for each Basic {X} Energy discarded that way." Expected value,
    not a guarantee -- see _remaining_deck_energy_fraction and the module
    docstring's Phase 2 note."""
    n_mill = int(m.group(1))
    per_card = int(m.group(2))
    energy_type = _SYMBOL_TO_ENERGY_TYPE.get(m.group(3)) if m.group(3) else None
    matching, total = _remaining_deck_energy_fraction(ctx, energy_type)
    if total == 0:
        return 0.0
    state = obs.current
    assert state is not None
    me = state.players[state.yourIndex]
    n = min(n_mill, me.deckCount)  # can't mill more than the real deck has
    expected_matches = n * matching / total
    return expected_matches * per_card


# Order doesn't matter for correctness (every matched pattern's contribution
# is summed, not first-match-wins) but is kept roughly in the order the
# investigation found each mechanism, for readability against the plan doc's
# sizing table.
#
# `deterministic` marks whether the bonus is *guaranteed* given the current
# game state (false for the three coin-flip patterns and the deck-mill
# pattern -- all four return an expected value, not a certainty: a coin-flip
# attack whose expected damage meets the KO threshold still only kills
# roughly half the time, and a deck-mill attack's exact discarded cards
# aren't knowable in advance even from a fully-known remaining-deck
# composition). Only the deterministic subset feeds `min_guaranteed_damage`,
# used by `lethal_this_turn`; `estimate_attack_damage` (a continuous
# magnitude signal, not a certainty claim) uses everything.
_PATTERNS: list[tuple[re.Pattern, object, bool]] = [
    (
        re.compile(r"[Ff]lip (\d+) coins?\. This attack does (\d+) damage for each heads"),
        _coin_n_bonus,
        False,
    ),
    (
        re.compile(r"[Ff]lip a coin\. This attack does (\d+) damage for each heads"),
        _coin_one_bonus,
        False,
    ),
    (
        re.compile(
            r"[Ff]lip a coin until you get tails\. "
            r"This attack does (\d+) damage for each heads"
        ),
        _coin_until_tails_bonus,
        False,
    ),
    (
        re.compile(r"[Pp]ut (\d+) damage counters? on .*? in any way you like"),
        _fixed_counters_bonus,
        True,
    ),
    (
        re.compile(
            r"This attack does (\d+) damage for each \{([A-Z]+)\} "
            r"Energy attached to this Pok.mon"
        ),
        _energy_self_typed_bonus,
        True,
    ),
    (
        re.compile(
            r"This attack does (\d+) damage for each Energy attached to this Pok.mon"
        ),
        _energy_self_any_bonus,
        True,
    ),
    (
        re.compile(
            r"This attack does (\d+) damage for each Basic(?: \{([A-Z]+)\})? "
            r"Energy card in (your opponent's|your) discard pile"
        ),
        _discard_damage_bonus,
        True,
    ),
    (
        re.compile(
            r"[Pp]ut (\d+) damage counters? on .*? for each Basic(?: \{([A-Z]+)\})? "
            r"Energy card in (your opponent's|your) discard pile"
        ),
        _discard_counters_bonus,
        True,
    ),
    (
        re.compile(r"This attack does (\d+) damage for each of your Benched Pok.mon"),
        _bench_count_bonus,
        True,
    ),
    (
        re.compile(
            r"This attack does (\d+) damage for each Prize card "
            r"(you have taken|your opponent has taken)"
        ),
        _prize_count_bonus,
        True,
    ),
    (
        re.compile(
            r"This attack does (\d+) damage for each of your Pok.mon in play "
            r"that has the (\w[\w -]*) attack"
        ),
        _teammate_attack_bonus,
        True,
    ),
    (
        re.compile(r"This attack does (\d+) damage to"),
        _fixed_damage_to_bonus,
        True,
    ),
    (
        re.compile(r"This attack does (\d+) damage for each card in your opponent's hand"),
        _hand_size_bonus,
        True,
    ),
    (
        re.compile(
            r"This attack does (\d+) damage for each Pok.mon Tool attached to "
            r"all of your Pok.mon"
        ),
        _tool_count_bonus,
        True,
    ),
    (
        re.compile(
            r"Discard the top (\d+) cards? of your deck, and this attack does "
            r"(\d+) damage for each Basic(?: \{([A-Z]+)\})? Energy card that "
            r"you discarded in this way"
        ),
        _deck_mill_energy_bonus,
        False,
    ),
]


def _normalize(text: str) -> str:
    # Normalize the curly right-single-quote PTCG text uses for possessives
    # ("opponent's") to a plain apostrophe once, so patterns above don't
    # need to embed ’ everywhere.
    return text.replace("’", "'")


def _sum_bonus(
    attack: Attack, obs: Observation, ctx: GameContext | None, deterministic_only: bool
) -> float:
    if not attack.text:
        return 0.0
    text = _normalize(attack.text)
    bonus = 0.0
    for pattern, handler, deterministic in _PATTERNS:
        if deterministic_only and not deterministic:
            continue
        for m in pattern.finditer(text):
            bonus += handler(m, attack, obs, ctx)
    return bonus


def estimate_attack_damage(
    attack: Attack, obs: Observation, ctx: GameContext | None = None
) -> float:
    """`attack.damage` refined by every text pattern that matches, summed
    (see module docstring for why summing, not replacing). Falls back to
    plain `attack.damage` when nothing matches -- an attack that's already
    correctly represented can't get worse. Includes expected-value patterns
    (coin flips, deck-mill) -- appropriate for a continuous magnitude/threat
    signal, not a certainty claim. `ctx` is optional: without it (or without
    a search-revealed deck), deck-mill patterns fall back to 0, same as
    before Phase 2 -- never worse than the pre-Phase-2 baseline."""
    total = attack.damage + _sum_bonus(attack, obs, ctx, deterministic_only=False)
    return max(0.0, min(total, MAX_REASONABLE_DAMAGE))


def min_guaranteed_damage(
    attack: Attack, obs: Observation, ctx: GameContext | None = None
) -> float:
    """Same as `estimate_attack_damage`, but excludes expected-value-only
    patterns (coin flips, deck-mill) -- use this wherever the caller needs a
    *certainty* claim (e.g. lethal_this_turn's "will this KO" boolean), not
    just a magnitude estimate. Treating an expected value as guaranteed
    would be actively misleading, not just imprecise."""
    total = attack.damage + _sum_bonus(attack, obs, ctx, deterministic_only=True)
    return max(0.0, min(total, MAX_REASONABLE_DAMAGE))
