"""Refines the static `Attack.damage` card-data field for attacks whose real
damage is (partly or entirely) computed from card text at runtime rather
than being a flat number -- see
docs/superpowers/plans/2026-07-20-attack-damage-estimator.md for the full
investigation and design (Mega Abomasnow ex's Hammer-lanche is the
motivating example: declared `damage: 0`, real damage 0-600 depending on a
deck-mill effect).

Deliberately depends only on `pkm.types.obs` and `pkm.data.card_data`, the
same restriction `deterministic_features.py` observes -- both get imported
by `pkm.rl.features`, so neither can import back from it without a cycle.

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
patterns, which return an expected value, not a guarantee. Use
`estimate_attack_damage` for a continuous magnitude/threat signal (e.g. the
`attack_damage` PER_OPTION feature); use `min_guaranteed_damage` wherever the
caller needs to claim certainty (e.g. `lethal_this_turn`'s "will this KO"
boolean) -- treating a coin flip's expected damage as guaranteed would be
actively misleading, not just imprecise.

Phase 1 only: closed-form/exact patterns computable from the observation
alone, no GameContext/DeckTracker involvement -- verified against every
attack in the live card database, not guessed. Deck-mill effects
(Hammer-lanche's own family -- damage depends on the attacker's unrevealed
remaining deck order) are Phase 2, not implemented here.
"""

import re
from re import Match

from pkm.data.card_data import Attack, get_card_by_id
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


def _coin_n_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    n, dmg = int(m.group(1)), int(m.group(2))
    return n * dmg * 0.5


def _coin_one_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    return int(m.group(1)) * 0.5


def _coin_until_tails_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    # Geometric distribution, P(tails) = 0.5: E[heads before first tails] = 1.
    return int(m.group(1)) * 1.0


def _fixed_counters_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    return int(m.group(1)) * 10.0


def _energy_self_any_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    attacker = _attacker(obs)
    if attacker is None:
        return 0.0
    return int(m.group(1)) * len(attacker.energies)


def _energy_self_typed_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    attacker = _attacker(obs)
    if attacker is None:
        return 0.0
    energy_type = _SYMBOL_TO_ENERGY_TYPE.get(m.group(2))
    if energy_type is None:
        return 0.0
    count = sum(1 for e in attacker.energies if e == energy_type)
    return int(m.group(1)) * count


def _discard_damage_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    per_card = int(m.group(1))
    energy_type = _SYMBOL_TO_ENERGY_TYPE.get(m.group(2)) if m.group(2) else None
    mine = m.group(3) == "your"
    return per_card * _energy_count_in_discard(obs, mine=mine, energy_type=energy_type)


def _discard_counters_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    per_card = int(m.group(1)) * 10.0
    energy_type = _SYMBOL_TO_ENERGY_TYPE.get(m.group(2)) if m.group(2) else None
    mine = m.group(3) == "your"
    return per_card * _energy_count_in_discard(obs, mine=mine, energy_type=energy_type)


def _bench_count_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    state = obs.current
    assert state is not None
    me = state.players[state.yourIndex]
    count = sum(1 for p in me.bench if p is not None)
    return int(m.group(1)) * count


def _prize_count_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    state = obs.current
    assert state is not None
    you = state.yourIndex
    mine = m.group(2) == "you have taken"
    player = state.players[you] if mine else state.players[1 - you]
    prizes_remaining = sum(1 for p in player.prize if p is not None)
    taken = max(0, 6 - prizes_remaining)
    return int(m.group(1)) * taken


def _teammate_attack_bonus(m: Match, attack: Attack, obs: Observation) -> float:
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


def _fixed_damage_to_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    """"This attack does N damage to ..." (1 of/2 of/each of your opponent's
    (Benched) Pokemon, the new Active Pokemon after a forced switch, etc.) --
    whatever the target-selection phrasing, N itself is always a fixed
    constant here, not scaled by anything. Distinct from the "for each ..."
    patterns above by construction: every attack in this card database uses
    either "damage to" (fixed target, fixed number) or "damage for each"
    (scaling), never both for the same clause."""
    return float(m.group(1))


def _hand_size_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    """Hand *contents* are hidden, but hand *size* (handCount) is always
    public -- no GameContext/DeckTracker deduction needed to count it."""
    per_card = int(m.group(1))
    opponent = obs.opponent
    return per_card * opponent.handCount


def _tool_count_bonus(m: Match, attack: Attack, obs: Observation) -> float:
    per_tool = int(m.group(1))
    pokes = board_pokemon(obs)[: 1 + MAX_BENCH]  # my active + bench only
    count = sum(len(p.tools) for p in pokes if p is not None)
    return per_tool * count


# Order doesn't matter for correctness (every matched pattern's contribution
# is summed, not first-match-wins) but is kept roughly in the order the
# investigation found each mechanism, for readability against the plan doc's
# sizing table.
#
# `deterministic` marks whether the bonus is *guaranteed* given the current
# game state (true for everything except the three coin-flip patterns, which
# return an expected value -- a coin-flip attack whose expected damage meets
# the KO threshold still only kills roughly half the time). Only the
# deterministic subset feeds `min_guaranteed_damage`, used by
# `lethal_this_turn`; `estimate_attack_damage` (a continuous magnitude
# signal, not a certainty claim) uses everything.
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
]


def _normalize(text: str) -> str:
    # Normalize the curly right-single-quote PTCG text uses for possessives
    # ("opponent's") to a plain apostrophe once, so patterns above don't
    # need to embed ’ everywhere.
    return text.replace("’", "'")


def _sum_bonus(attack: Attack, obs: Observation, deterministic_only: bool) -> float:
    if not attack.text:
        return 0.0
    text = _normalize(attack.text)
    bonus = 0.0
    for pattern, handler, deterministic in _PATTERNS:
        if deterministic_only and not deterministic:
            continue
        for m in pattern.finditer(text):
            bonus += handler(m, attack, obs)
    return bonus


def estimate_attack_damage(attack: Attack, obs: Observation) -> float:
    """`attack.damage` refined by every text pattern that matches, summed
    (see module docstring for why summing, not replacing). Falls back to
    plain `attack.damage` when nothing matches -- an attack that's already
    correctly represented can't get worse. Includes expected-value patterns
    (coin flips) -- appropriate for a continuous magnitude/threat signal,
    not a certainty claim."""
    total = attack.damage + _sum_bonus(attack, obs, deterministic_only=False)
    return max(0.0, min(total, MAX_REASONABLE_DAMAGE))


def min_guaranteed_damage(attack: Attack, obs: Observation) -> float:
    """Same as `estimate_attack_damage`, but excludes expected-value-only
    patterns (coin flips) -- use this wherever the caller needs a *certainty*
    claim (e.g. lethal_this_turn's "will this KO" boolean), not just a
    magnitude estimate. A coin-flip attack whose expected damage clears the
    KO threshold still only kills roughly half the time; claiming otherwise
    would be actively misleading, not just imprecise."""
    total = attack.damage + _sum_bonus(attack, obs, deterministic_only=True)
    return max(0.0, min(total, MAX_REASONABLE_DAMAGE))
