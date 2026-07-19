"""Tier-1 deterministic heuristics (plan.md §4): exact facts computable from
the current observation plus static card/attack data. No memory needed --
every function here ignores `ctx` and is a pure function of `obs` alone.

Deliberately depends only on `pkm.types.obs` and `pkm.data.card_data`, not on
`pkm.rl.features` -- these get registered into that module's PER_SLOT/
PER_OPTION lists, so importing back from here would cycle.

One correction to plan.md's literal wording, made here because it reflects
how this engine's data actually works (verified against
tests/fixtures/observations.json): `PokemonRef.hp` is already the *current*
remaining HP (it decreases as damage is taken), not the undamaged max with a
separate damage-counter tally. So `lethal_this_turn` compares `hp - damage`
directly; there is no separate damage-counters term to subtract.
"""

import math

import numpy as np

from pkm.data.card_data import get_attack_data, get_card_by_id
from pkm.heuristics.context import GameContext
from pkm.types.obs import MAX_BENCH, Observation, OptionType, board_pokemon


def lethal_this_turn(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    """1.0 for an attack option that would knock out the opponent's active
    Pokemon this turn, else 0.0. Ignores type effectiveness and any other
    damage-modifying effect -- see type_effectiveness for that signal;
    the network combines the two itself."""
    state = obs.current
    sel = obs.select
    assert state is not None and sel is not None
    opp_active = state.players[1 - state.yourIndex].active_pokemon
    attack_data = get_attack_data()

    out: list[float] = []
    for o in sel.option:
        lethal = False
        if o.type == OptionType.ATTACK and opp_active is not None:
            atk = attack_data.get(o.attackId or 0)
            if atk is not None and opp_active.hp - atk.damage <= 0:
                lethal = True
        out.append(1.0 if lethal else 0.0)
    return np.array(out, dtype=np.float32)


def _card_of(pokemon):
    return get_card_by_id(pokemon.id) if pokemon is not None else None


def type_effectiveness(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    """Weakness/resistance multiplier for an attack option, normalized by
    dividing by 2.0 (the max bucket): weak -> 1.0, neutral -> 0.5,
    resisted -> 0.25. 0.0 for non-attack options (distinguishable from the
    neutral bucket, which is 0.5, so there's no ambiguity)."""
    state = obs.current
    sel = obs.select
    assert state is not None and sel is not None
    you = state.yourIndex
    attacker = _card_of(state.players[you].active_pokemon)
    defender = _card_of(state.players[1 - you].active_pokemon)

    out: list[float] = []
    for o in sel.option:
        if o.type != OptionType.ATTACK or attacker is None or defender is None:
            out.append(0.0)
            continue
        multiplier = 1.0
        if defender.weakness is not None and attacker.energy_type == defender.weakness:
            multiplier = 2.0
        elif (
            defender.resistance is not None
            and attacker.energy_type == defender.resistance
        ):
            multiplier = 0.5
        out.append(multiplier / 2.0)
    return np.array(out, dtype=np.float32)


def enemy_threat(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    """Threat level of each opposing Pokemon, 0.0 everywhere on my own side.

    Threat scales with attached energy (the pace at which a Pokemon nears
    being able to attack) but with diminishing returns -- the first energy is
    the biggest jump in danger, each further one adds less. Concretely
    log(1 + 3n) / log(16): 0 -> 0.0, 1 -> 0.50, 2 -> 0.70, 3 -> 0.83,
    4 -> 0.93, 5 -> 1.0 (capped there). Damage output, prize value, and
    remaining HP are deliberate future refinements.
    """
    state = obs.current
    assert state is not None
    pokes = board_pokemon(obs)  # slots 0..MAX_BENCH mine, the rest opponent's
    out = np.zeros(len(pokes), dtype=np.float32)
    for i in range(1 + MAX_BENCH, len(pokes)):
        p = pokes[i]
        if p is not None:
            out[i] = min(math.log(1 + 3 * len(p.energies)) / math.log(16), 1.0)
    return out


def retreat_viable(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    """1.0 at a bench slot of mine whose card's retreat_cost is covered by
    my active's currently-attached energy count, else 0.0. 0.0 everywhere
    else (my own active slot, and the opponent's whole side -- their
    retreat is never a decision I make)."""
    state = obs.current
    assert state is not None
    you = state.yourIndex
    my_active = state.players[you].active_pokemon
    active_energy_count = len(my_active.energies) if my_active is not None else 0

    pokes = board_pokemon(obs)  # slot 0 = my active, slots 1..MAX_BENCH = my bench
    out = np.zeros(len(pokes), dtype=np.float32)
    for i in range(1, 1 + MAX_BENCH):
        p = pokes[i]
        if p is None:
            continue
        card = get_card_by_id(p.id)
        if card is not None and card.retreat_cost <= active_energy_count:
            out[i] = 1.0
    return out
