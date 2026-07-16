"""Declarative feature registry for the encoder's float feature vectors.

Registration order is the single source of truth for both the tensor layout
(within a scope) and the total width (STATE_FEATS/OPT_FEATS below) -- no
more hand-maintained integers. `pkm/rl/encoder.py` assembles the raw card-ID
arrays (board_cards/hand_cards/opt_card/...) itself and calls into this
module only for the float feature slices (state_feats/opt_feats).

`deterministic=True` marks a spec as a pure function of `obs` alone (it
ignores `ctx` and is always correct given the observation) -- true of every
feature registered here today. Specs that read `ctx.tracker` (accumulated
per-game memory, Task 6+) or a learned belief (the archetype head, Task 8)
are `deterministic=False`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from pkm.data import get_attack_data
from pkm.heuristics.context import GameContext
from pkm.types.obs import (
    MAX_BENCH,
    N_POKEMON_SLOTS,
    NUM_SELECT_TYPES,
    Observation,
    PokemonRef,
)

# OptionType values (see docs / official api.py). Also used by encoder.py's
# option-ID resolution, which imports these from here.
OPT_NUMBER = 0
OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_TOOL_CARD = 4
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14
OPT_SKILL = 15
OPT_SPECIAL_CONDITION = 16


class Scope(Enum):
    GLOBAL = auto()
    PER_SLOT = auto()
    PER_OPTION = auto()
    PER_DECK_CARD = auto()


@dataclass
class FeatureSpec:
    name: str
    width: int
    scope: Scope
    fn: Callable[[Observation, GameContext | None], np.ndarray]
    deterministic: bool


@dataclass(frozen=True)
class FeatureConfig:
    """Ablation: names in `disabled` get their output slice zero-masked,
    without changing the total assembled width."""

    disabled: frozenset[str] = frozenset()

    def is_enabled(self, name: str) -> bool:
        return name not in self.disabled


@dataclass(frozen=True)
class Norm:
    """Normalization divisors for feature encoding.

    Each field is the assumed maximum for the corresponding game quantity.
    Change these if the game's bounds evolve (e.g. larger bench, higher HP).
    """

    max_hp: float = 300.0
    max_energies: float = 5.0
    max_hand_count: float = 20.0
    max_deck_count: float = 60.0
    max_prize_count: float = 6.0
    max_discard_count: float = 60.0
    max_bench_count: float = 8.0
    max_turn: float = 30.0
    max_actions_per_turn: float = 20.0
    max_pick_count: float = 5.0
    max_energy_cost: float = 5.0
    max_damage_counters: float = 10.0
    max_damage: float = 300.0
    max_option_number: float = 20.0
    max_option_count: float = 5.0


NORM = Norm()


def board_pokemon(obs: Observation) -> list[PokemonRef | None]:
    """Flat list of the N_POKEMON_SLOTS board slots: me (active + bench),
    then opponent (active + bench), padded with None. Shared by the
    per-slot feature specs below and by encoder.py's board_cards array, so
    the slot ordering lives in exactly one place."""
    state = obs.current
    assert state is not None
    you = state.yourIndex
    me = state.players[you]
    opp = state.players[1 - you]

    pokes: list[PokemonRef | None] = []
    for player in (me, opp):
        p_list: list[PokemonRef | None] = [player.active_pokemon]
        p_list += list(player.bench)[:MAX_BENCH]
        p_list += [None] * (1 + MAX_BENCH - len(p_list))
        pokes.extend(p_list)
    return pokes


def _me_opp(obs: Observation):
    state = obs.current
    assert state is not None
    you = state.yourIndex
    return state.players[you], state.players[1 - you]


# --- GLOBAL features -------------------------------------------------------


def _status_conditions(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    me, opp = _me_opp(obs)
    out: list[float] = []
    for player in (me, opp):
        out.extend(
            [
                1.0 if player.poisoned else 0.0,
                1.0 if player.burned else 0.0,
                1.0 if player.asleep else 0.0,
                1.0 if player.paralyzed else 0.0,
                1.0 if player.confused else 0.0,
            ]
        )
    return np.array(out, dtype=np.float32)


def _zone_counts(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    me, opp = _me_opp(obs)
    out: list[float] = []
    for player in (me, opp):
        out.extend(
            [
                player.handCount / NORM.max_hand_count,
                player.deckCount / NORM.max_deck_count,
                len(player.prize) / NORM.max_prize_count,
                len(player.discard) / NORM.max_discard_count,
            ]
        )
    return np.array(out, dtype=np.float32)


def _bench_counts(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    me, opp = _me_opp(obs)
    out: list[float] = []
    for player in (me, opp):
        out.append(len(player.bench) / NORM.max_bench_count)
        out.append(player.benchMax / NORM.max_bench_count)
    return np.array(out, dtype=np.float32)


def _turn(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    state = obs.current
    assert state is not None
    return np.array([state.turn / NORM.max_turn], dtype=np.float32)


def _turn_action_count(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    state = obs.current
    assert state is not None
    return np.array(
        [state.turnActionCount / NORM.max_actions_per_turn], dtype=np.float32
    )


def _turn_flags(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    state = obs.current
    assert state is not None
    return np.array(
        [
            1.0 if state.energyAttached else 0.0,
            1.0 if state.supporterPlayed else 0.0,
            1.0 if state.stadiumPlayed else 0.0,
            1.0 if state.retreated else 0.0,
        ],
        dtype=np.float32,
    )


def _first_player_flags(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    state = obs.current
    assert state is not None
    you = state.yourIndex
    return np.array(
        [
            1.0 if state.firstPlayer == you else 0.0,
            1.0 if state.firstPlayer >= 0 else 0.0,
        ],
        dtype=np.float32,
    )


def _select_type_onehot(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    sel = obs.select
    assert sel is not None
    onehot = np.zeros(NUM_SELECT_TYPES, dtype=np.float32)
    if 0 <= sel.type < NUM_SELECT_TYPES:
        onehot[sel.type] = 1.0
    return onehot


def _select_counts(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    sel = obs.select
    assert sel is not None
    return np.array(
        [
            sel.minCount / NORM.max_pick_count,
            sel.maxCount / NORM.max_pick_count,
            sel.remainEnergyCost / NORM.max_energy_cost,
            sel.remainDamageCounter / NORM.max_damage_counters,
        ],
        dtype=np.float32,
    )


GLOBAL_FEATURES: list[FeatureSpec] = [
    FeatureSpec("status_conditions", 10, Scope.GLOBAL, _status_conditions, True),
    FeatureSpec("zone_counts", 8, Scope.GLOBAL, _zone_counts, True),
    FeatureSpec("bench_counts", 4, Scope.GLOBAL, _bench_counts, True),
    FeatureSpec("turn", 1, Scope.GLOBAL, _turn, True),
    FeatureSpec("turn_action_count", 1, Scope.GLOBAL, _turn_action_count, True),
    FeatureSpec("turn_flags", 4, Scope.GLOBAL, _turn_flags, True),
    FeatureSpec("first_player_flags", 2, Scope.GLOBAL, _first_player_flags, True),
    FeatureSpec(
        "select_type_onehot", NUM_SELECT_TYPES, Scope.GLOBAL, _select_type_onehot, True
    ),
    FeatureSpec("select_counts", 4, Scope.GLOBAL, _select_counts, True),
]


# --- PER_SLOT features -------------------------------------------------------


def _slot_present(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    return np.array([1.0 if p else 0.0 for p in board_pokemon(obs)], dtype=np.float32)


def _slot_hp(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    return np.array(
        [p.hp / NORM.max_hp if p else 0.0 for p in board_pokemon(obs)], dtype=np.float32
    )


def _slot_max_hp(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    return np.array(
        [p.maxHp / NORM.max_hp if p else 0.0 for p in board_pokemon(obs)],
        dtype=np.float32,
    )


def _slot_energy_count(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    return np.array(
        [len(p.energies) / NORM.max_energies if p else 0.0 for p in board_pokemon(obs)],
        dtype=np.float32,
    )


def _slot_appeared_this_turn(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    return np.array(
        [1.0 if (p and p.appearThisTurn) else 0.0 for p in board_pokemon(obs)],
        dtype=np.float32,
    )


PER_SLOT_FEATURES: list[FeatureSpec] = [
    FeatureSpec("slot_present", 1, Scope.PER_SLOT, _slot_present, True),
    FeatureSpec("slot_hp", 1, Scope.PER_SLOT, _slot_hp, True),
    FeatureSpec("slot_max_hp", 1, Scope.PER_SLOT, _slot_max_hp, True),
    FeatureSpec("slot_energy_count", 1, Scope.PER_SLOT, _slot_energy_count, True),
    FeatureSpec(
        "slot_appeared_this_turn", 1, Scope.PER_SLOT, _slot_appeared_this_turn, True
    ),
]


# --- PER_OPTION features -----------------------------------------------------


def _option_number(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    sel = obs.select
    assert sel is not None
    return np.array(
        [(o.number or 0) / NORM.max_option_number for o in sel.option], dtype=np.float32
    )


def _option_count(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    sel = obs.select
    assert sel is not None
    return np.array(
        [(o.count or 0) / NORM.max_option_count for o in sel.option], dtype=np.float32
    )


def _attack_damage(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    sel = obs.select
    assert sel is not None
    attack_data = get_attack_data()
    out: list[float] = []
    for o in sel.option:
        if o.type == OPT_ATTACK:
            atk = attack_data.get(o.attackId or 0)
            out.append(atk.damage / NORM.max_damage if atk else 0.0)
        else:
            out.append(0.0)
    return np.array(out, dtype=np.float32)


def _attack_cost(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    sel = obs.select
    assert sel is not None
    attack_data = get_attack_data()
    out: list[float] = []
    for o in sel.option:
        if o.type == OPT_ATTACK:
            atk = attack_data.get(o.attackId or 0)
            out.append(len(atk.energies) / NORM.max_energies if atk else 0.0)
        else:
            out.append(0.0)
    return np.array(out, dtype=np.float32)


def _option_is_mine(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    state = obs.current
    sel = obs.select
    assert state is not None and sel is not None
    you = state.yourIndex
    out: list[float] = []
    for o in sel.option:
        pi = you if o.playerIndex is None else o.playerIndex
        out.append(1.0 if pi == you else 0.0)
    return np.array(out, dtype=np.float32)


PER_OPTION_FEATURES: list[FeatureSpec] = [
    FeatureSpec("option_number", 1, Scope.PER_OPTION, _option_number, True),
    FeatureSpec("option_count", 1, Scope.PER_OPTION, _option_count, True),
    FeatureSpec("attack_damage", 1, Scope.PER_OPTION, _attack_damage, True),
    FeatureSpec("attack_cost", 1, Scope.PER_OPTION, _attack_cost, True),
    FeatureSpec("option_is_mine", 1, Scope.PER_OPTION, _option_is_mine, True),
]


STATE_FEATS = N_POKEMON_SLOTS * sum(s.width for s in PER_SLOT_FEATURES) + sum(
    s.width for s in GLOBAL_FEATURES
)
OPT_FEATS = sum(s.width for s in PER_OPTION_FEATURES)


# --- assembly ----------------------------------------------------------------


def assemble_global(
    obs: Observation, ctx: GameContext | None, config: FeatureConfig | None = None
) -> np.ndarray:
    config = config or FeatureConfig()
    parts = []
    for spec in GLOBAL_FEATURES:
        arr = np.asarray(spec.fn(obs, ctx), dtype=np.float32).reshape(spec.width)
        if not config.is_enabled(spec.name):
            arr = np.zeros_like(arr)
        parts.append(arr)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def assemble_per_slot(
    obs: Observation, ctx: GameContext | None, config: FeatureConfig | None = None
) -> np.ndarray:
    config = config or FeatureConfig()
    n = N_POKEMON_SLOTS
    cols = []
    for spec in PER_SLOT_FEATURES:
        arr = np.asarray(spec.fn(obs, ctx), dtype=np.float32).reshape(n, spec.width)
        if not config.is_enabled(spec.name):
            arr = np.zeros_like(arr)
        cols.append(arr)
    if not cols:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(cols, axis=1).reshape(-1)  # slot-major flatten


def assemble_per_option(
    obs: Observation, ctx: GameContext | None, config: FeatureConfig | None = None
) -> np.ndarray:
    config = config or FeatureConfig()
    sel = obs.select
    assert sel is not None
    n = len(sel.option)
    cols = []
    for spec in PER_OPTION_FEATURES:
        arr = np.asarray(spec.fn(obs, ctx), dtype=np.float32).reshape(n, spec.width)
        if not config.is_enabled(spec.name):
            arr = np.zeros_like(arr)
        cols.append(arr)
    if not cols:
        return np.zeros((n, 0), dtype=np.float32)
    return np.concatenate(cols, axis=1)


# --- checkpoint-compatibility stamp (persistence lands in Task 5) -----------


class FeatureStampMismatch(RuntimeError):
    pass


ALL_FEATURES: list[FeatureSpec] = (
    GLOBAL_FEATURES + PER_SLOT_FEATURES + PER_OPTION_FEATURES
)


def feature_stamp() -> tuple[tuple[str, str, int], ...]:
    """An ordered fingerprint of the registered feature list, keyed by
    (scope, name, width). Used to reject loading a checkpoint whose weights
    were trained against a different registry."""
    return tuple((s.scope.name, s.name, s.width) for s in ALL_FEATURES)


def check_stamp(expected: tuple[tuple[str, str, int], ...]) -> None:
    current = feature_stamp()
    if current != expected:
        raise FeatureStampMismatch(
            "feature registry mismatch: this checkpoint was stamped with a "
            f"different registered-feature list.\nexpected: {expected}\n"
            f"current:  {current}"
        )
