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

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

from pkm.data import get_attack_data
from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import CardLocation
from pkm.rl.deterministic_features import (
    lethal_this_turn,
    retreat_viable,
    type_effectiveness,
)
from pkm.types.obs import (
    N_POKEMON_SLOTS,
    NUM_SELECT_TYPES,
    Observation,
    board_pokemon,
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

    max_hp: float = 380.0  # max in card DB; engine/src/card/CardImpl.h grep
    max_energies: float = 5.0  # practical max; no C++ cap
    max_hand_count: float = 20.0  # practical max; no C++ cap
    max_deck_count: float = 60.0  # engine/src/core/Core.h:12 DECK_SIZE=60
    max_prize_count: float = 6.0  # engine/src/core/Core.h:15 PRIZE_SIZE=6
    max_discard_count: float = 60.0  # bounded by DECK_SIZE
    max_bench_count: float = 8.0  # engine/src/core/Core.h:14 BENCH_SIZE_MAX=8
    max_turn: float = 30.0  # practical max; engine/src/game/GameProc.h:809 hard cap=10000
    max_actions_per_turn: float = 20.0  # practical max; engine/src/game/GameProc.h:805 hard cap=10000
    max_pick_count: float = 5.0  # practical max; no C++ cap
    max_energy_cost: float = 5.0  # practical max; no C++ cap
    max_damage_counters: float = 10.0  # practical max; bounded by max_hp
    max_damage: float = 350.0  # max in card DB; engine/src/card/CardImpl.h grep
    max_option_number: float = 20.0  # practical max; no C++ cap
    max_option_count: float = 5.0  # practical max; no C++ cap


NORM = Norm()


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


# --- opponent-archetype belief (Task 8) ---------------------------------
#
# Tracked archetype classes, chosen deliberately now per plan.md §8.2 rule 4
# -- growing this list is a checkpoint-breaking change (the head's output
# width re-joins state_feats), so it's not meant to be painlessly resizable
# later. One reserved "Other" slot (index len(ARCHETYPE_CLASSES)) absorbs
# any future deck not in this list, without a width change for that case.
ARCHETYPE_CLASSES = ["00_basic", "01_psychic", "02_dragapult"]
ARCHETYPE_OUT = len(ARCHETYPE_CLASSES) + 1  # +1 for "Other"


def archetype_index(deck_name_or_path: str) -> int:
    """Map a deck name/path to its archetype class index, or the reserved
    "Other" index (len(ARCHETYPE_CLASSES)) if unrecognized."""
    stem = Path(deck_name_or_path).stem
    try:
        return ARCHETYPE_CLASSES.index(stem)
    except ValueError:
        return len(ARCHETYPE_CLASSES)


def _opponent_archetype_belief(obs: Observation, ctx: GameContext | None) -> np.ndarray:
    """Detached softmax belief carried on ctx, updated by the caller after
    each real decision (see pkm/rl/rollout.py:TorchPolicy.act) from the
    trunk's own archetype head -- one decision stale, never recomputed
    inside this pure function. Zero (uninformative) before the first
    update, or when ctx is None (e.g. MCTS's simulated tree)."""
    if ctx is not None and ctx.archetype_belief is not None:
        return np.asarray(ctx.archetype_belief, dtype=np.float32)
    return np.zeros(ARCHETYPE_OUT, dtype=np.float32)


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
    # Task 8: learned belief, not a deterministic fact -- see docstring.
    FeatureSpec(
        "opponent_archetype_belief",
        ARCHETYPE_OUT,
        Scope.GLOBAL,
        _opponent_archetype_belief,
        False,
    ),
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
    # Tier 1 (plan.md §4): bench retreat affordability.
    FeatureSpec("retreat_viable", 1, Scope.PER_SLOT, retreat_viable, True),
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
    # Tier 1 (plan.md §4): attack-option heuristics.
    FeatureSpec("lethal_this_turn", 1, Scope.PER_OPTION, lethal_this_turn, True),
    FeatureSpec("type_effectiveness", 1, Scope.PER_OPTION, type_effectiveness, True),
]


STATE_FEATS = N_POKEMON_SLOTS * sum(s.width for s in PER_SLOT_FEATURES) + sum(
    s.width for s in GLOBAL_FEATURES
)
OPT_FEATS = sum(s.width for s in PER_OPTION_FEATURES)


# --- deck ledger (Task 7) ----------------------------------------------------
#
# Not a FeatureSpec: this isn't a float feature slice, it's a raw
# (card_id, count) list the model pools through its own card_emb table
# (h_memory = sum_c unseen_count[c] * card_emb[c]), the same pattern as
# board_cards/hand_cards. Deliberately supersedes plan.md §5's flat 60-wide
# slot-indexed vector -- see pkm/rl/model.py for the pooling.


def deck_ledger(ctx: GameContext | None) -> tuple[np.ndarray, np.ndarray]:
    """Unique still-unseen card ids in my own deck and their counts, from
    ctx.tracker.by_location(CardLocation.DECK). Pure function of ctx alone
    (no obs dependency) -- empty arrays if ctx is None (e.g. MCTS's
    simulated tree, which must never touch a real per-game tracker)."""
    if ctx is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    tally = Counter(c.card_id for c in ctx.tracker.by_location(CardLocation.DECK))
    if not tally:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    ids = np.array(list(tally.keys()), dtype=np.int64)
    counts = np.array(list(tally.values()), dtype=np.float32)
    return ids, counts


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


# --- checkpoint-compatibility stamp -----------------------------------------


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


def stamp_json() -> str:
    """The current registry's stamp, JSON-encoded for embedding in an .npz
    array or a sidecar file next to a .pt checkpoint."""
    return json.dumps(feature_stamp())


def check_stamp_json(raw: str) -> None:
    expected = tuple(tuple(x) for x in json.loads(raw))
    check_stamp(expected)


def stamp_sidecar_path(checkpoint_path: str | Path) -> Path:
    """Sidecar path for a .pt checkpoint's feature stamp, e.g.
    ppo_latest.pt -> ppo_latest.pt.stamp.json."""
    return Path(str(checkpoint_path) + ".stamp.json")


def write_stamp_sidecar(checkpoint_path: str | Path) -> None:
    """Write the current registry's stamp alongside a saved .pt checkpoint.
    Call this right after torch.save()-ing a checkpoint meant to be reloaded
    later (e.g. via AgentProfile.latest_checkpoint)."""
    stamp_sidecar_path(checkpoint_path).write_text(stamp_json())


def check_stamp_sidecar(checkpoint_path: str | Path) -> None:
    """Raise if a stamp sidecar exists and doesn't match the current
    registry. A missing sidecar means the checkpoint predates stamping --
    can't verify it, so don't hard-fail on legacy checkpoints."""
    p = stamp_sidecar_path(checkpoint_path)
    if not p.is_file():
        return
    check_stamp_json(p.read_text())
