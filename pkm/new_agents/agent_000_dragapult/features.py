"""Observation -> numpy tensors for agent_000_dragapult (featurizer v1).

Pure, learning-free conversion from a typed :class:`~pkm.cabt.api.Observation`
into fixed-shape numpy arrays. This is the boundary between the engine/env
(runs in worker processes, no torch needed) and the model (converts numpy ->
torch on the learner).

**This is v1 and provisional.** The concrete choices below are the simplest
defensible ones, NOT locked decisions (see README `[DECIDE]` items):
  * board = 12 FIXED slots (own active + 5 bench, opp active + 5 bench) + mask.
  * our own hand / discard = count histogram over our 27-row vocab
    (opponent hand is hidden; opponent discard is open-vocab -> only counted).
  * options are emitted as (type + normalized raw fields); resolving an option
    to the board entity it references (for the pointer action head) is TODO.

Normalizers come from the deterministic ``spec.json`` so they never drift.
Bump ``FEATURE_VERSION`` on any layout change (checkpoints record it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pkm.cabt.api import Card, Observation, Option, Pokemon
from pkm.agents.agent_000_dragapult import deck

FEATURE_VERSION = "v1"

_SPEC = json.loads(Path(__file__).with_name("spec.json").read_text())
_MAX_HP = float(_SPEC["global_max"]["max_hp"])          # 380
_N_ENERGY = _SPEC["constants"]["n_energy_types"]        # 12
_BOARD_SLOTS = _SPEC["constants"]["board_slots"]        # 12
_VOCAB = deck.VOCAB_SIZE                                 # 27

# --- normalizers, with sources (no magic numbers) ---
# Rule / spec-derived (exact, meaningful denominators):
_DECK_SIZE = _SPEC["vocab"]["deck_size"]                # 60 — deck-construction rule
_BENCH_MAX = _SPEC["constants"]["bench_max"]            # 5  — max bench
_N_PRIZES = 6                                           # standard Prize count (game rule)
_MAX_AREA = 24                                          # max AreaType code (core/CardTypes.h:18)
# Arbitrary display scales: no natural maximum exists, so these are provisional
# ~typical magnitudes chosen only to keep inputs O(1). Tune later; not meaningful.
_TURN_SCALE = 50.0
_ACTION_SCALE = 10.0
_HAND_SCALE = 15.0
_INDEX_SCALE = 12.0     # provisional: index-within-area (bench/hand) magnitude
_NUMBER_SCALE = 64.0    # provisional: COUNT option numeric magnitude
_ATTACH_SCALE = 8.0     # provisional: tool/energy sub-index magnitude

# entity_feat column layout (F = 26); documented so downstream never guesses.
ENTITY_FEAT_COLS = (
    ["hp_norm", "maxhp_norm", "hp_frac", "appear_this_turn",
     "n_tools", "n_energy_cards", "evo_depth", "is_own", "is_active"]
    + [f"energy_{i}" for i in range(_N_ENERGY)]
    + ["poisoned", "burned", "asleep", "paralyzed", "confused"]
)
F = len(ENTITY_FEAT_COLS)  # 26

GLOBAL_COLS = [
    "turn", "turn_action_count", "we_are_first",
    "supporter_played", "stadium_played", "energy_attached", "retreated",
    "own_prize", "opp_prize", "own_deck", "opp_deck",
    "own_hand", "opp_hand", "stadium_present", "own_bench", "opp_bench",
]
G = len(GLOBAL_COLS)  # 16

# option_feat column layout (O). Provisional raw-field encoding.
OPTION_FEAT_COLS = [
    "has_area", "area", "index", "player_index", "tool_index", "energy_index",
    "count", "number", "has_inplay_area", "inplay_area", "inplay_index", "has_attack",
]
O = len(OPTION_FEAT_COLS)  # 12


@dataclass(slots=True)
class Features:
    """Fixed-shape featurized observation. Batch dim added later by the collator."""

    entity_id_row: npt.NDArray[np.int64]     # [12]      row into our 27-vocab (UNK for opp/unknown)
    entity_card_id: npt.NDArray[np.int64]    # [12]      raw card id (0 if empty) -> attribute lookup
    entity_feat: npt.NDArray[np.float32]     # [12, F]
    entity_mask: npt.NDArray[np.float32]     # [12]      1 = occupied
    hand_hist: npt.NDArray[np.float32]       # [27]      our hand as count histogram over vocab
    discard_hist: npt.NDArray[np.float32]    # [27]      our discard as count histogram over vocab
    globals: npt.NDArray[np.float32]         # [G]
    option_type: npt.NDArray[np.int64]       # [L]
    option_feat: npt.NDArray[np.float32]     # [L, O]
    select_type: int
    select_context: int
    min_count: int
    max_count: int

    @property
    def n_options(self) -> int:
        return int(self.option_type.shape[0])


def _entity_row(
    pk: Pokemon | None, is_own: bool, is_active: bool, status: dict[str, bool]
) -> tuple[int, int, npt.NDArray[np.float32]]:
    """Return (id_row, card_id, feature_vector) for one board slot; empty -> zeros."""
    feat = np.zeros(F, dtype=np.float32)
    if pk is None:
        return deck.UNK_ROW, 0, feat
    feat[0] = pk.hp / _MAX_HP
    feat[1] = pk.maxHp / _MAX_HP
    feat[2] = (pk.hp / pk.maxHp) if pk.maxHp else 0.0
    feat[3] = float(pk.appearThisTurn)
    feat[4] = float(len(pk.tools))
    feat[5] = float(len(pk.energyCards))
    feat[6] = float(len(pk.preEvolution))
    feat[7] = float(is_own)
    feat[8] = float(is_active)
    base = 9
    for e in pk.energies:
        feat[base + int(e)] += 1.0
    if is_active:  # status conditions apply to the active Pokémon
        feat[base + _N_ENERGY + 0] = float(status["poisoned"])
        feat[base + _N_ENERGY + 1] = float(status["burned"])
        feat[base + _N_ENERGY + 2] = float(status["asleep"])
        feat[base + _N_ENERGY + 3] = float(status["paralyzed"])
        feat[base + _N_ENERGY + 4] = float(status["confused"])
    return deck.row_of(pk.id), pk.id, feat


def _hist(cards: list[Card] | None) -> npt.NDArray[np.float32]:
    """Count histogram over our 27-row vocab (non-owned cards land on UNK)."""
    h = np.zeros(_VOCAB, dtype=np.float32)
    for c in cards or []:
        h[deck.row_of(c.id)] += 1.0
    return h


def _option_row(opt: Option) -> npt.NDArray[np.float32]:
    # PROVISIONAL: this encodes the option's enum/index fields as scaled scalars.
    # Per the embedding decision, `area`/`inPlayArea` (AreaType) and the option
    # `type` should become embedding indices, not normalized floats — this block
    # will be replaced when the pointer action head is built. Kept scalar for the
    # shape-only v1 featurizer. Area normalized by the true max AreaType code.
    f = np.zeros(O, dtype=np.float32)
    f[0] = float(opt.area is not None)
    f[1] = (int(opt.area) / _MAX_AREA) if opt.area is not None else 0.0
    f[2] = (opt.index / _INDEX_SCALE) if opt.index is not None else 0.0
    f[3] = float(opt.playerIndex) if opt.playerIndex is not None else 0.0  # 0/1 owner indicator (already O(1))
    f[4] = (opt.toolIndex / _ATTACH_SCALE) if opt.toolIndex is not None else 0.0
    f[5] = (opt.energyIndex / _ATTACH_SCALE) if opt.energyIndex is not None else 0.0
    f[6] = (opt.count / _ATTACH_SCALE) if opt.count is not None else 0.0
    f[7] = (opt.number / _NUMBER_SCALE) if opt.number is not None else 0.0
    f[8] = float(opt.inPlayArea is not None)
    f[9] = (int(opt.inPlayArea) / _MAX_AREA) if opt.inPlayArea is not None else 0.0
    f[10] = (opt.inPlayIndex / _INDEX_SCALE) if opt.inPlayIndex is not None else 0.0
    f[11] = float(opt.attackId is not None)
    return f


def featurize(obs: Observation) -> Features:
    """Convert a typed Observation (from our POV) into fixed-shape arrays.

    Requires ``obs.current`` (raises if called during the deck-selection phase,
    where ``current``/``select`` are None — the agent returns the deck there).
    """
    st = obs.current
    if st is None:
        raise ValueError("featurize() needs obs.current; deck-selection phase has none")
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
    my_status = {k: getattr(me, k) for k in ("poisoned", "burned", "asleep", "paralyzed", "confused")}
    op_status = {k: getattr(opp, k) for k in ("poisoned", "burned", "asleep", "paralyzed", "confused")}

    id_rows = np.zeros(_BOARD_SLOTS, dtype=np.int64)
    card_ids = np.zeros(_BOARD_SLOTS, dtype=np.int64)
    feats = np.zeros((_BOARD_SLOTS, F), dtype=np.float32)
    mask = np.zeros(_BOARD_SLOTS, dtype=np.float32)

    def put(slot: int, pk: Pokemon | None, is_own: bool, is_active: bool, status: dict[str, bool]) -> None:
        r, cid, f = _entity_row(pk, is_own, is_active, status)
        id_rows[slot] = r
        card_ids[slot] = cid
        feats[slot] = f
        mask[slot] = 1.0 if pk is not None else 0.0

    # slots 0..5 = us (active + 5 bench), 6..11 = opponent
    put(0, me.active[0] if me.active else None, True, True, my_status)
    for i in range(5):
        put(1 + i, me.bench[i] if i < len(me.bench) else None, True, False, my_status)
    put(6, opp.active[0] if opp.active else None, False, True, op_status)
    for i in range(5):
        put(7 + i, opp.bench[i] if i < len(opp.bench) else None, False, False, op_status)

    g = np.zeros(G, dtype=np.float32)
    g[0] = st.turn / _TURN_SCALE
    g[1] = st.turnActionCount / _ACTION_SCALE
    g[2] = float(st.firstPlayer == st.yourIndex)
    g[3] = float(st.supporterPlayed)
    g[4] = float(st.stadiumPlayed)
    g[5] = float(st.energyAttached)
    g[6] = float(st.retreated)
    g[7] = len(me.prize) / _N_PRIZES
    g[8] = len(opp.prize) / _N_PRIZES
    g[9] = me.deckCount / _DECK_SIZE
    g[10] = opp.deckCount / _DECK_SIZE
    g[11] = me.handCount / _HAND_SCALE
    g[12] = opp.handCount / _HAND_SCALE
    g[13] = float(len(st.stadium) > 0)
    g[14] = len(me.bench) / _BENCH_MAX
    g[15] = len(opp.bench) / _BENCH_MAX

    sel = obs.select
    if sel is not None and sel.option:
        opt_type = np.array([int(o.type) for o in sel.option], dtype=np.int64)
        opt_feat = np.stack([_option_row(o) for o in sel.option]).astype(np.float32)
        stype, sctx, mn, mx = int(sel.type), int(sel.context), sel.minCount, sel.maxCount
    else:
        opt_type = np.zeros(0, dtype=np.int64)
        opt_feat = np.zeros((0, O), dtype=np.float32)
        stype = sctx = mn = mx = 0

    return Features(
        entity_id_row=id_rows,
        entity_card_id=card_ids,
        entity_feat=feats,
        entity_mask=mask,
        hand_hist=_hist(me.hand),
        discard_hist=_hist(me.discard),
        globals=g,
        option_type=opt_type,
        option_feat=opt_feat,
        select_type=stype,
        select_context=sctx,
        min_count=mn,
        max_count=mx,
    )
