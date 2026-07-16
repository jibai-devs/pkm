"""Observation and option encoding: obs dicts -> numpy arrays for the network.

The state is encoded as card-ID slots (embedded by the model) plus float
features. Options are encoded per-option as (type, card, target card, attack)
IDs plus float features, so the model can score a variable-length option list.

The float feature slices (state_feats/opt_feats) are assembled from the
declarative registry in `pkm/rl/features.py`; this module owns the raw
card-ID arrays (board_cards/hand_cards/opt_card/opt_card2/opt_attack) and
option-type dispatch, which aren't part of that registry.
"""

from dataclasses import dataclass, field

import numpy as np

from pkm.heuristics.context import GameContext
from pkm.rl.features import (
    NORM,
    OPT_ABILITY,
    OPT_ATTACH,
    OPT_ATTACK,
    OPT_CARD,
    OPT_DISCARD,
    OPT_ENERGY,
    OPT_ENERGY_CARD,
    OPT_EVOLVE,
    OPT_FEATS,  # noqa: F401 -- re-exported for pkm.rl.model
    OPT_PLAY,
    OPT_SKILL,
    OPT_TOOL_CARD,
    STATE_FEATS,
    FeatureConfig,
    assemble_global,
    assemble_per_option,
    assemble_per_slot,
    board_pokemon,
    deck_ledger,
)
from pkm.types.obs import (
    MAX_HAND,
    N_BOARD_SLOTS,
    NUM_ATTACKS,
    NUM_CARDS,
    NUM_OPT_TYPES,
    GameState,
    Observation,
    PokemonRef,
    Select,
)

# AreaType values (see docs / official api.py)
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12


@dataclass
class EncodedDecision:
    """One decision point: encoded state + options, filled in by rollout/PPO."""

    board_cards: np.ndarray  # (N_BOARD_SLOTS,) int64
    hand_cards: np.ndarray  # (MAX_HAND,) int64
    state_feats: np.ndarray  # (STATE_FEATS,) float32
    deck_card_ids: np.ndarray  # (K,) int64 -- unique still-unseen card ids
    deck_card_counts: np.ndarray  # (K,) float32 -- their unseen counts
    opt_type: np.ndarray  # (N,) int64
    opt_card: np.ndarray  # (N,) int64
    opt_card2: np.ndarray  # (N,) int64
    opt_attack: np.ndarray  # (N,) int64
    opt_feats: np.ndarray  # (N, OPT_FEATS) float32
    min_count: int
    max_count: int
    # filled by the acting policy
    picks: list[int] = field(default_factory=list)
    stopped: bool = False  # True if the STOP action ended the pick sequence
    logprob: float = 0.0
    value: float = 0.0
    # filled by return computation
    potential: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0


def encode_state(
    obs: Observation,
    ctx: GameContext | None = None,
    config: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode observation into (board_cards, hand_cards, state_feats,
    deck_card_ids, deck_card_counts)."""
    state = obs.current
    assert state is not None and obs.select is not None
    me = state.players[state.yourIndex]

    pokes = board_pokemon(obs)
    board = np.zeros(N_BOARD_SLOTS, dtype=np.int64)
    for i, p in enumerate(pokes):
        board[i] = p.id if p else 0
    board[len(pokes)] = state.stadium[0].id if state.stadium and state.stadium[0] else 0

    hand = np.zeros(MAX_HAND, dtype=np.int64)
    for i, c in enumerate((me.hand or [])[:MAX_HAND]):
        hand[i] = c.id

    feats = np.concatenate(
        [
            assemble_per_slot(obs, ctx, config),
            assemble_global(obs, ctx, config),
        ]
    )
    assert feats.shape[0] == STATE_FEATS, feats.shape

    deck_card_ids, deck_card_counts = deck_ledger(ctx)

    return board, hand, feats, deck_card_ids, deck_card_counts


def _card_id_at(
    state: GameState,
    select: Select,
    player_index: int,
    area: int | None,
    index: int | None,
) -> int:
    """Best-effort resolution of (playerIndex, area, index) -> card ID."""
    if index is None:
        return 0
    players = state.players
    try:
        if area == AREA_DECK:
            c = (select.deck or [])[index]
        elif area == AREA_HAND:
            c = (players[player_index].hand or [])[index]
        elif area == AREA_DISCARD:
            c = players[player_index].discard[index]
        elif area == AREA_ACTIVE:
            c = players[player_index].active[index]
        elif area == AREA_BENCH:
            c = players[player_index].bench[index]
        elif area == AREA_PRIZE:
            c = players[player_index].prize[index]
        elif area == AREA_STADIUM:
            c = state.stadium[index]
        elif area == AREA_LOOKING:
            c = (state.looking or [])[index]
        else:
            return 0
        return c.id if c else 0
    except (TypeError, IndexError, AttributeError):
        return 0


def _pokemon_at(
    state: GameState, player_index: int, area: int | None, index: int | None
) -> PokemonRef | None:
    if index is None:
        return None
    try:
        if area == AREA_ACTIVE:
            return state.players[player_index].active[index]
        if area == AREA_BENCH:
            return state.players[player_index].bench[index]
    except (TypeError, IndexError, AttributeError):
        pass
    return None


def encode_options(
    obs: Observation,
    ctx: GameContext | None = None,
    config: FeatureConfig | None = None,
) -> dict[str, np.ndarray]:
    """Encode the option list into parallel arrays."""
    state = obs.current
    select = obs.select
    assert state is not None and select is not None
    you = state.yourIndex
    options = select.option
    n = len(options)

    opt_type = np.zeros(n, dtype=np.int64)
    opt_card = np.zeros(n, dtype=np.int64)
    opt_card2 = np.zeros(n, dtype=np.int64)
    opt_attack = np.zeros(n, dtype=np.int64)

    for i, o in enumerate(options):
        t = o.type
        opt_type[i] = t if 0 <= t < NUM_OPT_TYPES else 0
        pi = o.playerIndex
        pi = you if pi is None else pi
        area = o.area
        index = o.index
        card_id = 0
        card2_id = 0
        attack_id = 0

        if t == OPT_CARD:
            card_id = _card_id_at(state, select, pi, area, index)
        elif t == OPT_TOOL_CARD:
            p = _pokemon_at(state, pi, area, index)
            if p:
                card2_id = p.id
                tools = p.tools
                ti = o.toolIndex
                if ti is not None and ti < len(tools):
                    card_id = tools[ti].id
        elif t in (OPT_ENERGY_CARD, OPT_ENERGY):
            p = _pokemon_at(state, pi, area, index)
            if p:
                card2_id = p.id
                ecards = p.energyCards
                ei = o.energyIndex
                if ei is not None and ei < len(ecards):
                    card_id = ecards[ei].id
        elif t == OPT_PLAY:
            card_id = _card_id_at(state, select, you, AREA_HAND, index)
        elif t in (OPT_ATTACH, OPT_EVOLVE):
            card_id = _card_id_at(state, select, you, area, index)
            p = _pokemon_at(state, you, o.inPlayArea, o.inPlayIndex)
            if p:
                card2_id = p.id
        elif t in (OPT_ABILITY, OPT_DISCARD):
            card_id = _card_id_at(state, select, pi, area, index)
        elif t == OPT_ATTACK:
            attack_id = o.attackId or 0
            active = state.players[you].active_pokemon
            if active:
                card_id = active.id
        elif t == OPT_SKILL:
            card_id = o.cardId or 0

        opt_card[i] = card_id if 0 <= card_id < NUM_CARDS else 0
        opt_card2[i] = card2_id if 0 <= card2_id < NUM_CARDS else 0
        opt_attack[i] = attack_id if 0 <= attack_id < NUM_ATTACKS else 0

    return {
        "opt_type": opt_type,
        "opt_card": opt_card,
        "opt_card2": opt_card2,
        "opt_attack": opt_attack,
        "opt_feats": assemble_per_option(obs, ctx, config),
    }


def encode_decision(
    obs: Observation,
    ctx: GameContext | None = None,
    config: FeatureConfig | None = None,
) -> EncodedDecision:
    """Encode a full decision point (state + options)."""
    board, hand, feats, deck_ids, deck_counts = encode_state(obs, ctx, config)
    opts = encode_options(obs, ctx, config)
    sel = obs.select
    assert sel is not None
    return EncodedDecision(
        board_cards=board,
        hand_cards=hand,
        state_feats=feats,
        deck_card_ids=deck_ids,
        deck_card_counts=deck_counts,
        min_count=sel.minCount,
        max_count=sel.maxCount,
        **opts,
    )


def prize_potential(obs: Observation) -> float:
    """Prize differential from the to-move player's perspective, in [-1, 1].

    Taking a prize removes it from *your own* prize pile, so fewer prizes
    remaining for me than for the opponent means I'm ahead.
    """
    state = obs.current
    assert state is not None
    you = state.yourIndex
    me = state.players[you]
    opp = state.players[1 - you]
    return (len(opp.prize) - len(me.prize)) / NORM.max_prize_count
