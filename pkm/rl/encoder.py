"""Observation and option encoding: obs dicts -> numpy arrays for the network.

The state is encoded as card-ID slots (embedded by the model) plus float
features. Options are encoded per-option as (type, card, target card, attack)
IDs plus float features, so the model can score a variable-length option list.
"""

from dataclasses import dataclass, field

import numpy as np

from pkm.data import get_attack_data
from pkm.types.obs import (
    MAX_BENCH,
    MAX_HAND,
    N_BOARD_SLOTS,
    N_POKEMON_SLOTS,
    NUM_ATTACKS,
    NUM_CARDS,
    NUM_OPT_TYPES,
    NUM_SELECT_TYPES,
    AreaType,
    GameState,
    Observation,
    OptionType,
    PokemonRef,
    Select,
)


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


SLOT_FEATS = 5
GLOBAL_FEATS = 45
STATE_FEATS = N_POKEMON_SLOTS * SLOT_FEATS + GLOBAL_FEATS
OPT_FEATS = 5


@dataclass
class EncodedDecision:
    """One decision point: encoded state + options, filled in by rollout/PPO."""

    board_cards: np.ndarray  # (N_BOARD_SLOTS,) int64
    hand_cards: np.ndarray  # (MAX_HAND,) int64
    state_feats: np.ndarray  # (STATE_FEATS,) float32
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


def _pokemon_slot_feats(p: PokemonRef | None) -> list[float]:
    if p is None:
        return [0.0] * SLOT_FEATS
    return [
        1.0,
        p.hp / NORM.max_hp,
        p.maxHp / NORM.max_hp,
        len(p.energies) / NORM.max_energies,
        1.0 if p.appearThisTurn else 0.0,
    ]


def encode_state(obs: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode observation into (board_cards, hand_cards, state_feats)."""
    state = obs.current
    sel = obs.select
    assert state is not None and sel is not None
    you = state.yourIndex
    me = state.players[you]
    opp = state.players[1 - you]

    board = np.zeros(N_BOARD_SLOTS, dtype=np.int64)
    slot_feats: list[float] = []

    slot = 0
    for player in (me, opp):
        pokes: list[PokemonRef | None] = [player.active_pokemon]
        pokes += list(player.bench)[:MAX_BENCH]
        pokes += [None] * (1 + MAX_BENCH - len(pokes))
        for p in pokes:
            board[slot] = p.id if p else 0
            slot_feats.extend(_pokemon_slot_feats(p))
            slot += 1
    board[slot] = state.stadium[0].id if state.stadium and state.stadium[0] else 0

    hand = np.zeros(MAX_HAND, dtype=np.int64)
    for i, c in enumerate((me.hand or [])[:MAX_HAND]):
        hand[i] = c.id

    g: list[float] = []
    for player in (me, opp):
        g.extend(
            [
                1.0 if player.poisoned else 0.0,
                1.0 if player.burned else 0.0,
                1.0 if player.asleep else 0.0,
                1.0 if player.paralyzed else 0.0,
                1.0 if player.confused else 0.0,
            ]
        )
    for player in (me, opp):
        g.extend(
            [
                player.handCount / NORM.max_hand_count,
                player.deckCount / NORM.max_deck_count,
                len(player.prize) / NORM.max_prize_count,
                len(player.discard) / NORM.max_discard_count,
            ]
        )
    for player in (me, opp):
        g.append(len(player.bench) / NORM.max_bench_count)
        g.append(player.benchMax / NORM.max_bench_count)
    g.append(state.turn / NORM.max_turn)
    g.append(state.turnActionCount / NORM.max_actions_per_turn)
    g.extend(
        [
            1.0 if state.energyAttached else 0.0,
            1.0 if state.supporterPlayed else 0.0,
            1.0 if state.stadiumPlayed else 0.0,
            1.0 if state.retreated else 0.0,
        ]
    )
    g.append(1.0 if state.firstPlayer == you else 0.0)
    g.append(1.0 if state.firstPlayer >= 0 else 0.0)

    sel_onehot = [0.0] * NUM_SELECT_TYPES
    st = sel.type
    if 0 <= st < NUM_SELECT_TYPES:
        sel_onehot[st] = 1.0
    g.extend(sel_onehot)
    g.extend(
        [
            sel.minCount / NORM.max_pick_count,
            sel.maxCount / NORM.max_pick_count,
            sel.remainEnergyCost / NORM.max_energy_cost,
            sel.remainDamageCounter / NORM.max_damage_counters,
        ]
    )

    feats = np.array(slot_feats + g, dtype=np.float32)
    assert feats.shape[0] == STATE_FEATS, feats.shape
    return board, hand, feats


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
        if area == AreaType.DECK:
            c = (select.deck or [])[index]
        elif area == AreaType.HAND:
            c = (players[player_index].hand or [])[index]
        elif area == AreaType.TRASH:
            c = players[player_index].discard[index]
        elif area == AreaType.ACTIVE:
            c = players[player_index].active[index]
        elif area == AreaType.BENCH:
            c = players[player_index].bench[index]
        elif area == AreaType.PRIZE:
            c = players[player_index].prize[index]
        elif area == AreaType.STADIUM:
            c = state.stadium[index]
        elif area == AreaType.LOOKING:
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
        if area == AreaType.ACTIVE:
            return state.players[player_index].active[index]
        if area == AreaType.BENCH:
            return state.players[player_index].bench[index]
    except (TypeError, IndexError, AttributeError):
        pass
    return None


def encode_options(obs: Observation) -> dict[str, np.ndarray]:
    """Encode the option list into parallel arrays."""
    state = obs.current
    select = obs.select
    assert state is not None and select is not None
    you = state.yourIndex
    options = select.option
    n = len(options)

    attack_data = get_attack_data()

    opt_type = np.zeros(n, dtype=np.int64)
    opt_card = np.zeros(n, dtype=np.int64)
    opt_card2 = np.zeros(n, dtype=np.int64)
    opt_attack = np.zeros(n, dtype=np.int64)
    opt_feats = np.zeros((n, OPT_FEATS), dtype=np.float32)

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
        damage = 0.0
        cost = 0.0

        if t == OptionType.CARD:
            card_id = _card_id_at(state, select, pi, area, index)
        elif t == OptionType.TOOL_CARD:
            p = _pokemon_at(state, pi, area, index)
            if p:
                card2_id = p.id
                tools = p.tools
                ti = o.toolIndex
                if ti is not None and ti < len(tools):
                    card_id = tools[ti].id
        elif t in (OptionType.ENERGY_CARD, OptionType.ENERGY):
            p = _pokemon_at(state, pi, area, index)
            if p:
                card2_id = p.id
                ecards = p.energyCards
                ei = o.energyIndex
                if ei is not None and ei < len(ecards):
                    card_id = ecards[ei].id
        elif t == OptionType.PLAY:
            card_id = _card_id_at(state, select, you, AreaType.HAND, index)
        elif t in (OptionType.ATTACH, OptionType.EVOLVE):
            card_id = _card_id_at(state, select, you, area, index)
            p = _pokemon_at(state, you, o.inPlayArea, o.inPlayIndex)
            if p:
                card2_id = p.id
        elif t in (OptionType.ABILITY, OptionType.DISCARD):
            card_id = _card_id_at(state, select, pi, area, index)
        elif t == OptionType.ATTACK:
            attack_id = o.attackId or 0
            active = state.players[you].active_pokemon
            if active:
                card_id = active.id
            atk = attack_data.get(attack_id)
            if atk:
                damage = atk.damage / NORM.max_damage
                cost = len(atk.energies) / NORM.max_energies
        elif t == OptionType.SKILL:
            card_id = o.cardId or 0

        opt_card[i] = card_id if 0 <= card_id < NUM_CARDS else 0
        opt_card2[i] = card2_id if 0 <= card2_id < NUM_CARDS else 0
        opt_attack[i] = attack_id if 0 <= attack_id < NUM_ATTACKS else 0
        opt_feats[i] = [
            (o.number or 0) / NORM.max_option_number,
            (o.count or 0) / NORM.max_option_count,
            damage,
            cost,
            1.0 if pi == you else 0.0,
        ]

    return {
        "opt_type": opt_type,
        "opt_card": opt_card,
        "opt_card2": opt_card2,
        "opt_attack": opt_attack,
        "opt_feats": opt_feats,
    }


def encode_decision(obs: Observation) -> EncodedDecision:
    """Encode a full decision point (state + options)."""
    board, hand, feats = encode_state(obs)
    opts = encode_options(obs)
    sel = obs.select
    assert sel is not None
    return EncodedDecision(
        board_cards=board,
        hand_cards=hand,
        state_feats=feats,
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
