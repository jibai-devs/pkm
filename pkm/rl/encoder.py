"""Observation and option encoding: obs dicts -> numpy arrays for the network.

The state is encoded as card-ID slots (embedded by the model) plus float
features. Options are encoded per-option as (type, card, target card, attack)
IDs plus float features, so the model can score a variable-length option list.
"""

from dataclasses import dataclass, field

import numpy as np

from pkm.data import get_attack_data

# Vocabulary sizes (id 0 = pad/unknown; real ids start at 1)
NUM_CARDS = 1268
NUM_ATTACKS = 1557
NUM_OPT_TYPES = 17
NUM_SELECT_TYPES = 11

MAX_BENCH = 8
MAX_HAND = 25
# my active + my bench, opp active + opp bench, stadium
N_POKEMON_SLOTS = 2 * (1 + MAX_BENCH)
N_BOARD_SLOTS = N_POKEMON_SLOTS + 1
SLOT_FEATS = 5
GLOBAL_FEATS = 45
STATE_FEATS = N_POKEMON_SLOTS * SLOT_FEATS + GLOBAL_FEATS
OPT_FEATS = 5

# AreaType values (see docs / official api.py)
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

# OptionType values
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


def _pokemon_slot_feats(p: dict | None) -> list[float]:
    if p is None:
        return [0.0] * SLOT_FEATS
    return [
        1.0,
        p["hp"] / 300.0,
        p["maxHp"] / 300.0,
        len(p.get("energies") or []) / 5.0,
        1.0 if p.get("appearThisTurn") else 0.0,
    ]


def _active(player: dict) -> dict | None:
    act = player.get("active") or []
    return act[0] if act else None


def encode_state(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode observation into (board_cards, hand_cards, state_feats)."""
    state = obs["current"]
    you = state["yourIndex"]
    me = state["players"][you]
    opp = state["players"][1 - you]

    board = np.zeros(N_BOARD_SLOTS, dtype=np.int64)
    slot_feats: list[float] = []

    slot = 0
    for player in (me, opp):
        pokes = [_active(player)] + list(player.get("bench") or [])[:MAX_BENCH]
        pokes += [None] * (1 + MAX_BENCH - len(pokes))
        for p in pokes:
            board[slot] = p["id"] if p else 0
            slot_feats.extend(_pokemon_slot_feats(p))
            slot += 1
    stadium = state.get("stadium") or []
    board[slot] = stadium[0]["id"] if stadium else 0

    hand = np.zeros(MAX_HAND, dtype=np.int64)
    for i, c in enumerate((me.get("hand") or [])[:MAX_HAND]):
        hand[i] = c["id"]

    g: list[float] = []
    for player in (me, opp):
        g.extend(
            [
                1.0 if player["poisoned"] else 0.0,
                1.0 if player["burned"] else 0.0,
                1.0 if player["asleep"] else 0.0,
                1.0 if player["paralyzed"] else 0.0,
                1.0 if player["confused"] else 0.0,
            ]
        )
    for player in (me, opp):
        g.extend(
            [
                player["handCount"] / 20.0,
                player["deckCount"] / 60.0,
                len(player["prize"]) / 6.0,
                len(player["discard"]) / 60.0,
            ]
        )
    for player in (me, opp):
        g.append(len(player.get("bench") or []) / 8.0)
        g.append(player.get("benchMax", 5) / 8.0)
    g.append(state["turn"] / 30.0)
    g.append(state["turnActionCount"] / 20.0)
    g.extend(
        [
            1.0 if state["energyAttached"] else 0.0,
            1.0 if state["supporterPlayed"] else 0.0,
            1.0 if state["stadiumPlayed"] else 0.0,
            1.0 if state["retreated"] else 0.0,
        ]
    )
    g.append(1.0 if state["firstPlayer"] == you else 0.0)
    g.append(1.0 if state["firstPlayer"] >= 0 else 0.0)

    sel = obs["select"]
    sel_onehot = [0.0] * NUM_SELECT_TYPES
    st = sel["type"]
    if 0 <= st < NUM_SELECT_TYPES:
        sel_onehot[st] = 1.0
    g.extend(sel_onehot)
    g.extend(
        [
            sel["minCount"] / 5.0,
            sel["maxCount"] / 5.0,
            sel.get("remainEnergyCost", 0) / 5.0,
            sel.get("remainDamageCounter", 0) / 10.0,
        ]
    )

    feats = np.array(slot_feats + g, dtype=np.float32)
    assert feats.shape[0] == STATE_FEATS, feats.shape
    return board, hand, feats


def _card_id_at(
    state: dict, select: dict, player_index: int, area: int, index: int
) -> int:
    """Best-effort resolution of (playerIndex, area, index) -> card ID."""
    players = state["players"]
    try:
        if area == AREA_DECK:
            c = (select.get("deck") or [])[index]
        elif area == AREA_HAND:
            c = (players[player_index].get("hand") or [])[index]
        elif area == AREA_DISCARD:
            c = players[player_index]["discard"][index]
        elif area == AREA_ACTIVE:
            c = players[player_index]["active"][index]
        elif area == AREA_BENCH:
            c = players[player_index]["bench"][index]
        elif area == AREA_PRIZE:
            c = players[player_index]["prize"][index]
        elif area == AREA_STADIUM:
            c = state["stadium"][index]
        elif area == AREA_LOOKING:
            c = (state.get("looking") or [])[index]
        else:
            return 0
        return c["id"] if c else 0
    except (TypeError, IndexError, KeyError):
        return 0


def _pokemon_at(state: dict, player_index: int, area: int, index: int) -> dict | None:
    try:
        if area == AREA_ACTIVE:
            return state["players"][player_index]["active"][index]
        if area == AREA_BENCH:
            return state["players"][player_index]["bench"][index]
    except (TypeError, IndexError, KeyError):
        pass
    return None


def encode_options(obs: dict) -> dict[str, np.ndarray]:
    """Encode the option list into parallel arrays."""
    state = obs["current"]
    select = obs["select"]
    you = state["yourIndex"]
    options = select["option"]
    n = len(options)

    attack_data = get_attack_data()

    opt_type = np.zeros(n, dtype=np.int64)
    opt_card = np.zeros(n, dtype=np.int64)
    opt_card2 = np.zeros(n, dtype=np.int64)
    opt_attack = np.zeros(n, dtype=np.int64)
    opt_feats = np.zeros((n, OPT_FEATS), dtype=np.float32)

    for i, o in enumerate(options):
        t = o["type"]
        opt_type[i] = t if 0 <= t < NUM_OPT_TYPES else 0
        pi = o.get("playerIndex")
        pi = you if pi is None else pi
        area = o.get("area")
        index = o.get("index")
        card_id = 0
        card2_id = 0
        attack_id = 0
        damage = 0.0
        cost = 0.0

        if t == OPT_CARD:
            card_id = _card_id_at(state, select, pi, area, index)
        elif t == OPT_TOOL_CARD:
            p = _pokemon_at(state, pi, area, index)
            if p:
                card2_id = p["id"]
                tools = p.get("tools") or []
                ti = o.get("toolIndex")
                if ti is not None and ti < len(tools):
                    card_id = tools[ti]["id"]
        elif t in (OPT_ENERGY_CARD, OPT_ENERGY):
            p = _pokemon_at(state, pi, area, index)
            if p:
                card2_id = p["id"]
                ecards = p.get("energyCards") or []
                ei = o.get("energyIndex")
                if ei is not None and ei < len(ecards):
                    card_id = ecards[ei]["id"]
        elif t == OPT_PLAY:
            card_id = _card_id_at(state, select, you, AREA_HAND, index)
        elif t in (OPT_ATTACH, OPT_EVOLVE):
            card_id = _card_id_at(state, select, you, area, index)
            p = _pokemon_at(state, you, o.get("inPlayArea"), o.get("inPlayIndex"))
            if p:
                card2_id = p["id"]
        elif t in (OPT_ABILITY, OPT_DISCARD):
            card_id = _card_id_at(state, select, pi, area, index)
        elif t == OPT_ATTACK:
            attack_id = o.get("attackId") or 0
            active = _active(state["players"][you])
            if active:
                card_id = active["id"]
            atk = attack_data.get(attack_id)
            if atk:
                damage = atk.damage / 300.0
                cost = len(atk.energies) / 5.0
        elif t == OPT_SKILL:
            card_id = o.get("cardId") or 0

        opt_card[i] = card_id if 0 <= card_id < NUM_CARDS else 0
        opt_card2[i] = card2_id if 0 <= card2_id < NUM_CARDS else 0
        opt_attack[i] = attack_id if 0 <= attack_id < NUM_ATTACKS else 0
        opt_feats[i] = [
            (o.get("number") or 0) / 20.0,
            (o.get("count") or 0) / 5.0,
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


def encode_decision(obs: dict) -> EncodedDecision:
    """Encode a full decision point (state + options)."""
    board, hand, feats = encode_state(obs)
    opts = encode_options(obs)
    sel = obs["select"]
    return EncodedDecision(
        board_cards=board,
        hand_cards=hand,
        state_feats=feats,
        min_count=sel["minCount"],
        max_count=sel["maxCount"],
        **opts,
    )


def prize_potential(obs: dict) -> float:
    """Prize differential from the to-move player's perspective, in [-1, 1].

    Taking a prize removes it from *your own* prize pile, so fewer prizes
    remaining for me than for the opponent means I'm ahead.
    """
    state = obs["current"]
    you = state["yourIndex"]
    me = state["players"][you]
    opp = state["players"][1 - you]
    return (len(opp["prize"]) - len(me["prize"])) / 6.0
