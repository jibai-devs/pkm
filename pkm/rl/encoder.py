"""Observation and option encoding: obs dicts -> numpy arrays for the network.

The state is encoded as card-ID slots (embedded by the model) plus float
features. Options are encoded per-option as (type, card, target card, attack)
IDs plus float features, so the model can score a variable-length option list.
"""

from dataclasses import dataclass, field

import numpy as np

from pkm.data import get_attack_data
from pkm.data.card_data import CardData, get_card_by_id
from pkm.types.obs import GameState, Observation, Option, PokemonRef, Select

# CardType values (see obs_data_structure/OBSERVATION_SCHEMA.md)
CARD_TYPE_BASIC_ENERGY = 5
CARD_TYPE_SPECIAL_ENERGY = 6

BUDEW_CARD_ID = 235
DREEPY_CARD_ID = 119
DRAKLOAK_CARD_ID = 120
DRAGAPULT_EX_CARD_ID = 121
DREEPY_LINE_CARD_IDS = {DREEPY_CARD_ID, DRAKLOAK_CARD_ID, DRAGAPULT_EX_CARD_ID}
XEROSIC_MACHINATIONS_CARD_ID = 1197

# EnergyType values (see pkm/types/obs.py's EnergyType enum)
ENERGY_TYPE_FIRE = 2
ENERGY_TYPE_PSYCHIC = 5

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
    board_setup_potential: float = 0.0
    budew_setup_potential: float = 0.0
    energy_penalty: float = 0.0
    budew_bonus: float = 0.0
    wrong_type_energy_penalty: float = 0.0
    dragapult_attack_bonus: float = 0.0
    dreepy_spread_penalty: float = 0.0
    xerosic_bonus: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0


def _pokemon_slot_feats(p: PokemonRef | None) -> list[float]:
    if p is None:
        return [0.0] * SLOT_FEATS
    return [
        1.0,
        p.hp / 300.0,
        p.maxHp / 300.0,
        len(p.energies) / 5.0,
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
                player.handCount / 20.0,
                player.deckCount / 60.0,
                len(player.prize) / 6.0,
                len(player.discard) / 60.0,
            ]
        )
    for player in (me, opp):
        g.append(len(player.bench) / 8.0)
        g.append(player.benchMax / 8.0)
    g.append(state.turn / 30.0)
    g.append(state.turnActionCount / 20.0)
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
            sel.minCount / 5.0,
            sel.maxCount / 5.0,
            sel.remainEnergyCost / 5.0,
            sel.remainDamageCounter / 10.0,
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
            atk = attack_data.get(attack_id)
            if atk:
                damage = atk.damage / 300.0
                cost = len(atk.energies) / 5.0
        elif t == OPT_SKILL:
            card_id = o.cardId or 0

        opt_card[i] = card_id if 0 <= card_id < NUM_CARDS else 0
        opt_card2[i] = card2_id if 0 <= card2_id < NUM_CARDS else 0
        opt_attack[i] = attack_id if 0 <= attack_id < NUM_ATTACKS else 0
        opt_feats[i] = [
            (o.number or 0) / 20.0,
            (o.count or 0) / 5.0,
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
    return (len(opp.prize) - len(me.prize)) / 6.0


def dragapult_backup_potential(obs: Observation) -> float:
    """1.0 if Dragapult ex is active and able to attack right now, AND a
    bench Drakloak already has at least one Fire or Psychic energy attached
    (a charged backup ready to take over if the active one falls) — 0.0
    otherwise.

    A pure function of state (like prize_potential), not of what was
    picked — used as potential-based shaping so the reward lands on
    whichever decision actually *reaches* this setup, not as a flat bonus
    for every decision made while it happens to still hold.
    """
    state = obs.current
    sel = obs.select
    if state is None or sel is None:
        return 0.0
    you = state.yourIndex
    me = state.players[you]
    active = me.active_pokemon
    if active is None or active.id != DRAGAPULT_EX_CARD_ID:
        return 0.0
    # Same "trust the engine's own legality check" trick as elsewhere:
    # able to attack right now iff an attack option is actually offered.
    can_attack = any(o.type == OPT_ATTACK for o in sel.option)
    if not can_attack:
        return 0.0
    has_charged_drakloak = any(
        p is not None
        and p.id == DRAKLOAK_CARD_ID
        and any(e in (ENERGY_TYPE_FIRE, ENERGY_TYPE_PSYCHIC) for e in p.energies)
        for p in me.bench
    )
    return 1.0 if has_charged_drakloak else 0.0


def _active_energy_already_sufficient(obs: Observation) -> bool:
    """True if every one of the active Pokemon's attacks, plus retreat, is
    already offered as an option — i.e. more energy on it can't unlock
    anything new right now.

    Checking option *availability* rather than computing energy costs by
    hand means this automatically accounts for anything that changes what's
    needed (e.g. an effect that raises attack costs), since the engine has
    already factored that into what it offers.
    """
    state = obs.current
    sel = obs.select
    if state is None or sel is None:
        return False
    active = state.players[state.yourIndex].active_pokemon
    if active is None:
        return False
    card = get_card_by_id(active.id)
    if card is None or not card.attacks:
        return False
    required = {a.attack_id for a in card.attacks}
    offered = {
        o.attackId
        for o in sel.option
        if o.type == OPT_ATTACK and o.attackId is not None
    }
    can_retreat = any(o.type == OPT_RETREAT for o in sel.option)
    return required <= offered and can_retreat


def energy_overattach_penalty(obs: Observation, picks: list[int]) -> float:
    """-1.0 if `picks` attach an energy card to the active Pokemon despite it
    already being able to use every attack and retreat — that energy can't
    unlock anything, it's wasted. 0.0 otherwise.
    """
    sel = obs.select
    state = obs.current
    if sel is None or state is None:
        return 0.0
    if not _active_energy_already_sufficient(obs):
        return 0.0
    you = state.yourIndex
    active = state.players[you].active_pokemon
    if active is None:
        return 0.0
    for i in picks:
        if i < 0 or i >= len(sel.option):
            continue
        opt = sel.option[i]
        if opt.type != OPT_ATTACH:
            continue
        target = _pokemon_at(state, you, opt.inPlayArea, opt.inPlayIndex)
        if target is None or target.serial != active.serial:
            continue
        card_id = _card_id_at(state, sel, you, opt.area, opt.index)
        card = get_card_by_id(card_id) if card_id else None
        if card is not None and card.card_type in (
            CARD_TYPE_BASIC_ENERGY,
            CARD_TYPE_SPECIAL_ENERGY,
        ):
            return -1.0
    return 0.0


def budew_first_turn_attack_bonus(obs: Observation, picks: list[int]) -> float:
    """+1.0 if, going second, on your own first turn (the engine's turn
    counter is shared across both players, so that's turn 2, not turn 1),
    you attack with Budew as your active Pokemon. Budew's attack costs no
    energy, so this is purely a "did you take the free early disruption"
    check — no energy-sufficiency gating needed, unlike the penalty above.
    0.0 otherwise.
    """
    state = obs.current
    sel = obs.select
    if state is None or sel is None:
        return 0.0
    went_second = state.firstPlayer >= 0 and state.firstPlayer != state.yourIndex
    if not went_second or state.turn != 2:
        return 0.0
    active = state.players[state.yourIndex].active_pokemon
    if active is None or active.id != BUDEW_CARD_ID:
        return 0.0
    attacked = any(
        sel.option[i].type == OPT_ATTACK for i in picks if 0 <= i < len(sel.option)
    )
    return 1.0 if attacked else 0.0


def budew_active_second_potential(obs: Observation) -> float:
    """1.0 if, going second and still early (turn <= 2 -- before or on your
    own first turn), Budew is your active Pokemon -- 0.0 otherwise.

    A pure function of state, not of what was picked: potential-based
    shaping so the reward lands on whichever decision actually puts Budew
    into the active spot (setup, or a turn-1 switch), not as a flat bonus
    paid every step it happens to still be there. Sets up the free Itchy
    Pollen attack that budew_first_turn_attack_bonus rewards taking.
    """
    state = obs.current
    if state is None:
        return 0.0
    went_second = state.firstPlayer >= 0 and state.firstPlayer != state.yourIndex
    if not went_second or state.turn > 2:
        return 0.0
    active = state.players[state.yourIndex].active_pokemon
    return 1.0 if active is not None and active.id == BUDEW_CARD_ID else 0.0


def xerosic_machinations_bonus(obs: Observation, picks: list[int]) -> float:
    """+1.0 if `picks` play Xerosic's Machinations (discards the opponent's
    hand down to 3 cards) while they have 7+ cards -- a big swing. -3.0 if
    played while they have 4 or fewer -- it does almost nothing (or, at 3
    or below, literally nothing) and burns your Supporter for the turn, so
    that's a much worse mistake than the upside is a win. 0.0 otherwise
    (including the 5-6 card dead zone, and any turn it isn't played).
    """
    sel = obs.select
    state = obs.current
    if sel is None or state is None:
        return 0.0
    you = state.yourIndex
    opp_hand_count = state.players[1 - you].handCount
    for i in picks:
        if i < 0 or i >= len(sel.option):
            continue
        opt = sel.option[i]
        if opt.type != OPT_PLAY:
            continue
        card_id = _card_id_at(state, sel, you, AREA_HAND, opt.index)
        if card_id != XEROSIC_MACHINATIONS_CARD_ID:
            continue
        if opp_hand_count >= 7:
            return 1.0
        if opp_hand_count <= 4:
            return -3.0
    return 0.0


def _resolve_energy_attach(
    obs: Observation, opt: Option
) -> tuple[PokemonRef, CardData] | None:
    """If `opt` attaches an energy card to an in-play Pokemon, returns
    (target, card); otherwise None. Shared by the two checks below that both
    need to know "which Pokemon is this energy landing on, and which card."
    """
    if opt.type != OPT_ATTACH:
        return None
    state = obs.current
    sel = obs.select
    if state is None or sel is None:
        return None
    you = state.yourIndex
    target = _pokemon_at(state, you, opt.inPlayArea, opt.inPlayIndex)
    if target is None:
        return None
    card_id = _card_id_at(state, sel, you, opt.area, opt.index)
    card = get_card_by_id(card_id) if card_id else None
    if card is None or card.card_type not in (
        CARD_TYPE_BASIC_ENERGY,
        CARD_TYPE_SPECIAL_ENERGY,
    ):
        return None
    return target, card


def wrong_type_energy_penalty(obs: Observation, picks: list[int]) -> float:
    """-1.0 if `picks` attach energy to a Dreepy/Drakloak/Dragapult ex that
    will end up with exactly 2 energy after this attach, both the same
    type. Phantom Dive costs one Fire + one Psychic — a same-type pair at
    2 energy can never pay for it, so it's the wrong energy to attach here.
    """
    sel = obs.select
    if sel is None:
        return 0.0
    for i in picks:
        if i < 0 or i >= len(sel.option):
            continue
        resolved = _resolve_energy_attach(obs, sel.option[i])
        if resolved is None:
            continue
        target, card = resolved
        if target.id not in DREEPY_LINE_CARD_IDS or len(target.energies) != 1:
            continue  # only the 1 -> 2 transition matters
        if card.energy_type == target.energies[0]:
            return -1.0
    return 0.0


def dragapult_ex_attack_bonus(obs: Observation, picks: list[int]) -> float:
    """+1.0 if `picks` attack while Dragapult ex is the active Pokemon —
    encourages actually pulling the trigger once it's set up, rather than
    passively holding back."""
    sel = obs.select
    state = obs.current
    if sel is None or state is None:
        return 0.0
    active = state.players[state.yourIndex].active_pokemon
    if active is None or active.id != DRAGAPULT_EX_CARD_ID:
        return 0.0
    attacked = any(
        sel.option[i].type == OPT_ATTACK for i in picks if 0 <= i < len(sel.option)
    )
    return 1.0 if attacked else 0.0


def dreepy_energy_spread_penalty(obs: Observation, picks: list[int]) -> float:
    """-1.0 if `picks` attach energy to a Dreepy that already has some,
    while another Dreepy on the board has none — spreading the one
    attachment-per-turn across more Dreepy lines beats stacking one, since
    it's better to end up with two 1-energy Dreepy than one with 2 energy
    and another sitting empty.
    """
    sel = obs.select
    state = obs.current
    if sel is None or state is None:
        return 0.0
    you = state.yourIndex
    me = state.players[you]
    for i in picks:
        if i < 0 or i >= len(sel.option):
            continue
        resolved = _resolve_energy_attach(obs, sel.option[i])
        if resolved is None:
            continue
        target, _card = resolved
        if target.id != DREEPY_CARD_ID or not target.energies:
            continue
        board = [me.active_pokemon, *me.bench]
        has_empty_sibling = any(
            p is not None
            and p.id == DREEPY_CARD_ID
            and p.serial != target.serial
            and not p.energies
            for p in board
        )
        if has_empty_sibling:
            return -1.0
    return 0.0
