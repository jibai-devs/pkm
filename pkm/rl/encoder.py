"""Observation and option encoding: obs dicts -> numpy arrays for the network.

The state is encoded as card-ID slots (embedded by the model) plus float
features. Options are encoded per-option as (type, card, target card, attack)
IDs plus float features, so the model can score a variable-length option list.
"""

from dataclasses import dataclass, field

import numpy as np

from pkm.data import get_attack_data
from pkm.data.card_data import Attack, CardData, get_card_by_id
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
    Option,
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

# CardType values (see obs_data_structure/OBSERVATION_SCHEMA.md)
CARD_TYPE_ITEM = 1
CARD_TYPE_SUPPORTER = 3
CARD_TYPE_BASIC_ENERGY = 5
CARD_TYPE_SPECIAL_ENERGY = 6

BUDEW_CARD_ID = 235
DREEPY_CARD_ID = 119
DRAKLOAK_CARD_ID = 120
DRAGAPULT_EX_CARD_ID = 121
DREEPY_LINE_CARD_IDS = {DREEPY_CARD_ID, DRAKLOAK_CARD_ID, DRAGAPULT_EX_CARD_ID}
XEROSIC_MACHINATIONS_CARD_ID = 1197
PHANTOM_DIVE_ATTACK_ID = 154

# EnergyType values (see pkm/types/obs.py's EnergyType enum)
ENERGY_TYPE_COLORLESS = 0
ENERGY_TYPE_FIRE = 2
ENERGY_TYPE_PSYCHIC = 5


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
    board_setup_potential: float = 0.0
    budew_setup_potential: float = 0.0
    dreepy_line_field_potential: float = 0.0
    energy_penalty: float = 0.0
    budew_bonus: float = 0.0
    wrong_type_energy_penalty: float = 0.0
    dragapult_attack_bonus: float = 0.0
    dreepy_spread_penalty: float = 0.0
    xerosic_bonus: float = 0.0
    budew_bench_setup_bonus: float = 0.0
    dreepy_evolve_bonus: float = 0.0
    dreepy_bench_charge_bonus: float = 0.0
    dreepy_active_charge_bonus: float = 0.0
    wasted_resources_penalty: float = 0.0
    phantom_dive_bonus: float = 0.0
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
    can_attack = any(o.type == OptionType.ATTACK for o in sel.option)
    if not can_attack:
        return 0.0
    has_charged_drakloak = any(
        p is not None
        and p.id == DRAKLOAK_CARD_ID
        and any(e in (ENERGY_TYPE_FIRE, ENERGY_TYPE_PSYCHIC) for e in p.energies)
        for p in me.bench
    )
    return 1.0 if has_charged_drakloak else 0.0


def dreepy_line_field_potential(obs: Observation) -> float:
    """Ramps up from 0.0 to 1.0 as the number of Dreepy/Drakloak/Dragapult
    ex you have in play (active + bench, combined) goes from 0 to 3, then
    drops to -1.0 at 4 or more.

    A pure function of state, not of what was picked -- potential-based
    shaping so the reward lands on whichever action actually grows the
    line (mainly playing a new Dreepy from hand; evolving doesn't change
    the count), and the drop lands on whichever action pushes past 3,
    rather than paying out every step a 4th happens to still be sitting
    there. 3 copies is the deck's expected ceiling for this line -- a 4th
    is bench space that would serve the deck better elsewhere.
    """
    state = obs.current
    if state is None:
        return 0.0
    me = state.players[state.yourIndex]
    board = [me.active_pokemon, *me.bench]
    count = sum(1 for p in board if p is not None and p.id in DREEPY_LINE_CARD_IDS)
    if count >= 4:
        return -1.0
    return count / 3.0


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
        if o.type == OptionType.ATTACK and o.attackId is not None
    }
    can_retreat = any(o.type == OptionType.RETREAT for o in sel.option)
    return required <= offered and can_retreat


def _attack_cost_covered(attack: Attack, energies: list[int]) -> bool:
    """True if `energies` (attached energy types) pays every requirement in
    `attack`'s cost: specific types must be matched one-for-one, and any
    Colorless slots can be paid by whatever's left over."""
    pool = list(energies)
    generic = 0
    for req in attack.energies:
        if req == ENERGY_TYPE_COLORLESS:
            generic += 1
            continue
        if req in pool:
            pool.remove(req)
        else:
            return False
    return generic <= len(pool)


def _bench_energy_already_sufficient(target: PokemonRef) -> bool:
    """Bench equivalent of `_active_energy_already_sufficient`: the engine
    never offers attack options for a benched Pokemon, so "already enough"
    has to be computed directly from the card's attack costs instead of
    read off offered options."""
    card = get_card_by_id(target.id)
    if card is None or not card.attacks:
        return False
    return all(_attack_cost_covered(a, target.energies) for a in card.attacks)


def energy_overattach_penalty(obs: Observation, picks: list[int]) -> float:
    """-1.0 if `picks` attach an energy card to a Pokemon -- active or
    benched -- that already has enough energy to pay for every attack it
    knows, so the new energy can't unlock anything. The active Pokemon is
    checked via which attacks/retreat the engine actually offers (also
    covers effects that change costs); a benched Pokemon gets no such
    options from the engine, so it's checked directly against its card's
    attack costs. 0.0 otherwise.
    """
    sel = obs.select
    state = obs.current
    if sel is None or state is None:
        return 0.0
    you = state.yourIndex
    active = state.players[you].active_pokemon
    active_ready = _active_energy_already_sufficient(obs)
    for i in picks:
        if i < 0 or i >= len(sel.option):
            continue
        resolved = _resolve_energy_attach(obs, sel.option[i])
        if resolved is None:
            continue
        target, _card = resolved
        if active is not None and target.serial == active.serial:
            if active_ready:
                return -1.0
        elif _bench_energy_already_sufficient(target):
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
        sel.option[i].type == OptionType.ATTACK for i in picks if 0 <= i < len(sel.option)
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
        if opt.type != OptionType.PLAY:
            continue
        card_id = _card_id_at(state, sel, you, AreaType.HAND, opt.index)
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
    if opt.type != OptionType.ATTACH:
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
        sel.option[i].type == OptionType.ATTACK for i in picks if 0 <= i < len(sel.option)
    )
    return 1.0 if attacked else 0.0


def phantom_dive_attack_bonus(obs: Observation, picks: list[int]) -> float:
    """+1.0 if `picks` use Dragapult ex's Phantom Dive specifically (as
    opposed to its cheap Jet Headbutt) -- attacking always ends the turn,
    so this rewards actually closing the turn out with the deck's real
    finisher once it's paid for, rather than settling for chip damage.
    0.0 otherwise.
    """
    sel = obs.select
    if sel is None:
        return 0.0
    used = any(
        sel.option[i].type == OptionType.ATTACK
        and sel.option[i].attackId == PHANTOM_DIVE_ATTACK_ID
        for i in picks
        if 0 <= i < len(sel.option)
    )
    return 1.0 if used else 0.0


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


def budew_turn_bench_setup_bonus(obs: Observation, picks: list[int]) -> float:
    """+1.0 if `picks` attach energy to a bench Dreepy/Drakloak/Dragapult ex
    during the same turn Itchy Pollen is available (going second, turn 2,
    Budew active). Budew's attack costs no energy, so this turn's one energy
    attachment is otherwise wasted sitting in hand -- there's no reason not
    to spend it developing the bench threat that'll take over later. 0.0
    otherwise.
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
    for i in picks:
        if i < 0 or i >= len(sel.option):
            continue
        resolved = _resolve_energy_attach(obs, sel.option[i])
        if resolved is None:
            continue
        target, _card = resolved
        if target.id in DREEPY_LINE_CARD_IDS:
            return 1.0
    return 0.0


def dreepy_evolve_bonus(obs: Observation, picks: list[int]) -> float:
    """+1.0 if `picks` evolve a Dreepy into Drakloak. +2.0 instead if that
    Dreepy already had energy attached (the energy carries over onto
    Drakloak, so evolving a charged Dreepy is strictly better than evolving
    an empty one). 0.0 otherwise.
    """
    state = obs.current
    sel = obs.select
    if state is None or sel is None:
        return 0.0
    you = state.yourIndex
    for i in picks:
        if i < 0 or i >= len(sel.option):
            continue
        opt = sel.option[i]
        if opt.type != OptionType.EVOLVE:
            continue
        card_id = _card_id_at(state, sel, you, opt.area, opt.index)
        if card_id != DRAKLOAK_CARD_ID:
            continue
        target = _pokemon_at(state, you, opt.inPlayArea, opt.inPlayIndex)
        if target is None or target.id != DREEPY_CARD_ID:
            continue
        return 2.0 if target.energies else 1.0
    return 0.0


def dreepy_line_bench_charge_bonus(obs: Observation, picks: list[int]) -> float:
    """+1.0 if `picks` attach energy to a bench Dreepy/Drakloak/Dragapult ex
    such that it ends up with exactly 1 Fire, 1 Psychic, or one of each --
    clean progress toward the Fire+Psychic combo their strongest attacks
    need. -1.0 if the attach instead pushes it to 3 energy total: only 2
    are ever needed, so a third is wasted (worse still on the bench, where
    it can't even attack this turn). 0.0 otherwise.
    """
    sel = obs.select
    state = obs.current
    if sel is None or state is None:
        return 0.0
    you = state.yourIndex
    active = state.players[you].active_pokemon
    for i in picks:
        if i < 0 or i >= len(sel.option):
            continue
        resolved = _resolve_energy_attach(obs, sel.option[i])
        if resolved is None:
            continue
        target, card = resolved
        if target.id not in DREEPY_LINE_CARD_IDS:
            continue
        if active is not None and target.serial == active.serial:
            continue  # bench only
        resulting = [*target.energies, card.energy_type]
        n = len(resulting)
        if n >= 3:
            return -1.0
        type_set = set(resulting)
        if len(type_set) == n and type_set <= {ENERGY_TYPE_FIRE, ENERGY_TYPE_PSYCHIC}:
            return 1.0
    return 0.0


def dreepy_line_active_charge_bonus(obs: Observation, picks: list[int]) -> float:
    """+1.0 if `picks` attach energy to your *active* Dreepy/Drakloak/
    Dragapult ex such that it now has both 1 Fire and 1 Psychic energy --
    the exact combo their strongest attacks need -- and it didn't have both
    before this attach. Only fires on the attach that actually completes
    the combo, not on every attach made after it's already complete, so it
    can't be farmed by re-attaching once the active can already attack.

    Nothing else rewards this: energy_penalty only discourages *over*-
    attaching to an already-sufficient active, and
    dreepy_line_bench_charge_bonus explicitly skips the active Pokemon --
    so correctly powering up your own attacker had no direct reward at
    all. This fills that gap. 0.0 otherwise.
    """
    sel = obs.select
    state = obs.current
    if sel is None or state is None:
        return 0.0
    you = state.yourIndex
    active = state.players[you].active_pokemon
    if active is None or active.id not in DREEPY_LINE_CARD_IDS:
        return 0.0
    for i in picks:
        if i < 0 or i >= len(sel.option):
            continue
        resolved = _resolve_energy_attach(obs, sel.option[i])
        if resolved is None:
            continue
        target, card = resolved
        if target.serial != active.serial:
            continue  # active only
        before = set(target.energies)
        after = before | {card.energy_type}
        had_combo = ENERGY_TYPE_FIRE in before and ENERGY_TYPE_PSYCHIC in before
        has_combo = ENERGY_TYPE_FIRE in after and ENERGY_TYPE_PSYCHIC in after
        if has_combo and not had_combo:
            return 1.0
    return 0.0


def wasted_resources_attack_penalty(obs: Observation, picks: list[int]) -> float:
    """-1.0 if `picks` attack while an Item or Supporter card is still
    playable (offered as a PLAY option) and you have fewer than 2
    Dreepy/Drakloak/Dragapult ex on the bench -- attacking ends your turn,
    so anything playable that's still sitting in hand afterward is wasted,
    and that matters most while the bench threat isn't developed yet. 0.0
    otherwise (including once the bench is developed enough that pressing
    the attack now is worth more than one extra card play).
    """
    sel = obs.select
    state = obs.current
    if sel is None or state is None:
        return 0.0
    you = state.yourIndex
    me = state.players[you]
    attacked = any(
        sel.option[i].type == OptionType.ATTACK for i in picks if 0 <= i < len(sel.option)
    )
    if not attacked:
        return 0.0
    bench_count = sum(
        1 for p in me.bench if p is not None and p.id in DREEPY_LINE_CARD_IDS
    )
    if bench_count >= 2:
        return 0.0
    for o in sel.option:
        if o.type != OptionType.PLAY:
            continue
        card_id = _card_id_at(state, sel, you, AreaType.HAND, o.index)
        card = get_card_by_id(card_id) if card_id else None
        if card is not None and card.card_type in (CARD_TYPE_ITEM, CARD_TYPE_SUPPORTER):
            return -1.0
    return 0.0
