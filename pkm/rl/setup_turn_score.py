"""End-of-turn scoring for the Dragapult **setup** agent.

The setup agent is trained on a 2-turn episode, not a full match: the
first-turn agent plays turn 1 for both sides, the setup agent plays turn 2 for
both sides, and then this module scores the board the setup agent left behind
(after its attack). That terminal score is the dominant training signal; the
default agent's usual shaping terms ride along underneath at reduced weight.

**What a good turn-2 board looks like, and why.** `deck/03_pult_munki.csv`
carries no Rare Candy, so the only route to a Dragapult ex is
Dreepy -> Drakloak -> Dragapult ex, one evolution per turn. That makes a
**benched Drakloak the hard prerequisite** for attacking on turn 3: no
Drakloak in play at the end of turn 2 means no Phantom Dive on turn 3, full
stop. So the target state is:

  * bench   -- Drakloaks (carrying Fire/Psychic, which survives evolution),
               backed by Dreepies
  * active  -- a cheap, disposable staller (Budew, Munkidori); *not* a
               dreepy-line piece and *not* an ex, both of which hand the
               opponent tempo or a two-prize knockout
  * hand    -- a Dragapult ex to evolve into, plus energy
  * enemy   -- item-locked by Itchy Pollen, ideally damaged

Every term below is a module-level constant so the whole rubric can be tuned
in one place. Terms marked ``(spec)`` are the user-supplied rubric; those
marked ``(added)`` are proposed extensions -- set their weight to 0.0 to
disable one without touching the code.

Pure functions of the observation: no engine calls, no card database, so this
is cheap enough to call on every episode and importable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pkm.types.obs import EnergyType, Observation, Player, PokemonRef

# --- the cards this rubric reasons about -----------------------------------
DREEPY_CARD_ID = 119
DRAKLOAK_CARD_ID = 120
DRAGAPULT_EX_CARD_ID = 121
BUDEW_CARD_ID = 235
MUNKIDORI_CARD_ID = 112
FEZANDIPITI_EX_CARD_ID = 140
MEOWTH_EX_CARD_ID = 1071

DREEPY_LINE_CARD_IDS = {DREEPY_CARD_ID, DRAKLOAK_CARD_ID, DRAGAPULT_EX_CARD_ID}
# The bodies this deck wants holding the active spot while the line builds:
# cheap, one-prize, and disruptive rather than valuable.
STALLER_CARD_IDS = {BUDEW_CARD_ID, MUNKIDORI_CARD_ID}
# Two prizes if knocked out -- a liability in the active spot on turn 2.
EX_CARD_IDS = {DRAGAPULT_EX_CARD_ID, FEZANDIPITI_EX_CARD_ID, MEOWTH_EX_CARD_ID}
# Phantom Dive's cost. Only these two types advance the line; a Dark energy
# belongs on Munkidori.
COMBO_ENERGY_TYPES = {EnergyType.FIRE, EnergyType.PSYCHIC}

STARTING_PRIZE_COUNT = 6

# --- weights: the whole rubric, tunable in one place ------------------------

# (spec) Itchy Pollen locks the opponent out of items for a turn -- the single
# highest-tempo thing turn 2 can do while the line is still assembling.
W_ITCHY_POLLEN = 5.0

# (spec) Per prize taken this turn.
W_KNOCKOUT = 3.0

# (spec) A benched Drakloak, keyed by how many *useful* (Fire/Psychic)
# energies it holds. NOTE the ordering is deliberately non-monotonic: one
# energy scores above two, which spreads investment across several Drakloaks
# rather than loading one up. See the module notes in the review for the
# monotonic alternative.
W_BENCH_DRAKLOAK_BY_ENERGY: dict[int, float] = {0: 3.0, 1: 5.0, 2: 4.0}

# (spec) A benched Dreepy -- next turn's Drakloak.
W_BENCH_DREEPY = 1.0

# (spec) Cards in hand: 2 each up to the first six, 1 each beyond.
W_HAND_CARD = 2.0
W_HAND_CARD_BEYOND = 1.0
HAND_SIZE_KNEE = 6

# (spec) A Dragapult ex in hand is what a benched Drakloak evolves into.
W_DRAGAPULT_IN_HAND = 1.0

# (added) The actual next-turn payoff: a bench Drakloak that can attack the
# turn after this one, paired with a Dragapult ex in hand to evolve it into.
# Scored per matched pair, since one Dragapult ex evolves one Drakloak. This
# is the term that rewards *completing* a combo, and so partly offsets the
# 1-over-2 energy ordering above.
#
# Two ways to be ready, because a Pokemon gets one attachment per turn:
#   READY      -- already holds Fire+Psychic; evolve and attack, no attachment
#                 spent, and nothing left to draw or hold onto.
#   READY_NEXT -- holds one of the two, with the *other type in hand*; evolve,
#                 attach it, attack. Worth less: it consumes the turn's only
#                 attachment and depends on still having that card.
W_PHANTOM_DIVE_READY = 6.0
W_PHANTOM_DIVE_READY_NEXT = 4.0

# (added) Who is holding the active spot when the turn ends.
W_ACTIVE_STALLER = 3.0  # Budew / Munkidori: exactly the intended body
W_ACTIVE_DREEPY_LINE = -4.0  # a combo piece exposed to a knockout
W_ACTIVE_EX = -3.0  # two prizes on the line

# (added) Energy on a benched Dreepy still counts -- it survives evolution
# into Drakloak, so it is early progress rather than waste.
W_BENCH_DREEPY_ENERGY = 1.0

# (added) Fire/Psychic sitting on something outside the dreepy line is
# stranded: it can never pay for a Phantom Dive.
W_STRANDED_COMBO_ENERGY = -1.5

# (added) A third energy on a line member is wasted -- Phantom Dive needs two.
W_OVERCHARGED_LINE = -2.0

# (added) The deck expects at most three of the line in play; a fourth is
# bench space that a staller or Fezandipiti ex would use better. Mirrors
# `dreepy_line_field_potential`'s ceiling in pkm/rl/encoder.py.
MAX_USEFUL_LINE_COUNT = 3
W_LINE_OVERCOMMIT = -2.0

# (added) Chip damage left on the opponent's board. Over a 2-turn episode all
# of it is ours, so this needs no before/after comparison.
W_DAMAGE_DEALT_PER_10HP = 0.3

# (added) Energy in hand is next turn's attachment.
W_ENERGY_IN_HAND = 0.5

# PPO terminal rewards are conventionally O(1) while this rubric totals tens of
# points. Divide by this before it reaches the trainer so it stays comparable
# to the win/loss signal rather than swamping it.
SCORE_SCALE = 20.0


@dataclass
class TurnScore:
    """A scored board, plus the per-term breakdown that produced it.

    The breakdown is the point: a bare total is untunable, whereas the
    contributions make it obvious which term is driving a decision.
    """

    total: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        if value:
            self.parts[name] = self.parts.get(name, 0.0) + value
            self.total += value

    @property
    def normalized(self) -> float:
        """`total` rescaled toward the O(1) range PPO expects."""
        return self.total / SCORE_SCALE

    def report(self) -> str:
        lines = [f"{name:<26}{value:+7.2f}" for name, value in self.parts.items()]
        lines.append(
            f"{'TOTAL':<26}{self.total:+7.2f}  (scaled {self.normalized:+.3f})"
        )
        return "\n".join(lines)


# --- helpers ---------------------------------------------------------------


def _bench(player: Player) -> list[PokemonRef]:
    return [p for p in player.bench if p is not None]


def _in_play(player: Player) -> list[PokemonRef]:
    active = player.active_pokemon
    return ([active] if active else []) + _bench(player)


def _combo_energy_count(poke: PokemonRef) -> int:
    """How many Fire/Psychic energies this Pokemon holds."""
    return sum(1 for e in poke.energies if e in COMBO_ENERGY_TYPES)


def _hand_ids(player: Player) -> list[int]:
    return [c.id for c in (player.hand or []) if c is not None]


def _hand_size(player: Player) -> int:
    """Always `handCount`, never `len(hand)`.

    `handCount` is visible from either point of view and is always current,
    whereas the `hand` list is hidden for the non-acting player -- and where
    the caller has spliced one in (see the trainer's `_with_own_hand`) it may
    be a decision stale. Size and contents therefore come from different
    places on purpose.
    """
    return player.handCount


# The basic energy this deck runs, card id -> the type it provides. Spelled
# out rather than inferred: the card ids happen to equal their EnergyType
# values, which is a coincidence of the card database, not a rule.
BASIC_ENERGY_CARD_TYPES: dict[int, EnergyType] = {
    2: EnergyType.FIRE,
    5: EnergyType.PSYCHIC,
    7: EnergyType.DARKNESS,
}


def _is_energy_card_id(card_id: int) -> bool:
    return card_id in BASIC_ENERGY_CARD_TYPES


def _hand_energy_types(player: Player) -> set[EnergyType]:
    """Which basic energy types are sitting in hand, ready to attach."""
    return {
        BASIC_ENERGY_CARD_TYPES[cid]
        for cid in _hand_ids(player)
        if cid in BASIC_ENERGY_CARD_TYPES
    }


def _attachments_to_phantom_dive(
    poke: PokemonRef, hand_types: set[EnergyType]
) -> int | None:
    """Attachments still needed before `poke` can pay Fire+Psychic.

    0 = already there, 1 = one type missing and that type is in hand, None =
    not reachable next turn. There is no "2": a Pokemon gets a single
    attachment per turn, so anything needing two is at least two turns away.
    """
    missing = COMBO_ENERGY_TYPES - set(poke.energies)
    if not missing:
        return 0
    if len(missing) == 1 and next(iter(missing)) in hand_types:
        return 1
    return None


# --- the rubric ------------------------------------------------------------


def score_end_of_turn(
    obs: Observation,
    itchy_pollen_used: bool = False,
    prizes_taken: int | None = None,
    seat: int | None = None,
) -> TurnScore:
    """Score the board the setup agent leaves at the end of its turn.

    `obs` is the observation *after* the turn's attack resolves. `seat` names
    whose board to score, defaulting to the observation's own point of view.
    Pass it explicitly when scoring a seat other than the one to move -- the
    2-turn trainer scores the first player's board from an observation taken
    once play has already passed to the second player.

    `itchy_pollen_used` must be supplied by the caller: `Log` entries carry
    only a `type`, with no payload naming the attack, so which attack was used
    is not recoverable from the observation alone. The training loop knows it,
    because it chose the option.

    `prizes_taken` likewise defaults to inferring from the prize pile, which is
    only sound because this is a 2-turn episode starting from a full six.
    Pass it explicitly for any other setting.
    """
    score = TurnScore()
    state = obs.current
    if state is None:
        return score

    if seat is None:
        seat = state.yourIndex
    me = state.players[seat]
    opponent = state.players[1 - seat]

    _score_tempo(score, me, itchy_pollen_used, prizes_taken)
    _score_bench(score, me)
    _score_active(score, me)
    _score_hand(score, me)
    _score_readiness(score, me)
    _score_energy_discipline(score, me)
    _score_opponent(score, opponent)
    return score


def _score_tempo(
    score: TurnScore,
    me: Player,
    itchy_pollen_used: bool,
    prizes_taken: int | None,
) -> None:
    if itchy_pollen_used:
        score.add("itchy_pollen", W_ITCHY_POLLEN)
    if prizes_taken is None:
        prizes_taken = max(0, STARTING_PRIZE_COUNT - len(me.prize))
    if prizes_taken:
        score.add("knockouts", W_KNOCKOUT * prizes_taken)


def _score_bench(score: TurnScore, me: Player) -> None:
    """The bench is where this deck's turn 2 actually happens."""
    for poke in _bench(me):
        if poke.id == DRAKLOAK_CARD_ID:
            tier = min(_combo_energy_count(poke), 2)
            score.add("bench_drakloak", W_BENCH_DRAKLOAK_BY_ENERGY.get(tier, 0.0))
        elif poke.id == DREEPY_CARD_ID:
            score.add("bench_dreepy", W_BENCH_DREEPY)
            score.add(
                "bench_dreepy_energy",
                W_BENCH_DREEPY_ENERGY * _combo_energy_count(poke),
            )

    line_count = sum(1 for p in _in_play(me) if p.id in DREEPY_LINE_CARD_IDS)
    if line_count > MAX_USEFUL_LINE_COUNT:
        score.add(
            "line_overcommit",
            W_LINE_OVERCOMMIT * (line_count - MAX_USEFUL_LINE_COUNT),
        )


def _score_active(score: TurnScore, me: Player) -> None:
    """Who is left holding the active spot when the opponent's turn begins."""
    active = me.active_pokemon
    if active is None:
        return
    if active.id in STALLER_CARD_IDS:
        score.add("active_staller", W_ACTIVE_STALLER)
    if active.id in DREEPY_LINE_CARD_IDS:
        score.add("active_dreepy_line", W_ACTIVE_DREEPY_LINE)
    if active.id in EX_CARD_IDS:
        score.add("active_ex", W_ACTIVE_EX)


def _score_hand(score: TurnScore, me: Player) -> None:
    size = _hand_size(me)
    base = min(size, HAND_SIZE_KNEE) * W_HAND_CARD
    beyond = max(0, size - HAND_SIZE_KNEE) * W_HAND_CARD_BEYOND
    score.add("hand_size", base + beyond)

    hand = _hand_ids(me)
    pults = sum(1 for cid in hand if cid == DRAGAPULT_EX_CARD_ID)
    score.add("dragapult_in_hand", W_DRAGAPULT_IN_HAND * pults)
    energy = sum(1 for cid in hand if _is_energy_card_id(cid))
    score.add("energy_in_hand", W_ENERGY_IN_HAND * energy)


def _score_readiness(score: TurnScore, me: Player) -> None:
    """Bench Drakloaks that can actually attack the turn after this one.

    Scored as matched pairs: a Drakloak is only a Phantom Dive if there is a
    Dragapult ex in hand to evolve it into, and one Dragapult ex evolves one
    Drakloak.

    Two tiers, because a Pokemon gets one attachment per turn. A Drakloak
    already holding Fire+Psychic needs none. A Drakloak holding one of them,
    with the other type in hand, needs exactly one -- also ready, but it
    spends the turn's only attachment, so only **one** such Drakloak can be
    finished no matter how many are waiting.
    """
    hand = _hand_ids(me)
    pults = sum(1 for cid in hand if cid == DRAGAPULT_EX_CARD_ID)
    hand_types = _hand_energy_types(me)

    charged = 0
    one_away = 0
    for poke in _bench(me):
        if poke.id != DRAKLOAK_CARD_ID:
            continue
        needed = _attachments_to_phantom_dive(poke, hand_types)
        if needed == 0:
            charged += 1
        elif needed == 1:
            one_away += 1

    ready_now = min(charged, pults)
    # Capped at 1: the turn has a single energy attachment to spend, however
    # many one-away Drakloaks are sitting there.
    ready_next = min(one_away, max(0, pults - ready_now), 1)

    score.add("phantom_dive_ready", W_PHANTOM_DIVE_READY * ready_now)
    score.add("phantom_dive_ready_next", W_PHANTOM_DIVE_READY_NEXT * ready_next)


def _score_energy_discipline(score: TurnScore, me: Player) -> None:
    """Fire/Psychic is a scarce resource -- it should all be on the line."""
    stranded = 0
    overcharge = 0
    for poke in _in_play(me):
        useful = _combo_energy_count(poke)
        if poke.id in DREEPY_LINE_CARD_IDS:
            overcharge += max(0, useful - 2)
        else:
            stranded += useful
    score.add("stranded_energy", W_STRANDED_COMBO_ENERGY * stranded)
    score.add("overcharged_line", W_OVERCHARGED_LINE * overcharge)


def _score_opponent(score: TurnScore, opponent: Player) -> None:
    damage = sum(max(0, p.maxHp - p.hp) for p in _in_play(opponent))
    score.add("damage_dealt", W_DAMAGE_DEALT_PER_10HP * (damage / 10.0))


if __name__ == "__main__":  # pragma: no cover - hand-run tuning aid
    from pkm.types.obs import Observation as _Obs

    _serial = 0

    def _poke(card_id: int, energies=(), hp: int = 70, max_hp: int = 70) -> dict:
        global _serial
        _serial += 1
        return {
            "id": card_id,
            "playerIndex": 0,
            "serial": _serial,
            "hp": hp,
            "maxHp": max_hp,
            "energies": list(energies),
        }

    def _state(active, bench, hand, opp_active, prizes: int = 6) -> _Obs:
        me = {
            "active": [active] if active else [],
            "bench": list(bench),
            "hand": [
                {"id": c, "playerIndex": 0, "serial": 900 + i}
                for i, c in enumerate(hand)
            ],
            "handCount": len(hand),
            "discard": [],
            "prize": [None] * prizes,
            "deckCount": 30,
        }
        opp = {
            "active": [opp_active] if opp_active else [],
            "bench": [],
            "hand": None,
            "handCount": 5,
            "discard": [],
            "prize": [None] * 6,
            "deckCount": 30,
        }
        return _Obs.model_validate(
            {
                "current": {
                    "yourIndex": 0,
                    "players": [me, opp],
                    "result": -1,
                    "turn": 2,
                    "firstPlayer": 0,
                },
                "select": None,
            }
        )

    FIRE, PSY = int(EnergyType.FIRE), int(EnergyType.PSYCHIC)

    print("=== ideal turn-2 board ===")
    ideal = _state(
        active=_poke(BUDEW_CARD_ID),
        bench=[
            _poke(DRAKLOAK_CARD_ID, [FIRE, PSY]),
            _poke(DRAKLOAK_CARD_ID, [FIRE]),
            _poke(DREEPY_CARD_ID, [PSY]),
        ],
        hand=[DRAGAPULT_EX_CARD_ID, 2, 1121, 1086, 1152],
        opp_active=_poke(112, hp=40, max_hp=70),
    )
    print(score_end_of_turn(ideal, itchy_pollen_used=True).report())

    print("\n=== bad turn-2 board (line exposed, energy stranded) ===")
    bad = _state(
        active=_poke(DREEPY_CARD_ID, [FIRE]),
        bench=[_poke(MUNKIDORI_CARD_ID, [PSY, PSY])],
        hand=[1121, 1152],
        opp_active=_poke(112),
    )
    print(score_end_of_turn(bad).report())
