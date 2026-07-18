"""End-of-first-turn heuristic evaluation for the first-turn agent.

The MCTS in ``search.py`` simulates full sequences of our first turn and
scores the resulting position with :func:`evaluate`. The user-specified
first-turn strategy is encoded here as *outcome* weights rather than
procedural steps: the engine simulation already plays out what each item /
supporter actually does (Poffin fetches, Judge redraws, Crispin attaches...),
so rewarding the outcomes those plays are *for* — dreepy on the bench,
charged bench attackers, Budew attacking with Itchy Pollen, a hand that
guarantees a good second turn — makes the search discover the same
priorities (Poffin > Poke Pad > Ultra Ball etc.) as the written strategy,
while conditional nudges (Ultra Ball conservation, Xerosic's hand-size
condition) cover the rules that pure outcomes can't see.

All weights are module constants so they can be tuned in one place.
"""

from .cards import (
    BUDEW,
    DREEPY,
    DREEPY_LINE,
    DRAKLOAK,
    ENERGY_CARDS,
    ENERGY_DARK,
    ENERGY_FIRE,
    ENERGY_PSYCHIC,
    JUDGE,
    LILLIES_DETERMINATION,
    MEOWTH_EX,
    MUNKIDORI,
    POKE_PAD,
    ULTRA_BALL,
    count_in_play,
    hand_ids,
    start_priority_rank,
)

# --- weights (importance descending encodes the strategy's priorities) ---
W_ITCHY_POLLEN = 40.0  # went second: attacked with Budew (the headline goal)
W_DEAD_END = -35.0  # ended the turn in a dead-end hand
W_DREEPY = 30.0  # per dreepy-line member in play, up to 3
W_LONE_ACTIVE = -30.0  # lone Budew/Dreepy active with an empty bench
W_GOOD_SECOND_TURN = 25.0
W_OVERCROWD = -20.0  # a 4th dreepy-line member in play
W_XEROSIC_BAD = -15.0  # Xerosic's played with opponent hand < 6
W_BENCH_CHARGE = 12.0  # benched dreepy-liner charged fire/psychic-only
W_UB_WASTE = -10.0  # Ultra Ball used while a dreepy was already in play
W_BUDEW_ACTIVE_END = 10.0  # went second: Budew active at end of turn
W_ACTIVE_CHARGE = 8.0  # active dreepy-liner charged fire/psychic
W_XEROSIC_GOOD = 8.0  # Xerosic's played with opponent hand >= 6
W_MUNKI_CHARGE = 6.0  # active Munkidori charged dark/psychic
ACTIVE_PRIORITY_BONUS = [10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 0.0]
W_HAND_CARD = 1.5  # per card kept in hand (resource conservation)


def charged_ok(energies: list[int]) -> bool:
    """Energy set the strategy wants on a dreepy-liner: exactly one fire,
    one psychic, or one of each (Phantom Dive's cost) — nothing else."""
    if not energies or len(energies) > 2:
        return False
    if any(e not in (ENERGY_FIRE, ENERGY_PSYCHIC) for e in energies):
        return False
    return energies.count(ENERGY_FIRE) <= 1 and energies.count(ENERGY_PSYCHIC) <= 1


def good_second_turn(me: dict, went_first: bool) -> tuple[bool, bool]:
    """The strategy's <GOOD SECOND TURN> / dead-end evaluation.

    Returns (good, dead_end), judged on our own final-turn observation
    (hand fully visible).
    """
    hand = hand_ids(me)
    n = len(hand)
    dreepy_in_play = count_in_play(me, {DREEPY})
    drakloak_access = DRAKLOAK in hand or ULTRA_BALL in hand or POKE_PAD in hand
    draw_out = LILLIES_DETERMINATION in hand or JUDGE in hand
    meowth_route = MEOWTH_EX in hand or count_in_play(me, {MEOWTH_EX}) > 0
    can_evolve = dreepy_in_play > 0 and (drakloak_access or draw_out or meowth_route)
    good = can_evolve or (not went_first and LILLIES_DETERMINATION in hand)
    dead_end = n == 0 or (n <= 2 and not can_evolve and not draw_out)
    return good and not dead_end, dead_end


def can_itchy_pollen_setup(hand: list[int], active_id: int | None) -> bool:
    """Setup-time <CAN ITCHY POLLEN> estimate: Budew will be active, or the
    active can be retreated into Budew (energy in hand to attach, or Crispin
    to fetch one). Used only for the bench-placement setup decision — the
    real turn's retreat route is found by the search itself."""
    from .cards import CRISPIN

    if active_id == BUDEW:
        return True
    if BUDEW not in hand:
        return False
    return any(c in ENERGY_CARDS for c in hand) or CRISPIN in hand


def evaluate(final_obs: dict, events: dict, went_first: bool) -> float:
    """Score the end of our first turn (higher is better).

    ``final_obs`` is the last our-perspective observation of the turn (the
    state on which the turn-ending action was taken). ``events`` is the
    action log accumulated by the search: attacked / itchy_pollen /
    ultra_balls + dreepy_at_ub / xerosic_opp_hand.
    """
    state = final_obs["current"]
    me = state["players"][state["yourIndex"]]
    score = 0.0

    line_count = count_in_play(me, DREEPY_LINE)
    score += W_DREEPY * min(line_count, 3)
    if line_count >= 4:
        score += W_OVERCROWD

    active = (me.get("active") or [None])[0]
    bench = [p for p in (me.get("bench") or []) if p]
    if active is not None:
        score += ACTIVE_PRIORITY_BONUS[
            min(
                start_priority_rank(active["id"], went_first),
                len(ACTIVE_PRIORITY_BONUS) - 1,
            )
        ]
        if active["id"] in (BUDEW, DREEPY) and not bench:
            score += W_LONE_ACTIVE

    # energy placement per the attach priorities: charged bench dreepy-liners
    # beat a charged active one, Munkidori dark/psychic is the fallback
    for p in bench:
        if p["id"] in DREEPY_LINE and charged_ok(p.get("energies") or []):
            score += W_BENCH_CHARGE
    if active is not None:
        en = active.get("energies") or []
        if active["id"] in DREEPY_LINE and charged_ok(en):
            score += W_ACTIVE_CHARGE
        if (
            active["id"] == MUNKIDORI
            and en
            and all(e in (ENERGY_DARK, ENERGY_PSYCHIC) for e in en)
        ):
            score += W_MUNKI_CHARGE

    good, dead_end = good_second_turn(me, went_first)
    if good:
        score += W_GOOD_SECOND_TURN
    if dead_end:
        score += W_DEAD_END

    score += W_HAND_CARD * len(hand_ids(me))

    if not went_first:
        if events.get("itchy_pollen"):
            score += W_ITCHY_POLLEN
        elif active is not None and active["id"] == BUDEW:
            score += W_BUDEW_ACTIVE_END

    # Ultra Ball conservation: the strategy only spends it when no dreepy is
    # in play (or, going second, to put Budew on the bench for Itchy Pollen)
    ub_used = events.get("ultra_balls", 0)
    if ub_used and events.get("dreepy_at_ub", 0) >= 1:
        budew_fetched = not went_first and any(p["id"] == BUDEW for p in bench)
        if not budew_fetched:
            score += W_UB_WASTE * ub_used

    if "xerosic_opp_hand" in events:
        score += W_XEROSIC_GOOD if events["xerosic_opp_hand"] >= 6 else W_XEROSIC_BAD

    return score
