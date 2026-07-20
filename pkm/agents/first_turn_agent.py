"""Agent for the first turn of the game only (deck/03_pult_munki strategy).

``singaporean_middleman`` routes here for the whole of our own first turn:
the setup decisions (turn 0) and our first playing turn — engine turn 1 when
we go first, turn 2 when we go second (the counter is shared across both
players). Handoff back to the dragapult_default agent happens on the next turn change.

Setup decisions follow the user-specified starting-priority tables directly
(they're fixed lists — nothing to search). Every in-turn decision runs the
turn-scoped Monte Carlo tree search in ``turn1agent_dep/search.py``, which
simulates whole action sequences to the end of the turn on the engine's
SearchStep API and scores the outcomes with the first-turn rubric in
``turn1agent_dep/scoring.py`` (dreepy bench setup, Itchy Pollen, GOOD SECOND
TURN, dead-end avoidance). Any search failure falls back to random legal
picks — this agent must never crash the match.
"""

import random
import sys
import traceback
from typing import Callable

from pkm.types.obs import SelectContext, forced_picks

from .turn1agent_dep.cards import (
    BUDEW,
    DREEPY,
    hand_ids,
    start_priority_rank,
)
from .turn1agent_dep.scoring import can_itchy_pollen_setup
from .turn1agent_dep.search import Turn1Search, _opt_card_id


def _random_picks(sel: dict, rng: random.Random) -> list[int]:
    """Legal-but-arbitrary fallback picks.

    Clamped to the number of options on purpose: ``rng.sample`` raises when
    k > population, and this runs *inside* an exception handler, so an
    unclamped sample would crash the very match the fallback exists to save.
    """
    n = len(sel["option"])
    lo = min(max(int(sel.get("minCount") or 0), 0), n)
    k = min(max(int(sel.get("maxCount") or 0), lo), n)
    return rng.sample(range(n), k)


def _setup_active(obs: dict, sel: dict, went_first: bool) -> list[int]:
    """Pick the starting active by the strategy's priority table."""
    ranked = sorted(
        range(len(sel["option"])),
        key=lambda i: start_priority_rank(
            _opt_card_id(obs, sel["option"][i]), went_first
        ),
    )
    return ranked[: max(sel["minCount"], 1)]


def _setup_bench(obs: dict, sel: dict, went_first: bool) -> list[int]:
    """Bench placement at setup: all dreepies; going second, Budew too when
    the Itchy Pollen route looks live; pad to minCount by priority."""
    state = obs["current"]
    me = state["players"][state["yourIndex"]]
    active = (me.get("active") or [None])[0]
    active_id = active["id"] if active else None
    hand = hand_ids(me)

    want_budew = (
        not went_first
        and active_id != BUDEW
        and can_itchy_pollen_setup(hand, active_id)
    )
    picks: list[int] = []
    for i, opt in enumerate(sel["option"]):
        cid = _opt_card_id(obs, opt)
        if cid == DREEPY or (want_budew and cid == BUDEW):
            picks.append(i)
    picks = picks[: sel["maxCount"]]
    if len(picks) < sel["minCount"]:
        rest = sorted(
            (i for i in range(len(sel["option"])) if i not in picks),
            key=lambda i: start_priority_rank(
                _opt_card_id(obs, sel["option"][i]), went_first
            ),
        )
        picks += rest[: sel["minCount"] - len(picks)]
    return picks


def make_first_turn_agent(
    deck: list[int],
    n_determinizations: int = 2,
    # 40 simulations resolved in ~21ms/decision -- 0.35% of the 6s cap, so the
    # budget was never the binding constraint, the simulation count was. And
    # depth demonstrably matters on this turn: going 6 -> 40 took the rate of
    # leaving a *charged* Dreepy on the bench from 4% to 38%. 100 costs
    # ~50ms/decision, still ~1% of the cap.
    n_simulations: int = 100,
    time_budget_s: float = 6.0,
    seed: int | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> Callable[[dict], list[int]]:
    """Create the agent that plays only our first turn of the game.

    `log_sink` receives a line whenever the search fails and the random
    fallback kicks in (``singaporean_middleman`` passes its own sink through).
    """
    rng = random.Random(seed)
    search = Turn1Search(
        n_determinizations=n_determinizations,
        n_simulations=n_simulations,
        time_budget_s=time_budget_s,
        rng=rng,
    )

    def _report(exc: Exception) -> None:
        """Never degrade to random *silently*.

        The fallback keeps the match alive, but staying quiet about it hid a
        real determinization bug for a long time: turn 1 -- the highest-leverage
        turn -- would occasionally be played by coin flip and simply look like
        the bot "sometimes doing something dumb".
        """
        msg = (
            f"first_turn: search failed ({type(exc).__name__}: {exc}) "
            "-- falling back to RANDOM picks"
        )
        if log_sink is not None:
            log_sink(msg)
        else:
            print(msg, file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    def agent(obs: dict) -> list[int]:
        sel = obs["select"]
        if sel is None:
            return deck
        forced = forced_picks(sel)
        if forced is not None:
            return forced

        state = obs["current"]
        went_first = state.get("firstPlayer", -1) == state["yourIndex"]
        try:
            ctx = sel.get("context")
            if ctx == SelectContext.IS_FIRST:
                # the strategy is built around a second-turn Itchy Pollen —
                # prefer going second (option order is [first, second] per
                # the engine; index 1 = play second)
                return [1] if len(sel["option"]) > 1 else [0]
            if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
                return _setup_active(obs, sel, went_first)
            if ctx == SelectContext.SETUP_BENCH_POKEMON:
                return _setup_bench(obs, sel, went_first)
            return search.choose(obs, deck)
        except Exception as exc:
            # Deliberately broad: on Kaggle an uncaught exception forfeits the
            # match, which is strictly worse than one arbitrary pick. Narrowing
            # this would trade a bad move for a loss. The fix for silent
            # degradation is the report above, not a narrower catch.
            _report(exc)
            return _random_picks(sel, rng)

    return agent
