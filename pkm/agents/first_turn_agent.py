"""Agent for the first turn of the game only (deck/03_pult_munki strategy).

``singaporean_middleman`` routes here for the whole of our own first turn:
the setup decisions (turn 0) and our first playing turn — engine turn 1 when
we go first, turn 2 when we go second (the counter is shared across both
players). Handoff back to the neural agent happens on the next turn change.

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
    return rng.sample(range(len(sel["option"])), sel["maxCount"])


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
    n_simulations: int = 40,
    time_budget_s: float = 6.0,
    seed: int | None = None,
) -> Callable[[dict], list[int]]:
    """Create the agent that plays only our first turn of the game."""
    rng = random.Random(seed)
    search = Turn1Search(
        n_determinizations=n_determinizations,
        n_simulations=n_simulations,
        time_budget_s=time_budget_s,
        rng=rng,
    )

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
        except Exception:
            # never crash the match on a search failure
            return _random_picks(sel, rng)

    return agent
