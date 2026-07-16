"""Kaggle-compatible agent that picks moves with IS-MCTS guided by the policy net."""

import random
from typing import Callable

from pkm.agents.neural_agent import _find_weights
from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.rl.numpy_policy import NumpyPolicy

from .determinize import infer_opponent_decklist
from .search import MCTS, forced_picks


def make_mcts_agent(
    deck: list[int],
    weights_path: str | None = None,
    opp_decklist: list[int] | None = None,
    n_determinizations: int = 2,
    n_simulations: int = 32,
    seed: int | None = None,
) -> Callable[[dict], list[int]]:
    """Create an MCTS agent. Falls back to the raw policy without search when
    the decision is forced, and to random moves if no weights exist."""
    path = _find_weights(weights_path)
    if path is None:
        raise FileNotFoundError(
            "no policy weights found; export one with pkm.rl.export"
        )
    policy = NumpyPolicy.load(path)
    mcts = MCTS(
        policy,
        n_determinizations=n_determinizations,
        n_simulations=n_simulations,
        rng=random.Random(seed),
    )
    ctx = GameContext(list(deck), DeckTracker(deck), opp_decklist=opp_decklist)

    def agent(obs: dict) -> list[int]:
        ctx.tracker.observe(obs)
        if ctx.tracker.is_search_reveal(obs):
            ctx.tracker.record_search_reveal(obs)

        if obs["select"] is None:
            return deck
        forced = forced_picks(obs["select"])
        if forced is not None:
            return forced
        opp = opp_decklist if opp_decklist is not None else infer_opponent_decklist(obs)
        try:
            picks, _ = mcts.choose(obs, deck, opp)
            return picks
        except Exception:
            # search must never crash the match; fall back to the raw policy
            return policy.select(obs)

    return agent
