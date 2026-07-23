"""Kaggle-compatible agent that picks moves with IS-MCTS guided by the policy net."""

import random
from typing import Callable

from pkm.agents.dragapult_default_agent import (
    _find_weights,
    _load_archetype_classifier,
)
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
    archetype_weights_path: str | None = None,
) -> Callable[[dict], list[int]]:
    """Create an MCTS agent. Falls back to the raw policy without search when
    the decision is forced, and to random moves if no weights exist.

    archetype_weights_path (opt-in): when given (or auto-discovered, same
    lookup order as the main policy weights), biases infer_opponent_decklist's
    padding toward the believed archetype's staple composition. Load failure
    is non-fatal -- falls back to today's crude behavior, same as the
    `except Exception: return policy.select(obs, ctx)` safety net below.
    """
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
    archetype_classifier = _load_archetype_classifier(archetype_weights_path)

    def agent(obs: dict) -> list[int]:
        ctx.tracker.observe(obs)
        if ctx.tracker.is_search_reveal(obs):
            ctx.tracker.record_search_reveal(obs)

        if obs["select"] is None:
            return deck
        forced = forced_picks(obs["select"])
        if forced is not None:
            return forced
        if archetype_classifier is not None:
            from pkm.archetype.belief import compute_belief

            ctx.archetype_belief = compute_belief(obs, archetype_classifier)
        if opp_decklist is not None:
            opp = opp_decklist
        else:
            opp = infer_opponent_decklist(obs, classifier=archetype_classifier)
        try:
            picks, _ = mcts.choose(obs, deck, opp)
            return picks
        except Exception:
            # search must never crash the match; fall back to the raw policy
            return policy.select(obs, ctx)

    return agent
