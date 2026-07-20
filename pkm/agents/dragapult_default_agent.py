"""The default Dragapult agent: the exported policy net (numpy inference).

This is the workhorse `singaporean_middleman` routes to for every decision
outside our own first turn (see `first_turn_agent.py` for that one).

Weight lookup order: explicit path arg, $PKM_POLICY_PATH, policy.npz next to
the pkm package (bundled in the submission), /kaggle_simulations/agent/.
Falls back to random legal moves if no weights are found.
"""

import os
import random
from pathlib import Path
from typing import Callable

from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker


def _find_weights(explicit: str | None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("PKM_POLICY_PATH"),
        str(Path(__file__).resolve().parent.parent / "policy.npz"),
        "/kaggle_simulations/agent/pkm/policy.npz",
        "/kaggle_simulations/agent/policy.npz",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


def load_policy(weights_path: str | None = None):
    """Load the exported policy once, or None if no weights can be found.

    Split out of the factory so a caller that builds *many* agents can pay the
    weight load once. The turn planner does exactly this: it needs a fresh
    agent per plan (for a clean `DeckTracker`) but the weights never change.
    """
    path = _find_weights(weights_path)
    if path is None:
        return None
    from pkm.rl.numpy_policy import NumpyPolicy

    return NumpyPolicy.load(path)


def make_dragapult_default_agent(
    deck: list[int], weights_path: str | None = None, policy=None
) -> Callable[[dict], list[int]]:
    """Create an agent function that plays greedily with the trained policy.

    `policy` accepts an already-loaded policy (see `load_policy`) to skip
    re-reading the weights from disk; the per-game memory below is still built
    fresh on every call, which is the part that must never be shared.
    """
    if policy is None:
        policy = load_policy(weights_path)

    ctx = GameContext(list(deck), DeckTracker(deck))

    def agent(obs: dict) -> list[int]:
        ctx.tracker.observe(obs)
        if ctx.tracker.is_search_reveal(obs):
            ctx.tracker.record_search_reveal(obs)

        if obs["select"] is None:
            return deck
        if policy is None:
            sel = obs["select"]
            return random.sample(range(len(sel["option"])), sel["maxCount"])
        return policy.select(obs, ctx)

    return agent
