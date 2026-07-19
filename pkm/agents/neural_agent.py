"""Kaggle-compatible agent backed by an exported policy (numpy inference).

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


def _find_archetype_weights(explicit: str | None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("PKM_ARCHETYPE_PATH"),
        str(Path(__file__).resolve().parent.parent / "archetype.npz"),
        "/kaggle_simulations/agent/pkm/archetype.npz",
        "/kaggle_simulations/agent/archetype.npz",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


def _load_archetype_classifier(explicit: str | None):
    """Non-fatal: a missing/corrupt/stale (stamp-mismatched) archetype
    classifier must never take down the whole agent -- same safety net as
    pkm/mcts/agent.py's `except Exception: return policy.select(obs)`."""
    path = _find_archetype_weights(explicit)
    if path is None:
        return None
    try:
        from pkm.archetype.numpy_model import NumpyArchetypeClassifier

        return NumpyArchetypeClassifier.load(path)
    except Exception:
        return None


def make_neural_agent(
    deck: list[int],
    weights_path: str | None = None,
    archetype_weights_path: str | None = None,
) -> Callable[[dict], list[int]]:
    """Create an agent function that plays greedily with the trained policy."""
    path = _find_weights(weights_path)
    policy = None
    if path is not None:
        from pkm.rl.numpy_policy import NumpyPolicy

        policy = NumpyPolicy.load(path)

    archetype_classifier = _load_archetype_classifier(archetype_weights_path)

    ctx = GameContext(list(deck), DeckTracker(deck))

    def agent(obs: dict) -> list[int]:
        ctx.tracker.observe(obs)
        if ctx.tracker.is_search_reveal(obs):
            ctx.tracker.record_search_reveal(obs)

        if obs["select"] is None:
            return deck
        if archetype_classifier is not None:
            from pkm.archetype.belief import compute_belief

            ctx.archetype_belief = compute_belief(obs, archetype_classifier)
        if policy is None:
            sel = obs["select"]
            return random.sample(range(len(sel["option"])), sel["maxCount"])
        return policy.select(obs, ctx)

    return agent
