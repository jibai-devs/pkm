"""Kaggle-compatible agent backed by an exported policy (numpy inference).

Weight lookup order: explicit path arg, $PKM_POLICY_PATH, policy.npz next to
the pkm package (bundled in the submission), /kaggle_simulations/agent/.
Falls back to random legal moves if no weights are found.
"""

import os
import random
from pathlib import Path
from typing import Callable


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


def make_neural_agent(
    deck: list[int], weights_path: str | None = None
) -> Callable[[dict], list[int]]:
    """Create an agent function that plays greedily with the trained policy."""
    path = _find_weights(weights_path)
    policy = None
    if path is not None:
        from pkm.rl.numpy_policy import NumpyPolicy

        policy = NumpyPolicy.load(path)

    def agent(obs: dict) -> list[int]:
        if obs["select"] is None:
            return deck
        if policy is None:
            sel = obs["select"]
            return random.sample(range(len(sel["option"])), sel["maxCount"])
        return policy.select(obs)

    return agent
