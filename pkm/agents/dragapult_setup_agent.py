"""The Dragapult *setup* agent: plays each side's second turn.

Trained separately from the default agent (`pkm/rl/setup_train.py`) on a
2-turn episode whose reward is the end-of-turn board rubric in
`pkm/rl/setup_turn_score.py` -- build a bench of charged Drakloaks behind a
disposable staller, rather than play for a win it can't reach in two turns.
`singaporean_middleman` routes each seat's own second turn here.

Weight lookup order: explicit path arg, $PKM_SETUP_POLICY_PATH,
policy_setup.npz next to the pkm package (bundled in the submission),
/kaggle_simulations/agent/. It is a *separate* export from the default
agent's policy.npz -- same network shape, different training.

**If no setup weights are found it delegates to the default agent** rather
than falling back to random. An untrained-but-sensible policy is strictly
better than noise on the turn that sets up the whole game, and this keeps
the middleman safe to wire up before the first export exists.
"""

import os
from pathlib import Path
from typing import Callable

from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker


def _find_setup_weights(explicit: str | None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("PKM_SETUP_POLICY_PATH"),
        str(Path(__file__).resolve().parent.parent / "policy_setup.npz"),
        "/kaggle_simulations/agent/pkm/policy_setup.npz",
        "/kaggle_simulations/agent/policy_setup.npz",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


def make_dragapult_setup_agent(
    deck: list[int],
    weights_path: str | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> Callable[[dict], list[int]]:
    """Create the setup-turn agent, or the default agent if it has no weights."""
    path = _find_setup_weights(weights_path)
    if path is None:
        msg = (
            "dragapult_setup: no policy_setup.npz found -- "
            "delegating to dragapult_default"
        )
        if log_sink is not None:
            log_sink(msg)
        else:
            print(msg, flush=True)
        from .dragapult_default_agent import make_dragapult_default_agent

        return make_dragapult_default_agent(deck)

    from pkm.rl.numpy_policy import NumpyPolicy

    policy = NumpyPolicy.load(path)
    ctx = GameContext(list(deck), DeckTracker(deck))

    def agent(obs: dict) -> list[int]:
        ctx.tracker.observe(obs)
        if ctx.tracker.is_search_reveal(obs):
            ctx.tracker.record_search_reveal(obs)
        if obs["select"] is None:
            return deck
        return policy.select(obs, ctx)

    return agent
