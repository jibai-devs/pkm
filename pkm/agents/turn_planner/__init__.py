"""Plan a whole turn up front, then measure what reality did instead.

At the first decision of each turn, `TurnPlanner` forks the current game into
a simulated copy (unknowns randomized plausibly, the opponent played out
randomly), plays the turn to its end, and records every decision plus which
agent the router picked for it. As the real turn then plays out, each actual
decision is scored against the plan and the whole thing is written to JSON.

Enable by setting ``PKM_TURN_PLAN_DIR`` to an output directory; it is inert
otherwise. See `worker.py` for the engine-imposed fidelity limit.
"""

from .client import TurnPlanner, plan_dir

__all__ = ["TurnPlanner", "plan_dir"]
