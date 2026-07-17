"""Kaggle submission entry point for agent_000_dragapult.

Packed into a submission bundle by ``pkm new_agents 000_dragapult pack`` (which
copies this file to ``main.py`` and drops the trained ``weights.pt`` next to it).
Exposes a plain ``agent(obs) -> list[int]`` callable — it must be a plain function
(not a bound method) so kaggle's ``co_argcount`` check sees exactly one argument.

NOTE: inference uses torch (loads a ``PolicyValueModel``). No torch is bundled
(the >197 MiB limit forbids it), so this relies on torch being present in the
cabt sandbox. If it is not, add a numpy-only inference path instead.
"""

from pathlib import Path

from pkm.new_agents.agent_000_dragapult.agent import DragapultAgent

_DIR = Path(__file__).resolve().parent
_KAGGLE_DIR = Path("/kaggle_simulations/agent")


def _weights_path() -> str | None:
    for candidate in (_DIR / "weights.pt", _KAGGLE_DIR / "weights.pt"):
        if candidate.is_file():
            return str(candidate)
    return None


_weights = _weights_path()
_agent_obj = (
    DragapultAgent.from_checkpoint(_weights, greedy=True)
    if _weights
    else DragapultAgent(
        greedy=True
    )  # untrained fallback (should not happen in a bundle)
)


def agent(obs: dict) -> list[int]:
    return _agent_obj(obs)
