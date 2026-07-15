"""Routes each decision to one of several sub-agents. Only agent kaggle sees."""

from typing import Callable

from .neural_agent import make_neural_agent
from .random_agent import make_random_agent

AgentFn = Callable[[dict], list[int]]


def _select_agent(obs: dict, agents: dict[str, AgentFn]) -> str:
    """Template: pick a registered agent name for the upcoming turn."""
    return "neural"


def make_singaporean_middleman(
    deck: list[int],
    weights_path: str | None = None,
    agents: dict[str, AgentFn] | None = None,
    select_agent: Callable[[dict, dict[str, AgentFn]], str] = _select_agent,
) -> AgentFn:
    """Build the kaggle-facing agent that dispatches per turn."""
    registry: dict[str, AgentFn] = (
        agents
        if agents is not None
        else {
            "neural": make_neural_agent(deck, weights_path),
            "random": make_random_agent(deck),
        }
    )

    state = {"turn": None, "active": next(iter(registry))}

    def agent(obs: dict) -> list[int]:
        if obs["select"] is None:
            return deck

        turn = obs["current"]["turn"]
        if turn != state["turn"]:
            state["turn"] = turn
            state["active"] = select_agent(obs, registry)

        return registry[state["active"]](obs)

    return agent
