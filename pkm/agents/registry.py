"""Typed registries for profile-backed agent policies and strategies."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pkm.mcts.agent import make_mcts_agent

from .random_agent import make_random_agent

if TYPE_CHECKING:
    from .profile import AgentProfile

Agent = Callable[[dict], list[int]]


class PolicyFactory(Protocol):
    def __call__(
        self,
        profile: "AgentProfile",
        deck: list[int],
        weights_path: str | None = None,
    ) -> Agent: ...


class StrategyFactory(Protocol):
    def __call__(self, profile: "AgentProfile", policy: Agent) -> Agent: ...


def _random_factory(
    profile: "AgentProfile", deck: list[int], weights_path: str | None = None
) -> Agent:
    return make_random_agent(deck)


def _neural_factory(
    profile: "AgentProfile", deck: list[int], weights_path: str | None = None
) -> Agent:
    # Import through the factory module so callers can replace the policy
    # constructor without replacing the registry entry.
    from .factory import make_neural_agent

    return make_neural_agent(
        deck,
        weights_path or str(profile.exported_weights_path),
        require_weights=True,
    )


def _mcts_factory(
    profile: "AgentProfile", deck: list[int], weights_path: str | None = None
) -> Agent:
    resolved_weights = weights_path or str(profile.exported_weights_path)
    if not Path(resolved_weights).is_file():
        raise FileNotFoundError(
            "exported policy weights not found for configured agent: "
            f"{resolved_weights}"
        )
    return make_mcts_agent(
        deck,
        weights_path=resolved_weights,
    )


POLICY_FACTORIES: dict[str, PolicyFactory] = {
    "random": _random_factory,
    "neural": _neural_factory,
    "mcts": _mcts_factory,
}
STRATEGY_FACTORIES: dict[str, StrategyFactory] = {}


def require_policy(name: str) -> PolicyFactory:
    try:
        return POLICY_FACTORIES[name]
    except KeyError as error:
        raise ValueError(f"unknown policy {name!r}") from error


def require_strategy(name: str) -> StrategyFactory:
    try:
        return STRATEGY_FACTORIES[name]
    except KeyError as error:
        raise ValueError(f"unknown strategy {name!r}") from error
