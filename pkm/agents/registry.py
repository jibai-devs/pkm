"""Typed registries for profile-backed agent policies, strategies, and trainers."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

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


@dataclass(frozen=True)
class TrainingResult:
    """Outputs produced by a profile training facade."""

    checkpoint: Path
    metrics: Path | None = None
    iterations: int = 0


Trainer = Callable[..., TrainingResult]
PPO_TRAINER = "ppo"
EXPERT_TRAINER = "expert_iteration"


def _ppo_trainer(**kwargs: Any) -> TrainingResult:
    # Imported lazily: torch is not available in the Kaggle inference bundle.
    from pkm.rl.train import train_profile

    return train_profile(**kwargs)


def _expert_trainer(**kwargs: Any) -> TrainingResult:
    from pkm.rl.exit_train import train_profile

    return train_profile(**kwargs)


TRAINERS: dict[str, Trainer] = {
    PPO_TRAINER: _ppo_trainer,
    EXPERT_TRAINER: _expert_trainer,
}


def register_trainer(name: str, trainer: Trainer, *, replace: bool = False) -> None:
    """Register a profile trainer by name."""
    if not name:
        raise ValueError("trainer name must not be empty")
    if name in TRAINERS and not replace:
        raise ValueError(f"trainer {name!r} is already registered")
    TRAINERS[name] = trainer


def require_trainer(name: str) -> Trainer:
    try:
        return TRAINERS[name]
    except KeyError as error:
        raise ValueError(f"unknown trainer {name!r}") from error
