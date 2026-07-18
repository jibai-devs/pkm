"""Pluggable training methods.

A ``Trainer`` owns the two method-specific halves of one PPO/ExIt update:
``collect`` (self-play → samples, runs in rollout workers) and ``update`` (the
learn step). Everything else — checkpoint/resume, observers, the parallel pool,
eval, timing/utilization diagnostics — lives in the method-agnostic driver
(:func:`..train.train`). Methods register in :data:`TRAINERS`, keyed by
``cfg.train.method`` (mirrors ``shaping.SHAPERS``/``ESTIMATORS``).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch


@runtime_checkable
class Trainer(Protocol):
    def collect(
        self, model: torch.nn.Module, n_games: int, cfg: Any,
        gen: "torch.Generator | None" = None,
    ) -> tuple[list, dict]:
        """Self-play → (samples, stats). Runs inside rollout workers."""
        ...

    def update(
        self, model: torch.nn.Module, opt: torch.optim.Optimizer,
        samples: list, cfg: Any,
    ) -> dict:
        """One learn step over ``samples`` → per-update stats."""
        ...


def _ppo_trainer() -> Trainer:
    from pkm.new_agents.agent_000_dragapult.trainers.ppo import PpoTrainer
    return PpoTrainer()


def _exit_trainer() -> Trainer:
    from pkm.new_agents.agent_000_dragapult.trainers.exit import ExItTrainer
    return ExItTrainer()


# Lazy factories so importing this package doesn't pull torch-heavy modules
# until a method is actually selected.
TRAINERS: dict[str, Callable[[], Trainer]] = {
    "ppo": _ppo_trainer,
    "exit": _exit_trainer,
}


def get_trainer(cfg: Any) -> Trainer:
    method = cfg.train.method
    try:
        factory = TRAINERS[method]
    except KeyError:
        raise ValueError(
            f"unknown training method {method!r}; choose from {sorted(TRAINERS)}"
        ) from None
    return factory()
