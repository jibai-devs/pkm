"""Agent profile resolution.

An agent name (e.g. ``00_basic``, ``01_psychic``) maps to a deck and a set of
output directories for checkpoints, metrics, and TensorBoard logs.

Directory layout::

    agents/<name>/
        checkpoints/
        metrics/
        runs/
            ppo/
            exit/
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import spec as spec_module
from .registry import (
    EXPERT_TRAINER,
    TRAINERS,
    Trainer,
    TrainingResult,
    register_trainer,
    require_trainer,
)
from .spec import AgentSpec

if TYPE_CHECKING:
    from .registry import Agent

__all__ = [
    "AgentProfile",
    "EXPERT_TRAINER",
    "TRAINERS",
    "Trainer",
    "TrainingResult",
    "register_trainer",
    "require_trainer",
]


def _agents_root() -> Path:
    """Resolve the agents directory at call time so tests can relocate the root."""
    return spec_module.REPO_ROOT / "agents"


class AgentProfile:
    """Compatibility facade over a declarative :class:`AgentSpec`."""

    def __init__(self, name: str, _spec: AgentSpec | None = None) -> None:
        self.spec = _spec or AgentSpec.load(name)

    @classmethod
    def load(cls, name: str) -> "AgentProfile":
        return cls(name, _spec=AgentSpec.load(name))

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def deck_path(self) -> Path:
        return self.spec.deck_path

    @property
    def policy(self) -> str:
        return self.spec.policy

    @property
    def trainer(self) -> str:
        return self.spec.trainer

    @property
    def strategy(self) -> str | None:
        return self.spec.strategy

    @property
    def checkpoint_path(self) -> Path:
        return self.spec.checkpoint_path

    @property
    def base_dir(self) -> Path:
        return _agents_root() / self.name

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_path.parent

    @property
    def exit_checkpoint_path(self) -> Path:
        return self.spec.exit_checkpoint_path or self.checkpoint_dir / "exit_latest.pt"

    @property
    def exported_weights_path(self) -> Path:
        return self.spec.exported_weights_path

    @property
    def metrics_dir(self) -> Path:
        return self.spec.metrics_dir

    @property
    def runs_dir(self) -> Path:
        return self.spec.runs_dir

    @property
    def submissions_dir(self) -> Path:
        return self.spec.submissions_dir

    def load_deck(self) -> list[int]:
        return self.spec.load_deck()

    def make_agent(
        self,
        policy: str | None = None,
        weights_path: str | None = None,
    ) -> Agent:
        """Build this profile's plain Kaggle-compatible agent callable."""
        from .factory import make_profile_agent

        return make_profile_agent(self, policy=policy, weights_path=weights_path)

    def ensure_dirs(self) -> None:
        """Create the agent's directory tree if it doesn't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        (self.runs_dir / "ppo").mkdir(parents=True, exist_ok=True)
        (self.runs_dir / "exit").mkdir(parents=True, exist_ok=True)

    def _own_output_path(self, path: Path | None, label: str) -> Path | None:
        """Reject an output path that resolves into a different profile.

        Uses ``realpath`` rather than ``abspath``: a symlink planted inside this
        profile's own directory must not be usable to write into another one.
        """
        if path is None:
            return None
        candidate = Path(os.path.realpath(os.path.expanduser(str(path))))
        agents_root = Path(os.path.realpath(_agents_root()))
        if candidate == agents_root:
            raise ValueError(
                f"{label} {candidate} is the agents root; a profile may only write "
                f"to its own agents/{self.name}/ directory"
            )
        if agents_root in candidate.parents:
            owner = candidate.relative_to(agents_root).parts[0]
            if owner != self.name:
                raise ValueError(
                    f"{label} {candidate} belongs to agent profile {owner!r}, not "
                    f"{self.name!r}; a profile may only write to its own "
                    "agents/<name>/ directory"
                )
        return candidate

    def _run_trainer(self, name: str, **call: Any) -> TrainingResult:
        """Dispatch to a registered trainer, reporting argument mismatches clearly."""
        trainer = require_trainer(name)
        try:
            inspect.signature(trainer).bind(**call)
        except TypeError as error:
            raise TypeError(
                f"trainer {name!r} for agent profile {self.name!r} rejected the "
                f"supplied arguments: {error}"
            ) from error
        # Called outside the guard above so a TypeError raised *inside* the
        # trainer is not mislabelled as an argument mismatch.
        return trainer(**call)

    def train(
        self,
        *,
        iterations: int = 50,
        games: int = 8,
        lr: float = 3e-4,
        seed: int = 0,
        resume_path: Path | None = None,
        checkpoint_dir: Path | None = None,
        metrics_path: Path | None = None,
        log_dir: Path | None = None,
        **hyperparams: Any,
    ) -> TrainingResult:
        """Train this profile with its configured trainer and output paths.

        Trainer-specific hyperparameters (PPO's ``gamma``, ``pool_size``, ...) are
        forwarded through ``hyperparams`` so that a profile configured with a
        non-PPO trainer is not handed PPO-only arguments.
        """
        self.ensure_dirs()
        resume = resume_path
        if resume is None:
            ppo_resume = self.ppo_init()
            resume = Path(ppo_resume) if ppo_resume else None
        output_dir = (
            self._own_output_path(checkpoint_dir, "checkpoint directory")
            or self.checkpoint_dir
        )
        return self._run_trainer(
            self.trainer,
            deck_path=self.deck_path,
            checkpoint_path=output_dir / self.checkpoint_path.name,
            checkpoint_dir=output_dir,
            metrics_dir=self.metrics_dir,
            runs_dir=self.runs_dir,
            metrics_path=self._own_output_path(metrics_path, "metrics path"),
            log_dir=self._own_output_path(log_dir, "log directory"),
            resume_path=resume,
            iterations=iterations,
            games_per_iter=games,
            lr=lr,
            seed=seed,
            **hyperparams,
        )

    def train_exit(
        self,
        *,
        iterations: int = 3,
        games: int = 4,
        lr: float = 1e-4,
        n_simulations: int = 24,
        n_determinizations: int = 2,
        seed: int = 0,
        resume: bool = False,
        resume_path: Path | None = None,
        checkpoint_dir: Path | None = None,
        metrics_path: Path | None = None,
        log_dir: Path | None = None,
        **hyperparams: Any,
    ) -> TrainingResult:
        """Train this profile with the expert-iteration trainer."""
        self.ensure_dirs()
        if resume_path is not None:
            initialization = resume_path
        elif resume:
            exit_resume = self.exit_init()
            initialization = Path(exit_resume) if exit_resume else None
        else:
            initialization = self.checkpoint_path
        output_dir = (
            self._own_output_path(checkpoint_dir, "checkpoint directory")
            or self.checkpoint_dir
        )
        return self._run_trainer(
            EXPERT_TRAINER,
            deck_path=self.deck_path,
            checkpoint_path=output_dir / self.exit_checkpoint_path.name,
            checkpoint_dir=output_dir,
            metrics_dir=self.metrics_dir,
            runs_dir=self.runs_dir,
            metrics_path=self._own_output_path(metrics_path, "metrics path"),
            log_dir=self._own_output_path(log_dir, "log directory"),
            resume_path=initialization,
            iterations=iterations,
            games_per_iter=games,
            lr=lr,
            n_simulations=n_simulations,
            n_determinizations=n_determinizations,
            seed=seed,
            **hyperparams,
        )

    def latest_checkpoint(self, phase: str = "ppo") -> Path | None:
        """Return the latest checkpoint for *phase* (``ppo`` or ``exit``), or None."""
        if phase == "ppo":
            p = self.checkpoint_path
        elif phase == "exit":
            p = self.exit_checkpoint_path
        else:
            p = self.checkpoint_dir / f"{phase}_latest.pt"
        return p if p.is_file() else None

    def ppo_init(self) -> str | None:
        """Checkpoint to resume PPO from, if one exists."""
        p = self.latest_checkpoint("ppo")
        return str(p) if p else None

    def exit_init(self) -> str:
        """Checkpoint to initialize expert iteration from (prefer exit, fall back to ppo)."""
        p = self.latest_checkpoint("exit")
        if p is None:
            p = self.latest_checkpoint("ppo")
        return str(p) if p else ""

    @staticmethod
    def list_agents() -> list[str]:
        """Return sorted list of agent profile names."""
        agents_root = _agents_root()
        if not agents_root.is_dir():
            return []
        return sorted(
            d.name
            for d in agents_root.iterdir()
            if d.is_dir() and (d / "profile.yaml").is_file()
        )
