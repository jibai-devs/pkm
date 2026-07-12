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

from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .spec import AgentSpec, REPO_ROOT

if TYPE_CHECKING:
    from .registry import Agent

AGENTS_DIR = REPO_ROOT / "agents"


@dataclass(frozen=True)
class TrainingResult:
    """Outputs produced by a profile training facade."""

    checkpoint: Path
    metrics: Path | None = None
    iterations: int = 0


Trainer = Callable[..., TrainingResult]
EXPERT_TRAINER = "expert_iteration"


def _ppo_trainer(**kwargs: Any) -> TrainingResult:
    from pkm.rl.train import train_profile

    return train_profile(**kwargs)


def _expert_trainer(**kwargs: Any) -> TrainingResult:
    from pkm.rl.exit_train import train_profile

    return train_profile(**kwargs)


TRAINERS: dict[str, Trainer] = {
    "ppo": _ppo_trainer,
    EXPERT_TRAINER: _expert_trainer,
}
EXIT_TRAINER: Trainer | None = _expert_trainer


def register_trainer(name: str, trainer: Trainer, *, replace: bool = False) -> None:
    """Register a profile trainer by name."""
    global EXIT_TRAINER
    if not name:
        raise ValueError("trainer name must not be empty")
    if name in TRAINERS and not replace:
        raise ValueError(f"trainer {name!r} is already registered")
    TRAINERS[name] = trainer
    if name == EXPERT_TRAINER:
        EXIT_TRAINER = trainer


def _registered_trainers() -> tuple[dict[str, Trainer], Trainer]:
    """Load built-in trainers lazily so low-level modules stay importable."""
    global EXIT_TRAINER
    if EXIT_TRAINER is not None:
        TRAINERS[EXPERT_TRAINER] = EXIT_TRAINER
    else:
        EXIT_TRAINER = TRAINERS[EXPERT_TRAINER]
    return TRAINERS, EXIT_TRAINER


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
        return AGENTS_DIR / self.name

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

    def train(
        self,
        *,
        iterations: int = 50,
        games: int = 8,
        lr: float = 3e-4,
        gamma: float = 0.99,
        shaping: float = 0.2,
        pool_size: int = 8,
        eval_every: int = 5,
        eval_games: int = 20,
        seed: int = 0,
        resume_path: Path | None = None,
        checkpoint_dir: Path | None = None,
        metrics_path: Path | None = None,
        log_dir: Path | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Train this profile with its configured trainer and output paths."""
        self.ensure_dirs()
        trainers, _ = _registered_trainers()
        try:
            trainer = trainers[self.trainer]
        except KeyError as error:
            raise ValueError(f"unknown trainer {self.trainer!r}") from error
        resume = resume_path
        if resume is None:
            ppo_resume = self.ppo_init()
            resume = Path(ppo_resume) if ppo_resume else None
        output_dir = checkpoint_dir or self.checkpoint_dir
        output_checkpoint = output_dir / self.checkpoint_path.name
        return trainer(
            deck_path=self.deck_path,
            checkpoint_path=output_checkpoint,
            checkpoint_dir=output_dir,
            metrics_dir=self.metrics_dir,
            runs_dir=self.runs_dir,
            metrics_path=metrics_path,
            log_dir=log_dir,
            resume_path=resume,
            iterations=iterations,
            games_per_iter=games,
            lr=lr,
            gamma=gamma,
            shaping_coef=shaping,
            pool_size=pool_size,
            eval_every=eval_every,
            eval_games=eval_games,
            seed=seed,
            **kwargs,
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
        **kwargs: Any,
    ) -> TrainingResult:
        """Train this profile with the expert-iteration trainer."""
        self.ensure_dirs()
        _, trainer = _registered_trainers()
        if resume_path is not None:
            initialization = resume_path
        elif resume:
            exit_resume = self.exit_init()
            initialization = Path(exit_resume) if exit_resume else None
        else:
            initialization = self.checkpoint_path
        output_dir = checkpoint_dir or self.checkpoint_dir
        output_checkpoint = output_dir / self.exit_checkpoint_path.name
        return trainer(
            deck_path=self.deck_path,
            checkpoint_path=output_checkpoint,
            checkpoint_dir=output_dir,
            metrics_dir=self.metrics_dir,
            runs_dir=self.runs_dir,
            metrics_path=metrics_path,
            log_dir=log_dir,
            resume_path=initialization,
            iterations=iterations,
            games_per_iter=games,
            lr=lr,
            n_simulations=n_simulations,
            n_determinizations=n_determinizations,
            seed=seed,
            **kwargs,
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
        if not AGENTS_DIR.is_dir():
            return []
        return sorted(
            d.name
            for d in AGENTS_DIR.iterdir()
            if d.is_dir() and (d / "profile.yaml").is_file()
        )
