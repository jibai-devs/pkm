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
from pathlib import Path
from collections.abc import Callable
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
TRAINERS: dict[str, Trainer] = {}
EXIT_TRAINER: Trainer | None = None


def _registered_trainers() -> tuple[dict[str, Trainer], Trainer]:
    """Load built-in trainers lazily so low-level modules stay importable."""
    global EXIT_TRAINER
    if "ppo" not in TRAINERS:
        from pkm.rl.train import train_profile

        TRAINERS["ppo"] = train_profile
    if EXIT_TRAINER is None:
        from pkm.rl.exit_train import train_profile

        EXIT_TRAINER = train_profile
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
        **kwargs: Any,
    ) -> TrainingResult:
        """Train this profile with its configured trainer and output paths."""
        self.ensure_dirs()
        trainers, _ = _registered_trainers()
        try:
            trainer = trainers[self.trainer]
        except KeyError as error:
            raise ValueError(f"unknown trainer {self.trainer!r}") from error
        resume = str(resume_path) if resume_path is not None else self.ppo_init()
        return trainer(
            deck_path=self.deck_path,
            checkpoint_path=self.checkpoint_path,
            checkpoint_dir=self.checkpoint_dir,
            metrics_dir=self.metrics_dir,
            runs_dir=self.runs_dir,
            resume_path=Path(resume) if resume else None,
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
        resume_path: Path | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Train this profile with the expert-iteration trainer."""
        self.ensure_dirs()
        _, trainer = _registered_trainers()
        resume = str(resume_path) if resume_path is not None else self.exit_init()
        return trainer(
            deck_path=self.deck_path,
            checkpoint_path=self.exit_checkpoint_path,
            checkpoint_dir=self.checkpoint_dir,
            metrics_dir=self.metrics_dir,
            runs_dir=self.runs_dir,
            resume_path=Path(resume) if resume else None,
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
