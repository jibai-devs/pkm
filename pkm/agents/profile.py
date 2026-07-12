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

from pathlib import Path
from typing import TYPE_CHECKING

from .spec import AgentSpec, REPO_ROOT

if TYPE_CHECKING:
    from .registry import Agent

AGENTS_DIR = REPO_ROOT / "agents"


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

    def latest_checkpoint(self, phase: str = "ppo") -> Path | None:
        """Return the latest checkpoint for *phase* (``ppo`` or ``exit``), or None."""
        configured_name = self.checkpoint_path.name
        configured_phase = configured_name.removesuffix("_latest.pt")
        if phase == configured_phase or (
            phase == "ppo" and configured_phase not in {"ppo", "exit"}
        ):
            p = self.checkpoint_path
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
