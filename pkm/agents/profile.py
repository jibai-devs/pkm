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

from pathlib import Path

AGENTS_DIR = Path("agents")


class AgentProfile:
    """Resolves an agent name to its deck and output directories."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.base_dir = AGENTS_DIR / name
        self.deck_path = Path(f"deck/{name}.csv")
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.metrics_dir = self.base_dir / "metrics"
        self.runs_dir = self.base_dir / "runs"
        self.reward_weights_path = self.base_dir / "reward_weights.json"

    def ensure_dirs(self) -> None:
        """Create the agent's directory tree if it doesn't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        (self.runs_dir / "ppo").mkdir(parents=True, exist_ok=True)
        (self.runs_dir / "exit").mkdir(parents=True, exist_ok=True)

    def latest_checkpoint(self, phase: str = "ppo") -> Path | None:
        """Return the latest checkpoint for *phase* (``ppo`` or ``exit``), or None."""
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
            if d.is_dir() and (d / "checkpoints").is_dir()
        )
