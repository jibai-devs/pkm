"""Immutable declarative agent profile specifications."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pkm.data.deck import Deck

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_FIELDS = {
    "name",
    "deck",
    "policy",
    "trainer",
    "strategy",
    "checkpoint",
    "exit_checkpoint",
}
REQUIRED_FIELDS = PROFILE_FIELDS - {"exit_checkpoint"}


def _require_string(values: dict[str, Any], field: str, profile_name: str) -> str:
    value = values[field]
    if not isinstance(value, str):
        raise ValueError(
            f"Agent profile {profile_name!r} field {field!r} must be a string, "
            f"got {type(value).__name__}"
        )
    return value


@dataclass(frozen=True)
class AgentSpec:
    """The immutable configuration loaded from an agent profile YAML file."""

    name: str
    deck_path: Path
    policy: str
    trainer: str
    strategy: str | None
    checkpoint_path: Path
    exit_checkpoint_path: Path | None = None

    @classmethod
    def load(cls, name: str) -> "AgentSpec":
        profile_path = REPO_ROOT / "agents" / name / "profile.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"Agent profile not found: {name!r} ({profile_path})"
            )

        with profile_path.open() as profile_file:
            values: Any = yaml.safe_load(profile_file)
        if not isinstance(values, dict):
            raise ValueError(f"Agent profile {name!r} must contain a YAML mapping")

        unknown = sorted(set(values) - PROFILE_FIELDS)
        if unknown:
            raise ValueError(
                f"Agent profile {name!r} has unknown field(s): {', '.join(unknown)}"
            )
        missing = sorted(REQUIRED_FIELDS - set(values))
        if missing:
            raise ValueError(
                f"Agent profile {name!r} is missing required field(s): {', '.join(missing)}"
            )
        configured_name = _require_string(values, "name", name)
        if configured_name != name:
            raise ValueError(
                f"Agent profile name mismatch: requested {name!r}, configured {configured_name!r}"
            )
        deck = _require_string(values, "deck", name)
        policy = _require_string(values, "policy", name)
        trainer = _require_string(values, "trainer", name)
        checkpoint = _require_string(values, "checkpoint", name)
        exit_checkpoint = values.get("exit_checkpoint")
        if exit_checkpoint is not None and not isinstance(exit_checkpoint, str):
            raise ValueError(
                f"Agent profile {name!r} field 'exit_checkpoint' must be a string, "
                f"got {type(exit_checkpoint).__name__}"
            )
        strategy = values["strategy"]
        if strategy is not None and not isinstance(strategy, str):
            raise ValueError(
                f"Agent profile {name!r} field 'strategy' must be a string or null, "
                f"got {type(strategy).__name__}"
            )

        from .registry import POLICY_FACTORIES, STRATEGY_FACTORIES

        if policy not in POLICY_FACTORIES:
            raise ValueError(f"Agent profile {name!r} has unknown policy {policy!r}")
        if strategy is not None and strategy not in STRATEGY_FACTORIES:
            raise ValueError(
                f"Agent profile {name!r} has unknown strategy {strategy!r}"
            )

        return cls(
            name=name,
            deck_path=(REPO_ROOT / deck).resolve(),
            policy=policy,
            trainer=trainer,
            strategy=strategy,
            checkpoint_path=(REPO_ROOT / checkpoint).resolve(),
            exit_checkpoint_path=(REPO_ROOT / exit_checkpoint).resolve()
            if exit_checkpoint
            else None,
        )

    def load_deck(self) -> list[int]:
        """Load and validate this profile's deck, returning a fresh card list."""
        path = self.deck_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Deck for agent profile {self.name!r} not found: {path}"
            )
        return list(Deck.from_csv(path).card_ids)

    @property
    def exported_weights_path(self) -> Path:
        """Path to the policy export owned by this profile."""
        return self.checkpoint_path.parent / "policy.npz"

    @property
    def metrics_dir(self) -> Path:
        return (REPO_ROOT / "agents" / self.name / "metrics").resolve()

    @property
    def runs_dir(self) -> Path:
        return (REPO_ROOT / "agents" / self.name / "runs").resolve()

    @property
    def submissions_dir(self) -> Path:
        return (REPO_ROOT / "agents" / self.name / "submissions").resolve()
