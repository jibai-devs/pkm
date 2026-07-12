"""Immutable declarative agent profile specifications."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pkm.data.deck import Deck

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_FIELDS = {"name", "deck", "policy", "trainer", "strategy", "checkpoint"}
REQUIRED_FIELDS = PROFILE_FIELDS - {"strategy"}


@dataclass(frozen=True)
class AgentSpec:
    """The immutable configuration loaded from an agent profile YAML file."""

    name: str
    deck_path: Path
    policy: str
    trainer: str
    strategy: str | None
    checkpoint_path: Path

    @classmethod
    def load(cls, name: str) -> "AgentSpec":
        profile_path = REPO_ROOT / "agents" / name / "profile.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(f"Agent profile not found: {name!r} ({profile_path})")

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
        if values["name"] != name:
            raise ValueError(
                f"Agent profile name mismatch: requested {name!r}, configured {values['name']!r}"
            )

        return cls(
            name=name,
            deck_path=Path(values["deck"]),
            policy=str(values["policy"]),
            trainer=str(values["trainer"]),
            strategy=None if values.get("strategy") is None else str(values["strategy"]),
            checkpoint_path=Path(values["checkpoint"]),
        )

    def load_deck(self) -> list[int]:
        """Load and validate this profile's deck, returning a fresh card list."""
        path = REPO_ROOT / self.deck_path
        if not path.is_file():
            raise FileNotFoundError(f"Deck for agent profile {self.name!r} not found: {path}")
        return list(Deck.from_csv(path).card_ids)

    @property
    def metrics_dir(self) -> Path:
        return Path("agents") / self.name / "metrics"

    @property
    def runs_dir(self) -> Path:
        return Path("agents") / self.name / "runs"

    @property
    def submissions_dir(self) -> Path:
        return Path("agents") / self.name / "submissions"
