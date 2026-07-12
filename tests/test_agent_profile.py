from itertools import product
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from pkm.agents import AgentProfile, AgentSpec


def test_dragapult_profile_resolves_paths_from_repository_root(monkeypatch, tmp_path):
    profile = AgentProfile.load("02_dragapult")

    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]

    assert profile.name == "02_dragapult"
    assert profile.deck_path == repo_root / "agents/02_dragapult/deck.csv"
    assert profile.policy == "neural"
    assert profile.trainer == "ppo"
    assert profile.strategy is None
    assert profile.checkpoint_path == (
        repo_root / "agents/02_dragapult/checkpoints/ppo_latest.pt"
    )
    assert profile.metrics_dir == repo_root / "agents/02_dragapult/metrics"
    assert profile.runs_dir == repo_root / "agents/02_dragapult/runs"
    assert profile.submissions_dir == repo_root / "agents/02_dragapult/submissions"
    assert all(
        path.is_absolute()
        for path in (
            profile.deck_path,
            profile.checkpoint_path,
            profile.metrics_dir,
            profile.runs_dir,
            profile.submissions_dir,
        )
    )


def test_profile_spec_is_frozen():
    profile = AgentProfile.load("02_dragapult")

    assert isinstance(profile.spec, AgentSpec)
    assert is_dataclass(profile.spec)
    with pytest.raises(FrozenInstanceError):
        profile.spec.name = "other"


def test_profile_load_deck_validates_and_returns_new_lists():
    profile = AgentProfile.load("02_dragapult")

    first = profile.load_deck()
    second = profile.load_deck()

    assert len(first) == 60
    assert first == second
    assert first is not second


def test_legacy_constructor_loads_named_profile():
    profile = AgentProfile("02_dragapult")

    assert profile.name == "02_dragapult"
    assert profile.deck_path.is_absolute()


def test_missing_profile_has_clear_error():
    with pytest.raises(FileNotFoundError, match="profile.*does-not-exist"):
        AgentProfile.load("does-not-exist")


def test_missing_deck_has_clear_error(tmp_path, monkeypatch):
    profile_dir = tmp_path / "agents" / "broken"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        "name: broken\n"
        "deck: agents/broken/missing.csv\n"
        "policy: neural\n"
        "trainer: ppo\n"
        "checkpoint: agents/broken/checkpoints/ppo_latest.pt\n"
        "strategy: null\n"
    )
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)

    profile = AgentProfile.load("broken")
    with pytest.raises(FileNotFoundError, match="deck.*missing.csv"):
        profile.load_deck()


def test_invalid_deck_length_has_clear_error(tmp_path, monkeypatch):
    profile_dir = tmp_path / "agents" / "short"
    profile_dir.mkdir(parents=True)
    (profile_dir / "deck.csv").write_text("1\n" * 59)
    (profile_dir / "profile.yaml").write_text(
        "name: short\n"
        "deck: agents/short/deck.csv\n"
        "policy: neural\n"
        "trainer: ppo\n"
        "checkpoint: agents/short/checkpoints/ppo_latest.pt\n"
        "strategy: null\n"
    )
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="60 cards.*59"):
        AgentProfile.load("short").load_deck()


def test_unknown_profile_fields_have_clear_error(tmp_path, monkeypatch):
    profile_dir = tmp_path / "agents" / "unknown"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        "name: unknown\n"
        "deck: agents/unknown/deck.csv\n"
        "policy: neural\n"
        "trainer: ppo\n"
        "checkpoint: agents/unknown/checkpoints/ppo_latest.pt\n"
        "strategy: null\n"
        "unexpected: true\n"
    )
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="unknown field.*unexpected"):
        AgentProfile.load("unknown")


def test_missing_strategy_has_clear_error(tmp_path, monkeypatch):
    profile_dir = tmp_path / "agents" / "missing-strategy"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        "name: missing-strategy\n"
        "deck: agents/missing-strategy/deck.csv\n"
        "policy: neural\n"
        "trainer: ppo\n"
        "checkpoint: agents/missing-strategy/checkpoints/ppo_latest.pt\n"
    )
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="missing required field.*strategy"):
        AgentProfile.load("missing-strategy")


def _write_profile(profile_dir: Path, values: dict[str, object]) -> None:
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        "\n".join(f"{key}: {value}" for key, value in values.items()) + "\n"
    )


def _valid_profile_values(name: str = "custom") -> dict[str, object]:
    return {
        "name": name,
        "deck": f"agents/{name}/deck.csv",
        "policy": "neural",
        "trainer": "ppo",
        "checkpoint": f"agents/{name}/checkpoints/ppo_latest.pt",
        "strategy": "null",
    }


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    list(
        product(
            ("name", "deck", "policy", "trainer", "checkpoint"),
            ("true", "[]", "{}", "null"),
        )
    ),
)
def test_string_profile_fields_reject_invalid_yaml_types(
    tmp_path, monkeypatch, field, invalid_value
):
    values = _valid_profile_values()
    values[field] = invalid_value
    _write_profile(tmp_path / "agents" / "custom", values)
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match=f"field.*{field}.*string"):
        AgentProfile.load("custom")


@pytest.mark.parametrize("invalid_value", ("true", "[]", "{}"))
def test_strategy_rejects_invalid_yaml_types(tmp_path, monkeypatch, invalid_value):
    values = _valid_profile_values()
    values["strategy"] = invalid_value
    _write_profile(tmp_path / "agents" / "custom", values)
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="field.*strategy.*string or null"):
        AgentProfile.load("custom")


def test_configured_checkpoint_path_is_authoritative(tmp_path, monkeypatch):
    values = _valid_profile_values()
    values["checkpoint"] = "agents/custom/state/selected.pt"
    profile_dir = tmp_path / "agents" / "custom"
    _write_profile(profile_dir, values)
    checkpoint_path = tmp_path / "agents" / "custom" / "state" / "selected.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("checkpoint")
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)

    profile = AgentProfile.load("custom")

    assert profile.checkpoint_path == checkpoint_path
    assert profile.checkpoint_dir == checkpoint_path.parent
    assert profile.latest_checkpoint("ppo") == checkpoint_path
    assert profile.ppo_init() == str(checkpoint_path)
    assert profile.exit_init() == str(checkpoint_path)

    working_dir = tmp_path / "working"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    profile.ensure_dirs()

    assert profile.metrics_dir.is_dir()
    assert (profile.runs_dir / "ppo").is_dir()
    assert (profile.runs_dir / "exit").is_dir()


def test_profile_discovery_does_not_require_runtime_directories(tmp_path, monkeypatch):
    agents_dir = tmp_path / "agents"
    profile_dir = agents_dir / "profile-only"
    _write_profile(profile_dir, _valid_profile_values("profile-only"))
    monkeypatch.setattr("pkm.agents.profile.AGENTS_DIR", agents_dir)

    assert "profile-only" in AgentProfile.list_agents()
