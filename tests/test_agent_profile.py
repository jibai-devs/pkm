from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from pkm.agents import AgentProfile, AgentSpec


def test_dragapult_profile_resolves_paths_from_repository_root(monkeypatch, tmp_path):
    profile = AgentProfile.load("02_dragapult")

    monkeypatch.chdir(tmp_path)

    assert profile.name == "02_dragapult"
    assert profile.deck_path == Path("agents/02_dragapult/deck.csv")
    assert profile.policy == "neural"
    assert profile.trainer == "ppo"
    assert profile.strategy is None
    assert profile.checkpoint_path == Path("agents/02_dragapult/checkpoints/ppo_latest.pt")
    assert profile.metrics_dir == Path("agents/02_dragapult/metrics")
    assert profile.runs_dir == Path("agents/02_dragapult/runs")
    assert profile.submissions_dir == Path("agents/02_dragapult/submissions")


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
    assert profile.deck_path == Path("agents/02_dragapult/deck.csv")


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
