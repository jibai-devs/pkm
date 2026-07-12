from pathlib import Path

import pytest

from pkm.agents import AgentProfile
from pkm.agents.registry import POLICY_FACTORIES, STRATEGY_FACTORIES


def test_profile_make_agent_returns_kaggle_callable(monkeypatch):
    profile = AgentProfile.load("02_dragapult")
    expected = list(range(60))
    monkeypatch.setattr(profile, "load_deck", lambda: expected)
    monkeypatch.setattr(
        "pkm.agents.factory.make_neural_agent",
        lambda deck, weights, **kwargs: (
            lambda obs: deck if obs["select"] is None else [0]
        ),
    )

    policy = profile.make_agent()

    assert callable(policy)
    assert policy({"select": None}) == expected
    assert policy({"select": {"option": ["move"], "maxCount": 1}}) == [0]


def test_random_profile_agent_selects_legal_options(tmp_path, monkeypatch):
    profile = _profile(tmp_path, monkeypatch, policy="random")

    policy = profile.make_agent()

    assert policy({"select": None}) == [1] * 60
    assert len(policy({"select": {"option": ["a", "b"], "maxCount": 1}})) == 1


def test_neural_profile_requires_exported_weights(tmp_path, monkeypatch):
    profile = _profile(tmp_path, monkeypatch, policy="neural")

    with pytest.raises(FileNotFoundError, match="exported policy weights"):
        profile.make_agent()


def test_neural_profile_uses_profile_owned_export(tmp_path, monkeypatch):
    profile = _profile(tmp_path, monkeypatch, policy="neural")
    profile.checkpoint_dir.mkdir()
    profile.exported_weights_path.write_bytes(b"profile weights")
    seen = {}

    def factory(deck, weights, **kwargs):
        seen["weights"] = weights
        seen["require_weights"] = kwargs["require_weights"]
        return lambda obs: deck

    monkeypatch.setattr("pkm.agents.factory.make_neural_agent", factory)

    policy = profile.make_agent()

    assert profile.exported_weights_path == profile.checkpoint_dir / "policy.npz"
    assert seen == {
        "weights": str(profile.exported_weights_path),
        "require_weights": True,
    }
    assert policy({"select": None}) == [1] * 60


def test_profile_passes_resolved_profile_to_strategy_factory(tmp_path, monkeypatch):
    profile = _profile(tmp_path, monkeypatch, policy="random", strategy=None)
    seen = {}

    def factory(resolved_profile, deck):
        seen["profile"] = resolved_profile
        seen["deck"] = deck
        return lambda obs: [0]

    monkeypatch.setitem(POLICY_FACTORIES, "test", factory)
    profile.spec = profile.spec.__class__(
        name=profile.spec.name,
        deck_path=profile.spec.deck_path,
        policy="test",
        trainer=profile.spec.trainer,
        strategy=profile.spec.strategy,
        checkpoint_path=profile.spec.checkpoint_path,
    )

    policy = profile.make_agent()

    assert seen == {"profile": profile, "deck": [1] * 60}
    assert policy({"select": {"option": ["move"], "maxCount": 1}}) == [0]


@pytest.mark.parametrize("name", ["unknown", "custom"])
def test_unknown_policy_names_fail_during_profile_loading(tmp_path, monkeypatch, name):
    _write_profile(tmp_path, name=name, policy="not-registered")
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="unknown policy"):
        AgentProfile.load(name)


def test_unknown_strategy_names_fail_during_profile_loading(tmp_path, monkeypatch):
    _write_profile(tmp_path, strategy="not-registered")
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="unknown strategy"):
        AgentProfile.load("custom")


def test_strategy_registry_starts_empty():
    assert STRATEGY_FACTORIES == {}


def _profile(tmp_path, monkeypatch, *, policy, strategy=None):
    _write_profile(tmp_path, policy=policy, strategy=strategy)
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)
    return AgentProfile.load("custom")


def _write_profile(tmp_path: Path, *, name="custom", policy="random", strategy=None):
    profile_dir = tmp_path / "agents" / name
    profile_dir.mkdir(parents=True)
    (profile_dir / "deck.csv").write_text("1\n" * 60)
    strategy_value = "null" if strategy is None else strategy
    (profile_dir / "profile.yaml").write_text(
        f"name: {name}\n"
        f"deck: agents/{name}/deck.csv\n"
        f"policy: {policy}\n"
        "trainer: ppo\n"
        f"checkpoint: agents/{name}/checkpoints/ppo_latest.pt\n"
        f"strategy: {strategy_value}\n"
    )
