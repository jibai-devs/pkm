from pathlib import Path

import pytest

from pkm.rl import play


class _Profile:
    deck_path = Path("profile-deck.csv")

    def __init__(self, agent):
        self._agent = agent
        self.calls = []

    def make_agent(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("policy") == "random":
            return "random-agent"
        return self._agent


def test_profile_play_resolution_uses_profile_factory(monkeypatch):
    def expected(obs):
        return [0]

    profile = _Profile(expected)

    def fail_legacy(*args, **kwargs):
        raise AssertionError("profile play must not use legacy factories")

    monkeypatch.setattr(play, "make_neural_agent", fail_legacy)

    assert play.make_agent_by_name("neural", [1] * 60, None, profile=profile) is expected


def test_profile_play_resolution_respects_independent_player_names():
    def neural(obs):
        return [0]

    profile = _Profile(neural)

    p0 = play.make_agent_by_name("neural", [1] * 60, None, profile=profile)
    p1 = play.make_agent_by_name("random", [1] * 60, None, profile=profile)

    assert p0 is neural
    assert p1 == "random-agent"
    assert profile.calls == [{"policy": "neural"}, {"policy": "random"}]


def test_profile_play_resolution_passes_explicit_weights_override():
    profile = _Profile(None)

    play.make_agent_by_name(
        "neural", [1] * 60, "/tmp/override.npz", profile=profile
    )

    assert profile.calls == [
        {"policy": "neural", "weights_path": "/tmp/override.npz"}
    ]


def test_profile_play_resolution_propagates_missing_export_error(monkeypatch):
    error = FileNotFoundError("exported policy weights not found")
    profile = _Profile(None)

    def fail_make_agent(**kwargs):
        raise error

    monkeypatch.setattr(profile, "make_agent", fail_make_agent)

    with pytest.raises(FileNotFoundError, match="exported policy weights"):
        play.make_agent_by_name("neural", [1] * 60, None, profile=profile)


def test_low_level_play_resolution_keeps_legacy_factory(monkeypatch):
    def expected(obs):
        return [0]

    monkeypatch.setattr(play, "make_random_agent", lambda deck: expected)

    assert play.make_agent_by_name("random", [1] * 60, None) is expected
