from pathlib import Path

import pytest

from pkm.rl import play


class _Profile:
    deck_path = Path("profile-deck.csv")

    def __init__(self, agent):
        self._agent = agent

    def make_agent(self):
        return self._agent


def test_profile_play_resolution_uses_profile_factory(monkeypatch):
    def expected(obs):
        return [0]

    profile = _Profile(expected)

    def fail_legacy(*args, **kwargs):
        raise AssertionError("profile play must not use legacy factories")

    monkeypatch.setattr(play, "make_neural_agent", fail_legacy)

    assert play.make_agent_by_name("neural", [1] * 60, None, profile=profile) is expected


def test_profile_play_resolution_propagates_missing_export_error(monkeypatch):
    error = FileNotFoundError("exported policy weights not found")
    profile = _Profile(None)

    def fail_make_agent():
        raise error

    monkeypatch.setattr(profile, "make_agent", fail_make_agent)

    with pytest.raises(FileNotFoundError, match="exported policy weights"):
        play.make_agent_by_name("neural", [1] * 60, None, profile=profile)


def test_low_level_play_resolution_keeps_legacy_factory(monkeypatch):
    def expected(obs):
        return [0]

    monkeypatch.setattr(play, "make_random_agent", lambda deck: expected)

    assert play.make_agent_by_name("random", [1] * 60, None) is expected
