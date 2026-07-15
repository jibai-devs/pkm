import json
from pathlib import Path

import pytest

from pkm.types.obs import (
    LogType,
    Observation,
    Option,
    OptionType,
    SelectContext,
    SelectType,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)


@pytest.mark.parametrize("key", sorted(FIXTURE["observations"]))
def test_every_real_observation_parses(key):
    raw = FIXTURE["observations"][key]
    obs = Observation.model_validate(raw)
    assert obs.select is not None
    assert obs.current is not None
    assert len(obs.current.players) == 2


def test_round_trip_drops_nothing():
    raw = FIXTURE["observations"]["0:0"]
    dumped = Observation.model_validate(raw).model_dump(exclude_none=True)
    assert dumped["current"]["turn"] == raw["current"]["turn"]
    assert len(dumped["select"]["option"]) == len(raw["select"]["option"])


def test_kaggle_extra_keys_do_not_break_validation():
    raw = dict(FIXTURE["observations"]["0:0"])
    raw["step"] = 12
    raw["remainingOverageTime"] = 600
    obs = Observation.model_validate(raw)
    assert obs.select is not None


def test_select_enums_are_zero_based():
    # The first prompt of every game is YesNo + IsFirst ("do you go first?").
    assert SelectType.MAIN == 0
    assert SelectType.YES_NO == 9
    assert SelectContext.IS_FIRST == 41
    raw = FIXTURE["observations"].get("9:41")
    if raw is not None:
        sel = Observation.model_validate(raw).select
        assert sel.kind is SelectType.YES_NO
        assert sel.context_kind is SelectContext.IS_FIRST


def test_option_and_log_enums_are_not_offset():
    assert OptionType.ATTACK == 13
    assert OptionType.END == 14
    assert LogType.HP_CHANGE == 16
    assert LogType.RESULT == 23


def test_unknown_option_type_still_parses():
    opt = Option.model_validate({"type": 99})
    assert opt.type == 99
    assert opt.kind is None


def test_hidden_information_is_optional():
    obs = Observation.model_validate(FIXTURE["observations"]["0:0"])
    opponent = obs.opponent
    # The opponent's hand is hidden; prizes may be face-down.
    assert opponent.hand is None or isinstance(opponent.hand, list)
    assert len(opponent.prize) <= 6


def test_me_and_opponent_follow_your_index():
    obs = Observation.model_validate(FIXTURE["observations"]["0:0"])
    you = obs.current.yourIndex
    assert obs.me is obs.current.players[you]
    assert obs.opponent is obs.current.players[1 - you]
