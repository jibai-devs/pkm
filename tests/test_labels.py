import json
from pathlib import Path

import pytest

from pkm.obs import Observation, Option
from pkm.tui.labels import energy_cost, log_label, option_label

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)
MAIN = Observation.model_validate(FIXTURE["observations"]["0:0"])


def test_energy_cost_symbols():
    assert energy_cost([5, 5, 0]) == "[P][P][☆]"
    assert energy_cost([]) == "—"


def test_every_real_option_gets_a_nonempty_label():
    # Labels must never be blank: a blank line is an unpickable option.
    for raw in FIXTURE["options"].values():
        label = option_label(MAIN, Option.model_validate(raw))
        assert label and label.strip()


def test_attack_label_names_the_attack_and_damage():
    for raw in FIXTURE["options"].values():
        if raw["type"] == 13:  # OptionType.ATTACK
            label = option_label(MAIN, Option.model_validate(raw))
            assert "Attack:" in label
            assert "dmg" in label
            return
    pytest.skip("no attack option in fixture")


def test_end_and_retreat_labels():
    assert option_label(MAIN, Option.model_validate({"type": 14})) == "End turn"
    assert option_label(MAIN, Option.model_validate({"type": 12})) == "Retreat"


def test_yes_no_labels():
    assert option_label(MAIN, Option.model_validate({"type": 1})) == "Yes"
    assert option_label(MAIN, Option.model_validate({"type": 2})) == "No"


def test_play_option_names_the_hand_card():
    hand = MAIN.me.hand
    assert hand, "fixture should have a visible hand"
    label = option_label(MAIN, Option.model_validate({"type": 7, "index": 0}))
    assert label.startswith("Play ")
    assert len(label) > len("Play ")


def test_unknown_option_type_falls_back_and_stays_pickable():
    label = option_label(MAIN, Option.model_validate({"type": 99}))
    assert "99" in label


def test_every_real_log_gets_a_nonempty_label():
    from pkm.obs import Log

    for raw in FIXTURE["logs"].values():
        assert log_label(MAIN, Log.model_validate(raw)).strip()
