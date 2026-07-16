import json
from pathlib import Path

import numpy as np

from pkm.rl.deterministic_features import (
    lethal_this_turn,
    retreat_viable,
    type_effectiveness,
)
from pkm.types.obs import Observation, OptionType

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)
OBS = FIXTURE["observations"]

# Hippopotas (1048, energy_type=FIGHTING/6) knows "Bite" (attack id 1514,
# damage 60). Real card/attack data from pkm/data/card_data.py, used to
# build hand-constructed attack decisions -- the real fixture has no
# captured attack-selection observations to reuse.
ATTACKER_ID = 1048
ATTACK_ID = 1514
ATTACK_DAMAGE = 60

# Meowth ex (1071): weakness=FIGHTING(6), resistance=None.
WEAK_TARGET_ID = 1071
# Dusclops (132): weakness=DARKNESS(7), resistance=FIGHTING(6).
RESIST_TARGET_ID = 132
# Dreepy (119): weakness=None, resistance=None.
NEUTRAL_TARGET_ID = 119


def _attack_obs(
    target_id: int, target_hp: int, target_max_hp: int = 300
) -> Observation:
    raw = {
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [
                {"type": int(OptionType.ATTACK), "attackId": ATTACK_ID},
                {"type": int(OptionType.END)},
            ],
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": [],
        "current": {
            "turn": 1,
            "turnActionCount": 1,
            "yourIndex": 0,
            "firstPlayer": -1,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [
                {
                    "active": [
                        {
                            "id": ATTACKER_ID,
                            "serial": 1,
                            "playerIndex": 0,
                            "hp": 100,
                            "maxHp": 100,
                            "appearThisTurn": False,
                            "energies": [6],
                            "energyCards": [],
                            "tools": [],
                            "preEvolution": [],
                        }
                    ],
                    "bench": [],
                    "benchMax": 5,
                    "deckCount": 50,
                    "discard": [],
                    "prize": [],
                    "handCount": 0,
                    "hand": [],
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
                {
                    "active": [
                        {
                            "id": target_id,
                            "serial": 2,
                            "playerIndex": 1,
                            "hp": target_hp,
                            "maxHp": target_max_hp,
                            "appearThisTurn": False,
                            "energies": [],
                            "energyCards": [],
                            "tools": [],
                            "preEvolution": [],
                        }
                    ],
                    "bench": [],
                    "benchMax": 5,
                    "deckCount": 50,
                    "discard": [],
                    "prize": [],
                    "handCount": 0,
                    "hand": None,
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
            ],
        },
        "search_begin_input": None,
    }
    return Observation.model_validate(raw)


# --- lethal_this_turn --------------------------------------------------------


def test_lethal_this_turn_true_when_damage_meets_or_exceeds_hp():
    obs = _attack_obs(WEAK_TARGET_ID, target_hp=ATTACK_DAMAGE)
    out = lethal_this_turn(obs, None)
    assert out.shape == (2,)
    assert out[0] == 1.0  # the OPT_ATTACK option
    assert out[1] == 0.0  # OPT_END is never lethal


def test_lethal_this_turn_false_when_damage_insufficient():
    obs = _attack_obs(WEAK_TARGET_ID, target_hp=ATTACK_DAMAGE + 1)
    out = lethal_this_turn(obs, None)
    assert out[0] == 0.0
    assert out[1] == 0.0


# --- type_effectiveness -------------------------------------------------------


def test_type_effectiveness_weak():
    obs = _attack_obs(WEAK_TARGET_ID, target_hp=200)
    out = type_effectiveness(obs, None)
    assert out[0] == 1.0  # weak: 2.0x multiplier, normalized /2
    assert out[1] == 0.0


def test_type_effectiveness_resisted():
    obs = _attack_obs(RESIST_TARGET_ID, target_hp=200)
    out = type_effectiveness(obs, None)
    assert out[0] == 0.25  # resisted: 0.5x multiplier, normalized /2
    assert out[1] == 0.0


def test_type_effectiveness_neutral():
    obs = _attack_obs(NEUTRAL_TARGET_ID, target_hp=200)
    out = type_effectiveness(obs, None)
    assert out[0] == 0.5  # neutral: 1.0x multiplier, normalized /2
    assert out[1] == 0.0


# --- retreat_viable ------------------------------------------------------------


def test_retreat_viable_true_when_energy_covers_retreat_cost():
    # "9:43": my active has 2 attached energy; my one bench mon (Meowth ex,
    # retreat_cost=1) sits at slot 1 (slot 0 is my active).
    obs = Observation.model_validate(OBS["9:43"])
    out = retreat_viable(obs, None)
    assert out[1] == 1.0
    assert out[0] == 0.0  # my active itself, not a bench slot
    assert np.all(out[9:] == 0.0)  # opponent's side is never mine to retreat


def test_retreat_viable_false_when_energy_insufficient():
    # "8:40": my active has 0 attached energy; 5 bench mons, all
    # retreat_cost=1 -- none are affordable right now.
    obs = Observation.model_validate(OBS["8:40"])
    out = retreat_viable(obs, None)
    assert np.all(out[1:6] == 0.0)
    assert out[0] == 0.0


def test_retreat_viable_zero_for_empty_bench_slots():
    obs = Observation.model_validate(OBS["9:43"])
    out = retreat_viable(obs, None)
    assert np.all(out[2:9] == 0.0)  # my unoccupied bench slots
