"""Deterministic card attribute table for the hybrid card encoder.

Builds a static ``[max_card_id + 1, A]`` matrix of per-card attributes from the
engine's card/attack tables (`all_card_data` / `all_attack`). Row ``i`` is the
attribute vector for card ID ``i`` (all-zero for gaps / id 0 = "no card").

This is the *attribute channel* of the card encoder: it lets any card — including
open-vocabulary opponent cards never seen in training — get a meaningful vector,
in contrast to the learned per-ID embedding which only covers our own 27-vocab.

Normalisers come from the deterministic ``spec.json`` (global maxima), so nothing
here is a magic number. Column layout is documented in ``ATTR_COLS``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pkm.new_agents.agent_000_dragapult.cabt import (
    Attack,
    CardData,
    all_attack,
    all_card_data,
)

_SPEC = json.loads(Path(__file__).with_name("spec.json").read_text())
_MAX_HP = float(_SPEC["global_max"]["max_hp"])  # 380
_MAX_RETREAT = float(_SPEC["global_max"]["max_retreat_cost"])  # 4
_MAX_DMG = float(_SPEC["global_max"]["max_damage"])  # 350
_MAX_ATK = float(_SPEC["global_max"]["max_attacks_per_card"])  # 2
_N_CARD = _SPEC["constants"]["n_card_types"]  # 7
_N_ENERGY = _SPEC["constants"]["n_energy_types"]  # 12

# Documented column layout of each card's attribute vector.
ATTR_COLS = (
    ["hp_norm", "retreat_norm"]
    + [f"cardtype_{i}" for i in range(_N_CARD)]  # one-hot CardType
    + [f"energy_{i}" for i in range(_N_ENERGY)]  # one-hot energyType
    + [f"weak_{i}" for i in range(_N_ENERGY)]  # one-hot weakness (zeros if none)
    + [f"resist_{i}" for i in range(_N_ENERGY)]  # one-hot resistance (zeros if none)
    + ["basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec"]
    + ["max_atk_dmg_norm", "n_attacks_norm"]
)
A = len(ATTR_COLS)


def _attr_vector(cd: CardData, atk_by_id: dict[int, Attack]) -> npt.NDArray[np.float32]:
    v = np.zeros(A, dtype=np.float32)
    i = 0
    v[i] = cd.hp / _MAX_HP
    i += 1
    v[i] = cd.retreatCost / _MAX_RETREAT
    i += 1
    v[i + int(cd.cardType)] = 1.0
    i += _N_CARD
    v[i + int(cd.energyType)] = 1.0
    i += _N_ENERGY
    if cd.weakness is not None:
        v[i + int(cd.weakness)] = 1.0
    i += _N_ENERGY
    if cd.resistance is not None:
        v[i + int(cd.resistance)] = 1.0
    i += _N_ENERGY
    for flag in (cd.basic, cd.stage1, cd.stage2, cd.ex, cd.megaEx, cd.tera, cd.aceSpec):
        v[i] = float(flag)
        i += 1
    dmgs = [atk_by_id[a].damage for a in cd.attacks if a in atk_by_id]
    v[i] = (max(dmgs) / _MAX_DMG) if dmgs else 0.0
    i += 1
    v[i] = len(cd.attacks) / _MAX_ATK
    i += 1
    return v


@lru_cache(maxsize=1)
def build_card_attr_table() -> npt.NDArray[np.float32]:
    """Return the ``[max_id + 1, A]`` attribute matrix (cached)."""
    cards = all_card_data()
    atk_by_id = {a.attackId: a for a in all_attack()}
    max_id = max(c.cardId for c in cards)
    table = np.zeros((max_id + 1, A), dtype=np.float32)
    for cd in cards:
        table[cd.cardId] = _attr_vector(cd, atk_by_id)
    return table
