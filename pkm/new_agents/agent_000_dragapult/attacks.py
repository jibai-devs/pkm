"""Deterministic attack attribute table + encoder for the option (move) channel.

An option that is an attack carries only an ``attackId``. Like a raw card id,
that integer is meaningless as a scalar — so we look it up in a static
``[max_attack_id + 1, AA]`` attribute matrix (damage, energy cost, cost types)
and encode *that* with a small MLP. This is the attack analogue of the card
attribute channel (:mod:`.cards`): it gives every attack — including
open-vocabulary opponent moves never seen in training — a meaningful vector,
without needing a closed attack vocabulary.

Normalisers come from the deterministic ``spec.json`` (global maxima).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from pkm.new_agents.agent_000_dragapult.cabt import all_attack

_SPEC = json.loads(Path(__file__).with_name("spec.json").read_text())
_MAX_DMG = float(_SPEC["global_max"]["max_damage"])  # 350
_MAX_COST = float(_SPEC["global_max"]["max_energies_per_attack"])
_N_ENERGY = _SPEC["constants"]["n_energy_types"]  # 12

# Attribute column layout of each attack's vector.
ATK_ATTR_COLS = ["damage_norm", "cost_norm"] + [f"energy_{i}" for i in range(_N_ENERGY)]
AA = len(ATK_ATTR_COLS)  # 14

D_ATK = 32  # attack embedding dim (provisional)


@lru_cache(maxsize=1)
def build_attack_attr_table() -> npt.NDArray[np.float32]:
    """Return the ``[max_attack_id + 1, AA]`` attribute matrix (cached).

    Row ``i`` is the attribute vector for attack id ``i`` (all-zero for gaps and
    for id 0 = "no attack"). Column 0 = damage, 1 = energy-cost count, then a
    per-energy-type count of the attack's cost.
    """
    attacks = all_attack()
    max_id = max(a.attackId for a in attacks)
    table = np.zeros((max_id + 1, AA), dtype=np.float32)
    for a in attacks:
        v = np.zeros(AA, dtype=np.float32)
        v[0] = a.damage / _MAX_DMG
        v[1] = len(a.energies) / _MAX_COST
        for e in a.energies:
            v[2 + int(e)] += 1.0
        table[a.attackId] = v
    return table


class AttackEncoder(nn.Module):
    """Attribute-channel move encoder: attack id -> vector (any attack)."""

    def __init__(self, d_atk: int = D_ATK):
        super().__init__()
        self.d_atk = d_atk
        attr = torch.from_numpy(build_attack_attr_table().astype(np.float32))
        self.register_buffer("attr", attr)  # [max_atk_id + 1, AA] static
        self.mlp = nn.Sequential(
            nn.Linear(AA, d_atk), nn.ReLU(), nn.Linear(d_atk, d_atk)
        )

    def forward(self, attack_id: torch.Tensor) -> torch.Tensor:
        # id 0 = "no attack" -> zero attribute row -> a constant (bias) vector.
        return self.mlp(self.attr[attack_id])  # [..., d_atk]
