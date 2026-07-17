"""Deterministically derive the dimensioning constants ("max variables") this
agent needs to size its tensors and embedding tables.

Everything here is derived from the engine's *static* card/attack tables
(``all_card_data`` / ``all_attack``) plus the fixed decklist — so it is fully
deterministic and reproducible (unlike gameplay-derived maxima such as the
number of options in a selection, which are stochastic and bounded at runtime).

Run:  ``uv run python -m pkm.agents.agent_000_dragapult.build_spec``
Writes ``spec.json`` next to this file and prints a summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pkm.cabt.api import CardType, EnergyType, SpecialConditionType, all_attack, all_card_data
from pkm.agents.agent_000_dragapult import deck

SPEC_PATH = Path(__file__).with_name("spec.json")


def build_spec() -> dict[str, Any]:
    cards = all_card_data()
    attacks = all_attack()
    atk_by_id = {a.attackId: a for a in attacks}

    # --- global maxima over the FULL card table (opponent cards can be anything,
    #     so normalisers are global, not deck-only) ---
    global_max = {
        "max_hp": max(c.hp for c in cards),
        "max_retreat_cost": max(c.retreatCost for c in cards),
        "max_attacks_per_card": max((len(c.attacks) for c in cards), default=0),
        "max_damage": max((a.damage for a in attacks), default=0),
        "max_energies_per_attack": max((len(a.energies) for a in attacks), default=0),
    }

    # --- our own deck's maxima (for reference / tighter per-agent bounds) ---
    own_cards = [c for c in cards if c.cardId in set(deck.DISTINCT_IDS)]
    own_attack_ids = {aid for c in own_cards for aid in c.attacks}
    own_attacks = [atk_by_id[i] for i in own_attack_ids if i in atk_by_id]
    own_max = {
        "max_hp": max((c.hp for c in own_cards), default=0),
        "max_retreat_cost": max((c.retreatCost for c in own_cards), default=0),
        "max_attacks_per_card": max((len(c.attacks) for c in own_cards), default=0),
        "max_damage": max((a.damage for a in own_attacks), default=0),
        "max_energies_per_attack": max((len(a.energies) for a in own_attacks), default=0),
        "num_own_attacks": len(own_attack_ids),
    }

    spec = {
        "agent": "agent_000_dragapult",
        "source": {"num_cards": len(cards), "num_attacks": len(attacks)},
        # fixed enum cardinalities (constants, listed so downstream never guesses)
        "constants": {
            "n_energy_types": len(EnergyType),
            "n_card_types": len(CardType),
            "n_special_conditions": len(SpecialConditionType),
            "bench_max": 5,
            "board_slots": 12,  # 2 x (1 active + 5 bench)
        },
        # closed vocabulary tied to this deck
        "vocab": {
            "distinct_ids": deck.DISTINCT_IDS,
            "vocab_size": deck.VOCAB_SIZE,  # includes UNK
            "unk_row": deck.UNK_ROW,
            "deck_size": len(deck.DECK_60),
        },
        "global_max": global_max,
        "own_max": own_max,
    }
    return spec


def main() -> None:
    spec = build_spec()
    SPEC_PATH.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {SPEC_PATH}")
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
