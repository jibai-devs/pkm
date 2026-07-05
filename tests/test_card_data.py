"""Test card data loading from cabt engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pkm.data.card_data import (
    get_card_data,
    get_attack_data,
    get_pokemon_cards,
    get_energy_cards,
)


def test_card_data_loads():
    cards = get_card_data()
    assert len(cards) > 0, "Should have cards"
    print(f"Loaded {len(cards)} cards")


def test_attack_data_loads():
    attacks = get_attack_data()
    assert len(attacks) > 0, "Should have attacks"
    print(f"Loaded {len(attacks)} attacks")


def test_pokemon_cards():
    pokemon = get_pokemon_cards()
    assert len(pokemon) > 0, "Should have pokemon cards"
    for p in pokemon[:3]:
        print(f"  Pokemon: id={p.card_id}, name={p.name}, hp={p.hp}, basic={p.basic}")


def test_energy_cards():
    energies = get_energy_cards()
    assert len(energies) > 0, "Should have energy cards"
    for e in energies[:5]:
        print(f"  Energy: id={e.card_id}, name={e.name}, type={e.energy_type}")


if __name__ == "__main__":
    test_card_data_loads()
    test_attack_data_loads()
    test_pokemon_cards()
    test_energy_cards()
    print("All tests passed!")
