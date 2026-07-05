import json
from dataclasses import dataclass

from kaggle_environments.envs.cabt.cg.sim import lib
import ctypes


@dataclass
class Attack:
    attack_id: int
    name: str
    text: str
    damage: int
    energies: list[int]


@dataclass
class CardData:
    card_id: int
    name: str
    card_type: int
    energy_type: int
    hp: int
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    mega_ex: bool
    tera: bool
    ace_spec: bool
    evolves_from: int | None
    weakness: str | None
    resistance: str | None
    retreat_cost: int
    attacks: list[Attack]
    skills: list[dict]


_CARD_DATA: dict[int, CardData] | None = None
_ATTACK_DATA: dict[int, Attack] | None = None


def _load_card_data() -> dict[int, CardData]:
    global _CARD_DATA
    if _CARD_DATA is not None:
        return _CARD_DATA

    # Load attacks first so we can reference them
    all_attacks = _load_attack_data()

    lib.AllCard.restype = ctypes.c_char_p
    raw = lib.AllCard()
    cards_json = json.loads(raw)

    _CARD_DATA = {}
    for c in cards_json:
        # attacks field is a list of attack IDs
        attack_ids = c.get("attacks", [])
        attacks = [all_attacks[aid] for aid in attack_ids if aid in all_attacks]

        card = CardData(
            card_id=c["cardId"],
            name=c["name"],
            card_type=c["cardType"],
            energy_type=c["energyType"],
            hp=c["hp"],
            basic=c["basic"],
            stage1=c["stage1"],
            stage2=c["stage2"],
            ex=c["ex"],
            mega_ex=c["megaEx"],
            tera=c["tera"],
            ace_spec=c["aceSpec"],
            evolves_from=c.get("evolvesFrom"),
            weakness=c.get("weakness"),
            resistance=c.get("resistance"),
            retreat_cost=c["retreatCost"],
            attacks=attacks,
            skills=c.get("skills", []),
        )
        _CARD_DATA[card.card_id] = card

    return _CARD_DATA


def _load_attack_data() -> dict[int, Attack]:
    global _ATTACK_DATA
    if _ATTACK_DATA is not None:
        return _ATTACK_DATA

    lib.AllAttack.restype = ctypes.c_char_p
    raw = lib.AllAttack()
    attacks_json = json.loads(raw)

    _ATTACK_DATA = {}
    for a in attacks_json:
        attack = Attack(
            attack_id=a["attackId"],
            name=a["name"],
            text=a.get("text", ""),
            damage=a.get("damage", 0),
            energies=a.get("energies", []),
        )
        _ATTACK_DATA[attack.attack_id] = attack

    return _ATTACK_DATA


def get_card_data() -> dict[int, CardData]:
    """Get all card data indexed by card ID."""
    return _load_card_data()


def get_attack_data() -> dict[int, Attack]:
    """Get all attack data indexed by attack ID."""
    return _load_attack_data()


def get_card_by_id(card_id: int) -> CardData | None:
    """Get a specific card by its ID."""
    return _load_card_data().get(card_id)


def get_pokemon_cards() -> list[CardData]:
    """Get all Pokemon cards."""
    return [c for c in _load_card_data().values() if c.card_type == 0]


def get_energy_cards() -> list[CardData]:
    """Get all energy cards."""
    return [c for c in _load_card_data().values() if c.card_type in (5, 6)]


def get_trainer_cards() -> list[CardData]:
    """Get all trainer cards (Item, Tool, Supporter, Stadium)."""
    return [c for c in _load_card_data().values() if c.card_type in (1, 2, 3, 4)]
