"""Card IDs and priority tables for deck/03_pult_munki.csv.

IDs verified against the card DB (pkm.data.card_data) on 2026-07-19.
"""

# --- Pokemon ---
MUNKIDORI = 112
DREEPY = 119
DRAKLOAK = 120
DRAGAPULT_EX = 121
FEZANDIPITI_EX = 140
BUDEW = 235
MOLTRES = 791
MEOWTH_EX = 1071

DREEPY_LINE = {DREEPY, DRAKLOAK, DRAGAPULT_EX}

# --- Trainers ---
UNFAIR_STAMP = 1080
BUDDY_BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
CRUSHING_HAMMER = 1120
ULTRA_BALL = 1121
POKE_PAD = 1152
BOSSS_ORDERS = 1182
XEROSICS_MACHINATIONS = 1197
CRISPIN = 1198
JUDGE = 1213
LILLIES_DETERMINATION = 1227

# --- Energy card IDs (deck copies) ---
FIRE_ENERGY_CARD = 2
PSYCHIC_ENERGY_CARD = 5
DARK_ENERGY_CARD = 7
ENERGY_CARDS = {FIRE_ENERGY_CARD, PSYCHIC_ENERGY_CARD, DARK_ENERGY_CARD}

# --- EnergyType wire values on PokemonRef.energies (pkm/types/obs.py) ---
ENERGY_FIRE = 2
ENERGY_PSYCHIC = 5
ENERGY_DARK = 7

# Starting-active priority, best first (user-specified strategy).
START_PRIORITY_FIRST = [BUDEW, MUNKIDORI, DREEPY, FEZANDIPITI_EX, MOLTRES, MEOWTH_EX]
START_PRIORITY_SECOND = [BUDEW, DREEPY, MUNKIDORI, MOLTRES, FEZANDIPITI_EX, MEOWTH_EX]


def start_priority_rank(card_id: int, went_first: bool) -> int:
    """Lower is better; unknown cards sort last."""
    order = START_PRIORITY_FIRST if went_first else START_PRIORITY_SECOND
    try:
        return order.index(card_id)
    except ValueError:
        return len(order)


def pokemon_in_play(player: dict) -> list[dict]:
    """All non-empty Pokemon of a raw player dict (active + bench)."""
    out = []
    for p in player.get("active") or []:
        if p:
            out.append(p)
    for p in player.get("bench") or []:
        if p:
            out.append(p)
    return out


def count_in_play(player: dict, card_ids: set[int]) -> int:
    return sum(1 for p in pokemon_in_play(player) if p["id"] in card_ids)


def hand_ids(player: dict) -> list[int]:
    return [c["id"] for c in (player.get("hand") or [])]
