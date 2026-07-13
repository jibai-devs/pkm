"""Render options and log entries as text a human can act on.

Resolving an option means turning ``(playerIndex, area, index)`` into an actual
card — the same resolution ``pkm/rl/encoder.py`` performs for embedding IDs, but
yielding names. Card data comes from the live engine via ``pkm.data``;
``replay/cards.json`` is a reference dump, not a runtime dependency.

Every OptionType has a renderer, plus a generic fallback. The fallback is not
decoration: an option type we failed to anticipate must still be pickable, or the
game soft-locks.
"""

from pkm.data import get_attack_data, get_card_data
from pkm.obs import (
    AreaType,
    CardRef,
    EnergyType,
    Log,
    LogType,
    Observation,
    Option,
    OptionType,
    PokemonRef,
)

ENERGY_SYMBOL = {
    EnergyType.COLORLESS: "☆",
    EnergyType.GRASS: "G",
    EnergyType.FIRE: "R",
    EnergyType.WATER: "W",
    EnergyType.LIGHTNING: "L",
    EnergyType.PSYCHIC: "P",
    EnergyType.FIGHTING: "F",
    EnergyType.DARKNESS: "D",
    EnergyType.METAL: "M",
    EnergyType.DRAGON: "N",
    EnergyType.ALL: "*",
    EnergyType.PSYCHIC_DARKNESS: "PD",
}

AREA_NAME = {
    AreaType.DECK: "deck",
    AreaType.HAND: "hand",
    AreaType.TRASH: "discard",
    AreaType.ACTIVE: "active",
    AreaType.BENCH: "bench",
    AreaType.PRIZE: "prize",
    AreaType.STADIUM: "stadium",
}


def card_name(card_id: int | None) -> str:
    if card_id is None:
        return "?"
    card = get_card_data().get(card_id)
    return card.name if card else f"Card#{card_id}"


def energy_cost(energies: list[int]) -> str:
    if not energies:
        return "—"
    return "".join(f"[{ENERGY_SYMBOL.get(e, '?')}]" for e in energies)


def attack_label(attack_id: int) -> str:
    atk = get_attack_data().get(attack_id)
    if atk is None:
        return f"Attack #{attack_id}"
    return f"Attack: {atk.name}  {atk.damage} dmg  {energy_cost(atk.energies)}"


def _card_at(
    obs: Observation, player_index: int, area: int | None, index: int | None
) -> CardRef | PokemonRef | None:
    """Resolve (player, area, index) to a card. Returns None if not resolvable."""
    if area is None or index is None or obs.current is None:
        return None
    try:
        player = obs.current.players[player_index]
        if area == AreaType.DECK:
            deck = obs.select.deck if obs.select else None
            return deck[index] if deck else None
        if area == AreaType.HAND:
            return player.hand[index] if player.hand else None
        if area == AreaType.TRASH:
            return player.discard[index]
        if area == AreaType.ACTIVE:
            return player.active[index]
        if area == AreaType.BENCH:
            return player.bench[index]
        if area == AreaType.PRIZE:
            return player.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
    except (IndexError, KeyError, TypeError):
        return None
    return None


def _place(
    obs: Observation, player_index: int, area: int | None, index: int | None
) -> str:
    """A short 'where' suffix, e.g. '(bench 2)' or "(opponent's active)"."""
    if area is None:
        return ""
    where = AREA_NAME.get(area, f"area {area}")  # AreaType is an IntEnum: int keys match
    owner = ""
    if obs.current is not None and player_index != obs.current.yourIndex:
        owner = "opponent's "
    if area == AreaType.BENCH and index is not None:
        return f" ({owner}{where} {index + 1})"
    return f" ({owner}{where})"


def _target(
    obs: Observation, player_index: int, area: int | None, index: int | None
) -> str:
    card = _card_at(obs, player_index, area, index)
    name = card_name(card.id) if card else "?"
    return f"{name}{_place(obs, player_index, area, index)}"


def _hand_card(obs: Observation, player_index: int, index: int | None) -> str:
    """Name of a hand card, with no '(hand)' suffix.

    Used where the hand is the only possible source (playing/attaching/evolving),
    so tagging the location would just be noise.
    """
    card = _card_at(obs, player_index, AreaType.HAND, index)
    return card_name(card.id) if card else "?"


def option_label(obs: Observation, opt: Option) -> str:
    you = obs.current.yourIndex if obs.current else 0
    owner = you if opt.playerIndex is None else opt.playerIndex
    kind = opt.kind

    if kind is OptionType.YES:
        return "Yes"
    if kind is OptionType.NO:
        return "No"
    if kind is OptionType.END:
        return "End turn"
    if kind is OptionType.RETREAT:
        return "Retreat"
    if kind is OptionType.NUMBER:
        return f"Choose {opt.number}"
    if kind is OptionType.ATTACK:
        return attack_label(opt.attackId) if opt.attackId is not None else "Attack"
    if kind is OptionType.PLAY:
        return f"Play {_hand_card(obs, you, opt.index)}"
    if kind is OptionType.CARD:
        return f"Choose {_target(obs, owner, opt.area, opt.index)}"
    if kind is OptionType.DISCARD:
        return f"Discard {_target(obs, owner, opt.area, opt.index)}"
    if kind is OptionType.ABILITY:
        return f"Ability of {_target(obs, owner, opt.area, opt.index)}"
    if kind is OptionType.ATTACH:
        source = _hand_card(obs, you, opt.index)
        dest = _target(obs, you, opt.inPlayArea, opt.inPlayIndex)
        return f"Attach {source} → {dest}"
    if kind is OptionType.EVOLVE:
        source = _hand_card(obs, you, opt.index)
        dest = _target(obs, you, opt.inPlayArea, opt.inPlayIndex)
        return f"Evolve {dest} → {source}"
    if kind is OptionType.SKILL:
        return f"Use {card_name(opt.cardId)}"
    if kind is OptionType.SPECIAL_CONDITION:
        conditions = ["Poison", "Burn", "Sleep", "Paralyze", "Confuse"]
        i = opt.specialConditionType
        name = conditions[i] if i is not None and i < len(conditions) else "?"
        return f"Inflict {name}"
    if kind in (OptionType.ENERGY, OptionType.ENERGY_CARD):
        holder = _target(obs, owner, opt.area, opt.index)
        pokemon = _card_at(obs, owner, opt.area, opt.index)
        energy = "energy"
        if isinstance(pokemon, PokemonRef) and opt.energyIndex is not None:
            cards = pokemon.energyCards
            if opt.energyIndex < len(cards):
                energy = card_name(cards[opt.energyIndex].id)
        return f"{energy} on {holder}"
    if kind is OptionType.TOOL_CARD:
        holder = _target(obs, owner, opt.area, opt.index)
        pokemon = _card_at(obs, owner, opt.area, opt.index)
        tool = "tool"
        if isinstance(pokemon, PokemonRef) and opt.toolIndex is not None:
            tools = pokemon.tools
            if opt.toolIndex < len(tools):
                tool = card_name(tools[opt.toolIndex].id)
        return f"{tool} on {holder}"

    # Unknown option type: still pickable.
    return f"Option (type={opt.type})"


def _extra(log: Log, key: str) -> int | None:
    value = (log.model_extra or {}).get(key)
    return value if isinstance(value, int) else None


def log_label(obs: Observation, log: Log) -> str:
    """One line for the event feed. 'You'/'Agent' from the viewer's perspective."""
    you = obs.current.yourIndex if obs.current else 0
    player = _extra(log, "playerIndex")
    who = "You" if player == you else "Agent"
    card = card_name(_extra(log, "cardId")) if _extra(log, "cardId") else ""
    kind = log.kind

    if kind is LogType.TURN_START:
        return f"— {who} start turn —"
    if kind is LogType.TURN_END:
        return f"— {who} end turn —"
    if kind is LogType.DRAW:
        return f"{who} drew {card}"
    if kind is LogType.DRAW_REVERSE:
        return f"{who} drew a card"
    if kind is LogType.SHUFFLE:
        return f"{who} shuffled"
    if kind is LogType.PLAY:
        return f"{who} played {card}"
    if kind is LogType.ATTACH:
        target = card_name(_extra(log, "cardIdTarget"))
        return f"{who} attached {card} to {target}"
    if kind is LogType.EVOLVE:
        target = card_name(_extra(log, "cardIdTarget"))
        return f"{who} evolved {target} into {card}"
    if kind is LogType.DEVOLVE:
        return f"{who} devolved {card}"
    if kind is LogType.ATTACK:
        attack_id = _extra(log, "attackId")
        atk = get_attack_data().get(attack_id) if attack_id is not None else None
        name = atk.name if atk else "an attack"
        damage = f" ({atk.damage})" if atk and atk.damage else ""
        return f"{who} attacked with {name}{damage}"
    if kind is LogType.HP_CHANGE:
        value = _extra(log, "value") or 0
        verb = "healed" if value < 0 else "took"
        return f"{card} {verb} {abs(value)} damage"
    if kind in (LogType.MOVE_CARD, LogType.MOVE_CARD_REVERSE):
        what = card if card else "a card"
        from_area = AREA_NAME.get(_extra(log, "fromArea"), "")
        to_area = AREA_NAME.get(_extra(log, "toArea"), "")
        if from_area and to_area:
            return f"{who} moved {what} from {from_area} to {to_area}"
        return f"{who} moved {what}"
    if kind is LogType.SWITCH:
        incoming = card_name(_extra(log, "cardIdBench"))
        outgoing = card_name(_extra(log, "cardIdActive"))
        return f"{who} switched {incoming} in for {outgoing}"
    if kind is LogType.CHANGE:
        return f"{who} replaced their active Pokémon"
    if kind is LogType.MOVE_ATTACHED:
        return f"{who} moved an attached card"
    if kind is LogType.COIN:
        head = (log.model_extra or {}).get("head")
        return f"{who} flipped {'heads' if head else 'tails'}"
    if kind is LogType.HAS_BASIC_POKEMON:
        has = (log.model_extra or {}).get("hasBasicPokemon")
        verb = "have" if who == "You" else "has"
        return f"{who} {verb} a basic Pokémon" if has else f"{who} {verb} no basic Pokémon"
    if kind in (
        LogType.POISONED,
        LogType.BURNED,
        LogType.ASLEEP,
        LogType.PARALYZED,
        LogType.CONFUSED,
    ):
        condition = kind.name.capitalize()
        recovered = (log.model_extra or {}).get("isRecover")
        return f"{card or who}: {condition}{' recovered' if recovered else ''}"
    if kind is LogType.RESULT:
        result = _extra(log, "result")
        if result == 2:
            return "=== Draw ==="
        return "=== You win! ===" if result == you else "=== You lose ==="

    return f"event (type={log.type})"
