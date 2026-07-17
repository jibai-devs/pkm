"""Typed data model for the ``cabt`` engine + dict -> dataclass converters.

**Vendored, self-contained typed API for this agent.** The engine
(:mod:`pkm.engine`) returns observations and card/attack metadata as raw JSON
``dict``s. This module mirrors the engine's data model as frozen, slotted
dataclasses (with real ``IntEnum``s) and provides the ``to_*`` converters plus
:func:`all_card_data` / :func:`all_attack`.

Field names are kept in the engine's original ``camelCase`` so the typed objects
line up 1:1 with the raw JSON. This is a near-verbatim vendoring of the reference
``pkm.cabt.api`` module; the only adaptation is the data source — instead of
calling the native ``lib.AllCard()`` / ``lib.AllAttack()`` directly, it reads the
already-parsed dicts from :func:`pkm.engine.all_cards` / :func:`pkm.engine.all_attacks`
(same ``libcg`` underneath). Battle entry points are re-exported from
:mod:`pkm.engine` so the whole engine surface this agent needs lives behind one
import.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from pkm.engine import all_attacks as _engine_all_attacks
from pkm.engine import all_cards as _engine_all_cards
from pkm.engine import battle_finish, battle_select, battle_start  # noqa: F401  (re-export)

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class _WireEnum(IntEnum):
    """``IntEnum`` tolerant of undocumented engine codes.

    The engine emits some enum codes not in the published docs (observed:
    ``AreaType`` code 14 as the ``toArea`` of ``MOVE_CARD`` logs). Rather than
    raise, synthesize a pseudo-member that still behaves as the right int with a
    readable ``UNKNOWN_<n>`` name and does not pollute the member map.
    """

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, int):
            pseudo = int.__new__(cls, value)
            pseudo._name_ = f"UNKNOWN_{value}"
            pseudo._value_ = value
            return pseudo
        return None


class AreaType(_WireEnum):
    """Where a card is located (``Option.area``, ``Log.fromArea``, ...).

    Source: C++ ``core/CardTypes.h:18`` (``enum class AreaType``), serialized raw.
    """

    ALL = 0
    DECK = 1
    HAND = 2
    DISCARD = 3  # C++ "Trash"
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12
    PLAYING = 13  # item/supporter being played
    DECK_BOTTOM = 14  # bottom of deck
    ME = 15
    EFFECTED = 16
    EFFECTED_PRE_TARGET = 17
    SELECTED_LIST = 18
    TRIGGER_SUBJECT = 19
    TRIGGER_OBJECT = 20
    ATTACH = 21  # the Pokémon a card is attached to
    TURN_PLAY = 22  # cards used this turn
    ATTACK_PRE_MY_TURN = 23
    TEMPORARY = 24


class EnergyType(_WireEnum):
    """Energy types (``Pokemon.energies``, ``CardData.energyType``, ...).

    Source: C++ ``core/EnergyTypes.h`` is a bit-flag (Grass=1<<0 ... All=511);
    the API remaps to this sequential 0-11 index.
    """

    COLORLESS = 0
    GRASS = 1
    FIRE = 2
    WATER = 3
    LIGHTNING = 4
    PSYCHIC = 5
    FIGHTING = 6
    DARKNESS = 7
    METAL = 8
    DRAGON = 9
    RAINBOW = 10  # every type
    TEAM_ROCKET = 11  # Psychic or Darkness


class CardType(_WireEnum):
    """Card categories (``CardData.cardType``).

    Source: C++ ``core/CardTypes.h`` (``enum class CardType``), serialized raw.
    """

    POKEMON = 0
    ITEM = 1
    TOOL = 2
    SUPPORTER = 3
    STADIUM = 4
    BASIC_ENERGY = 5
    SPECIAL_ENERGY = 6


class SpecialConditionType(_WireEnum):
    """Special conditions (``Option.specialConditionType``).

    Source: C++ ``core/CardTypes.h`` (``SelectSpecialConditionType``), raw.
    """

    POISON = 0
    BURN = 1
    SLEEP = 2
    PARALYZE = 3
    CONFUSE = 4


class SelectType(_WireEnum):
    """Category of a selection (``SelectData.type``).

    Source: C++ ``core/ApiTypes.h:10``. The core enum starts with a ``None``
    sentinel that the API omits, so API value = core - 1 (Main -> 0).
    """

    MAIN = 0
    CARD = 1
    ATTACHED_CARD = 2
    CARD_OR_ATTACHED_CARD = 3
    ENERGY = 4
    SKILL = 5
    ATTACK = 6
    EVOLVE = 7
    COUNT = 8
    YES_NO = 9
    SPECIAL_CONDITION = 10


class SelectContext(_WireEnum):
    """Specific purpose of a selection (``SelectData.context``).

    Source: C++ ``core/ApiTypes.h:40``. Like SelectType, the core enum starts
    with a ``None`` sentinel the API omits, so API value = core - 1 (IsFirst -> 41).
    """

    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    SWITCH = 3
    TO_ACTIVE = 4
    TO_BENCH = 5
    TO_FIELD = 6
    TO_HAND = 7
    DISCARD = 8
    TO_DECK = 9
    TO_DECK_BOTTOM = 10
    TO_PRIZE = 11
    NOT_MOVE = 12
    DAMAGE_COUNTER = 13
    DAMAGE_COUNTER_ANY = 14
    DAMAGE = 15
    REMOVE_DAMAGE_COUNTER = 16
    HEAL = 17
    EVOLVES_FROM = 18
    EVOLVES_TO = 19
    DEVOLVE = 20
    ATTACH_FROM = 21
    ATTACH_TO = 22
    DETACH_FROM = 23
    LOOK = 24
    EFFECT_TARGET = 25
    DISCARD_ENERGY_CARD = 26
    DISCARD_TOOL_CARD = 27
    SWITCH_ENERGY_CARD = 28
    DISCARD_CARD_OR_ATTACHED_CARD = 29
    DISCARD_ENERGY = 30
    TO_HAND_ENERGY = 31
    TO_DECK_ENERGY = 32
    SWITCH_ENERGY = 33
    SKILL_ORDER = 34
    ATTACK = 35
    DISABLE_ATTACK = 36
    EVOLVE = 37
    DRAW_COUNT = 38
    DAMAGE_COUNTER_COUNT = 39
    REMOVE_DAMAGE_COUNTER_COUNT = 40
    IS_FIRST = 41
    MULLIGAN = 42
    ACTIVATE = 43
    FIRST_EFFECT = 44
    MORE_DEVOLVE = 45
    COIN_HEAD = 46
    AFFECT_SPECIAL_CONDITION = 47
    RECOVER_SPECIAL_CONDITION = 48


class OptionType(_WireEnum):
    """What a single option refers to (``Option.type``).

    Source: C++ ``core/ApiTypes.h:146`` (``SelectOptionType``), serialized raw.
    """

    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class LogType(_WireEnum):
    """Event kinds emitted in ``Observation.logs`` (``Log.type``).

    Source: C++ ``core/ApiTypes.h:166`` (``enum class LogType``), serialized raw.
    """

    SHUFFLE = 0
    HAS_BASIC_POKEMON = 1
    TURN_START = 2
    TURN_END = 3
    DRAW = 4
    DRAW_REVERSE = 5
    MOVE_CARD = 6
    MOVE_CARD_REVERSE = 7
    SWITCH = 8
    CHANGE = 9
    PLAY = 10
    ATTACH = 11
    EVOLVE = 12
    DEVOLVE = 13
    MOVE_ATTACHED = 14
    ATTACK = 15
    HP_CHANGE = 16
    POISONED = 17
    BURNED = 18
    ASLEEP = 19
    PARALYZED = 20
    CONFUSED = 21
    COIN = 22
    RESULT = 23


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class Card:
    """A basic card entity."""

    id: int
    serial: int
    playerIndex: int


@dataclass(slots=True, frozen=True)
class Pokemon:
    """Detailed Pokémon state on the field."""

    id: int
    serial: int
    hp: int
    maxHp: int
    appearThisTurn: bool
    energies: list[EnergyType]
    energyCards: list[Card]
    tools: list[Card]
    preEvolution: list[Card]
    playerIndex: int | None = None


@dataclass(slots=True, frozen=True)
class PlayerState:
    """A single player's board state."""

    active: list[Pokemon | None]
    bench: list[Pokemon | None]
    benchMax: int
    deckCount: int
    discard: list[Card]
    prize: list[Card | None]
    handCount: int
    hand: list[Card] | None
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool


@dataclass(slots=True, frozen=True)
class State:
    """Whole-board state (``Observation.current``)."""

    turn: int
    turnActionCount: int
    yourIndex: int
    firstPlayer: int
    supporterPlayed: bool
    stadiumPlayed: bool
    energyAttached: bool
    retreated: bool
    result: int
    stadium: list[Card]
    looking: list[Card | None] | None
    players: list[PlayerState]


@dataclass(slots=True, frozen=True)
class Option:
    """A single selectable option. Present fields depend on ``type``."""

    type: OptionType
    number: int | None = None
    area: AreaType | None = None
    index: int | None = None
    playerIndex: int | None = None
    toolIndex: int | None = None
    energyIndex: int | None = None
    count: int | None = None
    inPlayArea: AreaType | None = None
    inPlayIndex: int | None = None
    attackId: int | None = None
    cardId: int | None = None
    serial: int | None = None
    specialConditionType: SpecialConditionType | None = None


@dataclass(slots=True, frozen=True)
class SelectData:
    """The decision presented to the agent (``Observation.select``)."""

    type: SelectType
    context: SelectContext
    minCount: int
    maxCount: int
    remainDamageCounter: int
    remainEnergyCost: int
    option: list[Option]
    deck: list[Card] | None
    contextCard: Card | None
    effect: Card | None


@dataclass(slots=True, frozen=True)
class Log:
    """A single event entry. Present fields depend on ``type``."""

    type: LogType
    playerIndex: int | None = None
    hasBasicPokemon: bool | None = None
    cardId: int | None = None
    serial: int | None = None
    fromArea: AreaType | None = None
    toArea: AreaType | None = None
    cardIdActive: int | None = None
    serialActive: int | None = None
    cardIdBench: int | None = None
    serialBench: int | None = None
    cardIdBefore: int | None = None
    serialBefore: int | None = None
    cardIdAfter: int | None = None
    serialAfter: int | None = None
    cardIdTarget: int | None = None
    serialTarget: int | None = None
    attackId: int | None = None
    value: int | None = None
    putDamageCounter: bool | None = None
    isRecover: bool | None = None
    head: bool | None = None
    result: int | None = None
    reason: int | None = None


@dataclass(slots=True, frozen=True)
class Observation:
    """Current state and available choices presented to the agent."""

    select: SelectData | None
    logs: list[Log]
    current: State | None
    search_begin_input: str | None = None


@dataclass(slots=True, frozen=True)
class SearchState:
    """State of a search process (SDK search API)."""

    observation: Observation
    searchId: int


@dataclass(slots=True, frozen=True)
class Skill:
    """A card's skill / ability."""

    name: str
    text: str


@dataclass(slots=True, frozen=True)
class CardData:
    """Static metadata for a card (SDK ``all_card_data``)."""

    cardId: int
    name: str
    cardType: CardType
    retreatCost: int
    hp: int
    weakness: EnergyType | None
    resistance: EnergyType | None
    energyType: EnergyType
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    megaEx: bool
    tera: bool
    aceSpec: bool
    evolvesFrom: str | None
    skills: list[Skill]
    attacks: list[int]


@dataclass(slots=True, frozen=True)
class Attack:
    """Static metadata for an attack (SDK ``all_attack``)."""

    attackId: int
    name: str
    text: str
    damage: int
    energies: list[EnergyType]


# --------------------------------------------------------------------------- #
# Converters (raw dict -> dataclass)
# --------------------------------------------------------------------------- #


def _area(v: int | None) -> AreaType | None:
    return None if v is None else AreaType(v)


def _card(d: dict) -> Card:
    return Card(id=d["id"], serial=d["serial"], playerIndex=d["playerIndex"])


def _card_or_none(d: dict | None) -> Card | None:
    return None if d is None else _card(d)


def _card_list(lst: list) -> list[Card]:
    """Cards that are always face-up (discard, stadium, attached cards)."""
    return [_card(c) for c in lst]


def _card_list_or_none(lst: list | None) -> list[Card] | None:
    """Face-up cards, or ``None`` (own hand visible / opponent's hidden; deck)."""
    return None if lst is None else [_card(c) for c in lst]


def _nullable_card_list(lst: list) -> list[Card | None]:
    """Cards that may be face-down (``None``), e.g. prize."""
    return [_card_or_none(c) for c in lst]


def _nullable_card_list_or_none(lst: list | None) -> list[Card | None] | None:
    """Possibly-hidden list of possibly-face-down cards, e.g. ``looking``."""
    return None if lst is None else [_card_or_none(c) for c in lst]


def _pokemon(d: dict | None) -> Pokemon | None:
    if d is None:
        return None
    return Pokemon(
        id=d["id"],
        serial=d["serial"],
        hp=d["hp"],
        maxHp=d["maxHp"],
        appearThisTurn=d["appearThisTurn"],
        energies=[EnergyType(e) for e in d["energies"]],
        energyCards=_card_list(d["energyCards"]),
        tools=_card_list(d["tools"]),
        preEvolution=_card_list(d["preEvolution"]),
        playerIndex=d.get("playerIndex"),
    )


def _player(d: dict) -> PlayerState:
    return PlayerState(
        active=[_pokemon(x) for x in d["active"]],
        bench=[_pokemon(x) for x in d["bench"]],
        benchMax=d["benchMax"],
        deckCount=d["deckCount"],
        discard=_card_list(d["discard"]),
        prize=_nullable_card_list(d["prize"]),
        handCount=d["handCount"],
        hand=_card_list_or_none(d["hand"]),
        poisoned=d["poisoned"],
        burned=d["burned"],
        asleep=d["asleep"],
        paralyzed=d["paralyzed"],
        confused=d["confused"],
    )


def _state(d: dict | None) -> State | None:
    if d is None:
        return None
    return State(
        turn=d["turn"],
        turnActionCount=d["turnActionCount"],
        yourIndex=d["yourIndex"],
        firstPlayer=d["firstPlayer"],
        supporterPlayed=d["supporterPlayed"],
        stadiumPlayed=d["stadiumPlayed"],
        energyAttached=d["energyAttached"],
        retreated=d["retreated"],
        result=d["result"],
        stadium=_card_list(d["stadium"]),
        looking=_nullable_card_list_or_none(d["looking"]),
        players=[_player(p) for p in d["players"]],
    )


def _option(d: dict) -> Option:
    sct = d.get("specialConditionType")
    return Option(
        type=OptionType(d["type"]),
        number=d.get("number"),
        area=_area(d.get("area")),
        index=d.get("index"),
        playerIndex=d.get("playerIndex"),
        toolIndex=d.get("toolIndex"),
        energyIndex=d.get("energyIndex"),
        count=d.get("count"),
        inPlayArea=_area(d.get("inPlayArea")),
        inPlayIndex=d.get("inPlayIndex"),
        attackId=d.get("attackId"),
        cardId=d.get("cardId"),
        serial=d.get("serial"),
        specialConditionType=None if sct is None else SpecialConditionType(sct),
    )


def _select(d: dict | None) -> SelectData | None:
    if d is None:
        return None
    return SelectData(
        type=SelectType(d["type"]),
        context=SelectContext(d["context"]),
        minCount=d["minCount"],
        maxCount=d["maxCount"],
        remainDamageCounter=d["remainDamageCounter"],
        remainEnergyCost=d["remainEnergyCost"],
        option=[_option(o) for o in d["option"]],
        deck=_card_list_or_none(d.get("deck")),
        contextCard=_card_or_none(d.get("contextCard")),
        effect=_card_or_none(d.get("effect")),
    )


def _log(d: dict) -> Log:
    return Log(
        type=LogType(d["type"]),
        playerIndex=d.get("playerIndex"),
        hasBasicPokemon=d.get("hasBasicPokemon"),
        cardId=d.get("cardId"),
        serial=d.get("serial"),
        fromArea=_area(d.get("fromArea")),
        toArea=_area(d.get("toArea")),
        cardIdActive=d.get("cardIdActive"),
        serialActive=d.get("serialActive"),
        cardIdBench=d.get("cardIdBench"),
        serialBench=d.get("serialBench"),
        cardIdBefore=d.get("cardIdBefore"),
        serialBefore=d.get("serialBefore"),
        cardIdAfter=d.get("cardIdAfter"),
        serialAfter=d.get("serialAfter"),
        cardIdTarget=d.get("cardIdTarget"),
        serialTarget=d.get("serialTarget"),
        attackId=d.get("attackId"),
        value=d.get("value"),
        putDamageCounter=d.get("putDamageCounter"),
        isRecover=d.get("isRecover"),
        head=d.get("head"),
        result=d.get("result"),
        reason=d.get("reason"),
    )


def to_observation(obs: dict) -> Observation:
    """Convert a raw observation ``dict`` (as returned by the engine) into a
    typed :class:`Observation`.

    ``select`` and ``current`` are ``None`` during the initial deck-selection
    phase, and are converted to ``None`` accordingly.
    """
    return Observation(
        select=_select(obs.get("select")),
        logs=[_log(entry) for entry in obs.get("logs", [])],
        current=_state(obs.get("current")),
        search_begin_input=obs.get("search_begin_input"),
    )


def to_card_data(d: dict) -> CardData:
    """Convert a raw card-data ``dict`` into a typed :class:`CardData`."""
    weakness = d.get("weakness")
    resistance = d.get("resistance")
    return CardData(
        cardId=d["cardId"],
        name=d["name"],
        cardType=CardType(d["cardType"]),
        retreatCost=d["retreatCost"],
        hp=d["hp"],
        weakness=None if weakness is None else EnergyType(weakness),
        resistance=None if resistance is None else EnergyType(resistance),
        energyType=EnergyType(d["energyType"]),
        basic=d["basic"],
        stage1=d["stage1"],
        stage2=d["stage2"],
        ex=d["ex"],
        megaEx=d["megaEx"],
        tera=d["tera"],
        aceSpec=d["aceSpec"],
        evolvesFrom=d.get("evolvesFrom"),
        skills=[Skill(name=s["name"], text=s["text"]) for s in d.get("skills", [])],
        attacks=list(d["attacks"]),
    )


def to_attack(d: dict) -> Attack:
    """Convert a raw attack ``dict`` into a typed :class:`Attack`."""
    return Attack(
        attackId=d["attackId"],
        name=d["name"],
        text=d.get("text", ""),
        damage=d.get("damage", 0),
        energies=[EnergyType(e) for e in d.get("energies", [])],
    )


# --------------------------------------------------------------------------- #
# Global metadata lookups (SDK ``all_card_data`` / ``all_attack``)
# --------------------------------------------------------------------------- #
#
# Backed by :func:`pkm.engine.all_cards` / :func:`pkm.engine.all_attacks`, which
# read the native ``AllCard`` / ``AllAttack`` exports (cached statically in the
# engine). We parse the dicts into typed dataclasses once and cache the result.

_card_data_cache: list[CardData] | None = None
_attack_cache: list[Attack] | None = None


def all_card_data() -> list[CardData]:
    """Return metadata for every card in the engine's card table (cached)."""
    global _card_data_cache
    if _card_data_cache is None:
        _card_data_cache = [to_card_data(d) for d in _engine_all_cards()]
    return list(_card_data_cache)


def all_attack() -> list[Attack]:
    """Return metadata for every attack in the engine's attack table (cached)."""
    global _attack_cache
    if _attack_cache is None:
        _attack_cache = [to_attack(d) for d in _engine_all_attacks()]
    return list(_attack_cache)
