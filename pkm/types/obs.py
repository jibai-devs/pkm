"""Typed observation models for the cabt engine.

The engine hands Python a plain dict (actually a kaggle ``Struct``, a dict
subclass with extra ``step`` / ``remainingOverageTime`` keys). These models are
the typed contract the TUI speaks: parse once at the boundary with
``Observation.model_validate(raw)``, and nothing inward of that sees a dict.

Enum numbering (verified against the live engine):

* ``SelectType`` and ``SelectContext`` are **0-based on the wire**. The API
  serializes them as ``max(0, enum - 1)``, so the 1-based tables in
  ``obs_data_structure/OBSERVATION_SCHEMA.md`` are shifted by one relative to
  what actually arrives. ``SelectType.MAIN == 0``.
* ``OptionType`` and ``LogType`` are **not** offset.

Unknown enum values must never crash validation — the engine may emit an option
type we have not catalogued. So the wire fields stay ``int`` and each model
exposes a ``kind`` property that returns the enum or ``None``.
"""

from enum import IntEnum

from pydantic import BaseModel, ConfigDict


# C++ source: engine/src/core/ApiTypes.h — enum class SelectType : unsigned char
# Wire values are max(0, enum - 1), so None is collapsed and Main=0.
class SelectType(IntEnum):
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


# C++ source: engine/src/core/ApiTypes.h — enum class SelectContext : unsigned char
# Wire values are max(0, enum - 1), so None is collapsed and Main=0.
class SelectContext(IntEnum):
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


# C++ source: engine/src/core/ApiTypes.h — enum class SelectOptionType : unsigned char
# Not offset on wire.
class OptionType(IntEnum):
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


# C++ source: engine/src/core/CardTypes.h — enum class AreaType : unsigned char
# Full enum; values beyond LOOKING are internal to the engine and rarely appear in API JSON.
class AreaType(IntEnum):
    ALL = 0
    DECK = 1
    HAND = 2
    TRASH = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12
    PLAYING = 13
    DECK_BOTTOM = 14
    ME = 15
    EFFECTED = 16
    EFFECTED_PRE_TARGET = 17
    SELECTED_LIST = 18
    TRIGGER_SUBJECT = 19
    TRIGGER_OBJECT = 20
    ATTACH = 21
    TURN_PLAY = 22
    ATTACK_PRE_MY_TURN = 23
    TEMPORARY = 24


# C++ source: engine/src/core/EnergyTypes.h — enum class EnergyType : unsigned short
# These are EnergyTypeIndex values (sequential 0-11), not bitmask values.
class EnergyType(IntEnum):
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
    ALL = 10
    PSYCHIC_DARKNESS = 11


# C++ source: engine/src/core/ApiTypes.h — enum class LogType : unsigned char
# Not offset on wire.
class LogType(IntEnum):
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


# --- Domain constants: vocabulary sizes and game limits ---

NUM_CARDS = 1268
"""Card-ID vocabulary size (id 0 = pad/unknown; real ids start at 1)."""

NUM_ATTACKS = 1557
"""Attack-ID vocabulary size."""

NUM_OPT_TYPES = 17
"""Number of :class:`OptionType` values (including 0)."""

NUM_SELECT_TYPES = 11
"""Number of :class:`SelectType` values (including 0)."""

MAX_BENCH = 8
"""Maximum number of benched Pokémon."""

MAX_HAND = 25
"""Maximum cards in hand."""

N_POKEMON_SLOTS = 2 * (1 + MAX_BENCH)
"""my active + my bench + opp active + opp bench."""

N_BOARD_SLOTS = N_POKEMON_SLOTS + 1
"""Pokémon slots + stadium."""


def _as_enum[E: IntEnum](enum_cls: type[E], value: int | None) -> E | None:
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow")


class CardRef(_Model):
    id: int
    serial: int
    playerIndex: int


class PokemonRef(CardRef):
    hp: int
    maxHp: int
    appearThisTurn: bool = False
    energies: list[int] = []
    energyCards: list[CardRef] = []
    tools: list[CardRef] = []
    preEvolution: list[CardRef] = []


class Option(_Model):
    type: int
    # type-specific; all optional because they vary by OptionType
    attackId: int | None = None
    area: int | None = None
    index: int | None = None
    playerIndex: int | None = None
    inPlayArea: int | None = None
    inPlayIndex: int | None = None
    energyIndex: int | None = None
    toolIndex: int | None = None
    number: int | None = None
    count: int | None = None
    cardId: int | None = None
    serial: int | None = None
    specialConditionType: int | None = None

    @property
    def kind(self) -> OptionType | None:
        return _as_enum(OptionType, self.type)


class Select(_Model):
    type: int
    context: int
    minCount: int
    maxCount: int
    remainDamageCounter: int = 0
    remainEnergyCost: int = 0
    option: list[Option] = []
    deck: list[CardRef | None] | None = None
    contextCard: CardRef | None = None
    effect: CardRef | None = None

    @property
    def kind(self) -> SelectType | None:
        return _as_enum(SelectType, self.type)

    @property
    def context_kind(self) -> SelectContext | None:
        return _as_enum(SelectContext, self.context)

    def forced_picks(self) -> list[int] | None:
        """Return forced selection indices if this decision offers no real choice."""
        n = len(self.option)
        if n == 1 and self.minCount >= 1:
            return [0]
        if n == self.minCount == self.maxCount:
            return list(range(n))
        return None


def forced_picks(sel: dict) -> list[int] | None:
    """Return forced selection indices if the decision offers no real choice.

    Works on a raw select dict so callers don't need to parse the full
    observation to pydantic.  Same logic regardless of whether ``sel`` is
    a dict or a :class:`Select` instance (both expose ``option``,
    ``minCount``, ``maxCount``).
    """
    n = len(sel["option"])
    if n == 1 and sel["minCount"] >= 1:
        return [0]
    if n == sel["minCount"] == sel["maxCount"]:
        return list(range(n))
    return None


class Player(_Model):
    active: list[PokemonRef | None] = []
    bench: list[PokemonRef | None] = []
    benchMax: int = 5
    deckCount: int = 0
    discard: list[CardRef | None] = []
    prize: list[CardRef | None] = []
    handCount: int = 0
    hand: list[CardRef] | None = None  # None = hidden (opponent)
    poisoned: bool = False
    burned: bool = False
    asleep: bool = False
    paralyzed: bool = False
    confused: bool = False

    @property
    def active_pokemon(self) -> PokemonRef | None:
        return self.active[0] if self.active else None

    @property
    def conditions(self) -> list[str]:
        names = ("poisoned", "burned", "asleep", "paralyzed", "confused")
        return [n for n in names if getattr(self, n)]


class Log(_Model):
    type: int

    @property
    def kind(self) -> LogType | None:
        return _as_enum(LogType, self.type)


class GameState(_Model):
    turn: int = 0
    turnActionCount: int = 0
    yourIndex: int = 0
    firstPlayer: int = -1
    supporterPlayed: bool = False
    stadiumPlayed: bool = False
    energyAttached: bool = False
    retreated: bool = False
    result: int = -1
    stadium: list[CardRef | None] = []
    looking: list[CardRef | None] | None = None
    players: list[Player] = []


class Observation(_Model):
    select: Select | None = None
    logs: list[Log] = []
    current: GameState | None = None
    search_begin_input: str | None = None

    @property
    def me(self) -> Player:
        return self.current.players[self.current.yourIndex]

    @property
    def opponent(self) -> Player:
        return self.current.players[1 - self.current.yourIndex]


class SearchState:
    """Typed view over one search result (``{"observation": ..., "searchId": ...}``).

    Tree traversal touches ``search_id`` and the raw dict (``raw_observation``)
    only — near-zero cost. ``observation`` validates to a full :class:`Observation`
    lazily and caches, so the expensive parse is paid once per node, on encode,
    exactly matching the codebase's "dict at the seam, pydantic at the ML
    boundary" design.
    """

    __slots__ = ("_raw", "_obs")

    def __init__(self, raw: dict):
        self._raw = raw
        self._obs: Observation | None = None

    @property
    def search_id(self) -> int:
        return self._raw["searchId"]

    @property
    def raw_observation(self) -> dict:
        return self._raw["observation"]

    @property
    def observation(self) -> Observation:
        if self._obs is None:
            self._obs = Observation.model_validate(self._raw["observation"])
        return self._obs
