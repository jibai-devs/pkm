# Human TUI Battle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a human play a full match against a trained agent in the terminal via `just play human neural`.

**Architecture:** `env.run()` drives the game on a worker thread; a Textual app owns the screen on the main thread; the "human agent" is a queue bridge that blocks inside kaggle's agent call until the player picks options. The observation is parsed once into pydantic models at the session boundary; nothing inward of that sees a dict.

**Tech Stack:** Python 3.12, pydantic (new dep), Textual (already a dep), kaggle-environments, pytest.

**Spec:** `docs/superpowers/specs/2026-07-13-human-tui-battle-design.md`

---

## Background the engineer needs

Read these before starting — they encode findings that are **not** obvious from the source and were verified by running the engine:

1. **`select.type` and `select.context` are 0-based on the wire, and the tables in
   `obs_data_structure/OBSERVATION_SCHEMA.md` are 1-based.** The API serializes them
   as `max(0, enum - 1)`. So `SelectType.MAIN == 0`, `YES_NO == 9`. The first prompt
   of every game is `(type=9, context=41)` = YesNo + IsFirst ("do you go first?").
   **`OptionType` and `LogType` are NOT offset** — `option.type == 13` is Attack,
   `log.type == 16` is HpChange, `log.type == 23` is Result.

2. **`obs_data_structure/example_obs.json` is hand-written and wrong** (it numbers
   HpChange as 17). Do not test against it. Task 1 captures a real fixture.

3. **kaggle passes the agent a `Struct` (a dict subclass) with two extra keys**,
   `step` and `remainingOverageTime`. Models must allow extra fields.

4. **The engine will time a human out and make them lose.** Each agent call deducts
   its duration from a cumulative 600 s budget; overdraw it and you get status
   `TIMEOUT`, reward `-1`. `env.run` separately aborts after `runTimeout` (2000 s).
   Both are disarmed by passing `actTimeout: 1e9, runTimeout: 1e9` to `make()`.

5. **The TUI must never call `print()`.** kaggle wraps every agent call in
   `redirect_stdout(...)`, which rebinds `sys.stdout` process-wide, and our human
   agent blocks *inside* that block while the player thinks. Textual renders to
   `sys.__stderr__` and is unaffected, but a stray `print()` vanishes into kaggle's
   buffer. Use `textual.log` instead.

---

## File structure

| File | Responsibility |
|---|---|
| `pkm/obs.py` | **Create.** Pydantic models + IntEnums for the observation. The typed contract. |
| `pkm/tui/__init__.py` | **Create.** Package marker. |
| `pkm/tui/labels.py` | **Create.** `Option` → readable text; `Log` → readable text. Card-name resolution. |
| `pkm/tui/session.py` | **Create.** `GameSession` protocol + `ThreadedEnvSession` (the engine hook). |
| `pkm/tui/widgets.py` | **Create.** `BoardPanel`, `HandBar`, `EventLog`, `PromptPane`, `ConfirmScreen`. |
| `pkm/tui/app.py` | **Create.** `BattleApp` — layout, keybindings, worker pump, result screen. |
| `pkm/rl/play.py` | **Modify.** `human` branch in `play_match`; reject `human` in `win_rate`. |
| `pkm/cli/__init__.py` | **Modify.** Help text for `--p0`/`--p1`. |
| `tests/fixtures/capture_observations.py` | **Create.** Script that plays a scripted game and records every distinct shape. |
| `tests/fixtures/observations.json` | **Create (generated).** The real fixture. |
| `tests/test_obs.py`, `tests/test_labels.py`, `tests/test_session.py`, `tests/test_tui_app.py` | **Create.** |

`just play human neural` already works with the current justfile (`play p0 p1 agent`), so **no justfile change is needed**.

---

## Task 0: Add pydantic

**Files:** Modify `pyproject.toml`

- [ ] **Step 1: Add the dependency**

```bash
uv add pydantic
```

- [ ] **Step 2: Verify it imports and the suite still passes**

```bash
python -c "import pydantic; print(pydantic.VERSION)"
just test
```
Expected: a 2.x version prints; existing tests pass.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pydantic for typed observations"
```

---

## Task 1: Capture a real observation fixture

Tests must run against shapes the engine actually emits, not against the hand-written example.

**Files:**
- Create: `tests/fixtures/capture_observations.py`
- Create: `tests/fixtures/observations.json` (generated, committed)

- [ ] **Step 1: Write the capture script**

Create `tests/fixtures/capture_observations.py`:

```python
"""Regenerate tests/fixtures/observations.json from the live engine.

Plays scripted random games and records one observation per distinct
(select.type, select.context) pair, plus one example of every option type and
log type seen. Run:  python tests/fixtures/capture_observations.py
"""

import json
import random
from pathlib import Path

from kaggle_environments.envs.cabt.cg.game import (
    battle_finish,
    battle_select,
    battle_start,
)

from pkm.data import Deck

OUT = Path(__file__).parent / "observations.json"


def capture(seeds: tuple[int, ...] = (3, 11, 42)) -> dict:
    deck = Deck.from_csv("deck/02_dragapult.csv").card_ids
    observations: dict[str, dict] = {}
    options: dict[str, dict] = {}
    logs: dict[str, dict] = {}

    for seed in seeds:
        random.seed(seed)
        obs, _ = battle_start(deck, deck)
        try:
            for _ in range(600):
                for entry in obs["logs"]:
                    logs.setdefault(str(entry["type"]), entry)
                if obs["current"]["result"] >= 0:
                    break
                sel = obs["select"]
                observations.setdefault(f'{sel["type"]}:{sel["context"]}', obs)
                for opt in sel["option"]:
                    options.setdefault(str(opt["type"]), opt)
                picks = random.sample(range(len(sel["option"])), sel["maxCount"])
                obs = battle_select(picks)
        finally:
            battle_finish()

    return {
        "observations": observations,
        "options": options,
        "logs": logs,
    }


if __name__ == "__main__":
    data = capture()
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}: {len(data['observations'])} observations, "
          f"{len(data['options'])} option types, {len(data['logs'])} log types")
```

- [ ] **Step 2: Generate the fixture**

Run: `python tests/fixtures/capture_observations.py`
Expected: prints something like `wrote .../observations.json: 15 observations, 12 option types, 15 log types`. The counts vary by seed; that's fine. It **must** include the `0:0` (Main) observation — check with:

```bash
python -c "import json; d=json.load(open('tests/fixtures/observations.json')); print(sorted(d['observations'])); print(sorted(d['options'], key=int))"
```
Expected: the observation keys include `'0:0'` and the option keys include `'13'` (Attack) and `'14'` (End).

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/capture_observations.py tests/fixtures/observations.json
git commit -m "test: capture real engine observation fixtures"
```

---

## Task 2: Pydantic observation models

**Files:**
- Create: `pkm/obs.py`
- Test: `tests/test_obs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_obs.py`:

```python
import json
from pathlib import Path

import pytest

from pkm.obs import (
    LogType,
    Observation,
    Option,
    OptionType,
    SelectContext,
    SelectType,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)


@pytest.mark.parametrize("key", sorted(FIXTURE["observations"]))
def test_every_real_observation_parses(key):
    raw = FIXTURE["observations"][key]
    obs = Observation.model_validate(raw)
    assert obs.select is not None
    assert obs.current is not None
    assert len(obs.current.players) == 2


def test_round_trip_drops_nothing():
    raw = FIXTURE["observations"]["0:0"]
    dumped = Observation.model_validate(raw).model_dump(exclude_none=True)
    assert dumped["current"]["turn"] == raw["current"]["turn"]
    assert len(dumped["select"]["option"]) == len(raw["select"]["option"])


def test_kaggle_extra_keys_do_not_break_validation():
    raw = dict(FIXTURE["observations"]["0:0"])
    raw["step"] = 12
    raw["remainingOverageTime"] = 600
    obs = Observation.model_validate(raw)
    assert obs.select is not None


def test_select_enums_are_zero_based():
    # The first prompt of every game is YesNo + IsFirst ("do you go first?").
    assert SelectType.MAIN == 0
    assert SelectType.YES_NO == 9
    assert SelectContext.IS_FIRST == 41
    raw = FIXTURE["observations"].get("9:41")
    if raw is not None:
        sel = Observation.model_validate(raw).select
        assert sel.kind is SelectType.YES_NO
        assert sel.context_kind is SelectContext.IS_FIRST


def test_option_and_log_enums_are_not_offset():
    assert OptionType.ATTACK == 13
    assert OptionType.END == 14
    assert LogType.HP_CHANGE == 16
    assert LogType.RESULT == 23


def test_unknown_option_type_still_parses():
    opt = Option.model_validate({"type": 99})
    assert opt.type == 99
    assert opt.kind is None


def test_hidden_information_is_optional():
    obs = Observation.model_validate(FIXTURE["observations"]["0:0"])
    opponent = obs.opponent
    # The opponent's hand is hidden; prizes may be face-down.
    assert opponent.hand is None or isinstance(opponent.hand, list)
    assert len(opponent.prize) <= 6


def test_me_and_opponent_follow_your_index():
    obs = Observation.model_validate(FIXTURE["observations"]["0:0"])
    you = obs.current.yourIndex
    assert obs.me is obs.current.players[you]
    assert obs.opponent is obs.current.players[1 - you]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_obs.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pkm.obs'`

- [ ] **Step 3: Write the models**

Create `pkm/obs.py`:

```python
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
```

Note on `Log`: its per-type fields (`cardId`, `attackId`, `value`, …) arrive as
extra keys, which `extra="allow"` keeps accessible via `log.model_extra`. Task 3
reads them through a helper rather than declaring 24 optional fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_obs.py -q`
Expected: PASS (all parametrized cases).

- [ ] **Step 5: Lint**

Run: `ruff check pkm/ tests/ && ruff format pkm/ tests/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add pkm/obs.py tests/test_obs.py
git commit -m "feat: typed pydantic observation models"
```

---

## Task 3: Human-readable labels

Turns options and logs into text a player can act on. This is the module that makes the game legible; the generic fallback is what guarantees an unknown option type can't soft-lock the UI.

**Files:**
- Create: `pkm/tui/__init__.py`, `pkm/tui/labels.py`
- Test: `tests/test_labels.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_labels.py`:

```python
import json
from pathlib import Path

import pytest

from pkm.obs import Observation, Option
from pkm.tui.labels import energy_cost, log_label, option_label

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)
MAIN = Observation.model_validate(FIXTURE["observations"]["0:0"])


def test_energy_cost_symbols():
    assert energy_cost([5, 5, 0]) == "[P][P][☆]"
    assert energy_cost([]) == "—"


def test_every_real_option_gets_a_nonempty_label():
    # Labels must never be blank: a blank line is an unpickable option.
    for raw in FIXTURE["options"].values():
        label = option_label(MAIN, Option.model_validate(raw))
        assert label and label.strip()


def test_attack_label_names_the_attack_and_damage():
    for raw in FIXTURE["options"].values():
        if raw["type"] == 13:  # OptionType.ATTACK
            label = option_label(MAIN, Option.model_validate(raw))
            assert "Attack:" in label
            assert "dmg" in label
            return
    pytest.skip("no attack option in fixture")


def test_end_and_retreat_labels():
    assert option_label(MAIN, Option.model_validate({"type": 14})) == "End turn"
    assert option_label(MAIN, Option.model_validate({"type": 12})) == "Retreat"


def test_yes_no_labels():
    assert option_label(MAIN, Option.model_validate({"type": 1})) == "Yes"
    assert option_label(MAIN, Option.model_validate({"type": 2})) == "No"


def test_play_option_names_the_hand_card():
    hand = MAIN.me.hand
    assert hand, "fixture should have a visible hand"
    label = option_label(MAIN, Option.model_validate({"type": 7, "index": 0}))
    assert label.startswith("Play ")
    assert len(label) > len("Play ")


def test_unknown_option_type_falls_back_and_stays_pickable():
    label = option_label(MAIN, Option.model_validate({"type": 99}))
    assert "99" in label


def test_every_real_log_gets_a_nonempty_label():
    from pkm.obs import Log

    for raw in FIXTURE["logs"].values():
        assert log_label(MAIN, Log.model_validate(raw)).strip()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_labels.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pkm.tui'`

- [ ] **Step 3: Implement labels**

Create `pkm/tui/__init__.py`:

```python
"""Terminal UI for human-vs-agent battles."""
```

Create `pkm/tui/labels.py`:

```python
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


def _place(obs: Observation, player_index: int, area: int | None, index: int | None) -> str:
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


def _target(obs: Observation, player_index: int, area: int | None, index: int | None) -> str:
    card = _card_at(obs, player_index, area, index)
    name = card_name(card.id) if card else "?"
    return f"{name}{_place(obs, player_index, area, index)}"


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
        return f"Play {_target(obs, you, AreaType.HAND, opt.index)}"
    if kind is OptionType.CARD:
        return f"Choose {_target(obs, owner, opt.area, opt.index)}"
    if kind is OptionType.DISCARD:
        return f"Discard {_target(obs, owner, opt.area, opt.index)}"
    if kind is OptionType.ABILITY:
        return f"Ability of {_target(obs, owner, opt.area, opt.index)}"
    if kind is OptionType.ATTACH:
        source = _target(obs, you, opt.area, opt.index)
        dest = _target(obs, you, opt.inPlayArea, opt.inPlayIndex)
        return f"Attach {source} → {dest}"
    if kind is OptionType.EVOLVE:
        source = _target(obs, you, opt.area, opt.index)
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
    if kind is LogType.MOVE_CARD:
        return f"{who} moved {card}"
    if kind is LogType.MOVE_CARD_REVERSE:
        return f"{who} moved a card"
    if kind is LogType.SWITCH:
        return f"{who} switched Pokémon"
    if kind is LogType.CHANGE:
        return f"{who} replaced their active Pokémon"
    if kind is LogType.MOVE_ATTACHED:
        return f"{who} moved an attached card"
    if kind is LogType.COIN:
        head = (log.model_extra or {}).get("head")
        return f"{who} flipped {'heads' if head else 'tails'}"
    if kind is LogType.HAS_BASIC_POKEMON:
        has = (log.model_extra or {}).get("hasBasicPokemon")
        return f"{who} {'has' if has else 'has no'} basic Pokémon"
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_labels.py -q`
Expected: PASS.

- [ ] **Step 5: Eyeball a real prompt** (labels are for humans; read them)

```bash
python - <<'PY'
import json
from pathlib import Path
from pkm.obs import Observation
from pkm.tui.labels import option_label
d = json.loads(Path("tests/fixtures/observations.json").read_text())
for key, raw in sorted(d["observations"].items()):
    obs = Observation.model_validate(raw)
    print(f"--- select {key} (pick {obs.select.minCount}-{obs.select.maxCount})")
    for i, o in enumerate(obs.select.option[:6]):
        print(f"  [{i + 1}] {option_label(obs, o)}")
PY
```
Expected: every line reads like a game action ("Attack: Phantom Dive  130 dmg  [P][P][☆]", "Attach Basic {P} Energy → Dragapult ex (active)"). No blank labels, no `?` where a card name should be. **If a label reads badly, fix it now** — this is the whole point of the module.

- [ ] **Step 6: Commit**

```bash
git add pkm/tui/__init__.py pkm/tui/labels.py tests/test_labels.py
git commit -m "feat: human-readable option and log labels"
```

---

## Task 4: Game session (engine hook)

The seam between the engine and the UI. `GameSession` is the interface; `ThreadedEnvSession` implements it over `env.run` on a worker thread. Keeping this interface narrow is what makes the fallback (a direct `battle_select` loop) a one-file swap.

**Files:**
- Create: `pkm/tui/session.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_session.py`:

```python
import json
from pathlib import Path

import pytest

from pkm.obs import Observation
from pkm.tui.session import (
    Failed,
    Finished,
    HUMAN,
    Prompt,
    ThreadedEnvSession,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)
RAW_MAIN = FIXTURE["observations"]["0:0"]


def test_human_agent_returns_deck_when_select_is_none():
    session = ThreadedEnvSession(deck=[1] * 60, human_index=0, opponent="random")
    assert session.human_agent({"select": None}) == [1] * 60


def test_human_agent_blocks_then_returns_the_submitted_picks():
    session = ThreadedEnvSession(deck=[1] * 60, human_index=0, opponent="random")

    # Pre-load a pick, then call the agent: it should consume the prompt and
    # return our picks without ever touching the engine.
    session.submit([2])
    picks = session.human_agent(RAW_MAIN)

    assert picks == [2]
    event = session.next_event(timeout=1)
    assert isinstance(event, Prompt)
    assert isinstance(event.obs, Observation)
    assert event.obs.select is not None


def test_quit_aborts_the_blocked_agent():
    from pkm.tui.session import _Abort

    session = ThreadedEnvSession(deck=[1] * 60, human_index=0, opponent="random")
    session.quit()
    with pytest.raises(_Abort):
        session.human_agent(RAW_MAIN)


def test_worker_failure_surfaces_as_a_failed_event():
    session = ThreadedEnvSession(deck=[1] * 60, human_index=0, opponent="random")

    def boom():
        raise RuntimeError("engine exploded")

    session._run_env = boom  # simulate the engine dying
    session.start()
    event = session.next_event(timeout=5)

    assert isinstance(event, Failed)
    assert "engine exploded" in str(event.error)


@pytest.mark.slow
def test_full_game_against_random_through_the_real_engine():
    """Drive a real game with a scripted 'always pick the first options' human.

    This is the only test that can catch a genuine deadlock between the Textual
    thread and env.run.
    """
    from pkm.data import Deck

    deck = Deck.from_csv("deck/02_dragapult.csv").card_ids
    session = ThreadedEnvSession(
        deck=deck,
        human_index=0,
        opponent="random",
        html_path=None,
        replay_path=None,
    )
    session.start()

    prompts = 0
    while True:
        event = session.next_event(timeout=60)
        if isinstance(event, Finished):
            assert event.rewards[0] in (-1, 0, 1)
            break
        if isinstance(event, Failed):
            raise AssertionError(f"session failed: {event.error}")
        assert isinstance(event, Prompt)
        prompts += 1
        select = event.obs.select
        session.submit(list(range(select.minCount or 1))[: select.maxCount] or [0])

    assert prompts > 5, "a real game should ask the human more than a few questions"
    assert HUMAN == "human"
```

- [ ] **Step 2: Register the `slow` marker**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["slow: runs a full game through the real engine"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_session.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pkm.tui.session'`

- [ ] **Step 4: Implement the session**

Create `pkm/tui/session.py`:

```python
"""The engine hook: run a match while a human answers the prompts.

``env.run()`` owns the game loop and calls agent functions synchronously, so a
human player cannot sit on Textual's event-loop thread. Instead ``env.run`` goes
on a worker thread and the "human agent" is a queue bridge: it pushes the parsed
observation to the UI and blocks until the UI posts picks back.

    main thread (Textual)                worker thread (env.run)
      BattleApp                            human_agent(obs)  <- kaggle calls this
         |                                     |  events.put(Prompt)
         |  <---------- events ---------------  |  picks.get()   [blocks]
         |  render, user chooses                |
         |  ----------- picks ---------------> |
         |                                      |  returns [0] -> battle_select

``GameSession`` is deliberately narrow (``start`` / ``next_event`` / ``submit`` /
``quit``). If this threaded approach ever proves unworkable, a session driving
``battle_start`` / ``battle_select`` directly implements the same four methods and
the UI does not change.

Two engine limits are disarmed at ``make()`` (both verified by measurement, see
the design doc): the cumulative 600 s overage clock, which would otherwise hand
the human a loss for thinking, and ``runTimeout``, which would abort the episode.
"""

import json
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Protocol

from pkm.obs import Observation

HUMAN = "human"

# Large enough to disable kaggle's timeouts. actTimeout has minimum 0 in the
# schema, so this must be large-positive, not negative.
NO_TIMEOUT = 1e9


@dataclass(frozen=True)
class Prompt:
    """The engine is asking the human to choose."""

    obs: Observation


@dataclass(frozen=True)
class Finished:
    """The game ended normally."""

    rewards: tuple[int | None, int | None]
    html_path: str | None = None
    replay_path: str | None = None


@dataclass(frozen=True)
class Failed:
    """The worker died. Without this the UI would wait on a queue forever."""

    error: BaseException


Event = Prompt | Finished | Failed


class _Quit:
    """Sentinel posted on the picks queue when the user quits."""


class _Abort(Exception):
    """Raised inside the blocked agent to unwind env.run."""


class GameSession(Protocol):
    human_index: int

    def start(self) -> None: ...
    def next_event(self, timeout: float | None = None) -> Event: ...
    def submit(self, picks: list[int]) -> None: ...
    def quit(self) -> None: ...


class ThreadedEnvSession:
    """Runs kaggle's env.run on a worker thread; bridges the human via queues."""

    def __init__(
        self,
        deck: list[int],
        human_index: int,
        opponent: str,
        weights: str | None = None,
        html_path: str | None = "result.html",
        replay_path: str | None = "replay.json",
    ) -> None:
        self.deck = deck
        self.human_index = human_index
        self.opponent = opponent
        self.weights = weights
        self.html_path = html_path
        self.replay_path = replay_path
        self._events: queue.Queue[Event] = queue.Queue()
        self._picks: queue.Queue[list[int] | _Quit] = queue.Queue()
        self._thread: threading.Thread | None = None

    # -- the human "agent" -------------------------------------------------

    def human_agent(self, obs: dict) -> list[int]:
        """Called by kaggle on the worker thread. Blocks until the user picks.

        Note kaggle has redirect_stdout active for the whole duration of this
        call — never print() from here or anywhere the UI runs.
        """
        if obs["select"] is None:
            return self.deck  # deck submission is not a decision
        self._events.put(Prompt(Observation.model_validate(obs)))
        picks = self._picks.get()
        if isinstance(picks, _Quit):
            raise _Abort
        return picks

    # -- GameSession -------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def next_event(self, timeout: float | None = None) -> Event:
        return self._events.get(timeout=timeout)

    def submit(self, picks: list[int]) -> None:
        self._picks.put(picks)

    def quit(self) -> None:
        self._picks.put(_Quit())

    # -- worker ------------------------------------------------------------

    def _worker(self) -> None:
        try:
            self._events.put(self._run_env())
        except _Abort:
            return  # user quit; unwind quietly
        except BaseException as exc:  # noqa: BLE001 - must reach the screen
            self._events.put(Failed(exc))

    def _run_env(self) -> Finished:
        from kaggle_environments import make

        from pkm.rl.play import make_agent_by_name

        opponent_agent = make_agent_by_name(self.opponent, self.deck, self.weights)
        agents: list[Callable[[dict], list[int]]] = [None, None]  # type: ignore[list-item]
        agents[self.human_index] = self.human_agent
        agents[1 - self.human_index] = opponent_agent

        env = make(
            "cabt",
            configuration={
                "decks": [self.deck, self.deck],
                "actTimeout": NO_TIMEOUT,
                "runTimeout": NO_TIMEOUT,
            },
        )
        env.run(agents)

        final = env.steps[-1]
        if self.html_path:
            with open(self.html_path, "w") as f:
                f.write(env.render(mode="html"))
        if self.replay_path:
            data = env.toJSON()
            with open(self.replay_path, "w") as f:
                f.write(data) if isinstance(data, str) else json.dump(data, f)

        return Finished(
            rewards=(final[0].reward, final[1].reward),
            html_path=self.html_path,
            replay_path=self.replay_path,
        )
```

- [ ] **Step 5: Run the fast tests**

Run: `python -m pytest tests/test_session.py -q -m "not slow"`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the slow integration test — this is the deadlock check**

Run: `python -m pytest tests/test_session.py -q -m slow`
Expected: PASS within ~30 s. If it **hangs**, the queue handshake is wrong — do not
paper over it with a timeout; fix the handshake.

- [ ] **Step 7: Commit**

```bash
git add pkm/tui/session.py tests/test_session.py pyproject.toml
git commit -m "feat: threaded game session bridging the engine and a human"
```

---

## Task 5: Widgets

**Files:**
- Create: `pkm/tui/widgets.py`

No unit test of its own — widgets are exercised by the app pilot test in Task 6. Keep them dumb: they take models and render text.

- [ ] **Step 1: Implement the widgets**

Create `pkm/tui/widgets.py`:

```python
"""Dumb, stateless-ish widgets. They take models and render text."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Label, RichLog, Static

from pkm.obs import Observation, Player, PokemonRef
from pkm.tui.labels import card_name, energy_cost, option_label

HP_BAR_WIDTH = 16


def _hp_bar(pokemon: PokemonRef) -> str:
    if pokemon.maxHp <= 0:
        return ""
    filled = max(0, round(HP_BAR_WIDTH * pokemon.hp / pokemon.maxHp))
    return "█" * filled + "░" * (HP_BAR_WIDTH - filled)


def _pokemon_line(pokemon: PokemonRef | None, prefix: str) -> str:
    if pokemon is None:
        return f"{prefix} —"
    tools = ", ".join(card_name(t.id) for t in pokemon.tools)
    bits = [
        f"{prefix} {card_name(pokemon.id)}",
        f"{pokemon.hp}/{pokemon.maxHp}",
        _hp_bar(pokemon),
        energy_cost(pokemon.energies),
    ]
    if tools:
        bits.append(f"tool: {tools}")
    return "  ".join(bits)


class BoardPanel(Static):
    """One player's side of the board."""

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = title

    def show(self, player: Player) -> None:
        lines = [_pokemon_line(player.active_pokemon, "ACTIVE")]
        bench = [p for p in player.bench if p is not None]
        if bench:
            lines += [_pokemon_line(p, f"BENCH {i + 1}") for i, p in enumerate(bench)]
        else:
            lines.append("BENCH  —")
        prizes_left = sum(1 for p in player.prize if p is not None)
        lines.append(
            f"prizes {prizes_left}  deck {player.deckCount}  "
            f"hand {player.handCount}  discard {len(player.discard)}"
        )
        conditions = player.conditions
        if conditions:
            lines.append("status: " + ", ".join(conditions))
        self.update("\n".join(lines))


class HandBar(Static):
    """The human's hand, as names."""

    def show(self, player: Player) -> None:
        if not player.hand:
            self.update("hand: (empty)")
            return
        self.update(" · ".join(card_name(c.id) for c in player.hand))


class EventLog(RichLog):
    """Scrolling feed of translated log entries."""

    def add(self, line: str) -> None:
        self.write(line)


class PromptPane(Static):
    """The option list. Multi-select via toggling; Enter submits.

    Picks are indices into the *current* select.option list and are cleared on
    every new prompt — the engine has no rollback, so there is no cross-prompt
    state to keep.
    """

    picks: reactive[list[int]] = reactive(list, always_update=True)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.obs: Observation | None = None

    def show(self, obs: Observation) -> None:
        self.obs = obs
        self.picks = []
        self._render()

    def toggle(self, index: int) -> None:
        if self.obs is None or self.obs.select is None:
            return
        if not 0 <= index < len(self.obs.select.option):
            return
        picks = list(self.picks)
        if index in picks:
            picks.remove(index)
        elif len(picks) < self.obs.select.maxCount:
            picks.append(index)
        self.picks = picks
        self._render()

    def is_submittable(self) -> bool:
        if self.obs is None or self.obs.select is None:
            return False
        return self.obs.select.minCount <= len(self.picks) <= self.obs.select.maxCount

    def hint(self) -> str:
        if self.obs is None or self.obs.select is None:
            return ""
        select = self.obs.select
        missing = select.minCount - len(self.picks)
        if missing > 0:
            return f"pick {missing} more"
        if len(self.picks) > select.maxCount:
            return f"pick at most {select.maxCount}"
        return "Enter to confirm"

    def _render(self) -> None:
        if self.obs is None or self.obs.select is None:
            self.update("waiting for the agent…")
            return
        select = self.obs.select
        span = (
            f"pick {select.minCount}"
            if select.minCount == select.maxCount
            else f"pick {select.minCount}-{select.maxCount}"
        )
        self.border_title = f"CHOOSE ({span}) — {self.hint()}"
        lines = []
        for i, option in enumerate(select.option):
            mark = "x" if i in self.picks else " "
            key = str(i + 1) if i < 9 else " "
            lines.append(f"[{mark}] {key}. {option_label(self.obs, option)}")
        self.update("\n".join(lines))


class ConfirmScreen(ModalScreen[bool]):
    """Confirm an irreversible choice (attack / end turn). There is no undo."""

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(self.question)
            yield Button("Confirm", variant="primary", id="yes")
            yield Button("Cancel", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")
```

- [ ] **Step 2: Lint and typecheck imports**

Run: `ruff check pkm/ && python -c "import pkm.tui.widgets"`
Expected: clean, no import error.

- [ ] **Step 3: Commit**

```bash
git add pkm/tui/widgets.py
git commit -m "feat: TUI board, hand, log and prompt widgets"
```

---

## Task 6: The Battle app

**Files:**
- Create: `pkm/tui/app.py`
- Test: `tests/test_tui_app.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tui_app.py`:

```python
import asyncio
import json
import queue
from pathlib import Path

from pkm.obs import Observation
from pkm.tui.app import BattleApp
from pkm.tui.session import Finished, Prompt

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)


class FakeSession:
    """A scripted session: no engine, no threads beyond the app's own pump."""

    def __init__(self, events):
        self.human_index = 0
        self._events = queue.Queue()
        for event in events:
            self._events.put(event)
        self.submitted: list[list[int]] = []
        self.quit_called = False

    def start(self) -> None:
        pass

    def next_event(self, timeout=None):
        return self._events.get(timeout=timeout)

    def submit(self, picks):
        self.submitted.append(picks)

    def quit(self) -> None:
        self.quit_called = True


def _main_obs() -> Observation:
    return Observation.model_validate(FIXTURE["observations"]["0:0"])


def _run(coro):
    asyncio.run(coro)


def test_pressing_a_number_then_enter_submits_that_pick():
    obs = _main_obs()
    session = FakeSession([Prompt(obs), Finished(rewards=(1, -1))])
    app = BattleApp(session, confirm_irreversible=False)

    async def go():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("1")
            await pilot.press("enter")
            await pilot.pause()

    _run(go())
    assert session.submitted == [[0]]


def test_enter_does_not_submit_below_min_count():
    obs = _main_obs()
    obs.select.minCount = 2
    obs.select.maxCount = 2
    session = FakeSession([Prompt(obs)])
    app = BattleApp(session, confirm_irreversible=False)

    async def go():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("1")
            await pilot.press("enter")  # only 1 of 2 picked: must not submit
            await pilot.pause()

    _run(go())
    assert session.submitted == []


def test_finished_event_shows_the_result():
    session = FakeSession([Finished(rewards=(1, -1))])
    app = BattleApp(session, confirm_irreversible=False)

    async def go():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()

    _run(go())
    assert app.result_text is not None
    assert "win" in app.result_text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tui_app.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pkm.tui.app'`

- [ ] **Step 3: Implement the app**

Create `pkm/tui/app.py`:

```python
"""BattleApp — the human's screen.

Never call print() from this module or anything it touches: kaggle wraps the
agent call in redirect_stdout, which is active process-wide while the human is
thinking, so prints vanish into its capture buffer. Use textual.log.
"""

from textual import log
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header

from pkm.obs import Observation, OptionType
from pkm.tui.session import Event, Failed, Finished, GameSession, Prompt
from pkm.tui.labels import log_label
from pkm.tui.widgets import BoardPanel, ConfirmScreen, EventLog, HandBar, PromptPane

IRREVERSIBLE = {OptionType.ATTACK, OptionType.END}


class BattleApp(App[None]):
    """Human vs agent. One prompt pane drives every decision."""

    CSS = """
    Screen { layout: vertical; }
    #board { height: 1fr; }
    #panels { width: 2fr; }
    BoardPanel { border: round $accent; padding: 0 1; height: 1fr; }
    EventLog { border: round $secondary; width: 1fr; }
    HandBar { border: round $accent; padding: 0 1; height: auto; }
    PromptPane { border: round $success; padding: 0 1; height: auto; min-height: 6; }
    #confirm-box { align: center middle; width: 50; height: auto;
                   border: thick $warning; background: $surface; padding: 1 2; }
    """

    BINDINGS = [
        ("q", "quit_game", "Quit"),
        ("enter", "submit", "Confirm"),
        *[(str(n), f"toggle({n - 1})", "") for n in range(1, 10)],
    ]

    def __init__(self, session: GameSession, confirm_irreversible: bool = True) -> None:
        super().__init__()
        self.session = session
        self.confirm_irreversible = confirm_irreversible
        self.result_text: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="board"):
            with Vertical(id="panels"):
                yield BoardPanel("AGENT", id="opponent")
                yield BoardPanel("YOU", id="me")
            yield EventLog(id="events", markup=False)
        yield HandBar(id="hand")
        yield PromptPane(id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#hand", HandBar).border_title = "YOUR HAND"
        self.session.start()
        self.run_worker(self._pump, thread=True, exclusive=True)

    # -- worker pump: blocks on the session, hands events to the UI thread --

    def _pump(self) -> None:
        while True:
            event = self.session.next_event()
            self.call_from_thread(self._handle, event)
            if isinstance(event, (Finished, Failed)):
                return

    def _handle(self, event: Event) -> None:
        if isinstance(event, Prompt):
            self._show(event.obs)
        elif isinstance(event, Finished):
            self._finish(event)
        elif isinstance(event, Failed):
            log.error(f"session failed: {event.error}")
            self.result_text = f"Error: {event.error}"
            self.query_one("#prompt", PromptPane).update(self.result_text)

    def _show(self, obs: Observation) -> None:
        self.query_one("#me", BoardPanel).show(obs.me)
        self.query_one("#opponent", BoardPanel).show(obs.opponent)
        self.query_one("#hand", HandBar).show(obs.me)

        # obs.logs is already a delta: the engine sends only what happened since
        # the last observation. Write all of it, every time.
        events = self.query_one("#events", EventLog)
        for entry in obs.logs:
            events.add(log_label(obs, entry))

        self.query_one("#prompt", PromptPane).show(obs)
        self.sub_title = f"turn {obs.current.turn}"

    def _finish(self, event: Finished) -> None:
        mine = event.rewards[self.session.human_index]
        if mine is None:
            self.result_text = "Game over (no result)"
        elif mine > 0:
            self.result_text = "You win!"
        elif mine < 0:
            self.result_text = "You lose."
        else:
            self.result_text = "Draw."
        artifacts = " · ".join(
            p for p in (event.html_path, event.replay_path) if p
        )
        suffix = f"\nwrote {artifacts}" if artifacts else ""
        self.query_one("#prompt", PromptPane).update(
            f"{self.result_text}{suffix}\n\npress q to exit"
        )

    # -- actions -----------------------------------------------------------

    def action_toggle(self, index: int) -> None:
        self.query_one("#prompt", PromptPane).toggle(index)

    def action_submit(self) -> None:
        prompt = self.query_one("#prompt", PromptPane)
        if not prompt.is_submittable():
            self.bell()
            return
        picks = list(prompt.picks)

        # NB: push_screen_wait() would raise NoActiveWorker here — it may only be
        # awaited inside a worker. Use the callback form instead.
        if self.confirm_irreversible and self._is_irreversible(prompt):
            self.push_screen(
                ConfirmScreen("This can't be undone. Confirm?"),
                lambda confirmed: self._send(picks) if confirmed else None,
            )
            return
        self._send(picks)

    def _send(self, picks: list[int]) -> None:
        self.query_one("#prompt", PromptPane).update("waiting for the agent…")
        self.session.submit(picks)

    def _is_irreversible(self, prompt: PromptPane) -> bool:
        if prompt.obs is None or prompt.obs.select is None:
            return False
        options = prompt.obs.select.option
        return any(options[i].kind in IRREVERSIBLE for i in prompt.picks)

    def action_quit_game(self) -> None:
        self.session.quit()
        self.exit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tui_app.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pkm/tui/app.py tests/test_tui_app.py
git commit -m "feat: BattleApp textual UI"
```

---

## Task 7: Wire `human` into `pkm play`

**Files:**
- Modify: `pkm/rl/play.py`
- Modify: `pkm/cli/__init__.py` (help text only)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_session.py`:

```python
def test_win_rate_rejects_human():
    from pkm.rl.play import win_rate

    with pytest.raises(ValueError, match="human"):
        win_rate("human", "random", games=5)


def test_make_agent_by_name_rejects_human():
    # human needs a session to talk to; it cannot be built as a plain agent.
    from pkm.rl.play import make_agent_by_name

    with pytest.raises(ValueError, match="human"):
        make_agent_by_name("human", [1] * 60, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session.py -q -k human`
Expected: FAIL — `make_agent_by_name` raises the generic "unknown agent" message, and `win_rate` starts a real game.

- [ ] **Step 3: Modify `pkm/rl/play.py`**

Replace the `make_agent_by_name` function (currently at `pkm/rl/play.py:21-32`) with:

```python
def make_agent_by_name(
    name: str, deck: list[int], weights: str | None
) -> Callable[[dict], list[int]]:
    if name == HUMAN:
        raise ValueError(
            "human has no standalone agent: it needs a TUI session "
            "(handled by play_match)"
        )
    if name == "random":
        return make_random_agent(deck)
    if name == "neural":
        return make_neural_agent(deck, weights)
    if name == "mcts":
        from pkm.mcts.agent import make_mcts_agent

        return make_mcts_agent(deck, weights_path=weights)
    raise ValueError(f"unknown agent: {name!r} (expected random|neural|mcts|human)")
```

Add the import near the top of the file (after the existing `from pkm.data import Deck`):

```python
from pkm.tui.session import HUMAN
```

Add this function above `play_match`:

```python
def play_human_match(
    p0: str,
    p1: str,
    deck_path: str = "deck/02_dragapult.csv",
    weights: str | None = None,
    html_path: str | None = "result.html",
    replay_path: str | None = "replay.json",
) -> None:
    """Play one match with a human at the keyboard, in a Textual TUI."""
    from pkm.tui.app import BattleApp
    from pkm.tui.session import ThreadedEnvSession

    if p0 == HUMAN and p1 == HUMAN:
        raise ValueError("only one human player is supported")

    human_index = 0 if p0 == HUMAN else 1
    opponent = p1 if human_index == 0 else p0

    deck = Deck.from_csv(deck_path).card_ids
    session = ThreadedEnvSession(
        deck=deck,
        human_index=human_index,
        opponent=opponent,
        weights=weights,
        html_path=html_path,
        replay_path=replay_path,
    )
    BattleApp(session).run()
```

At the top of `play_match`, add the dispatch:

```python
    if HUMAN in (p0, p1):
        return play_human_match(
            p0,
            p1,
            deck_path=deck_path,
            weights=weights,
            html_path=html_path,
            replay_path=replay_path,
        )
```

At the top of `win_rate`, add:

```python
    if HUMAN in (p0, p1):
        raise ValueError("human play does not support --games > 1")
```

- [ ] **Step 4: Update the CLI help text**

In `pkm/cli/__init__.py`, in the `play` command, change the two option helps:

```python
    p0: str = typer.Option("neural", help="player 0 agent: random|neural|mcts|human"),
    p1: str = typer.Option("random", help="player 1 agent: random|neural|mcts|human"),
```

And in `pkm/rl/play.py`, the same two options in its `main`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py -q -k human`
Expected: PASS (2 tests).

- [ ] **Step 6: Verify the whole suite is green**

Run: `just test && ruff check pkm/ tests/`
Expected: all tests pass, lint clean.

- [ ] **Step 7: Commit**

```bash
git add pkm/rl/play.py pkm/cli/__init__.py tests/test_session.py
git commit -m "feat: play as human via 'just play human neural'"
```

---

## Task 8: Play it, then document it

A TUI that passes its tests can still be unusable. Play a real game before calling this done.

- [ ] **Step 1: Play a real game against random**

Run: `just play human random`
Expected, and check each one:
- The board shows both actives with HP bars, benches, and counters.
- The first prompt is the coin-flip-ish "do you go first?" YesNo.
- `1`-`9` toggle, `Enter` confirms, attacks/end-turn ask for confirmation.
- Options read like game actions, not like raw indices.
- The game reaches a result screen and reports `result.html` / `replay.json`.
- `q` exits cleanly at any point, with no traceback and no hung process.

- [ ] **Step 2: Play a real game against the trained agent**

Run: `just play human neural`
Expected: same, and the agent's moves appear in the event log.

- [ ] **Step 3: Confirm the replay artifacts are real**

```bash
just replay-react
```
Expected: the React viewer opens your hand-played game.

- [ ] **Step 4: Update AGENTS.md**

Add to the "Custom Agents" section and to Project Structure:

```markdown
## Human Play (TUI)
Play against a trained agent yourself, in the terminal:
```bash
just play human neural            # you vs the neural agent (both 02_dragapult)
just play human random            # you vs random
just play neural human 01_psychic # you as player 2
```
`1`-`9` toggle options, `Enter` confirms (irreversible moves ask twice), `q` quits.
The match writes `result.html` + `replay.json` like any other, so you can rewatch
it in the React replay viewer.

Implementation: `pkm/tui/` (Textual), typed observations in `pkm/obs.py`.
Two kaggle limits are disarmed for human play — a cumulative 600s "overage clock"
that would otherwise hand you a loss for thinking, and `runTimeout`. See
`docs/superpowers/specs/2026-07-13-human-tui-battle-design.md`.
```

Add to Project Structure:
```markdown
- `pkm/obs.py` — pydantic models for the observation (typed contract for the TUI)
- `pkm/tui/` — Textual human-vs-agent battle UI (session, labels, widgets, app)
```

- [ ] **Step 5: Update CLAUDE.md Active Context**

Replace the stale worktree-related lines with:

```markdown
- Human TUI battle: `just play human neural`. Code in `pkm/tui/`, typed obs in `pkm/obs.py`.
- `select.type`/`select.context` are 0-based on the wire (the schema doc's tables are 1-based); `OptionType`/`LogType` are not offset.
- Human play must disarm kaggle's 600s overage clock + runTimeout (`actTimeout/runTimeout = 1e9`), else the player loses on time.
```

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md CLAUDE.md
git commit -m "docs: document human TUI battle mode"
```

---

## Self-review notes

**Spec coverage:** every spec section maps to a task — models (Task 2), labels
(Task 3), session + three failure paths (Task 4), widgets/interaction (Task 5),
app (Task 6), CLI (Task 7), testing (Tasks 2/3/4/6), docs (Task 8). The fixture
requirement added during spec correction is Task 1.

**Known gaps, accepted deliberately:**
- No undo (the engine has no rollback) — mitigated by the confirm step.
- Quitting mid-game leaks the C battle (kaggle only calls `battle_finish()` on a
  clean finish). Acceptable: the process exits immediately, one game per process.
- `pkm/rl/encoder.py` keeps its own bare-int `OPT_*` / `AREA_*` constants. Not
  unified with `pkm/obs.py`: encoder is the hot PPO/MCTS rollout path and is out
  of scope for this change.
