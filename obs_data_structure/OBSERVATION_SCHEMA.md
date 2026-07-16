# Pokemon TCG AI Battle Simulator — Observation JSON Schema

This document describes the complete JSON data structure returned by `GetBattleData()` and passed to Python via ctypes as the `obs` dict.

> **Source files:** `ToJson.h`, `ApiJson.h`, `ApiTypes.h`, `CardTypes.h`, `EnergyTypes.h`

---

## Top-Level Structure

```json
{
  "select": { ... } | null,
  "logs": [ ... ],
  "current": { ... },
  "search_begin_input": "..." | null
}
```

| Field                | Type             | Description                                                                                           |
|----------------------|------------------|-------------------------------------------------------------------------------------------------------|
| `select`             | `object \| null` | The current action the player must take. `null` when the game is over or it's not this player's turn. |
| `logs`               | `array<Log>`     | Array of game events that occurred since the last observation.                                        |
| `current`            | `object`         | Full current game state snapshot.                                                                     |
| `search_begin_input` | `string \| null` | **Added by Python** (not from C++). Base64-encoded binary state blob for the agent/search API.        |

---

## `select` — Action Prompt

Tells the agent what kind of decision to make and what options are available.

```json
{
  "type": 1,
  "context": 1,
  "minCount": 0,
  "maxCount": 1,
  "remainDamageCounter": 0,
  "remainEnergyCost": 0,
  "option": [ ... ],
  "deck": null,
  "contextCard": null,
  "effect": null
}
```

| Field                 | Type                     | Description                                                                                              |
|-----------------------|--------------------------|----------------------------------------------------------------------------------------------------------|
| `type`                | `int`                    | The `SelectType` enum (0-indexed, with `None` subtracted). See [SelectType table](#selecttype).          |
| `context`             | `int`                    | The `SelectContext` enum (0-indexed, with `None` subtracted). See [SelectContext table](#selectcontext). |
| `minCount`            | `int`                    | Minimum number of options the agent must select.                                                         |
| `maxCount`            | `int`                    | Maximum number of options the agent can select.                                                          |
| `remainDamageCounter` | `int`                    | Remaining damage counters to place (used in `Count`-type selections).                                    |
| `remainEnergyCost`    | `int`                    | Remaining energy cost to satisfy (used in energy discard selections).                                    |
| `option`              | `array<Option>`          | Available choices for the agent.                                                                         |
| `deck`                | `array<CardRef> \| null` | Cards visible from the deck during deck-selection contexts. `null` normally.                             |
| `contextCard`         | `CardRef \| null`        | The card that triggered the current selection (e.g., ability activation).                                |
| `effect`              | `CardRef \| null`        | The card whose effect is currently being resolved.                                                       |

### `option` — Selectable Choices

Each option has a `type` field indicating its kind, plus type-specific parameters.

```json
{"type": 13, "attackId": 42}
```

| OptionType         | Value | Extra Fields                                           | Description                                            |
|--------------------|-------|--------------------------------------------------------|--------------------------------------------------------|
| `Number`           | 0     | `number: int`                                          | A numeric choice (e.g., damage counter count).         |
| `Yes`              | 1     | —                                                      | Affirmative choice.                                    |
| `No`               | 2     | —                                                      | Negative/decline choice.                               |
| `Card`             | 3     | `area`, `index`, `playerIndex`                         | A card selection from a specific area.                 |
| `ToolCard`         | 4     | `area`, `index`, `playerIndex`, `toolIndex`            | A tool card attached to a Pokemon.                     |
| `EnergyCard`       | 5     | `area`, `index`, `playerIndex`, `energyIndex`          | A specific energy card on a Pokemon.                   |
| `Energy`           | 6     | `area`, `index`, `playerIndex`, `energyIndex`, `count` | Energy type selection with count.                      |
| `Play`             | 7     | `index: int`                                           | Play a card from hand (index in hand).                 |
| `Attach`           | 8     | `area`, `index`, `inPlayArea`, `inPlayIndex`           | Attach a card to an in-play Pokemon.                   |
| `Evolve`           | 9     | `area`, `index`, `inPlayArea`, `inPlayIndex`           | Evolve a Pokemon.                                      |
| `Ability`          | 10    | `area`, `index`                                        | Activate an ability.                                   |
| `Discard`          | 11    | `area`, `index`                                        | Discard a card.                                        |
| `Retreat`          | 12    | —                                                      | Retreat the active Pokemon.                            |
| `Attack`           | 13    | `attackId: int`                                        | Use an attack. `attackId` maps to the attack database. |
| `End`              | 14    | —                                                      | End the turn / pass.                                   |
| `Skill`            | 15    | `cardId`, `serial`                                     | Use a skill (ability/play/attach).                     |
| `SpecialCondition` | 16    | `specialConditionType: int`                            | Choose a special condition to inflict.                 |

#### Card Option Field Meanings

| Field         | Type  | Description                                                                                                                   |
|---------------|-------|-------------------------------------------------------------------------------------------------------------------------------|
| `area`        | `int` | `AreaType` enum value: 1=Deck, 2=Hand, 3=Trash, 4=Active, 5=Bench, 6=Prize, 7=Stadium, 8=Energy, 9=Tool, 10=PreEvolution, ... |
| `index`       | `int` | Index of the card within that area (0-based).                                                                                 |
| `playerIndex` | `int` | 0=Player 0, 1=Player 1.                                                                                                       |
| `inPlayArea`  | `int` | For attach/evolve: the area of the target Pokemon (4=Active, 5=Bench).                                                        |
| `inPlayIndex` | `int` | For attach/evolve: index of the target Pokemon within that area.                                                              |
| `energyIndex` | `int` | Index of the energy among attached energies on a Pokemon.                                                                     |
| `toolIndex`   | `int` | Index of the tool among attached tools on a Pokemon.                                                                          |

---

## `current` — Game State Snapshot

```json
{
  "turn": 6,
  "turnActionCount": 2,
  "yourIndex": 0,
  "firstPlayer": 0,
  "supporterPlayed": false,
  "stadiumPlayed": false,
  "energyAttached": true,
  "retreated": false,
  "result": -1,
  "stadium": [],
  "looking": null,
  "players": [ ... ]
}
```

| Field             | Type                             | Description                                                                                                                                      |
|-------------------|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `turn`            | `int`                            | Current turn number (starts at 1).                                                                                                               |
| `turnActionCount` | `int`                            | Number of actions taken this turn.                                                                                                               |
| `yourIndex`       | `int`                            | Which player should act: `0` or `1`.                                                                                                             |
| `firstPlayer`     | `int`                            | Which player went first: `0` or `1`.                                                                                                             |
| `supporterPlayed` | `bool`                           | Whether a Supporter has been played this turn.                                                                                                   |
| `stadiumPlayed`   | `bool`                           | Whether a Stadium has been played this turn.                                                                                                     |
| `energyAttached`  | `bool`                           | Whether an energy has been attached this turn (from hand).                                                                                       |
| `retreated`       | `bool`                           | Whether the active Pokemon has retreated this turn.                                                                                              |
| `result`          | `int`                            | Game result: `-1` = ongoing, `0` = player 0 wins, `1` = player 1 wins, `2` = draw.                                                               |
| `stadium`         | `array<CardRef>`                 | Currently in-play Stadium card(s). Usually 0 or 1.                                                                                               |
| `looking`         | `array<CardRef \| null> \| null` | Cards currently being looked at (by abilities like search effects). `null` if not looking. Array may contain `null` entries for face-down cards. |
| `players`         | `array<Player>`                  | Exactly 2 player state objects.                                                                                                                  |

---

## `players[i]` — Player State

```json
{
  "active": [ ... ],
  "bench": [ ... ],
  "benchMax": 5,
  "deckCount": 38,
  "discard": [ ... ],
  "prize": [ ... ],
  "handCount": 5,
  "hand": [ ... ] | null,
  "poisoned": false,
  "burned": true,
  "asleep": false,
  "paralyzed": false,
  "confused": false
}
```

| Field       | Type                        | Description                                                                                                    |
|-------------|-----------------------------|----------------------------------------------------------------------------------------------------------------|
| `active`    | `array<PokemonRef \| null>` | Pokemon in the active (battle) spot. Usually 1 entry. `null` if face-down and not in addName mode.             |
| `bench`     | `array<PokemonRef \| null>` | Pokemon on the bench. Up to `benchMax` entries.                                                                |
| `benchMax`  | `int`                       | Maximum bench size (default 5, can be modified by effects up to 8).                                            |
| `deckCount` | `int`                       | Number of cards remaining in the deck.                                                                         |
| `discard`   | `array<CardRef \| null>`    | Cards in the discard pile (trash).                                                                             |
| `prize`     | `array<CardRef \| null>`    | Prize cards. Always 6 slots; `null` for taken prizes. Opponent's prizes show `null` for face-down cards.       |
| `handCount` | `int`                       | Number of cards in hand.                                                                                       |
| `hand`      | `array<CardRef> \| null`    | Cards in hand. **`null` for the opponent's hand** (hidden information). Full list only for the viewing player. |
| `poisoned`  | `bool`                      | Whether this player's active Pokemon is poisoned.                                                              |
| `burned`    | `bool`                      | Whether this player's active Pokemon is burned.                                                                |
| `asleep`    | `bool`                      | Whether this player's active Pokemon is asleep.                                                                |
| `paralyzed` | `bool`                      | Whether this player's active Pokemon is paralyzed.                                                             |
| `confused`  | `bool`                      | Whether this player's active Pokemon is confused.                                                              |

---

## CardRef — Card Reference

Represents a specific card instance. `null` when the card is face-down (reverse) and not in addName mode.

```json
{"id": 101, "serial": 5, "playerIndex": 0}
```

| Field         | Type  | Description                                                                                   |
|---------------|-------|-----------------------------------------------------------------------------------------------|
| `id`          | `int` | The card's master ID from the card database. Maps to `AllCard[].cardId`.                      |
| `serial`      | `int` | Unique instance index for this card in the current game. Used to distinguish identical cards. |
| `playerIndex` | `int` | Owner: `0` or `1`.                                                                            |

---
  
## PokemonRef — Pokemon Reference

Extends `CardRef` with live battle state. `null` when face-down and not in addName mode.

```json
{
  "id": 101,
  "serial": 5,
  "playerIndex": 0,
  "hp": 170,
  "maxHp": 200,
  "appearThisTurn": false,
  "energies": [1, 1, 0],
  "energyCards": [ ... ],
  "tools": [ ... ],
  "preEvolution": [ ... ]
}
```

| Field            | Type             | Description                                                                                                                       |
|------------------|------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `id`             | `int`            | Card master ID.                                                                                                                   |
| `serial`         | `int`            | Unique instance index.                                                                                                            |
| `playerIndex`    | `int`            | Owner: `0` or `1`.                                                                                                                |
| `hp`             | `int`            | Current remaining HP.                                                                                                             |
| `maxHp`          | `int`            | Maximum HP (may be modified by effects).                                                                                          |
| `appearThisTurn` | `bool`           | Whether this Pokemon was placed this turn (can't evolve or retreat normally).                                                     |
| `energies`       | `array<int>`     | Energy types currently attached, as `EnergyTypeIndex` values. Each entry is one energy unit. See [EnergyType table](#energytype). |
| `energyCards`    | `array<CardRef>` | The actual energy cards attached (for identifying specific energy cards).                                                         |
| `tools`          | `array<CardRef>` | Tool cards attached to this Pokemon.                                                                                              |
| `preEvolution`   | `array<CardRef>` | Evolution chain beneath this Pokemon (face-up only). First = basic, then stage1, stage2.                                          |

---

## `logs` — Game Event Log

Array of game events. Each log entry has a `type` field and type-specific data.

```json
[
  {"type": 2, "playerIndex": 0},
  {"type": 4, "playerIndex": 0, "cardId": 101, "serial": 5},
  {"type": 15, "playerIndex": 0, "cardId": 101, "serial": 5, "attackId": 42}
]
```

### LogType Enum

> **C++ source:** `engine/src/core/ApiTypes.h` — `enum class LogType : unsigned char`

```cpp
enum class LogType : unsigned char {
  Shuffle, HasBasicPokemon, TurnStart, TurnEnd, Draw, DrawReverse,
  MoveCard, MoveCardReverse, Switch, Change, Play, Attach, Evolve,
  Devolve, MoveAttached, Attack, HpChange, Poisoned, Burned, Asleep,
  Paralyzed, Confused, Coin, Result,
};
```

> Values are implicit (auto-incremented from 0).

| Value | Name              | Fields                                                                                          | Description                                                                                                                    |
|-------|-------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| 0     | `Shuffle`         | `playerIndex`                                                                                   | Deck was shuffled.                                                                                                             |
| 1     | `HasBasicPokemon` | `playerIndex`, `hasBasicPokemon`                                                                | Mulligan check result.                                                                                                         |
| 2     | `TurnStart`       | `playerIndex`                                                                                   | A player's turn begins.                                                                                                        |
| 3     | `TurnEnd`         | `playerIndex`                                                                                   | A player's turn ends.                                                                                                          |
| 4     | `Draw`            | `playerIndex`, `cardId`, `serial`                                                               | A card was drawn (visible). Opponent draws appear as `DrawReverse` (type 5).                                                   |
| 5     | `DrawReverse`     | `playerIndex`                                                                                   | Opponent drew a card (card details hidden).                                                                                    |
| 6     | `MoveCard`        | `playerIndex`, `cardId`, `serial`, `fromArea`, `toArea`                                         | A card moved between areas. Opponent's face-down moves appear as `MoveCardReverse` (type 7).                                   |
| 7     | `MoveCardReverse` | `playerIndex`, `fromArea`, `toArea`                                                             | Opponent moved a face-down card (card details hidden).                                                                         |
| 8     | `Switch`          | `playerIndex`, `cardIdActive`, `serialActive`, `cardIdBench`, `serialBench`                     | Active and bench Pokemon were switched.                                                                                        |
| 9     | `Change`          | `playerIndex`, `cardIdBefore`, `serialBefore`, `cardIdAfter`, `serialAfter`                     | Pokemon was replaced (e.g., by KO).                                                                                            |
| 10    | `Play`            | `playerIndex`, `cardId`, `serial`                                                               | A trainer/supporter/stadium was played.                                                                                        |
| 11    | `Attach`          | `playerIndex`, `cardId`, `serial`, `cardIdTarget`, `serialTarget`                               | A card was attached to a Pokemon (includes target).                                                                            |
| 12    | `Evolve`          | `playerIndex`, `cardId`, `serial`, `cardIdTarget`, `serialTarget`                               | A Pokemon evolved.                                                                                                             |
| 13    | `Devolve`         | `playerIndex`, `cardId`, `serial`, `cardIdTarget`, `serialTarget`                               | A Pokemon devolved.                                                                                                            |
| 14    | `MoveAttached`    | `playerIndex`, `cardId`, `serial`, `cardIdBefore`, `serialBefore`, `cardIdAfter`, `serialAfter` | An attached card was moved between Pokemon.                                                                                    |
| 15    | `Attack`          | `playerIndex`, `cardId`, `serial`, `attackId`                                                   | An attack was used.                                                                                                            |
| 16    | `HpChange`        | `playerIndex`, `cardId`, `serial`, `value`, `putDamageCounter`                                  | HP changed. `value` is positive for damage, negative for healing. `putDamageCounter` indicates if damage counters were placed. |
| 17    | `Poisoned`        | `playerIndex`, `isRecover`, `cardId`, `serial`                                                  | Poison status applied or recovered.                                                                                            |
| 18    | `Burned`          | `playerIndex`, `isRecover`, `cardId`, `serial`                                                  | Burn status applied or recovered.                                                                                              |
| 19    | `Asleep`          | `playerIndex`, `isRecover`, `cardId`, `serial`                                                  | Sleep status applied or recovered.                                                                                             |
| 20    | `Paralyzed`       | `playerIndex`, `isRecover`, `cardId`, `serial`                                                  | Paralysis status applied or recovered.                                                                                         |
| 21    | `Confused`        | `playerIndex`, `isRecover`, `cardId`, `serial`                                                  | Confusion status applied or recovered.                                                                                         |
| 22    | `Coin`            | `playerIndex`, `head`                                                                           | A coin was flipped. `head=true` = heads.                                                                                       |
| 23    | `Result`          | `result`, `reason`                                                                              | Game ended. `result`: 0/1=winner player index, 2=draw. `reason`: cause code.                                                   |

### AreaType Values (used in `fromArea`/`toArea`)

> **C++ source:** `engine/src/core/CardTypes.h` — `enum class AreaType : unsigned char`

```cpp
enum class AreaType : unsigned char {
  All, Deck, Hand, Trash, Active, Bench, Prize, Stadium,
  Energy, Tool, PreEvolution, Player, Looking, Playing, DeckBottom,
  Me, Effected, EffectedPreTarget, SelectedList, TriggerSubject,
  TriggerObject, Attach, TurnPlay, AttackPreMyTurn, Temporary,
};
```

> Values are implicit (auto-incremented from 0).

| Value | Name         | Description                       |
|-------|--------------|-----------------------------------|
| 0     | All          | (rarely used)                     |
| 1     | Deck         | Deck                              |
| 2     | Hand         | Hand                              |
| 3     | Trash        | Discard pile                      |
| 4     | Active       | Active (battle) spot              |
| 5     | Bench        | Bench                             |
| 6     | Prize        | Prize cards                       |
| 7     | Stadium      | Stadium slot                      |
| 8     | Energy       | Attached energy                   |
| 9     | Tool         | Attached tool                     |
| 10    | PreEvolution | Evolution cards beneath a Pokemon |

---

## `search_begin_input` — Binary State (Python-added)

This field is **not** part of the C++ JSON output. Python adds it after parsing:

```python
Battle.obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
```

It contains a base64-encoded binary serialization of the complete game state, used by the tree search / agent API. The binary format is defined in `Binary.h`.

---

## Enum Reference Tables

### SelectType

> **C++ source:** `engine/src/core/ApiTypes.h` — `enum class SelectType : unsigned char`

```cpp
enum class SelectType : unsigned char {
  None, Main, Card, AttachedCard, CardOrAttachedCard,
  Energy, Skill, Attack, Evolve, Count, YesNo, SpecialCondition,
};
```

> Values are implicit (auto-incremented from 0). C++ enum members without explicit assignments get sequential integers starting at 0.

| Value | Name               | Description                                                         |
|-------|--------------------|---------------------------------------------------------------------|
| 0     | None               | (subtracted — never appears in API JSON)                            |
| 1     | Main               | Main phase: choose actions (play cards, attack, retreat, end turn). |
| 2     | Card               | Select a card from a specific area.                                 |
| 3     | AttachedCard       | Select an attached card (energy/tool).                              |
| 4     | CardOrAttachedCard | Select either a card or an attached card.                           |
| 5     | Energy             | Select energy to discard/detach.                                    |
| 6     | Skill              | Select a skill/ability to activate.                                 |
| 7     | Attack             | Select an attack to use.                                            |
| 8     | Evolve             | Select an evolution target.                                         |
| 9     | Count              | Select a numeric count (damage counters, cards, etc.).              |
| 10    | YesNo              | Binary yes/no choice.                                               |
| 11    | SpecialCondition   | Choose a special condition to inflict.                              |

> **Note:** In API mode, `type` is serialized as `max(0, enum_value - 1)`, so the `None` entry is collapsed and values start at 0 for `Main`.

### SelectContext

> **C++ source:** `engine/src/core/ApiTypes.h` — `enum class SelectContext : unsigned char`

```cpp
enum class SelectContext : unsigned char {
  None, Main, SetupActivePokemon, SetupBenchPokemon, Switch, ToActive,
  ToBench, ToField, ToHand, Discard, ToDeck, ToDeckBottom, ToPrize,
  NotMove, DamageCounter, DamageCounterAny, Damage, RemoveDamageCounter,
  Heal, EvolvesFrom, EvolvesTo, Devolve, AttachFrom, AttachTo, DetachFrom,
  Look, EffectTarget, DiscardEnergyCard, DiscardToolCard, SwitchEnergyCard,
  DiscardCardOrAttachedCard, DiscardEnergy, ToHandEnergy, ToDeckEnergy,
  SwitchEnergy, SkillOrder, Attack, DisableAttack, Evolve, DrawCount,
  DamageCounterCount, RemoveDamageCounterCount, IsFirst, Mulligan, Activate,
  FirstEffect, MoreDevolve, CoinHead, AffectSpecialCondition,
  RecoverSpecialCondition,
};
```

> Values are implicit (auto-incremented from 0).

| Value | Name                      | Description                           |
|-------|---------------------------|---------------------------------------|
| 0     | None                      | (subtracted)                          |
| 1     | Main                      | Normal main phase.                    |
| 2     | SetupActivePokemon        | Initial setup: choose active Pokemon. |
| 3     | SetupBenchPokemon         | Initial setup: choose bench Pokemon.  |
| 4     | Switch                    | Switch Pokemon (retreat).             |
| 5     | ToActive                  | Move to active spot.                  |
| 6     | ToBench                   | Move to bench.                        |
| 7     | ToField                   | Move to field.                        |
| 8     | ToHand                    | Return to hand.                       |
| 9     | Discard                   | Discard.                              |
| 10    | ToDeck                    | Return to deck.                       |
| 11    | ToDeckBottom              | Send to bottom of deck.               |
| 12    | ToPrize                   | Send to prizes.                       |
| 13    | NotMove                   | Cannot move.                          |
| 14    | DamageCounter             | Place damage counters.                |
| 15    | DamageCounterAny          | Place damage counters freely.         |
| 16    | Damage                    | Deal damage.                          |
| 17    | RemoveDamageCounter       | Remove damage counters.               |
| 18    | Heal                      | Heal HP.                              |
| 19    | EvolvesFrom               | Select evolves-from target.           |
| 20    | EvolvesTo                 | Select evolves-to target.             |
| 21    | Devolve                   | Devolve.                              |
| 22    | AttachFrom                | Select attachment source.             |
| 23    | AttachTo                  | Select attachment target.             |
| 24    | DetachFrom                | Detach from.                          |
| 25    | Look                      | Look at cards.                        |
| 26    | EffectTarget              | Select effect target.                 |
| 27    | DiscardEnergyCard         | Discard energy card.                  |
| 28    | DiscardToolCard           | Discard tool card.                    |
| 29    | SwitchEnergyCard          | Switch energy cards.                  |
| 30    | DiscardCardOrAttachedCard | Discard card or attached card.        |
| 31    | DiscardEnergy             | Discard energy.                       |
| 32    | ToHandEnergy              | Energy to hand.                       |
| 33    | ToDeckEnergy              | Energy to deck.                       |
| 34    | SwitchEnergy              | Switch energy.                        |
| 35    | SkillOrder                | Order of skill resolution.            |
| 36    | Attack                    | Attack context.                       |
| 37    | DisableAttack             | Disable an attack.                    |
| 38    | Evolve                    | Evolution context.                    |
| 39    | DrawCount                 | Draw count selection.                 |
| 40    | DamageCounterCount        | Damage counter count.                 |
| 41    | RemoveDamageCounterCount  | Remove damage counter count.          |
| 42    | IsFirst                   | Who goes first.                       |
| 43    | Mulligan                  | Mulligan handling.                    |
| 44    | Activate                  | Ability activation.                   |
| 45    | FirstEffect               | First effect resolution.              |
| 46    | MoreDevolve               | Additional devolution.                |
| 47    | CoinHead                  | Coin flip choice.                     |
| 48    | AffectSpecialCondition    | Apply special condition.              |
| 49    | RecoverSpecialCondition   | Recover from special condition.       |

### EnergyType

> **C++ source:** `engine/src/core/EnergyTypes.h` — `enum class EnergyType : unsigned short` (bitmask flags)

```cpp
enum class EnergyType : unsigned short {
  Colorless = 0,       Grass = 1 << 0,   Fire = 1 << 1,
  Water = 1 << 2,      Lightning = 1 << 3, Psychic = 1 << 4,
  Fighting = 1 << 5,   Darkness = 1 << 6, Metal = 1 << 7,
  Dragon = 1 << 8,     All = (1 << 9) - 1,
};
// Also: Psychic | Darkness = 80 (index 11)
```

| Index | Name             | Bitmask |
|-------|------------------|---------|
| 0     | Colorless        | 0       |
| 1     | Grass            | 1       |
| 2     | Fire             | 2       |
| 3     | Water            | 4       |
| 4     | Lightning        | 8       |
| 5     | Psychic          | 16      |
| 6     | Fighting         | 32      |
| 7     | Darkness         | 64      |
| 8     | Metal            | 128     |
| 9     | Dragon           | 256     |
| 10    | All (Rainbow)    | 511     |
| 11    | Psychic+Darkness | 80      |

### CardType (used in AllCard)

> **C++ source:** `engine/src/core/CardTypes.h` — `enum class CardType : unsigned char`

```cpp
enum class CardType : unsigned char {
  Pokemon, Item, Tool, Supporter, Stadium, BasicEnergy, SpecialEnergy,
};
```

> Values are implicit (auto-incremented from 0).

| Value | Name                |
|-------|---------------------|
| 0     | Pokemon             |
| 1     | Item (Goods)        |
| 2     | Tool (Pokemon Tool) |
| 3     | Supporter           |
| 4     | Stadium             |
| 5     | Basic Energy        |
| 6     | Special Energy      |

### PokemonType (used in AllCard)

> **C++ source:** `engine/src/core/CardTypes.h` — `enum class PokemonType : unsigned char`

```cpp
enum class PokemonType : unsigned char {
  NotPokemon, Normal, PokemonItem, Ex, MegaEx,
};
```

> Values are implicit (auto-incremented from 0).

| Value | Name                       |
|-------|----------------------------|
| 0     | NotPokemon                 |
| 1     | Normal                     |
| 2     | PokemonItem (fossil/doll)  |
| 3     | Ex (Pokemon ex)            |
| 4     | MegaEx (Mega Evolution ex) |

### EvolutionType (used in AllCard)

> **C++ source:** `engine/src/core/CardTypes.h` — `enum class EvolutionType : unsigned char`

```cpp
enum class EvolutionType : unsigned char {
  NoEvolutionType, Basic, Stage1, Stage2,
};
```

> Values are implicit (auto-incremented from 0).

| Value | Name                            |
|-------|---------------------------------|
| 0     | NoEvolutionType (not a Pokemon) |
| 1     | Basic                           |
| 2     | Stage1                          |
| 3     | Stage2                          |

---

## AllCard — Card Database API

Returned by `ApiAllCard()`. One entry per unique card in the game.

```json
[
  {
    "cardId": 101,
    "name": "Charizard ex",
    "cardType": 0,
    "pokemonType": 3,
    "evolutionType": 3,
    "retreatCost": 3,
    "hp": 330,
    "weakness": 3,
    "resistance": null,
    "energyType": 2,
    "basic": false,
    "stage1": false,
    "stage2": true,
    "ex": true,
    "megaEx": false,
    "tera": false,
    "aceSpec": false,
    "evolvesFrom": "Charmeleon",
    "skills": [
      {"name": "Blaze", "text": "Once during your turn, ..."}
    ],
    "attacks": [42, 87]
  }
]
```

| Field           | Type             | Description                                       |
|-----------------|------------------|---------------------------------------------------|
| `cardId`        | `int`            | Unique card ID.                                   |
| `name`          | `string`         | English card name.                                |
| `cardType`      | `int`            | CardType enum (see above).                        |
| `pokemonType`   | `int`            | PokemonType enum.                                 |
| `evolutionType` | `int`            | EvolutionType enum.                               |
| `retreatCost`   | `int`            | Colorless energy cost to retreat.                 |
| `hp`            | `int`            | Base HP (0 for non-Pokemon).                      |
| `weakness`      | `int \| null`    | EnergyType index of weakness, or `null`.          |
| `resistance`    | `int \| null`    | EnergyType index of resistance, or `null`.        |
| `energyType`    | `int`            | EnergyType index of this Pokemon's type.          |
| `basic`         | `bool`           | Is a Basic Pokemon.                               |
| `stage1`        | `bool`           | Is a Stage 1 Pokemon.                             |
| `stage2`        | `bool`           | Is a Stage 2 Pokemon.                             |
| `ex`            | `bool`           | Is a Pokemon ex.                                  |
| `megaEx`        | `bool`           | Is a Mega Evolution ex.                           |
| `tera`          | `bool`           | Is a Terastal Pokemon.                            |
| `aceSpec`       | `bool`           | Is an ACE SPEC card.                              |
| `evolvesFrom`   | `string \| null` | Name of the Pokemon this evolves from, or `null`. |
| `skills`        | `array<Skill>`   | Abilities and special skills.                     |
| `attacks`       | `array<int>`     | Attack IDs this card has. Map to AllAttack.       |

---

## AllAttack — Attack Database API

Returned by `ApiAllAttack()`.

```json
[
  {
    "attackId": 42,
    "name": "Flame Wing",
    "text": "Discard a Fire Energy from this Pokemon.",
    "damage": 130,
    "energies": [2, 2, 0]
  }
]
```

| Field      | Type         | Description                                                                 |
|------------|--------------|-----------------------------------------------------------------------------|
| `attackId` | `int`        | Unique attack ID.                                                           |
| `name`     | `string`     | English attack name.                                                        |
| `text`     | `string`     | Attack effect text (may contain `\n` for newlines).                         |
| `damage`   | `int`        | Base damage (0 for non-damage attacks).                                     |
| `energies` | `array<int>` | Energy cost as EnergyType index values. Each entry = 1 energy of that type. |

---

## SelectOptionType Enum Reference

> **C++ source:** `engine/src/core/ApiTypes.h` — `enum class SelectOptionType : unsigned char`

```cpp
enum class SelectOptionType : unsigned char {
  Number, Yes, No, Card, ToolCard, EnergyCard, Energy,
  Play, Attach, Evolve, Ability, Discard, Retreat, Attack,
  End, Skill, SpecialCondition,
};
```

> Values are implicit (auto-incremented from 0).

| Value | Name             |
|-------|------------------|
| 0     | Number           |
| 1     | Yes              |
| 2     | No               |
| 3     | Card             |
| 4     | ToolCard         |
| 5     | EnergyCard       |
| 6     | Energy           |
| 7     | Play             |
| 8     | Attach           |
| 9     | Evolve           |
| 10    | Ability          |
| 11    | Discard          |
| 12    | Retreat          |
| 13    | Attack           |
| 14    | End              |
| 15    | Skill            |
| 16    | SpecialCondition |

---

## SpecialConditionType

> **C++ source:** `engine/src/core/CardTypes.h` — `enum class SelectSpecialConditionType : unsigned char`

```cpp
enum class SelectSpecialConditionType : unsigned char {
  Poison, Burn, Sleep, Paralyze, Confuse,
};
```

> Values are implicit (auto-incremented from 0).

| Value | Name     |
|-------|----------|
| 0     | Poison   |
| 1     | Burn     |
| 2     | Sleep    |
| 3     | Paralyze |
| 4     | Confuse  |

---

## BadStatusType

> **C++ source:** `engine/src/core/CardTypes.h` — `enum class BadStatusType : unsigned char`

```cpp
enum class BadStatusType : unsigned char {
  None, Asleep, Paralyzed, Confused,
};
```

> Values are implicit (auto-incremented from 0).

| Value | Name      | Description         |
|-------|-----------|---------------------|
| 0     | None      | No bad status       |
| 1     | Asleep    | Sleep (ねむり)       |
| 2     | Paralyzed | Paralysis (マヒ)    |
| 3     | Confused  | Confusion (こんらん) |

---

## Hidden Information Rules

- **Opponent's hand**: Always `null` (only `handCount` is visible).
- **Opponent's prizes**: `null` for face-down entries (you know count but not which cards).
- **Face-down cards** (reverse): `null` in `CardRef`/`PokemonRef` positions.
- **Opponent's draws**: Appear as `DrawReverse` (type 5) — only `playerIndex`, no card details.
- **Opponent's face-down moves**: Appear as `MoveCardReverse` (type 7) — no card details.
- **`deck` field in `select`**: Only populated when selecting from deck; otherwise `null`.
- **`looking` field**: Shows cards being inspected by effects; `null` if not looking. May contain `null` entries for cards the current player shouldn't see.

---

## Web vs API Mode Differences

The same game state can be serialized in two modes:

| Aspect                       | API Mode (`ToJsonApi`) | Web/Vis Mode (`ToJsonWeb`/`ToJsonVis`)           |
|------------------------------|------------------------|--------------------------------------------------|
| `type`/`context` in `select` | Integer enum values    | String enum names (e.g., `"Main"`, `"Attack"`)   |
| Log `type`                   | Integer                | String (e.g., `"Draw"`, `"Attack"`)              |
| `yourIndex` in `current`     | `selectPlayer` value   | `selectPlayer` (API) or `2` (vis = both players) |
| `lookingCount`               | Not included           | Included (web only)                              |
| `selected`                   | Not included           | Included (vis only)                              |
| `name` field on cards        | Not included           | Included when `playerIndex==2` (addName mode)    |
| Opponent hand                | `null`                 | Full list when `playerIndex==2`                  |
| Opponent prizes              | `null` for face-down   | Full list when `playerIndex==2`                  |

The example JSON in this directory uses **API mode** (integer enums), which is what Python receives from `GetBattleData()`.
