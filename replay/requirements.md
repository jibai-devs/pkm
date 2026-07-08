# Replay Visualization — Requirements & Context

## Goal

Build a visualization app that replays Pokemon TCG matches step-by-step from `replay.json` files. The viewer should support forward/backward navigation, pause, speed control, and display relevant game state and statistics at each step.

## Data Format

### `replay.json` — Game replay log (Kaggle Simulation format)

```
{
  "id", "name", "title", "version", "module_version",
  "configuration": {
    "decks": [deck0[60], deck1[60]]   — card IDs per player
  },
  "specification": { ... },           — Kaggle env spec
  "steps": [
    [entry_p0, entry_p1],             — step 0
    [entry_p0, entry_p1],             — step 1
    ...
  ]
}
```

Each step entry (per player perspective):
```json
{
  "action": [],                       — option indices chosen
  "reward": -1|0|1,                   — final reward (only meaningful at end)
  "status": "ACTIVE"|"DONE",
  "info": {},
  "observation": {
    "step": N,
    "remainingOverageTime": float,
    "select": {...}|null,             — raw prompt
    "logs": [...],                    — raw logs
    "current": {...}|null             — raw game state
  },
  "visualize": [{                     — rich per-player data
    "select": {
      "type": "YesNo"|"Card"|...,
      "context": "IsFirst"|"SetupActivePokemon"|...,
      "option": [...]                 — available choices
    },
    "logs": [event, ...],             — game events this step
    "current": {
      "turn": int,
      "turnActionCount": int,
      "yourIndex": 0|1,
      "firstPlayer": 0|1|-1,
      "supporterPlayed": bool,
      "stadiumPlayed": bool,
      "energyAttached": bool,
      "retreated": bool,
      "result": -1|0|1,              — -1=in progress, 0=draw, 1=p0 wins
      "stadium": [card],
      "players": [{
        "active": [card, ...],        — active pokemon(s) with HP, energies
        "bench": [card, ...],
        "benchMax": 5,
        "deckCount": int,
        "handCount": int,
        "hand": [card, ...],          — only visible for that player's POV
        "discard": [card, ...],
        "prize": [card, ...],
        "poisoned": bool,
        "burned": bool,
        "asleep": bool,
        "paralyzed": bool,
        "confused": bool
      }, ...]
    },
    "selected": [...]|null,
    "action": [...]
  }]
}
```

Card object:
```json
{
  "id": int,        — card_id (matches cards.json)
  "serial": int,    — unique instance ID in this game
  "playerIndex": 0|1,
  "name": string,
  "hp": int,        — current HP (in active/bench)
  "maxHp": int,
  "energies": [int, ...],
  "energyCards": [card, ...],
  "appearThisTurn": bool
}
```

### Log event types

| Type | Fields | Meaning |
|------|--------|---------|
| `Draw` | playerIndex, cardId, serial | Player drew a card |
| `Play` | playerIndex, cardId, serial | Card played from hand |
| `Attach` | area, index, inPlayArea, inPlayIndex | Energy attached |
| `Ability` | area, index | Ability activated |
| `Attack` | playerIndex, cardId, serial, attackId | Pokemon attacked |
| `HpChange` | playerIndex, cardId, serial, value, putDamageCounter | HP changed (negative = damage) |
| `MoveCard` | playerIndex, cardId, serial, fromArea, toArea | Card moved between zones |
| `HasBasicPokemon` | playerIndex, hasBasicPokemon | Mulligan check |
| `End` | — | Turn ended |

Area codes: 0=deck, 1=hand, 2=bench, 3=discard, 4=active, 5=prize, 8=attached(?)

### `cards.json` — Card database (59K lines)

```json
{
  "cards": [{
    "card_id": int,
    "name": string,
    "card_type": int,       — 0=pokemon, 5=energy, etc.
    "energy_type": int,
    "hp": int,
    "basic": bool,
    "stage1": bool,
    "stage2": bool,
    "ex": bool,
    "mega_ex": bool,
    "tera": bool,
    "ace_spec": bool,
    "evolves_from": string|null,
    "weakness": int|null,
    "resistance": int|null,
    "retreat_cost": int,
    "attacks": [{
      "attack_id": int,
      "name": string,
      "text": string,
      "damage": int,
      "energies": [int, ...]
    }],
    "skills": [...]
  }]
}
```

## Existing assets

- `replay/replay.json` — sample replay (~608K lines)
- `replay/cards.json` — card database with attack metadata
- `result.html` — Kaggle Simulation Player (generic, canvas-based, forward-only)

## Core features

1. **Step-by-step playback** — navigate through every game step
2. **Forward/backward** — step forward and rewind
3. **Play/pause** — auto-advance with adjustable speed
4. **Timeline scrubbing** — jump to any step via slider
5. **Game state rendering** — show active, bench, hand, discard, prize, deck count
6. **Event log** — scrolling text of what happened each step
7. **Statistics** — damage dealt, KOs, cards played, prizes taken
8. **Diff highlighting** — visually indicate what changed between steps
9. **Card detail lookup** — resolve card IDs to names/attacks via cards.json

## Constraints

- Replay files can be large (600K+ lines) — consider streaming/parsing strategy
- Must work offline (no server required)
- Should open with minimal setup (browser-only or `npm run dev`)
