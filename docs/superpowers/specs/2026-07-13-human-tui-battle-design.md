# Human TUI Battle — Design

**Date:** 2026-07-13
**Status:** Approved, ready for planning

A Textual TUI that lets a human play a full Pokémon TCG match against one of our
trained agents, launched with `just play human neural 02_dragapult`. Exactly one
human player is supported.

---

## Goals

- Play a complete, legal match by hand against `random` / `neural` / `mcts`.
- Every decision the engine can ask for is answerable — no soft-locks, ever.
- The observation is strongly typed with pydantic at the TUI boundary.
- A hand-played game produces the same `result.html` / `replay.json` artifacts a
  headless game does.

## Non-goals

- Undo / rollback (the engine has no rollback).
- Hot-seat two-human play.
- Deckbuilding in the TUI.
- Re-typing the training path (`encoder.py`, agents, MCTS keep raw dicts).

---

## Engine constraints (verified empirically, 2026-07-13)

Two limits in `kaggle_environments` will end a human game if left at their
defaults. Both were confirmed by running real matches, not read off the source.

| Limit | Where | Behaviour | Measured |
|---|---|---|---|
| Overage clock | `agent.py:220` | Each act deducts `duration - actTimeout` from a **cumulative 600 s** budget (`remainingOverageTime`, defaulted in `cabt.json`). The first act that overdraws it returns `DeadlineExceeded` → status `TIMEOUT` → **reward -1**. | With `actTimeout=0` and a 0.05 s/act sleep, the clock burned 600 → 593.6 s over one game. A human thinking 30 s/turn loses on time in ~20 moves. |
| Run timeout | `core.py:302` | `env.run` aborts the entire episode once wall clock exceeds `runTimeout` (default 2000 s). | With `runTimeout=5` and a 0.5 s/act sleep, `env.run` **raised `DeadlineExceeded`** at 5.0 s, killing the game at step 21. |

**Fix, verified:** construct the env with
`configuration={"decks": [deck, deck], "actTimeout": 1e9, "runTimeout": 1e9}`.
With those values a full game ran to completion and `remainingOverageTime` stayed
at **exactly 600** — nothing consumed. Note `actTimeout` has `minimum: 0` in the
schema, so it must be large-positive; a negative value raises `InvalidArgument`.

### stdout hazard (verified safe)

`kaggle_environments/agent.py:184` wraps every agent call in
`redirect_stdout(out_buffer)` / `redirect_stderr(err_buffer)`. These rebind
`sys.stdout` / `sys.stderr` **process-wide**, and our human agent blocks *inside*
that `with` block for the entire time the player is thinking.

Textual is immune: `textual/drivers/linux_driver.py:58` writes to `sys.__stderr__`
— the pristine stream captured at interpreter start, which `redirect_stderr` does
not touch. Rendering cannot be swallowed.

**The rule this imposes: the TUI must never call `print()`.** Any `print` issued
while the human agent is blocked disappears into kaggle's capture buffer. Use
Textual's own logging.

---

## Architecture

```
pkm/obs.py            pydantic models + IntEnums for the observation
pkm/tui/session.py    GameSession protocol + ThreadedEnvSession (the engine hook)
pkm/tui/labels.py     Option -> readable text; Log -> readable text
pkm/tui/widgets.py    BoardPanel, HandBar, EventLog, PromptPane
pkm/tui/app.py        BattleApp (Textual App), result screen, keybindings
pkm/rl/play.py        + 'human' branch (the only existing file that changes)
```

Dependencies run one way — UI → session → engine. Nothing existing depends on any
of it. New dependency: `pydantic`. `textual>=8.2.8` is already in `pyproject.toml`.

### `pkm/obs.py` — the typed contract

Models: `Observation`, `Select`, `Option`, `GameState`, `Player`, `PokemonRef`,
`CardRef`, `Log`. Enums: `SelectType`, `SelectContext`, `OptionType`, `AreaType`,
`EnergyType`, `LogType`.

```python
class Observation(BaseModel):
    model_config = ConfigDict(extra="allow")  # kaggle adds step, remainingOverageTime

    select: Select | None = None
    logs: list[Log] = []
    current: GameState | None = None
    search_begin_input: str | None = None
```

#### Enum numbering (verified against the live engine, 2026-07-13)

**`SelectType` and `SelectContext` are off-by-one from the tables in
`OBSERVATION_SCHEMA.md`.** The doc's own footnote says API mode serializes them as
`max(0, enum - 1)`, collapsing the `None` entry — so the tables are 1-based but the
wire format is 0-based. Confirmed by playing real games: the first prompt of every
game is `(type=9, context=41)`, which decodes as `YesNo` + `IsFirst` ("do you go
first?") only under 0-based numbering. Every other observed pair decodes correctly
0-based too: `(0,0)`=Main+Main, `(1,1)`=Card+SetupActivePokemon, `(1,3)`=Card+Switch,
`(4,30)`=Energy+DiscardEnergy, `(5,34)`=Skill+SkillOrder, `(7,37)`=Evolve+Evolve.

So: `SelectType.MAIN = 0`, `CARD = 1`, … `YES_NO = 9`, `SPECIAL_CONDITION = 10`.
(This matches `encoder.py`'s `NUM_SELECT_TYPES = 11`.)

**`OptionType` and `LogType` are NOT offset** — they are exactly as the doc's tables
say. Verified in live games: `option.type == 13` really is Attack (carries
`attackId`), `log.type == 16` really is HpChange (carries `value` /
`putDamageCounter`), `log.type == 23` really is Result.

**Consequence: `obs_data_structure/example_obs.json` is a hand-written
approximation and is NOT a valid test fixture** — it numbers HpChange as 17, which
contradicts both the doc and the engine. Tests use a fixture captured from the live
engine instead (Task 1).

`Observation.model_validate(raw)` is called **once**, at the session boundary.
Inward of that, nothing sees a dict; `option.type == OptionType.ATTACK` replaces
`o["type"] == 13`.

Hidden information is modelled honestly, so the type checker forces us to handle
it: `Player.hand: list[CardRef] | None` (opponent's hand is always `None`),
`Player.prize: list[CardRef | None]` (taken/face-down prizes are `None`),
`Player.active: list[PokemonRef | None]`.

`Option` is a single model with all type-specific fields optional
(`attackId`, `area`, `index`, `playerIndex`, `inPlayArea`, `inPlayIndex`,
`energyIndex`, `toolIndex`, `number`, `count`, `cardId`, `serial`,
`specialConditionType`) rather than a discriminated union — the engine may emit
option types we haven't catalogued, and an unknown one must still parse and stay
pickable.

**Deliberate duplication:** `pkm/rl/encoder.py:31-57` already defines `OPT_*` /
`AREA_*` as bare ints. Encoder is the hot PPO/MCTS rollout path and is explicitly
out of scope, so `pkm/obs.py` defines its enums independently rather than making
encoder import pydantic. This is a knowing, contained duplication.

### `pkm/tui/labels.py` — making the game legible

Turns an `Option` into text a human can act on. The work is resolving
`(playerIndex, area, index)` to a real card — the same resolution
`encoder._card_id_at` performs, but yielding names (via `pkm.data.get_card_data()`
/ `get_attack_data()`, which read the live engine card DB; `replay/cards.json` is
a reference dump, not a runtime dependency).

```
{"type": 8, "area": 2, "index": 3, "inPlayArea": 5, "inPlayIndex": 1}
  -> "Attach Basic {P} Energy → Drakloak (bench 2)"
{"type": 13, "attackId": 42}
  -> "Attack: Phantom Dive  130 dmg  [P][P][☆]"
```

Every one of the 17 `OptionType`s gets a renderer, plus a generic fallback
(`"Option 3 (type=16)"`). **The fallback is what guarantees no soft-lock:** an
option type we failed to anticipate degrades to something pickable instead of
crashing. Log entries get the same treatment for the event feed.

### `pkm/tui/session.py` — the engine hook

`env.run` owns the game loop on a worker thread; Textual owns the screen on the
main thread; two queues join them.

```
main thread (Textual)                worker thread (env.run)
  BattleApp                            human_agent(obs)  <- kaggle calls this
     |                                     |  prompts.put(Observation)
     |  <---------- prompts --------------- |  picks.get()   [blocks here]
     |  render board + options              |      (kaggle's redirect_stdout
     |  user presses [1], Enter             |       is active this whole time)
     |  ----------- picks --------------->  |
     |                                      |  returns [0] -> battle_select
```

The human agent is the entire engine hook:

```python
def human_agent(obs: dict) -> list[int]:
    if obs["select"] is None:
        return deck                      # deck submission is not a decision
    prompts.put(Observation.model_validate(obs))
    picks = picks_q.get()                # blocks until the user chooses
    if isinstance(picks, Quit):
        raise _Abort
    return picks
```

`GameSession` is a Protocol with `next_prompt()` and `submit(picks)`.
`ThreadedEnvSession` implements it over the queues above. **This is the seam that
makes the fallback cheap** — if the threaded approach proves unworkable, a
`DirectEngineSession` driving `battle_start` / `battle_select` / `battle_finish`
implements the same two methods, and the UI does not change.

### Failure paths

All three must be handled or the app hangs on a queue nobody will write to.

| Path | Handling |
|---|---|
| **Worker dies** (engine error, invalid pick, our bug) | Worker wraps `env.run` in try/except, pushes `Failed(exc)` onto `prompts`. App renders the error and stops. |
| **User quits mid-game** (`q`) | App puts `Quit` on `picks`; the blocked agent raises `_Abort`, unwinding `env.run`. Worker is a daemon thread, so a wedged engine call cannot keep the process alive. `battle_finish()` does not run on this path (kaggle only calls it on clean finish), leaking the C battle — acceptable, since the process exits immediately and each game is its own process. |
| **Game ends** | `env.run` returns; worker pushes `Finished(env)`. App reads `env.steps[-1][i].reward` → Victory / Defeat / Draw. kaggle's `finish()` has already built the visualizer payload, so we write `result.html` and `replay.json` exactly as `play_match` does today. |

---

## Interaction

One prompt pane drives every decision, because `select` is uniform: an option
list plus `minCount` / `maxCount`.

```
┌─ AGENT (02_dragapult) ────────────┐┌─ LOG ─────────────┐
│ ACTIVE  Dusknoir       130/220 🔥 ││ T6 you drew Rare  │
│   [P][P][D]  tool: Rescue Board   ││ Agent attacked    │
│ BENCH   Duskull 60/60             ││  Phantom Dive 130 │
│         Drakloak 100/100          ││ Dusknoir KO'd     │
│ prizes 3  deck 40  hand 7         ││ …                 │
├─ YOU ─────────────────────────────┤│                   │
│ ACTIVE  Dragapult ex   170/200    ││                   │
│ BENCH   Drakloak, Dreepy          ││                   │
│ prizes 4  deck 38  hand 5         │└───────────────────┘
├─ YOUR HAND ────────────────────────────────────────────┤
│ Boss's Orders · Ultra Ball · Basic {P} ×2 · Dreepy     │
├─ CHOOSE (pick 1) ──────────────────────────────────────┤
│ [1] Attack: Phantom Dive  130 dmg  [P][P][☆]           │
│ [2] Attack: Jet Headbutt   70 dmg  [☆][☆]             │
│ [3] Retreat                                            │
│ [4] End turn                                           │
└────────────────────────────────────────────────────────┘
```

- `1`-`9` (and `↑`/`↓` + `Enter`) toggle options.
- `Enter` submits, enabled only when `minCount <= len(picks) <= maxCount`;
  otherwise it shows a hint ("pick 2 more").
- This one mechanism covers attacking, retreating, playing cards, setup, *and*
  the fiddly cases — discard 2 cards, distribute 6 damage counters
  (`maxCount: 6`) — with no special-casing.
- Selection is **stateless per prompt**: picks are indices into the current
  `select.option` list and are cleared on each new prompt.
- There is no undo, so irreversible choices (**attack**, **end turn**) get a
  confirm step.
- `?` toggles help. `q` quits.

Board panels show, per player: active Pokémon (name, HP bar, attached energy,
tools, status conditions), bench, and counters for prizes / deck / hand /
discard. The event log renders `obs.logs` through `labels.py`.

---

## CLI surface

`human` becomes an agent name alongside `random` / `neural` / `mcts`. Finding it
in either slot is what flips `pkm play` from headless into TUI mode. Both players
use the profile's deck (a mirror match — exactly how `just eval` works today).

```bash
just play human neural              # you vs the trained agent, both 02_dragapult
just play human random              # you vs random
just play neural human 01_psychic   # you as player 2 on the psychic deck
just play neural random             # unchanged: headless, as today
```

`make_agent_by_name` cannot build a human agent (it has no session to talk to),
so `play_match` grows a branch: if either side is `human`, construct the
`ThreadedEnvSession` and run `BattleApp`; otherwise run headless as it does now.
`win_rate` (games > 1) rejects `human` with a clear error.

---

## Testing

The UI is thin; the logic under it is pure and testable without a terminal.

| Unit | Test |
|---|---|
| `pkm/obs.py` | Parse a fixture **captured from the live engine** (`tests/fixtures/observations.json`, generated in Task 1 by playing a scripted random game and recording every distinct select/option/log shape). Not `example_obs.json` — that file is hand-written and has the wrong LogType numbering. A `model_validate` → `model_dump` round-trip proves no field is silently dropped. Plus: an unknown `OptionType` still parses, and the extra kaggle keys (`step`, `remainingOverageTime`) don't break validation. |
| `labels.py` | Table test: one constructed `Option` per `OptionType`, asserting the rendered string names the right card and target. Plus the unknown-type fallback. |
| `session.py` | Tested against a **fake engine** — a scripted list of observations, no C library, no threads — covering the prompt/submit handshake and all three sentinels (`Failed`, `Quit`, `Finished`). |
| threading | One slow integration test drives a real `human` vs `random` game through the real `ThreadedEnvSession` with a scripted "always pick option 0" auto-player. This is the only test that can catch a genuine deadlock. |
| `app.py` | Textual `run_test()` pilot: boot against the fake session, press `1`, press `Enter`, assert the pick reached the session. |

Existing `just test` must stay green; no test may depend on a real TTY.
