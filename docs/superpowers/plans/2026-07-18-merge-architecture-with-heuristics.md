# Merge guide: `feature/heuristics-integration` architecture + `refactor-to-prepare-for-heuristics-integration` heuristics

> **For agentic workers:** this doc is self-contained. Every claim below was
> verified by diffing the two actual branches (`git merge-base`, `git diff`,
> `git merge-tree --write-tree`) as of the commits named in "Branches" below.
> If something here doesn't match what you find in the repo, trust the repo
> and flag the mismatch rather than silently reconciling it.

## Branches

- **`feature/heuristics-integration`** (pushed to `origin`, tip `380c829`) —
  the general architecture: `GameContext`/`DeckTracker` (per-game memory),
  the `FeatureSpec` registry (declarative input features + ablation +
  checkpoint stamping), a pooled-embedding deck-ledger feature family, a
  trunk split (board/deck-ledger/belief), and a detached opponent-archetype
  auxiliary head. Full design doc:
  `docs/superpowers/plans/2026-07-16-heuristics-integration-architecture.md`.
- **`refactor-to-prepare-for-heuristics-integration`** (on `origin`, tip
  `841b85f`) — hand-tuned, deck-specific (Dreepy/Drakloak/Dragapult ex/Budew/
  Xerosic) reward-shaping heuristics, plus unrelated training-infra work
  (parallel rollout, train-vs-fixed-opponent mode, an agent-listing CLI, a
  TUI opponent-hand-spy debug view) that happened to land on the same
  branch.
- **Common ancestor:** `39a1964` (`git merge-base` of the two above).

## The principle

These two branches solve **different problems** and the merge should stay
that way — modular, not tangled:

- `feature/heuristics-integration` decides **what the network sees**
  (input features) and **what it's allowed to bypass** (hard rules). Lives
  in `pkm/heuristics/`, `pkm/rl/features.py`, `pkm/rl/model.py`.
- `refactor-to-prepare-for-heuristics-integration`'s heuristic content
  decides **what reward the network is trained to chase** — a parallel
  registry pattern (`pkm/rl/reward_terms.py`) of potential-based and
  direct reward-shaping terms, each a pure function of `(obs, picks)`.
  Lives in `pkm/rl/reward_terms.py` and ~15 functions appended to
  `pkm/rl/encoder.py`.

Verified: **`pkm/rl/model.py` has a zero-line diff between the two
branches' common ancestor and `refactor-to-prepare-for-heuristics-integration`.**
The reward-shaping heuristics never touch the network architecture,
`GameContext`, or `DeckTracker` — they're pure functions computed once per
decision in `rollout.py` and combined into the PPO return in
`ppo.py:compute_returns`. This is why a clean modular merge is possible:
graft the reward-shaping subsystem onto the architecture branch without
touching the architecture at all.

**Also verified (do this, don't `git merge`):** a raw merge of the two
branches produces real conflicts in `pkm/rl/train.py` and `pkm/cli/__init__.py`
because `refactor-to-prepare-for-heuristics-integration` interleaves the
wanted reward-weight wiring with *unwanted* parallel-rollout and
train-vs-fixed-opponent code in the same functions. Untangling that from a
merge conflict is harder than just porting the specific wanted pieces by
hand, per the file-by-file instructions below. Do **not** `git merge` or
`git rebase` the branches directly.

## What to bring in

### 1. New file, verbatim

Copy `pkm/rl/reward_terms.py` unmodified from
`refactor-to-prepare-for-heuristics-integration`:

```bash
git show refactor-to-prepare-for-heuristics-integration:pkm/rl/reward_terms.py \
  > pkm/rl/reward_terms.py
```

This is a registry of `(term_name, EncodedDecision_attr)` pairs — a
`POTENTIAL_TERMS` list (potential-based, e.g. prize differential) and a
`DIRECT_TERMS` list (action-conditioned bonuses/penalties) — plus
`DEFAULT_WEIGHTS`, `load_weights(path)`, `write_default_weights_file(path)`.
**`DEFAULT_WEIGHTS["shaping"] = 0.2`, every other term defaults to `0.0`** —
this exactly reproduces `feature/heuristics-integration`'s current
hardcoded `shaping_coef=0.2` default, so an agent profile with no
`reward_weights.json` trains identically to today. No behavior change for
existing agents unless they opt in.

### 2. `pkm/rl/encoder.py` — port ~15 heuristic functions

`feature/heuristics-integration` already refactored this file's *feature*
side onto the `FeatureSpec` registry (Task 4, commit `ecbf929`) — that
happened after `refactor-to-prepare-for-heuristics-integration` forked, so
its `encoder.py` still looks pre-refactor. The heuristic functions
themselves don't depend on that refactor at all (pure functions of
`obs`/`picks`), so this is a mechanical append, not a structural merge.

Get the source:
```bash
git show refactor-to-prepare-for-heuristics-integration:pkm/rl/encoder.py \
  > /tmp/their_encoder.py
```

**Append these functions** (everything after `prize_potential` in their
file) to the end of `feature/heuristics-integration`'s `pkm/rl/encoder.py`,
after its own `prize_potential`:

```
dragapult_backup_potential          dreepy_line_field_potential
_active_energy_already_sufficient   _attack_cost_covered
_bench_energy_already_sufficient    energy_overattach_penalty
budew_first_turn_attack_bonus       budew_active_second_potential
xerosic_machinations_bonus          _resolve_energy_attach
wrong_type_energy_penalty           dragapult_ex_attack_bonus
phantom_dive_attack_bonus           dreepy_energy_spread_penalty
dreepy_evolve_bonus                 dreepy_line_bench_charge_bonus
dreepy_line_active_charge_bonus     drakloak_backup_ready_bonus
wasted_resources_attack_penalty     budew_turn_bench_setup_bonus
```
(the last one, `budew_turn_bench_setup_bonus`, is referenced by
`rollout.py`'s import list — grep their `encoder.py` for its body if it
isn't in your copy; it may be adjacent to `budew_first_turn_attack_bonus`.)

**Constant/import fixes needed** (checked against the current
`feature/heuristics-integration` `encoder.py`, which already defines or
imports most of what these functions need):

- Add to the existing `from pkm.rl.features import (...)` block:
  **`OPT_RETREAT`** (value 12 — every other `OPT_*` these functions use,
  `OPT_ATTACK`/`OPT_ATTACH`/`OPT_EVOLVE`/`OPT_PLAY`, is already imported).
- Add a new import: `from pkm.data.card_data import Attack, CardData, get_card_by_id`.
- `AREA_HAND` and friends, `_card_id_at`, `_pokemon_at` already exist in
  `feature/heuristics-integration`'s `encoder.py` at the same names — no
  change needed. **Note:** `AREA_LOOKING = 12` already exists there too, so
  the `AreaType.LOOKING` addition their branch made to `pkm/types/obs.py`
  is likely redundant on this side — verify, don't blindly port it.
- Their card-ID/attack-ID constants block (`BUDEW_CARD_ID`, `DREEPY_CARD_ID`,
  `DRAKLOAK_CARD_ID`, `DRAGAPULT_EX_CARD_ID`, `DREEPY_LINE_CARD_IDS`,
  `XEROSIC_MACHINATIONS_CARD_ID`, `PHANTOM_DIVE_ATTACK_ID`,
  `ENERGY_TYPE_COLORLESS`/`FIRE`/`PSYCHIC`) has no equivalent on this side —
  port it verbatim, placed near the top of the file with the other
  constants.

**Add these fields to `EncodedDecision`** (currently ends with
`true_archetype: int = -1` in `feature/heuristics-integration`):

```python
board_setup_potential: float = 0.0
budew_setup_potential: float = 0.0
dreepy_line_field_potential: float = 0.0
energy_penalty: float = 0.0
budew_bonus: float = 0.0
wrong_type_energy_penalty: float = 0.0
dragapult_attack_bonus: float = 0.0
dreepy_spread_penalty: float = 0.0
xerosic_bonus: float = 0.0
budew_bench_setup_bonus: float = 0.0
dreepy_evolve_bonus: float = 0.0
dreepy_bench_charge_bonus: float = 0.0
dreepy_active_charge_bonus: float = 0.0
wasted_resources_penalty: float = 0.0
phantom_dive_bonus: float = 0.0
drakloak_backup_ready_bonus: float = 0.0
```

### 3. `pkm/rl/rollout.py` — populate the new fields

In `TorchPolicy.__call__` (or wherever `d.potential = prize_potential(parsed)`
currently sits), add the population lines — copy verbatim from
`refactor-to-prepare-for-heuristics-integration`'s `pkm/rl/rollout.py`:

```bash
git diff 39a1964 refactor-to-prepare-for-heuristics-integration -- pkm/rl/rollout.py
```

Take only the ~17 lines that assign `d.<field> = <heuristic_fn>(parsed, res.picks)`
(or `(parsed)` for the two potential-based ones) and the matching import
additions from `.encoder import (...)`. **Do not** take the rest of that
diff — `GameSpec`, `make_game_specs`, `play_one`, `aggregate_result` are
the parallel-rollout machinery (see "What to leave out" below).

### 4. `pkm/rl/ppo.py` — replace `compute_returns`'s body

`feature/heuristics-integration`'s `ppo.py` has two functions:
`compute_returns` (untouched since the branch point — still the old flat
`shaping_coef: float` signature) and `ppo_update` (Task 8 added the
archetype-loss term here — **do not touch `ppo_update`**, it auto-merges
cleanly and has no overlap with this change).

Replace `compute_returns` with
`refactor-to-prepare-for-heuristics-integration`'s version (get it via
`git show refactor-to-prepare-for-heuristics-integration:pkm/rl/ppo.py`) —
it changes the signature from `shaping_coef: float = 0.2` to
`weights: dict[str, float] | None = None, win_reward: float = 1.0`, and
loops over `POTENTIAL_TERMS`/`DIRECT_TERMS` from `reward_terms.py` instead
of hardcoding the one prize-differential term. Add
`from .reward_terms import DIRECT_TERMS, POTENTIAL_TERMS` to `ppo.py`'s
imports.

`win_reward` is optional — a scale-wins-only knob their branch added
alongside this. Take it if convenient (it's a one-line, self-contained
addition to the terminal-reward line), skip it if you want the smallest
possible diff; it isn't load-bearing for the reward-terms registry itself.

### 5. `pkm/rl/train.py` — one call-site change, nothing else

Do **not** port `refactor-to-prepare-for-heuristics-integration`'s
`train.py` changes wholesale — that diff (403 lines) interleaves the
wanted reward-weight wiring with unwanted parallel-rollout and
vs-fixed-opponent training-mode code in the same function bodies.

The only change needed:
- Add `from .reward_terms import DEFAULT_WEIGHTS, load_weights` to the
  imports.
- In `train()`'s signature, replace `shaping_coef: float = 0.2` with
  `weights: dict[str, float] | None = None`.
- At the top of `train()`, resolve the effective weights:
  `effective_weights = {**DEFAULT_WEIGHTS, **(weights or {})}`.
- The one `compute_returns(..., shaping_coef=shaping_coef)` call site
  (around line 158) becomes
  `compute_returns(..., weights=effective_weights)`.
- In `main()` (the typer CLI entry point), replace the
  `shaping: float = typer.Option(0.2, ...)` parameter with something that
  loads a JSON weights file — simplest correct version:
  ```python
  weights: str | None = typer.Option(
      None, "--weights",
      help="path to a JSON file of {term: weight} overrides — see "
           "pkm/rl/reward_terms.py for term names and defaults. Defaults "
           "to the agent's own reward_weights.json when --agent is given.",
  ),
  ```
  then inside `main()`, resolve it before calling `train()`:
  ```python
  from .reward_terms import load_weights
  weights_path = weights or (profile.reward_weights_path if agent else None)
  resolved_weights = load_weights(weights_path)
  ```
  and pass `weights=resolved_weights` to `train()`.

Leave every other part of `train.py` untouched — the wandb logging, the
archetype aux-loss stamping, checkpoint-stamping calls (Tasks 8/5) all stay
exactly as they are.

### 6. `pkm/agents/profile.py` — one-line addition

```python
self.reward_weights_path = self.base_dir / "reward_weights.json"
```
next to the existing `self.checkpoint_dir`/`self.metrics_dir` assignments
in `AgentProfile.__init__`.

## What to leave out (and why)

Everything below is real work on
`refactor-to-prepare-for-heuristics-integration`, but it's training
*infrastructure*, not a heuristic, and isn't part of this merge:

| File / feature | Why it's out of scope here |
|---|---|
| `pkm/rl/parallel_rollout.py`, `GameSpec`/`make_game_specs`/`play_one`/`aggregate_result` in `rollout.py`, `--workers` CLI flag | Multi-process self-play rollout — orthogonal to heuristics, changes the training loop's control flow. Bring in separately if/when parallelism is actually needed. |
| `evaluate_vs_agent`, `play_vs_fixed_opponent`, `--eval-vs`/`--vs-agent` in `train.py`/`cli/__init__.py` | Train-against-another-agent's-checkpoint mode — a real feature, but a different one (multi-agent training), not a heuristic. |
| `pkm/cli/agent.py` (`pkm agent list`) | Standalone devtool listing agent profiles/checkpoints/reward-weight status. Nice to have, doesn't depend on anything else here — safe to port later on its own with no interaction risk. |
| `pkm/tui/session.py`'s `AgentNote`/`OpponentHand`/hand-spy wiring | Human-play TUI debug view for watching `singaporean_middleman`'s deduced prize list live. TUI-only, no training/architecture interaction. |
| `pkm/agents/singaporean_middleman.py` | A dispatcher stub — `_select_agent()` always returns `"neural"`, no actual routing logic exists yet. Nothing to port; `DeckTracker`, the one real thing it prototypes, is already in `GameContext` (Task 1). |
| `battle.sh`, `deck/03_pult_munki.csv`, `deck/04_mega_abomasnow.csv`, `agents/*/metrics/*.csv`, `agents/*/train_loop.log`, `agents/*/reward_weights.json` | Their own training-run artifacts and deck files — bring over the deck CSVs only if you intend to actually train that deck; the reward-weight *values* tuned for `03_pult_munki` belong in that agent's own `reward_weights.json`, not in this architecture branch's defaults. |
| `replay/02_vite_web_app/` deletions, `replay/05_vite_react_app/public/cards.json` update | Unrelated repo cleanup / data refresh that happened to ride along on the same branch. |
| `pkm/types/obs.py`'s `_as_enum` `TypeVar` rewrite | Cosmetic Python-typing-compat change from their parallel-rollout work (commit `81811e2`). Harmless either way; not required for anything above. |

## Procedure

```bash
git fetch origin feature/heuristics-integration refactor-to-prepare-for-heuristics-integration
git checkout -b integrate/heuristics-into-architecture feature/heuristics-integration
```

Then, in order (each is independently testable — run `python -m pytest
tests/ -q` after each step):

1. Add `pkm/rl/reward_terms.py` (§1). No existing tests touch this yet.
2. Port the encoder functions + `EncodedDecision` fields (§2). Add a test
   mirroring their intent if none exists — check whether
   `refactor-to-prepare-for-heuristics-integration` has tests for these
   functions (`git log --oneline refactor-to-prepare-for-heuristics-integration -- tests/`)
   and port those too if so.
3. Wire `rollout.py` (§3). Smoke-test: `pkm play --p0 neural --p1 random`
   completes without exceptions.
4. Replace `compute_returns` in `ppo.py` (§4). Existing PPO tests
   (`tests/test_rl.py` or similar) must still pass with default weights —
   this is the regression check that `DEFAULT_WEIGHTS` truly reproduces
   the old hardcoded behavior.
5. Wire `train.py` + CLI (§5) and `profile.py` (§6).
6. Full suite green (`just test`), then a short real training smoke run:
   `pkm train --agent <profile> --iterations 2 --games 4` with no
   `reward_weights.json` present (confirms the default path still works),
   then again after writing a `reward_weights.json` with one non-default
   term (confirms the override path works).
7. Re-run `tests/test_numpy_torch_parity.py` — `pkm/rl/model.py` isn't
   touched by this merge, so it should already pass untouched, but this is
   the standing gate for anything that touches `pkm/rl/`
   (`docs/superpowers/plans/2026-07-16-heuristics-integration-architecture.md`,
   Guardrails section).

## Open questions to flag back, not silently resolve

- Whether to also port `win_reward` (§4) — small, optional, your call.
- Whether `AreaType.LOOKING` needs porting to `obs.py` at all, given it
  already exists in `encoder.py`'s local `AREA_LOOKING` — check whether
  `pkm/types/obs.py`'s `AreaType` enum (used elsewhere beyond `encoder.py`)
  is missing it before deciding.
- Whether any of "What to leave out" is actually wanted later (parallel
  rollout in particular looks like real, reusable infra) — this doc scopes
  it out of *this* merge only, not forever.
