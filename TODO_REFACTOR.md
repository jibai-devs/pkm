# TODO: pkm Refactor Checklist

**Why this exists:** The training code (~4800 lines Python) works, but has accumulated
several patterns that will hurt as the codebase grows. This document catalogs every
issue with exact file:line references, explains why it's bad, and proposes the fix.
Work top-to-bottom; each item is independent unless noted.

---

## Grand Master Plan

```
Phase 1 — Quick wins (no architecture change)
  [x] 1. Canonical forced-pick check
  [x] 2. Encoder magic numbers -> dataclass
  [ ] 3. Numpy parity test

Phase 2 — Extract shared code
  [ ] 4. Move encoder/features to pkm.features
  [ ] 5. Training harness (shared loop)

Phase 3 — Harder refactors
  [ ] 6. Consolidate dual forward-pass (model.py / numpy_policy.py)
  [ ] 7. Engine global state safety
  [ ] 8. Split play.py execution modes
  [ ] 9. Stop leaking private _find_weights
```

---

## 1. Canonical forced-pick check

**Status:** [x] done (2026-07-16)

**What:** Three separate implementations of "this decision has no real choice, skip it":

| Location | Lines | Interface |
|----------|-------|-----------|
| `pkm/mcts/search.py` | `#26-33` | `_forced_picks(sel: dict) -> list[int] \| None` |
| `pkm/rl/rollout.py` | `#36-40` | inlined in `TorchPolicy.act()` on pydantic `Observation` |
| `pkm/rl/numpy_policy.py` | `#142-146` | inlined in `NumpyPolicy.select()` on pydantic `Observation` |

**Why it's bad:**
- The logic is identical (`n == 1 && minCount >= 1` or `n == minCount == maxCount`)
  but the interfaces diverge: `_forced_picks` uses dict access (`sel["minCount"]`),
  the others use pydantic attrs (`sel.minCount`).
- If the forced-pick heuristic changes (e.g. a new edge case), you must update three
  places. There is no test that keeps them in sync.
- `exit_train.py:81` imports the private `_forced_picks` from `mcts/search.py` —
  leaking an underscore-prefixed name across package boundaries.

**Fix:**
Both implementations now live in `pkm/types/obs.py`:

- **Method on `Select`** (`obs.py:239-247`) — `self.forced_picks()` for pydantic callers
  that already have a parsed `Select` object.
- **Standalone function** (`obs.py:251-262`) — `forced_picks(sel: dict)` for dict callers
  in hot loops (MCTS, exit_train) that skip pydantic parsing.

`pkm/mcts/search.py` re-exports the standalone function so existing dict-based
callers (`agent.py`, `exit_train.py`, `tests/test_mcts.py`) keep working.

All six call sites use the canonical implementation:
- `pkm/mcts/search.py` — `forced_picks(obs["select"])` (standalone, via re-export)
- `pkm/mcts/agent.py` — `forced_picks(obs["select"])` (via `search.py` re-export)
- `pkm/rl/exit_train.py` — `forced_picks(obs["select"])` (via `search.py` re-export)
- `pkm/rl/rollout.py` — `sel.forced_picks()` (pydantic method)
- `pkm/rl/numpy_policy.py` — `sel.forced_picks()` (pydantic method)
- `tests/test_mcts.py` — `forced_picks(sel)` (via `search.py` re-export)

---

## 2. Encoder magic numbers -> dataclass

**Status:** [x] done (2026-07-16)

**What:** Normalization constants are hardcoded inline throughout the encoder:

| Location | Line(s) | Expression | Meaning |
|----------|---------|------------|---------|
| `pkm/rl/encoder.py` | `#91` | `p.hp / 300.0` | Max HP |
| `pkm/rl/encoder.py` | `#92` | `p.maxHp / 300.0` | Max HP |
| `pkm/rl/encoder.py` | `#93` | `len(p.energies) / 5.0` | Max energies |
| `pkm/rl/encoder.py` | `#139` | `player.handCount / 20.0` | Max hand size |
| `pkm/rl/encoder.py` | `#140` | `player.deckCount / 60.0` | Deck size |
| `pkm/rl/encoder.py` | `#141` | `len(player.prize) / 6.0` | Prize count |
| `pkm/rl/encoder.py` | `#142` | `len(player.discard) / 60.0` | Max discard |
| `pkm/rl/encoder.py` | `#146` | `len(player.bench) / 8.0` | Max bench |
| `pkm/rl/encoder.py` | `#147` | `player.benchMax / 8.0` | Max bench |
| `pkm/rl/encoder.py` | `#148` | `state.turn / 30.0` | Max turns |
| `pkm/rl/encoder.py` | `#149` | `state.turnActionCount / 20.0` | Max actions/turn |
| `pkm/rl/encoder.py` | `#168` | `sel.minCount / 5.0` | Max pick count |
| `pkm/rl/encoder.py` | `#169` | `sel.maxCount / 5.0` | Max pick count |
| `pkm/rl/encoder.py` | `#170` | `sel.remainEnergyCost / 5.0` | Max energy cost |
| `pkm/rl/encoder.py` | `#171` | `sel.remainDamageCounter / 10.0` | Max damage counters |
| `pkm/rl/encoder.py` | `#294` | `atk.damage / 300.0` | Max attack damage |
| `pkm/rl/encoder.py` | `#295` | `len(atk.energies) / 5.0` | Max energies |
| `pkm/rl/encoder.py` | `#303` | `(o.number or 0) / 20.0` | Max option number |
| `pkm/rl/encoder.py` | `#304` | `(o.count or 0) / 5.0` | Max option count |
| `pkm/rl/encoder.py` | `#346` | `(len(opp.prize) - len(me.prize)) / 6.0` | Prize diff norm |

**Why it's bad:**
- If the game's bounds change (new max HP, larger bench, etc.), these silently
  produce wrong inputs. No error, just degraded training.
- The numbers are undocumented — a newcomer has no idea why 300, why 5, why 20.
- `300` appears three times (HP, damage) but could easily diverge.

**Fix:**
Domain constants (`NUM_CARDS`, `NUM_ATTACKS`, `MAX_BENCH`, `MAX_HAND`, etc.) moved
to `pkm/types/obs.py` where the data model lives. `encoder.py` imports them from there.

A `Norm` dataclass in `pkm/rl/encoder.py:36-61` replaces all ~20 inline divisors.
Single instance `NORM = Norm()` at module level. Every normalization is now
`NORM.max_hp`, `NORM.max_energies`, etc. — self-documenting and easy to override.

---

## 3. Numpy parity test

**Status:** [ ] not started

**What:** `pkm/rl/numpy_policy.py` is a hand-ported numpy reimplementation of
`pkm/rl/model.py`'s PyTorch forward pass. The docstring at `numpy_policy.py:1-4`
literally says "Must stay in sync with pkm/rl/model.py."

**Why it's bad:**
- Any change to the network architecture requires manually updating both files.
- There are **zero tests** asserting parity between the two implementations.
- A subtle drift (different activation, wrong transpose, off-by-one in padding)
  would silently produce a Kaggle agent that disagrees with the training model.
- The numpy version reimplements `_encode_state` (`numpy_policy.py:33-42`),
  `_encode_options` (`numpy_policy.py:44-56`), `_logits` (`numpy_policy.py:58-67`),
  `value` (`numpy_policy.py:69-73`), `priors` (`numpy_policy.py:75-85`),
  `act_greedy` (`numpy_policy.py:87-106`), and `sample_picks` (`numpy_policy.py:108-135`)
  — all must match their PyTorch counterparts in `model.py:87-150`.

**Fix:**
Add a test that:
1. Creates a `PolicyValueNet` with random weights
2. Exports to `NumpyPolicy` via the dict conversion
3. Generates a random `EncodedDecision` fixture
4. Asserts `model.act(d).picks == numpy_policy.act_greedy(d)` (greedy parity)
5. Asserts `abs(model.value(h) - numpy_policy.value(d)) < 1e-5` (value parity)
6. Asserts `allclose(model priors, numpy_policy priors)` (prior parity)

**Files to change:**
- `tests/test_numpy_parity.py` — new file
- (optionally) CI config to run it on every commit

---

## 4. Move encoder to `pkm.features`

**Status:** [ ] not started

**What:** `pkm/rl/encoder.py` is imported by three packages that shouldn't depend on RL:

| Importer | Line | Import |
|----------|------|--------|
| `pkm/mcts/search.py` | `#17` | `from pkm.rl.encoder import encode_decision` |
| `pkm/rl/exit_train.py` | `#33` | `from pkm.rl.encoder import EncodedDecision, encode_decision` |
| `pkm/rl/numpy_policy.py` | `#11` | `from .encoder import EncodedDecision, encode_decision` |

**Why it's bad:**
- MCTS is conceptually a search algorithm, not an RL component. It imports from
  `pkm.rl` for the encoder, creating a circular conceptual dependency:
  RL training -> MCTS (for expert iteration targets) -> RL (for encoder + policy).
- `pkm/rl/encoder.py` also depends on `pkm.data` (for `get_attack_data`) and
  `pkm.types.obs` — it's really a shared feature-engineering layer, not RL-specific.

**Fix:**
1. Create `pkm/features/__init__.py` with the encoder contents.
2. Update imports in:
   - `pkm/rl/model.py:33-41` — `from pkm.features import ...`
   - `pkm/rl/rollout.py:15` — `from pkm.features import ...`
   - `pkm/rl/numpy_policy.py:11` — `from pkm.features import ...`
   - `pkm/rl/exit_train.py:33` — `from pkm.features import ...`
   - `pkm/mcts/search.py:17` — `from pkm.features import ...`
3. Keep a re-export in `pkm/rl/encoder.py` for backward compat during transition:
   `from pkm.features import *`

**Files to change:**
- `pkm/features/__init__.py` — new (move encoder contents)
- `pkm/rl/encoder.py` — becomes a re-export shim
- `pkm/rl/model.py:33` — update import
- `pkm/rl/rollout.py:15` — update import
- `pkm/rl/numpy_policy.py:11` — update import
- `pkm/rl/exit_train.py:33` — update import
- `pkm/mcts/search.py:17` — update import

---

## 5. Training harness (shared loop)

**Status:** [ ] not started

**What:** `pkm/rl/train.py` and `pkm/rl/exit_train.py` have nearly identical
boilerplate for the training loop:

| Concern | `train.py` lines | `exit_train.py` lines |
|---------|------------------|-----------------------|
| Typer CLI setup | `#229-278` (50 lines) | `#308-351` (44 lines) |
| Agent profile resolution | `#252-260` | `#328-336` |
| CSV writer setup | `#94-98` | `#226-230` |
| MetricLog + wandb setup | `#100-119` | `#232-249` |
| Iteration loop + timing | `#125-221` | `#251-300` |
| Checkpoint save | `#204-205,222` | `#300` |
| Resource cleanup | `#223-225` | `#302-304` |

**Why it's bad:**
- Adding a new training algorithm (e.g. DPO, RLHF, REINFORCE) means copying ~150
  lines of boilerplate and hoping you don't miss a CSV field or a log call.
- The CSV field lists are separate (`CSV_FIELDS` at `train.py:42-57` vs
  `EXIT_CSV_FIELDS` at `exit_train.py:183-192`) but overlap heavily.
- The wandb config dicts (`train.py:107-118` vs `exit_train.py:239-248`) duplicate
  the same structure.

**Fix:**
Extract a `TrainingHarness` class or `run_training_loop()` function in a new
`pkm/rl/harness.py`:

```python
@dataclass
class TrainingConfig:
    algo: str
    deck_path: str
    iterations: int
    checkpoint_dir: str
    metrics_path: str
    log_dir: str
    seed: int
    wandb_project: str | None
    wandb_run_name: str | None
    # algo-specific config stored as dict
    algo_config: dict

def run_training_loop(
    config: TrainingConfig,
    step_fn: Callable[[int], dict[str, float]],  # iter -> metrics
    eval_fn: Callable[[int], float | None] | None = None,
    save_fn: Callable[[Path], None] | None = None,
) -> None:
    """Shared training loop: CSV, MetricLog, timing, checkpointing."""
    ...
```

Then `train.py` and `exit_train.py` each become ~80 lines: define `step_fn`,
`eval_fn`, `save_fn`, and call `run_training_loop`.

**Files to change:**
- `pkm/rl/harness.py` — new
- `pkm/rl/train.py` — refactor to use harness
- `pkm/rl/exit_train.py` — refactor to use harness

---

## 6. Consolidate dual forward-pass

**Status:** [ ] not started

**What:** `pkm/rl/model.py` (PyTorch, 290 lines) and `pkm/rl/numpy_policy.py`
(numpy, 147 lines) implement the same network. The numpy version reimplements:

| Method | `model.py` | `numpy_policy.py` |
|--------|-----------|-------------------|
| State encoding | `encode_state` `#87-99` | `_encode_state` `#33-42` |
| Option encoding | `encode_options` `#101-120` | `_encode_options` `#44-56` |
| Logit scoring | `option_logits` `#122-141` | `_logits` `#58-67` |
| Value head | `value` `#143-150` | `value` `#69-73` |
| Greedy acting | `act` `#154-197` (greedy branch) | `act_greedy` `#87-106` |
| Sampling | `act` `#154-197` (sample branch) | `sample_picks` `#108-135` |
| First-pick priors | (not exposed) | `priors` `#75-85` |

**Why it's bad:**
- Every architecture change is a manual two-file sync with no safety net.
- The numpy version has subtle differences: `priors()` (`numpy_policy.py:75-85`)
  has no PyTorch equivalent — it's used only by MCTS and was added ad-hoc.
- `act_greedy` and `sample_picks` are separate methods in numpy but branches of
  a single `act()` in PyTorch — the decomposition doesn't match.

**Fix (two options):**

**Option A (recommended): Write forward pass in numpy, wrap in PyTorch.**
Move the core forward logic to a shared numpy module. `NumpyPolicy` uses it
directly. `PolicyValueNet` wraps it with `torch.from_numpy` calls for gradient
computation. This eliminates the sync problem entirely.

**Option B (lighter): Generate numpy from torch.**
Add an export step that traces the PyTorch model and generates the numpy weights
dict with the correct layer ordering. The numpy inference code stays but is
tested for parity (see item 3).

**Files to change:**
- `pkm/rl/model.py` — major rewrite (option A) or minor (option B)
- `pkm/rl/numpy_policy.py` — major rewrite (option A) or tested (option B)
- `tests/test_numpy_parity.py` — new (option B)

---

## 7. Engine global state safety

**Status:** [ ] not started

**What:** The engine uses process-global mutable singletons:

| Location | Line(s) | Global |
|----------|---------|--------|
| `pkm/engine/loader.py` | `#46-50` | `Battle.battle_ptr`, `Battle.obs` (class-level attrs) |
| `pkm/engine/api.py` | `#106` | `_agent_ptr: int \| None = None` |
| `pkm/engine/api.py` | `#109-113` | `_get_agent_ptr()` lazy init |

`battle_start` sets `Battle.battle_ptr` (`api.py:60`), and every subsequent
`battle_select` / `battle_finish` reads it. `search_begin` uses a separate
`_agent_ptr` singleton.

**Why it's bad:**
- Only one battle can exist at a time. Nesting (e.g. MCTS simulation inside a
  training rollout) silently corrupts state.
- `battle_finish()` at `api.py:81-83` frees `battle_ptr` but doesn't null it —
  a subsequent `battle_select` would use a dangling pointer.
- The `try/finally` pattern in `rollout.py:80-91` and `exit_train.py:78-107`
  is the only defense, but it's easy to forget.
- `search_begin`/`search_end` has its own global (`_agent_ptr`) that could
  conflict if two searches run concurrently.

**Fix:**
Add a context manager that asserts non-reentrancy:

```python
from contextlib import contextmanager

_battle_active = False

@contextmanager
def battle_context(deck0, deck1):
    global _battle_active
    if _battle_active:
        raise RuntimeError("Cannot nest battles (engine is process-global)")
    _battle_active = True
    try:
        obs, start = battle_start(deck0, deck1)
        yield obs
    finally:
        battle_finish()
        _battle_active = False
```

This doesn't fix the underlying C library limitation, but it turns silent
corruption into a loud error. Same pattern for `search_context`.

**Files to change:**
- `pkm/engine/api.py` — add `battle_context`, `search_context`
- `pkm/rl/rollout.py:74,91` — use `battle_context`
- `pkm/rl/exit_train.py:72,107` — use `battle_context`
- `pkm/mcts/search.py:174,193` — use `search_context`

---

## 8. Split `play.py` execution modes

**Status:** [ ] not started

**What:** `pkm/rl/play.py` contains two fundamentally different execution paths:

| Function | Lines | Mechanism |
|----------|-------|-----------|
| `play_match` / `win_rate` | (kaggle-env runner) | `kaggle_environments.make()` |
| `play_human_match` | (TUI path) | Direct engine API via `ThreadedEnvSession` |

They share a CLI entry point via `make_agent_by_name()` dispatch.

**Why it's bad:**
- `play_match` returns `Environment | None` depending on whether it's a human
  match — two different return types in one function.
- The human-play path needs timeout disarm (`actTimeout`/`runTimeout` = `1e9`)
  and `textual.log` instead of `print` — these are easy to break if someone
  modifies the shared code.
- Agent creation logic (`make_agent_by_name`) is tangled with match execution.

**Fix:**
Split into three files:
- `pkm/rl/play.py` — CLI entry point + `make_agent_by_name()` only
- `pkm/rl/match.py` — `play_match()`, `win_rate()` (kaggle-env runner)
- `pkm/tui/match.py` — `play_human_match()` (already in the `pkm.tui` package)

**Files to change:**
- `pkm/rl/play.py` — slim down to CLI + dispatch
- `pkm/rl/match.py` — new (kaggle-env match runner)
- `pkm/tui/match.py` — new or move human-play code here

---

## 9. Stop leaking private `_find_weights`

**Status:** [ ] not started

**What:** `pkm/mcts/agent.py:6` imports `_find_weights` (underscore-prefixed,
private by convention) from `pkm/agents/neural_agent.py`.

**Why it's bad:**
- Private APIs can change without notice. `mcts/agent.py` is a consumer of an
  implementation detail.
- The weight-lookup logic (explicit path -> env var -> package-relative ->
  kaggle path) is useful to any agent, not just the neural one.

**Fix:**
Move `_find_weights` to `pkm/agents/__init__.py` or a new `pkm/agents/weights.py`
as a public function `find_weights()`. Update both import sites:
- `pkm/agents/neural_agent.py:14` — `from pkm.agents.weights import find_weights`
- `pkm/mcts/agent.py:6` — `from pkm.agents.weights import find_weights`

**Files to change:**
- `pkm/agents/weights.py` — new (move + rename to `find_weights`)
- `pkm/agents/neural_agent.py:14` — update import
- `pkm/mcts/agent.py:6` — update import
