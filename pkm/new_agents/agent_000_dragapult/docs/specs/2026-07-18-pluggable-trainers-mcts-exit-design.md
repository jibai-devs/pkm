# Design: Pluggable training methods + MCTS expert iteration

**Date:** 2026-07-18
**Agent:** `agent_000_dragapult`
**Status:** Approved design (pre-implementation)
**Scope:** Everything lives inside `pkm/new_agents/agent_000_dragapult/`. The engine
is treated as an existing dependency reached only through the agent's own `cabt.py`
seam — no changes outside the package.

## 1. Goal

Add **MCTS expert iteration (ExIt)** as a second self-play training method, while:

1. keeping classic **PPO** working (the existing update-458 run must still resume
   bit-for-bit), and
2. introducing a **clean interface that supports multiple training methods**, so a
   third method is easy to add later.

This first iteration optimizes for **architecture + a correct minimal ExIt**: a clean
pluggable-trainer seam plus a real-but-minimal MCTS ExIt loop (perfect-information
self-play, modest simulation count) that exercises the whole cycle end-to-end.
Strength work (IS-MCTS, chance nodes, bootstrapped targets) is explicitly deferred
and logged in §7.

## 2. Why now / what already fits

Grounding facts from the current code:

- The network is already **AlphaZero-shaped**: `model.evaluate()` returns
  `(priors, value)` no-grad (`model.py:155`) — exactly an MCTS node's need.
- The engine exposes a real **forward simulator**: `search_begin / search_step /
  search_end / search_release` (`pkm/engine/api.py:133-237`), reached via the agent's
  `cabt.py`. `search_begin(obs, predicted opponent deck/prize/hand/active…)` → node;
  `search_step(id, picks)` → next node.
- **Self-play is the easy determinization case**: both seats play the same known
  `DECK_60` (`train.py:62`), so hidden information is only *which* known cards sit in
  hand/prize/deck-order — not open-vocab opponent decks.
- There is a **precedent for config-selected pluggable strategy**: `shaping.py`'s
  `SHAPERS`/`ESTIMATORS` registries, keyed by config strings, serialized into the
  config hash. The trainer registry mirrors this pattern.
- The PPO-specific surface is **narrow**: `train.train()` hardcodes
  `rollout → ppo_update → checkpoint` (`train.py:340-354`); everything else in the
  loop (checkpoint/resume, observers, parallel pool, eval, timing/utilization
  diagnostics) is method-agnostic.

## 3. Architecture — the `Trainer` seam (Approach A)

A **single `Trainer` protocol, one implementation per method**, behind a registry
keyed by `cfg.train.method`. The three things that differ between PPO and ExIt —
rollout records, target labels, loss — are all owned by the Trainer; the driver keeps
everything shared.

### 3.1 `trainers/__init__.py` (new subpackage)

```python
class Trainer(Protocol):
    def collect(self, model, n_games, cfg, gen=None) -> tuple[list, dict]: ...  # runs in workers
    def update(self, model, opt, samples, cfg) -> dict: ...                      # the learn step

TRAINERS: dict[str, Callable[[], Trainer]] = {"ppo": PpoTrainer, "exit": ExItTrainer}
```

- `collect` returns `(samples, stats)`. `samples` is *any* picklable list; each method
  defines its own record type. It ships over the worker queue exactly like `Step` does
  today.
- `update` returns the per-update `stats` dict already consumed by the observers /
  console. Both methods reuse the same optimizer, grad-clip, checkpoint, and the
  `rollout_util`/`core_util`/`serial_frac` diagnostics.

### 3.2 `train.py` becomes a thin, method-agnostic driver

The loop changes from hardcoding `collect_rollout` + `ppo_update` to:

```python
trainer = TRAINERS[cfg.train.method]()
...
samples, roll_stats = (
    pool.collect(trainer, games_per_update) if pool else trainer.collect(model, games, cfg)
)
upd_stats = trainer.update(model, opt, samples, cfg)
```

Checkpoint/resume, observer notification, timing, and eval are untouched.

### 3.3 `parallel.py` — one generic change

`_worker` builds `trainer = TRAINERS[cfg.train.method]()` once and calls
`trainer.collect(...)` instead of importing `collect_rollout`.
`ParallelRollout.collect(trainer, total_games)` gains the trainer parameter and passes
it through. Each spawn-worker already owns its own engine and its own `AgentStart`
agent-ptr, so MCTS-in-workers needs no extra plumbing.

### 3.4 `config.py` + checkpoint migration

- Add `method: str = "ppo"` to `TrainConfig` (a selector, same shape as the existing
  `shaping` / `advantage` keys).
- Fix the resume guard (`train.py:257`, `"config hash mismatch on resume"`) so additive
  schema changes do not break old checkpoints: validate the checkpoint's **stored**
  config dict against its **stored** hash (always matches an untampered file), and
  back-fill missing keys to defaults in `Config.from_dict`. The update-458 PPO
  checkpoint then resumes bit-for-bit; new runs record `method` normally.

## 4. PPO trainer (`trainers/ppo.py`)

The current `play_game` / `collect_rollout` / `ppo_update` move here almost verbatim:

- `PpoTrainer.collect` = self-play sampling → `Step`s → `assign_targets` (existing
  `shaping.py` shaping + GAE).
- `PpoTrainer.update` = the existing clipped-PPO epoch loop.
- `Step` remains PPO's sample type.

A **PPO regression smoke test** asserts this relocated path matches the pre-refactor
behavior (finite losses over updates + resume round-trip), proving the old way is
intact.

## 5. MCTS + ExIt (`mcts.py`, `trainers/exit.py`)

### 5.1 `cabt.py` — surface the search seam

Add `search_begin / search_step / search_end / search_release` to the existing
`pkm.engine` re-export line (they are already exposed at the package level), alongside
the already-typed `SearchState`. Purely widening the agent's seam; no engine changes.

### 5.2 `mcts.py` (new) — PUCT search, method-agnostic

A tree over engine search nodes, guided by the network:

- **Node** = one engine search node (`searchId` + its observation), holding per-child
  `N, W, Q, P` over that node's legal options.
- **Selection:** `argmax  Q(a) + c_puct · P(a) · √(ΣN) / (1 + N(a))`.
- **Expansion:** `search_step(id, [a])` → child `SearchState`; `featurize(child.obs)` →
  `model.evaluate()` → `(priors, value)`; priors seed the child's `P`, `value` is
  backed up.
- **Backup:** negamax — flip the value's sign when the child's acting seat differs from
  the parent's (zero-sum). Terminal nodes (battle finished in the returned state) back
  up the true ±1.
- **Budget / knobs:** `n_simulations` (default 32), `c_puct`. `search_end()` /
  `search_release()` free nodes each move so memory stays flat.
- **Returns** the root visit distribution `π(a) = N(a)^(1/τ) / Σ` — the improved policy
  target.

### 5.3 Determinization — pluggable, v1 = single-sample (K=1)

`search_begin` needs predicted opponent deck/prize/hand/active. The engine only returns a
**per-seat masked view** (`GetBattleData`, engine/api.py:44) — the opponent's hand is
hidden and there is no full-state accessor — so true "oracle" state is not available.
Because both seats play the known `DECK_60`, v1 instead **samples one consistent
assignment** of the known remaining cards (the DECK_60 multiset minus everything publicly
visible in the acting seat's view) into the opponent's hidden zones (deck / hand / prize),
respecting the observed counts. This is honest **IS-MCTS with K=1**. It sits behind a small
`determinize(obs, seat, cfg, gen) -> predictions` function so full **IS-MCTS** (sample K
worlds, search each, average visits) drops in later without touching `mcts.py`.

### 5.4 `trainers/exit.py` — `ExItTrainer`

- **`collect`:** play a self-play game; at each acting decision run `mcts.search(...)` →
  record `ExItSample(features, policy_target=π, seat)`, then play a move sampled from `π`
  and advance the real game with `battle_select`. At game end write
  `value_target = z` (the ±1 outcome from each seat's view, via the existing
  `_seat_reward`) onto that seat's samples. Value target is the Monte-Carlo outcome for
  v1 — no bootstrap/GAE (those are PPO's concern).
- **`update`:** supervised, reusing the same optimizer/grad-clip. Loss =
  `CE(policy_logits, π_target) + value_coef · MSE(value, z)` (+ optional entropy reg).
  No clipping, no `old_logprob`, no ratio epochs — though it can iterate a few epochs
  over the collected batch.

### 5.5 `ExItSample` record

Fields: `features, policy_target (vector over options), value_target, seat`. Picklable,
ships over the worker queue like `Step`. `Step` stays PPO's; each trainer owns its record.

## 6. Config, CLI, testing

### 6.1 Config knobs (added to `TrainConfig`, inert unless `method="exit"`)

- `method: str = "ppo"`
- `mcts_simulations: int = 32`, `mcts_c_puct: float = 1.25`,
  `mcts_temperature: float = 1.0`, `determinization: str = "sample"`

All flat (matching the current style), serialized into the hash — safe via the resume
migration (§3.4).

### 6.2 CLI + justfile

- `train` gains `--method` (default `ppo`) plus `--mcts-*` options.
- New recipe `train-exit`, mirroring `train-fast`'s `train`/`resume` mode contract,
  injecting `--method exit` and the MCTS knobs.
- `resume` needs nothing new — method + knobs restore from the checkpoint.

### 6.3 Testing

- **PPO regression:** smoke run through `PpoTrainer` matches the pre-refactor path.
- **Back-compat:** load an old-schema checkpoint blob (no `method`/`mcts_*`) and resume
  without a hash error.
- **MCTS unit:** on a captured `search_begin` fixture, a few sims produce a valid `π`
  (sums to 1, mass only on legal options) and visit counts grow with `n_simulations`.
- **ExIt smoke:** 2 updates × 2 games, tiny sim count — finite CE/MSE losses + checkpoint
  round-trip.
- **Parallel ExIt:** a 2-worker collect returns samples (confirms MCTS-in-workers with a
  per-process engine).

## 7. Known limitations & future improvements (logged, not built)

These are the deliberate v1 simplifications. Each has a clean upgrade path and should be
mirrored as a code comment at its site, matching the repo's `[DECIDE]` convention.

1. **Single-sample determinization, K=1** (§5.3) — one sampled world per search, so the
   search can be lucky/unlucky about hidden cards. Upgrade: **full IS-MCTS** — sample K
   worlds, search each, average visits, behind the existing `determinize()` seam.
2. **Chance resolved inside `search_step`** — coin flips / draws are treated as
   environment transitions rather than explicit **chance nodes**. Slightly biases visit
   counts; explicit chance nodes are a later refinement.
3. **Monte-Carlo value target `z`** (§5.4) — replace/augment with **n-step or
   bootstrapped** value targets.
4. **Fixed multi-select count** carried over from `policy.py` (`[DECIDE]`) — MCTS models a
   single option index per node; revisit multi-select search.
5. **Synchronous collector unchanged** — MCTS makes rollout the heavy, parallelizable
   part, so the worker pool finally scales near-linearly (see the Amdahl analysis: the
   serial PPO update stops being the bottleneck once search dominates the rollout).
6. **Inference-time MCTS** (README §9.1) — reuses `mcts.py` + the `cabt` search seam; a
   near-free strength follow-up once training ExIt lands.

## 8. File manifest

**New**
- `trainers/__init__.py` — `Trainer` protocol + `TRAINERS` registry
- `trainers/ppo.py` — `PpoTrainer` (relocated PPO code)
- `trainers/exit.py` — `ExItTrainer` + `ExItSample`
- `mcts.py` — PUCT search over engine search nodes
- tests for the above (MCTS unit, ExIt smoke, PPO regression, back-compat, parallel ExIt)

**Changed**
- `train.py` — slim method-agnostic driver
- `parallel.py` — generic `collect(trainer, …)`
- `config.py` — `method` + `mcts_*` knobs + hash-migration on resume
- `cabt.py` — re-export `search_*`
- `cli.py` — `--method` / `--mcts-*` options
- `justfile` — `train-exit` recipe
