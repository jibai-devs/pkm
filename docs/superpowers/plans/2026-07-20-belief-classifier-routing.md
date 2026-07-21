# Belief-Classifier Routing — Scoping Plan

**Status: Phase 1 and Phase 2 code DONE (2026-07-20), both committed
(`1a0c011`, `6749a65`). Phase 2 not yet run for real. Phase 3's own open
question is now MOOT** — see
`docs/superpowers/plans/2026-07-20-phase3-retrain-and-measure.md` Finding 1:
Milestone 8's checkpoint no longer exists (silently overwritten by
Milestone 9's own same-numbered snapshot files, both scripts sharing one
`checkpoints/` directory with independently-restarting iteration counters).
So "restart from Milestone 8" was never actually an available choice by the
time this question got asked — only "resume from `ppo_latest.pt`" remains.
Phase 3 here is now folded into that other plan file's bundled
retrain-and-measure scope (it covers this fix's Phase 3 together with
attack-damage-estimator's, per both plans' explicit "don't retrain twice"
guidance) — **this file's own Phase 3 section below is kept for the
historical decision record, but the actual next action lives in the other
plan file.**

**Phase 2 result:** `pkm/rl/population_train.py` (`run_pop_iteration`,
`population_train`) and `pkm/rl/parallel_rollout.py` (`_play_pop_chunk`,
`collect_pop_parallel`) now accept `archetype_classifier`, attached to
**both** `TorchPolicy` sides of every pairing (decision #2 below: resolved,
symmetric — population training has no frozen side). New `pkm
population-train --archetype-belief [--archetype-weights ...]` flag,
**opt-in, default off** (unlike `eval-vs-pool`'s default-on — this changes
what's being learned, so it stays an explicit choice like `train.py`'s
equivalent flag). Both the CLI shim (`pkm/cli/__init__.py`) and the
underlying `pkm/rl/population_train.py:main` were updated (the
hand-duplicated-shim gotcha, again). Verified: `ruff check` clean, full test
suite passes (150 tests), plus 3 new spy tests confirming
`compute_belief`-eligible construction reaches both `TorchPolicy` instances
in a pairing for both the sequential path (`run_pop_iteration`) and the
parallel path (`_play_pop_chunk`, called directly) when the classifier is
set, and neither when it isn't (`tests/test_population_train.py`).
**Deliberately not run for real** against the actual `03_pult_munki`/pool-bot
checkpoints — `population_train()`'s `finally` block unconditionally saves
every roster member's weights on exit, and clobbering hours of
only-exists-locally training work for a smoke test wasn't worth the risk
when the unit tests already exercise the real code path faithfully. A real
run (even a short one) should use throwaway agent/deck names, or accept that
it will perturb the real roster's checkpoints (by design — that's what this
flag is for).

**Phase 1 result:** `pkm/rl/eval_vs_pool.py` now wires a
`NumpyArchetypeClassifier` into both sides of every game, defaulting **on**
(decision #1 below resolved: yes, default on, since this tool exists to
measure production behavior). `--no-archetype-belief` reproduces the old
baseline. Full 26-bot re-eval: **62.7% overall** (was 60.1%), **35% vs
`pool_400_mega_abomasnow_ex`** (was 5%). Confirms the finding: the old 5%
was a measurement artifact, not a real result. See `AGENTS.md` →
"Abomasnow Matchup Investigation" for the up-to-date numbers.

**Origin:** same investigation as
`docs/superpowers/plans/2026-07-20-attack-damage-estimator.md` (see that file
for the sibling finding — a variable-damage feature blind spot). This plan
covers the *other* finding from that session: `AGENTS.md` → "Abomasnow
Matchup Investigation (2026-07-20)", Finding 1, plus What's Next #10 and #11.

---

## The story, in one paragraph

The Milestone 9 snapshot eval said `03_pult_munki` wins only 5% of games
against `pool_400_mega_abomasnow_ex`. That number doesn't reproduce: playing
20 real games through the actual production path (exported `policy.npz`,
`pkm play`) gave 45%. The gap traces to what input distribution each eval
path feeds the `opponent_archetype_belief` feature — and once that thread got
pulled, it turned out **three different code paths disagree** about whether
this feature should be live or zero, and at least one of them contradicts
what a checkpoint was actually trained on.

## The three paths, and what each one does today

| Path | Constructs `TorchPolicy(...)` with a classifier? | Result |
|---|---|---|
| `pkm/agents/neural_agent.py` — **production / `pkm play` / actual Kaggle submission** | Yes, always, auto-loaded from `pkm/archetype.npz` if the file exists (`_find_archetype_weights`, `_load_archetype_classifier`) | Live, non-zero belief every decision |
| `pkm/rl/train.py` (solo PPO training, e.g. Milestone 8) | Opt-in via `--archetype-belief [--archetype-weights ...]`, attached **only to the trainee's** `TorchPolicy`, never the frozen opponent's (`pkm/rl/rollout.py:play_one`) | Live belief when the flag is set (Milestone 8 used it); zero otherwise |
| `pkm/rl/eval_vs_pool.py` (`pkm eval-vs-pool`) | **No** — `TorchPolicy(model, greedy=True)` for both sides, no classifier arg at all, no flag to add one | Always zero belief, no way to change it |
| `pkm/rl/population_train.py` (Milestone 9) — both the sequential path (`run_pop_iteration`, inline `TorchPolicy(...)` construction) and the parallel path (`parallel_rollout.py:_play_pop_chunk`) | **No** — same as above, and there's no `--archetype-belief` flag on `population-train` at all (checked `pkm/cli/__init__.py:population_train_cmd`) | Always zero belief, no way to change it |

So: production always computes real belief. Milestone 8 (solo PPO) trained
with real belief. But Milestone 9 (population training) — which **started
from** the Milestone 8 checkpoint and ran 2375 further iterations on top of
it — trained the entire time with belief silently forced to zero, because
nothing in `population_train.py` ever attaches a classifier. Whatever the
network learned in Milestone 8 about interpreting a real belief vector then
got 2375 iterations of exposure to a completely different (constant-zero)
version of that same input slot. And `eval-vs-pool`, the tool used to measure
all of this, also always uses zero — so it's been silently evaluating on yet
a third distribution, one that doesn't match either training regime *or*
deployment.

## Why fixing this isn't just "add the flag everywhere"

Two decisions have to be made deliberately, not defaulted into:

**1. Does `eval-vs-pool` default to live belief, or stay an explicit opt-in?**
**RESOLVED (Phase 1, 2026-07-20): defaults on.** Every other place this flag
exists (`train.py`, and this plan's proposed addition to
`population_train.py`) is opt-in-during-training, which makes sense there —
you don't want to silently change what a training run is learning against.
But `eval-vs-pool`'s entire purpose is to *measure* what a checkpoint
actually does, and production always computes live belief. An eval tool
that defaults to a distribution production never uses is measuring the
wrong thing by default. Implemented: `eval-vs-pool` auto-loads
`pkm/archetype.npz` the same way `neural_agent.py` does (matching production
automatically, no flag required to get a faithful number), with an explicit
`--no-archetype-belief` escape hatch to still get the old zero-belief
baseline for comparison.

**2. In population training, which side(s) of a pairing get belief?**
**RESOLVED (Phase 2, 2026-07-20): both, symmetrically.**
`pkm/rl/rollout.py`'s existing convention (solo training) is: attach the
classifier only to the trainee's policy, never the frozen opponent's — there
*is* no "opponent" concept in population training, though. Per
`population_train.py`'s own docstring: "each side updating its own live
policy from that game's outcome" — every roster member is simultaneously a
trainee. The natural generalization is symmetric: attach the classifier to
**both** `TorchPolicy` instances in every pairing, so each member computes
its own live belief about whoever it's currently facing. This is a
deliberate departure from the solo-training asymmetric convention, worth
calling out explicitly in the code so a future reader doesn't "fix" it back
to match `rollout.py`'s pattern by mistake.

**3. What happens to the already-run Milestone 9 checkpoint?**
**RESOLVED BY ELIMINATION (2026-07-20): resume from `ppo_latest.pt`, redo
from Milestone 8 is not actually available.** The iter-2375 checkpoint was
trained entirely on belief≡0 for 2375 iterations, after starting from a
checkpoint (Milestone 8) that saw real belief. This originally posed a real
choice — redo from Milestone 8's starting point vs. resume iter-2375 with
belief newly turned on. It turned out not to be a live choice: Milestone 8's
checkpoint was silently overwritten during Milestone 9's own run (both
`train.py` and `population_train.py` write numbered snapshots to the same
directory with the same naming scheme, but independently-restarting
iteration counters — Milestone 9's own iteration 2000 clobbered the file
Milestone 8's iteration 2000 had written). See
`docs/superpowers/plans/2026-07-20-phase3-retrain-and-measure.md` Finding 1
for the full verification. So resuming from `ppo_latest.pt` isn't just the
recommended option, it's the only one, short of a full from-scratch retrain.

## Design

No new module needed here (unlike the attack-damage plan) — this is
plumbing, not new logic. `pkm/archetype/numpy_model.py:NumpyArchetypeClassifier`
and `pkm/archetype/belief.py:compute_belief` already exist and are already
proven to work (Part 2a) and to pickle cleanly across `ProcessPoolExecutor`
workers (confirmed under `train.py --workers 2`, per `AGENTS.md`). This is
entirely about getting the same classifier object threaded into the two
paths that currently never receive one.

### `pkm/rl/eval_vs_pool.py`
- Load a `NumpyArchetypeClassifier` (same pattern as `train.py`'s
  `--archetype-belief` handling — see `pkm/rl/train.py` lines ~357-361) once,
  before the per-bot loop.
- Pass it to **both** `TorchPolicy(model, greedy=True, archetype_classifier=...)`
  constructions (lines ~56 and ~64) — both sides should get live belief to
  match production, since in deployment neither the anchor nor its opponent
  is a "frozen" concept the way solo training's opponent pool is.
- New CLI options on `pkm eval-vs-pool` (`pkm/cli/__init__.py:eval_vs_pool_cmd`
  *and* `pkm/rl/eval_vs_pool.py`'s own `main()` — remember the hand-duplicated
  shim gotcha, both need the flag or it silently no-ops from the actual `pkm`
  entry point): `--archetype-belief/--no-archetype-belief` (default: **on**,
  pending the decision above) and `--archetype-weights` (default
  `pkm/archetype.npz`).

### `pkm/rl/population_train.py` + `pkm/rl/parallel_rollout.py`
- `population_train()` gains `archetype_classifier=None` param (mirrors
  `train.py`'s signature shape).
- `run_pop_iteration()` gains the same param, threads it into:
  - the sequential branch's `TorchPolicy(roster[...].model)` calls (both
    sides — see decision #2 above) — line ~197-199.
  - the parallel branch: `collect_pop_parallel()` needs a new
    `archetype_classifier` parameter, forwarded to `_play_pop_chunk` (either
    as a fixed `executor.submit` arg shared across all chunks, since it's
    the same object for every game — *not* duplicated per-`PopGame` tuple
    the way per-game state dicts are, to avoid needless pickling of the same
    classifier weights repeatedly) — `pkm/rl/parallel_rollout.py` lines
    ~148-161 (`collect_pop_parallel`) and ~103-126 (`_play_pop_chunk`).
- CLI: `pkm/cli/__init__.py:population_train_cmd` gains
  `--archetype-belief [--archetype-weights pkm/archetype.npz]`, **opt-in,
  default off** (unlike the eval-vs-pool default-on proposal above — this is
  a training-distribution decision, and flipping it retroactively changes
  what's being learned, so it should stay explicit). Same "must update both
  the CLI shim and the underlying function" trap as before.

## Phasing

**Phase 1 — `eval-vs-pool` fix.** Smaller, self-contained, no training
implications (it's read-only measurement). Gets a trustworthy number
immediately: re-run `pkm eval-vs-pool --agent 03_pult_munki --games 20` with
belief wired in, across all 26 pool bots, and see whether the 60.1%
Milestone-9-snapshot average holds up the same way the Abomasnow-specific
number didn't.

**Phase 2 — `population_train.py` + `parallel_rollout.py` plumbing.** Add the
flag and the threading described above. Validate with a small smoke test
(few iterations, throwaway roster) the same way Part 3c's `--archetype-pool`
flag was validated, plus a monkeypatch spy test analogous to
`tests/test_rl.py::test_play_one_classifier_reaches_trainee_not_opponent` —
here asserting `compute_belief` gets called for **both** `TorchPolicy`s in a
pairing when the flag is on, and for neither when it's off. Test the
parallel path (`_play_pop_chunk`) separately from the sequential path — this
codebase has already had the parallel/sequential paths drift out of sync
once (Part 3c's own history), so don't assume testing one covers the other.

**Phase 3 — decide and execute the Milestone 9 redo question.** Once Phase 2
is validated, get an explicit answer to open question #3 above (resume
iter-2370 with belief newly on, vs. restart from Milestone 8) before running
anything long. This is a multi-hour-plus training run either way, so getting
the decision right before starting matters more than for the code changes.

## Explicit non-goals (for now)

- Not touching `pkm/rl/train.py`'s existing solo-training belief wiring —
  it's already correct and already validated (Part 2a).
- Not deciding the Milestone-9-redo question in this plan — Phase 3 above
  exists precisely to force that conversation before spending compute on it.
- Not bundled with the attack-damage-estimator fix
  (`2026-07-20-attack-damage-estimator.md`) even though both surfaced from
  the same investigation — they're independent bugs with independent retrain
  implications; conflating them into one retrain would make it impossible to
  attribute any resulting win-rate change to either fix individually.

## Where to resume

Nothing is implemented. The next concrete action is Phase 1: add
`archetype_classifier` construction + wiring to
`pkm/rl/eval_vs_pool.py:eval_vs_pool()` and its two `TorchPolicy(...)` call
sites, plus the CLI flag in both `pkm/cli/__init__.py:eval_vs_pool_cmd` and
`pkm/rl/eval_vs_pool.py:main()`. Re-run the full 26-pool-bot eval before
touching `population_train.py` at all, since Phase 1's result should inform
how urgent Phase 2/3 actually are.
