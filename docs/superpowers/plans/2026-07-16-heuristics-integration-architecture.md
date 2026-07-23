# Heuristics & Learned-Belief Architecture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **You are picking this up cold.** This doc was produced from an extended
> design conversation and is meant to be fully self-contained — every claim
> about the current codebase below was verified by reading the actual files
> as of commit `e2d8357` on `master`. If something here doesn't match what
> you find in the repo, trust the repo and flag the mismatch rather than
> silently reconciling it.

**Goal:** Give the RL architecture a real seam for three capabilities that
currently have nowhere principled to live: deterministic per-decision facts
("this attack is lethal"), a per-game memory of what's been seen (deck/prize
tracking), and a learned belief about hidden information (opponent deck
archetype). Land it in an order where each piece is validated (tests pass,
measured win-rate lift, or standalone accuracy) before the next is built on
top of it.

**Architecture:** A `GameContext` (one instance per match, holding a ported
`DeckTracker`) threads through every place a decision gets encoded. A
declarative `FeatureSpec` registry replaces the current hand-appended float
lists in the encoder, with per-feature ablation and checkpoint stamping. The
network trunk splits into feature-family sub-encoders (board / deck-ledger /
learned-belief) instead of one flat concatenation, and the deck-ledger
family reuses the network's own learned card-embedding table rather than a
raw slot-indexed count vector. An opponent-archetype classifier is added as
a detached auxiliary head off the shared trunk. MCTS reads `GameContext`
once at the real root and never inside its simulated tree.

**Tech Stack:** Python 3.12, PyTorch, NumPy (torch-free mirror for Kaggle
inference), pydantic (typed observation), pytest, Optuna (existing sweep
infra), the `cabt` C engine via `pkm/engine/`.

---

## Session Status (as of 2026-07-17, branch `feature/heuristics-integration`)

**Tasks 1–8 are implemented, tested, and committed.** Task 9 was
deliberately **not** implemented — see below. Full suite: 110 passed, 1
skipped, at commit `380c829`.

| Task | Commit | Status |
|---|---|---|
| 1. Port `DeckTracker` | `d029c5b` | Done |
| 2. `GameContext` | `6ad4b9c` | Done |
| 3. Wire `GameContext` into call sites | `51d08a6` | Done |
| 4. `FeatureSpec` registry | `ecbf929` | Done |
| 5. Checkpoint stamping + numpy/torch parity | `f537fe6` | Done |
| 6. Tier-1 deterministic features | `4841bf8` | Done except retrain/measure |
| 7. Deck-ledger + board split | `92c0877` | Done except retrain/measure |
| 8. Opponent-archetype auxiliary head | `380c829` | Done except retrain/measure |
| 9. Hard-rule extension to `forced_picks` | — | **Skipped, see below** |

### What's next session (read this first)

1. **Three deferred retrain-and-measure steps** (Task 6 Step 5, Task 7 Step
   7, Task 8 Step 9) — none of the win-rate/accuracy lift claims this plan
   is supposed to validate have been empirically measured yet. All the
   *architecture* is in place and unit-tested, but nobody has run
   `pkm train` with these features on vs. ablated and looked at the
   numbers. Do this before trusting any of it in a real submission.
   `pkm/policy.npz` locally right now (if present) is just a throwaway
   random-init export used to smoke-test the pipeline end-to-end — not a
   trained checkpoint.
2. **Task 9 decision needed.** Attempted during this session and
   deliberately abandoned: every candidate hard rule considered either
   turned out to already be covered by the existing structural checks
   (`n==1`, `minCount==maxCount==n`), or required Pokémon TCG rules
   certainty("is taking a lethal attack ever wrong?", "are these two
   options truly interchangeable?") that wasn't available in this session
   — and a *wrong* forced-pick rule is worse than no rule, since it
   silently overrides the network's decision with no way to detect the
   error from outside. Options for next session: (a) come with a specific
   verified rule in hand, (b) accept a narrower, mechanically-certain rule
   (a sketch was drafted — see the question/answer trail in the
   conversation that produced this update — "all reveal options resolve
   to the same card_id" is mechanically safe but low-payoff), or (c)
   formally close out Task 9 as not-worth-building and consider the plan
   complete at Task 8.
3. **`AGENTS.md` still needs its "Current Progress"/"What's Working"
   tables updated** per this plan's own Verification checklist and
   `AGENTS.md`'s own convention — not yet done as of this status update.

### Deviations from the plan's stated File Map (flagged, not silent)

The plan's own header says to flag mismatches rather than silently
reconcile them. Each task above touched at least one file beyond what its
own "Files:" list named, always because the literal file list undercounted
what correctness required:

- **Task 5** also touched `pkm/rl/numpy_policy.py`, `pkm/rl/train.py`,
  `pkm/rl/exit_train.py` — the .npz/.pt read/write points the stamping
  actually needed, not just `export.py`/`profile.py`.
- **Task 7** also touched `pkm/rl/rollout.py`, `pkm/rl/exit_train.py`,
  `pkm/agents/neural_agent.py`, `pkm/mcts/agent.py` — without threading
  `ctx` through these, the deck ledger would be permanently empty at both
  train and play time (Tasks 3/4 had left that threading as an unused
  stub, `ctx=None` always).
- **Task 8**: the actual `pi_loss`/`v_loss` combine site is
  `pkm/rl/ppo.py`, not `pkm/rl/train.py` as the plan names — the aux loss
  was wired in there instead. Also touched `pkm/heuristics/context.py`
  (`GameContext` gained `archetype_belief`) and `pkm/rl/rollout.py` — not
  in the plan's file list, but the re-injection mechanism the plan
  requires (a real per-game belief, updated after each decision) needed
  somewhere to live.
- **Task 8's re-injection mechanism** was a genuine design decision the
  plan didn't specify (it named the shape — detached softmax, GLOBAL
  scope — not the wiring). Chosen: ctx-mediated, one-decision-stale
  belief, not same-decision two-pass re-encoding. See Task 8's section
  below and the commit message on `380c829` for the reasoning.
- **Task 7's board split** ("separate my slots from opponent's slots
  before pooling") was underspecified in the same way; the user chose
  mean-pooling each side (matching the existing hand-pooling pattern)
  over a same-width regroup-only alternative. See commit `92c0877`.

---

## Context & Motivation

Three things need a home in this codebase, and none of them fit today:

1. **Deterministic heuristics** — exact facts computable from the current
   board: "would this attack knock out the target," "how many copies of
   card X have I not seen yet." Already speced in `plan.md` (repo root,
   currently untracked — read it first, this plan implements it with one
   deliberate correction, see Task 6).
2. **A per-game memory** — a prototype already exists, unmerged, on branch
   `refactor-to-prepare-for-heuristics-integration`
   (`pkm/heuristics/deck_tracker.py`, class `DeckTracker`): it identifies
   every one of your own 60 deck cards by its engine-assigned `serial` the
   first time it's seen in a public zone, and — critically — deduces prize
   pile contents when a search effect reveals your *entire* remaining deck
   (by elimination: whatever's still unbound has nowhere left to be). That
   branch diverged from `master` at commit `39a1964`, *before* the
   vendored-engine work and the `obs.py` restructure, so it cannot be
   merged as-is — it needs a targeted port (Task 1).
3. **A learned opponent-archetype belief** — doesn't exist yet. Already
   anticipated in `plan.md` §8 as a shared-trunk auxiliary head, detached
   before re-injection so PPO gradient can't corrupt it into predicting
   "whatever helps win rate" instead of the true archetype.

### The mental model (read this before touching code)

```
                    ┌───────────────────────────┐
                    │       GameContext            │   one per match,
                    │  (owns a fresh DeckTracker)  │   never reused across games
                    └─────────────┬────────────────┘
                                  │ facts
                ┌─────────────────┴─────────────────┐
                ▼                                    ▼
    ┌────────────────────────┐        ┌─────────────────────────┐
    │  DETERMINISTIC FACTS     │        │  LEARNED BELIEFS          │
    │  plain game-logic math,  │        │  opponent-archetype head, │
    │  always correct if       │        │  detached before          │
    │  correctly computed      │        │  re-injection              │
    └────────────┬─────────────┘        └─────────────┬─────────────┘
                │                                     │
                └────────────────┬────────────────────┘
                                  ▼
                      ┌───────────────────────┐
                      │      THE TRUNK           │
                      │  (PolicyValueNet, h)      │
                      └─────────────┬────────────┘
                                    │ guided by
                                    ▼
                      ┌───────────────────────┐
                      │      SEARCH (MCTS)      │
                      │  reads GameContext ONCE  │
                      │  at the root, never       │
                      │  inside the tree           │
                      └───────────────────────┘
```

The same `GameContext`/tracker fact is consumed **two ways**, not built
twice: as a soft, learned input feature (via the registry, Task 4+), and —
for facts certain enough to not leave to gradient descent — as a hard rule
extending `forced_picks` (Task 9). One computation, two readers.

---

## Guardrails

- `just test` (`python -m pytest tests/ -q`) must stay green after every task.
- Never construct or reuse a `DeckTracker`/`GameContext` across game
  boundaries — self-play runs many games back-to-back; a leaked reference
  silently contaminates one game's prize knowledge into an unrelated game.
  Every task touching this must include the no-leak test (Task 1).
- `GameContext.tracker.observe()` must **never** be called from inside
  `pkm/mcts/search.py`'s `_expand`/`_simulate` — those run on hypothetical
  *determinized* branches, not real history. Feeding imagined observations
  into the real tracker silently corrupts the AI's actual beliefs about the
  live game. MCTS may read `GameContext` read-only, once, before search
  starts (to bias `sample_determinization`).
- Every checkpoint-breaking change (any tensor width change) is an accepted
  cost, per `plan.md` §3 — training volume so far is small (~200 PPO iters,
  1 expert-iteration run) so retraining from scratch is cheap. This is only
  safe *because* of checkpoint stamping (Task 5) — never skip that task to
  save time.
- Do not build a general `Stage`/`Pipeline` framework
  (`docs/ideas/agent-composition-and-refactor.md` §5) before a second real
  hard-rule heuristic exists to justify it. Task 9 extends `forced_picks`
  directly; it does not introduce a new framework.
- Do not build the multi-branch trunk's learned gating weights (Task 7)
  before at least two real feature families exist and Task 6's flat
  version has measured a lift. Building tuning infrastructure before there
  is anything to tune is waste.
- Do **not** build full entity self-attention over the board (AlphaStar-
  style) as part of this plan — see "Explicitly deferred" below. If you
  find yourself reaching for `nn.MultiheadAttention` anywhere in this plan,
  stop and re-read that section.
- `pkm/rl/numpy_policy.py` is a **hand-written mirror** of
  `pkm/rl/model.py`'s forward pass, used for torch-free Kaggle inference.
  There is currently no automated check that the two agree. Any task that
  changes `model.py`'s forward pass **must** update `numpy_policy.py` in
  the same task and must not be marked done until the new parity test
  (Task 5) passes for it.

### Explicitly deferred (do not build in this plan)

- **Full entity self-attention** over board/state. Correct eventual shape,
  wrong ROI now: self-play volume is on the order of a few thousand games
  total, and there is no CI parity harness yet for the numpy mirror — every
  new layer type widens that untested gap. Revisit once (a) Task 5's parity
  test exists and is trusted, and (b) self-play volume grows by roughly an
  order of magnitude.
- **Contrastive/metric-learning archetype embedding** (predicting a point in
  deck-embedding space instead of a fixed-class softmax, to generalize to
  un-enumerated archetypes). Real complexity — its own training procedure.
  Only worth it after the fixed-class version (Task 8) is built, validated,
  and has hit its ceiling.
- **General opponent-side deck ledger** (assuming their decklist from a
  guessed archetype and running Tier-2-style math against the assumption) —
  `plan.md` §6.1 already calls this out as depending on the archetype
  classifier existing and being validated first. Do not build ahead of Task 8.

---

## Current State Audit (verified against `master` @ `e2d8357`)

Read these before starting; this plan assumes you have:

- `plan.md` (repo root, untracked) — the deterministic-feature tier design
  (Tiers 1–4), the `FeatureSpec` registry sketch, checkpoint-stamping
  rationale, and the auxiliary-head design rules (§8). This plan implements
  it with **one deliberate correction**: `plan.md` §5 locks in a flat
  60-slot count vector for the deck ledger; Task 6 below replaces that with
  a pooled-embedding representation instead. Do not build the flat version.
- `docs/ideas/agent-composition-and-refactor.md` — composition vocabulary
  (Pipeline/Injection/Delegation), the code map, and the "don't build the
  cathedral" philosophy this plan follows for Tasks 7 and 9. Note: its
  smell #3 ("forced-decision handling copy-pasted 4×") is **already fixed**
  on `master` — `pkm/types/obs.py:266` (`Select.forced_picks()`) and
  `:276` (module-level `forced_picks(sel: dict)`) are now the single
  canonical implementation (commit `8f8bb2a`). Don't re-fix it.

Key files and what they currently do:

| File | Current role |
|---|---|
| `pkm/rl/encoder.py` | `encode_state`/`encode_options`/`encode_decision` — hand-appended float lists into `EncodedDecision`. `STATE_FEATS`/`OPT_FEATS` (lines 57-60) are hand-maintained constants. `Norm` dataclass (line 29) holds normalization divisors. Pure functions of `Observation` only — **no access to any per-game memory**. |
| `pkm/rl/model.py` | `PolicyValueNet` — `card_emb`/`attack_emb`/`opt_type_emb` (line 67-69, `EMB_CARD=32`), `encode_state` (line 87) flattens board + mean-pools hand + concats `state_feats` into one `state_fc1` (line 73). Policy head scores options against `h`; value head is 2-layer MLP off `h` (line 82). |
| `pkm/rl/numpy_policy.py` | `NumpyPolicy` — hand-replays the above forward pass in raw numpy for the Kaggle bundle. Docstring literally says "must stay in sync with model.py." No automated parity check exists. |
| `pkm/mcts/search.py` | `MCTS._expand`/`_simulate` — calls `encode_decision` on **hypothetical determinized** branches (`node.state.observation`), not real game history. This is the boundary Guardrails above protects. |
| `pkm/rl/rollout.py` | `play_game` — one of two duplicated per-game loops (`battle_start`/`while`/`battle_select`/`battle_finish`). Used by `train.py`. |
| `pkm/rl/exit_train.py` | `play_exit_game` (line 64) — the *other* duplicated per-game loop, used by Phase 2. |
| `pkm/agents/neural_agent.py` | `make_neural_agent(deck, weights_path)` returns a closure called once per match by `pkm/rl/play.py`. Stateless today. |
| `pkm/mcts/agent.py` | `make_mcts_agent(...)` — same shape as above, plus owns the `MCTS` instance. |
| `pkm/tui/session.py` | Drives the human-play loop; not yet read in detail — treat as a 5th per-game construction site, confirm during Task 3. |
| `pkm/heuristics/deck_tracker.py` (branch only) | `DeckTracker` — `CardLocation` enum, `CardState` dataclass, `observe(obs)`, `is_search_reveal(obs)`, `record_search_reveal(obs)`, `by_location(loc)`, `known_prizes()`. Imports `from pkm.types.obs import Observation` — verify this import path is still correct after the port (obs.py has moved/changed since the branch's fork point). |
| `pkm/agents/singaporean_middleman.py` (branch only) | Already prototypes the right per-episode construction pattern: `state = {"tracker": DeckTracker(deck)}` built inside the agent factory closure. Reuse this pattern for `GameContext`; do **not** port the dispatcher/routing logic itself — that's a separate, out-of-scope concern (delegation, not a heuristic input). |
| `pkm/types/obs.py` | `forced_picks` (both the free function, line 276, and `Select.forced_picks()`, line 266) — canonical, already deduplicated. `NUM_CARDS=1268`, `N_BOARD_SLOTS`, `MAX_HAND=25`, `MAX_BENCH=8` etc. live here. |
| `pkm/agents/profile.py` | `AgentProfile` — `checkpoint_dir`, `latest_checkpoint(phase)`, `ppo_init()`/`exit_init()`. Checkpoint stamping (Task 5) needs to hook in here. |

---

## File Map

### New files

- `pkm/heuristics/__init__.py` — ported from the branch.
- `pkm/heuristics/deck_tracker.py` — ported `DeckTracker` (Task 1).
- `pkm/heuristics/context.py` — `GameContext` dataclass (Task 2).
- `pkm/rl/features.py` — `FeatureSpec`, `Scope`, registry, `FeatureConfig`,
  checkpoint stamping helpers (Task 4).
- `pkm/rl/deterministic_features.py` — Tier-1 feature implementations
  (Task 6): `lethal_this_turn`, `type_effectiveness`, `retreat_viable`.
- `tests/test_deck_tracker.py` — ported/adapted tests + the no-leak
  invariant test (Task 1).
- `tests/test_game_context.py` — construction/lifecycle tests (Task 2).
- `tests/test_feature_registry.py` — registry mechanics, ablation,
  checkpoint stamp mismatch (Task 4).
- `tests/test_numpy_torch_parity.py` — forward-pass parity between
  `PolicyValueNet` and `NumpyPolicy` on random inputs (Task 5).
- `tests/test_tier1_features.py` — fixture-based exact-value tests
  (Task 6).
- `tests/test_deck_ledger.py` — pooled-embedding ledger feature test
  (Task 7).
- `tests/test_archetype_head.py` — standalone classification accuracy test
  (Task 8).

### Existing files to modify

- `pkm/rl/encoder.py` — `encode_state`/`encode_options` become registry-
  driven loops (Task 4); gain `ctx: GameContext` parameter throughout.
- `pkm/rl/model.py` — trunk split into family sub-encoders (Task 7);
  auxiliary head added (Task 8).
- `pkm/rl/numpy_policy.py` — mirror every `model.py` change, same task.
- `pkm/rl/rollout.py`, `pkm/rl/exit_train.py` — construct `GameContext`
  per game, call `ctx.tracker.observe(obs)` before each encode (Task 3).
- `pkm/agents/neural_agent.py`, `pkm/mcts/agent.py` — construct
  `GameContext` inside the factory closure (Task 3).
- `pkm/tui/session.py` — same, human-play session (Task 3).
- `pkm/mcts/search.py` — confirm/enforce the tracker-boundary guardrail
  (Task 3); optionally read `GameContext` read-only at `choose()`'s root to
  bias `sample_determinization` (stretch, not required for Task 3 to land).
- `pkm/rl/train.py` — combine archetype-head loss with `pi_loss`/`v_loss`
  (Task 8).
- `pkm/rl/export.py`, `pkm/agents/profile.py` — checkpoint stamping
  read/write (Task 5).
- `pkm/types/obs.py` — extend `forced_picks`-style logic for the first hard
  rule (Task 9); do not generalize into a framework.
- `AGENTS.md` — document the new architecture once landed (repo convention:
  "update this file whenever something significant changes").

---

## Task 1: Port `DeckTracker` onto `master`

**Files:**
- Create: `pkm/heuristics/__init__.py`, `pkm/heuristics/deck_tracker.py`
- Create: `tests/test_deck_tracker.py`

- [x] **Step 1: Diff the source.** On the `refactor-to-prepare-for-heuristics-integration`
  branch, `git show refactor-to-prepare-for-heuristics-integration:pkm/heuristics/deck_tracker.py`.
  Compare its `from pkm.types.obs import Observation` against current
  `master`'s `pkm/types/obs.py` — confirm `Observation`, `Player`,
  `PokemonRef`, `CardRef` still have the fields `DeckTracker.observe` reads
  (`hand`, `discard`, `prize`, `active`/`bench`, `energyCards`, `tools`,
  `preEvolution`, `stadium`, `yourIndex`, `deckCount`). Port only this file
  and its direct dependency on `pkm.types.obs` — do **not** bring over
  `pkm/agents/singaporean_middleman.py`, `pkm/rl/parallel_rollout.py`,
  `battle.sh`, or the replay-viewer deletions from that branch; they're
  unrelated changes that happened to land on the same branch.

- [x] **Step 2: Write failing tests.** Port the intent of the branch's
  behavior into `tests/test_deck_tracker.py` using
  `tests/fixtures/observations.json` (the real captured-engine fixture —
  do **not** use `example_obs.json`, which is hand-written and wrong per
  `CLAUDE.md`). Cover: binding a card by serial once seen in hand/board/
  discard/prize, `is_search_reveal` returning `False` for a filtered search
  (length mismatch vs `deckCount`) and `True` only for a genuine full-deck
  reveal, `record_search_reveal` correctly deducing prizes by elimination,
  and — the critical one — **construct two `DeckTracker` instances for two
  distinct fixture "games" and assert neither's state leaks into the
  other.**

- [x] **Step 3: Run and confirm failure** (`python -m pytest
  tests/test_deck_tracker.py -q`) — expected FAIL, module doesn't exist yet.

- [x] **Step 4: Port the file**, fixing only the import path if
  `pkm.types.obs` has changed shape since the branch's fork point.

- [x] **Step 5: Run and confirm pass.**

- [x] **Step 6: Commit.**
  ```bash
  git add pkm/heuristics/__init__.py pkm/heuristics/deck_tracker.py tests/test_deck_tracker.py
  git commit -m "port DeckTracker from refactor-to-prepare-for-heuristics-integration"
  ```

## Task 2: `GameContext`

**Files:**
- Create: `pkm/heuristics/context.py`
- Create: `tests/test_game_context.py`

- [x] **Step 1: Write failing tests.** `GameContext(my_deck, tracker,
  opp_decklist=None)` constructs cleanly; two `GameContext`s built from the
  same deck list produce independent `DeckTracker` instances (no shared
  mutable default).

  ```python
  @dataclass
  class GameContext:
      my_deck: list[int]
      tracker: DeckTracker
      opp_decklist: list[int] | None = None
  ```

- [x] **Step 2: Run, confirm failure.**

- [x] **Step 3: Implement** exactly the dataclass above — this task is
  intentionally thin. Do not add a factory/builder yet; that's Task 3's job
  once you see all 5 call sites side by side and know what's actually
  shared.

- [x] **Step 4: Run, confirm pass.**

- [x] **Step 5: Commit.**

## Task 3: Wire `GameContext` into every per-game call site

**Files:**
- Modify: `pkm/rl/rollout.py`, `pkm/rl/exit_train.py`,
  `pkm/agents/neural_agent.py`, `pkm/mcts/agent.py`, `pkm/tui/session.py`
- Modify (guardrail only, see Step 5): `pkm/mcts/search.py`

This task is pure plumbing — no new features, no behavior change other than
`GameContext` existing and being kept current. `encode_decision` does **not**
change signature yet (that's Task 4); for now just construct and update the
context alongside each loop.

- [x] **Step 1: `rollout.py:play_game`.** Construct `GameContext(deck, DeckTracker(deck))`
  per side (two contexts, one per player, since each player only tracks
  their own deck) right after `battle_start`, before the `while` loop. Call
  `ctx[player].tracker.observe(obs)` each iteration, and
  `ctx[player].tracker.record_search_reveal(obs)` when
  `ctx[player].tracker.is_search_reveal(obs)` — mirror the exact call
  sequence already proven in `pkm/agents/singaporean_middleman.py`'s
  `agent()` closure (lines 98-106 on the source branch).

- [x] **Step 2: `exit_train.py:play_exit_game`.** Same pattern, second
  duplicated loop. (Do not use this task to merge the two loops into one
  shared driver — that's the `agent-composition-and-refactor.md` §6
  `collect_trajectory` refactor, out of scope here. Just don't let the
  duplication grow a third way by hand-copying incorrectly.)

- [x] **Step 3: `neural_agent.py:make_neural_agent` and
  `mcts/agent.py:make_mcts_agent`.** Each already builds one closure per
  match (called once per game via `pkm/rl/play.py`). Construct
  `GameContext` inside the factory function body, exactly like
  `singaporean_middleman.py`'s `state = {"tracker": DeckTracker(deck)}`.
  Update it (`observe`, reveal-check) at the top of the returned
  `agent(obs)` function, before any decision logic runs.

- [x] **Step 4: `pkm/tui/session.py`.** Read the file first (not yet
  inspected as of this plan being written) to find the session's per-match
  construction point; apply the same pattern.

- [x] **Step 5: Enforce the MCTS boundary.** In `pkm/mcts/search.py`,
  confirm `_expand`/`_simulate` take no `GameContext` parameter and cannot
  reach one — this should be true by construction if you didn't add one,
  but add a comment at `_Node.__init__` and `MCTS._expand` explicitly
  stating why (imagined branches must never touch the real tracker) so the
  next person doesn't "fix" this by threading `ctx` through by habit.

- [x] **Step 6: Sanity match.** Run `pkm play --p0 mcts --p1 neural` (or
  `just play mcts neural`) end to end — must complete without exceptions.
  This is the practical check that the tracker wiring didn't break MCTS's
  determinization path.

- [x] **Step 7: Run full suite** (`just test`), then commit.

## Task 4: `FeatureSpec` registry + `FeatureConfig` ablation

**Files:**
- Create: `pkm/rl/features.py`
- Modify: `pkm/rl/encoder.py`
- Create: `tests/test_feature_registry.py`

Behavior-preserving: re-implements today's exact features through the new
mechanism. Output must be numerically identical to the pre-refactor encoder
on the same fixture inputs — this task adds no new signal.

- [x] **Step 1: Write failing tests.** A `FeatureSpec(name, width, scope,
  fn, deterministic)` where `scope ∈ {GLOBAL, PER_SLOT, PER_OPTION,
  PER_DECK_CARD}`; registering specs and summing their widths reproduces
  today's `STATE_FEATS`/`OPT_FEATS` constants; a `FeatureConfig` that
  disables one spec zero-masks its output slice without changing total
  width; loading a checkpoint stamped with a different registered-feature
  list raises loudly (don't implement the stamping mechanism itself yet if
  it's easier to stub — but the *test* for "must fail loudly on mismatch"
  belongs here since it's the registry's contract).

  ```python
  class Scope(Enum):
      GLOBAL = auto()
      PER_SLOT = auto()
      PER_OPTION = auto()
      PER_DECK_CARD = auto()

  @dataclass
  class FeatureSpec:
      name: str
      width: int
      scope: Scope
      fn: Callable[[Observation, GameContext], np.ndarray]
      deterministic: bool
  ```

  Note the `fn` signature takes **`(obs, ctx)`**, not just `obs` as
  originally sketched in `plan.md` §2 — this is the one change needed to
  make `GameContext`-backed features (Tasks 6-8) expressible at all.
  Deterministic Tier-1 features simply ignore `ctx`.

- [x] **Step 2: Run, confirm failure.**

- [x] **Step 3: Implement the registry** in `pkm/rl/features.py`: a
  `GLOBAL_FEATURES: list[FeatureSpec]`, `PER_SLOT_FEATURES: list[FeatureSpec]`,
  `PER_OPTION_FEATURES: list[FeatureSpec]` module-level list (registration
  order = tensor layout order, and is the single source of truth for
  width — no more hand-maintained integers). Re-express every scalar
  currently appended by hand in `encode_state`/`encode_options`
  (`pkm/rl/encoder.py` lines 129-208, 261-347) as one `FeatureSpec` each.

- [x] **Step 4: Rewrite `encode_state`/`encode_options`** to assemble
  their output by iterating registered specs in order and concatenating,
  instead of the current hand-built list. `STATE_FEATS`/`OPT_FEATS`
  (`encoder.py:57-60`) become `sum(f.width for f in ...)`, computed once at
  import time.

- [x] **Step 5: Numerical equivalence check.** Run the existing encoder
  tests (`tests/test_obs.py` and any encoder-specific tests) against
  `tests/fixtures/observations.json` before and after this refactor and
  diff the output arrays — must be bit-identical. This is the actual
  acceptance criterion for "behavior-preserving," not just "tests still
  pass."

- [x] **Step 6: Run full suite, commit.**

## Task 5: Checkpoint stamping + numpy/torch parity test

**Files:**
- Modify: `pkm/rl/export.py`, `pkm/agents/profile.py`
- Create: `tests/test_numpy_torch_parity.py`

Two independent safety nets, bundled into one task because both are
prerequisites for every task after this point being *safely* buildable —
land them before touching the trunk (Task 7) or the auxiliary head
(Task 8).

- [x] **Step 1: Write failing checkpoint-stamp test.** Saving a checkpoint
  records the ordered `(name, width)` list from Task 4's registry
  (alongside the existing `.pt` state dict, and in the exported `.npz` for
  Kaggle). Loading a checkpoint against a registry whose stamp doesn't
  match raises a clear error naming the mismatch, rather than silently
  loading garbage into misaligned tensor slices.

- [x] **Step 2: Write failing parity test.** Build a `PolicyValueNet` with
  random-initialized weights, export it via the existing `pkm/rl/export.py`
  path into a `NumpyPolicy`, run both on the same batch of random
  `EncodedDecision`s (respecting real shapes: `NUM_CARDS`, `NUM_ATTACKS`
  bounds etc. from `pkm/types/obs.py`), and assert `value()`/`priors()`
  agree within float32 tolerance. This is the CI parity check
  `docs/ideas/agent-composition-and-refactor.md` §2 names as currently
  missing and the biggest fragility in the codebase.

- [x] **Step 3: Run, confirm both fail.**

- [x] **Step 4: Implement stamping** — hook into `pkm/rl/export.py`
  (checkpoint → `.npz`) and `pkm/agents/profile.py` (`latest_checkpoint`/
  `ppo_init`/`exit_init`) so the check happens automatically on load, not
  just when a test calls it directly.

- [x] **Step 5: Implement/confirm parity** — if the parity test fails on
  today's *unmodified* `model.py`/`numpy_policy.py` pair, that's a real bug
  to fix now, before any further architecture changes make debugging it
  harder.

- [x] **Step 6: Run full suite, commit.**

## Task 6: Tier-1 deterministic features

**Files:**
- Create: `pkm/rl/deterministic_features.py`
- Modify: `pkm/rl/features.py` (registration)
- Create: `tests/test_tier1_features.py`

Per `plan.md` §4: `lethal_this_turn` (`PER_OPTION`, attack options only),
`type_effectiveness` (`PER_OPTION`, attack options only), `retreat_viable`
(`PER_SLOT`, bench). All pure functions of `(obs, ctx)` that ignore `ctx`
(no memory needed) and reuse `pkm/data/card_data.py`'s `weakness`/
`resistance`/`retreat_cost`/`energy_type`/attack `damage`/`energies`.

- [x] **Step 1: Write fixture-based failing tests** — one hand-computed
  expected value per feature against `tests/fixtures/observations.json`
  (or a purpose-built minimal fixture if the existing one doesn't cover a
  lethal/type-matchup/retreat scenario — check first before adding a new
  fixture). These are exact-value tests, not statistical — no
  approximate-equality tolerance needed.

- [x] **Step 2: Run, confirm failure.**

- [x] **Step 3: Implement each feature function**, register as
  `FeatureSpec(deterministic=True)` entries in `pkm/rl/features.py`.

- [x] **Step 4: Run, confirm pass. Run full suite.**

- [ ] **Step 5: Retrain and measure.** Using the existing profile/training
  CLI (`pkm train --agent <profile> --eval-every N`, see `AGENTS.md` → "RL
  Training"), run once with these features enabled and once with them
  ablated via `FeatureConfig` (Task 4), same seed/iteration count. Record
  the eval-win-rate delta in this checkbox item's commit message or a
  short note in `AGENTS.md`'s "What's Next" — this measurement is what
  justifies (or doesn't) proceeding to Task 7.

- [x] **Step 6: Commit.**

## Task 7: Deck-ledger feature family via pooled card embeddings

**Files:**
- Modify: `pkm/rl/model.py`, `pkm/rl/numpy_policy.py`, `pkm/rl/features.py`
- Create: `tests/test_deck_ledger.py`

**This is the one deliberate deviation from `plan.md`.** §5 there locks in
a flat, fixed-width-60, slot-indexed count vector for the deck ledger. Do
**not** build that. Build this instead:

```
h_memory = Σ_c  unseen_count[c] · card_emb[c]
```

— a count-weighted pool over the network's *existing* `card_emb` table
(`pkm/rl/model.py:67`, `EMB_CARD=32`), where `unseen_count[c]` comes from
`ctx.tracker.by_location(CardLocation.DECK)` (Task 1/2). This reuses
learned card identity instead of an arbitrary per-deck slot position,
produces a fixed 32-wide vector regardless of decklist composition (no new
embedding table, no 60-wide cap), and — unlike the slot-indexed version —
generalizes across decks for free, which matters directly for `AGENTS.md`'s
"multi-deck training" roadmap item.

- [x] **Step 1: Write failing tests.** Given a small fixture decklist and a
  hand-set tracker state (some cards seen, some not), the pooled vector
  equals the hand-computed weighted sum of the corresponding `card_emb`
  rows. Also test the degenerate case (nothing seen yet → pool equals the
  full-decklist-weighted sum; everything seen → zero vector).

- [x] **Step 2: Run, confirm failure.**

- [x] **Step 3: Split the board tower.** While in `model.py`'s
  `encode_state`, also do the board-family split flagged in the design
  conversation: separate "my slots" from "opponent's slots" before pooling,
  rather than the current flat positional concatenation
  (`model.py:87-99`). This is bundled into this task because both changes
  touch `encode_state`'s structure together; keep them as separate commits
  if you want cleaner history, but land both before moving on — don't
  restructure the trunk twice.

- [x] **Step 4: Implement the pooled deck-ledger family and the board
  split** in `model.py`, combining family outputs into `h` via **simple
  concatenation for now** — do not add learned gating weights in this task
  (see Guardrails: gating comes after ≥2 families exist and Task 6 has a
  measured baseline, which it now does from Step 5 above).

- [x] **Step 5: Update `numpy_policy.py` to match exactly**, then re-run
  Task 5's parity test — it must still pass. If it doesn't, the mirror is
  out of sync; fix it before proceeding.

- [x] **Step 6: Run full suite.**

- [ ] **Step 7: Retrain and measure**, isolated from Task 6's already-
  measured lift (same protocol as Task 6 Step 5).

- [x] **Step 8: Commit.**

## Task 8: Opponent-archetype auxiliary head

**Files:**
- Modify: `pkm/rl/model.py`, `pkm/rl/numpy_policy.py`, `pkm/rl/train.py`,
  `pkm/rl/features.py`
- Create: `tests/test_archetype_head.py`

Per `plan.md` §8. Pick the initial archetype class list now, deliberately —
growing it later is a breaking change (§3 accepted cost, but still worth
minimizing churn). Candidate list based on decks that exist today:
`00_basic`, `01_psychic`, `02_dragapult`, plus one `Other` bucket to absorb
anything unrecognized. Confirm with whoever owns training-run scheduling
before locking this in if more decks are expected soon.

- [x] **Step 1: Write failing standalone-accuracy test.** Add
  `archetype_fc1`/`archetype_fc2` off `h` (`model.py`, same shape pattern
  as existing `value_fc1`/`value_fc2` at lines 82-83), output width =
  `N_ARCHETYPES + 1`. With the rest of the network **frozen** (or just a
  freshly initialized trunk, per `plan.md` §8.2 rule 3 — "pretrain/validate
  standalone before wiring into the re-injection path"), train the head
  alone against ground-truth deck identity (both decks known during
  self-play) and assert classification accuracy clears some non-trivial
  threshold (e.g. well above the `1/N_ARCHETYPES` random baseline) on a
  held-out set of self-play games.

- [x] **Step 2: Run, confirm failure** (head doesn't exist).

- [x] **Step 3: Implement the head and its standalone training path.** Do
  not wire re-injection yet — this step only proves the head can learn the
  label at all.

- [x] **Step 4: Run, confirm the accuracy threshold passes.**

- [x] **Step 5: Wire the detached re-injection.** `.detach()` the head's
  softmax output before it joins `h_belief` and gets registered as a
  `GLOBAL`-scope `FeatureSpec` (Task 4's registry) feeding back into the
  trunk. This is the non-negotiable line from `plan.md` §8.2 rule 1 — get
  this wrong and the head silently degrades into predicting "whatever
  helps win rate," which is both wrong and unfalsifiable from the outside.
  Add a unit test asserting `.grad` does not flow from the policy/value
  loss back through the archetype head's parameters (construct a tiny
  forward+backward pass and check `archetype_fc1.weight.grad` is `None` or
  zero when only `pi_loss`/`v_loss` are backpropped).

- [x] **Step 6: Add the auxiliary loss term.** In `pkm/rl/train.py`, at the
  point `pi_loss`/`v_loss` already combine, add `+ λ · aux_loss`
  (cross-entropy) with its own tunable `λ`. Update `numpy_policy.py` to
  mirror the new head structure (even though the head's *output* feeds
  back as a detached feature and therefore only needs to exist for
  inference, not gradient, in the numpy mirror).

- [x] **Step 7: Re-run Task 5's parity test** — must still pass.

- [x] **Step 8: Run full suite.**

- [ ] **Step 9: Retrain and measure**, isolated from Tasks 6/7's lift.

- [x] **Step 10: Commit.**

## Task 9: First hard-certain rule, extending `forced_picks`

**Files:**
- Modify: `pkm/types/obs.py`
- Create/modify tests alongside the existing `forced_picks` tests.

Per Guardrails: do not build a `Stage`/`Pipeline` framework. Extend the
already-canonical `forced_picks` (`pkm/types/obs.py:266`/`:276`) with one
concrete, certain rule that reads the same `ctx.tracker` call as Task 6/7's
soft features — one computation, hard-rule consumer. Concrete first
candidate: if `ctx.tracker` shows zero remaining copies of the only attack
option that could matter this turn (or an equivalent certain-and-narrow
case identified while implementing Task 6's `lethal_this_turn`), short-
circuit rather than leaving it to the network. Pick the exact condition
during implementation once Task 6 exists to draw on — don't over-specify it
here; the point of this task is proving the "same fact, two consumers"
pattern once, not enumerating rules.

> **Status: not attempted, deliberately.** Once `lethal_this_turn` (Task 6)
> existed to draw on, no candidate condition survived scrutiny as a genuine
> *no-real-choice* fact rather than a strategic judgment call in disguise
> ("always take the kill" is a policy preference, not a certainty — forcing
> it would silently override the network instead of teaching it) or
> something the existing structural checks already cover. Encoding the
> wrong rule here is worse than skipping the task: forced picks bypass the
> network entirely with no way to detect the error from outside. See
> "What's next session" above for how to unblock this.

- [ ] **Step 1: Write a failing test** for the specific chosen condition
  against a fixture (or hand-built `Observation`) demonstrating the forced
  outcome.
- [ ] **Step 2: Run, confirm failure.**
- [ ] **Step 3: Implement**, reading from `ctx.tracker`/`GameContext`
  exactly as Task 6/7's registry features do — do not duplicate the
  tracker query logic.
- [ ] **Step 4: Run, confirm pass. Run full suite.**
- [ ] **Step 5: Commit.**

---

## Verification (cumulative, check after every task)

- `just test` (`python -m pytest tests/ -q`) green.
- `just lint` (`ruff check pkm/ tests/`, `ruff format --check pkm/ tests/`)
  clean.
- `pkm play --p0 mcts --p1 neural --agent <profile>` completes without
  exceptions (confirms the `GameContext`/MCTS boundary holds).
- After Task 5 lands: `tests/test_numpy_torch_parity.py` passing is a hard
  gate for every subsequent task that touches `model.py`.
- After each of Tasks 6/7/8: an eval-win-rate comparison (feature/head on
  vs. ablated), not just "trains without crashing" — record the delta.
  **Not yet done for any of the three as of this update — see "Session
  Status" at the top of this doc.**
- Before considering this plan complete: update `AGENTS.md`'s "Current
  Progress" and "What's Working" tables per its own stated convention
  ("update this file whenever something significant changes"). **Not yet
  done.**

## Open questions to flag back (do not silently resolve)

- ~~Exact archetype class list (Task 8)~~ — **Resolved:** `00_basic`,
  `01_psychic`, `02_dragapult`, plus one reserved "Other" slot. Locked in
  at commit `380c829`.
- ~~Whether `pkm/tui/session.py`'s structure (Task 3, Step 4) matches the
  "one closure per match" pattern~~ — **Resolved:** it does; wired the
  same way as the other 4 call sites at commit `51d08a6`.
- The exact condition for Task 9's first hard rule — **still open.** No
  candidate survived scrutiny this session as a genuine no-choice fact
  rather than a strategic judgment or an already-covered structural case.
  See the note under Task 9 above and "What's next session" up top.
