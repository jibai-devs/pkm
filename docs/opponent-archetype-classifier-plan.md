# Opponent-Archetype Classifier + RL Integration

## Context

`staples.json` holds per-archetype staple-card composition data (~10 meta archetypes,
~15-20 staples each, with copy-count distributions) scraped from limitlesstcg. The
existing design note (`docs/ideas/multi-phase-policy-and-opponent-modeling.md`)
already reasoned through *why* this should be supervised learning feeding the shared
RL trunk (not an independent RL model): opponent-archetype identity has real ground
truth, unlike turn-type classification, and should bias two things — the state
encoder (soft belief signal) and MCTS's hidden-card determinization (better guesses
at what the opponent is holding). That doc flagged the data's location as unresolved;
it's now confirmed at repo root. This plan makes it concrete and buildable.

Two hard constraints from the existing codebase shape everything below:
- **No real opponent match logs exist** — only static decklist compositions. Training
  data must be synthetically generated from `staples.json`'s presence-percentage
  distributions.
- **Kaggle submission is numpy-only inference** (`pkm/rl/numpy_policy.py`, no torch at
  eval time). Since the classifier is wired into the live decision pipeline (encoder +
  MCTS), it must also ship a numpy-forward twin, exported the same way as the policy
  net (`pkm/rl/export.py` → `pkm/policy.npz`).

Part 1 (classifier) must be built and verified in complete isolation before Part 2
(RL integration) consumes its output — each Part 2 integration point is landed as a
separately-revertable, opt-in change so a regression in one doesn't force reverting
the other.

## Part 1 — Opponent-Archetype Classifier (Supervised)

### New files
- `pkm/data/card_aliases.py` — hand-maintained `ALIASES: dict[tuple[name,set,number], card_id]` override table for name-collision resolution.
- `pkm/data/archetypes.py` — loads `staples.json`, resolves staple names to internal `card_id`s, exposes `Archetype`/`StapleCard` dataclasses + `get_archetypes()` (cached, same pattern as `pkm/data/card_data.py:get_card_data()`).
- `pkm/rl/archetype_gen.py` — synthetic decklist + partial-reveal dataset generator (stdlib/numpy only).
- `pkm/rl/archetype_model.py` — torch `ArchetypeClassifier` (bag-of-cards embedding + pooling, mirrors `PolicyValueNet`'s style in `pkm/rl/model.py`, but **its own** small `card_emb` — do not share `PolicyValueNet.card_emb`, keeps the two training loops decoupled).
- `pkm/rl/numpy_archetype.py` — numpy-forward twin, mirrors `pkm/rl/numpy_policy.py`.
- `pkm/rl/archetype_train.py` / `pkm/rl/archetype_export.py` — training loop + `.npz` export, mirroring `pkm/rl/train.py` / `pkm/rl/export.py`.
- `pkm/cli/archetype.py` — `pkm archetype gen-data / train / export / eval`, registered like the existing `deck`/`cards` sub-apps.
- `tests/test_archetypes.py` — new test module (see below).

### Key mechanics
- **Name → card_id resolution** (`pkm/data/archetypes.py`): exact single-name match is the common case (zero-maintenance). Zero matches or multiple matches fall back to `card_aliases.py` keyed by `(name, set, number)` — the staple's set+number is the authoritative disambiguator even though the engine DB itself has no set/number field. `load_archetypes_with_report()` returns a resolution report (auto/alias/unresolved counts) used to iteratively hand-populate aliases until unresolved ≈ 0. This directly addresses the collision risk found during exploration: `pkm/data/card_data.py`'s `CardData` has no `set`/`number` field, only `name`.
- **Synthetic data** (`pkm/rl/archetype_gen.py`): (1) parse each staple's `tooltip` text into a per-copy-count probability table, sample actual copies per staple per synthetic decklist; (2) pad to 60 cards with archetype-appropriate basic energy; (3) simulate partial observability by sampling a random reveal-fraction (0-50%) subset of the 60-card list as the "revealed so far" multiset (order-invariant simplification — documented as the main external-validity assumption); (4) generate an explicit off-meta/"unknown" negative class by mixing staples across archetypes or sampling uniform-random legal decks. Dataset = `(X: bag-of-card-id-counts, y: archetype id or NUM_ARCHETYPES="unknown")`.
- **Classifier**: card-id embedding (own small table, e.g. dim 16) → count-weighted sum-pool (permutation-invariant, same idea as hand-pooling in `pkm/rl/model.py`) → small MLP → softmax over `NUM_ARCHETYPES + 1`. Small enough (~tens of K params) to be a non-issue against the 197.7 MiB Kaggle bundle cap.
- **Legally-visible input only**: revealed counts come from opponent `discard` + in-play `active`/`bench` (+ attached energy/tools) + revealed `prize`s per `pkm/types/obs.py:Player` — never `hand` (always `None` for the opponent in the observation contract, confirmed during exploration).

### Tests to add
Resolution-report completeness, tooltip-distribution parsing (against literal strings pulled from `staples.json`), sampled-decklist legality (length 60, max-4-copies, ≥1 Basic), dataset shape/class-balance, a training smoke test (loss decreases), and a torch/numpy parity test (`belief()` output matches within `1e-4`) — the last one is a hard gate before Part 2 ever touches the numpy classifier.

### Verification (must pass before Part 2 starts)
1. Held-out synthetic accuracy, broken out by reveal-fraction bucket — should rise from near-chance at 0% revealed toward high accuracy by ~25-50% revealed.
2. Off-meta calibration — held-out "unknown" examples should get diffuse/low-confidence predictions, not confident misclassification into a real archetype.
3. Alias-table completeness (unresolved staples ≈ 0).
4. Torch/numpy parity test passes.

## Part 2 — Feed Classifier Output into the RL System

Two additive, independently-toggleable integration points. **Before touching either**, read `pkm/rl/rollout.py` and `pkm/rl/play.py` in full to find every call site of `encode_state`/`encode_decision` (not yet inspected this session) — all must thread an optional `belief` parameter.

### 2a — Belief vector into the state encoder

**Status: encoder/`TorchPolicy` plumbing done and unit-tested since Parts
1-2 shipped (`a836ebd`); actually exercised during a real PPO training run
for the first time on 2026-07-19.** The gap: `TorchPolicy(model,
archetype_classifier=...)` and `compute_belief` were only ever driven
directly in `tests/test_archetype_integration.py` — `pkm/rl/train.py` never
constructed a classifier or passed one to any `TorchPolicy` it built, so
every training run to date (including the original `03_pult_munki` 1000-iter
run) saw an all-zero belief feature regardless of the dim-4→26 resize having
already happened. Closed by threading an optional `archetype_classifier`
through `play_one` (`pkm/rl/rollout.py`) → `_play_chunk`/`collect_parallel`
(`pkm/rl/parallel_rollout.py`) → `train()`, surfaced as `pkm train
--archetype-belief [--archetype-weights pkm/archetype.npz]` (opt-in, off by
default). Attached only to the trainee's `TorchPolicy`, never a frozen
opponent's (self-mirror or Part 3c pool bot) — verified by
`tests/test_rl.py::test_play_one_classifier_reaches_trainee_not_opponent`
(a monkeypatch spy on `TorchPolicy.__init__`). Smoke-tested standalone,
combined with `--archetype-pool`, and under `--workers 2` (confirms
`NumpyArchetypeClassifier` — plain numpy arrays, no torch/file-handle state —
pickles cleanly across worker processes).

- `pkm/rl/encoder.py`: add `NUM_ARCHETYPES`/`BELIEF_DIM` constants (from `pkm.data.archetypes`), extend `STATE_FEATS` by `BELIEF_DIM`, add an optional `belief: np.ndarray | None = None` param to `encode_state`/`encode_decision` (default → zero vector, so existing callers are unaffected unless they opt in). `pkm/rl/model.py` needs **no changes** — `STATE_IN` derives from `STATE_FEATS` automatically.
- New `pkm/rl/belief.py`: `compute_belief(obs, classifier) -> np.ndarray`, built on the same "what's visible" logic already implemented in `pkm/mcts/determinize.py` (`_visible_counter`) — reuse/import it rather than re-deriving visibility rules in two places.
- Test updates: `tests/test_rl.py::test_encoder_shapes` gets its expected `STATE_FEATS` constant bumped (mechanical, not weakened); add `test_encoder_belief_default_zero` and `test_encoder_belief_injection`.

### 2b — MCTS determinization biasing
- `pkm/mcts/determinize.py`: `infer_opponent_decklist` gains an optional `classifier` param. When provided, compute belief from currently-visible opponent cards, then weight the *composition* of the padded/estimated 60-card decklist toward archetypes' staple distributions (`copies * presence_pct`, weighted by `belief[a]`) instead of today's crude basics/energy-only padding. **No change needed to `sample_determinization` itself** — it already draws uniformly from whatever decklist it's given, so biasing composition upstream is sufficient and keeps `test_determinization_counts` (which only checks zone lengths, not composition) passing untouched.
- `pkm/mcts/agent.py`: `make_mcts_agent` gains an optional `archetype_weights_path` param; loads a `NumpyArchetypeClassifier` once at construction (mirrors `NumpyPolicy.load`), passes it through. Classifier load failure is non-fatal — falls back to today's crude behavior, consistent with the existing `except Exception: return policy.select(obs)` safety net in that file.
- Test updates: add `test_infer_opponent_decklist_with_classifier` (statistical property — archetype-biased decklists contain more of that archetype's staples on average across seeds, not exact equality).

### Verification for Part 2
1. All updated/new unit tests pass; pre-existing tests (`test_encoder_shapes`, `test_determinization_counts`, `test_infer_opponent_decklist`, `test_mcts_choose_legal`, `test_act_evaluate_consistency`) keep passing with no weakened assertions.
2. **Ablation win-rate comparison** using existing `pkm play --games N` / `pkm train`'s eval-vs-checkpoint-pool machinery: run (a) baseline/no-belief/uniform-determinization, (b) belief-in-encoder only, (c) determinization-biasing only, (d) both — against a fixed opponent pool, enough games (50-100) to see past self-play noise. A regression in any of (b)/(c)/(d) vs (a) is the signal to revert that specific opt-in flag.
3. MCTS-specific: compare value-head calibration (`abs(V(s) - actual_outcome)`) with/without archetype-biased determinization over a few `exit-train` iterations — better hidden-info guesses should reduce MCTS target noise.
4. End-to-end smoke test: `make_mcts_agent(..., archetype_weights_path=<real exported path>)` driving real decisions through the real engine, confirming the classifier path never silently falls into the exception-swallowing fallback.

## Part 3 — Real-Decklist Pool Bots + Cross-Archetype Opponent Sampling

> **2026-07-19 addendum.** Part 1+2 shipped in commit `a836ebd` with a known
> accepted breakage: the belief-feature resize invalidates existing policy
> checkpoints, left broken "pending a full retrain in a follow-up." Part 3 is
> that follow-up's prerequisite — it gives both the classifier and RL
> self-play *real* opponent decklists to train/retrain against, instead of
> only the synthetic staple-sampled decklists Part 1 generates. Confirmed via
> code inspection: `pkm/archetype/gen.py`'s dataset generator is currently
> 100% `staples.json`-derived and has no connection to real decklists;
> `pkm/rl/train.py` currently hardcodes one deck for both self-play sides
> (`train.py:97-102`, comment already flags "no multi-deck opponent pool yet
> -- AGENTS.md 'What's Next' #5"); `GameSpec` (`rollout.py:176-183`) has no
> deck field. Part 3 closes both gaps.

### New files (beyond Part 1/2)
- `pkm/archetype/build_pool_deck.py` — converts a sourced `entries.json`
  (`[{"name", "set", "number", "count"}, ...]`, scratch file, not committed)
  into a legal 60-card `deck/pool_<archetype_id>_<slug>.csv`, reusing
  `archetypes.py`'s name→card_id resolution machinery but with best-effort
  multi-match picking (unlike the classifier's strict resolution, which
  leaves ambiguous staples unresolved) and basic-energy padding/trimming to
  exactly 60. **Already built.**
- `agents/pool_<archetype_id>_<slug>/` — one per-archetype training profile
  directory (checkpoints/metrics), via the existing `AgentProfile` (Part 3b).
  **Correction (2026-07-19):** the name must be flat (`pool_284_dragapult_ex`,
  matching `deck/pool_284_dragapult_ex.csv`), not nested under `agents/pool/`
  — `AgentProfile.__init__` derives `base_dir=agents/<name>` and
  `deck_path=deck/<name>.csv` directly from the profile name with no
  subdirectory support, so nesting would require changing `pkm/agents/profile.py`
  for no benefit. The original wording above (`agents/pool/<id>_<slug>/`) was
  never actually implemented against and is superseded by this line.
- `pkm/rl/population_train.py` — new, additive orchestration layer for
  simultaneous multi-agent training (see "3b+3c" below). Reuses
  `rollout.py:play_game`/`ppo.py:ppo_update` unchanged; does not modify
  `train.py`'s existing single-deck path.

### 3a — Real decklists for 25 archetypes
Source a real, legal decklist per `staples.json` archetype (25 total), convert
via `python -m pkm.archetype.build_pool_deck <entries.json> <archetype_id>
<slug>`, and check the printed notes for unresolved cards / >4-copy warnings.

**Status: 25/25 done (2026-07-19).** The remaining 12 (Raging Bolt ex (280),
Ogerpon Box (339), Lillie's Clefairy ex (326), Alakazam Powerful Hand (350),
Metagross Metal Maker (361), Mega Lopunny ex (353), Marnie's Grimmsnarl ex
(329), Mega Starmie ex (362), Mega Greninja ex (370), Ogerpon Meganium (351),
Sylveon Safeguard (373), Archaludon ex (315)) were sourced from real
limitlesstcg tournament decklists (one URL per archetype, user-supplied) and
converted with a new scraper, `pkm/archetype/scrape_decklist.py`
(`python -m pkm.archetype.scrape_decklist <limitlesstcg_list_url>
<archetype_id> <slug>`), which parses `div.decklist-card[data-set][data-number]`
+ `.card-count`/`.card-name` into `entries.json` shape in-memory and reuses
`build_pool_deck.build_deck()`/new `write_pool_deck()` unchanged — no separate
resolution path. `requests`/`beautifulsoup4` were added as real `pyproject.toml`
dependencies for this (previously only transitive).

Worst-case unresolved-card count across all 12 was 3/60 (Archaludon ex, 5%) —
well under a 15-20% concern threshold discussed with the user; the recurring
gap is `Special Red Card` (CRI/82, a secret-rare trainer genuinely absent from
the engine's card pool), not the "some cards may not exist" risk flagged for
Metagross/Sylveon specifically, both of which resolved with ≤1 unresolved
card. All 25 pass `build_deck()`'s legality checks (60 cards, ≥1 Basic,
padded/trimmed as needed); several `multi-match` best-effort picks remain
(same pre-existing reprint-collision class documented in
`pkm/archetype/aliases.py`, not fixed here since `build_pool_deck.py`
deliberately best-effort-picks rather than blocking on them for pool bots).

### 3b — Train 25 simple pool bots

**Status: done (2026-07-19), via the fallback path (below), not simultaneous
population training.** All 25 `deck/pool_*.csv` trained solo, 100 iterations/
16 games each, `pkm/rl/train.py` **unmodified**, into
`agents/pool_<archetype_id>_<slug>/checkpoints/`. Eval-vs-random after 100
iters ranged 65-100% across all 25 (worst: `pool_361_metagross_metal_maker`
65%, `pool_362_mega_starmie_ex` 70%; most landed 85-100%) — every bot learned
something over random, none stuck near-chance. Spot-checked one
(`pool_361_metagross_metal_maker`) through export → `make_neural_agent` →
5 games vs random (3/5 win) to confirm the checkpoint→npz→agent pipeline
works, consistent with §Verification #2 below.

Confirmed concretely in the process: the "known breakage" flagged in the
2026-07-19 addendum above is real and specific — `03_pult_munki`'s committed
checkpoint is stamped `opponent_archetype_belief` dim=4 (the old dormant
3-class+Other trunk aux head), current registry expects dim=26 (25 real
archetypes + Other, from this plan's live Part 1 classifier) — so that
checkpoint cannot be exported or resumed from until the full retrain (3c
below, then Milestone 8) happens.

No lightweight-but-competent agent exists in this codebase today —
`random_agent` and an unweighted `neural_agent` are both effectively random
(`neural_agent.py` falls back to random legal moves when no weights file is
found). Decision: produce each pool bot via a **short PPO run** (~50-100
iterations, well under `03_pult_munki`'s 1000) using `pkm/rl/train.py`
**unmodified** — one run per `deck/pool_*.csv`, exported the standard way
(`pkm/rl/export.py`) into `agents/pool/<archetype_id>_<slug>/checkpoints/`.

### 3c — Cross-archetype opponent sampling

**Status: implementation done (2026-07-19); the actual `03_pult_munki`
retrain against the pool has not been run yet (Milestone 8).**

`GameSpec` (`pkm/rl/rollout.py`) gained an optional `opponent_deck` field
(default `None` = mirror `deck`, so every pre-3c `GameSpec` construction is
unaffected). `make_game_specs` gained `archetype_pool`/`archetype_pool_prob`
params: when given a list of `(deck, state_dict)` pairs, `archetype_pool_prob`
of games draw both an opponent deck and its matching policy instead of
self-mirroring — rolled *before* the existing `pool_prob` check, so
`pool_prob`'s meaning is unchanged when `archetype_pool_prob` is 0 (default).
`play_one` now sends each side its own deck instead of always `(deck, deck)`.
New `pkm/rl/opponent_pool.py:load_pool_bots()` scans `agents/pool_*/` for
trained checkpoints (skips untrained profiles rather than erroring; raises
`FeatureStampMismatch` via `check_stamp_sidecar` for a stale one, same
fail-loud convention as `AgentProfile.latest_checkpoint`). `pkm/rl/train.py`
wires it behind `--archetype-pool [--archetype-pool-prob 0.2]`, opt-in,
defaulting to today's single-deck behavior — consistent with how 2a/2b were
landed. This is the concrete, scoped-down version of the generalized
opponent pool sketched in `docs/ideas/general-agent-architecture.md`
(checkpoint pool + random + other profile policies) and closes
`docs/ideas/rl-improvements.md`'s "Multi-deck training" item.

Tests: `test_make_game_specs_no_archetype_pool_unchanged` (mechanical
backward-compat guard), `test_make_game_specs_cross_archetype_pool`,
`test_play_one_cross_archetype_deck` (real engine game confirming each side
gets its own deck), `test_load_pool_bots_skips_untrained_profiles` — all in
`tests/test_rl.py`. Smoke-tested end-to-end with a throwaway agent profile
against the real Part 3b pool-bot checkpoints (5 iterations, no crash,
completed games).

**Gotcha hit while wiring this up, worth flagging for any future CLI-flag
addition:** `pkm/cli/__init__.py`'s `train` command is a hand-duplicated
typer shim over `pkm/rl/train.py`'s own CLI (so `pkm train --help` shows
the top-level app, not `pkm.rl.train`'s) — it does not forward unknown
kwargs, so a new flag added only to `pkm/rl/train.py` silently doesn't
exist from the actual `pkm train` entry point until also added to the
`pkm/cli/__init__.py` shim. Cost about 10 minutes of "why doesn't my flag
show up in `--help`" before finding the duplicate signature.

### 3b+3c — Simultaneous population training (design decision, 2026-07-19)

> Supersedes the sequential reading of 3b→3c above ("train 25 bots solo,
> *then* sample them as a frozen pool"). Decision: train the main agent
> (`03_pult_munki`) and all 25 pool bots **together**, from shared games,
> each side updating its own live policy from that game's outcome — not a
> frozen-checkpoint opponent pool. 3a (real decklists) is unaffected and
> still a prerequisite; this only changes how 3b/3c turn those decklists
> into trained agents.

> **Status update (2026-07-19, later same day): this is now the plan for
> the next training run, not a conditional fallback.** Earlier the same day,
> the decision was to ship the cheaper frozen-pool version first (3b solo +
> 3c `--archetype-pool`, both done — see status notes above) and treat this
> section as a "fallback-of-the-fallback," used only if the frozen-pool
> retrain underperformed. That frozen-pool retrain is currently running
> (`03_pult_munki`, 2000 iterations, `--archetype-pool-prob 0.4
> --archetype-belief`). Independent of how that run turns out, the explicit
> ask now is to build this simultaneous population-training design and use
> it for the **next** training run after this one. `pkm/rl/population_train.py`
> is still unbuilt — the design below is the spec to implement, not yet code.
> Nothing about the currently-running retrain changes; this affects what
> comes after it.

**Why this is additive, not a rewrite of the existing trainer.** `play_game()`
(`pkm/rl/rollout.py:129`) already plays two independent `TorchPolicy` objects
against two independent decks and returns `GameResult.trajectories: tuple[list,
list]` — one player's experience never touches the other's. `ppo_update(model,
optimizer, decisions)` (`pkm/rl/ppo.py:82`) is already fully generic per model
— it has no notion of "the" training run. Today's `train.py` just never
exploits this: the opponent is always a frozen `state_dict` (no optimizer),
and both trajectories get merged into one `data` list for one model. Population
training only needs a new orchestration layer on top — `train.py`'s existing
single-deck path (used by `03_pult_munki` today) is untouched.

**Design:**
- `PopulationMember` dataclass: `name`, `deck` (card ids), `model`
  (`PolicyValueNet`), `optimizer`, `weights` (reward-shaping dict, from that
  member's own `AgentProfile.reward_weights_path`), `archetype_label` (from
  `archetype_index(deck_path)` — see note below), `profile` (`AgentProfile`,
  for checkpoint/metrics paths). Roster = the anchor (`03_pult_munki`) + all
  `agents/pool_*` members, loaded once at startup.
- Matchmaking: each iteration, the anchor plays `games_per_pairing` (new
  knob, default e.g. 2-4) games against **each** pool bot — guarantees every
  bot gets fresh data every iteration instead of a random subset. Total
  games/iteration = `games_per_pairing * len(pool)`. Bot-vs-bot games are out
  of scope for v1 (the ask was "Dragapult vs. every other deck," not a full
  round-robin); can be added later as another matchup-generation function
  without touching the per-member update logic.
- New `PopSpec` (not a reuse of `GameSpec` — that stays exactly as-is for
  `train.py`'s pool-of-past-checkpoints path): `(member_a_idx, member_b_idx,
  collect: (bool, bool))`.
- After rollout, trajectories are bucketed **per member name**, not into one
  shared list — `data: dict[str, list[EncodedDecision]]`. Each member's PPO
  update runs on only its own bucket.
- Update cadence: buffer a member's trajectories across iterations until
  either its bucket reaches a minimum sample count or `update_every`
  iterations pass, whichever first — bounds how stale ("off-policy") a
  member's own batch gets relative to the policy that generated it, same
  concern `pool_size` already manages for frozen opponents in `train.py`,
  just now applying to the *learner* too. Default a small bound (e.g. 1-3
  iterations) rather than leaving it unbounded.
- `dec.true_archetype`: set from each trajectory owner's *own*
  `archetype_index(deck_path)`, not one constant for the whole run. Low
  stakes either way — per `pkm/rl/features.py:233-238`, this feeds only the
  dormant 3-class trunk aux head (`00_basic`/`01_psychic`/`02_dragapult`),
  not the live belief feature (that's the standalone Part 1 classifier via
  `ctx.archetype_belief`). Every pool-bot deck resolves to the reserved
  "Other" class, which is fine — it was already dormant before this change.
- Parallel rollout: `parallel_rollout.py:_play_chunk` currently closes over
  one shared `deck`/`current_state`; population training needs each spec to
  carry both members' decks and state dicts, since no single shared deck
  exists anymore. New `_play_pop_chunk` alongside the existing one rather
  than modifying it in place.
- Checkpoints/metrics/eval-vs-random: reuse each member's own
  `AgentProfile` dirs unchanged — no new convention needed.

**Tradeoffs to watch (from the earlier discussion, now concrete):**
1. **Memory** — 26 live models × Adam optimizer state (2x params) instead of
   1 live + cheap frozen `state_dict`s. Model is small per AGENTS.md ("tens
   of K params" scale for the archetype classifier; `PolicyValueNet` is
   larger but still modest), so likely fine — worth a one-time memory check
   with the full 26-member roster before committing to it.
2. **Compute** — total games/iteration scales with `games_per_pairing * 25`,
   not `games_per_iter` as today; tune `games_per_pairing` down if wall-clock
   becomes the bottleneck rather than silently starving bots of data.
3. **On-policy drift** — bounded by `update_every`/buffer size above; if
   training destabilizes, shrinking that bound is the first lever, not
   architecture changes.

**Fallback:** if population training proves unstable early (e.g. very weak
freshly-initialized pool bots give the anchor a degenerate easy-win signal,
or vice versa), fall back to the frozen-pool design that's already built and
running (3b solo pool bots + 3c `--archetype-pool`/`--archetype-belief`, see
status notes above) — that path stays available precisely because
`pkm/rl/population_train.py` is required to be additive enough for both
approaches to coexist as separate entry points, not one replacing the other
outright.

### Tests to add
- `build_deck()` legality (exactly 60 cards, resolution notes) for each of
  the 12 new archetypes.
- `test_population_trajectory_routing` — a mixed anchor-vs-bot game's
  trajectories land in the correct member's bucket, never cross-contaminate.
- `test_population_matchmaking_coverage` — every roster member (besides the
  anchor) gets exactly `games_per_pairing` games per iteration.
- `test_population_train_noop_on_solo_path` — `train.py`'s existing
  single-deck/single-model flow (used by `03_pult_munki` today) is bit-for-bit
  unaffected by the new module existing (no shared mutable state, no import
  side effects).

### Verification for Part 3
1. ~~`ls deck/pool_*.csv | wc -l` == 25; no unresolved-card notes.~~ **Done
   (2026-07-19)** — 25/25, worst case 3 unresolved cards (Archaludon ex),
   none blocking legality.
2. Each pool bot's exported `policy.npz` loads via
   `make_neural_agent(deck, weights_path=...)` and completes a legal game
   against `random_agent`.
3. `pytest tests/` (full suite) passes with no weakened assertions —
   specifically confirms `train.py`'s existing solo path is unaffected by the
   new population-training module.
4. Population-specific: each pool bot's eval-vs-random win rate trends up
   over iterations (not just the anchor's) — confirms bots are actually
   learning from their thinner per-iteration batches, not stuck near-random.
5. Ablation win-rate comparison (same methodology as Part 2's Verification
   §2): anchor's win rate vs. a fixed held-out opponent set, population
   training vs. the original frozen-pool design (§3b/3c as first written),
   before flipping the default at Milestone 8.

---

## Milestones (sequenced)
0. Prep: inspect `pkm/rl/rollout.py`, `pkm/rl/play.py`, `pkm/rl/logging.py`; confirm `.gitignore` covers generated dataset/checkpoint paths.
1. Part 1 data plumbing: `card_aliases.py` + `archetypes.py`, iterate resolution report to near-zero unresolved.
2. Part 1 classifier: `archetype_gen.py` → `archetype_model.py`/`archetype_train.py` → `numpy_archetype.py`/`archetype_export.py`; run full Part 1 verification before proceeding.
3. ~~Part 2a encoder integration behind opt-in `belief` param~~ **Done** —
   encoder/`TorchPolicy` integration shipped with Parts 1-2 (`a836ebd`);
   train-time wiring (`pkm train --archetype-belief`) closed 2026-07-19 (see
   status note above — no training run had actually exercised this until
   then). Ablation (a) vs (b) still not run.
4. Part 2b MCTS determinization biasing behind opt-in `archetype_weights_path`; ablation (a) vs (c) vs (d), plus value-calibration check.
5. If ablations are neutral-or-positive with no regressions for Part 2: confirm final Kaggle bundle size with `pkm/archetype.npz` added, then proceed to Part 3 (below) before finalizing defaults/docs.
6. ~~Part 3a: source + build the remaining 12 pool decklists~~ **Done (2026-07-19)** — all 25 are legal 60-card decks with near-zero unresolved-card notes (worst case 3/60).
7. ~~Part 3b+3c (population training)~~ **Initially superseded (2026-07-19
   morning) — the frozen-pool fallback path was taken first** (decided up
   front, not after instability): Part 3b done as 25 solo PPO runs via
   unmodified `train.py` (65-100% eval-vs-random); Part 3c done as
   `GameSpec.opponent_deck` + `pkm/rl/opponent_pool.py` +
   `pkm train --archetype-pool`, opt-in. **Reopened the same day (afternoon):**
   the explicit ask is now to build `pkm/rl/population_train.py` and use it
   for the training run *after* Milestone 8 — see the §3b+3c status update
   above. Not superseded after all, just resequenced to come after the
   frozen-pool retrain rather than replacing it.
8. **In progress (2026-07-19).** Full retrain of `03_pult_munki`, fresh run
   (not resumed — its old checkpoint predates the belief-feature resize and
   can't load): `pkm train --agent 03_pult_munki --iterations 2000 --games 16
   --eval-every 10 --archetype-pool --archetype-pool-prob 0.4
   --archetype-belief` — the "full retrain in a follow-up" `a836ebd`'s commit
   message deferred, now meaningful because both real opponent diversity
   (Part 3c) and a live belief signal (Part 2a) exist to retrain with/against.
   `--archetype-pool-prob` bumped from the 0.2 default to 0.4 and iterations
   doubled from the original run's 1000 to 2000, both user calls, not
   ablation-derived. Old checkpoint preserved at
   `agents/03_pult_munki/checkpoints_pre_belief_resize/` (renamed, not
   deleted — gitignored so not recoverable via git). Determinization-biasing
   (2b) stays irrelevant here — it's MCTS-only and `03_pult_munki` hasn't run
   Phase 2 expert iteration yet.
9. **Next: build `pkm/rl/population_train.py`** per the §3b+3c design above
   (`PopulationMember`/`PopSpec`, per-member trajectory bucketing,
   `_play_pop_chunk`), smoke-test with a small roster (anchor + 2-3 bots)
   before scaling to all 25, then use it for the *next* `03_pult_munki`
   training run — independent of how Milestone 8's frozen-pool retrain turns
   out. Land the "Tests to add" list under §3b+3c first. If this later
   destabilizes, Milestone 8's frozen-pool design is the documented fallback
   (see §3b+3c "Fallback" above), not a redesign.
10. **Flip defaults + update docs**, once 8 and 9 have both been run and
    compared (ablation win-rate comparison, §3b+3c's own Verification #5):
    turn belief-in-encoder (2a) and `--archetype-pool`/population training on
    by default for new training runs. Update
    `docs/ideas/multi-phase-policy-and-opponent-modeling.md` to reflect what
    was actually built, `docs/ideas/rl-improvements.md` (mark "Multi-deck
    training" done), and `AGENTS.md`.

## Critical files
- `staples.json`, `pkm/data/card_data.py` — source data + card DB to match against
- `pkm/rl/encoder.py`, `pkm/rl/model.py` — encoder hook (Part 2a)
- `pkm/mcts/determinize.py`, `pkm/mcts/agent.py` — determinization hook (Part 2b)
- `pkm/rl/numpy_policy.py`, `pkm/rl/export.py` — pattern to mirror for numpy export
- `pkm/rl/rollout.py` (`play_game`, `GameResult`, `GameSpec`), `pkm/rl/ppo.py`
  (`ppo_update`), `pkm/rl/train.py`, `pkm/agents/profile.py` (`AgentProfile`)
  — existing primitives Part 3b+3c's population trainer builds on top of,
  unmodified
- `pkm/rl/parallel_rollout.py` — `_play_chunk` pattern to mirror for
  `_play_pop_chunk`
- `tests/test_rl.py`, `tests/test_mcts.py` — invariants to preserve
- `docs/ideas/multi-phase-policy-and-opponent-modeling.md` — doc to finalize at the end
