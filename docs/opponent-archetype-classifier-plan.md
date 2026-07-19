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

## Milestones (sequenced)
0. Prep: inspect `pkm/rl/rollout.py`, `pkm/rl/play.py`, `pkm/rl/logging.py`; confirm `.gitignore` covers generated dataset/checkpoint paths.
1. Part 1 data plumbing: `card_aliases.py` + `archetypes.py`, iterate resolution report to near-zero unresolved.
2. Part 1 classifier: `archetype_gen.py` → `archetype_model.py`/`archetype_train.py` → `numpy_archetype.py`/`archetype_export.py`; run full Part 1 verification before proceeding.
3. Part 2a encoder integration behind opt-in `belief` param; ablation (a) vs (b).
4. Part 2b MCTS determinization biasing behind opt-in `archetype_weights_path`; ablation (a) vs (c) vs (d), plus value-calibration check.
5. If ablations are neutral-or-positive with no regressions: flip defaults on for new training runs, update `docs/ideas/multi-phase-policy-and-opponent-modeling.md` to reflect what was actually built, confirm final Kaggle bundle size with `pkm/archetype.npz` added.

## Critical files
- `staples.json`, `pkm/data/card_data.py` — source data + card DB to match against
- `pkm/rl/encoder.py`, `pkm/rl/model.py` — encoder hook (Part 2a)
- `pkm/mcts/determinize.py`, `pkm/mcts/agent.py` — determinization hook (Part 2b)
- `pkm/rl/numpy_policy.py`, `pkm/rl/export.py` — pattern to mirror for numpy export
- `tests/test_rl.py`, `tests/test_mcts.py` — invariants to preserve
- `docs/ideas/multi-phase-policy-and-opponent-modeling.md` — doc to finalize at the end
