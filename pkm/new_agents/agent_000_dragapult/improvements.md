# agent_000 improvements — what to bring over from agent_001_transformer

Concrete, prioritized list of ideas to port from **agent_001_transformer** (the
near-verbatim AlphaZero notebook port) into **agent_000_dragapult** (our
engineered PPO agent). Ranked by *(value ÷ effort)*.

Context: the two agents share the same engine seam
(`agent_000_dragapult.cabt`) and the same determinization idea. They diverge in
**featurization**, **the policy head**, and **how training targets are made**.
agent_000 is stuck at Kaggle **600** — an *eval ceiling*, not capacity: training
was mirror self-play but the metric was vs random. agent_001's structural answer
is **MCTS-in-the-training-loop (expert iteration)**.

Key enabling fact: agent_000's `mcts.search()` **already returns a root
visit-count policy** (`mcts.py:170-180`) — the exact target AlphaZero training
consumes. `determinize.py` already handles hidden info, and `featurize`/`collate`
already feed the net. The expensive part is built.

---

## Bring these over — ranked

### ① Expert-iteration training loop (MCTS-in-the-loop) — **do first**

- **What:** a trainer that plays self-play where *every move is an
  `mcts.search()` call*, stores `(featurized obs, root-visit-policy π, game
  outcome z)`, and fits the net to π (cross-entropy) + z (value regression).
- **Why:** direct fix for the Kaggle-600 eval ceiling — search becomes the
  teacher on every step instead of policy-gradient vs a weak self.
- **Why it's #1:** the hard 80% already exists. `mcts.search()` returns the
  visit policy π you need (`mcts.py:170-180`); `determinize.py` handles hidden
  info; `featurize`/`collate` already feed the net. This is orchestration of
  existing parts, not new ML.
- **From:** `agent_001/train.py:101-164` (`collect_selfplay` + `train_epoch`).
- **Land in:** `agent_000/trainers/` (new trainer beside the PPO one — it's
  already a registry; see `test_trainers_registry.py`).
- **Effort:** medium. **Risk:** low — additive, gated behind a trainer choice;
  PPO stays.

### ② Combination-scoring / STOP-conditioned policy head (multi-select)

- **What:** replace (or add a mode to) the marginal pointer so multi-select
  decisions are scored as a *set*, not N independent options.
- **Why:** the one place agent_001 is genuinely *more correct*. PTCG constantly
  asks "pick N of these," and agent_000's `model.py:14-21` explicitly flags this
  as its top unfinished TODO ("STOP token + running pick summary").
- **Recommendation:** don't copy agent_001's "enumerate up to 64 combinations"
  verbatim (`net.py:303-355`, `426-438`) — that's exponential and caps at 64.
  Take the *idea* (score-the-slate) but implement the STOP-token autoregressive
  version already designed in the docstring. Composes cleanly with ①.
- **From:** concept from `agent_001/net.py` decoder; design already spec'd in
  `agent_000/model.py:14-21`.
- **Land in:** `agent_000/model.py` + `features.py` (option masking).
- **Effort:** high. **Risk:** medium — model change → retrain, config-hash bump,
  checkpoint incompatibility.

### ③ TD(λ) value targets

- **What:** blend terminal outcome with bootstrapped node values along the
  trajectory instead of the raw ±1 outcome (agent_001 uses λ=0.9).
- **Why:** lower-variance value target → more stable value head, faster
  convergence. Cheap, well-understood, drops straight into the ① loop.
- **From:** `agent_001/train.py:112-118`.
- **Land in:** the new expert-iter trainer.
- **Effort:** low. **Risk:** low.

### ④ Play the loop's search over W determinized worlds

- **What:** in expert-iter self-play, call
  `mcts.search_worlds(..., n_worlds=W)` instead of single-world `search()`.
- **Why:** IS-MCTS world-averaging is *already written* (`mcts.py:183-216`) but
  only reachable at inference. Single-world targets are biased toward one guessed
  layout; averaging de-biases the *training* policy target too. Free lever.
- **From:** nothing to port — it's our own `search_worlds`.
- **Land in:** the ① trainer.
- **Effort:** trivial. **Risk:** low (cost linear in W — trade against sim count).

---

## Consider, lower priority

### ⑤ Sparse `EmbeddingBag` offset-featurizer pattern — *learn from, don't adopt*

agent_001's `SparseVector` (`net.py:149-186`) makes adding a feature a one-liner
(allocate an offset range, deposit activations) with no `FEATURE_VERSION`
migration. **But** it throws away agent_000's real structural priors (grounded
pointers, hybrid open-vocab card encoder). Don't replace the encoder. Lesson
worth keeping: for fast prototyping of a new feature, a sparse side-channel is
cheaper than a typed dataclass field.

### ⑥ Combination odometer enumerator — **skip**

`net.py:426-438` is a clean legal-combination enumerator, but it's the
exponential-with-cap-64 approach ② is meant to replace. Only a stopgap if ②'s
autoregressive head proves too slow.

---

## Do NOT bring over

- **AlphaZero's Huber-on-everything loss / dropping PPO** — we have a working PPO
  stack (GAE, entropy, reward shaping, aux losses). Add expert iteration
  *alongside* it via the registry; let them compete on the leaderboard.
- **The hard-coded deck registry / smaller harness** — agent_000's config-hash +
  multi-deck superset vocab + test suite is strictly more capable.
- **`EmbeddingBag(22000, d)` encoder** — mostly one-hot card IDs; far weaker
  prior than the set-attention encoder.

---

## Suggested sequence

```
①  expert-iter trainer   (reuses mcts.search → π target)   ← biggest ceiling lever, least new code
   └─ ③ TD(λ) targets      (inside ①)                       ← cheap accuracy add
   └─ ④ search_worlds W>1   (inside ①)                       ← free, our own code
②  STOP-token multi-select head                            ← real modeling gap, but retrain + config bump
```

Do ①③④ as one focused change (same trainer file, reuses existing code), measure
vs the 600 baseline, *then* decide whether ② is worth the retrain.

## Cross-references

- agent_000 network: `encoder.py:80-156`, `model.py:138-257`
- agent_000 existing MCTS (visit policy + IS-MCTS worlds): `mcts.py:101-216`
- agent_001 network (EmbeddingBag encoder + combination decoder): `../agent_001_transformer/net.py:97-143`
- agent_001 training loop (self-play + TD(λ) + Huber): `../agent_001_transformer/train.py:101-164`
- agent_001 architectural contrast: `../agent_001_transformer/__init__.py`
