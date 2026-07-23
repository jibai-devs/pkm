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

### ① Expert-iteration training loop (MCTS-in-the-loop) — ✅ DONE (pre-existing)

**Status: already shipped in `trainers/exit.py` (`ExItTrainer`, `method="exit"`)
before this roadmap.** Matches the spec point-for-point: every move runs
`mcts.search_worlds(...)` → stores `(featurized obs, root-visit-policy π,
outcome z)` → fits `cross-entropy(logits, π) + value_coef·MSE(value, z)`;
registered in `TRAINERS["exit"]`; CLI `train --method exit`; tested
(`test_exit_trainer.py`). ③ (TD(λ)) and ④ (W-worlds) were layered on top
(2026-07-22).

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

### ② Combination-scoring / STOP-conditioned policy head (multi-select) — ✅ SHIPPED (2026-07-22)

**Status: done, opt-in.** Autoregressive STOP-token head landed as
`ModelConfig.policy_head` (`"marginal"` default = v1, `"autoreg"` = new). Enable
with `train --policy-head autoreg`. Implementation: `model.AutoregPolicyHead`
(conditions each pick on a pooled summary of already-picked options + a learned
STOP logit); sampler/logprob/entropy in `policy.py`
(`sample_action_autoreg` / `batched_action_logprob_autoreg` /
`batched_entropy_autoreg`); wired into PPO rollout+update (`trainers/ppo.py`) and
inference (`agent.py`). Key design: `policy_from_state`/`forward`/`evaluate` keep
returning step-0 per-option logits, so **MCTS, the ExIt trainer, and
inference-time MCTS were untouched**. Config-hashed + checkpoint-compatible (old
ckpts backfill to `"marginal"`); packed bundles rebuild the head automatically
(it's in the serialized `model_config`). Tests: `test_autoreg_head.py` (19,
incl. sample↔recompute logprob consistency, STOP legality, full inference game).
**Not yet done:** actually *train* an autoreg run and A/B it on the leaderboard
vs marginal.

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

### ③ TD(λ) value targets — ✅ SHIPPED (2026-07-22)

**Status: done, opt-in.** `TrainConfig.exit_value_target` (`"mc"` default = v1
raw outcome, `"tdlambda"` = the blend) + `exit_lambda` (0.9). Enable with
`train --method exit --exit-value-target tdlambda`. `mcts.search` now optionally
returns the MCTS-refined root value (`return_value=True`, `@overload`-typed;
default off so all existing callers are unchanged) as the bootstrap; the blend
runs backward per seat in `exit._assign_value_targets`. Note ① (the
expert-iteration loop itself) was **already implemented** in `trainers/exit.py`
before this — ③ adds the value-target scheme on top.

- **What:** blend terminal outcome with bootstrapped node values along the
  trajectory instead of the raw ±1 outcome (agent_001 uses λ=0.9).
- **Why:** lower-variance value target → more stable value head, faster
  convergence. Cheap, well-understood, drops straight into the ① loop.
- **From:** `agent_001/train.py:112-118`.
- **Land in:** the new expert-iter trainer.
- **Effort:** low. **Risk:** low.

### ④ Play the loop's search over W determinized worlds — ✅ SHIPPED (2026-07-22)

**Status: done, opt-in.** `TrainConfig.mcts_worlds` (default 1 = v1
single-world). Enable with `train --method exit --mcts-worlds 4`.
`exit._play_game` now calls `mcts.search_worlds(n_worlds=cfg.train.mcts_worlds)`
(which also averages the returned root value across worlds when TD(λ) is on).

- **What:** in expert-iter self-play, call
  `mcts.search_worlds(..., n_worlds=W)` instead of single-world `search()`.
- **Why:** IS-MCTS world-averaging is *already written* (`mcts.py:183-216`) but
  only reachable at inference. Single-world targets are biased toward one guessed
  layout; averaging de-biases the *training* policy target too. Free lever.
- **From:** nothing to port — it's our own `search_worlds`.
- **Land in:** the ① trainer.
- **Effort:** trivial. **Risk:** low (cost linear in W — trade against sim count).

### ⑤ Cross-attention policy head (successor to ②) — 📋 PROPOSED

- **What:** replace the autoregressive head's hard-wired single referenced
  entity + pooled-pick summary with a **learned cross-attention** layer: each
  option (the *query*) attends over the encoder's full per-entity embedding set
  (`ent [B,12,d_entity]`, the keys/values) and, autoregressively, over the
  already-picked options. The attended context vector feeds the scorer instead
  of the one `_gather_entity` slot.
- **Why:** this is the genuinely-more-powerful version of the grounding we
  approximate today. Right now ② (`model.AutoregPolicyHead`) grounds an option
  in exactly the *one* board slot it names (`_gather_entity`, a 1:1 gather) plus
  a mean-pool of already-picked option vectors (`g`). Cross-attention lets an
  option weigh **every** entity on the board — "should I attach this energy?"
  can look at my attacker, the opponent's active, and my bench together — which
  a single gathered slot and a mean-pool cannot express. It's the mechanism
  agent_001 uses (`../agent_001_transformer/net.py:88-94`, `DecoderLayer`
  cross-attends `encoder_out`), ported to *our* structured per-entity encoder.
- **Why it composes cheaply:** the head only has to keep the existing interface
  — `policy_from_state(...) -> [B,L]` step-0 logits and `policy_step(state, ent,
  b, picked_mask) -> (opt_logits[B,L], stop_logit[B])`. Every downstream
  consumer (the autoregressive sampler / PPO logprob / entropy in `policy.py`,
  MCTS priors, the ExIt target, inference) then works **unchanged**, exactly as
  they already do for ②. So this is a third `policy_head` value, not a
  cross-cutting change.
- **Sketch:**
  ```
  # per option: query = option vec; keys/values = board entities
  xattn = nn.MultiheadAttention(embed_dim=d_opt, num_heads=n_heads,
                                kdim=d_entity, vdim=d_entity, batch_first=True)
  context, _ = xattn(query=opt[B,L,d_opt], key=ent[B,12,d_entity],
                     value=ent[B,12,d_entity])           # [B,L,d_opt]
  opt_logit = opt_scorer([opt, state, ctx, context, g])  # g = pooled picked set
  stop_logit = stop_scorer([state, ctx, g])
  ```
  Attend over all 12 fixed slots (occupied + empty) with **no key mask** — the
  board tensor is always 12 slots so there is always ≥1 valid key (no
  fully-masked-row NaN, unlike a mask-empties approach); empty-slot embeddings
  are legitimate signal ("this bench slot is open").
- **From:** concept from `agent_001/net.py` decoder; grounding target already in
  `encoder.py` (`ent [B,12,d_entity]`). **Land in:** new `CrossAttnPolicyHead` in
  `model.py`, built when `ModelConfig.policy_head == "xattn"`; thread
  `n_heads` into `PolicyValueModel`; add `"xattn"` to the head validation +
  `--policy-head` help; extend `test_autoreg_head.py` to parametrize its
  behavioural tests over `("autoreg", "xattn")` (same guarantees must hold).
- **Effort:** medium (one new module + build wiring; the sampler/trainer/
  inference code is untouched). **Risk:** medium — new params → retrain,
  config-hash bump, checkpoint-incompatible; `d_opt` must be divisible by
  `n_heads` (holds for all presets: 64/4, 128/8, 192/8, 256/8).
- **Open choice:** pure cross-attention head vs *also* keeping the `_gather_entity`
  slot as an extra scorer input (belt-and-suspenders). Recommend starting pure
  and adding the gather back only if it helps.

---

## Consider, lower priority

### ⑥ Sparse `EmbeddingBag` offset-featurizer pattern — *learn from, don't adopt*

agent_001's `SparseVector` (`net.py:149-186`) makes adding a feature a one-liner
(allocate an offset range, deposit activations) with no `FEATURE_VERSION`
migration. **But** it throws away agent_000's real structural priors (grounded
pointers, hybrid open-vocab card encoder). Don't replace the encoder. Lesson
worth keeping: for fast prototyping of a new feature, a sparse side-channel is
cheaper than a typed dataclass field.

### ⑦ Combination odometer enumerator — **skip**

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
①  expert-iter trainer   (reuses mcts.search → π target)   ← ✅ DONE (pre-existing)
   └─ ③ TD(λ) targets      (inside ①)                       ← ✅ DONE (opt-in)
   └─ ④ search_worlds W>1   (inside ①)                       ← ✅ DONE (opt-in)
②  STOP-token multi-select head                            ← ✅ DONE (opt-in, pooled summary)
   └─ ⑤ cross-attention head (successor to ②)               ← 📋 PROPOSED (learned attention over entities)
```

①③④ and ② are shipped (all opt-in, config-hashed). Remaining: (a) *train* an
autoreg and/or TD(λ)+W-worlds run and A/B vs the 600 baseline; (b) implement ⑤
(cross-attention) if the pooled-summary grounding in ② proves too weak.

## Cross-references

- agent_000 network: `encoder.py:80-156`, `model.py:138-257`
- agent_000 existing MCTS (visit policy + IS-MCTS worlds): `mcts.py:101-216`
- agent_001 network (EmbeddingBag encoder + combination decoder): `../agent_001_transformer/net.py:97-143`
- agent_001 training loop (self-play + TD(λ) + Huber): `../agent_001_transformer/train.py:101-164`
- agent_001 architectural contrast: `../agent_001_transformer/__init__.py`
