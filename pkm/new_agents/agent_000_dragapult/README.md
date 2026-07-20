# agent_000_dragapult — design notes & TODO

A Dragapult ex **specialist** for the cabt competition. Everything here is bound
to one fixed decklist (`deck.py`): the card vocabulary, tensor dimensions, and
learned embeddings are meaningless away from this deck.

> **How to train / read metrics / stop & resume: [`TRAINING.md`](TRAINING.md).**
> Run it with `pkm new_agents 000_dragapult train …` (or the `justfile` here).

Related docs: **state representation coverage map (what's embedded, and where in
the code)** in [`state-representation.md`]; engine/data model in
[`../../../../docs/cabt-engine-reference.md`],
cross-cutting infra checklist in [`../../../../research/infra-todo.md`], overall
RL design in [`../../../../research/rl-agent-design-notes.md`].

> Status legend: **[DONE]** built & verified · **[NEXT]** planned · **[DECIDE]**
> open choice, do not build until discussed.

---

## 1. What this agent is
- One deck, one specialist policy. We only ever pilot our own deck, so we do
  **not** need cross-deck generalization on our side — build a specialist.
- Inference contract (what we ultimately submit): `agent(obs_dict) -> list[int]`
  returning chosen option **indices** into `obs["select"]["option"]`.
- Training needs more than the inference agent (optimizers, buffers, self-play,
  possibly extra networks) — those live here too but are training-only.

## 2. State representation — the idea

Chosen direction (our lean; the encoder details below are still **[DECIDE]**):
an **entity/set** representation, not a flat hand-engineered vector.

- The board *is* a set of typed entities (≤12 Pokémon = 2 × (1 active + 5 bench)),
  each with attributes (HP, energy, tools, evolution). Encode each entity to a
  vector; process the set order-invariantly (attention / pooling); emit a board
  summary **plus one embedding per entity**.
- Why entity/set over flat: it mirrors the real structure, is order-invariant,
  is sample-efficient, and — crucially — the **action head points at the
  per-entity embeddings** (an engine `CARD` option references a board/hand slot;
  an `ATTACK` option references the active Pokémon), so state and action rep are
  coupled. A flat vector throws those embeddings away.
- Memory: the obs is fairly Markov (the "this-turn" flags live in `State`), and
  `logs` give deltas since our last decision. Start **single-frame + light belief
  features**; reach for recurrence only if needed. **[DECIDE]**

### Card encoding (hybrid) — highest-leverage piece
Two regimes because of the specialist asymmetry:
- **Our cards** (closed vocab, 26 IDs): a learned `nn.Embedding` row per ID.
- **Opponent / unseen cards** (open vocab, ~1200+ IDs): no dedicated row — they
  map to a single **UNK** row for the *learned* table, but are still
  distinguished by an **attribute encoder** built from `all_card_data()`
  (HP / type / cardType / retreat / attack damage+energy).
- This is why the `AllCard` binding mattered: it lets *any* card, even one never
  seen in training, get a meaningful vector.

### How the embeddings are learnt
No separate step — the embedding table is just weights, trained **end-to-end**
with the RL loss: init random → card ID looked up in forward pass → loss →
autograd gives a gradient only for rows whose IDs appeared in the batch →
optimizer updates them. Frequently-played cards (Dreepy, Ultra Ball) get lots of
signal; rare ones (Budew ×1) get little — which is fine, and another reason the
attribute channel helps.

### Is the model bounded to the deck?
Yes, with a nuance: the **learned per-ID table is fixed** (27 rows = 26 own +
UNK); changing the deck changes the vocab → re-fit that table (a retrain). But
the model is **not blind** to opponent cards — the attribute channel covers the
open vocab. Global normalizers are computed over *all* cards so opponent cards
scale correctly.

## 3. Action representation — the idea
- The engine presents only **legal** options each decision (`select.option`,
  `minCount..maxCount`), so the legal mask is free.
- Lean: a **pointer/scoring head over the presented options**, encoding each
  option by its type + the entity it references, scored against the select
  context. Multi-select (`maxCount > 1`) and `COUNT` handling: **[DECIDE]**.
- Alternative not chosen yet: fixed templated action head. **[DECIDE]**

## 4. Agent (inference) vs Training (learner)
- **Agent** holds only what inference needs: the network(s), optional
  inference-time search, belief state. Frozen weights, ≤2 s/turn, hidden-info
  only.
- **Learner** holds training-only machinery: optimizer(s), rollout/replay buffer,
  loss/update, self-play/league, target/teacher nets. Keep multiple-network
  support (not one monolith) — model is a container of named submodules.
  Method choice (model-free vs search-centric) is **[DECIDE]**.

---

## 5. What's built so far

| File | Status | What |
|------|--------|------|
| `deck.py` | **[DONE]** | Hard-coded 60-card deck; `DECK_60`, `DISTINCT_IDS` (26), `ID_TO_ROW`, `UNK_ROW=26`, `VOCAB_SIZE=27`, `row_of()`. Single source of truth for the deck. |
| `build_spec.py` | **[DONE]** | Deterministic dimensioning script over the static card/attack tables + deck. Re-run: `uv run python -m pkm.agents.agent_000_dragapult.build_spec`. |
| `spec.json` | **[DONE]** | Generated output (below). |
| `features.py` | **[DONE, v1 provisional]** | `Observation` → numpy arrays. Fixed-12 board slots + mask, per-entity F=26, globals G=16, own hand/discard as 27-vocab histograms, options as type + O=12 raw fields. Verified finite & fixed-shape over 6528 live obs. `FEATURE_VERSION="v1"`. |
| `cards.py` | **[DONE]** | Deterministic `[max_id+1, A=54]` card attribute table from `all_card_data`/`all_attack` (hp/retreat/cardType/energy/weak/resist/stage flags/attack dmg). The attribute channel of the hybrid card encoder; normalizers from `spec.json`. |
| `encoder.py` | **[DONE, v1 provisional]** | torch `StateEncoder`: hybrid `CardEncoder` (learned 27-vocab `nn.Embedding` + attribute MLP) → per-entity vectors → CLS-token attention over the 12-slot set → `state` `[B,128]` + per-entity `[B,12,64]`. `collate_states` batches Features. Forward+backward verified finite (all grads) over 1361 obs incl. empty-board setup states. |
| `model.py` | **[DONE, v1 provisional]** | torch `PolicyValueModel`: `OptionEncoder` (option type emb + field proj) scored against `state` + decision-context (`SelectType`/`SelectContext`) emb → per-option **policy logits** (padding masked to -inf); `value_head` → scalar. `collate` adds padded options + mask. Trunk/heads split (`encode`/`policy_from_state`/`value`/`evaluate`), injectable encoder. Verified: logits/value finite, 0 prob-mass on padding, all grads finite (~79k params). |
| `agent.py` | **[DONE, v1]** | `DragapultAgent` + module-level `agent(obs)->list[int]`. Handles deck-selection phase (returns `DECK_60`) and `minCount..maxCount` count (top-k greedy / sample-without-replacement). `from_checkpoint` loader. Verified: 20 full self-play games, 242 decisions all valid (in-range, distinct, count in `[min,max]`). ~8.6 games/s (net inference now the bottleneck vs 141/s random). |
| `config.py` | **[DONE; TrainConfig provisional]** | Frozen dataclasses `ModelConfig`/`TrainConfig`/`RunConfig`/`Config` with dict (JSON) serialization + stable `hash()` for checkpoints. `build_model(cfg)` threads model dims into the network (single wiring point). Verified round-trip + config-driven sizing. TrainConfig fields are placeholder PPO-style (algorithm not yet chosen). |
| `policy.py` | **[DONE, v1]** | Action distribution over presented options: sequential without-replacement (Plackett–Luce) `sample_action` + `batched_action_logprob` + `batched_entropy`. k=1 reduces to a categorical; count fixed to `maxCount` (v1, ~4% multi-select). |
| `train.py` | **[DONE, v1 baseline]** | PPO + self-play: `play_game`/`collect_rollout` (one model both seats) → per-seat GAE (terminal ±1) → clipped `ppo_update`; `TrainState` checkpoint/resume (weights, optimizer, RNG, update idx, config+hash). Smoke-verified: finite losses over updates + resume round-trip. Uses `ParallelRollout` when `num_workers > 1`, else the single-process loop. |
| `parallel.py` | **[WRITTEN, UNVERIFIED]** | `ParallelRollout` — synchronous multiprocess self-play (spawn workers, one engine each; broadcast weights → gather `Step`s). Not yet run. Uses `torch.multiprocessing`, so real runs spawn `torch_shm_manager`; a stdlib-`multiprocessing`+numpy-weights variant (no torch IPC binary) is the noted alternative. |
| `eval.py` | **[DONE, v1]** | Win-rate harness (the learn-check): agent (greedy) vs a fixed opponent, **alternating seats** to cancel first-player bias, from the agent's perspective. `RandomAgent` baseline; `winrate_vs_random`; wired into `train(eval_every=…)`. Verified: untrained=50%, random-v-random≈50%. **Finding: fixed seat-0 wins ~56% → real first-player advantage; always alternate/average over seats.** |

### Locked dimensioning numbers (`spec.json`)
- `vocab_size = 27` (26 own + UNK), `board_slots = 12`, `bench_max = 5`
- enum sizes: `n_energy_types = 12`, `n_card_types = 7`, `n_special_conditions = 5`
- global normalizers (all 1267 cards): `max_hp = 380`, `max_retreat = 4`,
  `max_damage = 350`, `max_energies_per_attack = 5`, `max_attacks_per_card = 2`
- our deck maxima: `max_hp = 320`, `max_damage = 200`, 13 distinct attacks

---

## 6. TODO

### Immediate (mechanical, low-decision)
- [ ] `features.py` — pure `Observation` → tensors (entity matrix + mask +
  globals + option encodings). No learning; verify shapes against a live game.
- [ ] Decide featurizer output contract + **version it** (checkpoints record it).

### Needs a decision before building
- [ ] **[DECIDE]** entity slots: fixed-12 vs padded variable-length set.
- [ ] **[DECIDE]** set encoder: attention vs pooling (and dims).
- [ ] **[DECIDE]** card embedding dim + how the learned + attribute channels combine.
- [ ] **[DECIDE]** hand/discard encoding: per-card set vs count-histogram.
- [ ] **[DECIDE]** belief/opponent features from `logs` (any? which?).
- [ ] **[DECIDE]** action head: pointer-over-options vs templated; multi-select handling.
- [ ] **[DECIDE]** single-frame vs recurrence.
- [ ] **[DECIDE]** learning method: model-free (PPO) vs search-centric vs hybrid.
- [ ] **[DECIDE]** reward: terminal ±1 vs weak potential shaping; γ, GAE.

### Build after decisions
- [ ] `encoder.py` — entity/set encoder + hybrid card encoder.
- [ ] `model.py` — network container (multiple submodules), policy + value heads.
- [x] `config.py` — hyperparams (batch size, lr, γ, λ, clip, entropy, workers, seeds, feature-spec version) + `build_model`. *(done; TrainConfig provisional until algorithm chosen; pluggable-head registry still deferred here)*
- [x] `train.py` — PPO + self-play learner loop + **checkpoint/resume** (weights, optimizer, RNG, update idx, config hash). *(done; league state + centralized critic are future work)*
- [x] `agent.py` — inference `agent(obs)->indices`, loads a frozen checkpoint. *(done; inference-time search still optional/later)*

### Infra hooks (shared, tracked in `research/infra-todo.md`)
- [ ] `CabtEnv` wrapper (one engine per process) + multiprocessing vectorization.
- [ ] Opponent league / eval panel.
- [ ] Save/restore round-trip test.

---

## 7. Decisions log (alternatives considered)

Record of choices made + what we deliberately did *not* do, so we can revisit.

### D1 — Categorical encoding: `nn.Embedding` (chosen)
- **Chosen:** learned `nn.Embedding` for card IDs, attack IDs, and the small
  categorical enums (`AreaType`, `OptionType`, `SelectType`/`SelectContext`,
  `CardType`, etc.).
- **Alternative considered — one-hot encoding:** rejected as the primary
  mechanism, but worth remembering.
  - An embedding is mathematically *the same thing* as one-hot × a learned
    matrix, so no expressiveness is lost by choosing embeddings.
  - One-hot's only edges are simplicity and zero params at tiny cardinality;
    embedding wins on speed (gather vs wide matmul), compactness, and being
    **attribute-seedable** (needed for the open-vocab opponent cards).
  - **When we might still reach for one-hot / revisit:** 2–3-value flags where a
    module isn't worth it; a quick baseline; or if an embedding for some tiny
    enum shows no benefit over a fixed one-hot in ablation.
- **Note:** energy is already effectively one-hot (the 12-dim per-Pokémon energy
  histogram) and statuses are booleans — those stay as-is; the embedding decision
  is about the *identity/enum* fields.

### D2 — `AreaType` completed from C++ source (chosen)
- Trust the api-doc enum values everywhere **except `AreaType`**, which we
  completed to `0..24` from `core/CardTypes.h:18`. Alternative (tolerant
  `_WireEnum` fallback only) kept as a safety net, not the fix. See
  `../../../../docs/engine-enum-sources.md`.

### D3 — Set encoder: CLS-token attention (chosen), provisional dims
- **Chosen:** prepend a learnable CLS token to the 12 entity slots; attention
  pools the board (CLS output = board summary) and emits per-entity embeddings.
- **Why not masked mean-pool:** a fully-empty board (setup) makes attention
  softmax over an all-masked row → NaN in the forward AND (crucially) the
  backward pass; `nan_to_num` on the output fixes forward only. The always-valid
  CLS token guarantees ≥1 key per row, killing the NaN at the source.
- **Provisional dims:** `D_CARD=32, D_ENTITY=64, D_GLOBAL=64, D_STATE=128,
  N_HEADS=4` — first-pass, to tune.
- **Known v1 limitation:** only Pokémon appear as board entities, so the learned
  `own_emb` is exercised only for on-board Pokémon; trainer/energy cards flow
  through the hand/discard count histograms (no learned per-ID vector yet).
  Revisit if trainer/energy identity needs a learned embedding. `[DECIDE]`

### D4 — Action head: pointer/scorer over presented options (chosen, provisional)
- **Chosen:** score each engine-presented option (`select.option`) with a shared
  scorer over `[option-encoding, state, decision-context]` → per-option logits;
  padding (batching only) masked to `-inf`. Legality is intrinsic (engine offers
  only legal options).
- **Deferred / provisional:**
  - **Entity gather** — the scorer does *not* yet pull the board entity an option
    references (the encoder's per-entity embeddings). Planned upgrade. `[DECIDE]`
  - **Multi-select** (`maxCount > 1`) — model emits per-option logits only;
    choosing *k* is left to the sampling layer (the agent). `[DECIDE]`

### D5 — Modularity: trunk + method-exposed heads (chosen); registry deferred
- **Done:** `PolicyValueModel` exposes `encode` / `policy_from_state` /
  `value` / `evaluate` so a training loss, an **MCTS driver** (`evaluate` →
  priors+value, no-grad), or a swapped head can call each piece without reaching
  into internals. The trunk runs once; heads operate on the cached `state`.
  Complexity is tunable by **injecting** a custom `StateEncoder`
  (`PolicyValueModel(encoder=StateEncoder(d_state=256, n_heads=8, ...))`).
- **Deferred (→ `config.py`):** a *formal* pluggable-heads container + model
  registry (method declares which heads it uses) and threading all dims through
  config instead of module constants. Current setup is "modular enough to
  experiment," not yet "swap-a-head-by-config". `[DECIDE]`
- **MCTS note:** the net already emits `(prior, value)` (AlphaZero shape); the
  real search work is the driver over the engine's `Search*` API with
  determinization for the imperfect-info / random-draw chance nodes — the value
  is an **expectation** over hidden state + draws, learned from outcomes
  (model-free) or sampled via determinization (search).

### D6 — Option masking: finite sentinel, not -inf (chosen)
- Policy masks padding options with `MASK_FILL = -1e9`, not `float("-inf")`.
  `exp(-1e9)` underflows to 0 (real rows still put ~0 mass on padding), but a
  fully-masked row — a hypothetical 0-option decision collated at train time —
  yields a finite uniform distribution instead of NaN-poisoned gradients. Found
  by code review as a latent trap for the not-yet-written `train.py` (inference
  is unaffected: the agent slices to real options). Verified.
- **Deferred nit:** `spec.json` is read independently in `features.py` and
  `cards.py` — a shared loader would remove drift risk. Low value. `[DECIDE]`

### D7 — Learning algorithm: PPO + self-play first (chosen)
- **Chosen:** model-free **PPO with action masking + self-play** as the first
  learner, **terminal ±1 reward** (win/loss), no shaping to start. Fast to a
  working baseline, fast inference (well under 2 s), imperfect-info handled
  implicitly (value learns `E[return | obs]`).
- The value head is the through-line reused by every later method.

## 9. Future improvements (roadmap — logged, not yet built)

Ordered; each reuses the parts before it (see D5/D7). The network already emits
`(priors, value)` via `model.evaluate()`, so the whole ladder is unlocked by the
same trunk+heads.

1. **Hybrid: inference-time MCTS** — ✅ **SHIPPED (2026-07-20).** The *same PPO
   net* is wrapped in MCTS at decision time (policy=priors, value=leaf eval) via
   the engine's `Search*` API + determinization for hidden cards. No retrain.
   Opt-in via an `InferenceConfig` baked into the packed bundle: pack with
   `--inference mcts -K <sims>` (K=0 or `--inference policy` → plain policy);
   measure locally with `eval --inference mcts -K <sims>`; pack both variants at
   once with `scripts/pack_variants.sh`. Search rides Kaggle's own `libcg.so`
   search symbols, so no vendored engine is shipped. **Watch the per-turn +
   cumulative 600 s time budget** — tune K to fit. `agent._mcts_pick`, `mcts.py`.
2. **Expert iteration / AlphaZero-style** — move search *into* training: MCTS
   produces improved policy targets, net imitates them (+ outcome value). New
   training loop/loss, but reuses the net + search driver. Needs **IS-MCTS /
   determinization** for imperfect info (the hard, expensive part) → do last.
3. **Auxiliary losses on the shared trunk** — predict opponent's next card /
   game outcome / legal options. Cheap (tiny heads, privileged self-play labels,
   dropped at inference); each needs a loss weight + ablation. Add any time.
4. **Reward shaping** `[DONE, pluggable]` — `shaping.py` splits reward
   (`SHAPERS`) from advantage estimation (`ESTIMATORS`), both chosen by
   `TrainConfig.shaping`/`advantage` (serialized in the checkpoint + config
   hash) and exposed as `--shaping`/`--shaping-coef`. Shipped: `terminal`
   (sparse ±1) and `prize_potential` (policy-invariant `coef·(γ·Φ(s')−Φ(s))`
   on the prize differential, Ng 1999). **Default is now `prize_potential`.**
   *Other shaping ideas to consider* (each = one new `SHAPERS` entry, ablate
   with `--shaping-coef`): board-HP differential, energy-in-play advantage,
   bench-development / active-Pokémon survival, tempo (KOs per turn),
   hand-size/card-advantage potential. All should stay **potential-based**
   (telescoping) to preserve policy invariance. New estimators (TD(λ),
   V-trace, GAE-with-truncation-bootstrap) drop into `ESTIMATORS` the same way.
5. **Representation upgrades** (see `state-representation.md` gaps) — effects via
   attack/ability embeddings; opponent discard; hand/discard as count-weighted
   embeddings; attachment identity.
6. **Opponent league** (see `../../../../research/infra-todo.md`) — scripted
   archetype bots → frozen checkpoints → PSRO/PBT; the diversity that buys
   robustness to an unknown field.
7. **Centralized critic** — give the value head privileged full-state info at
   train time only (we control both seats in self-play). Cheap, usually helps.
8. **Throughput** — vectorized/multiprocess self-play (one engine per process)
   with batched net inference (net is the bottleneck: ~8.6 vs 141 games/s).
9. **Deck hedge** — if unsure of the decklist, train several separate
   specialists and pick the strongest (not a generalist in one net).

## 10. Open questions
- Fixed-12 slots vs padded set as the first featurizer? (affects everything downstream)
- Commit to pointer action head now, or prototype both?
- One shared trunk vs several independent networks — how many, and which?
- Persist the replay buffer for exact resume, or accept approximate resume?
- Which learning method do we prototype first on a single mirror match?
