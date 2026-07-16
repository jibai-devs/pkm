# Reinforcement Self-Play Plan (CABT / Pokémon TCG AI Battle Challenge)

Two phases, built in order. Phase 1 gets a working model-free self-play pipeline;
Phase 2 layers search on top for stronger training targets (expert iteration).

## The core difficulty: the action space

Each decision, the engine hands the agent `select.option` — a *variable-length*
list of `Option` objects — and expects a list of indices back. A fixed softmax
head can't represent this. We use a **pointer/scoring architecture**:

- Encode the game state into a vector `h` (card-ID embeddings for
  active/bench/hand plus scalar features).
- Encode each option `o_i` (embed its `type`, `cardId`, `attackId`, plus scalars).
- Score each with `f(h, o_i)`, softmax over the legal options only, sample.

Multi-select steps (`minCount..maxCount`) are decomposed into **sequential
picks**: sample an option, mask it, re-score, repeat. A learned virtual STOP
option becomes available once `minCount` picks are made; sampling stops at
`maxCount`. The joint log-prob is the sum of the sequential softmax log-probs
(Plackett–Luce style), which PPO can optimize directly.

## Phase 1 — PPO self-play (no search API)

Pipeline (`pkm/rl/`):

1. **Rollouts**: drive games directly via `battle_start` / `battle_select`
   (faster than `kaggle_environments.make`, and gives us both players'
   observations). Both players use the current policy; opponents are also
   sampled from a **checkpoint pool** to prevent self-play cycling.
2. **Transitions**: per decision store encoded state, encoded options, chosen
   index sequence, log-prob, value estimate — *for both players* (each game
   yields a winning and a losing trajectory; halves variance for free).
3. **Reward**: terminal +1 / −1 / 0. Plus optional *potential-based* shaping
   `φ(s) = 0.2 · (opp_prizes_remaining − my_prizes_remaining_deficit)` — i.e.
   prize differential — added as `γφ(s') − φ(s)` so the optimal policy is
   unchanged but credit assignment over long games improves.
4. **Update**: GAE(λ) advantages + PPO clip. Entropy bonus for exploration.
5. **Eval**: every N iterations play vs. the random agent and vs. the oldest
   pool checkpoint; log win rates; save checkpoints.

Files:

| file | contents |
|---|---|
| `pkm/rl/encoder.py` | obs dict → integer id arrays + float features (numpy) |
| `pkm/rl/model.py` | `PolicyValueNet` (torch): embeddings, state MLP, option scorer, value head; sequential multi-select sampling with STOP |
| `pkm/rl/rollout.py` | self-play game driver, transition collection |
| `pkm/rl/ppo.py` | GAE + PPO update |
| `pkm/rl/train.py` | training loop, opponent pool, eval, checkpoints |
| `pkm/rl/export.py` | export weights to `.npz` for numpy-only inference |
| `pkm/rl/numpy_policy.py` | torch-free forward pass (submission has a 197.7 MiB size cap; don't bundle torch) |
| `pkm/agents/neural_agent.py` | kaggle-env-compatible agent from a checkpoint |

Encoding notes:

- Card vocabulary: ids 1..1267 (embedding table 1268×32, id 0 = pad/unknown).
- Attack vocabulary: ids 1..1556.
- State features: my active (embed + hp frac + energy count + statuses), 5 bench
  slots (embed + hp frac + energy count), hand (bag of embeddings), counts
  (hand/deck/prize/discard), opponent mirror (minus hand contents), stadium,
  turn scalars and once-per-turn flags.

## Phase 2 — IS-MCTS + expert iteration (uses the search API)

The game is imperfect-information: opponent hand/deck/prizes and our own
deck order/prizes are hidden. `search_begin` accepts a **determinization** —
predicted card IDs for every hidden zone — and returns a fully-observable
simulation we can advance with `search_step`.

Correct bindings (the official `cg/api.py` from the competition, recovered and
mirrored in `pkm/search.py`):

- `agent_ptr = lib.AgentStart()` once per process.
- `lib.SearchBegin(agent_ptr, search_begin_input, len, your_deck, your_prize,
  opp_deck, opp_prize, opp_hand, opp_active, manual_coin) -> char* (ApiResult JSON)`
  — the `search_begin_input` ASCII blob comes from the observation.
- `lib.SearchStep(agent_ptr, search_id: int64, select, n) -> char*`.
- `lib.SearchEnd(agent_ptr)` after each real decision; `lib.SearchRelease`
  frees individual nodes. **Call these diligently or memory grows.**

Components (`pkm/mcts/`):

1. **`determinize.py` — belief tracker.** From our own decklist and the
   observation, compute the multiset of unseen cards
   (decklist − hand − board − discard − revealed prizes), then randomly deal
   them into deck/prizes (self) and deck/hand/prizes (opponent, from their
   inferred decklist: cards they've revealed + a prior; in training self-play
   we know the exact opponent decklist since we chose it). Sampling one deal =
   one determinization.
2. **`search.py` — PUCT MCTS.** For each of D determinizations: `search_begin`
   → root; run S simulations. Selection by PUCT with policy-network priors;
   leaf evaluation by the value network (no random rollouts); backup from the
   perspective of the player to move. Chance events (coin flips, shuffles)
   re-randomize inside the engine per `search_step`, so an edge's child is a
   *sample*; v1 keeps the first sampled child per edge (approximation, noted).
   Root visit counts are **summed across determinizations** and the most
   visited action is played.
3. **`agent.py`** — agent function: budgeted MCTS per decision, falls back to
   raw policy on trivial decisions (single option) and low time budgets.
4. **`pkm/rl/exit_train.py` — expert iteration.** Self-play where both players
   move by MCTS. Store `(state, options, visit distribution, final outcome)`;
   train the network with cross-entropy to the visit distribution + value MSE.
   The improved network gives better priors/evals → stronger search → better
   targets. Repeat.

**Submission strategy**: search at inference is expensive and Kaggle enforces
per-move time limits. The distilled *policy network* (one forward pass per
decision, numpy) is the safe submission; enable a small search on top only if
measured per-move budget allows.

## GAE — Generalized Advantage Estimation

GAE estimates **how much better an action was than the policy's average** at a
given state — the *advantage*. It blends multi-step returns using a single
parameter λ (lambda) to trade off bias vs variance.

### TD error

At step `t` the one-step temporal-difference error is:

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
```

This is the "surprise" — actual reward plus discounted next-value minus
predicted value. A single δ is noisy but low-variance.

### Multi-step returns

Instead of using only 1-step δ, we could use 2-step, 3-step, ... or full
episode returns. Each length has different bias/variance characteristics:

- **1-step** (TD(0)): low variance, high bias (bootstraps heavily)
- **N-step** (TD(N)): less bias, more variance
- **Full episode** (Monte Carlo): unbiased, highest variance

### The GAE formula

GAE computes an exponentially-weighted average of all n-step advantages:

```
A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

- **λ = 0** → only 1-step TD (pure bootstrap, lowest variance)
- **λ = 1** → full Monte Carlo returns (no bootstrap, lowest bias)
- **λ ≈ 0.95** → practical sweet spot used in most PPO implementations

### Implementation

GAE is computed **backwards** through the trajectory in O(n):

```python
gae = 0.0
for t in reversed(range(n)):
    next_value = trajectory[t + 1].value if t + 1 < n else 0.0
    delta = rewards[t] + gamma * next_value - trajectory[t].value
    gae = delta + gamma * lam * gae
    trajectory[t].advantage = gae
    trajectory[t].ret = gae + trajectory[t].value
```

Each step accumulates: `gae_t = δ_t + γλ · gae_{t+1}`. The result is stored on
each `EncodedDecision` so PPO can weight the policy gradient by how surprising
each action was.

### How it fits in the pipeline

1. **Rollout** collects a trajectory of `EncodedDecision`s (state, action,
   logprob, value prediction)
2. **`compute_returns()`** walks the trajectory backwards, fills `advantage` and
   `ret` on each decision via GAE
3. **`ppo_update()`** reads those fields to compute the PPO clip loss and
   update the network — actions with high advantage get reinforced, actions
   with low/negative advantage get suppressed

## Verification milestones

1. Rewritten `pkm/search.py` round-trips: `search_begin` on a live observation
   returns a root, `search_step` advances it, terminal states show `result`.
2. Phase 1 smoke: loss decreases; win rate vs. random agent > 60% after a
   short run (random-vs-random is 50% by construction).
3. Phase 2 smoke: MCTS agent beats the raw policy network head-to-head.
