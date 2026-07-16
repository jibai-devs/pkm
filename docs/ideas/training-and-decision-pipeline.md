# How the Network Makes Decisions and Gets Trained

**Status:** Reference doc (Jul 2026). Covers the full pipeline from raw
observation to gradient update.

---

## Part 1: Decision Pipeline (Inference)

### Step 1: Raw observation arrives

The engine hands you a dict (like `obs_data_structure/example_obs.json`). It
has:
- `select` — what you need to decide (options, min/max count)
- `current` — full board state (both players' Pokémon, hands, prizes, etc.)

### Step 2: Check if it's a forced decision

`pkm/rl/rollout.py:37-40` — before running the network:

```python
# only 1 option and you must pick it? → just pick it, skip network
if n == 1 and sel.minCount >= 1:
    return [0], None
# must pick exactly all options? → pick all, skip network
if n == sel.minCount == sel.maxCount:
    return list(range(n)), None
```

No point wasting compute on decisions with no choice.

### Step 3: Encode the observation → numpy arrays

`pkm/rl/encoder.py:319-332` — `encode_decision(obs)` converts the dict into:

**State encoding** (`encoder.py:98-177`):
- `board_cards` — 18 card IDs (your active + 8 bench, opponent active + 8
  bench, stadium). Each gets embedded to 32 dims by the network.
- `hand_cards` — up to 25 card IDs in your hand.
- `state_feats` — 135 floats: HP fractions, energy counts, statuses
  (poisoned/burned/etc), hand/deck/prize/discard counts, turn number,
  once-per-turn flags, select type one-hot.

**Option encoding** (`encoder.py:230-314`):
- `opt_type` — what kind of option (attack=13, end turn=14, play card=7, etc.)
- `opt_card` — the card involved (card ID)
- `opt_card2` — target card (e.g., which Pokémon to attach energy to)
- `opt_attack` — attack ID (for attack options)
- `opt_feats` — 5 floats: number, count, damage, cost, is-self

### Step 4: Network forward pass

`pkm/rl/model.py:155-197` — `act()`:

**4a. Encode state → h**
```python
h = self.encode_state(board, hand, feats)  # → 128-dim vector
v = self.value(h)                          # → scalar in [-1, +1]
```

The state encoder (`model.py:87-99`):
- Embed each board card ID → 32-dim vector (18 cards → 576 dims)
- Embed hand cards → average pooling into 32 dims (bag of embeddings)
- Concatenate with scalar features → 641-dim input
- Two linear layers (641 → 512 → 128) with ReLU → `h`

**4b. Encode each option → o_i**
```python
opts = self.encode_options(...)  # → (n_options, 64) tensor
```

Each option: embed its card + target card + attack + type → concat with
features → 64-dim vector.

**4c. Score and pick (the loop)**

This is the key part — `model.py:179-195`:

```python
while len(picks) < max_count:
    # STOP is legal once we've picked min_count items
    available[0, n] = len(picks) >= min_count

    # score every option + STOP
    logits = self.option_logits(h, opts, picked_sum, available)

    # softmax → probabilities
    logp = F.log_softmax(logits, dim=-1)

    # pick: greedy (argmax) or sample (multinomial)
    idx = int(torch.multinomial(logp.exp(), 1)[0, 0])

    # if STOP → done
    if idx == n:
        break

    # otherwise record the pick, mask it out, add to picked_sum
    picks.append(idx)
    picked_sum = picked_sum + opts[0, idx]
    available[0, idx] = False
```

The scoring in `option_logits` (`model.py:118-141`):

```python
# for each option, compute: MLP([h, option_embedding, picked_so_far]) → score
hx = h.expand to match each option
px = picked_sum.expand to match each option
x = concat([hx, option_row, px])  # → 256-dim
logits = score_fc2(ReLU(score_fc1(x)))  # → scalar per option
```

So each option gets scored by: "given the board state `h`, and what I've
already picked `picked_sum`, how good is this option?"

### Step 5: Return the picks

```python
return ActResult(picks=[2, 0], stopped=False, logprob=-1.23, value=0.45)
```

The `picks` list (e.g., `[2, 0]`) is what gets sent back to the engine via
`battle_select([2, 0])`.

### Visual summary

```
obs (dict)
    │
    ├─ encode_state ──→ board_cards, hand_cards, state_feats
    │                        │              │            │
    │                   card_emb         card_emb     raw floats
    │                        │              │            │
    │                        └──────┬───────┘            │
    │                               │                    │
    │                          concat + MLP ──→ h (128-dim) ──→ value head ──→ V(s) = 0.45
    │
    └─ encode_options ──→ opt_type, opt_card, opt_attack, opt_feats
                              │          │         │          │
                         type_emb    card_emb  attack_emb   raw floats
                              │          │         │          │
                              └────┬─────┘─────────┘──────────┘
                                   │
                              concat + MLP ──→ opts (n × 64-dim)
                                   │
                    ┌──────────────┘
                    │
         option_logits(h, opts, picked_sum, mask)
                    │
              MLP([h, opt_i, picked_sum]) → score_i  (for each option + STOP)
                    │
              softmax → [0.1, 0.6, 0.2, 0.1]
                    │
              sample or argmax → pick index 2
                    │
              (loop if multi-pick: mask option 2, re-score remaining)
```

---

## Part 2: Training Pipeline (PPO)

### The full training loop

One PPO iteration (`pkm/rl/train.py:104-143`):

```
1. COLLECT: play N games, record (obs, action, logprob, value, potential)
             for every decision by the learning agent

2. COMPUTE RETURNS: for each trajectory, walk backward computing GAE
             → each decision now has: advantage, return

3. UPDATE: for 3 epochs, shuffle into mini-batches of 256
             → compute loss, backprop, step optimizer
```

### Step 1: Collect rollouts

`pkm/rl/rollout.py:68-101` — `play_game()`:

For each decision in the game:
1. `TorchPolicy.act(obs)` runs the network forward (no grad)
2. Records: `logprob` (of the action taken), `value` (V(s) prediction),
   `potential` (prize differential for shaping)
3. The trajectory is a list of `EncodedDecision` objects, one per decision

After the game ends, `result` gives +1/-1/0.

### Step 2: Compute returns — the GAE delta

`pkm/rl/ppo.py:10-40` — `compute_returns()`:

First, compute per-step rewards (shaping only, the real reward is terminal):

```python
# shaping reward at each non-terminal step
rewards[t] = shaping_coef * (gamma * phi(s_{t+1}) - phi(s_t))

# terminal step: the actual win/loss minus shaping at final state
rewards[n-1] = terminal_reward - shaping_coef * phi(s_{n-1})
```

Where `phi(s)` = prize differential (`encoder.py:335-346`).

Then, backward pass to compute GAE (Generalized Advantage Estimation):

```python
gae = 0.0
for t in reversed(range(n)):
    next_value = trajectory[t + 1].value if t + 1 < n else 0.0
    delta = rewards[t] + gamma * next_value - trajectory[t].value
    gae = delta + gamma * lam * gae
    trajectory[t].advantage = gae
    trajectory[t].ret = gae + trajectory[t].value
```

**What delta means:**

`delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`

This is the **TD error** (temporal difference error). It answers: "how wrong was
my value prediction?"

- `r_t` — reward at this step (mostly from shaping, except terminal)
- `gamma * V(s_{t+1})` — discounted value of the next state (what I think
  the future is worth from here)
- `V(s_t)` — value I predicted at this state (what I thought the future was
  worth)

If delta > 0: "things turned out better than I predicted" → this action was
better than expected → positive advantage → reinforce it.

If delta < 0: "things turned out worse than I predicted" → this action was
worse than expected → negative advantage → discourage it.

**What GAE does:**

Instead of just using delta_t, GAE accumulates it with exponential decay:

`gae_t = delta_t + gamma * lambda * gae_{t+1}`

This blends one-step estimates (accurate but noisy) with multi-step estimates
(smooth but biased). `lambda=0.95` balances the two. Walking backward means
each step's advantage includes credit/blame from all future steps.

**What `ret` is:**

`ret_t = gae_t + V(s_t)` — the target for training the value head. It's "what
the advantage was + what I already predicted" = "what the actual return should
have been."

### Step 3: PPO update — where the gradient flows

`pkm/rl/ppo.py:43-94` — `ppo_update()`:

For each mini-batch:

```python
# 1. Recompute logprobs and values WITH gradient tracking
logprobs, entropies, values = model.evaluate(batch)

# 2. Policy ratio: how much has the policy changed?
ratio = exp(logprob_new - logprob_old)

# 3. Clipped surrogate objective
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-eps, 1+eps) * advantage
policy_loss = -min(surr1, surr2).mean()

# 4. Value loss: predicted value vs actual return
value_loss = MSE(values, returns)

# 5. Entropy bonus: encourage exploration
entropy = mean(per_step_entropy)

# 6. Combined loss
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

# 7. Gradient step
optimizer.zero_grad()
loss.backward()                    # ← gradients flow here
clip_grad_norm_(model.parameters(), 0.5)
optimizer.step()
```

**Where the gradient goes:**

`loss.backward()` computes gradients for every parameter. The three loss
components push different things:

| Loss component | Gradient pushes | Which parameters |
|---|---|---|
| `policy_loss` | "actions with high advantage → higher probability" | score_fc1, score_fc2, state_fc1/2, card_emb, etc. |
| `value_loss` | "predicted value closer to actual return" | value_fc1, value_fc2, state_fc1/2, card_emb |
| `entropy` | "keep probabilities spread out, don't collapse to one action" | score_fc1, score_fc2 |

Both policy and value heads share the state encoder (`state_fc1`, `state_fc2`,
`card_emb`), so gradients from both flow through the same parameters. The
encoder learns representations useful for both picking actions AND estimating
value.

**The clipping mechanism:**

```python
ratio = exp(logprob_new - logprob_old)
surr1 = ratio * advantage           # unclipped
surr2 = clamp(ratio, 0.8, 1.2) * advantage  # clipped
policy_loss = -min(surr1, surr2)
```

If the policy changed too much (ratio outside [0.8, 1.2]), the gradient is
clipped to zero in that direction. This prevents catastrophic policy updates
that destroy what was learned.

---

## Part 3: Code Modularity Analysis

### Current coupling map

```
encoder.py ─────────────────────────────┐
  encode_state()     → numpy arrays     │
  encode_options()   → numpy arrays     │
  EncodedDecision    → dataclass        │
  prize_potential()  → float            │
                                         │
model.py ───────────────────────────────┤
  PolicyValueNet                         │
    encode_state()   ← takes numpy arrays from encoder
    encode_options() ← takes numpy arrays from encoder
    option_logits()  ← takes h, opts, picked_sum from encode_*
    value()          ← takes h from encode_state
    act()            ← orchestrates: encode → score → pick
    evaluate()       ← re-runs forward pass with grad for training
                                         │
rollout.py ──────────────────────────────┤
  TorchPolicy.act()  ← calls encoder then model
  play_game()        ← drives TorchPolicy, records trajectory
                                         │
ppo.py ──────────────────────────────────┤
  compute_returns()  ← reads EncodedDecision.value/.potential
  ppo_update()       ← calls model.evaluate(), computes loss
                                         │
train.py ────────────────────────────────┘
  train()            ← orchestrates: rollout → compute_returns → ppo_update
```

### What's coupled and why

**1. Encoder ↔ Model (tightly coupled)**

The encoder produces numpy arrays with specific shapes that the model's
embedding tables expect. `board_cards` must be `(N_BOARD_SLOTS,)` int64,
`opt_type` must be `(N,)` int64, etc. The constants (`N_BOARD_SLOTS`,
`NUM_CARDS`, `OPT_FEATS`) are shared via imports.

If you swap the model (e.g., attention-based), you'd likely need to change
how options are encoded (maybe as a single graph instead of parallel arrays).

**2. Model ↔ PPO (loosely coupled)**

PPO calls `model.evaluate(batch)` which returns `(logprobs, entropies, values)`.
As long as a new model implements `evaluate()` with the same signature, PPO
works unchanged. This is the cleanest seam.

**3. Encoder ↔ Rollout (tightly coupled)**

`TorchPolicy.act()` calls `encode_decision(obs)` directly. If you change
what features are encoded, rollout doesn't need to change — but the model's
input dimensions must match.

**4. PPO ↔ EncodedDecision (moderately coupled)**

`compute_returns()` reads `d.value`, `d.potential`, and writes `d.advantage`,
`d.ret`. The `EncodedDecision` dataclass is the shared data format between
rollout, PPO, and model.

### What you can swap today

| Component | Swap difficulty | What to change |
|---|---|---|
| **Value head** (MLP → something else) | Easy | Override `value()` in a subclass |
| **Policy scoring** (MLP → attention) | Medium | Override `option_logits()`, may need new option encoding |
| **State encoder** (MLP → transformer) | Medium | Override `encode_state()`, change `STATE_IN` |
| **PPO algorithm** (PPO → SAC/DQN) | Medium | New file, reuse `model.evaluate()` |
| **Reward shaping** (prize diff → something else) | Easy | Change `prize_potential()` in encoder |
| **Option encoding** (parallel arrays → graph) | Hard | Changes encoder + model input layer |
| **Training loop** (self-play → something else) | Medium | New file, reuse rollout + model |

### Suggested refactoring for swappability

To make parts easily swappable:

1. **Define interfaces (Protocols)** for model, encoder, and trainer:
   ```python
   class ModelProtocol(Protocol):
       def evaluate(self, batch) -> tuple[Tensor, Tensor, Tensor]: ...
       def act(self, d: EncodedDecision, **kw) -> ActResult: ...

   class EncoderProtocol(Protocol):
       def encode(self, obs: Observation) -> EncodedDecision: ...
   ```

2. **Separate the state encoder from the model** — currently
   `encode_state()` lives on `PolicyValueNet` as a method. Extract it to a
   standalone module so the encoder can be swapped independently.

3. **Make the option encoder swappable** — the parallel-arrays format
   (`opt_type`, `opt_card`, etc.) is baked into both encoder and model. A
   graph-based model would need a different format. Define a protocol for
   "give me option representations" that both sides implement.

4. **Separate the scoring head** — `option_logits()` is the policy head.
   Making it a pluggable module (MLP, attention, dot-product) lets you swap
   architectures without touching the rest.

5. **Make the value head pluggable** — already clean: `value(h) -> scalar`.
   Just override in a subclass.

6. **Isolate the loss function** — PPO's loss is in `ppo_update()`. To try
   SAC or DQN, you'd write a new update function with the same signature:
   `(model, optimizer, decisions) -> stats`.

### Concrete next steps

If you want to make this modular, the minimal refactor is:

1. Extract `encode_state` and `encode_options` from `PolicyValueNet` into
   `encoder.py` as standalone functions (they're already mostly there —
   `encoder.py` has `encode_state()` and `encode_options()` that return
   numpy; `model.py` has methods that wrap them in torch).
2. Define a `ModelProtocol` with `evaluate()` and `act()`.
3. Have `PolicyValueNet` implement it.
4. Have `ppo_update()` and `exit_update()` accept any `ModelProtocol`.
5. Make the training loop accept a model factory and an update function.
