# Training and Inference Architecture

## The Network: `pkm/rl/model.py`

```
PolicyValueNet
├── card_emb: Embedding(NUM_CARDS, 32)     — embed card IDs
├── attack_emb: Embedding(NUM_ATTACKS, 16) — embed attack IDs
├── opt_type_emb: Embedding(NUM_OPT_TYPES, 8) — embed option types
├── stop_vec: Parameter(64)                — learned STOP embedding
│
├── State Encoder (obs -> h)
│   ├── board cards -> card_emb -> flatten  (N_BOARD_SLOTS * 32 dims)
│   ├── hand cards  -> card_emb -> masked mean-pool (32 dims)
│   ├── state feats (scalars)              (STATE_FEATS dims)
│   └── concat -> Linear -> ReLU -> Linear -> ReLU -> h (128-dim)
│
├── Policy Head (h + option -> logits)
│   ├── each option: card_emb + card2_emb + attack_emb + opt_type_emb + feats -> Linear -> ReLU (64-dim)
│   ├── append learned STOP row
│   ├── for each: concat(h, option_enc, picked_sum) -> Linear -> ReLU -> Linear -> logit
│   └── mask illegal -> softmax -> sample/argmax
│
└── Value Head (h -> V(s))
    └── h -> Linear -> ReLU -> Linear -> tanh -> scalar in [-1,+1]
```

---

## Phase 1 Training: PPO Self-Play (`pkm/rl/train.py`)

**Loop per iteration:**

```
1. model.eval()
2. For each of N games:
   a. 40% chance: pick random past checkpoint from pool as opponent
      60% chance: mirror match (current vs current, collect both sides)
   b. play_game() -> battle_start -> loop battle_select -> battle_finish
   c. Each decision: obs -> encode_decision -> model.act() -> picks
      - model.act() at model.py:154-197:
        - encode_state -> h
        - encode_options -> opts
        - LOOP: score (opts + STOP) against h, softmax, sample
        - accumulate logprob, track picks
   d. record EncodedDecision with (picks, logprob, value, potential)
   e. compute_returns() with GAE + potential-based shaping
      - r_t = 0.2 * (gamma * phi(s_{t+1}) - phi(s_t))   [shaping]
      - r_terminal = outcome - 0.2 * phi(s_terminal)
      - GAE: delta = r + gamma*V(s') - V(s); advantage = discounted sum
3. model.train()
4. ppo_update():
   - For K epochs, minibatch over all decisions:
     - Recompute logprobs, values via model.evaluate()
     - ratio = exp(new_logprob - old_logprob)
     - policy_loss = -min(ratio*adv, clip(ratio)*adv)
     - value_loss = MSE(V, returns)
     - loss = policy + 0.5*value - 0.01*entropy
     - clip grad, step optimizer
5. Append current state_dict to opponent pool (FIFO, size=8)
6. Every 5 iters: eval vs random, save checkpoint
```

---

## Export: `pkm/rl/export.py`

```
torch checkpoint (.pt)  -->  state_dict  -->  numpy arrays  -->  .npz file
                                                        pkm/policy.npz
```

Simply `np.savez_compressed(path, **state_dict_as_numpy)`. Every weight key maps to a `.npy` inside the zip.

---

## Inference (Kaggle): `pkm/rl/numpy_policy.py` + `pkm/agents/neural_agent.py`

**No torch at runtime.** The numpy policy replays the exact same forward pass:

```
neural_agent.py: agent(obs)
├── obs["select"] is None -> return deck (60 card IDs)
└── else -> NumpyPolicy.select(obs)
         -> encode_decision(obs)       [same encoder as training]
         -> act_greedy(decision):
            1. _encode_state: card_emb lookup (numpy), concat, Linear->ReLU->Linear->ReLU -> h
            2. _encode_options: same embeddings, Linear->ReLU -> opts (N x 64)
            3. LOOP picks:
               - concat(h, each_opt+STOP, picked_sum) -> Linear->ReLU->Linear -> logits
               - argmax (greedy)
               - if STOP: break
               - else: append pick, update picked_sum, mask out picked option
            4. return picks
```

**Weight lookup order** (`neural_agent.py:14-25`):

1. explicit `weights_path` arg
2. `$PKM_POLICY_PATH` env var
3. `pkm/policy.npz` next to the package (bundled in submission)
4. `/kaggle_simulations/agent/pkm/policy.npz`
5. `/kaggle_simulations/agent/policy.npz`

Falls back to random if no weights found.

---

## Phase 2 Training: Expert Iteration (`pkm/rl/exit_train.py`)

```
For each iteration:
1. Export model -> NumpyPolicy (for MCTS priors)
2. For each game:
   a. Both players use MCTS (IS-MCTS with determinization)
   b. MCTS uses network as: prior provider + leaf evaluator
   c. Collect: (decision, MCTS visit distribution, outcome)
   d. Early game: sample from visit dist (temperature)
3. exit_update():
   - Single-pick decisions: KL(MCTS_target || network_probs) + MSE(V, outcome)
   - Multi-pick decisions: behavior-clone the MCTS-chosen sequence
```

---

## Data Flow Summary

```
TRAINING (torch):
  obs dict -> Observation (pydantic) -> encode_decision() -> EncodedDecision
  EncodedDecision -> model.act() -> picks, logprob, value
  Trajectory -> compute_returns() -> advantages
  Decisions -> ppo_update() -> model.evaluate() -> gradient step
  Checkpoint -> export_npz() -> policy.npz

INFERENCE (numpy, Kaggle):
  obs dict -> Observation (pydantic) -> encode_decision() -> EncodedDecision
  EncodedDecision -> NumpyPolicy.act_greedy() -> picks
  (weights loaded from policy.npz, no torch dependency)
```
