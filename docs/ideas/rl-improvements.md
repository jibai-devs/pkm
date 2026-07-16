# RL Techniques & Improvement Ideas

**Status:** Ideas from discussion (Jul 2026). Not yet prioritized or scheduled.

## Current Training Overview

### How the reward signal works

Reward is **sparse and binary**: win = +1, loss = -1, draw = 0. No intermediate
rewards. The agent makes ~50-100 decisions per game but only learns the outcome
at the end.

Credit assignment happens through:

1. **GAE (Generalized Advantage Estimation)** — the value head predicts expected
   outcome from each state. Advantage = actual outcome - predicted outcome. Moves
   that led to a better-than-expected result get positive advantage (reinforced);
   worse-than-expected get negative (discouraged).
2. **Potential-based reward shaping** — prize differential (your prizes taken vs
   opponent's) provides a dense local signal. Added as `γφ(s') - φ(s)` at each
   step, which is provably optimal-policy-preserving. Coefficient: `shaping_coef=0.2`.

### What each trainer does

- **PPO (Phase 1, `pkm/rl/train.py`)** — network picks moves directly. Collect
  rollout, compute advantages, do 3 epochs of mini-batch SGD (batch size 256),
  discard data. Opponent sampled from checkpoint pool (size 8) to prevent
  cycling. No search.

- **Expert Iteration (Phase 2, `pkm/rl/exit_train.py`)** — MCTS picks moves (32
  simulations per move, 2 determinizations). Network trained to match MCTS visit
  distributions (policy head) and game outcomes (value head). Stronger net ->
  better priors -> stronger search -> better targets.

### How MCTS works (search.py)

MCTS does **not** try every action. It runs a fixed budget of simulations,
guided by the network:

1. **Determinization** — sample a plausible world state (guess opponent's
   hand/deck/prizes, shuffle unknowns in your own deck).
2. **Simulation** — walk from root to leaf using PUCT selection:
   - `q` (exploitation) = average value of this action from past simulations
   - `u` (exploration) = `c_puct * prior * sqrt(parent_visits) / (1 + visits)`
   - Pick `argmax(q + u)`, go deeper
3. **Expansion** — at unexpanded node, ask network for value estimate + action
   priors.
4. **Backup** — propagate the value back up the path.
5. **Aggregate** — sum visit counts across determinizations, pick most-visited.

### Network architecture: V(s) not Q(s, a)

The network (`pkm/rl/model.py`) has two heads:

- **Policy head** (lines 116-135): scores each legal option against the state
  embedding `h` via a 2-layer MLP, producing logits over (options + STOP).
  The agent samples or argmaxes from the softmax. Multi-pick decisions are
  sequential: pick one, mask it, re-score, repeat.

- **Value head** (lines 137-146): V(s) predictor. Two linear layers on `h`,
  tanh squashed to [-1, +1]. Predicts the expected final outcome (win=+1,
  loss=-1) from the current state, **regardless of action taken**.

This is a **state value function V(s)**, not an action-value function Q(s, a).

| | V(s) — what we have | Q(s, a) — alternative |
|---|---|---|
| Predicts | Value of the state | Value of taking action a in state s |
| Output | One number per state | One number per (state, action) pair |
| Used for | Leaf evaluation in MCTS, advantage baseline in PPO | Could replace PUCT scoring, enable offline RL |
| Cost | One forward pass per state | One per (state, action) pair — much more expensive |

**Why V(s) and not Q(s, a)?** With variable-length action spaces, Q would
require scoring every option per state anyway — essentially duplicating the
policy head's work. V + separate policy scoring is the AlphaZero approach: the
policy head says "which action," the value head says "how good is this state."
They share the state encoder (`h`) so computation is reused.

**How V(s) is used in PPO** (`pkm/rl/ppo.py:47-51`):

```python
delta = rewards[t] + gamma * next_value - trajectory[t].value
gae = delta + gamma * lam * gae
trajectory[t].advantage = gae
```

The value prediction is subtracted as a baseline: "how much better was the
actual outcome than I expected?" This reduces variance in the policy gradient
without changing its expected direction.

**How V(s) is used in MCTS** (`pkm/mcts/search.py:162-163`):

```python
v = self.policy.value(d)        # leaf evaluation
v0 = v if node.player == 0 else -v  # flip for opponent perspective
```

The value head evaluates leaf nodes when simulation reaches an unexpanded node:
"if we stop searching here, how good is this position?" This replaces the
random rollouts used in classic MCTS.

---

## Experience Replay

### What it is

Store past transitions `(state, action, reward, next_state, done)` in a buffer.
During training, sample random mini-batches from this buffer instead of using
only the most recent rollout.

### Why it helps

- **Breaks temporal correlation** — sequential game states are correlated; random
  sampling decorrelates batches, stabilizing gradients.
- **Data efficiency** — each transition reused across many updates.
- **Off-policy learning** — can learn from data generated by older policies.

### Uniform replay (simple version)

```python
buffer = deque(maxlen=100_000)
buffer.append((s, a, r, s2, done))
batch = random.sample(buffer, batch_size)
```

### Prioritized Experience Replay (PER)

Weight transitions by TD-error — transitions the model is most surprised by get
sampled more often. Uses a sum-tree for efficient weighted sampling.

### Current status in this codebase

PPO already does a micro form of replay: reuses rollout data for K=3 epochs
before discarding. But the buffer is tiny (one rollout) and short-lived.

**Adding a full replay buffer to PPO is tricky** because PPO's clipped objective
assumes data came from π_θ_old. Very old data has stale logprobs, causing the
importance ratio to explode. Would need importance weighting or a switch to a
fully off-policy algorithm.

**Best fit: use replay with expert iteration.** MCTS visit targets are
supervision signals (not importance-ratio-dependent), so mixing old and new
training data is safe.

---

## Offline RL from Replay Logs

### The opportunity

We have many replay logs from past games (various agents, possibly including
human play). These contain full game observations at each decision point.

### What's in a replay log

Each observation has:
- Full game state (`current`) — board, hands, prizes, bench, statuses
- Legal options (`select`) — the action space at that decision
- Game history (`logs`) — all moves played so far
- Search state blob (`search_begin_input`)

### What's NOT in a replay log

- The action that was taken (must be reconstructed from `logs`)
- Old policy logprobs (needed for PPO, not for offline RL)
- Advantage estimates

### Methods that work without logprobs

| Method | What it needs | Complexity |
|---|---|---|
| **Behavior cloning** | (obs, action) pairs | Low — cross-entropy loss |
| **DAgger** | (obs, action) + online queries to expert | Medium — needs live expert |
| **CQL (Conservative Q-Learning)** | (obs, action, reward, next_obs) | High — full offline RL |
| **IQL (Implicit Q-Learning)** | (obs, action, reward, next_obs) | High — avoids querying Q |

### Practical approach: supervised pretraining

1. Parse replay logs to extract `(obs, action)` pairs
2. Run `encode_decision` on each observation
3. Train policy head with cross-entropy loss against recorded actions
4. Use as warm start for PPO or expert iteration

This gives better-than-random initialization without any self-play.

### Why offline RL is promising

Replays come from multiple agents (different skill levels, different policies).
CQL's conservative penalty specifically handles mixed-quality data — it avoids
overestimating values for out-of-distribution actions. You can learn from both
expert and suboptimal demonstrations.

---

## Other Techniques Worth Considering

### Quick wins (low effort, high impact)

1. **Larger batches** — try 32-64 games/iteration instead of 8. PPO benefits
   from large batches. More diverse data -> better gradient estimates -> more
   stable updates.

2. **KL penalty instead of clip** — measure policy change directly, penalize
   large shifts. More principled than hard clipping. Some papers show better
   convergence.

3. **Learning rate schedule** — cosine annealing or step decay. Prevents
   late-training collapse when the policy is mostly converged.

4. **Value function clipping** — clip the value head update too, not just
   policy. Prevents value function overfitting to a single batch.

### Medium effort

5. **Curriculum opponent selection** — don't sample pool uniformly. Weight
   toward stronger opponents as training progresses. Start vs random, gradually
   introduce harder opponents.

6. **Auxiliary losses** — force the network to build better representations:
   - Predict opponent's hand contents
   - Predict next card to be drawn
   - Predict prize card count
   These multi-task objectives improve the shared encoder.

7. **Network architecture upgrades**:
   - Attention over bench/active Pokémon (variable-length sets)
   - Separate encoders for board state vs hand vs options
   - Wider MLP, more embedding dimensions
   - Graph neural network over the board state

### High effort, high reward

8. **MCTS at inference time** — exit_train already trains for this. The big
   AlphaZero insight: MCTS at play time + network for priors/evaluation beats
   raw policy by a huge margin. Kaggle time limits are the constraint.

9. **Distributed training** — N self-play workers in parallel, aggregate
   gradients. Scales linearly with workers.

10. **Multi-deck training** — sample opponent decks from a pool for robustness.
    Currently trains against a single deck.

11. **MuZero-style learned world model** — instead of using the game engine for
    MCTS rollouts, learn a model of state transitions. Enables planning without
    the engine (useful if engine access is limited).

---

## Interactive Training (Human-in-the-Loop)

### The idea

During TUI play, use the human's moves as a training signal:
- Network picks same move as human -> reward
- Network picks different move -> penalty

### Assessment

This is essentially **online behavior cloning**. It can work for bootstrapping
but has significant limitations:

| Concern | Detail |
|---|---|
| Skill ceiling | Network can at best mimic the human, not exceed them |
| Data rate | TUI play is slow; need thousands of decisions for training |
| Inconsistency | Human may play differently on same board state across games |
| Better alternative | Parse existing replay logs instead of playing live |

### When interactive IS useful

- **Reward shaping** — human flags critical decisions during play
- **Curriculum** — play specific situations the agent struggles with
- **Debugging** — compare network choices vs human choices, identify weaknesses

### Recommendation

Don't train live. Parse replays for behavior cloning warm-start, then let
self-play (PPO or expert iteration) take over. Self-play is faster, more
consistent, and can surpass human level.

---

## Summary: Priority Ranking

| # | Technique | Effort | Impact | Prerequisite |
|---|---|---|---|---|
| 1 | Parse replays -> supervised pretrain | Weekend | Medium | Replay parser |
| 2 | More games per iteration (8->32) | Config change | Medium | None |
| 3 | Exit_train on replay data | Low | Medium | Replay parser |
| 4 | Bigger network + attention | Medium | High | Architecture work |
| 5 | Experience replay for exit_train | Medium | Medium | Buffer implementation |
| 6 | Offline RL (CQL/IQL) pretrain | High | High | Replay parser + new algo |
| 7 | MCTS at inference | Medium | Very high | Time budget analysis |
| 8 | Distributed self-play | High | High | Infrastructure |
| 9 | Multi-deck training | Medium | Medium | Deck pool |
| 10 | Auxiliary prediction heads | Medium | Medium | Network refactor |
