# RL Self-Play Guide

How the reinforcement-learning stack in this repo came to be, how it works,
and how to operate it: training, checkpoints, weights, replays, visualization.

Companion docs: [`RL_PLAN.md`](RL_PLAN.md) is the design document written
before implementation; this guide is the operator's manual written after.

---

## 1. The research process (what happened)

### 1.1 Environment archaeology

The competition engine (cabt) is a closed C library shipped inside
`kaggle_environments`. Reading the installed package showed:

- `cg/sim.py` loads `libcg.so` and declares ctypes prototypes for the
  *battle* API (`BattleStart`, `Select`, `GetBattleData`...), but **not** for
  the search API.
- `nm -D libcg.so` confirmed `SearchBegin/SearchStep/SearchEnd/SearchRelease`
  exist in the binary, so search is available locally — the Python wrapper
  just isn't shipped.
- Every observation carries a `search_begin_input` field: an ASCII blob the
  engine hands you specifically so you can start forward simulations from
  the current (imperfect-information) state.

### 1.2 Recovering the official API

The docs site (matsuoinstitute.github.io/cabt) documents the *dataclasses*
but not the C signatures. The repo's original `pkm/search.py` guessed that
`SearchBegin` takes a battle pointer — **that was wrong**. GitHub code search
for `search_begin_input` surfaced competitor repos containing the official
competition `cg/api.py`, which revealed the real contract:

- `lib.AgentStart() -> void*` — one agent handle per process; every Search*
  call takes it as the first argument.
- `SearchBegin(agent_ptr, search_begin_input, len, your_deck, your_prize,
  opp_deck, opp_prize, opp_hand, opp_active, manual_coin) -> char*` returning
  ApiResult JSON `{state: {observation, searchId}, error}`.
- `SearchStep(agent_ptr, search_id: int64, select, n)` — search ids are
  **int64**, each step returns a *new* state id (the tree is ids).
- `SearchEnd(agent_ptr)` frees all search memory — call it after every real
  decision or memory grows.

`pkm/search.py` was rewritten against this and verified by simulating 30
decisions into the future from a live mid-game state.

### 1.3 Phase 1 — model-free self-play (PPO)

Built the encoder → pointer network → rollout → PPO pipeline (§2). Key
verification steps, in order:

1. One self-play game runs end-to-end in **~0.08 s** (~170 decisions) —
   the C engine is fast enough that no parallelism was needed yet.
2. The batched training-time `evaluate()` reproduces the rollout-time
   log-probs to 2e-7 (catches any masking/padding bug in the pointer head).
3. A 40-iteration × 16-game run (~40 s total) raised the greedy policy's
   win rate vs the random agent from ~50% (by construction) to **80–90%**.

### 1.4 Phase 2 — search-guided self-play (IS-MCTS + expert iteration)

Built the determinizer + PUCT search + expert-iteration trainer (§2.3–2.4).
Verification:

1. Determinizations are accepted by the engine and MCTS returns legal picks
   (including legal *empty* selections on `minCount=0` decisions).
2. MCTS (2 det × 32 sims) beats random **95%**.
3. MCTS (3 det × 64 sims) beats the raw greedy policy **63%** over 30 games.
   The edge grows with search budget and — more importantly — with value-net
   quality, which is exactly what longer expert-iteration training improves.

### 1.5 Gotchas discovered along the way

- `uv add torch` silently **downgraded kaggle-environments** 1.30.2 → 1.29.3
  (index shadowing), which dropped the bundled HTML match visualizer. Fixed
  by marking the pytorch-cpu index `explicit = true` in `pyproject.toml` so
  only torch resolves from it.
- `env.toJSON()` returns a dict (not a string) in current versions.
- The Kaggle submission bundle is capped at **197.7 MiB** — too small to ship
  torch. Hence the numpy inference path (§4.2).

---

## 2. How it works

### 2.1 The action-space problem and the pointer network

Each decision, the engine presents `select.option` — a variable-length list —
and expects a list of indices back (between `minCount` and `maxCount` of
them). A fixed softmax head can't represent this, so `pkm/rl/model.py` uses a
pointer/scoring architecture:

```
observation ──► encoder.py ──► board card-ID slots (19) ┐
                               hand card-IDs (≤25)      ├─► embeddings ─► MLP ─► state vector h (128)
                               float features (135)     ┘
each option ──► (type, card, target-card, attack) IDs + floats ─► option vector o_i (64)

score(h, o_i, Σ picked so far) ─► logit per option ─► softmax over legal options only
value(h) ─► tanh scalar (win-probability estimate for the player to move)
```

Multi-select decisions are decomposed into **sequential picks**: sample an
option, mask it, re-score (the "already picked" sum lets the net condition on
its earlier picks), repeat. A learned **STOP row** becomes legal once
`minCount` picks are made. The decision's log-prob is the sum of per-pick
log-probs, so PPO can optimize it like any action.

Trivial decisions (1 option forced, or min == max == all) bypass the network
entirely — they're pure noise for credit assignment.

### 2.2 Phase 1: PPO self-play (`pkm/rl/train.py`)

- Games run directly on `battle_start`/`battle_select` (not the kaggle env
  wrapper) — faster, and gives both players' observations, so **each game
  yields a winning and a losing trajectory**.
- Reward: terminal ±1, plus *potential-based shaping* on the prize
  differential (`ppo.py`) — it densifies credit over ~100-decision games
  without changing the optimal policy.
- Opponents: the current policy plays itself, and with probability
  `pool_prob` a **past checkpoint** sampled from a pool — the standard
  defence against self-play cycling.
- Update: GAE(λ) + PPO clip + entropy bonus.

### 2.3 Determinization (`pkm/mcts/determinize.py`)

The game is imperfect-information; `search_begin` demands concrete card IDs
for every hidden zone. The determinizer computes the multiset of unseen cards
(decklist − hand − board − attachments − discard − revealed prizes) and deals
it randomly into the hidden zones (own deck order + face-down prizes;
opponent hand/deck/prizes/face-down active), respecting engine constraints
(face-down active must be a Pokémon; at setup the opponent deck must contain
a Basic). In training self-play both decklists are known exactly; at
inference `infer_opponent_decklist()` builds a crude estimate from cards seen.

### 2.4 IS-MCTS + expert iteration (`pkm/mcts/search.py`, `pkm/rl/exit_train.py`)

For each real decision: sample D determinizations, build one PUCT tree per
determinization on the search API, sum root visit counts across trees, play
the most-visited action. Leaves are evaluated by the value network (no random
rollouts); priors come from the policy network; multi-pick nodes use a small
set of policy-sampled candidate sequences as their action set. Chance events
re-randomize per `search_step`, so each edge's child is a sample (we keep the
first — a standard approximation).

Expert iteration closes the loop: MCTS-vs-MCTS self-play games produce
(state, visit distribution, outcome) targets; the network is trained toward
them (cross-entropy + value MSE); the improved network makes the next round
of search stronger. `exit_train.py` initializes from the PPO checkpoint.

---

## 3. How to train

### Phase 1 (do this first — fast, builds the foundation)

```bash
python -m pkm.rl.train --iterations 200 --games 16 --eval-every 10
```

Reads `deck.csv`, writes `checkpoints/ppo_iterNNNN.pt` + `checkpoints/ppo_latest.pt`.
Useful flags: `--lr 3e-4 --gamma 0.99 --shaping 0.2 --pool-size 8 --seed 0`.

Reading the log:

```
iter  33 | games 16 (W/L/D 10/6/0) | decisions 1584 | samples 932 | pi_loss -0.0064 | v_loss 0.1043 | ent 1.404 | clip 0.01 | 0.9s
iter  35 | eval vs random: 83.3% (30 games)
```

- `ent` (policy entropy) should decline slowly; a crash to ~0 means premature
  collapse (raise `entropy_coef` in `ppo.py`).
- `clip` above ~0.2 means updates too large (lower lr or epochs).
- `eval vs random` is the real progress signal; W/L/D in mirror games hovers
  near 50% by construction.

### Phase 2 (after Phase 1 plateaus)

```bash
python -m pkm.rl.exit_train --iterations 20 --games 8 --sims 32 --dets 2
```

Initializes from `checkpoints/ppo_latest.pt` (override with `--init`), writes
`checkpoints/exit_latest.pt`. Games here are slower (search per move); scale
`--sims`/`--dets` with your patience. Watch `pi_loss`/`v_loss` decline, and
periodically re-measure MCTS-vs-raw-policy head-to-head (§5).

---

## 4. Weights, checkpoints, and the submission

### 4.1 Checkpoints (`checkpoints/`, gitignored)

Plain torch `state_dict`s. `ppo_latest.pt` / `exit_latest.pt` are the live
pointers; `ppo_iterNNNN.pt` are periodic snapshots (they double as the
opponent pool if you resume). Load one with:

```python
model = PolicyValueNet()
model.load_state_dict(torch.load("checkpoints/exit_latest.pt", weights_only=True))
```

### 4.2 Exported weights for inference (`pkm/policy.npz`)

The submission cannot include torch (197.7 MiB cap), so inference re-implements
the exact forward pass in numpy (`pkm/rl/numpy_policy.py`, parity verified to
~1e-6 in `tests/test_rl.py`). Export whenever you have a better checkpoint:

```bash
python -m pkm.rl.export checkpoints/exit_latest.pt pkm/policy.npz
```

`make_neural_agent()` finds weights via: explicit path → `$PKM_POLICY_PATH` →
`pkm/policy.npz` → `/kaggle_simulations/agent/…`. The npz is ~1 MiB, so the
bundle (`./submit.sh`, which tars `main.py deck.csv pkm/`) stays tiny.
Note: `pkm/policy.npz` is gitignored as a build artifact — regenerate it, or
remove the ignore line if you'd rather version it.

### 4.3 Agents available

| agent | file | speed | use |
|---|---|---|---|
| `make_random_agent` | `pkm/agents/random_agent.py` | instant | baseline |
| `make_neural_agent` | `pkm/agents/neural_agent.py` | ~1 ms/move | safe submission |
| `make_mcts_agent` | `pkm/mcts/agent.py` | ~10–500 ms/move | strongest; check Kaggle's per-move budget before submitting |

---

## 5. Matches, replay logs, visualization

`pkm/rl/play.py` runs any pairing of `random | neural | mcts`:

```bash
# one rendered match: writes result.html + replay.json
python -m pkm.rl.play --p0 neural --p1 random

# watch the searcher play
python -m pkm.rl.play --p0 mcts --p1 neural --html mcts_vs_neural.html

# head-to-head win rate, no replay files
python -m pkm.rl.play --p0 mcts --p1 neural --games 30
```

- **`result.html`** — self-contained match visualizer (the official cabt
  board UI); open it in any browser and step through the game. Requires
  `kaggle-environments>=1.30.2` (pinned in `pyproject.toml`; older versions
  don't bundle the visualizer assets).
- **`replay.json`** — the full kaggle-environments episode record: per-step
  observations, actions, rewards, statuses. Reload for analysis:

  ```python
  import json
  replay = json.load(open("replay.json"))
  replay["rewards"]          # final result, e.g. [1, -1]
  len(replay["steps"])       # decision count
  replay["steps"][k]         # both players' (observation, action, status) at step k
  ```

- Training rollouts themselves don't write replays (they'd dominate runtime);
  to inspect training behaviour, export the checkpoint and replay it via
  `pkm.rl.play`.

---

## 6. Current measured baselines (deck.csv mirror, CPU)

| matchup | result |
|---|---|
| PPO greedy (40 quick iters) vs random | 80–90% |
| MCTS 2×32 vs random | 95% |
| MCTS 3×64 vs PPO greedy | 63% (30 games) |

Next steps, in order of expected value: a long Phase 1 run (thousands of
games), then sustained Phase 2 iterations, then re-measure the search edge
and decide sims/dets against Kaggle's real per-move time budget.
