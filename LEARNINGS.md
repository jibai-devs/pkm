# LEARNINGS.md

## Architecture Decisions (2026-07-05)

### Agent Profile Design
- Agent profiles are named directories under `agents/<name>/` that bundle deck + strategy + checkpoints + results
- Profile YAML stores: strategy type, deck path, creation time, notes
- Checkpoints named `<profile_name>_MM_DD_HH_MM_SS.pt` with `latest.pt` symlink
- Built-in agent names (`random`, `neural`, `mcts`) still work as legacy/defaults
- Play resolves agent name: check profiles first, fall back to built-in names
- Per-player deck support needed — currently both sides always share the same deck

### CLI Design
- Migrating from argparse to typer (already in pyproject.toml dependencies)
- Rich output for progress, tables, colored status
- Subcommands: `pkm play`, `pkm train`, `pkm exit-train`, `pkm export`, `pkm profile`
- Typer supports `--p0-weights` / `--p0-deck` style flags for overrides

## Environment & Config
- `typer>=0.26.8` already in pyproject.toml dependencies
- `rice>=0.4.0` in dependencies (terminal rendering library, not yet used much)
- Kaggle env needs `make("cabt", configuration={"decks": [deck, deck]})` — both sides get a deck list
- Agents must be plain functions `def agent(obs: dict) -> list[int]` for kaggle compatibility
- Kaggle may execute an agent with a working directory other than `/kaggle_simulations/agent`; bundled resources must be resolved relative to `__file__`, not only the process working directory
- Kaggle simulation submission runs `main.py` as `__main__`; a guarded local smoke-test call can execute instead of the callable `main(obs)` and must not load files absent from the bundle

## Codebase Conventions
- Agent factories: `make_<type>_agent(deck, **kwargs)` pattern
- `pkm/agents/base.py:make_agent(deck, strategy_fn)` is the base factory (closure-based)
- Checkpoints are `torch.save(model.state_dict(), path)` — state dicts only
- Metrics use `csv.DictWriter` with `writeheader()` + per-row `writerow()` + `flush()`
- Justfile is the primary task runner

## RL Training Insights (2026-07-16)

### Reward signal
- Reward is sparse binary: +1 (win), -1 (loss), 0 (draw). No intermediate rewards.
- Credit assignment via GAE + potential-based shaping on prize differential (`shaping_coef=0.2`).
- Early game moves get weak signal (value head prediction only); late game moves get strong signal.
- Shaping is mathematically optimal-policy-preserving (potential-based theorem).

### PPO vs Expert Iteration
- PPO (Phase 1): network picks moves directly, 3 epochs of mini-batch SGD (256) per rollout, data discarded after. Opponent pool (size 8) prevents cycling.
- Expert Iteration (Phase 2, `exit_train.py`): MCTS picks moves (32 sims, 2 determinizations), network trained toward MCTS visit distributions + game outcomes.
- PPO's mini-batch loop IS a form of replay (reuse rollout for K epochs), but buffer is tiny and short-lived.

### MCTS (search.py)
- Does NOT try every action. Fixed budget of simulations guided by network priors.
- PUCT selection balances exploitation (q) and exploration (prior-based bonus).
- Handles imperfect information via determinization: sample hidden info, search in each sample, aggregate visit counts.
- `sample_determinization` in `determinize.py` creates plausible world states from deck knowledge.

### Network architecture (model.py)
- Uses V(s) (state value), NOT Q(s,a) (action-value). One scalar output per state.
- Policy head: scores each option against state embedding h via 2-layer MLP, softmax over (options + STOP).
- Value head: V(s) = tanh(MLP(h)), predicts expected final outcome from state alone, not conditioned on action.
- V(s) used as baseline in PPO (advantage = actual - predicted) and leaf eval in MCTS.
- Why not Q(s,a): variable-length action spaces make Q expensive; V + policy scoring is the AlphaZero approach.

### Experience replay
- Full replay buffer is hard to add to PPO (stale logprobs break importance ratio).
- Works well with expert iteration (MCTS targets are supervision, no importance ratio needed).
- Prioritized Experience Replay (PER) weights by TD-error for better sample efficiency.

### Offline RL from replay logs
- Replay logs have full obs + legal options + game history, but NOT old logprobs.
- Can reconstruct actions from `logs` array in the observation.
- Behavior cloning (supervised `(obs, action)` pairs) is simplest warm-start.
- CQL/IQL handle mixed-quality data from multiple agents without needing logprobs.
- Best approach: parse replays -> supervised pretrain -> fine-tune with self-play.

### Interactive (human-in-the-loop) training
- Conceptually sound but impractical: slow data rate, skill ceiling, inconsistency.
- Better to parse existing replay logs than train live.
- Useful for: reward shaping hints, curriculum design, debugging.

### Key improvements to try (priority order)
1. Parse replays -> supervised pretrain (warm start)
2. More games per iteration (8 -> 32)
3. Exit_train on replay data (MCTS re-evaluates, targets stay fresh)
4. Bigger network + attention over board
5. Experience replay buffer for exit_train
6. Offline RL (CQL/IQL) pretrain
7. MCTS at inference time
8. Distributed self-play

### Training flow (PPO)
- Delta (TD error) = r_t + gamma * V(s_{t+1}) - V(s_t). Measures "was outcome better or worse than predicted?"
- GAE accumulates delta backward with decay: gae_t = delta_t + gamma * lambda * gae_{t+1}
- Advantage = GAE (positive → reinforce action, negative → discourage)
- Return target = gae + V(s) (what value head should have predicted)
- Policy loss: clipped surrogate (ratio * advantage, clamped to prevent large updates)
- Value loss: MSE(predicted value, return target)
- Entropy bonus: keeps exploration alive, prevents policy collapse
- All three losses share gradients through the state encoder (card_emb, state_fc1/2)

### Modularity / swappability
- Encoder ↔ Model: tightly coupled (numpy array shapes must match embedding dims)
- Model ↔ PPO: cleanly coupled via model.evaluate() → (logprobs, entropies, values)
- PPO ↔ EncodedDecision: moderately coupled (reads .value, .potential, writes .advantage, .ret)
- Easiest to swap: value head (subclass), reward shaping (change prize_potential), PPO algo (new update fn)
- Hardest to swap: option encoding format (changes both encoder and model input layer)
- Key refactoring: extract encode_state/encode_options from model into standalone, define ModelProtocol, make trainer accept model factory + update fn
