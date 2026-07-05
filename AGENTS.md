# Agents

## Current Progress (as of Jul 2026)

| Phase | Status | Details |
|-------|--------|---------|
| Phase 1 — PPO self-play | **Done (200 iters)** | 80% win rate vs random. Checkpoint: `checkpoints/ppo_latest.pt` |
| Phase 2 — Expert iteration | **Started (1 run)** | MCTS self-play + distillation. Checkpoint: `checkpoints/exit_latest.pt` |
| Metrics & monitoring | **Done** | CSV logging + Plotly notebook |
| Kaggle submission | **Ready** | `just submit` exports weights + bundles |

### What's Working
- Pointer/scoring policy network handles variable-length action spaces
- PPO self-play with checkpoint pool opponent sampling
- Potential-based reward shaping (prize differential)
- IS-MCTS with determinization for imperfect information
- Expert iteration (MCTS targets -> network training)
- Numpy-only inference for Kaggle submission (no torch at eval time)
- CSV metric logging for all training runs

### What's Next
1. **Hyperparameter sweep** — LR, games/iter, pool size, eval frequency
2. **Longer exit training** — run 50+ iters of expert iteration from the PPO baseline
3. **MCTS vs neural eval** — measure if MCTS agent beats raw policy head-to-head
4. **Larger model** — wider MLP, more embedding dims, attention over options
5. **Multi-deck training** — sample opponent decks from a pool for robustness
6. **Submission** — `just submit` and check Kaggle leaderboard

## Build & Run
```bash
uv sync                    # install deps
python main.py             # run a battle
./submit.sh                # create Kaggle submission bundle
```

## Lint & Typecheck
```bash
ruff check .               # lint
ruff format .              # format
pytest tests/              # run tests
```

## Project Structure
- `pkm/data/card_data.py` — card/attack metadata from cabt C library
- `pkm/data/deck.py` — Deck class (CSV load/save, 60-card validation)
- `pkm/agents/base.py` — `make_agent(deck, strategy_fn)` factory
- `pkm/agents/random_agent.py` — random legal move agent
- `pkm/agents/neural_agent.py` — greedy trained-policy agent (numpy inference, no torch)
- `pkm/search.py` — correct bindings to the engine's SearchBegin/SearchStep API
- `pkm/rl/` — encoders, pointer-style policy/value net, PPO self-play, expert iteration
- `pkm/mcts/` — determinization + IS-MCTS over the search API
- `pkm/strategies/` — future strategy implementations
- `main.py` — battle runner entry point
- `deck.csv` — sample deck (60 card IDs, one per line)
- `submit.sh` — creates `submission.tar.gz` for Kaggle
- `docs/RL_PLAN.md` — RL self-play design (Phase 1 PPO, Phase 2 IS-MCTS/ExIt)

## RL Training
Prefer the `justfile` (run `just` to list recipes): `just train` / `just resume`
(Phase 1 PPO), `just exit-train` / `just exit-resume` (Phase 2), `just export`,
`just play mcts neural`, `just eval mcts neural 30`, `just submit`.
Underlying commands:
```bash
python -m pkm.rl.train --iterations 50 --games 16 [--init checkpoints/ppo_latest.pt]
python -m pkm.rl.exit_train --iterations 5 --games 8    # Phase 2: expert iteration (init from ppo_latest.pt)
python -m pkm.rl.export checkpoints/ppo_latest.pt pkm/policy.npz  # export for torch-free inference
python -m pkm.rl.play --p0 mcts --p1 neural             # replay -> result.html + replay.json
```
- Checkpoints land in `checkpoints/`; `pkm/policy.npz` is bundled in the submission (no torch needed at inference).
- `pkm/search.py` signatures were recovered from the official competition `cg/api.py` (SearchBegin needs `lib.AgentStart()` handle + the observation's `search_begin_input`, returns ApiResult JSON; search ids are int64).

## Metrics & Monitoring
Training logs are saved to CSV during training:
- `metrics/ppo_train.csv` — PPO self-play (iter, wins, losses, pi_loss, v_loss, entropy, clip_frac, eval_win_rate)
- `metrics/exit_train.csv` — expert iteration (iter, pi_loss, v_loss)

Run the Plotly notebook for interactive charts:
```bash
cd notebooks && jupyter notebook training_monitor.ipynb
# or: jupyter notebook notebooks/training_monitor.ipynb
```

## Custom Agents
Agents are plain functions with signature `def agent(obs: dict) -> list[int]`.
To add your own agent:
1. Create `pkm/agents/your_agent.py` with a `make_your_agent(deck, **kwargs)` factory
2. Add a branch in `pkm/rl/play.py:make_agent_by_name()`
3. Run: `just play your_agent neural` or `just eval your_agent neural 30`

The `make_agent(deck, strategy_fn)` base factory in `pkm/agents/base.py` handles deck submission boilerplate — your strategy_fn only needs to handle `obs["select"] is not None`.

## cabt Engine API
- `from kaggle_environments.envs.cabt.cg.sim import lib` → `lib.AllCard()`, `lib.AllAttack()`
- `from kaggle_environments.envs.cabt.cg.game import battle_start, battle_select, battle_finish`
- Agents must be plain functions (not class instances) for kaggle-env compatibility
- `obs["select"] is None` → return deck (60 card IDs)
- Otherwise return list of option indices from `obs["select"]["option"]`

## Kaggle Submission
- Bundle: `tar -czvf submission.tar.gz main.py deck.csv pkm/`
- Max size: 197.7 MiB
- Daily limit: 5 submissions
- Only latest 2 are active
- Files land in `/kaggle_simulations/agent/`
