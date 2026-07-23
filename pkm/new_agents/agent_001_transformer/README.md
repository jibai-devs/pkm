# agent_001_transformer

A from-scratch **AlphaZero-style transformer agent**, ported near-verbatim from a
Kaggle community notebook (`../agent_000_dragapult/references/example.py`) and
rewired to drive this repo's libcg seam instead of the Kaggle `cg-lib` dataset.

It is deliberately a *different architecture* from `agent_000_dragapult`, kept
around as a comparison point:

| | agent_000_dragapult | **agent_001_transformer** |
|---|---|---|
| encoder | structured per-entity attention + CLS | `EmbeddingBag` bag-of-features → TransformerEncoder |
| card identity | learned own-vocab **+** attribute channel | single shared sparse index |
| action head | marginal pointer, grounded on referenced entity | transformer **decoder** scoring enumerated option *combinations* |
| targets | visit-count policy + GAE / MC value | advantage-style policy + TD(λ) value |
| determinization | pluggable, samples own deck | opponent filled with Snorlax/energy placeholder |

## Layout

- `net.py` — model (`MyModel`/`DecoderLayer`), sparse featurizers, PUCT `mcts_agent`, `sample_deck`. Shared by train + inference.
- `train.py` — self-play + MCTS training loop; writes `{state_dict, dims}` checkpoints.
- `submit_main.py` — Kaggle entry point (`agent(obs) -> list[int]`), packed to `main.py`.
- `pack.py` — bundles `main.py` + `weights.pth` + `pkm/` into a `.tar.gz` (flattened for `/kaggle_simulations/agent/`).

## Run

Everything is self-contained in this directory. The two scripts encode the
"blessed" way (GPU env fix, cuda, logging, pack+submit):

```bash
# train on the GPU: iters=20, sims=10 (args override). Tees a log into ./logs/.
bash scripts/train.sh [ITERS] [SIMS]

# pack ./out/latest.pth into a Kaggle bundle and submit it
bash scripts/submit.sh "my submission message"
```

`scripts/train.sh` exports `LD_LIBRARY_PATH=/run/opengl-driver/lib` (the NixOS
libcuda path — without it torch silently falls back to CPU). Prefer running it
inside the shared `pkm-train` tmux session so a long run survives detach.

Underlying entry points, if you want to drive them directly:

```bash
python -m pkm.new_agents.agent_001_transformer.train \
    --iters 20 --eval-games 50 --selfplay-games 100 --sims 10 \
    --device cuda --out pkm/new_agents/agent_001_transformer/out
python -m pkm.new_agents.agent_001_transformer.pack \
    --checkpoint pkm/new_agents/agent_001_transformer/out/latest.pth
```

Checkpoints + bundles land under `./out/` (gitignored); run logs under `./logs/`
(tracked); leaderboard history in `./submission_log.md`. Bundle is ~46 MiB (well
under Kaggle's 197.7 MiB); like agent_000 it relies on **torch** being present
in the cabt sandbox (no torch is bundled).

## Notes / caveats

- **Inference uses MCTS** (`SEARCH_COUNT`, default 10) at every decision, so
  per-move latency scales with sims — tune down if you hit Kaggle's time limit.
- Value/eval win-rate is measured **vs random**, which saturates; a head-to-head
  eval (as agent_000 has) would be the honest ceiling test.
- The original notebook lives, unmodified, at
  `../agent_000_dragapult/references/example.py`; the locally-runnable copy is
  `example_local.py` (same file, imports rewired to `cabt`).
