# deck_embedding — session notes, motive & how to continue

Companion to `README.md` (which is the usage reference). This file is the
narrative: **why** this exists, **what** was done, and **where to go next**.

## Motive

Goal: learn a vector for every Pokémon-TCG deck such that **distance ≈ matchup
behaviour**, from real competition games. Uses:
- **archetype discovery** — cluster the deck space into playstyles/metagame map;
- **deck similarity** — "decks that play like this one";
- **matchup estimates** — P(deck A beats deck B);
- **features** for the RL agent / analysis (opponent-deck context, win-rate models).

Why a **Set Transformer two-tower**: a deck is a *multiset* of cards
(permutation-invariant, variable size, meaning from card–card synergy). A Set
Transformer models exactly that; the two-tower head with an **antisymmetric**
matchup logit guarantees `P(A>B) = 1 − P(B>A)` and starts as a pure
Elo/Bradley-Terry strength model, learning rock-paper-scissors interaction on
top. So the embedding means "similar matchup behaviour", not just "similar list".

## What was done this session

1. **Moved the code** into `pkm/etc/deck_embedding/` (was loose in the kaggle
   crawl dir). All paths now anchor to `__file__` (`deck_data.DATA_DIR` etc.), so
   scripts run from any cwd. Data stays in
   `pkm_data/kaggle_replays/pokemon-tcg-ai-battle/deck_embedding/`.

2. **No-crawl data pipeline.** Kaggle now publishes episodes as one dataset per
   day (`…-episodes-<date>`), indexed by a manifest (`…-episodes-index`). Built:
   - `daily.py` (this dir): `list` + `build` — **stream** a day's compressed zip
     (~1 GB) member-by-member with orjson into `decks_with_outcomes.parquet`,
     never unzipping the ~21 GB day.
   - **Extended `replaydb`** (the pkm_data repo): `ReplayDB.ingest(episodes,…)`
     — one shred-and-write loop shared by file-ingest, stream-ingest and crawl —
     plus a `replaydb daily` subapp (`list`/`ingest`) that streams a day into the
     full 12-table lossless db. Verified byte-identical to file ingest.
   - `deck_data.episode_deck_rows()` is the single JSON→rows extractor; three
     builders (from zip stream, from a JSON dir, from a shredded db) all agree.

3. **Data for 2026-07-22:** shredded to `db_2026-07-22/` (4,639 episodes) and
   built `decks_with_outcomes.parquet` (**4,633 decisive episodes, 9,266 decks,
   208 card types**). NB: the raw day is ~21 GB of JSON; parallel-unzip with
   `xargs -P $(nproc) unzip` did it in ~14 s.

4. **NixOS GPU fix.** The uv/pip torch wheel couldn't find the driver's
   `libcuda.so.1` (NixOS puts it in `/run/opengl-driver/lib`), so CUDA silently
   fell back to CPU. Fixed **in `flake.nix`** (devShell `LD_LIBRARY_PATH` now
   includes it) → direnv exposes it, `--device cuda` just works.

5. **Faster data pipeline.** Vectorised `collate` with `pad_sequence` over
   pre-tensorised decks (+ optional `num_workers`, pinned non-blocking H2D). The
   old per-element Python collate starved the GPU; ~**3–4× faster** (200 epochs:
   ~10 min → ~3 min on an RTX 3090). NB the model is tiny, so the *data* loop was
   the bottleneck, not GPU compute.

6. **Trained two models** (see README training log). Both plateau around
   **val ≈ 0.60–0.63** — see "the ceiling" below.

7. **Clustering + viz.** `cluster_viz.py` (marimo + plotly): 2D PCA + numpy
   KMeans over the **unique** decklists, sized by popularity, coloured by
   cluster/win-rate, with a file-picker to compare 64-d vs 16-d embeddings.

## Results

| model | params | emb | train acc | best val_acc |
|---|---|---|---|---|
| default | 160K | 64 | 0.647 | **0.634** |
| bigger encoder | 769K | 16 | 0.646 | 0.613 |

## Key finding — the ceiling (read before "improving")

- **The 2026-07-22 field is ~66 unique decklists**, one dominant archetype family
  (shared core cards `1086/1152/1182/1227/1097/1079`) + a few outliers. So
  clustering shows weak structure (silhouette ~0.2; HDBSCAN → ~2 dense groups).
- **Accuracy is capped by signal, not capacity.** A 160K and a 769K net both land
  ~0.60–0.63 val, because deck-list → win/loss is only weakly predictive (play
  skill + RNG dominate). Making the model bigger did **not** help.
- Embedding *size* (64 vs 16) is a representation choice: 16-d is tighter/cheaper
  for clustering & plots at a small val cost.

## How to continue (highest-leverage first)

1. **More data (the real lever).** One day = one narrow meta. Pull several and
   append, then retrain — expect more archetypes and a firmer signal:
   ```bash
   uv run --script daily.py build 2026-07-18 2026-07-19 2026-07-20 2026-07-21 2026-07-22 --append
   uv run --script train_deck_model.py train --epochs 200 --device cuda   # defaults = the proven config
   ```
2. **Richer card features** (roadmap #7 in README): seed `card_emb` from card
   text/type so unseen cards aren't cold-start zeros and synergy is learnable
   beyond co-occurrence. Most likely to raise the ceiling.
3. **Resume/continue training** (`--init-from`, not built yet) — README roadmap #1.
4. **A `cluster` command** wrapping the notebook's KMeans + medoid naming.
5. Lower-LR larger nets are stable now (emb16 big ran fine at lr 3e-4) — but only
   worth it once (1)/(2) lift the signal.

## Reproduce / inspect

- Everything is CLI-parameterized — no code edits to try a new config.
- Training numbers, exact commands, weight shapes, GPU-on-NixOS notes, and the
  file/data map are all in `README.md`.
- Viz: `marimo edit cluster_viz.py` (runs from the project `.venv`).
