# deck_embedding — deck vectors & matchup model

Learn a vector for every Pokémon-TCG deck such that **distance ≈ matchup
behaviour** (not just "similar card list"), trained on real game outcomes
crawled/downloaded from the Kaggle *pokemon-tcg-ai-battle* competition.

This directory is **code only**. The data (parquet, checkpoints, shredded
databases) lives beside the crawl in the `pkm_data` repo — see
[Where everything is](#where-everything-is).

---

## TL;DR — the happy path

```bash
cd pkm/etc/deck_embedding                      # scripts self-anchor, but cwd is fine

# 0. what episode-days are published?
uv run --script daily.py list

# 1a. build deck vectors for a day (streaming; ~1 GB zip, no 21 GB unzip)
uv run --script daily.py build 2026-07-22                 # -> decks_with_outcomes.parquet
uv run --script daily.py build 2026-07-20 2026-07-21 2026-07-22 --append   # bigger set

# 1b. …or, if you also want the FULL game state (all tables), shred into a db
python -m replaydb daily ingest 2026-07-22 --db db_2026-07-22   # run from the kaggle dir
#     then the trainer can build decks_with_outcomes from that db automatically

# 2. train the two-tower matchup model
uv run --script train_deck_model.py train --epochs 200 --emb-dim 128

# 3. embed every deck with the trained checkpoint
uv run --script train_deck_model.py embed                 # -> deck_embeddings.parquet

# 4. explore / nearest-deck interactively
marimo edit db_explorer.py
```

Everything (data path, db path, checkpoints) is anchored to the repo via
`deck_data.py`, so the commands work from any working directory.

---

## Where everything is

**Code** (this repo, `pkm_new`, tracked): `pkm/etc/deck_embedding/`

| file | what it is |
|---|---|
| `deck_embed.py` | the model **library** — Set-Transformer two-tower + `build_vocab`, `train`, `embed_decks`, `nearest`, `load_matchups_from_parquet`. Importable; run standalone for a synthetic smoke test. |
| `deck_data.py` | data bridge + path anchors. `episode_deck_rows()` (one episode → rows), `build_decks_with_outcomes()` (from a shredded `db/`), `build_decks_with_outcomes_from_replays()` (from JSON files), `OUTCOMES_SCHEMA`. Defines `REPO_ROOT/KAGGLE_DIR/DB_DIR/DATA_DIR`. |
| `train_deck_model.py` | typer CLI: `train` and `embed`. |
| `daily.py` | typer CLI: `list` and `build` — stream the official daily datasets into `decks_with_outcomes.parquet`. Reuses `replaydb.daily`'s fetch primitives. |
| `db_explorer.py` | marimo notebook: browse decks/cards, train, and show nearest decks. |

**Shredder / crawl** (the `pkm_data` repo, at `kaggle_replays/pokemon-tcg-ai-battle/replaydb/`):

| file | what it is |
|---|---|
| `daily.py` | the `replaydb daily` subapp (`list`, `ingest`) + importable fetch primitives (`load_manifest`, `download_day_zip`, `iter_episodes_from_zip`). |
| `db.py` | `ReplayDB` — the Parquet-table database. `ReplayDB.ingest(episodes, …)` is the one shred-and-write loop shared by file-ingest, stream-ingest, and crawl. |
| `encode.py` / `decode.py` / `schema.py` | the lossless JSON⇄Parquet codec (`Shredder`). |
| `cli.py` / `crawl.py` | top-level CLI and the (now-superseded) leaderboard crawler. |

**Data** (`pkm_data` repo, **git-ignored** — regenerable, large):
`kaggle_replays/pokemon-tcg-ai-battle/`

| path | what it is |
|---|---|
| `deck_embedding/decks_with_outcomes.parquet` | **the deck vectors' training input** — long format `episode_id, player, card_id, count, won`. |
| `deck_embedding/deck_model.pt` | trained checkpoint (`state_dict` + `idx2id` vocab + `config`). |
| `deck_embedding/deck_embeddings.parquet` | `deck_key, embedding[]` — output of `embed`. |
| `deck_embedding/decks_with_outcomes.crawl_5200ep.parquet` | backup of the earlier crawl-derived set (5,200 episodes). |
| `db/` | the original crawl database (~5,218 episodes). |
| `db_2026-06-16/`, `db_2026-07-22/` | per-day lossless shreds (all 12 tables). |
| `episodes_daily/` | scratch for downloaded day zips (auto-cleaned). |

---

## How it works

### 1. Data source — no more crawling

Kaggle publishes the competition's episodes as **one dataset per day**
(`kaggle/pokemon-tcg-ai-battle-episodes-<date>`), indexed by a tiny manifest
(`…-episodes-index`, one row/day: date, slug, episode_count, bytes, scores).
That replaces the reverse-engineered leaderboard crawler.

The snag: a day is **~5–8k episodes / ~21 GB unzipped**, but only **~1 GB
compressed**. So we **stream**: download the compressed zip (one bounded file),
then iterate its members with `zipfile` and parse each episode with `orjson`
**one at a time**. The 21 GB unzipped form never lands on disk — peak disk is
the zip plus whatever we're writing. Two consumers share these primitives:

- `replaydb daily ingest` → shreds each episode into the **full** lossless db
  (12 tables) via `ReplayDB.ingest`.
- `deck_embedding/daily.py build` → keeps only **decks + outcomes** via
  `deck_data.episode_deck_rows`.

Both were verified **byte-identical** to the original file-based
`replaydb ingest` across all 12 tables.

### 2. Decks + outcomes

A registered deck is `steps[1][player].action` — a 60-card list. We count copies
→ `(card_id, count)`, join the terminal `rewards` → `won`, and keep only
decisive games where both players registered a valid 60-card deck. Result is the
long-format `decks_with_outcomes.parquet`. (`deck_data` can produce this three
ways — from a streamed zip, from a directory of JSONs, or from an already-shredded
`db/` — and all agree.)

### 3. The model (`deck_embed.py`)

A deck is a **multiset**: permutation-invariant, variable size, meaning comes
from card–card synergy. That's a **Set Transformer** (Lee et al. 2019):

```
cards (id+count embeddings)
   → ISAB blocks         # self-attention among cards = synergy, O(n·m) via inducing points
   → PMA pooling         # attention pooling to ONE deck vector
   → deck embedding e
```

Two decks are compared by a **two-tower** head with an **antisymmetric** logit,
so `P(A beats B) = 1 − P(B beats A)` by construction:

```
logit(A>B) = strength(eₐ) − strength(e_b)          # Bradley–Terry / Elo part
           + eₐᵀ W e_b − e_bᵀ W eₐ                  # learned rock-paper-scissors interaction
```

`W` starts at zero, so training *begins* as a pure strength model and learns
matchup interaction on top. Loss is BCE on real outcomes (each game added in
both orientations). Because the objective is *who beats whom*, embedding
distance means **matchup similarity**, which is what makes clustering meaningful.

### 4. Embed & explore

`train_deck_model.py embed` runs every deck through the trained encoder →
`deck_embeddings.parquet`. `deck_embed.nearest()` does top-k cosine/euclidean
neighbours; `db_explorer.py` wires that into a marimo UI.

---

## Common tasks

**Bigger training set (fix overfitting on one small day):**
```bash
uv run --script daily.py build 2026-07-18 2026-07-19 2026-07-20 2026-07-21 2026-07-22 --append
uv run --script train_deck_model.py train --epochs 200 --emb-dim 128
```

**Build deck vectors from a shredded db instead of re-downloading:**
```python
import deck_data as dd
dd.build_decks_with_outcomes(db="…/db_2026-07-22",
                             out=dd.DATA_DIR / "decks_with_outcomes.parquet")
```
(Or just point `train --db …/db_2026-07-22`; it builds the parquet if missing.)

**Cluster decks** (not yet a command — use the embeddings parquet):
```python
import polars as pl, numpy as np
from sklearn.cluster import KMeans
df = pl.read_parquet(".../deck_embedding/deck_embeddings.parquet")
X = np.array(df["embedding"].to_list())
labels = KMeans(n_clusters=12, n_init="auto").fit_predict(X)   # -> archetypes
```

---

## What can be improved (roadmap)

1. **Resume / continue training** *(not built yet)* — `train` starts from
   scratch. Add `--init-from CKPT` that loads `state_dict` (and reuses the saved
   `idx2id` vocab) before the optimiser loop; keep an epoch counter in the
   checkpoint. `_load_checkpoint` in `train_deck_model.py` is 90 % of the code.
2. **A `cluster` command** — wrap KMeans/HDBSCAN over `deck_embeddings.parquet`,
   emit `deck_key → cluster` + medoid decks per cluster (auto-named archetypes).
3. **More data by default** — a single day overfits (train acc ≫ val acc). Add a
   `daily.py build --last N` that walks the manifest and appends the N most
   recent days; consider dedup of identical deck lists across days.
4. **Regularisation** — no dropout / weight decay today. Add both; the val curve
   suggests it would help materially on small sets.
5. **Pure-stream download** — we drop the compressed zip on disk (bounded). A
   forward-only `stream-unzip` over the HTTP body would avoid even that, at the
   cost of a dependency.
6. **Shred speed** — the bottleneck is the ~8 M-row/day Python shred loop, not
   JSON parsing. Batched Arrow builders or multiprocessing over zip-member ranges
   would speed `replaydb daily ingest`; `orjson` already handles the parse side.
7. **Richer deck features** — feed card *text*/type embeddings into the card
   embedding table so unseen cards aren't cold-start zeros.

---

## Status

- Streaming download + shred + build: **done and verified** (byte-identical to
  file ingest, all 12 tables).
- Trained model on the 2026-07-22 day — see the **Training log** below.

### GPU on NixOS (important)

The uv/pip `torch` wheel bundles its own CUDA runtime but can't find the NVIDIA
**driver** stub `libcuda.so.1`, which NixOS keeps in `/run/opengl-driver/lib`
(not `/usr/lib`). Without help, `torch.cuda.is_available()` is False and training
silently runs on CPU. **Fixed in the flake** — `flake.nix`'s `devShell` now appends
`/run/opengl-driver/lib` to `LD_LIBRARY_PATH`, so `use flake` + direnv exposes
`libcuda.so.1` in every shell and `--device cuda` just works (run `direnv reload`
once after pulling). If CUDA still reports unavailable, check
`CUDA_VISIBLE_DEVICES` isn't set to an empty string (that hides all GPUs).

Alternatives if you don't use the flake: `export
LD_LIBRARY_PATH=/run/opengl-driver/lib` per-shell; `programs.nix-ld.enable =
true` (system-wide, cleanest); `nixGL`; or an FHS/`cudaSupport` nix env.

### Training log

**How the current model was trained** (2026-07-22, RTX 3090):

```bash
export LD_LIBRARY_PATH=/run/opengl-driver/lib; unset CUDA_VISIBLE_DEVICES
uv run --script train_deck_model.py train \
  --data …/deck_embedding/decks_with_outcomes.parquet \
  --epochs 200 --emb-dim 64 --dim 64 --heads 4 --n-blocks 2 --m 16 \
  --batch-size 256 --lr 1e-3 --val-frac 0.1 --seed 0 --device cuda \
  --out …/deck_embedding/deck_model_2026-07-22.pt
# then embeddings:
uv run --script train_deck_model.py embed \
  --checkpoint …/deck_model_2026-07-22.pt --out …/deck_embeddings.parquet --device cuda
```

| date | data | episodes / decks | config | result | checkpoint |
|---|---|---|---|---|---|
| 2026-06-16 | 06-16 | 1,274 / 2,548 | emb64 dim64 h4 b2, 128 ep | overfit — val_acc peaked 0.669 (tiny set) | superseded |
| 2026-07-22 (a) | 07-22 | 4,633 / 9,266 | **emb128 dim128 h8 b3**, 200 ep, lr 1e-3 | **collapsed** — loss stuck at ln2 (0.693), val≈0.50. Net too deep/wide for the LR (unstable). Discarded. | — |
| 2026-07-22 (b) | 07-22 | 4,633 / 9,266 | emb64 dim64 h4 b2, 200 ep, lr 1e-3 | learned: train acc 0.63, val ~0.59 (best 0.607) | superseded by (c) |
| 2026-07-22 (c) | 07-22 | 4,633 / 9,266 | same, vectorised pipeline | **train acc 0.647, val ~0.60 (best 0.634)** · 200 ep in **178 s** (was ~10 min) on RTX 3090 | `deck_model_2026-07-22.pt` ✅ |

**Artifacts** (in `…/pokemon-tcg-ai-battle/deck_embedding/`):
`deck_model_2026-07-22.pt` (weights, 668 KB) · `deck_embeddings.parquet`
(9,266 × 64-d) · `decks_with_outcomes.parquet` (training input).

**Interpretation:** deck-list alone is only a *weak* predictor of match outcome
(val ~0.59 vs 0.50 chance) — play skill and RNG dominate. So treat the embedding
as a **deck-similarity / clustering** space (which it captures), not a strong
matchup oracle. Bigger lever for accuracy: richer card features (roadmap #7) and
a lower-LR larger net (the collapse shows LR must drop with depth).
