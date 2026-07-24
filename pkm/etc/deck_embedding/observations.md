# Deck / card embeddings — observations & standing workflow

> **When you (human or Claude) work on card/deck embeddings, read this first and
> follow the workflow at the bottom.** It records which model, how the clustering
> works, what we observed, and the exact commands. Companions: `README.md`
> (usage reference), `NOTES.md` (session narrative). Keep this file current.

## What this is

A Set-Transformer two-tower model that learns a vector per Pokémon-TCG **deck**
so that distance ≈ matchup behaviour, trained on real competition games. Used
for archetype discovery (clustering), deck similarity, matchup estimates, and as
features for the RL agent. Full motive in `NOTES.md`.

## Data

Kaggle publishes episodes as **one dataset per day**, indexed by a manifest
(`kaggle/pokemon-tcg-ai-battle-episodes-index`). As of 2026-07-24 the manifest
lists **37 days · 191,821 episodes** (~723 GiB *unzipped*, but the compressed
download is far smaller and we never unzip — see the sync tool below).

| parquet (in `deck_embedding/`) | deck-instances | **unique decklists** | note |
|---|--:|--:|---|
| `decks_with_outcomes.parquet` | 9,266 | **66** | one day (2026-07-22), one meta |
| `decks_with_outcomes.crawl_5200ep.parquet` | 10,400 | **482** | older leaderboard crawl, broader field |

The raw field is ~99% duplicate re-submissions — **always dedup to unique
decklists** before clustering/plotting.

## Models trained (2026-07-22 data)

| checkpoint | dim / heads / blocks / emb | params | best val_acc | embeddings file |
|---|---|--:|--:|---|
| `deck_model_2026-07-22.pt` (default) | 64 / 4 / 2 / **64** | ~160K | **0.634** | `deck_embeddings.parquet` |
| `deck_model_2026-07-22_emb16.pt` (bigger encoder) | 128 / 8 / 3 / **16** | ~769K | 0.613 | `deck_embeddings_emb16.parquet` |

Config lives inside each `.pt` (`ckpt["config"]`). Clustering reads the
**embeddings parquet**, never the `.pt` directly.

## How the clustering works (`deck_cluster.py`)

Auto-`k`, no manual choice:

1. L2-normalise every deck vector → **cosine space** (euclidean on unit vectors
   is monotonic in cosine distance).
2. For `k = 2..k_max`: k-means (k-means++ init, 8 restarts, keep lowest inertia).
3. Score each partition by **mean silhouette** `s = (b−a)/max(a,b)` where `a` =
   mean distance to own cluster, `b` = mean distance to nearest other cluster.
4. Keep the `k` with the highest silhouette.
5. **Medoid** per cluster = deck closest to the cluster's mean direction (its
   representative decklist). **Labels** = each cluster's headline **Pokémon** —
   highest-marquee Pokémon present (`mega_ex>tera>ex>stage2`, hp tie-break, from
   `cards.json` metadata), tie-broken by in-cluster frequency (`_card_marquee`).
   No TF-IDF anywhere — deck *similarity* should be Jaccard/overlap or
   co-occurrence, not edit distance.

**Win-rate** is empirical, not predicted: from `decks_with_outcomes.won` (engine
terminal reward; decisive games only, exactly one winner), aggregated
games-weighted per cluster. Because every game has one winner the field averages
to ~0.50 by construction — only sizeable swings from 0.5 are meaningful.

## Observations

- **16-d clusters better than 64-d** despite lower val-acc. 64-d → **k=3,
  silhouette 0.18** (mushy, one big blob). 16-d → **k=11, silhouette 0.20** with
  legible archetypes: Mega Lopunny ex, Marnie's Grimmsnarl ex control, Team
  Rocket's Wobbuffet, Mega Kangaskhan ex, and weak decks like Xerosic's
  Machinations control (win-rate 0.23) and a Gravity Mountain deck (0.29). The
  tighter bottleneck forces archetype identity into fewer directions → cleaner
  cosine geometry. For clustering/viz prefer the **16-d** embeddings; for raw
  matchup prediction the 64-d is marginally better.
- **Accuracy is signal-capped (~0.60–0.63), not capacity-capped.** 160K and 769K
  nets land in the same band because deck-list → win/loss is only weakly
  predictive — pilot skill + RNG dominate. Bigger models did not help.
- **One day = one narrow meta.** Silhouette stays ~0.18 flat on the 64-d/one-day
  data — the honest signal that the field is basically one archetype family. The
  real lever is **more days** (see sync tool) so more archetypes appear.
- **Embedding is 64/16-d, not < 60**, because it's a *learned function of the
  multiset* (permutation-invariant, fixed-width), not a compression of the
  60-card list. Size is a representation choice, unrelated to deck length.

## Player identity

Episodes carry **display names / usernames** (`info.TeamNames`,
`info.Agents[].Name`), stored by the shredder in the `games` table as
`p0_name` / `p1_name` (+ `p0_thumb`/`p1_thumb`) — e.g. `Tony Li`, `Luca`,
`懒惰的金枪鱼`, `Team KASA.` (142 unique in 2026-07-22). **We keep the
usernames as they arrive; there is no numeric player/team ID in the episode
data** (teamId/submissionId only existed in the old leaderboard-crawl layer).
Names are **not** in `decks_with_outcomes.parquet` — join on `episode_id`
(player 0→`p0_name`, 1→`p1_name`) if you want per-pilot analysis. Treat names as
weak identifiers (they can change/collide) and as public-but-personal data — do
not paste them into external services.

## Retrain from scratch vs update?

**Retrain from scratch when the corpus grows.** The card **vocabulary is derived
from the data** (`DeckVocab` over the card set), so more days → new cards → a
larger embedding table → the old checkpoint's shapes no longer match and can't be
loaded anyway. The model is also tiny (~3 min on GPU), and continue-training
(`--init-from`) isn't built. So: pull all data, rebuild `decks_with_outcomes`,
train fresh. (If the vocab happens not to grow, warm-starting would be possible
once `--init-from` exists — roadmap in README.)

## Tools (all in `pkm/etc/deck_embedding/` unless noted)

| tool | what |
|---|---|
| `deck_data.py` | JSON→rows extractor + `decks_with_outcomes` builders (from db / JSON dir / zip stream) |
| `deck_embed.py` | the model (Set-Transformer encoder + two-tower antisymmetric matchup), vectorised dataloader |
| `train_deck_model.py` | `train` / `embed` CLI (net size, GPU, workers all flagged) |
| `daily.py` | `list` / `build` — download → **disk-extract → parallel-parse across cores** → `decks_with_outcomes*.parquet`; resumable (sidecar `.days.json`), reuses cached zips, `build all` for every day |
| **`deck_cluster.py`** | **auto-`k` clustering → `deck_clusters.parquet` + labelled summary** |
| **`cluster_viz.py`** | **marimo + plotly explorer: auto-k, PCA scatter, per-cluster deck gallery with card art** |
| `replaydb daily sync` (pkm_data) | download+shred **every** published day into one db, incrementally (done-marker, throttle, cleanup) |
| `replay/fetch_card_images.py` (repo root) | download card face PNGs → `replay/card_images/<id>.png` |

## Standing workflow

```bash
# 0. from the deck_embedding dir; GPU works via flake.nix (direnv), --device cuda
cd pkm/etc/deck_embedding
VENV=/home/df/projects/zeke/pkm_new/.venv/bin/python   # reuse project venv; DO NOT add packages

# 1. DATA — pull deck+outcome rows for EVERY published day (fast path for training).
#    Disk-extract + PARALLEL parse across cores (NOT streaming — decompression is
#    single-threaded). Resumable: sidecar `<out>.days.json` skips done days, reuses
#    cached zips, cleans up per day. `all` = every published day.
$VENV daily.py build all --workers 21 --out .../decks_with_outcomes_all.parquet
#   …one day:  $VENV daily.py build 2026-07-22
#   full lossless 12-table db instead of just decks:  $VENV -m replaydb daily ingest <days> --db db_daily

# 2. BUILD the deck→outcome table from the db (rebuild after new data)
cd ../../../pkm/etc/deck_embedding
$VENV deck_data.py            # or train_deck_model.py builds it on demand

# 3. TRAIN fresh (from scratch — vocab may grow). ALWAYS launch training in tmux.
$VENV train_deck_model.py train --epochs 200 --device cuda                       # 64-d default
$VENV train_deck_model.py train --emb-dim 16 --dim 128 --heads 8 --n-blocks 3 \
      --lr 3e-4 --epochs 200 --device cuda --out .../deck_model_emb16.pt          # 16-d

# 4. EMBED every deck with the checkpoint
$VENV train_deck_model.py embed --checkpoint <ckpt.pt> --out <deck_embeddings*.parquet>

# 5. CARD ART (once) — fetch images for every card id used
IDS=$($VENV -c "import polars as pl,deck_data as dd;print(' '.join(map(str,sorted(pl.read_parquet(dd.DATA_DIR/'decks_with_outcomes.parquet')['card_id'].unique().to_list()))))")
$VENV ../../../replay/fetch_card_images.py --ids $IDS

# 6. CLUSTER (auto-k)
$VENV deck_cluster.py                                            # 64-d, auto-k
$VENV deck_cluster.py --embeddings <...emb16.parquet> --out <...clusters_emb16.parquet>

# 7. VISUALISE — reuse the project venv, no sandbox
$VENV -m marimo edit cluster_viz.py --no-sandbox                # http://localhost:2718
```

**Decks are always SORTED by `card_id`** — in the dataset parquet (every builder
`.sort(...)`s, `episode_deck_rows` emits `sorted(cnt.items())`) AND going into
training (`load_matchups_from_parquet` sorts the frame + re-sorts each deck).
The Set-Transformer is permutation-invariant so this doesn't change results, but
sorted input is required for determinism/reproducibility — keep it that way.

**Always:** dedup to unique decklists before clustering; remember win-rate is
empirical and ~0.5-centred; record any model/flag change in
`docs/model_configurations.md` and each training command in
`pkm/new_agents/train_cmd_log.md`; keep this file + `README.md` + `NOTES.md`
current.
