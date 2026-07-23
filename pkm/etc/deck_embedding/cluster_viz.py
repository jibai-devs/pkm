# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "polars",
#     "numpy",
#     "plotly",
# ]
# ///
"""Interactive deck-embedding explorer — project the learned deck vectors to 2D
and cluster them, with plotly.

    marimo edit cluster_viz.py          # (or launched for you)

Reads `deck_embeddings.parquet` + `decks_with_outcomes.parquet` from
`deck_data.DATA_DIR`. Points are UNIQUE decklists (the raw set is ~99% duplicate
re-submissions), sized by how many games they were played in and coloured by
cluster / win-rate.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import marimo as mo
    import numpy as np
    import plotly.express as px
    import polars as pl

    import deck_data as dd
    return dd, mo, np, pl, px


@app.cell
def _(dd, mo):
    import glob

    # pick which embeddings to explore (e.g. 64-d vs 16-d)
    _files = sorted(p.rsplit("/", 1)[-1] for p in
                    glob.glob(str(dd.DATA_DIR / "deck_embeddings*.parquet")))
    emb_file = mo.ui.dropdown(_files, value=_files[0], label="embeddings file")
    emb_file
    return (emb_file,)


@app.cell
def _(dd, emb_file, mo, pl):
    # Per (episode, player): the sorted card list + that game's result.
    per_deck = (
        pl.read_parquet(dd.DATA_DIR / "decks_with_outcomes.parquet")
        .group_by("episode_id", "player")
        .agg(pl.col("card_id").sort().alias("cards"), pl.col("won").first().alias("won"))
        .with_columns((pl.col("episode_id") * 10 + pl.col("player")).alias("deck_key"))
        .with_columns(pl.col("cards").cast(pl.List(pl.Utf8)).list.join(",").alias("sig"))
    )
    emb = pl.read_parquet(dd.DATA_DIR / emb_file.value)
    m = emb.join(per_deck, on="deck_key", how="inner")

    # Collapse to unique decklists: one embedding, popularity, win-rate, top cards.
    agg = (
        m.group_by("sig")
        .agg(
            pl.col("embedding").first(),
            pl.len().alias("games"),
            pl.col("won").mean().alias("winrate"),
            pl.col("cards").first(),
            pl.col("deck_key").first(),
        )
        .with_columns(pl.col("cards").list.head(8).cast(pl.List(pl.Utf8))
                      .list.join(", ").alias("top_cards"))
    )
    mo.md(f"**{len(agg)} unique decklists** from **{len(m):,}** game-instances "
          f"(the day's field is heavily duplicated).")
    return (agg,)


@app.cell
def _(mo):
    k = mo.ui.slider(2, 12, value=3, label="clusters (k)")
    color = mo.ui.dropdown(["cluster", "winrate", "games"], value="cluster", label="colour by")
    mo.hstack([k, color], justify="start", gap=1)
    return color, k


@app.cell
def _(np):
    # PCA + KMeans in plain numpy (keeps the venv dependency-free — no sklearn).
    def pca2(Xn):
        Xc = Xn - Xn.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T

    def kmeans(X, k, iters=100, seed=0):
        rng = np.random.default_rng(seed)
        C = X[rng.choice(len(X), k, replace=False)].copy()
        lab = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        return lab
    return kmeans, pca2


@app.cell
def _(agg, color, k, kmeans, mo, np, pca2, pl, px):
    X = np.array(agg["embedding"].to_list(), dtype="float32")
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # cosine space

    pts = pca2(Xn)
    labels = kmeans(Xn, k.value)
    df = agg.with_columns(
        pl.Series("x", pts[:, 0]), pl.Series("y", pts[:, 1]),
        pl.Series("cluster", [str(int(v)) for v in labels]),
    )
    # plotly express takes a plain dict of columns — avoids pandas/pyarrow.
    data = {c: df[c].to_list() for c in
            ("x", "y", "games", "winrate", "cluster", "top_cards", "deck_key")}
    fig = px.scatter(
        data, x="x", y="y",
        color="cluster" if color.value == "cluster" else color.value,
        size="games", size_max=45, hover_name="deck_key",
        hover_data={"games": True, "winrate": ":.2f", "top_cards": True, "x": False, "y": False},
        color_continuous_scale="Viridis",
        title=f"{len(df)} unique decks · PCA · k={k.value} · colour={color.value}",
    )
    fig.update_layout(height=640)
    mo.ui.plotly(fig)
    return (df,)


@app.cell
def _(df, mo, pl):
    # Per-cluster summary: size, total games, mean win-rate.
    summary = (
        df.group_by("cluster")
        .agg(pl.len().alias("decks"), pl.col("games").sum().alias("games"),
             pl.col("winrate").mean().round(3).alias("mean_winrate"))
        .sort("games", descending=True)
    )
    mo.vstack([mo.md("### clusters"), mo.ui.table(summary, selection=None)])
    return


if __name__ == "__main__":
    app.run()
