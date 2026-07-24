# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "polars",
#     "numpy",
#     "plotly",
#     "typer",
#     "rich",
# ]
# ///
"""Interactive deck-embedding explorer — auto-cluster the learned deck vectors,
project them to 2D, and inspect real decks (with card art) per cluster.

    marimo edit cluster_viz.py

Reads `deck_embeddings*.parquet` + `decks_with_outcomes.parquet` from
`deck_data.DATA_DIR`. Points are UNIQUE decklists (the raw field is ~99%
duplicate re-submissions), sized by games played, coloured by cluster / win-rate.
Clustering (silhouette-swept auto-`k`, distinctive-card labels) is imported from
`deck_cluster.py` so the notebook and the CLI never drift. Card images come from
`replay/card_images/<id>.png` (run `replay/fetch_card_images.py` to populate).
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import marimo as mo
    import numpy as np
    import plotly.express as px
    import polars as pl

    import deck_cluster as dc
    import deck_data as dd

    IMG_DIR = dd.REPO_ROOT / "replay" / "card_images"
    NAMES = dc._card_names(dc._cards_json())
    return IMG_DIR, NAMES, dc, dd, glob, mo, np, pl, px


@app.cell
def _(dd, glob, mo):
    # pick which embeddings to explore; default to the 16-d model (clusters best).
    _files = sorted(p.rsplit("/", 1)[-1] for p in
                    glob.glob(str(dd.DATA_DIR / "deck_embeddings*.parquet")))
    _default = next((f for f in _files if "emb16" in f), _files[0])
    emb_file = mo.ui.dropdown(_files, value=_default, label="embeddings file")
    emb_file
    return (emb_file,)


@app.cell
def _(dd, emb_file, mo, pl):
    # Per (episode, player): sorted card list + result + per-card counts.
    _base = pl.read_parquet(dd.DATA_DIR / "decks_with_outcomes.parquet").with_columns(
        (pl.col("episode_id") * 10 + pl.col("player")).alias("deck_key")
    )
    per_deck = (
        _base.group_by("deck_key")
        .agg(
            pl.col("card_id").sort().alias("cards"),
            pl.col("won").first().alias("won"),
            pl.col("card_id").alias("ci"), pl.col("count").alias("cn"),
        )
        .with_columns(pl.col("cards").cast(pl.List(pl.Utf8)).list.join(",").alias("sig"))
    )
    # deck_key -> [(card_id, count), …] for the art gallery, SORTED BY card_id
    COUNTS = {r["deck_key"]: sorted(zip(r["ci"], r["cn"]), key=lambda t: t[0])
              for r in per_deck.select("deck_key", "ci", "cn").iter_rows(named=True)}

    emb = pl.read_parquet(dd.DATA_DIR / emb_file.value)
    m = emb.join(per_deck, on="deck_key", how="inner")
    agg = (
        m.group_by("sig")
        .agg(
            pl.col("embedding").first(),
            pl.len().alias("games"),
            pl.col("won").mean().alias("winrate"),
            pl.col("cards").first(),
            pl.col("deck_key").first(),
        )
        .sort("games", descending=True)
    )
    mo.md(f"**{len(agg)} unique decklists** from **{len(m):,}** game-instances "
          f"· embeddings: `{emb_file.value}`")
    return COUNTS, agg


@app.cell
def _(mo):
    auto = mo.ui.checkbox(value=True, label="auto-k (silhouette sweep)")
    k = mo.ui.slider(2, 14, value=3, label="k (when not auto)")
    color = mo.ui.dropdown(["cluster", "winrate", "games"], value="cluster", label="colour by")
    mo.hstack([auto, k, color], justify="start", gap=1)
    return auto, color, k


@app.cell
def _(NAMES, agg, auto, dc, k, np, pl):
    X = np.array(agg["embedding"].to_list(), dtype="float64")
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # cosine space
    sigs_cards = [[int(c) for c in row] for row in agg["cards"].to_list()]

    if auto.value:
        chosen, labels, curve = dc._auto_k(Xn, k_max=14, seed=0)
    else:
        labels, _inertia = dc._kmeans(Xn, k.value, seed=0)
        chosen = k.value
        curve = [(k.value, dc._silhouette(Xn, labels), 0.0)]
    sil = next(s for kk, s, _ in curve if kk == chosen)
    arche = dc._labels_per_cluster(sigs_cards, labels, NAMES)

    # 2D PCA projection for the scatter
    Xc = Xn - Xn.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    pts = Xc @ Vt[:2].T
    df = agg.with_columns(
        pl.Series("x", pts[:, 0]), pl.Series("y", pts[:, 1]),
        pl.Series("cluster", [str(int(v)) for v in labels]),
        pl.Series("archetype", [arche[int(v)] for v in labels]),
    )
    return arche, chosen, curve, df, labels, sil


@app.cell
def _(chosen, curve, mo, sil):
    _rows = "".join(
        f"<tr><td style='text-align:right;padding:0 8px'>{kk}</td>"
        f"<td style='text-align:right;padding:0 8px'>{s:.3f}</td>"
        f"<td><div style='background:#4c9;height:10px;width:{max(0, s) * 300:.0f}px'></div></td>"
        f"<td style='padding-left:6px'>{'← chosen' if kk == chosen else ''}</td></tr>"
        for kk, s, _ in curve
    )
    mo.md(f"### auto-k → **{chosen} clusters** (silhouette **{sil:.3f}**)\n\n"
          f"<table><tr><th>k</th><th>sil</th><th></th><th></th></tr>{_rows}</table>")
    return


@app.cell
def _(color, df, mo, px):
    data = {c: df[c].to_list() for c in
            ("x", "y", "games", "winrate", "cluster", "archetype", "deck_key")}
    fig = px.scatter(
        data, x="x", y="y",
        color="cluster" if color.value == "cluster" else color.value,
        size="games", size_max=45, hover_name="archetype",
        hover_data={"games": True, "winrate": ":.2f", "deck_key": True,
                    "x": False, "y": False},
        color_continuous_scale="Viridis",
        title=f"{len(df)} unique decks · PCA 2D · colour={color.value}",
    )
    fig.update_layout(height=560)
    mo.ui.plotly(fig)
    return


@app.cell
def _(agg, arche, chosen, labels, mo, pl):
    cl = agg.with_columns(pl.Series("cluster", labels.tolist()))
    summary = (
        cl.group_by("cluster")
        .agg(pl.len().alias("decks"), pl.col("games").sum().alias("games"),
             (pl.col("winrate") * pl.col("games")).sum().alias("_w"))
        .with_columns((pl.col("_w") / pl.col("games")).round(3).alias("winrate"))
        .drop("_w")
        .with_columns(pl.col("cluster").replace_strict(arche).alias("archetype"))
        .sort("games", descending=True)
    )
    cluster_pick = mo.ui.dropdown(
        {f"cluster {r['cluster']} · {r['archetype'][:40]} "
         f"({r['games']} games, wr {r['winrate']})": r["cluster"]
         for r in summary.iter_rows(named=True)},
        label="inspect cluster", value=None,
    )
    mo.vstack([mo.md(f"### {chosen} clusters"),
               mo.ui.table(summary, selection=None), cluster_pick])
    return (cluster_pick,)


@app.cell
def _(IMG_DIR, NAMES, agg, arche, labels, mo, pl):
    def _():
        # Contact-sheet GRID: every cluster at once, shown by its most common cards.
        _sigs = [[int(c) for c in row] for row in agg["cards"].to_list()]
        _lab = labels.tolist()
        _uniq = sorted(set(_lab))
        _count = {}
        for _c in _uniq:
            _idx = [i for i in range(len(_lab)) if _lab[i] == _c]
            _cnt = {}
            for _i in _idx:
                for _cid in set(_sigs[_i]):
                    _cnt[_cid] = _cnt.get(_cid, 0) + 1
            _count[_c] = _cnt

        def _top_ids(c, top=7):
            out = []
            for cid in sorted(_count[c], key=lambda x: _count[c][x], reverse=True):
                if NAMES.get(cid, "").startswith("Basic "):
                    continue
                out.append(cid)
                if len(out) == top:
                    break
            out.sort()
            return out

        def _thumb(cid):
            p = IMG_DIR / f"{cid}.png"
            return (mo.image(str(p), width=52, rounded=True) if p.is_file()
                    else mo.md(f"`{NAMES.get(cid, cid)}`"))

        _order = (agg.with_columns(pl.Series("cluster", _lab))
                  .group_by("cluster")
                  .agg(pl.col("games").sum().alias("g"),
                       (pl.col("winrate") * pl.col("games")).sum().alias("_w"))
                  .with_columns((pl.col("_w") / pl.col("g")).alias("wr"))
                  .sort("g", descending=True))
        _rows = []
        for r in _order.iter_rows(named=True):
            c = r["cluster"]
            head = mo.md(f"**c{c}** · {arche[c]}<br>{r['g']} games · wr {r['wr']:.2f}")
            strip = mo.hstack([_thumb(cid) for cid in _top_ids(c)],
                              justify="start", gap=0.2)
            _rows.append(mo.hstack([head, strip], justify="start", gap=1, widths=[2, 5]))
        return mo.vstack([mo.md("### all clusters — most common cards (grid)"), *_rows], gap=0.5)


    _()
    return


@app.cell
def _(COUNTS, IMG_DIR, NAMES, agg, cluster_pick, labels, mo, pl):
    # mo.image serves each PNG as a virtual file (URL), so the cell output stays
    # tiny — base64-inlining ~60 cards blew past marimo's 16 MB output cap.
    def _card(cid: int, cnt: int):
        p = IMG_DIR / f"{cid}.png"
        nm = NAMES.get(cid, f"#{cid}")
        cap = f"{nm[:16]} ×{cnt}"
        if p.is_file():
            return mo.image(str(p), width=76, rounded=True, caption=cap)
        return mo.md(f"`{nm}` ×{cnt}")

    def _deck(deck_key: int):
        cards = COUNTS.get(deck_key, [])
        rows = [mo.hstack([_card(cid, cnt) for cid, cnt in cards[i:i + 8]],
                          justify="start", gap=0.4)
                for i in range(0, len(cards), 8)]
        return mo.vstack(rows, gap=0.6)

    if cluster_pick.value is None:
        out = mo.md("*pick a cluster above to see its representative decks with card art.*")
    else:
        c = cluster_pick.value
        sub = (agg.with_columns(pl.Series("cluster", labels.tolist()))
               .filter(pl.col("cluster") == c).sort("games", descending=True).head(3))
        blocks = []
        for i, r in enumerate(sub.iter_rows(named=True)):
            tag = "most-played (representative)" if i == 0 else f"#{i + 1} most-played"
            blocks.append(mo.md(f"**{tag}** · deck_key `{r['deck_key']}` · "
                                f"{r['games']} games · win-rate {r['winrate']:.2f}"))
            blocks.append(_deck(r["deck_key"]))
        out = mo.vstack([mo.md(f"## cluster {c} — top decks"), *blocks])
    out
    return


if __name__ == "__main__":
    app.run()
