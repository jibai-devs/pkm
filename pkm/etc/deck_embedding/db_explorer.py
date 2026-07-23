# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "polars",
#     "pyarrow",
#     "torch",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import polars as pl

    return Path, pl


@app.cell
def _(Path):
    # This notebook lives at <repo>/pkm/etc/deck_embedding/; the crawl data lives
    # at <repo>/pkm_data/kaggle_replays/pokemon-tcg-ai-battle/. Anchor to __file__
    # so every path below resolves no matter where marimo is launched from.
    KAGGLE_DIR = (
        Path(__file__).resolve().parents[3]
        / "pkm_data" / "kaggle_replays" / "pokemon-tcg-ai-battle"
    )
    return (KAGGLE_DIR,)


@app.cell
def _(KAGGLE_DIR, pl):
    # Read the decks parquet: one row per (episode_id, player, card_id, count).
    decks = pl.read_parquet(KAGGLE_DIR / "parquet" / "decks.parquet")
    return (decks,)


@app.cell
def _(decks):
    decks
    return


@app.cell
def _(decks, pl):
    # Regenerate each full 60-card deck: repeat every card_id `count` times, then
    # collect back per (episode_id, player) into one sorted list of 60 numbers.
    deck_vectors = (
        decks.with_columns(pl.col("card_id").repeat_by("count").alias("card_id"))
        .explode("card_id")
        .group_by("episode_id", "player", maintain_order=True)
        .agg(pl.col("card_id").sort().alias("deck"))
    )
    return (deck_vectors,)


@app.cell
def _(deck_vectors):
    deck_vectors
    return


@app.cell
def _(KAGGLE_DIR, pl):
    # The cards table is db/cards/*.parquet — 66.5M rows across 209 parts, so
    # scan lazily and only ever collect a bounded slice (never read_parquet it).
    cards = pl.scan_parquet(str(KAGGLE_DIR / "db" / "cards" / "*.parquet"))
    return (cards,)


@app.cell
def _(cards):
    cards.head(100).collect()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(Path, mo):
    # deck_data.py / deck_embed.py are siblings of this notebook; make them
    # importable no matter where marimo is launched.
    import sys

    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    import deck_data as dd
    import deck_embed as de

    mo.md("## Train the deck-matchup model")
    return dd, de


@app.cell
def _(dd, de):
    # Ensure the deck+outcome bridge exists; build it from db/ if missing.
    # dd.DATA_DIR / dd.DB_DIR are anchored to deck_data.py's location.
    bridge = dd.DATA_DIR / "decks_with_outcomes.parquet"
    if not bridge.is_file():
        dd.build_decks_with_outcomes(db=dd.DB_DIR, out=bridge)

    matchups, decks_by_key, vocab = de.load_matchups_from_parquet(str(bridge))
    return decks_by_key, matchups, vocab


@app.cell
def _(matchups, mo, vocab):
    mo.md(f"""
    Loaded **{len(matchups):,}** matchups over **{len(matchups) // 2:,}** "
        f"episodes · vocab **{vocab.size}** cards.
    """)
    return


@app.cell
def _(mo):
    epochs = mo.ui.slider(1, 50, value=10, label="epochs")
    batch = mo.ui.dropdown(["64", "128", "256", "512"], value="256", label="batch size")
    lr = mo.ui.dropdown(["3e-4", "1e-3", "3e-3", "1e-2"], value="1e-3", label="learning rate")
    emb_dim = mo.ui.dropdown(["16", "32", "64", "128"], value="64", label="embedding dim")
    train_button = mo.ui.run_button(label="Train")
    mo.vstack([mo.hstack([epochs, batch, lr, emb_dim], justify="start", gap=1), train_button])
    return batch, emb_dim, epochs, lr, train_button


@app.cell
def _(batch, de, emb_dim, epochs, lr, matchups, mo, train_button, vocab):
    # Only trains when the button is clicked; downstream cells wait on this.
    mo.stop(not train_button.value, mo.md("*Pick hyperparameters and click **Train**.*"))

    _d = int(emb_dim.value)
    encoder = de.DeckEncoder(vocab.size, dim=_d, emb_dim=_d)
    model, history = de.train(
        de.TwoTowerMatchup(encoder, emb_dim=_d),
        matchups,
        epochs=epochs.value,
        batch_size=int(batch.value),
        lr=float(lr.value),
        verbose=False,
    )
    mo.md(
        f"Trained **{epochs.value}** epochs — final matchup accuracy "
        f"**{history[-1]['acc']:.3f}**, loss **{history[-1]['loss']:.4f}**."
    )
    return history, model


@app.cell
def _(history):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ep = [h["epoch"] for h in history]
    ax[0].plot(ep, [h["loss"] for h in history], marker="o")
    ax[0].set_title("loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["acc"] for h in history], marker="o", color="green")
    ax[1].set_title("matchup accuracy"); ax[1].set_xlabel("epoch")
    fig.tight_layout()
    fig
    return


@app.cell
def _(de, decks_by_key, model):
    # Embed every deck with the trained encoder (for clustering / nearest-deck).
    deck_keys = list(decks_by_key.keys())
    embeddings = de.embed_decks(model, [decks_by_key[k] for k in deck_keys])
    return deck_keys, embeddings


@app.cell
def _(deck_keys, mo):
    which = mo.ui.dropdown(
        options=[str(k) for k in deck_keys[:500]],
        value=str(deck_keys[0]),
        label="Deck (key = episode_id*10 + player)",
    )
    topk = mo.ui.slider(1, 20, value=6, label="neighbours")
    mo.hstack([which, topk], justify="start", gap=1)
    return topk, which


@app.cell
def _(de, deck_keys, embeddings, mo, topk, which):
    _qi = deck_keys.index(int(which.value))
    _idx, _dist = de.nearest(embeddings[_qi:_qi + 1], embeddings, k=topk.value + 1)
    rows = [
        {"rank": r, "deck_key": deck_keys[j], "cosine_dist": round(float(d), 4)}
        for r, (j, d) in enumerate(zip(_idx[0].tolist(), _dist[0].tolist()))
    ]
    mo.vstack([mo.md(f"### Nearest decks to `{which.value}`"), mo.ui.table(rows, selection=None)])
    return


if __name__ == "__main__":
    app.run()
