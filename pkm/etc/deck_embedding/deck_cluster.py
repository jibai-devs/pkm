# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer",
#     "rich",
#     "numpy",
#     "polars",
# ]
# ///
"""Cluster the learned deck embeddings into archetypes — **auto-`k`**.

You don't pick the number of clusters: it sweeps k and keeps the one with the
best mean silhouette (cosine space), then labels each cluster by its most common
cards. Writes a tidy `deck_clusters.parquet` (one row per unique decklist) and
prints a summary.

    uv run --script deck_cluster.py                       # auto-k on deck_embeddings.parquet
    uv run --script deck_cluster.py --embeddings deck_embeddings_emb16.parquet
    uv run --script deck_cluster.py --k 4                 # force k (skip the sweep)
    uv run --script deck_cluster.py --help

Points are UNIQUE decklists (the raw field is ~99% duplicate re-submissions), so
this answers "how many distinct decks/archetypes are in this data", weighting by
how many games each list was played in only for the summary stats.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_data as dd  # noqa: E402

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

DEFAULT_EMBEDDINGS = dd.DATA_DIR / "deck_embeddings.parquet"
DEFAULT_DATA = dd.DATA_DIR / "decks_with_outcomes.parquet"
DEFAULT_OUT = dd.DATA_DIR / "deck_clusters.parquet"


def _cards_json() -> Path:
    """Best-effort path to cards.json (card_id -> name); returns first that exists."""
    for c in (dd.REPO_ROOT / "replay" / "cards.json",
              dd.REPO_ROOT / "pkm_data" / "replay" / "cards.json"):
        if c.is_file():
            return c
    return dd.REPO_ROOT / "replay" / "cards.json"


def _card_names(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text())
    cards = blob.get("cards", blob) if isinstance(blob, dict) else blob
    return {int(c["card_id"]): c["name"] for c in cards if "card_id" in c and "name" in c}


def _card_marquee(path: Path) -> dict[int, float]:
    """cid -> 'marquee score' for POKÉMON only (hp>0), so a cluster can be named
    by its headline Pokémon. Pure card metadata (ex/mega/tera/hp), NOT corpus
    statistics — mega_ex > tera > ex > stage2 > basic, hp as a tie-break."""
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text())
    cards = blob.get("cards", blob) if isinstance(blob, dict) else blob
    out: dict[int, float] = {}
    for c in cards:
        if "card_id" not in c:
            continue
        hp = c.get("hp") or 0
        if hp <= 0:                       # energy / trainer / stadium — not a Pokémon
            continue
        tier = (5 if c.get("mega_ex") else 4 if c.get("tera") else 3 if c.get("ex")
                else 2 if c.get("stage2") else 1)
        out[int(c["card_id"])] = tier + hp / 10000.0   # hp<<1 so it only breaks ties
    return out


# --- clustering primitives (plain numpy — keeps the venv dependency-free) -------

def _kmeans(X: np.ndarray, k: int, iters: int = 100, restarts: int = 8, seed: int = 0):
    """K-means with k-means++ init and a few restarts; returns (labels, inertia)."""
    best_lab, best_inertia = None, np.inf
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        # k-means++ seeding
        C = X[rng.integers(len(X))][None]
        for _ in range(k - 1):
            d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1).min(1)
            probs = d2 / (d2.sum() + 1e-12)
            C = np.vstack([C, X[rng.choice(len(X), p=probs)]])
        lab = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            new = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.stack([X[new == j].mean(0) if (new == j).any() else C[j]
                             for j in range(k)])
            if (new == lab).all():
                lab = new
                break
            lab, C = new, newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if inertia < best_inertia:
            best_lab, best_inertia = lab, inertia
    return best_lab, best_inertia


def _silhouette(X: np.ndarray, lab: np.ndarray) -> float:
    """Mean silhouette over all points (euclidean on unit vectors == cosine order)."""
    n = len(X)
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None]) ** 2).sum(-1), 0.0))
    uniq = np.unique(lab)
    if len(uniq) < 2:
        return -1.0
    s = np.zeros(n)
    for i in range(n):
        same = lab == lab[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, lab == j].mean() for j in uniq if j != lab[i])
        s[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(s.mean())


def _auto_k(X: np.ndarray, k_max: int, seed: int):
    """Sweep k=2..k_max, return (best_k, labels, [(k, silhouette, inertia)…])."""
    curve, best = [], (-1.0, None, None)
    k_hi = min(k_max, len(X) - 1)
    for k in range(2, k_hi + 1):
        lab, inertia = _kmeans(X, k, seed=seed)
        sil = _silhouette(X, lab)
        curve.append((k, sil, inertia))
        if sil > best[0]:
            best = (sil, k, lab)
    return best[1], best[2], curve


# --- cluster labelling ----------------------------------------------------------

def _labels_per_cluster(sigs_cards: list[list[int]], lab: np.ndarray,
                        names: dict[int, str], top: int = 2) -> dict[int, str]:
    """Name each cluster by its headline POKÉMON — the highest-marquee Pokémon
    present (mega/tera/ex/stage2/hp), tie-broken by how common it is in the
    cluster. Uses card metadata, not corpus statistics."""
    marquee = _card_marquee(_cards_json())
    uniq = sorted(set(lab.tolist()))
    out = {}
    for c in uniq:
        idx = np.where(lab == c)[0]
        cnt: dict[int, int] = {}
        for i in idx:
            for cid in set(sigs_cards[i]):
                cnt[cid] = cnt.get(cid, 0) + 1
        poke = [cid for cid in cnt if cid in marquee]
        poke.sort(key=lambda cid: (marquee[cid], cnt[cid]), reverse=True)
        picked = [names.get(cid, f"#{cid}") for cid in poke[:top]]
        out[c] = " · ".join(picked) if picked else "(no Pokémon)"
    return out


@app.command()
def cluster(
    embeddings: Annotated[Path, typer.Option(help="deck_embeddings*.parquet")] = DEFAULT_EMBEDDINGS,
    data: Annotated[Path, typer.Option(help="decks_with_outcomes.parquet")] = DEFAULT_DATA,
    k: Annotated[int, typer.Option(help="clusters; 0 = auto (silhouette sweep)")] = 0,
    k_max: Annotated[int, typer.Option(help="max k to try when auto")] = 12,
    min_games: Annotated[int, typer.Option(help="drop decklists played in fewer games")] = 1,
    seed: Annotated[int, typer.Option(help="random seed")] = 0,
    cards: Annotated[Path | None, typer.Option(help="cards.json for names")] = None,
    out: Annotated[Path, typer.Option(help="write deck_clusters.parquet")] = DEFAULT_OUT,
) -> None:
    """Auto-cluster the deck embeddings into archetypes and dump the decks."""
    names = _card_names(cards or _cards_json())

    # per (episode, player): sorted card list + result; one embedding per row.
    per_deck = (
        pl.read_parquet(data)
        .group_by("episode_id", "player")
        .agg(pl.col("card_id").sort().alias("cards"), pl.col("won").first().alias("won"))
        .with_columns((pl.col("episode_id") * 10 + pl.col("player")).alias("deck_key"))
        .with_columns(pl.col("cards").cast(pl.List(pl.Utf8)).list.join(",").alias("sig"))
    )
    emb = pl.read_parquet(embeddings)
    m = emb.join(per_deck, on="deck_key", how="inner")

    # collapse to unique decklists
    agg = (
        m.group_by("sig")
        .agg(
            pl.col("embedding").first(),
            pl.len().alias("games"),
            pl.col("won").mean().alias("winrate"),
            pl.col("cards").first(),
            pl.col("deck_key").first(),
        )
        .filter(pl.col("games") >= min_games)
        .sort("games", descending=True)
    )
    n = agg.height
    if n < 2:
        console.print("[red]not enough unique decklists to cluster[/red]")
        raise typer.Exit(1)

    X = np.array(agg["embedding"].to_list(), dtype="float64")
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)  # cosine space
    sigs_cards = [[int(c) for c in row] for row in agg["cards"].to_list()]

    if k and k >= 2:
        lab, _ = _kmeans(X, k, seed=seed)
        sil = _silhouette(X, lab)
        curve = [(k, sil, 0.0)]
        chosen = k
    else:
        chosen, lab, curve = _auto_k(X, k_max, seed)
        sil = next(s for kk, s, _ in curve if kk == chosen)

    assert lab is not None
    labels = _labels_per_cluster(sigs_cards, lab, names)

    # header + silhouette sweep
    hdr = Table.grid(padding=(0, 2))
    hdr.add_column(style="dim"); hdr.add_column(style="bold")
    hdr.add_row("embeddings", f"{embeddings.name}  ({X.shape[1]}-d)")
    hdr.add_row("unique decklists", f"{n:,}  (from {m.height:,} game-instances)")
    hdr.add_row("k selection", "forced" if (k and k >= 2) else "auto (max silhouette)")
    hdr.add_row("chosen k", f"[green]{chosen}[/green]  ·  silhouette {sil:.3f}")
    console.print(Panel(hdr, title="[bold]deck clustering", border_style="cyan"))

    if len(curve) > 1:
        sweep = Table(title="silhouette sweep", header_style="bold")
        sweep.add_column("k", justify="right"); sweep.add_column("silhouette", justify="right")
        sweep.add_column("", justify="left")
        best_k = chosen
        for kk, s, _ in curve:
            bar = "█" * max(0, int(round(s * 40)))
            sweep.add_row(f"{kk}", f"{s:.3f}", f"[green]{bar}[/green]" +
                          ("  [bold]<- chosen[/bold]" if kk == best_k else ""))
        console.print(sweep)

    # medoid per cluster (unit vector closest to the cluster mean direction)
    medoid_key: dict[int, int] = {}
    for c in sorted(set(lab.tolist())):
        idx = np.where(lab == c)[0]
        centroid = X[idx].mean(0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        medoid_key[c] = int(agg["deck_key"][int(idx[np.argmax(X[idx] @ centroid)])])

    # per-cluster summary, ordered by total games played
    cl = agg.with_columns(pl.Series("cluster", lab.tolist()))
    summary = (
        cl.group_by("cluster")
        .agg(
            pl.len().alias("decks"),
            pl.col("games").sum().alias("games"),
            (pl.col("winrate") * pl.col("games")).sum().alias("_wsum"),
        )
        .with_columns((pl.col("_wsum") / pl.col("games")).round(3).alias("winrate"))
        .drop("_wsum")
        .sort("games", descending=True)
    )
    tbl = Table(title=f"{chosen} clusters", header_style="bold")
    for col, j in (("cluster", "right"), ("decks", "right"), ("games", "right"),
                   ("winrate", "right"), ("archetype (key Pokémon)", "left")):
        tbl.add_column(col, justify=j)
    for r in summary.iter_rows(named=True):
        c = r["cluster"]
        tbl.add_row(str(c), f"{r['decks']:,}", f"{r['games']:,}",
                    f"{r['winrate']:.3f}", labels.get(c, ""))
    console.print(tbl)

    # write tidy per-deck clusters parquet
    top_names = [
        ", ".join(names.get(cid, f"#{cid}") for cid in cs[:8]) for cs in sigs_cards
    ]
    result = cl.with_columns(
        pl.Series("archetype", [labels[c] for c in lab.tolist()]),
        pl.Series("is_medoid", [int(agg["deck_key"][i]) == medoid_key[lab[i]] for i in range(n)]),
        pl.Series("top_cards", top_names),
        pl.col("cards").list.len().alias("n_cards"),
    ).select("cluster", "archetype", "deck_key", "games", "winrate",
             "is_medoid", "n_cards", "top_cards", "sig")
    result.write_parquet(out)
    console.print(f"[green]wrote[/green] {result.height:,} decklists x cluster -> {out}")
    console.print("[dim]medoid rows (is_medoid=1) are each cluster's representative decklist.[/dim]")


if __name__ == "__main__":
    app()
