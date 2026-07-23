# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer",
#     "rich",
#     "torch",
#     "numpy",
#     "polars",
# ]
# ///
"""Train the Set-Transformer two-tower deck-matchup model — typer + rich CLI.

Uses the GPU when available and falls back to CPU automatically (`--device auto`).

    uv run --script train_deck_model.py train                 # sensible defaults
    uv run --script train_deck_model.py train --epochs 40 --emb-dim 128
    uv run --script train_deck_model.py embed
    uv run --script train_deck_model.py train --help

Data comes from `deck_embedding/decks_with_outcomes.parquet`; if it's missing it
is built from `db/` on the fly (see deck_data.build_decks_with_outcomes). All
default paths are anchored to this file's location (see deck_data.DATA_DIR /
DB_DIR), so the script runs from any working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TextColumn, TimeElapsedColumn)
from rich.table import Table

# Make the sibling modules importable regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_data as dd       # noqa: E402
import deck_embed as de      # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()

# Anchored to this file's location via deck_data (see DATA_DIR / DB_DIR there),
# so defaults resolve the same from any working directory.
DEFAULT_DATA = dd.DATA_DIR / "decks_with_outcomes.parquet"
DEFAULT_DB = dd.DB_DIR
DEFAULT_MODEL = dd.DATA_DIR / "deck_model.pt"
DEFAULT_EMBEDDINGS = dd.DATA_DIR / "deck_embeddings.parquet"


def _load_data(data: Path, db: Path):
    """Load matchups, building the bridge parquet from db/ if it's absent."""
    if not data.is_file():
        console.print(f"[yellow]{data} not found — building it from {db}/[/yellow]")
        with console.status("[cyan]reconstructing decks + outcomes from db/…"):
            dd.build_decks_with_outcomes(db=db, out=data)
    with console.status(f"[cyan]loading {data}…"):
        return de.load_matchups_from_parquet(str(data))


def _split(matchups, val_frac: float, seed: int):
    """Deterministic train/val split that keeps a game's two mirror orientations
    together (load_matchups_from_parquet appends them as consecutive pairs), so
    no game leaks across the split."""
    import random
    pairs = list(range(0, len(matchups) - 1, 2))     # even index = start of a mirror pair
    random.Random(seed).shuffle(pairs)
    n_val = int(len(pairs) * val_frac)
    take = lambda ps: [matchups[i] for p in ps for i in (p, p + 1)]
    return take(pairs[n_val:]), take(pairs[:n_val])


@app.command()
def train(
    data: Annotated[Path, typer.Option(help="decks_with_outcomes.parquet")] = DEFAULT_DATA,
    db: Annotated[Path, typer.Option(help="db/ dir used to build data if missing")] = DEFAULT_DB,
    epochs: Annotated[int, typer.Option(help="training epochs")] = 20,
    batch_size: Annotated[int, typer.Option(help="minibatch size")] = 256,
    lr: Annotated[float, typer.Option(help="Adam learning rate")] = 1e-3,
    emb_dim: Annotated[int, typer.Option(help="deck embedding dim")] = 64,
    dim: Annotated[int, typer.Option(help="transformer hidden dim")] = 64,
    heads: Annotated[int, typer.Option(help="attention heads")] = 4,
    n_blocks: Annotated[int, typer.Option(help="ISAB blocks")] = 2,
    m: Annotated[int, typer.Option(help="inducing points per ISAB")] = 16,
    val_frac: Annotated[float, typer.Option(help="held-out fraction")] = 0.1,
    seed: Annotated[int, typer.Option(help="random seed")] = 0,
    device: Annotated[str, typer.Option(help="'auto' | 'cuda' | 'cpu'")] = "auto",
    out: Annotated[Path, typer.Option(help="save checkpoint here (.pt)")] = DEFAULT_MODEL,
) -> None:
    """Train the two-tower matchup model and save a checkpoint."""
    import torch

    dev = de.resolve_device(device)
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "—"
    matchups, decks_by_key, vocab = _load_data(data, db)
    tr, val = _split(matchups, val_frac, seed)

    torch.manual_seed(seed)
    enc = de.DeckEncoder(vocab.size, dim=dim, heads=heads, m=m, n_blocks=n_blocks, emb_dim=emb_dim)
    model = de.TwoTowerMatchup(enc, emb_dim=emb_dim)
    n_params = sum(p.numel() for p in model.parameters())

    cfg = Table.grid(padding=(0, 2))
    cfg.add_column(style="dim"); cfg.add_column(style="bold")
    cfg.add_row("device", f"{dev}" + (f"  ([green]{gpu}[/green])" if dev == "cuda" else "  (CPU)"))
    cfg.add_row("matchups", f"{len(tr):,} train / {len(val):,} val")
    cfg.add_row("decks / vocab", f"{len(decks_by_key):,} decks · {vocab.size} cards")
    cfg.add_row("model", f"{n_params:,} params · dim={dim} heads={heads} blocks={n_blocks} emb={emb_dim}")
    cfg.add_row("optim", f"Adam lr={lr} · batch={batch_size} · epochs={epochs}")
    console.print(Panel(cfg, title="[bold]deck-matchup trainer", border_style="cyan"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("training", total=epochs)

        def on_epoch(rec: dict[str, float]) -> None:
            desc = f"epoch {int(rec['epoch']) + 1}/{epochs} · loss {rec['loss']:.4f} · acc {rec['acc']:.3f}"
            if "val_acc" in rec:
                desc += f" · [green]val_acc {rec['val_acc']:.3f}[/green]"
            prog.update(task, advance=1, description=desc)

        model, history = de.train(model, tr, epochs=epochs, batch_size=batch_size, lr=lr,
                                   device=dev, val_matchups=val or None, verbose=False,
                                   on_epoch=on_epoch)

    # per-epoch metrics table
    t = Table(title="training history", header_style="bold")
    for c in ("epoch", "loss", "acc", "val_loss", "val_acc"):
        t.add_column(c, justify="right")
    for rec in history:
        t.add_row(str(int(rec["epoch"])), f"{rec['loss']:.4f}", f"{rec['acc']:.3f}",
                  f"{rec.get('val_loss', float('nan')):.4f}", f"{rec.get('val_acc', float('nan')):.3f}")
    console.print(t)

    ckpt = {
        "state_dict": model.to("cpu").state_dict(),
        "idx2id": vocab.idx2id,
        "config": {"dim": dim, "heads": heads, "m": m, "n_blocks": n_blocks, "emb_dim": emb_dim},
    }
    torch.save(ckpt, out)
    best = max((r.get("val_acc", r["acc"]) for r in history), default=0.0)
    console.print(f"[green]saved checkpoint[/green] -> {out}  ·  best val_acc [bold]{best:.3f}[/bold]")


def _load_checkpoint(path: Path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    idx2id = ck["idx2id"]
    vocab = de.DeckVocab(idx2id=idx2id, id2idx={c: i for i, c in enumerate(idx2id)})
    c = ck["config"]
    enc = de.DeckEncoder(vocab.size, dim=c["dim"], heads=c["heads"], m=c["m"],
                         n_blocks=c["n_blocks"], emb_dim=c["emb_dim"])
    model = de.TwoTowerMatchup(enc, emb_dim=c["emb_dim"])
    model.load_state_dict(ck["state_dict"])
    return model, vocab


@app.command()
def embed(
    checkpoint: Annotated[Path, typer.Option(help="trained .pt checkpoint")] = DEFAULT_MODEL,
    data: Annotated[Path, typer.Option(help="decks_with_outcomes.parquet")] = DEFAULT_DATA,
    db: Annotated[Path, typer.Option(help="db/ dir used to build data if missing")] = DEFAULT_DB,
    out: Annotated[Path, typer.Option(help="write embeddings parquet")] = DEFAULT_EMBEDDINGS,
    device: Annotated[str, typer.Option(help="'auto' | 'cuda' | 'cpu'")] = "auto",
) -> None:
    """Embed every deck with a trained checkpoint and write a parquet."""
    import polars as pl

    dev = de.resolve_device(device)
    model, _vocab = _load_checkpoint(checkpoint)
    _matchups, decks_by_key, _v = _load_data(data, db)
    keys = list(decks_by_key.keys())
    with console.status(f"[cyan]embedding {len(keys):,} decks on {dev}…"):
        emb = de.embed_decks(model, [decks_by_key[k] for k in keys], device=dev)

    df = pl.DataFrame({"deck_key": keys}).with_columns(
        pl.Series("embedding", emb.tolist())
    )
    df.write_parquet(out)
    console.print(f"[green]wrote[/green] {df.height:,} × {emb.shape[1]}-d embeddings -> {out}")


if __name__ == "__main__":
    app()
