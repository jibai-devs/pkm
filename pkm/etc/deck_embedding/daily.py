# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer",
#     "rich",
#     "polars",
#     "kaggle",
#     "orjson",
#     "pyarrow",
# ]
# ///
"""Fetch official daily episode datasets and build the deck-embedding table —
without ever unzipping a full ~21 GB day to disk.

Kaggle now publishes the competition's episodes as one dataset per day
(`kaggle/pokemon-tcg-ai-battle-episodes-<date>`), indexed by a tiny manifest
(`kaggle/pokemon-tcg-ai-battle-episodes-index`). That replaces the old
leaderboard crawler entirely.

The catch: a day is ~5–8k episodes and ~21 GB *unzipped*, but only ~1 GB
*compressed*. So this tool downloads the compressed zip (one bounded file) and
**streams** it: each episode JSON is decompressed into memory one at a time,
reduced to its handful of deck+outcome rows (via
``deck_data.episode_deck_rows``), then discarded. Peak disk ≈ the compressed
zip; peak memory ≈ one episode. The accumulating table is tiny (~30 rows/episode).

    uv run --script daily.py list                       # what days exist
    uv run --script daily.py build 2026-06-16           # one day -> decks_with_outcomes.parquet
    uv run --script daily.py build 2026-06-16 2026-06-17 --append
    uv run --script daily.py build 2026-06-16 --limit 200 --keep-zip   # quick test

Auth: uses your ~/.kaggle/kaggle.json (same as the `kaggle` CLI).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TextColumn, TimeElapsedColumn)
from rich.table import Table

# Sibling module — the single source of truth for JSON -> deck rows + schema.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_data as dd  # noqa: E402

# The manifest + streaming-zip fetch primitives live in replaydb (one source of
# truth), as does the full lossless shred (`python -m replaydb daily ingest`).
# This tool is the deck-embedding-specific consumer: it streams the very same
# zips but keeps only decks+outcomes instead of the full game state.
sys.path.insert(0, str(dd.KAGGLE_DIR))
from replaydb.daily import (download_day_zip, iter_episodes_from_zip,  # noqa: E402
                            load_manifest)

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()


@app.command(name="list")
def list_days() -> None:
    """Show the days available in the manifest (date, episodes, size, scores)."""
    m = load_manifest().sort("date")
    t = Table(title="published daily episode datasets", header_style="bold")
    for c, j in (("date", "left"), ("episodes", "right"), ("size", "right"),
                 ("top_avg", "right"), ("median_avg", "right")):
        t.add_column(c, justify=j)
    total_ep = 0
    for r in m.iter_rows(named=True):
        total_ep += int(r["episode_count"])
        gib = r["total_bytes"] / 1024**3
        t.add_row(r["date"], f"{r['episode_count']:,}", f"{gib:,.1f} GiB",
                  f"{r['top_avg_score']:.0f}", f"{r['median_avg_score']:.0f}")
    console.print(t)
    console.print(f"[dim]{m.height} days · {total_ep:,} episodes total "
                  f"(unzipped ~{m['total_bytes'].sum() / 1024**3:,.0f} GiB; "
                  f"compressed download is far smaller)[/dim]")


@app.command()
def build(
    dates: Annotated[list[str], typer.Argument(help="one or more YYYY-MM-DD days from `list`")],
    out: Annotated[Path, typer.Option(help="output parquet")] = dd.DATA_DIR / "decks_with_outcomes.parquet",
    decisive_only: Annotated[bool, typer.Option(help="keep only games with a {+1,-1} result")] = True,
    append: Annotated[bool, typer.Option(help="merge into an existing `out` instead of replacing")] = False,
    keep_zip: Annotated[bool, typer.Option(help="keep the downloaded zips instead of deleting")] = False,
    limit: Annotated[int, typer.Option(help="cap episodes per day (0 = all), for quick tests")] = 0,
    cache: Annotated[Path, typer.Option(help="where to stash zips while working")] = dd.KAGGLE_DIR / "episodes_daily" / "_zips",
) -> None:
    """Stream one or more days into a decks-with-outcomes parquet."""
    rows: list[dd.DeckRow] = []
    per_day: list[tuple[str, int, int]] = []                # (date, episodes_seen, rows_kept)

    for date in dates:
        with console.status(f"[cyan]downloading {date} (compressed zip)…"):
            zip_path = download_day_zip(date, cache)
        zsize = zip_path.stat().st_size / 1024**2
        kept0 = len(rows)
        n_ep = 0
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task(f"streaming {date} ({zsize:,.0f} MB zip)", total=limit or None)
            for ep in iter_episodes_from_zip(zip_path):
                rows.extend(dd.episode_deck_rows(ep, decisive_only))
                n_ep += 1
                prog.update(task, advance=1)
                if limit and n_ep >= limit:
                    break
        per_day.append((date, n_ep, len(rows) - kept0))
        if not keep_zip:
            zip_path.unlink(missing_ok=True)

    df = dd.rows_to_frame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    if append and out.is_file():
        existing = pl.read_parquet(out)
        df = pl.concat([existing, df]).unique().sort("episode_id", "player", "card_id")
    df.write_parquet(out)

    if not keep_zip and cache.is_dir() and not any(cache.iterdir()):
        cache.rmdir()

    t = Table(title="build summary", header_style="bold")
    for c in ("day", "episodes seen", "rows kept"):
        t.add_column(c, justify="right")
    for date, seen, kept in per_day:
        t.add_row(date, f"{seen:,}", f"{kept:,}")
    console.print(t)
    n_eps = df.select("episode_id").n_unique()
    n_decks = df.select("episode_id", "player").unique().height
    console.print(f"[green]wrote[/green] {out}  ·  {df.height:,} rows · "
                  f"{n_eps:,} episodes · {n_decks:,} decks"
                  + ("  [dim](merged with existing)[/dim]" if append else ""))
    console.print("[dim]want the full game state (all tables), not just decks? "
                  "use `python -m replaydb daily ingest`[/dim]")


if __name__ == "__main__":
    app()
