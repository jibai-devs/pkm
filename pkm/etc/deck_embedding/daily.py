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

import json
import multiprocessing as mp
import os
import shutil
import sys
import zipfile
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

# The manifest + zip-download primitive live in replaydb (one source of truth),
# as does the full lossless shred (`python -m replaydb daily ingest`). This tool
# is the deck-embedding-specific consumer: it extracts the same zips to disk and
# parses them across CPU cores, keeping only decks+outcomes.
sys.path.insert(0, str(dd.KAGGLE_DIR))
from replaydb.daily import download_day_zip, load_manifest  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()

DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 2)


# --------------------------------------------------------------------------- #
# Parallel disk-based parsing. Extraction lays every episode JSON on disk, then
# a process pool reads+parses them across cores — decompressing a zip is
# single-threaded (the bottleneck), so we don't stream; we shred from disk.
# --------------------------------------------------------------------------- #

def _parse_files(args: tuple[list[str], bool]) -> list[tuple]:
    """Worker: read a batch of episode JSON files from disk and return deck rows."""
    paths, decisive_only = args
    import orjson

    rows: list[tuple] = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                ep = orjson.loads(f.read())
        except (OSError, ValueError):
            continue
        rows.extend(dd.episode_deck_rows(ep, decisive_only))
    return rows


def _chunks(items: list, n: int) -> list[list]:
    """Split into ~n roughly equal contiguous chunks (n = pool size × oversample)."""
    if n <= 0:
        return [items]
    k, r = divmod(len(items), n)
    out, i = [], 0
    for j in range(n):
        size = k + (1 if j < r else 0)
        if size:
            out.append(items[i:i + size])
        i += size
    return out


def _done_days(out: Path) -> set[str]:
    """Days already merged into `out` (sidecar so re-runs skip finished days)."""
    f = out.with_suffix(out.suffix + ".days.json")
    if not f.is_file():
        return set()
    try:
        return set(json.loads(f.read_text()).get("days", []))
    except (OSError, ValueError):
        return set()


def _mark_day(out: Path, date: str) -> None:
    f = out.with_suffix(out.suffix + ".days.json")
    days = _done_days(out)
    days.add(date)
    f.write_text(json.dumps({"days": sorted(days)}, indent=2))


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
    dates: Annotated[list[str], typer.Argument(help="one or more YYYY-MM-DD days from `list` (or 'all')")],
    out: Annotated[Path, typer.Option(help="output parquet")] = dd.DATA_DIR / "decks_with_outcomes.parquet",
    decisive_only: Annotated[bool, typer.Option(help="keep only games with a {+1,-1} result")] = True,
    workers: Annotated[int, typer.Option(help="parallel parse workers")] = DEFAULT_WORKERS,
    redo: Annotated[bool, typer.Option(help="re-ingest days even if already merged (ignore sidecar)")] = False,
    keep_zip: Annotated[bool, typer.Option(help="keep the downloaded zips instead of deleting")] = False,
    keep_extracted: Annotated[bool, typer.Option(help="keep the unzipped JSON on disk after parsing")] = False,
    limit: Annotated[int, typer.Option(help="cap episodes per day (0 = all), for quick tests")] = 0,
    cache: Annotated[Path, typer.Option(help="where to stash zips while working")] = dd.KAGGLE_DIR / "episodes_daily" / "_zips",
    extract_root: Annotated[Path, typer.Option(help="scratch dir for unzipped JSON (per-day, deleted after)")] = dd.KAGGLE_DIR / "episodes_daily" / "_extracted",
) -> None:
    """Download + **disk-extract + parallel-parse** days into decks_with_outcomes.

    Per day: download the compressed zip (reusing a cached one if present),
    unzip every episode JSON to disk, parse them across ``--workers`` cores
    (decompression is single-threaded, so we shred from disk not from a stream),
    merge the day's rows into ``out``, record the day in a sidecar, then delete
    the day's scratch files. Resumable: re-runs skip days already in the sidecar,
    so it also picks up NEW days. Pass ``all`` to do every published day.
    """
    if len(dates) == 1 and dates[0].lower() == "all":
        dates = load_manifest().sort("date", descending=True)["date"].to_list()
    done = set() if redo else _done_days(out)
    todo = [d for d in dates if d not in done]
    console.print(f"[bold]{len(dates)}[/bold] requested · [green]{len(done & set(dates))}[/green] "
                  f"already done · [yellow]{len(todo)}[/yellow] to build "
                  f"· {workers} workers")
    if not todo:
        console.print("[green]nothing to do — all requested days already merged.[/green]")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    per_day: list[tuple[str, int, int]] = []                # (date, episodes, rows_kept)

    for i, date in enumerate(todo, 1):
        console.rule(f"[bold cyan]{date}  ({i}/{len(todo)})")
        # reuse a cached zip if we already have one for this day
        cached = sorted(cache.glob(f"*{date}*.zip"))
        if cached:
            zip_path = cached[-1]
            console.print(f"[dim]reusing cached zip {zip_path.name}[/dim]")
        else:
            with console.status(f"[cyan]downloading {date} (compressed zip)…"):
                zip_path = download_day_zip(date, cache)

        exdir = extract_root / date
        if exdir.exists():
            shutil.rmtree(exdir)
        exdir.mkdir(parents=True)
        with console.status(f"[cyan]unzipping {date} to disk…"):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(exdir)
        files = sorted(str(p) for p in exdir.rglob("*.json"))
        if limit:
            files = files[:limit]

        rows: list[dd.DeckRow] = []
        batches = _chunks(files, workers * 4)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task(f"parsing {date} ({len(files):,} eps, {workers}w)", total=len(batches))
            with mp.Pool(workers) as pool:
                for chunk_rows in pool.imap_unordered(
                        _parse_files, [(b, decisive_only) for b in batches]):
                    rows.extend(chunk_rows)
                    prog.advance(task)

        # merge this day into out, then mark it done (so Ctrl-C keeps finished days)
        df_day = dd.rows_to_frame(rows)
        if out.is_file():
            df_day = pl.concat([pl.read_parquet(out), df_day]).unique()
        df_day.sort("episode_id", "player", "card_id").write_parquet(out)
        _mark_day(out, date)
        per_day.append((date, len(files), len(rows)))
        console.print(f"[green]{date}: {len(files):,} episodes → {len(rows):,} rows[/green] "
                      f"(table now {df_day.height:,} rows)")

        if not keep_extracted:
            shutil.rmtree(exdir, ignore_errors=True)
        if not keep_zip:
            zip_path.unlink(missing_ok=True)

    for scratch in (cache, extract_root):
        if scratch.is_dir() and not any(scratch.iterdir()):
            scratch.rmdir()

    t = Table(title="build summary", header_style="bold")
    for c in ("day", "episodes", "rows kept"):
        t.add_column(c, justify="right")
    for date, seen, kept in per_day:
        t.add_row(date, f"{seen:,}", f"{kept:,}")
    console.print(t)
    df = pl.read_parquet(out)
    console.print(f"[green]wrote[/green] {out}  ·  {df.height:,} rows · "
                  f"{df.select('episode_id').n_unique():,} episodes · "
                  f"{df.select('episode_id', 'player').unique().height:,} decks")
    console.print("[dim]full game state (all 12 tables), not just decks? "
                  "→ `python -m replaydb daily ingest`[/dim]")


if __name__ == "__main__":
    app()
