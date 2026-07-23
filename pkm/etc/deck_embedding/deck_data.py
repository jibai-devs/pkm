# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars",
# ]
# ///
"""Bridge: reconstruct decks from `db/` and join them to game outcomes.

The deck a player registered is `steps[1].action` -- a 60-element list of
card_ids (see tools/build_parquet.py). In the shredded database that is the
`actions` table at `step == 1`; the outcome is `db/games.reward{0,1}`
(+1 win / -1 loss). This is the `db -> decks` bridge flagged in HANDOFF #7.4,
and it is the input every deck-embedding method needs.

Output `decks_with_outcomes.parquet`, long format, one row per card per player
per episode -- exactly what `deck_embed.load_matchups_from_parquet` reads::

    episode_id: i64   player: i8   card_id: i64   count: i64   won: bool

The source `db/` lives in the kaggle-replay crawl dir; the derived parquet is
written next to the other deck-embedding artifacts in that dir's
`deck_embedding/` subfolder. Both are resolved relative to this file's location
(`DB_DIR` / `DATA_DIR`), so the script works from any working directory.

Importable::

    from deck_data import build_decks_with_outcomes
    df = build_decks_with_outcomes()        # uses DB_DIR / DATA_DIR defaults

or run standalone::

    uv run --script deck_data.py            # writes deck_embedding/decks_with_outcomes.parquet
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import polars as pl

# This file now lives at <repo>/pkm/etc/deck_embedding/; the crawl data it reads
# lives at <repo>/pkm_data/kaggle_replays/pokemon-tcg-ai-battle/. Anchor both to
# __file__ so the paths hold regardless of the working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
KAGGLE_DIR = REPO_ROOT / "pkm_data" / "kaggle_replays" / "pokemon-tcg-ai-battle"
DB_DIR = KAGGLE_DIR / "db"                          # source shredded database
DATA_DIR = KAGGLE_DIR / "deck_embedding"            # derived artifacts live here

# Long-format schema every deck-embedding consumer expects.
OUTCOMES_SCHEMA = {
    "episode_id": pl.Int64,
    "player": pl.Int8,
    "card_id": pl.Int32,
    "count": pl.UInt32,
    "won": pl.Boolean,
}


def build_decks_with_outcomes(db: str | Path = DB_DIR,
                              out: str | Path | None = DATA_DIR / "decks_with_outcomes.parquet",
                              decisive_only: bool = True) -> pl.DataFrame:
    """Reconstruct 60-card decks from db/actions and join db/games outcomes.

    - Keeps only episodes where BOTH players have a valid 60-card deck.
    - ``decisive_only`` drops draws / missing rewards (keeps games whose two
      rewards are exactly {+1, -1}), so ``won`` is unambiguous.
    Returns the long-format DataFrame and, unless ``out`` is None, writes it.
    """
    db = Path(db)

    # --- decks: actions at step 1, one row per (episode, player, card) with count
    decks = (
        pl.scan_parquet(db / "actions" / "*.parquet")
        .filter(pl.col("step") == 1)
        .group_by("episode_id", "player", "value")
        .agg(pl.len().alias("count"))
        .rename({"value": "card_id"})
    )

    # keep only (episode, player) whose copies sum to exactly 60
    valid_pp = (
        decks.group_by("episode_id", "player")
        .agg(pl.col("count").sum().alias("deck_size"))
        .filter(pl.col("deck_size") == 60)
    )
    # ...and only episodes where BOTH players are valid
    valid_ep = (
        valid_pp.group_by("episode_id")
        .agg(pl.len().alias("n_players"))
        .filter(pl.col("n_players") == 2)
        .select("episode_id")
    )
    decks = decks.join(valid_ep, on="episode_id", how="inner")

    # --- outcomes: games.reward0/1 -> long (episode, player, reward)
    games = pl.scan_parquet(db / "games" / "*.parquet").select("episode_id", "reward0", "reward1")
    outcomes = pl.concat([
        games.select("episode_id", pl.lit(0, dtype=pl.Int8).alias("player"),
                     pl.col("reward0").alias("reward")),
        games.select("episode_id", pl.lit(1, dtype=pl.Int8).alias("player"),
                     pl.col("reward1").alias("reward")),
    ])

    if decisive_only:
        # both rewards present and exactly one winner
        decisive = (
            games.filter(pl.col("reward0").is_not_null() & pl.col("reward1").is_not_null())
            .filter((pl.col("reward0") + pl.col("reward1")) == 0)
            .filter(pl.col("reward0") != 0)
            .select("episode_id")
        )
        outcomes = outcomes.join(decisive, on="episode_id", how="inner")

    df = (
        decks.join(outcomes, on=["episode_id", "player"], how="inner")
        .with_columns((pl.col("reward") > 0).alias("won"))
        .select("episode_id", pl.col("player").cast(pl.Int8), "card_id", "count", "won")
        .sort("episode_id", "player", "card_id")
        .collect()
    )

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
    return df


DeckRow = tuple[int, int, int, int, bool]  # (episode_id, player, card_id, count, won)


def episode_deck_rows(episode: dict, decisive_only: bool = True) -> list[DeckRow]:
    """Extract decks-with-outcomes rows from ONE Kaggle episode dict.

    The single source of truth for the JSON → (episode_id, player, card_id,
    count, won) mapping, shared by the file-based and streaming builders. Reads
    only the 60-card registered decks at ``steps[1][p].action`` and the terminal
    ``rewards`` — nothing else — so it's cheap to run per episode.

    Returns ``[]`` (episode skipped) when either player lacks a valid 60-card
    deck, or — with ``decisive_only`` — when the two rewards aren't exactly
    {+1, -1}.
    """
    rewards = (list(episode.get("rewards") or []) + [None, None])[:2]
    r0, r1 = rewards
    if decisive_only and (r0 is None or r1 is None or (r0 + r1) != 0 or r0 == 0):
        return []

    # both players must have registered a valid 60-card deck at step 1
    step1 = episode["steps"][1]
    decks_pp: dict[int, collections.Counter] = {}
    for p in (0, 1):
        a = step1[p].get("action")
        if isinstance(a, list) and len(a) == 60:
            decks_pp[p] = collections.Counter(a)
    if len(decks_pp) != 2:
        return []

    eid = int(episode["info"]["EpisodeId"])
    rows: list[DeckRow] = []
    for p, cnt in decks_pp.items():
        won = rewards[p] is not None and rewards[p] > 0
        for cid, n in sorted(cnt.items()):
            rows.append((eid, p, int(cid), int(n), bool(won)))
    return rows


def rows_to_frame(rows: list[DeckRow]) -> pl.DataFrame:
    """Assemble accumulated deck rows into the canonical long-format frame."""
    cols = list(zip(*rows)) if rows else ([], [], [], [], [])
    return pl.DataFrame(
        {"episode_id": cols[0], "player": cols[1], "card_id": cols[2],
         "count": cols[3], "won": cols[4]},
        schema=OUTCOMES_SCHEMA,
    ).sort("episode_id", "player", "card_id")


def build_decks_with_outcomes_from_replays(
    replays: str | Path,
    out: str | Path | None = DATA_DIR / "decks_with_outcomes.parquet",
    decisive_only: bool = True,
) -> pl.DataFrame:
    """Build the decks-with-outcomes table straight from episode replay JSONs.

    Each file under ``replays`` is one Kaggle episode — the format published in
    the official ``pokemon-tcg-ai-battle-episodes-<date>`` datasets, byte-shape
    identical to the crawler's ``replays/``. This is the no-crawl path: point it
    at a downloaded daily dataset instead of reconstructing from a shredded db/.
    Reads one file at a time (see ``episode_deck_rows``), so memory stays flat.
    """
    replays = Path(replays)
    rows: list[DeckRow] = []
    for f in sorted(replays.glob("*.json")):
        with open(f) as fh:
            rows.extend(episode_deck_rows(json.load(fh), decisive_only))

    df = rows_to_frame(rows)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
    return df


def _main() -> None:
    df = build_decks_with_outcomes()
    n_eps = df.select("episode_id").n_unique()
    n_decks = df.select("episode_id", "player").unique().height
    wins = df.select("episode_id", "player", "won").unique()["won"]
    print(f"wrote decks_with_outcomes.parquet: {df.height:,} rows")
    print(f"  episodes : {n_eps:,}")
    print(f"  decks    : {n_decks:,}  (won: {int(wins.sum()):,} / lost: {int((~wins).sum()):,})")


if __name__ == "__main__":
    _main()
