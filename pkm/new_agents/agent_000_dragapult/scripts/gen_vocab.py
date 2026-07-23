"""Idempotent generator for the card-embedding vocabulary snapshot.

The vocabulary (which card IDs get a learned embedding row, and the row each
maps to) is derived purely from the deck registry in ``deck.py`` — the union of
distinct card IDs over every registered deck, sorted, plus one UNK row. This
script materialises that derivation into an inspectable, committed
``vocab.json`` next to the agent package.

It is a **pure function of the deck definitions**: running it twice produces
byte-identical output. The network reads its shape from ``deck.py`` directly (not
this file), so ``vocab.json`` is a human-facing snapshot + drift guard, never a
runtime dependency. ``test_deck_vocab.py`` asserts the committed snapshot matches
what ``deck.py`` currently computes.

Run:  ``uv run python -m pkm.new_agents.agent_000_dragapult.scripts.gen_vocab``
      (add ``--check`` to verify the snapshot is up to date without writing).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pkm.new_agents.agent_000_dragapult import deck

VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.json"


def build_vocab() -> dict[str, Any]:
    """Compute the vocabulary snapshot from the deck registry (deterministic)."""
    return {
        # The one number that sizes the embedding table + hand-histogram width.
        "vocab_size": deck.VOCAB_SIZE,  # == n_unique + 1 (UNK)
        "n_unique": len(deck.DISTINCT_IDS),
        "unk_row": deck.UNK_ROW,
        "distinct_ids": deck.DISTINCT_IDS,
        # id -> (row, name), sorted by row, for eyeballing which card owns which row.
        "rows": [
            {"row": deck.ID_TO_ROW[cid], "id": cid, "name": deck.NAME_BY_ID.get(cid, "?")}
            for cid in deck.DISTINCT_IDS
        ],
        # Per-deck summary so the counts that size the NN are visible per source.
        "decks": {
            name: {
                "n_cards": sum(c for _i, _n, c in d),
                "n_distinct": len({cid for cid, _n, _c in d}),
                "ids": sorted(cid for cid, _n, _c in d),
            }
            for name, d in deck.DECKS.items()
        },
    }


def _serialise(vocab: dict[str, Any]) -> str:
    return json.dumps(vocab, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify vocab.json matches deck.py without writing; exit 1 on drift",
    )
    args = ap.parse_args()

    vocab = build_vocab()
    text = _serialise(vocab)

    if args.check:
        current = VOCAB_PATH.read_text(encoding="utf-8") if VOCAB_PATH.exists() else ""
        if current != text:
            raise SystemExit(
                f"{VOCAB_PATH.name} is stale — run gen_vocab to regenerate."
            )
        print(f"{VOCAB_PATH.name} is up to date (vocab_size={vocab['vocab_size']}).")
        return

    VOCAB_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {VOCAB_PATH}")
    print(
        f"vocab_size={vocab['vocab_size']} "
        f"(n_unique={vocab['n_unique']}, unk_row={vocab['unk_row']})"
    )
    for name, info in vocab["decks"].items():
        print(f"  {name:12s} {info['n_cards']} cards, {info['n_distinct']} distinct")


if __name__ == "__main__":
    main()
