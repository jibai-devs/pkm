#!/usr/bin/env python3
"""Download PTCG card face images from the Kaggle visualizer's static host.

The visualizer serves each card face as a PNG at:

    https://ptcgvis.heroz.jp/img/<album>/<card_id>.png

Card ids come from ``replay/cards.json`` (contiguous 1..1267). See
``replay/CARD_IMAGES.md`` for the full rules. Stdlib only (urllib), so no deps.

Politeness: one request at a time, a small delay between requests, bounded
retries, and it skips files already on disk so re-runs are cheap. The art is
© Pokémon/Nintendo/Creatures/GAME FREAK — keep it local/personal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ALBUM = "bqucewmzuceknw"
URL_TEMPLATE = "https://ptcgvis.heroz.jp/img/{album}/{card_id}.png"

# The endpoint checks Referer + a browser-like User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
    "Referer": "https://ptcgvis.heroz.jp/Visualizer/Replay/86895993/0",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
}

REPLAY_DIR = Path(__file__).resolve().parent


def load_card_names(cards_json: Path) -> dict[int, str]:
    """Map card_id -> name from cards.json (best-effort; empty on failure)."""
    try:
        data = json.loads(cards_json.read_text())
    except (OSError, ValueError):
        return {}
    return {c["card_id"]: c.get("name", "") for c in data.get("cards", [])}


def resolve_ids(args: argparse.Namespace, names: dict[int, str]) -> list[int]:
    if args.ids:
        return list(args.ids)
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 1
        end = args.end if args.end is not None else (max(names) if names else start)
        return list(range(start, end + 1))
    if names:
        return sorted(names)
    # Fallback if cards.json is missing: known contiguous range 1..1267.
    return list(range(1, 1268))


def fetch_one(card_id: int, album: str, retries: int, timeout: float) -> bytes | None:
    """Fetch a single card PNG. Returns bytes, or None on 404/permanent failure."""
    url = URL_TEMPLATE.format(album=album, card_id=card_id)
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                ctype = resp.headers.get("Content-Type", "")
                if not ctype.startswith("image/") or not body:
                    print(f"  ! id={card_id} unexpected content-type={ctype!r}")
                    return None
                return body
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None  # no image for this id; not an error worth retrying
            print(f"  ! id={card_id} HTTP {exc.code} (attempt {attempt}/{retries})")
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  ! id={card_id} {exc} (attempt {attempt}/{retries})")
        if attempt < retries:
            time.sleep(min(2.0 * attempt, 5.0))  # simple backoff
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--album", default=DEFAULT_ALBUM,
                   help=f"album path segment (default: {DEFAULT_ALBUM})")
    p.add_argument("--out", type=Path, default=REPLAY_DIR / "card_images",
                   help="output directory (default: replay/card_images)")
    p.add_argument("--cards-json", type=Path, default=REPLAY_DIR / "cards.json",
                   help="cards.json for id->name (default: replay/cards.json)")
    p.add_argument("--ids", type=int, nargs="+", help="explicit card ids to fetch")
    p.add_argument("--start", type=int, help="start id (inclusive)")
    p.add_argument("--end", type=int, help="end id (inclusive)")
    p.add_argument("--delay", type=float, default=0.15,
                   help="seconds between requests (default: 0.15)")
    p.add_argument("--retries", type=int, default=3, help="retries per id (default: 3)")
    p.add_argument("--timeout", type=float, default=30.0, help="per-request timeout secs")
    p.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = p.parse_args(argv)

    names = load_card_names(args.cards_json)
    ids = resolve_ids(args, names)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(ids)} card image(s) → {args.out}  (album={args.album})")
    downloaded, skipped, missing = 0, 0, []
    index: dict[str, str] = {}

    for i, card_id in enumerate(ids, 1):
        dest = args.out / f"{card_id}.png"
        name = names.get(card_id, "")
        if dest.exists() and not args.force:
            skipped += 1
            if name:
                index[str(card_id)] = name
            continue

        body = fetch_one(card_id, args.album, args.retries, args.timeout)
        if body is None:
            missing.append(card_id)
        else:
            dest.write_bytes(body)
            downloaded += 1
            if name:
                index[str(card_id)] = name
            label = f" ({name})" if name else ""
            print(f"[{i}/{len(ids)}] id={card_id}{label}  {len(body):,} bytes")

        time.sleep(args.delay)

    # Merge id->name index for everything present on disk.
    index_path = args.out / "index.json"
    try:
        if index_path.exists():
            existing = json.loads(index_path.read_text())
            existing.update(index)
            index = existing
    except ValueError:
        pass
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, "
          f"{len(missing)} missing. Index → {index_path}")
    if missing:
        preview = ", ".join(map(str, missing[:20]))
        more = " …" if len(missing) > 20 else ""
        print(f"Missing ids ({len(missing)}): {preview}{more}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
