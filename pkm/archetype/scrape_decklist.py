"""Scrape a single limitlesstcg.com tournament decklist page
(https://limitlesstcg.com/decks/list/<id>) into entries.json shape, then
build a legal deck/pool_<archetype_id>_<slug>.csv the same way
build_pool_deck.py does. Part of Part 3a's real-decklist sourcing (see
docs/opponent-archetype-classifier-plan.md Part 3a).

Page structure (confirmed by fetching a live page): each card is
    <div class="decklist-card" data-set="TEF" data-number="123" ...>
        <span class="card-count">2</span>
        <span class="card-name">Raging Bolt ex</span>
    </div>
Basic energies use the engine's actual set/number ("MEE"/"1".."8"), which
pkm/archetype/aliases.py already maps to card_ids 1-8 -- no special-casing
needed here.

Usage:
    python -m pkm.archetype.scrape_decklist <limitlesstcg_url> <archetype_id> <slug> [out_dir]
"""

import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from pkm.archetype.build_pool_deck import build_deck, write_pool_deck

USER_AGENT = "Mozilla/5.0 (compatible; pkm-archetype-scraper/1.0)"


def fetch_entries(url: str) -> list[dict]:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    entries: list[dict] = []
    for card_div in soup.select("div.decklist-card"):
        count_span = card_div.select_one("span.card-count")
        name_span = card_div.select_one("span.card-name")
        if count_span is None or name_span is None:
            continue
        entries.append(
            {
                "name": name_span.get_text(strip=True),
                "set": card_div.get("data-set", ""),
                "number": card_div.get("data-number", ""),
                "count": int(count_span.get_text(strip=True)),
            }
        )
    return entries


def main() -> None:
    url, archetype_id, slug = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("deck")

    entries = fetch_entries(url)
    if not entries:
        raise SystemExit(f"no cards found on {url} -- page structure may have changed")

    total = sum(e["count"] for e in entries)
    if total != 60:
        print(f"NOTE: scraped {total} cards (before padding/trimming), expected 60")

    card_ids, notes = build_deck(entries)
    out_path, notes = write_pool_deck(card_ids, notes, archetype_id, slug, out_dir)

    print(f"wrote {out_path} ({len(card_ids)} cards) from {url}")
    for note in notes:
        print(f"  NOTE: {note}")


if __name__ == "__main__":
    main()
