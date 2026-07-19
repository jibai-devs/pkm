# Card images — how the PTCG visualizer serves art

The Kaggle visualizer (`ptcgvis.heroz.jp`) renders each card face from a static
PNG endpoint. The saved replay in `06_copy/` only ships the HTML + JS bundle, so
the art shows as black squares offline. This is how to fetch the real images.

## URL pattern

```
https://ptcgvis.heroz.jp/img/<album>/<card_id>.png
```

- `<album>` — an opaque path segment (currently `bqucewmzuceknw`). It is **not**
  per-card; the same album serves every card. Treat it as a constant that the
  site may rotate; if downloads start 404ing, grab a fresh one from your browser
  devtools Network tab on any replay page and pass it via `--album`.
- `<card_id>` — the integer `card_id` from `replay/cards.json`. IDs are
  **contiguous 1–1267** (1267 cards total: energies, Pokémon, trainers, stadiums).

Example (verified: `200`, `image/png`, ~84 KB):

```
https://ptcgvis.heroz.jp/img/bqucewmzuceknw/1231.png
```

## Request headers that matter

The endpoint checks the `Referer` (and a normal browser `User-Agent`). A request
with no `Referer` may be rejected. Use:

```
Referer:    https://ptcgvis.heroz.jp/Visualizer/Replay/<any-replay-id>/0
User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36
```

## Rules of the road

- **Only GET, only images.** No auth, no cookies, no query params.
- **Be polite:** one request at a time with a small delay (default 150 ms) and a
  low retry count. This is someone else's static host, not an API — do not
  parallel-blast 1267 requests.
- **Resume-friendly:** skip files already on disk so re-runs are cheap.
- **Licensing:** the art is © Pokémon/Nintendo/Creatures/GAME FREAK. The site
  asks that images not be reprinted. Keep downloads for **local, personal use**
  (e.g. rendering your own replay viewer) — do not redistribute.

## Usage

```bash
# all 1267 cards → replay/card_images/
python3 replay/fetch_card_images.py

# a subset / custom album / custom out dir
python3 replay/fetch_card_images.py --ids 1231 1079 120 --out /tmp/cards
python3 replay/fetch_card_images.py --start 1 --end 100 --album bqucewmzuceknw
python3 replay/fetch_card_images.py --delay 0.3 --retries 5
```

Output: `<out>/<card_id>.png` plus an `index.json` mapping id → name (from
`cards.json`) for the ids that downloaded successfully.
