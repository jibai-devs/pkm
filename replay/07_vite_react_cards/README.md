# PTCG Replay Viewer — image edition (07)

Fork of `../05_vite_react_app` that renders **real card art** instead of text
names, in the board arrangement of `../06_copy/board_layout.html`.

## Run
```bash
bun install
bun run dev            # http://localhost:5175   (or: just replay-cards)
bun run test           # vitest unit tests
```

## Card images (backend)
- **local** (default): `/cards/<id>.png`, served from `public/cards` → a symlink
  to `../../../pkm_data/replay/cards`. Populate it with `just fetch-cards`
  (downloads 1267 PNGs, skips existing). Fully offline.
- **cdn**: `https://ptcgvis.heroz.jp/img/<album>/<id>.png` — automatic fallback
  when a local file is missing; force it via the header **Art** toggle or
  `VITE_CARD_BACKEND=cdn`. Album override: `VITE_CARD_ALBUM=<album>`.
- If both fail, the card shows its text name (never a broken image).

## Toggles (header)
- **Art**: local ↔ cdn image source.
- **Hidden**: `realistic` (opponent hand / prizes / deck as card-backs) ↔
  `reveal all` (opponent hand shown as art).

Everything else — timeline, log, stats, diff, keyboard controls, replay loading —
is inherited from 05; see `../05_vite_react_app/README.md` for the data contract.
