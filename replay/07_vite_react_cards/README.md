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

## Interactive play (`?mode=play`)
This app doubles as a live game GUI: open `?mode=play` to play a real match
against a bot, reusing the same board components. It needs the Python play server
(`pkm/web/server.py`) running for its `/api` calls.

```bash
# from repo root:
just play-web-build         # build + serve UI+API at :8000, open /?mode=play
# or for hot-reload dev (two terminals):
just play-web               # Python API bridge on :8000
just play-web-dev           # this Vite dev server on :5175/?mode=play (proxies /api)
```

Play-mode code lives in `src/live/` (`api.ts` fetch client, `useLiveGame.ts`
state machine + long-poll loop, `liveStep.ts` obs→MergedStep adapter) plus
`src/components/PreGame.tsx` and `src/components/OptionsPane.tsx`. Option labels
are rendered server-side; see repo `AGENTS.md` → "Human Play (Browser / React
GUI)" for the full architecture.
