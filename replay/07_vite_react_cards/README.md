# PTCG Replay Viewer (Vite + React + TS)

Step-by-step replay viewer for Pokemon TCG matches (Kaggle Simulation format),
implementing Option 5 from `../ideas_and_recommendations.md`.

## Run

```bash
bun install
bun run dev          # http://localhost:5175
# or: bun run build && bun run preview
```

`replay.json` and `cards.json` are symlinked into `public/` (from `../`), so the
viewer loads them over `fetch` with no server. Fully offline.

### Loading a different replay

Precedence: **file picker > `?replay=` > `VITE_REPLAY` > default `/replay.json`**.

- **File picker / drag-drop** — click "Load replay…" (top-right) or drop a `.json`
  onto the page. Reads **any file on disk** via `FileReader`; works in the built
  app too. The only option for files the dev server doesn't serve.
- **Query param** — `?replay=/other.json` (or `?file=`). Shareable, no restart;
  must be a path the dev server serves (under `public/`) or a URL.
- **Env var (CLI)** — `VITE_REPLAY=/other.json bun run dev`, or
  `just replay-react file=/other.json`. Same fetch-reachability constraint.
  (Vite doesn't expose shell env to the client by default, so `vite.config.ts`
  injects it via `define` as `__REPLAY_URL__`.) `VITE_CARDS` overrides the card DB.

## Features

- Forward/back, play/pause, first/last, and a timeline scrubber (`Timeline`)
- Speeds 0.5×–8×; keyboard: `←`/`→` step, `Space` play/pause, `Home`/`End` jump
- Full-information board: **both** players' hands are shown (see merge note below)
- Per-card HP bars, energy pips, status chips; hover a card for a detail popover
  (attacks, cost, weakness, retreat) resolved from `cards.json`
- Event log per step, color-coded by kind
- Cumulative match stats (damage, attacks, cards played, prizes)
- Diff highlighting: damaged/healed/new cards flash; a Diff panel lists changes
- `?step=N` deep-links (1-based) and stays in sync with the URL

## How the data actually works (verified against the real files)

- `replay.json` ≈ 19.6 MB, **284 steps**; each `steps[n]` is `[entry_p0, entry_p1]`.
- The rich per-step board is at **`entry.observation.current`** (NOT a `visualize`
  array as the old `requirements.md` says). Each step is a **full materialized
  snapshot**, so backward/scrub/jump are pure index reads — no log replay.
- Logs are at **`entry.observation.logs`** and use **numeric `type` codes**.
- Each entry only reveals **its own** player's hand. To show both, `mergeStep`
  takes player *i*'s slice from entry *i* (see `src/data/stepState.ts`).
- Replay card objects carry only dynamic state (`id`, `serial`, `hp`, `energies`,
  ...) and **not even the name** — everything human-readable comes from
  `cards.json` keyed by `id` (`src/data/cardDb.ts`).
- Energy-type codes (calibrated from the "Basic {X} Energy" cards):
  `0=C 1=G 2=R 3=W 4=L 5=P 6=F 7=D 8=M` (`src/data/energy.ts`).

### Log type codes — best-effort

The engine (binary `cabt`) ships no public enum, so `src/data/events.ts` maps
numeric log types by field signature. Confident: `1` HasBasicPokemon, `6/7`
MoveCard, `8` Switch, `15` Attack, `16` HpChange. Uncertain ones (`0/2/3` phase,
`4/10` reveal/play, `11/12` ability/evolve) are rendered in an italic "guess"
style and fall back to raw JSON for anything unmapped — the log never lies.

## Structure

```
src/
  data/      types, loaders, stepState (POV merge), diff, stats, events, energy
  state/     usePlayback hook (position + auto-advance)
  components/ Board, PlayerBoard, Card, Timeline, LogPanel, StatsPanel, DiffPanel
  App.tsx    load -> compute stats -> wire playback + panels
```
