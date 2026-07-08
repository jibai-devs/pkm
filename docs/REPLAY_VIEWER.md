# Replay Viewer

Step-by-step viewer for Pokemon TCG match replays (Kaggle Simulation format).
Two implementations live under `replay/`; the **React/TS** one
(`replay/05_vite_react_app/`) is the maintained viewer.

## Quick start

```bash
just replay-react                    # React/TS viewer  -> http://localhost:5175
just replay                          # older vanilla-JS viewer (02_vite_web_app)
```

Or directly:

```bash
cd replay/05_vite_react_app
bun install
bun run dev                          # dev server (hot reload)
bun run build && bun run preview     # production build, then serve dist/
```

`replay/replay.json` and `replay/cards.json` are symlinked into the app's
`public/`, loaded over `fetch` — no backend, fully offline.

## Generating a replay to view

```bash
just play                            # play one match -> writes replay.json + result.html
```

## Loading a different replay

Precedence: **file picker > `?replay=` > `VITE_REPLAY` > default `/replay.json`**.

| Way | How | Reaches |
|-----|-----|---------|
| File picker / drag-drop | click **"Load replay…"** (top-right) or drop a `.json` on the page | **any file on disk** (uses `FileReader`; works in the built app too) |
| URL query param | `http://localhost:5175/?replay=/foo.json` (or `?file=`) | files the dev server serves (under `public/`) or a URL |
| Env var / CLI | `just replay-react file=/foo.json` — or `VITE_REPLAY=/foo.json bun run dev` | same fetch-reachability constraint |

Env var caveat: the browser can only `fetch` served files, so a bare path must
live under the app's `public/`. For arbitrary files anywhere on disk, use the
in-app picker. `VITE_CARDS` overrides the card DB the same way.

Bonus: `?step=N` deep-links to a step (1-based) and the URL stays in sync as you
navigate — so the current position is shareable and survives a reload.

## Controls

| | |
|-|-|
| Play / pause | **Space**, or the ▶/⏸ button |
| Step back / forward | **← / →**, or ◀ / ▶ |
| First / last step | **Home / End**, or ⏮ / ⏭ |
| Jump anywhere | drag the timeline scrubber |
| Speed | 0.5×–8× buttons |
| Card details | hover a card (attacks, cost, weakness, retreat) |

## What it shows

Full-information board (both players' hands revealed), per-card HP bars, energy
pips and status; discard pile (newest on top); a per-step event log; cumulative
match stats; and diff highlighting (damaged/healed/new cards flash, plus a Diff
panel). Game-over shows the winner; the first step is the pre-game setup.

## Data notes (for maintainers)

- Rich per-step state is at `steps[n][player].observation.current` — each step is
  a **full snapshot**, so navigation is pure indexing (no log replay).
- Logs are at `observation.logs` with **numeric type codes**; some are decoded
  best-effort (the engine ships no enum) and shown in an italic "guess" style.
- Replay card objects carry only dynamic state (`id`, `hp`, `energies`, …) — even
  the **name** comes from `cards.json`, keyed by `id`.
- See `replay/05_vite_react_app/README.md` for the full data contract, the POV
  merge, energy-code map, and code structure. Design/options:
  `replay/ideas_and_recommendations.md`. Backlog is tracked in the task list.
