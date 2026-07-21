# Image-based PTCG replay viewer (`replay/07_vite_react_cards`)

**Date:** 2026-07-19
**Status:** Approved design, pre-implementation
**Author:** brainstorming session

## Problem

The existing replay viewer (`replay/05_vite_react_app`) renders each card as a
**text name** (`<div className="card-name">{name}</div>`). We now have the real
card art (1267 PNGs downloaded into `pkm_data/replay/cards/`, plus a fetch script)
and a hand-built board mockup (`replay/06_copy/board_layout.html`) that the user
approved as the target look. We want a replay viewer that renders that mockup's
board — real card art, correct zone arrangement — driven by live replay data.

## Goals

- A **fork** of `05_vite_react_app` at `replay/07_vite_react_cards`; `05` stays
  untouched as the text-only viewer.
- Cards render as **real art** with HP/energy/tag overlays, matching
  `board_layout.html`.
- Card images come from a **pluggable backend**: local files first, CDN fallback.
- The **whole-board arrangement** matches the mockup (see Layout), not 05's layout.
- A **hidden-info toggle** (realistic card-backs ↔ full-info reveal).
- Everything else 05 does is preserved: step parsing, timeline scrubber, event
  log, cumulative stats, diff flashes, keyboard controls, flexible replay loading.

## Non-goals (YAGNI)

- No changes to `05_vite_react_app`, the engine, or replay-log parsing logic.
- No new data pipeline: `stepState.ts`, `cardDb.ts`, `energy.ts`, `events.ts`
  carry over verbatim.
- No card-back PNG asset (drawn in CSS).
- Committing/pushing the 137 MB of PNGs to the `pkm_data` HF LFS submodule is a
  **separate, user-gated step** — not part of this implementation.

## Architecture

Fork inherits 05's structure:

```
replay/07_vite_react_cards/
├── public/
│   ├── replay.json        -> symlink ../../replay.json   (as in 05)
│   ├── cards.json         -> symlink ../../cards.json     (as in 05)
│   └── cards/             -> symlink ../../../pkm_data/replay/cards  (NEW)
├── src/
│   ├── data/
│   │   ├── cardArt.ts     (NEW — image URL resolver + backend logic)
│   │   ├── cardDb.ts      (unchanged from 05)
│   │   ├── stepState.ts   (unchanged)
│   │   ├── energy.ts      (unchanged)
│   │   └── events.ts      (unchanged)
│   ├── components/
│   │   ├── Card.tsx       (MODIFIED — art face + overlays + backs)
│   │   ├── PlayerBoard.tsx(REWRITTEN — mockup arrangement)
│   │   ├── Board.tsx      (MODIFIED — center stadium strip, mirrored sides)
│   │   ├── Timeline.tsx / LogPanel.tsx / StatsPanel.tsx / DiffPanel.tsx (unchanged)
│   │   └── Header controls (MODIFIED — backend + reveal toggles)
│   ├── state/            (MODIFIED — add backend + reveal UI state)
│   └── styles.css        (MODIFIED — art cards, backs, board grid)
└── vite.config.ts / package.json / etc. (copied; VITE_CARD_* added)
```

### Component 1 — `src/data/cardArt.ts` (image backend)

Single responsibility: given a `card_id` and the current backend, produce image
URL(s) and manage fallback.

- `localUrl(id) => "/cards/<id>.png"` (Vite serves `public/cards/`).
- `cdnUrl(id) => "https://ptcgvis.heroz.jp/img/<album>/<id>.png"`.
  - `album` default `"bqucewmzuceknw"`, overridable via `import.meta.env.VITE_CARD_ALBUM`.
- **3-tier fallback**, implemented in the `<img>` render path:
  1. `src` = primary backend URL (local by default).
  2. `onError` → swap `src` to the *other* backend URL once. A module-level
     `Set<number>` of ids that already failed-over prevents infinite loops.
  3. Both fail → component renders the **text-name card** (05's current markup)
     as the final fallback. Never show a broken-image icon.
- Backend precedence: runtime toggle (state) > `VITE_CARD_BACKEND` env
  (`local`|`cdn`, default `local`).

**Interface:** `resolveCardArt(id, backend) => { primary: string; fallback: string }`
plus a React helper/hook for the `<img>` + error handling. Consumers (`Card.tsx`)
never touch URLs directly.

### Component 2 — `src/components/Card.tsx` (art rendering)

The card art is the face; overlays sit on top (as in `board_layout.html`):

- `<img>` art via the cardArt resolver, sized by `variant`
  (`active`|`bench`|`hand`|`mini`).
- Overlays: HP badge (red when `hpDelta < 0` or below threshold), energy pips
  (existing `EnergyPips`), `ex`/status tags.
- Diff flashes (`flash-dmg`/`flash-heal`/`flash-new`) → border/glow on the art.
- Hover popover (attacks/cost/weakness/retreat from `cards.json`) preserved.
- **Card back** variant: a CSS-drawn blue diagonal pattern (from the mockup) for
  hidden cards — no image request.

### Component 3 — Board layout (`PlayerBoard.tsx` + `Board.tsx`)

Port `board_layout.html`'s arrangement. Per player, a 3-column band
`[prizes] [field] [deck/discard]`:

- **Prizes**: 2×3 grid of card-backs (taken prizes render as empty/greyed).
- **Field** (vertical): bench row **behind** the active, active in a gold frame;
  the two players are **mirrored** (opponent bench-then-active top-down, you
  active-then-bench). Bench shows filled slots + dashed empty slots up to
  `benchMax`.
- **Deck/Discard**: a vertical pile column; deck = back, discard = top-card art.
- **Hand**: fanned, overlapping cards (`margin-left` negative; hover lifts).
- **Stadium**: a dedicated shared strip between the two sides (`Board.tsx`).

Board data comes from `mergeStep` (unchanged): player *i*'s slice from entry *i*.

### Component 4 — UI state & toggles (`src/state/`, header)

Two new pieces of UI state, alongside 05's existing step/play state:

- `cardBackend: "local" | "cdn"` — header toggle; initialised from env.
- `revealHidden: "realistic" | "full-info"` — header toggle.
  - `realistic`: opponent hand + prizes + deck as card-backs.
  - `full-info`: opponent hand revealed as art (05's spy-view merge); prizes/deck
    stay counts + backs in both modes.

### Component 5 — fetch script + `just` target

`replay/fetch_card_images.py` already exists, is tested, and populated
`pkm_data/replay/cards` (1267/1267). It is the "populate local backend" tool:
fetch-if-missing (skips existing), `--album` overridable, writes `index.json`.
Add a `just fetch-cards` target wrapping it and document in the app README.

## Data flow

```
replay.json ──parse(stepState)──▶ merged step snapshot
                                        │
                        ┌───────────────┼────────────────┐
                        ▼               ▼                ▼
                   PlayerBoard      LogPanel         StatsPanel
                        │
                    Card.tsx ──id──▶ cardArt.resolve(id, backend)
                        │                    │
                        │            local /cards/<id>.png
                        │              └─onError─▶ cdn heroz url
                        │                          └─onError─▶ text name
                        ▼
                   cardDb.get(id) ──▶ name, attacks, hp, ex, ... (overlays + popover)
```

## Error handling

- **Missing local image** → automatic CDN fallback (self-heals gaps in the
  downloaded set).
- **CDN unreachable / offline** → text-name card; app stays fully functional
  offline as long as local files exist (they do: 1267/1267).
- **Unknown card id** (not in `cards.json`) → `#<id>` label, no crash (05 behaviour).
- **Malformed replay** → 05's `ErrorBoundary` carries over.

## Testing

- Reuse 05's data-layer tests (stepState/cardDb) unchanged.
- New unit test for `cardArt.resolveCardArt`: backend precedence + fallback URL
  construction (`local` → `/cards/121.png`, `cdn` → correct heroz URL, album
  override).
- Manual verification: load default `replay.json`, scrub steps, confirm art
  renders, toggle backend (local↔cdn), toggle reveal (realistic↔full-info),
  confirm offline mode (block network → local still renders, unknown id → text).
- Screenshot the running app (headless Chrome, as done for the mockup) as the
  completion artifact.

## Success criteria

1. `replay/07_vite_react_cards` runs (`bun run dev`) independently of 05.
2. Board matches `board_layout.html`'s arrangement, driven by real replay data.
3. Cards show real art with correct HP/energy/tag overlays; hover popover works.
4. Local-first with automatic CDN fallback; header toggles for backend + reveal.
5. Fully offline with the downloaded card set; graceful text fallback otherwise.
6. `05_vite_react_app` unchanged and still works.
