# Replay Visualization — Ideas & Recommendations

## Approach Options

### Option 1: Single HTML file
**Directory:** `replay/01_single_html/`
**Status:** Not started

**Stack:** Vanilla JS + DOM, all in one `.html`

Pros:
- Zero build step — just open in browser
- Easy to share (single file, ~1000-2000 lines)
- No dependencies to install
- Works anywhere with a browser

Cons:
- Gets unwieldy past ~2000 lines
- No hot reload, slow iteration
- No component reuse
- No TypeScript, no linting
- Hard to maintain long-term

Best for: Quick prototype, one-off sharing

---

### Option 2: Vite + vanilla JS web app (RECOMMENDED)
**Directory:** `replay/02_vite_web_app/`
**Status:** Scaffolded (Bun + Vite + vanilla JS)

**Stack:** Vite + vanilla JS

Pros:
- Instant dev server with hot reload
- Component-based architecture
- Import replay.json via fetch
- Can bundle into single HTML later if needed
- Proper project structure, linting, formatting
- Separate from Python training code

Cons:
- Requires `npm install` / `npm run dev`
- Node.js dependency
- Slightly more setup than single file

Best for: A proper viewer you'll use repeatedly and extend over time

Suggested structure:
```
replay/02_vite_web_app/
  index.html
  src/
    main.js              — entry point
    replay.js            — replay parser/loader
    playback.js          — playback state (stepIndex, speed, play/pause)
    components/
      PlayerView.js      — one player's active/bench/hand/discard
      LogPanel.js        — scrolling event log
      Timeline.js        — slider + playback controls
      StatsPanel.js      — match statistics
    utils/
      cardDb.js          — cards.json loader + lookup
      events.js          — log event formatting
  public/
    replay.json          — loaded via fetch
    cards.json
```

---

### Option 3: Jupyter/Plotly notebook
**Directory:** `replay/03_jupyter_plotly/`
**Status:** Not started

**Stack:** Python + Plotly + ipywidgets

Pros:
- Same language as training code
- Plotly timeline slider for scrubbing
- Can parse replay natively
- Good for quick analysis and exploration

Cons:
- Not great for rich card rendering
- Clunky for real-time playback
- Requires Jupyter setup
- Not easily shareable as standalone

Best for: Quick data exploration, not a polished viewer

---

### Option 4: Terminal TUI
**Directory:** `replay/04_terminal_tui/`
**Status:** Not started

**Stack:** Python + Rich or Textual

Pros:
- Runs in terminal, no browser needed
- Keyboard controls (arrows, space)
- Fast to build
- Lightweight

Cons:
- No visual card art
- Limited layout options
- Hard to show rich game state
- Not great for sharing

Best for: Quick debugging during development

---

### Option 5: Vite + React app (SELECTED)
**Directory:** `replay/05_vite_react_app/`
**Status:** Built & working — `cd 05_vite_react_app && bun install && bun run dev`
(http://localhost:5175). Full-info board, playback/scrub, event log, stats,
diff highlighting, card hover popovers, `?step=N` deep-links. See its README.

**Stack:** Vite + React + TypeScript

Same "proper web app" niche as Option 2, but with React instead of vanilla JS.
The decisive requirement is **diff highlighting** (core feature #8): a keyed,
reactive renderer makes "flash what changed between steps" fall out naturally —
key each card by its `serial` and React handles enter/exit/HP-change transitions
instead of hand-diffing the DOM.

Pros:
- Diff highlighting & stats come from keyed reactive rendering, not manual DOM diffing
- Card component reused across active/bench/hand/discard/prize (rendered ~20×/step)
- TypeScript catches the deeply-nested replay shape at compile time
- Hover → card-detail popover (attacks, weakness, retreat) is clean with components
- Scales to the "nice" version (timeline heatmap, damage charts, KO markers)

Cons:
- Build step + `node_modules` (mitigated: `npm run dev`, allowed by constraints)
- Slightly heavier than vanilla JS for a viewer that could stay simple

Best for: A polished, extensible viewer where per-step diffing is a first-class feature

**Data-shape findings (verified against the real files):**
- `replay.json` is **19.6 MB, 284 steps** (~69 KB/step); `cards.json` is 1.4 MB.
- Rich per-step state lives at **`steps[n][player].observation.current`** (with
  `.players`, `.stadium`, `.turn`, ...) — *not* in a `visualize` array as the old
  `requirements.md` describes. Each step carries a **full materialized snapshot**.
- Consequence: backward/scrub/jump are pure array indexing — **no log replay needed**
  to reconstruct state. `logs` are used only for the event feed and diff annotations.
- Card resolution: **`cards.json` is required, not optional** — replay card objects
  carry only dynamic state (`id`, `serial`, `hp`/`maxHp`, `energies`, `tools`, ...)
  and *not even the name*. Name/attacks/weakness/retreat/type come only from
  `cards.json`, keyed by `id`. v1 loads it at runtime (1.4 MB, once); slimming it to
  only the ~60–120 IDs used in the replay is a future optimization.

---

## Recommendation

**Option 2 (Vite web app)** is the best fit because:

1. The replay data is deeply nested JSON — DOM rendering is natural for text-heavy card info
2. Vite gives instant dev feedback with hot reload
3. Component-based: each panel (board, hand, log, stats, timeline) is independent
4. Can export as single HTML bundle later if needed
5. Keeps viewer separate from Python training code (`replay/02_vite_web_app/`)
6. Run with `just replay` — starts Bun + Vite dev server

## UI Layout Concept

```
┌─────────────────────────────────────────────────────────┐
│  [◀◀] [▶/⏸] [▶▶]  ─────●──────────────  Step 42/186   │
│  Speed: [0.5x] [1x] [2x] [4x]    Turn: 7              │
├────────────────────────────┬────────────────────────────┤
│        Player 0            │         Player 1           │
│  ┌──────────────────────┐  │  ┌──────────────────────┐  │
│  │  Active: Dragapult ex │  │  │  Active: Latias ex   │  │
│  │  HP: ████████░░ 220   │  │  │  HP: ██████░░░░ 150  │  │
│  │  Energy: {R}{D}{D}    │  │  │  Energy: {P}{P}{C}   │  │
│  │  Status: -            │  │  │  Status: Poisoned    │  │
│  └──────────────────────┘  │  └──────────────────────┘  │
│  Bench: [Duskull] [Munki]  │  Bench: [Dreepy] [Budew]   │
│  Hand: 7  Deck: 38  Disc: 3│  Hand: 5  Deck: 41  Disc: 2│
│  Prize: 4                  │  Prize: 3                  │
├────────────────────────────┴────────────────────────────┤
│  Event Log                                              │
│  ─────────────────────────────────────────────────────  │
│  Turn 7 — P0's Dragapult ex used Phantom Dive (220 dmg) │
│  P1's Zeraora took 200 damage → Knocked Out!            │
│  P0 took a prize card                                   │
│  ── P1 attached Basic {P} Energy to Latias ex           │
│  ── P1 played Boss's Orders                            │
├────────────────────────────┬────────────────────────────┤
│  Match Stats               │  Diff (this step)          │
│  ─────────────────         │  ──────────────────        │
│  Damage dealt:  P0: 420    │  - Zeraora KO'd (P1 bench) │
│                 P1: 180    │  - Prize: P0 took 1        │
│  KOs:           P0: 2      │  - Hand: P0 +1 (prize)    │
│                 P1: 0      │  - Active: P1 → Latias ex  │
│  Cards played:  P0: 12     │                            │
│                 P1: 9      │                            │
│  Supporters:    P0: 3      │                            │
│                 P1: 2      │                            │
└────────────────────────────┴────────────────────────────┘
```

## Key Design Decisions

### Playback model
- `steps[]` is loaded into memory as an array
- `currentStep` index tracks position (0 to steps.length-1)
- Each step has 2 player perspectives — show the "active" player's view
- Forward: `currentStep++`, backward: `currentStep--`
- Play/pause: setInterval that increments currentStep
- Timeline slider: `<input type="range" min="0" max="steps.length-1">`

### Card resolution
- Load `cards.json` into a Map<card_id, cardData>
- When rendering a card from the game state, look up by `card.id`
- Show: name, type, HP bar, energies, status icons
- On hover/click: show full details (attacks, weakness, retreat cost)

### Diff computation
- Compare `steps[n].visualize` vs `steps[n-1].visualize`
- Track: HP changes, cards moved between zones, new cards on bench, KOs
- Highlight changed elements with a brief flash or border color

### Event log formatting
- Parse each log entry into a human-readable string
- Color code by type: draw=blue, attack=red, ability=green, play=yellow, KO=bold red
- Auto-scroll to latest event

### Statistics tracking
- Pre-compute cumulative stats for all steps on load
- For each step, store: total damage per player, KOs, cards played, supporters, energy attachments
- Display current step's stats in sidebar

## Future extensions

- **Card images** — if card art URLs are available, render actual card images
- **Search/filter log** — filter events by type or player
- **Bookmarks** — save interesting moments (big attacks, KOs)
- **Side-by-side POV** — show both players' hands simultaneously
- **Export** — save current view as PNG or share link
- **Multi-replay** — load and compare different replays
