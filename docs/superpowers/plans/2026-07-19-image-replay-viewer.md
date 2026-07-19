# Image-based PTCG Replay Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fork `replay/05_vite_react_app` into `replay/07_vite_react_cards`, a replay viewer that renders real card art (board arrangement matching `replay/06_copy/board_layout.html`) with a local-first → CDN → text image backend and header toggles.

**Architecture:** Copy 05 verbatim, then layer four changes: a pure `cardArt` URL resolver (unit-tested), an `<img>`-based `Card` with 3-tier fallback, a rewritten `Board`/`PlayerBoard` matching the approved mockup, and two header toggles (image backend, reveal-hidden) threaded through React state. All of 05's data parsing (`stepState`, `cardDb`, `energy`, `events`, `diff`, `stats`, `loadReplay`) is reused unchanged.

**Tech Stack:** Vite 7, React 18, TypeScript 5, bun, vitest + jsdom + @testing-library/react (added for the fork).

## Global Constraints

- **Do NOT modify `replay/05_vite_react_app`.** It stays as the text-only viewer. All work happens under `replay/07_vite_react_cards`.
- **Never show a broken-image icon.** Image failures fall back local↔cdn once, then to the text-name card.
- **Fully offline** with the downloaded card set (`pkm_data/replay/cards/`, 1267 PNGs already present). CDN is fallback only.
- **CDN URL shape (exact):** `https://ptcgvis.heroz.jp/img/<album>/<id>.png`, default `<album>` = `bqucewmzuceknw`, overridable via `import.meta.env.VITE_CARD_ALBUM`.
- **Local URL shape (exact):** `/cards/<id>.png` (served from `public/cards/`, a symlink to `pkm_data/replay/cards/`).
- **Bench** can hold up to 8 (render filled slots + dashed empties up to `benchMax`, which defaults to 5).
- **Hidden zones affect only the hand.** Active/bench/discard/stadium are always public art. In `realistic` mode a non-viewer's hand renders as face-down backs (`handCount` of them); prizes and deck are always face-down backs in both modes. In `full-info` mode every hand renders as face-up art.
- **Viewer POV** = `current.yourIndex`.

---

### Task 1: Scaffold the fork

Copy 05 to the new location, wire the symlinks (including the new `cards` link), rename the package, add the test toolchain, and confirm it boots and builds identically to 05 (still text cards at this point).

**Files:**
- Create: `replay/07_vite_react_cards/` (full copy of `replay/05_vite_react_app/`, excluding `node_modules`, `dist`, `bun.lock`, `package-lock.json`, `tsconfig.tsbuildinfo`)
- Modify: `replay/07_vite_react_cards/package.json` (name + test deps + test script)
- Create: `replay/07_vite_react_cards/public/cards` (symlink)
- Create: `replay/07_vite_react_cards/vitest.config.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: a runnable app at `replay/07_vite_react_cards` (port 5175), plus `bun run test` wired to vitest.

- [ ] **Step 1: Copy the app (exclude build artifacts)**

```bash
cd /home/df/projects/zeke/pkm_new/replay
rsync -a --exclude node_modules --exclude dist --exclude bun.lock \
  --exclude package-lock.json --exclude tsconfig.tsbuildinfo \
  05_vite_react_app/ 07_vite_react_cards/
```

- [ ] **Step 2: Fix the symlinks**

The rsync copies 05's symlinks verbatim; `replay.json`/`cards.json` targets (`../../replay.json`, `../../cards.json`) still resolve from the new dir (same depth). Add the new `cards` link and verify all three resolve.

```bash
cd /home/df/projects/zeke/pkm_new/replay/07_vite_react_cards/public
ln -sfn ../../../pkm_data/replay/cards cards
# verify (all three must print real paths, not "No such file"):
ls -lL replay.json cards.json cards/121.png
```
Expected: `replay.json`, `cards.json`, and `cards/121.png` all resolve to existing files.

- [ ] **Step 3: Rename package + add test toolchain**

Edit `replay/07_vite_react_cards/package.json` — change `"name"`, add a `test` script and the vitest devDeps:

```json
{
  "name": "07_vite_react_cards",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "typescript": "^5.6.3",
    "vite": "^7.1.5",
    "vitest": "^2.1.8",
    "jsdom": "^25.0.1",
    "@testing-library/react": "^16.1.0",
    "@testing-library/jest-dom": "^6.6.3"
  }
}
```

- [ ] **Step 4: Add vitest config (jsdom env)**

Create `replay/07_vite_react_cards/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
  },
});
```

- [ ] **Step 5: Install and verify boot + build**

```bash
cd /home/df/projects/zeke/pkm_new/replay/07_vite_react_cards
bun install
bun run build
```
Expected: `tsc -b` and `vite build` both succeed (0 errors), `dist/` produced.

- [ ] **Step 6: Commit**

```bash
cd /home/df/projects/zeke/pkm_new
git add replay/07_vite_react_cards
git commit -m "feat(replay): scaffold 07_vite_react_cards fork of 05 + test toolchain"
```

---

### Task 2: `cardArt.ts` image-URL backend (TDD)

A pure module that maps a `card_id` + backend to primary/fallback URLs and reads defaults from env. Fully unit-tested — no React.

**Files:**
- Create: `replay/07_vite_react_cards/src/data/cardArt.ts`
- Test: `replay/07_vite_react_cards/src/data/cardArt.test.ts`

**Interfaces:**
- Consumes: `import.meta.env.VITE_CARD_ALBUM`, `import.meta.env.VITE_CARD_BACKEND`.
- Produces:
  - `type CardBackend = "local" | "cdn"`
  - `localCardUrl(id: number): string` → `/cards/<id>.png`
  - `cdnCardUrl(id: number, album?: string): string` → heroz URL
  - `defaultBackend(): CardBackend`
  - `resolveCardArt(id: number, backend: CardBackend): { primary: string; fallback: string }`

- [ ] **Step 1: Write the failing test**

Create `src/data/cardArt.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { resolveCardArt, localCardUrl, cdnCardUrl } from "./cardArt";

describe("cardArt", () => {
  it("localCardUrl builds /cards/<id>.png", () => {
    expect(localCardUrl(121)).toBe("/cards/121.png");
  });

  it("cdnCardUrl uses the default album", () => {
    expect(cdnCardUrl(121)).toBe(
      "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
    );
  });

  it("local backend: primary is local, fallback is cdn", () => {
    expect(resolveCardArt(121, "local")).toEqual({
      primary: "/cards/121.png",
      fallback: "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
    });
  });

  it("cdn backend: primary is cdn, fallback is local", () => {
    expect(resolveCardArt(121, "cdn")).toEqual({
      primary: "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
      fallback: "/cards/121.png",
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/df/projects/zeke/pkm_new/replay/07_vite_react_cards && bun run test`
Expected: FAIL — cannot resolve `./cardArt`.

- [ ] **Step 3: Write the implementation**

Create `src/data/cardArt.ts`:

```ts
// Maps a card_id to its face-art image URL for either backend.
// local  -> /cards/<id>.png   (public/cards symlink -> pkm_data/replay/cards)
// cdn    -> https://ptcgvis.heroz.jp/img/<album>/<id>.png
export type CardBackend = "local" | "cdn";

const DEFAULT_ALBUM = "bqucewmzuceknw";

export function cardAlbum(): string {
  return (import.meta.env.VITE_CARD_ALBUM as string | undefined) || DEFAULT_ALBUM;
}

export function localCardUrl(id: number): string {
  return `/cards/${id}.png`;
}

export function cdnCardUrl(id: number, album: string = cardAlbum()): string {
  return `https://ptcgvis.heroz.jp/img/${album}/${id}.png`;
}

export function defaultBackend(): CardBackend {
  return (import.meta.env.VITE_CARD_BACKEND as string | undefined) === "cdn"
    ? "cdn"
    : "local";
}

export interface CardArtUrls {
  primary: string;
  fallback: string;
}

// The active backend's URL is `primary`; the other backend is `fallback`
// (used by <img> onError before finally giving up to the text card).
export function resolveCardArt(id: number, backend: CardBackend): CardArtUrls {
  const local = localCardUrl(id);
  const cdn = cdnCardUrl(id);
  return backend === "cdn"
    ? { primary: cdn, fallback: local }
    : { primary: local, fallback: cdn };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun run test`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add replay/07_vite_react_cards/src/data/cardArt.ts replay/07_vite_react_cards/src/data/cardArt.test.ts
git commit -m "feat(replay): cardArt image-url resolver with local/cdn backends"
```

---

### Task 3: Card art rendering + 3-tier fallback

Replace the text-only `Card` face with `<img>` art (overlays kept on top), add a `CardBack` for face-down zones, and a render test proving the correct `src` and the text fallback.

**Files:**
- Modify: `replay/07_vite_react_cards/src/components/Card.tsx` (whole file)
- Modify: `replay/07_vite_react_cards/src/styles.css` (append card-art rules)
- Test: `replay/07_vite_react_cards/src/components/Card.test.tsx`

**Interfaces:**
- Consumes: `resolveCardArt`, `CardBackend` (Task 2); `CardInstance`, `CardDb`, `energyType` (existing).
- Produces:
  - `Card` props now include `backend: CardBackend`.
  - `export function CardBack(): JSX.Element` — CSS card-back, no image request.

- [ ] **Step 1: Write the failing test**

Create `src/components/Card.test.tsx`:

```tsx
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { Card, CardBack } from "./Card";
import { CardDb } from "../data/cardDb";
import type { CardInstance } from "../data/types";

const db = new CardDb([
  { card_id: 121, name: "Dragapult ex", card_type: 0, energy_type: 0, hp: 320,
    basic: false, stage1: false, stage2: true, ex: true, evolves_from: null,
    weakness: null, resistance: null, retreat_cost: 2, attacks: [] },
]);
const inst: CardInstance = { id: 121, serial: 1, playerIndex: 0, hp: 200, maxHp: 320, energies: [2, 2, 5] };

afterEach(cleanup);

describe("Card art", () => {
  it("renders the local image by default", () => {
    render(<Card card={inst} db={db} variant="active" backend="local" />);
    const img = screen.getByRole("img") as HTMLImageElement;
    expect(img.getAttribute("src")).toBe("/cards/121.png");
  });

  it("renders the cdn image when backend is cdn", () => {
    render(<Card card={inst} db={db} variant="active" backend="cdn" />);
    const img = screen.getByRole("img") as HTMLImageElement;
    expect(img.getAttribute("src")).toBe(
      "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
    );
  });

  it("swaps to the fallback url on first error", () => {
    render(<Card card={inst} db={db} variant="active" backend="local" />);
    const img = screen.getByRole("img") as HTMLImageElement;
    fireEvent.error(img);
    expect(img.getAttribute("src")).toBe(
      "https://ptcgvis.heroz.jp/img/bqucewmzuceknw/121.png",
    );
  });

  it("keeps the card name in the DOM as text fallback", () => {
    render(<Card card={inst} db={db} variant="active" backend="local" />);
    expect(screen.getByText("Dragapult ex")).toBeTruthy();
  });

  it("CardBack renders no image", () => {
    render(<CardBack />);
    expect(screen.queryByRole("img")).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun run test`
Expected: FAIL — `Card` has no `backend` prop / no `<img>` / `CardBack` not exported.

- [ ] **Step 3: Rewrite `Card.tsx`**

Replace the whole file with:

```tsx
import { useEffect, useState } from "react";
import type { CardDb } from "../data/cardDb";
import { energyType } from "../data/energy";
import type { CardInstance } from "../data/types";
import { resolveCardArt, type CardBackend } from "../data/cardArt";

interface Props {
  card: CardInstance;
  db: CardDb;
  variant: "active" | "bench" | "hand" | "mini";
  backend: CardBackend;
  hpDelta?: number;
  appeared?: boolean;
}

function EnergyPips({ energies }: { energies: number[] }) {
  if (!energies?.length) return null;
  return (
    <span className="pips">
      {energies.map((code, i) => {
        const t = energyType(code);
        return (
          <span key={i} className="pip" style={{ background: t.color, color: t.fg }} title={t.name}>
            {t.letter}
          </span>
        );
      })}
    </span>
  );
}

// Local-first (or cdn-first) <img> that swaps to the other backend once on
// error, then hides itself so the text name underneath shows. Never a broken
// image icon.
function CardArt({ id, backend, alt }: { id: number; backend: CardBackend; alt: string }) {
  const { primary, fallback } = resolveCardArt(id, backend);
  const [src, setSrc] = useState(primary);
  const [dead, setDead] = useState(false);
  useEffect(() => { setSrc(primary); setDead(false); }, [primary]);
  if (dead) return null;
  return (
    <img
      className="card-art"
      src={src}
      alt={alt}
      draggable={false}
      onError={() => (src !== fallback ? setSrc(fallback) : setDead(true))}
    />
  );
}

// Face-down card (opponent hand in realistic mode, prizes, deck). CSS-drawn.
export function CardBack({ variant = "mini" }: { variant?: Props["variant"] }) {
  return <div className={`card card-${variant} card-back`} aria-hidden="true" />;
}

export function Card({ card, db, variant, backend, hpDelta, appeared }: Props) {
  const def = db.get(card.id);
  const name = def?.name ?? `#${card.id}`;
  const hpPct = card.maxHp ? Math.max(0, (card.hp / card.maxHp) * 100) : 0;
  const damaged = (hpDelta ?? 0) < 0;
  const healed = (hpDelta ?? 0) > 0;

  const cls = [
    "card", `card-${variant}`, "has-art",
    hpDelta ? (damaged ? "flash-dmg" : healed ? "flash-heal" : "") : "",
    appeared ? "flash-new" : "",
  ].filter(Boolean).join(" ");

  return (
    <div className={cls} tabIndex={0}>
      <CardArt id={card.id} backend={backend} alt={name} />
      <div className="card-name">{name}</div>
      {def?.ex && <span className="card-tag">ex</span>}

      {variant !== "hand" && card.maxHp > 0 && (
        <div className="hp">
          <div className="hp-bar">
            <div className="hp-fill" style={{
              width: `${hpPct}%`,
              background: hpPct > 50 ? "#4caf50" : hpPct > 25 ? "#e0a03d" : "#e2493a",
            }} />
          </div>
          <span className="hp-text">
            {card.hp}/{card.maxHp}
            {hpDelta ? <span className={damaged ? "delta-dmg" : "delta-heal"}> {hpDelta > 0 ? "+" : ""}{hpDelta}</span> : null}
          </span>
        </div>
      )}
      <EnergyPips energies={card.energies ?? []} />

      {def && (
        <div className="card-pop">
          <div className="pop-title">{def.name}</div>
          <div className="pop-meta">
            HP {def.hp} · retreat {def.retreat_cost}
            {def.weakness != null ? ` · weak ${energyType(def.weakness).letter}` : ""}
          </div>
          {def.attacks?.map((a) => (
            <div key={a.attack_id} className="pop-atk">
              <span className="pop-atk-cost">{a.energies.map((c) => energyType(c).letter).join("")}</span>{" "}
              <b>{a.name}</b> {a.damage ? `— ${a.damage}` : ""}
              {a.text ? <div className="pop-atk-text">{a.text}</div> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Append card-art CSS**

Append to `src/styles.css`:

```css
/* --- card art (07 image viewer) --- */
.card.has-art { position: relative; overflow: visible; }
.card-art {
  position: absolute; inset: 0; width: 100%; height: 100%;
  object-fit: cover; border-radius: 6px; z-index: 0;
}
/* text name sits UNDER the art; visible only if the image fails to load */
.card.has-art .card-name {
  position: absolute; inset: 0; z-index: -1;
  display: flex; align-items: center; justify-content: center;
  text-align: center; font-size: 9px; padding: 2px;
}
.card.has-art .hp, .card.has-art .pips, .card-tag { position: relative; z-index: 1; }
.card-tag {
  position: absolute; top: 3px; left: 3px; z-index: 2;
  background: #111a; color: #fff; border-radius: 4px; font-size: 8px; padding: 0 3px;
}
.card-back {
  background: repeating-linear-gradient(45deg, #1b3a8f 0 8px, #12296b 8px 16px);
  border: 1px solid #0d1f52; border-radius: 6px;
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bun run test`
Expected: PASS (Task 2's 4 + Task 3's 5 = 9 tests). Note: `PlayerBoard`/`Board` still pass `Card` without `backend` — that is fixed in Task 4/5; the app won't typecheck yet, which is expected until Task 5. Tests import `Card` directly so they pass now.

- [ ] **Step 6: Commit**

```bash
git add replay/07_vite_react_cards/src/components/Card.tsx replay/07_vite_react_cards/src/components/Card.test.tsx replay/07_vite_react_cards/src/styles.css
git commit -m "feat(replay): image-based Card with local->cdn->text fallback + CardBack"
```

---

### Task 4: Board arrangement matching the mockup

Rewrite `PlayerBoard` and `Board` to the `board_layout.html` arrangement: per-player band `[prizes 2×3] [field: bench+active] [deck/discard piles]`, mirrored top/bottom, fanned hand, center stadium strip. Threads `backend` and hidden-info props (defined here, supplied by Task 5).

**Files:**
- Modify: `replay/07_vite_react_cards/src/components/PlayerBoard.tsx` (whole file)
- Modify: `replay/07_vite_react_cards/src/components/Board.tsx` (whole file)
- Modify: `replay/07_vite_react_cards/src/styles.css` (append board-layout rules)

**Interfaces:**
- Consumes: `Card`, `CardBack` (Task 3); `MergedStep`, `PlayerState`, `CardDb`, `StepDiff` (existing); `CardBackend` (Task 2).
- Produces:
  - `PlayerBoard` props: `{ player, index, db, diff, active, side: "top" | "bottom", backend: CardBackend, revealHand: boolean }`
    - `revealHand` — true → render `player.hand` as face-up `Card`s; false → render `player.handCount` `CardBack`s.
  - `Board` props: `{ step, db, diff, backend: CardBackend, reveal: "realistic" | "full-info" }`

- [ ] **Step 1: Rewrite `PlayerBoard.tsx`**

```tsx
import type { CardDb } from "../data/cardDb";
import type { StepDiff } from "../data/diff";
import type { PlayerState } from "../data/types";
import type { CardBackend } from "../data/cardArt";
import { Card, CardBack } from "./Card";

interface Props {
  player: PlayerState;
  index: number;
  db: CardDb;
  diff: StepDiff;
  active: boolean;
  side: "top" | "bottom";
  backend: CardBackend;
  revealHand: boolean;
}

const STATUS_FLAGS: [keyof PlayerState, string][] = [
  ["poisoned", "Poisoned"], ["burned", "Burned"], ["asleep", "Asleep"],
  ["paralyzed", "Paralyzed"], ["confused", "Confused"],
];

export function PlayerBoard({ player, index, db, diff, active, side, backend, revealHand }: Props) {
  const statuses = STATUS_FLAGS.filter(([k]) => player[k]);
  const cardProps = (serial: number) => ({
    hpDelta: diff.changedHp.get(serial),
    appeared: diff.appeared.has(serial),
  });
  const activeCards = (player.active ?? []).filter(Boolean) as NonNullable<PlayerState["active"][number]>[];
  const benchCards = (player.bench ?? []).filter(Boolean) as NonNullable<PlayerState["bench"][number]>[];
  const benchEmpties = Math.max(0, player.benchMax - benchCards.length);

  const prizes = (
    <div className="zone prizes-zone">
      <div className="zone-label">Prizes ({player.prize.length})</div>
      <div className="prizes-grid">
        {Array.from({ length: player.prize.length }).map((_, i) => <CardBack key={i} variant="mini" />)}
      </div>
    </div>
  );

  const piles = (
    <div className="zone piles-zone">
      <div className="pile">
        <div className="zone-label">Deck ({player.deckCount})</div>
        {player.deckCount > 0 ? <CardBack variant="mini" /> : <div className="empty">—</div>}
      </div>
      <div className="pile">
        <div className="zone-label">Discard ({player.discard.length})</div>
        {player.discard.length > 0
          ? <Card card={player.discard[player.discard.length - 1]} db={db} variant="mini" backend={backend} />
          : <div className="empty">—</div>}
      </div>
    </div>
  );

  const benchRow = (
    <div className="zone">
      <div className="zone-label">Bench ({benchCards.length}/{player.benchMax})</div>
      <div className="row bench-row">
        {benchCards.map((c) => <Card key={c.serial} card={c} db={db} variant="bench" backend={backend} {...cardProps(c.serial)} />)}
        {Array.from({ length: benchEmpties }).map((_, i) => <div key={`e${i}`} className="slot">bench</div>)}
      </div>
    </div>
  );

  const activeRow = (
    <div className="zone">
      <div className="zone-label">Active</div>
      <div className="row active-row">
        {activeCards.length
          ? activeCards.map((c) => <div key={c.serial} className="active-frame"><Card card={c} db={db} variant="active" backend={backend} {...cardProps(c.serial)} /></div>)
          : <div className="slot">active</div>}
      </div>
    </div>
  );

  const hand = (
    <div className="zone hand-zone">
      <div className="zone-label">Hand ({player.handCount})</div>
      <div className="row hand-fan">
        {revealHand
          ? player.hand.map((c) => <Card key={c.serial} card={c} db={db} variant="hand" backend={backend} />)
          : Array.from({ length: player.handCount }).map((_, i) => <CardBack key={i} variant="hand" />)}
      </div>
    </div>
  );

  // top: bench above active (active nearest centre); bottom: active above bench.
  const field = (
    <div className="zone field-zone">
      {side === "top" ? <>{benchRow}{activeRow}</> : <>{activeRow}{benchRow}</>}
      {statuses.length > 0 && (
        <div className="statuses">
          {statuses.map(([, label]) => <span key={label} className="status-chip">{label}</span>)}
        </div>
      )}
    </div>
  );

  return (
    <section className={`pl side-${side} ${active ? "pl-active" : ""}`}>
      <header className="pl-head">
        <span className="player-tag">P{index}</span>
        {active && <span className="turn-flag">▶ acting</span>}
        <span className="counts">Hand {player.handCount} · Deck {player.deckCount} · Discard {player.discard.length} · Prize {player.prize.length}</span>
      </header>
      {side === "top" && hand}
      <div className="pl-band">
        {prizes}
        {field}
        {piles}
      </div>
      {side === "bottom" && hand}
    </section>
  );
}
```

- [ ] **Step 2: Rewrite `Board.tsx`**

```tsx
import type { CardDb } from "../data/cardDb";
import type { StepDiff } from "../data/diff";
import type { MergedStep } from "../data/stepState";
import type { CardBackend } from "../data/cardArt";
import { Card } from "./Card";
import { PlayerBoard } from "./PlayerBoard";

interface Props {
  step: MergedStep;
  db: CardDb;
  diff: StepDiff;
  backend: CardBackend;
  reveal: "realistic" | "full-info";
}

function winnerText(rewards: [number, number]): string {
  if (rewards[0] === rewards[1]) return "Draw";
  return rewards[0] > rewards[1] ? "P0 wins" : "P1 wins";
}

export function Board({ step, db, diff, backend, reveal }: Props) {
  const cur = step.current;
  const done = step.statuses.includes("DONE");

  if (!cur) {
    return (
      <div className="board board-end">
        <div className="end-card"><h2>Game starting…</h2><p>Initial setup — step forward to begin.</p></div>
      </div>
    );
  }

  const stadium = (cur.stadium ?? []).filter(Boolean);
  const viewer = cur.yourIndex;
  const revealHand = (i: number) => reveal === "full-info" || i === viewer;

  return (
    <div className="board board-cards">
      {done && <div className="winner-banner">Game over — {winnerText(step.rewards)}</div>}

      <PlayerBoard player={cur.players[0]} index={0} db={db} diff={diff}
        active={step.activePlayer === 0} side="top" backend={backend} revealHand={revealHand(0)} />

      <div className="stadium-strip">
        <span className="zone-label">Stadium</span>
        {stadium.length > 0
          ? <Card card={stadium[0]!} db={db} variant="mini" backend={backend} />
          : <span className="empty">— none —</span>}
        <span className="board-meta">
          Turn {cur.turn} · Action {cur.turnActionCount}
          {cur.supporterPlayed && <span className="flagged">supporter</span>}
          {cur.energyAttached && <span className="flagged">energy</span>}
          {cur.retreated && <span className="flagged">retreated</span>}
        </span>
      </div>

      <PlayerBoard player={cur.players[1]} index={1} db={db} diff={diff}
        active={step.activePlayer === 1} side="bottom" backend={backend} revealHand={revealHand(1)} />
    </div>
  );
}
```

- [ ] **Step 3: Append board-layout CSS**

Append to `src/styles.css`:

```css
/* --- 07 board layout (mockup arrangement) --- */
.board-cards { --cw: 68px; --ch: 95px; display: flex; flex-direction: column; gap: 8px; }
.board-cards .card { width: var(--cw); height: var(--ch); }
.board-cards .card-hand { width: calc(var(--cw) * 0.92); height: calc(var(--ch) * 0.92); }
.pl { border: 1px solid rgba(255,255,255,.14); border-radius: 10px; padding: 8px 10px; background: rgba(0,0,0,.12); }
.pl-active { box-shadow: inset 0 0 0 1px #ffd24a55; }
.pl-head { display: flex; gap: 10px; align-items: center; font-size: 11px; opacity: .8; margin-bottom: 6px; }
.pl-band { display: grid; grid-template-columns: auto 1fr auto; gap: 16px; align-items: center; }
.field-zone { display: flex; flex-direction: column; gap: 8px; align-items: center; }
.zone { display: flex; flex-direction: column; gap: 4px; align-items: center; }
.zone-label { font-size: 9px; letter-spacing: 1px; text-transform: uppercase; opacity: .6; }
.row { display: flex; gap: 8px; }
.prizes-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px; }
.prizes-grid .card { width: calc(var(--cw) * 0.66); height: calc(var(--ch) * 0.66); }
.piles-zone { flex-direction: row; gap: 12px; align-items: flex-start; }
.pile { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.slot { width: var(--cw); height: var(--ch); border: 1.5px dashed rgba(255,255,255,.28);
  border-radius: 6px; display: flex; align-items: center; justify-content: center;
  font-size: 9px; opacity: .5; }
.active-frame { outline: 3px solid #ffd24a; outline-offset: 2px; border-radius: 8px; }
.hand-fan .card { margin-left: -16px; transition: transform .1s; }
.hand-fan .card:first-child { margin-left: 0; }
.hand-fan .card:hover { transform: translateY(-10px); z-index: 5; }
.stadium-strip { display: flex; align-items: center; justify-content: center; gap: 14px;
  padding: 6px; border-top: 1px dashed rgba(255,255,255,.2); border-bottom: 1px dashed rgba(255,255,255,.2); }
```

- [ ] **Step 4: Verify typecheck (App still passes old Board props — expected fail here)**

Run: `bun run test`
Expected: PASS (9 tests unchanged — components are unit-tested in isolation). `bun run build` will still FAIL because `App.tsx` calls `<Board step db diff />` without `backend`/`reveal`; that is fixed in Task 5. Do not attempt to build yet.

- [ ] **Step 5: Commit**

```bash
git add replay/07_vite_react_cards/src/components/PlayerBoard.tsx replay/07_vite_react_cards/src/components/Board.tsx replay/07_vite_react_cards/src/styles.css
git commit -m "feat(replay): mockup board arrangement (prizes/bench/active/piles/stadium)"
```

---

### Task 5: Header toggles + wiring (backend, reveal)

Add two pieces of UI state in `App.tsx`, render header toggles, and pass `backend`/`reveal` into `Board`. This closes the typecheck gap from Task 4 and makes the app build and run.

**Files:**
- Modify: `replay/07_vite_react_cards/src/App.tsx` (Viewer component + a new `ViewControls` component)
- Modify: `replay/07_vite_react_cards/src/styles.css` (append toggle styles)

**Interfaces:**
- Consumes: `defaultBackend`, `CardBackend` (Task 2); `Board` props (Task 4).
- Produces: nothing downstream.

- [ ] **Step 1: Add state + controls in `App.tsx`**

In `src/App.tsx`, add the import:

```tsx
import { defaultBackend, type CardBackend } from "./data/cardArt";
```

Inside `Viewer`, after `const pb = usePlayback(...)`, add:

```tsx
  const [backend, setBackend] = useState<CardBackend>(defaultBackend());
  const [reveal, setReveal] = useState<"realistic" | "full-info">("realistic");
```

Change the `<Board .../>` call to:

```tsx
          <Board step={step} db={db} diff={diff} backend={backend} reveal={reveal} />
```

Add the controls into the header `topline`, after `<FilePicker .../>`:

```tsx
          <ViewControls backend={backend} setBackend={setBackend} reveal={reveal} setReveal={setReveal} />
```

Ensure `useState` is imported (it already is in App.tsx).

- [ ] **Step 2: Add the `ViewControls` component**

Append to `src/App.tsx` (bottom, alongside `FilePicker`):

```tsx
function ViewControls({
  backend, setBackend, reveal, setReveal,
}: {
  backend: CardBackend;
  setBackend: (b: CardBackend) => void;
  reveal: "realistic" | "full-info";
  setReveal: (r: "realistic" | "full-info") => void;
}) {
  return (
    <div className="view-controls">
      <label className="ctl">
        Art
        <select value={backend} onChange={(e) => setBackend(e.target.value as CardBackend)}>
          <option value="local">local</option>
          <option value="cdn">cdn</option>
        </select>
      </label>
      <label className="ctl">
        Hidden
        <select value={reveal} onChange={(e) => setReveal(e.target.value as "realistic" | "full-info")}>
          <option value="realistic">realistic</option>
          <option value="full-info">reveal all</option>
        </select>
      </label>
    </div>
  );
}
```

- [ ] **Step 3: Append toggle CSS**

Append to `src/styles.css`:

```css
.view-controls { display: flex; gap: 12px; align-items: center; margin-left: auto; }
.view-controls .ctl { display: flex; gap: 5px; align-items: center; font-size: 11px; opacity: .85; }
.view-controls select { font-size: 11px; }
```

- [ ] **Step 4: Build + test**

Run: `cd /home/df/projects/zeke/pkm_new/replay/07_vite_react_cards && bun run build && bun run test`
Expected: build succeeds (0 TS errors), 9 tests pass.

- [ ] **Step 5: Manual + screenshot verification**

```bash
cd /home/df/projects/zeke/pkm_new/replay/07_vite_react_cards
bun run build && bun run preview --port 4175 &
sleep 3
google-chrome-stable --headless --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size=1400,1100 --virtual-time-budget=6000 \
  --screenshot=/tmp/07_board.png "http://localhost:4175/?step=95"
```
Then Read `/tmp/07_board.png` and confirm: real card art on both sides, prizes as 2×3 backs, active in gold frame, fanned hands, stadium strip, HP/energy overlays. Toggle checks: `local` art renders offline; switching to `cdn` still renders; `reveal all` shows opponent hand as art, `realistic` shows it as backs. Kill the preview server when done.

- [ ] **Step 6: Commit**

```bash
git add replay/07_vite_react_cards/src/App.tsx replay/07_vite_react_cards/src/styles.css
git commit -m "feat(replay): header toggles for image backend + hidden-info reveal"
```

---

### Task 6: `just fetch-cards` target + README

Wire the already-built fetch script into the justfile and document the new app.

**Files:**
- Modify: `justfile` (add `fetch-cards` + `replay-cards` targets)
- Create: `replay/07_vite_react_cards/README.md`
- Modify: `replay/CARD_IMAGES.md` (add the `just fetch-cards` one-liner) — optional if already clear.

**Interfaces:**
- Consumes: `replay/fetch_card_images.py` (exists).
- Produces: `just fetch-cards`, `just replay-cards`.

- [ ] **Step 1: Check existing justfile replay targets**

Run: `grep -n "replay" /home/df/projects/zeke/pkm_new/justfile`
Expected: shows `replay-react` (the 05 target) — mirror its style.

- [ ] **Step 2: Add justfile targets**

Add to `justfile` (match the file's existing recipe style; adjust the `replay-react`-like body to point at 07):

```makefile
# Download all card face images into pkm_data/replay/cards (skips existing)
fetch-cards *ARGS:
    python3 replay/fetch_card_images.py --out pkm_data/replay/cards --cards-json replay/cards.json {{ARGS}}

# Run the image-based replay viewer (07) on http://localhost:5175
replay-cards file="":
    cd replay/07_vite_react_cards && VITE_REPLAY="{{file}}" bun run dev
```

- [ ] **Step 3: Write the app README**

Create `replay/07_vite_react_cards/README.md`:

```markdown
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
```

- [ ] **Step 4: Verify the targets run**

```bash
cd /home/df/projects/zeke/pkm_new
just fetch-cards --ids 121   # should print "1 skipped" (already downloaded)
```
Expected: runs the script, reports the card already present (skipped).

- [ ] **Step 5: Commit**

```bash
git add justfile replay/07_vite_react_cards/README.md replay/CARD_IMAGES.md
git commit -m "docs(replay): just fetch-cards / replay-cards targets + 07 README"
```

---

## Self-Review

**Spec coverage:**
- Fork at `07_vite_react_cards`, 05 untouched → Task 1 + Global Constraints. ✅
- `cardArt.ts` backend (local/cdn, album env, precedence) → Task 2. ✅
- Card art + overlays + backs + 3-tier fallback → Task 3. ✅
- Mockup board arrangement (prizes 2×3, bench-behind-active mirrored, fanned hand, pile columns, stadium strip) → Task 4. ✅
- Hidden-info toggle (realistic ↔ full-info) + backend toggle → Task 5. ✅
- Fetch script + `just` target + README → Task 6. ✅
- Offline-first, no broken-image icon, bench up to 8 → Global Constraints, enforced in Tasks 3–4. ✅
- Testing: `cardArt` unit tests (Task 2), `Card` render/fallback tests (Task 3), build + screenshot verification (Task 5). ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✅

**Type consistency:** `CardBackend` from `cardArt.ts` used identically in Tasks 3–5. `Card` gains `backend` (Task 3), consumed by `PlayerBoard`/`Board` (Task 4), supplied from `App` state (Task 5). `revealHand: boolean` on `PlayerBoard` derived from `Board`'s `reveal` + `yourIndex`. `CardBack` exported in Task 3, used in Task 4. Signatures match across tasks. ✅

**Known intermediate breakage (intentional, documented):** After Task 3 the app does not `build` (App passes old Board props); after Task 4 it still does not build. Both are noted in-task; Task 5 closes the gap. Unit tests pass throughout because components are tested in isolation. This is acceptable for subagent-driven execution where each task's own verification is unit tests, with the full build gated at Task 5.
