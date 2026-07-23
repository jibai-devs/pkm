import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Board } from "./components/Board";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { DiffPanel } from "./components/DiffPanel";
import { LogPanel } from "./components/LogPanel";
import { ReplayChooser } from "./components/ReplayChooser";
import { SubAgentPanel } from "./components/SubAgentPanel";
import { StatsPanel } from "./components/StatsPanel";
import { Timeline } from "./components/Timeline";
import { CardDb, loadCardDb } from "./data/cardDb";
import { defaultBackend, type CardBackend } from "./data/cardArt";
import { computeDiff } from "./data/diff";
import {
  loadReplay,
  readReplayFile,
  resolveCardsUrl,
  resolveReplayUrl,
} from "./data/loadReplay";
import { computeStats } from "./data/stats";
import { mergeStep } from "./data/stepState";
import type { Replay } from "./data/types";
import { usePlayback } from "./state/usePlayback";
import { LiveApp } from "./live/LiveApp";

function isPlayMode(): boolean {
  return new URLSearchParams(window.location.search).get("mode") === "play";
}

export default function App() {
  // ?mode=play swaps the replay viewer for the interactive game against a bot
  // (same board components, live server-driven state). Everything below is the
  // unchanged replay viewer.
  if (isPlayMode()) return <LiveApp />;
  return <ReplayApp />;
}

function ReplayApp() {
  const [db, setDb] = useState<CardDb | null>(null);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [source, setSource] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // Initial load: cards DB (constant) + the resolved replay URL.
  useEffect(() => {
    const url = resolveReplayUrl();
    Promise.all([loadReplay(url), loadCardDb(resolveCardsUrl())])
      .then(([r, d]) => {
        setDb(d);
        setReplay(r);
        setSource(url.split("/").pop() || url);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Swap in a user-picked local file (any .json on disk).
  const loadFile = useCallback(async (file: File) => {
    try {
      const r = await readReplayFile(file);
      setReplay(r);
      setSource(file.name);
      setError(null);
    } catch (e) {
      setError(`${file.name}: ${e instanceof Error ? e.message : e}`);
    }
  }, []);

  if (error && !replay) return <div className="fatal">Failed to load: {error}</div>;
  if (!db || !replay) return <div className="loading">Loading replay…</div>;
  return (
    <Viewer
      key={source} // remount -> reset playback when the replay changes
      replay={replay}
      db={db}
      source={source}
      error={error}
      onPickFile={loadFile}
    />
  );
}

interface ViewerProps {
  replay: Replay;
  db: CardDb;
  source: string;
  error: string | null;
  onPickFile: (file: File) => void;
}

function initialStep(): number {
  const p = new URLSearchParams(window.location.search).get("step");
  const n = p ? parseInt(p, 10) : 1;
  return Number.isFinite(n) ? n - 1 : 0; // ?step is 1-based to match the UI
}

function Viewer({ replay, db, source, error, onPickFile }: ViewerProps) {
  const pb = usePlayback(replay.steps.length, initialStep());
  const [backend, setBackend] = useState<CardBackend>(defaultBackend());
  // Default to full information: these replays are used to debug our own
  // agents, where seeing what the opponent actually held is the whole point.
  // Switch the header's Hidden control back to "realistic" to view a game the
  // way a player would have experienced it.
  const [reveal, setReveal] = useState<"realistic" | "full-info">("full-info");
  const [swap, setSwap] = useState(false);
  const stats = useMemo(() => computeStats(replay), [replay]);

  const step = useMemo(() => mergeStep(replay, pb.index), [replay, pb.index]);
  const prev = useMemo(
    () => (pb.index > 0 ? mergeStep(replay, pb.index - 1) : null),
    [replay, pb.index],
  );
  const diff = useMemo(() => computeDiff(prev, step), [prev, step]);

  // Bottom side = the "player" (the viewer's own side), swappable via the toggle.
  const viewer = step.current?.yourIndex ?? 1;
  const bottomIndex = ((swap ? 1 - viewer : viewer) ? 1 : 0) as 0 | 1;

  // Keep ?step= in sync so the current position is shareable / survives reload.
  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("step", String(pb.index + 1));
    window.history.replaceState(null, "", url);
  }, [pb.index]);

  // Keyboard: ← → step, space play/pause, home/end.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      switch (e.key) {
        case " ": // space: play/pause (preventDefault stops page scroll + button re-activation)
          e.preventDefault();
          pb.togglePlay();
          break;
        case "ArrowRight":
          if (tag === "INPUT") return; // let the focused scrubber handle its own arrows
          e.preventDefault();
          pb.step(1);
          break;
        case "ArrowLeft":
          if (tag === "INPUT") return;
          e.preventDefault();
          pb.step(-1);
          break;
        case "Home":
          pb.setIndex(0);
          break;
        case "End":
          pb.setIndex(pb.count - 1);
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pb]);

  // Drag-and-drop a replay file anywhere on the app.
  const [dragging, setDragging] = useState(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onPickFile(file);
  };

  return (
    <div
      className={`app ${dragging ? "dragging" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <header className="topbar">
        <div className="topline">
          <h1>{replay.title || replay.name || "PTCG Replay"}</h1>
          <ReplayChooser />
          <FilePicker source={source} error={error} onPickFile={onPickFile} />
          <ViewControls backend={backend} setBackend={setBackend} reveal={reveal} setReveal={setReveal}
            swap={swap} onSwap={() => setSwap((s) => !s)} />
          <a className="mode-link" href="?mode=play">▶ Play vs bot</a>
        </div>
        <Timeline pb={pb} turn={step.current?.turn ?? null} />
      </header>

      <main className="main">
        <ErrorBoundary resetKey={pb.index}>
          <Board step={step} db={db} diff={diff} backend={backend} reveal={reveal} bottomIndex={bottomIndex} />
        </ErrorBoundary>
        <aside className="sidebar">
          {replay.subAgentLog && (
            <SubAgentPanel log={replay.subAgentLog[pb.index]} index={pb.index} />
          )}
          <LogPanel step={step} db={db} />
          <StatsPanel stats={stats[pb.index]} />
          <DiffPanel diff={diff} />
        </aside>
      </main>

      {dragging && <div className="drop-hint">Drop a replay .json to load it</div>}
    </div>
  );
}

function FilePicker({
  source,
  error,
  onPickFile,
}: {
  source: string;
  error: string | null;
  onPickFile: (file: File) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <div className="source">
      <button className="file-btn" onClick={() => ref.current?.click()}>
        Load replay…
      </button>
      <input
        ref={ref}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPickFile(f);
          e.target.value = ""; // allow re-picking the same file
        }}
      />
      <span className="source-name" title={source}>{source}</span>
      {error && <span className="source-err">{error}</span>}
    </div>
  );
}

function ViewControls({
  backend, setBackend, reveal, setReveal, swap, onSwap,
}: {
  backend: CardBackend;
  setBackend: (b: CardBackend) => void;
  reveal: "realistic" | "full-info";
  setReveal: (r: "realistic" | "full-info") => void;
  swap: boolean;
  onSwap: () => void;
}) {
  return (
    <div className="view-controls">
      <button className="swap-btn" onClick={onSwap} title="Swap which player is on the bottom">
        ⇅ Swap sides{swap ? " (flipped)" : ""}
      </button>
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
