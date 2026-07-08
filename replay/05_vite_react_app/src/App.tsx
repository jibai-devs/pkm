import { useEffect, useMemo, useState } from "react";
import { Board } from "./components/Board";
import { DiffPanel } from "./components/DiffPanel";
import { LogPanel } from "./components/LogPanel";
import { StatsPanel } from "./components/StatsPanel";
import { Timeline } from "./components/Timeline";
import { CardDb, loadCardDb } from "./data/cardDb";
import { computeDiff } from "./data/diff";
import { loadReplay } from "./data/loadReplay";
import { computeStats, type StepStats } from "./data/stats";
import { mergeStep } from "./data/stepState";
import type { Replay } from "./data/types";
import { usePlayback } from "./state/usePlayback";

interface Loaded {
  replay: Replay;
  db: CardDb;
  stats: StepStats[];
}

export default function App() {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([loadReplay(), loadCardDb()])
      .then(([replay, db]) => setData({ replay, db, stats: computeStats(replay) }))
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="fatal">Failed to load: {error}</div>;
  if (!data) return <div className="loading">Loading replay…</div>;
  return <Viewer {...data} />;
}

function initialStep(): number {
  const p = new URLSearchParams(window.location.search).get("step");
  const n = p ? parseInt(p, 10) : 1;
  return Number.isFinite(n) ? n - 1 : 0; // ?step is 1-based to match the UI
}

function Viewer({ replay, db, stats }: Loaded) {
  const pb = usePlayback(replay.steps.length, initialStep());

  // Keep ?step= in sync so the current position is shareable / survives reload.
  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("step", String(pb.index + 1));
    window.history.replaceState(null, "", url);
  }, [pb.index]);

  const step = useMemo(() => mergeStep(replay, pb.index), [replay, pb.index]);
  const prev = useMemo(
    () => (pb.index > 0 ? mergeStep(replay, pb.index - 1) : null),
    [replay, pb.index],
  );
  const diff = useMemo(() => computeDiff(prev, step), [prev, step]);

  // Keyboard: ← → step, space play/pause, home/end.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") pb.step(1);
      else if (e.key === "ArrowLeft") pb.step(-1);
      else if (e.key === " ") { e.preventDefault(); pb.togglePlay(); }
      else if (e.key === "Home") pb.setIndex(0);
      else if (e.key === "End") pb.setIndex(pb.count - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pb]);

  return (
    <div className="app">
      <header className="topbar">
        <h1>{replay.title || replay.name || "PTCG Replay"}</h1>
        <Timeline pb={pb} turn={step.current?.turn ?? null} />
      </header>

      <main className="main">
        <Board step={step} db={db} diff={diff} />
        <aside className="sidebar">
          <LogPanel step={step} db={db} />
          <StatsPanel stats={stats[pb.index]} />
          <DiffPanel diff={diff} />
        </aside>
      </main>
    </div>
  );
}
