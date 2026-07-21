import { useEffect, useMemo, useRef, useState } from "react";
import { Board } from "../components/Board";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { LogPanel } from "../components/LogPanel";
import { OptionsPane } from "../components/OptionsPane";
import { PreGame } from "../components/PreGame";
import { defaultBackend, type CardBackend } from "../data/cardArt";
import { CardDb, loadCardDb } from "../data/cardDb";
import { computeDiff } from "../data/diff";
import { resolveCardsUrl } from "../data/loadReplay";
import type { MergedStep } from "../data/stepState";
import { fetchConfig } from "./api";
import { liveMergedStep } from "./liveStep";
import type { LiveConfig } from "./types";
import { useLiveGame } from "./useLiveGame";

const RESULT_TEXT: Record<string, string> = {
  win: "You win! 🎉",
  lose: "You lose.",
  draw: "Draw.",
  opponent_crashed: "⚠️ Opponent bot crashed — not a real win.",
  you_errored: "⚠️ Your side errored out.",
  unknown: "Game over.",
};

// Extra context shown under the headline for the non-clean endings.
const RESULT_NOTE: Record<string, string> = {
  opponent_crashed:
    "The bot failed to make a move (its saved weights don't match the current " +
    "model — see AGENTS.md). Try the Random opponent, which works, or retrain a " +
    "compatible policy.",
  you_errored: "The engine rejected a move from your side.",
};

export function LiveApp() {
  const [db, setDb] = useState<CardDb | null>(null);
  const [config, setConfig] = useState<LiveConfig | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([loadCardDb(resolveCardsUrl()), fetchConfig()])
      .then(([d, c]) => {
        setDb(d);
        setConfig(c);
      })
      .catch((e) => setLoadError(String(e)));
  }, []);

  if (loadError) return <div className="fatal">Failed to load: {loadError}</div>;
  if (!db || !config) return <div className="loading">Loading…</div>;
  return <LiveTable db={db} config={config} />;
}

function LiveTable({ db, config }: { db: CardDb; config: LiveConfig }) {
  const game = useLiveGame();
  const [backend, setBackend] = useState<CardBackend>(defaultBackend());

  // Build the MergedStep from the latest observation; keep the previous one so
  // the Board can flash HP/appear/disappear diffs between decisions.
  const prevStepRef = useRef<MergedStep | null>(null);
  const stepIndexRef = useRef(0);

  const step = useMemo<MergedStep | null>(() => {
    if (!game.obs) return null;
    const done = game.phase === "finished";
    return liveMergedStep(game.obs, game.logs, stepIndexRef.current, {
      done,
      rewards: game.rewards,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [game.obs, game.logs, game.phase, game.rewards]);

  const diff = useMemo(
    () => (step ? computeDiff(prevStepRef.current, step) : null),
    [step],
  );

  useEffect(() => {
    if (step && game.phase === "prompt") {
      prevStepRef.current = step;
      stepIndexRef.current += 1;
    }
  }, [step, game.phase]);

  if (game.phase === "pregame" || game.phase === "connecting") {
    return (
      <PreGame
        config={config}
        busy={game.phase === "connecting"}
        error={game.error}
        onStart={game.start}
      />
    );
  }

  const bottomIndex = game.humanIndex;
  const waiting = game.phase === "waiting";

  return (
    <div className="app live-app">
      <header className="topbar">
        <div className="topline">
          <h1>Play vs bot</h1>
          <div className="live-status">
            {waiting && <span className="thinking">Bot is thinking…</span>}
            {game.phase === "prompt" && <span className="your-turn">Your move</span>}
            {game.phase === "error" && (
              <span className="live-err">Error: {game.error}</span>
            )}
          </div>
          <div className="view-controls">
            <label className="ctl">
              Art
              <select
                value={backend}
                onChange={(e) => setBackend(e.target.value as CardBackend)}
              >
                <option value="local">local</option>
                <option value="cdn">cdn</option>
              </select>
            </label>
            <button className="file-btn" onClick={game.quit}>
              New game
            </button>
            <a className="mode-link" href="./">↩ Replays</a>
          </div>
        </div>
      </header>

      <main className="main">
        <ErrorBoundary resetKey={stepIndexRef.current}>
          {step && diff ? (
            <Board
              step={step}
              db={db}
              diff={diff}
              backend={backend}
              reveal="realistic"
              bottomIndex={bottomIndex}
            />
          ) : (
            <div className="board board-end">
              <div className="end-card">
                <h2>Setting up…</h2>
              </div>
            </div>
          )}
        </ErrorBoundary>

        <aside className="sidebar">
          {game.phase === "finished" ? (
            <div className="panel result-panel">
              <div className="panel-title">Result</div>
              <div className="result-text">
                {RESULT_TEXT[game.result ?? "unknown"]}
              </div>
              {game.result && RESULT_NOTE[game.result] && (
                <div className="result-note">{RESULT_NOTE[game.result]}</div>
              )}
              <button className="start-btn" onClick={game.quit}>
                Play again
              </button>
            </div>
          ) : (
            game.phase === "prompt" &&
            game.prompt && (
              <div className="panel">
                <div className="panel-title">Your options</div>
                <OptionsPane
                  prompt={game.prompt}
                  disabled={false}
                  onSubmit={game.submit}
                />
              </div>
            )
          )}
          {game.notes.length > 0 && (
            <div className="panel notes-panel">
              <div className="panel-title">Opponent notes</div>
              {game.notes.slice(-6).map((n, i) => (
                <div key={i} className="note-row">
                  {n}
                </div>
              ))}
            </div>
          )}
          {step && <LogPanel step={step} db={db} />}
        </aside>
      </main>
    </div>
  );
}
