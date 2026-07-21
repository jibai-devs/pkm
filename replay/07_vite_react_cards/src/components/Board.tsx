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
  /** Which player index sits at the bottom (the "player" side). */
  bottomIndex: 0 | 1;
}

function winnerText(rewards: [number, number]): string {
  if (rewards[0] === rewards[1]) return "Draw";
  return rewards[0] > rewards[1] ? "P0 wins" : "P1 wins";
}

export function Board({ step, db, diff, backend, reveal, bottomIndex }: Props) {
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

  const topIndex = (bottomIndex === 0 ? 1 : 0) as 0 | 1;

  return (
    <div className="board board-cards">
      {done && <div className="winner-banner">Game over — {winnerText(step.rewards)}</div>}

      <PlayerBoard player={cur.players[topIndex]} index={topIndex} db={db} diff={diff}
        active={step.activePlayer === topIndex} side="top" backend={backend} revealHand={revealHand(topIndex)} />

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

      <PlayerBoard player={cur.players[bottomIndex]} index={bottomIndex} db={db} diff={diff}
        active={step.activePlayer === bottomIndex} side="bottom" backend={backend} revealHand={revealHand(bottomIndex)} />
    </div>
  );
}
