import type { CardDb } from "../data/cardDb";
import type { StepDiff } from "../data/diff";
import type { MergedStep } from "../data/stepState";
import { PlayerBoard } from "./PlayerBoard";

interface Props {
  step: MergedStep;
  db: CardDb;
  diff: StepDiff;
}

function winnerText(rewards: [number, number]): string {
  if (rewards[0] === rewards[1]) return "Draw";
  return rewards[0] > rewards[1] ? "P0 wins" : "P1 wins";
}

export function Board({ step, db, diff }: Props) {
  const cur = step.current;
  const done = step.statuses.includes("DONE");

  // Only the very first step has a null board (env reset), not a game-over.
  if (!cur) {
    return (
      <div className="board board-end">
        <div className="end-card">
          <h2>Game starting…</h2>
          <p>Initial setup — step forward to begin.</p>
        </div>
      </div>
    );
  }

  const stadium = (cur.stadium ?? []).filter(Boolean);
  return (
    <div className="board">
      {done && (
        <div className="winner-banner">Game over — {winnerText(step.rewards)}</div>
      )}
      <div className="board-meta">
        <span>Turn {cur.turn}</span>
        <span>Action {cur.turnActionCount}</span>
        {cur.supporterPlayed && <span className="flagged">supporter</span>}
        {cur.energyAttached && <span className="flagged">energy</span>}
        {cur.retreated && <span className="flagged">retreated</span>}
        {stadium.length > 0 && (
          <span className="stadium">Stadium: {db.name(stadium[0]!.id)}</span>
        )}
      </div>
      <div className="players">
        <PlayerBoard
          player={cur.players[0]}
          index={0}
          db={db}
          diff={diff}
          active={step.activePlayer === 0}
        />
        <PlayerBoard
          player={cur.players[1]}
          index={1}
          db={db}
          diff={diff}
          active={step.activePlayer === 1}
        />
      </div>
    </div>
  );
}
