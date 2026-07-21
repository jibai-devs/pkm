import { mergeStep } from "./stepState";
import type { Replay } from "./types";

export interface PlayerStats {
  damage: number; // cumulative damage dealt to the opponent
  attacks: number; // cumulative attacks used
  cardsPlayed: number; // cumulative "play card" events
  prizesTaken: number; // 6 - prizes remaining (derived from state)
}
export type StepStats = [PlayerStats, PlayerStats];

const INITIAL_PRIZES = 6;

// Precompute cumulative stats for every step in one pass over the logs.
export function computeStats(replay: Replay): StepStats[] {
  const out: StepStats[] = [];
  const acc: StepStats = [zero(), zero()];

  for (let n = 0; n < replay.steps.length; n++) {
    const step = mergeStep(replay, n);
    for (const ev of step.logs) {
      const p = ev.playerIndex;
      if (p !== 0 && p !== 1) continue;
      switch (ev.type) {
        case 15: // attack
          acc[p].attacks += 1;
          break;
        case 10: // play card
          acc[p].cardsPlayed += 1;
          break;
        case 16: // HP change: damage is dealt BY the opponent of the card owner
          if ((ev.value ?? 0) < 0) {
            const attacker = (1 - p) as 0 | 1;
            acc[attacker].damage += Math.abs(ev.value ?? 0);
          }
          break;
      }
    }
    // prizes taken is a snapshot, not accumulated
    const prizesTaken = ([0, 1] as const).map((i) => {
      const remaining = step.current?.players?.[i]?.prize?.length ?? INITIAL_PRIZES;
      return Math.max(0, INITIAL_PRIZES - remaining);
    });
    out.push([
      { ...acc[0], prizesTaken: prizesTaken[0] },
      { ...acc[1], prizesTaken: prizesTaken[1] },
    ]);
  }
  return out;
}

function zero(): PlayerStats {
  return { damage: 0, attacks: 0, cardsPlayed: 0, prizesTaken: 0 };
}
