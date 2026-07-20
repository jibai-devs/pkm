// Adapt a single live observation into the MergedStep the replay Board/LogPanel
// already know how to render.
//
// A live obs is single-POV (the human's own side): their hand is populated, the
// opponent's is null — exactly the "realistic" hidden-info view. So there is no
// two-entry merge to do (unlike replay data); we just normalize both players and
// wrap it. Logs are accumulated by the caller (each obs carries only the delta
// since the last one), so we take the running list as an argument.
import type { LogEvent, PlayerState } from "../data/types";
import { normalizePlayer, type MergedStep } from "../data/stepState";
import type { Observation } from "../data/types";

export function liveMergedStep(
  obs: Observation,
  logs: LogEvent[],
  index: number,
  opts: { done?: boolean; rewards?: [number, number] } = {},
): MergedStep {
  const cur = obs.current;
  const current = cur
    ? {
        ...cur,
        players: [
          normalizePlayer(cur.players?.[0]),
          normalizePlayer(cur.players?.[1]),
        ] as [PlayerState, PlayerState],
      }
    : null;
  const done = opts.done ?? false;
  return {
    index,
    current,
    activePlayer: cur?.yourIndex ?? null,
    logs,
    select: obs.select ?? null,
    // Statuses drive the Board's "game over" banner; while playing the human's
    // side is ACTIVE (they're the one being prompted).
    statuses: done ? ["DONE", "DONE"] : ["ACTIVE", "INACTIVE"],
    rewards: opts.rewards ?? [0, 0],
  };
}
