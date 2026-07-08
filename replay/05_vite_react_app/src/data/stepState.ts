import type {
  CurrentState,
  LogEvent,
  PlayerState,
  Replay,
  SelectPrompt,
  Step,
} from "./types";

// A fully-merged view of one step: both players' hands are revealed by taking
// each player's slice from their OWN point-of-view entry (in the raw data each
// entry only reveals its own hand).
export interface MergedStep {
  index: number;
  current: CurrentState | null;
  activePlayer: number | null; // which entry has status === "ACTIVE"
  logs: LogEvent[]; // from the acting player's POV (richest single source)
  select: SelectPrompt | null;
  statuses: [string, string];
  rewards: [number, number];
}

function mergeCurrent(step: Step): CurrentState | null {
  const c0 = step[0].observation.current;
  const c1 = step[1].observation.current;
  const base = c0 ?? c1;
  if (!base) return null;
  const povCurrents = [c0, c1];
  const players = ([0, 1] as const).map((i) => {
    // player i as seen from their own POV entry (hand revealed), else fallback
    const own = povCurrents[i]?.players?.[i];
    return own ?? base.players?.[i] ?? emptyPlayer();
  }) as [PlayerState, PlayerState];
  return { ...base, players };
}

function emptyPlayer(): PlayerState {
  return {
    active: [],
    bench: [],
    benchMax: 5,
    deckCount: 0,
    handCount: 0,
    hand: [],
    discard: [],
    prize: [],
    poisoned: false,
    burned: false,
    asleep: false,
    paralyzed: false,
    confused: false,
  };
}

export function mergeStep(replay: Replay, index: number): MergedStep {
  const step = replay.steps[index];
  const active =
    step[0].status === "ACTIVE" ? 0 : step[1].status === "ACTIVE" ? 1 : null;
  const src = active ?? 0;
  return {
    index,
    current: mergeCurrent(step),
    activePlayer: active,
    logs: step[src].observation.logs ?? step[0].observation.logs ?? [],
    select: step[src].observation.select ?? null,
    statuses: [step[0].status, step[1].status],
    rewards: [step[0].reward, step[1].reward],
  };
}

// All card serials currently in play (active + bench + stadium). Used for diffs.
export function inPlaySerials(cur: CurrentState | null): Set<number> {
  const s = new Set<number>();
  if (!cur) return s;
  for (const p of cur.players) {
    for (const c of [...(p.active ?? []), ...(p.bench ?? [])]) {
      if (c) s.add(c.serial);
    }
  }
  for (const c of cur.stadium ?? []) if (c) s.add(c.serial);
  return s;
}
