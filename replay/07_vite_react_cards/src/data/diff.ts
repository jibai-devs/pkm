import type { MergedStep } from "./stepState";
import { inPlaySerials } from "./stepState";

// What changed between step n-1 and step n. HP deltas come from this step's
// type-16 logs (authoritative); appeared/disappeared from comparing the set of
// in-play serials. Used both for the Diff panel and for per-card flash cues.
export interface StepDiff {
  changedHp: Map<number, number>; // serial -> hp delta (negative = damage)
  appeared: Set<number>; // serials that entered play this step
  disappeared: Set<number>; // serials that left play (KO / retreat / discard)
}

export function computeDiff(prev: MergedStep | null, cur: MergedStep): StepDiff {
  const changedHp = new Map<number, number>();
  for (const ev of cur.logs) {
    if (ev.type === 16 && ev.serial != null && ev.value != null) {
      changedHp.set(ev.serial, (changedHp.get(ev.serial) ?? 0) + ev.value);
    }
  }

  const prevSet = inPlaySerials(prev?.current ?? null);
  const curSet = inPlaySerials(cur.current);
  const appeared = new Set<number>();
  const disappeared = new Set<number>();
  for (const s of curSet) if (!prevSet.has(s)) appeared.add(s);
  for (const s of prevSet) if (!curSet.has(s)) disappeared.add(s);

  return { changedHp, appeared, disappeared };
}

export const EMPTY_DIFF: StepDiff = {
  changedHp: new Map(),
  appeared: new Set(),
  disappeared: new Set(),
};
