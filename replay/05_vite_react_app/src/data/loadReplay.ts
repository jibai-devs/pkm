import type { Replay } from "./types";

// Load and parse replay.json once into memory. ~19.6 MB / 284 steps parses
// in ~1-2s; each step already carries a full materialized state snapshot, so
// there is no streaming/log-replay needed for navigation.
export async function loadReplay(url = "/replay.json"): Promise<Replay> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  const data: Replay = await res.json();
  if (!Array.isArray(data.steps)) throw new Error("replay.json has no steps[]");
  return data;
}
