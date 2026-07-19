import type { Replay } from "./types";

// Validate/parse an already-decoded JSON object into a Replay.
export function parseReplay(data: unknown): Replay {
  const r = data as Replay;
  if (!r || !Array.isArray(r.steps)) {
    throw new Error("not a replay file — missing steps[]");
  }
  return r;
}

// Load and parse replay.json once into memory. ~19.6 MB / 284 steps parses
// in ~1-2s; each step already carries a full materialized state snapshot, so
// there is no streaming/log-replay needed for navigation.
export async function loadReplay(url = "/replay.json"): Promise<Replay> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return parseReplay(await res.json());
}

// Read a replay from a user-picked local File (works with any file on disk,
// unlike fetch which only reaches files the dev server serves).
export async function readReplayFile(file: File): Promise<Replay> {
  return parseReplay(JSON.parse(await file.text()));
}

// Resolve which replay/cards URL to fetch, in precedence order:
//   ?replay= (or ?file=) query param  >  VITE_REPLAY env  >  default
export function resolveReplayUrl(): string {
  const q = new URLSearchParams(window.location.search);
  return q.get("replay") || q.get("file") || __REPLAY_URL__ || "/replay.json";
}
export function resolveCardsUrl(): string {
  return __CARDS_URL__ || "/cards.json";
}
