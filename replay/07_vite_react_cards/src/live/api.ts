// Thin fetch client for the play server (pkm/web/server.py). All paths are
// relative to the current origin: in dev, Vite proxies /api to the Python
// server (see vite.config.ts); in prod the Python server serves this SPA too.
import type { LiveConfig, LiveEvent, StartRequest } from "./types";

async function postJson(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error((data as { error?: string }).error ?? `${path}: ${res.status}`);
  }
  return data;
}

export async function fetchConfig(): Promise<LiveConfig> {
  const res = await fetch("/api/config");
  if (!res.ok) throw new Error(`/api/config: ${res.status}`);
  return res.json();
}

export async function startGame(req: StartRequest): Promise<void> {
  await postJson("/api/start", req);
}

export async function submitPicks(picks: number[]): Promise<void> {
  await postJson("/api/submit", { picks });
}

export async function quitGame(): Promise<void> {
  await postJson("/api/quit", {});
}

// One long-poll. Resolves with the next event (may be a heartbeat if the server
// timed out waiting). `signal` lets the caller abort in-flight on unmount.
export async function nextEvent(signal?: AbortSignal): Promise<LiveEvent> {
  const res = await fetch("/api/event", { signal });
  if (!res.ok) throw new Error(`/api/event: ${res.status}`);
  return res.json();
}
