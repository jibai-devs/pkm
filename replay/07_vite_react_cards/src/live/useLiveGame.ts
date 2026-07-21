import { useCallback, useEffect, useRef, useState } from "react";
import type { LogEvent, Observation } from "../data/types";
import { nextEvent, quitGame, startGame, submitPicks } from "./api";
import type { LivePrompt, StartRequest } from "./types";

// Where the game is in its lifecycle. The long-poll pump runs only in "waiting";
// it parks the machine in "prompt" (human's turn) or a terminal state.
export type Phase =
  | "pregame" // choosing opponent/deck, no game running
  | "connecting" // POST /api/start in flight
  | "waiting" // bot / engine is thinking; long-poll outstanding
  | "prompt" // human must choose from `prompt.options`
  | "finished"
  | "error";

export interface LiveGame {
  phase: Phase;
  humanIndex: 0 | 1;
  obs: Observation | null;
  prompt: LivePrompt | null;
  logs: LogEvent[];
  notes: string[];
  result: "win" | "lose" | "draw" | "opponent_crashed" | "you_errored" | "unknown" | null;
  rewards: [number, number];
  error: string | null;
  start: (req: StartRequest) => Promise<void>;
  submit: (picks: number[]) => Promise<void>;
  quit: () => Promise<void>;
}

export function useLiveGame(): LiveGame {
  const [phase, setPhase] = useState<Phase>("pregame");
  const [humanIndex, setHumanIndex] = useState<0 | 1>(0);
  const [obs, setObs] = useState<Observation | null>(null);
  const [prompt, setPrompt] = useState<LivePrompt | null>(null);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [notes, setNotes] = useState<string[]>([]);
  const [result, setResult] = useState<LiveGame["result"]>(null);
  const [rewards, setRewards] = useState<[number, number]>([0, 0]);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  useEffect(() => {
    abortRef.current = new AbortController();
    return () => {
      mounted.current = false;
      abortRef.current?.abort();
    };
  }, []);

  // Drain events until one parks the machine (prompt / finished / error).
  // Heartbeats and opponent notes don't park it — keep polling.
  const pump = useCallback(async () => {
    while (mounted.current) {
      let ev;
      try {
        ev = await nextEvent(abortRef.current?.signal);
      } catch (e) {
        if (!mounted.current || (e as Error).name === "AbortError") return;
        setError(String(e));
        setPhase("error");
        return;
      }
      if (!mounted.current) return;
      if (ev.type === "heartbeat") continue;
      if (ev.type === "note") {
        setNotes((n) => [...n, ev.message]);
        continue;
      }
      if (ev.type === "prompt") {
        setHumanIndex(ev.humanIndex);
        setObs(ev.obs);
        // obs.logs is a delta (only what happened since the last obs) — append.
        setLogs((prev) => [...prev, ...(ev.obs.logs ?? [])]);
        setPrompt(ev.prompt);
        setPhase("prompt");
        return;
      }
      if (ev.type === "finished") {
        setResult(ev.result);
        setRewards([ev.rewards[0] ?? 0, ev.rewards[1] ?? 0]);
        setPhase("finished");
        return;
      }
      if (ev.type === "error") {
        setError(ev.message);
        setPhase("error");
        return;
      }
    }
  }, []);

  const start = useCallback(
    async (req: StartRequest) => {
      setError(null);
      setLogs([]);
      setNotes([]);
      setResult(null);
      setObs(null);
      setPrompt(null);
      setRewards([0, 0]);
      setHumanIndex(req.humanIndex);
      setPhase("connecting");
      try {
        await startGame(req);
      } catch (e) {
        setError(String(e));
        setPhase("error");
        return;
      }
      setPhase("waiting");
      void pump();
    },
    [pump],
  );

  const submit = useCallback(
    async (picks: number[]) => {
      setPrompt(null);
      setPhase("waiting");
      try {
        await submitPicks(picks);
      } catch (e) {
        setError(String(e));
        setPhase("error");
        return;
      }
      void pump();
    },
    [pump],
  );

  const quit = useCallback(async () => {
    try {
      await quitGame();
    } catch {
      // best effort — we're tearing the game down anyway
    }
    setPhase("pregame");
    setPrompt(null);
    setObs(null);
  }, []);

  return {
    phase,
    humanIndex,
    obs,
    prompt,
    logs,
    notes,
    result,
    rewards,
    error,
    start,
    submit,
    quit,
  };
}
