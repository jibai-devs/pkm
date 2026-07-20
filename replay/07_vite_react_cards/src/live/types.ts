// Wire protocol for interactive play — mirrors pkm/web/server.py.
// The server sends one event per `GET /api/event` long-poll.
import type { Observation } from "../data/types";

export interface LiveOption {
  index: number;
  label: string; // rendered server-side by pkm/tui/labels.option_label
  type: number;
  irreversible: boolean; // attack / end-turn: the UI should confirm before submit
}

export interface LivePrompt {
  minCount: number;
  maxCount: number;
  type: number;
  context: number;
  options: LiveOption[];
}

export type LiveEvent =
  | { type: "prompt"; humanIndex: 0 | 1; obs: Observation; prompt: LivePrompt }
  | { type: "note"; message: string }
  | {
      type: "finished";
      result: "win" | "lose" | "draw" | "opponent_crashed" | "you_errored" | "unknown";
      rewards: (number | null)[];
    }
  | { type: "error"; message: string }
  | { type: "heartbeat" };

export interface LiveConfig {
  opponents: string[];
  decks: string[];
  defaultOpponent: string;
  defaultDeck: string;
}

export interface StartRequest {
  opponent: string;
  deck: string;
  humanIndex: 0 | 1;
}
