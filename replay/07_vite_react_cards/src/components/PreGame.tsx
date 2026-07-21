import { useState } from "react";
import type { LiveConfig, StartRequest } from "../live/types";

// Pre-game screen: pick the bot opponent, the deck (both sides play it, same as
// the TUI), and which side you sit on. Then start.
interface Props {
  config: LiveConfig;
  busy: boolean;
  error: string | null;
  onStart: (req: StartRequest) => void;
}

const OPPONENT_LABELS: Record<string, string> = {
  neural: "Neural (trained policy)",
  mcts: "MCTS (search)",
  random: "Random",
  singaporean_middleman: "Middleman (heuristics)",
};

export function PreGame({ config, busy, error, onStart }: Props) {
  const [opponent, setOpponent] = useState(config.defaultOpponent);
  const [deck, setDeck] = useState(config.defaultDeck);
  const [humanIndex, setHumanIndex] = useState<0 | 1>(0);

  return (
    <div className="pregame">
      <div className="pregame-card">
        <h1>Play vs bot</h1>
        <p className="pregame-sub">
          Pick an opponent and deck. Both players use the same deck, just like the
          terminal battle.
        </p>

        <label className="pregame-field">
          <span>Opponent</span>
          <select value={opponent} onChange={(e) => setOpponent(e.target.value)}>
            {config.opponents.map((o) => (
              <option key={o} value={o}>
                {OPPONENT_LABELS[o] ?? o}
              </option>
            ))}
          </select>
        </label>

        <label className="pregame-field">
          <span>Deck</span>
          <select value={deck} onChange={(e) => setDeck(e.target.value)}>
            {config.decks.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>

        <label className="pregame-field">
          <span>Your side</span>
          <select
            value={humanIndex}
            onChange={(e) => setHumanIndex(Number(e.target.value) as 0 | 1)}
          >
            <option value={0}>Player 1 (go first)</option>
            <option value={1}>Player 2</option>
          </select>
        </label>

        <button
          className="start-btn"
          disabled={busy || !deck}
          onClick={() => onStart({ opponent, deck, humanIndex })}
        >
          {busy ? "Starting…" : "Start game"}
        </button>
        {error && <div className="pregame-err">{error}</div>}
      </div>
    </div>
  );
}
