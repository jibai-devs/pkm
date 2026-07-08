import type { CardDb } from "../data/cardDb";
import type { StepDiff } from "../data/diff";
import type { PlayerState } from "../data/types";
import { Card } from "./Card";

interface Props {
  player: PlayerState;
  index: number;
  db: CardDb;
  diff: StepDiff;
  active: boolean; // is this the acting player this step
}

const STATUS_FLAGS: [keyof PlayerState, string][] = [
  ["poisoned", "Poisoned"],
  ["burned", "Burned"],
  ["asleep", "Asleep"],
  ["paralyzed", "Paralyzed"],
  ["confused", "Confused"],
];

export function PlayerBoard({ player, index, db, diff, active }: Props) {
  const statuses = STATUS_FLAGS.filter(([k]) => player[k]);
  const cardProps = (serial: number) => ({
    hpDelta: diff.changedHp.get(serial),
    appeared: diff.appeared.has(serial),
  });

  return (
    <section className={`player ${active ? "player-active" : ""}`}>
      <header className="player-head">
        <span className="player-tag">P{index}</span>
        {active && <span className="turn-flag">▶ acting</span>}
        <span className="counts">
          Hand {player.handCount} · Deck {player.deckCount} · Discard {player.discard.length} · Prize {player.prize.length}
        </span>
      </header>

      <div className="zone-label">Active</div>
      <div className="row active-row">
        {(player.active ?? []).filter(Boolean).map((c) => (
          <Card key={c!.serial} card={c!} db={db} variant="active" {...cardProps(c!.serial)} />
        ))}
        {(player.active ?? []).filter(Boolean).length === 0 && <div className="empty">—</div>}
      </div>

      <div className="zone-label">Bench ({(player.bench ?? []).filter(Boolean).length}/{player.benchMax})</div>
      <div className="row bench-row">
        {(player.bench ?? []).filter(Boolean).map((c) => (
          <Card key={c!.serial} card={c!} db={db} variant="bench" {...cardProps(c!.serial)} />
        ))}
        {(player.bench ?? []).filter(Boolean).length === 0 && <div className="empty">—</div>}
      </div>

      {statuses.length > 0 && (
        <div className="statuses">
          {statuses.map(([, label]) => (
            <span key={label} className="status-chip">{label}</span>
          ))}
        </div>
      )}

      {player.hand.length > 0 && (
        <>
          <div className="zone-label">Hand ({player.hand.length})</div>
          <div className="row hand-row">
            {player.hand.map((c) => (
              <Card key={c.serial} card={c} db={db} variant="hand" />
            ))}
          </div>
        </>
      )}

      {player.discard.length > 0 && (
        <>
          <div className="zone-label">Discard ({player.discard.length})</div>
          <div className="row discard-row">
            {player.discard.map((c) => (
              <Card key={c.serial} card={c} db={db} variant="hand" {...cardProps(c.serial)} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}
