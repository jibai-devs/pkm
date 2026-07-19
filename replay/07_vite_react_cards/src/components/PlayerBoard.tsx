import type { CardDb } from "../data/cardDb";
import type { StepDiff } from "../data/diff";
import type { PlayerState } from "../data/types";
import type { CardBackend } from "../data/cardArt";
import { Card, CardBack } from "./Card";

interface Props {
  player: PlayerState;
  index: number;
  db: CardDb;
  diff: StepDiff;
  active: boolean;
  side: "top" | "bottom";
  backend: CardBackend;
  revealHand: boolean;
}

const STATUS_FLAGS: [keyof PlayerState, string][] = [
  ["poisoned", "Poisoned"], ["burned", "Burned"], ["asleep", "Asleep"],
  ["paralyzed", "Paralyzed"], ["confused", "Confused"],
];

export function PlayerBoard({ player, index, db, diff, active, side, backend, revealHand }: Props) {
  const statuses = STATUS_FLAGS.filter(([k]) => player[k]);
  const cardProps = (serial: number) => ({
    hpDelta: diff.changedHp.get(serial),
    appeared: diff.appeared.has(serial),
  });
  const activeCards = (player.active ?? []).filter(Boolean) as NonNullable<PlayerState["active"][number]>[];
  const benchCards = (player.bench ?? []).filter(Boolean) as NonNullable<PlayerState["bench"][number]>[];
  const benchEmpties = Math.max(0, player.benchMax - benchCards.length);

  const prizes = (
    <div className="zone prizes-zone">
      <div className="zone-label">Prizes ({player.prize.length})</div>
      <div className="prizes-grid">
        {Array.from({ length: player.prize.length }).map((_, i) => <CardBack key={i} variant="mini" />)}
      </div>
    </div>
  );

  const piles = (
    <div className="zone piles-zone">
      <div className="pile">
        <div className="zone-label">Deck ({player.deckCount})</div>
        {player.deckCount > 0 ? <CardBack variant="mini" /> : <div className="empty">—</div>}
      </div>
      <div className="pile">
        <div className="zone-label">Discard ({player.discard.length})</div>
        {player.discard.length > 0
          ? <Card card={player.discard[player.discard.length - 1]} db={db} variant="mini" backend={backend} />
          : <div className="empty">—</div>}
      </div>
    </div>
  );

  const benchRow = (
    <div className="zone">
      <div className="zone-label">Bench ({benchCards.length}/{player.benchMax})</div>
      <div className="row bench-row">
        {benchCards.map((c) => <Card key={c.serial} card={c} db={db} variant="bench" backend={backend} {...cardProps(c.serial)} />)}
        {Array.from({ length: benchEmpties }).map((_, i) => <div key={`e${i}`} className="slot">bench</div>)}
      </div>
    </div>
  );

  const activeRow = (
    <div className="zone">
      <div className="zone-label">Active</div>
      <div className="row active-row">
        {activeCards.length
          ? activeCards.map((c) => <div key={c.serial} className="active-frame"><Card card={c} db={db} variant="active" backend={backend} {...cardProps(c.serial)} /></div>)
          : <div className="slot">active</div>}
      </div>
    </div>
  );

  const hand = (
    <div className="zone hand-zone">
      <div className="zone-label">Hand ({player.handCount})</div>
      <div className="row hand-fan">
        {revealHand
          ? player.hand.map((c) => <Card key={c.serial} card={c} db={db} variant="hand" backend={backend} />)
          : Array.from({ length: player.handCount }).map((_, i) => <CardBack key={i} variant="hand" />)}
      </div>
    </div>
  );

  // top: bench above active (active nearest centre); bottom: active above bench.
  const field = (
    <div className="zone field-zone">
      {side === "top" ? <>{benchRow}{activeRow}</> : <>{activeRow}{benchRow}</>}
      {statuses.length > 0 && (
        <div className="statuses">
          {statuses.map(([, label]) => <span key={label} className="status-chip">{label}</span>)}
        </div>
      )}
    </div>
  );

  return (
    <section className={`pl side-${side} ${active ? "pl-active" : ""}`}>
      <header className="pl-head">
        <span className="player-tag">P{index}</span>
        {active && <span className="turn-flag">▶ acting</span>}
        <span className="counts">Hand {player.handCount} · Deck {player.deckCount} · Discard {player.discard.length} · Prize {player.prize.length}</span>
      </header>
      {side === "top" && hand}
      <div className="pl-band">
        {prizes}
        {field}
        {piles}
      </div>
      {side === "bottom" && hand}
    </section>
  );
}
