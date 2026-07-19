import { useEffect, useState } from "react";
import type { CardDb } from "../data/cardDb";
import { energyType } from "../data/energy";
import type { CardInstance } from "../data/types";
import { resolveCardArt, type CardBackend } from "../data/cardArt";

interface Props {
  card: CardInstance;
  db: CardDb;
  variant: "active" | "bench" | "hand" | "mini";
  backend: CardBackend;
  hpDelta?: number;
  appeared?: boolean;
}

function EnergyPips({ energies }: { energies: number[] }) {
  if (!energies?.length) return null;
  return (
    <span className="pips">
      {energies.map((code, i) => {
        const t = energyType(code);
        return (
          <span key={i} className="pip" style={{ background: t.color, color: t.fg }} title={t.name}>
            {t.letter}
          </span>
        );
      })}
    </span>
  );
}

// Local-first (or cdn-first) <img> that swaps to the other backend once on
// error, then reports itself dead via onDead so the parent Card can render a
// visible text-name fallback in its place. Never a broken image icon.
function CardArt({ id, backend, alt, onDead }: { id: number; backend: CardBackend; alt: string; onDead: () => void }) {
  const { primary, fallback } = resolveCardArt(id, backend);
  const [src, setSrc] = useState(primary);
  useEffect(() => { setSrc(primary); }, [primary]);
  return (
    <img
      className="card-art"
      src={src}
      alt={alt}
      draggable={false}
      onError={() => (src !== fallback ? setSrc(fallback) : onDead())}
    />
  );
}

// Face-down card (opponent hand in realistic mode, prizes, deck). CSS-drawn.
export function CardBack({ variant = "mini" }: { variant?: Props["variant"] }) {
  return <div className={`card card-${variant} card-back`} aria-hidden="true" />;
}

export function Card({ card, db, variant, backend, hpDelta, appeared }: Props) {
  const def = db.get(card.id);
  const name = def?.name ?? `#${card.id}`;
  const hpPct = card.maxHp ? Math.max(0, (card.hp / card.maxHp) * 100) : 0;
  const damaged = (hpDelta ?? 0) < 0;
  const healed = (hpDelta ?? 0) > 0;
  const [artDead, setArtDead] = useState(false);
  useEffect(() => { setArtDead(false); }, [card.id, backend]);

  const cls = [
    "card", `card-${variant}`, "has-art",
    hpDelta ? (damaged ? "flash-dmg" : healed ? "flash-heal" : "") : "",
    appeared ? "flash-new" : "",
  ].filter(Boolean).join(" ");

  return (
    <div className={cls} tabIndex={0}>
      {artDead
        ? <div className="card-name">{name}</div>
        : <CardArt id={card.id} backend={backend} alt={name} onDead={() => setArtDead(true)} />}
      {def?.ex && <span className="card-tag">ex</span>}

      {variant !== "hand" && card.maxHp > 0 && (
        <div className="hp">
          <div className="hp-bar">
            <div className="hp-fill" style={{
              width: `${hpPct}%`,
              background: hpPct > 50 ? "#4caf50" : hpPct > 25 ? "#e0a03d" : "#e2493a",
            }} />
          </div>
          <span className="hp-text">
            {card.hp}/{card.maxHp}
            {hpDelta ? <span className={damaged ? "delta-dmg" : "delta-heal"}> {hpDelta > 0 ? "+" : ""}{hpDelta}</span> : null}
          </span>
        </div>
      )}
      <EnergyPips energies={card.energies ?? []} />

      {def && (
        <div className="card-pop">
          <div className="pop-title">{def.name}</div>
          <div className="pop-meta">
            HP {def.hp} · retreat {def.retreat_cost}
            {def.weakness != null ? ` · weak ${energyType(def.weakness).letter}` : ""}
          </div>
          {def.attacks?.map((a) => (
            <div key={a.attack_id} className="pop-atk">
              <span className="pop-atk-cost">{a.energies.map((c) => energyType(c).letter).join("")}</span>{" "}
              <b>{a.name}</b> {a.damage ? `— ${a.damage}` : ""}
              {a.text ? <div className="pop-atk-text">{a.text}</div> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
