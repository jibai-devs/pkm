import type { CardDb } from "../data/cardDb";
import { energyType } from "../data/energy";
import type { CardInstance } from "../data/types";

interface Props {
  card: CardInstance;
  db: CardDb;
  variant: "active" | "bench" | "hand" | "mini";
  hpDelta?: number; // from step diff -> flash + show delta
  appeared?: boolean; // entered play this step -> flash
}

function EnergyPips({ energies }: { energies: number[] }) {
  if (!energies?.length) return null;
  return (
    <span className="pips">
      {energies.map((code, i) => {
        const t = energyType(code);
        return (
          <span
            key={i}
            className="pip"
            style={{ background: t.color, color: t.fg }}
            title={t.name}
          >
            {t.letter}
          </span>
        );
      })}
    </span>
  );
}

export function Card({ card, db, variant, hpDelta, appeared }: Props) {
  const def = db.get(card.id);
  const name = def?.name ?? `#${card.id}`;
  const hpPct = card.maxHp ? Math.max(0, (card.hp / card.maxHp) * 100) : 0;
  const damaged = (hpDelta ?? 0) < 0;
  const healed = (hpDelta ?? 0) > 0;

  const cls = [
    "card",
    `card-${variant}`,
    hpDelta ? (damaged ? "flash-dmg" : healed ? "flash-heal" : "") : "",
    appeared ? "flash-new" : "",
  ]
    .filter(Boolean)
    .join(" ");

  // Card art is served from public/cards/<id>.png (fetched by
  // replay/fetch_card_images.py). If an id has no image on disk the <img>
  // 404s -> onError hides just the image, leaving the name/hp/pips fallback.
  const imgSrc = `/cards/${card.id}.png`;

  return (
    <div className={cls} tabIndex={0}>
      <img
        className="card-img"
        src={imgSrc}
        alt={name}
        loading="lazy"
        onError={(e) => {
          e.currentTarget.style.display = "none";
        }}
      />
      <div className="card-name">{name}</div>
      {variant !== "hand" && card.maxHp > 0 && (
        <div className="hp">
          <div className="hp-bar">
            <div
              className="hp-fill"
              style={{
                width: `${hpPct}%`,
                background: hpPct > 50 ? "#4caf50" : hpPct > 25 ? "#e0a03d" : "#e2493a",
              }}
            />
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
          <img
            className="pop-img"
            src={imgSrc}
            alt=""
            onError={(e) => {
              e.currentTarget.style.display = "none";
            }}
          />
          <div className="pop-title">{def.name}</div>
          <div className="pop-meta">
            HP {def.hp} · retreat {def.retreat_cost}
            {def.weakness != null ? ` · weak ${energyType(def.weakness).letter}` : ""}
          </div>
          {def.attacks?.map((a) => (
            <div key={a.attack_id} className="pop-atk">
              <span className="pop-atk-cost">
                {a.energies.map((c) => energyType(c).letter).join("")}
              </span>{" "}
              <b>{a.name}</b> {a.damage ? `— ${a.damage}` : ""}
              {a.text ? <div className="pop-atk-text">{a.text}</div> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
