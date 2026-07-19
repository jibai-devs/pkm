import type { CardDb } from "./cardDb";
import type { LogEvent } from "./types";

// Log events use NUMERIC type codes (not the string names in requirements.md).
// This map is BEST-EFFORT, decoded from field signatures + frequencies in the
// sample replay — the engine (binary `cabt`) ships no public enum. Unknown or
// uncertain types fall back to a raw rendering, so the log never lies.
export const AREA_NAMES: Record<number, string> = {
  0: "deck",
  1: "hand",
  2: "bench",
  3: "discard",
  4: "active",
  5: "prize",
  8: "attached",
};

export type LogKind =
  | "phase"
  | "draw"
  | "play"
  | "move"
  | "switch"
  | "ability"
  | "evolve"
  | "attack"
  | "hp"
  | "mulligan"
  | "unknown";

export interface LogTypeDef {
  kind: LogKind;
  label: string;
  confident: boolean;
}

export const LOG_TYPES: Record<number, LogTypeDef> = {
  0: { kind: "phase", label: "Turn / phase marker", confident: false },
  1: { kind: "mulligan", label: "Has basic Pokemon?", confident: true },
  2: { kind: "phase", label: "Phase", confident: false },
  3: { kind: "phase", label: "Phase", confident: false },
  4: { kind: "draw", label: "Reveal / draw", confident: false },
  5: { kind: "draw", label: "Draw (hidden)", confident: false },
  6: { kind: "move", label: "Move card", confident: true },
  7: { kind: "move", label: "Move card (hidden)", confident: true },
  8: { kind: "switch", label: "Switch active/bench", confident: true },
  10: { kind: "play", label: "Play card", confident: false },
  11: { kind: "ability", label: "Ability / effect", confident: false },
  12: { kind: "evolve", label: "Evolve / attach", confident: false },
  15: { kind: "attack", label: "Attack", confident: true },
  16: { kind: "hp", label: "HP change", confident: true },
};

export function logKind(ev: LogEvent): LogKind {
  return LOG_TYPES[ev.type]?.kind ?? "unknown";
}

const area = (a?: number) => (a == null ? "?" : AREA_NAMES[a] ?? `area${a}`);

export interface FormattedLog {
  kind: LogKind;
  player?: number;
  text: string;
  confident: boolean;
}

// Render one log event as a readable line. `db` resolves card ids to names.
export function formatLog(ev: LogEvent, db: CardDb): FormattedLog {
  const def = LOG_TYPES[ev.type];
  const kind = def?.kind ?? "unknown";
  const p = ev.playerIndex;
  const pl = p == null ? "" : `P${p}`;
  const name = (id?: number) => (id == null ? "?" : db.name(id));

  let text: string;
  switch (ev.type) {
    case 1:
      text = `${pl} mulligan check — has basic Pokemon: ${ev.hasBasicPokemon}`;
      break;
    case 4:
      text = `${pl} revealed/drew ${name(ev.cardId)}`;
      break;
    case 5:
      text = `${pl} drew a card`;
      break;
    case 6:
      text = `${pl} moved ${name(ev.cardId)}: ${area(ev.fromArea)} → ${area(ev.toArea)}`;
      break;
    case 7:
      text = `${pl} moved a card: ${area(ev.fromArea)} → ${area(ev.toArea)}`;
      break;
    case 8:
      text = `${pl} switched ${name(ev.cardIdActive)} ⇄ ${name(ev.cardIdBench)} (active/bench)`;
      break;
    case 10:
      text = `${pl} played ${name(ev.cardId)}`;
      break;
    case 11:
      text = `${pl} ${name(ev.cardId)} used ability on ${name(ev.cardIdTarget)}`;
      break;
    case 12:
      text = `${pl} ${name(ev.cardId)} → ${name(ev.cardIdTarget)} (evolve/attach)`;
      break;
    case 15:
      text = `${pl} ${name(ev.cardId)} attacked (attack #${ev.attackId})`;
      break;
    case 16: {
      const v = ev.value ?? 0;
      const verb = v < 0 ? "took" : "healed";
      text = `${pl} ${name(ev.cardId)} ${verb} ${Math.abs(v)} ${v < 0 ? "damage" : "HP"}`;
      break;
    }
    case 0:
    case 2:
    case 3:
      text = `${pl} ${def?.label ?? "phase"}`;
      break;
    default:
      text = `${pl} [type ${ev.type}] ${JSON.stringify(ev)}`;
  }
  return { kind, player: p, text, confident: def?.confident ?? false };
}
