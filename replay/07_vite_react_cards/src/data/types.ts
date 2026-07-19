// Types for the Pokemon TCG replay format (Kaggle Simulation).
// Verified against the real replay.json / cards.json in ../../.

// ---- replay.json: in-play card instance (DYNAMIC state only) ----
// NOTE: these objects do NOT contain the card name or any static metadata.
// Resolve `id` against cards.json (CardDb) for name/attacks/type/etc.
export interface CardInstance {
  id: number; // card_id -> look up in cards.json
  serial: number; // unique instance id within this game (stable across steps)
  playerIndex: 0 | 1;
  hp: number;
  maxHp: number;
  energies: number[]; // attached energy as energy-type codes (e.g. [0,5,5])
  energyCards?: CardInstance[];
  tools?: CardInstance[];
  preEvolution?: unknown;
  appearThisTurn?: boolean;
}

export interface PlayerState {
  active: (CardInstance | null)[];
  bench: (CardInstance | null)[];
  benchMax: number;
  deckCount: number;
  handCount: number;
  hand: CardInstance[]; // only populated in this player's OWN POV entry
  discard: CardInstance[];
  prize: CardInstance[];
  poisoned: boolean;
  burned: boolean;
  asleep: boolean;
  paralyzed: boolean;
  confused: boolean;
}

export interface CurrentState {
  turn: number;
  turnActionCount: number;
  yourIndex: 0 | 1;
  firstPlayer: number;
  result: number; // -1 in progress, 0 draw, 1 p0 wins (per requirements.md)
  supporterPlayed: boolean;
  stadiumPlayed: boolean;
  energyAttached: boolean;
  retreated: boolean;
  stadium: CardInstance[];
  players: [PlayerState, PlayerState];
}

export interface LogEvent {
  type: number; // numeric code, see events.ts LOG_TYPES
  playerIndex?: number;
  cardId?: number;
  serial?: number;
  fromArea?: number;
  toArea?: number;
  value?: number;
  putDamageCounter?: boolean;
  attackId?: number;
  hasBasicPokemon?: boolean;
  cardIdActive?: number;
  serialActive?: number;
  cardIdBench?: number;
  serialBench?: number;
  cardIdTarget?: number;
  serialTarget?: number;
  [k: string]: unknown;
}

export interface SelectPrompt {
  type?: number;
  context?: number;
  option?: unknown[];
}

export interface Observation {
  step: number;
  remainingOverageTime: number;
  select: SelectPrompt | null;
  logs: LogEvent[] | null;
  current: CurrentState | null;
}

export interface StepEntry {
  action: unknown;
  reward: number;
  status: "ACTIVE" | "INACTIVE" | "DONE" | string;
  info: unknown;
  observation: Observation;
}

export type Step = [StepEntry, StepEntry];

export interface Replay {
  id?: string;
  name?: string;
  title?: string;
  configuration?: { decks?: [number[], number[]] };
  steps: Step[];
  rewards?: number[];
  statuses?: unknown[];
}

// ---- cards.json: STATIC card dictionary ----
export interface CardAttack {
  attack_id: number;
  name: string;
  text: string;
  damage: number;
  energies: number[];
}

export interface CardDef {
  card_id: number;
  name: string;
  card_type: number; // 0 = pokemon, 5 = energy, ...
  energy_type: number;
  hp: number;
  basic: boolean;
  stage1: boolean;
  stage2: boolean;
  ex: boolean;
  mega_ex?: boolean;
  tera?: boolean;
  ace_spec?: boolean;
  evolves_from: string | null;
  weakness: number | null;
  resistance: number | null;
  retreat_cost: number;
  attacks: CardAttack[];
  skills?: unknown[];
}

export interface CardsFile {
  cards: CardDef[];
}
