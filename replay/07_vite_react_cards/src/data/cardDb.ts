import type { CardDef, CardsFile } from "./types";

// Static card dictionary from cards.json, keyed by card_id.
// Required even to show a card's NAME (replay card objects carry only dynamic state).
export class CardDb {
  private byId: Map<number, CardDef>;

  constructor(defs: CardDef[]) {
    this.byId = new Map(defs.map((c) => [c.card_id, c]));
  }

  get(id: number): CardDef | undefined {
    return this.byId.get(id);
  }

  name(id: number): string {
    return this.byId.get(id)?.name ?? `#${id}`;
  }
}

export async function loadCardDb(url = "/cards.json"): Promise<CardDb> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  const data: CardsFile = await res.json();
  return new CardDb(data.cards);
}
