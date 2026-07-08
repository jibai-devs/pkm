export async function loadCardDb(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load card database: ${res.status}`)
  const data = await res.json()

  const byId = new Map()
  for (const card of data.cards) {
    byId.set(card.card_id, card)
  }

  const attacksById = new Map()
  if (Array.isArray(data.attacks)) {
    for (const atk of data.attacks) {
      attacksById.set(atk.attack_id, atk)
    }
  }

  return { byId, attacksById, raw: data }
}

export function getCardName(db, cardId) {
  const card = db.byId.get(cardId)
  return card?.name ?? `Card #${cardId}`
}

export const ENERGY_SYMBOLS = {
  0: '{C}', 1: '{G}', 2: '{R}', 3: '{W}', 4: '{L}',
  5: '{P}', 6: '{D}', 7: '{M}', 8: '{F}', 9: '{Y}', 10: '{N}',
}

export const ENERGY_COLORS = {
  0: '#a8a878', 1: '#78c850', 2: '#f08030', 3: '#6890f0',
  4: '#f8d030', 5: '#f85888', 6: '#705848', 7: '#b8b8d0',
  8: '#c03028', 9: '#ee99ac', 10: '#a890f0',
}

export const ENERGY_LETTERS = {
  0: 'C', 1: 'G', 2: 'R', 3: 'W', 4: 'L',
  5: 'P', 6: 'D', 7: 'M', 8: 'F', 9: 'Y', 10: 'N',
}

export const CARD_TYPE_NAMES = {
  0: 'Pokémon', 1: 'Trainer', 2: 'Supporter', 3: 'Stadium',
  4: 'Tool', 5: 'Energy', 6: 'Special Energy',
}
