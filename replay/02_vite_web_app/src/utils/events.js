import { getCardName, ENERGY_LETTERS, CARD_TYPE_NAMES } from './cardDb.js'

const AREA_NAMES = {
  0: 'deck', 1: 'hand', 2: 'bench', 3: 'discard',
  4: 'active', 5: 'prize', 8: 'attached', 9: 'lost zone',
}

export function formatEvent(log, cardDb) {
  const type = log.type
  const p = log.playerIndex != null ? `P${log.playerIndex}` : ''

  switch (type) {
    case 'Draw':
      return `${p} drew ${getCardName(cardDb, log.cardId)}`
    case 'Play':
      return `${p} played ${getCardName(cardDb, log.cardId)}`
    case 'Attach':
      return `${p} attached energy to ${getCardName(cardDb, log.cardId)}`
    case 'Attack': {
      const atk = cardDb.attacksById?.get(log.attackId)
      const atkName = atk?.name ?? `Attack #${log.attackId}`
      return `${p}'s ${getCardName(cardDb, log.cardId)} used ${atkName}`
    }
    case 'HpChange': {
      const dmg = log.value
      const name = getCardName(cardDb, log.cardId)
      if (dmg < 0) {
        const ko = log.putDamageCounter ? ' (KO!)' : ''
        return `${p}'s ${name} took ${Math.abs(dmg)} damage${ko}`
      }
      return `${p}'s ${name} healed ${dmg} HP`
    }
    case 'MoveCard': {
      const name = getCardName(cardDb, log.cardId)
      const from = AREA_NAMES[log.fromArea] ?? `zone ${log.fromArea}`
      const to = AREA_NAMES[log.toArea] ?? `zone ${log.toArea}`
      return `${p}'s ${name} moved from ${from} to ${to}`
    }
    case 'Ability':
      return `${p} activated an ability`
    case 'HasBasicPokemon':
      return `${p} ${log.hasBasicPokemon ? 'has' : 'does not have'} a Basic Pokémon`
    case 'TurnEnd':
      return `${p}'s turn ended`
    case 'TurnStart':
      return `${p}'s turn started`
    case 'End':
      return 'Game ended'
    default:
      return `[${type}] ${JSON.stringify(log)}`
  }
}

export function getEventColor(log) {
  switch (log.type) {
    case 'Draw': return '#3b82f6'
    case 'Play': return '#eab308'
    case 'Attack': return '#ef4444'
    case 'HpChange': return log.value < 0 ? '#ef4444' : '#22c55e'
    case 'Ability': return '#22c55e'
    case 'MoveCard': return '#a855f7'
    case 'TurnEnd':
    case 'TurnStart': return '#6b7280'
    default: return '#9ca3af'
  }
}
