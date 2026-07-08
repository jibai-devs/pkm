import { ENERGY_COLORS, ENERGY_LETTERS } from '../utils/cardDb.js'

export function renderPlayerView(container, vizEntry, playerIndex, cardDb, step) {
  const current = vizEntry?.current
  if (!current) {
    container.innerHTML = `<div class="player-view"><h3>Player ${playerIndex}</h3><p>No data</p></div>`
    return
  }

  const player = current.players?.[playerIndex]
  if (!player) {
    container.innerHTML = `<div class="player-view"><h3>Player ${playerIndex}</h3><p>No player data</p></div>`
    return
  }

  const isWinner = current.result === (playerIndex === 0 ? 1 : -1)
  const isLoser = current.result === (playerIndex === 0 ? -1 : 1)
  const resultBadge = isWinner ? ' 🏆 WINNER' : isLoser ? ' ❌ LOST' : ''

  const activeHtml = player.active.map(c => renderCard(c, true)).join('')
  const benchHtml = player.bench.map(c => renderCard(c, false)).join('')
  const handHtml = player.hand?.map(c => renderCardMini(c, true)).join('') ?? ''
  const discardHtml = player.discard.map(c => renderCardMini(c, true)).join('')
  const prizeCount = player.prize.length

  const statusIcons = []
  if (player.poisoned) statusIcons.push('☠️')
  if (player.burned) statusIcons.push('🔥')
  if (player.asleep) statusIcons.push('💤')
  if (player.paralyzed) statusIcons.push('⚡')
  if (player.confused) statusIcons.push('❓')
  const statusHtml = statusIcons.length ? `<div class="status-icons">${statusIcons.join(' ')}</div>` : ''

  container.innerHTML = `
    <div class="player-view ${playerIndex === 0 ? 'player-0' : 'player-1'}">
      <h3>Player ${playerIndex}${resultBadge}</h3>
      ${statusHtml}
      <div class="zone active-zone">
        <span class="zone-label">Active</span>
        <div class="zone-cards">${activeHtml || '<span class="empty">—</span>'}</div>
      </div>
      <div class="zone bench-zone">
        <span class="zone-label">Bench (${player.bench.length}/${player.benchMax})</span>
        <div class="zone-cards">${benchHtml || '<span class="empty">—</span>'}</div>
      </div>
      <div class="zone-info">
        <span>Hand: ${player.handCount}</span>
        <span>Deck: ${player.deckCount}</span>
        <span>Discard: ${player.discard.length}</span>
        <span>Prize: ${prizeCount}</span>
      </div>
      ${handHtml ? `<div class="zone hand-zone"><span class="zone-label">Hand</span><div class="zone-cards">${handHtml}</div></div>` : ''}
      ${discardHtml ? `<div class="zone discard-zone"><span class="zone-label">Discard</span><div class="zone-cards">${discardHtml}</div></div>` : ''}
    </div>
  `
}

function renderCard(card, isActive) {
  const hpPct = card.maxHp > 0 ? (card.hp / card.maxHp) * 100 : 0
  const hpColor = hpPct > 50 ? '#22c55e' : hpPct > 20 ? '#eab308' : '#ef4444'
  const energies = (card.energies || []).map(e =>
    `<span class="energy-pip" style="background:${ENERGY_COLORS[e] ?? '#888'}" title="${ENERGY_LETTERS[e] ?? e}">${ENERGY_LETTERS[e] ?? e}</span>`
  ).join('')

  return `
    <div class="card card-active" title="ID: ${card.id} | Serial: ${card.serial}">
      <div class="card-name">${card.name}</div>
      <div class="card-hp-bar">
        <div class="card-hp-fill" style="width:${hpPct}%; background:${hpColor}"></div>
      </div>
      <div class="card-hp-text">${card.hp}/${card.maxHp} HP</div>
      <div class="card-energies">${energies || '<span class="energy-none">No energy</span>'}</div>
    </div>
  `
}

function renderCardMini(card, showName) {
  return `<span class="card-mini" title="ID: ${card.id}">${showName ? card.name : `#${card.id}`}</span>`
}
