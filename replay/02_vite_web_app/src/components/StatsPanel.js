import { getCardName } from '../utils/cardDb.js'

export function renderStatsPanel(container, replay, step, cardDb) {
  const viz = replay.steps[0]?.[0]?.visualize
  if (!viz) {
    container.innerHTML = '<div class="stats-panel"><h3>Stats</h3><p>No data</p></div>'
    return
  }

  const stats = computeStats(viz, step)

  container.innerHTML = `
    <div class="stats-panel">
      <h3>Match Statistics</h3>
      <table class="stats-table">
        <tr><th></th><th>P0</th><th>P1</th></tr>
        <tr><td>Damage dealt</td><td>${stats.damage[0]}</td><td>${stats.damage[1]}</td></tr>
        <tr><td>KOs</td><td>${stats.kos[0]}</td><td>${stats.kos[1]}</td></tr>
        <tr><td>Cards played</td><td>${stats.played[0]}</td><td>${stats.played[1]}</td></tr>
        <tr><td>Energy attached</td><td>${stats.attached[0]}</td><td>${stats.attached[1]}</td></tr>
        <tr><td>Prizes taken</td><td>${stats.prizesTaken[0]}</td><td>${stats.prizesTaken[1]}</td></tr>
      </table>
      <div class="turn-info">
        <span>Turn: ${stats.turn}</span>
        <span>First player: P${stats.firstPlayer}</span>
      </div>
    </div>
  `
}

function computeStats(viz, upToStep) {
  const stats = {
    damage: [0, 0],
    kos: [0, 0],
    played: [0, 0],
    attached: [0, 0],
    prizesTaken: [0, 0],
    turn: 0,
    firstPlayer: -1,
  }

  for (let i = 0; i <= upToStep && i < viz.length; i++) {
    const v = viz[i]
    const cur = v.current
    if (!cur) continue

    stats.turn = cur.turn ?? stats.turn
    if (cur.firstPlayer >= 0) stats.firstPlayer = cur.firstPlayer

    for (const log of v.logs ?? []) {
      const p = log.playerIndex
      switch (log.type) {
        case 'HpChange':
          if (p != null && log.value < 0) {
            stats.damage[p === 0 ? 1 : 0] += Math.abs(log.value)
          }
          break
        case 'Play':
          if (p != null) stats.played[p]++
          break
        case 'Attach':
          if (p != null) stats.attached[p]++
          break
        case 'MoveCard':
          if (p != null) {
            if (log.fromArea === 4 && log.toArea === 3) {
              stats.kos[p === 0 ? 1 : 0]++
            }
            if (log.fromArea === 5 && log.toArea === 1) {
              stats.prizesTaken[p]++
            }
          }
          break
      }
    }
  }

  return stats
}
