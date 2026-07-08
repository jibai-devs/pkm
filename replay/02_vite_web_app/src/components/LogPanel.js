import { formatEvent, getEventColor } from '../utils/events.js'

export function renderLogPanel(container, vizEntries, cardDb) {
  const logs = vizEntries?.logs ?? []

  if (logs.length === 0) {
    container.innerHTML = '<div class="log-panel"><h3>Event Log</h3><p class="log-empty">No events this step</p></div>'
    return
  }

  const lines = logs.map(log => {
    const color = getEventColor(log)
    const text = formatEvent(log, cardDb)
    return `<div class="log-entry" style="border-left: 3px solid ${color}; padding-left: 8px; margin: 4px 0;">${text}</div>`
  }).join('')

  container.innerHTML = `
    <div class="log-panel">
      <h3>Event Log (${logs.length})</h3>
      <div class="log-entries">${lines}</div>
    </div>
  `
}
