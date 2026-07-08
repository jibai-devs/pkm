import './style.css'
import { loadReplay } from './replay.js'
import { loadCardDb } from './utils/cardDb.js'
import { createPlayback } from './playback.js'
import { renderTimeline } from './components/Timeline.js'
import { renderLogPanel } from './components/LogPanel.js'
import { renderPlayerView } from './components/PlayerView.js'
import { renderStatsPanel } from './components/StatsPanel.js'

async function init() {
  const [replay, cardDb] = await Promise.all([
    loadReplay('/replay.json'),
    loadCardDb('/cards.json'),
  ])

  const playback = createPlayback(replay.totalSteps)
  const viz = replay.steps[0][0].visualize

  const app = document.getElementById('app')
  app.innerHTML = `
    <header id="header"></header>
    <div id="controls"></div>
    <div id="board">
      <div id="player0" class="player"></div>
      <div id="player1" class="player"></div>
    </div>
    <div id="bottom">
      <div id="log"></div>
      <div id="stats"></div>
    </div>
  `

  function render() {
    const step = playback.step
    const vizEntry = viz[step]

    renderTimeline(document.getElementById('controls'), playback)
    renderPlayerView(document.getElementById('player0'), vizEntry, 0, cardDb, step)
    renderPlayerView(document.getElementById('player1'), vizEntry, 1, cardDb, step)
    renderLogPanel(document.getElementById('log'), vizEntry, cardDb)
    renderStatsPanel(document.getElementById('stats'), replay, step, cardDb)

    const turn = vizEntry?.current?.turn ?? '?'
    document.getElementById('header').innerHTML = `
      <h1>Pokemon TCG Replay</h1>
      <span class="step-info">Step ${step} / ${replay.totalSteps - 1} &mdash; Turn ${turn}</span>
    `
  }

  playback.onChange(render)

  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return
    switch (e.key) {
      case ' ':
        e.preventDefault()
        playback.togglePlay()
        break
      case 'ArrowLeft':
        e.preventDefault()
        playback.backward()
        break
      case 'ArrowRight':
        e.preventDefault()
        playback.forward()
        break
      case 'Home':
        e.preventDefault()
        playback.jumpToStart()
        break
      case 'End':
        e.preventDefault()
        playback.jumpToEnd()
        break
      case 'ArrowUp':
        e.preventDefault()
        playback.cycleSpeed()
        break
    }
  })

  render()
}

init()
