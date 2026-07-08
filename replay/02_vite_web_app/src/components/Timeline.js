export function renderTimeline(container, playback) {
  container.innerHTML = `
    <div class="timeline">
      <div class="timeline-controls">
        <button id="btn-start" title="Jump to start">⏮</button>
        <button id="btn-back" title="Step backward">◀</button>
        <button id="btn-play" title="Play/Pause">▶</button>
        <button id="btn-forward" title="Step forward">▶</button>
        <button id="btn-end" title="Jump to end">⏭</button>
        <button id="btn-speed" title="Cycle speed">${playback.speed}x</button>
      </div>
      <input type="range" id="timeline-slider" min="0" max="${playback.totalSteps - 1}" value="${playback.step}" />
      <span class="step-label">Step ${playback.step} / ${playback.totalSteps - 1}</span>
    </div>
  `

  container.querySelector('#btn-start').onclick = () => playback.jumpToStart()
  container.querySelector('#btn-back').onclick = () => playback.backward()
  container.querySelector('#btn-forward').onclick = () => playback.forward()
  container.querySelector('#btn-end').onclick = () => playback.jumpToEnd()
  container.querySelector('#btn-speed').onclick = () => playback.cycleSpeed()

  const playBtn = container.querySelector('#btn-play')
  playBtn.textContent = playback.playing ? '⏸' : '▶'
  playBtn.onclick = () => playback.togglePlay()

  const slider = container.querySelector('#timeline-slider')
  slider.value = playback.step
  slider.oninput = (e) => playback.setStep(Number(e.target.value))

  container.querySelector('.step-label').textContent =
    `Step ${playback.step} / ${playback.totalSteps - 1}`
}
