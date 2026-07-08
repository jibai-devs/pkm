export function createPlayback(totalSteps) {
  const listeners = new Set()
  let step = 0
  let playing = false
  let speed = 1
  let timer = null

  const SPEEDS = [0.25, 0.5, 1, 2, 4]
  const BASE_INTERVAL = 1000

  function notify() {
    for (const fn of listeners) fn()
  }

  function clamp(v) {
    return Math.max(0, Math.min(totalSteps - 1, v))
  }

  function tick() {
    if (!playing) return
    const next = step + 1
    if (next >= totalSteps) {
      playing = false
      clearInterval(timer)
      timer = null
      notify()
      return
    }
    step = next
    notify()
  }

  function startTimer() {
    if (timer) clearInterval(timer)
    timer = setInterval(tick, BASE_INTERVAL / speed)
  }

  return {
    get step() { return step },
    get playing() { return playing },
    get speed() { return speed },
    get totalSteps() { return totalSteps },

    setStep(s) {
      step = clamp(s)
      notify()
    },

    forward() {
      step = clamp(step + 1)
      notify()
    },

    backward() {
      step = clamp(step - 1)
      notify()
    },

    jumpToStart() {
      step = 0
      notify()
    },

    jumpToEnd() {
      step = totalSteps - 1
      notify()
    },

    togglePlay() {
      playing = !playing
      if (playing) {
        if (step >= totalSteps - 1) step = 0
        startTimer()
      } else {
        if (timer) clearInterval(timer)
        timer = null
      }
      notify()
    },

    setSpeed(s) {
      speed = s
      if (playing) startTimer()
      notify()
    },

    cycleSpeed() {
      const idx = SPEEDS.indexOf(speed)
      speed = SPEEDS[(idx + 1) % SPEEDS.length]
      if (playing) startTimer()
      notify()
    },

    onChange(fn) {
      listeners.add(fn)
      return () => listeners.delete(fn)
    },

    destroy() {
      if (timer) clearInterval(timer)
      listeners.clear()
    },
  }
}
