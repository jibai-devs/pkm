export async function loadReplay(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load replay: ${res.status}`)
  const data = await res.json()

  const vizEntries = data.steps[0]?.[0]?.visualize
  const totalSteps = vizEntries ? vizEntries.length : data.steps.length

  return {
    ...data,
    totalSteps,
  }
}
