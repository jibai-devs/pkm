import { useEffect, useState } from "react";

// Best-effort dropdown over the ian_tools self-play batch. Reads the manifest
// written by generate_vite_replays.py (/ian/index.json); if it isn't there
// (viewing some other replay), the chooser simply hides itself. Switching
// navigates to ?replay=/ian/<file> and lets the app reload that replay.

interface Entry {
  file: string;
  title: string;
  outcome: string;
  steps: number;
}

export function ReplayChooser() {
  const [entries, setEntries] = useState<Entry[] | null>(null);

  useEffect(() => {
    fetch("/ian/index.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setEntries(Array.isArray(d) ? d : null))
      .catch(() => setEntries(null));
  }, []);

  if (!entries || entries.length === 0) return null;

  const current = new URLSearchParams(window.location.search).get("replay") ?? "";
  const selected =
    entries.find((e) => current.endsWith(e.file))?.file ?? entries[0].file;

  return (
    <label className="replay-chooser">
      <span>batch:</span>
      <select
        value={selected}
        onChange={(e) => {
          const url = new URL(window.location.href);
          url.searchParams.set("replay", `/ian/${e.target.value}`);
          url.searchParams.delete("step"); // start the new game from the top
          window.location.assign(url.toString());
        }}
      >
        {entries.map((e, i) => (
          <option key={e.file} value={e.file}>
            {`#${String(i + 1).padStart(2, "0")} · ${e.outcome} · ${e.steps} steps`}
          </option>
        ))}
      </select>
    </label>
  );
}
