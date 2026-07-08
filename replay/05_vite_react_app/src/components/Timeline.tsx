import type { Playback } from "../state/usePlayback";

const SPEEDS = [0.5, 1, 2, 4, 8];

export function Timeline({ pb, turn }: { pb: Playback; turn: number | null }) {
  const pct = pb.count > 1 ? (pb.index / (pb.count - 1)) * 100 : 0;
  return (
    <div className="timeline">
      <div className="controls">
        <button onClick={() => pb.setIndex(0)} title="First">⏮</button>
        <button onClick={() => pb.step(-1)} title="Back">◀</button>
        <button className="play" onClick={pb.togglePlay} title="Play/Pause">
          {pb.playing ? "⏸" : "▶"}
        </button>
        <button onClick={() => pb.step(1)} title="Forward">▶</button>
        <button onClick={() => pb.setIndex(pb.count - 1)} title="Last">⏭</button>

        <span className="step-label">
          Step {pb.index + 1}/{pb.count}
          {turn != null ? ` · Turn ${turn}` : ""}
        </span>

        <span className="speeds">
          Speed:
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={pb.speed === s ? "sp on" : "sp"}
              onClick={() => pb.setSpeed(s)}
            >
              {s}×
            </button>
          ))}
        </span>
      </div>

      <input
        className="scrubber"
        type="range"
        min={0}
        max={100}
        step={100 / Math.max(1, pb.count - 1)}
        value={pct}
        onChange={(e) => pb.toPct(Number(e.target.value))}
      />
    </div>
  );
}
