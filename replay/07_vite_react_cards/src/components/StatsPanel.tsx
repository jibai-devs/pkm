import type { StepStats } from "../data/stats";

function Bar({ label, a, b }: { label: string; a: number; b: number }) {
  return (
    <tr>
      <td className="stat-label">{label}</td>
      <td className="stat-p0">{a}</td>
      <td className="stat-p1">{b}</td>
    </tr>
  );
}

export function StatsPanel({ stats }: { stats: StepStats }) {
  const [p0, p1] = stats;
  return (
    <div className="panel stats-panel">
      <div className="panel-title">Match Stats</div>
      <table className="stats">
        <thead>
          <tr>
            <th></th>
            <th className="stat-p0">P0</th>
            <th className="stat-p1">P1</th>
          </tr>
        </thead>
        <tbody>
          <Bar label="Prizes taken" a={p0.prizesTaken} b={p1.prizesTaken} />
          <Bar label="Damage dealt" a={p0.damage} b={p1.damage} />
          <Bar label="Attacks" a={p0.attacks} b={p1.attacks} />
          <Bar label="Cards played" a={p0.cardsPlayed} b={p1.cardsPlayed} />
        </tbody>
      </table>
    </div>
  );
}
