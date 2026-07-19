import type { StepDiff } from "../data/diff";

export function DiffPanel({ diff }: { diff: StepDiff }) {
  const hp = [...diff.changedHp.entries()];
  const nothing =
    hp.length === 0 && diff.appeared.size === 0 && diff.disappeared.size === 0;
  return (
    <div className="panel diff-panel">
      <div className="panel-title">Diff (this step)</div>
      <div className="diff-list">
        {nothing && <div className="log-empty">No board changes.</div>}
        {hp.map(([serial, delta]) => (
          <div key={`hp${serial}`} className={delta < 0 ? "diff-dmg" : "diff-heal"}>
            HP {delta > 0 ? "+" : ""}{delta} on serial #{serial}
          </div>
        ))}
        {diff.appeared.size > 0 && (
          <div className="diff-new">+{diff.appeared.size} card(s) entered play</div>
        )}
        {diff.disappeared.size > 0 && (
          <div className="diff-gone">−{diff.disappeared.size} card(s) left play (KO/retreat)</div>
        )}
      </div>
      <div className="diff-note">serials are per-game instance ids · hover a card for its name</div>
    </div>
  );
}
