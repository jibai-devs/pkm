// Shows the middleman's routing log for the current decision: which sub-agent
// (first_turn / neural / random) actually made this pick, plus any side notes
// it logged (went first/second, prize deductions). Data comes from the replay's
// optional `subAgentLog`, aligned index-for-index with steps[].

const WHO = /decision made by:\s*(\S+)/;

export function SubAgentPanel({
  log,
  index,
}: {
  log: string[] | undefined;
  index: number;
}) {
  const lines = log ?? [];
  const whoLine = lines.find((l) => WHO.test(l));
  const who = whoLine?.match(WHO)?.[1] ?? null;
  const rest = lines.filter((l) => l !== whoLine);

  return (
    <div className="panel subagent-panel">
      <div className="panel-title">Sub-agent · step {index + 1}</div>
      {who ? (
        <div className={`subagent-who who-${who}`}>{who}</div>
      ) : (
        <div className="subagent-none">
          {lines.length ? "(no routing line)" : "(no sub-agent log for this step)"}
        </div>
      )}
      {rest.length > 0 && (
        <div className="subagent-notes">
          {rest.map((l, i) => (
            <div key={i} className="subagent-note">
              {l}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
