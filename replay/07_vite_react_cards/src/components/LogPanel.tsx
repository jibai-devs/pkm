import type { CardDb } from "../data/cardDb";
import { formatLog } from "../data/events";
import type { MergedStep } from "../data/stepState";

export function LogPanel({ step, db }: { step: MergedStep; db: CardDb }) {
  const events = step.logs.map((ev) => formatLog(ev, db));
  return (
    <div className="panel log-panel">
      <div className="panel-title">Event Log · step {step.index + 1}</div>
      <div className="log-list">
        {events.length === 0 && <div className="log-empty">No events this step.</div>}
        {events.map((e, i) => (
          <div
            key={i}
            className={`log-row kind-${e.kind} ${e.confident ? "" : "log-guess"}`}
            title={e.confident ? "" : "best-effort decode of numeric log type"}
          >
            {e.text}
          </div>
        ))}
      </div>
      {step.select && (
        <div className="prompt">
          Prompt: type {step.select.type ?? "?"} / ctx {step.select.context ?? "?"} ·{" "}
          {step.select.option?.length ?? 0} option(s)
        </div>
      )}
    </div>
  );
}
