import { useEffect, useMemo, useState } from "react";
import type { LivePrompt } from "../live/types";

// The interactive counterpart of the TUI's PromptPane (pkm/tui/widgets.py):
// toggle options up to maxCount, submit when the count is in [min, max]. Attacks
// and end-turn have no undo, so a confirm step guards them.
interface Props {
  prompt: LivePrompt;
  disabled: boolean;
  onSubmit: (picks: number[]) => void;
}

export function OptionsPane({ prompt, disabled, onSubmit }: Props) {
  const [picks, setPicks] = useState<number[]>([]);
  const [confirming, setConfirming] = useState(false);

  // A fresh prompt clears any prior selection (the engine has no rollback, so
  // there is no cross-prompt state to keep).
  useEffect(() => {
    setPicks([]);
    setConfirming(false);
  }, [prompt]);

  const { minCount, maxCount, options } = prompt;
  const submittable = picks.length >= minCount && picks.length <= maxCount;
  const needsConfirm = useMemo(
    () => picks.some((i) => options.find((o) => o.index === i)?.irreversible),
    [picks, options],
  );

  const toggle = (index: number) => {
    if (disabled) return;
    setPicks((prev) => {
      if (prev.includes(index)) return prev.filter((i) => i !== index);
      if (prev.length >= maxCount) {
        // At the cap: for a single-pick prompt, replace; otherwise ignore.
        return maxCount === 1 ? [index] : prev;
      }
      return [...prev, index];
    });
  };

  const doSubmit = () => {
    if (!submittable || disabled) return;
    if (needsConfirm && !confirming) {
      setConfirming(true);
      return;
    }
    onSubmit([...picks].sort((a, b) => a - b));
  };

  const hint = !submittable
    ? picks.length < minCount
      ? `pick ${minCount - picks.length} more`
      : `pick at most ${maxCount}`
    : minCount === maxCount
      ? "ready"
      : `${picks.length} selected`;
  const span =
    minCount === maxCount ? `choose ${minCount}` : `choose ${minCount}–${maxCount}`;

  return (
    <div className="options-pane">
      <div className="options-header">
        <span className="options-span">{span}</span>
        <span className="options-hint">{hint}</span>
      </div>
      <div className="options-list">
        {options.map((o) => {
          const selected = picks.includes(o.index);
          return (
            <button
              key={o.index}
              className={`option-btn${selected ? " selected" : ""}${o.irreversible ? " irreversible" : ""}`}
              onClick={() => toggle(o.index)}
              disabled={disabled}
            >
              <span className="option-check">{selected ? "☑" : "☐"}</span>
              <span className="option-key">{o.index + 1}</span>
              <span className="option-label">{o.label}</span>
              {o.irreversible && <span className="option-flag">no undo</span>}
            </button>
          );
        })}
      </div>
      {confirming ? (
        <div className="options-confirm">
          <span>This can't be undone. Confirm?</span>
          <button className="confirm-yes" onClick={doSubmit} disabled={disabled}>
            Confirm
          </button>
          <button className="confirm-no" onClick={() => setConfirming(false)}>
            Cancel
          </button>
        </div>
      ) : (
        <button
          className="submit-btn"
          onClick={doSubmit}
          disabled={!submittable || disabled}
        >
          Submit
        </button>
      )}
    </div>
  );
}
