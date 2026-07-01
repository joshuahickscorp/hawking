/*
  StateTimeline.tsx — time-travel for the agent's STATE. Because RWKV-7's state is a constant-size
  serializable snapshot, scrubbing to a past step is instant (no re-prefill) and forking a new branch
  from any point is a memcpy. A row of step dots: click to scrub (scrub_to_event), "fork from here" to
  branch (fork_session). Both intents already exist; the backend snapshots state in plan 2.
*/
import { useState } from "react";
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Icon } from "./icons";

export function StateTimeline() {
  const tools = useStore((s) => s.tools);
  const sessionId = useStore((s) => s.sessionId);
  const [sel, setSel] = useState<number | null>(null);
  const steps = tools.slice(-14);
  if (steps.length === 0) return null;

  const at = sel == null ? steps.length - 1 : Math.min(sel, steps.length - 1);
  const scrub = (i: number) => {
    setSel(i);
    void sendIntent(intent.scrubToEvent(sessionId, steps[i].call_id));
  };
  const forkHere = () => void sendIntent(intent.forkSession(sessionId, steps[at].call_id));
  const live = () => setSel(null);

  return (
    <div className="statetl" role="group" aria-label="Agent state timeline">
      <span className="statetl__label" title="The agent's state is a serializable snapshot — scrubbing is instant.">state</span>
      <div className="statetl__dots">
        {steps.map((s, i) => (
          <button
            key={s.call_id + s.ts}
            className={[
              "statetl__dot",
              i <= at && "statetl__dot--past",
              i === at && "statetl__dot--at",
            ]
              .filter(Boolean)
              .join(" ")}
            title={s.message}
            aria-label={s.message}
            onClick={() => scrub(i)}
          />
        ))}
      </div>
      <span className="statetl__msg">{steps[at]?.message}</span>
      <div className="statetl__actions">
        {sel != null ? (
          <button className="statetl__btn" onClick={live} title="Return to the latest state">live</button>
        ) : null}
        <button className="statetl__btn statetl__fork" onClick={forkHere} title="Fork a new branch from this point (instant — the state is a snapshot)">
          <Icon name="fork" size={12} strokeWidth={1.6} />
          fork from here
        </button>
      </div>
    </div>
  );
}
