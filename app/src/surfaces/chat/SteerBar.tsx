/*
  chat/SteerBar.tsx: the persistent steer bar shown while a run is active (doctrine C10: see/steer/gate).
  The agent is interruptible and steerable, never fire-and-forget. A persistent steer input redirects
  the run mid-flight (Custom:redirect_run), and the transport verbs Pause/Resume/Cancel are the run FSM
  controls. The bar wears the gold rim while the run breathes; the steer field is the calm escape hatch.

  Sends (handed up as callbacks the surface wires to sendIntent):
    redirect -> Custom:redirect_run{run_id, text}
    Pause    -> PauseRun{run_id}   Resume -> ResumeRun{run_id}   Cancel -> CancelRun{run_id}
*/
import { useState } from "react";
import type { RunPhase } from "../../store";
import { Panel } from "../../ui";
import { ctlStyle } from "./parts";

export function SteerBar({
  phase,
  onRedirect,
  onPause,
  onResume,
  onCancel,
}: {
  phase: RunPhase;
  onRedirect: (text: string) => void;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
}) {
  const [steer, setSteer] = useState("");
  const paused = phase === "paused";
  const breathing = phase === "executing" || phase === "planning" || phase === "awaiting";

  const fire = () => {
    const t = steer.trim();
    if (!t) return;
    onRedirect(t);
    setSteer("");
  };

  return (
    <Panel
      active={breathing}
      pad="var(--s2) var(--s3)"
      style={{ display: "flex", alignItems: "center", gap: "var(--s2)", maxWidth: 720, margin: "0 auto var(--s2)" }}
    >
      <span
        aria-hidden
        title={paused ? "run paused" : "run active"}
        style={{
          flex: "0 0 auto",
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: paused ? "var(--warning)" : "var(--radiation)",
          ...(breathing ? { animation: "radiation-breathe 2s ease-in-out infinite" } : null),
        }}
      />
      <span style={{ flex: "0 0 auto", fontSize: "var(--text-xs)", color: "var(--text-low)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {PHASE_LABEL[phase]}
      </span>

      <input
        value={steer}
        onChange={(e) => setSteer(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            fire();
          }
        }}
        placeholder="Steer the agent: actually, use X"
        style={{
          flex: 1,
          minWidth: 0,
          background: "transparent",
          border: "none",
          outline: "none",
          color: "var(--text-hi)",
          font: "inherit",
          fontSize: "var(--text-sm)",
          padding: "2px 0",
        }}
      />

      <button onClick={fire} disabled={!steer.trim()} style={ctlStyle(!!steer.trim())} title="redirect this run">
        Steer
      </button>
      {paused ? (
        <button onClick={onResume} style={ctlStyle(false)} title="resume run">
          Resume
        </button>
      ) : (
        <button onClick={onPause} style={ctlStyle(false)} title="pause run">
          Pause
        </button>
      )}
      <button onClick={onCancel} style={{ ...ctlStyle(false), color: "var(--danger)" }} title="cancel run">
        Cancel
      </button>
    </Panel>
  );
}

const PHASE_LABEL: Record<RunPhase, string> = {
  idle: "Idle",
  planning: "Planning",
  executing: "Running",
  paused: "Paused",
  awaiting: "Awaiting",
  done: "Done",
  failed: "Failed",
};
