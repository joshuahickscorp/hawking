import { useState } from "react";
import type { RunPhase } from "../../store";
import { Volume } from "../../ui";
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
    <Volume
      raised
      alive={breathing}
      className="steerbar"
    >
      <span
        aria-hidden
        title={paused ? "run paused" : "run active"}
        className={breathing ? "alive" : undefined}
        style={{
          flex: "0 0 auto",
          width: 14,
          textAlign: "center",
          fontSize: "11px",
          borderRadius: "var(--radius-pill)",
          color: paused ? "var(--text-2)" : "var(--light)",
        }}
      >
        {paused ? "❙❙" : "●"}
      </span>
      <span
        style={{
          flex: "0 0 auto",
          fontWeight: 500,
          fontSize: "12px",
          color: "var(--mute)",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
        }}
      >
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
        placeholder="Redirect this run"
        className="t-body"
        style={{ flex: 1, padding: "var(--ma-1) 0" }}
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
      <button onClick={onCancel} style={{ ...ctlStyle(false), color: "var(--bad)" }} title="cancel run">
        Cancel
      </button>
    </Volume>
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
