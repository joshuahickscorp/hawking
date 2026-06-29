import { useState } from "react";
import type { RunPhase } from "../../store";
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

  const fire = () => {
    const t = steer.trim();
    if (!t) return;
    onRedirect(t);
    setSteer("");
  };

  return (
    <div className="steerbar">
      <span
        aria-hidden
        title={paused ? "run paused" : "run active"}
        style={{
          flex: "0 0 auto",
          width: 14,
          textAlign: "center",
          fontSize: "11px",
          color: paused ? "var(--text-muted)" : "var(--accent)",
        }}
      >
        {paused ? "❙❙" : "●"}
      </span>
      <span
        style={{
          flex: "0 0 auto",
          fontWeight: 600,
          fontSize: "11px",
          color: "var(--text-dim)",
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
        className="steerbar__input"
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
      <button onClick={onCancel} style={{ ...ctlStyle(false), color: "var(--red)" }} title="cancel run">
        Cancel
      </button>
    </div>
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
