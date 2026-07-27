/*
  SteerBar.tsx: the slim state strip above the composer while a turn is in flight.

  It used to carry its OWN "Redirect this run" input plus a Steer button. That is now a duplicate of
  the composer, which steers on Enter whenever a run is live (catalog `steer`, toolbar binding moved
  from steer.redirect to the composer), so the second input is retired: one text box, one meaning.
  What stays is the honest run state plus the lifecycle verbs that have no other home here.
*/
import type { RunPhase } from "../../store";
import { ctlStyle } from "./parts";

export function SteerBar({
  phase,
  onPause,
  onResume,
  onCancel,
}: {
  phase: RunPhase;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
}) {
  const paused = phase === "paused";

  return (
    <div className="steerbar">
      {/* The live region is the run STATE, and only the state. It used to be the whole bar, so every
          phase change re-announced Resume / Pause / Cancel as status text. */}
      <span
        role="status"
        style={{ display: "flex", alignItems: "center", gap: "var(--ma-2)", flex: 1, minWidth: 0 }}
      >
        <span
          aria-hidden
          style={{
            flex: "0 0 auto",
            width: 14,
            textAlign: "center",
            fontSize: "var(--fs-label)",
            color: paused ? "var(--text-muted)" : "var(--accent)",
          }}
        >
          {paused ? "❙❙" : "●"}
        </span>
        <span
          style={{
            flex: "0 0 auto",
            fontWeight: 600,
            fontSize: "var(--fs-label)",
            color: "var(--text-dim)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
          }}
        >
          {PHASE_LABEL[phase]}
        </span>

        <span style={{ flex: 1, minWidth: 0, fontSize: "var(--fs-small)", color: "var(--text-dim)" }}>
          {STEER_HINT[phase]}
        </span>
      </span>

      {paused ? (
        <button onClick={onResume} style={ctlStyle(false)} title="resume run">
          Resume
        </button>
      ) : (
        <button onClick={onPause} style={ctlStyle(false)} title="pause run">
          Pause
        </button>
      )}
      <button onClick={onCancel} style={{ ...ctlStyle(false), color: "var(--red)" }} title="cancel run, the turn stops">
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

// Says where steering lives now, so the retired input is not missed.
const STEER_HINT: Record<RunPhase, string> = {
  idle: "",
  planning: "type below to steer the plan",
  executing: "type below to steer this run",
  paused: "type below to steer, then resume",
  awaiting: "type below to answer or steer",
  done: "",
  failed: "",
};
