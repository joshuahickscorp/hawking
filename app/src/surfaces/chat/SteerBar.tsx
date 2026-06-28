/*
  chat/SteerBar.tsx: the persistent steer bar shown while a run is active (doctrine: see/steer/gate).
  The agent is interruptible and steerable, never fire-and-forget. A persistent steer input redirects
  the run mid-flight (Custom:redirect_run), and the transport verbs Pause/Resume/Cancel are the run FSM
  controls. The bar breathes with light while the run is live; the steer field is the calm escape hatch.

  Sends (handed up as callbacks the surface wires to sendIntent):
    redirect -> Custom:redirect_run{run_id, text}
    Pause    -> PauseRun{run_id}   Resume -> ResumeRun{run_id}   Cancel -> CancelRun{run_id}
*/
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
      pad="var(--ma-2) var(--ma-3)"
      style={{ display: "flex", alignItems: "center", gap: "var(--ma-3)", maxWidth: 700, margin: "0 auto var(--ma-3)" }}
    >
      {/* The run state, read as light + glyph, never as an invented hue. Paused is a glyph + neutral
          text (no orange); a live run breathes the steady light of the agent at work. */}
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
        placeholder="Steer the agent: actually, use X"
        className="t-body"
        style={{
          flex: 1,
          minWidth: 0,
          background: "transparent",
          border: "none",
          outline: "none",
          color: "var(--text-1)",
          font: "inherit",
          padding: "var(--ma-1) 0",
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
      {/* Cancel is destructive: the oxide pigment, glyph-paired by its plain label. */}
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
