/*
  chat/structure.tsx: the inline structure that lives INSIDE the assistant stream.
  Three calm, material devices, re-housed from the Cline/OpenCode plan-act + per-step chat UX into
  the v3 doctrine (grayscale concrete volumes, Geist Mono telemetry voice, light as the only accent,
  glyph+label markers, no spinners, no churn):

    PlanCard  <- projection_patch:plan       : ordered steps with per-step status + approve/edit/reorder.
    ToolChip  <- tool_progress{call_id,msg}   : one calm chip per tool call, no per-token churn.
    DiffChipRow <- projection_patch:diff_chip : a produced diff -> a chip that opens the hunk review.
    InlineGate <- security_gate{gate,message} : the lit approval the doctrine wants rendered inline.

  Steer/approve verbs go out as the registered intents: approve_plan / edit_plan_step / reorder_plan
  (Custom), accept_diff / reject_diff (enum), approve_gate (Custom). Nothing here owns transport;
  every action is an Intent handed up via the callbacks the surface wires to sendIntent.
*/
import { useState, type CSSProperties } from "react";
import type { ToolEvent } from "../../store";
import { Gate, Volume } from "../../ui";
import {
  blockLabel,
  chip,
  ctlStyle,
  STEP_MARK,
  type DiffChip,
  type PlanPatch,
  type PlanStep,
} from "./parts";

// ---- PlanCard: the plan rendered as ordered steps with status (the Cline plan-act card, re-skinned). ----
export function PlanCard({
  plan,
  onApprove,
  onEditStep,
  onReorder,
}: {
  plan: PlanPatch;
  onApprove: () => void;
  onEditStep: (step: PlanStep, title: string) => void;
  onReorder: (from: number, to: number) => void;
}) {
  const steps = plan.steps ?? [];
  if (steps.length === 0) return null;
  const awaiting = plan.awaiting_approval === true;
  const done = steps.filter((s) => s.status === "done").length;

  return (
    // While the plan waits for you, the whole volume breathes (the agent needs you, read as light).
    <Volume alive={awaiting} pad="var(--ma-4)">
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-3)", marginBottom: "var(--ma-3)" }}>
        <span style={blockLabel}>Plan</span>
        <span className="t-micro">
          {done}/{steps.length}
        </span>
        {awaiting ? (
          <button onClick={onApprove} style={{ ...ctlStyle(true), marginLeft: "auto" }} title="approve plan (Cmd+Enter)">
            Approve
          </button>
        ) : null}
      </div>
      <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: "var(--ma-1)" }}>
        {steps.map((s, i) => (
          <PlanStepRow
            key={s.id ?? i}
            step={s}
            index={i}
            last={i === steps.length - 1}
            onEdit={(title) => onEditStep(s, title)}
            onUp={i > 0 ? () => onReorder(i, i - 1) : undefined}
            onDown={i < steps.length - 1 ? () => onReorder(i, i + 1) : undefined}
          />
        ))}
      </ol>
    </Volume>
  );
}

function PlanStepRow({
  step,
  index,
  last,
  onEdit,
  onUp,
  onDown,
}: {
  step: PlanStep;
  index: number;
  last: boolean;
  onEdit: (title: string) => void;
  onUp?: () => void;
  onDown?: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(step.title);
  const mark = STEP_MARK[step.status ?? "pending"];
  const active = step.status === "active";

  const commit = () => {
    const t = draft.trim();
    if (t && t !== step.title) onEdit(t);
    setEditing(false);
  };

  return (
    <li style={{ display: "flex", alignItems: "flex-start", gap: "var(--ma-3)", padding: "var(--ma-1) 0", position: "relative" }}>
      {/* the spine: status glyph + a shadow-line connector down to the next step. The active step's
          marker breathes (the .alive keyframe), the agent's current move read as light. */}
      <span
        aria-hidden
        title={step.status ?? "pending"}
        className={active ? "alive" : undefined}
        style={{
          flex: "0 0 auto",
          width: 16,
          textAlign: "center",
          color: mark.color,
          ...(active ? { borderRadius: "var(--radius-pill)" } : null),
        }}
      >
        {mark.glyph}
      </span>
      {!last ? (
        <span aria-hidden style={{ position: "absolute", left: 7, top: 24, bottom: -4, width: 1, background: "var(--line)" }} />
      ) : null}

      <div style={{ flex: 1, minWidth: 0 }}>
        {editing ? (
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit();
              if (e.key === "Escape") setEditing(false);
            }}
            className="t-body"
            style={{
              width: "100%",
              background: "var(--concrete-4)",
              border: "none",
              outline: "none",
              color: "var(--text-1)",
              font: "inherit",
              padding: "var(--ma-1) var(--ma-2)",
              borderRadius: "var(--radius)",
              boxShadow: "var(--hairline)",
            }}
          />
        ) : (
          <button
            onClick={() => {
              setDraft(step.title);
              setEditing(true);
            }}
            title="edit step"
            className="t-body"
            style={{
              textAlign: "left",
              width: "100%",
              color: active ? "var(--text-1)" : step.status === "done" ? "var(--text-3)" : "var(--text-2)",
              textDecoration: step.status === "skipped" ? "line-through" : undefined,
            }}
          >
            <span style={{ color: "var(--text-3)", marginRight: "var(--ma-3)" }}>{index + 1}</span>
            {step.title}
          </button>
        )}
        {step.detail ? (
          <div className="t-micro" style={{ paddingLeft: 22, marginTop: "var(--ma-1)" }}>{step.detail}</div>
        ) : null}
      </div>

      {/* reorder handles (mouse-rich where it is spatial); shown dim, brighten on hover via focus ring */}
      <span style={{ flex: "0 0 auto", display: "flex", gap: 2 }}>
        {onUp ? (
          <button onClick={onUp} title="move up" style={reorderBtn}>
            ↑
          </button>
        ) : null}
        {onDown ? (
          <button onClick={onDown} title="move down" style={reorderBtn}>
            ↓
          </button>
        ) : null}
      </span>
    </li>
  );
}

const reorderBtn: CSSProperties = {
  color: "var(--text-3)",
  fontSize: "12px",
  padding: "0 var(--ma-1)",
  lineHeight: 1.4,
};

// ---- ToolChip: one calm chip per tool call (no churn). Bound to tool_progress{call_id,message}. ----
export function ToolChip({ tool }: { tool: ToolEvent }) {
  // The message is the agent's own present-tense narration ("edit guard.rs: moved drop past retry").
  const verb = tool.message.split(/[:\s]/, 1)[0] || "tool";
  return (
    <span style={chip} title={tool.call_id}>
      <span aria-hidden style={{ color: "var(--text-3)" }}>
        ▸
      </span>
      <span style={{ color: "var(--text-3)" }}>{verb}</span>
      <span style={{ color: "var(--text-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {tool.message.slice(verb.length).replace(/^[:\s]+/, "")}
      </span>
    </span>
  );
}

export function ToolChipRow({ tools }: { tools: ToolEvent[] }) {
  if (tools.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-3)" }}>
      <span style={blockLabel}>Tools</span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--ma-2)" }}>
        {tools.map((t, i) => (
          <ToolChip key={t.call_id + i} tool={t} />
        ))}
      </div>
    </div>
  );
}

// ---- DiffChipRow: a produced diff -> a chip that opens the hunk diff review. ----
export function DiffChipRow({
  chips,
  onOpen,
  onAccept,
  onReject,
}: {
  chips: DiffChip[];
  onOpen: (chip: DiffChip) => void;
  onAccept: (chip: DiffChip) => void;
  onReject: (chip: DiffChip) => void;
}) {
  if (chips.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-3)" }}>
      <span style={blockLabel}>Diffs</span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--ma-2)" }}>
        {chips.map((c) => {
          const settled = c.status === "applied" || c.status === "rejected";
          const file = c.path.split("/").pop() ?? c.path;
          return (
            <span key={c.diff_id} style={{ ...chip, paddingRight: settled ? "var(--ma-3)" : "var(--ma-1)" }}>
              <button onClick={() => onOpen(c)} title={c.path} style={{ display: "inline-flex", alignItems: "center", gap: "var(--ma-2)", color: "inherit" }}>
                <span aria-hidden style={{ color: "var(--text-3)" }}>
                  ◫
                </span>
                <span style={{ color: "var(--text-2)" }}>{file}</span>
                {/* added/removed counts in the diff pigments, each glyph-paired so color is never alone */}
                {c.added != null ? <span style={{ color: "var(--ok)" }}>+{c.added}</span> : null}
                {c.removed != null ? <span style={{ color: "var(--bad)" }}>-{c.removed}</span> : null}
              </button>
              {c.status === "applied" ? (
                <span style={{ color: "var(--ok)" }}>● applied</span>
              ) : c.status === "rejected" ? (
                <span style={{ color: "var(--text-3)" }}>rejected</span>
              ) : c.status === "stale" ? (
                // "stale" is a needs-you state, not a third color: a neutral glyph + --mute text.
                <span style={{ color: "var(--mute)" }}>⟳ stale</span>
              ) : (
                <span style={{ display: "inline-flex", gap: "var(--ma-1)" }}>
                  <button onClick={() => onAccept(c)} title="accept (Cmd+Enter)" style={ctlStyle(true)}>
                    a
                  </button>
                  <button onClick={() => onReject(c)} title="reject (Cmd+Backspace)" style={ctlStyle(false)}>
                    r
                  </button>
                </span>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ---- InlineGate: the SecurityGate as a lit approval inline in the stream (doctrine: see/steer/gate). ----
export function InlineGate({
  gate,
  message,
  onApprove,
  onDismiss,
}: {
  gate: string;
  message: string;
  onApprove: () => void;
  onDismiss: () => void;
}) {
  return (
    // The agent needs you: the volume holds the steady light of a threshold (alive breathe + lit glyph).
    <Volume alive pad="var(--ma-4)">
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-2)", marginBottom: "var(--ma-2)" }}>
        <span aria-hidden style={{ color: "var(--light)" }}>
          ◈
        </span>
        <span style={{ ...blockLabel, color: "var(--text-2)" }}>Approval</span>
        <span className="t-micro">{gate}</span>
      </div>
      <div className="t-body" style={{ color: "var(--text-1)", marginBottom: "var(--ma-4)" }}>{message}</div>
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-3)" }}>
        {/* the lit capsule: the one tactile control, holds steady, states plainly what happens */}
        <Gate onClick={onApprove} title="approve this action">
          Approve
        </Gate>
        <button onClick={onDismiss} style={ctlStyle(false)}>
          Dismiss
        </button>
      </div>
    </Volume>
  );
}
