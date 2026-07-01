/*
  chat/structure.tsx: the inline structure that lives INSIDE the assistant stream.
  Four flat Codex-style devices, re-housed into the VS Code / ChatGPT visual language
  (flat surfaces, thin borders, the accent only on the engaged control):

    PlanCard  <- projection_patch:plan       : ordered steps with per-step status + approve/edit/reorder.
    ToolChip  <- tool_progress{call_id,msg}   : one calm pill per tool call, no per-token churn.
    DiffChipRow <- projection_patch:diff_chip : a produced diff -> a "file edited" row + accept/reject/open.
    InlineGate <- security_gate{gate,message} : an approval card with an accent primary button.

  Steer/approve verbs go out as the registered intents: approve_plan / edit_plan_step / reorder_plan
  (Custom), accept_diff / reject_diff (enum), approve_gate (Custom). Nothing here owns transport;
  every action is an Intent handed up via the callbacks the surface wires to sendIntent.
*/
import { useState, type CSSProperties } from "react";
import type { ToolEvent } from "../../store";
import {
  blockLabel,
  chip,
  ctlStyle,
  STEP_MARK,
  type DiffChip,
  type PlanPatch,
  type PlanStep,
} from "./parts";

// ---- PlanCard: the plan rendered as ordered steps with status, as a flat card. ----
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
    <div className="chat-card">
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-3)", marginBottom: "var(--ma-3)" }}>
        <span style={blockLabel}>Plan</span>
        <span className="chat-card__meta">
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
    </div>
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
      {/* the spine: status glyph + a thin connector down to the next step. */}
      <span
        aria-hidden
        title={step.status ?? "pending"}
        style={{
          flex: "0 0 auto",
          width: 16,
          textAlign: "center",
          color: mark.color,
        }}
      >
        {mark.glyph}
      </span>
      {!last ? (
        <span aria-hidden style={{ position: "absolute", left: 7, top: 24, bottom: -4, width: 1, background: "var(--border)" }} />
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
            style={{
              width: "100%",
              background: "var(--input-bg)",
              border: "1px solid var(--input-border)",
              outline: "none",
              color: "var(--text)",
              font: "inherit",
              fontSize: "13px",
              padding: "var(--ma-1) var(--ma-2)",
              borderRadius: "var(--radius-sm)",
            }}
          />
        ) : (
          <button
            onClick={() => {
              setDraft(step.title);
              setEditing(true);
            }}
            title="edit step"
            style={{
              textAlign: "left",
              width: "100%",
              fontSize: "13px",
              lineHeight: 1.5,
              color: active ? "var(--text-strong)" : step.status === "done" ? "var(--text-dim)" : "var(--text)",
              textDecoration: step.status === "skipped" ? "line-through" : undefined,
            }}
          >
            <span style={{ color: "var(--text-dim)", marginRight: "var(--ma-3)" }}>{index + 1}</span>
            {step.title}
          </button>
        )}
        {step.detail ? (
          <div className="chat-card__meta" style={{ paddingLeft: 22, marginTop: "var(--ma-1)" }}>{step.detail}</div>
        ) : null}
      </div>

      {/* reorder handles */}
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
  color: "var(--text-dim)",
  fontSize: "12px",
  padding: "0 var(--ma-1)",
  lineHeight: 1.4,
};

// ---- ToolChip: one calm pill per tool call (no churn). Bound to tool_progress{call_id,message}. ----
export function ToolChip({ tool }: { tool: ToolEvent }) {
  const verb = tool.message.split(/[:\s]/, 1)[0] || "tool";
  return (
    <span style={chip} title={tool.call_id}>
      <span aria-hidden style={{ color: "var(--text-dim)" }}>
        ▸
      </span>
      <span style={{ color: "var(--text)" }}>{verb}</span>
      <span style={{ color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {tool.message.slice(verb.length).replace(/^[:\s]+/, "")}
      </span>
    </span>
  );
}

export function ToolChipRow({ tools }: { tools: ToolEvent[] }) {
  if (tools.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-2)" }}>
      <span style={blockLabel}>Tools</span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--ma-2)" }}>
        {tools.map((t, i) => (
          <ToolChip key={t.call_id + i} tool={t} />
        ))}
      </div>
    </div>
  );
}

// ---- DiffChipRow: a produced diff -> a "file edited" row that opens the hunk diff review. ----
export function DiffChipRow({
  chips,
  onOpen,
  onAccept,
  onReject,
}: {
  chips: DiffChip[];
  onOpen: (chip: DiffChip) => void;
  // Accept and reject are owned by the editor and the hunk panel, not the Executor. When these are
  // omitted (the Executor case) the chip is open-only: it routes you to the editor to review.
  onAccept?: (chip: DiffChip) => void;
  onReject?: (chip: DiffChip) => void;
}) {
  if (chips.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-2)" }}>
      <span style={blockLabel}>Diffs</span>
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-2)" }}>
        {chips.map((c) => {
          const file = c.path.split("/").pop() ?? c.path;
          return (
            <div key={c.diff_id} className="diff-row">
              <button onClick={() => onOpen(c)} title={c.path} className="diff-row__open">
                <span aria-hidden className="diff-row__icon">
                  ◫
                </span>
                <span className="diff-row__file">{file}</span>
                {c.added != null ? <span style={{ color: "var(--git-add)" }}>+{c.added}</span> : null}
                {c.removed != null ? <span style={{ color: "var(--git-del)" }}>-{c.removed}</span> : null}
              </button>
              {c.status === "applied" ? (
                <span style={{ color: "var(--green)", fontSize: "12px" }}>● applied</span>
              ) : c.status === "rejected" ? (
                <span style={{ color: "var(--text-dim)", fontSize: "12px" }}>rejected</span>
              ) : c.status === "stale" ? (
                <span style={{ color: "var(--git-mod)", fontSize: "12px" }}>⟳ stale</span>
              ) : onAccept && onReject ? (
                <span style={{ display: "inline-flex", gap: "var(--ma-1)" }}>
                  <button onClick={() => onAccept(c)} title="accept (Cmd+Enter)" style={ctlStyle(true)}>
                    Accept
                  </button>
                  <button onClick={() => onReject(c)} title="reject (Cmd+Backspace)" style={ctlStyle(false)}>
                    Reject
                  </button>
                </span>
              ) : (
                <button onClick={() => onOpen(c)} title="open to review in the editor" style={{ color: "var(--text-dim)", fontSize: "12px", background: "transparent" }}>
                  review
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- InlineGate: the SecurityGate as an approval card inline in the stream. ----
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
    <div className="chat-card chat-card--gate">
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-2)", marginBottom: "var(--ma-2)" }}>
        <span aria-hidden style={{ color: "var(--accent)" }}>
          ◈
        </span>
        <span style={blockLabel}>Approval</span>
        <span className="chat-card__meta">{gate}</span>
      </div>
      <div style={{ color: "var(--text)", fontSize: "13px", lineHeight: 1.5, marginBottom: "var(--ma-4)" }}>{message}</div>
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-2)" }}>
        <button className="gate" onClick={onApprove} title="approve this action">
          Approve
        </button>
        <button onClick={onDismiss} style={ctlStyle(false)}>
          Dismiss
        </button>
      </div>
    </div>
  );
}
