/*
  chat/structure.tsx: the inline structure that lives INSIDE the assistant stream.
  Three calm, material, gold-rim devices, re-housed from the Cline/OpenCode plan-act + per-step
  chat UX into the HIDE doctrine (near-black panels, Geist Mono telemetry voice, gold rim-light,
  shape+label markers, no spinners, no churn):

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
import { Panel } from "../../ui";
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
    <Panel active={awaiting} pad="var(--s3)" style={{ background: "var(--surface-0)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "var(--s2)", marginBottom: "var(--s2)" }}>
        <span style={blockLabel}>Plan</span>
        <span style={{ fontSize: "var(--text-xs)", color: "var(--text-low)" }}>
          {done}/{steps.length}
        </span>
        {awaiting ? (
          <button onClick={onApprove} style={{ ...ctlStyle(true), marginLeft: "auto" }} title="approve plan (Cmd+Enter)">
            Approve
          </button>
        ) : null}
      </div>
      <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 2 }}>
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
    </Panel>
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
    <li style={{ display: "flex", alignItems: "flex-start", gap: "var(--s2)", padding: "3px 0", position: "relative" }}>
      {/* the spine: index marker + a hairline connector down to the next step */}
      <span
        aria-hidden
        title={step.status ?? "pending"}
        style={{
          flex: "0 0 auto",
          width: 16,
          textAlign: "center",
          color: mark.color,
          ...(active ? { animation: "radiation-breathe 2.2s ease-in-out infinite" } : null),
        }}
      >
        {mark.glyph}
      </span>
      {!last ? (
        <span aria-hidden style={{ position: "absolute", left: 7, top: 22, bottom: -3, width: 1, background: "var(--rim)" }} />
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
              background: "var(--surface-2)",
              border: "none",
              outline: "none",
              color: "var(--text-hi)",
              font: "inherit",
              fontSize: "var(--text-sm)",
              padding: "2px 6px",
              borderRadius: "var(--radius)",
              boxShadow: "inset 0 0 0 1px var(--rim)",
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
              fontSize: "var(--text-sm)",
              color: active ? "var(--text-hi)" : step.status === "done" ? "var(--text-low)" : "var(--text-mid)",
              textDecoration: step.status === "skipped" ? "line-through" : undefined,
            }}
          >
            <span style={{ color: "var(--text-low)", marginRight: "var(--s2)" }}>{index + 1}</span>
            {step.title}
          </button>
        )}
        {step.detail ? (
          <div style={{ fontSize: "var(--text-xs)", color: "var(--text-low)", paddingLeft: 22 }}>{step.detail}</div>
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
  color: "var(--text-low)",
  fontSize: "var(--text-xs)",
  padding: "0 3px",
  lineHeight: 1.4,
};

// ---- ToolChip: one calm chip per tool call (no churn). Bound to tool_progress{call_id,message}. ----
export function ToolChip({ tool }: { tool: ToolEvent }) {
  // The message is the agent's own present-tense narration ("edit guard.rs: moved drop past retry").
  const verb = tool.message.split(/[:\s]/, 1)[0] || "tool";
  return (
    <span style={chip} title={tool.call_id}>
      <span aria-hidden style={{ color: "var(--radiation)" }}>
        ▸
      </span>
      <span style={{ color: "var(--text-low)" }}>{verb}</span>
      <span style={{ color: "var(--text-mid)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {tool.message.slice(verb.length).replace(/^[:\s]+/, "")}
      </span>
    </span>
  );
}

export function ToolChipRow({ tools }: { tools: ToolEvent[] }) {
  if (tools.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--s1)" }}>
      <span style={blockLabel}>Tools</span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--s2)" }}>
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
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--s1)" }}>
      <span style={blockLabel}>Diffs</span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--s2)" }}>
        {chips.map((c) => {
          const settled = c.status === "applied" || c.status === "rejected";
          const file = c.path.split("/").pop() ?? c.path;
          return (
            <span key={c.diff_id} style={{ ...chip, paddingRight: settled ? 9 : 3 }}>
              <button onClick={() => onOpen(c)} title={c.path} style={{ display: "inline-flex", alignItems: "center", gap: "var(--s2)", color: "inherit" }}>
                <span aria-hidden style={{ color: "var(--radiation)" }}>
                  ◫
                </span>
                <span style={{ color: "var(--text-mid)" }}>{file}</span>
                {c.added != null ? <span style={{ color: "var(--diff-add-fg)" }}>+{c.added}</span> : null}
                {c.removed != null ? <span style={{ color: "var(--diff-del-fg)" }}>-{c.removed}</span> : null}
              </button>
              {c.status === "applied" ? (
                <span style={{ color: "var(--success)" }}>applied</span>
              ) : c.status === "rejected" ? (
                <span style={{ color: "var(--text-low)" }}>rejected</span>
              ) : c.status === "stale" ? (
                <span style={{ color: "var(--warning)" }}>stale</span>
              ) : (
                <span style={{ display: "inline-flex", gap: 2 }}>
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
    <Panel active pad="var(--s3)" style={{ background: "var(--surface-0)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "var(--s2)", marginBottom: "var(--s1)" }}>
        <span aria-hidden style={{ color: "var(--radiation-bright)" }}>
          ◈
        </span>
        <span style={{ ...blockLabel, color: "var(--radiation-bright)" }}>Approval</span>
        <span style={{ fontSize: "var(--text-xs)", color: "var(--text-low)" }}>{gate}</span>
      </div>
      <div style={{ fontSize: "var(--text-sm)", color: "var(--text-hi)", marginBottom: "var(--s2)" }}>{message}</div>
      <div style={{ display: "flex", gap: "var(--s2)" }}>
        <button onClick={onApprove} style={ctlStyle(true)}>
          Approve
        </button>
        <button onClick={onDismiss} style={ctlStyle(false)}>
          Dismiss
        </button>
      </div>
    </Panel>
  );
}
