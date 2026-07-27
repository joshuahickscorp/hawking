/*
  chat/structure.tsx: the inline structure that lives INSIDE the assistant stream.
  Four flat pill-style devices, re-housed into the VS Code / ChatGPT visual language
  (flat surfaces, thin borders, the accent only on the engaged control):

    PlanCard  <- projection_patch:plan       : the durable host plan record (crates/hide-backend
                                               plan_domain.rs PlanRecord), ordered steps with the
                                               declared contract + live state, expandable per step.
    ToolChip  <- tool_progress{call_id,msg}   : one calm pill per tool call, no per-token churn.
    DiffChipRow <- projection_patch:diff_chip : a produced diff -> a "file edited" row that OPENS the
                                               review. Accept/reject belong to the Editor and the
                                               HunkReview surfaces (consolidation decision 3.1), so
                                               the in-chat pair is retired: one open/review control.
    InlineGate <- security_gate{gate,message} : an approval card with an accent primary button.

  Every plan verb resolves a catalog command id through the ONE spine (store.runCommand):
  approve_plan / edit_plan_step / reorder_plan / skip_step / repair_step / create_side_chat.
  Nothing here invents a verb and nothing here owns transport.

  WRITE BLOCKING is preserved end to end: a step the host marked write_blocked (any autonomy that is
  not full_auto) shows a gated badge with its own glyph AND its own words, and every effectful action
  is refused for it. That is distinct from a blocked step and from a failed step, by glyph and text,
  never by color alone.
*/
import { useEffect, useRef, useState, type CSSProperties } from "react";
import type { ToolEvent } from "../../store";
import { runCommand, useStore } from "../../store";
import type { IntentAck } from "../../wire";
import { Icon } from "../../shell/icons";
import { blockLabel, ctlStyle, type DiffChip } from "./parts";

/* ---- The plan projection, as the host emits it ------------------------------------------------
   Shape mirror of crates/hide-backend/src/plan_domain.rs (PlanRecord / PlanStepRecord). Fields stay
   optional because a projection_patch is a partial state diff: an early patch may carry a subset.
*/
export type PlanStepState = "pending" | "ready" | "running" | "blocked" | "completed" | "failed" | "skipped";
export type PlanVerification = "pending" | "passed" | "failed" | "skipped";

export interface PlanProjectionStep {
  id: string;
  text: string;
  status?: PlanStepState;
  dependencies?: string[];
  acceptance?: string;
  effects?: string[];
  related_files?: string[];
  owner_agent?: string;
  verification?: PlanVerification;
  blocker?: string;
  approved?: boolean;
  /** The step's effects are gated by the run autonomy (suggest_only / read_only). */
  write_blocked?: boolean;
}

export interface PlanProjection {
  plan_id?: string;
  title?: string;
  objective?: string;
  status?: "draft" | "active" | "completed" | "failed" | "superseded";
  autonomy?: "full_auto" | "suggest_only" | "read_only";
  approved?: boolean;
  steps?: PlanProjectionStep[];
}

/** Status -> glyph + words + tone. The glyph and the label carry the state; color only reinforces. */
export const PLAN_STEP_MARK: Record<PlanStepState, { glyph: string; label: string; color: string }> = {
  pending: { glyph: "○", label: "pending", color: "var(--text-dim)" },
  ready: { glyph: "◇", label: "ready", color: "var(--text)" },
  running: { glyph: "◉", label: "running", color: "var(--accent)" },
  blocked: { glyph: "⊗", label: "blocked", color: "var(--git-mod)" },
  completed: { glyph: "●", label: "completed", color: "var(--green)" },
  failed: { glyph: "✕", label: "failed", color: "var(--red)" },
  skipped: { glyph: "/", label: "skipped", color: "var(--text-dim)" },
};

/** The gated marker, deliberately a different glyph AND different words from blocked / failed. */
export const WRITE_BLOCKED_MARK = { glyph: "⊘", label: "write blocked" };

/* ---- Plan actions, every one resolved through the command spine ---- */

export type PlanActionId =
  | "approve_plan"
  | "approve_step"
  | "edit"
  | "skip"
  | "repair"
  | "reorder"
  | "side_chat"
  | "fork_alternative";

export interface PlanActionSpec {
  id: PlanActionId;
  label: string;
  /** Catalog command id (src/generated/command_catalog.json). */
  command: string;
  /** True when firing it causes a write. Refused for a write-blocked step. */
  effectful: boolean;
}

export const PLAN_ACTIONS: PlanActionSpec[] = [
  { id: "approve_plan", label: "Approve plan", command: "approve_plan", effectful: false },
  { id: "approve_step", label: "Approve this step", command: "approve_plan", effectful: false },
  { id: "edit", label: "Edit step text", command: "edit_plan_step", effectful: false },
  { id: "skip", label: "Skip step, with a reason", command: "skip_step", effectful: false },
  // repair appends a real edit step (write_fs) to the durable plan, so it is an effectful verb.
  { id: "repair", label: "Queue a repair step", command: "repair_step", effectful: true },
  { id: "reorder", label: "Reorder steps", command: "reorder_plan", effectful: false },
  { id: "side_chat", label: "Side chat about this step", command: "create_side_chat", effectful: false },
  { id: "fork_alternative", label: "Fork an alternative, fresh context", command: "create_side_chat", effectful: false },
];

export const planActionSpec = (id: PlanActionId): PlanActionSpec =>
  PLAN_ACTIONS.find((a) => a.id === id) as PlanActionSpec;

/** The per-step context menu. Reorder lives on the row's existing up/down handles, not here. */
export const PLAN_STEP_MENU: PlanActionId[] = [
  "approve_step",
  "edit",
  "skip",
  "repair",
  "side_chat",
  "fork_alternative",
];

/** Whether an entry can fire for this step right now. Offered-but-dead is worse than disabled. */
export function planActionEnabled(id: PlanActionId, step: PlanProjectionStep): boolean {
  // The write-block gate: a gated step never offers an effectful verb.
  if (step.write_blocked && planActionSpec(id).effectful) return false;
  const status = step.status ?? "pending";
  const terminal = status === "completed" || status === "skipped";
  switch (id) {
    case "approve_step":
      return !step.approved;
    case "edit":
    case "skip":
      return !terminal;
    case "repair":
      return step.verification === "failed" || status === "failed";
    default:
      return true;
  }
}

/** Why an entry is refused, so the disabled row explains itself instead of just dimming. */
export function planActionReason(id: PlanActionId, step: PlanProjectionStep): string {
  const spec = planActionSpec(id);
  if (planActionEnabled(id, step)) return spec.label;
  if (step.write_blocked && spec.effectful)
    return `${spec.label}: write blocked by the run autonomy`;
  if (id === "repair") return `${spec.label}: only a step whose verification failed can be repaired`;
  if (id === "approve_step") return `${spec.label}: already approved`;
  return `${spec.label}: not available for a ${step.status ?? "pending"} step`;
}

export interface PlanActionCtx {
  sessionId: string;
  step?: PlanProjectionStep;
  /** edit_plan_step */
  text?: string;
  /** skip_step: the host records it as the step's blocker, so it is required here. */
  reason?: string;
  /** reorder_plan: the full permutation of step ids. */
  order?: string[];
}

/**
 * THE dispatch point for every plan gesture. The card header, the row handles, the context menu and
 * the palette all land on the same catalog ids. Throws (never silently no-ops) so the caller can
 * surface a notice.
 */
export async function runPlanAction(id: PlanActionId, ctx: PlanActionCtx): Promise<IntentAck> {
  const session_id = ctx.sessionId;
  const step_id = ctx.step?.id;
  if (id !== "approve_plan" && id !== "reorder" && !step_id)
    throw new Error(`${planActionSpec(id).label} needs a step`);
  if (ctx.step && !planActionEnabled(id, ctx.step)) throw new Error(planActionReason(id, ctx.step));
  switch (id) {
    case "approve_plan":
      // No step_id = approve the whole plan (and every step) on the durable record.
      return runCommand("approve_plan", { session_id });
    case "approve_step":
      return runCommand("approve_plan", { session_id, step_id });
    case "edit": {
      const text = (ctx.text ?? "").trim();
      if (!text) throw new Error("Edit step needs the new text");
      return runCommand("edit_plan_step", { session_id, step_id, text });
    }
    case "skip": {
      const reason = (ctx.reason ?? "").trim();
      if (!reason) throw new Error("Skip step needs a reason");
      return runCommand("skip_step", { session_id, step_id, reason });
    }
    case "repair":
      return runCommand("repair_step", { session_id, step_id });
    case "reorder": {
      const order = ctx.order ?? [];
      if (order.length === 0) throw new Error("Reorder needs the step order");
      return runCommand("reorder_plan", { session_id, order });
    }
    case "side_chat":
      return runCommand("create_side_chat", { session_id, inherit: true });
    case "fork_alternative":
      return runCommand("create_side_chat", { session_id, inherit: false });
  }
}

/** Move step `i` by `d` and return the resulting full order (reorder_plan wants a permutation). */
export function reorderedIds(steps: PlanProjectionStep[], i: number, d: -1 | 1): string[] {
  const order = steps.map((s) => s.id);
  const j = i + d;
  if (j < 0 || j >= order.length) return order;
  [order[i], order[j]] = [order[j], order[i]];
  return order;
}

/** The screen-reader sentence for a step: number, text, state, verification, gate. */
export function stepAriaLabel(step: PlanProjectionStep, index: number): string {
  const mark = PLAN_STEP_MARK[step.status ?? "pending"];
  const bits = [`Step ${index + 1}: ${step.text}`, mark.label];
  if (step.verification) bits.push(`verification ${step.verification}`);
  if (step.write_blocked) bits.push(WRITE_BLOCKED_MARK.label);
  if (step.approved) bits.push("approved");
  return `${bits.join(", ")}. Expand for detail, Shift+F10 for step actions.`;
}

// ---- PlanCard: the durable plan rendered as ordered steps, as a flat card. ----
export function PlanCard({ plan, sessionId }: { plan: PlanProjection; sessionId: string }) {
  const steps = plan.steps ?? [];
  const pushNotice = useStore((s) => s.pushNotice);
  if (steps.length === 0) return null;
  const done = steps.filter((s) => s.status === "completed").length;

  const fire = async (id: PlanActionId, ctx: Omit<PlanActionCtx, "sessionId">) => {
    try {
      const ack = await runPlanAction(id, { sessionId, ...ctx });
      if (!ack.accepted)
        pushNotice({ kind: "error", code: "rejected", message: ack.message ?? `${planActionSpec(id).label} rejected` });
    } catch (err) {
      pushNotice({ kind: "error", code: "plan", message: (err as Error).message });
    }
  };

  return (
    <div className="chat-card">
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-3)", marginBottom: "var(--ma-3)" }}>
        <span style={blockLabel}>Plan</span>
        <span className="chat-card__meta">{plan.title ?? plan.objective ?? ""}</span>
        <span className="chat-card__meta">
          {done}/{steps.length}
        </span>
        {plan.approved !== true ? (
          <button
            onClick={() => void fire("approve_plan", {})}
            style={{ ...ctlStyle(true), marginLeft: "auto" }}
            aria-label="Approve plan"
            title="Approve the whole plan. Effects stay gated by the run autonomy."
          >
            Approve
          </button>
        ) : (
          <span className="chat-card__meta" style={{ marginLeft: "auto" }} aria-label="Plan approved">
            ✓ approved
          </span>
        )}
      </div>
      <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: "var(--ma-1)" }}>
        {steps.map((s, i) => (
          <PlanStepRow
            key={s.id ?? i}
            step={s}
            index={i}
            last={i === steps.length - 1}
            onFire={fire}
            onMove={(d) => void fire("reorder", { order: reorderedIds(steps, i, d) })}
            canUp={i > 0}
            canDown={i < steps.length - 1}
          />
        ))}
      </ol>
    </div>
  );
}

type InputMode = "none" | "edit" | "skip";

function PlanStepRow({
  step,
  index,
  last,
  onFire,
  onMove,
  canUp,
  canDown,
}: {
  step: PlanProjectionStep;
  index: number;
  last: boolean;
  onFire: (id: PlanActionId, ctx: Omit<PlanActionCtx, "sessionId">) => Promise<void>;
  onMove: (d: -1 | 1) => void;
  canUp: boolean;
  canDown: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [menu, setMenu] = useState(false);
  const [mode, setMode] = useState<InputMode>("none");
  const [draft, setDraft] = useState("");
  const headRef = useRef<HTMLButtonElement>(null);
  const mark = PLAN_STEP_MARK[step.status ?? "pending"];
  const active = step.status === "running";
  const detailId = `plan-step-detail-${step.id}`;

  // Focus never strands: the menu and the inline input both hand it back to the row head.
  const restore = () => headRef.current?.focus();
  const closeMenu = () => {
    setMenu(false);
    restore();
  };
  const cancelInput = () => {
    setMode("none");
    restore();
  };
  const commitInput = () => {
    const v = draft.trim();
    const m = mode;
    setMode("none");
    restore();
    if (!v) return;
    if (m === "edit") {
      if (v !== step.text) void onFire("edit", { step, text: v });
    } else if (m === "skip") {
      void onFire("skip", { step, reason: v });
    }
  };

  const pick = (id: PlanActionId) => {
    setMenu(false);
    if (id === "edit" || id === "skip") {
      setDraft(id === "edit" ? step.text : "");
      setMode(id);
      return;
    }
    restore();
    void onFire(id, { step });
  };

  return (
    <li style={{ display: "flex", alignItems: "flex-start", gap: "var(--ma-3)", padding: "var(--ma-1) 0", position: "relative" }}>
      {/* the spine: status glyph + a thin connector down to the next step. */}
      <span aria-hidden title={mark.label} style={{ flex: "0 0 auto", width: 16, textAlign: "center", color: mark.color }}>
        {mark.glyph}
      </span>
      {!last ? (
        <span aria-hidden style={{ position: "absolute", left: 7, top: 24, bottom: -4, width: 1, background: "var(--border)" }} />
      ) : null}

      <div style={{ flex: 1, minWidth: 0 }}>
        {mode !== "none" ? (
          <input
            autoFocus
            aria-label={mode === "edit" ? `Edit step ${index + 1}` : `Reason for skipping step ${index + 1}`}
            placeholder={mode === "skip" ? "Why this step is skipped (required)" : undefined}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={mode === "edit" ? commitInput : cancelInput}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitInput();
              if (e.key === "Escape") cancelInput();
            }}
            style={{
              width: "100%",
              background: "var(--input-bg)",
              border: "1px solid var(--input-border)",
              outline: "none",
              color: "var(--text)",
              font: "inherit",
              fontSize: "var(--fs-ui)",
              padding: "var(--ma-1) var(--ma-2)",
              borderRadius: "var(--radius-sm)",
            }}
          />
        ) : (
          <button
            ref={headRef}
            type="button"
            aria-expanded={open}
            aria-controls={detailId}
            aria-label={stepAriaLabel(step, index)}
            title="Click for detail. Right-click or Shift+F10 for step actions."
            onClick={() => setOpen((v) => !v)}
            onContextMenu={(e) => {
              e.preventDefault();
              setMenu(true);
            }}
            onKeyDown={(e) => {
              if (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) {
                e.preventDefault();
                setMenu(true);
              }
            }}
            style={{
              textAlign: "left",
              width: "100%",
              fontSize: "var(--fs-ui)",
              lineHeight: 1.5,
              color: active ? "var(--text-strong)" : step.status === "completed" ? "var(--text-dim)" : "var(--text)",
              textDecoration: step.status === "skipped" ? "line-through" : undefined,
            }}
          >
            <span aria-hidden style={{ color: "var(--text-dim)", marginRight: "var(--ma-3)" }}>
              {index + 1}
            </span>
            {step.text}
            {step.write_blocked ? (
              <span style={gateBadge} title="Effects gated by the run autonomy: this step cannot write yet.">
                {WRITE_BLOCKED_MARK.glyph} {WRITE_BLOCKED_MARK.label}
              </span>
            ) : null}
          </button>
        )}

        {open ? <StepDetail id={detailId} step={step} /> : null}
      </div>

      {/* reorder handles (existing controls, now sending the full permutation reorder_plan wants) */}
      <span style={{ flex: "0 0 auto", display: "flex", gap: 2 }}>
        {canUp ? (
          <button onClick={() => onMove(-1)} aria-label={`Move step ${index + 1} up`} title="Move up" style={reorderBtn}>
            ↑
          </button>
        ) : null}
        {canDown ? (
          <button onClick={() => onMove(1)} aria-label={`Move step ${index + 1} down`} title="Move down" style={reorderBtn}>
            ↓
          </button>
        ) : null}
      </span>

      {menu ? <StepMenu step={step} onPick={pick} onClose={closeMenu} /> : null}
    </li>
  );
}

/** The per-step context menu, reusing the shared popover styling (no new permanent control).
 *  Opened from the keyboard (Shift+F10), so focus MOVES into it: without that its own Escape handler
 *  never fired and the menu could be stranded open. Same contract as CodeActions: focus the first
 *  entry on open, Escape closes, an outside click closes, and closing hands focus back to the row. */
function StepMenu({
  step,
  onPick,
  onClose,
}: {
  step: PlanProjectionStep;
  onPick: (id: PlanActionId) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.querySelector<HTMLButtonElement>(".hc__addmenu__item:not([disabled])")?.focus();
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [onClose]);

  return (
    <div
      ref={ref}
      className="hc__addmenu"
      role="menu"
      aria-label="Step actions"
      style={STEP_MENU_POS}
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          onClose();
        }
      }}
    >
      {PLAN_STEP_MENU.map((id) => {
        const on = planActionEnabled(id, step);
        return (
          <button
            key={id}
            className="hc__addmenu__item"
            role="menuitem"
            type="button"
            disabled={!on}
            aria-disabled={!on}
            style={{ opacity: on ? 1 : 0.4, cursor: on ? "pointer" : "default" }}
            title={planActionReason(id, step)}
            onClick={() => onPick(id)}
          >
            {planActionSpec(id).label}
          </button>
        );
      })}
    </div>
  );
}

const STEP_MENU_POS: CSSProperties = { bottom: "auto", top: "100%", left: "auto", right: 0 };

/** The expandable detail: the declared contract plus the live state, as a definition list. */
export function StepDetail({ id, step }: { id?: string; step: PlanProjectionStep }) {
  const mark = PLAN_STEP_MARK[step.status ?? "pending"];
  const rows: [string, string][] = [
    ["Status", `${mark.glyph} ${mark.label}${step.approved ? ", approved" : ""}`],
    ["Verification", step.verification ?? "pending"],
  ];
  if (step.acceptance) rows.push(["Acceptance", step.acceptance]);
  if (step.dependencies?.length) rows.push(["Depends on", step.dependencies.join(", ")]);
  if (step.effects?.length) rows.push(["Effects", step.effects.join(", ")]);
  if (step.related_files?.length) rows.push(["Files", step.related_files.join(", ")]);
  if (step.owner_agent) rows.push(["Owner", step.owner_agent]);
  if (step.blocker) rows.push(["Blocker", step.blocker]);
  if (step.write_blocked)
    rows.push([
      WRITE_BLOCKED_MARK.label,
      "Effects are gated by the run autonomy. This step cannot write, and no effectful action is offered for it.",
    ]);

  return (
    <dl id={id} className="chat-card__meta" style={{ margin: "var(--ma-1) 0 0", paddingLeft: 22, display: "grid", gridTemplateColumns: "auto 1fr", gap: "2px var(--ma-3)" }}>
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: "contents" }}>
          <dt style={{ color: "var(--text-dim)" }}>{k}</dt>
          <dd style={{ margin: 0, minWidth: 0, overflowWrap: "anywhere" }}>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

const reorderBtn: CSSProperties = {
  color: "var(--text-dim)",
  fontSize: "var(--fs-small)",
  padding: "0 var(--ma-1)",
  lineHeight: 1.4,
};

const gateBadge: CSSProperties = {
  marginLeft: "var(--ma-2)",
  padding: "0 var(--ma-1)",
  borderRadius: "var(--radius-sm)",
  border: "1px solid var(--border-strong)",
  color: "var(--text-muted)",
  fontSize: "var(--fs-small)",
  whiteSpace: "nowrap",
};

// ---- Tool rows: one quiet inline row per tool call, collapsible, in the transcript flow
//      (Claude Code texture: "Read a file, ran a command >"). Bound to tool_progress{call_id,message}. ----
function ToolRow({ tool }: { tool: ToolEvent }) {
  const [open, setOpen] = useState(false);
  const text = tool.message.charAt(0).toUpperCase() + tool.message.slice(1);
  return (
    <div className="toolrow-inline">
      <button className="toolrow-inline__head" type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="toolrow-inline__text">{text}</span>
        <Icon name={open ? "chevron-down" : "chevron-right"} size={13} />
      </button>
      {open ? <div className="toolrow-inline__body">{tool.call_id}</div> : null}
    </div>
  );
}

export function ToolChipRow({ tools }: { tools: ToolEvent[] }) {
  if (tools.length === 0) return null;
  return (
    <div className="toolflow">
      {tools.map((t, i) => (
        <ToolRow key={t.call_id + i} tool={t} />
      ))}
    </div>
  );
}

// ---- DiffChipRow: a produced diff -> a "file edited" row that opens the hunk diff review. ----
// ONE control per chip: open/review. The old in-chat Accept/Reject pair never rendered (no surface
// ever passed the handlers) and real review lives in the Editor + HunkReview, so it is retired.
export function DiffChipRow({ chips, onOpen }: { chips: DiffChip[]; onOpen: (chip: DiffChip) => void }) {
  if (chips.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-2)" }}>
      <span style={blockLabel}>Diffs</span>
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--ma-2)" }}>
        {chips.map((c) => {
          const file = c.path.split("/").pop() ?? c.path;
          return (
            <div key={c.diff_id} className="diff-row">
              <button
                onClick={() => onOpen(c)}
                title={c.path}
                aria-label={`Review ${c.path} in the editor`}
                className="diff-row__open"
              >
                <span aria-hidden className="diff-row__icon">
                  ◫
                </span>
                <span className="diff-row__file">{file}</span>
                {c.added != null ? <span style={{ color: "var(--git-add)" }}>+{c.added}</span> : null}
                {c.removed != null ? <span style={{ color: "var(--git-del)" }}>-{c.removed}</span> : null}
              </button>
              {c.status === "applied" ? (
                <span style={{ color: "var(--green)", fontSize: "var(--fs-small)" }}>● applied</span>
              ) : c.status === "rejected" ? (
                <span style={{ color: "var(--text-dim)", fontSize: "var(--fs-small)" }}>✕ rejected</span>
              ) : c.status === "stale" ? (
                <span style={{ color: "var(--git-mod)", fontSize: "var(--fs-small)" }}>⟳ stale</span>
              ) : null}
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
      <div style={{ color: "var(--text)", fontSize: "var(--fs-ui)", lineHeight: 1.5, marginBottom: "var(--ma-4)" }}>{message}</div>
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
