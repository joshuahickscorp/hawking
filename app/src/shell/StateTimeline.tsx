/*
  StateTimeline.tsx - the ONE history surface.

  The row itself is unchanged in shape: a "state" label, the step dots, the step message, and the
  "live" return. Clicking a dot SELECTS that step as the boundary the history menu addresses; it is a
  local selection and the header no longer claims otherwise. `scrub_to_event` was retired: the host
  recorded the intent and no arm acted on it (`BackendHost::scrub_to_event` is a read-only projection
  query that is never reached from the intent path and has no surface to render a past projection),
  so it was a control that could not work. What changed before that is that the separate permanent
  "fork from here" BUTTON is gone. Fork, and every other time-travel verb the host supports, lives
  behind ONE menu on the row, so the control count went DOWN while the capability went up (campaign
  rule: no button per backend command).

  Every entry names a catalog command id (src/generated/command_catalog.json, mirrored from
  crates/hide-protocol command_catalog) and carries the payload crates/hide-backend
  handle_goal_checkpoint_intent parses, so the palette, a shortcut and this menu can never mean
  different things:

    create              -> checkpoint_create   { session_id, at_event, label }
    fork from here      -> fork_session        { session_id, at_event }
    inspect             -> checkpoint_inspect  { checkpoint_id }   (integrity + coverage drift)
    compare             -> checkpoint_compare  { checkpoint_id, session_id }
    replay              -> checkpoint_replay   { checkpoint_id }
    fork from checkpoint-> checkpoint_fork     { checkpoint_id }
    restore             -> checkpoint_restore  { checkpoint_id }
    rewind conversation -> checkpoint_rewind   { checkpoint_id, target: "conversation" }
    rewind code         -> checkpoint_rewind   { checkpoint_id, target: "code" }
    rewind both         -> checkpoint_rewind   { checkpoint_id, target: "both" }

  A rewind reverts work, so its target is named in the label (never a bare "rewind") and it takes an
  explicit second confirmation click before anything is sent.

  The boundary verbs (create, fork from this step) address `ToolEvent.event_id`, the id of the
  RECORDED event the step is, carried on the tool_progress UiEvent. They used to be handed the tool
  `call_id`, which `replay.rs seq_of_event` resolves as NotFound, so BOTH always failed on a live
  host and, since checkpoint_create is the only producer of a checkpoint id, the other seven entries
  could never enable. A step with no recorded event (streamed process output) is blocked with a
  stated reason instead of being addressed with something the host cannot resolve.

  The checkpoint verbs address `store.lastCheckpointId`, the id folded from the host's
  checkpoint_created Custom UiEvent. REMAINING GAP, reported rather than faked: the store keeps the
  last id, not a checkpoint LIST, and a rewind's `detail.invalidated_receipts` still sits past the
  200-character truncation of the Custom info notice, so this surface cannot list or diff receipts.
*/
import { useRef, useState } from "react";
import { runCommand, useStore } from "../store";
import type { CommandArgs, ToolEvent } from "../store";
import { ackState, heldNote } from "../wire";
import { Icon } from "./icons";

export type TimelineActionId =
  | "checkpoint_here"
  | "fork_event"
  | "inspect"
  | "compare"
  | "replay"
  | "fork_checkpoint"
  | "restore"
  | "rewind_conversation"
  | "rewind_code"
  | "rewind_both";

export type RewindTarget = "conversation" | "code" | "both";

export interface TimelineAction {
  id: TimelineActionId;
  label: string;
  /** The catalog command id this entry resolves. Nothing here invents a verb. */
  command: string;
  /** Named explicitly so "rewind" is never ambiguous about what it reverts. */
  target?: RewindTarget;
  /** Needs a host-minted checkpoint id; disabled with a stated reason until one exists. */
  needsCheckpoint: boolean;
  /** Reverts work, so it asks for an explicit second click before it is sent. */
  destructive: boolean;
  hint: string;
}

export const TIMELINE_ACTIONS: TimelineAction[] = [
  {
    id: "checkpoint_here",
    label: "Create checkpoint here",
    command: "checkpoint_create",
    needsCheckpoint: false,
    destructive: false,
    hint: "Seal an integrity-verified restore point (blake3) at this step",
  },
  {
    id: "fork_event",
    label: "Fork from this step",
    command: "fork_session",
    needsCheckpoint: false,
    destructive: false,
    hint: "Branch a new session whose history is this one folded up to this step",
  },
  {
    id: "inspect",
    label: "Inspect checkpoint",
    command: "checkpoint_inspect",
    needsCheckpoint: true,
    destructive: false,
    hint: "Re-verify the sealed integrity and report coverage drift since the boundary",
  },
  {
    id: "compare",
    label: "Compare current versus checkpoint",
    command: "checkpoint_compare",
    needsCheckpoint: true,
    destructive: false,
    hint: "Diff this session's code state against the checkpoint boundary",
  },
  {
    id: "replay",
    label: "Replay from checkpoint",
    command: "checkpoint_replay",
    needsCheckpoint: true,
    destructive: false,
    hint: "Re-apply the recorded history forward onto a fresh lineage. Drops nothing",
  },
  {
    id: "fork_checkpoint",
    label: "Fork from checkpoint",
    command: "checkpoint_fork",
    needsCheckpoint: true,
    destructive: false,
    hint: "Branch a new independent session seeded at the checkpoint boundary",
  },
  {
    id: "restore",
    label: "Restore checkpoint",
    command: "checkpoint_restore",
    needsCheckpoint: true,
    destructive: false,
    hint: "Open a new session whose history is the source folded to the boundary. The source is untouched",
  },
  {
    id: "rewind_conversation",
    label: "Rewind conversation only",
    command: "checkpoint_rewind",
    target: "conversation",
    needsCheckpoint: true,
    destructive: true,
    hint: "Drop the conversation after the boundary and keep every code change",
  },
  {
    id: "rewind_code",
    label: "Rewind code only",
    command: "checkpoint_rewind",
    target: "code",
    needsCheckpoint: true,
    destructive: true,
    hint: "Revert files changed after the boundary and keep the conversation. Verification receipts covering those files are invalidated",
  },
  {
    id: "rewind_both",
    label: "Rewind conversation and code",
    command: "checkpoint_rewind",
    target: "both",
    needsCheckpoint: true,
    destructive: true,
    hint: "Revert both after the boundary. Verification receipts covering the reverted files are invalidated",
  },
];

export const actionById = (id: TimelineActionId): TimelineAction =>
  TIMELINE_ACTIONS.find((a) => a.id === id) as TimelineAction;

export interface TimelineCtx {
  sessionId: string;
  /** The RECORDED event id of the selected step, the boundary a create or a fork uses. Empty when
   *  the step is not a recorded event, which blocks the boundary verbs rather than faking one. */
  atEvent: string;
  checkpointId: string | null;
  /** Human label for a new checkpoint (the step message, trimmed). */
  label: string;
}

/** A command plan is data, so a test asserts exactly which command and payload an entry means. */
export interface TimelinePlan {
  id: string;
  args: CommandArgs;
}

export function timelinePlan(action: TimelineAction, ctx: TimelineCtx): TimelinePlan {
  switch (action.id) {
    case "checkpoint_here":
      return {
        id: action.command,
        args: { session_id: ctx.sessionId, at_event: ctx.atEvent, label: ctx.label || "checkpoint" },
      };
    case "fork_event":
      return { id: action.command, args: { session_id: ctx.sessionId, at_event: ctx.atEvent } };
    case "compare":
      return { id: action.command, args: { checkpoint_id: ctx.checkpointId, session_id: ctx.sessionId } };
    default:
      return {
        id: action.command,
        args: action.target
          ? { checkpoint_id: ctx.checkpointId, target: action.target }
          : { checkpoint_id: ctx.checkpointId },
      };
  }
}

/** Build the menu's context from the SELECTED step. Exported so a test can assert the boundary id
 *  is the recorded event id (what the host resolves), never the tool call id (what it refuses). */
export function timelineCtx(
  step: ToolEvent | undefined,
  sessionId: string,
  checkpointId: string | null,
): TimelineCtx {
  return {
    sessionId,
    atEvent: step?.event_id ?? "",
    checkpointId,
    label: (step?.message ?? "").trim().slice(0, 60),
  };
}

/** Why an entry cannot run right now, in words. Empty string means it can. */
export function blockedReason(action: TimelineAction, ctx: TimelineCtx): string {
  if (action.needsCheckpoint && !ctx.checkpointId)
    return "No checkpoint yet. Create one first, so there is a sealed boundary to address";
  if (!action.needsCheckpoint && !ctx.atEvent)
    return "This step is not a recorded event, so there is no boundary the host can resolve";
  return "";
}

type RunState = "idle" | "pending" | "done" | "failed";

export function StateTimeline() {
  const tools = useStore((s) => s.tools);
  const sessionId = useStore((s) => s.sessionId);
  // The newest host-minted checkpoint id, folded by the store from checkpoint_created.
  const checkpointId = useStore((s) => s.lastCheckpointId);
  const [sel, setSel] = useState<number | null>(null);
  const [menu, setMenu] = useState(false);
  const [armed, setArmed] = useState<TimelineActionId | null>(null);
  const [run, setRun] = useState<{ id: TimelineActionId; state: RunState; message: string } | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);

  const steps = tools.slice(-14);
  if (steps.length === 0) return null;

  const at = sel == null ? steps.length - 1 : Math.min(sel, steps.length - 1);
  const ctx: TimelineCtx = timelineCtx(steps[at], sessionId, checkpointId);

  const fire = async (action: TimelineAction) => {
    // Destructive verbs take a second, explicit click that names the target.
    if (action.destructive && armed !== action.id) {
      setArmed(action.id);
      return;
    }
    setArmed(null);
    setMenu(false);
    setRun({ id: action.id, state: "pending", message: `${action.label}...` });
    try {
      const plan = timelinePlan(action, ctx);
      const ack = await runCommand(plan.id, plan.args);
      // A held verb (checkpoint_restore / checkpoint_rewind are ApprovalPolicy::Ask) has NOT run:
      // it stays pending until the approval gate is answered, so it must not render as done.
      const state = ackState(ack);
      setRun(
        state === "held"
          ? { id: action.id, state: "pending", message: heldNote(action.label) }
          : state === "accepted"
            ? { id: action.id, state: "done", message: `${action.label}: accepted` }
            : { id: action.id, state: "failed", message: ack.message ?? `${action.label} was refused` },
      );
    } catch (err) {
      setRun({ id: action.id, state: "failed", message: (err as Error).message });
    }
    trigger.current?.focus(); // focus returns to the control that opened the menu
  };

  return (
    <div className="statetl" role="group" aria-label="Agent state timeline">
      <span className="statetl__label" title="Pick the step the history menu addresses as its boundary.">
        state
      </span>
      <div className="statetl__dots">
        {steps.map((s, i) => (
          <button
            key={s.call_id + s.ts}
            className={["statetl__dot", i <= at && "statetl__dot--past", i === at && "statetl__dot--at"]
              .filter(Boolean)
              .join(" ")}
            title={s.message}
            aria-label={`Select step ${i + 1} of ${steps.length}, ${s.message}`}
            aria-current={i === at}
            onClick={() => setSel(i)}
          />
        ))}
      </div>
      <span className="statetl__msg">{steps[at]?.message}</span>
      {run ? (
        <span
          role="status"
          className="statetl__msg"
          style={{ flex: "0 1 auto", color: run.state === "failed" ? "var(--red)" : "var(--text-3)" }}
        >
          {run.state === "pending" ? "working: " : run.state === "failed" ? "failed: " : ""}
          {run.message}
        </span>
      ) : null}
      <div
        className="statetl__actions"
        onBlur={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
            setMenu(false);
            setArmed(null);
          }
        }}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            setMenu(false);
            setArmed(null);
          }
        }}
      >
        {sel != null ? (
          <button
            className="statetl__btn"
            type="button"
            onClick={() => setSel(null)}
            title="Return to the latest state"
            aria-label="Return to the latest state"
          >
            live
          </button>
        ) : null}
        <span className="hc__add">
          <button
            ref={trigger}
            className="statetl__btn"
            type="button"
            aria-haspopup="menu"
            aria-expanded={menu}
            aria-label={`History actions for this step${checkpointId ? `, checkpoint ${checkpointId}` : ", no checkpoint selected"}`}
            title="Checkpoint, rewind, replay, fork and compare, all from this one history menu"
            onClick={() => {
              setArmed(null);
              setMenu((v) => !v);
            }}
          >
            <Icon name="history" size={12} strokeWidth={1.6} />
            history
          </button>
          {menu ? (
            <div className="hc__addmenu" role="menu" aria-label="History actions" style={DROP_DOWN}>
              <span role="presentation" style={HEAD}>
                {checkpointId ? `checkpoint ${checkpointId}` : "no checkpoint yet"}
              </span>
              {TIMELINE_ACTIONS.map((a) => {
                const why = blockedReason(a, ctx);
                const on = !why;
                const arming = armed === a.id;
                const label = arming ? `Confirm: ${a.label}` : a.label;
                return (
                  <button
                    key={a.id}
                    className="hc__addmenu__item"
                    role="menuitem"
                    type="button"
                    disabled={!on}
                    aria-disabled={!on}
                    data-command={a.command}
                    style={{ opacity: on ? 1 : 0.4, cursor: on ? "pointer" : "default", gap: "var(--ma-4)" }}
                    title={on ? a.hint : why}
                    aria-label={
                      on
                        ? arming
                          ? `Confirm ${a.label}. ${a.hint}`
                          : `${a.label}. ${a.hint}`
                        : `${a.label}, unavailable. ${why}`
                    }
                    onClick={() => void fire(a)}
                  >
                    <span style={{ flex: 1 }}>{label}</span>
                    {a.destructive ? (
                      <span style={{ color: arming ? "var(--red)" : "var(--text-3)", fontSize: "var(--fs-small)" }}>
                        {arming ? "click again" : "reverts work"}
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </div>
          ) : null}
        </span>
      </div>
    </div>
  );
}

// The timeline sits at the TOP of the editor area, so its menu drops down, not up.
const DROP_DOWN = { bottom: "auto", top: "calc(100% + var(--ma-2))", left: "auto", right: 0, minWidth: 280 } as const;
const HEAD = {
  padding: "2px var(--ma-3)",
  color: "var(--text-3)",
  fontSize: "var(--fs-small)",
  overflow: "hidden",
  textOverflow: "ellipsis",
} as const;
