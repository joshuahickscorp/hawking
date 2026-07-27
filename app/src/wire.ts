/*
  wire.ts: TS mirrors of crates/hide-core/src/api.rs. THE contract.
  Field names, tags, and snake_case follow the Rust serde exactly. When this file and a
  doc disagree, the Rust code wins (api.rs uses #[serde(tag="type", content="data", rename_all="snake_case")]).
  Keep in lockstep with api.rs; this is the one place wire drift bites.
*/

// ids.rs newtypes all serialize as plain strings.
export type SessionId = string;
export type RunId = string;
export type EventId = string;
export type BlobId = string;

// types.rs BlobRef
export interface BlobRef {
  id: BlobId;
  hash: string;
  size_bytes: number;
  media_type: string | null;
}

// api.rs Intent  (tag = "type", content = "data")
export type Intent =
  | { type: "submit_turn"; data: { session_id: SessionId; text: string; attachments: BlobRef[] } }
  | { type: "cancel_run"; data: { run_id: RunId } }
  | { type: "pause_run"; data: { run_id: RunId } }
  | { type: "resume_run"; data: { run_id: RunId } }
  // hunk_id is additive and optional (Rust: #[serde(default)] Option<String>); absent or null means
  // the whole diff, a value targets exactly one hunk.
  | { type: "accept_diff"; data: { run_id: RunId; diff_id: string; hunk_id?: string | null } }
  | { type: "reject_diff"; data: { run_id: RunId; diff_id: string; hunk_id?: string | null } }
  | { type: "scrub_to_event"; data: { session_id: SessionId; event_id: EventId } }
  | { type: "fork_session"; data: { session_id: SessionId; at_event: EventId } }
  | { type: "open_file"; data: { path: string; line: number | null } }
  | { type: "run_command"; data: { argv: string[]; cwd: string | null } }
  | { type: "custom"; data: { name: CustomName; payload: unknown } };

// api.rs IntentAck. THREE outcomes, not two. `held` is optional on the wire (Rust #[serde(default)]),
// so an older host that never sends it reads as false, i.e. the previous two-state meaning.
export interface IntentAck {
  accepted: boolean;
  held?: boolean;
  event_seq: number | null;
  message: string | null;
}

/** The one place an ack is read as an outcome. A surface that branched on `accepted` alone showed a
 *  command PARKED at an approval gate as finished, and flipped optimistic UI as though the effect
 *  had run, so this returns the third state explicitly instead of leaving it inside `message`. */
export type AckState = "accepted" | "held" | "refused";
export const ackState = (ack: IntentAck): AckState =>
  !ack.accepted ? "refused" : ack.held ? "held" : "accepted";

/** The wording every surface shows for a held command, so "not done yet" reads the same everywhere. */
export const heldNote = (label: string): string => `${label}: held, waiting for your approval`;

// api.rs UiEventKind  (tag = "type", content = "data"; Custom is an untagged Value)
export type UiEventKind =
  | { type: "projection_patch"; data: { projection: ProjectionName; patch: unknown } }
  | { type: "token_batch"; data: { stream_id: string; text: string } }
  | { type: "runtime_status"; data: { status: RuntimeState; detail: string | null } }
  // `event_id` is the id of the RECORDED event this step is, present only when there is one
  // (streamed process output is not a recorded event). It is the id `seq_of_event` resolves, so it
  // is what a boundary verb (fork_session, checkpoint_create) must be given; `call_id` never was.
  | { type: "tool_progress"; data: { call_id: string; message: string; event_id?: string | null } }
  | { type: "security_gate"; data: { gate: string; message: string } }
  | { type: "error"; data: { code: string; message: string } }
  | { type: "custom"; data: unknown };

// api.rs UiEvent
export interface UiEvent {
  seq: number;
  session_id: SessionId | null;
  kind: UiEventKind;
}

// RuntimeSupervisor states (00-vision §3.6). status != "ready" gates the composer.
export type RuntimeState = "down" | "booting" | "ready" | "degraded" | "failed";

// The connectors (00-vision §3.5). callConnector is typed against this union. `home` serves the
// launcher's retrospective digest (pulled on connect, since it is folded from the log, not streamed).
export type ConnectorId = "runtime" | "code_index" | "context" | "personalization" | "research" | "fs" | "home";

/*
  The Custom-name registry (00-vision §3.8). Intent::Custom{name} is the escape hatch
  for every steer/observe action without a dedicated enum variant. Host and FE must agree
  on the exact string. Do NOT add a name anywhere without adding it here.

  ONE rule for this list, enforced by crates/hide-protocol (`WIRE_CUSTOM_NAMES` mirrors it byte for
  byte, and crates/hide-backend asserts every entry has an arm in `HANDLED_CUSTOM_NAMES`): a name
  lives here ONLY if `host.rs` `handle_intent` actually acts on it. A reserved-but-unhandled name is
  a control that cannot work, so the contract does not carry one.

  RETIRED by the contract-cleanup stage (16 names, none of them handled anywhere in crates/, all of
  them already removed from every surface that used to fire them):
    inline_edit, mention_in_chat, quick_fix, queue_turn, rerun_step, fleet_run,
    resolve_conflict, pin_span, unpin_span, switch_profile, switch_model, toggle_confidence,
    focus_run, dismiss, create_pr, switch_branch.
  `queue_turn` and `switch_branch` were already ordered removed by
  docs/hide-impl/consolidation/HIDE_CONSOLIDATION_DECISIONS.md section 5.
  `save_file` was on that list and is back, this time with a host arm: it is the ONE save path, and
  it exists precisely so the save stops being a raw connector write with no gate.
  RETIRED by the diff collapse: `edit_hunk`, which read only {diff_id, hunk_id} and ran the same
  host apply_hunk as `accept_diff{hunk_id}`.
  RETIRED by the reachability stage: `open_folder` and `compact_context`. Both had an EMPTY host arm
  and were listed in `HANDLED_CUSTOM_NAMES` only so the honest negative ack could not fire for them;
  the claimed downstream readers (the desktop shell re-root, the context compiler watermark gate) do
  not read either record. The workspace root is owned by app/src-tauri, and the real compaction is
  budget-driven span admission inside the context compiler (crates/hawking-context compiler.rs), not
  a watermark gate and not a request the app makes.
*/
export const CUSTOM_NAMES = [
  // AI IDE (Terminal / Search / Diff)
  "pty_input",
  "pty_resize",
  "run_search",
  "revert_diff",
  // The editor save (Editor.tsx Cmd+S). A command, not a raw connector write: the host runs it
  // through the permission-gated applier and holds a refused write at the approval gate.
  "save_file",
  // AI Chat (Composer / PlanCard)
  "redirect_run",
  "approve_plan",
  "edit_plan_step",
  "reorder_plan",
  // Notifications / any panel
  "approve_gate",
  "deny_gate",
  // Home / launcher (the courtyard front door)
  "new_session",
  "open_session",
  "create_worktree",
  // Host-handled names, host.rs refs inline in crates/hide-protocol/src/command.rs.
  "create_side_chat",
  "merge_side_chat",
  "goal_set",
  "goal_clear",
  "checkpoint_create",
  "checkpoint_restore",
  "approve_effect",
  "deny_effect",
  "skip_step",
  "repair_step",
  "checkpoint_rewind",
  "checkpoint_replay",
  "checkpoint_fork",
  "checkpoint_compare",
  "checkpoint_inspect",
  "promote_run",
  "resume_run_foreground",
  // Contract reconciliation: host.rs handle_intent dispatches these over Intent::Custom (the
  // memory / goal-eval / workspace-trust / environment arm), so their CommandSpecs bind Custom
  // and the names have to be on the wire contract for runCommand to send them.
  "memory_add",
  "memory_supersede",
  "memory_record_outcome",
  "memory_revalidate",
  "goal_evaluate",
  "workspace_set_repo_trust",
  "environment_switch",
  // The StatusBar Problems counter's producer: host.rs handle_static_analysis_intent takes
  // { sources: [{path,text}] } or { paths: [...] } over Intent::Custom, so the counter can fill
  // itself instead of only ever reading a projection nothing in this app wrote.
  "run_static_analysis",
  // The task-scoped write lease. `grant_write_lease` is approval-gated (its whole point is that a
  // human said yes to a task), so it is held at the security gate like every other Ask effect;
  // `revoke_write_lease` is a de-escalation and runs immediately, which is why it is the control the
  // StatusBar lease popover offers.
  "grant_write_lease",
  "revoke_write_lease",
  // The process controls. A client could START a sandboxed process and then had no way to attach to
  // it after navigating away, stop it, or keep its output: the host methods existed and no wire name
  // reached them. Not being able to stop what you started is the safety half of that gap.
  "attach_process",
  "stop_process",
  "capture_process_artifact",
  // The sealed diff review receipt. It had no wire name because nothing the app could do produced a
  // diff to seal; the save path records one now.
  "export_review_receipt",
] as const;
export type CustomName = (typeof CUSTOM_NAMES)[number];

/*
  ProjectionPatch{projection} discriminators (00-vision §3.8, 01-surfaces §A.4).
  The panel-slice names the FE routes on after kind === "projection_patch".
*/
export const PROJECTION_NAMES = [
  // chat
  "turn",
  "plan",
  "tool",
  "diff_chip",
  // ide
  "diff",
  "file_external",
  "editor",
  // context stack
  "context_manifest",
  "retrieval",
  "memory",
  // universal
  "timeline",
  // problems
  "build",
  "test",
  "diagnostics",
  // checkpoints
  "sourcecontrol",
  // workstation
  "fleet",
  "run",
  "merge",
  // home / launcher (the courtyard: the session list and the retrospective digest)
  "home",
  "sessions",
  // notifications
  "turn_ended",
  "plan_waiting",
  // status bar
  "status",
] as const;
export type ProjectionName = (typeof PROJECTION_NAMES)[number];

/* Convenience constructors so callers never hand-assemble the tagged shape. */
export const intent = {
  submitTurn: (session_id: SessionId, text: string, attachments: BlobRef[] = []): Intent => ({
    type: "submit_turn",
    data: { session_id, text, attachments },
  }),
  cancelRun: (run_id: RunId): Intent => ({ type: "cancel_run", data: { run_id } }),
  pauseRun: (run_id: RunId): Intent => ({ type: "pause_run", data: { run_id } }),
  resumeRun: (run_id: RunId): Intent => ({ type: "resume_run", data: { run_id } }),
  acceptDiff: (run_id: RunId, diff_id: string, hunk_id: string | null = null): Intent => ({
    type: "accept_diff",
    data: { run_id, diff_id, hunk_id },
  }),
  rejectDiff: (run_id: RunId, diff_id: string, hunk_id: string | null = null): Intent => ({
    type: "reject_diff",
    data: { run_id, diff_id, hunk_id },
  }),
  // No `scrubToEvent` builder. `Intent::ScrubToEvent` stays in the union because the router still
  // records it (hide-core api.rs, hide-backend commands.rs), but the scrub verb is retired from
  // every surface, so a constructor here was a builder for an intent nothing may send.
  forkSession: (session_id: SessionId, at_event: EventId): Intent => ({
    type: "fork_session",
    data: { session_id, at_event },
  }),
  openFile: (path: string, line: number | null = null): Intent => ({ type: "open_file", data: { path, line } }),
  runCommand: (argv: string[], cwd: string | null = null): Intent => ({ type: "run_command", data: { argv, cwd } }),
  custom: (name: CustomName, payload: unknown = {}): Intent => ({ type: "custom", data: { name, payload } }),
};
