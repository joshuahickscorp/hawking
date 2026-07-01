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
  | { type: "accept_diff"; data: { run_id: RunId; diff_id: string } }
  | { type: "reject_diff"; data: { run_id: RunId; diff_id: string } }
  | { type: "scrub_to_event"; data: { session_id: SessionId; event_id: EventId } }
  | { type: "fork_session"; data: { session_id: SessionId; at_event: EventId } }
  | { type: "open_file"; data: { path: string; line: number | null } }
  | { type: "run_command"; data: { argv: string[]; cwd: string | null } }
  | { type: "custom"; data: { name: CustomName; payload: unknown } };

// api.rs IntentAck
export interface IntentAck {
  accepted: boolean;
  event_seq: number | null;
  message: string | null;
}

// api.rs UiEventKind  (tag = "type", content = "data"; Custom is an untagged Value)
export type UiEventKind =
  | { type: "projection_patch"; data: { projection: ProjectionName; patch: unknown } }
  | { type: "token_batch"; data: { stream_id: string; text: string } }
  | { type: "runtime_status"; data: { status: RuntimeState; detail: string | null } }
  | { type: "tool_progress"; data: { call_id: string; message: string } }
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

// The five connectors (00-vision §3.5). callConnector is typed against this union.
export type ConnectorId = "runtime" | "code_index" | "context" | "personalization" | "research" | "fs";

/*
  The Custom-name registry (00-vision §3.8). Intent::Custom{name} is the escape hatch
  for every steer/observe action without a dedicated enum variant. Host and FE must agree
  on the exact string. Do NOT add a name anywhere without adding it here.
*/
export const CUSTOM_NAMES = [
  // AI IDE (Editor / Explorer / Terminal / Search / Problems / Diff)
  "save_file",
  "inline_edit",
  "mention_in_chat",
  "pty_input",
  "pty_resize",
  "run_search",
  "quick_fix",
  "revert_diff",
  "edit_hunk",
  // AI Chat (Composer / PlanCard)
  "queue_turn",
  "redirect_run",
  "approve_plan",
  "edit_plan_step",
  "reorder_plan",
  // AI Workstation (Timeline / Fleetview / Merge-review)
  "rerun_step",
  "fleet_run",
  "resolve_conflict",
  // Context Stack
  "pin_span",
  "unpin_span",
  "switch_profile",
  "switch_model",
  "toggle_confidence",
  // Notifications / any panel
  "approve_gate",
  "deny_gate",
  "focus_run",
  "dismiss",
  // Workspace / onboarding
  "open_folder",
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
  acceptDiff: (run_id: RunId, diff_id: string): Intent => ({ type: "accept_diff", data: { run_id, diff_id } }),
  rejectDiff: (run_id: RunId, diff_id: string): Intent => ({ type: "reject_diff", data: { run_id, diff_id } }),
  scrubToEvent: (session_id: SessionId, event_id: EventId): Intent => ({
    type: "scrub_to_event",
    data: { session_id, event_id },
  }),
  forkSession: (session_id: SessionId, at_event: EventId): Intent => ({
    type: "fork_session",
    data: { session_id, at_event },
  }),
  openFile: (path: string, line: number | null = null): Intent => ({ type: "open_file", data: { path, line } }),
  runCommand: (argv: string[], cwd: string | null = null): Intent => ({ type: "run_command", data: { argv, cwd } }),
  custom: (name: CustomName, payload: unknown = {}): Intent => ({ type: "custom", data: { name, payload } }),
};
