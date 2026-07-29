/*
  wire.ts: FE contract over hide-core wire types.

  Structural types (Intent, IntentAck, UiEvent, BlobRef, ids, RuntimeState) are
  generated from crates/hide-core by app/src/generated/gen_wire.mjs into
  wire_types.generated.ts. Regenerate with `node app/src/generated/gen_wire.mjs`.

  CUSTOM_NAMES stays here: crates/hide-protocol reads this file and asserts a
  byte-order mirror against WIRE_CUSTOM_NAMES. PROJECTION_NAMES, ack helpers,
  and intent builders are FE-only and live here too.
*/

export type {
  SessionId,
  RunId,
  EventId,
  BlobId,
  BlobRef,
  IntentAck,
  UiEvent,
  RuntimeState,
} from "./generated/wire_types.generated";

import type {
  BlobRef,
  EventId,
  Intent as GenIntent,
  IntentAck,
  RunId,
  SessionId,
  UiEventKind as GenUiEventKind,
} from "./generated/wire_types.generated";

/** Ack outcome. Surfaces must not treat `accepted && held` as finished. */
export type AckState = "accepted" | "held" | "refused";
export const ackState = (ack: IntentAck): AckState =>
  !ack.accepted ? "refused" : ack.held ? "held" : "accepted";

export const heldNote = (label: string): string => `${label}: held, waiting for your approval`;

export type ConnectorId = "runtime" | "code_index" | "context" | "personalization" | "research" | "fs" | "home";

/*
  The Custom-name registry. Intent::Custom{name} is the escape hatch for steer/observe
  actions without a dedicated enum variant. Host and FE must agree on the exact string.
  hide-protocol WIRE_CUSTOM_NAMES mirrors this list (wire_custom_names_mirror_wire_ts).
*/
export const CUSTOM_NAMES = [
  "pty_input",
  "pty_resize",
  "run_search",
  "revert_diff",
  "save_file",
  "redirect_run",
  "approve_plan",
  "edit_plan_step",
  "reorder_plan",
  "approve_gate",
  "deny_gate",
  "new_session",
  "open_session",
  "create_worktree",
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
  "memory_add",
  "memory_supersede",
  "memory_record_outcome",
  "memory_revalidate",
  "goal_evaluate",
  "workspace_set_repo_trust",
  "environment_switch",
  "run_static_analysis",
  "grant_write_lease",
  "revoke_write_lease",
  "attach_process",
  "stop_process",
  "capture_process_artifact",
  "export_review_receipt",
] as const;
export type CustomName = (typeof CUSTOM_NAMES)[number];

export const PROJECTION_NAMES = [
  "turn",
  "plan",
  "tool",
  "diff_chip",
  "diff",
  "file_external",
  "editor",
  "context_manifest",
  "retrieval",
  "memory",
  "timeline",
  "build",
  "test",
  "diagnostics",
  "sourcecontrol",
  "fleet",
  "run",
  "merge",
  "home",
  "sessions",
  "turn_ended",
  "plan_waiting",
  "status",
] as const;
export type ProjectionName = (typeof PROJECTION_NAMES)[number];

/** Intent with CustomName-branded custom payloads (generated Intent uses string). */
export type Intent =
  | Exclude<GenIntent, { type: "custom" }>
  | { type: "custom"; data: { name: CustomName; payload: unknown } };

/** UiEventKind refined with ProjectionName / RuntimeState discriminators. */
export type UiEventKind =
  | { type: "projection_patch"; data: { projection: ProjectionName; patch: unknown } }
  | { type: "token_batch"; data: { stream_id: string; text: string } }
  | {
      type: "runtime_status";
      data: { status: import("./generated/wire_types.generated").RuntimeState; detail: string | null };
    }
  | { type: "tool_progress"; data: { call_id: string; message: string; event_id?: string | null } }
  | { type: "security_gate"; data: { gate: string; message: string } }
  | { type: "error"; data: { code: string; message: string } }
  | { type: "custom"; data: unknown };

// Keep GenUiEventKind referenced so drift in the generator still typechecks against use.
type _AssertKind = GenUiEventKind;

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
  forkSession: (session_id: SessionId, at_event: EventId): Intent => ({
    type: "fork_session",
    data: { session_id, at_event },
  }),
  openFile: (path: string, line: number | null = null): Intent => ({ type: "open_file", data: { path, line } }),
  runCommand: (argv: string[], cwd: string | null = null): Intent => ({ type: "run_command", data: { argv, cwd } }),
  custom: (name: CustomName, payload: unknown = {}): Intent => ({ type: "custom", data: { name, payload } }),
};
