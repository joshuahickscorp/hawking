/*
  home/actions.ts: the courtyard's semantic actions, resolved through the ONE command spine.

  Every entry names a catalog command id (src/generated/command_catalog.json, mirrored from
  crates/hide-protocol command_catalog), so a chip, a menu item, a keyboard gesture and the palette
  all resolve the same CommandSpec. Nothing here invents a verb, and nothing here builds an Intent.

  Deliberately ABSENT, with the reason (docs/hide-impl/consolidation/HIDE_CONSOLIDATION_DECISIONS.md):
    steer a background run  -> already bound in the chat composer (chat/actions.ts `steer` routes the
                               same host capability over the reachable `redirect_run` custom intent,
                               Mod+/) and redirect_run addresses a run by id, so a promoted run is
                               steered there. A second control would duplicate an existing action.
    fork a background run   -> `fork_session` needs an EventId and the State Timeline is the only
                               surface holding real event ids, so fork stays there.
    switch model            -> no host model-switch capability exists (decision 3.4), so the three
                               empty-payload copies are RETIRED, not re-pointed.
    voice / transcription   -> no transcription capability in the catalog (decision 3.2), so the
                               HomeComposer mic is retired rather than kept as a recorder that
                               discards its own recording.
*/
import type { BlobRef, IntentAck } from "../../wire";
import { ackState } from "../../wire";
import type { RunPhase } from "../../store";
import type { CommandPlan } from "../contextstack/state";

/* ---- A. Goals ---------------------------------------------------------------------------------
   The durable goal domain (host goal_* over the KV store) is the ONE goal authority (decision
   section 1), so the composer keeps no private goal state beyond the text it last sent.
*/

export const goalPlan = {
  /** Bind an acceptance condition to this session. */
  set: (session_id: string, condition: string): CommandPlan => ({
    id: "goal_set",
    args: { session_id, condition },
  }),
  /** Drop it, so nothing keeps grading the run. */
  clear: (session_id: string): CommandPlan => ({ id: "goal_clear", args: { session_id } }),
  /** The deterministic acceptance check against real verification results. */
  evaluate: (session_id: string): CommandPlan => ({ id: "goal_evaluate", args: { session_id } }),
};

export const GOAL_HINT =
  "A goal is an acceptance condition. The run can stop exactly when it is met instead of drifting past done.";

/* ---- B. Attachments ---------------------------------------------------------------------------
   submit_turn has always carried an attachments field on the contract (wire.ts Intent, api.rs
   SubmitTurn) and the composer staged File objects and dropped them at submit. These fill the field.

   The contract stage threaded `attachments` through store.ts intentFor("submit_turn"), so the
   composer no longer needs its own Intent builder: it calls runCommand("submit_turn", { text,
   attachments }) like every other gesture, and `submitTurnWith` is retired.
*/

const hex = (buf: ArrayBuffer) =>
  Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

/** Real content digest of a staged file, so a BlobRef never carries an invented hash. */
export async function fileDigest(file: File): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return "";
  return `sha256:${hex(await subtle.digest("SHA-256", await file.arrayBuffer()))}`;
}

/** Staged File objects as contract BlobRefs. Empty in, empty out. */
export async function stageAttachments(files: File[]): Promise<BlobRef[]> {
  return Promise.all(
    files.map(async (f) => ({
      id: `file:${f.name}`,
      hash: await fileDigest(f),
      size_bytes: f.size,
      media_type: f.type || null,
    })),
  );
}

/* ---- C. Workspace trust -----------------------------------------------------------------------
   A repo enters the workspace graph UNTRUSTED (trust before config). While it is untrusted its
   instruction and policy refs are INERT, so adding a folder is a security decision and is never
   taken for the user.

   The trust decision is also the folder's ENTRY into the graph: `workspace_add_repo` has no wire
   name, so a trust call that carried only an id hit a repo that was never there and the host
   answered with nothing at all. The chosen folder's path rides along, and the host creates the node
   (untrusted) before applying the decision to it.
*/

export type Trust = "trusted" | "untrusted";

export const TRUST_MEANING: Record<Trust, string> = {
  untrusted: "Keep untrusted. Instruction and policy files in this folder stay inert.",
  trusted: "Trust this folder. Its instruction and policy files become active for every run here.",
};

/** The repo id the host graph keys on: the folder name, as the host records it. */
export const repoIdFor = (path: string): string => path.replace(/\/+$/, "").split("/").pop() || path;

export const trustPlan = (repo_id: string, root_path: string, trust: Trust): CommandPlan => ({
  id: "workspace_set_repo_trust",
  args: { repo_id, root_path, trust },
});

/* ---- C2. The worktree request -----------------------------------------------------------------
   `create_worktree` now has a CommandSpec and goes through runCommand, and its notice is derived
   from the ACK and never from optimism: the host records the request and then holds the unsandboxed
   `git worktree add` at an approval gate, so an accepted request is not a finished worktree.

   The notice names no branch. `spawn_worktree_add` creates a NEW branch (`hide/<slug>`) in a sibling
   directory, so the old wording ("worktree on main") described an operation the host never performs.
*/

export const worktreeNotice = (ack: IntentAck): { kind: "info" | "error"; code: string; message: string } => {
  switch (ackState(ack)) {
    case "held":
      return {
        kind: "info",
        code: "worktree",
        message: "worktree requested on a new hide/ branch, approve the gate to run it",
      };
    case "accepted":
      return { kind: "info", code: "worktree", message: "worktree requested on a new hide/ branch" };
    default:
      return { kind: "error", code: "worktree", message: ack.message ?? "worktree refused" };
  }
};

/* ---- D. Background jobs -----------------------------------------------------------------------
   A promoted run keeps running (no restart) and its control gestures route by RUN id, which is why
   pause / resume / stop reuse the existing run commands instead of inventing job verbs.
*/

export const jobPlan = {
  /** Promote the live interactive run to a durable background job. */
  promote: (run_id: string, session_id: string): CommandPlan => ({
    id: "promote_run",
    args: { run_id, session_id },
  }),
  pause: (run_id: string): CommandPlan => ({ id: "pause_run", args: { run_id } }),
  resume: (run_id: string): CommandPlan => ({ id: "resume_run", args: { run_id } }),
  stop: (run_id: string): CommandPlan => ({ id: "cancel_run", args: { run_id } }),
  /** Reattach a promoted job to this window and continue it in the foreground. */
  foreground: (job_id: string): CommandPlan => ({ id: "resume_run_foreground", args: { job_id } }),
};

/**
 * The job lifecycle events the host publishes (host.rs publish_job). They arrive as Custom UiEvents,
 * and store.ts has no job slice, so they land in the notices strip as the first 200 characters of
 * their JSON. `job_id` is the record's first field and `kind` is the envelope's, so both survive the
 * truncation and can be read back exactly.
 *
 * ponytail: regex over a truncated notice, because the FE has no job slice. Upgrade path: a `jobs`
 * slice in store.ts folded from these Custom events, then this reader deletes.
 */
export const JOB_EVENT_LABEL: Record<string, string> = {
  job_created: "queued",
  job_promoted: "running in the background",
  job_status: "status changed",
  job_cancelled: "stopped",
  job_resumed_foreground: "back in the foreground",
};

export interface JobSighting {
  jobId: string;
  event: string;
  label: string;
}

const JOB_EVENT_RE = /"kind"\s*:\s*"(job_[a-z_]+)"/;
const JOB_ID_RE = /"job_id"\s*:\s*"([^"]+)"/;

export function readJobNotice(message: string): JobSighting | null {
  const ev = JOB_EVENT_RE.exec(message);
  const id = JOB_ID_RE.exec(message);
  if (!ev || !id) return null;
  return { jobId: id[1], event: ev[1], label: JOB_EVENT_LABEL[ev[1]] ?? ev[1] };
}

/** The environment a switch actually landed on, read back from the same Custom notice path. */
const ENV_RE = /"kind"\s*:\s*"environment_switch"[\s\S]*?"new_env"\s*:\s*"([^"]+)"/;
export function readEnvironmentNotice(message: string): string | null {
  return ENV_RE.exec(message)?.[1] ?? null;
}

export type JobPhase = "idle" | "pending" | "active" | "paused" | "blocked" | "failed" | "done";

/** Phase from real store state. An open security gate outranks the run phase: it is what blocks. */
export function jobPhase(runPhase: RunPhase, blocked: boolean): JobPhase {
  if (blocked) return "blocked";
  switch (runPhase) {
    case "planning":
      return "pending";
    case "executing":
      return "active";
    case "paused":
      return "paused";
    case "awaiting":
      return "blocked";
    case "done":
      return "done";
    case "failed":
      return "failed";
    default:
      return "idle";
  }
}

export const JOB_PHASE_LABEL: Record<JobPhase, string> = {
  idle: "idle",
  pending: "planning",
  active: "running",
  paused: "paused",
  blocked: "waiting on you",
  failed: "failed",
  done: "done",
};

/** Shape, never colour: each phase reads differently with styles off. */
export const JOB_PHASE_GLYPH: Record<JobPhase, string> = {
  idle: "..",
  pending: "o",
  active: ">>",
  paused: "||",
  blocked: "!",
  failed: "x",
  done: "ok",
};

export interface JobView {
  phase: JobPhase;
  /** The durable job id, once the host has minted one. Null while the run is still foreground only. */
  jobId: string | null;
  /** The last job lifecycle event, in words. */
  jobEvent: string | null;
  /** An approval the run is blocked on. */
  approval: string | null;
  /** The current verification read (the same diagnostics projection the status bar binds). */
  verification: string | null;
  /** The newest process / tool line, so the row says what the run is actually doing. */
  process: string | null;
}

/** The whole row in words. Nothing in this row is carried by colour alone. */
export function jobLabel(v: JobView): string {
  const parts = [`Background run, ${JOB_PHASE_LABEL[v.phase]}`];
  if (v.jobId) parts.push(`job ${v.jobId}${v.jobEvent ? `, ${v.jobEvent}` : ""}`);
  if (v.approval) parts.push(`approval needed, ${v.approval}`);
  if (v.verification) parts.push(`verification, ${v.verification}`);
  if (v.process) parts.push(`last step, ${v.process}`);
  return parts.join(". ");
}

/** Whether a control can fire right now. Offered but dead is worse than visibly unavailable. */
export function jobActionEnabled(
  id: "promote" | "pause" | "resume" | "stop" | "foreground",
  v: JobView,
  hasRun: boolean,
): boolean {
  switch (id) {
    case "promote":
      return hasRun && v.phase !== "done" && v.phase !== "failed" && !v.jobId;
    case "pause":
      return hasRun && (v.phase === "active" || v.phase === "pending");
    case "resume":
      return hasRun && v.phase === "paused";
    case "stop":
      return hasRun && v.phase !== "done" && v.phase !== "failed";
    case "foreground":
      return !!v.jobId;
  }
}

/* ---- E. Environment + the workspace graph read ------------------------------------------------
   environment_switch keeps the session and its log and re-scopes fs roots and tool permissions. The
   target must already exist in the host workspace graph, and no projection enumerates it, so the
   field is an id the user names rather than a list this app cannot honestly populate.
*/

export const ENVIRONMENT_NOTE =
  "An environment is a node in the host workspace graph. Switching keeps this session and its history, and re-scopes file roots and tool permissions.";

export const environmentPlan = (
  session_id: string,
  env_id: string,
  reason = "switched from settings",
): CommandPlan => ({ id: "environment_switch", args: { session_id, env_id, reason } });

export interface WorkspaceRead {
  root?: string;
  repo?: string;
  branch?: string;
  worktrees?: string[];
}

/** The multi-repo workspace read the host actually reports, as label / value rows. */
export function workspaceRows(ws: WorkspaceRead | undefined): [string, string][] {
  const rows: [string, string][] = [
    ["root", ws?.root || "no folder opened"],
    ["repo", ws?.repo || "none"],
    ["branch", ws?.branch || "no branch"],
  ];
  const wt = ws?.worktrees ?? [];
  rows.push(["worktrees", wt.length ? wt.join(", ") : "none"]);
  return rows;
}
