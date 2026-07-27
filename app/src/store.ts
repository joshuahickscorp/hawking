/*
  store.ts: one coherent Zustand store, the derived cache of the UiEvent stream.
  The host owns truth; this store is a fold of the event log (constitution principle 3).
  The only outbound mutation channel is an Intent (sent via ipc); there is no third path. The
  connector route is a READ channel and the host enforces that with an ALLOWLIST of the read methods
  (connectors.rs CONNECTOR_READ_METHODS), so anything that mutates or escapes the workspace is
  refused there and has to arrive as an Intent, where the approval gate is
  (crates/hide-serve/src/lib.rs post_connector).

  Slices follow 01-surfaces §A.4 names. Each tracks lastAppliedSeq so reconnect replays cleanly
  (catch up via GET ?after_seq=lastAppliedSeq, then resume live). The EventRouter dispatches each
  UiEventKind to its slice. Every failure is surfaced into notifyStore, never swallowed.
*/
import { create } from "zustand";
import { callConnector, sendIntent, subscribeUi, TRANSPORT_KIND } from "./ipc";
import { ackState, CUSTOM_NAMES, intent } from "./wire";
import type {
  BlobRef,
  CustomName,
  Intent,
  IntentAck,
  ProjectionName,
  RuntimeState,
  UiEvent,
} from "./wire";
import catalogJson from "./generated/command_catalog.json";

// ---- Slice shapes (kept flat and minimal; surfaces extend their own slice in the next pass) ----

export type RunPhase = "idle" | "planning" | "executing" | "paused" | "awaiting" | "done" | "failed";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  streaming: boolean;
}

export interface FleetRun {
  id: string;
  objective: string;
  state: "active" | "waiting" | "done" | "failed";
  step: number;
  steps: number;
}

// Home / launcher slices (the courtyard front door). A session summary is one row in the recents
// rail; the digest is the retrospective activity read (what happened, never a budget cap).
export interface SessionSummary {
  id: string;
  title: string;
  state: "active" | "idle" | "done" | "failed";
  updated_ms: number;
  turns?: number;
  branch?: string; // git branch or worktree the session runs on
}

export interface HomeDigest {
  sessions: number;
  messages: number;
  tokens?: number; // omitted by the live engine (never persisted to the log); the launcher hides it then
  active_days: number;
  streak_current: number;
  streak_longest: number;
  peak_hour: number; // 0..23
  favorite_model: string;
  heatmap?: number[]; // flat activity counts, row-major (cols x 7), most-recent column last
  heatmap_cols?: number;
}

// The workspace the launcher targets: root folder, repo, current branch, and any worktrees. Drives
// the composer's context chips (Local . repo . branch . worktree).
export interface HomeWorkspace {
  root?: string;
  repo?: string;
  branch?: string;
  worktrees?: string[];
}

export interface HomeState {
  user?: { name?: string; plan?: string };
  workspace?: HomeWorkspace;
  digest?: HomeDigest;
}

export interface ContextManifest {
  model?: { id: string; arch: string; ctx: number; profile: string; sampling: string };
  budget?: { total: number; used: number; free: number; segments: { source: string; tokens: number }[] };
  retrieved?: { path: string; range: string; relevance: number }[];
  tools?: { name: string; ok: boolean }[];
  memory?: { fact: string; confidence: number }[];
  dropped?: { title: string; would_be_tokens: number; reason: string }[];
  // Spine A: the live, measured context picture from the engine (host emits this as
  // projection_patch{context_manifest}). The ceiling is native x the measured .tq
  // multiplier, never a constant; `live` is the dynamic occupancy/recall reading.
  arch?: string;
  ctx_len_native?: number;
  ctx_len_effective?: number;
  tq_multiplier?: number;
  tq_estimated?: boolean;
  recurrent_state_bytes?: number;
  live?: {
    effective_ceiling_tokens: number;
    used_tokens_estimate?: number;
    occupancy: number;
    recall_fidelity?: number;
    watermark: "normal" | "soft" | "warn" | "critical";
    estimated?: boolean;
  };
}

export interface ToolEvent {
  call_id: string;
  message: string;
  ts: number;
  /** The RECORDED event this step is, when there is one. This, not `call_id`, is the id the host
   *  resolves (replay.rs seq_of_event), so it is the only thing a boundary verb may be given. */
  event_id?: string;
}

export interface Notice {
  id: string;
  kind: "error" | "info";
  code: string;
  message: string;
}

export interface SecurityGate {
  gate: string;
  message: string;
  /** Set when the pause is a paused EFFECTFUL STEP of the running turn (the host's
   *  `approval_requested` Custom event) rather than a held shell command. Same slice, same two
   *  handlers, same two presentations; only the intent the decision dispatches differs
   *  (`approve_effect`/`deny_effect` instead of `approve_gate`/`deny_gate`). Without this the
   *  shipped SuggestOnly autonomy deadlocked: the turn paused and no surface could answer. */
  effect?: { run_id: string; step_id: string };
  /** A decision is in flight. The gate stays up until the host records it, so this is what keeps a
   *  second press from sending a second decision. */
  deciding?: boolean;
}

// A patch is a shallow merge by convention (the host emits state-diffs). Unknown projections
// land in a generic bag so nothing is lost before its panel exists.
type Patch = Record<string, unknown>;

interface State {
  // chat / run (Chat surface)
  sessionId: string; // active session, tracked from UiEvent.session_id (not hardcoded per surface)
  messages: ChatMessage[];
  streams: Record<string, string>; // stream_id -> active assistant message id
  runPhase: RunPhase;
  activeRunId: string | null;
  chatSeq: number;

  // runtime (status bar pill + banner)
  runtimeStatus: RuntimeState;
  runtimeDetail: string | null;
  runtimeSeq: number;

  // tool feed (Timeline)
  tools: ToolEvent[];
  toolSeq: number;

  // context stack (the differentiator)
  manifest: ContextManifest | null;
  manifestRing: Record<number, ContextManifest>; // keyed by seq for scrub coupling (FE-8)
  contextSeq: number;

  // workstation fleet
  fleet: FleetRun[];
  fleetSeq: number;

  // home / launcher (the courtyard front door)
  home: HomeState | null;
  homeSeq: number;
  sessions: SessionSummary[];
  sessionsSeq: number;

  // editor / diff / timeline / problems / generic projections (stubs that hold real patches)
  projections: Partial<Record<ProjectionName, Patch>>;
  projectionSeq: number;

  // The last host-minted ids carried by a Custom UiEvent (checkpoint_created, session_forked,
  // side_chat_created). Menus address a REAL record with these instead of guessing, and a landed
  // action is confirmed by the id changing. Not a full record slice on purpose.
  lastCheckpointId: string | null;
  lastForkedSession: string | null;
  lastSideChat: string | null;
  /** The session the side chat BRANCHED FROM. The `side_chat_created` event arrives under the new
   *  session id and the store adopts it as active, so `sessionId` is the side chat by the time a
   *  merge is offered: merging against it sent the summary to the side chat itself. The record
   *  carries the real parent, so the merge addresses that. */
  lastSideChatParent: string | null;

  // security gate (blocking, never auto-dismissed: FE-5)
  gate: SecurityGate | null;
  /** The gates raised while `gate` was already up, oldest first. There is one overlay, so a second
   *  gate used to overwrite the first and its id became unreachable from every surface: the parked
   *  effect could then never be approved or denied. Two saves under the shipped write policy is
   *  enough to reach it. They queue instead, and the next one opens when this one is decided. */
  gateQueue: SecurityGate[];

  // notifications (errors + surfaced transport failures)
  notices: Notice[];

  // ---- actions (all internal; user actions go out as Intents elsewhere) ----
  apply(ev: UiEvent): void;
  pushNotice(n: Omit<Notice, "id">): void;
  dismissNotice(id: string): void;
  // RETIRED: dismissGate(). It cleared the gate with no decision recorded, had no caller, and was
  // exactly the shape the Escape defect took. A gate leaves the screen one way now: decided.
  // The ONE pair of gate handlers. Both presentations (the shell overlay and the inline capsule)
  // call these, so an approve can never mean two different things. Both are one line over
  // decideGate, which is where the recorded-before-closed rule lives.
  approveGate(): void;
  denyGate(): void;
  decideGate(approve: boolean): void;
  pushUserMessage(text: string): void;
  // Launching a fresh session from the courtyard: clear the local transcript optimistically so the
  // conversation view is empty while the host mints the real session_id (arrives on the event stream).
  startNewSession(): void;
  // Reopening a recorded session. Lives here so the recents rail and the palette open a session the
  // same way instead of each keeping a copy of the mock/live branch.
  openSession(id: string): void;
}

let _id = 0;
const nextId = () => `m${++_id}`;

/** Raise a gate against the ONE overlay. Already showing one, so this one waits its turn instead of
 *  replacing it: the replaced gate's id lived nowhere else, so its parked effect could never be
 *  approved or denied by anything afterwards. A gate id already on screen or already queued is
 *  ignored, because the reconnect catch-up replays the same `approval_requested` it already sent. */
const raiseGate = (s: State, g: SecurityGate): Partial<State> => {
  if (!s.gate) return { gate: g };
  if (s.gate.gate === g.gate || s.gateQueue.some((q) => q.gate === g.gate)) return {};
  return { gateQueue: [...s.gateQueue, g] };
};

export const useStore = create<State>((set, get) => ({
  sessionId: "ses_local000000000000000000",
  messages: [],
  streams: {},
  runPhase: "idle",
  activeRunId: null,
  chatSeq: 0,

  runtimeStatus: "down",
  runtimeDetail: null,
  runtimeSeq: 0,

  tools: [],
  toolSeq: 0,

  manifest: null,
  manifestRing: {},
  contextSeq: 0,

  fleet: [],
  fleetSeq: 0,

  home: null,
  homeSeq: 0,
  sessions: [],
  sessionsSeq: 0,

  projections: {},
  projectionSeq: 0,

  lastCheckpointId: null,
  lastForkedSession: null,
  lastSideChat: null,
  lastSideChatParent: null,

  gate: null,
  gateQueue: [],
  notices: [],

  pushUserMessage: (text) =>
    set((s) => ({ messages: [...s.messages, { id: nextId(), role: "user", text, streaming: false }] })),

  pushNotice: (n) => {
    const id = nextId();
    set((s) => ({ notices: [...s.notices.slice(-19), { ...n, id }] }));
    // Auto-expire so a transient error never lives forever in the status bar; errors linger a little
    // longer than info so they are not missed. Cleared early if the user acts or a newer notice lands.
    setTimeout(() => get().dismissNotice(id), n.kind === "error" ? 8000 : 4000);
  },
  dismissNotice: (id) => set((s) => ({ notices: s.notices.filter((x) => x.id !== id) })),

  // Approve tells the engine the gate is cleared (it releases and runs the held command); deny tells
  // it to drop the held command. Both carry the gate id the SecurityGate was emitted with. When the
  // pause is an effectful STEP rather than a held command, the same decision goes out as
  // approve_effect / deny_effect against the run and step the host named.
  //
  // The gate closes only once the host has RECORDED the decision. It used to clear synchronously,
  // before and regardless of what the dispatch returned, so a refused or dropped decision dismissed
  // the app's one security-facing control while the effect stayed parked with nothing left to
  // answer it. A decision that did not land keeps the prompt up and says why.
  approveGate: () => get().decideGate(true),
  denyGate: () => get().decideGate(false),
  decideGate: (approve) => {
    const g = get().gate;
    if (!g || g.deciding) return; // one decision in flight, so a second press cannot send a second
    const effect = g.effect;
    const name = effect ? (approve ? "approve_effect" : "deny_effect") : approve ? "approve_gate" : "deny_gate";
    set({ gate: { ...g, deciding: true } });
    // `still` guards the late answer: a newer gate may have arrived while this one was in flight,
    // and neither closing it nor putting the old one back may clobber it.
    const still = () => get().gate?.gate === g.gate;
    const reopen = (message: string) => {
      if (still()) set({ gate: { ...g, deciding: false } });
      get().pushNotice({ kind: "error", code: "gate", message });
    };
    void runCommand(name, effect ?? { gate: g.gate })
      .then((ack) => {
        if (ackState(ack) !== "accepted") return reopen(ack.message ?? "the host did not record that decision");
        // Decided, so the next queued gate takes the overlay rather than being dropped.
        if (still()) set((s) => ({ gate: s.gateQueue[0] ?? null, gateQueue: s.gateQueue.slice(1) }));
      })
      .catch((e) => reopen(e instanceof Error ? e.message : String(e)));
  },

  // The store is the fold of ONE session, so the tool feed goes with the transcript. It used to
  // survive the switch, and the timeline then offered boundary dots belonging to a session the
  // history verbs would not be sent with.
  startNewSession: () =>
    set({ messages: [], streams: {}, tools: [], runPhase: "idle", activeRunId: null }),

  // Open a recent: the conversation loads in place. On mock there is no host to rebuild it, so the
  // session's task is replayed as a live exchange and the demo is a working chat.
  openSession: (id) => {
    const { sessions, sessionId, startNewSession, pushUserMessage } = get();
    startNewSession();
    if (TRANSPORT_KIND === "mock") {
      const task = sessions.find((s) => s.id === id)?.title ?? "continue the session";
      pushUserMessage(task);
      void runCommand("submit_turn", { session_id: sessionId, text: task }).catch(noticeFailure("session"));
    } else {
      void runCommand("open_session", { session_id: id }).catch(noticeFailure("session"));
    }
  },

  // THE EventRouter: route by kind, then for projection_patch by projection name.
  apply: (ev) => {
    const k = ev.kind;
    if (ev.session_id) set({ sessionId: ev.session_id }); // track the active session for fork/scrub/turn
    switch (k.type) {
      case "token_batch":
        set((s) => appendStream(s, k.data.stream_id, k.data.text, ev.seq));
        break;

      case "runtime_status":
        set({ runtimeStatus: k.data.status, runtimeDetail: k.data.detail, runtimeSeq: ev.seq });
        break;

      case "tool_progress":
        set((s) => ({
          tools: [
            ...s.tools.slice(-49),
            {
              call_id: k.data.call_id,
              message: k.data.message,
              ts: Date.now(),
              event_id: k.data.event_id ?? undefined,
            },
          ],
          toolSeq: ev.seq,
        }));
        break;

      case "security_gate":
        set((s) => raiseGate(s, { gate: k.data.gate, message: k.data.message }));
        break;

      case "error":
        get().pushNotice({ kind: "error", code: k.data.code, message: k.data.message });
        break;

      case "projection_patch":
        set((s) => routeProjection(s, k.data.projection, k.data.patch as Patch, ev.seq));
        break;

      case "custom": {
        // Custom events route by the `kind` discriminator inside the value. The three record kinds
        // that mint an id a menu needs to address keep that id here; everything else (and the record
        // detail) still surfaces as info so nothing is silently dropped.
        const c = k.data as {
          kind?: string;
          record?: { checkpoint_id?: string; session_id?: string; parent_session_id?: string };
          run_id?: string;
          step_id?: string;
          summary?: string;
          effects?: string[];
          event_id?: string;
          role?: string;
          text?: string;
        };
        const rec = c.record ?? {};
        // A paused effectful step (host announce_approval_request). It raises the SAME gate the
        // security overlay and the inline capsule already render, tagged with the run and step the
        // decision has to name, so approve/deny reach the host's ApprovalHub and the turn resumes.
        if (c.kind === "approval_requested" && c.run_id && c.step_id)
          set((s) =>
            raiseGate(s, {
              gate: c.step_id!,
              message: approvalMessage(c.summary, c.effects),
              effect: { run_id: c.run_id!, step_id: c.step_id! },
            }),
          );
        // A replayed transcript line (replay.rs `transcript_message`, from the durable
        // `user.intent.submit_turn` / `agent.message` pair). This is what makes a recorded session
        // RENDER: open_session and the reconnect catch-up both replay these, and with no arm here
        // they became 200-char JSON blobs in the status bar while the conversation stayed empty.
        // Deduped on the durable event id, since a line already on screen may replay.
        if (c.kind === "transcript_message" && typeof c.text === "string" && c.text) {
          const mid = `ev_${c.event_id ?? c.text}`;
          set((s) =>
            s.messages.some((m) => m.id === mid)
              ? {}
              : {
                  messages: [
                    ...s.messages,
                    { id: mid, role: c.role === "user" ? "user" : "assistant", text: c.text!, streaming: false },
                  ],
                  chatSeq: ev.seq,
                },
          );
          break; // routed, so it is not also a status-bar notice
        }
        if (c.kind === "checkpoint_created" && rec.checkpoint_id) set({ lastCheckpointId: rec.checkpoint_id });
        if (c.kind === "session_forked" && rec.session_id) set({ lastForkedSession: rec.session_id });
        // The parent rides the same record (services.rs SessionRecord.parent_session_id) and is kept
        // with the branch id: this event switches the app's active session TO the side chat, so the
        // merge would otherwise name the side chat as its own parent.
        if (c.kind === "side_chat_created" && rec.session_id)
          set({ lastSideChat: rec.session_id, lastSideChatParent: rec.parent_session_id ?? null });
        // Anything still unrouted gets a READABLE line, never a raw payload.
        //
        // This used to be `JSON.stringify(k.data).slice(0, 200)`, guarded by a single exclusion for
        // `search_results`. An exclusion list of one is not a rule: the moment another kind started
        // flowing (open_session, once the transcript route landed) the status bar filled with
        // `{"event_id":"evt_...","kind":"user.intent.custom.open_session","payload":{...` again.
        // Two rules replace it, both about the class rather than the next offender.
        //
        // 1. `user.intent.*` is the user's OWN action replayed back at them. The durable log echoes
        //    every intent, and the catch-up replays the lot on connect, so noticing them turns a
        //    reconnect into a wall of the user's own history. Silent.
        // 2. Everything else names the event and drops the payload. Nothing is silently swallowed
        //    (the kind still reaches the notice area) but no user is ever shown serialized JSON.
        const routed = c.kind === "search_results" || c.kind === "approval_requested";
        const replayedOwnAction = (c.kind ?? "").startsWith("user.intent.");
        if (!routed && !replayedOwnAction) {
          const label = (c.kind ?? "event").replace(/[._]/g, " ");
          get().pushNotice({ kind: "info", code: "custom", message: label });
        }
        break;
      }
    }
  },
}));

/** What the approval prompt says: the step's own summary plus the effects it declared, so the
 *  operator approves a named action rather than a bare "approve?". */
export function approvalMessage(summary: string | undefined, effects: string[] | undefined): string {
  const what = (summary ?? "").trim() || "a step in this turn";
  const fx = (effects ?? []).filter(Boolean);
  return fx.length ? `${what} (${fx.join(", ")})` : what;
}

// Append coalesced tokens to the open assistant message for a stream_id (creating it on first token).
function appendStream(s: State, streamId: string, text: string, seq: number): Partial<State> {
  let msgId = s.streams[streamId];
  let messages = s.messages;
  if (!msgId) {
    msgId = nextId();
    messages = [...messages, { id: msgId, role: "assistant", text: "", streaming: true }];
  }
  messages = messages.map((m) => (m.id === msgId ? { ...m, text: m.text + text } : m));
  return { messages, streams: { ...s.streams, [streamId]: msgId }, chatSeq: seq, runPhase: "executing" };
}

// Route a projection patch to its owning slice. Known projections update typed slices;
// the rest merge into the generic projections bag so a future panel finds its state already folded.
function routeProjection(s: State, projection: ProjectionName, patch: Patch, seq: number): Partial<State> {
  switch (projection) {
    case "context_manifest":
    case "retrieval":
    case "memory": {
      const manifest = { ...(s.manifest ?? {}), ...patch } as ContextManifest;
      return { manifest, manifestRing: { ...s.manifestRing, [seq]: manifest }, contextSeq: seq };
    }
    case "fleet": {
      const runs = Array.isArray((patch as { runs?: unknown }).runs)
        ? ((patch as { runs: FleetRun[] }).runs)
        : s.fleet;
      return { fleet: runs, fleetSeq: seq };
    }
    case "home": {
      // Shallow-merge, but merge nested workspace/digest so a partial patch (a new branch, a fresh
      // digest) never wipes the other half of the courtyard read.
      const prev = s.home ?? {};
      const p = patch as HomeState;
      const home: HomeState = {
        ...prev,
        ...p,
        workspace: p.workspace ? { ...(prev.workspace ?? {}), ...p.workspace } : prev.workspace,
        digest: p.digest ? { ...(prev.digest ?? {}), ...p.digest } : prev.digest,
        user: p.user ? { ...(prev.user ?? {}), ...p.user } : prev.user,
      };
      return { home, homeSeq: seq };
    }
    case "sessions": {
      const items = Array.isArray((patch as { items?: unknown }).items)
        ? (patch as { items: SessionSummary[] }).items
        : s.sessions;
      return { sessions: items, sessionsSeq: seq };
    }
    case "turn": {
      const phase = (patch as { phase?: RunPhase }).phase;
      const runId = (patch as { run_id?: string }).run_id ?? s.activeRunId;
      const finalizing = phase === "done" || phase === "failed";
      return {
        runPhase: phase ?? s.runPhase,
        activeRunId: runId,
        chatSeq: seq,
        // close any open streaming message when the turn ends
        messages: finalizing ? s.messages.map((m) => ({ ...m, streaming: false })) : s.messages,
        streams: finalizing ? {} : s.streams,
      };
    }
    default:
      return {
        projections: { ...s.projections, [projection]: { ...(s.projections[projection] ?? {}), ...patch } },
        projectionSeq: seq,
      };
  }
}

/* ---- The ONE command spine ------------------------------------------------------------------
   crates/hide-protocol command_catalog() is the authority; src/generated/command_catalog.json is a
   byte-identical copy of its golden (drift fails src/generated/catalog.test.ts). Buttons, keyboard
   shortcuts, context menus, and the palette all resolve the SAME id here and dispatch through the
   SAME runCommand, so no surface can re-declare its own binding.
*/

/** The fields of the generated CommandSpec the FE reads (see crates/hide-sdk/goldens/commands.d.ts). */
export interface CommandSpec {
  id: string;
  title: string;
  description: string;
  category: string;
  primary_surface: string;
  available_surfaces: string[];
  required_selection: "none" | "text" | "file" | "hunk" | "plan_step" | "any";
  keyboard_shortcut: string | null;
  command_palette: boolean;
  context_menu: boolean;
  toolbar_binding: string | null;
  backend_binding:
    | { kind: "intent"; target: string }
    | { kind: "custom"; target: string }
    | { kind: "rpc"; target: string }
    | { kind: "local_only" };
}

export const COMMANDS = catalogJson as unknown as CommandSpec[];
export const commandById = (id: string): CommandSpec | undefined => COMMANDS.find((c) => c.id === id);

export type CommandArgs = Record<string, unknown>;

/** Fields no store slice holds, so the caller must supply them (a palette entry cannot invent a diff
 *  id or a file path). Everything else fills from live state.
 *
 *  The second block is the SAME rule applied to the Custom bindings: each entry is a field the host
 *  refuses without (every `missing("...")` arm in crates/hide-backend/src/host.rs). It used to cover
 *  only the seven Intent-bound ids, which is why the palette offered a row per checkpoint verb that
 *  could never carry a checkpoint id. `session_id` is never listed: runCommand fills it below. The
 *  one exception is `open_session`, where the session id IS the argument, so the fill-in would
 *  otherwise turn "open that session" into "reopen the one already open". */
const REQUIRED_ARGS: Record<string, string[]> = {
  submit_turn: ["text"],
  accept_diff: ["diff_id"],
  reject_diff: ["diff_id"],
  fork_session: ["at_event"],
  open_file: ["path"],
  run_command: ["argv"],

  steer: ["run_id", "text"],
  merge_side_chat: ["side_chat", "parent", "summary"],
  goal_set: ["condition"],
  checkpoint_restore: ["checkpoint_id"],
  checkpoint_rewind: ["checkpoint_id", "target"],
  checkpoint_replay: ["checkpoint_id"],
  checkpoint_fork: ["checkpoint_id"],
  checkpoint_compare: ["checkpoint_id"],
  checkpoint_inspect: ["checkpoint_id"],
  memory_add: ["claim"],
  memory_supersede: ["old_id", "replacement"],
  memory_record_outcome: ["memory_id", "success"],
  // The host also accepts a bare `memory_id`; no surface offers that form, and the Context Stack
  // sends a scope.
  memory_revalidate: ["scope"],
  workspace_set_repo_trust: ["repo_id", "trust"],
  environment_switch: ["env_id"],
  edit_plan_step: ["step_id", "text"],
  skip_step: ["step_id"],
  repair_step: ["step_id"],
  reorder_plan: ["order"],
  promote_run: ["run_id"],
  resume_run_foreground: ["job_id"],
  pty_input: ["data"],
  pty_resize: ["cols", "rows"],
  run_search: ["query"],
  revert_diff: ["diff_id"],
  // The buffer being written. Both come from the editor, which is why this one is surface-owned
  // rather than a palette row.
  save_file: ["path", "content"],
  // The names the contract-cleanup stage gave a CommandSpec that address a LIVE object (a gate, a
  // paused step, a session): they carry that id or they cannot work, which is also what keeps them
  // out of the palette. `create_worktree` needs nothing, so it is not listed. `compact_context` and
  // `open_folder` were on this list and are retired: neither has a catalog row, a wire name or a
  // host arm any more.
  approve_gate: ["gate"],
  deny_gate: ["gate"],
  approve_effect: ["run_id"],
  deny_effect: ["run_id"],
  open_session: ["session_id"],
  // The host arm refuses an empty payload ("give 'sources' or 'paths'"). This app has file paths,
  // not editor buffers, so `paths` is the form it sends and the one the argument rule guards.
  run_static_analysis: ["paths"],
  // The lease names the repo it is scoped to, which is also the trust decision the host re-reads
  // before granting. It carries an argument, so it is not a bare palette gesture; `revoke` needs
  // nothing and stays offered everywhere.
  grant_write_lease: ["repo_id"],
  // The process controls each address ONE named process: a bare gesture cannot pick which, and
  // guessing "the latest" would stop the wrong one.
  attach_process: ["process"],
  stop_process: ["process"],
  capture_process_artifact: ["process"],
  // The diff being sealed.
  export_review_receipt: ["diff_id"],
};

/** Commands that act on a live run; without one there is nothing to address. */
const RUN_SCOPED = ["cancel_run", "pause_run", "resume_run", "accept_diff", "reject_diff"];

// The FE posts /v1/hide/intent and nothing else, so only Intent and Custom bindings are reachable.
const dispatchable = (c: CommandSpec) =>
  c.backend_binding.kind === "intent" || c.backend_binding.kind === "custom";

// Invocable from a bare gesture: no selection to satisfy and no argument to supply.
const selfContained = (c: CommandSpec) =>
  c.required_selection === "none" && !REQUIRED_ARGS[c.id]?.length;

/** Palette entries, derived from the catalog. Commands whose selection or arguments cannot be
 *  satisfied from a bare palette gesture are filtered out rather than offered broken. */
export function paletteCommands(): CommandSpec[] {
  return COMMANDS.filter((c) => c.command_palette && dispatchable(c) && selfContained(c));
}

/** A chord the COMPOSER itself binds on its textarea (Chat.tsx: Mod+Enter starts a turn, Mod+/
 *  steers), so the shell must not rebind it and the two cannot fight. A button binding such as
 *  `chat.new` owns no chord at all, which is how Mod+Shift+N ended up advertised in two menus and
 *  bound nowhere. */
const composerOwned = (c: CommandSpec) => (c.toolbar_binding ?? "").startsWith("composer.");

/** Shell-level keyboard bindings, derived from the catalog. Surface-owned chords stay with their
 *  surface (the composer above; the diff review binds accept/reject only while it holds focus). */
export function shortcutCommands(): CommandSpec[] {
  return COMMANDS.filter(
    (c) => c.keyboard_shortcut && dispatchable(c) && selfContained(c) && !composerOwned(c),
  );
}

/** Local-only shell commands: pure FE view state, no backend binding. Declared here so the ONE spine
 *  owns every binding (App.tsx supplies the handlers) and the collision check sees them. */
export const SHELL_COMMANDS: { id: string; title: string; shortcut?: string }[] = [
  { id: "go.chat", title: "Go to Chat" },
  { id: "go.code", title: "Go to Code" },
  { id: "toggle.chat", title: "Toggle Executor", shortcut: "Mod+I" },
  { id: "toggle.float", title: "Executor: Float / Dock" },
  { id: "toggle.panel", title: "Toggle Terminal", shortcut: "Mod+J" },
  { id: "toggle.sidebar", title: "Toggle Navigator", shortcut: "Mod+B" },
  { id: "toggle.palette", title: "Command Palette", shortcut: "Mod+P" },
  { id: "open.settings", title: "Settings", shortcut: "Mod+," },
  // The six conversation side panels. They were mouse-only icon buttons on the Chat stage, which is
  // the one place a control can hide from the keyboard entirely. No chords: six more would collide
  // with the editor's, and the palette is already a keyboard path.
  { id: "panel.terminal", title: "Panel: Terminal" },
  { id: "panel.diff", title: "Panel: Diff" },
  { id: "panel.preview", title: "Panel: Preview" },
  { id: "panel.tools", title: "Panel: Tools" },
  { id: "panel.artifacts", title: "Panel: Artifacts" },
  { id: "panel.context", title: "Panel: Context Stack" },
  // The permission mode governs the security gate, so it is a named command per mode (not a blind
  // cycle) and each row says what it does. Bypass auto-approves every gated command.
  { id: "perm.ask", title: "Permissions: ask before each gated step" },
  { id: "perm.bypass", title: "Permissions: bypass, auto-approve every gate" },
];

/** Every shell-bound shortcut, local plus catalog, with the label of the command that owns it. THE
 *  keyboard map: Settings renders this table and the palette looks its rows up in it, so no surface
 *  hand-writes a second one. The collision test asserts these are unique. */
export function boundShortcuts(): { id: string; title: string; shortcut: string }[] {
  return [
    ...SHELL_COMMANDS.flatMap((c) => (c.shortcut ? [{ id: c.id, title: c.title, shortcut: c.shortcut }] : [])),
    ...shortcutCommands().map((c) => ({
      id: c.id,
      title: c.title,
      shortcut: c.keyboard_shortcut as string,
    })),
  ];
}

/** The chords a SURFACE owns: declared in the catalog, but needing a selection or an argument the
 *  surface holds (a hunk, the open buffer), so the shell must not bind them and `boundShortcuts`
 *  cannot list them. They are still keyboard bindings, so Settings shows them as such instead of
 *  leaving them invisible: Cmd+S was bound inside Monaco and appeared in no table at all. */
export function surfaceShortcuts(): { id: string; title: string; shortcut: string; surface: string }[] {
  return COMMANDS.filter((c) => c.keyboard_shortcut && dispatchable(c) && !selfContained(c)).map((c) => ({
    id: c.id,
    title: c.title,
    shortcut: c.keyboard_shortcut as string,
    surface: c.primary_surface,
  }));
}

/** Match a catalog shortcut string ("Mod+Shift+F"); Mod is Cmd on macOS, Ctrl elsewhere. */
export function matchesShortcut(
  shortcut: string,
  e: Pick<KeyboardEvent, "key" | "metaKey" | "ctrlKey" | "shiftKey" | "altKey">,
): boolean {
  const parts = shortcut.split("+");
  const key = parts[parts.length - 1].toLowerCase();
  const has = (m: string) => parts.includes(m);
  return (
    has("Mod") === (e.metaKey || e.ctrlKey) &&
    has("Shift") === e.shiftKey &&
    has("Alt") === e.altKey &&
    e.key.toLowerCase() === key
  );
}

const CUSTOM_SET = new Set<string>(CUSTOM_NAMES);

// Build the typed api.rs Intent for an Intent-bound command, filling session and run ids from live
// state when the caller omits them.
function intentFor(name: string, a: CommandArgs): Intent {
  const s = useStore.getState();
  const runId = (a.run_id as string) ?? s.activeRunId ?? "";
  const sessionId = (a.session_id as string) ?? s.sessionId;
  switch (name) {
    case "submit_turn":
      // Attachments ride the spine too. Dropping them here is what forced the courtyard composer to
      // keep its own raw sendIntent; the argument is threaded, so nothing needs a private path.
      return intent.submitTurn(sessionId, String(a.text), (a.attachments as BlobRef[]) ?? []);
    case "cancel_run":
      return intent.cancelRun(runId);
    case "pause_run":
      return intent.pauseRun(runId);
    case "resume_run":
      return intent.resumeRun(runId);
    case "accept_diff":
      return intent.acceptDiff(runId, String(a.diff_id), (a.hunk_id as string) ?? null);
    case "reject_diff":
      return intent.rejectDiff(runId, String(a.diff_id), (a.hunk_id as string) ?? null);
    case "fork_session":
      return intent.forkSession(sessionId, String(a.at_event));
    case "open_file":
      return intent.openFile(String(a.path), (a.line as number) ?? null);
    case "run_command":
      return intent.runCommand(a.argv as string[], (a.cwd as string) ?? null);
    default:
      throw new Error(`no Intent constructor for ${name}`);
  }
}

/** The refusal path for a fire-and-forget `runCommand`. A guard that throws becomes a visible notice
 *  instead of a silent no-op, so a click never looks like it worked when the spine refused it. */
export const noticeFailure = (code: string) => (err: unknown) =>
  useStore.getState().pushNotice({ kind: "error", code, message: (err as Error).message });

/** THE single dispatch point. Every surface resolves a command id here; nothing else builds an
 *  Intent for a catalog command. Refuses honestly (a thrown Error the caller surfaces as a notice)
 *  rather than sending something the host cannot act on. */
export async function runCommand(id: string, args: CommandArgs = {}): Promise<IntentAck> {
  const spec = commandById(id);
  if (!spec) throw new Error(`unknown command: ${id}`);
  const b = spec.backend_binding;
  // ponytail: the FE speaks /v1/hide/intent only. Rpc-bound commands need an /rpc client that does
  // not exist yet; refuse loudly instead of pretending. Add the client when a surface needs one.
  if (b.kind === "rpc")
    throw new Error(`${spec.title} needs the elevated rpc channel (${b.target}), which this app does not speak`);
  if (b.kind === "local_only") throw new Error(`${spec.title} is a local action with no backend call`);

  const missing = (REQUIRED_ARGS[id] ?? []).filter((k) => args[k] == null);
  if (missing.length) throw new Error(`${spec.title} needs ${missing.join(", ")}`);
  if (RUN_SCOPED.includes(id) && !(args.run_id ?? useStore.getState().activeRunId))
    throw new Error(`${spec.title} needs an active run`);

  if (b.kind === "custom") {
    if (!CUSTOM_SET.has(b.target)) throw new Error(`custom name not on the wire contract: ${b.target}`);
    // Same session fill-in the Intent path gets, so a custom-bound caller cannot forget it.
    const payload = args.session_id == null ? { ...args, session_id: useStore.getState().sessionId } : args;
    return sendIntent(intent.custom(b.target as CustomName, payload));
  }
  return sendIntent(intentFor(b.target, args));
}

/**
 * Whether the open session has anything to show, and therefore whether the Chat stage (its panel
 * bar, and the `panel.*` palette rows that open into it) exists.
 *
 * It used to be `messages.length > 0` in two places, which meant the panel bar - the ONLY mount of
 * the Context Stack - could not open on a host with no served model, because the composer that
 * fills `messages` is disabled until the runtime is ready. A replayed session is a session with
 * something to show: its tool feed is real recorded work even when the transcript is empty.
 */
export const hasSessionActivity = (s: State): boolean => s.messages.length > 0 || s.tools.length > 0;

// The reconnect cursor: the highest seq any slice has applied.
export function lastAppliedSeq(s: State): number {
  return Math.max(
    s.chatSeq,
    s.runtimeSeq,
    s.toolSeq,
    s.contextSeq,
    s.fleetSeq,
    s.homeSeq,
    s.sessionsSeq,
    s.projectionSeq,
  );
}

// Boot: subscribe the store to the live UiEvent stream. Returns an unsubscribe fn.
// Transport errors are surfaced into notifyStore, never swallowed.
export function connectStore(): () => void {
  const apply = useStore.getState().apply;
  const unsub = subscribeUi(
    (ev) => apply(ev),
    (err) => useStore.getState().pushNotice({ kind: "error", code: "transport", message: err.message }),
    lastAppliedSeq(useStore.getState()),
    // The backfill's scope, read at each (re)connect rather than captured once: the store renders
    // one session, and an unscoped catch-up on a host with a long log would splice every other
    // session's transcript and tool feed into it.
    () => useStore.getState().sessionId,
  );
  // The live runtime emits its status once at boot, before the UI connects, and that event is not
  // replayable, so readiness is READ on connect. It reads the supervisor's real state: this used to
  // synthesize a "ready" event whenever the STATIC role registry was non-empty, which it always is,
  // so the status dot and the composer said ready on a host with no engine at all.
  if (TRANSPORT_KIND === "live") {
    void callConnector<{ state?: RuntimeState; detail?: string | null }>("runtime", "state", {})
      .then((r) => {
        // A real status that already arrived over the socket outranks this read.
        if (!r?.state || useStore.getState().runtimeStatus !== "down") return;
        apply({
          seq: 0,
          session_id: null,
          kind: { type: "runtime_status", data: { status: r.state, detail: r.detail ?? null } },
        });
      })
      .catch(() => void 0);
    // Seed the courtyard: the home digest + session list are folded from the event log by the host and
    // pulled here on connect (they are not part of the replayed event stream). Failures surface via the
    // transport notice path, not here.
    void callConnector<{ home?: unknown; sessions?: { items?: SessionSummary[] }; status?: unknown }>("home", "digest", {})
      .then((r) => {
        if (r?.home)
          apply({ seq: 0, session_id: null, kind: { type: "projection_patch", data: { projection: "home", patch: r.home } } });
        if (r?.sessions)
          apply({ seq: 0, session_id: null, kind: { type: "projection_patch", data: { projection: "sessions", patch: r.sessions } } });
        // The write lease rides the same read. It is held in host process memory and published only
        // live, so a reloaded tab was silently under-reporting an authorization that was still being
        // honoured, with no way to revoke what it could not see.
        if (r?.status)
          apply({ seq: 0, session_id: null, kind: { type: "projection_patch", data: { projection: "status", patch: r.status } } });
        // Reload durability for the conversation itself. The socket forwards only what is published
        // from this instant on, so a fresh tab starts blank while the host still holds the whole
        // session. Reopen the most recent one through the SAME open_session route the recents rail
        // uses (the host replays its durable events) instead of inventing a second hydration path.
        // Only when nothing is on screen yet, so a reconnect mid-conversation never yanks the user
        // out of the session they are in.
        // Newest first (digest.rs sorts on updated_us), under `items` the way the patch carries it.
        const recent = r?.sessions?.items?.[0]?.id;
        const s = useStore.getState();
        if (recent && s.messages.length === 0 && s.tools.length === 0) s.openSession(recent);
      })
      .catch(() => void 0);
  }
  return unsub;
}
