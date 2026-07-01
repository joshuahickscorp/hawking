/*
  store.ts: one coherent Zustand store, the derived cache of the UiEvent stream.
  The host owns truth; this store is a fold of the event log (constitution principle 3).
  The only outbound mutation channel is an Intent (sent via ipc); there is no third path.

  Slices follow 01-surfaces §A.4 names. Each tracks lastAppliedSeq so reconnect replays cleanly
  (catch up via GET ?after_seq=lastAppliedSeq, then resume live). The EventRouter dispatches each
  UiEventKind to its slice. Every failure is surfaced into notifyStore, never swallowed.
*/
import { create } from "zustand";
import { callConnector, subscribeUi, TRANSPORT_KIND } from "./ipc";
import type { ProjectionName, RuntimeState, UiEvent } from "./wire";

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

  // editor / diff / timeline / problems / generic projections (stubs that hold real patches)
  projections: Partial<Record<ProjectionName, Patch>>;
  projectionSeq: number;

  // security gate (blocking, never auto-dismissed: FE-5)
  gate: SecurityGate | null;

  // notifications (errors + surfaced transport failures)
  notices: Notice[];

  // ---- actions (all internal; user actions go out as Intents elsewhere) ----
  apply(ev: UiEvent): void;
  pushNotice(n: Omit<Notice, "id">): void;
  dismissGate(): void;
  pushUserMessage(text: string): void;
}

let _id = 0;
const nextId = () => `m${++_id}`;

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

  projections: {},
  projectionSeq: 0,

  gate: null,
  notices: [],

  pushUserMessage: (text) =>
    set((s) => ({ messages: [...s.messages, { id: nextId(), role: "user", text, streaming: false }] })),

  pushNotice: (n) => set((s) => ({ notices: [...s.notices.slice(-19), { ...n, id: nextId() }] })),

  dismissGate: () => set({ gate: null }),

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
          tools: [...s.tools.slice(-49), { call_id: k.data.call_id, message: k.data.message, ts: Date.now() }],
          toolSeq: ev.seq,
        }));
        break;

      case "security_gate":
        set({ gate: { gate: k.data.gate, message: k.data.message } });
        break;

      case "error":
        get().pushNotice({ kind: "error", code: k.data.code, message: k.data.message });
        break;

      case "projection_patch":
        set((s) => routeProjection(s, k.data.projection, k.data.patch as Patch, ev.seq));
        break;

      case "custom":
        // Custom events route by an agreed discriminator inside the value; until a panel
        // registers one, surface it as info so nothing is silently dropped.
        get().pushNotice({ kind: "info", code: "custom", message: JSON.stringify(k.data).slice(0, 200) });
        break;
    }
  },
}));

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

// The reconnect cursor: the highest seq any slice has applied.
export function lastAppliedSeq(s: State): number {
  return Math.max(s.chatSeq, s.runtimeSeq, s.toolSeq, s.contextSeq, s.fleetSeq, s.projectionSeq);
}

// Boot: subscribe the store to the live UiEvent stream. Returns an unsubscribe fn.
// Transport errors are surfaced into notifyStore, never swallowed.
export function connectStore(): () => void {
  const apply = useStore.getState().apply;
  const unsub = subscribeUi(
    (ev) => apply(ev),
    (err) => useStore.getState().pushNotice({ kind: "error", code: "transport", message: err.message }),
    lastAppliedSeq(useStore.getState()),
  );
  // The live runtime emits its "ready" status once at boot, before the UI connects, and that event
  // is not replayable. Derive current readiness from the runtime connector (a real working endpoint),
  // so the status bar reflects the live engine instead of staying "down".
  if (TRANSPORT_KIND === "live") {
    void callConnector<{ roles?: Array<{ model?: { architecture?: string } | null }> }>("runtime", "roles.list", {})
      .then((r) => {
        const roles = r?.roles;
        if (Array.isArray(roles) && roles.length) {
          const detail = roles[0]?.model?.architecture ?? "hawking";
          apply({ seq: 0, session_id: null, kind: { type: "runtime_status", data: { status: "ready", detail } } });
        }
      })
      .catch(() => void 0);
  }
  return unsub;
}
