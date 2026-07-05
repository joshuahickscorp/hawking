/*
  ipc.ts: the ONLY module that touches the HTTP/WS transport to hide-serve.
  Everything above this (store, router, surfaces) imports the typed seam only, so the
  transport (and any future desktop wrapper or remote host) is swappable without touching UI.
  Endpoints (00-vision §3.7): POST /v1/hide/intent, WS|GET /v1/hide/events, POST /v1/hide/connector.

  Failures are SURFACED, never swallowed: a rejected ack is returned as-is (accepted:false is a
  200 body), and a transport error is reported through onError so the UI shows it.
*/
import type { ConnectorId, Intent, IntentAck, UiEvent } from "./wire";

const BASE = import.meta.env.VITE_HIDE_BASE ?? "http://127.0.0.1:8744"; // hide-serve loopback
const WS_BASE = BASE.replace(/^http/, "ws");

// Default to the mock transport in dev so the app runs ALIVE with no backend.
// Set VITE_HIDE_TRANSPORT=live to bind the real hide-serve.
const USE_MOCK = (import.meta.env.VITE_HIDE_TRANSPORT ?? "mock") !== "live";

export interface Transport {
  sendIntent(intent: Intent): Promise<IntentAck>;
  /** Subscribe to the ordered UiEvent stream. Returns an unsubscribe fn. afterSeq backfills the gap first. */
  subscribeUi(onEvent: (ev: UiEvent) => void, onError: (err: Error) => void, afterSeq?: number): () => void;
  callConnector<T = unknown>(id: ConnectorId, method: string, params: unknown): Promise<T>;
}

// -------------------------------------------------------------------------------------------------
// Live transport: real fetch + WebSocket against hide-serve.
// -------------------------------------------------------------------------------------------------
class LiveTransport implements Transport {
  async sendIntent(intent: Intent): Promise<IntentAck> {
    const r = await fetch(`${BASE}/v1/hide/intent`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(intent),
    });
    if (!r.ok) throw new Error(`intent transport failed: ${r.status} ${r.statusText}`);
    return (await r.json()) as IntentAck; // accepted:false is a 200 body, not an HTTP error
  }

  subscribeUi(onEvent: (ev: UiEvent) => void, onError: (err: Error) => void, afterSeq?: number): () => void {
    let closed = false;
    let ws: WebSocket | null = null;
    let notified = false; // report a socket drop once per outage, not on every 1s reconnect tick

    const open = async (fromSeq: number) => {
      if (closed) return;
      try {
        // Pull catch-up first (GET ?after_seq=N) to fill any gap, then resume the live socket.
        if (fromSeq > 0) {
          const gap = await this.catchUp(fromSeq);
          for (const ev of gap) onEvent(ev);
          fromSeq = gap.length ? gap[gap.length - 1].seq : fromSeq;
        }
      } catch (e) {
        onError(e instanceof Error ? e : new Error(String(e)));
      }
      if (closed) return;
      ws = new WebSocket(`${WS_BASE}/v1/hide/events`);
      let lastSeq = fromSeq;
      ws.onopen = () => {
        notified = false;
      };
      ws.onmessage = (e) => {
        try {
          const ev = JSON.parse(e.data as string) as UiEvent;
          // Keep the reconnect cursor monotonic: a stray out-of-order (lower-seq) event must not
          // rewind it, or a reconnect would re-request / skip the gap. catchUp(lastSeq) on the next
          // open() backfills anything the socket missed during the disconnect window.
          lastSeq = Math.max(lastSeq, ev.seq);
          onEvent(ev);
        } catch (err) {
          onError(err instanceof Error ? err : new Error(String(err)));
        }
      };
      ws.onerror = () => {
        if (notified) return;
        notified = true;
        onError(new Error("event socket error; reconnecting"));
      };
      ws.onclose = () => {
        if (closed) return;
        // Reconnect: backfill from lastSeq, then resume live.
        setTimeout(() => open(lastSeq), 1000);
      };
    };

    void open(afterSeq ?? 0);
    return () => {
      closed = true;
      ws?.close();
    };
  }

  private async catchUp(afterSeq: number): Promise<UiEvent[]> {
    const r = await fetch(`${BASE}/v1/hide/events?after_seq=${afterSeq}`);
    if (!r.ok) throw new Error(`catch-up failed: ${r.status}`);
    return (await r.json()) as UiEvent[];
  }

  async callConnector<T = unknown>(id: ConnectorId, method: string, params: unknown): Promise<T> {
    const r = await fetch(`${BASE}/v1/hide/connector`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id, method, params }),
    });
    if (!r.ok) throw new Error(`connector ${id}.${method} failed: ${r.status}`);
    return (await r.json()) as T;
  }
}

// -------------------------------------------------------------------------------------------------
// Mock transport: replays a scripted, believable agent run so the shell is ALIVE with no backend.
// RuntimeStatus ready -> a SubmitTurn token stream -> ToolProgress -> context manifest patch ->
// fleet patches. Tokens stream coalesced, paced like ~real decode.
// -------------------------------------------------------------------------------------------------
const MOCK_SESSION = "ses_mock0000000000000000000";
const MOCK_RUN = "run_mock0000000000000000000";
const MOCK_STREAM = "str_mock0000000000000000000";

const MOCK_MANIFEST = {
  model: { id: "qwen2.5-7b", arch: "qwen", ctx: 32768, profile: "Standard", sampling: "greedy" },
  budget: { total: 16384, used: 14210, free: 2174, segments: [
    { source: "system", tokens: 1200 },
    { source: "code", tokens: 6100 },
    { source: "tools", tokens: 3400 },
    { source: "memory", tokens: 980 },
    { source: "history", tokens: 2530 },
  ] },
  retrieved: [
    { path: "crates/pool/src/guard.rs", range: "42-88", relevance: 0.91 },
    { path: "crates/pool/src/lib.rs", range: "12-30", relevance: 0.74 },
  ],
  tools: [
    { name: "read", ok: true },
    { name: "grep", ok: true },
    { name: "edit", ok: false },
  ],
  memory: [{ fact: "DB uses sqlx", confidence: 1.0 }],
  dropped: [{ title: "cargo build log", would_be_tokens: 4200, reason: "low relevance" }],
  // Spine A: the live, measured context picture (the host emits this for real; the mock
  // mirrors the shape so the ambient Context-Stack line renders).
  arch: "qwen",
  ctx_len_native: 32768,
  ctx_len_effective: 131072,
  tq_multiplier: 4.0,
  tq_estimated: true,
  live: { effective_ceiling_tokens: 131072, used_tokens_estimate: 14210, occupancy: 0.11, watermark: "normal" as const, estimated: true },
};

// Courtyard mock: a believable retrospective digest + a session list so Home is ALIVE with no backend.
// Deterministic (no Math.random) so re-renders never churn the heatmap.
const HM_COLS = 18;
const MOCK_HEATMAP = Array.from({ length: HM_COLS * 7 }, (_, i) => {
  const col = Math.floor(i / 7);
  const row = i % 7;
  const recency = col / HM_COLS; // later columns are busier
  const weekday = row >= 1 && row <= 5 ? 1 : 0.4;
  const jitter = ((i * 2654435761) % 7) / 7; // deterministic 0..1
  return Math.max(0, Math.round(recency * recency * 9 * weekday * (0.5 + jitter)));
});

const MOCK_HOME = {
  user: { name: "Joshua-Hicks", plan: "Max" },
  workspace: { root: "/Users/scammermike/Downloads/hawking", repo: "hawking", branch: "main", worktrees: [] as string[] },
  digest: {
    sessions: 1182,
    messages: 119695,
    tokens: 222_900_000,
    active_days: 32,
    streak_current: 4,
    streak_longest: 17,
    peak_hour: 7,
    favorite_model: "qwen2.5-7b",
    heatmap: MOCK_HEATMAP,
    heatmap_cols: HM_COLS,
  },
};

const MIN = 60_000;
// Demo sessions: readable tasks that double as the first turn, so clicking one opens a working chat.
const MOCK_SESSIONS = {
  items: [
    { id: "ses_guard", title: "Fix the pool guard drop order", state: "active" as const, updated_ms: Date.now() - 3 * MIN, branch: "main" },
    { id: "ses_retry", title: "Add a retry test for the exhausted pool", state: "idle" as const, updated_ms: Date.now() - 40 * MIN },
    { id: "ses_tok", title: "Port the tokenizer to Rust", state: "done" as const, updated_ms: Date.now() - 3 * 60 * MIN },
    { id: "ses_ladder", title: "Wire the condense ladder end to end", state: "idle" as const, updated_ms: Date.now() - 20 * 60 * MIN },
    { id: "ses_serve", title: "Serve the 32B .tq resident", state: "done" as const, updated_ms: Date.now() - 2 * 24 * 60 * MIN },
  ],
};

class MockTransport implements Transport {
  private listeners = new Set<(ev: UiEvent) => void>();
  private seq = 0;
  private timers: ReturnType<typeof setTimeout>[] = [];

  private emit(kind: UiEvent["kind"], session_id: string | null = MOCK_SESSION) {
    const ev: UiEvent = { seq: ++this.seq, session_id, kind };
    for (const l of this.listeners) l(ev);
  }

  private later(ms: number, fn: () => void) {
    this.timers.push(setTimeout(fn, ms));
  }

  async sendIntent(intent: Intent): Promise<IntentAck> {
    // Mirror the host's validation so the mock rejects exactly what hide-backend would.
    if (intent.type === "submit_turn") {
      if (!intent.data.text.trim()) return { accepted: false, event_seq: null, message: "empty turn" };
      this.scriptTurn(intent.data.text);
    }
    if (intent.type === "run_command" && intent.data.argv.length === 0)
      return { accepted: false, event_seq: null, message: "empty argv" };
    if (intent.type === "open_file" && !intent.data.path.trim())
      return { accepted: false, event_seq: null, message: "blank path" };
    if (intent.type === "custom" && !String(intent.data.name).trim())
      return { accepted: false, event_seq: null, message: "blank custom name" };
    return { accepted: true, event_seq: this.seq, message: null };
  }

  // The scripted assistant turn: acknowledge the task, then stream a plausible plan token-batch by
  // token-batch, then a tool move, the manifest, and the turn end. The reply references the user text so
  // each demo session reads as its own conversation.
  private scriptTurn(userText: string) {
    const task = userText.trim() || "the task";
    const reply =
      `Starting on: ${task}. I will read the relevant files first, sketch the smallest change that ` +
      `works, then add a regression test before applying anything.`;
    this.emit({ type: "projection_patch", data: { projection: "turn", patch: { run_id: MOCK_RUN, phase: "planning" } } });
    const words = reply.split(" ");
    let t = 120;
    for (let i = 0; i < words.length; i += 2) {
      const chunk = words.slice(i, i + 2).join(" ") + " ";
      this.later(t, () => this.emit({ type: "token_batch", data: { stream_id: MOCK_STREAM, text: chunk } }));
      t += 42;
    }
    this.later(t + 60, () => this.emit({ type: "tool_progress", data: { call_id: "call_read_1", message: "reading the workspace for context" } }));
    this.later(t + 200, () => this.emit({ type: "projection_patch", data: { projection: "context_manifest", patch: MOCK_MANIFEST } }));
    this.later(t + 320, () => this.emit({ type: "projection_patch", data: { projection: "turn", patch: { run_id: MOCK_RUN, phase: "done" } } }));
  }

  subscribeUi(onEvent: (ev: UiEvent) => void, _onError: (err: Error) => void, _afterSeq?: number): () => void {
    this.listeners.add(onEvent);
    // Ambient boot stream: runtime comes up, the courtyard digest + session list arrive, the fleet
    // shows two parallel agents.
    this.later(40, () => this.emit({ type: "runtime_status", data: { status: "booting", detail: "starting hawking serve" }, }, null));
    this.later(120, () => this.emit({ type: "projection_patch", data: { projection: "home", patch: MOCK_HOME } }, null));
    this.later(140, () => this.emit({ type: "projection_patch", data: { projection: "sessions", patch: MOCK_SESSIONS } }, null));
    this.later(420, () => this.emit({ type: "runtime_status", data: { status: "ready", detail: "qwen2.5-7b @ 41 tps" } }, null));
    this.later(700, () =>
      this.emit({ type: "projection_patch", data: { projection: "fleet", patch: { runs: [
        { id: "run_a", objective: "refactor pool guard", state: "active", step: 3, steps: 6 },
        { id: "run_b", objective: "add retry tests", state: "waiting", step: 2, steps: 4 },
      ] } } }, null),
    );
    this.later(1600, () =>
      this.emit({ type: "projection_patch", data: { projection: "fleet", patch: { runs: [
        { id: "run_a", objective: "refactor pool guard", state: "active", step: 4, steps: 6 },
        { id: "run_b", objective: "add retry tests", state: "done", step: 4, steps: 4 },
      ] } } }, null),
    );
    return () => {
      this.listeners.delete(onEvent);
      for (const t of this.timers) clearTimeout(t);
      this.timers = [];
    };
  }

  async callConnector<T = unknown>(id: ConnectorId, method: string, _params: unknown): Promise<T> {
    if (id === "context" && method === "compile") return { prompt: "", manifest: MOCK_MANIFEST } as T;
    if (id === "runtime" && method === "roles.list")
      return [{ role: "code", model: "qwen2.5-7b" }, { role: "plan", model: "qwen2.5-7b" }] as T;
    if (id === "code_index" && method === "search") return [] as T;
    return null as T;
  }
}

export const transport: Transport = USE_MOCK ? new MockTransport() : new LiveTransport();
export const TRANSPORT_KIND: "mock" | "live" = USE_MOCK ? "mock" : "live";

// The seam the rest of the app imports. Nothing else touches fetch/WebSocket.
export const sendIntent = (i: Intent) => transport.sendIntent(i);
export const subscribeUi = (
  onEvent: (ev: UiEvent) => void,
  onError: (err: Error) => void,
  afterSeq?: number,
) => transport.subscribeUi(onEvent, onError, afterSeq);
export const callConnector = <T = unknown>(id: ConnectorId, method: string, params: unknown) =>
  transport.callConnector<T>(id, method, params);
