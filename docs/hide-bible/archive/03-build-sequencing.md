# Front-End Build Sequencing — Skeleton First

> Part of the HIDE front-end bible. The 11-crate backend is **already built and tested** (the
> agent loop is real); see [`SCAFFOLD_STATUS.md`](../SCAFFOLD_STATUS.md). This doc is the
> actionable, ordered plan to build the UI **on top of** that backend — the React/TS/Vite web app
> plus the one piece of new Rust glue (`hide-serve`, a thin axum HTTP/WS server wrapping
> `BackendHost`, mirroring `crates/hawking-serve`). It does **not** re-plan any backend work;
> that is done.

---

## 0. Where we start from (the only thing not yet built)

Everything below the wire is finished. `BackendHost` boots, supervises `hawking serve`, runs the
Planner→Executor→Verifier loop, persists an event log, scrubs/forks sessions, and exposes a
transport-agnostic command surface. What does **not** exist yet, and is the entire subject of this
doc, is:

1. The **`hide-serve` crate** — a small axum HTTP/WS server that constructs
   `BackendHost::open_workspace` and exposes the `/v1/hide/*` endpoints (Wire-A `POST
   /v1/hide/intent`, Wire-B `WS /v1/hide/events` + `GET ?after_seq=N`, `POST /v1/hide/connector`).
   It mirrors `crates/hawking-serve` (the already-proven axum + SSE server in this repo). This is
   the transport seam recorded as a *deliberately deferred* item in `SCAFFOLD_STATUS.md` — we are
   filling it now. (A desktop wrapper — Electron/Tauri/PWA — is a separate, deferred packaging
   choice that does **not** touch UI code; see [`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) §5.)
2. The **React/TS/Vite web app** (`app/src`): the typed HTTP/WS client, the store, and the panels.

There is **no** backend redesign here. If a panel needs something, it already has a binding on the
real contract (see [`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) for the wire). Our job is
projection.

**The guiding rule (inherited from the backend constitution, restated for the FE):** *the headless
path is the truth; the UI is a projection, never the source of correctness.* Every panel renders
`UiEvent`s and emits `Intent`s. No panel mutates authoritative state locally. (Constitution §0.3
principle 1 & 3; in our world: the store is a cache folded from the event stream, and the only way
to change anything is to send an `Intent`.)

---

## 1. Scaffold the web app + the `hide-serve` server (the host glue)

**Goal:** a web app that talks to `hide-serve`, which boots `BackendHost`, can receive one `Intent`
over HTTP, and can stream one `UiEvent` over a WebSocket. No panels yet. This eliminates the single
biggest unknown — a Vite web app + an axum Rust host co-existing in CI — before any feature work.

### 1.1 `hide-serve` (the Rust HTTP/WS server)

A thin axum server, mirroring `crates/hawking-serve`. It owns one `BackendHost`, exposes the intent
endpoint, the connector endpoint, and the event WebSocket (+ its pull twin), and forwards the bus.

```rust
// crates/hide-serve/src/lib.rs  (sketch — binds to the REAL hide-backend API; mirrors hawking-serve)
use hide_backend::BackendHost;
use hide_core::api::Intent;
use axum::{
    extract::{ws::{Message, WebSocket, WebSocketUpgrade}, Query, State},
    routing::{get, post},
    Json, Router,
};
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone)]
struct AppState { host: Arc<BackendHost> }

fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/hide/intent", post(hide_intent))         // Wire-A
        .route("/v1/hide/events", get(hide_events))          // Wire-B: WS upgrade OR pull (?after_seq=N)
        .route("/v1/hide/connector", post(hide_connector))   // connector RPC (§2.3)
        .with_state(state)
}

// Wire-A: deserialize Intent → CommandRouter::handle → serialize IntentAck.
async fn hide_intent(State(s): State<AppState>, Json(intent): Json<Intent>) -> Json<Value> {
    let ack = s.host.handle_intent(intent).await;     // accepted:false is a normal 200 body
    Json(serde_json::to_value(ack.unwrap_or_default()).unwrap())
}

// Wire-B: a GET with ?after_seq=N is the pull catch-up; otherwise it's a WebSocket upgrade.
async fn hide_events(
    State(s): State<AppState>,
    Query(q): Query<std::collections::HashMap<String, String>>,
    ws: Option<WebSocketUpgrade>,
) -> axum::response::Response {
    if let Some(after) = q.get("after_seq").and_then(|v| v.parse::<u64>().ok()) {
        let evs = s.host.ui_events(None, Some(after), None).await.unwrap_or_default();
        return Json(evs).into_response();
    }
    ws.unwrap().on_upgrade(move |socket| forward_bus(s, socket))
}

// Forward every UiEvent from the bus onto the socket (ordered; host-side coalesced + backpressured).
async fn forward_bus(s: AppState, mut socket: WebSocket) {
    let mut rx = s.host.subscribe_ui();               // broadcast::Receiver<UiEvent>
    while let Ok(ev) = rx.recv().await {
        let json = serde_json::to_string(&ev).unwrap();
        if socket.send(Message::Text(json)).await.is_err() { break; }
    }
}

async fn hide_connector(State(s): State<AppState>, Json(req): Json<ConnectorReq>) -> Json<Value> {
    Json(s.host.call_connector(&req.id, &req.method, req.params).await.unwrap_or(Value::Null))
}
#[derive(serde::Deserialize)] struct ConnectorReq { id: String, method: String, params: Value }

pub async fn run(root: &str, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let host = Arc::new(BackendHost::open_workspace(root)?);   // builds the whole service graph
    let app = router(AppState { host });
    let listener = tokio::net::TcpListener::bind(addr).await?; // 127.0.0.1 — loopback only
    axum::serve(listener, app).await?;
    Ok(())
}
```

Key facts this binds to (all real, in `hide-backend`/`hide-core`):
- `BackendHost::open_workspace(root)` → constructs the whole service graph, incl. the
  `RuntimeSupervisor` that boots/supervises `hawking serve`.
- `BackendHost::subscribe_ui() -> broadcast::Receiver<UiEvent>` is the bus tap. The `UiEventBus`
  (`ui_bus.rs`) already does render-coalescing of `publish_token`, so the FE receives **coalesced
  `TokenBatch`** events, not per-token spam.
- `handle_intent(Intent) -> IntentAck` is *literally* `CommandRouter::handle` behind the host;
  validation/rejection/interrupt-signalling already live there. We add zero logic — we wrap.
- `ui_events(session, after_seq, limit)` backs `GET /v1/hide/events?after_seq=N` (reconnect pull).
- The connector RPC endpoint, `POST /v1/hide/connector`, is detailed in §2.3.

**Binding & perms:** `hide-serve` binds `127.0.0.1` (loopback only — air-gap-safe). Pin CI to
`macos-14` (matches the existing runner). The PTY is hosted here via `portable-pty` and streamed
over its own WebSocket (see panel §3.5). A desktop wrapper (Electron/Tauri/PWA) is a deferred
packaging choice that would launch this server as a child process — it does **not** change any of
the above (see [`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) §5).

### 1.2 `app/src` (React + TS + Vite — the web app)

Standard Vite React-TS template. `pnpm`. Browser-renderable; no desktop-shell dependency. The only
structural decisions to lock now:
- **Store fabric:** Zustand-style slices, one slice per surface (see §2.2). One root store; panels
  subscribe to slices.
- **HTTP/WS client module** (`src/ipc/`): the *only* place `fetch`/`WebSocket` against `hide-serve`
  is touched. Everything else imports the typed client (§2.1). This keeps the transport swappable
  (and any future desktop wrapper invisible) and testable against a mock.

**Done when:** `pnpm build` exits 0 and `hide-serve` builds on `macos-14`; with `hide-serve`
running, the web app loads in a browser; a dev-only button calls
`sendIntent({ type:'custom', data:{ name:'ping', payload:{} } })` (a `fetch` to
`POST /v1/hide/intent`) and logs the returned `IntentAck`.

---

## 2. The HTTP/WS client layer + store wiring

This is the seam between the React tree and `BackendHost` (via `hide-serve`). Get it exactly right
once; every panel rides on it. The contract is owned by
[`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) — this section is the
*client implementation* of that contract.

### 2.1 The typed TS client surface

```ts
// src/ipc/client.ts — the ENTIRE surface the rest of the app may use.
import type { Intent, IntentAck, UiEvent } from './contract';   // mirrors hide-core::api
const BASE = import.meta.env.VITE_HIDE_BASE ?? 'http://127.0.0.1:8744';   // hide-serve
const WS_BASE = BASE.replace(/^http/, 'ws');

export async function sendIntent(intent: Intent): Promise<IntentAck> {
  const r = await fetch(`${BASE}/v1/hide/intent`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify(intent),
  });
  return r.json() as Promise<IntentAck>;          // accepted:false is a 200 body, not an HTTP error
}

// Open ONCE at app boot. Fans every UiEvent to the store reducer. Returns an unsubscribe fn.
export function onUiEvent(handler: (ev: UiEvent) => void): () => void {
  const ws = new WebSocket(`${WS_BASE}/v1/hide/events`);
  ws.onmessage = (e) => handler(JSON.parse(e.data) as UiEvent);
  return () => ws.close();
}

// Reconnect pull catch-up: GET the durable UiEvents after a seq cursor.
export async function catchUpUiEvents(afterSeq: number): Promise<UiEvent[]> {
  const r = await fetch(`${BASE}/v1/hide/events?after_seq=${afterSeq}`);
  return r.json() as Promise<UiEvent[]>;
}

export async function callConnector(
  id: 'runtime' | 'code_index' | 'context' | 'personalization' | 'research',
  method: string,
  params: unknown,
): Promise<unknown> {
  const r = await fetch(`${BASE}/v1/hide/connector`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ id, method, params }),
  });
  return r.json();
}
```

`contract.ts` is the hand-mirrored (or `ts-rs`-generated) TypeScript of `hide-core::api`:
`Intent` (tagged `{ type, data }`), `IntentAck { accepted, event_seq?, message? }`,
`UiEvent { seq, session_id?, kind }`, and `UiEventKind` (also `{ type, data }`-tagged):
`ProjectionPatch | TokenBatch | RuntimeStatus | ToolProgress | SecurityGate | Error | Custom`.
**Keep this file in lockstep with `crates/hide-core/src/api.rs`** — it is the one place wire drift
will bite. A CI check that diffs the generated TS against the Rust is cheap insurance.

### 2.2 Store slices fed by `UiEvent`s

One reducer routes on `UiEvent.kind.type`; each slice owns one concern. The mapping is fixed:

| `UiEventKind` | Drives slice | What it does |
|---|---|---|
| `ProjectionPatch{projection, patch}` | the panel named by `projection` | merge `patch` (a state-diff) into that panel's slice — this is how editor buffers, plan tree, diff sets, file tree, and the Context Stack all update |
| `TokenBatch{stream_id, text}` | `chatStore` / `timelineStore` | append coalesced text to the open stream for `stream_id` |
| `RuntimeStatus{status, detail}` | `runtimeStore` | serve up/down/degraded → Status Bar chip color + tps |
| `ToolProgress{call_id, message}` | `timelineStore` | live tool-call row updates |
| `SecurityGate{gate, message}` | `gateStore` | raise an approval modal/toast |
| `Error{code, message}` | `notifyStore` | toast |
| `Custom(Value)` | `notifyStore` (default) | forward to whichever panel registered for it |

Slice list: `chatStore`, `editorStore`, `diffStore`, `fileTreeStore`, `terminalStore`,
`contextStore`, `timelineStore`, `runtimeStore`, `gateStore`, `notifyStore`. Every write to an
authoritative field comes from a `UiEvent`; every user action goes out as an `Intent`. There is no
third path.

### 2.3 Connector access (`POST /v1/hide/connector`)

Add the server endpoint and the client wrapper. Connectors are read-mostly side channels the panels
use for synchronous-ish queries (search, profile switch, manifest compile) that are not turn
submissions:

```rust
// crates/hide-serve — body = { id, method, params } → BackendHost::call_connector → Value
async fn hide_connector(State(s): State<AppState>, Json(req): Json<ConnectorReq>) -> Json<Value> {
    Json(s.host.call_connector(&req.id, &req.method, req.params).await.unwrap_or(Value::Null))
}
#[derive(serde::Deserialize)] struct ConnectorReq { id: String, method: String, params: Value }
```

Real connectors (`connectors.rs`): `runtime` (`roles.list`, `route` — the latter a read-only routing
preview), `code_index` (`search`, `definition`, `references`, `file.add_text`, `file.index`,
`health`), `context` (`compile` → prompt + manifest), `personalization`
(`records.list`/`records.append`/`records.by_task`), `research`
(`runs.list`/`runs.latest`/`runs.append`/`runs.by_state`). Panels call these; they do
**not** reach into crates directly.

**Done when:** `onUiEvent` receives a live `RuntimeStatus` after `BackendHost` boots serve; a
`sendIntent({type:'open_file', data:{path,line}})` returns `accepted:true`; `callConnector('code_index','search',{q})` returns hits.

---

## 3. Skeleton panels in priority order

Build order is chosen so the **earliest possible end-to-end demo is chat streaming from serve**,
then each subsequent panel exercises one more slice of the contract. For each: what to harvest
(licensed FE source — see [`02-oss-harvest.md`](02-oss-harvest.md)),
and which contract pieces it exercises.

| # | Panel | Harvest from doc 02 | Intents it sends | UiEvents it consumes |
|---|---|---|---|---|
| 1 | **Chat** | message-list + composer patterns; SSE/stream render | `SubmitTurn`, `CancelRun`, `PauseRun`, `ResumeRun` | `TokenBatch`, `ProjectionPatch(chat)`, `RuntimeStatus` |
| 2 | **Editor** | **Monaco** (`monaco-editor`, MIT) | `OpenFile` | `ProjectionPatch(editor)` |
| 3 | **Diff Review** | Monaco `createDiffEditor` + Cline/Void hunk-UX (Apache-2.0, *reference only*) | `AcceptDiff`, `RejectDiff` | `ProjectionPatch(diff)`, `SecurityGate` |
| 4 | **File Tree** | tree-view component; file reads via `hide-serve` / `code_index` connector | `OpenFile` | `ProjectionPatch(file_tree)` |
| 5 | **Terminal** | **xterm.js** (MIT) over `portable-pty` (MIT) | `RunCommand` | dedicated PTY WebSocket + `ProjectionPatch(terminal)` |
| 6 | **Context Stack** | original; renders `ContextManifest` | `ScrubToEvent`, `Custom{pin/unpin/switch_profile}` | `ProjectionPatch(context)`, `RuntimeStatus` |
| 7 | **Agent Timeline** | OpenHands event-model *idea* (MIT, *reference only*) | `ScrubToEvent`, `ForkSession`, `CancelRun` | `ProjectionPatch(timeline)`, `ToolProgress`, `TokenBatch` |
| 8 | **Workstation** | original; grid of timeline cards | `SubmitTurn`(per lane), `CancelRun` | per-session `ProjectionPatch`, `RuntimeStatus` |

### 3.1 Chat (first — the walking-skeleton spine)

The minimal full loop: user types → `sendIntent({type:'submit_turn', data:{session_id, text, attachments:[]}})` →
the kernel generates against serve → host publishes coalesced `TokenBatch{stream_id, text}` →
chat appends. Cancel/pause/resume map to `CancelRun`/`PauseRun`/`ResumeRun` against the active
`run_id` (carried in `chatStore`, learned from the run's first `ProjectionPatch`). This single
panel proves the entire vertical: command in, bus out, store fold, render.

### 3.2 Editor (Monaco)

`monaco.editor.create`. `OpenFile{path, line?}` is sent on file-tree click or a Context-Stack
row click; the host streams buffer contents back as a `ProjectionPatch(editor)`. Read-mostly in
the skeleton; ghost-text/inline-edit are later. Monaco is also the substrate for Diff Review, so it
is built before #3.

### 3.3 Diff Review

When the agent proposes an edit, the host emits a `ProjectionPatch(diff)` describing the hunks.
Render in Monaco `createDiffEditor`. Per-hunk Accept → `sendIntent({type:'accept_diff', data:{run_id, diff_id}})`;
Reject → `reject_diff`. **All apply/revert logic stays in the backend** (`hide-tools` tiered edit);
the panel only sends the verdict and re-renders the resulting patch. A `SecurityGate` event may
precede the apply (write-permission ask).

### 3.4 File Tree

The workspace root is listed by the host (served through `hide-serve` / the `code_index` connector,
not a direct browser FS API); render a tree; click → `OpenFile`. Live external-change decorations
arrive as `ProjectionPatch(file_tree)`. Read-only in the skeleton.

### 3.5 Terminal

xterm.js front; `portable-pty` in `hide-serve`. Keystrokes → PTY stdin over a dedicated WebSocket;
PTY output → xterm. Agent-initiated commands route through `RunCommand{argv, cwd?}` /
`run_agent_to_terminal(...)` so the terminal shows what the agent ran. Opens in workspace root.

### 3.6 Context Stack (the differentiator)

The right-rail, **on by default**, ~320px. It renders the `ContextManifest` verbatim and live:
Model/profile, a stacked **budget bar**, retrieved files, symbols, tools called, memory injected,
KV/tier reuse, dropped candidates (with one-click pin), conflicts, compaction. It arrives as
`ProjectionPatch(context)` each turn; `contextStore` keeps a **ring of recent manifests** so that
scrubbing the timeline re-renders the manifest *as it was at that turn*. Steering actions (pin /
unpin / resolve / switch-profile) go out as `Intent::Custom{name, payload}` (e.g.
`{name:"pin_span", payload:{span_id}}`) — the backend honors them on the next turn. Profile data and
manifest compilation come from `callConnector('context', ...)`. This panel has no cloud analog; it
is the product's signature surface.

### 3.7 Agent Timeline

A vertical run timeline: Idle → Planning → Step 1..N → Done/Failed/Repair, tool-call rows expand
to args+result, repair shown distinctly. Fed by `ProjectionPatch(timeline)` + `ToolProgress`; live
streams via `TokenBatch`. The **scrub slider** maps to `seq` and issues
`ScrubToEvent{session_id, event_id}` — the backend (`scrub_to_event`) replays the projection to
that seq and pushes rebuilt state; editor, plan tree, diffs, **and the Context Stack** all rewind
together. A timeline node can `ForkSession{session_id, at_event}` to branch.

### 3.8 Workstation (parallel agents)

A grid where each cell is a compact Agent Timeline bound to a different `session_id`. Each lane
sends its own `SubmitTurn`/`CancelRun`; the store keys all slices by `session_id` (already present
on every `UiEvent`). No new contract — it is N timelines plus a layout. This is the last skeleton
panel because it depends on every prior slice keying cleanly by session.

---

## 4. Milestones

Each milestone has one binary **done-when**. These are the FE counterparts to the backend's
M0/M1 sequencing — but the backend is done, so they are purely UI + host glue.

### M-FE0 — Walking skeleton
The web app + the `hide-serve` server, with chat streaming over the WebSocket. `hide-serve` boots
`BackendHost`; chat streams. Build §1 (web app + `hide-serve` scaffold), §2 (HTTP/WS client +
store), and §3.1 (chat only).
**Done when:** in the running web app (browser or any wrapper) talking to `hide-serve`, typing a
message sends `SubmitTurn` (a `fetch` to `POST /v1/hide/intent`), and coalesced `TokenBatch` events
arriving on the `WS /v1/hide/events` socket render incrementally in the chat panel — against a
stubbed serve *or* a live one. The `IntentAck.accepted` and the streamed `stream_id` are observable
in devtools.

### M-FE1 — IDE surface
Add Editor (§3.2), Diff Review (§3.3), File Tree (§3.4), Terminal (§3.5).
**Done when:** the agent proposes an `edit_file`; the Diff Review panel shows the hunks in Monaco;
the user accepts one hunk (`AcceptDiff`) and rejects another (`RejectDiff`); the file on disk
reflects **only** the accepted hunk; the terminal runs `cargo build` and streams output live.

### M-FE2 — Context Stack + Timeline
Add the Context Stack (§3.6) and Agent Timeline (§3.7), including scrub/replay.
**Done when:** during a multi-step run the Context Stack updates live (retrieved files appear as
the agent searches; the budget bar fills); the Timeline shows each step; dragging the scrub slider
issues `ScrubToEvent` and the Context Stack **rewinds to that turn's manifest** from the ring.

### M-FE3 — Workstation
Add the parallel-agent grid (§3.8).
**Done when:** two sessions run concurrently in two lanes; each lane's timeline and stream update
independently and correctly (no cross-talk — verified by distinct `session_id` on every event);
cancelling one lane's `run_id` leaves the other running.

---

## 5. License / harvest CI gate + `THIRD_PARTY_NOTICES`

Track licenses from the first FE commit, not at ship time. The harvest inventory and obligations
are owned by [`02-oss-harvest.md`](02-oss-harvest.md);
this is the **gate** that enforces it.

FE-relevant bundled/invoked components and obligations:

| Component | License | Usage | Obligation |
|---|---|---|---|
| Monaco Editor | MIT | editor + diff (bundled) | MIT notice |
| xterm.js | MIT | terminal (bundled) | MIT notice |
| portable-pty | MIT | PTY host (`hide-serve`) | MIT notice |
| axum | MIT | HTTP/WS server (`hide-serve`) | MIT notice |
| Zustand | MIT | store | MIT notice |
| OpenHands event model | MIT | timeline design *reference only* | no code copied → no obligation |
| Cline / Void diff UX | Apache-2.0 | diff design *reference only* | no code copied → no obligation |

**The gate (CI, every commit):**
1. `cargo deny check licenses` over `crates/hide-serve` — allow MIT / Apache-2.0 / MPL-2.0; reject
   any new dep introducing GPL / AGPL / BUSL. (Same `deny.toml` posture as the backend.)
2. A JS license scan (e.g. `license-checker`) over `app/src` `node_modules` with the same allow-list.
3. `THIRD_PARTY_NOTICES.md` is regenerated by `tools/gen_notices.sh` (extended to walk both the
   Rust host deps **and** the bundled npm deps) and must be present before any release.
4. `license-header-check`: any file in `app/src/` that is *copied* (not npm-installed) third-party
   code must carry its original copyright header; the scan fails if a copied-code file lacks one.
   Components marked *reference only* above must contain **no** copied lines — that is what keeps
   them obligation-free.

**Done when:** all four steps exit 0 in CI and `THIRD_PARTY_NOTICES.md` lists every bundled MIT
component above.

---

## 6. Build-order summary (open this to start)

```
§1 hide-serve server + web-app scaffold ─┐
                                          ├─► M-FE0  chat walking skeleton
§2 HTTP/WS client + store slices ────────┘     (SubmitTurn → TokenBatch over WS → render)
                                              │
§3.2–3.5 editor/diff/tree/terminal ──────────► M-FE1  IDE surface
                                              │
§3.6–3.7 Context Stack + timeline ───────────► M-FE2  observability + replay
                                              │
§3.8 workstation grid ───────────────────────► M-FE3  parallel agents
                                              │
§5 license gate + NOTICES ── runs from commit 1, blocks every release
```

**Consistency contract for sibling docs.** Decisions made here that other front-end docs must not
contradict:
- The FE talks to the backend through **exactly three** `hide-serve` endpoints:
  `POST /v1/hide/intent` (Wire-A: one `Intent` → `IntentAck`), `WS /v1/hide/events` (Wire-B: the
  ordered `UiEvent` stream; `GET /v1/hide/events?after_seq=N` is its pull twin for reconnect), and
  `POST /v1/hide/connector` (connector RPC: `{id, method, params}`). These match the contract doc
  ([`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) §3.7 "Decisions"); no
  panel reaches the backend any other way. The desktop wrapper (Electron/Tauri/PWA) is deferred and
  never touches UI code (contract doc §5).
- **`ProjectionPatch` is the universal state-update mechanism**; `TokenBatch` is the only streaming
  text path; the seven `UiEventKind`s map 1:1 to the slices in §2.2.
- The store is a **fold of the event stream**; the only outbound mutation channel is `Intent`.
- Panel priority is **chat → editor → diff → tree → terminal → context → timeline → workstation**,
  and the milestone done-when checks above are the acceptance gates.
- Scrub/replay rewinds **all** panels via `ScrubToEvent`; the Context Stack must keep a manifest
  **ring** to support it.
