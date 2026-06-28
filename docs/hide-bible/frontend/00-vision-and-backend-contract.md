# Front-End Vision & the Backend Contract

> Part of the **HIDE front-end bible**. The backend is **built and tested** — 11 Rust crates, a real Planner→Executor→Verifier agent loop, a runnable headless `BackendHost`. See [`SCAFFOLD_STATUS.md`](../SCAFFOLD_STATUS.md). This document is the orientation doc for the team building the UI skeleton: what HIDE's front end *is*, the stack it's built on, and — the load-bearing bulk of this doc — the **exact contract** the front end binds to. Sibling docs ([`01-surfaces.md`](01-surfaces.md) — the three surfaces + Context Stack, [`02-oss-harvest.md`](02-oss-harvest.md) — the OSS harvest map, [`03-build-sequencing.md`](03-build-sequencing.md) — the build order) build on the contract fixed here.

---

## 1. What the HIDE front end is

HIDE (Hawking IDE) is a **local-first agentic coding IDE**. The model runs on your own Apple Silicon GPU via `hawking serve` — no API calls, no telemetry, no subscription, zero marginal cost per decode. The local runtime *is* the product, not a fallback. The front end is the developer-facing surface wrapped around a runtime (`hawking serve`, supervised) and an agent layer (`hide-kernel`'s real Planner→Executor→Verifier loop) that already exist.

### Exceed, not rival

HIDE does **not** try to match cloud IDEs on raw frontier-model capability — that axis is structurally closed to a local product. It **exceeds** them on the axes a cloud provider *cannot* reach because they bill per token and assemble the context window server-side in a black box. The FE thesis, in one line: **a cloud agent is a chat box bolted to a black box** — you type, it works where you can't see, it hands back a result. Because HIDE's runtime is ours and local, the FE does the one thing they can't: **show everything and let the user edit it live.** Concretely, the FE surfaces and makes *editable*:

- the exact tokens the model sees (the `ContextManifest`, rendered as the Context Stack right-rail),
- the KV/context budget and what got dropped to fit it (drag-to-pin it back),
- every retrieved file, symbol, tool call, and memory injection, as it happens,
- the plan before *and* during execution (approve, edit, reorder it),
- the run as a **scrubbable, replayable timeline** backed by the durable event log,
- the model's own logit-derived confidence, on demand.

This "observability + live steering" is the differentiator. It is felt as latency-rich UI (local IPC is ~ms, no network round-trip), persistent restorable workspace, and HITL-by-default interaction.

### The three named surfaces

The product front end is **three surfaces** over one backend, plus the Context Stack rail that threads through all of them:

| Surface | What it is | Primary job |
|---|---|---|
| **AI IDE** | Editor (Monaco) + per-hunk Diff Review + File Explorer + integrated Terminal (xterm/PTY) | Edit code; review and accept/reject agent edits; run commands. The workbench body. |
| **AI Chat** | Streaming conversation with the agent: plan cards, tool-call chips, inline diff chips | Talk to the agent; approve/steer/interrupt a run; selection-to-chat. |
| **AI Workstation** | Parallel-agents dashboard: many sessions/runs at once with state pills, progress, ⏸/⛔/open | Fan out N agents, watch them, triage overnight runs. "Spend lavishly, locally." |
| **Context Stack** *(rail, not a surface)* | Live `ContextManifest` render in the right rail | The observability moat: show the model's mind, live, on every surface. |

The shell is **not a hard-coded screen** — it ships a layout engine, an event router, and a store fabric. (Panels-as-extensions is the long-term design; the v1 skeleton wires the three surfaces + rail directly.)

---

## 2. The tech stack (and why)

The stack is fixed by the product brief and the built host. **Do not re-litigate it.**

| Layer | Choice | Why |
|---|---|---|
| **App shell / host** | **Tauri 2** (Rust host wrapping `hide-backend::BackendHost`) | Native binary, no cloud deps, air-gap-safe; the host is already built as a headless Rust library — Tauri is the thin window + IPC bridge around it. |
| **UI runtime** | **React + TypeScript** | Component model fits the panel fabric; TS gives us typed mirrors of the Rust `Intent`/`UiEvent` wire types. |
| **Editor / diff** | **Monaco** | Ships a first-class `DiffEditor` (side-by-side + inline) with view zones, decorations (zIndex stacking), and inline widgets — exactly what ghost-text, Cmd+K inline-edit overlays, and per-hunk diff review need. |
| **Terminal** | **xterm.js + PTY** (`tauri-plugin-pty` / portable-pty) | A real PTY, not a command passthrough — proven Tauri 2 + React + xterm pattern. |
| **State store** | **Zustand-style slices** | Lightweight, selector-based; the right shape for derived-cache stores fed by a router (no Redux ceremony). |

**Architectural constraint that drives all of §2:** the view holds **no authoritative state**. The event log in the Rust host is the system of record (constitution principle 3). Every store is a *derived cache* of the projection stream; on reload the FE replays from the log and rebuilds byte-identical. This is why the stack is "host owns truth, webview renders + dispatches intents" — it makes reload lossless and time-travel free.

---

## 3. THE BACKEND CONTRACT

This is the load-bearing section. The backend is built; the FE binds to **these exact types and methods**, not to any earlier design sketch. Source of truth: `crates/hide-core/src/api.rs` (the wire types) and `crates/hide-backend/src/{host,commands,ui_bus,connectors}.rs` (the host surface).

### 3.1 The two wires + the Tauri bridge

Two directions, mediated by Tauri IPC:

```
  ┌─────────────────────────── RUST HOST (hide-backend::BackendHost) ─────────────────────────┐
  │                                                                                            │
  │   hawking serve (HTTP/SSE) ──┐                                                              │
  │   OS / tools / files ────────┼─▶ event log (single writer, seq) ─▶ projections ─▶ UiEvent  │
  │                              │                                                  │          │
  │   CommandRouter::handle(Intent) ─▶ validate ─▶ append user.intent.* ─▶ IntentAck│          │
  │           ▲                                                            UiEventBus│ (publish │
  │           │ (Wire-A)                                            subscribe()──────┘  + coalesce)
  └───────────┼──────────────────────────────────────────────────────────────┬─────────────────┘
              │  #[tauri::command] hide_intent                                 │  ipc::Channel<UiEvent>
  ┌───────────┼──────────────────────────────────────────────────────────────▼─────────────────┐
  │  WEBVIEW (React + TS)                                                                         │
  │   user action ─▶ sendIntent(intent) ─▶ invoke('hide_intent',{intent}) ─▶ IntentAck           │
  │   channel.onmessage(UiEvent) ─▶ EventRouter ─▶ route by kind ─▶ Zustand stores ─▶ render      │
  │   callConnector(id,method,params) ─▶ invoke('hide_call_connector',…) ─▶ Value                 │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Wire-A (FE → host): `invoke('hide_intent', { intent })`.** A thin `#[tauri::command]` deserializes the `Intent`, calls `CommandRouter::handle(intent)` (which `BackendHost::handle_intent` delegates to), and serializes the `IntentAck`. `handle` is deliberately a plain transport-agnostic `async fn` — the Tauri wrapper does nothing but (de)serialize. **The host validates and can reject:** an empty `SubmitTurn`, empty-argv `RunCommand`, or blank-name `Custom` returns `IntentAck { accepted: false, message: Some(reason) }` and logs nothing. The FE must surface a rejected ack, not assume success.
- **Wire-B (host → FE): `ipc::Channel<UiEvent>`.** The FE opens one ordered per-window channel; the bridge forwards everything from `BackendHost::subscribe_ui()` (a `broadcast::Receiver<UiEvent>` off the `UiEventBus`). The bus does **render-coalescing** (consecutive `TokenBatch`es for one stream merge before publish) and has **bounded backpressure** (a slow subscriber gets a `Lagged` drop-oldest signal, never stalls the host). The FE adds its own rAF render-governor on top (see §4 and the state-stores sibling doc).

### 3.2 Wire-A — the `Intent` enum (every variant)

`Intent` (`api.rs`, `#[serde(tag="type", content="data", rename_all="snake_case")]`). Each `handle` returns `IntentAck { accepted: bool, event_seq: Option<u64>, message: Option<String> }`.

| Intent variant | Payload fields | UI action that sends it | Host behavior |
|---|---|---|---|
| `SubmitTurn` | `session_id`, `text`, `attachments: Vec<BlobRef>` | Chat composer submit; selection-to-chat | **Rejected if `text` is blank.** Logs `user.intent.submit_turn`; kicks the agent turn. |
| `CancelRun` | `run_id` | Steer bar ⛔ Stop; Workstation per-run stop | Signals `Interrupt::Abort` on the `InterruptHub` for that run, then logs. |
| `PauseRun` | `run_id` | Steer bar ⏸ Pause; status-bar agent pill | Signals `Interrupt::Pause`; logs. |
| `ResumeRun` | `run_id` | Resume a paused run | Clears the buffered pause; logs. |
| `AcceptDiff` | `run_id`, `diff_id` | Diff Review: accept hunk/file | **Rejected if `diff_id` blank.** Logs `accept_diff`. |
| `RejectDiff` | `run_id`, `diff_id` | Diff Review: reject hunk/file | **Rejected if `diff_id` blank.** Logs `reject_diff`. |
| `ScrubToEvent` | `session_id`, `event_id: EventId` | Timeline scrub slider | Logs `scrub_to_event`; pairs with `BackendHost::scrub_to_event(seq)` to rebuild the read-only past projection. |
| `ForkSession` | `session_id`, `at_event: EventId` | Timeline "fork session here…" | Logs `fork_session`; pairs with `BackendHost::fork_session(at_seq)`. |
| `OpenFile` | `path`, `line: Option<u32>` | Explorer click; provenance peek; go-to-def | **Rejected if `path` blank.** Logs `open_file`. |
| `RunCommand` | `argv: Vec<String>`, `cwd: Option<String>` | Terminal command; palette "run…" | **Rejected if `argv` empty.** Logs `run_command`; pairs with `BackendHost::run_command`. |
| `Custom` | `name`, `payload: Value` | Extension/HIDE-specific actions (profile switch, pin span, re-run step…) | **Rejected if `name` blank.** Logs `custom.<name>`. **This is the escape hatch** for FE actions without a dedicated variant. |

> **Note on time-travel naming.** The built API uses `event_id`/`at_event` (typed `EventId`) on `ScrubToEvent`/`ForkSession`. The host methods `scrub_to_event(seq)` / `fork_session(at_seq)` operate on the numeric `seq`. The FE carries the `EventId` in the intent; the host resolves it. Don't invent an "at_seq" intent field — it doesn't exist.

### 3.3 Wire-B — `UiEvent` and the `UiEventKind` variants (every kind)

`UiEvent { seq: u64, session_id: Option<SessionId>, kind: UiEventKind }`. The FE routes by `kind` (and filters by `session_id` per surface). `seq` is the cursor each store tracks as `last_applied_seq` for replay-on-reconnect.

| `UiEventKind` | Payload | What the FE renders/does |
|---|---|---|
| `ProjectionPatch` | `projection: String`, `patch: Value` | A state-diff for a named panel/projection. Route by `projection` name to the owning store, apply the patch. The general-purpose state-sync path (plan tree, diff state, context manifest, etc.). |
| `TokenBatch` | `stream_id: String`, `text: String` | Coalesced streamed tokens for a stream/session. Append to the chat/run buffer keyed by `stream_id`; the FE rAF-governor commits once per frame. |
| `RuntimeStatus` | `status: String`, `detail: Option<String>` | Serve up/down/degraded. Drives the status-bar runtime pill + a banner on `down`/`degraded`. `status` mirrors the supervisor states: `down`/`booting`/`ready`/`degraded`/`failed`. |
| `ToolProgress` | `call_id: String`, `message: String` | Live tool-call progress chip (in chat + timeline). The host publishes one per dispatched tool result. |
| `SecurityGate` | `gate: String`, `message: String` | An approval is needed (sandbox/permission gate). FE shows an approval prompt; the user's decision goes back as an intent (`Custom` or an Accept/Reject). |
| `Error` | `code: String`, `message: String` | Route to the notification + status stores; non-fatal inline, fatal as a banner. |
| `Custom(Value)` | free `Value` | Extension-defined events; route by an agreed discriminator inside the value. |

### 3.4 The `BackendHost` method surface (what the Tauri commands wrap)

The Tauri layer exposes these (already real on `host.rs`) as `#[tauri::command]`s or uses them internally:

| Host method | Signature (abbrev.) | FE use |
|---|---|---|
| `open_workspace(root)` | `-> Result<Self>` | App boot: open a project. |
| `subscribe_ui()` | `-> broadcast::Receiver<UiEvent>` | Bridged to the `ipc::Channel<UiEvent>` (Wire-B). |
| `handle_intent(Intent)` | `-> Result<IntentAck>` | Backs `invoke('hide_intent')` (Wire-A). |
| `call_connector(id, method, params)` | `(&str,&str,Value) -> Result<Value>` | Backs `callConnector` (§3.5). |
| `fleet_run(session, objective)` | `-> Result<String>` | Workstation: schedule a parallel kernel run; returns terminal status. |
| `generate_and_publish(session, base_url, prompt)` | `-> Result<String>` | Drives generation through the runtime client; publishes `TokenBatch`es onto Wire-B. |
| `scrub_to_event(session, seq)` | `-> Result<SessionProjection>` | Timeline scrub (read-only past view). |
| `fork_session(from, at_seq)` | `-> Result<(SessionId, SessionProjection)>` | Timeline fork. |
| `run_agent_to_terminal(session, objective, max_steps)` | `-> Result<AgentState>` | Drive a run to a terminal phase (Chat/IDE turn). |
| `run_command(session, argv, cwd)` | `-> Result<ToolResult>` | Terminal/command execution (shell.run tool). |
| `status()` | `-> BackendStatus` | Boot/settings: workspace root, capabilities, connector statuses, tool specs, model roles. |
| `health()` | `-> HealthReport` | Health panel: per-component Ok/Degraded/Failed checks. |
| `ui_events(session, after_seq, limit)` | `-> Result<Vec<UiEvent>>` | **Pull** catch-up/replay (the durable-log path) — used on reconnect to fill the gap before live Wire-B resumes. |

### 3.5 The connectors (`call_connector(id, method, params)`)

Connectors are the typed RPC surface for non-intent data the FE needs (search, context, roles…). Registered in `connectors.rs`; all reachable via `BackendHost::call_connector`. Methods take/return `serde_json::Value`.

| Connector `id` | Methods | What it powers in the FE |
|---|---|---|
| `runtime` | `roles.list`, `route` | Model role list (Context Stack "Model" panel, settings); routing decision preview (greedy/sampled, grammar). |
| `code_index` | `search`, `definition`, `references`, `file.add_text`, `file.index`, `health` | Search surface (IDE); go-to-def / find-refs; provenance/index health. |
| `context` | `compile` (→ `{ prompt, manifest }`) | The Context Stack: compile a prompt + `ContextManifest` for a task; the manifest is the rail's data source. Params: `task`, `max_input_tokens`, `search_limit`, optional `role`. |
| `personalization` | `records.list`, `records.append`, `records.by_task` | Logging accepted/rejected diffs (the flywheel corpus); personalization views. |
| `research` | `runs.list`, `runs.latest`, `runs.append`, `runs.by_state` | Research Lab surfaces (post-shell, but the connector is live). |

### 3.6 The supervisor / runtime-status surface

`hawking serve` is booted and supervised by the `RuntimeSupervisor` inside the host. Its state machine is `Down → Booting → Ready → Degraded → Failed` (with restart/backoff). The FE never talks to the supervisor directly — it **observes** it via `RuntimeStatus` UiEvents (Wire-B) and the `status()`/`health()` snapshots. The FE responsibilities:

- **Status-bar runtime pill** bound to the latest `RuntimeStatus.status`; click-through to detail.
- **Degraded/down banners**: on `degraded`/`failed`/`down`, show a non-modal banner; the host auto-restarts, so the banner clears on the next `ready`.
- **Gate the composer**: while `status != ready`, `SubmitTurn` may be rejected upstream — reflect "runtime not ready" in the UI rather than spinning.

### 3.7 The IPC client surface

The FE's single seam to the host — a concrete TS IPC client. Everything else (stores, router) sits on top of this. This is the **canonical IPC client surface** sibling docs point to; the store-wiring *build steps* live in [`03-build-sequencing.md`](03-build-sequencing.md) §2. (Types are TS mirrors of the Rust `serde` wire shapes in `api.rs`.)

```ts
// wire.ts — TS mirrors of crates/hide-core/src/api.rs
export type SessionId = string; export type RunId = string; export type EventId = string;
export type BlobRef = { /* mirror hide-core types::BlobRef */ };

export type Intent =
  | { type: "submit_turn";   data: { session_id: SessionId; text: string; attachments: BlobRef[] } }
  | { type: "cancel_run";    data: { run_id: RunId } }
  | { type: "pause_run";     data: { run_id: RunId } }
  | { type: "resume_run";    data: { run_id: RunId } }
  | { type: "accept_diff";   data: { run_id: RunId; diff_id: string } }
  | { type: "reject_diff";   data: { run_id: RunId; diff_id: string } }
  | { type: "scrub_to_event";data: { session_id: SessionId; event_id: EventId } }
  | { type: "fork_session";  data: { session_id: SessionId; at_event: EventId } }
  | { type: "open_file";     data: { path: string; line: number | null } }
  | { type: "run_command";   data: { argv: string[]; cwd: string | null } }
  | { type: "custom";        data: { name: string; payload: unknown } };

export type IntentAck = { accepted: boolean; event_seq: number | null; message: string | null };

export type UiEventKind =
  | { type: "projection_patch"; data: { projection: string; patch: unknown } }
  | { type: "token_batch";      data: { stream_id: string; text: string } }
  | { type: "runtime_status";   data: { status: string; detail: string | null } }
  | { type: "tool_progress";    data: { call_id: string; message: string } }
  | { type: "security_gate";    data: { gate: string; message: string } }
  | { type: "error";            data: { code: string; message: string } }
  | { type: "custom";           data: unknown };

export type UiEvent = { seq: number; session_id: SessionId | null; kind: UiEventKind };
```

```ts
// ipc.ts — the ONLY module that touches Tauri invoke/Channel.
import { invoke, Channel } from "@tauri-apps/api/core";

/** Wire-A: send an intent, get the host's ack (which may be a rejection). */
export async function sendIntent(intent: Intent): Promise<IntentAck> {
  return invoke<IntentAck>("hide_intent", { intent });
}

/** Wire-B: subscribe to the ordered UiEvent stream. Returns an unsubscribe fn. */
export function onUiEvent(handler: (ev: UiEvent) => void): () => void {
  const channel = new Channel<UiEvent>();
  channel.onmessage = handler;            // ordered; host-side coalesced + backpressured
  void invoke("hide_subscribe_ui", { channel });
  return () => void invoke("hide_unsubscribe_ui", { channel });
}

/** Typed RPC to a backend connector (runtime/code_index/context/personalization/research). */
export async function callConnector<T = unknown>(
  id: "runtime" | "code_index" | "context" | "personalization" | "research",
  method: string,
  params: unknown,
): Promise<T> {
  return invoke<T>("hide_call_connector", { id, method, params });
}
```

> The Tauri command names (`hide_intent`, `hide_subscribe_ui`, `hide_call_connector`) are the FE↔host contract the Tauri wrapper must expose; they're the only strings hard-coded outside `ipc.ts`. (`onUiEvent`'s unsubscribe path also invokes `hide_unsubscribe_ui`.) Everything above the IPC client (router, stores, components) imports only `sendIntent`/`onUiEvent`/`callConnector`.

### 3.8 The Custom-name registry (canonical)

`Intent::Custom{name, payload}` is the escape hatch for every steer/observe action the built `Intent` enum has no dedicated variant for, and `ProjectionPatch{projection}` is the named state-diff for every panel slice. Because both are string-keyed, **the host and FE must agree on the exact string for each logical action / slice.** This is the canonical registry; the surface doc ([`01-surfaces.md`](01-surfaces.md)) introduces these in context and points back here. **Don't add a `Custom` name or a `projection` discriminator anywhere without adding it to this table.**

**`Intent::Custom{name}` values:**

| `name` | Sent by (surface) | Action |
|---|---|---|
| `save_file` | AI IDE (Editor) | persist an editor buffer |
| `inline_edit` | AI IDE (Editor) | Cmd+K agentic inline edit |
| `mention_in_chat` | AI IDE (Explorer) | add a file as a chat context source |
| `pty_input` / `pty_resize` | AI IDE (Terminal) | terminal stdin / resize over the PTY mirror |
| `run_search` | AI IDE (Search) | issue a search query |
| `quick_fix` | AI IDE (Problems) | apply a diagnostic quick-fix |
| `revert_diff` | AI IDE (Diff Review) | undo an applied diff (compensating event) |
| `edit_hunk` | AI IDE (Diff Review) | edit the modified side before accept |
| `queue_turn` | AI Chat (Composer) | append a turn to the prompt queue |
| `redirect_run` | AI Chat (Composer/steer) | redirect a running turn |
| `approve_plan` | AI Chat (PlanCard) | approve a proposed plan |
| `edit_plan_step` | AI Chat (PlanCard) | edit a plan step |
| `reorder_plan` | AI Chat (PlanCard) | reorder plan steps |
| `rerun_step` | AI Workstation (Timeline) | re-run a timeline step |
| `fleet_run` | AI Workstation (Fleetview) | schedule a parallel kernel run (→ `fleet_run()`) |
| `resolve_conflict` | AI Workstation (Merge-review) | choose a merge-conflict resolution |
| `pin_span` / `unpin_span` | Context Stack | pin / unpin a context span into the next turn |
| `switch_profile` | Context Stack | change the model profile |
| `toggle_confidence` | Context Stack | toggle per-token confidence heat |
| `resolve_conflict` | Context Stack | resolve a context contradiction |
| `approve_gate` | Notifications / any panel | approve a `SecurityGate` (when not an Accept/Reject) |
| `focus_run` / `dismiss` | Notifications | focus / dismiss a notification |

**`ProjectionPatch{projection}` discriminators** (the panel-slice names the FE routes on after `kind`):
`turn`, `plan`, `tool`, `diff_chip` (chat); `diff`, `file_external`, `editor` (IDE); `context_manifest`, `retrieval`, `memory` (Context Stack); `timeline` (Agent-Run Timeline, universal); `build`, `test`, `diagnostics` (Problems); `sourcecontrol` (checkpoints); `fleet`, `run`, `merge` (Workstation); `turn_ended`, `plan_waiting` (Notifications); `status` (Status Bar). The set is owned jointly by host + FE; the per-panel binding map lives in [`01-surfaces.md`](01-surfaces.md) §A.4.

---

## 4. How the three surfaces map onto the contract

Each surface is a composition of intents it sends, UiEvent kinds it consumes, and connectors it calls. (Detailed component shapes live in the sibling surface doc; this is the binding map.)

| | **AI IDE** | **AI Chat** | **AI Workstation** |
|---|---|---|---|
| **Sends (Intent)** | `OpenFile`, `RunCommand`, `AcceptDiff`/`RejectDiff`, `Custom`(inline-edit, ghost-text) | `SubmitTurn`, `PauseRun`/`ResumeRun`/`CancelRun`, `ScrubToEvent`/`ForkSession`, `Custom`(redirect, edit-plan) | `SubmitTurn` (fan-out objectives), `PauseRun`/`ResumeRun`/`CancelRun` per run |
| **Consumes (UiEventKind)** | `ProjectionPatch`(diff/editor/files), `ToolProgress`, `RuntimeStatus`, `Error` | `TokenBatch`, `ProjectionPatch`(plan/chat), `ToolProgress`, `SecurityGate`, `Error` | `ProjectionPatch`(run state across sessions), `RuntimeStatus`, `Error` |
| **Calls (connector)** | `code_index`(`search`/`definition`/`references`), `runtime` | `context`(`compile`), `runtime`(`route`), `personalization`(log accept/reject) | `runtime`(`roles.list`), (fleet via `BackendHost::fleet_run`) |
| **Host methods** | `run_command`, `scrub_to_event` | `generate_and_publish`, `run_agent_to_terminal`, `fork_session` | `fleet_run`, `status` |

**Context Stack (rail, all surfaces):** consumes `ProjectionPatch{projection:"context*"}` for the live `ContextManifest`; calls `context.compile` to (re)build it; `Custom` intents for pin/drop/profile-switch; reads `runtime.roles.list` for the Model panel. On a Timeline scrub (`ScrubToEvent`), the rail rewinds to that event's manifest — the "what did it see when it decided *that*" superpower.

**Cross-cutting (every surface):** `RuntimeStatus` → status pill + banner; `Error` → notifications; every `UiEvent.seq` advances the owning store's `last_applied_seq` so reconnect replays cleanly (open Channel, request `ui_events(after_seq)` catch-up, resume live).

---

## Decisions sibling docs must stay consistent with

1. **Wire types are fixed by `api.rs`** — `Intent`/`IntentAck`/`UiEvent`/`UiEventKind` exactly as enumerated (snake_case `type`/`data` tagging; time-travel uses `event_id`/`at_event: EventId`, not `at_seq`). Don't introduce new variants in docs; use `Custom{name,payload}` for FE-specific actions.
2. **Three Tauri command strings only:** `hide_intent` (Wire-A: an `Intent` → `IntentAck`), `hide_subscribe_ui` (opens the Wire-B `ipc::Channel<UiEvent>`), `hide_call_connector` (connector RPC: `id`, `method`, `params`). (`hide_unsubscribe_ui` closes a Wire-B channel.) The IPC client (`ipc.ts`) is the sole module that touches `invoke`/`Channel`.
3. **The host owns truth; stores are derived caches** keyed by `last_applied_seq`. Reconnect = open Channel + `ui_events(after_seq)` pull catch-up + live resume.
4. **Two streaming layers:** host-side `UiEventBus` coalescing (per `stream_id`) **and** an FE rAF render-governor. Don't render per token.
5. **Connectors are the non-intent RPC surface** (`runtime`/`code_index`/`context`/`personalization`/`research`); the Context Stack's data comes from `context.compile` → `{prompt, manifest}`.
6. **Runtime is observed, not controlled:** `RuntimeStatus` states are `down/booting/ready/degraded/failed`; gate the composer on `ready`.
