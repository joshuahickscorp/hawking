# The Three Surfaces + the Context Stack

> Part of the **HIDE front-end bible**. The backend is already built and tested (11 Rust crates, a real agent loop); see [SCAFFOLD_STATUS.md](../SCAFFOLD_STATUS.md). This doc is what the team builds the UI skeleton from. It is standalone — it inlines the front-end design that used to live in the archived numbered chapters and binds it to the **real** backend contract (`hide-core::api`, `hide-backend::BackendHost`), not the old design sketch.

---

## 0. The binding contract (read this first)

Everything in this doc talks to the host over two wires plus a connector RPC. These are the **real, built** types — do not invent kinds.

**Wire-A — `Intent` the FE sends** (`crates/hide-core/src/api.rs`, serde `{type, data}`), each returns `IntentAck{accepted, event_seq?, message?}`:

```ts
type Intent =
  | { type: "submit_turn";    data: { session_id: string; text: string; attachments: BlobRef[] } }
  | { type: "cancel_run";     data: { run_id: string } }
  | { type: "pause_run";      data: { run_id: string } }
  | { type: "resume_run";     data: { run_id: string } }
  | { type: "accept_diff";    data: { run_id: string; diff_id: string } }
  | { type: "reject_diff";    data: { run_id: string; diff_id: string } }
  | { type: "scrub_to_event"; data: { session_id: string; event_id: string } }
  | { type: "fork_session";   data: { session_id: string; at_event: string } }
  | { type: "open_file";      data: { path: string; line?: number } }
  | { type: "run_command";    data: { argv: string[]; cwd?: string } }
  | { type: "custom";         data: { name: string; payload: unknown } };

interface IntentAck { accepted: boolean; event_seq?: number; message?: string }
```

`Custom{name,payload}` is the **escape hatch** for every steer/observe action the old design named but the built enum does not have a dedicated variant for (`PinSpan`, `EditPlanStep`, `SwitchProfile`, `RerunStep`, `RevertDiff`, `Dictate`, …). The **canonical registry** of every `Custom` `name` string lives in the contract doc ([`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) §3.8), so the host's future `Custom` router and the FE agree on one string per logical action. **Decision FE-1: new steer verbs are `Custom`, not new enum variants, until the host promotes them.**

**Wire-B — `UiEvent{seq, session_id?, kind}` the FE receives** (`UiEventKind`, serde `{type, data}`):

```ts
interface UiEvent { seq: number; session_id?: string; kind: UiEventKind }
type UiEventKind =
  | { type: "projection_patch"; data: { projection: string; patch: unknown } } // state-diff for a panel
  | { type: "token_batch";      data: { stream_id: string; text: string } }    // coalesced stream
  | { type: "runtime_status";   data: { status: string; detail?: string } }    // up/down/degraded
  | { type: "tool_progress";    data: { call_id: string; message: string } }
  | { type: "security_gate";    data: { gate: string; message: string } }      // approval needed
  | { type: "error";            data: { code: string; message: string } }
  | { type: "custom";           data: unknown };
```

This is the entire surface. The old design's ~30 event kinds (`plan.*`, `diff.*`, `context.manifest`, `memory.written`, `test.status`, …) **all arrive as `ProjectionPatch`** with a `projection` discriminator naming the panel slice (e.g. `"plan"`, `"diff"`, `"context_manifest"`, `"timeline"`). **Decision FE-2: the FE routes first on `kind`, then for `projection_patch` on `data.projection`.** The set of `projection` names is the real binding map (§A.4) and is owned jointly by host + FE.

**The host surface the wires sit on** (`crates/hide-backend/src/host.rs`, `BackendHost`):

| Host method | Used by | Notes |
|---|---|---|
| `open_workspace(root)` | app boot | constructs the host for a workspace |
| `subscribe_ui() -> broadcast::Receiver<UiEvent>` | the IPC bridge | the live Wire-B push stream (ordered; lagging subscriber gets `Lagged`, not a stall) |
| `handle_intent(Intent) -> IntentAck` | every panel | wraps `CommandRouter::handle`; ack-then-events |
| `call_connector(id, method, params) -> Value` | Context Stack, Search, Model | the connector RPC (§0.1) |
| `ui_events(session, after_seq, limit)` | reconnect/catch-up | the **pull** replay API (Wire-B's durable twin) |
| `fleet_run(session, objective)` | Workstation fleetview | schedules a parallel kernel run via the fleet |
| `generate_and_publish(...)` | (host-internal) | drives a completion and publishes `TokenBatch`es |
| `scrub_to_event(seq)` / `fork_session(at_seq)` | Timeline | time-travel (read-only fold / child session) |
| `run_agent_to_terminal(...)` | Chat/Workstation | runs the kernel loop to a terminal phase |
| `status()` / `health()` | status bar, boot gate | capability + connector + tool inventory |

The Tauri seam (not yet built, per SCAFFOLD_STATUS): `CommandRouter::handle` is transport-agnostic; a `#[tauri::command] hide_intent(intent)` wraps it, and the `UiEventBus` (`ui_bus.rs`: `subscribe()`/`publish()`/`publish_token()` with render-coalescing) is bridged to the webview via a `tauri::ipc::Channel<UiEvent>`. **Decision FE-3: Wire-B is the ordered `Channel`, NOT `emit/listen`.** See [`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) §3.7 for the IPC client wrapper, and [`03-build-sequencing.md`](03-build-sequencing.md) §2 for the store wiring.

### 0.1 The connectors (`call_connector(id, method, params)`)

`crates/hide-backend/src/connectors.rs`. These are request/response (not streamed) and back the right-rail + search + model picker:

| Connector `id` | Methods the FE calls | Feeds |
|---|---|---|
| `runtime` | model roles, route/role lookup | Context Stack "Model" section, status pill |
| `code_index` | `search`, `definition`, `references` | Search panel, go-to-definition, retrieval rows |
| `context` | compile prompt + manifest | **Context Stack** (the manifest render) |
| `personalization` | profile/preferences | profiles, autonomy defaults |
| `research` | `runs.list`, `runs.append`, … | Research tab |

---

## A. The workbench shell

The frame all three surfaces live in: a **six-region workbench** on a dockable, splittable pane model, defaulting to a calm three-column layout that opens into full observability on demand. (Design rationale, now archived under ../archive/.)

### A.1 The six regions

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ TITLE BAR   hide ▸ project ▸ branch   ·  [runtime ● Ready 41 tps]  · ⌘K palette│
├──┬────────────────────┬──────────────────────────────────┬────────────────────┤
│A │ PRIMARY SIDEBAR    │  EDITOR GROUP(S) (tabbed, split)  │  CONTEXT STACK      │
│C │ (swappable viewlet)│  ┌────────────────────────────┐  │  (RIGHT RAIL — the  │
│T │  ▸ Explorer        │  │ auth.rs ✎ │ pool.rs │ +Diff │  │   differentiator)   │
│I │  ▸ Search          │  ├────────────────────────────┤  │  ▸ Model            │
│V │  ▸ Source Control  │  │   Monaco editor / diff     │  │  ▸ Budget ▓▓▓▓▓░    │
│I │  ▸ Agent Runs      │  │   (ghost-text, inline-edit,│  │  ▸ Retrieved (6) ▸  │
│T │  ▸ Memory*         │  │    hunk gutters)           │  │  ▸ Symbols / Tools  │
│Y │  ▸ Chat (or → R)   │  │                            │  │  ▸ Dropped (12) ▸   │
│  │                    │  └────────────────────────────┘  │  ──────────────     │
│BAR                    │                                  │  CHAT (default dock)│
├──┴────────────────────┴──────────────────────────────────┴────────────────────┤
│ BOTTOM PANEL:  Terminal │ Problems │ Test Output │ Agent Timeline │ Output      │
├──────────────────────────────────────────────────────────────────────────────┤
│ STATUS BAR:  ⎇ branch · ⚠2 ●0 · Ln42 Col8 · Rust · [agent: planning…⏸] · 41tps │
└──────────────────────────────────────────────────────────────────────────────┘
```

| # | Region | Default | Toggle |
|---|---|---|---|
| 1 | **Activity Bar** | left edge | viewlet switchers + badges (run count, problems, notifications) |
| 2 | **Primary Sidebar** | left ~260px | `Cmd+B`; one active viewlet; remembers width per viewlet |
| 3 | **Editor Group(s)** | center, fills | `Cmd+\` split; drag-tab to split; grid of tab groups |
| 4 | **Context Stack** (right rail) | right ~320px, **on by default** | `Cmd+Alt+B`; Chat docks beneath it |
| 5 | **Bottom Panel** | bottom ~30% | `Cmd+J`; tabbed; Agent Timeline lives here |
| 6 | **Status Bar** | bottom edge | always; agent-state pill is a global steer affordance |

The three *surfaces* in the brief are **arrangements of these regions**, not separate windows: **AI IDE** = Explorer + Editor + Bottom Panel (§B); **AI Chat** = Chat dock foregrounded (§C); **AI Workstation** = Agent Runs viewlet + Agent Timeline + fleetview (§D). They share one shell, one store fabric, one event stream. The **Context Stack** (§E) is present in all three.

### A.2 Pane model + layout store

```ts
type DockId = "sidebar.primary" | "rail.context" | "panel.bottom" | "editor.grid";
type LayoutNode =
  | { kind: "leaf";  panelId: string; tabs: string[]; activeTab: string }
  | { kind: "split"; orientation: "horizontal" | "vertical"; children: LayoutNode[]; sizes: number[] };
interface WorkspaceLayout {
  schema_version: 1;
  editorGrid: LayoutNode;
  docks: Record<DockId, { open: boolean; size_px: number; activeViewlet?: string }>;
  panelOrder: string[];
  floats: { panelId: string; windowLabel: string }[];   // torn-off panels (Tauri multi-window)
}
```

`layoutStore` owns this. It is **view-state, not log-state** — it persists to `<workspace>/.hide/` (next to the host's state), NOT through the event log. **Decision FE-4: layout never round-trips through Wire-A/B.** Tear-off uses Tauri multi-window; the new window re-subscribes via its own `Channel<UiEvent>`. Named presets (`Focus`, `Code`, `Agent`, `Review`) are commands.

### A.3 Command palette + keyboard model

Keyboard-first: every action is a command with a palette entry and a re-bindable key. `commandStore` holds the registry (read-only, not event-fed). The palette (`Cmd+Shift+P`) fuzzy-searches commands; Quick Open (`Cmd+P`) takes mode prefixes in one box — `>` commands, `@` file symbols, `#` workspace symbols (via `code_index` connector), `:` go-to-line, **`§` agent actions** (HIDE-specific: "scrub to event…", "fork session here…", "re-run step…", "switch profile…"). The `§` namespace is where Timeline/Context-Stack actions become keyboard-reachable. Seed keymap in §A.5.

### A.4 The panel inventory + the binding map (real names)

**v1 shell** (build first): Editor, Chat, Agent-Run Timeline, Diff Review, **Context Stack**, Terminal, File Explorer, Search, Command Palette, Status Bar, Problems, Notifications.
**Later** (designed, not v1): Memory viewlet/editor (`Custom` pin path exists in v1 via the rail), Test Explorer tree, Model Lab (placeholder, gated on HF distribution / 32B `.tq`), Research tab (§F), multiplayer presence, voice composer, energy dashboard.

Binding map — **mapped to the REAL `UiEventKind` + the `projection` discriminator**:

| Panel | Receives (`UiEventKind` → `projection`) | Sends (`Intent`) | Store(s) |
|---|---|---|---|
| **Editor (Monaco)** | `projection_patch:{diff, file_external}`, `token_batch` (inline/confidence) | `OpenFile`, `Custom:save_file`, `Custom:inline_edit`, `AcceptDiff`/`RejectDiff` (gutter) | `editorStore` |
| **Chat** | `token_batch`, `projection_patch:{turn, plan, tool, diff_chip}`, `error` | `SubmitTurn`, `Custom:queue_turn`, `Custom:redirect_run`, `Custom:approve_plan`, `Custom:edit_plan_step`, `PauseRun`/`ResumeRun`/`CancelRun` | `chatStore`, `runStore` |
| **Agent-Run Timeline** | **all** `kind`s (universal); ordered by `seq` | `ScrubToEvent`, `ForkSession`, `ResumeRun`, `Custom:rerun_step` | `timelineStore`, `runStore` |
| **Diff Review** | `projection_patch:{diff, file_external}` | `AcceptDiff`, `RejectDiff`, `Custom:revert_diff`, `Custom:edit_hunk` | `diffStore`, `sourceControlStore`, `editorStore` |
| **Context Stack** | `projection_patch:{context_manifest, retrieval, memory}`, `runtime_status`, `token_batch` (confidence) | `Custom:{pin_span, unpin_span, resolve_conflict, switch_profile, toggle_confidence}`; reads via `context`/`runtime` connectors | `contextStore`, `runtimeStore` |
| **Terminal** | PTY data (direct Tauri channel), `tool_progress` (agent-shell mirror) | `RunCommand`, `Custom:pty_input`, `Custom:pty_resize` | `terminalStore` |
| **File Explorer** | `projection_patch:{file_external, diff}`, `tool_progress` (file refs) | `OpenFile`, `Custom:mention_in_chat` | `fileTreeStore`, `sourceControlStore` |
| **Search** | results via `code_index` connector | `Custom:run_search`, `OpenFile` | `searchStore` |
| **Status Bar** | `runtime_status`, `projection_patch:{turn, plan}` | `PauseRun`, `CancelRun` | `statusStore` |
| **Problems** | `projection_patch:{build, test, diagnostics}` | `OpenFile`, `Custom:quick_fix` | `diagnosticsStore` |
| **Notifications** | `projection_patch:{turn_ended, test, plan_waiting}`, `runtime_status`, `error`, `security_gate` | `Custom:focus_run`, `Custom:dismiss` | `notificationStore` |
| **Workstation / fleetview** | `projection_patch:{fleet, run, merge}` | `Custom:fleet_run` → `fleet_run()`; `PauseRun`/`CancelRun` per run | `fleetStore`, `runStore` |

`SecurityGate` is consumed by **Notifications + Chat + the relevant panel** (it's an approval prompt: "shell.run needs approval", "diff write to protected path"); approving sends the gated intent (e.g. `AcceptDiff`, `ResumeRun`, or a `Custom:approve_gate{gate}`). **Decision FE-5: `SecurityGate` is a blocking modal/inline prompt, never auto-dismissed.**

### A.5 Seed keymap

| Action | Key (`Cmd`=⌘) | `when` |
|---|---|---|
| Command Palette / Quick Open / Agent actions | `Cmd+Shift+P` / `Cmd+P` / `Cmd+Shift+§` | always |
| Toggle Sidebar / Context Stack / Bottom Panel | `Cmd+B` / `Cmd+Alt+B` / `Cmd+J` | always |
| Focus Chat / Selection→Chat | `Cmd+Shift+L` / `Cmd+L` | always / editor |
| Inline Edit | `Cmd+K` | editor selection |
| Accept ghost-text / word | `Tab` / `Cmd+Right` | suggestion visible |
| Accept hunk / Reject hunk | `Cmd+Enter` / `Cmd+Backspace` | diff focus |
| Approve plan | `Cmd+Enter` | plan focus |
| Queue turn / Override turn | `Cmd+Enter` / `Shift+Cmd+Enter` | composer && running |
| Interrupt agent | `Esc Esc` | running |
| Why? (provenance) | `Cmd+Alt+/` | editor |
| Split editor / Save / Close tab | `Cmd+\` / `Cmd+S` / `Cmd+W` | editor |
| Timeline scrub back/fwd/live/fork | `Cmd+[` / `Cmd+]` / `Cmd+End` / `Cmd+Shift+F` | timeline |

### A.6 Theming

One token set (CSS variables) drives the shell, Monaco (its theme API, synced), and xterm (its theme object, synced) so they never drift. Ships dark (default) + light + high-contrast. Accessibility is first-class: keyboard-complete, Monaco accessible mode, ARIA live regions that announce on `token_batch`/turn boundaries (not per token), `prefers-reduced-motion` honored, diff/plan/confidence use shape+label not color alone.

---

## B. The AI IDE surface (editor / diff / files / terminal)

Monaco-centered editing with agent affordances, plus Diff Review, Explorer, Terminal, Search, Status Bar. (Design rationale, now archived under ../archive/.)

### B.1 Component tree

```
<IdeSurface>
 ├─ <ActivityBar/>                      // viewlet switch + badges
 ├─ <PrimarySidebar viewlet>            // Explorer | Search | SourceControl
 │   ├─ <FileExplorer onOpen=(p)=>intent.openFile(p)/>
 │   ├─ <SearchPanel onMatch=(p,l)=>intent.openFile(p,l)/>
 │   └─ <SourceControl/>
 ├─ <EditorGrid layout={layoutStore.editorGrid}>
 │   └─ <EditorGroup tabs activeTab>
 │       ├─ <MonacoEditor model decorations ghostText inlineEditWidget/>
 │       └─ <MonacoDiffEditor diff={diffStore.get(diffId)} onHunk=…/>   // diff tab
 ├─ <BottomPanel tab>
 │   ├─ <Terminal xterm onData onResize/>     // human PTY + agent-shell mirror
 │   ├─ <ProblemsPanel/>  <TestOutput/>  <AgentTimeline/>   // §D
 └─ <StatusBar agentPill tps problems cursor onPause onCancel/>
```

Key props sketch:

```ts
interface MonacoEditorProps {
  modelUri: string; language: string;
  decorations: Decoration[];                 // hunk gutters, confidence heat
  ghostText?: { range: Range; text: string };
  inlineEditWidget?: { anchor: Range; status: "prompt"|"generating"|"overlay" };
  onInlineEditSubmit(instruction: string): void;   // → Custom:inline_edit
  onAcceptCompletion(): void; onAcceptDiffHunk(diffId: string, hunkId: string): void;
}
interface DiffReviewProps {
  diffId: string; hunks: Hunk[]; mode: "side-by-side" | "inline"; editableModified: boolean;
  onAccept(hunkId?: string): void;  // hunkId omitted ⇒ whole file → AcceptDiff{run_id,diff_id}
  onReject(hunkId?: string): void;  // → RejectDiff
  onRevert(): void;                 // → Custom:revert_diff
}
```

### B.2 Store slices owned

- `editorStore` — open Monaco models, decorations, `ghostText`, `inlineEditWidget`, `confidenceHeat`. Fed by `projection_patch:{diff, file_external}` and `token_batch`.
- `diffStore` — `diffs{ diff_id → { path, hunks[], status } }`. Fed by `projection_patch:diff`.
- `sourceControlStore` — review aggregate (multibuffer), checkpoint list, git status.
- `fileTreeStore` — tree, `touchedByRun`, git badges. Fed by `projection_patch:{file_external,diff}` + `tool_progress`.
- `terminalStore` — xterm instances, `agentSessionId`.
- `searchStore`, `diagnosticsStore`, `statusStore`.

### B.3 Intents sent / events consumed

- **Editor**: sends `OpenFile`, `Custom:save_file`, `Custom:inline_edit`, `AcceptDiff`/`RejectDiff` (gutter). Consumes `projection_patch:{diff,file_external}`, `token_batch`.
- **Diff Review**: the agent's edits arrive as `projection_patch:diff` (a `diff_id` with hunks); per-hunk **Accept** → `AcceptDiff{run_id,diff_id}` (the host applies; a follow-up `projection_patch:diff` flips status to `applied`). **Reject** → `RejectDiff`. **Undo** → `Custom:revert_diff` (a compensating event upstream). Multibuffer review aggregates every `diff_id` for a `run_id` into one scroll (`sourceControlStore`). The modified side is editable before accept; the edit becomes a user-authored revision.
- **Terminal**: human input over a direct Tauri PTY channel (not Wire-A); `RunCommand{argv,cwd}` for agent/explicit commands (host routes to `shell.run`, recorded as a `tool.call`/`tool.result` pair and mirrored back as `tool_progress`).
- **Explorer/Search**: `OpenFile`; Search queries the `code_index` connector directly.

### B.4 State machine — inline-edit / ghost-text

```
            type in editor
  Idle ────────────────────────▶ GhostPending ──(provider returns)──▶ GhostShown
   ▲  ◀──Esc / cursor moves────────┘                                     │
   │                                                  Tab(accept) │ Esc(dismiss)
   │                                                       ▼       ▼
   │                                                   Accepted   Idle      (GhostShown is LOCAL —
   │                                                   (insert,             no event until accepted)
   │   Cmd+K on selection                              no event)
   └────────────▶ EditPrompt ──submit──▶ EditGenerating ──(projection_patch:diff)──▶ EditOverlay
         ▲ Esc(cancel)   │                  │ Esc(abort → CancelRun)                    │
         └───────────────┘                  ▼                              ┌────────────┴───────────┐
                                          Aborted                   Accept→AcceptDiff→applied   Reject→RejectDiff
                                                                     → Idle                      → Idle
```

Ghost-text is pre-commit/local (cheap completions stay off the log); the `Cmd+K` path is **agentic** — it produces a `projection_patch:diff` that is reviewable/undoable. Abort flips the host's `abort` flag via `CancelRun`.

### B.5 State machine — diff-review (per file + aggregate)

```
  projection_patch:diff (proposed)
       │
       ▼
   Proposed ──open tab / inline gutters──▶ Reviewing ──edit modified side──▶ Reviewing'(user-edited)
       │                                       │
       │ projection_patch:file_external        │
       ▼                                        ├─Accept hunk→AcceptDiff{hunk}→PartiallyApplied─loop─▶ Applied
    Stale ──re-diff/rebase──▶ Reviewing         ├─Accept all →AcceptDiff{file}─────────────────────▶ Applied
                                                └─Reject     →RejectDiff──────────────────────────▶ (no apply)
                                                                                       Applied ──Undo──▶
                                                                              Custom:revert_diff → Reverted
```

`Stale` handles the file changing under a pending diff (`projection_patch:file_external`) — never apply onto drifted content. Checkpoints are the events themselves (the host's CAS post-image); reverts are compensating events, and because shell tool calls are also recorded, a revert can honestly surface "these terminal commands also ran after this point" (the Cursor beat).

---

## C. The AI Chat surface

The conversation + plan + steer surface. Default docked beneath the Context Stack; can dock to the sidebar. (Design rationale, now archived under ../archive/.)

### C.1 Component tree

```
<ChatSurface>
 ├─ <MessageList>
 │   ├─ <UserMessage/> <AssistantMessage streaming/>
 │   ├─ <PlanCard steps editable onApprove onEditStep onReorder/>      // plan card
 │   ├─ <ToolChip callId status onExpand/>     // from tool_progress
 │   ├─ <DiffChip diffId onOpen=()=>openDiffReview(diffId)/>
 │   └─ <SecurityGatePrompt gate message onApprove onDeny/>
 ├─ <QueuedTurnChips turns/>                   // prompt queue
 └─ <Composer mode={idle|running}>
     // idle: textarea + @mention + /slash + model/profile picker + submit
     // running ⇒ STEER BAR: ⏸ Pause · ⤺ Redirect · ✎ Edit plan · ⛔ Stop
```

```ts
interface ComposerProps {
  running: boolean;
  onSubmit(text: string, attachments: BlobRef[]): void;     // → SubmitTurn
  onQueue(text: string): void;                              // → Custom:queue_turn (append)
  onOverride(text: string): void;                           // → Custom:redirect_run / Shift+Cmd+Enter
  onPause(): void; onResume(): void; onStop(): void;        // → PauseRun / ResumeRun / CancelRun
}
interface PlanCardProps {
  steps: { id: string; title: string; status: "pending"|"active"|"done"|"failed"|"skipped" }[];
  onApprove(): void;                                        // → Custom:approve_plan
  onEditStep(id: string, title: string): void;             // → Custom:edit_plan_step
  onReorder(order: string[]): void;                         // → Custom:reorder_plan
}
```

### C.2 Stores / intents / events

- **Stores:** `chatStore` (messages, streaming buffers per `stream_id`, composer, queued turns, plan cards), `runStore` (the active-run FSM, §C.4).
- **Streaming:** assistant tokens arrive as `token_batch{stream_id,text}`. The FE keys the streaming buffer on `stream_id` and appends; a render-rate governor flushes one React commit per animation frame (§C.3) so a 120 tok/s stream never thrashes the UI. (`generate_and_publish` in `host.rs` is the host side that emits these, coalesced.)
- **Turn model:** a user message → `SubmitTurn{session_id, text, attachments}` → `IntentAck{event_seq}`; the assistant turn streams back as `token_batch` + `projection_patch:{turn,plan,tool}`. `@`-mentions add a pinned context source (a `Custom:pin_span` alongside the turn); `/`-slash-commands dispatch their command's intent.
- **Steer:** `PauseRun`/`ResumeRun`/`CancelRun` are real enum variants. Redirect/edit-plan/queue are `Custom` (FE-1). `SecurityGate` renders inline as an approval prompt.

### C.3 The token governor (the one piece of real FE plumbing here)

```ts
const buffers = new Map<string /*stream_id*/, string>();
let scheduled = false;
function onTokenBatch(ev: { stream_id: string; text: string }) {
  buffers.set(ev.stream_id, (buffers.get(ev.stream_id) ?? "") + ev.text);
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(() => {
      for (const [sid, text] of buffers) chatStore.getState().appendStream(sid, text);
      buffers.clear(); scheduled = false;
    });
  }
}
```

One paint per frame, zero dropped content (the host already coalesced upstream; this is purely a paint optimization). Lives in the shared IPC client ([`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) §3.7) with the store wiring in [`03-build-sequencing.md`](03-build-sequencing.md) §2, not in Chat alone — the Timeline and Context-Stack confidence heat consume the same `token_batch`.

### C.4 State machine — plan-steer

```
  SubmitTurn / turn
     │
     ▼
  Planning ──(projection_patch:plan)──▶ PlanReady ──[autonomy=suggest-only]──▶ AwaitingApproval
     │                                     │                                        │
     │ [autonomy=auto]                     │                          Approve▶  Edit/Reorder   Reject
     ▼                                     │                          Executing  (→PlanReady)   →Idle
  Executing ◀───────────────────────────-─┘
     │   ├─(projection_patch:plan step active/done) render
     │   ├─ Pause (⏸ / Esc Esc) ─▶ Paused ──Resume──▶ Executing       // PauseRun / ResumeRun
     │   ├─ Redirect(text) ─▶ Replanning ─▶ Executing                  // Custom:redirect_run
     │   ├─ Stop (⛔) ─▶ Stopped                                       // CancelRun
     │   ├─ SecurityGate ─▶ AwaitingApproval ──approve──▶ Executing    // per-step / risky-action
     │   └─(turn ended) ─▶ Done
     ▼
  [from any past state]  scrub/fork via Timeline → replay (read-only) or fork (child session)
```

`runStore` owns this per `run_id`. `AwaitingApproval` is reached both by `autonomy=suggest-only` plans and by `SecurityGate` events; both resolve through the same approval affordance (status pill, steer bar, or inline prompt). **Decision FE-6: the run FSM is the single source of the agent-state pill** that the status bar (§B), notifications, and fleetview all read.

---

## D. The AI Workstation surface (parallel agents)

The Agent-Run Timeline, fleetview (multi-run grid), parallel-agent orchestration, and merge-review. (Design rationale, now archived under ../archive/.)

### D.1 The Agent-Run Timeline (scrub / replay / step cards)

Turns the host's append-only event log into a visual, scrubbable filmstrip — the observability spine no cloud agent or TUI has.

```
<AgentTimeline mode={live|review}>
 ├─ <ScrubSlider seq onScrub=(seq)=>intent.scrubToEvent(seq)/>
 ├─ <StepCardLane cards={timelineStore.cards}>
 │    <StepCard kind cause onOpen onWhy/>   // Turn|Plan|Tool|Diff|Test|Context|Runtime|Error
 └─ <ReplayBar onResume=()=>intent.resumeRun() onFork=(seq)=>intent.forkSession(seq) onWhy/>
```

Step-card kinds map to `UiEvent`s by `(kind, projection)`: **Turn/Plan** ← `projection_patch:{turn,plan}`; **Thinking** ← `token_batch`; **Tool** ← `tool_progress`; **Diff** ← `projection_patch:diff`; **Test/Build** ← `projection_patch:{test,build}`; **Context** ← `projection_patch:context_manifest`; **Runtime** ← `runtime_status`; **Error** ← `error`. The Timeline is the **universal consumer** — it subscribes to every `kind`. Cards thread by `cause` (the host's `Event.cause`, surfaced as `tool_progress.call_id`/projection ids) so a card shows "← caused by Plan step 3 ← Turn 1" (the provenance spine behind "Why?" — `Cmd+Alt+/`).

**Scrub / replay / fork** bind to the real host methods:

- **Scrub** → `ScrubToEvent{session_id, event_id}` → host `scrub_to_event(seq)` returns a read-only `SessionProjection` folded to that seq; the FE rebuilds editor buffers, plan tree, diffs, and the Context Stack **as they were at that seq**. A "replay mode" banner makes it read-only; effects never re-fire (the host fold is pure).
- **Resume from here** → `ResumeRun` → live execution re-attaches and appends new events from `seq` onward.
- **Fork from here** → `ForkSession{session_id, at_event}` → host `fork_session(at_seq)` returns a `(SessionId, SessionProjection)` child; original intact. **Edit-then-fork** (scrub → edit plan/pin via `Custom` → fork) re-executes deterministically from the edited state — the signature demo.

```ts
interface TimelineStore {
  cards: StepCard[];            // windowed, ordered by seq, threaded by cause
  scrubSeq: number | null;
  mode: "live" | "review";
  lastAppliedSeq: number;       // reconnect cursor → ui_events(after_seq)
}
```

Scrub beyond the in-memory window requests a log range via the **pull** API `ui_events(session, after_seq, limit)`; live tail auto-scrolls to head, detaches into review on scroll-back with a "Jump to live ⟶".

### D.2 Fleetview (multi-run grid)

The home for many parallel agents (fan out 8 refactors, watch them here). Launched via `fleet_run(session, objective)`; each cell is a run with a state pill, active plan step k/n, elapsed, last event, resource band, and quick `⏸/⛔/open`.

```
<Workstation>
 ├─ <AgentRunsViewlet>                          // sidebar list of all sessions/runs
 ├─ <FleetGrid runs={fleetStore.runs}>
 │    <RunCell pill activeStep k n elapsed lastEvent onPause onCancel onOpen/>
 └─ <MergeReview aggregate={fleetStore.mergeAggregate}/>   // §D.3
```

- **Launch:** `Custom:fleet_run{objective, pattern?}` → host `fleet_run()` schedules a parallel kernel run under the FleetGovernor (worktree-isolated, admission-gated). The orchestration *patterns* (single, fan-out/map-reduce, pipeline, tournament/best-of-N, planner→workers→merger, speculative) are chosen by the planner or the user; the FE renders the chosen pattern and per-run status — it does **not** decide the pattern (backend `hide-fleet`).
- **Store:** `fleetStore` (`runs{ id → status }`, `mergeAggregate`), fed by `projection_patch:{fleet, run, merge}`. Per-run control reuses `PauseRun`/`CancelRun`.
- **Resource honesty:** fleetview surfaces the per-run resource band (RAM/thermal headroom) the Governor reports, so the user sees why a run is queued vs admitted.

### D.3 Merge-review

When parallel runs finish, results funnel through an **integration branch** (fan-out: combine all disjoint-footprint results; tournament: oracle-first select one winner). The FE renders the funnel outcome:

```
<MergeReview>
 ├─ <IntegrationStatus adopted[] dropped[] conflicts[] suiteGreen/>   // ← projection_patch:merge
 ├─ <ConflictList conflicts onResolve>
 │    // structured(tree-sitter) → 3-way → LLM-resolver-run → human; each emits an event
 │    <ConflictHunk both diffs onKeepA onKeepB onMerge/>
 └─ <PromoteBar enabled={suiteGreen} onPromote/>   // the ONLY effect-commit to the user branch
```

Conflicts that reach the human are shown in the **Diff Review** UI (§B) — the same Monaco-diff component, fed two candidate diffs. The user branch only ever receives a fully-integrated, full-suite-green result. `merge.*` arrive as `projection_patch:merge`; resolution choices go back as `Custom:resolve_conflict{by}`.

### D.4 Workstation / remote (later)

The headline thin-client story (laptop drives a Mac-Studio agent server) reuses the **same** intent-in/events-out model over a WebSocket (JSON-RPC, ACP-shaped, carrying `UiEvent` verbatim; server-authoritative, reconnect resumes from `seq`). On the FE this is a **transport swap inside the IPC client** ([`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) §3.7) — the surfaces above are unchanged; `subscribe_ui` becomes a WS subscription and `handle_intent` an RPC call. The host's `remote_protocol` capability is currently `false` (SCAFFOLD_STATUS); FE work here waits on it. **Decision FE-7: surfaces bind to the IPC client interface, never to Tauri `invoke` directly, so the remote swap is a one-file change.**

---

## E. The Context Stack right-rail (THE DIFFERENTIATOR)

Renders the live context manifest verbatim, every turn — the exact answer to "what is the model looking at, why, and what did it leave out." No cloud agent can show this; we own the window assembly (the `context` connector). (Design rationale, now archived under ../archive/.)

### E.1 What it shows

The manifest arrives as `projection_patch:context_manifest` (the host's `context` connector compiles it; `hawking-context` produces the budgeted manifest). Sections:

| Section | Renders | Interaction → Intent |
|---|---|---|
| **Model** | model id/arch/ctx, profile, greedy-vs-sampled, seed; live `runtime_status` | click → `Custom:switch_profile` |
| **Budget bar** | `{total, used, free, reservations}` stacked bar, colored by source | hover segment → tokens + source |
| **Retrieved files** | code spans + `code_index` hits (path:range, relevance) | click → `OpenFile`; drag → `Custom:pin_span` |
| **Symbols** | symbol spans + provenance | click → go-to-definition (`code_index`) |
| **Tools called** | tool spans + `tool_progress` (name, ok/fail, bytes) | click → expand output, link to Timeline step |
| **Memory injected** | memory spans (fact, confidence, provenance) | click → Memory editor (later) |
| **KV / tiers** | prefix-reuse tokens, bank hit, tiers touched | read-only |
| **Confidence** (opt-in) | per-token logprob from `token_batch` confidence path | toggle → `Custom:toggle_confidence` (heat in editor/chat) |
| **Dropped** | dropped candidates `{title, would_be_tokens, reason}` | **one-click pin** → `Custom:pin_span` (forces into next turn) |
| **Conflicts** | surfaced contradictions | inline resolve → `Custom:resolve_conflict` |

```
┌─ CONTEXT STACK ─────────────────────────────┐
│ MODEL  qwen ▸ profile: Standard ▾  16k/32k  │
│ BUDGET ▓▓▓▓▓▓▓▓▓▓▓▓░  14,210 / 16,384        │
│        [sys 1.2k][code 6.1k][tools 3.4k]…    │
│ ▾ RETRIEVED (6)  • auth.rs:42-88 .91 ⊙pin    │
│ ▾ SYMBOLS (12)   ▾ TOOLS (3) ✓read ✓grep ✗   │
│ ▾ MEMORY (2)     • "DB uses sqlx" 1.0 📌      │
│ ▾ KV  bank HIT · reuse 1,200 tok · gpu·ram   │
│ ▾ DROPPED (12) ▸ why?  • cargo log 4.2k ⊕pin │
│ ⚠ CONFLICT: pinned arch fact vs new scan ▸   │
└──────────────────────────────────────────────┘
```

### E.2 Stores / binding / live steering

```ts
interface ContextStore {
  currentManifest: ContextManifest | null;
  manifestRing: ContextManifest[];   // recent manifests, indexed by turn/seq — for scrub coupling
  lastAppliedSeq: number;
}
```

- **Bound to:** `projection_patch:{context_manifest, retrieval, memory}`, `runtime_status`, `token_batch` (confidence). Pull the full manifest via the `context` connector on demand; live deltas via `projection_patch`.
- **Live update cadence:** re-renders on each turn boundary; the budget bar animates as the window fills during prefill; sections collapse with counts (never render 127 dropped rows until opened).
- **Steering writes** (the "edit what the agent sees, live" promise): pin/unpin/resolve/switch-profile emit `Custom` intents → the host appends events → the `context` connector's compiler honors them next turn.
- **Replay coupling (the superpower):** when the Timeline scrubs to seq N, the rail renders `manifestRing[N]` — "what did it see when it made *that* decision." **Decision FE-8: `contextStore.manifestRing` is keyed by the same seq the Timeline scrubs to**, so a sibling doc wiring scrub must publish the manifest per turn, not just the latest.

---

## F. The Research tab (secondary, later)

A first-class tab beside the IDE/Chat/Workstation surfaces, but **post-shell** — the research engine is usable headless via the `research` connector (`runs.list`/`runs.append`) before this UI exists. (Design rationale, now archived under ../archive/.) Panels: **Library** (ingested sources, parse-confidence badges), **Graph** (interactive KG, click node → provenance), **Research Runs** (launch/monitor via `research` connector, run ledger as a timeline, resume paused overnight runs), **Reports** (cited, every sentence a provenance chip), **Lit Maps**, **Experiments**, **Notes/Canvas**, **Review queue**. UX laws: provenance always one click away; measured vs inferred vs speculative visually distinct; contradictions shown, never hidden. Store: `researchStore`, fed by the `research` connector + `projection_patch:research`. Not v1; do not let it gate the three core surfaces.

---

## Appendix — decisions a sibling doc must stay consistent with

1. **FE-1** New steer verbs (`pin_span`, `edit_plan_step`, `switch_profile`, `redirect_run`, `queue_turn`, `rerun_step`, `revert_diff`, `resolve_conflict`, `fleet_run`, …) ride `Intent::Custom{name,payload}` with one canonical `name` string each, until the host promotes them to real variants. The contract doc ([`00-vision-and-backend-contract.md`](00-vision-and-backend-contract.md) §3.8) owns the canonical name registry.
2. **FE-2** Route on `UiEventKind` first; for `projection_patch`, route on `data.projection`. The `projection` name set (§A.4) is the joint host+FE binding map.
3. **FE-3 / FE-7** Wire-B is the ordered `tauri::ipc::Channel<UiEvent>` from `subscribe_ui()`/`UiEventBus`, never `emit/listen`. Surfaces bind to an IPC-client interface, never to `invoke` directly, so the remote (WS) transport is a one-file swap.
4. **FE-4** Layout (`WorkspaceLayout`) is view-state on disk under `<workspace>/.hide/`, never through the event log.
5. **FE-5** `SecurityGate` is a blocking approval prompt; approving sends the gated intent.
6. **FE-6** The per-`run_id` run FSM in `runStore` is the single source of the agent-state pill (status bar, notifications, fleetview all read it).
7. **FE-8** `contextStore.manifestRing` is keyed by the seq the Timeline scrubs to; the manifest is published per turn so scrub re-renders the rail historically.
8. **Reconnect** Every store carries `lastAppliedSeq`; on webview (re)load the FE catches up via the pull API `ui_events(session, after_seq, limit)`, then resumes the live `subscribe_ui()` stream. Nothing durable is lost (the host log is authoritative).
