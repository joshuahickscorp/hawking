# OSS Harvest Map — What to Rip From Each AI IDE

> Part of the **HIDE front-end bible**. The backend is already built (11 Rust crates, real agent loop — see [`SCAFFOLD_STATUS.md`](../SCAFFOLD_STATUS.md)); this doc is the front-end build team's map of which open-source AI-IDE *UI/interaction* components to harvest, what to merely study, and the license rules that gate any port into shipped `app/` code.

This document is self-contained. It supersedes and inlines the front-end-relevant material from the now-archived backend chapters (the competitive matrix and the HCI/UX-and-IDE-surface chapter, now archived under ../archive/). Where a harvested component lands in a *surface* — **AI IDE**, **AI Chat**, **AI Workstation**, or the **Context Stack** right-rail — it references the surface inventory in the sibling doc [`01-surfaces.md`](01-surfaces.md). Where it binds to the backend, it binds to the **real, shipped contract** (`crates/hide-core/src/api.rs` `Intent` / `UiEvent`; `crates/hide-backend` `BackendHost`), not the old design sketch.

---

## 0. The license rule (read first — it gates everything below)

This is a hard CI-enforced rule. Treat it as load-bearing, not advisory.

| Bucket | Licenses | What you may do |
|---|---|---|
| **PORT-OK** | **MIT, Apache-2.0** | Adapt/incorporate source into shipped `app/` (the React/TS/Vite web app) and Rust host code (`hide-serve`). Copyright header in every ported file + a `THIRD_PARTY_NOTICES.md` entry. |
| **STUDY-ONLY** | **AGPL-3.0** (Zed) | Read published docs/blogs/behavior only. **Never** copy, port, link, paste, or "reimplement from a peek at" the source — AGPL would force HIDE's proprietary FE open. No exceptions regardless of snippet size. |
| **NEVER-TOUCH** | Proprietary (Cursor, Copilot/Copilot Workspace, Claude Code) | No source exists / is closed. Study observable UX only. No lifting. |

Mechanics the FE build must honor:

- **`THIRD_PARTY_NOTICES.md`** at repo root is the canonical attribution file. It is *generated at build time* from a structured `harvest.toml` manifest (component → license → version → list of FE source files that incorporate it). Apache-2.0 ports additionally require a "modifications stated" line per file.
- **CI license gate.** A CI job **fails the build** if any FE source file under a harvest path (e.g. `app/src/diff/`, `app/src/layout/`, `app/src/timeline/`) lacks the required origin-header comment, or if `harvest.toml` references a license outside PORT-OK. The gate also scans for any file whose structure is flagged as resembling Zed/GPUI and rejects it.
- **Contributor confirmation.** Any PR adding harvested UI must confirm it does not draw on AGPL (Zed) or proprietary (Cursor/Copilot) source.

If in doubt: **port only MIT/Apache-2.0; everything else is inspiration you re-implement clean.**

---

## 1. Consolidated harvest table

One row per source. "FE component to harvest" is strictly front-end (UI/interaction); backend ports (diff-apply algorithm, repo-map, MCP client, event schema) are **already built** and live in the backend crates per `SCAFFOLD_STATUS.md` — they are listed here only where the *UI shape* travels with them, and are tagged **(backend-done)**. "Target HIDE FE module" names the `app/` module the FE team creates.

| Source | License | FE component to harvest | Mode | Target HIDE FE module / surface |
|---|---|---|---|---|
| **Void** (Void Editor Contributors) | Apache-2.0 | Monaco `DiffEditor` wrapper; hunk-level accept/reject controls; ghost-text rendering for streaming suggestions; collapsible dock/panel layout with stored widths + split-view config — *the closest existing UI to ours* | **Port** | `app/src/diff/DiffView.tsx`, `app/src/layout/` → **AI IDE** |
| **Cline** (Cline Bot, Inc.) | MIT | Plan↔Act mode toggle UX; per-step approval controls; auto-approve category toggles (read-only / write / terminal / MCP); MCP server list + tool-discovery panel UX. *(Tiered diff/apply matcher itself = backend-done.)* | **Port (UI) / study** | `app/src/chat/PlanActBar.tsx`, `app/src/settings/AutoApprove.tsx`, `app/src/mcp/McpPanel.tsx` → **AI Chat** + **AI IDE** |
| **OpenHands** (All Hands AI) | MIT | Event-stream timeline + replay observability rendering — the visual filmstrip over an append-only event log; action↔observation card pairing with `cause` threading. *(Event schema port itself = backend-done in `hide-core/src/event.rs`.)* | **Port (UI) / study** | `app/src/timeline/AgentRunTimeline.tsx` → **AI Workstation** + **AI IDE** |
| **Continue** (Continue Dev, Inc.) | Apache-2.0 | Context-provider UX glue: `@`-mention pickers (`@file`/`@symbol`/`@docs`/`@terminal`) in the composer; per-message "what's in context" affordance. *(Retrieval algorithm = backend-done.)* | **Study → small port** | `app/src/chat/MentionPicker.tsx` → **AI Chat** + **Context Stack** |
| **Aider** (Paul Gauthier) | Apache-2.0 | Repo-map *view* (ranked symbol map rendering) + architect/editor two-mode selector UX. *(Repo-map ranking algorithm = backend-done.)* | **Study** | `app/src/context/RepoMapView.tsx`, mode selector in **AI Chat** |
| **Goose** (Block, Inc.) | Apache-2.0 | Desktop-agent UX shape; MCP-client UX (install/configure an MCP server, browse its tools, see per-tool status). *(`rmcp` client itself = backend-done.)* | **Study** | feeds `app/src/mcp/McpPanel.tsx` → **AI IDE** |
| **Kilo Code** (Kilo Code Contributors) | Apache-2.0 | Checkpoint/undo *shadow-git* UX: per-run snapshot list, one-click "revert to checkpoint," snapshot→event-range index rendering. *(Checkpoint engine = backend-done in `hide-tools`.)* | **Study → Port (UI)** | `app/src/sourcecontrol/CheckpointList.tsx` → **AI IDE** |
| **OpenCode** (OpenCode Contributors) | MIT | Plan/act step-through interaction model + session list / resume / export UX (numbered steps, step-level confirm). | **Study** | `app/src/sessions/SessionBrowser.tsx` → **AI Chat** + **AI Workstation** |
| **Zed** (Zed Industries) | **AGPL-3.0** | Multibuffer diff-review bar (aggregate many-file edits into one scroll, per-hunk Keep/Reject, editable unified diff); "agent following" cursor; context-window-usage display. | **STUDY ONLY** — never copy | inspiration for `app/src/diff/MultibufferReview.tsx` (clean reimpl) → **AI IDE** |
| **Cursor** (Anysphere) | Proprietary | Three-mode model (Tab / Cmd+K inline-edit / Composer-agent); "functional minimalism, editor-grade not chat-grade" visual language; instant-apply concept. | **NEVER lift** — study UX | inspiration only, all surfaces |
| **GitHub Copilot / Workspace** (Microsoft) | Proprietary | Issue→plan→PR flow; plan-then-implement legible-checkpoint UX. | **NEVER lift** — study UX | inspiration for **AI Workstation** batch flow |

---

## 2. Per-source: what UI to take and how it maps onto our surfaces

Each subsection names the exact interaction to harvest and binds it to the **real** `Intent`/`UiEvent` contract the FE sends/receives. The wire types (from `crates/hide-core/src/api.rs`):

- **`Intent`** (FE → host, via `BackendHost::handle_intent` → `IntentAck{accepted, event_seq?, message?}`): `SubmitTurn{session_id,text,attachments}`, `CancelRun{run_id}`, `PauseRun{run_id}`, `ResumeRun{run_id}`, `AcceptDiff{run_id,diff_id}`, `RejectDiff{run_id,diff_id}`, `ScrubToEvent{session_id,event_id}`, `ForkSession{session_id,at_event}`, `OpenFile{path,line?}`, `RunCommand{argv,cwd?}`, `Custom{name,payload}`.
- **`UiEvent{seq,session_id?,kind}`** (host → FE, via `BackendHost::subscribe_ui()` broadcast, forwarded over the `WS /v1/hide/events` WebSocket served by `hide-serve`): `ProjectionPatch{projection,patch}`, `TokenBatch{stream_id,text}`, `RuntimeStatus{status,detail?}`, `ToolProgress{call_id,message}`, `SecurityGate{gate,message}`, `Error{code,message}`, `Custom(Value)`.

> **Contract note for all sibling docs.** The built contract is *deliberately thinner* than the old design sketch (now archived under ../archive/). Rich per-panel state (plan tree, context manifest, diff set, file tree, timeline cards) arrives as **`ProjectionPatch{projection, patch}`** — one named projection per panel, JSON-diff applied into a store slice — **not** as ~30 distinct typed event kinds. Steering actions that the old sketch named as first-class intents (`PinSpan`, `EditPlanStep`, `RedirectRun`, `SwitchProfile`, `ApproveStep`…) are **not** first-class in the shipped `Intent` enum; they ride on **`Intent::Custom{name, payload}`** (e.g. `name:"pin_span"`). Harvested UIs must emit `Custom` for these, and consume the relevant `ProjectionPatch` projection — see §3. Do not invent typed intents the host doesn't accept.

### 2.1 Void — Monaco diff UX + dock layout (the closest UI to ours) → **AI IDE**

**What to take.** Void is the open-source local-first Cursor clone; its UI is the nearest existing thing to HIDE's. Port two pieces:

1. **`DiffEditor` wrapper + hunk controls.** Side-by-side by default; flip to inline when the editor width drops below a breakpoint (Monaco `useInlineViewWhenSpaceIsLimited`). Per-hunk **✓ Accept / ✗ Reject** rendered as gutter actions (Monaco decorations + view zones), plus ghost-text rendering for streaming suggestions.
2. **Dock/panel layout** — collapsible left sidebar / right rail / bottom panel with persisted widths and split-view config.

**How it binds.** A proposed edit is delivered as a `ProjectionPatch{projection:"diff", patch}` (the diff set per file). Accept/reject emit the **real** typed intents `AcceptDiff{run_id,diff_id}` / `RejectDiff{run_id,diff_id}`; the host applies via the backend tiered applier and pushes back a `ProjectionPatch{projection:"diff"|"editor"}` reflecting the applied/reverted state. Layout state is FE-local view state (persisted to `<workspace>/.hide/`), **not** an event.

> **License:** Apache-2.0 → PORT-OK. Header + `THIRD_PARTY_NOTICES.md` entry; state modifications.

### 2.2 Cline / Roo Code — plan/act, per-step approval, MCP UI → **AI Chat** + **AI IDE**

**What to take.**
- **Plan↔Act toggle.** A mode switch in the chat composer: *Plan* (the agent reasons and proposes a numbered step list, pausing before execution) vs *Act* (executes). The plan renders as **editable step cards** in chat.
- **Per-step approval + auto-approve categories.** Each step waits for ✓ before running; a settings surface toggles auto-approve per category (read-only ops / file writes / terminal / MCP).
- **MCP UI.** A panel listing configured MCP servers and the tools each exposes, with per-tool status.

**How it binds.** Plan cards render from `ProjectionPatch{projection:"plan"}`. Approve/edit/reorder/insert steps emit `Intent::Custom{name:"approve_step"|"edit_plan_step"|"reorder_plan", payload}` (these are not first-class intents). The MCP panel lists tools via `BackendHost::call_connector` against the relevant connector and renders `ToolProgress{call_id,message}` for live status; `SecurityGate{gate,message}` events drive the approval prompts the auto-approve categories gate. Backed by the event log, an undo is a compensating event, not a side file.

> **License:** MIT → PORT-OK (UI). Header + notice entry.

### 2.3 OpenHands — event-stream timeline + replay → **AI Workstation** + **AI IDE**

**What to take.** OpenHands renders the agent as an append-only event stream and exposes it as an observable, replayable history. Harvest the **timeline rendering**: a lane of **step cards** in `seq` order, action cards paired with their observation/result, threaded by `cause`. This is the visual face of HIDE's durable log and the spine of the **Agent-Run Timeline** (sibling doc `01`).

**How it binds.** The timeline subscribes to the projection that aggregates run events — `ProjectionPatch{projection:"timeline"}` — and to `TokenBatch{stream_id,text}` (streamed reasoning), `ToolProgress`, `RuntimeStatus`, `Error`. The scrub slider issues the **real** `ScrubToEvent{session_id,event_id}` intent; the host replays the projection to that event (`BackendHost::scrub_to_event(seq)`) and pushes the rebuilt projection — **effects are never re-fired**. "Fork from here" issues the **real** `ForkSession{session_id,at_event}` intent (`BackendHost::fork_session(at_seq)`). Resume uses `ResumeRun{run_id}`.

> **License:** MIT → PORT-OK (UI). Header + notice entry. *(Event schema port itself already shipped in `hide-core/src/event.rs`.)*

### 2.4 Continue — retrieval / context UX glue → **AI Chat** + **Context Stack**

**What to take.** Continue's per-message **context-provider pickers**: `@`-mentions in the composer (`@file`, `@symbol`, `@docs`, `@terminal`, `@diff`) that let the user explicitly add what enters context. Harvest the *picker UX and the "what's in context" affordance*, not the retrieval engine (already backend-done).

**How it binds.** An `@`-mention adds a user-pinned context source — emitted as `Intent::Custom{name:"pin_span", payload:{...}}`; the Context Stack then reflects it via `ProjectionPatch{projection:"context"}` (the manifest render). Fetching candidate files/symbols for the picker uses `BackendHost::call_connector("code_index", "search"|"definition", params)`.

> **License:** Apache-2.0 → PORT-OK (small UI port). Header + notice entry.

### 2.5 Aider — repo-map view + architect/editor modes → **AI Chat** / **Context Stack**

**What to take.** Two UI ideas only (ranking algorithm is backend-done):
- A **repo-map view** — a compact, ranked symbol map of the codebase rendered as a navigable list (click → open at range).
- The **architect/editor mode selector** — a two-mode toggle (one model plans, one applies) surfaced as a chat-level mode picker.

**How it binds.** The repo-map view pulls its ranked entries via `call_connector("code_index", ...)` and/or a `ProjectionPatch{projection:"context"}` section; clicking a symbol issues `OpenFile{path,line}`. The mode selector is **not** a routing setter — the `runtime` connector has no mutating route method (`route` is a read-only routing-decision *preview*, `roles.list` just enumerates roles). Switching architect/editor mode sets FE client state and/or emits `Intent::Custom{name:"switch_profile", payload}`; the host's router honors it on the next turn. Use `call_connector("runtime", "roles.list", params)` to populate the picker and `"route"` only to *preview* the decision.

> **License:** Apache-2.0 → study (no UI code to port; reimplement the view clean). Document the reference in `harvest.toml`.

### 2.6 Goose — desktop-agent UX + MCP client UX → **AI IDE**

**What to take.** Goose's **MCP client UX** (Block, Rust): install/configure a local stdio or remote HTTP MCP server, browse its discovered tools, see per-tool call status — and its general desktop-agent UX shape. Study it to design `McpPanel.tsx` (shared with the Cline harvest).

**How it binds.** Same as §2.2's MCP panel: `call_connector` for listing/configuring, `ToolProgress` for live status, `SecurityGate` for the per-tool permission prompt (HIDE gates every MCP tool through the host permission model — the panel must render that gate, unlike Goose which trusts configured servers).

> **License:** Apache-2.0 → study. `rmcp` client port already backend-done; only the UX is referenced.

### 2.7 Kilo Code — checkpoint / undo shadow-git UX → **AI IDE**

**What to take.** The **checkpoint UX**: a per-run snapshot list, a one-click "revert to checkpoint," and the snapshot→event-range index rendered so the user can see what each checkpoint covers. HIDE's checkpoints are **terminal-aware** (the beat over Cursor): the revert UI must also surface the tool/terminal side-effects that ran after a checkpoint, honestly ("reverting here undoes 2 file edits; note these commands also ran: …").

**How it binds.** Checkpoint list renders from `ProjectionPatch{projection:"sourcecontrol"|"timeline"}`. Revert issues `Intent::Custom{name:"revert_to_checkpoint", payload:{event_id}}` (or, for the deep case, the **real** `ScrubToEvent` + `ForkSession` pair). Undo surfaces as a compensating event in the timeline, not a deletion.

> **License:** Apache-2.0 → study → port (UI). Header + notice entry on any ported view code.

### 2.8 OpenCode — plan/act + session UX → **AI Chat** + **AI Workstation**

**What to take.** OpenCode's **session browser**: session list → resume-by-id → export, keyboard-navigable; and its **plan/act step-through** (numbered steps, step-level confirmation between steps). Feeds HIDE's session browser and supervised-step autonomy UX.

**How it binds.** Session list/switch is host-level (`BackendHost` session registry); resume issues `ResumeRun{run_id}`; a forked session uses `ForkSession`. Plan/act step-through shares the §2.2 plan-card binding (`ProjectionPatch{projection:"plan"}` + `Custom` step intents).

> **License:** MIT → study (TUI implementation not applicable to a WebView; reimplement clean).

### 2.9 Zed — multibuffer diff-review bar (**STUDY ONLY, AGPL**)

**What to study (never copy).** The **multibuffer review** pattern: many-file agent edits aggregate into **one scrollable view** with per-hunk **Keep/Reject** and an **editable unified diff** accepted before commit; plus **agent following** (editor follows the agent's cursor) and a **context-window-usage display**. HIDE reimplements this clean as `MultibufferReview.tsx`, going further: each hunk is one node in the causally-linked timeline (trace a hunk back through `cause` to the plan step to the user turn), and the single "context-window-usage" number becomes the full **Context Stack**.

**Hard rule.** AGPL-3.0. No source contact. Reference only published docs/behavior. A PR resembling GPUI structure must be rejected and rewritten. Binds the same as §2.1 (`AcceptDiff`/`RejectDiff` + `ProjectionPatch{projection:"diff"}`).

### 2.10 Cursor & GitHub Copilot/Workspace (**NEVER lift, proprietary**)

Study observable UX only. From **Cursor**: the three-mode division (Tab predictive completion / Cmd+K inline edit / Composer agent) and the "editor-grade, functional-minimalism" visual language inform the **AI IDE** surface; the instant-apply concept is noted but HIDE achieves apply reliability via the backend tiered applier, not a trained full-file rewriter. From **Copilot Workspace**: the legible issue→plan→PR checkpoint flow informs the **AI Workstation** batch-job UX. No code, no asset, no lifted layout.

---

## 3. What HIDE adds on top (so harvested UIs bind correctly)

Three FE conventions every harvested component must follow — they reconcile the harvested UX with the *built* backend:

1. **Panels render from `ProjectionPatch`, not typed events.** Each harvested panel owns one named projection (`"diff"`, `"plan"`, `"timeline"`, `"context"`, `"editor"`, `"filetree"`, `"sourcecontrol"`, `"status"`). The IPC client applies the JSON `patch` into the matching store slice; the component is a pure render of that slice. (Streamed tokens are the exception: `TokenBatch{stream_id,text}` coalesced into the chat/timeline directly.)
2. **Steering = `Intent::Custom`.** Only ten intents are first-class (§2 header). Pin/unpin, edit/reorder/approve plan step, redirect mid-run, switch profile, revert-to-checkpoint all ride `Intent::Custom{name,payload}`. A harvested UI that assumed a typed intent must be rewired to `Custom`. Every intent returns `IntentAck{accepted, event_seq?, message?}` — render the rejection `message` (this is where Cline's auto-approve / `SecurityGate` denials surface).
3. **Connectors for non-turn data.** Search, definition/references, context-manifest compile, runtime `roles.list`/`route` (the latter a read-only routing preview), personalization, research all go through `BackendHost::call_connector(id, method, params)` — *not* a turn. The `@`-mention picker, repo-map view, and MCP panel all use this path.

---

## 4. Build-order note

Harvest priority follows the surface build order in [`01-surfaces.md`](01-surfaces.md) and the panel sequencing in [`03-build-sequencing.md`](03-build-sequencing.md). The minimum to stand up the **AI IDE** + **AI Chat** skeleton: **Void** (diff + layout), **Cline** (plan/act + approval), **OpenHands** (timeline). **Continue/Aider** (context glue, repo-map view) and **Goose/Cline** (MCP panel) follow with the **Context Stack**. **Kilo Code** (checkpoints) and **OpenCode** (session browser) land with the **AI Workstation**. **Zed** is studied throughout but never blocks a build (clean reimpl). Every port lands with its `THIRD_PARTY_NOTICES.md` row and header in the same PR, or the CI license gate fails it.
