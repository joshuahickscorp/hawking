# 07 · HCI, UX & The IDE Surface

> **Purpose (one line).** Specify the **app shell** of HIDE — the workbench layout, the complete panel/component inventory, the interaction state machines (inline-edit, plan-steer, diff-review), the keyboard/command system, and the front-end store/event-binding architecture — engineered to **exceed Claude Code and Cursor on the local plane** by making the agent *totally observable and steerable*: every token, tool call, retrieved file, KV byte, and plan step is shown live, editable mid-run, and scrubbable from a durable event log, with **zero latency tax** because it is all local and ours.

**Status:** DESIGN. This is the **first chapter the user builds from** — the shell is the product's body, and the *Context Stack* right-rail is its differentiator. It is downstream of two binding contracts and re-uses them verbatim:

- **Chapter 01 — System Architecture** fixed the **`Event` envelope** (§4.6), the ~30-kind taxonomy (`turn.*`/`plan.*`/`token`/`tool.*`/`diff.*`/`test.*`/`context.*`/`memory.*`/`runtime.*`/`error`), the delivery model (**ordered `tauri::ipc::Channel<UiEvent>`, NOT `emit/listen`**, with `token`→`token_batch` render-coalescing and a backpressure ladder), and the **extension manifest** (§7.2) that makes every panel a registered contribution. This chapter is the *consumer* of that event stream and the *renderer* of those projections. ([ch.01](01-system-architecture.md))
- **Chapter 04 — Context Engineering & Memory** fixed the **`ContextManifest`** (Appendix A.1): the per-turn record of retained spans, dropped candidates, signals, budget, KV/bank accounting, conflicts, and compaction events. **The Context Stack right-rail renders this verbatim.** ([ch.04](04-context-engineering-and-memory.md))

Runtime-completion items (32B `.tq` residency, native `.tq` serving) are **runtime testing, not shell-gating** — the shell binds to today's HTTP surface (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/hawking/generate`, `/v1/hawking/tokens`, `/healthz`, `/metrics`) which is **verified in-tree** (`crates/hawking-serve/src/http.rs`). The **Model Lab / Store** (HF distribution UI) is designed as a **placeholder panel and marked LATER**.

---

## Table of contents

1. [Purpose & scope](#1-purpose--scope)
2. [Tenets](#2-tenets)
3. [State of the art + limits (cited)](#3-state-of-the-art--limits-cited)
4. [The Hawking design (concrete)](#4-the-hawking-design-concrete)
   - 4.1 [The workbench layout & pane model](#41-the-workbench-layout--pane-model)
   - 4.2 [The panel/component inventory (v1 shell vs later)](#42-the-panelcomponent-inventory-v1-shell-vs-later)
   - 4.3 [The Context Stack right-rail (the differentiator)](#43-the-context-stack-right-rail-the-differentiator)
   - 4.4 [The Agent-Run Timeline (scrub / replay / step cards)](#44-the-agent-run-timeline-scrub--replay--step-cards)
   - 4.5 [Diff Review (Monaco diff + hunk accept/reject + checkpoints)](#45-diff-review-monaco-diff--hunk-acceptreject--checkpoints)
   - 4.6 [Chat, Editor, Terminal, Explorer, Search, Status Bar](#46-chat-editor-terminal-explorer-search-status-bar)
   - 4.7 [Interaction paradigms (inline-edit, selection-to-chat, plan-steer, interrupt/redirect, provenance)](#47-interaction-paradigms)
   - 4.8 [Key interaction state machines](#48-key-interaction-state-machines)
   - 4.9 [The keyboard model & command system](#49-the-keyboard-model--command-system)
   - 4.10 [The front-end state architecture (stores, SSE→store coalescing, event→panel routing)](#410-the-front-end-state-architecture)
   - 4.11 [Notifications & ambient UX for long / overnight runs](#411-notifications--ambient-ux-for-long--overnight-runs)
   - 4.12 [Theming & accessibility](#412-theming--accessibility)
5. [How we EXCEED (cloud literally cannot do this)](#5-how-we-exceed-cloud-literally-cannot-do-this)
6. [Failure modes / edge cases / mitigations](#6-failure-modes--edge-cases--mitigations)
7. [Extensibility / plugin points (panels as extensions)](#7-extensibility--plugin-points-panels-as-extensions)
8. [Bleeding-edge / moonshots (ranked)](#8-bleeding-edge--moonshots-ranked)
9. [Open questions / dials](#9-open-questions--dials)
10. [Cross-references](#10-cross-references)
- [Appendix A — The panel inventory table (binding)](#appendix-a--the-panel-inventory-table-binding)
- [Appendix B — The front-end store/event-binding map (binding)](#appendix-b--the-front-end-storeevent-binding-map-binding)
- [Appendix C — The default keymap (binding seed)](#appendix-c--the-default-keymap-binding-seed)
- [Appendix D — Source register](#appendix-d--source-register)

---

## 1. Purpose & scope

This chapter specifies **what the human sees and does** — the surface where HIDE's local superpowers become felt experience. The brief is blunt: *exceed Claude Code / Cursor on the local plane; no further UX design work needed after this chapter.* So we are exhaustive: panel inventory, layout spec to the pixel-band, state machines, keymap, and the data bindings that wire the front-end to ch.01's event stream.

The thesis of the whole product lives here. A cloud agent is a **chat box bolted to a black box**: you type, it works somewhere you cannot see, and it hands back a result. HIDE's runtime is *ours and local*, so we can do the one thing they structurally cannot — **show everything and let the user edit it live**:

- the exact tokens the model sees (the `ContextManifest`, rendered as the Context Stack),
- the KV/context budget and what got dropped to fit (drag-to-pin it back),
- every retrieved file, symbol, tool call, and memory injection, as it happens,
- the plan *before and during* execution (approve it, edit it, reorder it),
- the run as a **scrubbable, replayable timeline** backed by the durable event log,
- the model's own confidence (logit-derived) on a token, when the user wants it.

**In scope (this chapter owns the design of):**

- The **workbench layout**: activity bar → swappable left sidebar → tabbed editor group → **right-rail Context Stack** → bottom panel, on a **dockable / splittable pane model**.
- The **complete panel/component inventory**, split **v1-shell vs later**, each with the *events and stores it binds to*: Chat, Editor (Monaco), Agent-Run Timeline, Diff Review, Terminal (xterm+PTY), File Explorer, Search, Command Palette, Status Bar, Context Stack.
- The **interaction paradigms**: inline edit / ghost-text, selection-to-chat, plan-review-and-steer, interrupt/redirect mid-run, "why did you do that" provenance, accept/reject/undo with checkpoints.
- The **observability & steering surfaces** that make the agent transparent and controllable.
- The **keyboard-first model** and the **command system**.
- **Notifications / ambient UX** for long and overnight runs.
- **Theming / accessibility.**
- The **front-end state architecture**: stores, the SSE→store token coalescing, event→panel routing.

**Out of scope (delegated):**

- The agent reasoning loop and plan *generation* policy — **ch.02**. This chapter renders and steers plans; it does not decide them.
- The diff *computation*, apply/merge semantics, and checkpoint *engine* — **ch.03**. This chapter renders the diff and routes accept/reject intents; ch.03 owns the apply.
- Retrieval *ranking*, the symbol graph, the index — **ch.05**. This chapter renders `context.retrieval` hits; ch.05 produces them.
- Runtime kernels, sampler, grammar, KV surgery — **ch.06**. This chapter consumes `runtime.*`/`token` and exposes the dials.
- The architecture, event schema, IPC transport, plugin spine — **ch.01** (bound here, not redefined).
- HF model distribution / `.tq` packaging UI — **deferred**; a Model Lab placeholder is sketched (§4.2, §7) and marked LATER.

**The over-engineering mandate, restated for UX.** Every panel is an **extension** declared by the ch.01 manifest (`[[contributes.panels]]`), mounted at a registry-known dock point, and bound to event kinds via the capability spine. The litmus test from ch.01 §2 applies: *to add a panel, does anyone touch `core/`?* No — they register a contribution. The shell ships a **layout engine, a router, and a store fabric**, not a hard-coded screen.

---

## 2. Tenets

These nine UX tenets sit under ch.01's ten architectural tenets and cite them.

| # | Tenet | Consequence |
|---|---|---|
| **U1** | **Show the model's mind.** Everything the agent sees, retrieves, spends, and decides is visible by default — the Context Stack is a first-class rail, not a debug toggle. | The differentiator is *observability*. We render the `ContextManifest` (ch.04 A.1) verbatim, live. |
| **U2** | **Steer, don't just watch.** Every observable is *editable*: pin a dropped span, edit a plan step, interrupt and redirect mid-run, correct a memory. | HITL is the default interaction grammar, not an exception (cf. LangGraph `interrupt()`, AgentScope real-time steering — §3). |
| **U3** | **The log is the UI's truth (T2).** The view holds **no authoritative state**; it renders projections pushed over `Channel<UiEvent>` and sends intents back. A reload loses nothing. | Stores are *derived caches*; on reconnect we replay from the log. Time-travel is free because the substrate already records everything (ch.01 §4.5). |
| **U3.5** | **Render rate ≠ generation rate.** A 120 tok/s stream never floods a 60 fps UI; `token`→`token_batch` coalescing decouples them (ch.01 §4.4/§4.9). | The UI is smooth under flood by construction; the *log* keeps every token, the *render* is batched. |
| **U4** | **Keyboard-first, mouse-optional.** Every action has a command-palette entry and a keybinding; nothing is mouse-only. | A `command` is an extension kind (ch.01 §7.1); the palette is the universal entry point (cf. VS Code Quick Open — §3). |
| **U5** | **Diffs are reviewed, never dumped.** Agent edits land as a Monaco diff with per-hunk accept/reject and a checkpoint behind every applied hunk. | We beat Cursor's file-only checkpoints with event-log-backed, per-tool-call, *terminal-aware* checkpoints (ch.01 §4.5; §3). |
| **U6** | **Latency is a moat — spend it on richness.** No network round-trip for UI; local IPC is ~ms (`Channel` 10 MB ≈ 5 ms on macOS — ch.01 §3). | We can afford persistent panels, live re-render of the manifest every turn, and optimistic UI a cloud IDE cannot. |
| **U7** | **Persistent, restorable workspace.** Layout, open tabs, panel state, scroll, and the active run survive quit/crash/sleep and restore exactly. | Workspace state is itself a projection; "resume at turn N" rehydrates the surface (ch.01 §4.12). |
| **U8** | **Progressive disclosure.** The default surface is calm (chat + editor); depth (timeline scrub, manifest internals, logit confidence, KV tiers) is one keystroke away, never in your face. | Beginners get Cursor-grade calm; power users get full observability. Functional minimalism by default (cf. Cursor "editor-grade, not chat-grade" — §3). |
| **U9** | **Ambient for the long haul.** Overnight and multi-agent runs report through OS notifications, a runs dashboard, and a dock badge — you can walk away and come back to a reviewable timeline. | "Spend lavishly, locally" (T9) means runs are long; the UX must make *unattended* runs safe and reviewable. |

---

## 3. State of the art + limits (cited)

Tagged **[PROVEN]** (shipping in a real tool we can point at) / **[SPECULATIVE]** (research or emerging). Full register in [Appendix D](#appendix-d--source-register).

### 3.1 Cursor — the editor-grade bar to clear

Cursor's UX is the commercial state of the art for AI-native editing, organized around **three interaction modes**: **Tab** (predictive multi-token / next-action autocomplete), **Cmd+K Inline Edit** (surgical in-editor edits in a small prompt box over the selection), and **Composer / Agent** (multi-file, codebase-aware agentic edits that pick files, run terminal commands, check errors, and iterate). Its design philosophy in the 2026 builds is explicitly **"functional minimalism… editor-grade UX, not chat-grade"** — monochrome palette, contextual toolbars that vanish, typography tuned for dense review ([DeployHQ Cursor guide], [CallMissed Composer 2026]). **[PROVEN]**

> **Limit we beat.** Cursor's agent works in a **black box**: you see the resulting diff and a terse activity log, *not* the live context window, KV budget, retrieval set, or per-step plan, and you cannot edit them mid-run. Its checkpoints are **file-only and git-separate** and **do not revert terminal commands** (ch.01 §3). HIDE renders the *whole* context manifest live and makes checkpoints event-log-backed and terminal-aware.

### 3.2 Claude Code — the steerable-TUI patterns

Claude Code is the reference for **terminal-grade agent UX** and the source of several patterns we lift into a GUI: **Plan Mode** (the agent drafts a plan — literally a markdown file — and the human approves before any execution; "separates thinking from doing"), **permission prompts** with **allowlists** and an **Auto mode** (a separate classifier blocks only risky actions — scope escalation, unknown infra), and **interrupt-and-steer** (type *during* execution to redirect; an open feature request extends this to a **prompt queue** with "override current task" vs "append next task") ([Anthropic best practices], [Ronacher "What is Plan Mode"], [Anthropic auto-mode], [claude-code#25845]). **[PROVEN]**

> **Limit we beat.** A TUI is line-oriented and ephemeral: the plan, the tool log, and the diffs scroll away; there is no persistent, scrubbable timeline, no side-by-side diff with per-hunk control, no live context panel, no manifest. HIDE keeps Claude Code's *grammar* (plan-then-act, allowlists, interrupt-to-steer) and gives it a **durable, scrubbable, multi-pane body**.

### 3.3 Cline / Roo Code — plan/act, per-step approval, checkpoints

Cline pioneered the **Plan ↔ Act** toggle (Plan reads and reasons, Act executes with **per-step approval**) and **auto-approve categories** (read-only ops, file writes, terminal, browser, MCP). It creates a **checkpoint after *each individual tool call*** — every file write, terminal command, or web request gets its own checkpoint with one-click undo. **Roo Code** (a mid-2025 fork) added a **polished side-by-side Diff view** and **Modes** (Architect / Act / Ask) ([Cline.bot], [Roo Code review], [DevToolReviews]). **[PROVEN]**

> **Adoption.** HIDE's **autonomy ladder** (suggest-only ↔ auto-apply-with-tests) and **per-tool-call checkpoint** model come directly from here — but backed by the event log (ch.01), so undo is a *compensating event* (`diff.reverted`) and the checkpoint is replayable, not a side file.

### 3.4 Zed — the multibuffer diff-review bar

Zed's 2025 agent-panel overhaul shipped **multibuffer review**: agent edits across many files aggregate into **one scrollable view** with **per-hunk Keep / Reject** controls and an **editable unified diff** you accept/reject before committing; plus **agent following** (the editor follows the agent's cursor) and **context-window-usage display** for external agents ([Zed agent panel], [Zed 2025 recap], [Zed DeepWiki 8.1]). **[PROVEN]**

> **Adoption + beat.** HIDE adopts the **multibuffer review** pattern (§4.5) and goes further: the diff is one node in a **causally-linked timeline** (you can trace a hunk back through `diff.proposed → plan.step → turn.user`), and the "context-window-usage" display becomes the **full Context Stack**, not a single number.

### 3.5 OpenHands — event-stream observability & replay

OpenHands models the agent as a **pure function from event history to next event**, with an **append-only event stream as the single source of truth** enabling **deterministic replay**, fault recovery, and debugging; observability is exposed via **OpenTelemetry tracing** and interactive workspace surfaces (browser-VSCode, VNC) for human inspection ([OpenHands ICLR 2025], [OpenHands SDK arXiv:2511.03690], [OpenHands observability docs], [DeepWiki event storage & replay]). **[PROVEN]**

> **Adoption + beat.** This is the *exact substrate ch.01 already specifies* (event-sourced, replayable, effects-recorded-not-refired). OpenHands is cloud-deployed Python **with no native editor and no panel/provider plugin spine**. HIDE binds the same event model to a **real Monaco editor, a scrubbable timeline UI, and the ch.01 manifest plugin spine** — on-device.

### 3.6 Command palette & keyboard-first workbench

The **command palette** (Ctrl/Cmd+Shift+P) is the canonical keyboard-first surface: a single searchable text box over **all** commands with **fuzzy matching**, plus **Quick Open** (Cmd+P) for fuzzy file navigation; it solves discovery and keeps hands on the keyboard ([VS Code command palette], [UX Patterns: Command Palette], [uxpatterns.dev]). The **VS Code workbench** is the reference multi-pane layout: **Title bar, Activity Bar (movable), Primary Side Bar (viewlets), Editor Area (tabbed, splittable groups), Secondary Side Bar / Auxiliary Bar, Panel Area (bottom), Status Bar** — with a **dockable/splittable** model where views move between docks via `View: Move View` ([VS Code user interface], [VS Code custom layout], [Workbench Layer architecture]). **[PROVEN]**

> **Adoption.** HIDE's layout (§4.1) is this workbench, with the **Auxiliary Bar repurposed as the always-on Context Stack** and a **palette that searches commands, files, symbols, *and agent actions* (replay to event, re-run step)** in one box.

### 3.7 Human-in-the-loop / steerable-agent interaction research

The 2025 HITL literature converges on **interrupt + persistence + state** as the steering primitives: **graph-level interrupts** (halt at predefined points for approval) and **node-level interrupts** (request input mid-execution), both **checkpointed so resume is lossless**; **real-time steering** "gracefully pauses the ongoing ReAct loop on an external signal" (AgentScope uses asyncio cancellation); and interaction-design work on letting users **co-plan, observe real-time actions, take over when stuck, and approve/reject critical executions** ([LangChain interrupt blog], [LangGraph HITL], [AgentScope arXiv:2508.16279], [Interruptible Agents arXiv:2604.00892]). **[PROVEN-substrate / SPECULATIVE-UX]**

> **Adoption + beat.** ch.01's `abort: Arc<AtomicBool>` (verified in `engine.rs`) + the intent/event model give us **interrupt** natively; ch.01's checkpoint/replay gives us **persistence**; this chapter contributes the **UX**: a steer bar, plan editing, redirect-mid-run, and a prompt queue (Claude Code's pattern, given a GUI).

### 3.8 Voice coding (emerging)

Voice is an emerging accessibility + speed lever: humans speak 3–5× faster than they type (150+ WPM vs 40–80), and 2025 tools (Wispr Flow, BridgeVoice, SuperWhisper — local Whisper) put a **mic button in the composer** to dictate prompts to agents ([Wispr Flow], [BridgeVoice], [Addy Osmani "Speech-to-Code"]). **[PROVEN-niche]**

> **Local advantage.** A **local Whisper** runs as a `model-provider`/tool extension (ch.01 §7) — voice dictation with **zero audio egress**, an accessibility win cloud cannot match on privacy.

### 3.9 Front-end stack ground truth (this repo's chosen stack)

The stack is fixed (brief + ch.01): **Tauri 2 + React + TypeScript + Monaco + xterm.js**, Rust host emitting a typed event stream, runtime serving SSE. The ecosystem confirms feasibility: **Monaco** ships a `DiffEditor` (side-by-side or inline) with **view zones, decorations (zIndex stacking), and inline widgets** for ghost-text and inline-edit overlays ([Monaco IDiffEditorOptions], [monaco-editor/react]); **xterm.js + portable-pty/`tauri-plugin-pty`** is a proven Tauri terminal pattern (Terax: Tauri 2 + React 19 + xterm.js + Zustand, 7 MB) ([tauri-plugin-pty], [Terax]); **Zustand** is the lightweight store of choice for exactly this shape ([Tauri state mgmt], [Terax]). **[PROVEN]**

---

## 4. The Hawking design (concrete)

### 4.1 The workbench layout & pane model

HIDE is a **six-region workbench** on a **dockable, splittable pane model**, defaulting to a calm three-column reading layout (U8) that opens into full observability on demand.

#### 4.1.1 The six regions (wireframe)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TITLE BAR  ⌘  hide ▸ project ▸ branch  ·  [runtime: qwen-7b ● Ready 41 tps]  · ⌘K palette │  ← global status seam
├──┬───────────────────────┬──────────────────────────────────────┬──────────────────────┤
│A │  PRIMARY SIDEBAR       │  EDITOR GROUP(S)  (tabbed, splittable)│  CONTEXT STACK        │
│C │  (swappable viewlet)   │  ┌────────────────────────────────┐  │  (RIGHT RAIL — the    │
│T │                        │  │ auth.rs ✎ │ db/pool.rs │ +Diff  │  │   differentiator)     │
│I │  ▸ Explorer            │  ├────────────────────────────────┤  │                       │
│V │  ▸ Search              │  │                                │  │  ▸ Model  qwen-7b     │
│I │  ▸ Source Control      │  │   Monaco editor / Monaco diff  │  │     ● Ready · greedy  │
│T │  ▸ Agent Runs          │  │   (ghost-text, inline-edit,    │  │  ▸ Budget ▓▓▓▓▓░ 14.2k│
│Y │  ▸ Memory              │  │    hunk gutters, decorations)  │  │     /16k · resp 2k    │
│  │  ▸ Chat (or docked R)  │  │                                │  │  ▸ Retrieved (6) ▸    │
│B │                        │  │                                │  │  ▸ Symbols (12) ▸     │
│A │                        │  │                                │  │  ▸ Tools called (3) ▸ │
│R │                        │  │                                │  │  ▸ Memory inj. (2) ▸  │
│  │                        │  │                                │  │  ▸ KV/tiers gpu·ram   │
│  │                        │  │                                │  │  ▸ Dropped (12) ▸     │
│  │                        │  │                                │  │  ─────────────────    │
│  │                        │  │                                │  │  CHAT  (default dock) │
│  │                        │  │                                │  │  > turn / plan / steer│
├──┴───────────────────────┴──────────────────────────────────────┴──────────────────────┤
│ BOTTOM PANEL (tabbed):  Terminal │ Problems │ Test Output │ Agent Timeline │ Output │ Debug │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ STATUS BAR:   branch  ⎇  · ⚠ 2  ● 0 · Ln 42,Col 8 · UTF-8 · Rust · [agent: planning…⏸] · 41 tps │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| # | Region | Default | Contains | Notes |
|---|---|---|---|---|
| 1 | **Activity Bar** | left edge, narrow | Viewlet switchers (Explorer, Search, SCM, Agent Runs, Memory, Chat, Model Lab*) + bottom: settings, account, notifications bell | Movable to top/bottom of sidebar (VS Code parity). Badges (run count, problem count, unread notifications). |
| 2 | **Primary Sidebar** | left, ~260 px, collapsible | the active **viewlet** (one of the above) | Swappable; remembers width per viewlet. Collapse with `Cmd+B`. |
| 3 | **Editor Group(s)** | center, fills | tabbed Monaco editors + Monaco diff tabs; **splittable** into a grid of groups | The core. Splits horizontal/vertical; each group has its own tab strip and active tab. |
| 4 | **Context Stack** (Auxiliary/right rail) | right, ~320 px, **on by default** | the live `ContextManifest` render + the **docked Chat** beneath it | §4.3. This is the rail that makes the agent observable. Toggle `Cmd+Alt+B`. Chat can also dock to the sidebar. |
| 5 | **Bottom Panel** | bottom, ~30% height, collapsible | tabbed: **Terminal, Problems, Test Output, Agent Timeline, Output, Debug** | §4.4/§4.6. Maximizable; the **Agent Timeline** lives here when expanded (or pops to a full editor tab). |
| 6 | **Status Bar** | bottom edge, 1 line | branch, problems counts, cursor pos, encoding, language, **agent state pill (+ interrupt)**, **live tps** | §4.6.6. The agent-state pill is a *global steer affordance* — click to pause/interrupt from anywhere. |

\* Model Lab is a **placeholder viewlet, marked LATER** (§7).

#### 4.1.2 The pane model (dockable / splittable)

A **layout tree** (the front-end mirror of VS Code's grid + ch.01's "panels are extensions") governs docking and splitting:

```ts
type DockId = "sidebar.primary" | "rail.context" | "panel.bottom" | "editor.grid";
type Orientation = "horizontal" | "vertical";

type LayoutNode =
  | { kind: "leaf"; panelId: string; tabs: TabId[]; activeTab: TabId }   // a tab group
  | { kind: "split"; orientation: Orientation; children: LayoutNode[]; sizes: number[] };

interface WorkspaceLayout {
  schema_version: 1;
  editorGrid: LayoutNode;             // the splittable editor area
  docks: Record<DockId, { open: boolean; size_px: number; activeViewlet?: string }>;
  panelOrder: string[];               // bottom-panel tab order
  floats: FloatingPanel[];            // torn-off panels (own OS window via Tauri multiwindow)
}
```

- **Splittable editor grid.** `Cmd+\` splits the active group; drag a tab to a group edge to create a split (drop-zone overlay, VS Code-parity). Each leaf is an independent tab group.
- **Dockable panels.** Any panel registered with `mount: "left-dock" | "right-dock" | "bottom-dock" | "editor"` (ch.01 §7.2 manifest) can be **moved between docks** via `View: Move View …` (palette) or dragged. A `panel` extension declares a *preferred* mount; the user can override and the choice persists in `WorkspaceLayout`.
- **Tear-off to a window.** Tauri 2 multi-window lets a panel (e.g. the Agent Timeline, or a second editor group on a second monitor) **float into its own OS window**; it re-subscribes to the same `Channel<UiEvent>` for that window (ch.01 Wire B is per-window). This is a desktop-integration win cloud cannot offer.
- **Persistence (U7).** `WorkspaceLayout` is serialized to `<workspace>/.hide/` (alongside ch.01's per-workspace state) and restored on open; it is *not* an event (it's view state, deliberately separate from the authoritative log per ch.01 T2) but it is durable and per-workspace.

#### 4.1.3 Layout presets

A `Cmd+K Z`-style "zen/focus" cycle and named presets ship as defaults:

| Preset | Sidebar | Context Stack | Bottom Panel | For |
|---|---|---|---|---|
| **Focus** | hidden | hidden | hidden | distraction-free single-file editing (U8) |
| **Code** (default) | Explorer | open | collapsed | day-to-day |
| **Agent** | Agent Runs | open (expanded) | Agent Timeline | watching/steering a run |
| **Review** | Source Control | open | Test Output | reviewing a diff + tests |
| **Debug** | Explorer | open | Debug + Terminal | debugging |

Presets are `command`-kind extensions (`layout.preset.agent`), so teams can ship their own.

---

### 4.2 The panel/component inventory (v1 shell vs later)

The full binding table is **[Appendix A](#appendix-a--the-panel-inventory-table-binding)**. Summary of the split and the rationale:

**v1 shell (build first — the central deliverable):**

1. **Editor (Monaco)** — the body. Tabs, splits, ghost-text, inline-edit, hunk gutters.
2. **Chat** — the conversation + plan + steer surface (docked right rail or sidebar).
3. **Agent-Run Timeline** — scrub/replay + step cards. *The observability spine made visual.*
4. **Diff Review** — Monaco diff + per-hunk accept/reject + checkpoint integration.
5. **Context Stack (right rail)** — the live `ContextManifest`. *The differentiator.*
6. **Terminal** — xterm.js + PTY.
7. **File Explorer** — tree, with `file.changed_external` live decorations.
8. **Search** — workspace text search (ripgrep-backed via a tool).
9. **Command Palette** — commands + files + symbols + agent actions.
10. **Status Bar** — global status + agent-state pill + tps.
11. **Problems** — diagnostics list (`build.status`/`test.status`/LSP).
12. **Notifications / Toasts** — ambient run reporting.

**Later (designed, marked; not v1):**

- **Memory viewlet/editor** — browse/edit/pin `.hide/memory/*` (ch.04 §4.7). High user value but ch.04-gated; ships shortly after v1.
- **Test Explorer** — structured test tree (v1 ships the Test Output *panel*; the tree is later).
- **Model Lab / Store** — HF distribution UI. **Placeholder only; LATER** (32B `.tq` is runtime testing).
- **Multiplayer presence** — CRDT cursors/avatars (ch.01 §8 moonshot).
- **Speculative/optimistic UI overlay** — render draft tokens ahead of verify (ch.01 §8 #6).
- **Voice composer** — local-Whisper mic in chat (§3.8).
- **Energy/observability dashboard** — J/tok, dispatch traces (ch.01 §4.6 `runtime.stats`, repo's `dispatch_samples`).

**Why this split.** v1 must *exceed Cursor/Claude Code on the local plane* — that demands the editor, chat, diff-review, terminal, and crucially the **Timeline + Context Stack** (the things they lack). Memory editing, Model Lab, multiplayer, and voice are *additive differentiators* that don't gate the core promise and depend on chapters (ch.04/ch.06) or features still in runtime testing.

---

### 4.3 The Context Stack right-rail (the differentiator)

**This is the panel that makes HIDE.** It renders the ch.04 `ContextManifest` (Appendix A.1) **verbatim and live**, every turn — the exact answer to "what is the model looking at, why, and what did it leave out." No cloud agent can show this because they don't own the window assembly; we do.

#### 4.3.1 What it shows (bound 1:1 to `ContextManifest`)

| Section | Renders (from `ContextManifest`) | Interaction |
|---|---|---|
| **Model** | `manifest.model` `{id, arch, ctx_len_native, ctx_len_effective}` + `manifest.profile` `{name, position_policy, working_set_mode, kv_precision}` + live `runtime.status`/`runtime.stats` | Click → switch profile (Tight/Standard/Long/Unbounded, ch.04 §4.9) → emits a profile-change intent. Shows greedy vs sampled, seed. |
| **Budget bar** | `manifest.budget` `{total, used, free, reservations{system,response,scratchpad}}` | A stacked bar colored by source kind; hover a segment → tokens + source. The "context/KV budget" the brief calls for. |
| **Retrieved files** | `manifest.spans[kind=Code]` + `context.retrieval.hits` | Each row: path:line-range, value, relevance. Click → open file at range. Drag → pin (`pin: user_pinned`). |
| **Symbols** | `manifest.spans[kind=Symbol]` | Symbol name + provenance; click → go-to-definition. |
| **Tools called** | `manifest.spans[kind=ToolOutput]` + `tool.call`/`tool.result` | Tool name, ok/fail, bytes; click → expand output (from `bytes_ref` blob); links to the timeline step. |
| **Memory injected** | `manifest.spans[kind=Memory]` + `memory.written` | Which memory facts entered the window, confidence, provenance ref. Click → open in Memory editor (later). |
| **KV / tiers** | `manifest.kv` `{prefix_reuse_tokens, bank_hit, tiers_touched, checkpoint_id}` | Shows banked-prefix reuse (the "prefilled once ever" win), which tiers (gpu/ram/disk) served this turn. |
| **Confidence** (opt-in) | per-token `logprob` (ch.01 `token` payload `logprob?`; runtime full-logits path — `GenStats.logits_materialized_*`) | Toggle "confidence heat" → low-confidence tokens highlighted in the editor/chat. **[runtime-side hook — opt-in]** (the streaming `StreamEvent::Token{id,text}` is token-only by default; confidence requires the full-logits readback path). |
| **Dropped** | `manifest.dropped[]` `{title, would_be_tokens, reason}` | "Dropped (12) ▸" expander; each shows *why* (no_fit/redundant/stale/low_value). **One-click pin** to force it into next turn. The brief's "see what the agent left out, edit it live." |
| **Conflicts** | `manifest.conflicts[]` | Surfaced contradictions (e.g. pinned arch fact vs new scan); inline "resolve" → keep A / keep B / merge. |
| **Compaction** | `manifest.compaction_events[]` | What got summarized, by which draft model, at what ratio (the "free local compaction" win). |

#### 4.3.2 Binding & behavior

- **Bound to:** `context.update`, `context.retrieval`, `memory.written`, `runtime.status`, `runtime.stats`, `token`/`token_batch` (for confidence), plus the per-turn `ContextManifest` delivered as a `context.manifest` UiEvent (the projection engine assembles it from ch.04's compiler output).
- **Store:** `contextStore` (Appendix B) — holds the *current* manifest + a ring of recent manifests (so scrubbing the timeline re-renders the manifest *as it was at that turn*).
- **Live update cadence:** re-renders on each `context.update`/turn boundary; the budget bar animates as the window fills during prefill. Under flood it coalesces like everything else (U3.5).
- **Steering writes:** pin/unpin/resolve emit intents (`PinSpan`, `UnpinSpan`, `ResolveConflict`, `SwitchProfile`) → ch.01 appends events → ch.04's compiler honors them next turn. **This is "edit what the agent sees, live."**
- **Replay coupling:** when the user scrubs the timeline (§4.4) to turn N, the Context Stack shows turn N's manifest (the ring), so you can answer "what did it see when it made *that* decision" — a debugging superpower.

#### 4.3.3 Wireframe (expanded)

```
┌─ CONTEXT STACK ─────────────────────────────┐
│ MODEL  qwen-7b · transformer · greedy seed42 │
│        profile: Standard ▾   16k eff / 32k   │
│ BUDGET ▓▓▓▓▓▓▓▓▓▓▓▓▓░░  14,210 / 16,384      │
│        [sys 1.2k][code 6.1k][tools 3.4k]…    │
│        reserved: response 2k · scratch 1k    │
│ ▾ RETRIEVED (6)                              │
│   • auth.rs:42-88        rel .91  ⊙pin       │
│   • db/pool.rs:1-40      rel .77  ⊙pin       │
│ ▾ SYMBOLS (12)   ▾ TOOLS (3)  ✓read ✓grep ✗  │
│ ▾ MEMORY INJECTED (2)                        │
│   • "DB uses sqlx" conf 1.0 (pinned) 📌      │
│ ▾ KV   bank HIT · reuse 1,200 tok · gpu·ram  │
│ ▾ DROPPED (12)  ▸ why?                       │
│   • full cargo log  4.2k  no_fit     ⊕pin    │
│   • dup auth.rs     1.1k  redundant  ⊕pin    │
│ ⚠ CONFLICT: pinned arch fact vs new scan ▸   │
└──────────────────────────────────────────────┘
```

---

### 4.4 The Agent-Run Timeline (scrub / replay / step cards)

The Timeline turns ch.01's append-only event log into a **visual, scrubbable, replayable filmstrip of the run** — the observability surface that no cloud agent and no TUI can match. It is the visual face of ch.01 §4.5 (deterministic replay, effects-recorded-not-refired).

#### 4.4.1 Structure

- **A horizontal (or vertical) lane of step cards**, one per significant event, in `seq` order, grouped by `run_id` and nested by `parent`/`cause` (the causal DAG). Plus a **scrub slider** mapped to `seq`.
- **Step-card kinds** (each maps to event kinds):

| Card | From event(s) | Shows |
|---|---|---|
| **Turn** | `turn.user` / `turn.assistant_started/ended` | user prompt; assistant boundary; stop reason |
| **Plan step** | `plan.step` / `plan.step_updated` | title, rationale, status (pending/active/done/failed/skipped); editable (§4.7) |
| **Thinking/tokens** | `token`/`token_batch` | streamed reasoning (collapsible); confidence heat opt-in |
| **Tool call** | `tool.call` → `tool.result` | tool, args, ok/fail, exit code, duration, output (expand); grant id (audit) |
| **Diff** | `diff.proposed` → `diff.applied`/`diff.reverted` | file, hunk count, accepted/rejected; click → open Diff Review |
| **Test/Build** | `test.status`/`build.status` | pass/fail counts, duration; click → Test Output |
| **Context** | `context.update`/`context.retrieval` | budget delta; retrieved set (mirrors the Context Stack at that point) |
| **Memory** | `memory.written` | fact written, scope |
| **Runtime** | `runtime.status`/`runtime.unavailable` | model state transitions, pauses |
| **Error** | `error` | taxonomy code, message, fatal? |

- **Causal threading.** Because every event carries `parent`/`cause` (ch.01 §4.6), a card can show **"← caused by Plan step 3 ← Turn 1"** and a hunk can be traced to the tool call to the plan step to the user turn. This is the **provenance spine** behind "why did you do that" (§4.7).

#### 4.4.2 Scrub & replay (the superpower)

- **Scrub slider** mapped to `seq`. Dragging it issues `ScrubToEvent(seq)` intents (ch.01 Wire A); the kernel **replays the projection to that seq** (snapshot + fold, ch.01 §4.5) and pushes the rebuilt UI state. **Editor buffers, plan tree, diffs, and the Context Stack all rewind to that moment.**
- **Effects are never re-fired** (ch.01 T3): scrubbing is a pure fold over recorded outcomes — *no* file write or shell command re-runs. The UI is honest about this: a "replay mode" banner shows you're viewing history, read-only.
- **Resume / fork from here.** Two buttons on any past event:
  - **Resume execution from here** → kernel transitions replay→live, re-attaches runtime+tools, appends *new* events from `seq` onward (`session.resumed{from_seq}`).
  - **Fork from here** → `session.forked{from_session, at_seq}`; new child session, original intact (git-branch semantics). The UI shows a branch in the timeline.
- **Edit-then-fork.** The killer move: scrub to a plan step, **edit the plan or the injected memory or a pinned span**, then **fork** — the downstream re-executes deterministically from the edited state (ch.01 T6 greedy bit-identity). This is "edit the agent's memory/plan mid-run" made literal.

#### 4.4.3 Live vs historical

- **Live tail:** while a run executes, the timeline auto-scrolls to the newest card (like a log tail), the active plan step pulses, and the scrub thumb sits at head.
- **Detach:** scroll back / grab the slider → detaches from tail into review mode (a "Jump to live ⟶" affordance returns). Same pattern as a chat scrollback or a `tail -f` you scrolled up in.
- **Binding:** the whole timeline is bound to **every** event kind (it's the universal consumer); store is `runStore` + `timelineStore` (Appendix B), which subscribe to the raw projection and the durable log (for scrub beyond the in-memory window, the store requests a log range via an intent).

#### 4.4.4 Wireframe

```
┌─ AGENT TIMELINE ─ run 7 ─────────────────────────────────────  [⏸ pause] [⟶ live] ─┐
│ seq ├────────────●────────────●──────────◐ (active) ─────────────────────────────┤  │
│ #4201 TURN  "fix the JWT expiry bug"                                                │
│ #4203  └ PLAN  ▸1 locate auth ✓  ▸2 read token.rs ✓  ▸3 patch ◐  ▸4 run tests ·   │
│ #4205     └ TOOL grep "jwt" → 3 hits  ✓ 12ms                                       │
│ #4210     └ TOOL read auth.rs:1-120  ✓                                             │
│ #4222     └ DIFF auth.rs  +4 −2  [2 hunks]  → review ▸   (caused by plan ▸3)       │
│ #4228     └ TEST  cargo test  ✓ 41 passed 0 failed  870ms                          │
│ [⟲ replay from #4222]   [⑂ fork from #4222]   [why ▸ trace]                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.5 Diff Review (Monaco diff + hunk accept/reject + checkpoints)

Agent edits **never auto-clobber the file** (U5). They arrive as `diff.proposed` events and render in a **Monaco DiffEditor** with **per-hunk accept/reject** and an **event-log-backed checkpoint** behind every applied hunk — adopting Zed's multibuffer review and Cline's per-tool-call checkpoint, beating Cursor's file-only/terminal-blind checkpoints.

#### 4.5.1 Surface

- **Single-file diff:** a Monaco `DiffEditor` tab (side-by-side default; **inline when width < breakpoint**, Monaco's `useInlineViewWhenSpaceIsLimited`). Gutter hunk controls: **✓ Accept hunk / ✗ Reject hunk**, with `keep`/`reject` decorations (Monaco decorations + view zones).
- **Multibuffer review (Zed-style):** when a run touches many files, a **"Review Changes" aggregate** lists all `diff.proposed` for the run as collapsible per-file diffs in one scroll, each with hunk controls + a top toolbar **Accept All / Reject All / Accept File / Reject File**. Bound to `sourceControlStore` + `diffStore`.
- **Editable diff:** the modified side is editable (Monaco) — the user can tweak the agent's edit *before* accepting (Zed parity). Edits are captured as a `diff.proposed` revision (the user becomes the author of the final hunk).

#### 4.5.2 Accept / reject / checkpoint flow

- **Accept hunk** → `AcceptDiff{diff_id, hunk_id}` intent → ch.03 applies the hunk → `diff.applied{diff_id, post_blob, on_disk_hash}` event → a **checkpoint** is the event itself (the log *is* the checkpoint; the post-image is a content-addressed blob, ch.01 §4.7). The editor updates; the timeline gains a "diff applied" sub-card.
- **Reject hunk** → `RejectDiff{diff_id, hunk_id}` → recorded as a decision (the agent sees it didn't land and can react, ch.02).
- **Undo** → `diff.reverted{diff_id, reason}` is a **compensating event** (ch.01 §4.5), not a deletion; restores the pre-image blob. Because it's an event, undo is itself replayable/auditable and survives restart.
- **Terminal-aware checkpoints (the beat over Cursor).** Because *tool calls* (including `shell.run`) are events with recorded outcomes, "revert to checkpoint" can present **both** file reverts **and** the terminal/build side effects that happened after it (ch.01 explicitly notes Cursor "does not revert terminal commands"). HIDE shows: "reverting here will undo 2 file edits; note these terminal commands also ran after this point: …" — honest, complete, and replayable.

#### 4.5.3 Binding

- **Bound to:** `diff.proposed`, `diff.applied`, `diff.reverted`, `file.changed_external` (to warn if the file moved under the diff).
- **Stores:** `diffStore` (proposed/applied hunks per file), `sourceControlStore` (the review aggregate + checkpoint list), `editorStore` (the Monaco models).

#### 4.5.4 State machine — see §4.8.3.

---

### 4.6 Chat, Editor, Terminal, Explorer, Search, Status Bar

The remaining v1 panels, each with bindings.

#### 4.6.1 Chat

- **Role.** The conversation + **plan-review/steer** surface. Default docked beneath the Context Stack (right rail); can dock to the sidebar.
- **Composer.** Multi-line input with: `@`-mentions (files/symbols/memory → adds a pinned `ContextSource`), `/`-commands (slash-commands are `command`-kind extensions), **drag-drop attachments**, a **model/profile picker**, a **mic button** (voice, later), and **submit / submit-and-queue** (Claude Code prompt-queue pattern: queue a follow-up while the agent runs).
- **Message rendering.** Streamed assistant tokens (from `token`/`token_batch`), collapsible "thinking", inline **plan cards** (editable, §4.7), inline **tool-call chips** (expand to output), inline **diff chips** (→ Diff Review). Selection-to-chat lands here (§4.7).
- **Steer bar.** While a run is live, the composer becomes a **steer bar**: `⏸ Pause`, `⤺ Redirect` (inject guidance into the running loop), `✎ Edit plan`, `⛔ Stop`. (§4.7, §4.8.2.)
- **Bound to:** `turn.*`, `plan.*`, `token`/`token_batch`, `tool.call`/`tool.result` (chips), `diff.proposed` (chips), `error`. **Store:** `chatStore` + `runStore`.

#### 4.6.2 Editor (Monaco)

- **Role.** The body: tabbed, splittable Monaco editor groups (§4.1.2).
- **Agent affordances layered on Monaco:**
  - **Ghost-text / inline completion** — `registerInlineCompletionsProvider`; Tab to accept, Esc to dismiss (Cursor Tab parity). Driven by a completion request to `/v1/hawking/generate` (FIM-shaped) via the kernel; rendered as a Monaco inline-completion decoration.
  - **Inline edit (Cmd+K)** — a small prompt widget anchored over the selection (Monaco view zone + content widget); on submit, the agent returns a hunk that renders **in-place as a diff overlay** with accept/reject (§4.8.1). Cursor Cmd+K parity, but the overlay is a real Monaco diff zone and the result is a `diff.proposed` event (reviewable, undoable, logged).
  - **Hunk gutters** — when a `diff.proposed` targets an open file, accept/reject controls render in the gutter without leaving the editor.
  - **Provenance peek** — `Cmd+Alt+/` on a line the agent wrote → a peek widget tracing it back through the causal DAG (§4.7).
  - **Confidence heat** (opt-in) — decorations coloring low-`logprob` tokens.
  - **Cursor-follow** — during a run, an "agent following" mode (Zed parity) scrolls the editor to where the agent is reading/editing (driven by `tool.call` file refs and `diff.proposed`).
- **Bound to:** `file.changed_external` (reload/conflict prompts), `diff.proposed`/`diff.applied` (gutters/overlays), `token` (inline-completion + confidence). **Store:** `editorStore`.

#### 4.6.3 Terminal (xterm.js + PTY)

- **Role.** A real terminal (xterm.js front + portable-pty/`tauri-plugin-pty` back), and the surface where agent `shell.run` tool calls *also* appear (so the human sees exactly what the agent ran).
- **Design.** Multiple tabs/splits; WebGL renderer; resize → PTY resize. **Agent commands** run in a labeled terminal session ("agent") with output streamed as `tool.progress`/`tool.result` *and* mirrored to the xterm view → the agent's shell activity is **visible and scrollback-able**, and a banner marks agent-initiated commands.
- **Bound to:** PTY data (direct Tauri command channel for the human terminal), `tool.call`/`tool.result`/`tool.progress` for the agent-shell mirror. **Store:** `terminalStore`.
- **Note.** Per the MCP/computer-use guidance and ch.01, shell commands the *agent* runs go through the capability-gated tool dispatcher (recorded as effects); the *human* terminal is a direct PTY. Both render in xterm.

#### 4.6.4 File Explorer

- **Role.** Workspace tree.
- **Design.** Standard tree; **live decorations** from `file.changed_external` (agent or external edits flagged), git status badges (`sourceControlStore`), and **"touched by run"** highlights (files the active run has read/edited, from `tool.call`/`diff.*`). `@`-mention a file straight from the tree into chat.
- **Bound to:** `file.changed_external`, `diff.applied`, `tool.call` (file refs). **Store:** `fileTreeStore`.

#### 4.6.5 Search

- **Role.** Workspace text search.
- **Design.** ripgrep-backed (via a built-in `search` tool / indexer seam), results grouped by file with inline preview + replace; regex + case + whole-word toggles. Feeds the palette's file/symbol search too. (ch.05 owns semantic/symbol search; this is the literal-text panel.)
- **Bound to:** search-tool results (a `tool.result` flavored stream). **Store:** `searchStore`.

#### 4.6.6 Status Bar

- **Role.** The global status seam and a **global steer affordance**.
- **Shows:** branch + sync state, problem counts (⚠/●), cursor Ln/Col, encoding, language mode, **the agent-state pill** (`idle | planning | executing | paused | waiting-approval | error`) with an **inline ⏸/⛔** so you can interrupt from anywhere, and **live tps** (`runtime.stats.dec_tps`).
- **Bound to:** `runtime.status`/`runtime.stats`, `turn.*`/`plan.*` (agent state), `build.status`/`test.status` (problems), editor selection. **Store:** `statusStore` (a thin projection of `runStore` + `runtimeStore`).

---

### 4.7 Interaction paradigms

The grammar of working *with* the agent. Each is a concrete affordance bound to events; together they deliver U1/U2 (show + steer).

#### 4.7.1 Inline edit & ghost-text

- **Ghost-text:** as you type, a dimmed completion appears ahead of the cursor (Monaco inline completion); **Tab** accepts, **Esc** dismisses, **Cmd+→** accepts one word. Cursor-Tab parity. (§4.8.1.)
- **Inline edit (Cmd+K):** select code → `Cmd+K` → type an instruction in the anchored widget → the agent returns a hunk rendered **in-place** as an accept/reject diff overlay. Beats Cursor by making the result a logged `diff.proposed` (reviewable in the timeline, undoable as a compensating event).

#### 4.7.2 Selection-to-chat

- Select code (editor) or text (any panel) → `Cmd+L` (or right-click → "Add to chat") → the selection becomes a **pinned context span** in the composer (a `ContextSource` candidate with `pin: user_pinned`, ch.04) and is quoted in the message. The user is *directly editing what the model sees* — the Context Stack updates to show the pin.

#### 4.7.3 Plan-review-and-steer (approve / edit the plan)

The single most important steering paradigm, lifted from Claude Code Plan Mode + Cline Plan/Act + LangGraph interrupts:

- **Plan-first autonomy:** in `suggest-only` / `plan-first` profiles (ch.01 §4.10), the agent emits `plan.step` events **and pauses before executing** (waiting-approval state). The plan renders as **editable step cards** in chat and the timeline.
- **The human can, before or during execution:**
  - **Approve** the plan (▶ Run) → agent proceeds.
  - **Edit a step** (rewrite title/rationale) → emits an intent that updates the plan the agent will follow (ch.02 honors it).
  - **Reorder / delete / insert** steps (drag, ⌫, +) → restructures the plan.
  - **Approve step-by-step** (Cline per-step approval) → each step waits for ✓ before running.
- **Binding:** `plan.step`/`plan.step_updated` render; `EditPlanStep`/`ReorderPlan`/`ApprovePlan`/`ApproveStep` intents steer. The edited plan is an event → replayable, forkable.

#### 4.7.4 Interrupt / redirect mid-run

- **Interrupt:** `Esc Esc` (or the status-bar ⏸, or the steer-bar Pause) → `CancelRun`/`PauseRun` intent → ch.01 flips the runtime's `abort: Arc<AtomicBool>` (verified in `engine.rs`) → the loop pauses gracefully (AgentScope-style cooperative cancel). State persists (you can resume).
- **Redirect:** while paused (or live, queued), type guidance → `RedirectRun{text}` injects a new instruction into the running loop's context (the agent re-plans from the steer). This is Claude Code's "interrupt and steer," given a GUI.
- **Prompt queue:** `Cmd+Enter` while running → **queue** a follow-up turn ("append next task") vs **Shift+Cmd+Enter** → **override** current task (claude-code#25845 pattern). The queue renders as pending chips below the composer.

#### 4.7.5 "Why did you do that?" — provenance

- **On any agent artifact** (a written line, a tool call, a diff, a memory write): `Cmd+Alt+/` (editor peek) or right-click → **"Why?"** → traces the causal DAG (`parent`/`cause`, ch.01 §4.6): *"This edit ← Plan step 3 'patch expiry' ← reading auth.rs:42 (tool grep) ← your turn 'fix JWT bug' ← retrieved because relevance 0.91 to the query."* Renders as a mini-timeline + the relevant Context-Stack snapshot at that turn. **No cloud agent can answer this** because they don't keep the causal event log.

#### 4.7.6 Accept / reject / undo with checkpoints

- Covered in §4.5: per-hunk accept/reject, compensating-event undo, terminal-aware checkpoint reverts, and timeline scrub/fork as the "deep undo."

#### 4.7.7 Memory editing (later, but designed)

- The Memory viewlet (ch.04 §4.7) opens `.hide/memory/*.md` as editable files; pinning a fact (📌) sets `confidence=1.0, source=user_edit`; conflicts surfaced in the Context Stack resolve here. **"Edit the agent's memory live"** is literally editing files + clicking pins. Marked **later** (ch.04-gated) but the Context Stack already exposes the read + pin path in v1.

---

### 4.8 Key interaction state machines

The three load-bearing state machines the brief asks for. Each is event-driven (ch.01) and shown as states + transitions.

#### 4.8.1 Inline-edit (Cmd+K / ghost-text) state machine

```
            type in editor
   Idle ───────────────────────▶ GhostPending ──(provider returns)──▶ GhostShown
    ▲   ◀──Esc / cursor moves──────┘                                     │
    │                                                       Tab(accept)  │ Esc(dismiss)
    │                                                            ▼        ▼
    │                                                        Accepted    Idle
    │                                                        (insert,
    │                                                         no event — local edit)
    │
    │   Cmd+K on selection
    └──────────────────────▶ EditPrompt ──(submit)──▶ EditGenerating ──(diff.proposed)──▶ EditOverlay
            ▲  Esc(cancel)        │                        │ Esc(abort→CancelRun)            │
            │                     │                        ▼                                 │
            └─────────────────────┘                    Aborted                    ┌──────────┴──────────┐
                                                                          Accept hunk(s)        Reject all
                                                                          → AcceptDiff           → RejectDiff
                                                                          → diff.applied         → Idle
                                                                          → Idle
```

- **GhostShown** is *pre-commit and local* (no event until accepted — it's a UI suggestion); the **Cmd+K path is agentic** (produces `diff.proposed` → reviewable/loggable). Distinguishing these is deliberate: lightweight completions stay out of the log; *edits* are first-class events.
- **Abort** during `EditGenerating` flips the runtime abort flag (ch.01).

#### 4.8.2 Plan-steer state machine

```
  turn.user
     │
     ▼
  Planning ──(plan.step…)──▶ PlanReady ──[autonomy = suggest-only]──▶ AwaitingApproval
     │                          │                                         │
     │ [autonomy=auto-apply]    │                              ┌──────────┼───────────────┐
     ▼                          │                        Approve▶      EditStep/Reorder   Reject
  Executing ◀───────────────────┘                        Executing      (loops back to    → Idle
     │   ▲                                                               PlanReady w/ edit)
     │   │ resume
  ┌──┴───┴────────────────────────────────────────┐
  │ Executing                                       │
  │   ├─(plan.step_updated active/done)─ render     │
  │   ├─ Pause(⏸ / Esc Esc / abort flag) ─▶ Paused ─┤──Resume──▶ Executing
  │   ├─ Redirect(text) ─▶ Replanning ─▶ Executing  │
  │   ├─ Stop(⛔ / CancelRun) ───────────▶ Stopped   │
  │   ├─ step needs approval ─▶ AwaitingApproval ────┘ (per-step mode)
  │   └─(turn.assistant_ended)─▶ Done                │
  └──────────────────────────────────────────────────┘
        │
        ▼
   [from any past state] scrub/fork via Timeline → replay (read-only) or fork(child session)
```

- **AwaitingApproval / Paused / Replanning** are the steering hooks; all are reachable from the status-bar pill and the steer bar (global affordances, U2).
- **Per-step approval** (Cline) is the `Executing → AwaitingApproval → Executing` sub-loop when `autonomy=suggest-each-step`.

#### 4.8.3 Diff-review state machine (per file, and aggregate)

```
  diff.proposed
       │
       ▼
   Proposed ──(open Diff Review tab / inline gutters)──▶ Reviewing
       │                                                    │
       │  file.changed_external (conflict)                  │  edit modified side
       ▼                                                    ▼
    Stale ──(re-diff / rebase)──▶ Reviewing            Reviewing'(user-edited hunk)
                                                            │
                              ┌─────────────────────────────┼───────────────────────────┐
                        Accept hunk                    Accept all                  Reject all/hunk
                        → AcceptDiff{hunk}             → AcceptDiff{file}          → RejectDiff
                        → diff.applied                 → diff.applied (n)          → (no apply)
                        → PartiallyApplied ──remaining──▶ (loop) ── all done ──▶ Applied
                                                                                      │
                                                                                  Undo (⌘Z / revert)
                                                                                      ▼
                                                                              diff.reverted (compensating)
                                                                                  → Reverted
```

- **Stale** handles the agent (or the user, or an external tool) changing the file out from under a pending diff (`file.changed_external`) — the diff re-bases or prompts; never silently applies onto drifted content.
- **PartiallyApplied** is the per-hunk reality: some hunks accepted, some pending — the aggregate review tracks per-file/per-hunk status.

---

### 4.9 The keyboard model & command system

Keyboard-first (U4): **every** action is a `command` (ch.01 §7.1) with a palette entry and a (re-bindable) keybinding. The mouse is always optional.

#### 4.9.1 The command system

- **Commands are extensions.** A `command` contribution declares `{id, title, keybinding?, when?}` (ch.01 §7.2 manifest). The core ships a base set; plugins add more; the registry is the single source (`commandStore`).
- **The Command Palette** (`Cmd+Shift+P`) is the universal entry: **fuzzy search over commands**, scoped by a `when`-clause context (e.g. diff-review commands only when a diff tab is focused), with **recently used** floated to top (VS Code parity). Running a command dispatches its intent.
- **Quick Open** (`Cmd+P`): fuzzy **file** open. Mode prefixes in the same box (VS Code-style):
  - `>` commands · `@` symbols in file · `#` workspace symbols · `:` go-to-line · `?` help · **`§` agent actions** (HIDE-specific: "replay to event…", "re-run step…", "fork session here…", "switch profile…").
- **The `§` agent-action namespace is the HIDE differentiator in the palette** — the timeline's scrub/replay/fork and the Context-Stack's pin/profile actions are all keyboard-reachable.

#### 4.9.2 The keymap (seed; full in Appendix C)

Chords use a leader where natural (`Cmd+K <x>`, VS Code/Cursor parity). Conflicts resolve by `when`-context.

| Action | Default | Notes |
|---|---|---|
| Command Palette | `Cmd+Shift+P` | commands |
| Quick Open (file) | `Cmd+P` | + mode prefixes |
| Toggle Sidebar | `Cmd+B` | |
| Toggle Context Stack | `Cmd+Alt+B` | the differentiator rail |
| Toggle Bottom Panel | `Cmd+J` | |
| Focus Chat | `Cmd+Shift+L` | |
| Selection → Chat | `Cmd+L` | adds pinned span |
| Inline Edit | `Cmd+K` | over selection |
| Accept ghost-text | `Tab` | / `Cmd+→` word |
| Accept hunk / Reject hunk | `Cmd+Enter` / `Cmd+Backspace` | in diff focus (`when` diff) |
| Approve plan / Run | `Cmd+Enter` | in plan focus |
| Queue follow-up turn | `Cmd+Enter` (running) | append; `Shift+Cmd+Enter` = override |
| Interrupt agent | `Esc Esc` | flips abort flag |
| Why? (provenance) | `Cmd+Alt+/` | causal trace |
| Split editor | `Cmd+\` | |
| Agent actions menu | `Cmd+Shift+§` | replay/fork/profile |
| Toggle confidence heat | `Cmd+K C` | logit-derived |

- **Re-bindable.** A `keybindings.json` (per ch.01 config layering, §4.10) overrides defaults; the palette shows the current binding next to each command.

---

### 4.10 The front-end state architecture

The view holds **no authoritative state** (U3 / ch.01 T2). It is a set of **Zustand stores** that are *derived caches* of the projection stream, plus a thin **router** that fans `UiEvent`s to stores, plus an **intent dispatcher** that sends user actions back as Tauri commands.

#### 4.10.1 The pipeline

```
 hawking-serve (SSE) ──┐
                       │  [RUST HOST / hide-kernel — ch.01]
 OS / tools / files ───┼─▶ event log (single writer, seq) ─▶ projection engine ─▶ UiEvent
                       │                                                            │
                       │                          ipc::Channel<UiEvent> (ORDERED)   │  (Wire B)
                       └────────────────────────────────────────────────────────────┘
                                                       │
              ┌────────────────────────────────────────▼─────────────────────────────────┐
              │  WEBVIEW (React + TS)                                                       │
              │   ┌─ channel.onmessage(UiEvent) ─▶ EventRouter ─▶ route by `kind` prefix ─┐ │
              │   │                                   │                                    │ │
              │   │   token/token_batch ─▶ tokenCoalescer ─▶ chatStore/runStore           │ │
              │   │   plan.* ───────────▶ runStore                                         │ │
              │   │   tool.* ───────────▶ runStore + timelineStore                         │ │
              │   │   diff.* ───────────▶ diffStore + sourceControlStore + editorStore     │ │
              │   │   context.* ────────▶ contextStore  (renders ContextManifest, ch.04)   │ │
              │   │   memory.* ─────────▶ memoryStore (later)                               │ │
              │   │   runtime.* ────────▶ runtimeStore + statusStore                        │ │
              │   │   file.changed_external ─▶ fileTreeStore + editorStore                  │ │
              │   │   error ────────────▶ notificationStore + statusStore                   │ │
              │   │   (ALL kinds) ──────▶ timelineStore (universal consumer)                │ │
              │   └────────────────────────────────────────────────────────────────────────┘ │
              │                                                                               │
              │   React components subscribe to store slices (selectors) ─▶ render            │
              │                                                                               │
              │   user action ─▶ intentDispatcher.invoke("hide_intent",{intent}) ─▶ (Wire A) ─┤─▶ kernel
              └───────────────────────────────────────────────────────────────────────────────┘
```

#### 4.10.2 The token coalescer (SSE→store, U3.5)

The host already coalesces `token`→`token_batch` under backpressure (ch.01 §4.4). The front-end adds a **render-rate governor** so even un-coalesced bursts never thrash React:

```ts
// tokenCoalescer: buffer incoming token/token_batch text per run_id; flush on rAF.
const buffers = new Map<RunId, string>();
let scheduled = false;
function onToken(ev: TokenEvent) {
  buffers.set(ev.run_id, (buffers.get(ev.run_id) ?? "") + ev.text);
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(() => {
      for (const [runId, text] of buffers) chatStore.getState().appendTokens(runId, text);
      buffers.clear(); scheduled = false;
    });
  }
}
```

- **One React commit per animation frame**, regardless of token rate → a 120 tok/s stream renders at ≤60 fps with zero dropped *content* (the buffer holds every token; only the *paint* is batched). This is U3.5 realized in ~12 lines.
- **The log is unaffected:** every token is durable upstream (ch.01); the coalescer is purely a paint optimization.

#### 4.10.3 The stores (Zustand; full map in Appendix B)

Each store is a small Zustand slice with a `last_applied_seq` cursor (so on reconnect it can request a replay range, U3) and pure reducers fed by the router. Stores **never** mutate themselves except via routed events or explicit local-UI state (e.g. which tab is active — that's view state, not authoritative).

Core stores: `chatStore`, `runStore`, `timelineStore`, `diffStore`, `contextStore`, `editorStore`, `terminalStore`, `fileTreeStore`, `searchStore`, `sourceControlStore`, `runtimeStore`, `statusStore`, `commandStore`, `notificationStore`, `layoutStore`, (`memoryStore` later).

#### 4.10.4 Reconnect / crash (U3, U7)

On WebView reload or crash, the front-end:
1. Opens the per-window `Channel<UiEvent>` (ch.01 Wire B).
2. Sends a `Hello{last_seqs}` intent with each store's `last_applied_seq` (0 if fresh).
3. The kernel replays the projection from those seqs (ch.01 §4.5) → the stores rebuild → the UI is byte-identical to before the crash. **Nothing durable is lost** (ch.01 T2). Layout restores from `<workspace>/.hide/` (§4.1.2).

---

### 4.11 Notifications & ambient UX for long / overnight runs

"Spend lavishly, locally" (T9) means runs can be long, parallel, and unattended. The UX must make walking-away safe and coming-back reviewable (U9).

- **OS notifications** (Tauri notification plugin): on `turn.assistant_ended`, `error{fatal}`, `runtime.unavailable`, `test.status{failed>0}`, and **"waiting for your approval"** (a paused plan needs you). Click → focuses the relevant run/timeline.
- **Dock/taskbar badge + Activity-Bar "Agent Runs" badge:** count of running + needs-attention runs.
- **The Runs dashboard** (Agent Runs viewlet): a list of all sessions/runs with state pills, progress (active plan step k/n), elapsed, last event, and quick **⏸/⛔/open**. The home for **many parallel agents** (T9) — fan out 8 refactors, watch them here.
- **Quiet hours / batching:** overnight, notifications batch into a digest ("3 runs done, 1 needs approval, 1 failed") rather than buzzing per event. A `notification` policy in config (ch.01 §4.10).
- **Resumability on return:** any run is scrub/replay/fork-able from the timeline (§4.4), so a morning review is "scrub the night's run, accept the good diffs, fork from the point it went wrong" — a workflow cloud's ephemeral sessions cannot offer.
- **Live activity, glanceable:** the status-bar pill + tps + a tiny sparkline of token throughput give an at-a-glance "is it alive and moving" without opening anything.
- **Binding:** `notificationStore` consumes `turn.assistant_ended`, `error`, `runtime.*`, `test.status`, `plan.step_updated{waiting}`; the Runs dashboard binds `runStore` across sessions.

---

### 4.12 Theming & accessibility

- **Theming.** A token-based theme system (CSS variables) shared by the shell, Monaco (its own theme API, synced), and xterm (its theme object, synced) so **one theme drives all three**. Ships dark (default) + light + high-contrast; `theme` is a contribution kind (community themes). Monaco/xterm themes are derived from the same palette tokens to avoid drift.
- **Accessibility (first-class, not bolted on):**
  - **Keyboard-complete** (U4): every action reachable without a mouse; visible focus rings; logical tab order; the palette as the universal fallback.
  - **Screen-reader:** Monaco's accessible mode; ARIA live regions for streamed tokens (announce assistant output politely, not per-token-spam — announce on `token_batch`/turn boundaries); ARIA roles on panels/timeline; the timeline navigable as a list with announced step kinds.
  - **Reduced motion:** respect `prefers-reduced-motion` — no pulsing/animation for the active step, instant (not animated) budget bar.
  - **Color independence:** diff add/remove, plan status, confidence heat all use shape/label + color (never color alone); high-contrast theme verified to WCAG AA.
  - **Font/zoom:** editor + UI font size independently scalable; respects OS text-size.
  - **Voice (later, §3.8):** local-Whisper dictation as an accessibility input — hands-free prompting with zero audio egress.

---

## 5. How we EXCEED (cloud literally cannot do this)

Each item ties to a seam above and to a ch.01/ch.04 contract.

| # | Superpower | Surface (this ch.) | Why cloud literally cannot |
|---|---|---|---|
| 1 | **See the model's entire context, live** | Context Stack renders `ContextManifest` verbatim (§4.3) | Cloud assembles the window server-side and never exposes it; they bill per token so they *can't* afford to show, let alone let you edit, the full window. |
| 2 | **Edit what the agent sees, live** | drag-to-pin dropped spans, resolve conflicts, selection-to-chat, switch profile (§4.3, §4.7.2) | There is no API to reach into a cloud model's context window and pin/drop a span mid-turn. |
| 3 | **Scrub & replay a run from a durable log** | Agent-Run Timeline scrub→`ScrubToEvent`→replay (§4.4) | Cloud sessions are non-deterministic and non-resumable at the event level; they keep truncated transcripts on their server, not a byte-immutable local log with effects recorded (ch.01 T3). |
| 4 | **Edit the agent's plan / memory mid-run, then fork** | plan-step editing + memory pins + fork-from-event (§4.4.2, §4.7.3, §4.7.7) | Cloud agents don't expose an editable plan/memory state you can mutate and deterministically re-run from (ch.01 T6 bit-identity). |
| 5 | **"Why did you do that?" causal provenance** | `Cmd+Alt+/` traces the `parent`/`cause` DAG (§4.7.5) | Cloud keeps no causal event graph you can walk from a written line back to the user turn. |
| 6 | **Per-hunk, terminal-aware, undoable checkpoints** | Diff Review + compensating-event undo (§4.5) | Cursor's checkpoints are file-only, git-separate, and *don't revert terminal commands* (ch.01 §3); HIDE's are event-log-backed and effect-complete. |
| 7 | **Zero-latency rich UI** | persistent panels, per-turn manifest re-render, optimistic UI (U6) | Cloud pays a network round-trip per interaction; a live context panel re-rendering every turn over the wire is infeasible. Local `Channel` is ~ms (ch.01 §3). |
| 8 | **Many parallel agents + overnight runs, reviewable** | Runs dashboard + batched notifications + morning scrub (§4.11) | Per-token billing + rate limits make "8 agents all night, every token logged" economically impossible; and they have no scrubbable timeline to review it. |
| 9 | **Model confidence on tap** | logit-derived confidence heat (§4.3, §4.6.2) | Cloud exposes at most top-k logprobs; HIDE can read the full logit row (`GenStats.logits_materialized_*`) for true per-token confidence. |
| 10 | **Tear-off panels, multi-window, full desktop integration** | float a timeline/editor to its own OS window (§4.1.2) | A browser tab cannot become native multi-window desktop surfaces bound to the same live event stream. |
| 11 | **Air-gapped, private voice + everything local** | local-Whisper composer, no egress (§3.8, §4.12) | A cloud IDE *is* the egress path; HIDE's default is air-gappable (ch.01 §5). |

**The one-line version:** *Cloud gives you a chat box over a black box. HIDE gives you a glass box you can reach into, rewind, and re-run.*

---

## 6. Failure modes / edge cases / mitigations

| # | Failure / edge case | Mitigation |
|---|---|---|
| F1 | **Event flood melts the UI** (120 tok/s × N runs, rapid tool progress). | Two-layer coalescing: host `token`→`token_batch` (ch.01 §4.9) **+** front-end rAF governor (§4.10.2) → one paint/frame. Latest-only for `tool.progress`/cursor. The log keeps everything; only paint is batched (U3.5). |
| F2 | **Huge diff** (agent rewrites a 5k-line file or 200 files). | Monaco diff **virtualizes** (renders viewport only); the multibuffer aggregate lazy-mounts per-file diffs (collapsed by default, expand on demand); hunk count + "+N −M" summary up front; "review file-by-file" mode caps mounted editors. Post-images are blob-refs (ch.01 §4.7), not inlined. |
| F3 | **Run-away agent in the UI** (loops, spawns endless tool calls, fills the timeline). | The producer-slows backpressure means it can't outrun durability (ch.01 §4.9); the timeline **groups repeated identical steps** ("× 14 grep") instead of 14 cards; the status-bar ⛔ and `Esc Esc` interrupt from anywhere; a per-run **step/cost budget** (config) auto-pauses to `AwaitingApproval` on breach. |
| F4 | **WebView crash / reload mid-stream.** | View holds no authoritative state (U3); on reconnect, `Hello{last_seqs}`→replay→identical UI (§4.10.4). Layout restores from disk (§4.1.2). Nothing durable lost (ch.01 T2). |
| F5 | **Runtime goes Degraded/Down mid-run.** | `runtime.unavailable` → the run **pauses** (not fails); status pill shows ⏸ "runtime restarting"; the supervisor restarts (ch.01 §4.3); on Ready the user resumes from the timeline. A blocking banner only on `Failed` with "switch provider" (model providers are extensions). |
| F6 | **Stale diff** (file changed under a pending proposal). | Diff-review `Stale` state (§4.8.3): re-diff/rebase or prompt; **never** apply onto drifted content. Driven by `file.changed_external`. |
| F7 | **Conflicting edits** (agent + human edit same file). | Last-writer is an event; the editor shows a conflict decoration; the diff rebases; `sourceControlStore` exposes the conflict for manual resolve. (Full CRDT merge is a ch.01 §8 moonshot.) |
| F8 | **Manifest too big to render** (hundreds of spans/dropped). | Context Stack sections are **collapsed with counts** by default (U8), virtualized lists on expand; "Dropped (127) ▸" never renders 127 rows until opened. |
| F9 | **Scrub to a seq beyond the in-memory window.** | `timelineStore` requests a log range via intent; the kernel folds from the nearest snapshot (ch.01 §4.5); a spinner during the (bounded, snapshot-backed) replay. |
| F10 | **Notification storm** (overnight, many runs). | Quiet-hours batching → digest (§4.11); per-kind notification policy (config). |
| F11 | **Plugin panel hangs / traps** (a third-party WASM panel). | Panel render is sandboxed (ch.01 §7.4 fuel/epoch/memory); a hung panel is killed and replaced with an error tile; the *shell* never blocks (the panel is one tile, not the app). 3 faults/60 s auto-disables (ch.01 F8). |
| F12 | **Two windows, one workspace, divergent layout.** | Layout is per-window view state; the authoritative log is shared (ch.01 `runtime.lock` shared-daemon). Both windows subscribe to the same projection; edits reconcile through the log, not the layout. |
| F13 | **Inline-completion latency feels laggy.** | Ghost-text requests are debounced + cancellable (flip abort on keystroke); a fast draft model / `/v1/hawking/generate` FIM path keeps it sub-100 ms locally; never blocks typing (suggestion is async overlay). |
| F14 | **Confidence heat misleads** (low logprob ≠ wrong). | Opt-in, off by default; labeled "model confidence (logit entropy), not correctness"; uses a calibrated bucket, not raw logprob; shape+color (U8/§4.12). |
| F15 | **Accessibility regressions from streaming spam.** | ARIA live announces on `token_batch`/turn boundaries, not per token; reduced-motion honored; keyboard-complete is a CI check (§4.12). |
| F16 | **Replay shows an effect that the UI implies is re-runnable.** | "Replay mode" banner makes read-only explicit; the only path to re-execute is the explicit **Resume/Fork** buttons (§4.4.2); mirrors ch.01 F11 (replay has no dispatcher access). |

---

## 7. Extensibility / plugin points (panels as extensions)

Every surface in this chapter is an **extension** bound to ch.01's manifest (§7.2) and capability spine — adding UI touches **zero** `core/` files.

1. **Panels** (`[[contributes.panels]]`, ch.01 §7.2): `{id, title, mount: left-dock|right-dock|bottom-dock|editor, bundle}`. A plugin ships a web bundle mounted at a dock; it subscribes (with `events:subscribe` capability) to declared event kinds and renders. *This is how the Clippy/Test/Memory/Model-Lab panels and third-party panels all plug in.* The shell provides the **mount point + the scoped event subscription + the store-read API**; the plugin provides the React (or WASM-UI) bundle.
2. **Commands** (`[[contributes.commands]]`): `{id, title, keybinding, when}` → appear in the palette, bind a key, dispatch an intent. The keymap and palette are *open* by construction.
3. **Slash-commands** in chat are `command`-kind contributions surfaced in the composer.
4. **Themes** are a contribution kind (palette tokens → shell + Monaco + xterm).
5. **Status-bar items**: a panel/command can contribute a status-bar segment (scoped) — e.g. a linter showing counts.
6. **Editor decorations / inline widgets**: a plugin with editor capability can contribute Monaco decorations (e.g. a coverage gutter) via a scoped editor API — not raw Monaco access (capability-gated).
7. **Context-Stack sections**: a `ContextSource` plugin (ch.04 §7) that contributes a new source also gets a **Context-Stack section** to render its spans (e.g. a "Tickets" source shows a Tickets section) — the rail is *open*, not a fixed list.
8. **The Model Lab / Store** is itself just a **panel extension marked LATER**; when HF distribution lands (post-32B), it registers like any panel. No shell rework.

**Binding guarantees** (inherited from ch.01 §7.2): panel event subscriptions are **capability-scoped** (a panel sees only the kinds it declared and was granted); sandboxed WASM-UI panels are **fuel/epoch/memory-bounded**; a panel **cannot** reach authoritative state or other panels' stores except through granted, scoped host APIs (no ambient authority, T4). The UI surface is therefore as extensible as VS Code and as safe as Zed (ch.01 §3).

---

## 8. Bleeding-edge / moonshots (ranked)

Ranked by **(impact × feasibility)**; tagged PROVEN-substrate / SPECULATIVE.

1. **Full glass-box Context Stack + live pinning (PROVEN substrate; M build, VERY HIGH impact).** The `ContextManifest` exists (ch.04 A.1); the rail renders it; pin/resolve/profile are intents. *This is the product's signature — build it first.* No one ships this.
2. **Scrubbable, forkable Agent-Run Timeline (PROVEN substrate; M build, VERY HIGH impact).** The event log + replay exist (ch.01 §4.5); this is the UI. Edit-then-fork (§4.4.2) is the demo that wins. **Do it.**
3. **Terminal-aware, event-log-backed checkpoints in Diff Review (PROVEN substrate; M build, HIGH).** Beats Cursor's headline limitation directly (file-only/terminal-blind). Compensating-event undo is already the ch.01 model.
4. **Causal "Why?" provenance peek (PROVEN substrate; S-M build, HIGH).** The `parent`/`cause` DAG is in every event (ch.01 §4.6); the peek is a graph walk + render. Uniquely ours.
5. **Optimistic / speculative UI (SPECULATIVE; M build, MED).** Render the draft model's tokens ahead of verify, reconcile on the authoritative event — the UI analog of spec-decode (ch.01 §8 #6). Local-only (needs raw draft tokens, T7). Risk: flicker; prototype behind a flag.
6. **Tear-off multi-window + remote-runtime "thin client" (PROVEN substrate; M build, MED-HIGH).** Tauri multiwindow + ch.01's HTTP runtime → timeline on monitor 2, editor on monitor 1, runtime on a beefier LAN box. The lock-file/provider abstraction already allows it.
7. **Local-Whisper voice composer (PROVEN-niche; M build, MED, HIGH accessibility).** Mic in chat, local Whisper as a tool/provider, zero egress (§3.8). Accessibility + speed; privacy beat over cloud voice.
8. **CRDT multiplayer presence (PROVEN substrate; H build, MED).** The event log *is* an op-log (ch.01 §8 #4); layer Zed/Automerge-style cursors/avatars for human+human or human+N-agents on one live session. Reserve now (ULID ids, `parent`), defer the merge engine.
9. **Speculative-execution preview ("dry-run this plan") (SPECULATIVE; H build, MED).** Fork to a shadow worktree, execute the plan there, render the *would-be* diffs/tests in the timeline without touching the real tree — "preview the agent's whole run before it runs." Ties to ch.01 fork + a sandbox worktree.
10. **Adaptive layout / "the UI follows the work" (SPECULATIVE; M build, LOW-MED).** Auto-switch layout presets by agent state (planning→Agent preset, diff arrives→Review preset), with a calm transition (reduced-motion-aware). Risky (surprising); opt-in.
11. **Live energy/observability dashboard (PROVEN substrate; M build, MED).** Surface `runtime.stats` + dispatch traces (the repo's `dispatch_samples`, `readback_bytes`, J/tok energy verdict) as a panel — "watch the GPU work." A unique local-first flex.

---

## 9. Open questions / dials

| # | Question / dial | Default | Trade-off |
|---|---|---|---|
| Q1 | **Chat dock: right-rail (under Context Stack) vs left sidebar vs bottom.** | right rail, under Context Stack | Keeps "what the agent sees" + "what it says" adjacent **vs** vertical space. User-movable (panel extension). |
| Q2 | **Context Stack on by default?** | **Yes** (it's the differentiator, U1) | Discoverability of the signature feature **vs** calm/space for beginners (U8). Mitigate: sections collapsed with counts. |
| Q3 | **Coalescing granularity (rAF vs every N ms vs per-panel).** | rAF (§4.10.2) | Smoothness vs fidelity; the *log* always has every token (ch.01 Q7). Per-panel override for the timeline. |
| Q4 | **Default autonomy.** | `suggest-with-tests` (plan-first, auto-apply low-risk, approve risky) | Safety vs flow. Cline-style categories; per-profile (ch.01 §4.10). |
| Q5 | **Inline completion model/path.** | draft model via `/v1/hawking/generate` (FIM) | Latency vs quality; a dedicated tiny completion model later (ch.06). |
| Q6 | **Confidence heat metric.** | calibrated bucket of token entropy, opt-in | Usefulness vs misleading-ness (F14). Needs the full-logits path (runtime-side, opt-in). |
| Q7 | **Timeline orientation (horizontal filmstrip vs vertical log).** | vertical in panel; horizontal when popped to editor tab | Density vs scannability. |
| Q8 | **Plugin panel UI tech: React-in-iframe vs WASM-UI vs web-component.** | sandboxed bundle at a mount (ch.01 §7.4) | Ecosystem familiarity (React) vs strict sandbox (WASM-UI). Default React bundle, capability-gated host API. |
| Q9 | **How much of the diff is editable inline before accept.** | full modified side editable (Zed parity) | Power vs accidental-edit risk; edits become user-authored `diff.proposed`. |
| Q10 | **Notification verbosity / quiet-hours defaults.** | digest overnight, per-event daytime | Noise vs awareness; per-kind policy (config). |
| Q11 | **Layout as event vs pure view-state.** | pure view-state, durable on disk (not in the log) | Cleanliness (log = authoritative agent state, ch.01 T2) vs "replay restores layout too." Chosen: separate, per ch.01. |
| Q12 | **Memory editor in v1 or v1.1.** | v1.1 (ch.04-gated); Context-Stack read+pin in v1 | Ship the core first; memory editing is high-value but additive (§4.2). |

---

## 10. Cross-references

- **ch.01 · System Architecture & Extensibility Spine.** This chapter is the **consumer** of ch.01's `Event` envelope and ~30-kind taxonomy (§4.6), renders projections delivered over the **ordered `Channel<UiEvent>`** (Wire B), sends user actions as **intents** (Wire A), and mounts every panel via the **manifest/capability spine** (§7.2). The scrub/replay/fork UX is the visual face of ch.01's **deterministic replay** (§4.5, effects-recorded-not-refired). Backpressure/coalescing inherits ch.01 §4.4/§4.9. **Binds:** `Event` schema, `ipc::Channel<UiEvent>`, the manifest, the intent set.
- **ch.02 · Agent Kernel & Reasoning Loop.** Owns plan *generation*, autonomy policy, and the agent loop; **this chapter renders and steers** (`plan.*` cards, approve/edit/reorder, interrupt/redirect via the `abort` flag, prompt queue). Outcome telemetry (accept/edit of diffs and pins) flows back to ch.02. **Binds:** `turn.*`/`plan.*` events; the steer intents (`ApprovePlan`/`EditPlanStep`/`RedirectRun`/`CancelRun`).
- **ch.03 · Editor Surface & Diff/Apply.** Owns diff *computation* and apply/merge + the checkpoint *engine*; **this chapter renders** the Monaco diff, routes per-hunk `AcceptDiff`/`RejectDiff`, and shows compensating-event undo. **Binds:** `diff.*`/`file.*` events. *(Note: if ch.03 is the Tools/MCP chapter in the final numbering, the diff-render contract still holds against `diff.*`.)*
- **ch.04 · Context Engineering & Memory.** Provides the **`ContextManifest`** (Appendix A.1) the **Context Stack renders verbatim** (§4.3), the **profiles** the model picker switches (§4.3.1), and the **memory** the Memory editor edits (§4.7.7, later). Pin/resolve/profile-switch intents feed ch.04's compiler. **Binds:** `ContextManifest`, `ContextProfile`, `MemoryRecord`; `context.*`/`memory.*` events.
- **ch.05 · Codebase Intelligence.** Produces the retrieval hits / symbol graph the Context Stack and Search render (`context.retrieval`, symbols); provides go-to-definition targets. **Binds:** `context.retrieval`, symbol provenance.
- **ch.06 · Model Runtime & Providers.** Owns the runtime internals behind the HTTP surface; provides `runtime.status`/`runtime.stats` (the status pill, tps, model section), the SSE token stream, the `abort` flag (interrupt), `json_mode` (constrained UI), the full-logits path (confidence heat), and the `embed()` powering relevance. **Binds:** `runtime.*`/`token` events; `ModelProvider`/`ProviderCaps`. **Verified in-tree:** `crates/hawking-serve/src/http.rs` (routes), `crates/hawking-core/src/engine.rs` (`StreamEvent`, `GenStats`, `abort`, `json_mode`, `logits_materialized_*`).
- **Repo runtime/front-end ground truth.** Stack fixed at Tauri 2 + React + TS + Monaco + xterm.js (brief + ch.01 §3); runtime HTTP surface confirmed in `crates/hawking-serve/src/http.rs`; streaming/abort/stats in `crates/hawking-core/src/engine.rs`.

---

## Appendix A — The panel inventory table (binding)

> The normative panel set. **`v1` = build first (the central shell deliverable); `later` = designed, marked, additive.** "Binds (events)" are ch.01 §4.6 kinds; "Store" are Appendix B stores. Mount points are ch.01 §7.2 manifest values.

| Component | Mount | v1/later | Binds (events) | Store(s) | Intents emitted |
|---|---|---|---|---|---|
| **Editor (Monaco)** | `editor` | **v1** | `file.changed_external`, `diff.proposed`, `diff.applied`, `token` (inline+confidence) | `editorStore` | `OpenFile`, `SaveFile`, `InlineEdit`, `AcceptCompletion`, `AcceptDiff`/`RejectDiff` (gutter) |
| **Chat** | `right-dock` (or `left-dock`) | **v1** | `turn.*`, `plan.*`, `token`/`token_batch`, `tool.*` (chips), `diff.proposed` (chips), `error` | `chatStore`, `runStore` | `SubmitTurn`, `QueueTurn`, `RedirectRun`, `ApprovePlan`, `EditPlanStep`, `PinSpan` |
| **Agent-Run Timeline** | `bottom-dock` / `editor` | **v1** | **ALL kinds** (universal consumer) | `timelineStore`, `runStore` | `ScrubToEvent`, `ResumeRun`, `ForkSession`, `RerunStep` |
| **Diff Review** | `editor` | **v1** | `diff.proposed`, `diff.applied`, `diff.reverted`, `file.changed_external` | `diffStore`, `sourceControlStore`, `editorStore` | `AcceptDiff`, `RejectDiff`, `RevertDiff`, `EditHunk` |
| **Context Stack (right rail)** | `right-dock` | **v1** | `context.manifest`(per-turn), `context.update`, `context.retrieval`, `memory.written`, `runtime.status`/`stats`, `token` (confidence) | `contextStore`, `runtimeStore` | `PinSpan`, `UnpinSpan`, `ResolveConflict`, `SwitchProfile`, `ToggleConfidence` |
| **Terminal** | `bottom-dock` | **v1** | PTY data (direct), `tool.*` (agent-shell mirror) | `terminalStore` | `RunCommand`, `PtyInput`, `PtyResize` |
| **File Explorer** | `left-dock` | **v1** | `file.changed_external`, `diff.applied`, `tool.call` (file refs) | `fileTreeStore`, `sourceControlStore` | `OpenFile`, `RevealInExplorer`, `MentionInChat` |
| **Search** | `left-dock` | **v1** | search-tool `tool.result` stream | `searchStore` | `RunSearch`, `ReplaceInFiles`, `OpenMatch` |
| **Command Palette** | overlay | **v1** | (reads `commandStore`, `fileTreeStore`, symbols) | `commandStore` | dispatches any command's intent |
| **Status Bar** | `status` | **v1** | `runtime.status`/`stats`, `turn.*`/`plan.*`, `build.status`/`test.status` | `statusStore` | `PauseRun`, `CancelRun`, `OpenProblems` |
| **Problems** | `bottom-dock` | **v1** | `build.status`, `test.status`, LSP diagnostics | `diagnosticsStore` | `OpenProblem`, `QuickFix` |
| **Notifications / Toasts** | overlay | **v1** | `turn.assistant_ended`, `error`, `runtime.*`, `test.status`, `plan.step_updated{waiting}` | `notificationStore` | `FocusRun`, `DismissNotification` |
| **Test Output** | `bottom-dock` | **v1** (panel) | `test.status`, `tool.result` (test runner) | `testStore` | `RerunTests`, `OpenFailure` |
| **Source Control** | `left-dock` | **v1** (basic) | `diff.applied`, `file.changed_external` | `sourceControlStore` | `Stage`, `Commit`, `RevertFile` |
| **Memory viewlet/editor** | `left-dock` / `editor` | **later** | `memory.written` | `memoryStore` | `EditMemory`, `PinMemory`, `QuarantineMemory`, `ResolveConflict` |
| **Test Explorer (tree)** | `left-dock` | **later** | `test.status` | `testStore` | `RunTest`, `RunSuite` |
| **Model Lab / Store** | `left-dock` / `editor` | **later (placeholder)** | `runtime.status`, (HF distribution events — TBD) | `modelLabStore` | `InstallModel`, `LoadModel`, `LoadLora` |
| **Multiplayer presence** | overlay/editor | **later (moonshot)** | CRDT presence events | `presenceStore` | `FollowPeer` |
| **Speculative UI overlay** | `editor` | **later (moonshot)** | draft `token` stream | `editorStore` | — |
| **Voice composer** | overlay (in Chat) | **later** | local-Whisper tool result | `chatStore` | `Dictate` |
| **Energy/Obs dashboard** | `bottom-dock` | **later** | `runtime.stats` (+ dispatch traces) | `runtimeStore` | `ToggleTrace` |

---

## Appendix B — The front-end store/event-binding map (binding)

> The normative Zustand store map. Every store is a derived cache of the projection stream with a `last_applied_seq` (for reconnect-replay, U3); reducers are pure and fed by the `EventRouter`. **Other chapters that emit events can rely on these stores being the canonical front-end sinks for their kinds.**

```
EventRouter (routes UiEvent by `kind` prefix) ─▶ stores:

chatStore           ← turn.*, token/token_batch (via tokenCoalescer), plan.* (cards), tool.* (chips), diff.proposed (chips), error
                      state: messages[], streaming buffers, composer, queued turns, plan cards
runStore            ← turn.*, plan.*, tool.*, runtime.status; the active-run state machine (§4.8.2)
                      state: runs{ id → {state, activePlanStep, steps[], elapsed} }
timelineStore       ← ALL kinds (universal); ordered by seq, threaded by parent/cause
                      state: cards[] (windowed), scrubSeq, mode: live|review, snapshots cursor
diffStore           ← diff.proposed, diff.applied, diff.reverted
                      state: diffs{ diff_id → {path, hunks[], status} }
sourceControlStore  ← diff.applied, file.changed_external
                      state: reviewAggregate, checkpoints[], git status
contextStore        ← context.manifest (per-turn), context.update, context.retrieval, memory.written, runtime.*
                      state: currentManifest (ContextManifest, ch.04 A.1), manifestRing[] (per-turn, for scrub)
editorStore         ← file.changed_external, diff.proposed/applied (gutters/overlays), token (inline/confidence)
                      state: openModels, decorations, ghostText, inlineEditWidget, confidenceHeat
terminalStore       ← PTY data (direct command channel), tool.* (agent-shell mirror)
                      state: terminals[] (xterm instances), agentSessionId
fileTreeStore       ← file.changed_external, diff.applied, tool.call (file refs)
                      state: tree, touchedByRun, gitBadges
searchStore         ← search-tool tool.result stream
                      state: query, results grouped by file
runtimeStore        ← runtime.status, runtime.stats, runtime.unavailable
                      state: model, state pill, tps, profile, tiersTouched
statusStore         ← (projection of runStore + runtimeStore + diagnosticsStore)
                      state: branch, problems counts, cursor, agent pill, tps
diagnosticsStore    ← build.status, test.status, LSP diagnostics
                      state: problems[]
testStore           ← test.status, tool.result (test runner)
notificationStore   ← turn.assistant_ended, error, runtime.*, test.status, plan.step_updated{waiting}
                      state: toasts[], badge counts, quiet-hours
commandStore        ← (reads registry; not event-fed) commands{} from manifest contributions
layoutStore         ← (local view-state, durable on disk — NOT the log) WorkspaceLayout (§4.1.2)
memoryStore (later) ← memory.written; reads .hide/memory/* via tool

Intent path (Wire A): components ─▶ intentDispatcher.invoke("hide_intent", {intent}) ─▶ kernel
  intents include: SubmitTurn, QueueTurn, RedirectRun, CancelRun, PauseRun, ResumeRun,
  ApprovePlan, ApproveStep, EditPlanStep, ReorderPlan, AcceptDiff, RejectDiff, RevertDiff,
  EditHunk, ScrubToEvent, ForkSession, RerunStep, PinSpan, UnpinSpan, ResolveConflict,
  SwitchProfile, ToggleConfidence, OpenFile, SaveFile, InlineEdit, AcceptCompletion,
  RunCommand, PtyInput, PtyResize, RunSearch, ReplaceInFiles, Hello{last_seqs}.
```

**Reconnect contract (U3/U7):** on WebView (re)load each store reports `last_applied_seq` in a `Hello` intent; the kernel replays the projection from the minimum seq; stores rebuild to byte-identical state (ch.01 §4.5/§4.12). Layout restores from disk independently.

---

## Appendix C — The default keymap (binding seed)

> The seed keymap (macOS `Cmd`; Linux/Windows `Ctrl`). Re-bindable via `keybindings.json` (ch.01 config layering §4.10). `when` resolves chord/context conflicts. This is a *seed* — `command` extensions add more.

| Command id | Key | `when` |
|---|---|---|
| `palette.commands` | `Cmd+Shift+P` | always |
| `palette.quickOpen` | `Cmd+P` | always |
| `palette.agentActions` | `Cmd+Shift+§` | always |
| `view.toggleSidebar` | `Cmd+B` | always |
| `view.toggleContextStack` | `Cmd+Alt+B` | always |
| `view.toggleBottomPanel` | `Cmd+J` | always |
| `view.splitEditor` | `Cmd+\` | editorFocus |
| `layout.preset.focus` | `Cmd+K Z` | always |
| `chat.focus` | `Cmd+Shift+L` | always |
| `chat.selectionToChat` | `Cmd+L` | editorTextFocus |
| `chat.submit` | `Enter` | chatComposerFocus && !running |
| `chat.queueTurn` | `Cmd+Enter` | chatComposerFocus && running |
| `chat.overrideTurn` | `Shift+Cmd+Enter` | chatComposerFocus && running |
| `editor.inlineEdit` | `Cmd+K` | editorTextFocus |
| `editor.acceptCompletion` | `Tab` | inlineSuggestVisible |
| `editor.acceptCompletionWord` | `Cmd+Right` | inlineSuggestVisible |
| `editor.dismissCompletion` | `Esc` | inlineSuggestVisible |
| `editor.whyProvenance` | `Cmd+Alt+/` | editorTextFocus |
| `editor.toggleConfidence` | `Cmd+K C` | editorFocus |
| `diff.acceptHunk` | `Cmd+Enter` | diffFocus |
| `diff.rejectHunk` | `Cmd+Backspace` | diffFocus |
| `diff.acceptAll` | `Cmd+Shift+Enter` | diffReviewFocus |
| `plan.approve` | `Cmd+Enter` | planFocus |
| `plan.editStep` | `Enter` | planStepFocus |
| `agent.interrupt` | `Esc Esc` | running |
| `agent.pause` | (status-pill click) | running |
| `timeline.scrubBack` | `Cmd+[` | timelineFocus |
| `timeline.scrubForward` | `Cmd+]` | timelineFocus |
| `timeline.jumpToLive` | `Cmd+End` | timelineFocus |
| `timeline.forkHere` | `Cmd+Shift+F` | timelineFocus |
| `terminal.toggle` | `Ctrl+\`` | always |
| `file.save` | `Cmd+S` | editorFocus |
| `editor.closeTab` | `Cmd+W` | editorFocus |

---

## Appendix D — Source register

Tagged at point of use as **[PROVEN]** (a shipping tool) / **[SPECULATIVE]** (research/emerging). Where a vendor doc makes a UX claim, it is cited as the vendor's description, not independently measured.

**Cursor.** DeployHQ, "Cursor guide" (Composer/Agent/Tab/Inline-edit), deployhq.com/guides/cursor. · CallMissed, "Cursor Composer in 2026" (functional-minimalism, editor-grade UX), callmissed.com/en/blog/cursor-composer-in-2026-how-it-reshaped-editing. · cursor.com (Tab, Cmd+K, Composer).

**Claude Code.** Anthropic, "Best practices for Claude Code," anthropic.com/engineering/claude-code-best-practices. · A. Ronacher, "What Actually Is Claude Code's Plan Mode?", lucumr.pocoo.org/2025/12/17/what-is-plan-mode/. · Anthropic, "How we built Claude Code auto mode," anthropic.com/engineering/claude-code-auto-mode. · "Prompt Queue with Steer Controls," github.com/anthropics/claude-code/issues/25845.

**Cline / Roo Code.** cline.bot (Plan/Act, auto-approve categories, per-tool-call checkpoints). · "Roo Code Review — The Cline Fork," openaitoolshub.org/en/blog/roo-code-review (side-by-side Diff, Modes). · "Cline vs Roo Code vs Continue (2026)," devtoolreviews.com.

**Zed.** "Agent Panel," zed.dev/docs/ai/agent-panel (multibuffer review, per-hunk Keep/Reject, editable unified diff, agent following, context-window usage). · "2025 Recap," zed.dev/2025. · "Agent Panel and UI," deepwiki.com/zed-industries/zed/8.1-agent-panel-and-ui.

**OpenHands.** "OpenHands: An Open Platform…," ICLR 2025 (event stream, replay). · "The OpenHands Software Agent SDK," arXiv:2511.03690 (pure-function-from-event-history, deterministic replay). · "Observability & Tracing," docs.openhands.dev/sdk/guides/observability. · "Event Storage and Replay," deepwiki.com/All-Hands-AI/OpenHands/12.2-event-storage-and-replay.

**Command palette / workbench.** "Command Palette in VS Code," code.visualstudio.com / stevekinney.com. · "Command Palette | UX Patterns," uxpatterns.dev/patterns/advanced/command-palette. · "User interface," code.visualstudio.com/docs/getstarted/userinterface. · "Custom Layout," code.visualstudio.com/docs/configure/custom-layout. · "Workbench Layer" architecture (Titlebar/ActivityBar/Sidebar/EditorArea/AuxiliaryBar/Panel/StatusBar).

**Human-in-the-loop / steering.** LangChain, "Making it easier to build human-in-the-loop agents with interrupt," langchain.com/blog. · "Architecting Human-in-the-Loop Agents… LangGraph" (graph- vs node-level interrupts, checkpointed resume). · "AgentScope 1.0," arXiv:2508.16279 (real-time steering via asyncio cancellation of the ReAct loop). · "When Users Change Their Mind: Evaluating Interruptible Agents," arXiv:2604.00892. · "Inspect, interrupt, redirect" interaction-design framing (co-plan, take over, approve/reject).

**Voice coding.** Wispr Flow, wisprflow.ai/vibe-coding (dictate code, 3× faster). · BridgeVoice, bridgemind.ai/products/bridgevoice. · A. Osmani, "Speech-to-Code: Vibe Coding with Voice," addyo.substack.com. · SuperWhisper (local Whisper, coding-optimized).

**Front-end stack.** Monaco, "IDiffEditorOptions" (side-by-side/inline, `useInlineViewWhenSpaceIsLimited`, decorations zIndex, view zones), microsoft.github.io/monaco-editor. · @monaco-editor/react, npmjs.com/package/@monaco-editor/react. · tauri-plugin-pty, crates.io/crates/tauri-plugin-pty. · "Terax: A 7MB AI-Native Terminal Built with Tauri 2 and Rust," betterstack.com (Tauri 2 + React 19 + xterm.js + Zustand). · Tauri, "State Management," v2.tauri.app/develop/state-management. · @xterm/xterm, npmjs.com.

**In-tree ground truth (HIDE's own).** `crates/hawking-serve/src/http.rs` (routes: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/hawking/generate`, `/v1/hawking/tokens`, `/healthz`, `/metrics`). · `crates/hawking-core/src/engine.rs` (`StreamEvent::{Token{id,text},Done{reason,stats}}`, `GenStats::dec_tps()`, `GenerateRequest.abort: Arc<AtomicBool>`, `json_mode`, `logits_materialized_*`/`readback_bytes`). · ch.01 §4.6 event taxonomy; ch.04 Appendix A.1 `ContextManifest`.
