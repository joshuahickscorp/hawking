# HIDE Two-Surface Architecture

Run date: 2026-07-19 · Grounding: `HIDE_LIVE_ARCHAEOLOGY.md`, `HIDE_CLAUDE_CODE_UX_GENOME.md`, `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json`.
This is the anchor document; `HIDE_CHAT_SPEC.md` and `HIDE_IDE_SPEC.md` elaborate the two surfaces against it.

## 1. The invariant: two views, one session

HIDE Chat (terminal-style agent session) and HIDE IDE (VS Code/Cursor-class workbench) are **not two products**. They are two renderings of a single durable session. The live archaeology confirms this is already the frontend's shape: `app/` has both a Chat chamber and a Code chamber over **one Zustand store and one typed wire contract** (`wire.ts`), sharing one event stream. What is missing is the backend that makes the shared session real (the `/v1/hide/*` boundary is packed; the store is mock-fed in dev).

The seven things the two surfaces share (never duplicated):

```text
one session identity          one context state          one memory system
one tool/effect ledger        one repository snapshot     one agent tree
one verification state        one event stream            one model/runtime state
```

Claude Code has already proven users want this: CLI and the IDE extension share ONE conversation history + checkpoints ([DOCUMENTED]; a chat started in the extension continues with `--resume` in the terminal). HIDE's differentiator is that its shared state is a **local warm capsule**, not a re-sent transcript, so a handoff between surfaces costs a pointer copy, not a re-prefill.

## 2. Layered architecture (the shared core under both surfaces)

Both surfaces are thin clients over one local backend. This is the target that reconnects the packed spine.

```text
  HIDE Chat  (terminal-style)        HIDE IDE  (workbench)
        \                               /
         \        one typed wire       /        <- wire.ts contract (intents, events, projections)
          \      (/v1/hide/*)         /
           v                         v
  ┌─────────────────────────────────────────────┐
  │  Session core (durable truth plane)          │  event log, repo snapshot, tool/effect ledger,
  │                                              │  checkpoints, memory, permission/trust state
  ├─────────────────────────────────────────────┤
  │  Agent kernel (flat loop) + fleet scheduler  │  see HIDE_AGENT_KERNEL_OPTIONS
  ├─────────────────────────────────────────────┤
  │  Context OS (index + compiler + memory)      │  see HIDE_CONTEXT_OS_SPEC
  ├─────────────────────────────────────────────┤
  │  Action plane: typed tools + MCP + LSP/DAP   │  see HIDE_TOOL_SKILL_PLUGIN_MCP_ABI
  ├─────────────────────────────────────────────┤
  │  Verification plane: oracles, tests, review  │
  ├─────────────────────────────────────────────┤
  │  Model plane: Hawking local lanes + router   │  see HIDE_LOCAL_MODEL_TOPOLOGY
  ├─────────────────────────────────────────────┤
  │  State/serving plane: prompt ABI, capsules,  │  see HIDE_STATE_CAPSULE_ABI, HIDE_SPEED_FRONTIER
  │  batching, prefix/state caches               │
  └─────────────────────────────────────────────┘
  Security + provenance wrap every boundary (HIDE_SECURITY_CONSTITUTION).
```

The wire contract is the single seam. Both surfaces send **intents** and render **projections** from one **event stream**; neither surface holds authoritative state. This is what lets a user act in either surface and see the effect in both instantly.

## 3. The wire contract (live, to be re-anchored)

The frontend already defines: 11 intents, 7 event kinds, 30 custom names, 26 projections (`wire.ts`). Its Rust source of truth (`hide-core/src/api.rs`) is packed, so the contract is currently **unanchored** - a reintegration must restore `hide-core` as the schema authority (or generate the TS from it) so the contract cannot silently drift. The contract already covers the two-surface primitives HIDE needs: `submit_turn`, `fork_session`, `scrub_to_event`, `compact_context`, `inline_edit`, context manifest, fleet, tool, diff projections.

## 4. Cross-surface transitions (Chat → IDE)

Each transition is instant and preserves focus, because both surfaces read the same store. [parity target `ide.two_surface_bridge`]

| Gesture in Chat | Effect in IDE | Backing state |
|---|---|---|
| Click a file reference | Open file at line in editor | repo snapshot id |
| Click a diff chip | Open native side-by-side diff | patch transaction |
| Click a tool-output line | Reveal in terminal/output panel | tool/effect ledger |
| Click a terminal command | Reveal + re-run affordance | tool ledger |
| Click an agent branch | Open that agent's lineage board | agent tree |
| Click a context item | Reveal its source (file/memory/test) | context manifest |
| Click a checkpoint | Open the timeline at that capsule | checkpoint/capsule |

## 5. Cross-surface transitions (IDE → Chat)

| Gesture in IDE | Effect in Chat / agent | Backing |
|---|---|---|
| Send selection | New turn with `@selection` attached | selection injected via bridge |
| Send symbol / diagnostics | Turn scoped to that symbol/problem | LSP diagnostics |
| Send terminal failure | Turn with the failing command + output | tool ledger |
| "Ask about this diff" | Turn scoped to the patch | patch transaction |
| "Fix this" on a problem | Kernel loop targeting the diagnostic | verification state |
| "Spawn agents on this issue" | Fleet fan-out from the current warm state | state capsule fork |
| Steer active agent | `redirect_run` intent | agent tree |

The IDE↔session bridge is a loopback, token-authed local server (matching Claude Code's `ide` MCP server: 0600 lock file, per-activation token) that injects active selection + open-file path + LSP diagnostics into the session, with a Read-deny path for sensitive files (`.env`). [DOCUMENTED parity]

## 6. Surface division of labor

| Concern | HIDE Chat | HIDE IDE |
|---|---|---|
| Primary input | one prompt bar; `/` palette; `@`, `!` | editor + palette + agent panel |
| Agent loop visibility | streamed transcript, collapsed tools, todos | agent panel + status bar + problems |
| Diff review | diff summary chips | native side-by-side accept/reject hunks |
| Terminal | integrated PTY panel | integrated PTY panel (shared) |
| Plan | plan card + graded approval | editable plan document + inline comments |
| Context | Context Stack (light-well) | Context Stack panel + provenance on hover |
| Fleet | fleet board cards | fleet/worktree board |
| Permissions | inline gate + modal | inline gate + modal |

Neither surface is a subset of the other for *state*; they differ only in *rendering density and input modality*. A power user lives in Chat; a review-heavy user lives in the IDE; both see identical truth.

## 7. Honest current status (from the archaeology)

- **Real and shared today:** the FE store, wire contract, both chambers, the no-metering doctrine, diff review gestures, palette, terminal UI shell.
- **Broken today:** the backend seam. `/v1/hide/*` is served by the packed `hide-serve`; the live `hawking-serve` serves a disjoint surface. In dev everything is mock-fed.
- **The reconnection (Phase 0/1 of the ladder):** restore `hide-core` (wire schema authority) + `hide-serve` (the `/v1/hide/*` boundary) or implement those routes on `hawking-serve`, wire the session core, and replace the 256-token single-shot turn with the flat kernel loop. This single move makes both surfaces real at once, because they already share one store.

## 8. What makes this structurally better than Claude Code's two surfaces

1. **One warm state, not one transcript.** Claude Code shares a JSONL history; a surface switch re-establishes context from text. HIDE shares a resident capsule; a switch is a pointer to the same warm KV/recurrent state. [gated on state-capsule exposure build items]
2. **Zero-round-trip bridge.** The IDE↔session bridge is local; diffs, selection reads, and diagnostics stream with no network. [structural]
3. **Fork a surface, not just a session.** From the IDE a user can fork the live warm state into N agents (best-of-N) that each render as fleet cards in both surfaces, at near-zero marginal cost. [gated on fork exposure]
4. **No cross-interface fragmentation.** Claude Code's web/desktop/CLI keep *separate* histories [verifier-noted]; HIDE has one local session core, so all surfaces are genuinely unified.
