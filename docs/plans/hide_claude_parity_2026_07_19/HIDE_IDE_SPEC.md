# HIDE IDE Spec

Run date: 2026-07-19
Grounding: `HIDE_TWO_SURFACE_ARCHITECTURE.md` (anchor), `HIDE_LIVE_ARCHAEOLOGY.md` §3.4/§3.5/§5 (code-verified), `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json` (entry `ide.two_surface_bridge`), `HIDE_STATE_CAPSULE_ABI.md` (state exposure gates), `HIDE_2026_COMPETITOR_MATRIX.json` (ACP lever). Frontier evidence: `hawking/docs/plans/hawking_ide_frontier_2026_07_19.md`.
Status: specification for the IDE surface; every load-bearing claim cites live code (file:line) or a sibling doc, and every Hawking mechanism carries an honest readiness label (real-and-wired / real-but-unwired / partial / stub / missing).

## 1. Scope: the IDE is the second view of one session, not a second product

The HIDE IDE (a VS Code/Cursor-class workbench) and HIDE Chat (a terminal-style agent session) are two renderings of one durable session over one Zustand store and one typed wire contract (`app/src/wire.ts`). The invariant, the shared-core layering, and the cross-surface transition tables live in `HIDE_TWO_SURFACE_ARCHITECTURE.md` and are not repeated here. This document specifies:

1. the workbench surface inventory against modern-IDE expectations (Bible §27), with live readiness (§2, §3);
2. the two hard build items unique to this surface: a real terminal PTY and a native inline-completion path (§4);
3. the build-vs-fork product decision: standalone vs VS Code fork vs extension vs Zed/ACP vs hybrid (§5);
4. the IDE-to-session loopback bridge (parity `ide.two_surface_bridge`, P0) (§6);
5. the strict parity/supremacy split, each supremacy claim gated on its build item (§7).

The load-bearing structural fact carries over from the archaeology: the FE workbench is substantially built and polished but backend-deferred, and the `/v1/hide/*` boundary it targets is packed (`HIDE_LIVE_ARCHAEOLOGY.md` §0, §3.4). So this surface is a reconnection plus two net-new mechanisms, not a greenfield build.

## 2. Live IDE surface truth (what is already built)

Verified in the active tree at `4fbca8bc`. The IDE chamber is `app/src/surfaces/ide/*` plus shared shell files. Monaco and xterm are already vendored and bundled air-gapped (no CDN fetch, `Editor.tsx:25-26`).

| FE surface | File | What is real | Readiness |
|---|---|---|---|
| Editor (Monaco) | `surfaces/ide/Editor.tsx:14-26` | Monaco editor on the open file; `Cmd/Ctrl+S` command (`:211`); HIDE grayscale theme, Geist Mono, light-only accent (`monacoTheme.ts`) | FE-real, backend-deferred (`OpenFile` -> `ProjectionPatch{editor}`) |
| Native diff editor + per-hunk review | `surfaces/ide/Editor.tsx:50-53`, `surfaces/ide/HunkReview.tsx:2-5` | Monaco `DiffEditor` inline-by-default with side-by-side toggle; per-hunk gesture (`j/k` navigate, `a`/`Cmd+Enter` accept, `r`/`Cmd+Backspace` reject) dispatching `AcceptDiff`/`RejectDiff`; color-plus-marker (`+`/`-`) so color is never the sole signal | FE-real, backend-deferred (apply/revert logic lives in packed `hide-tools`) |
| Explorer + folded search | `surfaces/ide/Explorer.tsx:2,31-38` | ARIA file tree (roving-tabindex), Xcode-style filter that switches to `code_index.search` hits; real `fs` connector tree with `MOCK_TREE` dev fallback | partial (tree wired to connector; search targets packed index) |
| Code actions | `surfaces/ide/CodeActions.tsx` | quick-action affordances on the editor | FE-real, backend-deferred |
| Integrated terminal | `surfaces/ide/Terminal.tsx:5,150-155` | xterm mounted, HIDE mono chrome, local line editor; a line dispatches `RunCommand{argv,cwd}` and echoes `queued ›`; agent shell output mirrored from `tool_progress` | **no PTY** (see §4.1); comment marks `(future) PTY WebSocket` |
| Command palette | `app/src/ui.tsx` (`CommandPalette`) | incremental palette | partial (MCP/skill/prompt unification needs the packed backend; parity `palette.unified`) |
| Settings | `surfaces/Settings.tsx` | settings surface | FE-real |
| Context Stack | `surfaces/ContextStack.tsx` | truthful context panel (SKILLS is a hardcoded const today) | FE-real, mock-fed |
| State timeline (scrub/fork) | `shell/StateTimeline.tsx` | `scrub_to_event` / `fork_session` intents | FE-real, backend "plan 2" |
| Status bar | `shell/StatusBar.tsx` | phase/model/transport live | partial (branch + problems hardcoded, `HIDE_LIVE_ARCHAEOLOGY.md` §3.4) |
| Agent panel | `shell/ChatPane.tsx`, `shell/FloatingChat.tsx` | the Chat chamber embedded in the workbench | FE-real, mock-fed |
| Fleet / worktree board | `HIDE_LIVE_ARCHAEOLOGY.md` §3.4 | optimistic fleet cards, Fork-and-Try-N seeds | FE-real, backend "plan 2" (real fork/scheduler packed in `hide-fleet`) |

## 3. Modern-IDE workbench inventory (Bible §27 expectations)

Each row is one workbench expectation. `Live status` uses the honest readiness labels; `Build item` is the smallest move to parity; `Supremacy gate` names the build item a Hawking-native win depends on (do not claim the win before its gate lands).

| Expectation | Parity requirement | Live status | Build item | Supremacy gate |
|---|---|---|---|---|
| Explorer | File tree with git badges, touched-by-run marks | partial (`Explorer.tsx`; tree from `fs` connector, badges not wired) | wire git status + `touchedByRun` from the tool/effect ledger | none (parity surface) |
| Tabs / editor | Multi-tab Monaco with persistence | FE-real, backend-deferred (`Editor.tsx`) | tab model + `ProjectionPatch{editor}` stream | none |
| Symbols / outline | Document + workspace symbol view | missing in FE | consume LSP `documentSymbol` (frontier §5.10) OR surface packed `hawking-index` tree-sitter defs | index-warm symbol lookup gated on `hawking-index` reintegration (real-but-unwired, `HIDE_LIVE_ARCHAEOLOGY.md` §3.5) |
| Search | Global content + symbol search | partial (`Explorer.tsx` folds search to `code_index.search`) | wire `code_index.search` to reintegrated `hawking-index` (FTS5 + graph + RRF) | hybrid RRF retrieval gated on `hawking-index` |
| Go-to-def / refs | Jump to definition and references | missing in FE (`Editor.tsx` has no def/ref wiring) | consume LSP `definition`/`references`; OR `hawking-index` SCIP ids + ref graph | index-native cross-repo refs gated on `hawking-index` |
| Diagnostics / problems | Live problems panel from language servers | partial (`StatusBar` problems hardcoded) | LSP `publishDiagnostics` ingestion + a Problems panel (frontier §5.10) | "Fix this" one-click loop gated on kernel reintegration (`hide-kernel`, real-but-unwired) |
| Integrated terminal | Interactive shell with a real PTY | **missing PTY** (`Terminal.tsx` queues intents only) | **PTY WebSocket + sandboxed `shell.run`** (see §4.1) | egress-off, capsule-snapshot-before-run gated on `HIDE_SECURITY_CONSTITUTION` + state capsule |
| Source control | Git status, stage, diff, commit | partial (git shown hardcoded; logic packed in `hide-tools` worktree trio) | surface `hide-tools` git ops behind the SC panel | worktree fleet SC gated on `hide-fleet` reintegration |
| Native diff editor | Side-by-side diff with per-hunk accept/reject | FE-real (`HunkReview.tsx`, `Editor.tsx:50-53`) | wire `AcceptDiff`/`RejectDiff` to `hide-tools` tiered edit applier | best-of-N candidate diffs from state forks gated on state routes (§7) |
| Test UI | Run/inspect tests, pass/fail tree | missing in FE | a test-run panel over `hide-tools` `proc` + reintegrated `hawking-eval` (pass@1 + Wilson CI) | private rotating capability suite gated on `hawking-eval` (real-but-unwired) |
| Command palette | One incremental palette over built-ins, skills, MCP prompts | partial (`ui.tsx`) | unify command sources once the backend registry is wired (parity `palette.unified`) | none |
| Settings | Layered settings with fixed precedence | FE-real (`Settings.tsx`) | Claude-compatible settings precedence (parity `config.settings_precedence`, see `HIDE_CLAUDE_CODE_CONFIGURATION_COMPATIBILITY.md`) | none |
| Inline FIM completion | Fill-in-the-middle ghost text on type | **missing** (no native FIM path, frontier §6 line 899) | **native FIM lane** (see §4.2) | zero-marginal-cost inline completion gated on the FIM build + warm state |
| Next-edit prediction | Predict the next edit location + content | missing | next-edit model + edit-speculation transaction (frontier §5.7) | suffix/file-as-draft speedups gated on `hawking-speculate` verifier (real-but-unwired) |
| Agent panel | Steerable agent session inside the workbench | FE-real, mock-fed (`ChatPane.tsx`) | reconnect the flat kernel loop (frontier Phase 0/1) | zero-latency interrupt-and-fork gated on state routes (parity `loop.interrupt_and_keep`) |
| Context Stack | Truthful context panel with provenance on hover | FE-real, mock-fed (`ContextStack.tsx`) | reintegrate `hawking-context` compiler as the feed | value-density compiled context gated on `hawking-context` (real-but-unwired) |
| Worktree / fleet board | Parallel-agent board with worktree isolation | FE-real, backend "plan 2" | reintegrate `hide-fleet` (job DAG, leases, merge) | uncapped local parallelism gated on `hide-fleet` + state fork (parity `session.background_supervisor`) |

Reading of the table: the workbench body (explorer, editor, diff, palette, settings, terminal shell, timeline, fleet cards) is FE-real. What is missing is (a) language intelligence ingestion (symbols/def/refs/diagnostics: an LSP-consumption build item, cheap), (b) two net-new mechanisms (PTY, FIM), and (c) the backend reconnection that every other doc already scopes. No row requires inventing a Hawking primitive that does not already exist somewhere in the tree or the sealed pack.

## 4. The two hard build items unique to this surface

Everything else on this surface is either FE-real or a reconnection covered by `HIDE_TWO_SURFACE_ARCHITECTURE.md` §7 and the frontier ladder. These two are net-new.

### 4.1 Integrated terminal needs a real PTY (parity blocker)

Live truth: `Terminal.tsx` mounts xterm but has no PTY. A typed line calls `sendIntent(intent.runCommand(argv, null))` and, on ack, writes `queued › <cmd>` into the buffer (`Terminal.tsx:150-155`); real shell output only appears as mirrored `tool_progress` rows from the agent's shell tool (`:115-163`). The source comment names the gap: input is meant to travel over a `(future) PTY WebSocket` (`Terminal.tsx:5`). So today the terminal cannot run an interactive human command, cannot carry a REPL, and cannot show a live cursor.

Parity requirement (Bible §27, and the two-surface division of labor lists a shared PTY panel, `HIDE_TWO_SURFACE_ARCHITECTURE.md` §6): a bidirectional PTY so a human command streams stdout/stderr live with full terminal semantics (resize, control chars, colors), shared by both surfaces.

Build item:
1. a PTY host on the local backend (pty spawn, resize, kill) exposed as a WebSocket framed channel (`stdin` up, `stdout`/`stderr`/`exit` down);
2. route the human path through the same sandboxed executor as the agent shell tool (`hide-tools` `shell.run` with watchdog, real-but-unwired, `HIDE_LIVE_ARCHAEOLOGY.md` §3.5), so human and agent commands share one confinement policy;
3. the FE flips `Terminal.tsx` from `runLine -> intent` to `stdin -> PTY socket`, keeping the `tool_progress` mirror for agent-run commands.

Security precondition (do not ship the PTY before this): the executor must be OS-sandboxed and the backend must bind authenticated loopback, not `0.0.0.0` (serve binds `0.0.0.0:8080` with no auth today, `HIDE_LIVE_ARCHAEOLOGY.md` §3.2 G10). See `HIDE_SECURITY_CONSTITUTION.md`.

Supremacy (gated, do not assert before the sandbox lands): a purely local runtime needs no egress, so the terminal can run with network default-fully-off as a hardware fact, and a workspace capsule snapshot taken before a risky command makes rollback near-zero-cost (parity `security.sandbox`; capsule mechanism in `HIDE_STATE_CAPSULE_ABI.md`).

### 4.2 Inline completion (FIM + next-edit) has no native path

Live truth: there is no verified native fill-in-the-middle path (frontier §6, capability row "Inline coding: no verified native FIM path"). `Editor.tsx` wires no ghost-text/inline-completion provider; the Bible plan lists `ghostText` and `inlineEditWidget` as intended `editorStore` fields (`docs/hide-bible/HIDE_PLAN.md:592`), but they are plan, not built. The serve runtime exposes only OpenAI-shaped chat/completions; FIM prompt assembly, low-latency ranking, and next-edit are absent.

Parity requirement: (a) inline FIM ghost text on type with syntax/type-aware acceptance and low-latency ranking; (b) next-edit prediction (predict the next edit location and content after an accepted change).

Build item:
1. a FIM serving lane: a fill-in-the-middle prompt ABI (prefix/suffix/cursor) on a code model, on a dedicated low-latency slot separate from the agent turn (so completion never contends with a running agent);
2. Monaco `registerInlineCompletionsProvider` in `Editor.tsx` feeding from that lane, with a debounce + accept/reject telemetry (Keep Rate, frontier §5.12);
3. next-edit as an edit-speculation transaction (frontier §5.7): the model proposes the next hunk, which is applied only through the same reviewable `AcceptDiff` gate, never silently.

Supremacy (gated on the FIM lane plus warm state): a resident model on a warm prefix serves FIM at zero marginal metered cost and no network round-trip, and can run best-of-N next-edit candidates from a cheap state fork; both gains are gated on the FIM build item plus state exposure (`HIDE_STATE_CAPSULE_ABI.md`) and on the lossless verifier if suffix/file-as-draft speculation is used (`hawking-speculate/verifier.rs`, real-but-unwired). Until the FIM lane exists this is a research bet, not a shipping claim.

## 5. Build-vs-fork: how to ship the workbench (Bible §27)

The question is which shell hosts the surface specified above. Five options, scored on speed-to-quality, control, binary size, licensing, and ability to express HIDE state/fleet UX. The competitor field has converged so that raw feature parity is table stakes; the differentiator ANY of these can and cannot express is the deciding axis (`HIDE_2026_COMPETITOR_MATRIX.json` field_synthesis).

| Option | Speed-to-quality | Control | Binary size | Licensing | Can express HIDE state/fleet UX? |
|---|---|---|---|---|---|
| A. Standalone (current: Tauri v2 + React + Monaco/xterm) | high: ~80% built, FE-real today (`app/`) | full | small (Tauri, system webview; Monaco already bundled `Editor.tsx:25-26`) | clean: Monaco MIT, xterm MIT, Void-derived diff/layout Apache-2.0 PORT-OK (`HIDE_PLAN.md:931`) | yes: owns the whole shell (StateTimeline, fleet board, Context Stack, no-metering doctrine, Ando/Geist look) |
| B. VS Code fork | low: fork, re-skin, and carry an upstream-merge tax forever | high but shared with upstream | large (full Electron + VS Code) | Code-OSS MIT but marketplace/branding restricted; heavy re-skin to escape VS Code identity | partial: possible, but the VS Code visual identity fights the doctrine Self-check (`HIDE_PLAN.md:156`, "could be mistaken for VS Code") |
| C. VS Code extension + native backend | high for parity, low for identity | low: bounded by the extension API and host chrome | small extension, but user runs VS Code (large) | fine, but you do not own the workbench | no: cannot express the observation-first shell, the fleet board, or the no-metering doctrine inside someone else's chrome |
| D. Zed / ACP integration | medium: the agent appears in an existing IDE with no chrome to build | low: Zed owns thread UI/history; agent owns runtime/model/tools | none (rides the host) | ACP is a neutral protocol; no auth/subscription handshake, no egress (`HIDE_2026_COMPETITOR_MATRIX.json` Zed+ACP `local_first_opening`) | no: you render as a listed agent, not as HIDE; the fleet/timeline/Context Stack UX is not yours to draw |
| E. Hybrid: A as the flagship, plus C/D reach | high | full for the flagship, bounded for reach | small flagship + thin adapters | clean | yes for the flagship; parity presence in others |

Recommendation: **E (hybrid), anchored on A.** Keep the standalone Tauri workbench as the flagship product because it is the only option that can express HIDE's differentiators (the observation-first shell, StateTimeline scrub/fork, the fleet board, the truthful Context Stack, and the enforced no-metering doctrine), it is already ~80% built and FE-real, its licensing is clean (Monaco/xterm MIT, Void-derived diff/layout Apache-2.0 PORT-OK, Zed studied but never copied because AGPL would force the FE open, `HIDE_PLAN.md:920,931,939`), and its binary is small because Monaco is already vendored and bundled air-gapped. Then, for reach, ship the two thin adapters that meet users who will not leave their editor:

- an ACP server (option D as an adapter, not a product) so a local Hawking agent appears natively in Zed and JetBrains alongside Claude Code and Codex, with no auth handshake and no egress, where warm-state resume beats cloud agents' re-prefill. This is flagged as the strongest interop lever in the competitor matrix, and Claude Code itself is only a listed ACP agent there, so this is direct parity plus a local-first wedge (`HIDE_2026_COMPETITOR_MATRIX.json` Zed+ACP; `hawking_ide_frontier` §5.10).
- the IDE-to-session loopback bridge (§6), which is exactly Claude Code's VS Code/JetBrains extension shape (parity `ide.two_surface_bridge`), so a HIDE session can attach to an existing VS Code/Cursor without HIDE forking or replacing it (option C as an adapter).

Reject B (VS Code fork): the binary and upstream-merge tax are large, and the VS Code visual identity structurally fights the doctrine, which is a build-time CI gate, not a preference (`HIDE_PLAN.md:156`). Reject C-only and D-only as the product: neither can render HIDE's state/fleet UX, which is the entire differentiator once feature parity is table stakes.

## 6. The IDE-to-session loopback bridge (parity `ide.two_surface_bridge`, P0)

This is the mechanism that makes an external editor (option C reach, and the flagship's own two chambers) a first-class client of the one session. The cross-surface transition tables (Chat->IDE and IDE->Chat gestures, with backing state) are specified in `HIDE_TWO_SURFACE_ARCHITECTURE.md` §4-5 and are not repeated here; this section specifies the transport and its trust properties.

Parity target (DOCUMENTED, from `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json` `ide.two_surface_bridge`): a loopback, token-authed local server matching Claude Code's `ide` MCP server: a `0600` lock file, a per-activation token, injecting active selection, open-file path, and LSP diagnostics into the session, with a Read-deny path for sensitive files (for example `.env`). CLI and IDE share ONE conversation history plus checkpoints.

Bridge contract:

```text
Bridge (loopback only, 127.0.0.1)
  auth:        per-activation token minted at IDE activation; lock file mode 0600
  bind:        loopback only; never 0.0.0.0 (serve binds 0.0.0.0 today: fix first, HIDE_LIVE_ARCHAEOLOGY §3.2 G10)
  inject up:   active_selection (@selection), open_file_path (@file#L-range), LSP diagnostics
  read-deny:   sensitive-file glob (.env, secrets) refused at the bridge, not the model
  shared:      one session id, one event stream, one checkpoint ring (both surfaces read the same store)
```

Honest readiness: the FE already has both chambers over one store and one wire (`HIDE_LIVE_ARCHAEOLOGY.md` §3.4, parity `hide_status: partial`), so the FE half is real. The backend bridge is packed (the `/v1/hide/*` boundary in `hide-serve`, and the `ide` server itself), so the bridge is **real-but-unwired** at the seam and **missing** as a running local server. Two preconditions gate it: (1) loopback + auth on the backend (serve binds `0.0.0.0` with no auth today), and (2) the wire schema re-anchored to `hide-core` so the contract cannot drift while an external editor depends on it (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §3).

Supremacy (structural, gated on state exposure): because the engine is local, selection/diff/diagnostics injection is a zero-round-trip local call, and, once the state routes land, both surfaces share ONE warm resident context capsule rather than re-injecting selection text per prompt (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §8; capsule and its three exposure build items in `HIDE_STATE_CAPSULE_ABI.md` §8). Zero-round-trip is structural today; shared-warm-context is gated on the capsule exposure work.

## 7. Parity vs supremacy on the IDE surface

Parity is reproducing the workbench behaviors in §3. Supremacy is what a local resident runtime does better; each claim is inert until its gate lands, and must not be presented as shipping before then.

| Supremacy claim (IDE surface) | Structural basis | Gating build item | Ship-before-gate? |
|---|---|---|---|
| Zero-latency interrupt, then fork both branches | no network to cancel; fork is a memcpy | state routes `/v1/hawking/state/{save,load,fork}` + session->slot affinity (`HIDE_STATE_CAPSULE_ABI.md` §8) | no |
| Both surfaces share one warm context (not re-injected text) | one resident capsule under one session | capsule exposure (state routes, affinity, GPU->CPU readback G-CAP-1) | no |
| Fork the workbench into N fleet agents at near-zero model cost | RWKV fork is a pointer copy of a resident capsule (`rwkv7.rs:376-378`, tested) | state routes + `hide-fleet` reintegration; RWKV lane only today (transformer capsule missing) | no |
| Best-of-N candidate diffs / plans executed in isolated forks; user picks the one that already passed | cheap forks + deterministic verify | state routes + `hide-kernel` oracle-gated verify + `hide-tools` sandbox | no |
| Inline FIM + next-edit at zero marginal cost, no round-trip | resident model on a warm prefix | FIM lane (§4.2) + warm state; lossless verifier for draft speculation | no |
| Terminal runs air-gapped with capsule snapshot rollback | local inference needs no egress | OS sandbox + capsule-before-run (`HIDE_SECURITY_CONSTITUTION.md`, §4.1) | no |
| Uncapped local parallelism on the fleet board (no metered quota) | no per-token meter; bounded by hardware | `hide-fleet` reintegration + state fork | no |
| Diff/selection/diagnostics inject with zero network round-trip | engine and bridge are local | loopback bridge (§6) | yes (structural, once bridge runs) |

Every "no" in the last column is a real-but-unwired or missing primitive per the archaeology lever ledger (`HIDE_LIVE_ARCHAEOLOGY.md` §5); presenting any of them as a current capability would violate the honest-readiness discipline. The one "yes" still requires the bridge to actually run on loopback.

## 8. Honest readiness summary and feed-forward

- **FE-real today:** editor (Monaco), native per-hunk diff review, explorer + folded search, code actions, palette, settings, Context Stack, state timeline, agent panel, fleet cards. The workbench body exists and is doctrine-clean.
- **Missing on this surface, cheap:** language-intelligence ingestion (symbols, go-to-def/refs, diagnostics/problems) via LSP consumption; a Problems panel; a test-run panel.
- **Net-new mechanisms (this doc's two hard items):** a real terminal PTY (§4.1, parity blocker, security-gated) and a native FIM + next-edit lane (§4.2, no path today).
- **Reconnection (covered elsewhere):** the `/v1/hide/*` boundary, the flat kernel loop feeding the agent panel, the `hawking-context` feed for the Context Stack, `hide-tools` for diff-apply and git, `hawking-index` for symbols/search, `hawking-eval` for the test UI. See `HIDE_TWO_SURFACE_ARCHITECTURE.md` §7 and the frontier ladder (`hawking_ide_frontier` Phase 0/1).
- **Product decision:** ship the standalone Tauri workbench as the flagship, plus an ACP server and the loopback bridge as thin reach adapters (§5, option E). Do not fork VS Code.
- **Supremacy discipline:** every IDE-surface supremacy claim except zero-round-trip injection is gated on state exposure, kernel/fleet reintegration, or the FIM build; none may ship as a capability before its gate (§7).
