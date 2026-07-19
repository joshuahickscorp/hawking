# HIDE Claude Code Parity and Supremacy - Final Research Report

Run date: 2026-07-19 · Research cutoff: 2026-07-19 · Repo truth pinned at `4fbca8bc` (main)
Research branch: `research/hide-claude-parity-2026-07-19` (isolated worktree; no merge to main)
Authoritative spec: `HIDE_CLAUDE_CODE_PARITY_SUPREMACY_BIBLE.md`

## 1. Executive verdict

HIDE can be built to feel immediately familiar to a Claude Code user and at least as polished, then made structurally better through local Hawking-native mechanisms. The reason is concrete and code-verified, not aspirational: **the harness that produces the Claude Code workflow developers love already exists in this repository as tested code; it is disconnected, not missing.** Parity is dominantly a reconnection problem, and every Hawking-native advantage exists today as a real primitive (mostly real-but-unwired), so the supremacy path is a sequence of exposure and integration steps, not invention.

The single most important finding: **the production vertical slice is currently broken, and the "pass state, not text" moat is real but dead-ended below the product boundary.** The React/Tauri frontend and the Hawking runtime are both genuinely built; the ~50k-line agent backend between them (context compiler, code index, planner/verifier kernel, typed tools, fleet) was sealed to a pack on 2026-07-18 and is recoverable only from git `5a99d0e2`. RWKV recurrent state is serializable, byte-exact-forkable, and unit-tested, yet no HTTP route, CLI subcommand, or live turn ever calls it.

## 2. What was verified (reconciliation highlights)

A read-only 6-agent audit reconciled every historical claim against current code with file:line evidence; a 5-agent adversarial pass independently re-verified the load-bearing claims. All were confirmed:

- **Two distinct "packs":** `hawking-packs` was absorbed into `hawking-seed-c` and retired (do not recreate); the `hawking-hide-desktop` backend (13 crates) is a separate, still-sealed pack. Correction to the handoff: the offline archive is gone from disk; git `5a99d0e2` is the only lifeline.
- **The serve boundary:** 9 routes, none for state; no `/v1/hide/*`; `max_seq_len` hardcoded 4096; CLI binds `0.0.0.0` with no auth; the `/context` `tq_multiplier` is an unset-env estimate (G1, G3, G9, G10 all confirmed).
- **The state moat:** `RwkvState` fork/serialize byte-exact and tested, zero production callers; GPU->CPU recurrent readback missing (exact live capture blocked); no transformer capsule.
- **The facade (S1-S5):** the last-built live turn was a single-shot raw-prompt, empty-history, 256-token generate; the reserve-then-fill compiler ran only behind a connector and its output was discarded; `compact_context` was logged not performed; state routes never existed server-side. All confirmed at `5a99d0e2`.
- **The frontend** is substantially built and doctrine-compliant (React 19 + Zustand + Monaco + xterm + Tauri v2, Ando/Geist-Mono, no budget meter), but mock-fed with several differentiators backend-deferred to "plan 2" and the wire contract unanchored (its Rust source is packed).

## 3. The parity picture

The clean-room study (11 facet scouts, each independently verified against July-2026 primary sources) produced the loved-experience genome, 14 workflow traces, a 32-entry behavioral parity spec, a verbatim-migration configuration-compatibility layer, a love/pain/captivity map, and a 2026 competitor matrix.

- The love concentrates in **steerable autonomy over a genuinely-understood repo** (interrupt-and-keep, plan gate, reversibility, legibility), which is mostly harness (reconstructable) plus model judgment (gated on a capable local coder).
- Captivity is **almost entirely habit and config**, so verbatim reading of `CLAUDE.md`, agents, skills, and MCP is the wedge that neutralizes switching cost.
- The largest tolerated pains (metering, egress, silent regression, context reprocessing) map one-to-one onto HIDE structural or economic advantages.

## 4. The supremacy path (honest and gated)

Six advantages, each gated on its build item (`HIDE_SUPREMACY_THESIS.md`):

1. No meter (economic; ships at the doctrine level).
2. No egress / air-gap (structural for inference; whole-run air-gap gated on egress enforcement).
3. No silent regression (structural; the March-April 2026 Claude Code regression is the counterexample).
4. Warm-state forks and best-of-N (structural, the signature advantage; real-but-unwired on the RWKV lane; gated on GPU readback + state routes + affinity; transformer lane blocked on a local coder).
5. Resident shared-state daemon (shared session ships at the spine; warm-slot no-reprefill gated on affinity).
6. ACP-native local hosting (a low-cost interop wedge).

Nothing is claimed as shipped. Every "fastest/most-capable/densest" claim is explicitly unearned until the reintegrated `hawking-eval` harness produces a receipt on the real app path.

## 5. The build path

The priority formula surfaces one dominant move: **reconnect the packed spine** (`hide-core` + `hide-serve` + `hawking-context` + `hawking-index` + `hide-kernel` + `hide-tools` + `hawking-eval`) into a flat execution-grounded loop, with the compiled ContextPack actually fed to generation and the 256-token single-shot turn replaced. This is the minimum lovable vertical, it makes both surfaces real at once (they already share one store), and it unblocks the state moat, the two-surface daemon, and every ecosystem item. The state moat then follows in Phase 2 (GPU readback + state routes + affinity), and a capability-dense local coder (Qwen3-Coder-Next-class) is the Phase 4 capability lever, isolated from the ship path.

## 6. Adversarial verification outcome

Five independent verifiers attacked the package (`wf_74fdbb5b-b03`, 0 errors):

- **Archaeology re-verified against live code:** all 7 load-bearing claims CONFIRMED. One internal contradiction found and fixed (a "seven crates are workspace members" phrasing vs the table's non-member note; six are members).
- **Supremacy falsification:** all 6 handoff falsification targets held (the package either correctly FALSIFIES the naive claim - "recurrent state replaces exact retrieval", "weight compression expands context", "KV quantization is quality-neutral" - or correctly frames "cheap forks make best-of-N worthwhile" and "more agents reduce wall clock" as OPEN, must-be-measured questions with harsh kill criteria). Two sharpenings applied (2.2 air-gap scope; 2.5 affinity gate).
- **Parity correctness:** attempts to falsify the DOCUMENTED claims (@import depth 4, WebSocket MCP transport, auto-mode measured numbers, CVE-2026-33068, checkpoint 100/30-day, the `#` shortcut handling) all FAILED to falsify: the parity claims are accurate. Two label fixes applied (hooks.lifecycle -> absent; a component-name traceability fix).
- **Hawking-fit:** could not falsify that any "reintegrate X" target exists and does what the doc claims; all 9 reintegration targets verified in the packed source; readiness labels consistent. Minor citation nits noted (below).
- **Consistency/completeness (MATERIAL_ISSUES, now resolved):** caught a real house-rule violation - 123 em/en dashes across 10 files (my earlier check used a broken BSD-grep pattern). All stripped; 0 remain. The final report existed only as a receipt reference; it is now authored (this document). README manifest updated to sealed.

Residual minor nits accepted (documented, not blocking): a debug_assert vs hard-assert nuance in the context-OS compiler citation; a "~44 GenerateRequest sites" figure quoted from a stale in-code comment (true count ~38-41); a `json_mode:false`-forcing citation that should point at `http.rs`/`scheduler.rs` rather than `driver.rs`; slightly divergent line-range citations for the same RWKV features across sibling docs. None change a conclusion; all are flagged in the receipt's known-limits.

## 7. Owner decisions surfaced (not blocking Phase 0)

1. First supported Apple hardware tiers (laptop / Pro-Max / Ultra).
2. Qwen3-Coder-Next as the first hybrid coder target, or evaluate another open coder alongside it.
3. Restore `hide-serve` (rehydrate the pack) vs implement `/v1/hide/*` fresh on `hawking-serve`.
4. How much of the sealed backend to restore vs rewrite around the smaller vertical slice.
5. Latency contracts for Instant / Interactive / Thorough / Background.
6. Which private real-work corpus is retained for evaluation.
7. Canonical product name in code and docs (Hawking IDE / HIDE / both).

## 8. Package contents

29 research artifacts plus this report and the receipt, under `docs/plans/hide_claude_parity_2026_07_19/`. All `.json`/`.jsonl` parse; zero em/en dashes; every load-bearing claim evidence-tagged; the whole pass read-only against live sources with writes only on the research branch. Provenance and metrics are in `HIDE_RESEARCH_RECEIPT.json`.

## 9. Required final statement

> HIDE now has a clean-room Claude Code parity specification, a verified map of its live implementation, and an evidence-backed supremacy path. The package identifies how to reproduce the workflow developers love, remove its cost and cloud constraints, and exceed it through Hawking-native state, context, fleets, tools, verification and local capability density.
