# HIDE Prioritized Build Ladder

Run date: 2026-07-19 · Grounding: `HIDE_PARITY_GAP_MATRIX.json`, `HIDE_SUPREMACY_THESIS.md` (dependency spine), `HIDE_LIVE_ARCHAEOLOGY.md`.
This is an implementation ladder, not authorization to implement in this research pass.

## 1. Priority formula (Bible §95)

```
        verified-capability-gain × Claude-parity-value × Hawking-uniqueness × user-love
P  =  ------------------------------------------------------------------------------------
        engineering-cost × runtime-dependency × security-risk × UX-risk
```

Applied qualitatively (H/M/L) per item below. The formula surfaces one dominant conclusion: **reconnecting the packed spine has the highest P of anything**, because its engineering cost is "lift + wire existing tested code" (not invention), its parity value is the whole loved workflow, and it unblocks nearly every other item.

## 2. Feature classification key (Bible §94)

`real+wired` · `real+unwired` (packed/below-boundary) · `partial` · `stub` · `missing` · `blocked-on-model` · `blocked-on-security` · `blocked-on-ux` · `research-only`.

Most parity items are `real+unwired`: the ladder is dominated by *reconnection*, not greenfield.

## 3. The ladder

### Phase 0 - Restore truth, security floor, and measurement

Goal: one honest, secure, reproducible baseline. Nothing lovable ships before this.

| Item | Class | P | Notes |
|---|---|---|---|
| Re-anchor the wire contract: restore `hide-core` as schema authority (or generate TS from it) | real+unwired | H | stops the FE contract drifting against a packed source (archaeology §3.4) |
| Reconnect the product boundary: restore `hide-serve` `/v1/hide/*` OR implement those routes on `hawking-serve` | real+unwired | H | the FE targets a missing backend today; this makes the shell real |
| Trust-before-config gate + loopback+auth serve (no `0.0.0.0`, resolve no mode before trust) | missing / blocked-on-security | H | CVE-2026-33068 class; serve binds `0.0.0.0` no auth today (G10) |
| Crash-safe event log + single-writer lock + end-to-end trace IDs | real+unwired | M | from packed `hide-backend` event bus |
| Reintegrate `hawking-eval` (pass@1 + Wilson CI) | real+unwired | H | cheap; unblocks every capability claim; only perf (`hawking-bench`) is live today |
| Build a small private golden task suite from real HIDE/Rust/TS tasks | missing | M | the release gate, not public benchmarks (contaminated) |

**Exit gate:** a stranger can open the app, trust a repo, submit a task, see a persisted response, restart, and replay it; failures are explicit; one critical-path trace per task; no unsandboxed exec path; a baseline number exists with no "infinite/fastest/multiplied-context" claim.

### Phase 1 - The minimum lovable vertical (a real local coding agent)

Goal: reconnect the spine into a flat loop. This is the daily-usable slice (`HIDE_MINIMUM_LOVABLE_VERTICAL.md`).

| Item | Class | P | Notes |
|---|---|---|---|
| Replace the 256-token single-shot turn with the flat kernel loop (RuntimePlanner, not StubPlanner) | real+unwired | H | the central defect fix (facade S1/S2) |
| Reintegrate `hawking-context` and FEED the compiled ContextPack to generation | real+unwired | H | fixes the discarded-context facade (S3) |
| Reintegrate `hawking-index` + wire kernel Grounding | real+unwired | H | living code index |
| Reintegrate `hide-tools` (verifying edit applier, sandboxed shell, search, git) + wire dispatch | real+unwired | H | transactional reviewable edits |
| Deterministic verify oracles (build/typecheck/test/lint as ProcessOracle) | real+unwired | H | "done" = proof |
| Interrupt-and-keep + soft steer on the live turn | ui_only->wired | H | top love gene; FE intents exist |
| Plan mode with executor-level write block + graded approval | ui_only->wired | H | second love gene |
| Persist user/assistant/tool/patch/verify/checkpoint events | real+unwired | M | durable transcript |
| Read CLAUDE.md tree + settings + agents verbatim (migration reader) | missing | M | the switching-cost wedge |
| Use the real tokenizer; expose honest context limits (drop the unset `tq_multiplier` estimate) | partial | M | fixes G9/G11 |

**Exit gate:** the agent solves a private multi-file Rust/TS task through the real app; edits are transactional and reviewable; interrupt/plan/rewind/resume/replay work; context receipts show exactly what reached the model; an existing Claude Code repo's CLAUDE.md loads unchanged.

### Phase 2 - Make the repeated loop fast (and expose the state moat)

Goal: eliminate avoidable critical-path work and unlock the signature advantage.

| Item | Class | P | Notes |
|---|---|---|---|
| PromptABI + ToolRegistryABI as a monitored SLO (deterministic serialization, stable tool order, append-only, cache-key telemetry) | missing | H | prefix-cache is a product ABI |
| Fix direct-admit / batch-one prefix reuse; token-prefix radix cache | partial | H | G4; the common local-agent path cold-prefills today |
| Session->slot affinity | missing | H | prerequisite for warm reuse and state routes (G2) |
| GPU->CPU recurrent readback (exact live capture) | missing / research | H | **the first hard gate on the state moat**; measure cost |
| `/v1/hawking/state/{save,load,fork}` HTTP routes | missing | H | exposes the moat (G1); the shell-side client seam already exists |
| Wire warm-state `.sstate` persistence | real+unwired | M | instant resume, no re-prefill |
| Port stop-strings / JSON-mask / streaming into the batched serve path | partial | M | T7/T8 |
| Reintegrate `hawking-orch` tool-spec-decode into the batched path (first-try-valid tool calls) | real+unwired | M | lossless; a Hawking-native win |
| Keep MCP/LSP/search/build/test warm | missing | M | round-trip elimination |

**Exit gate:** warm repeated turns show measured TTFT + edit-to-green improvement with no quality regression; state fork/restore is parity-correct and resource-bounded (RWKV lane); cache invalidation is explainable; tool latency is decomposed.

### Phase 3 - Quality scales with effort; the two surfaces and the demo

| Item | Class | P | Notes |
|---|---|---|---|
| Model selection vs effort as distinct axes; transparent phase-aware routing | partial | M | model topology |
| Warm-state fork best-of-N with execution tie-break (RWKV lane) | real+unwired | H | the signature demo (`HIDE_SIGNATURE_DEMO.md`) |
| Full IDE surface parity: PTY terminal, native diff review wired, symbols/problems | partial/ui_only | H | FE terminal has no PTY today |
| Subagents (file-defined) + isolated context + summary-only | real+unwired | M | |
| Skills (progressive disclosure) + hooks (trust-gated) + MCP host/server | missing/partial | M | ecosystem import |
| Local Agent SDK + headless `-p` with structured output | missing | M | automation/CI |
| Outcome-governed memory + revalidation | real+unwired | M | |

**Exit gate:** best-of-N produces a verified winning diff a user accepts, with no cloud bill and a provable no-egress run (the signature demo); the same session is steerable from both Chat and IDE.

### Phase 4 - Fleet, durability, and the local coder

| Item | Class | P | Notes |
|---|---|---|---|
| Background supervisor + dashboard + worktree isolation + conditional auto-PR | real+unwired | M | REDESIGN `hide-fleet` |
| Durable goals + triggers + recovery | real+unwired | M | |
| Transformer/Hybrid KV capsule | missing / blocked-on-model | M | generalizes the moat off RWKV |
| Qwen3-Coder-Next Hawking feasibility (kernels, MoE route, periodic attention, tokenizer/FIM/tool) | research / blocked-on-model | H | the capability-density coder; isolated from the ship path |
| `.tq` sub-4-bit in the default build; finish GPU bitslice kernel | partial / research | M | density lever |
| Inline FIM + next-edit | missing / blocked-on-model | M | |
| ACP server (appear natively in Zed/JetBrains) | missing | M | interop wedge, low cost |

**Exit gate:** parallel background work reduces wall-clock on eligible tasks within a merge-conflict budget; a local coder passes the private suite within the device envelope; capability-density receipts exist.

## 4. What NOT to restore blindly

- `hawking-research` (knowledge graph / arXiv): ARCHIVE for the coding-IDE slice; a scope trap.
- `hide-personalize` RLEF: REDESIGN; stub at its load-bearing core (LoRA grad / PPL / KV copy are seams). Defer the learning flywheel until Phase 4+ behind frozen evals.
- The rigid 12-phase kernel FSM: do not preserve it because it exists; the flat loop is the frontier direction (`HIDE_AGENT_KERNEL_OPTIONS.md`).
- The `tq_multiplier` context-window claim: drop it; it conflates weight compression with recall (G9/G11).

## 5. The single highest-P move

Reconnect the spine (Phase 0/1). It is `real+unwired`, so its cost is lift-and-wire; it delivers the entire loved workflow (max parity value + user love); and it unblocks the state moat, the two-surface daemon, and every ecosystem item. Everything else is downstream of it.
