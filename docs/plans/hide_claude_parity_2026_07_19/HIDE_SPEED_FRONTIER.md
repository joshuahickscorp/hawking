# HIDE Speed Frontier

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` (§3.1, §3.2, §3.3, §5), `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json`, `hawking_ide_frontier_2026_07_19.md` §5.2, §5.4-5.7 (Bible Book VIII).
Status: specification for the HIDE latency budget and its levers. Every target is a design budget (INFERRED) gated by the readiness of the lever in its row; every lever is tagged real-and-wired / real-but-unwired / partial / stub / missing. External numbers are labeled MEASURED with their harness and are NOT Apple results unless said so. Companions: `HIDE_STATE_CAPSULE_ABI.md` (the state moat this doc schedules), `HIDE_TOOL_SKILL_PLUGIN_MCP_ABI.md` (the action plane whose warm surfaces this doc times).

## 1. Thesis: optimize the critical path, not tokens per second

The unit of speed is the user-perceived critical path of one agent turn, not decode throughput. A turn that reads three files, edits one, and runs a test spends almost all of its wall-clock in reuse boundaries, tool round-trips, and cold-start stalls, not in the model's tokens per second. Two levers dominate, in order:

1. **Do not do the work.** Reuse a warm prefix or state instead of re-prefilling; answer from a deterministic index instead of an inference; skip a tool round-trip entirely when a cached artifact already holds the answer. Prompt caching is a product-level ABI, and its degradation is an incident, not a metric (DOCUMENTED, §5.2). Tool speed is largely a round-trip problem (§5.6).
2. **Overlap the work you must do.** Parallel read-only tools, warm servers, dependency-aware futures, and best-of-N forks convert serial latency into one wide step.

The structural advantage HIDE has over Claude Code is that the model runs locally, so **there is no network request on the critical path** for a diff, a selection read, a diagnostic, a tool dispatch, an interrupt, or a state fork (parity `ide.two_surface_bridge` / `loop.interrupt_and_keep` hawking_superiority; `HIDE_TWO_SURFACE_ARCHITECTURE.md` §8.2). That advantage is real only after the levers below are wired; today it is latent.

**Honest baseline.** The last-built live turn is a single-shot 256-token `generate` on the raw prompt with empty history (VERIFIED at the packed commit `5a99d0e2`, archaeology S2, `host.rs:848-863`; the active tree does not build this backend, so this is the last-built baseline, not a currently-running path), and batch-one direct admission cold-prefills every request (G4). The speed frontier is overwhelmingly reconnection and exposure of proven parts, not invention.

## 2. The latency budget

Design budgets (p50 / p95) for the reconnected vertical slice on high-memory Apple Silicon. Test/build/network wall-time is excluded from a row's budget: it belongs to the tool, not to the critical-path render (§5.6). Perception anchors: ~16 ms is one frame (imperceptible), ~100 ms feels instant, ~1 s holds attention.

| Interaction | Budget p50 / p95 | Critical path | Dominant lever (readiness) | Gate / evidence |
|---|---|---|---|---|
| App startup (launch -> interactive shell) | 400 ms / 900 ms | Tauri window + FE hydrate; model load async off-path | Lazy/async model load (partial: FE real, deferral is a build item) | FE shell real (archaeology §3.4) |
| Session resume (open recent -> restored context) | 120 ms / 300 ms | Load `.sstate` capsule, restore state, no re-prefill | Warm-state `.sstate` + capsule (real-but-unwired / missing) | `cache/sstate_disk.rs` zero callers; capsule exposure per `HIDE_STATE_CAPSULE_ABI.md` §8. Parity fallback = transcript re-read + re-prefill (`session.durable_transcript`) |
| First model token (warm prefix hit) | 120 ms / 400 ms | Match prefix, skip re-prefill of shared span | Prefix reuse / radix cache (partial: queue-path wired, direct-admit cold, G4) | serve `lib.rs:605-918`; ext. anchor 3.9x-27x TTFT vs cold prefill at 2K-16K (MEASURED, CUDA, NOT Apple, §5.4) |
| First model token (cold, no hit) | bounded by prefill of uncached suffix | Full prefill | none (this is the cost reuse removes) | G4: default `max_batch_size=1` cold-prefills (`lib.rs:509`, `:900-918`) |
| Tool dispatch (call parsed -> tool starts) | 5 ms / 20 ms | Parse completion, dispatch locally, no network | Local dispatch loop (parser wired API-shaping; dispatch loop real-but-unwired) | parser `tool_calls.rs` (archaeology §3.3); runner packed in `hide-tools` |
| Terminal feedback (keystroke / command echo) | 16 ms / 33 ms | PTY echo to xterm | Integrated PTY (missing) | Terminal has no PTY, commands only queued (archaeology §3.4) |
| Context compile (incremental, warm index) | 150 ms / 400 ms | Retrieve candidates, value-density knapsack fill | Incremental index + reserve-then-fill compiler (real-but-unwired) | `hawking-context` / `hawking-index` @`5a99d0e2` (archaeology §3.5) |
| Memory retrieval (FTS5 + cosine) | 25 ms / 80 ms | Lexical + vector lookup over local store | SQLite FTS5 + cosine (real-but-unwired) | `hawking-context` memory store, packed |
| State fork (memcpy clone of recurrent state) | 1 ms / 8 ms | O(state bytes) copy, no re-prefill | RWKV state fork (real-but-unwired) | `rwkv7.rs:376-378`; transformer/hybrid fork = missing (capsule ABI §6); ext. anchor sub-ms resident restore (MEASURED, CUDA, §5.4) |
| Agent spawn (fork worker from warm state) | 15 ms / 60 ms | Capsule fork + inherit warm servers | Warm-state fork + warm MCP/LSP inheritance (real-but-unwired / packed) | fork `rwkv7.rs:376-378`; subagent support packed `hide-kernel`. Parity fallback = cold context assembly (`subagents.fork_worker`) |
| File-open-from-chat (click ref -> editor at line) | 30 ms / 80 ms | FE nav + repo-snapshot lookup, local | Local repo snapshot id (real-and-wired FE) | Explorer/Editor real (archaeology §3.4); cross-surface transition (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §4) |
| Diff render (patch -> side-by-side) | 40 ms / 100 ms | Render patch transaction in Monaco | Local diff (real-and-wired FE) | DiffReview real (archaeology §3.4) |
| Hunk accept (Tab -> applied + verified) | 25 ms / 70 ms | `base_hash` check, apply, re-render | Tiered verifying edit applier (real-but-unwired) | `hide-tools` search_replace/apply_patch + base_hash optimistic concurrency (archaeology §3.5); FE gesture real |
| Test-result display (process exit -> rendered) | 40 ms / 100 ms | Stream-parse result, render panel (test runtime excluded) | Warm test worker + streaming parse (missing / FE real) | keep test workers warm (§5.6); FE display real |

**Reading the table:** where a lever is real-but-unwired or missing, the budget is the post-wiring SLO, and the current state is stated in the same row. No row's budget is claimed as MEASURED-on-Apple; all are design targets to be replaced by measured p50/p95 once the slice is reconnected (`hawking-bench` is the only live harness; `hawking-eval` is sealed, archaeology §6.5).

## 3. Prompt ABI: the cache-key contract as a monitored SLO

TTFT (row 3) and resume (row 2) both bottom out in one contract: whether the next turn's prefix byte-matches a warm one. Prompt caching is a product-level ABI whose hit-rate is a monitored SLO, and a drop is treated as an incident (DOCUMENTED, §5.2). HIDE must own this contract explicitly, because a cache hit requires an exact shared prefix and any drift in serialization, tool order, or a mid-session model swap silently breaks reuse (DOCUMENTED, §5.2).

**PromptABI (a versioned specification, PARITY):**

| Field | Rule | Why |
|---|---|---|
| Deterministic byte/token serialization | One canonical encoder; same logical input -> identical bytes -> identical tokens | Any nondeterminism is a silent cache miss (§5.2) |
| Layer ordering | Static system + stable tool registry first; project/memory next; dynamic conversation last | Only a stable head reuses; volatile content must sit at the tail (§5.2) |
| Versioned system/project blocks | Each block carries a content hash + version; a change bumps the version, it does not mutate in place | Reuse survives everything below the changed block |
| Append-only dynamic tail | New turns append; earlier messages are never edited (mirror the Codex append-state discipline) | Editing an earlier message invalidates the whole suffix (§5.2) |
| Tool registry as ABI | Stable deferred stubs; full schemas append after discovery; tool order fixed | Tool definitions can cost 55K-134K tokens (MEASURED, §5.6); reordering breaks the prefix |
| Session/model affinity | A cache key binds `{model_weights_id, arch_id, tokenizer_id, prompt_abi_version}`; a model swap starts a new key | Changing models mid-session breaks reuse (§5.2); these are the `IdentityBinding` fields of `HIDE_STATE_CAPSULE_ABI.md` §4 |
| Cache-key explanation | Every request can emit which prefix it matched, the shared length, and why it missed | Makes a regression debuggable instead of a mystery |
| Hit/miss + invalidation telemetry | Per-turn: matched prefix len, reused/loaded/evicted/recomputed tokens; hit-rate is an alerting SLO | A degradation must page like an incident (§5.2) |
| Test vectors | A frozen corpus proving equivalent logical inputs serialize byte-identically and non-equivalent ones do not | The only defense against silent drift; reuse the byte-ledger discipline already in `hawking-seed-c` |

Compaction must fork from the identical parent prefix and append the compaction request, never rewrite history (DOCUMENTED, §5.2); this is also what makes compaction reversible detail-hiding rather than lossy summarization (parity `cost.usage_transparency` hawking_superiority; capsule ABI `CommitBoundary.kind = Compaction`).

**Readiness: missing.** No PromptABI exists in the active tree. Serve prefix reuse is a special-case system-prefix path on the queue drain only (G4); the FE wire contract is unanchored because its Rust source of truth is packed (archaeology §3.4, `HIDE_TWO_SURFACE_ARCHITECTURE.md` §3). `prompt_abi_version` is already a required identity field of the state capsule, so the PromptABI and the capsule ABI must ship together.

## 4. The reuse ladder: three mechanisms, honest readiness

Reuse is not one thing; it is three mechanisms with different cost models (§5.4). Conflating them hides where the latency actually goes.

| Mechanism | What it reuses | Hawking readiness | Build item |
|---|---|---|---|
| Prefix KV (radix) | Token-prefix of the transformer/hybrid KV | partial: queue-path wired (`lib.rs:605-918`), direct-admit cold (G4) | Fix batch-one reuse; replace the system-prefix hint with a token-prefix radix cache |
| Hierarchical KV store | Cold KV blocks tiered GPU/CPU/SSD | missing: deferred (G5, `system_kv_bank.rs:20-23`) | Co-design layout + transfer + scheduler; model cache-load latency in admission |
| Execution-state capsule | The complete committed state (KV + recurrent + metadata) | real-but-unwired (RWKV) / missing (transformer/hybrid) | Per `HIDE_STATE_CAPSULE_ABI.md` §8: state routes, session affinity, `.sstate` persistence |

**Fix direct-admit / batch-one prefix reuse (G4, the single highest-leverage TTFT fix).** `copy_kv_prefix_to_slot` has exactly one call site, the queue-drain branch (`lib.rs:900-918`), and the default `max_batch_size=1` (`lib.rs:509`) means the first and only request always cold-prefills. Reuse must fire on direct admission, especially at batch size one, which is the interactive coding case (§5.4). This is a wiring fix on an existing primitive, not new work.

**Token-prefix radix caching.** Replace the single special-case system-prefix hint (`system_kv_bank` stores zero KV bytes, G5, `system_kv_bank.rs:12-18`) with a token-prefix radix or content-addressed cache, scheduled by cache affinity without starving interactive work, isolated by workspace and trust domain, and instrumented to observe admitted / reused / loaded / evicted / recomputed tokens (§5.4). SGLang RadixAttention and vLLM prefix caching are the architectural evidence (they are datacenter systems, not Apple results; §5.4).

**Hierarchical KV store (defer, but design honestly).** Strata (OSDI 2026) shows a naive GPU/CPU/SSD hierarchy becomes I/O-bound from fragmentation and load stalls, reporting up to 5x over vLLM-LMCache in its evaluated workloads (MEASURED, datacenter, NOT Apple, §5.4). If HIDE builds this, it co-designs cache layout, transfer size, and a scheduler that models cache-load latency, and it keeps short interactive requests from regressing.

**Execution-state capsule (the moat).** The full spec lives in `HIDE_STATE_CAPSULE_ABI.md`. For the speed budget, three facts matter: state fork is a memcpy with no re-prefill (`rwkv7.rs:376-378`, real-but-unwired), warm-state `.sstate` persistence exists with zero callers (`cache/sstate_disk.rs`, real-but-unwired), and **GPU->CPU recurrent readback is missing** (`rwkv7.rs:1720-1723`), so a live-GPU mid-stream capsule is not byte-exact yet (gate G-CAP-1, capsule ABI §5). Session->slot affinity is also missing (G2, anonymous `u32` slots, `http.rs:124,127`), and it is a prerequisite: a resume or fork must target a real warmed arena, not an anonymous one. Until G-CAP-1 and G2 land, rows 2, 8, and 9 hit their parity fallback (re-prefill / cold assembly), not their supremacy budget.

## 5. Warm surfaces: keep the world hot

Cold-starting an MCP server, an LSP, a search index, or a test worker on the critical path is a self-inflicted round-trip (§5.6). HIDE keeps them warm across turns and across sessions, and snapshots the resolved and authenticated server set so a fork inherits live connections instantly (parity `mcp.client_host_server` hawking_superiority). The action-plane contract for these surfaces is `HIDE_TOOL_SKILL_PLUGIN_MCP_ABI.md`; this doc only sets their latency obligation.

| Warm surface | Obligation | Readiness |
|---|---|---|
| Local MCP servers (stdio) | Long-lived subprocesses reused across sessions; connection set foldable into a capsule | real-but-unwired: JSON-RPC MCP client (stdio + Streamable HTTP) packed in `hide-tools` (archaeology §3.5; parity `mcp.client_host_server` packed_unwired) |
| LSP / DAP | Warm language servers for symbols, diagnostics, definitions | missing in active tree; bridge shape documented (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §5) |
| Search / repo index | Incremental daemon stays resident (see §7) | real-but-unwired: `hawking-index` MVCC daemon @`5a99d0e2` |
| Build / test workers | Reused processes; stream results, do not respawn per turn | missing (respawn cost is why row 12 needs a warm worker) |
| PTY terminal | Persistent shell, not per-command queue | missing (archaeology §3.4) |

## 6. Tool round-trips: parallel reads, futures, artifact handles, deferred schemas

The tool plane is where a turn most often stalls serially. The frontier findings (§5.6) map directly onto build items:

- **Three to five core tools always present; a stable deferred namespace index; on-demand schemas.** Deferred tool loading cut initial definition load in Anthropic's example and improved large-catalog selection (DOCUMENTED, §5.6). Full schemas append after discovery, preserving the prefix (§3).
- **Parallel read-only tools.** Independent reads (file, search, symbol) run concurrently with deterministic result-commit order; only reads parallelize (writes stay serialized and gated). Readiness: real-but-unwired (ignore-walker search, `proc` exec-nonzero-as-data, and the tool runner are packed in `hide-tools`; the parallelism is not on any live path, archaeology §3.5, master reconciliation).
- **Dependency-aware futures with cancellation.** Overlap tool execution with continued decoding via symbolic futures, no fine-tuning required (AsyncFC, §5.6), and cancel cleanly on interrupt (parity `loop.interrupt_and_keep`).
- **Programmatic tool control + artifact handles.** Keep intermediate results outside model context and replace many round-trips with one sandboxed control program: Anthropic reports a 37% average token reduction and elimination of 19+ inference passes in a 20-call example (MEASURED, §5.6). Large outputs return as artifact handles, not dumps into context.

Naive grammar masking can suppress tool selection if the model must first choose between prose and a tool tag; use two-stage or tag-dispatch decoding and measure tool-selection recall, not merely JSON validity (§5.6). The constrained-decode primitives are packed (see §8).

## 7. Incremental repo intelligence: compute deltas, not the world

Context compile (row 7) and memory retrieval (row 8) stay inside budget only if the index never rescans unchanged files. `hawking-index` already implements the right machine (real-but-unwired, packed @`5a99d0e2`, archaeology §3.5, §5 lever ledger):

- **BLAKE3 merkle change-gate:** a subtree whose hash is unchanged is skipped entirely; only changed paths re-parse.
- **AST / symbol / git deltas:** tree-sitter defs and refs, cAST chunking, SCIP ids, updated per changed file, not per repo.
- **Incremental MVCC daemon with crash recovery:** the index is a resident warm surface (§5), snapshot-consistent under concurrent edits.
- **Retrieval:** FTS5 + graph, PageRank repo-map, hybrid RRF retriever, feeding the reserve-then-fill compiler.

Speed implication: the first index build is a one-time cost; every subsequent turn pays only for the delta. This is what makes the 150 ms context-compile budget defensible, and it must beat deterministic lexical/symbol/graph/RepoMap baselines before any learned retriever is added (§5.1).

## 8. FIM and next-edit: the missing inline-completion path

Inline completion and localized next-edit are latency-critical (they sit under the cursor at typing speed), and they are the natural home for file-as-draft speculation (§9). Qwen3-Coder-Next ships native FIM support (DOCUMENTED, §5.3), but **Hawking has no native FIM path today** (missing): the FIM contract (tokenizer, special tokens, chat template) is an explicit build item in the architecture-fit study (§5.3). Until it lands, HIDE has no first-class fill-in-the-middle or next-edit lane, only full-turn generation. Gate: the Qwen3-Coder-Next FIM/tokenizer contract (SUPREMACY item, gated on the hybrid-model build, `HIDE_STATE_CAPSULE_ABI.md` §6 notes the same model needs its transformer/periodic-KV half built).

## 9. Model escalation: route from trajectory evidence

A cheap, small model on the critical path avoids expensive inference for easy steps; escalation is a speed lever, not only a cost lever. Route from trajectory evidence (files, tests, diffs, repeated failures, uncertainty, cache affinity), not only the initial prompt, and optimize success@time and success@cost under quality floors, not cheapest-call percentage (§5.3.1). Start with a transparent policy and an intentionally small model pool; collect counterfactual traces before training any router (§5.3.1). Readiness: partial (confidence-gated escalation via self-consistency vote is packed in `hawking-orch`, real-but-unwired, archaeology §3.5, §5 lever ledger).

## 10. Speculation rules: lossless or explicitly stageable

Speculation is the last mile of decode speed, and it is dangerous if it changes outputs or leaks intent. The rules are hard (§5.7).

| Rule | Enforcement | Hawking hook (readiness) |
|---|---|---|
| Token speculation must prove greedy / target-distribution equivalence for the configured mode | Accept iff `argmax == draft`, which is bit-identical to greedy (the lossless gate) | Exact-match verifier `verifier.rs:77-133` ("still lossless" path confirmed), real-but-unwired; runs single-seq only, never over serve (archaeology §3.3, T8) |
| Edit speculation stays a reviewable transaction | A speculated edit is applied only through the diff/hunk gate, never silently | Tiered verifying edit applier + `base_hash` optimistic concurrency, `hide-tools`, real-but-unwired (row 11) |
| Tool speculation only for authorized, local, side-effect-free, idempotent reads | A speculated tool call must be a whitelisted read; anything else is refused | Read-only allowlist (parity `perm.rule_engine`); dispatch loop packed |
| Never speculate external requests, secrets, messages, writes, deletes, purchases, or credentials | Hard deny; even a discarded external request leaks predicted intent | Ghost Tool Calls warning (§5.7); egress default-off makes the class physically absent (parity `security.sandbox` hawking_superiority) |
| Measure every speculation | Track acceptance, wasted work, rollback cost, memory overhead, critical-path savings, task-quality delta | `hawking-bench` (perf) live; task-quality needs `hawking-eval` reintegrated (archaeology §6.5) |

Candidate lossless lanes: suffix decoding for repetitive edit output; file-as-draft verification for localized edits (pairs with §8); schema-aware tool-call drafts (training-free lossless tool-spec-decode via schema jump-forward + prompt-lookup is packed in `hawking-orch/tool_spec_decode.rs`, partial, archaeology §3.3, §3.5); model-native multi-token prediction; safe local speculative reads; and best-of-N branches from cheap state forks (§4). PASTE and ToolSpec are 2026 preprints, not Hawking receipts (§5.7); nothing ships until its acceptance and quality delta are measured.

## 11. Parity vs supremacy: gate every claim

PARITY reproduces what Claude Code does; SUPREMACY is what the local runtime does structurally better. Every supremacy claim is gated on a named build item; none is asserted as shipping today.

| Claim | Class | Gating build item | Readiness of the gate |
|---|---|---|---|
| Prefix-cache discipline with hit-rate as a monitored SLO | PARITY | PromptABI (§3) | missing |
| Interactive TTFT via prefix reuse at batch-one | PARITY | Fix G4 direct-admit reuse (§4) | partial (queue-path wired) |
| Instant resume with no re-prefill | SUPREMACY | Warm `.sstate` + capsule + G2 affinity (`HIDE_STATE_CAPSULE_ABI.md` §8) | real-but-unwired / missing |
| Zero-latency interrupt (no request to cancel) | SUPREMACY | Live kernel turn to interrupt (parity `loop.interrupt_and_keep`) | ui_only (no live turn) |
| Sub-ms state fork -> best-of-N from one warm start | SUPREMACY | Fork exposure + G-CAP-1 GPU readback (§4) | real-but-unwired / missing |
| Zero-round-trip diff / selection / diagnostics | SUPREMACY | IDE loopback bridge wired (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §5) | partial (FE real, bridge packed) |
| Warm MCP/LSP inherited by a fork | SUPREMACY | Warm surfaces (§5) + capsule fold-in | real-but-unwired |
| Lossless token speculation | PARITY (correctness) / SUPREMACY (speed) | Wire `verifier.rs` into batched serve (§10) | real-but-unwired |
| FIM / next-edit inline completion | PARITY | Qwen3-Coder-Next FIM contract (§8) | missing |
| Reversible compaction, no lossy summary | SUPREMACY | Capsule `CommitBoundary=Compaction` (§3) | missing |

## 12. What to measure (speed as a monitored contract)

Speed is a contract only if it is instrumented. The minimum telemetry, doctrine-clean (no dollar meter; local performance framed as headroom, parity `cost.usage_transparency` / `loop.status_line` hawking_superiority):

- **Per-turn critical-path breakdown:** compile / TTFT / decode / tool / render, so a regression is attributable.
- **Prefix cache:** hit-rate (alerting SLO), matched prefix length, reused / loaded / evicted / recomputed tokens (§3, §4).
- **TTFT distribution:** p50 / p95, warm-hit vs cold, replacing the design budgets in §2 with measured values.
- **Speculation:** acceptance, wasted work, rollback cost, memory overhead, critical-path savings, task-quality delta (§10).
- **Warm surfaces:** MCP/LSP/index/test-worker warm-hit rate vs cold-start count (§5).
- **Local headroom, not a countdown:** tok/s, energy J/tok, resident GB, state-fork depth, parallelism (parity `loop.status_line` hawking_superiority).

Current harness reality: `hawking-bench` (perf) is live; `hawking-eval` (pass@1 + Wilson CI) is sealed and must be reintegrated before any quality-gated speed claim is credible (archaeology §6.5). Until the slice is reconnected, every number in §2 is a target, not a receipt.
