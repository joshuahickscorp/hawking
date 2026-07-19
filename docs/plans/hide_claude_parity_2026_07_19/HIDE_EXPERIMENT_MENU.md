# HIDE Experiment Menu

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` (readiness with file:line), `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json` (behavior ids), `HIDE_STATE_CAPSULE_ABI.md` (state moat discipline), `HIDE_TWO_SURFACE_ARCHITECTURE.md` (shared core). Model for this document: dossier `docs/plans/hawking_ide_frontier_2026_07_19.md` section 9, expanded for the parity plus supremacy frontier.
Status: research menu. This authorizes isolated experiments behind flags, not product implementation. Ordering and sizing are owned by `HIDE_PRIORITIZED_BUILD_LADDER.md`; the claims each bet unblocks are defined in `HIDE_SUPREMACY_THESIS.md`.

## 1. How to read this menu

An experiment bet is an isolated research question with one cheap first experiment and an explicit stop rule, so a negative result costs days, not a phase. Every bet carries three mandatory disciplines (dossier section 9, "feature flags, receipts, and rollback are mandatory for every bet"):

- **Feature flag:** the bet is default-off; the shipping default build is unchanged until the receipt clears a gate.
- **Receipt:** a named artifact under `reports/` produced from the real app or serve path, not a microbench in isolation (dossier decision 12: capability and speed claims require receipts from the real path).
- **Rollback:** flag-off deterministically restores the prior behavior; no bet leaves a half-wired path on the default route.

**PARITY vs SUPREMACY are separated.** A parity bet reproduces something Claude Code already does and repairs the broken vertical slice (`HIDE_TWO_SURFACE_ARCHITECTURE.md` section 7). A supremacy bet is a Hawking-native advantage, and it is gated on the specific build item it needs, never asserted from an unwired primitive (`HIDE_STATE_CAPSULE_ABI.md` section 10). Section 4 marks each bet's class; section 6 maps every supremacy claim to its prerequisite bets.

**Readiness key** (from the archaeology lever ledger, `HIDE_LIVE_ARCHAEOLOGY.md` section 5): real-and-wired / real-but-unwired / partial / stub / missing. A packed-and-unwired primitive is never counted as a shipping capability.

## 2. The dependency spine

The state moat is a chain, not a set of independent wins. Nothing about "fork a warm state into N agents" is load-bearing until the capture is exact and the routes exist. Read this order before scheduling:

```text
B1 GPU->CPU readback (exact live capture)     [missing, hard gate G-CAP-1]
      |
      v
B2 HTTP state routes + session->slot affinity [missing]
      |
      +--> B3 warm-state fork best-of-N        [fork real-but-unwired]
      |
      +--> B4 transformer/Hybrid KV capsule    [missing] --> needs B10 for Qwen3-Coder-Next

B9 flat kernel loop (replace 256-tok single-shot)  [good loop packed; live turn stub]
      |
      +--> B8 compiler output fed to generation (fix S3) [packed, discarded]
      +--> B5 tool-spec-decode into batched serve        [partial, packed]

B6 direct-admit prefix + radix cache   [partial]   (independent perf spine)
B7 .tq sub-4-bit default serving        [partial]   (independent footprint spine)
B10 Qwen3-Coder-Next feasibility        [missing]   (isolated arch branch)
B11 model-role router                   [partial]   (needs B9 loop to route within)
```

B1 is the first hard gate on the entire warm-state moat (`HIDE_STATE_CAPSULE_ABI.md` gate G-CAP-1). B9 is the first hard gate on capability parity: until the live turn stops being a 256-token single-shot, no context, tool, or verify bet has a real loop to run inside.

## 3. The bet menu

Columns: bet / why it matters / first experiment / kill-or-pause-when / readiness. IDs are stable references used by section 4 and by `HIDE_PRIORITIZED_BUILD_LADDER.md`.

| ID | Bet | Why it matters | First experiment | Kill or pause when | Readiness (archaeology) |
|---|---|---|---|---|---|
| **B1** | GPU->CPU recurrent readback for exact live-state capture | Blocks the whole warm-state moat; on the shipping macOS GPU decode path `self.state` is stale vs the live GPU arena, so mid-stream capture is not exact (`rwkv7.rs:1720-1723`, ABI gate G-CAP-1). Only a reverse of the one-way `copy_cpu_state_to_gpu_slot` (`rwkv7.rs:1897`) makes GPU capsules honest. | Implement a reverse readback at a committed token boundary; assert the read-back CPU state decodes bit-identical next-token logits vs the CPU oracle (mirror `tests/rwkv7_state_checkpoint_parity.rs`); measure capture and restore wall-clock and memory on M3 Ultra. | Pause if capture+restore cost exceeds the cold prefill it would replace (TTFT benefit erased); kill if read-back cannot be made bit-exact at a committed boundary. | **missing** (no reverse readback fn; `rwkv7.rs:1720-1723`) |
| **B2** | HTTP state save/load/fork routes + session->slot affinity | The primitives exist but no route reaches them (G1); slots are anonymous `u32` with no session id (G2), so a fork/handoff cannot target a real warmed arena. This is the wire that turns the capsule ABI into a capability. | Add `POST /v1/hawking/state/{save,load,fork}` over the existing engine seam (`engine.rs:339-358`); pin a client-stable session id to a warm slot; identity-verify on load/fork per ABI section 4 and refuse on mismatch (typed error, `providers/verify.rs` pattern). Bind loopback + auth first (serve currently binds `0.0.0.0`, no auth, G10). | Pause if session-affinity hit rate is low or slot pinning starves interactive tail latency; kill if routes cannot ship loopback-authenticated (security precondition, dossier 5.9). | **missing** (routes absent `http.rs:160-172`; anonymous slots `http.rs:124,127`) |
| **B3** | Warm-state fork best-of-N with execution tie-break | Fork is a memcpy clone with no re-prefill (`rwkv7.rs:376`); N cheap local branches reconciled by tests/build (not state averaging) is the structural answer to Claude's metered parallelism. Must be proven to beat spending the same wall-clock on one stronger model. | On a set of hard private tasks, fork one warm RWKV state into N branches, tie-break by deterministic oracle (build/tests), and record **verified quality gain per second** vs a single stronger-model run at matched wall-clock. | Kill when verified quality gain per second is inferior to one stronger model (dossier section 9); pause if verification cost dominates the cheap fork so best-of-N is net-negative. | fork **real-but-unwired** (`rwkv7.rs:376`); FE Fork-and-Try-N **ui_only** |
| **B4** | Transformer/Hybrid KV capsule for Qwen3-Coder-Next | `KvCache` is not serializable and `qwen_dense`/`deepseek_v2` never override the checkpoint seam, so transformer and hybrid sessions cannot be snapshot or forked at all. Qwen3-Coder-Next is a Hybrid (recurrent + periodic KV), so the flagship local coder needs this before its state is capsulable (ABI section 6). | Inventory the closed set of live buffers (KV + recurrent + periodic-attention + metadata) at a committed boundary; serialize and restore on one supported transformer; parity-test next-token logits and a long continuation. | Kill if capture/restore cost or memory erases the TTFT benefit vs cold prefill (dossier section 9); pause if `KvCache` serialization cannot be made byte-exact. | **missing** (`cache/mod.rs` not serializable) |
| **B5** | Reintegrate `hawking-orch` tool-spec-decode into the batched serve path | First-try-valid, streamed tool calls. Today tool SSE buffers to completion (T7), the batched path forces `json_mode:false` and does no draft/verify (T8), while the training-free lossless tool-spec-decode (schema jump-forward + prompt-lookup) sits packed at `5a99d0e2` and the exact-match verifier (accept iff `argmax == draft`, bit-identical to greedy) runs single-seq only. | Wire `tool_spec_decode.rs` (jump-forward + prompt-lookup) + the single-seq JSON constraint mask + the exact-match verify gate (`verifier.rs:77-133`) into `driver.rs` for one stable tool family; measure first-try-valid rate, tool-selection recall, acceptance, and tokens/s over the HTTP boundary. | Kill if tool-selection recall drops or acceptance is too low (dossier section 9); pause if lossless verify cannot be preserved on the batched path (output must stay bit-identical to greedy). | **partial** (orch primitive real+tested @`5a99d0e2`; verifier real-but-unwired `verifier.rs:77-133`; batched path `driver.rs:52-110` unconstrained) |
| **B6** | Fix direct-admit / batch-one prefix reuse + token-prefix radix cache | The sole `copy_kv_prefix_to_slot` call site is the queue-drain branch (`lib.rs:908`); with default `max_batch_size=1` (`lib.rs:386`) the first/only request cold-prefills (G4). A content-addressed radix cache replaces the single system-prefix hint (G5 stores zero KV bytes) and is the broad transformer/hybrid reuse lever. | Add prefix reuse on direct admission (batch size one); build a token-prefix radix cache over a shared repo prefix; benchmark reused/loaded/evicted/recomputed tokens and interactive TTFT, warm vs cold. | Kill if radix hit rate is low or eviction harms interactive tail latency (dossier section 9); pause if direct-admit reuse is not bit-identical to a cold prefill. | **partial** (queue-path wired `lib.rs:605-918`; direct-admit missing; radix cache missing) |
| **B7** | `.tq` sub-4-bit native serving in the default build | Both `qwen_dense` and `rwkv7` now read `.tq`, but only under the non-default `tq` cargo feature plus `HAWKING_*_TQ` env flags, and the GPU bitslice GEMV is staged (CPU RHT matvec is the parity oracle). Sub-4-bit is the lever that keeps a bigger model resident on Apple memory. | On one supported model, run `.tq` serving through the real serve path; gate output against reference precision (top-token agreement, perplexity, one coding task); measure resident GB and decode tok/s vs the bf16/Q4 baseline. | Pause if `.tq` fails the reference-quality gate or the GPU GEMV is not at parity with the CPU RHT oracle; kill if there is no end-to-end resident-footprint or quality win. | **partial** (feature+env gated; GPU bitslice staged) |
| **B8** | Reserve-then-fill compiler actually fed to generation (fix S3) | The reserve-then-fill context compiler runs only behind the `context` connector and its compiled prompt is discarded relative to generation (S3). Feeding the compiled ContextPack is the difference between a small exact cited working set and a raw prompt (dossier 5.1). | Restore `hawking-context` (@`5a99d0e2`); feed its compiled ContextPack (system/response/scratchpad reserved, value-density knapsack, head/tail order) to the live turn; A/B task success and context receipts (raw prompt vs compiled) on a private multi-file task. | Kill if the compiled ContextPack does not improve task success vs the raw prompt (no capability lift) or regresses; pause if compiler budgets diverge from the live tokenizer. | **unwired (packed)** (`hawking-context` @`5a99d0e2`; compiled prompt discarded S3) |
| **B9** | Flat kernel loop replacing the 256-token single-shot | The live `SubmitTurn` sends the raw prompt, empty history, `max_output_tokens=256` (S2, `host.rs:848-863`) under `StubPlanner` (S1). Replacing it with a flat compile-context -> tool-observation -> deterministic-verify loop is the single move that makes the polished FE real (dossier 5.8, decision 6). | Lift `hide-core` + `hide-serve` + `hide-kernel` (RuntimePlanner, not StubPlanner) + `hide-tools` from `5a99d0e2`; run one private multi-file Rust/TS task end-to-end through the real app with transactional reviewable edits, cancel, resume, replay; report pass@1 (reuse `hawking-eval`). | Pause (do not kill: this is P0 parity, not optional) if the flat loop cannot solve a task the stub cannot, or if it regresses reviewability; escalate loop design to `HIDE_AGENT_KERNEL_OPTIONS.md`. | good loop **unwired (packed)** (`hide-kernel` @`5a99d0e2`); live turn **stub** (S1/S2) |
| **B10** | Qwen3-Coder-Next Hawking architecture feasibility | Strongest capability-density fit (80B total / 3B active, hybrid Gated DeltaNet + periodic attention + sparse MoE, native 262k, FIM, tool format). Today the generic `qwen_moe` loader is a STUB and MoE ships only via `deepseek_v2`; Gated DeltaNet kernels do not exist. Isolated from the vertical-slice ship path (dossier Phase 1.8). | Reference-parity study: Gated DeltaNet kernel + state semantics, exact sparse MoE route, periodic-attention KV layout, tokenizer/chat-template/tool-parser/FIM contract; compare next-token output vs a reference runtime; measure Apple prefill/decode/memory/power. | Kill when Apple performance or quant quality fails the local-agent envelope (dossier section 9); pause if Gated DeltaNet kernel parity vs the reference runtime fails. | **missing** (`qwen_moe` stub; no Gated DeltaNet; MoE via `deepseek_v2` only) |
| **B11** | Local model-role router from trajectory evidence | Route separately for exploration, patching, diagnosis, review, explanation using tests, repeated failures, uncertainty, cache affinity, and complementarity, optimizing success@time under a quality floor (dossier 5.3.1). The `hawking-orch` role router and the provider registry are packed/unexposed and the adapters are declarative descriptors, not a runtime engine. | Ship a transparent rule policy over a deliberately small model pool inside the B9 loop; collect counterfactual traces; only then compare an offline-learned router against the rules out of sample. | Kill when the learned router does not beat transparent rules out of sample (dossier section 9); pause if the small pool shows no real complementarity. | **partial** (role router packed @`5a99d0e2`; registry ACTIVE_UNEXPOSED; adapters declarative) |

## 4. Per-bet operations (flag / receipt / rollback / class)

Every bet is default-off behind its flag, produces its receipt from the real path, and reverts cleanly on flag-off. Class marks PARITY (repair the slice / reproduce Claude Code) vs SUPREMACY (Hawking-native advantage) vs ENABLER (prerequisite that unblocks a supremacy bet without a user-visible behavior of its own).

| ID | Class | Feature flag (default off) | Receipt artifact | Rollback | Prerequisites |
|---|---|---|---|---|---|
| B1 | ENABLER | `HAWKING_RWKV_GPU_READBACK=1` | `reports/state/gpu_readback_capture.json` (bit-identity + capture/restore ms + GB) | flag off -> capsules stay `gpu_synced=false`, CPU/fresh-boundary only | none (root gate) |
| B2 | PARITY + ENABLER | serve `state-routes` feature + `HAWKING_SERVE_SESSION_AFFINITY=1` | `reports/serve/state_routes_affinity.json` (route parity + affinity hit rate + tail latency) | flag off -> routes absent, anonymous slots restored | B1 (for GPU-synced capsules); loopback+auth |
| B3 | SUPREMACY | `HIDE_FORK_BEST_OF_N=<N>` | `reports/eval/best_of_n_vs_stronger.json` (verified gain/sec vs one stronger model) | flag off -> single-branch turn | B1, B2 |
| B4 | SUPREMACY + PARITY | `kv-capsule` feature + `HAWKING_KV_CAPSULE=1` | `reports/state/kv_capsule_parity.json` (logit parity + long-continuation + capture cost) | flag off -> transformer/hybrid sessions non-capsulable | B1, B2 |
| B5 | PARITY + SUPREMACY | `tool-spec-decode` feature + `HAWKING_SERVE_TOOL_SPEC_DECODE=1` | `reports/serve/tool_spec_decode_firsttry.json` (first-try-valid % + recall + acceptance + tok/s) | flag off -> batched path unconstrained, tools buffer to completion | B9 (a loop to call tools) |
| B6 | PARITY + ENABLER | `HAWKING_DIRECT_ADMIT_PREFIX=1` + `HAWKING_SERVE_RADIX_CACHE=1` | `reports/serve/prefix_radix_cache.json` (reused/loaded/evicted tokens + warm/cold TTFT) | flag off -> queue-path-only reuse (`lib.rs:908`) | none |
| B7 | SUPREMACY | `tq` feature promoted behind `HAWKING_TQ_DEFAULT=1` | `reports/serve/tq_default_quality.json` (top-token + PPL + coding task + resident GB + tok/s) | flag off -> `.tq` stays non-default | GPU GEMV vs CPU RHT parity |
| B8 | PARITY | `HIDE_FEED_COMPILED_CONTEXT=1` | `reports/context/compiled_context_ab.json` (pass@1 raw vs compiled + context receipt) | flag off -> raw prompt to generation (current S3 behavior) | B9 |
| B9 | PARITY (P0) | `HIDE_FLAT_KERNEL=1` (RuntimePlanner vs StubPlanner) | `reports/eval/flat_loop_slice.json` (pass@1 + transactional-edit + resume/replay proof) | flag off -> 256-tok single-shot stub turn (S1/S2) | `hide-core`/`hide-serve`/`hide-kernel`/`hide-tools` restored |
| B10 | SUPREMACY | `arch-qwen3-coder-next` feature (build-isolated branch) | `reports/arch/qwen3_coder_next_feasibility.json` (reference parity + Apple prefill/decode/power) | branch isolated; never on the vertical-slice ship path | none (isolated) |
| B11 | SUPREMACY + PARITY | `HIDE_ROLE_ROUTER=rules\|learned` | `reports/router/role_router_ab.json` (success@time + escalation calibration, rules vs learned) | flag off / `rules` -> single default model | B9 |

## 5. Notes on the load-bearing bets

- **B1 is the honesty gate for the entire moat.** Until reverse readback lands, capsules from a live GPU session MUST carry `gpu_synced=false` and are valid only for CPU continuation or recompute-from-fresh-prefill (`HIDE_STATE_CAPSULE_ABI.md` G-CAP-1). No supremacy claim about mid-stream fork or instant resume may be made from the RWKV atom alone while B1 is open. The experiment is small (one reverse copy + one parity assertion) and its receipt is binary: bit-exact at a committed boundary, or not.
- **B3 is measured, never assumed.** Fork is cheap by construction (memcpy, no re-prefill), so the real budget is verification, not model compute (`HIDE_STATE_CAPSULE_ABI.md` section 10). The kill criterion is deliberately harsh: gain per second must beat one stronger model at matched wall-clock, because a pile of cheap-but-wrong branches is worse than one good answer. This bet also underwrites the parity behaviors `subagents.fork_worker`, `loop.side_query`, and `perm.plan_mode` best-of-N, so its receipt is reused across three claims.
- **B5 keeps the lossless invariant.** The exact-match verifier accepts a draft token iff `argmax == draft`, which is bit-identical to greedy (`verifier.rs:77-133`); reintegration must preserve that over the batched path, and tool-selection recall (not mere JSON validity) is the pass metric, because naive grammar masking can suppress the tool tag entirely (dossier 5.6). Speculation that cannot prove greedy-equivalence is paused, not shipped (dossier 5.7).
- **B4 gates the flagship coder.** State-fork supremacy ships first on the RWKV lane; Qwen3-Coder-Next (a Hybrid) cannot be forked until both halves of its capsule exist, so B4 and B10 together, not either alone, unblock the "fork a real coding model" claim. Keep B10 on an isolated branch so a feasibility failure never destabilizes the shipping slice.

## 6. Supremacy claims and their prerequisite bets

Every supremacy claim is gated on the specific bets below; the canonical claim register and the measured-vs-asserted status live in `HIDE_SUPREMACY_THESIS.md`. A claim is not assertable until all its prerequisite receipts clear. Behavior ids reference `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json`.

| Supremacy claim | Prerequisite bets | Anchored behavior id(s) |
|---|---|---|
| Zero-latency interrupt, branch both directions from the interrupt point | B1, B2, B3 | `loop.interrupt_and_keep` |
| Warm-fork side-query at zero marginal cost, out of history | B1, B2, B3 | `loop.side_query` |
| Instant warm-state resume, no re-prefill | B1, B2 (+ `SstateDiskCache` wired) | `session.durable_transcript`, `session.resume_picker` |
| Atomic, instantaneous "restore both", unlimited depth (no 100-cap) | B1, B2 (RWKV) / + B4 (transformer/hybrid) | `session.checkpoint_rewind` |
| State-fork best-of-N as a structural advantage over metered parallelism | B1, B2, B3 | `subagents.fork_worker`, `perm.plan_mode`, `teams.coordinated` |
| Fork a warm Qwen3-Coder-Next session | B4, B10 | ABI section 10 (Hybrid capsule) |
| First-try-valid, streamed tool calls, faster over the wire | B5 | `mcp.client_host_server`, tool T7/T8 |
| Warm repeated-turn TTFT elimination | B6 (+ B1/B2 capsule) | `cost.usage_transparency` (perf telemetry) |
| Small exact cited working set beats stuffed context | B8 (+ `hawking-index`) | `config.claude_md`, `cost.usage_transparency` |
| Local safety classifier as a warm fork, egress-off by default | B3 (+ `hide-security` OS enforcement) | `perm.auto_mode`, `security.sandbox` |
| Cheap-model-first routing without losing success@time | B11 | `sdk.headless`, product routing |
| Run a bigger model resident via sub-4-bit serving | B7 | `HIDE_LOCAL_MODEL_TOPOLOGY.md` |

## 7. Deferred and out of scope for this menu

These are real levers but downstream of the spine above; they enter the menu only after their prerequisites clear (sizing in `HIDE_PRIORITIZED_BUILD_LADDER.md`):

- **Large agent fleets / durable DAG** (dossier section 9 "large agent fleets"): the packed `hide-fleet` is REDESIGN (heavy, not HTTP-reachable); gated behind B3 proving best-of-N pays, plus single-writer/isolation from `HIDE_SECURITY_CONSTITUTION.md`. Kill when duplicate work, conflicts, or thermal contention dominate.
- **Async tool futures** (overlap decode and tool execution, dossier section 9): gated behind B5 and safe concurrent reads; kill when dependency errors or wasted work exceed critical-path gain.
- **Suffix / file-as-draft speculation** (dossier section 9): a localized-patch speed lane gated behind B5's lossless-verify path; kill when rewrite-heavy tasks collapse to baseline or memory cost is excessive.
- **Fused quantized KV/state** (longer local context, dossier section 9): gated behind B4; reproduce reference-quality on one model before adopting any compressed-KV result (dossier 5.5); kill when the coding/recall delta exceeds the floor or the kernel win is not end-to-end.
- **Warm-state disk persistence** (`SstateDiskCache`, built + tested, zero callers, `cache/sstate_disk.rs`): a small wire behind B2's `save`/`load` for instant cross-restart resume; folded into the resume claim above rather than run as its own bet.
