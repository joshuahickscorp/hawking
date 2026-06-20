# Paradigm plan final audit — completeness table

> **Generated:** 2026-06-02. Read-only audit; no code changed.
> **HEAD:** `c78d7dc` on branch `paradigm/exec`.
> **11 session commits** in this paradigm build. Golden hash (default decode) = `b480cc10faf9a8ec`.
> **Source authority:** `plans/paradigm_execution_plan.md` (the plan); `reports/paradigm_execution_log.md` (the running ledger); `reports/dead_levers.md` (the kill-ledger); `reports/phase_4_4_cross_vendor_scope.md` (Wave-2 scope doc); `paradigmshift.md` Part V.1/V.4.

---

## Legend

| Symbol | Meaning |
|---|---|
| DONE | Step completed, parity/ship gate passed, committed. |
| DONE-DOC | Step's output is a kill-ledger entry or design document; no code change required. |
| DONE-PARTIAL | The part of the step that is achievable without a blocked dependency is done; a defined residual is deferred. |
| KILLED | Step died by Kill Protocol; Type noted; entry in `reports/dead_levers.md`. |
| LEFT | Neither done, killed, nor explicitly in-flight — a genuine gap. |
| IN-FLIGHT | Defined deliverable with a complete validated draft; not yet committed. |

---

## PHASE 0 — Instrument & baseline

| Step | Status | Commit / pointer | Notes |
|---|---|---|---|
| **0.1** Paired baseline + noise floor | **DONE** | `paradigm/exec@64edfbd` (harness calibration run, no commit — harness output only) | Self-vs-self noise floor measured: ~3.0% second-position bias + ±5% scatter. Gating rule adopted. `reports/bench/phase0_noise_floor.json`. |
| **0.1** Clean-room absolute queued | **DONE** | User clean-room run 2026-06-01 (outside git; results in log §0 setup) | Results: 29.96 tps clean, 0.2021 J/tok, llama gap = 1.63×. |
| **0.2** Gap anatomy / profile | **DONE** | `paradigm/exec@64edfbd` (gpu_prod trace, no commit — analysis only) | 86.7% GPU = predec GEMVs @~52% peak; 616 disp/tok (324 trivial); 1.55× llama gap. Sets Phase-2 order. Scout specs persisted to `reports/scout_phase_*.md`. |
| **0.3** Per-phase energy attribution | **DONE** | `c78d7dc` (tooling fix + new `tools/bench/phase_joules.sh`) | `measure_joules.sh` macmon-leak fixed. `phase_joules.sh` adds per-phase J/tok aligned to TCB boundaries. Proxy (energy ≈ GPU-time) recorded in ledger; clean-room J/tok queued. |
| **0.4** Moat regression guards | **DONE** | `paradigm/exec@64edfbd` (no new code — existing tests confirmed green) | `prefix_cache_parity.rs` 3/3 ✓; `user_draft_parity_e2e.rs` 6/6 ✓ (237 s). Eagle5 slow path (NO-GO path) >60s/test → deferred per plan's timebox. Guards wired into `cargo test`. |

**PHASE 0 verdict: COMPLETE.** All four steps resolved, moats green, harness calibrated, baseline anchored.

---

## PHASE 1 — Bank the free wins

| Step | Status | Commit / pointer | Notes |
|---|---|---|---|
| **1.1** GPU greedy sampling default | **DONE** | `e99ed7f` | Flipped `use_tcb` gate: greedy (temp=0) now defaults to the full-Metal TCB path. Bit-identical: batch-hash byte-identical 5×64tok; `v1e_gpu_argmax_parity` 3/3; `integration_greedy_64` + moat guards green. Paired: old-default ~0.4 → new-default ~25–30 dec_tps (no regression; large default-user win). CLEAN-ROOM TODO queued. |
| **1.2** `--profile fast` (f16-scales + lever bundle) | **DONE** | `8af136e` | Named profile bundles: vocab-prune-32k + Q4K-LM-head + Q4K-FFN-down + predec + f16-scales. f16-scales is the one genuinely new lever. Kernel parity green (rel_L2 < 1e-2); 4/5 prompts byte-identical at 64 tok; paired +7.4% (contaminated; within prior +6–9%). Opt-in, not default — raw default stays bit-identical. Quality gate (logit-cosine ≥ 0.999 / PPL+0.05) not yet run — queued as oracle. |
| 1.2 — sub-steps: vocab-prune | **DONE-PARTIAL** (pre-existing) | Pre-existing `DISMANTLE_QWEN_VOCAB_PRUNE=32000` | Already in the locked bench fast-path at branch base; marginal delta ≈ 0 vs the 1.2 bundle. Included in `--profile fast`. |
| 1.2 — sub-steps: Q4K-LM-head | **DONE-PARTIAL** (pre-existing) | Pre-existing `DISMANTLE_QWEN_Q4K_LMHEAD=1` | Same: already in fast-path; bundled. |
| 1.2 — sub-steps: Q4K-FFN-down | **DONE-PARTIAL** (pre-existing) | Pre-existing `DISMANTLE_QWEN_FFN_DOWN_Q4K=1` | Plan flags spec-accept-lowering (~7→3). Already in moat-bench baseline; spec interaction measured pre-branch. Bundled. |
| 1.2 — precise quality gate (logit-cosine / PPL oracle) | **LEFT** | No commit | The plan requires logit-cosine ≥ ~0.999 AND PPL within ~+0.05. No turnkey harness exists; it has not been run. The lever is shipped opt-in on kernel rel_L2 + token-divergence bounds, with the formal quality oracle explicitly queued. This is the one real gap in Phase 1. |

**PHASE 1 verdict: EFFECTIVELY COMPLETE** with one documented gap: the formal f16-scales quality oracle (logit-cosine/PPL gate) is queued but unrun. The lever is opt-in, so this gap does not break the default path.

---

## PHASE 2 — Throughput structural wins

Note: Phase 2's order was reshaped by the 0.2 gap anatomy and the Phase 1→2 boundary reconciliation. The plan's default expected order (2.1 → 2.2 → 2.3) changed to: GEMV-efficiency (dead) → 2.2 fusion → 2.1-a f16-KV → 2.3 flash. The GEMV-efficiency lead recommended by 0.2 was confirmed structurally dead by the A5/A6/A10 reconciliation (`c54d299`).

| Step | Status | Commit / pointer | Notes |
|---|---|---|---|
| **2.1** f16 activations + f16/Q8 KV cache — activation (x_buf) half | **KILLED** (Type-1) | `dead_levers.md` — "f16 residual stream" entry; `dead_levers.md` "Decode-kernel micro-opt: vectorized uint4 / A5" cross-ref | f16 residual stream killed 2026-05-11 (accumulated error after 27 layers). Plan 2.1 calls for f16 activations through the decode path. `x_buf` stays f32 by structural necessity (f16 residual is Type-1). The achievable portion of 2.1 is the KV cache half (2.1-a). |
| **2.1-a** f16 KV cache — kernels | **DONE** | `ed6925e` | `mha_decode_f16kv` (single) + `mha_decode_f16kv_batched` + `memcpy_f32_to_f16_off` + 3 TCB wrappers + `tests/mha_decode_f16kv_parity.rs`. Parity 8/8 atol=1e-3 incl seq=2048. Unreachable from default path (no dispatch yet) → bit-identical. |
| **2.1-a** f16 KV cache — dispatch (arena + qwen_dense wiring) | **DONE** | `1fa6941` | Arena gains `k/v_cache_f16_buf` + `ensure_f16_kv` + `kv_f16_layer_byte_offset`. qwen_dense dispatch: `if f16_kv → mha_decode_f16kv[_batched]` (HALF-stride) else if flash else f32. Early-ensure in both fresh_arena blocks. Flag-OFF bit-identical (golden `b480cc10` + batch-hash byte-identical). Flag-ON coherent (1/8 batch-hash divergence = expected small f16 perturbation). Mutually exclusive w/ W4A8 + flash. CLEAN-ROOM TODO: long-ctx tps + J/tok + logit-cosine/PPL via `DISMANTLE_QWEN_F16_KV=1`. |
| **2.2** Reduce dispatch count / fuse kernels | **KILLED** (Type-1 dead-for-tps) | `2b5379d` (kill-ledger entry) | All three fusion forms evaluated: rope-q+rope-k = host-per-dispatch-overhead Type-1; add/rope epilogue fold = A10 FMA-recontraction 1-ULP bit-identity trap; KV-append memcpy elision (KV_DIRECT) = bit-identical but 0.83% GPU-time ≤ ±3% noise floor. 86.7% GEMV-bound; trivial ops ~9% GPU. Short-ctx dense tps structurally tapped. KV_DIRECT draft in `reports/wave2_result.json`, not landed (below gate). |
| **2.3** Flash-style decode attention | **DONE** (build-and-hold; default-off) | `1769de2` | `mha_decode_flash_f32`: online-softmax GQA decode; no score materialization; lifts the ~7800-tok cap. Parity 4/4 atol=1e-3 + rtol=1e-4 incl 4K. Default-off → bit-identical (golden + batch-hash). Short-ctx attention = 2.92% → no headline tps gain; payoff = long-context reach + cap removal. Wiring default-on at long context requires a long-context paired bench (queued clean-room). |

**PHASE 2 verdict:** 2.1-a f16-KV is end-to-end (kernels + dispatch, parity-verified, default-off, energy lever ready for clean-room measure). 2.2 is cleanly killed. 2.3 is a capability spike (default-off, correct). The f16 activations arm of 2.1 is Type-1 dead (f16 residual kills quality). **No genuine gaps** — all outcomes are recorded DONE, KILLED, or documented build-and-hold.

---

## PHASE 3 — Backend seam

| Step | Status | Commit / pointer | Notes |
|---|---|---|---|
| **3.1** Trait defs (`Backend` / op-traits / `ComputeBackend` / `Op` enum) | **DONE** | `728ab6d` | `backend/mod.rs`: `Backend` base + 10 op-traits + `Op` enum + `ComputeBackend` blanket-impl + `GemvSpec`/`WeightKind` + AWQ-scaled q8 norm verb. Defs only, no impl, no decode wiring → bit-identical (108 lib tests green). E0391 supertrait cycle fixed by base/bundle split. |
| **3.1** `MetalBackend` concrete impl (`backend/metal.rs`) | **DONE** | `da8da67` | 608 lines; implements all ~11 op-traits as thin wrappers over `kernels::*_tcb`; `type Buffer = PinnedBuffer`; GAT `Recorder<'a>` wraps `TokenCommandBuffer`. macOS-gated `mod metal;`. Dead code (not wired into any decode call-site) → bit-identical, 108 lib tests green. |
| **3.1** Per-family call-site routing (`DISMANTLE_BACKEND_SEAM`) | **IN-FLIGHT** | `reports/phase_3_1_seam_routing_checklist.md` + draft in `reports/wave3_result.json` | A routing checklist exists with adversarial corrections (Step-6 label inversion, batch range). The draft was flagged do-not-apply-as-drafted (depends on a `recorder_borrowing` pattern not in tree). Must be done one op-family at a time, each golden-hash-gated. This is the only uncommitted structural piece of 3.1. |
| **3.2** Op scheduler + CPU fallback (`backend/router.rs` + decode hook) | **IN-FLIGHT** | `reports/phase_3_2_decode_hook_spec.md` + `reports/wave3_result.json` (router-files stream) | `backend/router.rs` + `cpu_fallback_parity.rs` drafts exist. The `flush_and_reset` lifetime bug is identified (needs `ctx: &'a MetalContext` param; test needs `Engine` import). The `DISMANTLE_FORCE_CPU_OP` decode hook spec is `reports/phase_3_2_decode_hook_spec.md`. Router + hook + test must be done as one unit. |
| **3.3** CPU backend (dense Qwen reach path) | **DONE** | `2f47141` | `EngineConfig.force_cpu` + `DISMANTLE_FORCE_CPU=1`. `metal_ctx.is_some()` guards. New `tests/cpu_backend_parity.rs`: CPU vs Metal on qwen0.5b = 12/12 tokens identical. Metal path bit-identical (batch-hash). 108 lib tests green. |
| **3.3** CPU backend — MoE reach (off-macOS DeepSeek/Mixtral) | **LEFT** | `reports/phase_3_3_cpu_moe_scope.md` | Dense CPU path done. MoE (DeepSeek-V2) hard-errors off-macOS at the MLA attention branch (`mla_decode: Metal context unavailable`). Scope doc identifies fix options (suppress `mla_metal=true` off-macOS, or add CPU MLA path). No code written. |
| **3.3** Off-macOS build verification (non-macOS toolchain) | **LEFT** | — | `cargo check --target aarch64-unknown-linux-*` was explicitly queued for the user (needs a non-macOS toolchain); not run or verified in this session. The macOS guard lift (`cfg(target_os="macos")` in `Cargo.toml:27`) has not been touched. |

**PHASE 3 verdict:** Seam trait defs and MetalBackend impl are done. CPU dense reach path is done. The three open items are: (a) per-family call-site routing (draft exists, needs careful incremental apply); (b) the op-scheduler router + decode hook (draft exists, needs lifetime fix); (c) CPU MoE reach and off-macOS build verification. Items (a) and (b) are the `ggml_backend_sched` analogue that makes partial backends shippable — they are genuine gaps in Phase 3's portability goal.

---

## PHASE 4 — High-ceiling bets

| Step | Status | Commit / pointer | Notes |
|---|---|---|---|
| **4.1** QTIP-on-Metal trellis decode spike (decode cost gate) | **KILLED** (Type-1 by proxy) | `3ec51f7` (kill-ledger entry) | Clean Q3 proxy (36.3 GB/s = 24.2% of 150 GB/s peak; Q3 kernels 38–46% slower in µs than Q4_K predec despite ~half the bytes). Trellis re-adds the per-element ALU predec removed + adds a serial state dependence (`state[i]←state[i-1]`) with no Q4_K analog → same compute-bound wall. Direct `gemm_qtip_trellis_v1` kernel was deliberately NOT built (cheapest honest outcome). Type-2 reframe named (lane-independent sub-block + fused predec-of-seeds) with its cheap oracle in-hand. Closes the sub-Q4 byte-cut axis for decode speed. |
| **4.2** Custom on-disk format (DWA) | **NOT STARTED** (gated on 4.1) | — | 4.1 is KILLED → 4.2 is moot unless the 4.1 Type-2 reframe clears its oracle. Per plan: gated on 4.1. Correctly not started. |
| **4.3** 2–2.5-bit QTIP model | **NOT STARTED** (gated on 4.1/4.2) | — | Gated on 4.1 and 4.2. 4.1 Type-1 killed. QTIP 3-bit quality is leaning NO-GO (weight bracket `bits_needed=[+1.37,+0.44]`, both positive; decisive codec run not completed — `ALLOW_FRESH_QTIP_CODEC=False`). Type-2 (quality arm) is still open: the Colab decisive run (`05_combined_quality_gates.ipynb` with `ALLOW_FRESH_QTIP_CODEC=True`) is queued. However, a quality GO here is necessary-but-not-sufficient (the decode Type-2 oracle must also clear first). |
| **4.4** Cross-vendor GPU backend (CubeCL / WGPU) | **LEFT** (design-only; no code) | `reports/phase_4_4_cross_vendor_scope.md` | Scope doc delivers: the landed seam interface each backend must satisfy; CubeCL vs WGPU comparison (maturity, effort, Cargo.toml deps); recommended order (WGPU Rung 1 first); risks. No Cargo.toml dep added; user approval required per CLAUDE.md. No WgpuBackend or CubeclBackend code written. Prerequisites: 3.1 per-family routing + 3.2 router must land first. |

**PHASE 4 verdict:** 4.1 cleanly killed by protocol. 4.2 and 4.3 correctly not started (gated). 4.4 is a genuine gap — design is done but no implementation has begun, and its prerequisites (3.1 routing, 3.2 router) are themselves IN-FLIGHT.

---

## Energy / V.4 work (paradigmshift.md Part V.4)

| Item | Status | Commit / pointer | Notes |
|---|---|---|---|
| `measure_joules.sh` macmon-leak fix | **DONE** | `c78d7dc` | `pkill -P $sampler_pid` reaps the macmon pipe child; sampler exits cleanly and prints J/tok. |
| `phase_joules.sh` per-phase J/tok attribution | **DONE** | `c78d7dc` | New 344-line script. Aligns GPU-power samples to decode TCB phase boundaries (GEMV vs attention vs trivial-ops). The differentiated V.4 north-star instrument. User runs clean-room. |
| Clean-room per-phase J/tok measurement | **LEFT** | — | The instrument exists; the actual per-phase J/tok numbers have not been captured with Claude quit. Queued. |
| DVFS / race-to-idle probe | **LEFT** | — | Plan 0.3 + paradigmshift V.4 list testing GPU clock steerability. Not probed; V.4 research returned empty (no verified public literature). |
| ANE energy probe | **LEFT** | — | Flagged unverified in paradigmshift V.4. Silicon audit #6 (ANE/CoreML FFN) killed as ~4–7× slower than GPU Q4_K, but the *energy* question is separate. Not probed. |

**V.4 verdict:** Tooling is done (leak fix + per-phase script). The measurement itself + the deeper energy probes are genuine gaps, though the plan explicitly timeboxes this work and the proxy (energy ≈ GPU-time) is sufficient to gate levers.

---

## Consolidated gap table — things in the plan that are neither DONE nor KILLED nor in-flight

| Gap | Blocking what | Action required |
|---|---|---|
| **f16-scales formal quality gate** (logit-cosine ≥ 0.999 / PPL+0.05 oracle) | 1.2 ship gate technically open | Run `--profile fast` on a PPL/logit harness (same pattern as W4A8 quality; no turnkey harness exists yet). Low urgency — lever is already opt-in. |
| **3.1 per-family call-site routing** (`DISMANTLE_BACKEND_SEAM`) | 3.2 router; 4.4 cross-vendor backend | Apply the routing checklist one op-family at a time, each golden-hash-gated. Draft exists in `reports/wave3_result.json` but must be re-anchored (recorder_borrowing pattern issue). |
| **3.2 op-scheduler router + decode hook** | 4.4 cross-vendor; the `ggml_backend_sched` partial-backend-ship lesson | Fix the `flush_and_reset` lifetime bug in `backend/router.rs`; wire `DISMANTLE_FORCE_CPU_OP` hook per `phase_3_2_decode_hook_spec.md`; add `cpu_fallback_parity.rs` test with `rmsnorm_f32` target. |
| **3.3 CPU MoE reach** (off-macOS DeepSeek-V2 / Mixtral) | Off-macOS full-model run | Suppress `mla_metal=true` off-macOS (option a per `phase_3_3_cpu_moe_scope.md`) or add CPU MLA path. One of two options is scoped in the doc. |
| **3.3 off-macOS build verification** | Confirming the portability claim | User runs `cargo check --target aarch64-unknown-linux-gnu` (needs the target toolchain). The macOS hard-gate in `crates/dismantle-core/Cargo.toml:27` has not been lifted. |
| **4.4 WgpuBackend implementation** | Cross-vendor reach (AMD/Intel/external-GPU/Android) | Cargo.toml `wgpu = { version = "0.20", features = ["vulkan", "metal", "dx12"] }` needs user approval first. Then: scaffolding → cheap ops (WGSL) → attention → BackendGemv Q4_K. Scope in `reports/phase_4_4_cross_vendor_scope.md`. |
| **4.3 QTIP quality decisive run** | Closing the QTIP quality Type-2 | Re-run `colab/05_combined_quality_gates.ipynb` with `ALLOW_FRESH_QTIP_CODEC=True`. Expected outcome: decisive Type-1 (weight bracket leans NO-GO). Low urgency while 4.1 decode is Type-1. |
| **Per-phase J/tok clean-room measurement** | V.4 north-star instrument | User runs `tools/bench/phase_joules.sh` with Claude quit. |
| **flash-decode default-on at long context** | Long-context tps headline | Run long-context paired bench; wire `DISMANTLE_QWEN_FLASH_ATTN=1` as default at seq > threshold if CI excludes 0. Currently build-and-hold. |

---

## Clean-room TODO queue (user, Claude quit)

1. `tools/bench/clean_room_batch.sh` — `--profile fast` absolute tps + J/tok (the 1.2 ship claim).
2. `DISMANTLE_QWEN_F16_KV=1` long-ctx tps + J/tok + logit-cosine/PPL (the 2.1-a energy lever).
3. `DISMANTLE_QWEN_FLASH_ATTN=1` long-ctx tps + cap removal verification (the 2.3 capability).
4. `tools/bench/phase_joules.sh` per-phase J/tok attribution (V.4 north-star measurement).
5. `colab/05_combined_quality_gates.ipynb` with `ALLOW_FRESH_QTIP_CODEC=True` (decisive 4.3 quality kill).
6. `cargo check --target aarch64-unknown-linux-gnu` off-macOS build gate (3.3 portability claim).

---

## Moats status

| Moat | Guard test | Status |
|---|---|---|
| Prefix cache bit-identity | `tests/prefix_cache_parity.rs` (3 cases) | GREEN throughout (verified at 0.4, 1.1, and in batch-hash baseline checks) |
| Spec-on-code (n-gram user-draft) | `tests/user_draft_parity_e2e.rs` (6 cases) | GREEN throughout (verified at 0.4) |
| Default golden decode | `tests/integration_greedy_64.rs` + `batch-hash` b480cc10faf9a8ec | GREEN (verified at every shipped commit; bit-identical invariant held throughout all 11 commits) |

---

## Summary scorecard

| Phase | Steps | DONE | KILLED | LEFT / IN-FLIGHT |
|---|---|---|---|---|
| Phase 0 | 4 | 4 | 0 | 0 |
| Phase 1 | 4 (counting sub-steps) | 3 + 1 quality oracle open | 0 | 1 (quality oracle) |
| Phase 2 | 3 | 2 (2.1-a + 2.3 build-and-hold) | 1 (2.2 dead-for-tps; 2.1 activations Type-1) | 0 |
| Phase 3 | 6 items | 3 (trait defs, MetalBackend, CPU dense) | 0 | 3 (routing, router, MoE/off-macOS build) |
| Phase 4 | 4 | 0 | 1 (4.1 Type-1 by proxy) | 3 (4.2/4.3 gated-by-4.1 = correctly not started; 4.4 design-only) |
| V.4 energy | 5 items | 2 (tooling) | 0 | 3 (clean-room measure, DVFS probe, ANE probe) |

**The single most impactful open item is Phase 3's router + per-family routing (3.1/3.2)**, which is the `ggml_backend_sched` lesson: partial backends only become shippable once missing ops route to the CPU fallback. Without the router, the MetalBackend impl and the CPU backend exist but are not connected, and 4.4 cross-vendor cannot start. Everything else (quality oracles, clean-room runs, MoE reach) is either gated, lower-priority, or user-triggered.

