# Phase 3.2 spec — op scheduler + CPU fallback (scout, 2026-06-01)

> Builds on 3.1's seam (`reports/scout_phase_3_1_backend_seam.md`): one `Backend` supertrait bundling ~11 op-traits, concrete `MetalBackend`, `&mut Recorder` (TCB) threaded through every op. 3.2 adds **per-op routing** so a partial backend (one missing op) ships by falling back to CPU. This is a **PORTABILITY/reach lever, NOT a tps lever** — see caution.

## Confirmed substrate (verified rg/read, 2026-06-01)
- **A CPU reference layer already exists, platform-neutral (NOT macOS-gated):** top of `kernels/mod.rs` — `rmsnorm` (:9, f64 accum), `silu_mul` (:25), `gelu_mul` (:40), `softmax_inplace` (:68), `rope_inplace` (:92), `rope_inplace_scaled` (:129), `rope_inplace_longrope` (:183), `embed_lookup` (:229). The macOS gate starts at `mod metal_dispatch` (:367). **The fallback bodies are mostly already written** — 3.2 routes to them, it does not author them.
- **Unified-memory readback is FREE (charge hypothesis CONFIRMED).** `PinnedBuffer = ::metal::Buffer` (`metal/mod.rs:283`), allocated `StorageModeShared` (:571/:583/:626/:646). The CPU-view pattern `buf.contents() as *const f32` + `std::slice::from_raw_parts` is already pervasive on decode buffers (`qwen_dense.rs:3076`, `:3396` k/v cache, `:4821` x_norm, `:3669` k/v write-back as `*mut f32`). A Metal-produced buffer read by a CPU op needs **no copy/blit** — just `from_raw_parts(buf.contents(), n)`; the write-back is the same pointer mutated in place. The boundary is a pointer cast, not a DMA.
- **CPU→GPU seed (if ever needed):** `ctx.new_buffer_with_bytes` (:575) / in-place `contents() as *mut f32`. Not even needed for in-place ops (rmsnorm/rope mutate the shared buffer the next GPU op already binds).
- **Cleanest first op = rmsnorm** (cheap, isolated, f32→f32, 4.98% GPU, NOT the gemv moat). Decode sites: hoisted layer-0 `rmsnorm_metal_buf_tcb` (`qwen_dense.rs:3793`, dispatch fn `kernels/mod.rs:5244` — clean x_buf→x_norm_buf), in-loop `add_rmsnorm_fused_tcb` (`:4238`/`:4543`, dispatch `:5278`). **Route the un-fused `rmsnorm_metal_buf_tcb` site first** — the fused add+rmsnorm is one kernel and would need the CPU path to also do the residual add (do that second).
- **rope** is the equally-clean alternative (1.01% GPU, `rope_q_f32_inplace_tcb` at `:4109`/`:4119`, CPU `rope_inplace` :92). Use it as the *second* fallback-capable op to prove the router generalizes.

## ⭐ Key design decision: a `BackendOp` capability enum + per-op dispatch, NOT a graph
dismantle has **no IR/graph** (it's imperative `forward_token_greedy_tcb`). Do **not** build a `ggml_backend_sched` graph splitter — that's a mismatch. The right shape:
1. `enum Op { RmsNorm, Rope, Add, SiluMul, Gemv, Attention, … }` (the ~11 verbs from 3.1).
2. `trait Backend { fn supports(&self, op: Op) -> bool; … }` — `MetalBackend::supports` returns `true` for all (it's complete).
3. A thin `Router { primary: MetalBackend, cpu: CpuFallback }` whose per-op methods are: `if primary.supports(op) { primary.rmsnorm(rec, …) } else { rec.sync()?; cpu.rmsnorm(buf.as_slice_mut(), …) }`. **The router IS the scheduler** — one `match`/`supports` check per op, no graph.
4. The CPU fallback for an in-flight TCB op must **flush the recorder first** (`tcb.commit_and_wait` of the work queued so far) so the buffer the CPU reads is current, then CPU-mutate-in-place, then continue recording into a *fresh* TCB. This split-CB is the one real cost (see risk) — acceptable because fallback is the slow path, not the Metal path.

## Routing order (cheapest/safest first; gate after EACH)
1. **Router scaffold + `supports()` + a forced-fallback test hook** (`DISMANTLE_FORCE_CPU_OP=rmsnorm`) — no behavior change when unset. Prove the seam compiles and Metal path is untouched.
2. **rmsnorm fallback** (un-fused site `:3793` → `kernels::rmsnorm`). The flagship test.
3. **rope fallback** (`:4109` → `rope_inplace_scaled`) — proves generalization to a second op + an in-place mutate.
4. (defer) add / silu_mul fallbacks — trivial once 2–3 land; the fused `add_rmsnorm` site needs the CPU path to fold the residual add (note it, don't rush it).
- **Do NOT make gemv/attention fallback-capable in 3.2** — the gemv moat stays Metal-only (3.1 rule); a CPU gemv is 3.3's job and would tank decode if force-routed.

## The forced-fallback parity test (the deliverable test)
Clone the structure of `tests/integration_greedy_64.rs` (golden 64-tok SHA vs `tests/golden/_phase0_token_baseline_64.hashes`) into a new `tests/cpu_fallback_parity.rs`:
- **Leg A (bit-identity, the moat guard):** router with NO forced op → `integration_greedy_64` hash **byte-identical** to the existing golden. Nothing is forced ⇒ pure Metal ⇒ hashes MUST hold.
- **Leg B (atol parity, the fallback correctness):** `DISMANTLE_FORCE_CPU_OP=rmsnorm` → run the decode, compare the **logit vector** (or per-token greedy-64 IDs) against the Metal-only run at **atol=1e-3** (NOT bit-identity — see below). A unit-level mirror of `phase1_kernel_parity.rs:48` (`test_rmsnorm_matches_cpu`, already compares `kernels::rmsnorm` vs `rmsnorm_metal` at `ATOL`) is the cheap inner gate; the end-to-end greedy-64 leg is the integration gate.

## EXACT gates (per plan §3.2 + the parity floors)
- **Metal-only path:** **BIT-IDENTICAL.** `cargo test -p dismantle-core --test integration_greedy_64` (greedy-64 SHA) unchanged + `dismantle batch-hash --tokens 64` byte-identical (b3sum) vs a FRESH HEAD baseline captured before starting. If a hash moves with no op forced → the router indirection itself changed something → STOP.
- **Forced-CPU-fallback path:** **atol=1e-3** (fp16 floor) on logits / first-3-greedy-token match. **It is NOT bit-identical and must not be gated as such** — VERIFIED REASON: GPU `rmsnorm_f32` reduces partials in **f32** (threadgroup tree, `common.metal:102-112`); CPU `kernels::rmsnorm` accumulates in **f64** (`:13-17`). Different reduction precision ⇒ ~1e-6 drift ⇒ correct at atol=1e-3, never byte-equal. (rope: CPU uses `sin_cos`/`powf`, GPU uses Metal `sincos` — same atol story.)
- **Ship gate (3.2 is portability, not perf):** Metal-path bit-identical AND lib tests green AND forced-fallback leg parity-green at atol=1e-3. **No paired-tps gain is required or expected** — the gate is "Metal unchanged + fallback correct," per plan §3.2. (A paired-tps-within-noise check on the *unset* path guards against router overhead; concrete `MetalBackend` monomorphizes so it should be free — add `#[inline]` if a step dips, per 3.1.)

## Kill cross-check (`reports/dead_levers.md`)
Searched the ledger for every adjacent mechanism:
- **"CPU+GPU pipelining" (killed 2026-05-22, Type-1)** and **"Non-GEMM CPU P-cluster offload" (#5, Type-1)**: both kill *running CPU work to GAIN tps* (non-GEMM = 3.2% of wall, on the serial dependency chain, no free slot). **3.2 is DISTINCT and NOT a resurrection:** it does the opposite — it runs an op on CPU as a *correctness fallback for reach* (when a non-Metal backend lacks the op), explicitly accepting a **slowdown**. The kills say "CPU offload won't make decode faster"; 3.2 never claims it will. The Type-1 reality (CPU is slower on this chain) is exactly *why* fallback is the slow path and gemv is excluded. No conflict.
- **"Host-side per-dispatch overhead" family (exhausted 2026-05-24, Type-1)** and **ICB (Type-1)**: warn that splitting/adding command buffers costs ~nothing to gain but also nothing if dispatch count stays bounded. **Relevance:** the forced-fallback split-CB (flush→CPU→fresh-TCB) adds commits *only on the fallback path* — irrelevant to the Metal path (count unchanged → bit-identity holds). Not a kill of 3.2; a confirmation that the Metal path stays clean.
- **No ledger entry covers a *capability-routing seam itself*.** 3.2's mechanism (per-op `supports()` + CPU fallback for portability) is genuinely un-scoped territory — the 3.1 seam is its only precedent, and 3.1 is LIVE (in progress), not dead.
- Conclusion: **no Type-1 kill blocks 3.2.** The adjacent kills constrain the *scope* (don't route gemv/non-GEMM for speed) but do not kill the *portability fallback*, which has a different objective.

## ⚠️ STRATEGIC CAUTION (orchestrator)
- **This is a reach/portability lever — zero tps, by design.** Realistic ceiling: it makes a *future* partial non-Metal backend (3.3 CPU, 4.4 wgpu) shippable with missing ops; on the Metal-only product it is pure overhead that must net to zero (bit-identity gate). Do not let anyone bench it for a throughput delta.
- **Biggest risk: the split-CB flush.** Forcing a mid-token CPU op shatters the single-CB-per-token batching (3.1's load-bearing invariant) for that token. This is *fine for the fallback path* but means the forced-fallback test will show degraded tps — **assert correctness, never tps, on Leg B.** The real trap is if the router indirection leaks into the *unset* path and moves a golden hash; mitigate with concrete `MetalBackend` (no `dyn`) and the bit-identity gate after every routing step.
- **Second risk: scope creep into gemv.** The moat is the predec Q4_K GEMV (86.7% GPU). A "while I'm here, make gemv fallback-capable" is how 3.2 becomes a regression. Hold the line: rmsnorm + rope only; gemv fallback is 3.3.
- **Smallest viable win:** router scaffold + rmsnorm fallback + the two-leg test. That alone proves the `ggml_backend_sched` lesson holds in dismantle's imperative shape and unblocks 3.3. rope is the cheap confirmation that it generalizes.

## Files
- `kernels/mod.rs` — CPU fallback bodies (already present, platform-neutral): `rmsnorm` :9, `rope_inplace_scaled` :129, `silu_mul` :25, `softmax_inplace` :68; GPU TCB dispatch `rmsnorm_metal_buf_tcb` :5244, `add_rmsnorm_fused_tcb` :5278, `rope_q_f32_inplace_tcb` :5673.
- `model/qwen_dense.rs` — decode routing sites: rmsnorm `:3793` (un-fused, route FIRST), fused `:4238`/`:4543`; rope `:4109`/`:4119`; readback pattern (free) `:3076`/`:4821`.
- `metal/mod.rs` — `PinnedBuffer = metal::Buffer` :283/:1324, `StorageModeShared` alloc :571, `new_buffer_with_bytes` :575, non-macOS `PinnedBuffer` stub :1294 (router's `cfg` boundary).
- `shaders/common.metal:93` — GPU `rmsnorm_f32` (f32 reduction; the bit-identity-vs-atol proof).
- `backend/mod.rs` (neutral) + `backend/metal.rs` (macOS-gated) — from 3.1; add `enum Op`, `supports()`, `Router` here.
- NEW `tests/cpu_fallback_parity.rs` (clone `tests/integration_greedy_64.rs` + `phase1_kernel_parity.rs:48`); `tests/golden/_phase0_token_baseline_64.hashes` (the Leg-A invariant).
