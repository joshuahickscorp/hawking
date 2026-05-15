# Path-to-90 Stage 1 — A4 close: MTLFunctionConstantValues specialization (mla pilot)

**Status:** SHIPPED (+7.8% trimmed-median e2e on top of A5; +16.9% cumulative vs pristine main).
**Branch:** `claude/strange-proskuriakova-b5d48e`
**Base:** a964c3f (A5 close).
**Date:** 2026-05-15

## Result

| Metric | pristine main | A5 | **A4 (mla-only)** | Δ A4 vs A5 | Δ A4 vs main |
|---|---:|---:|---:|---:|---:|
| dec_tps (trimmed median, 5×3 trials) | 20.50 | 22.23 | **23.97** | +7.8% | +16.9% |
| dec_tps (warm-10 median) | 21.20 | 22.69 | **23.98** | +5.7% | +13.1% |
| dec_tps (min, cold-tail) | 4.85 | 12.76 | **21.06** | +65% | +334% |

The cold-tail jump is striking: A4's worst trial (21.06) beats A5's median (22.23 was warm-trimmed; the warm-10 median was 22.69 with a 12.76 cold tail). The specialized pipeline is JIT-compiled once at engine load, so unlike the non-fc kernel it doesn't pay any per-token JIT-warmup cost. This is real beyond the headline number.

**Parity:** bit-identical 3-token greedy decode on all 12 baseline prompts vs both pristine main and A5. All 39 lib tests pass.

## Pilot scope

Pilot specialized exactly **one** kernel — `mla_decode_kernel` (12.10% GPU time per Stage 0 attribution) — into `mla_decode_kernel_fc`. Six model-constant args (n_heads=16, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=512, scale=1/√192) move from per-dispatch buffer args to `[[function_constant(n)]]` declarations. Only `seq_len` remains a runtime arg.

The +7.8% e2e from one 12%-GPU-share kernel is well above the +3% bar; it confirms the function-constant lever works on this codebase. Extending the same pattern to the top MoE kernel (28% GPU) and the rmsnorm-attn kernel (10.5% GPU) is the natural next move — but **deferred** because:
- The top MoE kernels (`moe_batched_gemm_q4_indexed_v2t_gu_v2`, etc.) take `rows`/`cols` that differ per gate-vs-up vs down call (1408×2048 vs 2048×1408). The down-proj `cols=1408` does NOT divide cleanly by 256, so the inner loop bound `cols/256` would silently truncate if naively turned into a function constant. Needs the dispatcher to pick one of two pipelines per (matrix-shape) — possible but non-trivial.
- The rmsnorm_gemv_f16w_attn_pinned_v2t kernel has shape-bound loops but the same dispatcher serves q_proj, kv_a_proj, q_a_proj — three different (rows, cols) pairs. Same multi-variant compilation problem.

These extensions belong to **A4.2**, attended (or a future focused session). The current pilot ships and the infrastructure (`MetalContext::register_specialized_pipeline`) is reusable.

## Mechanism

1. **Shader** (`crates/dismantle-core/shaders/attn.metal`): added a parallel `mla_decode_kernel_fc` whose body is identical to `mla_decode_kernel` except that six args declared as `constant uint& [[buffer(n)]]` are removed in favor of `constant uint kFcN_heads [[function_constant(0)]]` etc. Only `seq_len` stays as a runtime buffer arg.

2. **Rust API** (`crates/dismantle-core/src/metal/mod.rs`): new `MetalContext::register_specialized_pipeline(fn_name, build_fcv)`. Compiles the function with a populated `FunctionConstantValues` and **injects the resulting `ComputePipelineState` into the regular pipeline cache under `fn_name`**. Subsequent `ctx.pipeline(fn_name)` calls — including from inside `TokenCommandBuffer::dispatch_threads` — return the specialized pipeline directly. No special dispatch path needed.

3. **Engine wire-up** (`crates/dismantle-core/src/model/deepseek_v2.rs`): added `mla_use_fc: bool` field gated on `mla_schedule == "metal-mla-fc"`. At engine load, the constants get baked into the pipeline once. Three former call sites of `mla_decode_and_o_proj_arena_tcb` collapse to a single new helper `self.dispatch_mla_decode_and_o_proj(...)` that routes to the fc or non-fc dispatcher based on the flag.

4. **Profile default** (`profiles/deepseek-v2-lite-q4.m3pro18.json`): default `mla_schedule` flipped from `"metal-mla"` → `"metal-mla-fc"`. The non-fc dispatcher remains in the codebase as a fallback (and is still exercised by the in-tree parity tests that go through `dispatch_gemv_f32` etc.). `shader_hash` updated to the new combined-source hash (the new function definition changed it).

## Why the win is bigger than the per-call CPU savings

The fc variant saves 6 `set_bytes` calls per dispatch — ~12 µs CPU/dispatch × 11.8 dispatches/token = ~140 µs/tok of CPU work. That's only ~0.7 ms/tok = ~1.5% e2e on its own. The actual +5.7-7.8% e2e win means the **GPU side benefitted** too, presumably from:
- Loop bounds known at compile time → MSL compiler unrolls Phase 0/1/3/4 inner loops
- `kFcQkNopeHeadDim * kFcKvLoraRank` etc folded to a constant → no per-call multiplication
- The dead-code branch `if (gid >= kFcN_heads) return;` reducible to a single immediate compare
- Pipeline-state cache hot path: the specialized pipeline lives at the same lookup key, so the existing `ctx.pipeline("mla_decode_kernel_fc")` cache hit pulls the optimized version with no per-dispatch lookup branching

The cold-trial reduction (12.76 → 21.06 dec_tps minimum) is the clearest signal: pipeline JIT is amortized to engine-load time, so first-token decode is no longer paying for it.

## Stage 1 cumulative progress

| Stage | dec_tps (trimmed-median) | Δ vs main | Δ vs llama.cpp |
|---|---:|---:|---:|
| pristine main (v2.2.0) | 20.50 | — | 0.39× |
| A5 (arena) | 22.23 | +8.4% | 0.42× |
| A5 + **A4 (mla-fc pilot)** | **23.97** | **+16.9%** | **0.46×** |
| llama.cpp comparator | 52.51 | +156% | 1.00× |

Closing the gap to llama.cpp from 0.39× → 0.46× in two landings. Engineering ceiling is at 70% practical bandwidth (~70 dec_tps for the engine-work-only stage); spec-decode + KV quant required to break that.

## Next: A1 — wire `flash_attn_decode_kernel`

Per the revised Stage 0 plan ordering, A1 is next. `flash_attn_decode_kernel` exists in `attn.metal` (~L220-370) but is not currently routed. Expected +4-8% e2e on the MLA attention path. The wire-up is structurally similar to A4: profile flag → dispatcher branch → parity gate.

After A1: A3 (residual+RMSNorm cross-layer fusion) → A2 (Q8 KV) → A6 (autotune polish).

## Files changed

- [crates/dismantle-core/shaders/attn.metal](../../../crates/dismantle-core/shaders/attn.metal) — added `mla_decode_kernel_fc`
- [crates/dismantle-core/src/metal/mod.rs](../../../crates/dismantle-core/src/metal/mod.rs) — added `register_specialized_pipeline`, `FunctionConstantValues` import, trace name for `mla_decode_kernel_fc`
- [crates/dismantle-core/src/kernels/mod.rs](../../../crates/dismantle-core/src/kernels/mod.rs) — added `mla_decode_and_o_proj_arena_fc_tcb` dispatcher
- [crates/dismantle-core/src/model/deepseek_v2.rs](../../../crates/dismantle-core/src/model/deepseek_v2.rs) — added `mla_use_fc` engine field, fcv registration at engine load, helper `dispatch_mla_decode_and_o_proj`; replaced 3 inline call sites with the helper
- [profiles/deepseek-v2-lite-q4.m3pro18.json](../../../profiles/deepseek-v2-lite-q4.m3pro18.json) — default `mla_schedule = "metal-mla-fc"`, new `shader_hash`
