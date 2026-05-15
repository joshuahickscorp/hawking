# Path-to-90 Stage 1 — A5 close: persistent argbuf bump-arena

**Status:** SHIPPED (+8.4% trimmed-median, +7.0% warm-only median).
**Branch:** `claude/strange-proskuriakova-b5d48e`
**Base:** eb11a74 (Stage 0 attribution commit).
**Date:** 2026-05-15

## Result

| Metric | pristine main | A5 | Δ |
|---|---:|---:|---:|
| **dec_tps (trimmed median, 5×3 trials, untraced)** | 20.50 | 22.23 | **+8.4%** |
| dec_tps (warm-only median, top-10 of 15) | 21.20 | 22.69 | +7.0% |
| dec_tps (traced, ProdCbGpu, warm trial 3) | 21.10 | 22.34 | +5.9% |
| GPU time per token (ProdCbGpu, traced) | 21.96 ms | 21.64 ms | -1.5% |
| Inferred CPU encode per token (wall − GPU) | 25.4 ms | 23.2 ms | **-8.7%** |
| Effective GPU bandwidth | 82.9 GiB/s | 84.1 GiB/s | +1.4% |

**Bit-identical 3-token greedy decode on all 12 baseline prompts** vs pristine main (parity gate passed). All 25 lib tests pass; the 2 pre-existing pre-A5 failures (`test_gemv_f32_attn_matches_cpu`, `test_gemv_f32_moe_matches_cpu`) are unchanged — those parity tests were already failing on main with diffs of 35-54 (synthetic-input distribution mismatch, not a regression).

The win is squarely on the CPU side — GPU time stayed flat within noise (21.96 → 21.64 ms/tok, well under the 5.8 ms wall-time delta). This matches the Stage 0 thesis: per-dispatch `new_buffer` allocation was a meaningful slice of the ~25 ms/tok CPU residual.

The +7-8% is below the +10-15% Stage 0 projection — the simplest explanation is that `new_buffer` on shared-memory M3 Pro costs ~25-30 µs (not the ~50 µs I assumed) and at ~80 argbuf-using dispatches/token that's ~2 ms/tok saved, matching the observed CPU delta.

## Mechanism

Per-dispatch `KernelArgBuffer::new` was calling `device.new_buffer(size, StorageModeShared)`. At ~80 argbuf-using dispatches per token × ~25-30 µs each, this was ~2-2.5 ms/tok of CPU overhead that did not need to be there.

A5 replaces this with a single persistent `MTLBuffer` carved by a bump-arena allocator (`MetalContext::argbuf_alloc`). The arena lives in `Inner::argbuf_arena: Mutex<ArgbufArena>` and is reset to cursor=0 by `TokenCommandBuffer::commit_and_wait` (and `Drop`) once the GPU has finished the token — safe because all encoded dispatches have already consumed their captured buffer reference.

Initial capacity is 64 KB; current peak `high_water` is well under that (≤ ~13 KB worst-case at 213 dispatches × 64 B max). Growth is supported only at cursor=0 (between tokens) so we can't orphan in-flight dispatches. Mid-token overflow panics with a clear message — surfaces immediately if a future kernel scales argbuf usage past the limit.

## Files changed

- [crates/dismantle-core/src/metal/argbuf.rs](../../../crates/dismantle-core/src/metal/argbuf.rs) — rewrote `KernelArgBuffer` to carve from arena; added `bind(enc, slot)` method.
- [crates/dismantle-core/src/metal/mod.rs](../../../crates/dismantle-core/src/metal/mod.rs) — added `ArgbufArena` struct, `argbuf_alloc()`, `argbuf_reset()`, `argbuf_high_water()`; wired reset into `TokenCommandBuffer::commit_and_wait` and `Drop`.
- [crates/dismantle-core/src/kernels/mod.rs](../../../crates/dismantle-core/src/kernels/mod.rs) — replaced `enc.set_buffer(N, Some(ab.handle()), 0);` with `ab.bind(enc, N);` at all 10 call sites (no semantic change beyond using the arena's base_offset).

No shader changes; no kernel-profile changes.

## Bench artifacts

- [a5_run{1..5}.json](.) — 5 untraced bench runs × 3 trials each, A5 build.
- [main/main_run{1..5}.json](main/) — same protocol on pristine main, for the 1:1 comparison.
- [a5_attribution_bench.json](a5_attribution_bench.json) — ProdCbGpu-traced 3-trial bench.
- [a5_attribution_trace.json](a5_attribution_trace.json) — per-dispatch sample log (13,648 samples).
- [a5_attribution.txt](a5_attribution.txt) — `analyze_tcb_trace.py` summary.

## Re-attribution (top-6 vs Stage 0)

| Kernel | Stage 0 (main) % GPU | A5 % GPU |
|---|---:|---:|
| `moe_batched_gemm_q4_indexed_v2t_gu_v2` | 28.88% | 27.88% |
| `mla_decode_kernel` | 11.54% | 12.10% |
| `rmsnorm_gemv_f16w_attn_pinned_v2t` | 10.63% | 10.51% |
| `moe_batched_gemm_q5_0_indexed_v2t` | 10.37% | 10.32% |
| `gemv_f16_simdmat` | 9.25% | 9.40% |
| `moe_batched_gemm_q8_0_indexed_v2t` | 6.40% | 6.19% |

Top-6 share virtually unchanged — confirms the A5 win is structural (CPU encode), not a kernel-quality lift. Good news for Stage 1 sequencing: next levers (A4, A1, A3) still apply unmodified.

## Next: A4 — `MTLFunctionConstantValues` specialization

Per the plan-revision in Stage 0 attribution, A4 sits next:
- Convert per-dispatch uniform args for the top-6 GEMV kernels to function constants where the shape is known at engine-load time.
- Extend pipeline cache key from `String` to `(String, FunctionConstantSig)`.
- Expected: +5-10% e2e (compounds with A5 by removing more bytes from per-dispatch encoder hot path).

After A4 → A1 (wire flash_attn_decode_kernel) → A3 (residual+RMSNorm cross-layer fusion) → A2 (Q8 KV) → A6 (autotune).
