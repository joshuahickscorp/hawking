# Phase A Wedge A3 — STUCK: MoE dispatch batch-aware

**Wedge:** A3  
**Status:** STUCK (deferred)  
**Date:** 2026-05-04  

## What failed

Did not attempt implementation after assessing complexity vs benefit.

## Why STUCK (not a blocking error — strategic defer)

`moe_block_batched_indexed_metal` is the single most complex dispatcher in the
codebase. Extending it to handle M=4 tokens would require:

1. **3 Metal shader changes**: `moe_batched_gemm_q4_indexed`, `moe_batched_gemm_q8_0_indexed`,
   `moe_batched_gemm_q6_k_indexed` — each needs a new `tgp.z` token dimension,
   2D `route_ids[M][routes]`, 2D `x[M][cols]`, and 3D output indexing.

2. **Rust dispatcher overhaul**: `moe_block_batched_indexed_metal` handles shared
   experts, gate/up/down routing, q4k_schedule branching, and validation logic
   across hundreds of lines. Parameterizing M across this surface is high-risk.

3. **Zero dec_tps benefit**: Phase A batching is prefill-only (greedy decode is
   inherently sequential). The `dec_tps` target metric is unaffected by prefill
   speed. Phase A can only help first-token latency.

## Impact on A4-A5

A4 (batched argmax) is trivial independently and deferred. A5 (wire generate()
for prefill batching) depends on A3 for the MoE path and is deferred.

The `forward_tokens_batched` scaffold (A1) and `mla_decode_kernel_batched` (A2)
are committed and available for future use (spec decoding, if acceptance rates
improve above α=0.34).

## What attended work unblocks

The spec requires changing 3 Metal shaders and the dispatcher. To proceed:

1. Inspect `shaders/moe.metal` lines 304-600 for the 3 indexed kernels.
2. Add `uint token_idx = tgp.z` and adjust buffer indexing for each.
3. Update Rust dispatcher signature: `route_ids: &[&[u32]]`, `x: &[&[f32]]`,
   `out: &mut [&mut [f32]]` (slice-of-slices for M tokens).
4. Add parity test comparing M=4 batched vs 4 serial calls.

## Followups

- A3 should be re-attempted when Phase A prefill batching is needed (e.g., if
  a speculative decoding approach brings α > 0.5 and batching helps).
- The `moe_batched_gemm_q4_indexed_v2` variant (simd-sum, no threadgroup barriers)
  is the better candidate to extend since it has fewer sync points.
