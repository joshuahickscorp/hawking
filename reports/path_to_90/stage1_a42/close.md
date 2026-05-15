# Path-to-90 Stage 1 — A4.2 close: MoE Q4 routed fc — REJECTED at +3% gate

**Status:** HALTED. Infrastructure landed as opt-in `gemm_q4_k_schedule = "v2t_gu_v2_fc"`; default unchanged.
**Branch:** `claude/strange-proskuriakova-b5d48e`
**Base:** 918c93c (A1 close).
**Date:** 2026-05-15

## Result

| Profile | dec_tps (trimmed-median) | Δ vs A4 baseline |
|---|---:|---:|
| A4 (mla-fc; default) | 23.97 | — |
| **A4.2 (mla-fc + moe_q4_gu_v2_fc)** | **21.30** | **−11.1%** |
| warm-10 median: A4 / A4.2 | 23.98 / 21.48 | −10.4% |
| max: A4 / A4.2 | 24.07 / 22.13 | −8.0% |
| min: A4 / A4.2 | 21.06 / 15.74 | −25.3% |

All buckets regress. Below the plan's +3% reject threshold. Halted; default profile unchanged.

**Parity:** bit-identical 3-token greedy decode on all 12 baseline prompts after the shared-expert fix (see "First-try bug" below).

## Why MoE-fc lost

Two structural differences from the MLA pilot make the function-constant pattern net-negative on this kernel:

1. **The inner loops are already small.** `moe_batched_gemm_q4_indexed_v2t_gu_v2` has its hot loops over `blocks_per_row = cols/256 = 8` and four sub-block iterations of 4 + 4 elements. The MSL compiler already auto-unrolls these (they're trivially small) and constant-propagates everything reachable from a `constant uint&` buffer arg. Burning `rows` and `cols` into function constants gave the compiler nothing it didn't already have.

2. **Register pressure flipped occupancy.** With model constants now compile-time literals, the compiler made different allocation choices for the per-thread `sg[8]`, `mg[8]`, `dsg[8]`, `dmu[8]`, `xl[8]` register arrays. The non-fc kernel happens to land at a register count that hits 100% occupancy on M3 Pro's Apple9 GPU family; the fc variant slips into a register count that triggers spilling or lower occupancy. The ~11% e2e regression with no other meaningful change in the trace is consistent with this.

The Stage 0 attribution gave A4.2 a misleadingly high prior because the MoE Q4 kernel is 28% of GPU. The win-per-percent-GPU model doesn't account for: (a) whether the compiler had already extracted the wins, and (b) whether the optimization helps register allocation or hurts it. **MLA decode was the right kernel for the fc pattern — many bounded loops, big inner work; MoE Q4 was the wrong kernel — small fixed loops, already auto-unrolled.**

## First-try bug (worth recording)

Initial parity run failed 9 of 12 prompts with completely different hashes. Root cause: the kernel hardcoded `kFcMoeRows = moe_intermediate (1408)` via function constants, but the SHARED-expert call site dispatches the same kernel with `rows = shared_mid = n_shared_experts × moe_intermediate (2816 for V2-Lite)`. The hardcoded value silently truncated the shared kernel's work to 1408 rows.

Fix: routed expert path (rows=1408) → fc kernel; shared expert path (rows=2816) → fall back to the non-fc kernel even under `v2t_gu_v2_fc`. Then bit-identical 12/12. The lesson: function constants only work cleanly when the call-site shape is genuinely unique. Mixed-shape callers need either a runtime arg (no specialization) or N specialized pipelines (one per shape).

## What stays in tree

- `moe_batched_gemm_q4_indexed_v2t_gu_v2_fc` shader — kept; it's correct and parity-tested.
- `encode_batched_gemv_fused_gu_v2_fc_tcb` dispatcher — kept.
- `gemm_q4_k_schedule = "v2t_gu_v2_fc"` profile value — kept as opt-in; routes routed-expert path to fc, shared-expert path falls back to non-fc.
- Default profile (`gemm_q4_k_schedule = "v2t_gu_v2"`) unchanged.
- `shader_hash` updated to `79d8d37838383a564382e1c9` (the new fc-kernel shader-source variant). The non-fc kernel is unchanged and remains the default.

Future revival path: investigate occupancy via Metal Profiler. If the fc kernel's register count can be brought back into the high-occupancy band (e.g., by tagging vector arrays as `thread` instead of letting the allocator pick, or by hand-rolling per-block scale arrays as `uchar4` instead of `uchar[8]`), the function-constant compile-time arithmetic might still net win. Skip unless re-attribution post-Stage-2/3 shows MoE Q4 as the gating kernel.

## Stage 1 cumulative (unchanged)

| Stage | dec_tps | Δ vs main | Δ vs llama.cpp |
|---|---:|---:|---:|
| pristine main (v2.2.0) | 20.50 | — | 0.39× |
| A5 (arena) | 22.23 | +8.4% | 0.42× |
| **A5 + A4 (mla-fc)** | **23.97** | **+16.9%** | **0.46×** |
| A1 (mla-flash) — rejected | 20.65 | +0.7% | 0.39× |
| A4.2 (moe-fc routed) — rejected | 21.30 | +3.9% | 0.41× |

Two rejections in a row from the Stage 0 plan's prioritized list. The negative results aren't a Stage 1 failure — they're calibration on which levers actually pay. With dispatch overhead largely cleared by A5 (the real Stage-0 finding), the remaining engine wins come from kernels whose inner loops are big enough to amortize compile-time specialization. A4 (MLA) was that; A1/A4.2 weren't.

## Next options

- **A3 — cross-layer residual+RMSNorm fusion.** Eliminates ~27 dispatches/token (one `add_inplace` per layer fuses into the next `rmsnorm_gemv_*` call). Doesn't depend on inner-loop speedups — it's a dispatch-count reduction, building on the same A5 thesis. Estimated +3-6% e2e. Lowest risk remaining lever.
- **Skip ahead to Stage 2 / Stage 3.** Engine work has captured the big wins (+16.9%). Spec-decode (Stage 3) can break the bandwidth ceiling; KV quant (Stage 2 B1/B2) opens long-context headroom. Both require multi-week elapsed time but offer 30-90% gains.
- **Pause and reassess.** Re-attribute via ProdCbGpu on the A4 build and pick the next lever evidence-first rather than from the original plan order.
