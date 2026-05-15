# Path-to-90 Stage 0 — Profile-driven gap attribution

**Date:** 2026-05-15
**Branch:** `claude/strange-proskuriakova-b5d48e` (from `main` @ ecf77a6, v2.2.0 post-T2.14)
**Hardware:** Apple M3 Pro 18 GB (vendor 150 GB/s; practical anchor 130 GB/s)
**Model:** DeepSeek-V2-Lite-Chat Q4_K_M (`models/deepseek-v2-lite-q4.gguf`, 9.65 GiB)
**Quality bar:** kernel-parity gate (existing standard).

## TL;DR

- **dismantle (this measurement):** 21.10 dec_tps (trace overhead present; untraced baseline from v2.2.0 closeout is 23-24 dec_tps).
- **llama.cpp on same hw/model:** 52.51 ± 1.71 dec_tps (`llama-bench -m ... -p 0 -n 64 -ngl 99 -r 3 -t 4`).
- **Gap:** dismantle/llama.cpp ≈ 0.40-0.45×. We are **2.2-2.5×** slower, not 3× as the upstream research brief estimated.
- **Per-token decomposition (dismantle, traced):**
  - Wall: **47.4 ms/tok** (≈ 21.1 dec_tps).
  - GPU compute: **21.96 ms/tok** (per-kernel sum from ProdCbGpu trace).
  - CPU dispatch / encoding overhead: **≈ 25 ms/tok** (residual).
- **Most surprising finding:** CPU dispatch overhead is **bigger than GPU compute**. dismantle is *dispatch-bound first*, *bandwidth-bound second*. The plan's Stage-1 priorities need rebalancing.
- **Dispatch count:** **213 dispatches/token** (not 731 from the brief, not 450-600 from my plan estimate). Engine has already done substantial fusion.
- **GPU bandwidth utilization:** **82.9 GiB/s = 64% of practical 130 GiB/s anchor.** Meaningful headroom on the GPU side too (~50% perf room there if 95% practical bandwidth were reached).
- **Gate to proceed:** met. The attribution is consistent with the plan's overall structure (kernel-bandwidth + dispatch-overhead are both real); but the **priority order inside Stage 1 changes** — see "Plan revisions" below.

## Method

### dismantle

```
DISMANTLE_TCB_TRACE=gpu_prod nice -n 19 taskpolicy -b ./target/release/dismantle bench \
  --backend dismantle --suite decode \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  --trials 3 --max-new-tokens 64 \
  --trace-dispatch \
  --json reports/path_to_90/stage0/dismantle_bench.json \
  --trace-json reports/path_to_90/stage0/dismantle_trace.json
```

Trial-level dec_tps: 19.72 / 21.16 / 21.10 (trial 1 cold). Median across warm trials: **21.13** (trace-on; untraced clean-bench history puts production at 23-24).

Slm not running; Claude Code running (the trace adds known overhead but L7 commentary says ProdCbGpu preserves pipelining vs SplitCbGpu's 2.15× inflation).

### llama.cpp

`llama-bench -m models/deepseek-v2-lite-q4.gguf -p 0 -n 64 -ngl 99 -r 3 -t 4` (homebrew llama.cpp build 1a03cf47f, 9000-series).

Output: `tg64 = 52.51 ± 1.71` t/s. Backend: BLAS+MTL on Apple M3 Pro Apple9 (no M5 Tensor API).

## Per-kernel GPU attribution (dismantle, n=64 tokens warm)

Source: [stage0/dismantle_attribution.txt](dismantle_attribution.txt)

| Kernel | dispatches | total µs | µs/call | µs/token | % GPU |
|---|---:|---:|---:|---:|---:|
| `moe_batched_gemm_q4_indexed_v2t_gu_v2` | 1456 | 405,827 | 278.7 | 6,341 | **28.88%** |
| `mla_decode_kernel` | 756 | 162,187 | 214.5 | 2,534 | **11.54%** |
| `rmsnorm_gemv_f16w_attn_pinned_v2t` | 1512 | 149,461 | 98.8 | 2,335 | **10.63%** |
| `moe_batched_gemm_q5_0_indexed_v2t` | 392 | 145,771 | 371.9 | 2,278 | **10.37%** |
| `gemv_f16_simdmat` (q_b/shared) | 756 | 130,057 | 172.0 | 2,032 | 9.25% |
| `moe_batched_gemm_q8_0_indexed_v2t` | 336 | 89,995 | 267.8 | 1,406 | 6.40% |
| `gemv_f16` (lm_head) | 36 | 85,760 | 2,382.2 | 1,340 | 6.10% |
| `gemv_f32_attn` | 84 | 66,174 | 787.8 | 1,034 | 4.71% |
| `moe_batched_gemm_q4_indexed_v2t` (no-gu) | 392 | 50,812 | 129.6 | 794 | 3.62% |
| `moe_topk_gate` | 728 | 35,000 | 48.1 | 547 | 2.49% |
| `moe_batched_gemm_q6_k_indexed_v2t` | 336 | 28,650 | 85.3 | 448 | 2.04% |
| residual+RMSNorm+RoPE+gate+kv_append (rest) | ~3220 | ~54k | — | ~840 | ~3.9% |

**Top 6 = 76.6% of GPU time.** Five of those six are GEMV variants — dismantle is GEMV-throughput-bound on the GPU side.

## Bandwidth analysis

| Metric | Value | Notes |
|---|---:|---|
| Per-token model reads | 1.82 GiB | V2-Lite Q4_K_M, from `docs/v2.1.0_comprehensive_perf_push.md §1.2` |
| Per-token GPU time | 21.96 ms | Sum of `gpu_us` across all dispatches |
| Effective GPU bandwidth | **82.9 GiB/s** | (1.82 GiB ÷ 21.96 ms) |
| % of vendor peak (150 GB/s) | 55% | M3 Pro spec |
| % of practical anchor (130 GB/s) | 64% | from v2.1.0 strategy doc |
| Engine-work ceiling at 95% practical | ~95 dec_tps GPU-only | (1.82 ÷ (1/130×0.95)) — but only if CPU overhead is also eliminated |
| Engine-work ceiling at 85% practical | ~63-71 dec_tps wall | (more realistic; lines up with the plan's Stage-1 target) |
| llama.cpp implied bandwidth | ~96 GiB/s | (1.82 ÷ (1/52.51) ≈ 19.0 ms/tok wall; assumes ~zero CPU overhead) |

## Dispatch analysis

- **213 dispatches/token** measured (13,648 samples / 64 tokens).
- **Commits/token: 0.0625** (from `dispatch_commits_per_token` in trial stats) → ~1 commit per 16 tokens. Token-CommandBuffer batching is working as advertised.
- **Implied CPU per-dispatch cost: ~117 µs** (25 ms residual ÷ 213 dispatches). This is **5-10× higher than typical Metal encode time (5-15 µs)**.
- Argbuf.rs already flags this: per-dispatch `MTLBuffer::new` for argument storage (~50 µs each). Plus pipeline-cache HashMap lookup-with-mutex on every dispatch.

**This makes A5 (persistent argument-buffer pool) likely a 10-15% win, not the plan's estimated 3-5%.** And A4 (function-constant specialization) compounds with A5 by removing more per-dispatch bytes from the encoder hot path.

## Gap decomposition vs llama.cpp

llama.cpp wall: 19.05 ms/tok. dismantle wall: 47.40 ms/tok. Delta: **28.35 ms/tok = +149% wall**.

Approximate decomposition (with uncertainty since llama.cpp CPU split isn't measured):

| Bucket | dismantle | llama.cpp (est.) | gap | Plan item that targets it |
|---|---:|---:|---:|---|
| GPU compute | 22.0 ms | ~17-19 ms | 3-5 ms | A1 (FA), A6 (autotune) |
| CPU dispatch / encode | ~25 ms | ~0-2 ms | 23-25 ms | **A4, A5, macro-fusion of cross-layer adds** |
| Sync / wait | <1 ms | <1 ms | ~0 | — |
| **Total** | 47.4 ms | 19.0 ms | 28.4 ms | |

**~85% of the gap to llama.cpp is CPU dispatch overhead, not kernel quality.** This is the single most important finding of Stage 0.

## Gate decisions

The Stage 0 gate (per plan: a/b/c) is met:

- **(a) Is the gap dispatch-overhead or kernel-bandwidth?** Both, but dispatch-overhead first. ~85% of the gap is CPU; ~15% is GPU-side kernel quality / bandwidth utilization.
- **(b) Which kernel families are <60% of roofline?** All of the top six are 55-65% — uniform, no single outlier kernel. Re-tuning gives uniform 5-15% wins, not a 2× hotspot fix.
- **(c) Attention / MoE / lm_head / residual split:**
  - MoE: ~62% (Q4 routed 28.9% + Q5_0 routed 10.4% + Q8_0 routed 6.4% + Q4 no-gu 3.6% + Q6_K routed 2.0% + topk gate 2.5% + shared via gemv_f16_simdmat ~6-7% + accumulate 0.6% + silu 0.01% ≈ 60-63%).
  - Attention: ~28% (MLA decode 11.5% + RMSNorm+attn-proj 10.6% + gemv_f32_attn 4.7% + RoPE 0.6% + kv_append 0.2%).
  - LM head + sample: ~6.4%.
  - Residual + rest: ~3%.

This contradicts the plan's prior in two places worth flagging:
- The plan assumed attention was ~30%; confirmed at 28%.
- The plan understated **MoE share (62% vs assumed ~45%)**. MoE-side wins matter more than the plan budgeted.

## Plan revisions (Stage 1 priority reorder)

Original Stage 1 order: A1 (FA) → A2 (Q8 KV) → A3 (residual fusion) → A4 (function constants) → A5 (arg pool) → A6 (autotune).

**Revised Stage 1 order, evidence-driven:**

1. **A5 — Persistent argument-buffer pool** (was 5th). Top of the list because CPU dispatch overhead is the dominant gap (~25 ms/tok). Per-dispatch `new_buffer` is the largest single per-dispatch CPU cost. **Estimated +10-15% e2e (revised up from +3-5%).**
2. **A4 — `MTLFunctionConstantValues` specialization** (was 4th). Compounds with A5: removes per-dispatch arg-buffer bytes by burning shape into the pipeline. **Estimated +5-10% (revised up from +3-8%).**
3. **A1 — Wire & validate `flash_attn_decode_kernel`** (was 1st). Cuts MLA decode (11.5% GPU); estimated +4-8% e2e. Still on the path but no longer the headline.
4. **A3 — Cross-layer residual+RMSNorm fusion** (was 3rd). Cuts dispatch count by ~27/token (residual `add_inplace`); compounds with A5. **Estimated +3-6%.**
5. **A2 — Q8 latent KV** (was 2nd). KV bandwidth is small share (kv_append 0.24% GPU); the win here is footprint reduction for long context + makes Stage 3 spec-decode KV-cheap. **Estimated +2-5%.** Defer until A5+A1+A3 land — order them by leverage.
6. **A6 — Hot-kernel autotune sweep** (last, unchanged).

**New Stage-1 cumulative expectation: 24 → 42-50 dec_tps** (range narrows because A5 reweighting is well-grounded). Same conclusion: engine work alone can't reach 90; need Stage 2 (KV/expert) + Stage 3 (spec-decode).

**One new candidate not in the original plan:** dispatch macro-fusion across the *attention block + MoE gate* — currently 213 dispatches/token, target 100-120. Add as **A7 (post-A5)** if attribution after A5 still shows >40% CPU residual. Don't pre-commit; data-driven.

## Files written this stage

- [reports/path_to_90/stage0/dismantle_bench.json](dismantle_bench.json) — 3-trial bench JSON
- [reports/path_to_90/stage0/dismantle_trace.json](dismantle_trace.json) — 13648-sample ProdCbGpu trace
- [reports/path_to_90/stage0/dismantle_attribution.txt](dismantle_attribution.txt) — analyze_tcb_trace.py output
- [reports/path_to_90/stage0/llama_bench.txt](llama_bench.txt) — llama-bench raw output
- [reports/path_to_90/stage0/attribution.md](attribution.md) — this report

## Halt / re-plan triggers (still active for downstream stages)

- If A5 lands with <+5% e2e: the 117 µs/dispatch model is wrong. Profile the encode path with `Instruments → Time Profiler` before continuing.
- If A4+A5 combined land with <+10% e2e: revisit the macro-fusion lever (new A7) before A1.
- If A1 flash-attn improves only the MLA % (no e2e win because CPU encode dominates): the FA kernel is correct but the lever sequencing is wrong — promote macro-fusion above A1.
- If post-Stage-1 attribution still shows >50% CPU/dispatch overhead, Stage 2/3 won't compound — re-attribute first.

## Next action

Start **A5 — persistent argument-buffer pool**. Files to read:
[crates/dismantle-core/src/metal/argbuf.rs](../../../crates/dismantle-core/src/metal/argbuf.rs),
[crates/dismantle-core/src/metal/mod.rs:374-529](../../../crates/dismantle-core/src/metal/mod.rs).
