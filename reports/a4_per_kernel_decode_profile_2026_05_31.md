# A4 — per-kernel Qwen-3B decode profile (2026-05-31)

**Workload:** Qwen2.5-3B-Instruct-Q4_K_M on M3 Pro 18 GB.
**Locked config:** `DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 DISMANTLE_QWEN_Q4K_PREDEC=1 DISMANTLE_QWEN_LMHEAD_PREDEC=1` (the A1-committed tree, `0e6eb14`).
**Source of record:** engine `DISMANTLE_TCB_TRACE=gpu_prod` per-dispatch GPU-timestamp trace (`reports/traces/qwen3b_decode_gpu_prod_2026_05_31.json`), analyzed by `tools/bench/analyze_tcb_trace.py --model qwen3b`. **§1 methodology gate: PASS** (INV1 BW ≤ peak; INV2 unmapped "other" = 0%; INV3 token count from `sample_*`).
**Run:** code prompt (`reports/a4_code_prompt.txt`), 128 new tokens, 1 trial, `nice -n 19 taskpolicy -b`, qwen3b profile. Clean untraced sibling = **31.0 dec_tps median** (3 trials; matches the A1 ≈31 baseline). Traced run 30.6 dec_tps.

## Why this is the gpu_prod trace, not a fresh xctrace MST

The MST capture pipeline **ran successfully** (`traces/mst_20260531_021428.trace`, exported to `traces/mst_20260531_021428_export/`) but **cannot attribute per-kernel for this workload** — twice confirmed:

1. The Metal-System-Trace "Metal System Trace" template does **not** enable the Shader Profiler counter instrument, so `metal-shader-profiler-intervals` / `gpu-shader-profiler-interval` export to **3 lines each (header only, empty)** — no per-function GPU intervals.
2. `metal-gpu-intervals` (1270 rows) is dominated by **Claude.app's compositor**: 612 Vertex + 605 Fragment intervals owned by PID 39990 ("Claude Helper") / PID 646 (WindowServer), all labeled `coreanimation.*` / `Render Command N`. Only **53 intervals are `Compute`, totaling 1.23 ms — and those are also Claude Helper's Blit commands**. dismantle's own decode process (launched under background QoS via `taskpolicy -b`) does not appear as a labeled GPU producer. `metal-object-label` carries only generic `Command Buffer N` / `Blit Command N` labels — dismantle does not set Metal debug labels on its compute pipelines.

So the MST gives Claude-polluted, type-only (Compute/Vertex/Fragment) GPU intervals with no kernel names. The engine's `gpu_prod` mode is the un-distorted ground truth used for the canonical budget (`reports/dismantle_budget_2026_05_24.md` §s2/s3): production single-CB-per-token, real per-dispatch GPU timestamps, real kernel names. **This profile uses gpu_prod** and supersedes the 2026-05-24 budget table (which predates the predec path being default-on).

## Per-kernel decode breakdown (gpu_prod, locked config)

per-token GPU busy = **21.73 ms**; achieved aggregate BW = **88.8 GiB/s = 59% of 150 peak (68% of 130 sustained)**.

| kernel | % GPU | calls/token | µs/call | role | achieved BW |
|---|---|---|---|---|---|
| **`gemm_q4_k_v4_predec_pair`** | **46.64%** | 36 | 281.5 | fused FFN gate+up (11008×2048 ×2), 1/layer | **~84 GB/s = 56% of peak** |
| **`gemm_q4_k_v4_predec_2r`** | **42.78%** | 163 | 57.0 | q/o/ffn_down(Q4 layers)/LM-head, 2-row predec | ~mid-50s % of peak |
| `mha_decode_f32` | 3.18% | 36 | 19.2 | attention | trivial (KV cache) |
| `add_rmsnorm_fused` | 3.02% | 72 | 9.1 | norm+residual | trivial |
| `add_inplace` | 1.18% | 108 | 2.4 | residual add | trivial |
| `rope_q_f32_inplace` | 0.80% | 72 | 2.4 | RoPE | trivial |
| `memcpy_f32_off` | 0.78% | 72 | 2.3 | buffer copy | trivial |
| `gemm_q6_k_fused_v2` | 0.73% | 18 | 8.9 | ffn_down Q6_K layers (Q4_K_M mix) | trivial |
| `moe_batched_silu_mul` | 0.67% | 36 | 4.0 | SiLU gate | trivial |
| `sample_argmax_f32`, `embed_lookup_f32`, `rmsnorm_f32` | <0.1% each | 1 | — | sampling/embed | trivial |

## The single dominant stall

**The pre-decoded Q4_K decode GEMVs are the entire decode budget: `_pair` (46.6%) + `_2r` (42.8%) = 89.4% of GPU time.** Attention is 3.2%, rmsnorm+add ~4.2% (much smaller than the V2-Lite MoE profile because the dense GEMVs dominate). The predec kernel has *replaced* the old `gemm_q4_k_m_v3_8r` (87.3% in the 2026-05-24 budget) as THE kernel — the A-series predec work shipped and this is now the wall.

**The single most dominant kernel is `gemm_q4_k_v4_predec_pair` (46.6%, 281.5 µs/call) at ~84 GB/s = 56% of 150-peak BW.** It is the fused FFN gate+up on the biggest shape (11008×2048 ×2 weight matrices). At 56% of peak it is **kernel-bound, not BW-bound** — there is a ~1.6–1.8× headroom to peak/sustained DRAM BW. This is the kernel A5/A6 must target.

## Why the efficiency is left on the table (shader inspection, read-only)

`crates/dismantle-core/shaders/quant.metal:1870–2059`. In both `_predec` and `_predec_pair`, the inner loop:
- reads weight nibbles as **single `uchar` loads** — `w_q4[bo + 16 + pi*32 + simd_lane]`, **1 byte per thread per `pi`**, 4 separate byte loads per block (no vector-width coalescing);
- unpacks scalar nibbles (`qb & 0x0F`, `qb >> 4`) one byte at a time;
- the `_pair` kernel already amortizes the `x` load and weight-header decode across gate+up, and `_2r` already runs 2 accumulator chains to hide DRAM latency — so the *cheap* dispatch-amortization and latency-hiding levers are spent. The remaining gap is **per-thread memory transaction width and ALU on the nibble unpack**.

## Recommendation for A5 / A6

- **A5 (vectorized uint4 nibble unpack)** → target `gemm_q4_k_v4_predec_pair` first (46.6%, biggest single slice), then `_2r`. Replace the 4× scalar `uchar` weight loads per block with a single **`uint4` (16-byte) vectorized load** of the 32-byte nibble plane (2× `uint4`, or `as_type` over a `device const uint4*` weight pointer), then unpack all 8 nibble-pairs from registers. This widens each thread's memory transaction (the current 1-byte loads under-fill the 16-byte cache line) and lets the compiler issue fewer, fatter loads — directly attacking the 56%→peak BW gap on the dominant kernel. Gate: bit-identical (the math is unchanged; only the load width/unpack changes).
- **A6 (threadgroup / occupancy tuning)** → both kernels are fixed at 256 threads/TG, 8 simdgroups, 8 rows/TG (`_pair`) / 16 rows/TG (`_2r`). On the 11008-row gate+up shape, sweep **rows-per-TG / rows-per-simdgroup (the `_2r`→`_3r`/`_4r` accumulator-chain count) and TG width** to raise occupancy and in-flight weight streams. **Caveat (the `_4r` lesson):** `_4r` previously died because it wasn't profile-targeted — now it is. A6 must bench each variant against this 281.5 µs/call `_pair` baseline and keep only what beats it; do not ship a wider-chain kernel on register-pressure speculation. Gate: bit-identical + paired bench vs this baseline.

## Artifacts
- `reports/traces/qwen3b_decode_gpu_prod_2026_05_31.json` — gpu_prod trace (source of the table).
- `reports/a4_clean_walltime.json` — untraced 3-trial clean baseline (31.0 dec_tps).
- `reports/a4_code_prompt.txt` — the code prompt used.
- `traces/mst_20260531_021428.trace` + `_export/` — the MST capture (Claude-polluted, no per-kernel attribution; kept for the record).
