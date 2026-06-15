# Per-kernel attribution — DeepSeek-V2-Lite Q4_K_M on M3 Pro 18 GB

Captured 2026-05-20 from `traces/mst_labeled_20260520_201202.trace` (the
first MST capture taken after the encoder-labeling pass landed at
[crates/dismantle-core/src/metal/mod.rs](../crates/dismantle-core/src/metal/mod.rs)).

## Method

- Production profile (`profiles/deepseek-v2-lite-q4.m3pro18.json`), 32
  decode tokens, 4-token prompt, Claude.app open (contamination caveat;
  fine for relative ranking).
- Extracted `metal-application-encoders-list` with XML id/ref resolution
  to recover the per-encoder kernel label (the 3rd `metal-object-label`
  slot — the un-prefixed one set by `encoder.setLabel(fn_name)`).
- Where an encoder contained multiple pipeline switches (the TCB
  attention-block fusion), the encoder's duration is split evenly
  across the listed kernels.
- 3,788 encoders total; 976 fused (multi-kernel), 2,812 single-kernel.
- 21 unique kernel names. Total encoder time: 11.26 ms.

**Caveat — what "encoder time" means here.** The encoders table's
Duration is the CPU-side encoder lifecycle (from
`new_compute_command_encoder` to `end_encoding`). For
`MetalContext::dispatch_threads` (the immediate-commit path), this
includes implicit GPU wait. For the TCB path (where 15+ pipeline
switches share one encoder), Duration is encoding-only and the GPU
runs after `commit_and_wait`. Treat this as a **relative ranking** of
where the engine spends time, not a per-kernel GPU profile.

For pure GPU time per kernel, the next pass needs to join encoder
labels against `metal-gpu-intervals` (322 MB, 319k rows) by
encoder-id. Queued.

## Top kernels by attributed encoder time

| Rank | Kernel | ms | % | calls | µs/call |
|---|---|---|---|---|---|
| 1 | moe_batched_gemm_q8_0_indexed_v2t | 1.72 | 15.3% | 77 | 22.33 |
| 2 | rmsnorm_f32 | 1.40 | 12.4% | 316 | 4.43 |
| 3 | add_inplace | 1.33 | 11.8% | 282 | 4.71 |
| 4 | moe_batched_gemm_q4_indexed_v2t_gu_v2 | 1.29 | 11.4% | 212 | **6.07** |
| 5 | moe_batched_gemm_q5_0_indexed_v2t | 1.11 | 9.9% | 73 | 15.21 |
| 6 | moe_batched_gemm_q4_indexed_v2t | 0.83 | 7.4% | 51 | **16.28** |
| 7 | moe_batched_gemm_q6_k_indexed_v2t | 0.73 | 6.5% | 45 | 16.32 |
| 8 | rmsnorm_gemv_f16w_attn_pinned_v2t | 0.54 | 4.8% | 310 | 1.73 |
| 9 | rope_slice_f32_inplace | 0.27 | 2.4% | 155 | 1.73 |
| 10 | rope_q_f32_inplace | 0.27 | 2.4% | 155 | 1.73 |
| 11 | kv_append_f32 | 0.27 | 2.4% | 155 | 1.73 |
| 12 | mla_decode_kernel | 0.27 | 2.4% | 155 | 1.73 |
| 13 | gemv_f16_simdmat | 0.27 | 2.4% | 155 | 1.73 |
| 14 | gemv_f32_attn | 0.22 | 1.9% | 102 | 2.15 |
| 15 | gemv_f32_moe | 0.20 | 1.7% | 121 | 1.61 |

(Per-call cost for items 9–13 is identical because they're all members
of the same fused TCB encoder; the split-evenly attribution charges
each equally. Their real ratios diverge once joined with
`metal-gpu-intervals`.)

## Aggregate findings

- **MoE batched GEMMs dominate: 50.5% combined** (q8_0 + q4_gu_v2 +
  q5_0 + q4 + q6_k). This is consistent with what
  [[v110_path30_findings]] said about the routed-expert path being the
  hot loop.
- **Attention (mla_decode_kernel) is only ~2.4%.** That validates last
  session's flash-attn finding: the lever was correctly identified as
  having small headroom on single-stream V2-Lite. The plan's predicted
  +5-9 tps was overshoot — attention isn't enough of the budget to
  give those gains at short context.
- **rmsnorm_f32 (12.4%) + add_inplace (11.8%) = 24% combined.** These
  are individually tiny (~4-5 µs) but called 280-316 times each. A
  fusion lever (merge rmsnorm into the next GEMM's prologue; merge
  add_inplace into the prior GEMM's epilogue) could compress a real
  fraction of total time. Note `rmsnorm_gemv_f16w_attn_pinned_v2t`
  already fuses one rmsnorm into the attn input GEMM — that pattern
  could extend to FFN inputs.

## The actionable lever — port `_v2t_gu_v2` to other quants

**Compare per-call times for the routed-expert MoE GEMMs:**

| variant | µs/call | speedup over `_v2t` |
|---|---|---|
| `moe_batched_gemm_q4_indexed_v2t` | 16.28 | 1.0× |
| `moe_batched_gemm_q4_indexed_v2t_gu_v2` | **6.07** | **2.68×** |
| `moe_batched_gemm_q8_0_indexed_v2t` | 22.33 | — (no gu_v2) |
| `moe_batched_gemm_q5_0_indexed_v2t` | 15.21 | — (no gu_v2) |
| `moe_batched_gemm_q6_k_indexed_v2t` | 16.32 | — (no gu_v2) |

q4's `_v2t_gu_v2` variant is **2.68× faster per call** than its sibling
`_v2t`. Inspect [crates/dismantle-core/shaders/moe.metal:626](../crates/dismantle-core/shaders/moe.metal:626) — `gu_v2` fuses the gate
GEMM and up GEMM for the same routed expert into one dispatch,
amortizing weight loads and threadgroup setup across both.

q5_0, q6_k, and q8_0 have no equivalent. Porting the `gu_v2` fusion
pattern to those three variants is the highest-EV next lever
identified from this trace.

**Estimated savings** (rough):
- q8_0: if it follows the same 2.68× ratio, 22.33 µs → 8.33 µs.
  77 calls × (22.33 − 8.33) = **1.08 ms saved per decode (32 tok)**.
- q5_0: 15.21 → 5.68. 73 × 9.53 = **0.70 ms saved**.
- q6_k: 16.32 → 6.09. 45 × 10.23 = **0.46 ms saved**.
- Combined: **~2.24 ms saved over 32 decode tokens = 70 µs/token**.
- At baseline 1573 ms / 32 tokens = 49.2 ms/token, 70 µs ≈ **+0.14%**.

Hmm — even if the per-call ratio holds, the absolute saving is tiny
because the encoder durations being measured include large CPU/wait
components. A real GPU-time joined attribution would either confirm
this is small (and the lever isn't worth it) or reveal the gap is in
GPU dispatches we can't see here. **The "join with gpu_intervals"
step is now the gating analysis before committing to the
gu_v2-portation engineering work.**

## Compounding lever check — fused-encoder share

- 976 fused encoders × ~15 kernels each ≈ 14,640 kernel dispatches
  that paid one encoder-creation overhead.
- 2,812 single-kernel encoders × 1 ≈ 2,812 dispatches at full
  encoder-creation overhead.
- The TCB attention-block fusion (the long `embed_lookup_f32 &
  rmsnorm_gemv_f16w_attn_pinned_v2t & … & moe_topk_gate &
  moe_batched_gemm_q4_indexed_v2t_gu_v2` encoder) is doing most of
  the work. Routed-expert dispatches stay as separate encoders.
- This means **ICB (#9 in plan) compresses encoder-creation for the
  2,812 single-kernel dispatches**, not the fused ones (which already
  amortize). 2,812 / 36 forward passes ≈ 78 single-kernel encoders
  per forward pass that ICB would compress. Each saves an encoder
  alloc + binding rebind, ~1-3 µs each on Apple Silicon. Estimated
  ICB win: ~80 × 2 µs = **160 µs/token = 0.3%**. Possibly underwhelming.

## Open items

- **Join encoder-id → `metal-gpu-intervals` duration** to get pure GPU
  time per kernel. The XML resolver from this report extends; needs
  streaming parse of the 322 MB intervals file.
- **Validate the gu_v2 portation gain estimate** before committing to
  kernel work. The 2.68× q4 ratio is real per-call, but the absolute
  per-token saving estimate (70 µs) is small. May not clear the +5%
  gate.
- **The +24% combined rmsnorm + add_inplace share** is the most
  attractive headline number. Investigating whether the existing TCB
  fusion can swallow more of those calls is a separate lever from the
  gu_v2 work.
