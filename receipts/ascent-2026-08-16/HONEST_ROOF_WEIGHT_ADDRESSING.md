# Honest roof — Qwen3.8 `weight_addressing`

GPU timestamps: completed `MTLCommandBuffer` `GPUEndTime − GPUStartTime` after wait.
Hardware: Apple M3 Ultra, 60 cores, 96 GiB unified memory.
Kernel: `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` and its addr/decode probes.

Timing label: **GPU_PROTECTED_CPU_CONTENDED**. The GPU lock was held, but the
box had concurrent CPU builds and a repeatedly respawned sealed-model supervisor.
Because this is unified memory, the relative finding is strong but the absolute
roof is provisional until a clean paired rerun. The measurement source was a
staged tree based on `8772941f`; that SHA was not the complete source tree.

## Verdict

`weight_addressing` **is bandwidth-saturated on this genome**. The resource is
**unique-once DRAM traffic of the geo_tpr64_tg128 Q4 grouped GEMV** (32 code
bytes + 2 f16 scale bytes per group of 64). It is **not** at 97.6 % of 411.51
GB/s. That number is not a ceiling for this access pattern.

| quantity | value |
|---|---|
| defended bytes | **13,611,663,360** (GEMV codes + f16 scales) |
| sealed addressing time | 21.293 ms |
| sealed addressing rate | **639.25 GB/s** |
| measured single-GEMV addr roof at 13.6 GB | **699.57 GB/s** |
| sealed / kernel roof | **91.4 %** |
| single-GEMV full at 13.6 GB | 666.68 GB/s (4.7 % ALU+decode tax) |
| 401-organ catalog addr / full | 530.65 / 505.81 GB/s |
| unique_once at 13.6 GB | 375.65 GB/s |
| refuted ceiling | 411.51 GB/s |

## The 97.6 % claim, taken apart

`13,618,141,856 / 33,912,333 ns / 411.51 = 97.58 %`.

1. **Bytes.** 13,618,141,856 is geometry-active (GEMV + norms + one embed row).
   Addressing does not load norms or the embed row.
2. **Time.** 33.91 ms is the whole production GPU token. DeltaNet, GQA, norms,
   SwiGLU, KV, and terminal FMA move none of those GEMV bytes.
3. **Ceiling.** 411.51 is the 512 MiB point of a sequential `unique_once`
   read-reduce. The same sweep's 1024 MiB point was 301.63 and was discarded.
   `unique_once` takes `uint nbytes` and cannot name a 13.6 GB point in one
   dispatch.

Correct attribution: **13,611,663,360 bytes / 21.293 ms = 639.25 GB/s**.

## Byte-count adjudication

| source | bytes | verdict |
|---|---|---|
| **geometry GEMV payload** | **13,611,663,360** | **defended.** What `geo_tpr64` streams. |
| ledger `active_bytes` | 13,618,141,856 | +6,475,776 norms + 2,720 embed row |
| `ACTIVE_BUDGET_BYTES` | 13,622,264,240 | manifest classes: 40 B HQ30UQ4 headers/tensor + extra mixer tensors |
| bandwidth receipt | 13,621,829,601 | `14,297,694,680 − 675,865,079`; embed figure is 434,639 B too large |

MLP manifest − geometry = 7,680 = 192 × 40 headers. lm_head extra = 40. Linear
extra after headers ≈ 7.9 MB of conv / A_log / dt_bias — not GEMV traffic.

## Curve (single Q4 GEMV, cols = 5120, GPU ns)

Working set grows; rate does **not** collapse the way unique_once was claimed to.

| payload | unique_once GB/s | Q4 addr | Q4 decode | Q4 full |
|---|---:|---:|---:|---:|
| 64 MiB | 268.3 | 817.1 | 632.8 | 608.0 |
| 128 MiB | 326.4 | 675.3 | 647.2 | 627.5 |
| 256 MiB | 286.8 | 692.1 | 668.0 | 646.8 |
| 512 MiB | 342.7 | 694.0 | 667.9 | 658.3 |
| 1024 MiB | 368.8 | 698.3 | 680.2 | 663.7 |
| 2048 MiB | 376.6 | 687.3 | 678.9 | 666.1 |
| 4096 MiB | 373.6 | 692.5 | 682.0 | 666.8 |
| 8192 MiB | 375.1 | 698.4 | 684.7 | 665.6 |
| **13.612 GB** | **375.7** | **699.6** | **683.8** | **666.7** |

unique_once plateaus near 375 GB/s from 2 GiB through 13.6 GB. Q4 GEMV plateaus
near 667 (full) / 700 (addr). The discarded 1024 MiB unique_once point is how
411.51 became a fake ceiling: the curve was collapsed to its flattering point,
and that point was the wrong shape.

This run did **not** reproduce 411.51 at 512 MiB (we measured 343). The original
used a 1 GiB buffer; this run held a 13.6 GB unique slab plus 13.6 GB of Q4
payload. The size dependence is the result, not any one number.

## Headroom that remains

- **Inside one GEMV:** decode+FMA are a 4.7 % tax. Not the 21.3 ms.
- **Dispatch topology:** 401 mixed organs in one CB run at 531 GB/s addr,
  24 % below the single-GEMV roof. Isolated class CBs (the ledger) sit at 639,
  between the two. Tiny `ba` (96×5120) and encoder boundaries are genome.
- **819 GB/s datasheet:** the kernel roof is 700, 85 % of published peak. The
  rest is not available to this access pattern without changing the genome.
- **Density** is still a lever. It is no longer the only lever justified by a
  fake 411.51 ceiling.

Authority receipt: `HONEST_ROOF_WEIGHT_ADDRESSING.json`.
