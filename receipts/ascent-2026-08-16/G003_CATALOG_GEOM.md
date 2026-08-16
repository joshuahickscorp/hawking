# G003 catalog reparse — before / after

Warm in-process rep 2 (expert_bind = 0). Three alternating process pairs.
GPU is completed MTLCommandBuffer `GPUEndTime − GPUStartTime`.

Greedy ids on every process: `[8420, 748, 264, 729, 429, 17431, 288, 264, 914, 320, 72, 1734]`.
silent_fallbacks `dense_w_materialized` / `host_expert_payload_bind` / `host_mixed_matvec` = 0.

## host_preparation BEFORE and AFTER

| component | before median | after median | delta |
|---|---:|---:|---:|
| **host_preparation** | **79.555 ms** | **24.987 ms** | **−54.568 ms** |
| catalog.load_packed | 64.183 ms | 8.597 ms | −55.586 ms |
| embed gather | 8.521 ms | 8.608 ms | +0.087 ms |
| buffer write/read | 4.711 ms | 4.868 ms | +0.157 ms |
| encode | 2.114 ms | 2.488 ms | +0.373 ms |
| expert_bind | 0 | 0 | 0 |
| gpu_matvec (MTL CB) | 35.564 ms | 35.489 ms | −0.075 ms |
| wall | 253.235 ms | 198.239 ms | −54.997 ms |

Acceptance: host_preparation falls ≥ 40 ms. Measured **54.568 ms**.
catalog.load_packed no longer dominates host_prep (64.2/79.6 → 8.6/25.0).
Remainder 8.6 ms is one embed `packed()` per token; GEMV `packed()` went from 11 144 calls/generate to 0 (`packed_skipped=11116`).

## Full warm spread (3 process reps / arm)

| arm | wall ns | GPU ns | host_prep ns |
|---|---|---|---|
| before | 252 058 383 … 256 897 610 | 35 562 378 … 35 581 190 | 78 441 154 … 80 128 179 |
| after | 192 074 799 … 198 246 792 | 35 480 432 … 35 568 841 | 22 544 260 … 25 500 341 |

## What stayed

encode (~2.1–2.5 ms), embed gather (~8.5–8.6 ms), buffer write/read (~4.7–4.9 ms), GPU matvec (~35.5 ms).
Only the per-GEMV catalog header walk was deleted.

Raw pair receipts: `receipts/ascent-2026-08-16/g003-catalog-geom/`.
