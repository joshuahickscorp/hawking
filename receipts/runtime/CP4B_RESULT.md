# CP4b — mlp_down batches at 2.27x, and the knee moved

Raw: `receipts/runtime/CP4B_AFFINE_DOWN_FORWARD.json`, `CP4B_AFFINE_DOWN_REVERSED.json`
Kernels: `qwen_affine_q2_matmul_r{R}k{K}_geo_tpr64_tg128`, `shaders/q80_mixed_decode.metal`
Harness: `crates/hawking-core/examples/ascension_qwen38_cp4b_multiposition_affine_down.rs`

Organ: `mlp_down`, **20.9%** of per-step GPU ns, second largest. Production kernel is the
generic single-tensor affine matvec `qwen_affine_q2_group32_matvec_geo_tpr64_tg128`, which
branches on a runtime `group_size`. Geometry is the TRANSPOSE of gate_up: rows 5120, cols
17408, so `groups_per_row` is 272 rather than 80 and each row is 3.4x longer.

9 reps, first discarded, arms alternated, zero loaded 27B residents (by RSS), under
`gpu_lane_lock.sh`, in both sweep orders.

| r | k | live floats | fwd pos | ns/pos baseline | ns/pos multi | fwd speedup | rev speedup |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 5 | 1 | 153,541 | 154,625 | 0.994 | 0.995 |
| 2 | 1 | 9 | 2 | 154,125 | 156,250 | 0.985 | 0.963 |
| 4 | 1 | 17 | 3 | 153,709 | 159,166 | 0.968 | 0.957 |
| 2 | 2 | 12 | 4 | 154,375 | 105,292 | 1.466 | 1.458 |
| 2 | 4 | 18 | 5 | 154,031 | 99,781 | 1.543 | 1.564 |
| 4 | 2 | 22 | 6 | 88,750 | 49,000 | 1.809 | 1.798 |
| **4** | **4** | **32** | 7 | 89,198 | 39,354 | **2.269** | **2.296** |
| 4 | 8 | 52 | 8 | 88,844 | 44,281 | 2.015 | 2.024 |

**ACCEPT. r4k4 = 2.27-2.30x, bit-identical** (`max_rel_err` exactly `0.000e0`, all eight cells,
both orders), 4 dispatches to 1. Orders agree to within 0.027 on every cell.

Controls: r1k1 = 0.994 / 0.995 (the two kernels do identical work at K=1). **R-only baselines
r2k1 = 0.985 / 0.963 and r4k1 = 0.968 / 0.957 are all slightly BELOW 1.0** — tiling alone is a
small net loss here, so the entire win is multi-token batching. Negative control rejected in
every K>1 cell; uniform-position control holds in all eight.

## The knee moved, and the reason is occupancy not registers

CP4 (gate_up) peaked at r4k4 with **60** live floats and showed no collapse. CP4b peaks at r4k4
with **32** and falls off at r4k8's 52. Same codec, same kernel skeleton, opposite conclusion
about where the ceiling is — so "live floats per thread" is not the governing variable.

The shapes say what is: gate_up is 17408 rows, so at R=4 it launches `ceil(17408/8)` = 2176
threadgroups. down_proj is 5120 rows, so the same R launches `ceil(5120/8)` = 640 — against 60
GPU cores. Raising R divides an already-thin grid, and on this organ that costs more than the
extra accumulators save. Note the R-only column: r4k1 is a LOSS on down_proj (0.957-0.968) where
it was near-neutral on gate_up. Rows, not registers.

This is the third time a knee model has failed to transfer. CP3 said registers; CP4 refuted it;
CP4b says the q4-era number was coincidence and the real constraint is grid width against core
count. **Treat every published knee as scoped to the shape it was measured on.**

> **SUPERSEDED by `CP6C_RESULT.md`.** The grid-width explanation above is wrong, and it is wrong
> in exactly the way this paragraph warns about. CP6c varied R at FIXED K on the real organ with
> real weights across 64 layers: doubling R halves the threadgroups (4352 to 2176) and doubles the
> accumulators (16 to 32), and the time moves by 2.0% at K=2 and 0.3% at K=4. Grid width does
> nothing across a 4x range. What CP4b actually observed was two organs of different SHAPES
> (17408 rows against 5120) having different knees, and the grid-width story was fitted to that
> rather than tested against it. The measured governing variable is K, through FMAs per weight
> loaded. The sentence about scoping a knee to its shape was right; the conclusion drawn one line
> earlier was not.

## Standing arithmetic, and what it is not

Measured organ shares and measured speedups, if and only if they integrate:

| organ | share | speedup | share after |
|---|---|---|---|
| mlp_gate_up | 35.9% | 2.49x | 14.4% |
| mlp_down | 20.9% | 2.27x | 9.2% |
| deltanet (contains in_proj) | 24.1% | 2.474x on in_proj ONLY | not separable |

gate_up and down alone would remove **~33% of per-step GPU ns**. That number is arithmetic on
organ measurements, NOT a prompt-wall measurement, and it is exactly the kind of projection this
mission has already had to retract once. All three kernel families still have **zero call sites**
in `qwen38_hybrid_decode.rs`. CP5 is where it is earned or lost.
