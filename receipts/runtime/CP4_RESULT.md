# CP4 — the dominant organ can be batched too, and it refutes CP3's prediction

Raw: `receipts/runtime/CP4_AFFINE_GATE_UP_FORWARD.json`,
`receipts/runtime/CP4_AFFINE_GATE_UP_REVERSED.json`
Kernels: `qwen_affine_q2_group64_matmul_gate_up_r{R}k{K}_geo_tpr64_tg128`,
`crates/hawking-core/shaders/q80_mixed_decode.metal`
Harness: `crates/hawking-core/examples/ascension_qwen38_cp4_multiposition_affine_gate_up.rs`

Organ: `mlp_gate_up`, **35.9%** of per-step GPU ns, the largest single organ.
Codec: `affine_q2_group64_fp16_scale_bias` (`.hgrafv01`), geometry read from the artifact's
own header — shape [17408, 5120], group 64, `code_bytes` 22,282,240, `scale_bytes` 2,785,280,
`bias_bytes` 2,785,280. Code plane taken from a real 27,853,103-byte segment.

Before CP4 there was no multi-position kernel for this codec at all. CP3's 2.474x was on
uniform-q4 DeltaNet in_proj (24.1%); `grep "kernel void qwen_affine_q2.*matmul"` returned
nothing, so 56.8% of the step — gate_up plus mlp_down — could not be batched by any existing
kernel.

## Result, measured in BOTH sweep orders

9 reps, first discarded, arms alternated A B A B, zero loaded 27B residents (checked by RSS),
under `tools/gpu_lane_lock.sh`. CP3b established that the first cells of any sweep sit on a
DVFS ramp, so any claim that compares CELLS to each other is run in both orders.

| r | k | live floats/thread | fwd position | fwd speedup | rev position | rev speedup |
|---|---|---|---|---|---|---|
| 1 | 1 | 9 | 1 | 0.999 | 7 | 0.997 |
| 2 | 1 | 17 | 2 | 0.998 | 6 | 1.001 |
| 4 | 1 | 33 | 3 | 0.915 | 5 | 0.913 |
| 2 | 2 | 22 | 4 | 1.612 | 4 | 1.620 |
| 4 | 2 | 42 | 6 | 1.656 | 2 | 1.655 |
| 2 | 4 | 32 | 5 | 2.432 | 3 | 2.477 |
| **4** | **4** | **60** | 7 | **2.492** | 1 | **2.491** |

Speedups are medians of per-rep paired ratios. The two orders agree to within 0.008 on six of
seven cells and 0.046 on the seventh, so the ordering carries nothing.

**ACCEPT. r4k4 = 2.49x, bit-identical** (`max_rel_err` exactly `0.000e0` in all seven cells,
both orders), 4 dispatches to 1.

Controls, all of which had to hold:
* **r1k1 = 0.999 / 0.997.** At K=1 the multi-position kernel and the production matvec do
  identical work; a win here would mean the "batching win" was a kernel-quality difference.
* **R-only baselines: r2k1 = 0.998 / 1.001, r4k1 = 0.915 / 0.913.** Kernel tiling alone buys
  NOTHING on this organ — r4k1 is a net LOSS. So the entire win is multi-token batching, not
  more rows per activation load. Without these rows the table would credit R and K together.
* **Negative control REJECTED** in every K>1 cell; **uniform-position control holds** in all
  seven.

## CP3's prediction is REFUTED

CP3 predicted, from its own register curve, that the affine knee would land at (2,4) or (4,2)
rather than (4,4), because this kernel accumulates gate AND up together — `2*R*K` accumulators
against CP3's `R*K`, so `live floats ≈ 2*R*K + K + 6*R`, putting r4k4 at 60 where CP3's q4
curve had already collapsed (r8k4 at 56 gave 1.106x, r16k1 at 50 gave 0.680x).

Measured: **r4k4 at 60 live floats is the BEST cell**, in both orders. No collapse.

So the CP3 Law "the knee is register pressure" is **narrowed in scope**: it describes the
uniform-q4 kernel family, not this machine. The likely reason it does not transfer is that
the affine kernel is far less bandwidth-bound — it reads a `ushort` per row per group where q4
reads a `uint`, and does two FMAs per weight rather than one. Its multi-position arm at r4k4
moves ~55.7 MB in 409 µs (~136 GB/s) against the baseline's ~219 GB/s, i.e. it is leaving
bandwidth on the table and is limited by arithmetic, which is exactly the regime where extra
accumulators are cheap. That explanation is a hypothesis; the refutation is not.

A prediction dying under its own test is the point of having made it explicit in the harness.

## What this is worth, and what it is not

`mlp_gate_up` is 35.9% of per-step GPU ns. At 2.49x that term falls to ~14.4%, i.e. ~21.5% of
step GPU ns removed — IF the batching survives integration. It is not integrated: these
kernels have **zero call sites** in `qwen38_hybrid_decode.rs`, exactly as the q4 family did
before CP3. An organ-level speedup is not a prompt-wall speedup, and CP5/CP6 are what decide
that.

`mlp_down` (20.9%) runs `qwen_affine_q2_group32_matvec_geo_tpr64_tg128` — group 32, a different
kernel — and still has no multi-position variant.

## Next

1. **CP5 multi-layer.** Both CP3 and CP4 won at one organ, one layer, in isolation. The claim
   that matters is a prompt wall, and it requires the recurrent state to be chunked so the
   projections can be batched across a real layer sequence — which is what CP0-CP2's WY
   composition was for and which nothing has yet exercised physically.
2. **group32 affine for mlp_down**, +20.9% of the step, same structural port.
3. Wider (R,K) for affine now that 60 live floats is known not to be the wall.
