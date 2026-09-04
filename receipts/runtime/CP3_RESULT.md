# CP3 — one organ, one layer, physical: multi-position batching WINS at the real shape

Raw: `receipts/runtime/CP3_MULTIPOSITION_INPROJ.json`
Harness: `crates/hawking-core/examples/ascension_qwen38_cp3_multiposition_inproj.rs`
Machine: Apple M3 Ultra, 60 GPU cores. GPU ns = `GPUEndTime - GPUStartTime` on completed
command buffers, never a CPU-wait proxy. 9 reps, first discarded, arms alternated A B A B.
Measured with ZERO loaded 27B residents (checked by resident memory, not process name) and
under `tools/gpu_lane_lock.sh`.

## The premise had to be corrected first

CP3 was framed as "wire the existing multi-position kernel into the biggest organ". That
premise is false, and it fails before any measurement.

`mlp_gate_up` is the largest organ at **35.9%** of per-step GPU ns
(`receipts/headless/_ORGAN_BANDWIDTH_raw.json`, 7 reps, noop control 125 ns). But the live
resident's own dispatch record shows it running
`qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128`, and `mlp_down` (20.9%)
running `qwen_affine_q2_group32_matvec_geo_tpr64_tg128`. The artifact agrees: **192
segments — gate/up/down across all 64 layers — are `.hgrafv01`,
`hawking.gravity.affine_scale_bias.v1`**, not uniform q4.

`grep "kernel void qwen_affine_q2.*matmul"` over every shader returns **nothing**. 42
affine-q2 kernels exist; not one is multi-position. The `matmul_r{R}k{K}` family is
uniform-q4 only, and has **zero call sites** in `qwen38_hybrid_decode.rs`.

So **56.8% of per-step GPU ns runs on a codec with no batching kernel at all.** CP3
retargets to the largest organ that does run uniform q4: DeltaNet `in_proj`, 24.1%,
rows 16384 x cols 5120, group 64. Code plane 41,943,040 B, taken from a real
44,564,520-byte `.hq30uq4` artifact segment.

## Result

Baseline arm: K dispatches of the production `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`,
one per position, each on its own command buffer — which is physically what the per-token
prefill loop at `qwen38_hybrid_decode.rs:7566` does.
Multi-position arm: one dispatch of `qwen_uniform_q4_group64_matmul_r{R}k{K}_geo_tpr64_tg128`.

| r | k | ns/pos baseline | ns/pos multi | speedup | live floats/thread |
|---|---|---|---|---|---|
| 1 | 1 | 128,750 | 128,959 | **0.998** | 5 |
| 2 | 2 | 74,500 | 40,000 | 1.863 | 12 |
| 4 | 2 | 74,938 | 39,625 | 1.891 | 20 |
| 2 | 4 | 74,896 | 35,125 | 2.132 | 20 |
| **4** | **4** | **74,520** | **30,125** | **2.474** | **32** |
| 8 | 4 | 74,927 | 67,719 | 1.106 | 56 |
| 16 | 4 | 75,062 | 75,396 | 0.996 | 104 |
| 4 | 8 | 78,286 | 55,594 | 1.408 | 56 |
| 8 | 8 | 77,016 | 60,375 | 1.276 | 96 |
| 16 | 1 | 74,500 | 109,541 | 0.680 | 50 |

**ACCEPT. r4k4 = 2.474x lower physical wall, 4 dispatches to 1, and the outputs are
BIT-IDENTICAL** (`max_rel_err` exactly `0.000e0`; the K=8 cells sit at 1.837e-6, float
reassociation). Not "the chunked path executed" — a lower complete wall with preserved
semantics.

Controls, all of which had to hold:
* **r1k1 = 0.998.** At K=1 the two kernel bodies do identical work — same thread map, same
  8-wide unpack, same reduction. If this cell had shown a win it would mean the "batching
  win" was really a kernel-quality difference. It shows none, so the win is batching.
* **Negative control REJECTED in all 12 K>1 cells.** Column k compared against position
  k+1's answer is rejected by the same check that accepts the real comparison, so the check
  has been observed to fail and is evidence.
* **Uniform-position control holds in all 16 cells**: all K columns identical must reproduce
  the K=1 answer, which catches a kernel that writes one column K times.

## Where it stops winning, and why

The knee is **(R=4, K=4)** and the mechanism is **register pressure, not the WY O(T^2 K)
term** — this organ has no WY construction at all. Live floats per thread is
`R*K + 2K + 2R`; the win survives to 32 and collapses by 50-56:

    r4k4    32 floats   2.474x
    r8k4    56 floats   1.106x
    r16k1   50 floats   0.680x
    r16k4  104 floats   0.996x

**The directive's T sweep {8, 16, 32, 64, 128} is entirely past this knee.** K=8 is already
worse than K=4 at every R measured. The four uninstantiated grid cells are reported in the
raw receipt with their budgets rather than dropped; r16k8 would need 176 live floats/thread
and is not worth instantiating on this evidence.

## CP3b — the baseline anomaly is RESOLVED, and it refutes the flattering number

Raw: `receipts/runtime/CP3B_REVERSED_ORDER.json` (`sweep_order: REVERSED`).

The first pass measured the R=1 cells (which ran first) at ~128,750 ns/position against a
steady ~74,500 across the twelve later cells. Two mechanisms explained that equally well and
implied opposite conclusions: cache reuse in the sequential arm, which a real 64-layer prefill
loop would not get, making 2.474x a FLOOR; or first-cells warm-up, making 2.474x the honest
number. Re-running with the sweep order reversed separates them, because under warm-up the
cost follows the ORDER and under cache it follows the CELL.

**It follows the order.**

| cell | fwd position | rev position | fwd ns/pos baseline | rev ns/pos baseline |
|---|---|---|---|---|
| r1k1 | 1 | 20 | 128,750 | **74,375** |
| r1k4 | 3 | 18 | 129,188 | **74,188** |
| r2k1 | 5 | 16 | 95,375 | 74,125 |
| r4k4 | 13 | 8 | 74,520 | 85,646 |
| r4k8 | 14 | 7 | 78,286 | **128,370** |
| r8k4 | 16 | 5 | 74,927 | **128,667** |
| r16k4 | 19 | 2 | 75,062 | **124,209** |

The expensive cells swapped places exactly with the order. So the cause is **first-cells
warm-up / DVFS ramp, not cache**, and the "vs serial-K1" ratio of 4.281x that the raw receipt
carries is **REFUTED** — it divided by a K1 reference measured during the ramp. It is not
claimed, and the earlier suspicion that 2.474x was a floor is withdrawn.

**2.474x is the number.** The paired design is what makes it survive: because both arms of a
cell run adjacently, the ramp cancels in the ratio, and the speedups agree across the two
orders to within noise —

    r4k4   2.474  /  2.479        r2k2   1.863  /  1.859
    r2k4   2.132  /  2.114        r16k1  0.680  /  0.678

Interleaving activations into `input[col*K + k]` cost 19,167 ns of host time against
~120,500 ns of GPU time saved (15.9%). In a real chunked prefill the activations arrive
batched, so most of that disappears; it is charged here anyway.

## Reprofile — what CP4 should be

Not "port this to gate_up". The dominant organ cannot use this kernel. The affine-q2
gate_up kernel (`q80_mixed_decode.metal`) is structurally the SAME skeleton — geo_tpr64,
kSplit 2, 2 rows/TG, 8-wide unpack, simd_sum — so an RxK affine variant is a direct port.
But it accumulates gate AND up together, so it needs **2*R*K accumulators**, double this
organ's cost at the same (R,K), plus a third half-plane read per group for the bias. On the
register curve measured above, the affine knee should therefore land nearer (R=2,K=4) or
(R=4,K=2), not (4,4). That is a prediction this receipt makes and CP4 can falsify.

Ranked by the directive-VIII rule:
1. **CP3b**, the cache-vs-warm-up discriminator. Cheapest, and it sets whether the affine
   work is worth 2.5x or 4x. Blocks nothing else.
2. **Multi-position affine-q2 gate_up.** 56.8% of step GPU ns, no kernel exists, structural
   port with a measured register budget to design against.
3. **CP5 multi-layer**, which is also the honest test of the cache question, because a real
   layer sequence evicts between positions exactly as production does.
