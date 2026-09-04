# CP6a — real catalog weights, all 64 layers: 1.892x, and one instrument caught lying

Raw: `receipts/runtime/CP6A_REAL_SESSION_ORGAN.json`
Harness: `crates/hawking-core/examples/ascension_qwen38_cp6a_real_session_chunked_organ.rs`
Runtime: `measure_isolated_organ_chunked`, added beside `measure_isolated_organ`.

Every prior CP measured buffers this crate filled. `CP5C_RESULT.md` listed "real artifact weight
bytes at these shapes" as untested. This is that test: weights come from the session's own
catalog via `affine()`, all 64 layers are encoded, and each arm sits in one command buffer — the
same instrument that produced the production organ number. Zero loaded residents, 50.8 GB free,
noop control 208 ns.

## The first run's baseline was broken, and the roof is what caught it

The first pass reported **0.226x**. `gate_up` across 64 layers is 3.565 GB and the machine's
measured roof is 778.8 GB/s, so no arm reading those weights can finish under 4.578 ms. That
run's pair baseline claimed 866,874 ns — **4112.7 GB/s, 5.3x a physical ceiling**. It was not
reading the weights.

The cause was a real defect, not a benchmark artifact, and it is fixed in
`qwen38_affine_gate_up_launch`: `Affine2Geo::Bitcast` has no unfused kernel and its arm fell
back to the production **SwiGLU** kernel, whose buffer layout the unfused caller does not bind —
`rows` was read from the up-output buffer, so `row < rows` failed for nearly every thread and
the dispatch returned fast and wrong. See the commit for the guard: no geo may resolve a
`with_swiglu=false` request to a swiglu-named kernel.

Without an independent physical bound, 0.226x would have read as a clean refutation of the whole
campaign.

## The measurement, with a baseline that reads the weights

All three arms are now under the roof:

| arm | GPU ns | positions | GB/s | per position |
|---|---|---|---|---|
| pair baseline (unfused, fixed) | 9,321,041 | 1 | 382.5 | 9,321,041 ns |
| SwiGLU-fused production | 7,183,124 | 1 | 496.3 | 7,183,124 ns |
| chunked r4k4 | 15,348,083 | 4 | 232.3 | 3,837,021 ns |

    fusion-matched, pair vs chunked      2.429x   <- what BATCHING alone buys
    against the fused path that runs     1.872x

**2.429x is the honest batching number** — same unfused pair kernel on both sides, one position
against four, on real catalog weights across all 64 layers, with weight traffic cut exactly 4x.

**1.872x is what you would get today** by replacing the fused production path with the batched
unfused kernel.

## The gap between them is the lever

Production's SwiGLU fusion plus its bitcast unpack are worth **1.298x** on their own
(9,321,041 → 7,183,124 ns). The multi-position kernel has neither. So:

    batching alone                          2.429x
    production's fusion + bitcast alone      1.298x
    batching, minus what it gives up         1.872x

**A SwiGLU-fused, bitcast multi-position kernel is the thing that turns 1.872x into ~2.4x in
production.** That is a concrete next kernel, not a hope: both halves are already written
separately.

The chunked arm runs at 232.3 GB/s against a 778.8 roof — 30%, where production reaches 63%.
Having cut traffic 4x it is no longer bandwidth-bound, so more positions will not help it and
the remaining headroom is arithmetic, which is exactly what fusion buys.

Stability: the fused baseline measured 7,248,624 ns in the first pass and 7,183,124 ns here,
0.9% apart, on separate quiet lanes.

## Caveats, stated rather than buried

* **This is a timing measurement only.** The isolated-organ instrument runs whatever the shared
  workspace holds, so it cannot check per-position outputs. Correctness for these kernels was
  established by CP4 — bit-identical at the real gate_up shape against the production affine
  matvec, with a negative control observed to reject — and CP6a does **not** re-establish it.
* **1.892x is an upper bound, not a matched comparison.** The production baseline fuses SwiGLU
  and the chunked kernel does not, so the baseline does slightly more work. The honest
  fusion-matched number needs a pair baseline that actually reads the weights, which is now a
  known defect rather than an assumption.
* Still a single organ. Not a prompt wall.

## What this changes on the frontier

The claim "batching wins on the real weights" is measured rather than projected: **2.429x**
fusion-matched for the largest organ, on the session's own catalog across all 64 layers.

`CP5_DECOMPOSITION.md` used 2.49x for gate_up, taken from CP4's single-layer harness. The
real-weight fusion-matched figure is 2.429x, so that projection holds almost exactly — but the
number that would apply to a drop-in replacement of the production path today is **1.872x**,
because the multi-position kernel gives up the SwiGLU fusion and bitcast unpack that production
already has.

Frontier items found here:
1. **Fixed:** `Affine2Geo::Bitcast` resolved an unfused gate_up request to a SwiGLU kernel with
   a mismatched ABI — production-reachable via `HAWKING_QWEN38_FUSE_MLP=pair`, and any past A/B
   run in that mode on a bitcast-geo catalog was measuring garbage.
2. **A SwiGLU-fused, bitcast multi-position gate_up kernel.** Worth 1.298x on top of the 1.872x,
   and both halves already exist separately. This is the highest-value next kernel.
3. The chunked organ is compute-bound at 30% of roof, so more positions will not help it — which
   is the same finding from the other direction.
