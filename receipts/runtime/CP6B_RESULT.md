# CP6b — fusion and batching attack the same cost. They do not compose.

Raw: `receipts/runtime/CP6B_FUSED_MULTIPOSITION.json`
Kernel: `qwen_affine_q2_group64_matmul_gate_up_swiglu_r{R}k{K}_geo_tpr64_tg128_bitcast`

CP6A_RESULT ended with a prediction, stated as the highest-value next kernel:

> A SwiGLU-fused, bitcast multi-position gate_up is what turns 1.872x into ~2.4x in production.
> That is a concrete next kernel, not a hope: both halves are already written separately.

The kernel was written. **The prediction is refuted.**

## Measured, real catalog weights, 64 layers, one quiet lane

| arm | GPU ns | positions | per position |
|---|---|---|---|
| unfused pair baseline | 9,379,499 | 1 | 9,379,499 |
| SwiGLU-fused + bitcast production | 7,194,124 | 1 | 7,194,124 |
| chunked r4k4, unfused | 15,277,875 | 4 | 3,819,469 |
| chunked r4k4, **fused + bitcast** | 15,224,750 | 4 | 3,806,188 |

    fusion + bitcast at ONE position       1.304x   (9,379,499 -> 7,194,124)
    fusion + bitcast at FOUR positions     1.003x   (15,277,875 -> 15,224,750)

    predicted   1.872x -> ~2.4x
    measured    1.884x -> 1.890x

Adding fusion and the bitcast unpack to the batched kernel bought **0.35%**.

## Why — and it is not a subtlety, it is arithmetic

Both optimisations attack the **per-unpacked-weight** cost, and batching has already amortised
exactly that.

    single position   1 FMA per weight unpacked   -> the unpack is 1/1 of the arithmetic
    r4k4             16 FMAs per weight unpacked  -> the unpack is 1/16 of it

The bitcast replaces an integer→float convert with an OR and a reinterpret. At one FMA per
weight that is a large share of the inner loop and worth 1.304x. At sixteen FMAs per weight the
same saving is a sixteenth as significant, and it disappears into noise.

The SwiGLU fusion saves writing one output buffer instead of two: 69,632 B per layer at one
position, 139,264 B at four — against 55,705,600 B of weights per layer, **0.12% and 0.25%**.
It was never going to be worth 1.3x at either width; the 1.304x measured at one position is the
bitcast, not the fusion.

**Batching and unpack-optimisation are substitutes, not complements.** Both make the weight
sweep cheaper per weight. Having done one, the other has little left to take.

## What this corrects

`CP6A_RESULT.md`'s closing recommendation is wrong and is superseded here. The gap between
2.456x (fusion-matched) and 1.890x (against the path that runs) is **not** recoverable by
porting fusion into the batched kernel. It is the honest cost of comparing against a baseline
that is already 1.304x better than the unfused form the batched kernel descends from.

**1.89x is the ceiling for this organ via batching**, on real weights, against the path
production actually runs. Not 2.4x.

## What that does to the projection

`CP5_DECOMPOSITION.md` projects 1.79-1.89x on step GPU ns from batching the position-independent
78.7%, using per-organ speedups of ~2.3-2.5x. Those were measured fusion-matched or on harness
buffers. Against the paths production really runs, gate_up delivers 1.89x, not 2.49x — so the
step-level projection should be re-derived from the production-relative figures, and it will come
out lower. The other organs have not been measured production-relative yet; gate_up is the only
one with a real-weights, real-baseline number.

## Where the remaining headroom is, if anywhere

The chunked arm sits at 232 GB/s against a 778.8 GB/s roof — 30%, where the fused production path
reaches 496 GB/s, 64%. Having cut weight traffic 4x it is bound by something other than DRAM, and
CP6b now rules out unpack cost as that something: removing the convert changed nothing. The
remaining candidates are register pressure at 32 live accumulators, threadgroup reduction cost
(`8*R*K` floats, and the reduction loop runs `R*K` simd_sums), and occupancy. Measuring which
would be the next honest step, and CP4b's finding — that grid width against core count governs,
not register count — is the standing prior to test against.
