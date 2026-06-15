# STRAND decode parallelism — the serial-trellis wall is NOT fundamental (research findings, 2026-06-09)

_Deep-research result (102 agents, adversarially verified — 21 claims confirmed, 4 killed). This
**corrects** the earlier session conclusion that "the serial 256-step trellis chain is a fundamental
wall." It is not. There is peer-reviewed, bit-exact, **integer-compatible** math that breaks it._

## The finding (Axis 1 — decode speed, strongly answered)

The serial decode recurrence `state[i] = f(state[i-1], symbol[i])` can be written **exactly** as a
chain of matrix-vector products over the **tropical (min-plus / max-plus) semiring**:

```
s_i = A_i ⊙ s_{i-1}      (A⊞B)_ij = ⋀_k (A_ik + B_kj)   neutral: +∞ (add), 0 (mul)
```

This semiring product is **associative** — the exact algebraic precondition for a **parallel-prefix /
scan**. The reformulation is **arithmetic-agnostic: it holds identically in integer arithmetic**, so it
is **COMPATIBLE with STRAND's integer-deterministic decode**. (Maleki PPoPP'16 / CACM; Theodosis–Maragos
SPAWC'18. 3-0 verified.)

### Bit-exact parallel algorithms
- **Maleki rank-convergence** (data-dependent, the demonstrated speedup): `rank(A⊙B) ≤ min(rank A, rank B)`
  in the tropical semiring, so partial-product rank is non-increasing and converges to 1 in ~30–112
  stages for real Viterbi codes; a rank-1 product sends any non-zero start vector to a result *parallel*
  (equal up to an additive constant) to the truth, and since `arg max` is shift-invariant, parallel
  vectors give the **identical traceback**. A fix-up loop repairs non-converged stages, worst-case
  degrading to sequential but **never wrong**. Exact iff **integer arithmetic + deterministic
  tie-breaking**. **Measured: up to ~24× on 64 cores** vs a SIMD baseline. (3-0 verified.)
- **Associativity-based exact SCAN** (data-independent, bit-exact *by construction*): the Blelloch scan
  over the transition matrices. The moat-safe route — but pays an `O(nk²)` / ~half-throughput "Scan
  penalty" (Maleki–Burtscher ASPLOS'18).

### THE strategic kicker — the moat *is* the speed precondition
The float variant of this parallelization was **REFUTED 0-3**: FP addition is non-associative, so a
parallel float recurrence is **not** bit-identical (≈1e-3 drift). **Bit-exact parallel decode requires
integer-only arithmetic** — which is *exactly* STRAND's float-free moat. Float-codebook trellis quants
(QTIP-with-float-LUT, etc.) **cannot** do the bit-exact parallel decode; STRAND can. The determinism
moat is not just "runs anywhere" — it is the precondition for a parallel decode the float competitors
are mathematically locked out of.

### Practical routes (verified)
| route | parallelism | bit-exact? | determinism verdict |
|---|---|---|---|
| **Windowing / overlap-save / tail-biting** | n/f tiles (multi-Gb/s GPU) | empirically exact; bit-exact **iff** overlap sized for provable sufficient-history (tail-biting) | COMPATIBLE with a throughput cost — STRAND must add the provable sizing |
| **Bitslicing** (batch, 32 streams/word) | inter-stream (= our block-parallel, extended to GPU) | YES — pure integer bitwise, 21.4 Gbps on V100 | ✅ fully moat-safe; does NOT reduce single-chain latency |
| **Tropical exact scan** | intra-chain | YES by construction | ✅ but ~half throughput (Scan penalty) |
| **Maleki rank-convergence** | intra-chain, ~24× | YES (int + det. ties), but data-dependent fix-up | ✅ *if* convergence is bounded for STRAND's trellis (open Q) |
| ~~Tensor cores~~ | per-stage float matmul only | NO (float; leaves CSelect+traceback serial) | ❌ off the deterministic path |

## Axis 2 (compression) — one result, mostly open
**TCQ is `O(2^L·T)` — linear in dimension, independent of bitrate** — vs VQ's `2^(kd)` blowup that caps
VQ at dimension ≤8. This complexity asymmetry is the mechanism that lets trellis VQ reach the **>100-D
regime where the space-filling/shaping gap closes** (QTIP: >3× reduction of the distortion gap to an
optimal 2-bit quantizer). *Open:* whether the gap-closing is realizable with a **frozen integer LUT** vs
needing trained/float codebooks; the quantitative shaping-gain limit (~1.53 dB / 0.255 bit), ECTCQ, and
incoherence-as-Gaussianizer were **not** covered by surviving claims.

## Axis 3 (side-info / fixed-point) — ZERO confirmed claims (open)
Minimum side-info, predictive/entropy scale coding, fixed-point state/scale precision bounds, and
incoherence-transform ops/quality were not answered — a follow-up research pass should target them.

## The concrete next experiments
1. **Measure STRAND's transition-matrix rank-convergence depth.** If STRAND's specific trellis provably
   reaches rank-1 within a bounded, fixed number of stages, a **fixed-overlap blocked decode is
   guaranteed exact without the data-dependent fix-up** — converting the 24× into a *deterministic*
   guarantee (open question #4). This is a small, decisive measurement.
2. **Implement + bench a windowed/tail-biting parallel decode** (sufficient-history overlap), CPU and
   GPU, vs the current 6× block-parallel baseline — does any exact-parallel route actually *beat* it
   once the determinism constraint is paid? (open question #3.)
3. **Follow-up research pass on Axes 2 & 3** (shaping-gain math, side-info minimum) — the gaps above.

## Key sources
Maleki et al., "Parallelizing dynamic programming through rank convergence" (PPoPP'16 / CACM) ·
Theodosis & Maragos, "Analysis of the Viterbi Algorithm Using Tropical Algebra and Geometry" (SPAWC'18) ·
Maleki & Burtscher, work-efficient parallel linear recurrences (ASPLOS'18) · GPU block-parallel Viterbi
(arXiv:1608.00066) · windowed/overlap GPU decode (arXiv:2011.09337) · bitsliced Viterbi (ACM TOPC
10.1145/3470642) · QTIP (NeurIPS'24, arXiv:2406.11235). Full claim-level evidence + votes in the
research task output.
