# STRAND compression + side-info — how much of the density gap is recoverable deterministically (research, 2026-06-09)

_Follow-up deep-research (104 agents, adversarially verified: 21 confirmed / 4 killed; settled 30-yr theorems
+ recent LLM-quant primaries). Answers the Axis 2/3 gaps the first pass left open. Bottom line: **the
granular ~0.1-0.25 bit/weight of the gap is recoverable with a FROZEN integer LUT; the shaping/boundary tail
+ the entropy-coding tail are the parts that resist a pure fixed-LUT decode.**_

## The gap, decomposed (Axis 2)
The ~0.2-0.35 bit/weight excess above the Shannon R-D bound splits **additively** (high-rate) into:
- **granular gain** — Voronoi-cell shape (the quantizer's fine structure), and
- **shaping/boundary gain** — the support-region shape,

each independently capped at **πe/6 = 1.53 dB ≈ 0.255 bit/sample** (Eyuboglu–Forney; derived as the
power ratio of a uniform vs a same-entropy Gaussian). It is an **n→∞ ceiling** — finite dimension realizes
well under it.

## How fast the gap closes (the money numbers)
- **Lattice shaping by dimension:** E8 (n=8) 0.65 dB · Barnes-Wall (n=16) 0.86 dB · Leech (n=24) 1.03 dB —
  even n=24 leaves ~0.5 dB (~0.083 bit) to the limit. Slow.
- **Trellis shaping by state count:** ~1 dB from a 4-state code, up to ~1.36 dB (Marcellin-Fischer/Forney);
  state count grows **very rapidly** toward 1.53 dB.
- **Realizable TCQ/ECTCQ gaps (canonical):** uniform source — TCQ within **0.21 dB (~0.035 bit)** of R-D;
  Gaussian source — a simple **8-state ECTCQ within ~0.5 dB (~0.08 bit)** at all rates. Larger trellises keep
  helping (256-state ECTCQ ~0.2 dB) — "8 states saturates" was **refuted**.
- **The LLM-quant headline (QTIP, NeurIPS'24), i.i.d. Gaussian @ 2 bpw:** scalar Lloyd-Max 0.118 MSE (~87%
  above bound) · 8D VQ (QuIP# E8P) 0.089 (~41%) · **high-dim L=16 (65,536-state) TCQ 0.069 (~9.5%, well inside
  0.1 bit)** · bound D_R = 0.063. With scalar V=1, TCQ provably **approaches D_R as state count grows**, and the
  Viterbi cost is **O(2^L·T)** — linear in dimension, independent of bitrate. **This is the lever: high state
  count / high dimension closes the granular gap, and it's exactly what STRAND's `gate-vectrellis` probe tests.**

## The determinism split (the load-bearing result for STRAND)
| component | recoverable with a FROZEN integer LUT decode? | gain |
|---|---|---|
| **Granular gain** — dense high-state/high-dim trellis cells + a **Gaussian-optimal (Lloyd-Max) companded codebook frozen at encode** | **YES — fully** (decode is integer LUT lookup) | most of ~0.1-0.25 bit/weight; a frozen Gaussian-optimal scalar codebook alone cuts MSE **54% vs absmax @ 3-bit** (PolarQuant) |
| **Shaping/boundary gain** (approach to 1.53 dB) | partially — finite-state trellis shaping is LUT-decodable, but the last ~0.5 dB is exponentially expensive in states | diminishing |
| **Entropy-coding tail** (ECTCQ's sub-R rate = conditional entropy of the codebook) | **NO** — needs a variable-length stage; recoverable deterministically only via an **integer range/ANS decoder** (still float-free, but not a pure LUT) | ~0.5→0.2 dB of the ECTCQ gain |
| **LDGM last 0.04-0.09 dB to 1.53 dB** | encode-side only (thousands of BP iters); decode unaffected | negligible (<0.02 bit/weight) |

⚠️ **Refuted, do not assume:** QTIP's "1MAD/3INST computed codes give a lookup-free frozen integer decode at
SOTA quality" was **refuted 0-3** — its specific computed-code decode-realizability is unconfirmed. STRAND's
**frozen LUT** is the confirmed determinism-compatible path (the granular row above).

## Side-info + incoherence (Axis 3)
- **Incoherence:** the **fast Walsh-Hadamard transform** Gaussianizes each coordinate to ≈N(0,1/d)
  (KS < 0.01 at d=128) at **O(d log d), no multiplies** — the cheapest incoherence option. Caveat: deterministic
  WHT has no *worst-case* incoherence guarantee; QuIP#/QTIP use the **randomized** Hadamard — STRAND can bake the
  random ±1 signs as a **frozen seed** (decode stays deterministic) to get the guarantee.
- **Scales:** per-block scale side-info + a **globally-shared centroid table** is the determinism-compatible
  normalization path (matches STRAND's structure). **Open:** the information-theoretic *minimum* scale rate, the
  gain from predictive/hierarchical/entropy-coded scales, and how coarsely scales can be quantized before PPL
  drops — not quantified by the verified set.
- **Open:** a provable fixed-point bit-width sufficiency theorem (state/scale/accumulator precision for a
  bit-exact + distortion-lossless decode) did not surface.

## What this means for STRAND — concrete, determinism-safe levers
1. **Push the trellis state count / vector dimension** (the `gate-vectrellis` experiment): the granular gain is
   the bulk of the recoverable gap and is frozen-LUT-compatible — QTIP lands within ~0.1 bit at high state count.
2. **Use a Gaussian-optimal (Lloyd-Max) frozen companded codebook** — a large (~54% MSE) determinism-safe win
   over absmax/uniform; ensure STRAND's frozen codebook is Gaussian-optimal, not uniform.
3. **Keep the FWHT incoherence**, bake the randomized signs as a frozen seed for the worst-case guarantee.
4. **Optional integer ANS stage** to recover the entropy-coding tail (~0.2-0.3 dB) while staying float-free —
   a deliberate "not a pure LUT, still deterministic" extension (decode-speed cost to weigh).
5. **Accept the residual:** the approach-to-1.53 dB shaping tail + heavy-tail residual is the part determinism
   genuinely leaves on the table; the honest recoverable budget is the **granular ~0.1-0.25 bit/weight**.

## Still open (a third research pass or measurement)
Residual heavy-tail gap on real LLM tensors after randomized-Hadamard (no verified number); the minimum
scale-side-info rate + predictive-scale savings; the fixed-point precision sufficiency theorem; and the
integer-ANS entropy-stage decode-speed cost.

## Key sources
Eyuboglu & Forney 1993 (lattice/trellis quantization, the shaping/granular decomposition + 1.53 dB) · Forney
MIT 6.451 Ch.14 · Marcellin & Fischer 1990 + Marcellin 1994 (TCQ/ECTCQ realizable gaps) · Kurkoski 2016
(lattice shaping by dimension) · Wang & He 2007 (LDGM, approaching 1.53 dB) · QTIP NeurIPS'24 (high-dim TCQ
R-D numbers) · PolarQuant 2026 (frozen Gaussian-optimal LUT, FWHT Gaussianization). Full claim-level
evidence/votes in the research task output.
