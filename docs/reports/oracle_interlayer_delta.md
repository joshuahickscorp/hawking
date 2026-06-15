# Oracle — §8.1 L1.3 cross-layer weight delta-encoding

**Verdict: NO-GO**

- Model: `models/qwen2.5-3b-instruct-q4_k_m.gguf` (36 layers)
- Mode: SAMPLED; pairs analyzed: [(0, 1), (17, 18), (34, 35)]
- Peak RSS during run: 1.59 GB
- Q4_K baseline = 4.5 bits/weight; Q6_K = 6.5625 bits/weight (verified from gguf GGML_QUANT_SIZES).

## Deciding numbers (means across analyzed pairs)

- Mean cosine(W[L], W[L+1]) across all types: **+0.0003** (GO gate: >= 0.3).
- Mean std(delta)/std(W[L+1]): **1.610** (> 1.0 means the delta needs MORE bits than the original at equal error).
- Mean top-64 SVD energy of the delta: **0.230** (fraction of delta energy in its 64 largest singular values; min(dim) is 256-2048, so 64 is a generous rank budget).
- Tensor types with mean cosine >= 0.3: 0/7.
- Tensor types where a delta encoding beats Q4_K/Q6_K bytes at equal error (majority of pairs): 0/7.

### Affine-delta sanity (forecloses a smarter codec)

A delta codec could learn a per-layer gain: store `W[L+1] ≈ α·W[L] + D` and pick
the least-squares-optimal α. Checked directly on q_proj/up_proj/down_proj pairs:

- optimal `α* ≈ 0` in every case (e.g. q_proj 0→1: α*=+0.0007, up_proj 0→1: α*=−0.0000),
  so the best affine model collapses to "store W[L+1] raw, ignore W[L]".
- `std(D_scaled)/std(W[L+1]) = 1.0000` exactly — the optimally-scaled residual is
  *no easier* to quantize than the original. There is no gain term that helps.

Mean-centered (Pearson) cosine is also ~0 (q_proj 0↔1: 0.00085; 0↔35: 0.00047),
ruling out a hidden DC-offset correlation. Self-cosine = 1.0 confirms the metric.

## Per-tensor-type summary

| proj | pairs | cos mean | cos range | std(D)/std(W) | delta top-64 E | base bpw | delta best bpw | delta wins |
|------|-------|---------|-----------|---------------|----------------|----------|----------------|------------|
| q_proj | 3 | +0.0006 | [+0.000, +0.001] | 1.478 | 0.243 | 4.500 | 5.061 | 0% |
| k_proj | 3 | -0.0027 | [-0.007, -0.001] | 1.464 | 0.513 | 4.500 | 5.048 | 0% |
| v_proj | 3 | +0.0004 | [+0.000, +0.001] | 1.388 | 0.373 | 5.875 | 6.347 | 0% |
| o_proj | 3 | -0.0002 | [-0.001, +0.001] | 1.402 | 0.174 | 4.500 | 4.988 | 0% |
| gate_proj | 3 | +0.0043 | [+0.001, +0.007] | 1.750 | 0.122 | 4.500 | 5.254 | 0% |
| up_proj | 3 | -0.0001 | [-0.000, -0.000] | 1.849 | 0.096 | 4.500 | 5.320 | 0% |
| down_proj | 3 | -0.0002 | [-0.001, +0.000] | 1.942 | 0.092 | 5.875 | 6.752 | 0% |

## Interpretation

L1.3 dies cheaply on Qwen2.5-3B with zero kernel written, because consecutive layers are essentially uncorrelated (mean cosine +0.0003, far below the 0.3 gate) — there is no shared structure for a delta to exploit; the delta has HIGHER variance than the original (std ratio 1.61 >= 1.0), so quantizing D at equal error costs MORE bits than storing W[L+1] directly — the delta is anti-compressible, not compressible; the delta is full-rank (only 23.0% of its energy in the top-64 singular values), so a low-rank store of D cannot approach the Q4_K byte budget either; no delta encoding beats the native quant bytes at equal error in the majority of tensor types (0/7).

This is the expected outcome for a well-trained transformer: each layer learns a distinct transform, so W[L+1] - W[L] is roughly the difference of two near-independent random-looking matrices, whose variance adds (std grows by ~sqrt(2)) and whose spectrum stays flat. Same discipline that killed block-256 FFN sparsity: measured, not assumed.

Note: the std-ratio comparison is range-model-independent — the assumed quant range k*std cancels, so the verdict rests only on the measured std(D) vs std(W) and the measured SVD spectrum, not on any tunable constant.

## Method

- Source: the Q4_K_M GGUF the engine actually serves; each tensor dequantized to f32 via gguf's own type-trait dequantizer (Q4_K/Q6_K). Measuring structure of the served weights is correct here.
- cosine: dot(flat W[L], flat W[L+1]) / (||.|| ||.||).
- delta low-rank: SVD via eig of the smaller Gram (min-dim^2); top-r energy = sum of r largest sigma^2 / total. Low-rank store cost = 16*r*(m+n)/(m*n) bits/weight at fp16 factors.
- delta quant cost at equal error: a uniform quantizer hits a target RMSE with bits = log2(range / (RMSE*sqrt(12))); holding the target = Q4_K's error on W[L+1], the delta needs base_bpw + log2(std(D)/std(W)) bits. std(D) > std(W) => more bits, i.e. anti-compressible.
- RAM: one pair resident at a time, del + gc.collect() between pairs; peak RSS 1.59 GB.
