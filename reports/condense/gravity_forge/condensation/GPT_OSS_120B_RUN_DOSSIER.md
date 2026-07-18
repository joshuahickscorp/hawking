# GPT-OSS-120B — scientific run dossier (layer 0, 128 experts)

Source: `openai/gpt-oss-120b` @ `b5c939de` (verified). Scope: layer 0, all 128 experts, `mlp1_weight`
(MXFP4) read directly, bounded 256×256 reference slices. This is an **execution + infrastructure proof**,
not a quality solution — divergence remains high and no full-model capability is claimed.

## 1. What the run showed (128 sealed checkpoints)

| metric | mean | sd | range |
|---|--:|--:|--:|
| untreated divergence | 0.9951 | 0.0056 | 0.977 – 1.009 |
| treated divergence (Doctor) | 0.8684 | 0.0205 | 0.825 – 0.919 |
| Doctor improvement | 0.1267 | 0.0197 | ≤ 0.176 (all ≥ 0) |

- Rank selected by the sub-bit ladder: **r=4 for 98 experts, r=8 for 30** — the ladder mostly refused to
  climb, because higher rank did not lower divergence enough to matter. That is the first clue.
- All 128 checkpoints stayed sub-bit (0.9895–0.9897 BPW); all 134 seals valid.
- Worst experts (treated div): 30 (0.919), 80 (0.915), 56 (0.911), 111 (0.908), 104 (0.906).

## 2. Why the ternary latent family plateaus (the key finding)

**The GPT-OSS experts are high-rank; low-rank factorization is the wrong geometry.**

On the real weights (SVD of the 256×256 slices):

| quantity | value |
|---|--:|
| effective rank for 90% energy | **≈ 104 / 256** |
| best possible rank-8 error (true SVD) | 0.916 |
| ternary rank-8 error | 0.986 |
| **ternary quantization penalty** (ternary − SVD) | **0.070** |
| best possible rank-64 error (true SVD) | 0.51 |

Of the ~0.99 rank-8 error, **low-rank truncation accounts for ~0.92 and ternary sign-quantization adds
only ~0.07**. The bottleneck is **rank, not quantization precision** — richer codebooks or more levels
*at the same low rank* cannot help, and even rank-64 SVD is still 0.51 (and no longer sub-bit as a
factorization). This is exactly why the ladder stayed at r=4.

## 3. Residual structure

After the rank-8 fit, the residual is **heavy-tailed (kurtosis 5.2, vs 3 for Gaussian)** with mild column
concentration (top-10% of columns hold ~15% of residual energy). Implication: a small **protected-island
outlier reserve** captures disproportionate residual and is a useful *secondary* lever — but it does not
address the primary rank deficiency.

## 4. Next Forge family — selected from evidence

Because the experts are high-rank, the winning geometry must be **full-rank** and compress along the
element/block axis, amortizing a shared codebook below one bit. A bounded Product-Quantization experiment
on the same real experts (sub-vectors of dim 8, K=256 centroids, codebook shared/amortized across experts):

| family | rel-error @ ~1.0 BPW amortized |
|---|--:|
| ternary rank-8 (current) | 0.985 ± 0.001 |
| **Product Quantization (sd=8, K=256)** | **0.543 ± 0.010** |

**PQ beats ternary by 45%, consistently across all 32/32 experts tested.** At 0.5 BPW amortized (K=16) PQ
still reaches ~0.785 — better than ternary at 0.99. PQ is full-rank (it quantizes sub-vectors, not a rank
subspace), so it does not hit the truncation wall.

**Selected next family: Product / Vector Quantization with a shared, amortized codebook** (full-rank,
sub-bit), optionally hybridized with a **protected-island outlier reserve** for the heavy tail. Low-rank
latent factorization is **rejected** for these high-rank experts.

## 5. Recommended next step (bounded, not a huge run)

Implement PQ as a first-class Forge family in the Seed's Forge registry (inspect/fit/pack/measure/execute/
validate/repairability), wire it through Candidate C's direct compact operator, then run **one bounded
layer-level PQ campaign** (layer 0, 128 experts, real output-divergence eval) to confirm the ~0.54 weight
error translates to a materially lower functional divergence — before expanding to all 36 layers or
attempting a capability run. The architecture stays frozen; only a new representation family is added.

## Boundary of the claim

This dossier proves Hawking can *scientifically interrogate* a real 120B parent — measure rank, decompose
error, and select the next geometry from evidence — under one controller with sealed evidence. It does not
claim GPT-OSS-120B is compressible below one bit with retained function; the answer to *that* question is
the PQ campaign this dossier motivates.
