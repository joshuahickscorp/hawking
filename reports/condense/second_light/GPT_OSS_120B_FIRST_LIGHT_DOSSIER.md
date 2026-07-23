# GPT-OSS-120B FIRST-LIGHT CALIBRATION DOSSIER

schema `hawking.second_light.first_light_calibration.v1`  sha256 `33ba88b136c30df2`  sealed 2026-07-18T23:52:58Z

## Classification

**FIRST-LIGHT CALIBRATION** (not FULL RUN).

> This dataset selected the next representation family. It did not constitute a full-model condensation or capability run. Low-rank ternary factorization is REJECTED as the principal geometry; the next run uses a full-rank Product/Vector Quantization family with shared amortized codebooks and a protected-island / residual Doctor reserve.

## What it was

- parent: `openai/gpt-oss-120b @ b5c939de`
- scope: layer 0, 128 experts, 256x256 bounded slices (mlp1 MXFP4)
- representation: low-rank ternary factorization (rejected) + PQ selection probe
- code commit: `7f237ed36a64`

## Plateau diagnosis (why ternary was rejected)

- effective rank: approximately 104 / 256 for 90% energy (experts are HIGH-RANK)
- SVD rank-8 error: 0.916  (rank-64: 0.51)
- ternary rank-8 error: 0.9854774866253138
- conclusion: of the ~0.99 error, low-rank truncation accounts for ~0.92 and ternary sign-quantization adds only ~0.07. The bottleneck is RANK, not quantization precision. Richer levels at the same low rank cannot help.

## Family selection (evidence-driven)

- selected: PRODUCT/VECTOR QUANTIZATION with a codebook shared+amortized across experts (full-rank, sub-bit), optionally hybridized with a protected-island outlier reserve for the heavy tail. Low-rank latent factorization is REJECTED for these high-rank experts.
- PQ subdim8 K256 @ 1 BPW rel-error: 0.5425698049366474
- ternary rank-8 rel-error: 0.9854774866253138
- PQ beats ternary on 32/32 experts (~approximately 45 percent lower error)

## Residual structure

- heavy-tailed: True  kurtosis: 5.19
- implication: a small protected-island outlier reserve captures disproportionate residual

## Divergence (bounded slice, proxy)

- untreated: {'max': 1.008770548002912, 'mean': 0.9950891258086177, 'min': 0.9765087689563521, 'sd': 0.005576657019256403}
- treated: {'max': 0.9190994910017254, 'mean': 0.8684329149033003, 'min': 0.8244588763686855, 'sd': 0.020542707998828225}
- output-space F2 proxy (true residual, real tokens): {'true_residual_mean_output_rel_div': 0.68792, 'activation_aware_mean': 0.65088, 'capability_parity': False}

## Honesty boundary

- is capability claim: False   is event horizon: False   authorizes escape: False
- weight error is a PROXY; no full-model capability was measured or claimed.
