# Q30 activation-aware family probe

- Recorded: `2026-08-09T19:36:08.950726Z`
- Status: `EARNED_ACTIVATION_AWARE_FAMILY_PROBE_COMPLETE_LOCAL_ONLY_OR_NEGATIVE`
- Ceiling: component BPW ≤ 1.5
- Coherence bar (surplus-first): output_cos ≥ 0.9, surplus ≥ 0.1, operator-recovery weight_cos ≥ 0.5
- Mean null (high-hit under ceiling): `0.9421903472852479`

## Verdict

MIXED/NEGATIVE for promotion: high-hit experts clear a surplus-first local output bar under BPW <= 1.5, but only as distribution-local matches (weight cosine < 0.5). Not evidence of a coherent full-model artifact. First surplus-first coherence-grade point (any expert, any BPW) at component_bpw=0.2037 (activation_pca_low_rank_q/r64_b3/up_proj expert 127, out=0.9265, null=0.4934, surplus=+0.4331, wt=0.1521). On high-hit experts, no surplus-first coherence row also recovers the operator (weight_cos>=0.5). Best high-hit weight cosine is 0.9846 at bpw=4.8698 (raw_weight_low_rank_q/r640_b4, surplus=+0.0223). Low-hit footnote: operator-recovery+surplus first appears at component_bpw=1.4609 on expert 127 (activation_weighted_svd_low_rank_q/r192_b4). Family ranking on real activations DOES invert vs weight-space: activation-PCA low-rank beats raw-weight low-rank on surplus-over-null for high-hit experts. Raw-weight low-rank typically fails to beat the constant-mean null on this capture. CRITICAL NULL TRAP: mean constant-mean null on high-hit under-ceiling rows is 0.9422. Absolute output cosine without null subtraction is inadmissible here (prior campaign constant-mean null of ~0.90). Best high-hit under-ceiling surplus=+0.1035 (out=0.9950, null=0.8915, wt=0.3813, bpw=1.4609) via activation_pca_low_rank_q/r192_b4 on gate_proj expert 1. On high-hit experts, within the tested grid (up to rank-640 / ~4+ BPW component), no point jointly clears surplus-first coherence and the operator-recovery weight-cosine cutoff. Exact BPW for joint high-hit reachability is ABOVE the highest tested anchor for this three-prompt capture. Footnote (low-hit experts only, not primary): joint surplus+operator first seen at component_bpw=1.4609 on expert 127 (activation_weighted_binary_residual/r192_b4); small-N activations can inflate surplus.

## Family summary — high-hit experts only (primary)

| family | n | mean out-cos | mean wt-cos | mean null | mean surplus | frac beats null | best surplus | local coh? | operator coh? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `activation_weighted_svd_low_rank_q` | 36 | 0.9812 | 0.4640 | 0.9422 | 0.0390 | 1.0000 | 0.0943 | no | no |
| `activation_pca_low_rank_q` | 36 | 0.9616 | 0.3852 | 0.9422 | 0.0194 | 0.7222 | 0.1035 | yes | no |
| `activation_weighted_binary_residual` | 63 | 0.8752 | 0.5522 | 0.9422 | -0.0670 | 0.1111 | 0.0195 | no | no |
| `raw_weight_low_rank_q` | 36 | 0.7873 | 0.6806 | 0.9422 | -0.1549 | 0.0000 | -0.0018 | no | no |

## Family summary — all selected experts (under ceiling)

| family | n | mean out-cos | mean wt-cos | mean null | mean surplus | frac beats null | best surplus | local coh? | operator coh? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `activation_pca_low_rank_q` | 57 | 0.9537 | 0.3139 | 0.8247 | 0.1290 | 0.8246 | 0.4709 | yes | no |
| `activation_weighted_svd_low_rank_q` | 48 | 0.9727 | 0.4677 | 0.8625 | 0.1102 | 1.0000 | 0.4831 | yes | yes |
| `activation_weighted_binary_residual` | 84 | 0.8689 | 0.5470 | 0.8625 | 0.0065 | 0.3095 | 0.3912 | yes | yes |
| `raw_weight_low_rank_q` | 48 | 0.7798 | 0.6823 | 0.8625 | -0.0827 | 0.1875 | 0.3777 | no | no |

## Full table (family × budget × tensor)

| family | budget | expert | component | bpw | under 1.5? | weight-cos | output-cos | null | surplus | beats null | coh | local-only |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---|---|---|
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 127 | `up_proj` | 1.4609 | yes | 0.6199 | 0.9765 | 0.4934 | +0.4831 | yes | yes | no |
| `activation_pca_low_rank_q` | `r192_b4` | 127 | `up_proj` | 0.2663 | yes | 0.1616 | 0.9643 | 0.4934 | +0.4709 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r384_b4` | 127 | `up_proj` | 0.2663 | yes | 0.1616 | 0.9643 | 0.4934 | +0.4709 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r512_b4` | 127 | `up_proj` | 0.2663 | yes | 0.1616 | 0.9643 | 0.4934 | +0.4709 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r640_b4` | 127 | `up_proj` | 0.2663 | yes | 0.1616 | 0.9643 | 0.4934 | +0.4709 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 127 | `up_proj` | 0.3724 | yes | 0.2077 | 0.9275 | 0.4934 | +0.4341 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r128_b3` | 127 | `up_proj` | 0.2037 | yes | 0.1521 | 0.9265 | 0.4934 | +0.4331 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r256_b3` | 127 | `up_proj` | 0.2037 | yes | 0.1521 | 0.9265 | 0.4934 | +0.4331 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r64_b3` | 127 | `up_proj` | 0.2037 | yes | 0.1521 | 0.9265 | 0.4934 | +0.4331 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 127 | `up_proj` | 0.7448 | yes | 0.4913 | 0.9227 | 0.4934 | +0.4293 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 127 | `up_proj` | 1.4896 | yes | 0.6229 | 0.9201 | 0.4934 | +0.4267 | yes | yes | no |
| `activation_weighted_binary_residual` | `r256_b3` | 127 | `up_proj` | 1.4844 | yes | 0.7927 | 0.8846 | 0.4934 | +0.3912 | yes | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 127 | `up_proj` | 1.5000 | yes | 0.7934 | 0.8845 | 0.4934 | +0.3911 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 127 | `up_proj` | 1.5000 | yes | 0.7934 | 0.8845 | 0.4934 | +0.3911 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 127 | `up_proj` | 1.5000 | yes | 0.7934 | 0.8845 | 0.4934 | +0.3911 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 127 | `up_proj` | 1.4609 | yes | 0.7916 | 0.8845 | 0.4934 | +0.3911 | yes | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 127 | `up_proj` | 1.2500 | yes | 0.7805 | 0.8764 | 0.4934 | +0.3831 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 127 | `up_proj` | 1.2500 | yes | 0.7805 | 0.8764 | 0.4934 | +0.3831 | yes | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 127 | `up_proj` | 1.4896 | yes | 0.8287 | 0.8711 | 0.4934 | +0.3777 | yes | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 127 | `up_proj` | 1.4609 | yes | 0.8322 | 0.8576 | 0.4934 | +0.3642 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 127 | `gate_proj` | 1.4609 | yes | 0.6283 | 0.9888 | 0.6545 | +0.3342 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 127 | `gate_proj` | 0.3724 | yes | 0.2147 | 0.9605 | 0.6545 | +0.3060 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 127 | `gate_proj` | 0.7448 | yes | 0.4996 | 0.9570 | 0.6545 | +0.3025 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r192_b4` | 127 | `gate_proj` | 0.2663 | yes | 0.1654 | 0.9565 | 0.6545 | +0.3019 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r384_b4` | 127 | `gate_proj` | 0.2663 | yes | 0.1654 | 0.9565 | 0.6545 | +0.3019 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r512_b4` | 127 | `gate_proj` | 0.2663 | yes | 0.1654 | 0.9565 | 0.6545 | +0.3019 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r640_b4` | 127 | `gate_proj` | 0.2663 | yes | 0.1654 | 0.9565 | 0.6545 | +0.3019 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 127 | `gate_proj` | 1.4896 | yes | 0.6268 | 0.9547 | 0.6545 | +0.3001 | yes | yes | no |
| `activation_pca_low_rank_q` | `r128_b3` | 127 | `gate_proj` | 0.2037 | yes | 0.1557 | 0.9316 | 0.6545 | +0.2771 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r256_b3` | 127 | `gate_proj` | 0.2037 | yes | 0.1557 | 0.9316 | 0.6545 | +0.2771 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r64_b3` | 127 | `gate_proj` | 0.2037 | yes | 0.1557 | 0.9316 | 0.6545 | +0.2771 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 127 | `down_proj` | 1.4609 | yes | 0.5973 | 0.9767 | 0.7221 | +0.2545 | yes | yes | no |
| `activation_weighted_binary_residual` | `r384_b4` | 127 | `gate_proj` | 1.5000 | yes | 0.7896 | 0.9045 | 0.6545 | +0.2500 | yes | yes | no |
| `activation_weighted_binary_residual` | `r512_b4` | 127 | `gate_proj` | 1.5000 | yes | 0.7896 | 0.9045 | 0.6545 | +0.2500 | yes | yes | no |
| `activation_weighted_binary_residual` | `r640_b4` | 127 | `gate_proj` | 1.5000 | yes | 0.7896 | 0.9045 | 0.6545 | +0.2500 | yes | yes | no |
| `activation_weighted_binary_residual` | `r256_b3` | 127 | `gate_proj` | 1.4844 | yes | 0.7888 | 0.9043 | 0.6545 | +0.2498 | yes | yes | no |
| `activation_weighted_binary_residual` | `r192_b4` | 127 | `gate_proj` | 1.4609 | yes | 0.7876 | 0.9030 | 0.6545 | +0.2484 | yes | yes | no |
| `raw_weight_low_rank_q` | `r128_b3` | 127 | `up_proj` | 0.7448 | yes | 0.6984 | 0.7391 | 0.4934 | +0.2457 | yes | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 127 | `gate_proj` | 1.2500 | yes | 0.7741 | 0.8962 | 0.6545 | +0.2417 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 127 | `gate_proj` | 1.2500 | yes | 0.7741 | 0.8962 | 0.6545 | +0.2417 | yes | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 127 | `gate_proj` | 1.4896 | yes | 0.8381 | 0.8826 | 0.6545 | +0.2281 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 127 | `down_proj` | 0.2663 | yes | 0.2629 | 0.9367 | 0.7221 | +0.2146 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r384_b4` | 127 | `down_proj` | 0.2663 | yes | 0.2629 | 0.9367 | 0.7221 | +0.2146 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r512_b4` | 127 | `down_proj` | 0.2663 | yes | 0.2629 | 0.9367 | 0.7221 | +0.2146 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r640_b4` | 127 | `down_proj` | 0.2663 | yes | 0.2629 | 0.9367 | 0.7221 | +0.2146 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 127 | `down_proj` | 0.3724 | yes | 0.1669 | 0.9299 | 0.7221 | +0.2078 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 127 | `down_proj` | 0.7448 | yes | 0.4390 | 0.9267 | 0.7221 | +0.2046 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 127 | `down_proj` | 1.4896 | yes | 0.6328 | 0.9227 | 0.7221 | +0.2006 | yes | yes | no |
| `raw_weight_low_rank_q` | `r192_b4` | 127 | `gate_proj` | 1.4609 | yes | 0.8419 | 0.8548 | 0.6545 | +0.2002 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 127 | `down_proj` | 0.2037 | yes | 0.2482 | 0.9144 | 0.7221 | +0.1922 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r256_b3` | 127 | `down_proj` | 0.2037 | yes | 0.2482 | 0.9144 | 0.7221 | +0.1922 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r64_b3` | 127 | `down_proj` | 0.2037 | yes | 0.2482 | 0.9144 | 0.7221 | +0.1922 | yes | yes | yes |
| `raw_weight_low_rank_q` | `r256_b3` | 127 | `down_proj` | 1.4896 | yes | 0.7200 | 0.8384 | 0.7221 | +0.1163 | yes | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 127 | `up_proj` | 0.3724 | yes | 0.5557 | 0.6076 | 0.4934 | +0.1142 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 1 | `gate_proj` | 1.4609 | yes | 0.3813 | 0.9950 | 0.8915 | +0.1035 | yes | yes | yes |
| `activation_weighted_binary_residual` | `r384_b4` | 127 | `down_proj` | 1.5000 | yes | 0.0201 | 0.8221 | 0.7221 | +0.0999 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 127 | `down_proj` | 1.5000 | yes | 0.0201 | 0.8221 | 0.7221 | +0.0999 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 127 | `down_proj` | 1.5000 | yes | 0.0201 | 0.8221 | 0.7221 | +0.0999 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 127 | `down_proj` | 1.4792 | yes | 0.0201 | 0.8192 | 0.7221 | +0.0971 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 1 | `gate_proj` | 1.4609 | yes | 0.5875 | 0.9857 | 0.8915 | +0.0943 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 127 | `down_proj` | 1.4583 | yes | 0.0201 | 0.8164 | 0.7221 | +0.0942 | yes | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 127 | `down_proj` | 1.4609 | yes | 0.6896 | 0.8051 | 0.7221 | +0.0829 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 1 | `gate_proj` | 0.3724 | yes | 0.2671 | 0.9726 | 0.8915 | +0.0811 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 1 | `gate_proj` | 0.7448 | yes | 0.3224 | 0.9720 | 0.8915 | +0.0806 | yes | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 1 | `gate_proj` | 1.4896 | yes | 0.3964 | 0.9709 | 0.8915 | +0.0794 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 1 | `gate_proj` | 0.3724 | yes | 0.1745 | 0.9695 | 0.8915 | +0.0780 | yes | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 127 | `gate_proj` | 0.7448 | yes | 0.7088 | 0.7307 | 0.6545 | +0.0762 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 1 | `gate_proj` | 0.7448 | yes | 0.4504 | 0.9671 | 0.8915 | +0.0756 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 1 | `gate_proj` | 1.4896 | yes | 0.5912 | 0.9654 | 0.8915 | +0.0739 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 104 | `gate_proj` | 1.4609 | yes | 0.3665 | 0.9946 | 0.9216 | +0.0730 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 104 | `gate_proj` | 1.4609 | yes | 0.6184 | 0.9939 | 0.9216 | +0.0723 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 104 | `up_proj` | 1.4609 | yes | 0.3643 | 0.9939 | 0.9232 | +0.0707 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 104 | `up_proj` | 1.4609 | yes | 0.6301 | 0.9932 | 0.9232 | +0.0700 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 104 | `gate_proj` | 0.3724 | yes | 0.1808 | 0.9854 | 0.9216 | +0.0639 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 104 | `up_proj` | 0.3724 | yes | 0.1783 | 0.9847 | 0.9232 | +0.0615 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 104 | `gate_proj` | 0.7448 | yes | 0.4777 | 0.9827 | 0.9216 | +0.0612 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 104 | `gate_proj` | 1.4896 | yes | 0.6221 | 0.9805 | 0.9216 | +0.0590 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 104 | `up_proj` | 0.7448 | yes | 0.4872 | 0.9799 | 0.9232 | +0.0567 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 104 | `up_proj` | 1.4896 | yes | 0.6390 | 0.9763 | 0.9232 | +0.0532 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 104 | `gate_proj` | 0.3724 | yes | 0.2393 | 0.9710 | 0.9216 | +0.0495 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 104 | `gate_proj` | 0.7448 | yes | 0.3033 | 0.9705 | 0.9216 | +0.0489 | yes | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 104 | `gate_proj` | 1.4896 | yes | 0.3831 | 0.9695 | 0.9216 | +0.0479 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 104 | `up_proj` | 0.3724 | yes | 0.2382 | 0.9703 | 0.9232 | +0.0471 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 104 | `up_proj` | 0.7448 | yes | 0.3026 | 0.9703 | 0.9232 | +0.0471 | yes | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 104 | `up_proj` | 1.4896 | yes | 0.3813 | 0.9699 | 0.9232 | +0.0467 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 45 | `gate_proj` | 1.4609 | yes | 0.3604 | 0.9941 | 0.9484 | +0.0457 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 45 | `gate_proj` | 1.4609 | yes | 0.6066 | 0.9940 | 0.9484 | +0.0456 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 45 | `gate_proj` | 0.3724 | yes | 0.1691 | 0.9874 | 0.9484 | +0.0390 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 1 | `down_proj` | 1.4609 | yes | 0.6080 | 0.9873 | 0.9489 | +0.0384 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 1 | `up_proj` | 1.4609 | yes | 0.3774 | 0.9949 | 0.9580 | +0.0369 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 45 | `gate_proj` | 0.7448 | yes | 0.4668 | 0.9848 | 0.9484 | +0.0364 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 1 | `up_proj` | 1.4609 | yes | 0.5985 | 0.9933 | 0.9580 | +0.0354 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 45 | `gate_proj` | 1.4896 | yes | 0.6127 | 0.9814 | 0.9484 | +0.0330 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 104 | `down_proj` | 1.4609 | yes | 0.5892 | 0.9889 | 0.9593 | +0.0296 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 1 | `up_proj` | 0.3724 | yes | 0.1679 | 0.9874 | 0.9580 | +0.0294 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 45 | `down_proj` | 1.4609 | yes | 0.5985 | 0.9899 | 0.9616 | +0.0283 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 1 | `down_proj` | 1.4609 | yes | 0.6296 | 0.9767 | 0.9489 | +0.0278 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 45 | `up_proj` | 1.4609 | yes | 0.3600 | 0.9947 | 0.9673 | +0.0274 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 1 | `up_proj` | 0.7448 | yes | 0.4589 | 0.9845 | 0.9580 | +0.0265 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 45 | `up_proj` | 1.4609 | yes | 0.6113 | 0.9932 | 0.9673 | +0.0259 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 1 | `up_proj` | 1.4896 | yes | 0.6060 | 0.9830 | 0.9580 | +0.0250 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 45 | `down_proj` | 0.3724 | yes | 0.1799 | 0.9827 | 0.9616 | +0.0211 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 45 | `gate_proj` | 0.3724 | yes | 0.2401 | 0.9695 | 0.9484 | +0.0211 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 45 | `gate_proj` | 0.7448 | yes | 0.2994 | 0.9691 | 0.9484 | +0.0207 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 1 | `down_proj` | 0.3724 | yes | 0.1895 | 0.9695 | 0.9489 | +0.0206 | yes | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 1 | `gate_proj` | 1.5000 | yes | 0.7883 | 0.9110 | 0.8915 | +0.0195 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 1 | `gate_proj` | 1.5000 | yes | 0.7883 | 0.9110 | 0.8915 | +0.0195 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 1 | `gate_proj` | 1.5000 | yes | 0.7883 | 0.9110 | 0.8915 | +0.0195 | yes | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 45 | `gate_proj` | 1.4896 | yes | 0.3790 | 0.9675 | 0.9484 | +0.0191 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 1 | `gate_proj` | 1.4609 | yes | 0.7869 | 0.9104 | 0.8915 | +0.0189 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 1 | `gate_proj` | 1.4844 | yes | 0.7878 | 0.9104 | 0.8915 | +0.0189 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 45 | `up_proj` | 0.3724 | yes | 0.1627 | 0.9859 | 0.9673 | +0.0185 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 104 | `down_proj` | 1.4609 | yes | 0.5358 | 0.9770 | 0.9593 | +0.0177 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 1 | `down_proj` | 0.7448 | yes | 0.4524 | 0.9663 | 0.9489 | +0.0174 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 45 | `down_proj` | 0.7448 | yes | 0.4442 | 0.9789 | 0.9616 | +0.0173 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 104 | `down_proj` | 0.3724 | yes | 0.1594 | 0.9759 | 0.9593 | +0.0166 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 45 | `up_proj` | 0.7448 | yes | 0.4675 | 0.9824 | 0.9673 | +0.0151 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 1 | `up_proj` | 0.3724 | yes | 0.2598 | 0.9729 | 0.9580 | +0.0150 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 1 | `down_proj` | 1.4896 | yes | 0.6174 | 0.9638 | 0.9489 | +0.0149 | yes | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 1 | `gate_proj` | 1.2500 | yes | 0.7780 | 0.9063 | 0.8915 | +0.0149 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 1 | `gate_proj` | 1.2500 | yes | 0.7780 | 0.9063 | 0.8915 | +0.0149 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 104 | `down_proj` | 0.7448 | yes | 0.4330 | 0.9739 | 0.9593 | +0.0146 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 1 | `up_proj` | 0.7448 | yes | 0.3178 | 0.9721 | 0.9580 | +0.0141 | yes | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 1 | `up_proj` | 1.4896 | yes | 0.3917 | 0.9713 | 0.9580 | +0.0134 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 45 | `down_proj` | 1.4896 | yes | 0.6283 | 0.9749 | 0.9616 | +0.0133 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 104 | `down_proj` | 1.4896 | yes | 0.6191 | 0.9716 | 0.9593 | +0.0123 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 45 | `up_proj` | 1.4896 | yes | 0.6204 | 0.9794 | 0.9673 | +0.0121 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 45 | `up_proj` | 0.3724 | yes | 0.2384 | 0.9737 | 0.9673 | +0.0064 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 45 | `up_proj` | 0.7448 | yes | 0.2977 | 0.9729 | 0.9673 | +0.0056 | yes | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 45 | `up_proj` | 1.4896 | yes | 0.3772 | 0.9719 | 0.9673 | +0.0046 | yes | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 1 | `gate_proj` | 1.4609 | yes | 0.8224 | 0.8897 | 0.8915 | -0.0018 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 127 | `down_proj` | 0.7448 | yes | 0.5563 | 0.7162 | 0.7221 | -0.0059 | no | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 1 | `down_proj` | 1.4896 | yes | 0.6530 | 0.9420 | 0.9489 | -0.0069 | no | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 1 | `gate_proj` | 1.4896 | yes | 0.8208 | 0.8744 | 0.8915 | -0.0171 | no | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 104 | `down_proj` | 1.4896 | yes | 0.5717 | 0.9420 | 0.9593 | -0.0173 | no | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 1 | `down_proj` | 0.7448 | yes | 0.5161 | 0.9301 | 0.9489 | -0.0188 | no | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 1 | `down_proj` | 0.3724 | yes | 0.3855 | 0.9199 | 0.9489 | -0.0290 | no | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 45 | `down_proj` | 1.4609 | yes | 0.5767 | 0.9326 | 0.9616 | -0.0290 | no | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 104 | `gate_proj` | 1.5000 | yes | 0.7825 | 0.8906 | 0.9216 | -0.0310 | no | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 104 | `gate_proj` | 1.5000 | yes | 0.7825 | 0.8906 | 0.9216 | -0.0310 | no | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 104 | `gate_proj` | 1.5000 | yes | 0.7825 | 0.8906 | 0.9216 | -0.0310 | no | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 104 | `gate_proj` | 1.4844 | yes | 0.7817 | 0.8902 | 0.9216 | -0.0314 | no | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 104 | `down_proj` | 0.7448 | yes | 0.4271 | 0.9265 | 0.9593 | -0.0327 | no | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 104 | `gate_proj` | 1.4609 | yes | 0.7805 | 0.8886 | 0.9216 | -0.0329 | no | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 45 | `down_proj` | 1.4896 | yes | 0.6046 | 0.9268 | 0.9616 | -0.0348 | no | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 104 | `up_proj` | 1.5000 | yes | 0.7853 | 0.8872 | 0.9232 | -0.0360 | no | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 104 | `up_proj` | 1.5000 | yes | 0.7853 | 0.8872 | 0.9232 | -0.0360 | no | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 104 | `up_proj` | 1.5000 | yes | 0.7853 | 0.8872 | 0.9232 | -0.0360 | no | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 1 | `up_proj` | 1.5000 | yes | 0.7896 | 0.9218 | 0.9580 | -0.0362 | no | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 1 | `up_proj` | 1.5000 | yes | 0.7896 | 0.9218 | 0.9580 | -0.0362 | no | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 1 | `up_proj` | 1.5000 | yes | 0.7896 | 0.9218 | 0.9580 | -0.0362 | no | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 1 | `up_proj` | 1.4844 | yes | 0.7892 | 0.9214 | 0.9580 | -0.0366 | no | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 104 | `up_proj` | 1.4844 | yes | 0.7848 | 0.8864 | 0.9232 | -0.0368 | no | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 1 | `up_proj` | 1.4609 | yes | 0.7884 | 0.9201 | 0.9580 | -0.0379 | no | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 104 | `up_proj` | 1.4609 | yes | 0.7836 | 0.8852 | 0.9232 | -0.0379 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 104 | `gate_proj` | 1.2500 | yes | 0.7674 | 0.8830 | 0.9216 | -0.0386 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 104 | `gate_proj` | 1.2500 | yes | 0.7674 | 0.8830 | 0.9216 | -0.0386 | no | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 104 | `down_proj` | 1.5000 | yes | 0.1830 | 0.9190 | 0.9593 | -0.0403 | no | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 104 | `down_proj` | 1.5000 | yes | 0.1830 | 0.9190 | 0.9593 | -0.0403 | no | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 104 | `down_proj` | 1.5000 | yes | 0.1830 | 0.9190 | 0.9593 | -0.0403 | no | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 104 | `down_proj` | 1.4792 | yes | 0.1829 | 0.9187 | 0.9593 | -0.0405 | no | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 104 | `down_proj` | 0.3724 | yes | 0.3168 | 0.9185 | 0.9593 | -0.0408 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 1 | `up_proj` | 1.2500 | yes | 0.7800 | 0.9171 | 0.9580 | -0.0408 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 1 | `up_proj` | 1.2500 | yes | 0.7800 | 0.9171 | 0.9580 | -0.0408 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 104 | `up_proj` | 1.2500 | yes | 0.7733 | 0.8804 | 0.9232 | -0.0428 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 104 | `up_proj` | 1.2500 | yes | 0.7733 | 0.8804 | 0.9232 | -0.0428 | no | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 104 | `down_proj` | 1.4583 | yes | 0.1827 | 0.9151 | 0.9593 | -0.0441 | no | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 104 | `gate_proj` | 1.4609 | yes | 0.8127 | 0.8757 | 0.9216 | -0.0458 | no | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 45 | `gate_proj` | 1.5000 | yes | 0.7867 | 0.9025 | 0.9484 | -0.0459 | no | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 45 | `gate_proj` | 1.5000 | yes | 0.7867 | 0.9025 | 0.9484 | -0.0459 | no | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 45 | `gate_proj` | 1.5000 | yes | 0.7867 | 0.9025 | 0.9484 | -0.0459 | no | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 45 | `gate_proj` | 1.4844 | yes | 0.7863 | 0.9014 | 0.9484 | -0.0470 | no | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 45 | `gate_proj` | 1.4609 | yes | 0.7855 | 0.9008 | 0.9484 | -0.0476 | no | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 104 | `gate_proj` | 1.4896 | yes | 0.8122 | 0.8726 | 0.9216 | -0.0489 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 45 | `gate_proj` | 1.2500 | yes | 0.7754 | 0.8970 | 0.9484 | -0.0514 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 45 | `gate_proj` | 1.2500 | yes | 0.7754 | 0.8970 | 0.9484 | -0.0514 | no | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 1 | `down_proj` | 1.5000 | yes | 0.0593 | 0.8959 | 0.9489 | -0.0530 | no | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 1 | `down_proj` | 1.5000 | yes | 0.0593 | 0.8959 | 0.9489 | -0.0530 | no | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 1 | `down_proj` | 1.5000 | yes | 0.0593 | 0.8959 | 0.9489 | -0.0530 | no | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 104 | `up_proj` | 1.4609 | yes | 0.7981 | 0.8683 | 0.9232 | -0.0549 | no | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 45 | `down_proj` | 0.7448 | yes | 0.4637 | 0.9057 | 0.9616 | -0.0559 | no | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 104 | `up_proj` | 1.4896 | yes | 0.8048 | 0.8666 | 0.9232 | -0.0565 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 127 | `gate_proj` | 0.3724 | yes | 0.5608 | 0.5953 | 0.6545 | -0.0592 | no | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 1 | `down_proj` | 1.4792 | yes | 0.0593 | 0.8872 | 0.9489 | -0.0617 | no | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 1 | `up_proj` | 1.4609 | yes | 0.8108 | 0.8940 | 0.9580 | -0.0640 | no | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 1 | `down_proj` | 1.4583 | yes | 0.0593 | 0.8843 | 0.9489 | -0.0646 | no | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 45 | `gate_proj` | 1.4609 | yes | 0.8148 | 0.8824 | 0.9484 | -0.0660 | no | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 45 | `up_proj` | 1.5000 | yes | 0.7869 | 0.8998 | 0.9673 | -0.0675 | no | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 45 | `up_proj` | 1.5000 | yes | 0.7869 | 0.8998 | 0.9673 | -0.0675 | no | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 45 | `up_proj` | 1.5000 | yes | 0.7869 | 0.8998 | 0.9673 | -0.0675 | no | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 1 | `down_proj` | 1.4896 | yes | 0.7645 | 0.8804 | 0.9489 | -0.0685 | no | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 45 | `up_proj` | 1.4844 | yes | 0.7864 | 0.8987 | 0.9673 | -0.0687 | no | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 45 | `up_proj` | 1.4609 | yes | 0.7857 | 0.8979 | 0.9673 | -0.0694 | no | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 45 | `gate_proj` | 1.4896 | yes | 0.8190 | 0.8770 | 0.9484 | -0.0714 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 45 | `up_proj` | 1.2500 | yes | 0.7771 | 0.8934 | 0.9673 | -0.0739 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 45 | `up_proj` | 1.2500 | yes | 0.7771 | 0.8934 | 0.9673 | -0.0739 | no | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 1 | `up_proj` | 1.4896 | yes | 0.8158 | 0.8779 | 0.9580 | -0.0801 | no | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 1 | `down_proj` | 1.4609 | yes | 0.7518 | 0.8666 | 0.9489 | -0.0823 | no | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 45 | `up_proj` | 1.4896 | yes | 0.8151 | 0.8798 | 0.9673 | -0.0875 | no | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 104 | `down_proj` | 1.4896 | yes | 0.7008 | 0.8717 | 0.9593 | -0.0876 | no | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 45 | `down_proj` | 0.3724 | yes | 0.3422 | 0.8732 | 0.9616 | -0.0884 | no | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 45 | `up_proj` | 1.4609 | yes | 0.8061 | 0.8762 | 0.9673 | -0.0911 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 127 | `down_proj` | 1.2500 | yes | 0.0199 | 0.6299 | 0.7221 | -0.0922 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 127 | `down_proj` | 1.2500 | yes | 0.0199 | 0.6299 | 0.7221 | -0.0922 | no | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 45 | `down_proj` | 1.5000 | yes | 0.0301 | 0.8580 | 0.9616 | -0.1036 | no | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 45 | `down_proj` | 1.5000 | yes | 0.0301 | 0.8580 | 0.9616 | -0.1036 | no | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 45 | `down_proj` | 1.5000 | yes | 0.0301 | 0.8580 | 0.9616 | -0.1036 | no | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 45 | `down_proj` | 1.4609 | yes | 0.7099 | 0.8559 | 0.9616 | -0.1057 | no | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 45 | `down_proj` | 1.4792 | yes | 0.0301 | 0.8547 | 0.9616 | -0.1069 | no | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 45 | `down_proj` | 1.4583 | yes | 0.0301 | 0.8504 | 0.9616 | -0.1112 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 1 | `gate_proj` | 0.7448 | yes | 0.6961 | 0.7795 | 0.8915 | -0.1120 | no | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 45 | `down_proj` | 1.4896 | yes | 0.7328 | 0.8453 | 0.9616 | -0.1163 | no | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 104 | `down_proj` | 1.4609 | yes | 0.6727 | 0.8378 | 0.9593 | -0.1215 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 104 | `gate_proj` | 0.7448 | yes | 0.6785 | 0.7894 | 0.9216 | -0.1322 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 127 | `down_proj` | 0.3724 | yes | 0.4196 | 0.5885 | 0.7221 | -0.1337 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 1 | `down_proj` | 0.7448 | yes | 0.6221 | 0.7988 | 0.9489 | -0.1501 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 104 | `up_proj` | 0.7448 | yes | 0.6645 | 0.7581 | 0.9232 | -0.1651 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 1 | `up_proj` | 0.7448 | yes | 0.6806 | 0.7782 | 0.9580 | -0.1798 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 104 | `down_proj` | 0.7448 | yes | 0.5484 | 0.7657 | 0.9593 | -0.1935 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 45 | `down_proj` | 0.7448 | yes | 0.5797 | 0.7591 | 0.9616 | -0.2025 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 45 | `up_proj` | 0.7448 | yes | 0.6749 | 0.7527 | 0.9673 | -0.2146 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 45 | `gate_proj` | 0.7448 | yes | 0.6854 | 0.7307 | 0.9484 | -0.2177 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 1 | `down_proj` | 0.3724 | yes | 0.4950 | 0.7135 | 0.9489 | -0.2354 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 1 | `gate_proj` | 0.3724 | yes | 0.5597 | 0.6428 | 0.8915 | -0.2487 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 45 | `down_proj` | 0.3724 | yes | 0.4488 | 0.7005 | 0.9616 | -0.2611 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 104 | `down_proj` | 1.2500 | yes | 0.1811 | 0.6857 | 0.9593 | -0.2736 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 104 | `down_proj` | 1.2500 | yes | 0.1811 | 0.6857 | 0.9593 | -0.2736 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 45 | `down_proj` | 1.2500 | yes | 0.0297 | 0.6792 | 0.9616 | -0.2824 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 45 | `down_proj` | 1.2500 | yes | 0.0297 | 0.6792 | 0.9616 | -0.2824 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 104 | `down_proj` | 0.3724 | yes | 0.4231 | 0.6662 | 0.9593 | -0.2931 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 1 | `down_proj` | 1.2500 | yes | 0.0589 | 0.6382 | 0.9489 | -0.3107 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 1 | `down_proj` | 1.2500 | yes | 0.0589 | 0.6382 | 0.9489 | -0.3107 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 104 | `gate_proj` | 0.3724 | yes | 0.5331 | 0.6050 | 0.9216 | -0.3166 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 45 | `gate_proj` | 0.3724 | yes | 0.5425 | 0.6250 | 0.9484 | -0.3234 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 1 | `up_proj` | 0.3724 | yes | 0.5376 | 0.6191 | 0.9580 | -0.3389 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 45 | `up_proj` | 0.3724 | yes | 0.5304 | 0.6138 | 0.9673 | -0.3535 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 104 | `up_proj` | 0.3724 | yes | 0.5157 | 0.5522 | 0.9232 | -0.3710 | no | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 127 | `up_proj` | 4.8698 | no | 0.9845 | 0.9870 | 0.4934 | +0.4936 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 127 | `up_proj` | 2.9219 | no | 0.6999 | 0.9760 | 0.4934 | +0.4826 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 127 | `up_proj` | 3.8958 | no | 0.7143 | 0.9759 | 0.4934 | +0.4825 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 127 | `up_proj` | 4.8698 | no | 0.7199 | 0.9759 | 0.4934 | +0.4825 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 127 | `up_proj` | 3.8958 | no | 0.9707 | 0.9750 | 0.4934 | +0.4816 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 127 | `up_proj` | 2.9219 | no | 0.9431 | 0.9601 | 0.4934 | +0.4667 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 127 | `gate_proj` | 4.8698 | no | 0.9847 | 0.9894 | 0.6545 | +0.3349 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 127 | `gate_proj` | 2.9219 | no | 0.7015 | 0.9885 | 0.6545 | +0.3340 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 127 | `gate_proj` | 3.8958 | no | 0.7141 | 0.9885 | 0.6545 | +0.3340 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 127 | `gate_proj` | 4.8698 | no | 0.7189 | 0.9885 | 0.6545 | +0.3340 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 127 | `gate_proj` | 3.8958 | no | 0.9727 | 0.9810 | 0.6545 | +0.3264 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 127 | `gate_proj` | 2.9219 | no | 0.9476 | 0.9609 | 0.6545 | +0.3064 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 127 | `down_proj` | 4.8698 | no | 0.9859 | 0.9907 | 0.7221 | +0.2686 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 127 | `down_proj` | 2.9219 | no | 0.7660 | 0.9760 | 0.7221 | +0.2539 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 127 | `down_proj` | 3.8958 | no | 0.8120 | 0.9758 | 0.7221 | +0.2536 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 127 | `down_proj` | 4.8698 | no | 0.8307 | 0.9757 | 0.7221 | +0.2535 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 127 | `down_proj` | 3.8958 | no | 0.9420 | 0.9713 | 0.7221 | +0.2491 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 127 | `down_proj` | 2.9219 | no | 0.8707 | 0.9356 | 0.7221 | +0.2135 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 1 | `gate_proj` | 2.7164 | no | 0.4693 | 0.9950 | 0.8915 | +0.1036 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 1 | `gate_proj` | 2.7164 | no | 0.4693 | 0.9950 | 0.8915 | +0.1036 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 1 | `gate_proj` | 2.7164 | no | 0.4693 | 0.9950 | 0.8915 | +0.1036 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 1 | `gate_proj` | 4.8698 | no | 0.9824 | 0.9886 | 0.8915 | +0.0972 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 1 | `gate_proj` | 2.9219 | no | 0.6865 | 0.9855 | 0.8915 | +0.0940 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 1 | `gate_proj` | 3.8958 | no | 0.7080 | 0.9855 | 0.8915 | +0.0940 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 1 | `gate_proj` | 4.8698 | no | 0.7154 | 0.9854 | 0.8915 | +0.0940 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 1 | `gate_proj` | 3.8958 | no | 0.9685 | 0.9819 | 0.8915 | +0.0904 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 104 | `gate_proj` | 2.7545 | no | 0.4625 | 0.9948 | 0.9216 | +0.0732 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 104 | `gate_proj` | 2.7545 | no | 0.4625 | 0.9948 | 0.9216 | +0.0732 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 104 | `gate_proj` | 2.7545 | no | 0.4625 | 0.9948 | 0.9216 | +0.0732 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 1 | `gate_proj` | 2.9219 | no | 0.9368 | 0.9645 | 0.8915 | +0.0730 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 104 | `gate_proj` | 2.9219 | no | 0.7108 | 0.9936 | 0.9216 | +0.0720 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 104 | `gate_proj` | 3.8958 | no | 0.7293 | 0.9935 | 0.9216 | +0.0720 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 104 | `gate_proj` | 4.8698 | no | 0.7391 | 0.9935 | 0.9216 | +0.0719 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 104 | `up_proj` | 2.7545 | no | 0.4591 | 0.9941 | 0.9232 | +0.0709 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 104 | `up_proj` | 2.7545 | no | 0.4591 | 0.9941 | 0.9232 | +0.0709 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 104 | `up_proj` | 2.7545 | no | 0.4591 | 0.9941 | 0.9232 | +0.0709 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 104 | `up_proj` | 2.9219 | no | 0.7290 | 0.9927 | 0.9232 | +0.0696 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 104 | `up_proj` | 3.8958 | no | 0.7494 | 0.9927 | 0.9232 | +0.0695 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 104 | `up_proj` | 4.8698 | no | 0.7601 | 0.9926 | 0.9232 | +0.0694 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 104 | `gate_proj` | 4.8698 | no | 0.9759 | 0.9870 | 0.9216 | +0.0655 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 104 | `up_proj` | 4.8698 | no | 0.9758 | 0.9847 | 0.9232 | +0.0615 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 104 | `gate_proj` | 3.8958 | no | 0.9589 | 0.9769 | 0.9216 | +0.0553 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 104 | `up_proj` | 3.8958 | no | 0.9566 | 0.9745 | 0.9232 | +0.0514 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 45 | `gate_proj` | 2.6099 | no | 0.4464 | 0.9940 | 0.9484 | +0.0456 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 45 | `gate_proj` | 2.6099 | no | 0.4464 | 0.9940 | 0.9484 | +0.0456 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 45 | `gate_proj` | 2.6099 | no | 0.4464 | 0.9940 | 0.9484 | +0.0456 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 45 | `gate_proj` | 2.9219 | no | 0.7052 | 0.9936 | 0.9484 | +0.0452 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 45 | `gate_proj` | 3.8958 | no | 0.7254 | 0.9935 | 0.9484 | +0.0451 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 45 | `gate_proj` | 4.8698 | no | 0.7346 | 0.9935 | 0.9484 | +0.0451 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 45 | `gate_proj` | 4.8698 | no | 0.9842 | 0.9892 | 0.9484 | +0.0408 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 1 | `down_proj` | 4.8698 | no | 0.9777 | 0.9893 | 0.9489 | +0.0404 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 1 | `down_proj` | 2.7164 | no | 0.7673 | 0.9885 | 0.9489 | +0.0396 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 1 | `down_proj` | 2.7164 | no | 0.7673 | 0.9885 | 0.9489 | +0.0396 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 1 | `down_proj` | 2.7164 | no | 0.7673 | 0.9885 | 0.9489 | +0.0396 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 1 | `down_proj` | 2.9219 | no | 0.7393 | 0.9869 | 0.9489 | +0.0380 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 1 | `down_proj` | 3.8958 | no | 0.7717 | 0.9867 | 0.9489 | +0.0378 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 1 | `down_proj` | 4.8698 | no | 0.7832 | 0.9867 | 0.9489 | +0.0378 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 1 | `up_proj` | 2.7164 | no | 0.4657 | 0.9958 | 0.9580 | +0.0378 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 1 | `up_proj` | 2.7164 | no | 0.4657 | 0.9958 | 0.9580 | +0.0378 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 1 | `up_proj` | 2.7164 | no | 0.4657 | 0.9958 | 0.9580 | +0.0378 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 1 | `up_proj` | 2.9219 | no | 0.7029 | 0.9931 | 0.9580 | +0.0351 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 1 | `up_proj` | 3.8958 | no | 0.7258 | 0.9931 | 0.9580 | +0.0351 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 1 | `up_proj` | 4.8698 | no | 0.7336 | 0.9931 | 0.9580 | +0.0351 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 1 | `down_proj` | 3.8958 | no | 0.9505 | 0.9814 | 0.9489 | +0.0325 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 104 | `gate_proj` | 2.9219 | no | 0.9279 | 0.9539 | 0.9216 | +0.0324 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 45 | `gate_proj` | 3.8958 | no | 0.9677 | 0.9806 | 0.9484 | +0.0322 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 1 | `up_proj` | 4.8698 | no | 0.9827 | 0.9882 | 0.9580 | +0.0303 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 104 | `down_proj` | 2.9219 | no | 0.7532 | 0.9887 | 0.9593 | +0.0295 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 104 | `down_proj` | 3.8958 | no | 0.8014 | 0.9887 | 0.9593 | +0.0294 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 104 | `down_proj` | 4.8698 | no | 0.8271 | 0.9887 | 0.9593 | +0.0294 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 45 | `down_proj` | 4.8698 | no | 0.9803 | 0.9901 | 0.9616 | +0.0285 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 45 | `down_proj` | 2.9219 | no | 0.7555 | 0.9898 | 0.9616 | +0.0282 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 45 | `down_proj` | 3.8958 | no | 0.7968 | 0.9897 | 0.9616 | +0.0281 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 45 | `down_proj` | 4.8698 | no | 0.8161 | 0.9897 | 0.9616 | +0.0281 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 45 | `up_proj` | 2.6099 | no | 0.4456 | 0.9949 | 0.9673 | +0.0276 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 45 | `up_proj` | 2.6099 | no | 0.4456 | 0.9949 | 0.9673 | +0.0276 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 45 | `up_proj` | 2.6099 | no | 0.4456 | 0.9949 | 0.9673 | +0.0276 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 104 | `down_proj` | 2.7545 | no | 0.7013 | 0.9861 | 0.9593 | +0.0268 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 104 | `down_proj` | 2.7545 | no | 0.7013 | 0.9861 | 0.9593 | +0.0268 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 104 | `down_proj` | 2.7545 | no | 0.7013 | 0.9861 | 0.9593 | +0.0268 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 104 | `up_proj` | 2.9219 | no | 0.9218 | 0.9498 | 0.9232 | +0.0266 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 45 | `up_proj` | 2.9219 | no | 0.7155 | 0.9929 | 0.9673 | +0.0256 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 45 | `up_proj` | 3.8958 | no | 0.7368 | 0.9929 | 0.9673 | +0.0255 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 45 | `up_proj` | 4.8698 | no | 0.7465 | 0.9928 | 0.9673 | +0.0255 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 104 | `down_proj` | 4.8698 | no | 0.9572 | 0.9840 | 0.9593 | +0.0248 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 1 | `up_proj` | 3.8958 | no | 0.9674 | 0.9814 | 0.9580 | +0.0234 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 45 | `up_proj` | 4.8698 | no | 0.9846 | 0.9896 | 0.9673 | +0.0223 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 45 | `gate_proj` | 2.9219 | no | 0.9355 | 0.9634 | 0.9484 | +0.0150 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 45 | `up_proj` | 3.8958 | no | 0.9667 | 0.9801 | 0.9673 | +0.0128 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 1 | `down_proj` | 2.9219 | no | 0.9000 | 0.9590 | 0.9489 | +0.0101 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 104 | `down_proj` | 3.8958 | no | 0.9093 | 0.9684 | 0.9593 | +0.0091 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 45 | `down_proj` | 3.8958 | no | 0.9401 | 0.9705 | 0.9616 | +0.0089 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 45 | `down_proj` | 2.6099 | no | 0.7171 | 0.9667 | 0.9616 | +0.0051 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 45 | `down_proj` | 2.6099 | no | 0.7171 | 0.9667 | 0.9616 | +0.0051 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 45 | `down_proj` | 2.6099 | no | 0.7171 | 0.9667 | 0.9616 | +0.0051 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 1 | `up_proj` | 2.9219 | no | 0.9330 | 0.9623 | 0.9580 | +0.0044 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 45 | `up_proj` | 2.9219 | no | 0.9324 | 0.9628 | 0.9673 | -0.0045 | no | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 104 | `down_proj` | 2.9219 | no | 0.8416 | 0.9418 | 0.9593 | -0.0175 | no | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 45 | `down_proj` | 2.9219 | no | 0.8761 | 0.9413 | 0.9616 | -0.0203 | no | no | no |

## Provenance

- Capture run: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/gate-up-residual-v1/current-hcli-route-capture/runs/74c918d500b2a8fdc17c2a4a417bf0e967b6a17709e7cdba486466c7c39e862a_8bd3bfb36e16be850dc5e1909e3f07a3b0ddc4f49634455e258a0eb3d8660037`
- Capture result sha256: `43845450bc8dc977d07d742125d88958ab7322ea92e09ef5048346f4eabad062`
- Hidden source: device-produced L0 post-attention RMSNorm (router input)
- Probes: literal_hawking, json_status, python_add
- Token-expert pairs: 9000
- Experts: [104, 1, 45, 127] hits={'104': 484, '1': 477, '45': 458, '127': 48}
- Model: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct`

## Claim boundary

- `no_server_started`: True
- `no_lease_issued`: True
- `no_full_model_pack`: True
- `no_gate_weakened`: True
- `cpu_numpy_only`: True
- `component_bpw_not_complete_model_bpw`: True
- `activations`: real current-HCLI L0 router-input hiddens from sealed route capture
- `down_proj_activations`: derived swiglu intermediate using true BF16 gate/up on those hiddens
- `not_a_runtime_admission`: True
- `not_a_capability_claim`: True
- `capture_is_three_prompt_prefix_only`: True

Component BPW is the honest per-tensor billed rate for the compressed factors/codes. It is not a complete-model BPW ledger. No gate was weakened.
