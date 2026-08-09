# Q30 activation-aware family probe

- Recorded: `2026-08-09T20:02:30.791985Z`
- Status: `EARNED_ACTIVATION_AWARE_FAMILY_PROBE_COMPLETE_OPERATOR_COHERENCE_UNDER_CEILING`
- Ceiling: component BPW ≤ 1.5
- Coherence bar (surplus-first): output_cos ≥ 0.9, surplus ≥ 0.1, operator-recovery weight_cos ≥ 0.5
- Mean null (high-hit under ceiling): `0.5710325777617785`

## Verdict

POSITIVE on high-hit experts: surplus-first coherence-grade cleared under component BPW <= 1.5 with weight cosine also above the 0.5 distribution-local cutoff. First surplus-first coherence-grade point (any expert, any BPW) at component_bpw=0.3724 (activation_weighted_svd_low_rank_q/r64_b3/gate_proj expert 127, out=0.9121, null=0.4124, surplus=+0.4997, wt=0.3122). On high-hit experts, first surplus-first row that also clears the operator-recovery weight-cosine cutoff is at component_bpw=0.7448 (activation_weighted_svd_low_rank_q/r128_b3, wt=0.5049, surplus=+0.3126). Family ranking on real activations DOES invert vs weight-space: activation-PCA low-rank beats raw-weight low-rank on surplus-over-null for high-hit experts. Raw-weight low-rank typically fails to beat the constant-mean null on this capture. Best high-hit under-ceiling surplus=+0.5569 (out=0.9693, null=0.4124, wt=0.7481, bpw=1.4609) via activation_weighted_svd_low_rank_q/r192_b4 on gate_proj expert 127. On high-hit experts, joint surplus+operator reachability first appears at component_bpw=0.7448 (activation_weighted_svd_low_rank_q/r128_b3).

## Family summary — high-hit experts only (primary)

| family | n | mean out-cos | mean wt-cos | mean null | mean surplus | frac beats null | best surplus | local coh? | operator coh? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `activation_pca_low_rank_q` | 57 | 0.9444 | 0.4166 | 0.5567 | 0.3877 | 1.0000 | 0.5563 | yes | yes |
| `activation_weighted_svd_low_rank_q` | 48 | 0.9278 | 0.5102 | 0.5756 | 0.3522 | 1.0000 | 0.5569 | yes | yes |
| `activation_weighted_binary_residual` | 84 | 0.8602 | 0.5328 | 0.5756 | 0.2847 | 0.9286 | 0.4827 | yes | yes |
| `raw_weight_low_rank_q` | 48 | 0.7585 | 0.6765 | 0.5756 | 0.1830 | 0.8542 | 0.4653 | yes | yes |

## Family summary — all selected experts (under ceiling)

| family | n | mean out-cos | mean wt-cos | mean null | mean surplus | frac beats null | best surplus | local coh? | operator coh? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `activation_pca_low_rank_q` | 57 | 0.9444 | 0.4166 | 0.5567 | 0.3877 | 1.0000 | 0.5563 | yes | yes |
| `activation_weighted_svd_low_rank_q` | 48 | 0.9278 | 0.5102 | 0.5756 | 0.3522 | 1.0000 | 0.5569 | yes | yes |
| `activation_weighted_binary_residual` | 84 | 0.8602 | 0.5328 | 0.5756 | 0.2847 | 0.9286 | 0.4827 | yes | yes |
| `raw_weight_low_rank_q` | 48 | 0.7585 | 0.6765 | 0.5756 | 0.1830 | 0.8542 | 0.4653 | yes | yes |

## Full table (family × budget × tensor)

| family | budget | expert | component | bpw | under 1.5? | weight-cos | output-cos | null | surplus | beats null | coh | local-only |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---|---|---|
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 127 | `gate_proj` | 1.4609 | yes | 0.7481 | 0.9693 | 0.4124 | +0.5569 | yes | yes | no |
| `activation_pca_low_rank_q` | `r192_b4` | 127 | `gate_proj` | 1.4077 | yes | 0.4157 | 0.9687 | 0.4124 | +0.5563 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r384_b4` | 127 | `gate_proj` | 1.4077 | yes | 0.4157 | 0.9687 | 0.4124 | +0.5563 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r512_b4` | 127 | `gate_proj` | 1.4077 | yes | 0.4157 | 0.9687 | 0.4124 | +0.5563 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r640_b4` | 127 | `gate_proj` | 1.4077 | yes | 0.4157 | 0.9687 | 0.4124 | +0.5563 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 127 | `up_proj` | 1.4609 | yes | 0.7471 | 0.9537 | 0.4007 | +0.5529 | yes | yes | no |
| `activation_pca_low_rank_q` | `r256_b3` | 127 | `gate_proj` | 1.0765 | yes | 0.3941 | 0.9395 | 0.4124 | +0.5271 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r192_b4` | 127 | `up_proj` | 1.4077 | yes | 0.4113 | 0.9207 | 0.4007 | +0.5199 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r384_b4` | 127 | `up_proj` | 1.4077 | yes | 0.4113 | 0.9207 | 0.4007 | +0.5199 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r512_b4` | 127 | `up_proj` | 1.4077 | yes | 0.4113 | 0.9207 | 0.4007 | +0.5199 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r640_b4` | 127 | `up_proj` | 1.4077 | yes | 0.4113 | 0.9207 | 0.4007 | +0.5199 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 127 | `up_proj` | 1.4896 | yes | 0.6608 | 0.9064 | 0.4007 | +0.5057 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 127 | `gate_proj` | 1.4896 | yes | 0.6460 | 0.9179 | 0.4124 | +0.5055 | yes | yes | no |
| `activation_pca_low_rank_q` | `r192_b4` | 109 | `gate_proj` | 1.4609 | yes | 0.4279 | 0.9649 | 0.4614 | +0.5035 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 109 | `gate_proj` | 1.4609 | yes | 0.7136 | 0.9645 | 0.4614 | +0.5031 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 127 | `gate_proj` | 0.7448 | yes | 0.4590 | 0.9131 | 0.4124 | +0.5007 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 127 | `gate_proj` | 0.3724 | yes | 0.3122 | 0.9121 | 0.4124 | +0.4997 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r256_b3` | 109 | `gate_proj` | 1.4896 | yes | 0.4491 | 0.9571 | 0.4614 | +0.4957 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r128_b3` | 127 | `gate_proj` | 0.7448 | yes | 0.3474 | 0.9050 | 0.4124 | +0.4926 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 109 | `up_proj` | 1.4609 | yes | 0.7090 | 0.9687 | 0.4789 | +0.4898 | yes | yes | no |
| `activation_pca_low_rank_q` | `r256_b3` | 127 | `up_proj` | 1.0765 | yes | 0.3895 | 0.8905 | 0.4007 | +0.4898 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 127 | `up_proj` | 0.7448 | yes | 0.4801 | 0.8897 | 0.4007 | +0.4890 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 127 | `gate_proj` | 0.3724 | yes | 0.2456 | 0.9006 | 0.4124 | +0.4882 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r192_b4` | 109 | `up_proj` | 1.4609 | yes | 0.4146 | 0.9644 | 0.4789 | +0.4855 | yes | yes | yes |
| `activation_weighted_binary_residual` | `r384_b4` | 127 | `up_proj` | 1.5000 | yes | 0.7944 | 0.8834 | 0.4007 | +0.4827 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 127 | `up_proj` | 1.5000 | yes | 0.7944 | 0.8834 | 0.4007 | +0.4827 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 127 | `up_proj` | 1.5000 | yes | 0.7944 | 0.8834 | 0.4007 | +0.4827 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 127 | `up_proj` | 1.4844 | yes | 0.7937 | 0.8829 | 0.4007 | +0.4822 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 127 | `up_proj` | 1.4609 | yes | 0.7926 | 0.8826 | 0.4007 | +0.4819 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 109 | `gate_proj` | 0.7448 | yes | 0.3316 | 0.9390 | 0.4614 | +0.4776 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r256_b3` | 109 | `up_proj` | 1.4896 | yes | 0.4366 | 0.9555 | 0.4789 | +0.4766 | yes | yes | yes |
| `activation_weighted_binary_residual` | `r128_b3` | 127 | `up_proj` | 1.2500 | yes | 0.7814 | 0.8772 | 0.4007 | +0.4765 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 127 | `up_proj` | 1.2500 | yes | 0.7814 | 0.8772 | 0.4007 | +0.4765 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 109 | `gate_proj` | 0.3724 | yes | 0.1974 | 0.9352 | 0.4614 | +0.4738 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 127 | `up_proj` | 0.3724 | yes | 0.3042 | 0.8697 | 0.4007 | +0.4690 | yes | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 127 | `gate_proj` | 1.5000 | yes | 0.7906 | 0.8795 | 0.4124 | +0.4671 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 127 | `gate_proj` | 1.5000 | yes | 0.7906 | 0.8795 | 0.4124 | +0.4671 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 127 | `gate_proj` | 1.5000 | yes | 0.7906 | 0.8795 | 0.4124 | +0.4671 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 127 | `gate_proj` | 1.4844 | yes | 0.7898 | 0.8794 | 0.4124 | +0.4670 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 127 | `gate_proj` | 1.4609 | yes | 0.7884 | 0.8788 | 0.4124 | +0.4664 | yes | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 127 | `up_proj` | 1.4896 | yes | 0.8287 | 0.8660 | 0.4007 | +0.4653 | yes | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 127 | `up_proj` | 1.4609 | yes | 0.8322 | 0.8603 | 0.4007 | +0.4595 | yes | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 127 | `gate_proj` | 1.2500 | yes | 0.7750 | 0.8719 | 0.4124 | +0.4595 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 127 | `gate_proj` | 1.2500 | yes | 0.7750 | 0.8719 | 0.4124 | +0.4595 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 109 | `up_proj` | 0.7448 | yes | 0.3258 | 0.9374 | 0.4789 | +0.4585 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r64_b3` | 109 | `up_proj` | 0.3724 | yes | 0.2043 | 0.9344 | 0.4789 | +0.4555 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r128_b3` | 127 | `up_proj` | 0.7448 | yes | 0.3433 | 0.8525 | 0.4007 | +0.4518 | yes | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 127 | `gate_proj` | 1.4609 | yes | 0.8419 | 0.8640 | 0.4124 | +0.4516 | yes | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 127 | `gate_proj` | 1.4896 | yes | 0.8381 | 0.8622 | 0.4124 | +0.4498 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 109 | `gate_proj` | 1.4896 | yes | 0.6471 | 0.9052 | 0.4614 | +0.4438 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 109 | `gate_proj` | 0.7448 | yes | 0.4236 | 0.9051 | 0.4614 | +0.4437 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 109 | `gate_proj` | 0.3724 | yes | 0.2917 | 0.9045 | 0.4614 | +0.4431 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r64_b3` | 127 | `up_proj` | 0.3724 | yes | 0.2516 | 0.8417 | 0.4007 | +0.4409 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 109 | `up_proj` | 1.4896 | yes | 0.6519 | 0.9126 | 0.4789 | +0.4338 | yes | yes | no |
| `activation_weighted_binary_residual` | `r384_b4` | 109 | `gate_proj` | 1.5000 | yes | 0.7916 | 0.8938 | 0.4614 | +0.4324 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 109 | `gate_proj` | 1.5000 | yes | 0.7916 | 0.8938 | 0.4614 | +0.4324 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 109 | `gate_proj` | 1.5000 | yes | 0.7916 | 0.8938 | 0.4614 | +0.4324 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 109 | `up_proj` | 0.7448 | yes | 0.4181 | 0.9113 | 0.4789 | +0.4324 | yes | yes | yes |
| `activation_weighted_binary_residual` | `r256_b3` | 109 | `gate_proj` | 1.4844 | yes | 0.7904 | 0.8935 | 0.4614 | +0.4320 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 109 | `up_proj` | 0.3724 | yes | 0.3035 | 0.9104 | 0.4789 | +0.4315 | yes | yes | yes |
| `activation_weighted_binary_residual` | `r192_b4` | 109 | `gate_proj` | 1.4609 | yes | 0.7889 | 0.8927 | 0.4614 | +0.4313 | yes | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 109 | `gate_proj` | 1.2500 | yes | 0.7716 | 0.8859 | 0.4614 | +0.4244 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 109 | `gate_proj` | 1.2500 | yes | 0.7716 | 0.8859 | 0.4614 | +0.4244 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 85 | `gate_proj` | 1.4609 | yes | 0.6984 | 0.9779 | 0.5585 | +0.4195 | yes | yes | no |
| `activation_pca_low_rank_q` | `r192_b4` | 85 | `gate_proj` | 1.4609 | yes | 0.3915 | 0.9764 | 0.5585 | +0.4180 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r192_b4` | 127 | `down_proj` | 1.4077 | yes | 0.6014 | 0.9700 | 0.5556 | +0.4144 | yes | yes | no |
| `activation_pca_low_rank_q` | `r384_b4` | 127 | `down_proj` | 1.4077 | yes | 0.6014 | 0.9700 | 0.5556 | +0.4144 | yes | yes | no |
| `activation_pca_low_rank_q` | `r512_b4` | 127 | `down_proj` | 1.4077 | yes | 0.6014 | 0.9700 | 0.5556 | +0.4144 | yes | yes | no |
| `activation_pca_low_rank_q` | `r640_b4` | 127 | `down_proj` | 1.4077 | yes | 0.6014 | 0.9700 | 0.5556 | +0.4144 | yes | yes | no |
| `raw_weight_low_rank_q` | `r256_b3` | 109 | `gate_proj` | 1.4896 | yes | 0.8263 | 0.8742 | 0.4614 | +0.4128 | yes | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 109 | `up_proj` | 1.5000 | yes | 0.7965 | 0.8912 | 0.4789 | +0.4123 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 109 | `up_proj` | 1.5000 | yes | 0.7965 | 0.8912 | 0.4789 | +0.4123 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 109 | `up_proj` | 1.5000 | yes | 0.7965 | 0.8912 | 0.4789 | +0.4123 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 109 | `up_proj` | 1.4844 | yes | 0.7959 | 0.8910 | 0.4789 | +0.4122 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 109 | `up_proj` | 1.4609 | yes | 0.7947 | 0.8904 | 0.4789 | +0.4115 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 127 | `down_proj` | 1.4609 | yes | 0.6380 | 0.9638 | 0.5556 | +0.4082 | yes | yes | no |
| `activation_weighted_binary_residual` | `r128_b3` | 109 | `up_proj` | 1.2500 | yes | 0.7831 | 0.8851 | 0.4789 | +0.4063 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 109 | `up_proj` | 1.2500 | yes | 0.7831 | 0.8851 | 0.4789 | +0.4063 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 85 | `up_proj` | 1.4609 | yes | 0.3802 | 0.9789 | 0.5772 | +0.4017 | yes | yes | yes |
| `raw_weight_low_rank_q` | `r192_b4` | 109 | `gate_proj` | 1.4609 | yes | 0.8249 | 0.8627 | 0.4614 | +0.4013 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 85 | `up_proj` | 1.4609 | yes | 0.6857 | 0.9786 | 0.5772 | +0.4013 | yes | yes | no |
| `activation_pca_low_rank_q` | `r256_b3` | 85 | `gate_proj` | 1.4896 | yes | 0.4171 | 0.9591 | 0.5585 | +0.4006 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r128_b3` | 85 | `gate_proj` | 0.7448 | yes | 0.3128 | 0.9470 | 0.5585 | +0.3885 | yes | yes | yes |
| `raw_weight_low_rank_q` | `r256_b3` | 109 | `up_proj` | 1.4896 | yes | 0.8210 | 0.8654 | 0.4789 | +0.3865 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 85 | `gate_proj` | 0.3724 | yes | 0.2014 | 0.9446 | 0.5585 | +0.3861 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r256_b3` | 85 | `up_proj` | 1.4896 | yes | 0.4073 | 0.9629 | 0.5772 | +0.3857 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r256_b3` | 127 | `down_proj` | 1.0765 | yes | 0.5665 | 0.9390 | 0.5556 | +0.3834 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 85 | `gate_proj` | 1.4896 | yes | 0.6807 | 0.9393 | 0.5585 | +0.3808 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 85 | `gate_proj` | 0.3724 | yes | 0.2485 | 0.9378 | 0.5585 | +0.3793 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 85 | `gate_proj` | 0.7448 | yes | 0.4317 | 0.9376 | 0.5585 | +0.3792 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r128_b3` | 85 | `up_proj` | 0.7448 | yes | 0.3033 | 0.9512 | 0.5772 | +0.3739 | yes | yes | yes |
| `raw_weight_low_rank_q` | `r192_b4` | 109 | `up_proj` | 1.4609 | yes | 0.8154 | 0.8520 | 0.4789 | +0.3731 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 85 | `up_proj` | 0.3724 | yes | 0.1966 | 0.9492 | 0.5772 | +0.3720 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r128_b3` | 127 | `down_proj` | 0.7448 | yes | 0.4822 | 0.9230 | 0.5556 | +0.3674 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 127 | `down_proj` | 1.4896 | yes | 0.6147 | 0.9199 | 0.5556 | +0.3643 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 127 | `down_proj` | 0.7448 | yes | 0.3631 | 0.9188 | 0.5556 | +0.3632 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 85 | `up_proj` | 0.3724 | yes | 0.2403 | 0.9403 | 0.5772 | +0.3630 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 127 | `down_proj` | 0.3724 | yes | 0.2173 | 0.9182 | 0.5556 | +0.3626 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 85 | `up_proj` | 0.7448 | yes | 0.4102 | 0.9393 | 0.5772 | +0.3621 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 85 | `up_proj` | 1.4896 | yes | 0.6744 | 0.9392 | 0.5772 | +0.3620 | yes | yes | no |
| `activation_pca_low_rank_q` | `r192_b4` | 76 | `gate_proj` | 1.4609 | yes | 0.4244 | 0.9826 | 0.6228 | +0.3599 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 76 | `gate_proj` | 1.4609 | yes | 0.7508 | 0.9772 | 0.6228 | +0.3545 | yes | yes | no |
| `activation_pca_low_rank_q` | `r192_b4` | 76 | `up_proj` | 1.4609 | yes | 0.4168 | 0.9787 | 0.6283 | +0.3504 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 76 | `up_proj` | 1.4609 | yes | 0.7525 | 0.9768 | 0.6283 | +0.3484 | yes | yes | no |
| `activation_pca_low_rank_q` | `r64_b3` | 127 | `down_proj` | 0.3724 | yes | 0.3508 | 0.9004 | 0.5556 | +0.3448 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r256_b3` | 76 | `gate_proj` | 1.4896 | yes | 0.4528 | 0.9622 | 0.6228 | +0.3395 | yes | yes | yes |
| `activation_weighted_binary_residual` | `r384_b4` | 85 | `gate_proj` | 1.5000 | yes | 0.7979 | 0.8927 | 0.5585 | +0.3342 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 85 | `gate_proj` | 1.5000 | yes | 0.7979 | 0.8927 | 0.5585 | +0.3342 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 85 | `gate_proj` | 1.5000 | yes | 0.7979 | 0.8927 | 0.5585 | +0.3342 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 85 | `gate_proj` | 1.4844 | yes | 0.7974 | 0.8926 | 0.5585 | +0.3342 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 85 | `gate_proj` | 1.4609 | yes | 0.7966 | 0.8922 | 0.5585 | +0.3337 | yes | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 76 | `up_proj` | 1.4896 | yes | 0.4468 | 0.9606 | 0.6283 | +0.3322 | yes | yes | yes |
| `raw_weight_low_rank_q` | `r128_b3` | 127 | `up_proj` | 0.7448 | yes | 0.6984 | 0.7328 | 0.4007 | +0.3320 | yes | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 85 | `gate_proj` | 1.2500 | yes | 0.7885 | 0.8888 | 0.5585 | +0.3303 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 85 | `gate_proj` | 1.2500 | yes | 0.7885 | 0.8888 | 0.5585 | +0.3303 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 76 | `gate_proj` | 0.7448 | yes | 0.3473 | 0.9458 | 0.6228 | +0.3230 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r64_b3` | 76 | `gate_proj` | 0.3724 | yes | 0.2435 | 0.9435 | 0.6228 | +0.3207 | yes | yes | yes |
| `raw_weight_low_rank_q` | `r128_b3` | 127 | `gate_proj` | 0.7448 | yes | 0.7088 | 0.7280 | 0.4124 | +0.3155 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 76 | `up_proj` | 0.7448 | yes | 0.3405 | 0.9424 | 0.6283 | +0.3141 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 76 | `gate_proj` | 1.4896 | yes | 0.6784 | 0.9368 | 0.6228 | +0.3140 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 76 | `gate_proj` | 0.7448 | yes | 0.5049 | 0.9354 | 0.6228 | +0.3126 | yes | yes | no |
| `activation_pca_low_rank_q` | `r64_b3` | 76 | `up_proj` | 0.3724 | yes | 0.2407 | 0.9393 | 0.6283 | +0.3109 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 76 | `gate_proj` | 0.3724 | yes | 0.2895 | 0.9337 | 0.6228 | +0.3109 | yes | yes | yes |
| `activation_weighted_binary_residual` | `r384_b4` | 85 | `up_proj` | 1.5000 | yes | 0.7990 | 0.8848 | 0.5772 | +0.3075 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 85 | `up_proj` | 1.5000 | yes | 0.7990 | 0.8848 | 0.5772 | +0.3075 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 85 | `up_proj` | 1.5000 | yes | 0.7990 | 0.8848 | 0.5772 | +0.3075 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 85 | `up_proj` | 1.4844 | yes | 0.7985 | 0.8843 | 0.5772 | +0.3070 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 85 | `up_proj` | 1.4609 | yes | 0.7978 | 0.8842 | 0.5772 | +0.3069 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 76 | `up_proj` | 1.4896 | yes | 0.6876 | 0.9340 | 0.6283 | +0.3056 | yes | yes | no |
| `activation_weighted_binary_residual` | `r128_b3` | 85 | `up_proj` | 1.2500 | yes | 0.7910 | 0.8804 | 0.5772 | +0.3032 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 85 | `up_proj` | 1.2500 | yes | 0.7910 | 0.8804 | 0.5772 | +0.3032 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 76 | `up_proj` | 0.7448 | yes | 0.5045 | 0.9313 | 0.6283 | +0.3029 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 76 | `up_proj` | 0.3724 | yes | 0.2811 | 0.9288 | 0.6283 | +0.3005 | yes | yes | yes |
| `activation_weighted_binary_residual` | `r384_b4` | 76 | `gate_proj` | 1.5000 | yes | 0.7912 | 0.9224 | 0.6228 | +0.2996 | yes | yes | no |
| `activation_weighted_binary_residual` | `r512_b4` | 76 | `gate_proj` | 1.5000 | yes | 0.7912 | 0.9224 | 0.6228 | +0.2996 | yes | yes | no |
| `activation_weighted_binary_residual` | `r640_b4` | 76 | `gate_proj` | 1.5000 | yes | 0.7912 | 0.9224 | 0.6228 | +0.2996 | yes | yes | no |
| `activation_weighted_binary_residual` | `r256_b3` | 76 | `gate_proj` | 1.4844 | yes | 0.7904 | 0.9222 | 0.6228 | +0.2994 | yes | yes | no |
| `activation_weighted_binary_residual` | `r192_b4` | 76 | `gate_proj` | 1.4609 | yes | 0.7893 | 0.9216 | 0.6228 | +0.2988 | yes | yes | no |
| `raw_weight_low_rank_q` | `r256_b3` | 127 | `down_proj` | 1.4896 | yes | 0.7200 | 0.8508 | 0.5556 | +0.2952 | yes | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 76 | `gate_proj` | 1.2500 | yes | 0.7777 | 0.9171 | 0.6228 | +0.2943 | yes | yes | no |
| `activation_weighted_binary_residual` | `r64_b3` | 76 | `gate_proj` | 1.2500 | yes | 0.7777 | 0.9171 | 0.6228 | +0.2943 | yes | yes | no |
| `activation_weighted_binary_residual` | `r256_b3` | 76 | `up_proj` | 1.4844 | yes | 0.7942 | 0.9223 | 0.6283 | +0.2940 | yes | yes | no |
| `activation_weighted_binary_residual` | `r192_b4` | 76 | `up_proj` | 1.4609 | yes | 0.7930 | 0.9223 | 0.6283 | +0.2940 | yes | yes | no |
| `activation_weighted_binary_residual` | `r384_b4` | 76 | `up_proj` | 1.5000 | yes | 0.7948 | 0.9223 | 0.6283 | +0.2939 | yes | yes | no |
| `activation_weighted_binary_residual` | `r512_b4` | 76 | `up_proj` | 1.5000 | yes | 0.7948 | 0.9223 | 0.6283 | +0.2939 | yes | yes | no |
| `activation_weighted_binary_residual` | `r640_b4` | 76 | `up_proj` | 1.5000 | yes | 0.7948 | 0.9223 | 0.6283 | +0.2939 | yes | yes | no |
| `raw_weight_low_rank_q` | `r128_b3` | 109 | `gate_proj` | 0.7448 | yes | 0.6891 | 0.7533 | 0.4614 | +0.2919 | yes | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 76 | `gate_proj` | 1.4896 | yes | 0.8647 | 0.9136 | 0.6228 | +0.2908 | yes | yes | no |
| `activation_weighted_binary_residual` | `r128_b3` | 76 | `up_proj` | 1.2500 | yes | 0.7827 | 0.9181 | 0.6283 | +0.2898 | yes | yes | no |
| `activation_weighted_binary_residual` | `r64_b3` | 76 | `up_proj` | 1.2500 | yes | 0.7827 | 0.9181 | 0.6283 | +0.2898 | yes | yes | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 85 | `down_proj` | 1.4609 | yes | 0.5902 | 0.9709 | 0.6829 | +0.2880 | yes | yes | no |
| `activation_pca_low_rank_q` | `r192_b4` | 85 | `down_proj` | 1.4609 | yes | 0.5720 | 0.9699 | 0.6829 | +0.2870 | yes | yes | no |
| `raw_weight_low_rank_q` | `r192_b4` | 76 | `gate_proj` | 1.4609 | yes | 0.8705 | 0.9087 | 0.6228 | +0.2860 | yes | yes | no |
| `raw_weight_low_rank_q` | `r256_b3` | 76 | `up_proj` | 1.4896 | yes | 0.8567 | 0.9067 | 0.6283 | +0.2784 | yes | yes | no |
| `raw_weight_low_rank_q` | `r192_b4` | 76 | `up_proj` | 1.4609 | yes | 0.8608 | 0.9057 | 0.6283 | +0.2774 | yes | yes | no |
| `raw_weight_low_rank_q` | `r128_b3` | 109 | `up_proj` | 0.7448 | yes | 0.6821 | 0.7495 | 0.4789 | +0.2706 | yes | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 85 | `down_proj` | 1.4896 | yes | 0.6172 | 0.9477 | 0.6829 | +0.2649 | yes | yes | no |
| `raw_weight_low_rank_q` | `r256_b3` | 85 | `gate_proj` | 1.4896 | yes | 0.7646 | 0.8185 | 0.5585 | +0.2600 | yes | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 127 | `down_proj` | 1.5000 | yes | 0.0177 | 0.8153 | 0.5556 | +0.2597 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 127 | `down_proj` | 1.5000 | yes | 0.0177 | 0.8153 | 0.5556 | +0.2597 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 127 | `down_proj` | 1.5000 | yes | 0.0177 | 0.8153 | 0.5556 | +0.2597 | yes | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 127 | `down_proj` | 1.4609 | yes | 0.6896 | 0.8136 | 0.5556 | +0.2580 | yes | no | no |
| `activation_pca_low_rank_q` | `r128_b3` | 85 | `down_proj` | 0.7448 | yes | 0.4495 | 0.9386 | 0.6829 | +0.2557 | yes | yes | yes |
| `activation_pca_low_rank_q` | `r64_b3` | 85 | `down_proj` | 0.3724 | yes | 0.3242 | 0.9328 | 0.6829 | +0.2499 | yes | yes | yes |
| `activation_weighted_binary_residual` | `r256_b3` | 127 | `down_proj` | 1.4792 | yes | 0.0177 | 0.7997 | 0.5556 | +0.2441 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 127 | `down_proj` | 1.4583 | yes | 0.0177 | 0.7964 | 0.5556 | +0.2409 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 85 | `down_proj` | 0.3724 | yes | 0.2236 | 0.9215 | 0.6829 | +0.2386 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 85 | `down_proj` | 0.7448 | yes | 0.3986 | 0.9202 | 0.6829 | +0.2373 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 85 | `down_proj` | 1.4896 | yes | 0.6290 | 0.9181 | 0.6829 | +0.2352 | yes | yes | no |
| `raw_weight_low_rank_q` | `r256_b3` | 85 | `up_proj` | 1.4896 | yes | 0.7563 | 0.8117 | 0.5772 | +0.2345 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 109 | `down_proj` | 1.4609 | yes | 0.5868 | 0.9739 | 0.7477 | +0.2262 | yes | yes | no |
| `raw_weight_low_rank_q` | `r192_b4` | 85 | `gate_proj` | 1.4609 | yes | 0.7400 | 0.7780 | 0.5585 | +0.2195 | yes | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 127 | `up_proj` | 0.3724 | yes | 0.5557 | 0.6185 | 0.4007 | +0.2177 | yes | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 127 | `gate_proj` | 0.3724 | yes | 0.5608 | 0.6222 | 0.4124 | +0.2098 | yes | no | no |
| `activation_pca_low_rank_q` | `r192_b4` | 76 | `down_proj` | 1.4609 | yes | 0.6867 | 0.9846 | 0.7804 | +0.2042 | yes | yes | no |
| `activation_pca_low_rank_q` | `r256_b3` | 109 | `down_proj` | 1.4896 | yes | 0.6303 | 0.9471 | 0.7477 | +0.1994 | yes | yes | no |
| `raw_weight_low_rank_q` | `r192_b4` | 85 | `up_proj` | 1.4609 | yes | 0.7270 | 0.7729 | 0.5772 | +0.1957 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 109 | `down_proj` | 1.4609 | yes | 0.5765 | 0.9414 | 0.7477 | +0.1937 | yes | yes | no |
| `activation_pca_low_rank_q` | `r128_b3` | 109 | `down_proj` | 0.7448 | yes | 0.4658 | 0.9359 | 0.7477 | +0.1882 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r192_b4` | 76 | `down_proj` | 1.4609 | yes | 0.6667 | 0.9666 | 0.7804 | +0.1862 | yes | yes | no |
| `activation_pca_low_rank_q` | `r64_b3` | 109 | `down_proj` | 0.3724 | yes | 0.3404 | 0.9263 | 0.7477 | +0.1787 | yes | yes | yes |
| `raw_weight_low_rank_q` | `r128_b3` | 76 | `up_proj` | 0.7448 | yes | 0.7277 | 0.8009 | 0.6283 | +0.1725 | yes | no | no |
| `activation_pca_low_rank_q` | `r256_b3` | 76 | `down_proj` | 1.4896 | yes | 0.7374 | 0.9516 | 0.7804 | +0.1712 | yes | yes | no |
| `activation_pca_low_rank_q` | `r128_b3` | 76 | `down_proj` | 0.7448 | yes | 0.5438 | 0.9452 | 0.7804 | +0.1648 | yes | yes | no |
| `raw_weight_low_rank_q` | `r128_b3` | 76 | `gate_proj` | 0.7448 | yes | 0.7386 | 0.7875 | 0.6228 | +0.1647 | yes | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 85 | `down_proj` | 1.5000 | yes | 0.0283 | 0.8418 | 0.6829 | +0.1589 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 85 | `down_proj` | 1.5000 | yes | 0.0283 | 0.8418 | 0.6829 | +0.1589 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 85 | `down_proj` | 1.5000 | yes | 0.0283 | 0.8418 | 0.6829 | +0.1589 | yes | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 127 | `down_proj` | 0.7448 | yes | 0.5563 | 0.7144 | 0.5556 | +0.1588 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 85 | `down_proj` | 1.4792 | yes | 0.0282 | 0.8415 | 0.6829 | +0.1586 | yes | no | no |
| `activation_pca_low_rank_q` | `r64_b3` | 76 | `down_proj` | 0.3724 | yes | 0.3961 | 0.9377 | 0.7804 | +0.1573 | yes | yes | yes |
| `raw_weight_low_rank_q` | `r64_b3` | 109 | `gate_proj` | 0.3724 | yes | 0.5414 | 0.6165 | 0.4614 | +0.1551 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 85 | `down_proj` | 1.4583 | yes | 0.0282 | 0.8362 | 0.6829 | +0.1533 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 76 | `down_proj` | 0.3724 | yes | 0.2150 | 0.9141 | 0.7804 | +0.1337 | yes | yes | yes |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 76 | `down_proj` | 0.7448 | yes | 0.4374 | 0.9135 | 0.7804 | +0.1331 | yes | yes | yes |
| `raw_weight_low_rank_q` | `r256_b3` | 85 | `down_proj` | 1.4896 | yes | 0.6951 | 0.8154 | 0.6829 | +0.1325 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 76 | `down_proj` | 1.4896 | yes | 0.6894 | 0.9113 | 0.7804 | +0.1309 | yes | yes | no |
| `raw_weight_low_rank_q` | `r256_b3` | 109 | `down_proj` | 1.4896 | yes | 0.7066 | 0.8613 | 0.7477 | +0.1137 | yes | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 85 | `down_proj` | 1.4609 | yes | 0.6620 | 0.7952 | 0.6829 | +0.1123 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r128_b3` | 109 | `down_proj` | 0.7448 | yes | 0.3832 | 0.8598 | 0.7477 | +0.1121 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r64_b3` | 109 | `down_proj` | 0.3724 | yes | 0.2047 | 0.8595 | 0.7477 | +0.1119 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r256_b3` | 109 | `down_proj` | 1.4896 | yes | 0.6060 | 0.8589 | 0.7477 | +0.1112 | yes | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 85 | `gate_proj` | 0.7448 | yes | 0.6067 | 0.6693 | 0.5585 | +0.1108 | yes | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 109 | `down_proj` | 1.5000 | yes | 0.0192 | 0.8574 | 0.7477 | +0.1097 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 109 | `down_proj` | 1.5000 | yes | 0.0192 | 0.8574 | 0.7477 | +0.1097 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 109 | `down_proj` | 1.5000 | yes | 0.0192 | 0.8574 | 0.7477 | +0.1097 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 109 | `down_proj` | 1.4792 | yes | 0.0191 | 0.8495 | 0.7477 | +0.1018 | yes | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 109 | `up_proj` | 0.3724 | yes | 0.5304 | 0.5793 | 0.4789 | +0.1005 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 109 | `down_proj` | 1.4583 | yes | 0.0191 | 0.8480 | 0.7477 | +0.1004 | yes | no | no |
| `activation_weighted_binary_residual` | `r384_b4` | 76 | `down_proj` | 1.5000 | yes | 0.0056 | 0.8738 | 0.7804 | +0.0934 | yes | no | no |
| `activation_weighted_binary_residual` | `r512_b4` | 76 | `down_proj` | 1.5000 | yes | 0.0056 | 0.8738 | 0.7804 | +0.0934 | yes | no | no |
| `activation_weighted_binary_residual` | `r640_b4` | 76 | `down_proj` | 1.5000 | yes | 0.0056 | 0.8738 | 0.7804 | +0.0934 | yes | no | no |
| `raw_weight_low_rank_q` | `r256_b3` | 76 | `down_proj` | 1.4896 | yes | 0.7998 | 0.8713 | 0.7804 | +0.0909 | yes | no | no |
| `activation_weighted_binary_residual` | `r256_b3` | 76 | `down_proj` | 1.4792 | yes | 0.0056 | 0.8682 | 0.7804 | +0.0878 | yes | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 109 | `down_proj` | 1.4609 | yes | 0.6758 | 0.8340 | 0.7477 | +0.0863 | yes | no | no |
| `activation_weighted_binary_residual` | `r192_b4` | 76 | `down_proj` | 1.4583 | yes | 0.0055 | 0.8649 | 0.7804 | +0.0845 | yes | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 127 | `down_proj` | 1.2500 | yes | 0.0174 | 0.6384 | 0.5556 | +0.0828 | yes | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 127 | `down_proj` | 1.2500 | yes | 0.0174 | 0.6384 | 0.5556 | +0.0828 | yes | no | no |
| `raw_weight_low_rank_q` | `r192_b4` | 76 | `down_proj` | 1.4609 | yes | 0.7695 | 0.8472 | 0.7804 | +0.0668 | yes | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 85 | `up_proj` | 0.7448 | yes | 0.5935 | 0.6294 | 0.5772 | +0.0522 | yes | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 127 | `down_proj` | 0.3724 | yes | 0.4196 | 0.5786 | 0.5556 | +0.0231 | yes | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 76 | `gate_proj` | 0.3724 | yes | 0.5928 | 0.6452 | 0.6228 | +0.0224 | yes | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 109 | `down_proj` | 0.7448 | yes | 0.5452 | 0.7651 | 0.7477 | +0.0174 | yes | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 76 | `up_proj` | 0.3724 | yes | 0.5772 | 0.6384 | 0.6283 | +0.0100 | yes | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 85 | `down_proj` | 0.7448 | yes | 0.5301 | 0.6807 | 0.6829 | -0.0021 | no | no | no |
| `raw_weight_low_rank_q` | `r128_b3` | 76 | `down_proj` | 0.7448 | yes | 0.6291 | 0.7758 | 0.7804 | -0.0046 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 85 | `down_proj` | 1.2500 | yes | 0.0277 | 0.6545 | 0.6829 | -0.0284 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 85 | `down_proj` | 1.2500 | yes | 0.0277 | 0.6545 | 0.6829 | -0.0284 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 85 | `gate_proj` | 0.3724 | yes | 0.4644 | 0.5269 | 0.5585 | -0.0316 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 109 | `down_proj` | 1.2500 | yes | 0.0188 | 0.6996 | 0.7477 | -0.0480 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 109 | `down_proj` | 1.2500 | yes | 0.0188 | 0.6996 | 0.7477 | -0.0480 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 85 | `up_proj` | 0.3724 | yes | 0.4464 | 0.5184 | 0.5772 | -0.0588 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 109 | `down_proj` | 0.3724 | yes | 0.4133 | 0.6647 | 0.7477 | -0.0829 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 85 | `down_proj` | 0.3724 | yes | 0.3945 | 0.5603 | 0.6829 | -0.1226 | no | no | no |
| `activation_weighted_binary_residual` | `r128_b3` | 76 | `down_proj` | 1.2500 | yes | 0.0054 | 0.6566 | 0.7804 | -0.1237 | no | no | no |
| `activation_weighted_binary_residual` | `r64_b3` | 76 | `down_proj` | 1.2500 | yes | 0.0054 | 0.6566 | 0.7804 | -0.1237 | no | no | no |
| `raw_weight_low_rank_q` | `r64_b3` | 76 | `down_proj` | 0.3724 | yes | 0.4818 | 0.6418 | 0.7804 | -0.1386 | no | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 127 | `up_proj` | 4.8698 | no | 0.9845 | 0.9871 | 0.4007 | +0.5863 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 127 | `up_proj` | 3.8958 | no | 0.9707 | 0.9770 | 0.4007 | +0.5763 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 127 | `gate_proj` | 4.8698 | no | 0.9847 | 0.9885 | 0.4124 | +0.5761 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 127 | `gate_proj` | 3.8958 | no | 0.9727 | 0.9796 | 0.4124 | +0.5672 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 127 | `up_proj` | 4.8698 | no | 0.8482 | 0.9656 | 0.4007 | +0.5648 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 127 | `up_proj` | 3.8958 | no | 0.8431 | 0.9650 | 0.4007 | +0.5642 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 127 | `up_proj` | 2.9219 | no | 0.8296 | 0.9632 | 0.4007 | +0.5625 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 127 | `gate_proj` | 4.8698 | no | 0.8425 | 0.9730 | 0.4124 | +0.5606 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 127 | `gate_proj` | 3.8958 | no | 0.8380 | 0.9728 | 0.4124 | +0.5604 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 127 | `gate_proj` | 2.9219 | no | 0.8261 | 0.9725 | 0.4124 | +0.5601 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 127 | `up_proj` | 2.9219 | no | 0.9431 | 0.9585 | 0.4007 | +0.5577 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 127 | `gate_proj` | 2.9219 | no | 0.9476 | 0.9583 | 0.4124 | +0.5459 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 109 | `gate_proj` | 4.8698 | no | 0.9840 | 0.9893 | 0.4614 | +0.5279 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 109 | `gate_proj` | 4.8698 | no | 0.6583 | 0.9857 | 0.4614 | +0.5243 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 109 | `gate_proj` | 3.8958 | no | 0.6079 | 0.9845 | 0.4614 | +0.5231 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 109 | `gate_proj` | 2.9219 | no | 0.5470 | 0.9830 | 0.4614 | +0.5216 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 109 | `gate_proj` | 3.8958 | no | 0.9692 | 0.9780 | 0.4614 | +0.5165 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 109 | `up_proj` | 4.8698 | no | 0.9839 | 0.9887 | 0.4789 | +0.5098 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 109 | `gate_proj` | 4.8698 | no | 0.8616 | 0.9675 | 0.4614 | +0.5061 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 109 | `gate_proj` | 3.8958 | no | 0.8548 | 0.9674 | 0.4614 | +0.5060 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 109 | `gate_proj` | 2.9219 | no | 0.8386 | 0.9671 | 0.4614 | +0.5057 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 109 | `up_proj` | 4.8698 | no | 0.6454 | 0.9837 | 0.4789 | +0.5048 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 109 | `up_proj` | 3.8958 | no | 0.5948 | 0.9826 | 0.4789 | +0.5037 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 109 | `up_proj` | 2.9219 | no | 0.5342 | 0.9809 | 0.4789 | +0.5021 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 109 | `gate_proj` | 2.9219 | no | 0.9415 | 0.9596 | 0.4614 | +0.4982 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 109 | `up_proj` | 3.8958 | no | 0.9679 | 0.9766 | 0.4789 | +0.4977 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 109 | `up_proj` | 4.8698 | no | 0.8657 | 0.9724 | 0.4789 | +0.4936 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 109 | `up_proj` | 3.8958 | no | 0.8582 | 0.9723 | 0.4789 | +0.4934 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 109 | `up_proj` | 2.9219 | no | 0.8405 | 0.9718 | 0.4789 | +0.4929 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 109 | `up_proj` | 2.9219 | no | 0.9370 | 0.9554 | 0.4789 | +0.4765 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 127 | `down_proj` | 4.8698 | no | 0.9859 | 0.9917 | 0.5556 | +0.4361 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 85 | `gate_proj` | 4.8698 | no | 0.9808 | 0.9864 | 0.5585 | +0.4279 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 85 | `gate_proj` | 4.8698 | no | 0.6257 | 0.9859 | 0.5585 | +0.4274 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 85 | `gate_proj` | 3.8958 | no | 0.5754 | 0.9848 | 0.5585 | +0.4264 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 85 | `gate_proj` | 2.9219 | no | 0.5140 | 0.9836 | 0.5585 | +0.4251 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 85 | `gate_proj` | 4.8698 | no | 0.9079 | 0.9825 | 0.5585 | +0.4240 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 85 | `gate_proj` | 3.8958 | no | 0.8925 | 0.9822 | 0.5585 | +0.4237 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 85 | `gate_proj` | 2.9219 | no | 0.8589 | 0.9816 | 0.5585 | +0.4231 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 127 | `down_proj` | 3.8958 | no | 0.9420 | 0.9705 | 0.5556 | +0.4149 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 127 | `down_proj` | 4.8698 | no | 0.8824 | 0.9673 | 0.5556 | +0.4117 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 127 | `down_proj` | 3.8958 | no | 0.8634 | 0.9671 | 0.5556 | +0.4115 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 85 | `up_proj` | 4.8698 | no | 0.6186 | 0.9882 | 0.5772 | +0.4110 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 127 | `down_proj` | 2.9219 | no | 0.8164 | 0.9665 | 0.5556 | +0.4109 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 85 | `up_proj` | 3.8958 | no | 0.5675 | 0.9877 | 0.5772 | +0.4104 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 85 | `up_proj` | 2.9219 | no | 0.5056 | 0.9868 | 0.5772 | +0.4096 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 85 | `up_proj` | 4.8698 | no | 0.9807 | 0.9858 | 0.5772 | +0.4085 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 85 | `gate_proj` | 3.8958 | no | 0.9502 | 0.9634 | 0.5585 | +0.4049 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 85 | `up_proj` | 4.8698 | no | 0.9116 | 0.9815 | 0.5772 | +0.4042 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 85 | `up_proj` | 3.8958 | no | 0.8949 | 0.9813 | 0.5772 | +0.4040 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 85 | `up_proj` | 2.9219 | no | 0.8586 | 0.9809 | 0.5772 | +0.4036 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 85 | `up_proj` | 3.8958 | no | 0.9481 | 0.9626 | 0.5772 | +0.3854 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 127 | `down_proj` | 2.9219 | no | 0.8707 | 0.9362 | 0.5556 | +0.3806 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 76 | `gate_proj` | 3.8958 | no | 0.9856 | 0.9913 | 0.6228 | +0.3685 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 76 | `gate_proj` | 4.8698 | no | 0.9856 | 0.9913 | 0.6228 | +0.3685 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 85 | `gate_proj` | 2.9219 | no | 0.8980 | 0.9266 | 0.5585 | +0.3681 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 76 | `gate_proj` | 4.8698 | no | 0.6554 | 0.9873 | 0.6228 | +0.3645 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 76 | `gate_proj` | 3.8958 | no | 0.6108 | 0.9867 | 0.6228 | +0.3639 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 76 | `gate_proj` | 2.9219 | no | 0.5529 | 0.9853 | 0.6228 | +0.3625 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 76 | `up_proj` | 4.8698 | no | 0.9852 | 0.9903 | 0.6283 | +0.3620 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 76 | `up_proj` | 3.8958 | no | 0.9852 | 0.9903 | 0.6283 | +0.3620 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 76 | `gate_proj` | 2.9219 | no | 0.9707 | 0.9817 | 0.6228 | +0.3589 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 76 | `gate_proj` | 4.8698 | no | 0.8333 | 0.9788 | 0.6228 | +0.3560 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 76 | `gate_proj` | 3.8958 | no | 0.8333 | 0.9788 | 0.6228 | +0.3560 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 76 | `gate_proj` | 2.9219 | no | 0.8262 | 0.9787 | 0.6228 | +0.3559 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 76 | `up_proj` | 4.8698 | no | 0.6492 | 0.9843 | 0.6283 | +0.3559 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 76 | `up_proj` | 3.8958 | no | 0.6022 | 0.9832 | 0.6283 | +0.3548 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 76 | `up_proj` | 2.9219 | no | 0.5452 | 0.9817 | 0.6283 | +0.3534 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 76 | `up_proj` | 2.9219 | no | 0.9681 | 0.9811 | 0.6283 | +0.3528 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 76 | `up_proj` | 4.8698 | no | 0.8425 | 0.9792 | 0.6283 | +0.3508 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 76 | `up_proj` | 3.8958 | no | 0.8425 | 0.9792 | 0.6283 | +0.3508 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 76 | `up_proj` | 2.9219 | no | 0.8344 | 0.9789 | 0.6283 | +0.3506 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 85 | `up_proj` | 2.9219 | no | 0.8926 | 0.9177 | 0.5772 | +0.3405 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 85 | `down_proj` | 4.8698 | no | 0.9704 | 0.9898 | 0.6829 | +0.3069 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 85 | `down_proj` | 3.8958 | no | 0.8865 | 0.9876 | 0.6829 | +0.3047 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 85 | `down_proj` | 4.8698 | no | 0.9747 | 0.9856 | 0.6829 | +0.3027 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 85 | `down_proj` | 2.9219 | no | 0.7815 | 0.9834 | 0.6829 | +0.3006 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 85 | `down_proj` | 4.8698 | no | 0.9174 | 0.9729 | 0.6829 | +0.2901 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 85 | `down_proj` | 3.8958 | no | 0.8890 | 0.9729 | 0.6829 | +0.2900 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 85 | `down_proj` | 2.9219 | no | 0.8276 | 0.9725 | 0.6829 | +0.2896 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 85 | `down_proj` | 3.8958 | no | 0.9232 | 0.9589 | 0.6829 | +0.2760 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 109 | `down_proj` | 4.8698 | no | 0.9781 | 0.9920 | 0.7477 | +0.2444 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 109 | `down_proj` | 4.8698 | no | 0.9814 | 0.9913 | 0.7477 | +0.2437 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 109 | `down_proj` | 3.8958 | no | 0.8956 | 0.9904 | 0.7477 | +0.2427 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 109 | `down_proj` | 2.9219 | no | 0.7931 | 0.9870 | 0.7477 | +0.2393 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 85 | `down_proj` | 2.9219 | no | 0.8473 | 0.9119 | 0.6829 | +0.2290 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 109 | `down_proj` | 3.8958 | no | 0.9321 | 0.9721 | 0.7477 | +0.2244 | yes | no | no |
| `raw_weight_low_rank_q` | `r640_b4` | 76 | `down_proj` | 4.8698 | no | 0.9872 | 0.9908 | 0.7804 | +0.2104 | yes | no | no |
| `raw_weight_low_rank_q` | `r512_b4` | 76 | `down_proj` | 3.8958 | no | 0.9872 | 0.9908 | 0.7804 | +0.2104 | yes | no | no |
| `activation_pca_low_rank_q` | `r512_b4` | 76 | `down_proj` | 3.8958 | no | 0.9881 | 0.9908 | 0.7804 | +0.2104 | yes | no | no |
| `activation_pca_low_rank_q` | `r640_b4` | 76 | `down_proj` | 4.8698 | no | 0.9881 | 0.9908 | 0.7804 | +0.2104 | yes | no | no |
| `activation_pca_low_rank_q` | `r384_b4` | 76 | `down_proj` | 2.9219 | no | 0.9216 | 0.9899 | 0.7804 | +0.2095 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 109 | `down_proj` | 4.8698 | no | 0.8994 | 0.9436 | 0.7477 | +0.1959 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 109 | `down_proj` | 3.8958 | no | 0.8730 | 0.9435 | 0.7477 | +0.1958 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 109 | `down_proj` | 2.9219 | no | 0.8118 | 0.9432 | 0.7477 | +0.1955 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 109 | `down_proj` | 2.9219 | no | 0.8579 | 0.9391 | 0.7477 | +0.1914 | yes | no | no |
| `raw_weight_low_rank_q` | `r384_b4` | 76 | `down_proj` | 2.9219 | no | 0.9476 | 0.9714 | 0.7804 | +0.1910 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r384_b4` | 76 | `down_proj` | 2.9219 | no | 0.8484 | 0.9671 | 0.7804 | +0.1867 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r512_b4` | 76 | `down_proj` | 3.8958 | no | 0.8694 | 0.9670 | 0.7804 | +0.1866 | yes | no | no |
| `activation_weighted_svd_low_rank_q` | `r640_b4` | 76 | `down_proj` | 4.8698 | no | 0.8694 | 0.9670 | 0.7804 | +0.1866 | yes | no | no |

## Provenance

- Capture run: `workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1/runs/main_20260809T195857Z`
- Capture result sha256: `4c721dc855ad3fbe0699e41244106da277c5983e1404ea6dafa9d43b90865037`
- Hidden source: device-produced L0 post-attention RMSNorm (router input)
- Probes: code_fib_iter, code_py_bisect, code_sql_window, code_go_http, code_ts_reduce, code_long_pq, prose_measure, prose_short_haiku, prose_argument, prose_narrative, json_status_strict, json_schema_example, json_array_table, json_nested_error, multi_turn_moe, multi_turn_debug, multi_turn_review, long_ctx_notes, long_ctx_log, math_bayes, math_rank, math_entropy, instr_checklist, instr_refuse, instr_compare, dialogue_lab, list_domains, mixed_api, mixed_regex, code_rust_parse, prose_policy, structured_yamlish
- Token-expert pairs: 31432
- Experts: [109, 85, 76, 127] hits={'109': 1623, '85': 1549, '76': 1145, '127': 248}
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
