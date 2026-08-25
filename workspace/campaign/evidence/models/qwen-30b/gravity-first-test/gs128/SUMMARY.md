# Qwen3-Coder-30B Gravity First Test — Summary

Source: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct`
Method: symmetric group absmax RTN, group_size=128
Layers: [0, 24]  Experts: [0, 7]

## By bit width

### 8-bit  complete_bpw≈8.1252  (n_weights=56623104)
- weight recon: mean_cos=0.999970 mean_rel_l2=0.007142
- expert FFN: mean_cos=0.999952 min_cos=0.999939 mean_rel_l2=0.009067 hold=4/4 notes=['near_lossless']
- attn matmul: mean_cos=0.999982 min_cos=0.999968 mean_rel_l2=0.005616
- layer 0 MoE combine: cos=0.999979 rel_l2=0.005261 holds=True (near_lossless)
- layer 24 MoE combine: cos=0.999939 rel_l2=0.010630 holds=True (near_lossless)

### 4-bit  complete_bpw≈4.1252  (n_weights=56623104)
- weight recon: mean_cos=0.991631 mean_rel_l2=0.129417
- expert FFN: mean_cos=0.984137 min_cos=0.980318 mean_rel_l2=0.166157 hold=1/4 notes=['borderline_capable', 'degraded_math_preserve_risk']
- attn matmul: mean_cos=0.994083 min_cos=0.989631 mean_rel_l2=0.102252
- layer 0 MoE combine: cos=0.993193 rel_l2=0.092865 holds=True (mild_degradation_likely_capable)
- layer 24 MoE combine: cos=0.979576 rel_l2=0.207756 holds=False (degraded_math_preserve_risk)

### 3-bit  complete_bpw≈3.1252  (n_weights=56623104)
- weight recon: mean_cos=0.957619 mean_rel_l2=0.299381
- expert FFN: mean_cos=0.917911 min_cos=0.899701 mean_rel_l2=0.397704 hold=0/4 notes=['collapse', 'degraded_math_preserve_risk']
- attn matmul: mean_cos=0.970011 min_cos=0.948764 mean_rel_l2=0.235878
- layer 0 MoE combine: cos=0.960156 rel_l2=0.211567 holds=False (degraded_math_preserve_risk)
- layer 24 MoE combine: cos=0.897101 rel_l2=0.483945 holds=False (collapse)

## Honest floor
- expert-FFN floor nbits: None
- note: no_level_held

## Math-Preserve-style hits
- none under the (weight_cos>=0.99 & ffn_cos<0.95) rule

