# Qwen3-Coder-30B Gravity First Test — Summary

Source: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct`
Method: symmetric group absmax RTN, group_size=64
Layers: [0, 24]  Experts: [0, 7]

## By bit width

### 8-bit  complete_bpw≈8.2502  (n_weights=56623104)
- weight recon: mean_cos=0.999975 mean_rel_l2=0.006436
- expert FFN: mean_cos=0.999960 min_cos=0.999949 mean_rel_l2=0.008274 hold=4/4 notes=['near_lossless']
- attn matmul: mean_cos=0.999985 min_cos=0.999974 mean_rel_l2=0.005038
- layer 0 MoE combine: cos=0.999983 rel_l2=0.004852 holds=True (near_lossless)
- layer 24 MoE combine: cos=0.999949 rel_l2=0.009811 holds=True (near_lossless)

### 4-bit  complete_bpw≈4.2502  (n_weights=56623104)
- weight recon: mean_cos=0.993211 mean_rel_l2=0.116639
- expert FFN: mean_cos=0.986851 min_cos=0.983388 mean_rel_l2=0.151378 hold=2/4 notes=['borderline_capable', 'degraded_math_preserve_risk', 'mild_degradation_likely_capable']
- attn matmul: mean_cos=0.995231 min_cos=0.991556 mean_rel_l2=0.091466
- layer 0 MoE combine: cos=0.994480 rel_l2=0.086756 holds=True (mild_degradation_likely_capable)
- layer 24 MoE combine: cos=0.982592 rel_l2=0.187101 holds=False (degraded_math_preserve_risk)

### 3-bit  complete_bpw≈3.2502  (n_weights=56623104)
- weight recon: mean_cos=0.965168 mean_rel_l2=0.270529
- expert FFN: mean_cos=0.931757 min_cos=0.914369 mean_rel_l2=0.360933 hold=0/4 notes=['degraded_math_preserve_risk']
- attn matmul: mean_cos=0.975284 min_cos=0.957186 mean_rel_l2=0.212551
- layer 0 MoE combine: cos=0.968864 rel_l2=0.194970 holds=False (degraded_math_preserve_risk)
- layer 24 MoE combine: cos=0.918648 rel_l2=0.439268 holds=False (degraded_math_preserve_risk)

## Honest floor
- expert-FFN floor nbits: None
- note: no_level_held

## Math-Preserve-style hits
- none under the (weight_cos>=0.99 & ffn_cos<0.95) rule

