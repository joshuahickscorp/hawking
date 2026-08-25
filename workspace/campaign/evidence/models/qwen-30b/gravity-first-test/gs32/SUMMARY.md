# Qwen3-Coder-30B Gravity First Test — Summary

Source: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct`
Method: symmetric group absmax RTN, group_size=32
Layers: [0, 24]  Experts: [0, 7]

## By bit width

### 8-bit  complete_bpw≈8.5002  (n_weights=56623104)
- weight recon: mean_cos=0.999979 mean_rel_l2=0.005698
- expert FFN: mean_cos=0.999968 min_cos=0.999959 mean_rel_l2=0.007393 hold=4/4 notes=['near_lossless']
- attn matmul: mean_cos=0.999988 min_cos=0.999979 mean_rel_l2=0.004463
- layer 0 MoE combine: cos=0.999987 rel_l2=0.004187 holds=True (near_lossless)
- layer 24 MoE combine: cos=0.999958 rel_l2=0.009004 holds=True (near_lossless)

### 4-bit  complete_bpw≈4.5002  (n_weights=56623104)
- weight recon: mean_cos=0.994685 mean_rel_l2=0.103273
- expert FFN: mean_cos=0.989593 min_cos=0.986725 mean_rel_l2=0.134562 hold=2/4 notes=['borderline_capable', 'degraded_math_preserve_risk', 'mild_degradation_likely_capable']
- attn matmul: mean_cos=0.996256 min_cos=0.993369 mean_rel_l2=0.080730
- layer 0 MoE combine: cos=0.995532 rel_l2=0.075818 holds=True (mild_degradation_likely_capable)
- layer 24 MoE combine: cos=0.985273 rel_l2=0.166363 holds=False (degraded_math_preserve_risk)

### 3-bit  complete_bpw≈3.5002  (n_weights=56623104)
- weight recon: mean_cos=0.972409 mean_rel_l2=0.239999
- expert FFN: mean_cos=0.944202 min_cos=0.929320 mean_rel_l2=0.325245 hold=0/4 notes=['degraded_math_preserve_risk']
- attn matmul: mean_cos=0.980388 min_cos=0.965686 mean_rel_l2=0.187939
- layer 0 MoE combine: cos=0.977994 rel_l2=0.175235 holds=False (degraded_math_preserve_risk)
- layer 24 MoE combine: cos=0.929932 rel_l2=0.406853 holds=False (degraded_math_preserve_risk)

## Honest floor
- expert-FFN floor nbits: None
- note: no_level_held

## Math-Preserve-style hits
- none under the (weight_cos>=0.99 & ffn_cos<0.95) rule

