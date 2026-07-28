# GLM-5.2 bounded real-activation basis pilot (revision 1)

- schema: `hawking.glm52.basis_pilot.v1`
- revision: `1`
- at: 2026-07-28T12:39:07Z
- seed: `738581`
- ranks: [16, 64, 128, 256, 512]
- elapsed_s: 320.615
- full_traversal_authorized: `False`

## Distinguishing story

RETAINING_MEAN_HELPS_CENTERED_RESIDUAL: uncentered and equal-byte explicit-mean are numerically tied (|median lift difference| < 0.0001). Retaining the mean helps vs centered residual on held-out real activations. The implementation choice between uncentered and explicit-mean remains unresolved when B≈C. Production centering is a real defect; this is not whole-model capability proof.

- story code: `RETAINING_MEAN_HELPS_CENTERED_RESIDUAL`
- mean median lift explicit_mean − centered: 0.08234207332134247
- mean median lift uncentered − centered: 0.08234201371669769
- mean median lift explicit_mean − uncentered: 5.960464477539063e-08
- numerical equivalence tolerance: 0.0001
- uncentered/explicit_mean tied: `True`

## Revision 0 evidence (preserved)

- label: `revision_0`
- sha256: `8928fad82967d2fc06e0ccad470f7e527b842faf766df20f4148e0db3f8ba357`
- at: 2026-07-28T12:16:25Z
- prior story: EXPLICIT_MEAN_HELPS: equal-byte explicit mean + residual beats centered residual on held-out real activations. Production centering is a real defect, but this is not whole-model capability proof.
- note: First measured pass preserved as evidence. Not promotion-valid: requested ranks conflated with capped ranks; down_proj measured only in production output space; floors mixed low-traffic diagnostics.

## Promotion-grade panel (uncapped ranks only)

Equal-byte median / min cosine. Points with `rank_capped` or `total_rank != requested_rank` are excluded. Included/excluded counts shown.

| rank | centered med/min (n_in/n_ex) | uncentered | explicit_mean |
|---:|---|---|---|
| 16 | 0.7901 / 0.5450 (11/0) | 0.9525 / 0.7757 (11/0) | 0.9525 / 0.7758 (11/0) |
| 64 | 0.8589 / 0.6661 (11/0) | 0.9695 / 0.8594 (11/0) | 0.9695 / 0.8594 (11/0) |
| 128 | 0.8763 / 0.6821 (11/0) | 0.9748 / 0.8937 (11/0) | 0.9748 / 0.8937 (11/0) |
| 256 | 0.9053 / 0.7167 (11/0) | 0.9793 / 0.9228 (11/0) | 0.9793 / 0.9228 (11/0) |
| 512 | 0.9370 / 0.7536 (11/0) | 0.9834 / 0.9491 (11/0) | 0.9834 / 0.9491 (11/0) |

## Separate panel aggregates @ rank 256 (uncapped)

### `promotion_grade_high_traffic_routed`

- centered: med=0.9053 min=0.7167 n_included=11 n_excluded=0
- uncentered: med=0.9793 min=0.9228 n_included=11 n_excluded=0
- explicit_mean: med=0.9793 min=0.9228 n_included=11 n_excluded=0

### `shared_mlp`

- centered: med=0.8861 min=0.8616 n_included=3 n_excluded=0
- uncentered: med=0.9362 min=0.9182 n_included=3 n_excluded=0
- explicit_mean: med=0.9362 min=0.9182 n_included=3 n_excluded=0

### `attention_router_controls`

- centered: med=0.8397 min=0.8114 n_included=2 n_excluded=0
- uncentered: med=0.9641 min=0.9325 n_included=2 n_excluded=0
- explicit_mean: med=0.9641 min=0.9325 n_included=2 n_excluded=0

### `low_traffic_diagnostics`

- centered: n_included=0 n_excluded=3
- uncentered: n_included=0 n_excluded=3
- explicit_mean: n_included=0 n_excluded=3

## Floor checks (promotion-grade panel only)

- math_live_capability / centered @ rank 256 panel=`promotion_grade_high_traffic_routed`: **FAILS** (min=0.7167, med=0.9053, n_in=11, n_ex=0)
- math_live_capability / uncentered @ rank 256 panel=`promotion_grade_high_traffic_routed`: **CLEARS** (min=0.9228, med=0.9793, n_in=11, n_ex=0)
- math_live_capability / explicit_mean @ rank 256 panel=`promotion_grade_high_traffic_routed`: **CLEARS** (min=0.9228, med=0.9793, n_in=11, n_ex=0)
- strong_capability / centered @ rank 512 panel=`promotion_grade_high_traffic_routed`: **FAILS** (min=0.7536, med=0.9370, n_in=11, n_ex=0)
- strong_capability / uncentered @ rank 512 panel=`promotion_grade_high_traffic_routed`: **CLEARS** (min=0.9491, med=0.9834, n_in=11, n_ex=0)
- strong_capability / explicit_mean @ rank 512 panel=`promotion_grade_high_traffic_routed`: **CLEARS** (min=0.9491, med=0.9834, n_in=11, n_ex=0)

## Down analyses @ rank 256

### production_output_side_down_negative_control (NOT promotional)

- `model.layers.5.mlp.experts.11.down_proj.weight` panel=promotion_grade_high_traffic_routed total_rank=256 capped=False cosine=0.3576 basis_width=6144 side=output
- `model.layers.38.mlp.experts.73.down_proj.weight` panel=promotion_grade_high_traffic_routed total_rank=256 capped=False cosine=0.2471 basis_width=6144 side=output
- `model.layers.74.mlp.experts.118.down_proj.weight` panel=promotion_grade_high_traffic_routed total_rank=256 capped=False cosine=0.2181 basis_width=6144 side=output
- `model.layers.5.mlp.experts.100.down_proj.weight` panel=low_traffic_diagnostics total_rank=164 capped=True cosine=0.1920 basis_width=6144 side=output
- `model.layers.38.mlp.shared_experts.down_proj.weight` panel=shared_mlp total_rank=256 capped=False cosine=0.3514 basis_width=6144 side=output

### activation_matched_input_side_down (promotion metric)

- `model.layers.5.mlp.experts.11.down_proj.weight` panel=promotion_grade_high_traffic_routed total_rank=256 capped=False cosine=0.9986 basis_width=2048 side=input
- `model.layers.38.mlp.experts.73.down_proj.weight` panel=promotion_grade_high_traffic_routed total_rank=256 capped=False cosine=0.9228 basis_width=2048 side=input
- `model.layers.74.mlp.experts.118.down_proj.weight` panel=promotion_grade_high_traffic_routed total_rank=256 capped=False cosine=0.9580 basis_width=2048 side=input
- `model.layers.5.mlp.experts.100.down_proj.weight` panel=low_traffic_diagnostics total_rank=164 capped=True cosine=0.9187 basis_width=2048 side=input
- `model.layers.38.mlp.shared_experts.down_proj.weight` panel=shared_mlp total_rank=256 capped=False cosine=0.9182 basis_width=2048 side=input

## Panel-total encoded bytes @ rank 256 (exact arithmetic)

- accounting scope: exact_arithmetic_per_tensor_payload_float16_header_plus_coefficients_plus_basis; sum over uncapped promotion-eligible points only
- is_physical_file_measurement: `False`
- promotion-grade total bytes: `46138048`

- `promotion_grade_high_traffic_routed`: n=11 total_bytes=46138048 mean_bpw=2.6667073567708335
- `shared_mlp`: n=3 total_bytes=12583104 mean_bpw=2.6667073567708335
- `attention_router_controls`: n=2 total_bytes=7471232 mean_bpw=9.666849772135416
- `low_traffic_diagnostics`: n=0 total_bytes=0 mean_bpw=None

## Inputs (hashes)

- pilot code: `dcce2d751da36e196ee4b4634d5ae35078f4ed59ac800533fc53d832efa7be5d`
- pack module: `321cbd1f071683702bfc61c293785eb44cfbf6bb6098bde0f039134c4b3f4832`

### Verified shards

- `model-00108-of-00282.safetensors` sha256=`d201fa9064deea80…` verified=True
- `model-00156-of-00282.safetensors` sha256=`abc8f728e74c908d…` verified=True
- `model-00157-of-00282.safetensors` sha256=`219d858b98458070…` verified=True
- `model-00112-of-00282.safetensors` sha256=`07c4b6c43744d2aa…` verified=True
- `model-00256-of-00282.safetensors` sha256=`d4c1064d459bb858…` verified=True

### Capsules

- `L05_L05.npz` sha256=`8fb88dc4e804afb9…`
- `L38_L41.npz` sha256=`a76f903af3283bdd…`
- `L70_L75.npz` sha256=`b8044c06032fd89a…`

## Missing classes

- **global_embed_tokens**: embed_tokens is not among the five resident pilot shards
- **global_lm_head**: lm_head is not among the five resident pilot shards
- **attention_o_proj_real_intermediate**: o_proj is [6144,16384]; capsules retain attention_input/output at hidden=6144 but not the 16384-wide attention intermediate. Gaussian input is forbidden for promotion, so o_proj is omitted rather than scored under a proxy.

## Per-tensor status (promotion arms @ rank 256)

- `model.layers.5.mlp.experts.11.gate_proj.weight` panel=promotion_grade_high_traffic_routed routes=3529 fit/hold=2823/706 centered=0.9449[ok] uncentered=0.9987[ok] explicit_mean=0.9987[ok]
- `model.layers.5.mlp.experts.11.up_proj.weight` panel=promotion_grade_high_traffic_routed routes=3529 fit/hold=2823/706 centered=0.9523[ok] uncentered=0.9986[ok] explicit_mean=0.9986[ok]
- `model.layers.5.mlp.experts.11.down_proj.weight` panel=promotion_grade_high_traffic_routed routes=3529 fit/hold=2823/706 centered=0.9890[ok] uncentered=0.9986[ok] explicit_mean=0.9986[ok] neg_ctrl=0.3576
- `model.layers.5.mlp.experts.165.gate_proj.weight` panel=promotion_grade_high_traffic_routed routes=3391 fit/hold=2713/678 centered=0.9132[ok] uncentered=0.9992[ok] explicit_mean=0.9992[ok]
- `model.layers.5.mlp.experts.165.up_proj.weight` panel=promotion_grade_high_traffic_routed routes=3391 fit/hold=2713/678 centered=0.9053[ok] uncentered=0.9992[ok] explicit_mean=0.9992[ok]
- `model.layers.38.mlp.experts.73.gate_proj.weight` panel=promotion_grade_high_traffic_routed routes=3227 fit/hold=2582/645 centered=0.8208[ok] uncentered=0.9610[ok] explicit_mean=0.9610[ok]
- `model.layers.38.mlp.experts.73.up_proj.weight` panel=promotion_grade_high_traffic_routed routes=3227 fit/hold=2582/645 centered=0.8675[ok] uncentered=0.9552[ok] explicit_mean=0.9552[ok]
- `model.layers.38.mlp.experts.73.down_proj.weight` panel=promotion_grade_high_traffic_routed routes=3227 fit/hold=2582/645 centered=0.9009[ok] uncentered=0.9228[ok] explicit_mean=0.9228[ok] neg_ctrl=0.2471
- `model.layers.74.mlp.experts.118.gate_proj.weight` panel=promotion_grade_high_traffic_routed routes=2577 fit/hold=2062/515 centered=0.7167[ok] uncentered=0.9793[ok] explicit_mean=0.9793[ok]
- `model.layers.74.mlp.experts.118.up_proj.weight` panel=promotion_grade_high_traffic_routed routes=2577 fit/hold=2062/515 centered=0.8374[ok] uncentered=0.9722[ok] explicit_mean=0.9722[ok]
- `model.layers.74.mlp.experts.118.down_proj.weight` panel=promotion_grade_high_traffic_routed routes=2577 fit/hold=2062/515 centered=0.9264[ok] uncentered=0.9580[ok] explicit_mean=0.9580[ok] neg_ctrl=0.2181
- `model.layers.5.mlp.experts.100.gate_proj.weight` panel=low_traffic_diagnostics routes=205 fit/hold=164/41 centered=0.8494[capped] uncentered=0.9358[capped] explicit_mean=0.9358[capped]
- `model.layers.5.mlp.experts.100.up_proj.weight` panel=low_traffic_diagnostics routes=205 fit/hold=164/41 centered=0.8525[capped] uncentered=0.9365[capped] explicit_mean=0.9365[capped]
- `model.layers.5.mlp.experts.100.down_proj.weight` panel=low_traffic_diagnostics routes=205 fit/hold=164/41 centered=0.8696[capped] uncentered=0.9187[capped] explicit_mean=0.9187[capped] neg_ctrl=0.1920
- `model.layers.38.mlp.shared_experts.gate_proj.weight` panel=shared_mlp routes=4096 fit/hold=3277/819 centered=0.8861[ok] uncentered=0.9699[ok] explicit_mean=0.9699[ok]
- `model.layers.38.mlp.shared_experts.up_proj.weight` panel=shared_mlp routes=4096 fit/hold=3277/819 centered=0.8616[ok] uncentered=0.9362[ok] explicit_mean=0.9362[ok]
- `model.layers.38.mlp.shared_experts.down_proj.weight` panel=shared_mlp routes=4096 fit/hold=3277/819 centered=0.8933[ok] uncentered=0.9182[ok] explicit_mean=0.9182[ok] neg_ctrl=0.3514
- `model.layers.38.self_attn.q_a_proj.weight` panel=attention_router_controls routes=4096 fit/hold=3277/819 centered=0.8679[ok] uncentered=0.9325[ok] explicit_mean=0.9325[ok]
- `model.layers.38.mlp.gate.weight` panel=attention_router_controls routes=4096 fit/hold=3277/819 centered=0.8114[ok] uncentered=0.9957[ok] explicit_mean=0.9957[ok]

## Safety

- full_parent_traversal_started: `False`
- teacher_capsules_modified: `False`
- prior_artifacts_modified: `False`
- MOP_touched: `False`
- production_defaults_changed: `False`
- ODYSSEY_LAUNCH_AUTHORIZED: `False`
- RAMANUJAN_RESEARCH_AUTHORIZED: `False`
- HIDE_KERNEL_TURN: `False`
- gaussian_proxy_used_for_selection: `False`
- full_traversal_authorized: `False`

## Remaining uncertainty

- Five shards / ~19 tensors cannot decide whole-model generation.
- Calibrated cosine floors came from Llama-1B; GLM absolute floors may differ.
- Shared layer bases transferred across experts were not the primary fit mode; route-conditioned per-expert bases were used for routed tensors.
- o_proj and global embed/lm_head remain unmeasured on real intermediates.
- Winning a basis arm does not authorize flipping production defaults.
- Uncentered and explicit-mean remain numerically tied; implementation choice unresolved when median difference is below tolerance.
- activation_matched_input_side_down may still fail floors; if so, down remains unsalvageable under equal-byte linear subspace codecs and needs a different representation (e.g. SwiGLU-joint).
- Byte totals are exact arithmetic payload estimates, not physical serialized pack-file measurements.

## Next safe action

Treat uncentered and equal-byte explicit-mean as numerically tied; do not declare one uniquely superior. Prefer an opt-in pack basis flag that retains the mean (either implementation), not a production default flip. Do not start a 282-shard traversal: full_traversal_authorized is false unless every preregistered promotion-grade floor clears. Never promote on beats_null, reconstruction error, or low-traffic diagnostics.

## Not claimed

A bounded tensor pilot does **not** prove whole-model capability. `beats_null`, reconstruction error, the invalid all-row diagnostic, low-traffic diagnostics, and the production output-side down negative control are non-promotional. Do not change production defaults merely because an arm wins. Do not declare explicit-mean uniquely superior when it is numerically tied with uncentered.
