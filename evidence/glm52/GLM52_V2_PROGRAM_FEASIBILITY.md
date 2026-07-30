# GLM-5.2 activation-aware pack v2 — program feasibility (revision 1)

Opt-in, source-body-free feasibility for the next representation program.
Corrects Generation B defects (relative admission, centered-mean loss,
output-side down, Gaussian proxies) without authorizing a traversal.

Revision 1: the rank-64 whole-population total is a **lower bound only** (not conservative). Top-level `within_target_bpw` is decided solely by the all-routed rank-128 **uncertainty bound** (byte feasibility, not quality proof).

## Safety fences (all false)

- `RAMANUJAN_RESEARCH_AUTHORIZED` = `False`
- `HIDE_KERNEL_TURN` = `False`
- `ODYSSEY_LAUNCH_AUTHORIZED` = `False`
- `full_parent_traversal_started` = `False`
- `full_traversal_authorized` = `False`
- `capable_artifact_claimed` = `False`
- `MOP_touched` = `False`

## Route-population status

- `full_route_population_classified`: **False**
- `route_population_evidence_sufficient_for_rank_assignment`: **False**
- `rank64_population_fit_is_lower_bound_only`: **True**
- `full_traversal_authorized`: **False**
- Routed experts (static census): **19,456**
- Static routed classification: `routed_gate/up/down` (not traffic labels)

## Census (sealed headers)

- Unique tensors: **59585** (expected 59585) OK
- Original weights: **753,329,940,480** OK
- Source payload bytes: **1,506,659,919,872** OK

## Preregistered candidate (not whole-model capability)

| Program | Rank | Floors | Role |
|---|---:|---|---|
| Neutral routed gate/up/down (census) | 64 or 128 by scenario | pilot floors retained | ledger-scenario rank only |
| High-traffic panel (pilot evidence) | 64 | panel min 0.85, median 0.96 | not whole-model traffic |
| Low-traffic diagnostics (pilot) | 128 | per-tensor 0.91 | not population map |
| Shared MLP gate/up/down | 256 | per-tensor 0.91, panel median 0.93 | preregistered |
| Router control | 128 | per-tensor 0.99 | preregistered |
| Attention `q_a_proj` only | 128 | per-tensor 0.91 | preregistered |
| All other classes | native | source payload width | unvalidated |

## All-routed rank-64 lower-bound ledger (NON-AUTHORIZING)

Not conservative. Optimistic whole-population scenario with every routed expert at rank 64.

- Unique bases: **39,219**
- Total bytes: **76,751,084,032**
- Complete BPW: **299808922/367836885** (0.815059)
- `authorizing`: **False**
- Scenario within target (informational): **True**
- Itemization reconciles: **True**

Component totals:

- `float16_basis_matrices`: 20,973,695,744
- `float16_coefficient_matrices`: 25,946,226,688
- `tensor_headers_metadata`: 15,040,256
- `native_source_payload`: 29,816,121,344
- `packaging_alignment`: 0

## All-routed rank-128 uncertainty-bound ledger (AUTHORIZING for BPW only)

Byte-feasibility uncertainty bound for the whole routed population at rank 128. **Not** proof that rank 128 is quality-sufficient for every expert. This total alone decides top-level `within_target_bpw`.

- Unique bases: **39,219**
- Total bytes: **122,653,547,008**
- Complete BPW: **479115418/367836885** (1.302521)
- Target: **49/50**
- `within_target_bpw` (top-level): **False**
- `authorizing`: **True**
- Itemization reconciles: **True**

Component totals:

- `float16_basis_matrices`: 41,374,790,400
- `float16_coefficient_matrices`: 51,447,595,008
- `tensor_headers_metadata`: 15,040,256
- `native_source_payload`: 29,816,121,344
- `packaging_alignment`: 0

## Route-population sensitivity (arithmetic, not traffic)

- Selection: `sorted_(layer, expert)_prefix`
- Expert unit: one expert = gate/up/down triplet with shared hidden basis and separate real-SwiGLU-input basis
- N routed experts: **19,456**
- Max rank-128 experts under target: **6,583** (6583/19456)

| Fraction @128 | N@128 | Total bytes | BPW | Within target |
|---:|---:|---:|---:|:---:|
| 0/1 | 0 | 76,751,084,032 | 0.815059 | True |
| 1/4 | 4,864 | 88,226,699,776 | 0.936925 | True |
| 1/2 | 9,728 | 99,702,315,520 | 1.058790 | False |
| 3/4 | 14,592 | 111,177,931,264 | 1.180656 | False |
| 1/1 | 19,456 | 122,653,547,008 | 1.302521 | False |

## Transfer-sharing scenario (NON-AUTHORIZING)

- Unique bases: **459**
- Total bytes: **56,419,758,592**
- Complete BPW (informational): **220389682/367836885**
- Cross-layer / cross-expert transfer remains **unvalidated**.
- This total must **not** affect any top-level decision.

## Scientific laws

- `basis_mode`: `uncentered`
- `centered_only_fitting_forbidden`: `True`
- `route_conditioned_routed_experts`: `True`
- `empty_route_fails_closed`: `True`
- `real_swiglu_inputs_for_down`: `True`
- `gaussian_proxy_forbidden_for_promotion`: `True`
- `beats_null_diagnostic_only`: `True`
- `absolute_floors_required`: `True`
- `budget_failure_never_reduces_floor`: `True`
- `native_fallback_at_source_payload_width_only`: `True`
- `transfer_sharing_non_authorizing`: `True`
- `rank64_population_fit_is_lower_bound_only`: `True`
- `within_target_bpw_decided_by_rank128_uncertainty_bound_only`: `True`

## Pilot checks (sealed receipt)

- `high_traffic_routed_gate_up_down` rank 64: clears=True detail={panel=promotion_grade_high_traffic_routed, measured_min=0.8594450950622559, measured_median=0.9695300459861755, floor_min=0.85, floor_median=0.96, clears=True}
- `low_traffic_routed_diagnostics` rank 128: clears=True detail={panel=low_traffic_diagnostics, measured_min=0.9154013395309448, measured_median=0.9323745369911194, per_tensor_floor=0.91, clears=True}
- `shared_mlp_gate_up_down` rank 256: clears=True detail={panel=shared_mlp, measured_min=0.9182274341583252, measured_median=0.9361572265625, per_tensor_floor=0.91, panel_median_floor=0.93, clears=True}
- `attention_input_q_a_proj` rank 128: clears=True detail={name=model.layers.38.self_attn.q_a_proj.weight, measured_cosine=0.9142497181892395, per_tensor_floor=0.91, clears=True}
- `router_control` rank 128: clears=True detail={name=model.layers.38.mlp.gate.weight, measured_cosine=0.9945491552352905, per_tensor_floor=0.99, clears=True}

## Unsupported / native islands

- `attention_o_proj`: n=79, bytes=15,904,800,768 — Capsules lack the 16384-wide attention intermediate; Gaussian input forbidden.
- `attention_other`: n=325, bytes=8,592,561,664 — No bounded real-input pilot for this attention projection.
- `global_embed_tokens`: n=1, bytes=1,903,165,440 — embed_tokens not in five-shard pilot; remains native.
- `global_lm_head`: n=1, bytes=1,903,165,440 — lm_head not in five-shard pilot; remains native.
- `norm`: n=342, bytes=2,400,768 — Norms stay native (vector pass-through).
- `router_bias`: n=76, bytes=77,824 — Router bias/e_score stay native.
- `dense_mlp`: n=9, bytes=1,358,954,496 — Dense early layers (0-2) unvalidated under MoE pilot program.
- `other`: n=1, bytes=150,994,944 — No real-input pilot; billed native at source payload width.

## Remaining uncertainties

- Teacher capsules cover only a subset of layers; uncovered layers unvalidated.
- No full-model traffic map: route population is not classified; rank assignment for the whole population is unproven.
- Rank-64 whole-population fit is a lower bound only; sealed low-traffic diagnostics required rank 128 to clear the 0.91 per-tensor floor.
- All-rank-128 uncertainty bound is a byte-feasibility envelope, not quality proof for every routed expert.
- attention.o_proj lacks real intermediate in current capsules.
- global embed_tokens and lm_head have no bounded real-input pilot.
- Dense MLP layers 0-2 unvalidated under the MoE program.
- Cross-layer basis transfer is unvalidated and non-authorizing.
- Feasibility is not whole-model capability; Generation B proved relative admission fails.

## Non-claims

- Does not prove representation quality on uncovered layers, unmeasured route traffic, globals, or attention output projections.
- Does not authorize full parent traversal, capability gate, HIDE kernel turn, Odyssey launch, Math-Frozen, or Ramanujan research.
- Does not claim a capable artifact.
- Does not treat a passing rank-mixture budget as proof of representation capability.
- Does not call the rank-64 whole-population total conservative.

## Next safe action

Route-population measurement is required before any full traversal. A passing rank-mixture budget alone would still not prove representation capability. Keep v2 opt-in; never lower absolute floors or ranks to force the uncertainty bound under target BPW. Do not start a full traversal from this receipt alone.

Receipt sha256: `c4c93bfab2bd24a886699decb740f58468ef8b1a57cab6c5a920bf8bcaf2db1d`
