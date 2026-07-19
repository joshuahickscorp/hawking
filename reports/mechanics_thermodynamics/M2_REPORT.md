# M2 - Generation-M run-all

Candidate: gen-M.runall.block0.experts83-86-12-48 | generated 2026-07-19T05:13:10Z
Energy UNAVAILABLE. Wall MEASURED (contaminated_by_concurrent_cpu_load). Mechanics ANALYTICAL. CPU authoritative.

## mlp1 / independent (control vs self)
- quality: {"rel_error_vs_dense_mean": 0.881873, "rel_error_vs_dense_max": 0.903648, "cosine_vs_dense_mean": 0.471004, "combine_divergence": {"n_inputs": 3, "n_experts_in_combine": 4, "top_k_used": 4, "mean_combine_div": 0.292276, "max_combine_div": 0.294938, "min_combine_div": 0.29028, "routing": "masked_to_loaded_expert_subset", "signal": "proxy_output_synthetic_activations", "capability_parity": false}}
- quality_gate: matched_or_better admissible=True
- mech: F32=1.387e+07 Llookup=2.074e+06 Klaunch=0 Ttemp=1.083e+06
- no_dense_shadow: True (temp 1.083e+06 < half_dense 3.318e+07)
- wall medians ms: {"M2_independent": 13.66946, "M2_shared": 12.35004, "M2_independent_metal": 14.23317, "M2_shared_metal": 10.21933}
- causal: n/a

## mlp1 / shared (control vs M1_independent)
- quality: {"rel_error_vs_dense_mean": 0.883356, "rel_error_vs_dense_max": 0.895855, "cosine_vs_dense_mean": 0.468557, "combine_divergence": {"n_inputs": 3, "n_experts_in_combine": 4, "top_k_used": 4, "mean_combine_div": 0.267447, "max_combine_div": 0.269879, "min_combine_div": 0.266213, "routing": "masked_to_loaded_expert_subset", "signal": "proxy_output_synthetic_activations", "capability_parity": false}}
- quality_gate: matched_or_better admissible=True
- mech: F32=5.023e+06 Llookup=2.074e+06 Klaunch=0 Ttemp=1.083e+06
- no_dense_shadow: True (temp 1.083e+06 < half_dense 3.318e+07)
- wall medians ms: {"M2_independent": 13.66946, "M2_shared": 12.35004, "M2_independent_metal": 14.23317, "M2_shared_metal": 10.21933}
- causal: {"control": "M2_shared vs M1_independent (per-expert-table)", "reuse_ratio_E": 4, "table_builds_independent": 8, "table_builds_shared": 2, "table_build_work_avoided_frac": 0.75, "wall_shared_over_independent": 0.9035, "flops_shared_over_independent": 0.3621, "quality_relerr_shared_minus_independent": 0.001483}

## mlp1 / layer_group_share (control vs M1_independent)
- quality: {"rel_error_vs_dense_mean": 0.881808, "rel_error_vs_dense_max": 0.895898, "cosine_vs_dense_mean": 0.471696, "combine_divergence": {"n_inputs": 3, "n_experts_in_combine": 4, "top_k_used": 4, "mean_combine_div": 0.274973, "max_combine_div": 0.277576, "min_combine_div": 0.272267, "routing": "masked_to_loaded_expert_subset", "signal": "proxy_output_synthetic_activations", "capability_parity": false}}
- quality_gate: matched_or_better admissible=True
- mech: F32=5.023e+06 Llookup=2.074e+06 Klaunch=0 Ttemp=1.083e+06
- no_dense_shadow: True (temp 1.083e+06 < half_dense 3.318e+07)
- wall medians ms: {"M2_independent": 13.66946, "M2_shared": 12.35004, "M2_independent_metal": 14.23317, "M2_shared_metal": 10.21933}
- causal: n/a

## mlp2 / independent (control vs self)
- quality: {"rel_error_vs_dense_mean": 0.385499, "rel_error_vs_dense_max": 0.551826, "cosine_vs_dense_mean": 0.944098, "combine_divergence": {"n_inputs": 3, "n_experts_in_combine": 4, "top_k_used": 4, "mean_combine_div": 0.292276, "max_combine_div": 0.294938, "min_combine_div": 0.29028, "routing": "masked_to_loaded_expert_subset", "signal": "proxy_output_synthetic_activations", "capability_parity": false}}
- quality_gate: matched_or_better admissible=True
- mech: F32=1.283e+07 Llookup=1.037e+06 Klaunch=0 Ttemp=5.645e+05
- no_dense_shadow: True (temp 5.645e+05 < half_dense 1.659e+07)
- wall medians ms: {"M2_independent": 6.61167, "M2_shared": 6.03913, "M2_independent_metal": 13.45296, "M2_shared_metal": 9.44337}
- causal: n/a

## mlp2 / shared (control vs M1_independent)
- quality: {"rel_error_vs_dense_mean": 0.362042, "rel_error_vs_dense_max": 0.511537, "cosine_vs_dense_mean": 0.946071, "combine_divergence": {"n_inputs": 3, "n_experts_in_combine": 4, "top_k_used": 4, "mean_combine_div": 0.267447, "max_combine_div": 0.269879, "min_combine_div": 0.266213, "routing": "masked_to_loaded_expert_subset", "signal": "proxy_output_synthetic_activations", "capability_parity": false}}
- quality_gate: matched_or_better admissible=True
- mech: F32=3.986e+06 Llookup=1.037e+06 Klaunch=0 Ttemp=5.645e+05
- no_dense_shadow: True (temp 5.645e+05 < half_dense 1.659e+07)
- wall medians ms: {"M2_independent": 6.61167, "M2_shared": 6.03913, "M2_independent_metal": 13.45296, "M2_shared_metal": 9.44337}
- causal: {"control": "M2_shared vs M1_independent (per-expert-table)", "reuse_ratio_E": 4, "table_builds_independent": 8, "table_builds_shared": 2, "table_build_work_avoided_frac": 0.75, "wall_shared_over_independent": 0.9134, "flops_shared_over_independent": 0.3106, "quality_relerr_shared_minus_independent": -0.023457}

## mlp2 / layer_group_share (control vs M1_independent)
- quality: {"rel_error_vs_dense_mean": 0.359519, "rel_error_vs_dense_max": 0.491709, "cosine_vs_dense_mean": 0.945979, "combine_divergence": {"n_inputs": 3, "n_experts_in_combine": 4, "top_k_used": 4, "mean_combine_div": 0.274973, "max_combine_div": 0.277576, "min_combine_div": 0.272267, "routing": "masked_to_loaded_expert_subset", "signal": "proxy_output_synthetic_activations", "capability_parity": false}}
- quality_gate: matched_or_better admissible=True
- mech: F32=3.986e+06 Llookup=1.037e+06 Klaunch=0 Ttemp=5.645e+05
- no_dense_shadow: True (temp 5.645e+05 < half_dense 1.659e+07)
- wall medians ms: {"M2_independent": 6.61167, "M2_shared": 6.03913, "M2_independent_metal": 13.45296, "M2_shared_metal": 9.44337}
- causal: n/a

