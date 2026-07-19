# M6 - Generation-M run-all

Candidate: gen-M.runall.block0.experts83-86-12-48 | generated 2026-07-19T05:13:10Z
Energy UNAVAILABLE. Wall MEASURED (contaminated_by_concurrent_cpu_load). Mechanics ANALYTICAL. CPU authoritative.

## mlp1 / residual_additive_lookup (control vs M4_fused)
- quality: {"rel_error_vs_dense_mean": 0.654399, "rel_error_vs_dense_max": 0.669088, "cosine_vs_dense_mean": 0.756163}
- quality_gate: matched_or_better admissible=True
- mech: F32=1.758e+07 Llookup=7.258e+06 Klaunch=0 Ttemp=1.083e+06
- no_dense_shadow: True (temp 1.083e+06 < half_dense 3.318e+07)
- wall medians ms: {"M6_residual_additive": 41.73163, "M4_fused": 20.05133}
- causal: {"control": "M6(richer additive codebooks) vs M4(base+islands+doctor) at EQUAL bits", "stages_m6": 7, "bits_m6": 59896320, "bits_m4": 58352932, "bits_ratio_m6_over_m4": 1.0264, "relerr_m6": 0.654399, "relerr_m4": 0.804737, "relerr_m6_minus_m4": -0.150338, "m6_wins_quality_at_equal_bits": true, "wall_m6_over_m4": 2.0812}

## mlp2 / residual_additive_lookup (control vs M4_fused)
- quality: {"rel_error_vs_dense_mean": 0.228441, "rel_error_vs_dense_max": 0.313693, "cosine_vs_dense_mean": 0.980044}
- quality_gate: worse_quality_rejected admissible=False
- mech: F32=1.395e+07 Llookup=3.629e+06 Klaunch=0 Ttemp=5.645e+05
- no_dense_shadow: True (temp 5.645e+05 < half_dense 1.659e+07)
- wall medians ms: {"M6_residual_additive": 25.18746, "M4_fused": 10.45288}
- causal: {"control": "M6(richer additive codebooks) vs M4(base+islands+doctor) at EQUAL bits", "stages_m6": 7, "bits_m6": 30865920, "bits_m4": 30054992, "bits_ratio_m6_over_m4": 1.027, "relerr_m6": 0.228441, "relerr_m4": 0.199821, "relerr_m6_minus_m4": 0.02862, "m6_wins_quality_at_equal_bits": false, "wall_m6_over_m4": 2.4096}

