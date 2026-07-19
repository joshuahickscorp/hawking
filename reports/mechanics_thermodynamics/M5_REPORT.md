# M5 - Generation-M run-all

Candidate: gen-M.runall.block0.experts83-86-12-48 | generated 2026-07-19T05:13:10Z
Energy UNAVAILABLE. Wall MEASURED (contaminated_by_concurrent_cpu_load). Mechanics ANALYTICAL. CPU authoritative.

## mlp1 / conditional_doctor (control vs M4_always_on)
- quality: {"rel_error_vs_dense_mean": 0.804737, "rel_error_vs_dense_max": 0.818959, "cosine_vs_dense_mean": 0.593607}
- quality_gate: conditional_safe admissible=True
- mech: F32=1.594e+07 Llookup=3.111e+06 Klaunch=0 Ttemp=3.076e+06
- no_dense_shadow: True (temp 3.076e+06 < half_dense 3.318e+07)
- wall medians ms: {"M5_conditional": 52.91933, "M4_always_on": 19.00221}
- causal: {"control": "M5(conditional) vs M4(always-on doctor)", "condition": "residual_syndrome (relative doctor contribution >= threshold)", "threshold": 0.02, "skip_frac": 0.0, "false_negative_rate": 0.0, "fn_gate": 0.05, "needed_corrections": 12, "skipped_total": 0, "false_negatives": 0, "hard_gate_fires_reject": false, "wall_conditional_over_alwayson": 2.7849}

## mlp2 / conditional_doctor (control vs M4_always_on)
- quality: {"rel_error_vs_dense_mean": 0.199821, "rel_error_vs_dense_max": 0.281188, "cosine_vs_dense_mean": 0.978706}
- quality_gate: conditional_safe admissible=True
- mech: F32=1.241e+07 Llookup=1.556e+06 Klaunch=0 Ttemp=1.567e+06
- no_dense_shadow: True (temp 1.567e+06 < half_dense 1.659e+07)
- wall medians ms: {"M5_conditional": 25.80633, "M4_always_on": 8.54479}
- causal: {"control": "M5(conditional) vs M4(always-on doctor)", "condition": "residual_syndrome (relative doctor contribution >= threshold)", "threshold": 0.02, "skip_frac": 0.0, "false_negative_rate": 0.0, "fn_gate": 0.05, "needed_corrections": 12, "skipped_total": 0, "false_negatives": 0, "hard_gate_fires_reject": false, "wall_conditional_over_alwayson": 3.0201}

