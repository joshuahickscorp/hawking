# M3 - Generation-M run-all

Candidate: gen-M.runall.block0.experts83-86-12-48 | generated 2026-07-19T05:13:10Z
Energy UNAVAILABLE. Wall MEASURED (contaminated_by_concurrent_cpu_load). Mechanics ANALYTICAL. CPU authoritative.

## mlp1 / islands_magnitude (control vs M2_shared)
- quality: {"rel_error_vs_dense_mean": 0.855364, "rel_error_vs_dense_max": 0.8682, "cosine_vs_dense_mean": 0.517946}
- quality_gate: islands_reduce_error admissible=True
- mech: F32=9.009e+06 Llookup=2.074e+06 Klaunch=0 Ttemp=3.076e+06
- no_dense_shadow: True (temp 3.076e+06 < half_dense 3.318e+07)
- wall medians ms: {"M3_islands_off": 12.42829, "M3_magnitude": 13.58483, "M3_activation_aware": 13.39612, "M3_sensitivity": 13.56967, "M3_residual_energy": 13.44212}
- causal: {"control": "M3(+islands) vs M2(islands-off, shared)", "strategy": "magnitude", "island_bits": 31896356, "n_islands_total": 692, "relerr_off": 0.883356, "relerr_on": 0.855364, "relerr_improvement": 0.027992, "quality_improved": true, "wall_on_over_off": 1.0931}

## mlp1 / islands_activation_aware (control vs M2_shared)
- quality: {"rel_error_vs_dense_mean": 0.855322, "rel_error_vs_dense_max": 0.868566, "cosine_vs_dense_mean": 0.518007}
- quality_gate: islands_reduce_error admissible=True
- mech: F32=9.009e+06 Llookup=2.074e+06 Klaunch=0 Ttemp=3.076e+06
- no_dense_shadow: True (temp 3.076e+06 < half_dense 3.318e+07)
- wall medians ms: {"M3_islands_off": 12.42829, "M3_magnitude": 13.58483, "M3_activation_aware": 13.39612, "M3_sensitivity": 13.56967, "M3_residual_energy": 13.44212}
- causal: {"control": "M3(+islands) vs M2(islands-off, shared)", "strategy": "activation_aware", "island_bits": 31896356, "n_islands_total": 692, "relerr_off": 0.883356, "relerr_on": 0.855322, "relerr_improvement": 0.028034, "quality_improved": true, "wall_on_over_off": 1.0779}

## mlp1 / islands_sensitivity (control vs M2_shared)
- quality: {"rel_error_vs_dense_mean": 0.855559, "rel_error_vs_dense_max": 0.868667, "cosine_vs_dense_mean": 0.517634}
- quality_gate: islands_reduce_error admissible=True
- mech: F32=9.009e+06 Llookup=2.074e+06 Klaunch=0 Ttemp=3.076e+06
- no_dense_shadow: True (temp 3.076e+06 < half_dense 3.318e+07)
- wall medians ms: {"M3_islands_off": 12.42829, "M3_magnitude": 13.58483, "M3_activation_aware": 13.39612, "M3_sensitivity": 13.56967, "M3_residual_energy": 13.44212}
- causal: {"control": "M3(+islands) vs M2(islands-off, shared)", "strategy": "sensitivity", "island_bits": 31896356, "n_islands_total": 692, "relerr_off": 0.883356, "relerr_on": 0.855559, "relerr_improvement": 0.027797, "quality_improved": true, "wall_on_over_off": 1.0918}

## mlp1 / islands_residual_energy (control vs M2_shared)
- quality: {"rel_error_vs_dense_mean": 0.855587, "rel_error_vs_dense_max": 0.869031, "cosine_vs_dense_mean": 0.517601}
- quality_gate: islands_reduce_error admissible=True
- mech: F32=9.009e+06 Llookup=2.074e+06 Klaunch=0 Ttemp=3.076e+06
- no_dense_shadow: True (temp 3.076e+06 < half_dense 3.318e+07)
- wall medians ms: {"M3_islands_off": 12.42829, "M3_magnitude": 13.58483, "M3_activation_aware": 13.39612, "M3_sensitivity": 13.56967, "M3_residual_energy": 13.44212}
- causal: {"control": "M3(+islands) vs M2(islands-off, shared)", "strategy": "residual_energy", "island_bits": 31896356, "n_islands_total": 692, "relerr_off": 0.883356, "relerr_on": 0.855587, "relerr_improvement": 0.027769, "quality_improved": true, "wall_on_over_off": 1.0816}

## mlp2 / islands_magnitude (control vs M2_shared)
- quality: {"rel_error_vs_dense_mean": 0.218502, "rel_error_vs_dense_max": 0.305529, "cosine_vs_dense_mean": 0.974504}
- quality_gate: islands_reduce_error admissible=True
- mech: F32=5.990e+06 Llookup=1.037e+06 Klaunch=0 Ttemp=1.567e+06
- no_dense_shadow: True (temp 1.567e+06 < half_dense 1.659e+07)
- wall medians ms: {"M3_islands_off": 5.58842, "M3_magnitude": 6.114, "M3_activation_aware": 6.38612, "M3_sensitivity": 6.27079, "M3_residual_energy": 6.44375}
- causal: {"control": "M3(+islands) vs M2(islands-off, shared)", "strategy": "magnitude", "island_bits": 16040016, "n_islands_total": 348, "relerr_off": 0.362042, "relerr_on": 0.218502, "relerr_improvement": 0.14354, "quality_improved": true, "wall_on_over_off": 1.094}

## mlp2 / islands_activation_aware (control vs M2_shared)
- quality: {"rel_error_vs_dense_mean": 0.218503, "rel_error_vs_dense_max": 0.305491, "cosine_vs_dense_mean": 0.974505}
- quality_gate: islands_reduce_error admissible=True
- mech: F32=5.990e+06 Llookup=1.037e+06 Klaunch=0 Ttemp=1.567e+06
- no_dense_shadow: True (temp 1.567e+06 < half_dense 1.659e+07)
- wall medians ms: {"M3_islands_off": 5.58842, "M3_magnitude": 6.114, "M3_activation_aware": 6.38612, "M3_sensitivity": 6.27079, "M3_residual_energy": 6.44375}
- causal: {"control": "M3(+islands) vs M2(islands-off, shared)", "strategy": "activation_aware", "island_bits": 16040016, "n_islands_total": 348, "relerr_off": 0.362042, "relerr_on": 0.218503, "relerr_improvement": 0.143539, "quality_improved": true, "wall_on_over_off": 1.1427}

## mlp2 / islands_sensitivity (control vs M2_shared)
- quality: {"rel_error_vs_dense_mean": 0.221166, "rel_error_vs_dense_max": 0.305295, "cosine_vs_dense_mean": 0.974018}
- quality_gate: islands_reduce_error admissible=True
- mech: F32=5.990e+06 Llookup=1.037e+06 Klaunch=0 Ttemp=1.567e+06
- no_dense_shadow: True (temp 1.567e+06 < half_dense 1.659e+07)
- wall medians ms: {"M3_islands_off": 5.58842, "M3_magnitude": 6.114, "M3_activation_aware": 6.38612, "M3_sensitivity": 6.27079, "M3_residual_energy": 6.44375}
- causal: {"control": "M3(+islands) vs M2(islands-off, shared)", "strategy": "sensitivity", "island_bits": 16040016, "n_islands_total": 348, "relerr_off": 0.362042, "relerr_on": 0.221166, "relerr_improvement": 0.140876, "quality_improved": true, "wall_on_over_off": 1.1221}

## mlp2 / islands_residual_energy (control vs M2_shared)
- quality: {"rel_error_vs_dense_mean": 0.22134, "rel_error_vs_dense_max": 0.305559, "cosine_vs_dense_mean": 0.973976}
- quality_gate: islands_reduce_error admissible=True
- mech: F32=5.990e+06 Llookup=1.037e+06 Klaunch=0 Ttemp=1.567e+06
- no_dense_shadow: True (temp 1.567e+06 < half_dense 1.659e+07)
- wall medians ms: {"M3_islands_off": 5.58842, "M3_magnitude": 6.114, "M3_activation_aware": 6.38612, "M3_sensitivity": 6.27079, "M3_residual_energy": 6.44375}
- causal: {"control": "M3(+islands) vs M2(islands-off, shared)", "strategy": "residual_energy", "island_bits": 16040016, "n_islands_total": 348, "relerr_off": 0.362042, "relerr_on": 0.22134, "relerr_improvement": 0.140702, "quality_improved": true, "wall_on_over_off": 1.1531}

