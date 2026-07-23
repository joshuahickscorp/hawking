# M4 - Generation-M run-all

Candidate: gen-M.runall.block0.experts83-86-12-48 | generated 2026-07-19T05:13:10Z
Energy UNAVAILABLE. Wall MEASURED (contaminated_by_concurrent_cpu_load). Mechanics ANALYTICAL. CPU authoritative.

## mlp1 / fused_pq_islands_doctor (control vs M4_separate)
- quality: {"rel_error_vs_dense_mean": 0.804737, "rel_error_vs_dense_max": 0.818959, "cosine_vs_dense_mean": 0.593607}
- quality_gate: fused_matches_unfused admissible=True
- mech: F32=1.594e+07 Llookup=3.111e+06 Klaunch=19 Ttemp=3.076e+06
- no_dense_shadow: True (temp 3.076e+06 < half_dense 3.318e+07)
- wall medians ms: {"M4_fused": 20.43042, "M4_separate": 19.63642}
- causal: {"control": "M4(fused) vs M4(separate-kernel); adds doctor over M3", "fuse_quality_match_rel": 0.0, "quality_matches_unfused": true, "doctor_bits": 9342976, "island_bits": 31896356, "launch_fused": 19.0, "launch_separate": 27.0, "launch_reduction": 8.0, "temp_fused": 3075840.0, "temp_separate": 3144960.0, "temp_reduction_bytes": 69120.0, "wall_fused_over_separate": 1.0404}

## mlp2 / fused_pq_islands_doctor (control vs M4_separate)
- quality: {"rel_error_vs_dense_mean": 0.199821, "rel_error_vs_dense_max": 0.281188, "cosine_vs_dense_mean": 0.978706}
- quality_gate: fused_matches_unfused admissible=True
- mech: F32=1.241e+07 Llookup=1.556e+06 Klaunch=19 Ttemp=1.567e+06
- no_dense_shadow: True (temp 1.567e+06 < half_dense 1.659e+07)
- wall medians ms: {"M4_fused": 9.68, "M4_separate": 9.19583}
- causal: {"control": "M4(fused) vs M4(separate-kernel); adds doctor over M3", "fuse_quality_match_rel": 0.0, "quality_matches_unfused": true, "doctor_bits": 5195776, "island_bits": 16040016, "launch_fused": 19.0, "launch_separate": 27.0, "launch_reduction": 8.0, "temp_fused": 1566720.0, "temp_separate": 1624320.0, "temp_reduction_bytes": 57600.0, "wall_fused_over_separate": 1.0527}

