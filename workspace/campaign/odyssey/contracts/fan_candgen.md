# DELEGATION — CANDIDATE SEARCH AS DATA (NEW candidate_families.json + tools/odyssey_candgen.py)
Encode the aggressive representation search space as DATA + a deterministic generator/pruner, so
the orchestrator produces & prunes hundreds of candidates with NO per-candidate LLM (steer S004 §61/§16/§17).
Repo /Users/scammermike/Downloads/hawking. Branch odyssey-i. Do NOT edit ctl.py/runner.

## BUILD
1. `workspace/campaign/odyssey/candidate_families.json` — machine-readable families with dimensions +
   Doctor-risk priors + native-availability, keyed by patient class (moe/dense/hybrid). Include:
   per_expert_mixed_quant {bit_classes:[1,2,3,4], group_sizes:[32,64,128], router_precision:[4,8], correction_budget:[0,0.02,0.05], metadata_codec:[raw,shared,entropy]}, sensitivity_driven_alloc, base_plus_correction, matryoshka_tiers, alt_group, scale_codec_joint. Each family: mechanism, conventionality(CONVENTIONAL/AGGRESSIVE/STRUCTURAL), cheapest_falsifier, expected_win, doctor_risk, applicability predicate.
2. `tools/odyssey_candgen.py`: `generate(patient_class, census, sensitivity, policy) -> [candidate_spec...]` — expands the grid, applies constraints (native availability, disk/mem, source-pass budget), and RANKS by expected info-gain/cost (order families by prior); `prune(candidates, results)` drops dominated ones (Pareto on complete_bpw vs Doctor delta vs active-bytes). Runner-spec strings must match the runner `--gravity` spec grammar (q<bits>-g<group>-experts, mixed-q2q3-experts, +correction, tiers). Deterministic; no model calls.

## Self-check (exit 0): generate() yields a non-empty ranked list for an moe patient; prune() removes a strictly-dominated candidate; specs are valid strings; ranking is stable.

## SCOPE
WRITE tools/odyssey_candgen.py
WRITE workspace/campaign/odyssey/candidate_families.json
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json, workspace/campaign/odyssey/patients/
VERIFY tools/odyssey_candgen.py by running `python3 tools/odyssey_candgen.py --self-check` — exit 0.
