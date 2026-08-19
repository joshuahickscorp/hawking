# DELEGATION — O003 PER-ORGAN / PER-EXPERT SENSITIVITY (§17)

Patient O003 = `moonshotai/Kimi-VL-A3B-Instruct` (moe; KimiVLForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-Instruct/snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375`. Repo: `/Users/scammermike/Downloads/hawking`.
A baseline already exists on this patient. Measure Doctor capability delta when
each organ (and each expert, if MoE) is zeroed or rounded.

Organs from census (MEASURED): embed, attn, router, expert, shared_expert, mlp_dense, norm, lm_head, other.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ tools/doctor_seal.py
READ workspace/campaign/odyssey/patients/O003/census.json
READ workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
Reuse tools/odyssey_patient_runner.py. If it has no organ-ablation path, add a
`--sensitivity` mode that, after the fast battery baseline:
1. For each organ in {embed, attn, router, expert, shared_expert, mlp_dense, norm, lm_head, other}, zero (and separately round-to-zero-bpw / 8-bit round) that organ.
2. Re-run the same fast battery + refusal controls.
3. Record capability delta vs the unablated battery (hits, refusals, seal verdict).
For MoE patients, also ablate a hot expert and a random expert and record per-expert delta.
Non-MoE: skip the expert loop; skip the route tap (`--skip-route` / `--route-tokens 0`).

Write receipts/odyssey-i/O003_SENSITIVITY.json and fill representation.per_organ_sensitivity (and
per_expert_sensitivity when MoE) on workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json. Label every delta MEASURED.

Do not delete canonical weights. Keep one load if possible; if reload is required,
re-admit via worker_gate each time.

## ACCEPTANCE
- receipts/odyssey-i/O003_SENSITIVITY.json exists with per_organ_sensitivity entries for each named organ and a
  baseline battery. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json representation.per_organ_sensitivity is non-null and schema-valid.

## SCOPE
WRITE tools/odyssey_patient_runner.py
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O003/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, tools/doctor_seal.py, workspace/campaign/odyssey/patients/O003/census.json, workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json
VERIFY tools/odyssey_patient_runner.py by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O003 --weights /Users/scammermike/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-Instruct/snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375 --runtime mlx --route-tokens 0 --out receipts/odyssey-i/O003_SENSITIVITY.json --packet workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json --sensitivity
Do not touch Genesis state or tools/odyssey/.
