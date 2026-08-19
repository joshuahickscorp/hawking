# DELEGATION — O005 PER-ORGAN / PER-EXPERT SENSITIVITY (§17)

Patient O005 = `Qwen/Qwen3-30B-A3B` (moe; Qwen3MoeForCausalLM),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. Repo: `/Users/scammermike/Downloads/hawking`.
A baseline already exists on this patient. Measure Doctor capability delta when
each organ (and each expert, if MoE) is zeroed or rounded.

Organs from census (MEASURED): embed, attn, router, expert, norm, lm_head.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ tools/doctor_seal.py
READ workspace/campaign/odyssey/patients/O005/census.json
READ workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py `--sensitivity`. After the fast battery baseline:
1. For each organ in {embed, attn, router, expert, norm, lm_head}, zero (and separately round-to-zero-bpw / 8-bit round) that organ.
2. Re-run the same fast battery + refusal controls.
3. Record capability delta vs the unablated battery (hits, refusals, seal verdict).
For MoE patients, also ablate a hot expert and a random expert and record per-expert delta.
Non-MoE: skip the expert loop; skip the route tap (`--skip-route` / `--route-tokens 0`).

Write receipts/odyssey-i/O005_SENSITIVITY.json and fill representation.per_organ_sensitivity (and
per_expert_sensitivity when MoE) on workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json. Label every delta MEASURED.

Do not delete canonical weights. Keep one load if possible; if reload is required,
re-admit via worker_gate each time.

## ACCEPTANCE
- receipts/odyssey-i/O005_SENSITIVITY.json exists with per_organ_sensitivity entries for each named organ and a
  baseline battery. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json representation.per_organ_sensitivity is non-null and schema-valid.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O005/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, tools/doctor_seal.py, workspace/campaign/odyssey/patients/O005/census.json, workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json
VERIFY receipts/odyssey-i/O005_SENSITIVITY.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O005 --weights /Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 --runtime mlx --route-tokens 0 --out receipts/odyssey-i/O005_SENSITIVITY.json --packet workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json --sensitivity
Do not touch Genesis state or tools/odyssey/.
