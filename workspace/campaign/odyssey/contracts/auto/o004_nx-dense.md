# DELEGATION — O004 NX ACCOUNTING (--nx-dense; gate profile: MLX/Metal)

Patient O004 = `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (dense; Mistral3ForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a`. Repo: `/Users/scammermike/Downloads/hawking`.
Bounded NX/execution attempt (steer S002). ACCOUNTING + minimal-primitive-design,
not a full Rust runtime (§14). Label DERIVED/MEASURED.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O004/census.json
READ workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py --nx-dense. Dense `--nx-dense`: report full-weight-sweep bytes/token as the dense NX floor and note there is no sparsity lever.
Write receipts/odyssey-i/O004_NX_dense.json (schema odyssey.patient.nx.v1) and refresh workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json nx.
Do not delete canonical weights. Never call this a Hawking NX win.

## ACCEPTANCE
- receipts/odyssey-i/O004_NX_dense.json exists with the theoretical-vs-measured (or state-vs-KV / dense-floor)
  accounting and §18 labels. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json still schema-valid.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O004/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O004/census.json, workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json
VERIFY receipts/odyssey-i/O004_NX_dense.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O004 --weights /Users/scammermike/.cache/huggingface/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a --runtime mlx --out receipts/odyssey-i/O004_NX_dense.json --packet workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json --skip-route --nx-dense
Do not touch Genesis state or tools/odyssey/.
