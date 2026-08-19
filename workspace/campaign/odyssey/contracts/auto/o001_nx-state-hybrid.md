# DELEGATION — O001 NX ACCOUNTING (--nx-state; gate profile: MLX/Metal)

Patient O001 = `tiiuae/Falcon-H1-7B-Instruct` (hybrid; FalconH1ForCausalLM),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/41e72f27effbab80cd45b6e884688452253a3686`. Repo: `/Users/scammermike/Downloads/hawking`.
Bounded NX/execution attempt (steer S002). ACCOUNTING + minimal-primitive-design,
not a full Rust runtime (§14). Label DERIVED/MEASURED.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O001/census.json
READ workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py --nx-state. Hybrid `--nx-state`: reuse SSM-vs-KV accounting already on the packet/external receipt; frame fixed-state residency as the NX lever. Emit state_bytes vs kv_bytes across ctx.
Write receipts/odyssey-i/O001_NX_state.json (schema odyssey.patient.nx.v1) and refresh workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json nx.
Do not delete canonical weights. Never call this a Hawking NX win.

## ACCEPTANCE
- receipts/odyssey-i/O001_NX_state.json exists with the theoretical-vs-measured (or state-vs-KV / dense-floor)
  accounting and §18 labels. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json still schema-valid.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O001/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O001/census.json, workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json
VERIFY receipts/odyssey-i/O001_NX_state.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O001 --weights /Users/scammermike/.cache/huggingface/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/41e72f27effbab80cd45b6e884688452253a3686 --runtime mlx --out receipts/odyssey-i/O001_NX_state.json --packet workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json --skip-route --nx-state
Do not touch Genesis state or tools/odyssey/.
