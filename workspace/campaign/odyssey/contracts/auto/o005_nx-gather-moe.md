# DELEGATION — O005 NX ACCOUNTING (--nx-gather; gate profile: MLX/Metal)

Patient O005 = `Qwen/Qwen3-30B-A3B` (moe; Qwen3MoeForCausalLM),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. Repo: `/Users/scammermike/Downloads/hawking`.
Bounded NX/execution attempt (steer S002). ACCOUNTING + minimal-primitive-design,
not a full Rust runtime (§14). Label DERIVED/MEASURED.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O005/census.json
READ workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
Reuse tools/odyssey_patient_runner.py --nx-gather. MoE `--nx-gather`: from the router over N tokens, compute THEORETICAL selected-expert bytes/token = topk/n_experts × expert_body_bytes; contrast with full-expert-body bytes and the dense-MLP-equivalent; report the ratio (the NX opportunity). Note whether mlx actually gathers or densely computes.
Write receipts/odyssey-i/O005_NX_gather.json (schema odyssey.patient.nx.v1) and refresh workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json nx.
Do not delete canonical weights. Never call this a Hawking NX win.

## ACCEPTANCE
- receipts/odyssey-i/O005_NX_gather.json exists with the theoretical-vs-measured (or state-vs-KV / dense-floor)
  accounting and §18 labels. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json still schema-valid.

## SCOPE
WRITE tools/odyssey_patient_runner.py
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O005/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O005/census.json, workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json
VERIFY tools/odyssey_patient_runner.py by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O005 --weights /Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 --runtime mlx --out receipts/odyssey-i/O005_NX_gather.json --packet workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json --nx-gather
Do not touch Genesis state or tools/odyssey/.
