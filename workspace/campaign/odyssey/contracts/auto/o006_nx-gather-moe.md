# DELEGATION — O006 NX ACCOUNTING (--nx-gather; gate profile: MLX/Metal)

Patient O006 = `Qwen/Qwen3-VL-30B-A3B-Instruct` (moe; Qwen3VLMoeForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`. Repo: `/Users/scammermike/Downloads/hawking`.
Bounded NX/execution attempt (steer S002). ACCOUNTING + minimal-primitive-design,
not a full Rust runtime (§14). Label DERIVED/MEASURED.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O006/census.json
READ workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py --nx-gather. MoE `--nx-gather`: from the router over N tokens, compute THEORETICAL selected-expert bytes/token = topk/n_experts × expert_body_bytes; contrast with full-expert-body bytes and the dense-MLP-equivalent; report the ratio (the NX opportunity). Note whether mlx actually gathers or densely computes.
Write receipts/odyssey-i/O006_NX_gather.json (schema odyssey.patient.nx.v1) and refresh workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json nx.
Do not delete canonical weights. Never call this a Hawking NX win.

## ACCEPTANCE
- receipts/odyssey-i/O006_NX_gather.json exists with the theoretical-vs-measured (or state-vs-KV / dense-floor)
  accounting and §18 labels. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json still schema-valid.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O006/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O006/census.json, workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json
VERIFY receipts/odyssey-i/O006_NX_gather.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O006 --weights /Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c --runtime mlx --out receipts/odyssey-i/O006_NX_gather.json --packet workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json --nx-gather
Do not touch Genesis state or tools/odyssey/.
