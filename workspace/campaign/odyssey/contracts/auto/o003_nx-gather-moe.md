# DELEGATION — O003 NX ACCOUNTING (--nx-gather; gate profile: MLX/Metal)

Patient O003 = `moonshotai/Kimi-VL-A3B-Instruct` (moe; KimiVLForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-Instruct/snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375`. Repo: `/Users/scammermike/Downloads/hawking`.
Bounded NX/execution attempt (steer S002). ACCOUNTING + minimal-primitive-design,
not a full Rust runtime (§14). Label DERIVED/MEASURED.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O003/census.json
READ workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py --nx-gather. MoE `--nx-gather`: from the router over N tokens, compute THEORETICAL selected-expert bytes/token = topk/n_experts × expert_body_bytes; contrast with full-expert-body bytes and the dense-MLP-equivalent; report the ratio (the NX opportunity). Note whether mlx actually gathers or densely computes.
Write receipts/odyssey-i/O003_NX_gather.json (schema odyssey.patient.nx.v1) and refresh workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json nx.
Do not delete canonical weights. Never call this a Hawking NX win.

## ACCEPTANCE
- receipts/odyssey-i/O003_NX_gather.json exists with the theoretical-vs-measured (or state-vs-KV / dense-floor)
  accounting and §18 labels. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json still schema-valid.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O003/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O003/census.json, workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json
VERIFY receipts/odyssey-i/O003_NX_gather.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O003 --weights /Users/scammermike/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-Instruct/snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375 --runtime mlx --out receipts/odyssey-i/O003_NX_gather.json --packet workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json --nx-gather
Do not touch Genesis state or tools/odyssey/.
