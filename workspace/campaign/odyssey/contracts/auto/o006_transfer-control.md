# DELEGATION — O006 TRANSFER CONTROL vs O005 (§41)

Patient O006 = `Qwen/Qwen3-VL-30B-A3B-Instruct` (moe; Qwen3VLMoeForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`. Reference O005 = `Qwen/Qwen3-30B-A3B`
(moe; Qwen3MoeForCausalLM). Repo: `/Users/scammermike/Downloads/hawking`.

Run the O005-style runner on the sibling, then diff route/representation against
the named reference and write a transfer-matrix delta.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O006/census.json
READ workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json
READ workspace/campaign/odyssey/patients/O005/census.json
READ workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json
READ receipts/odyssey-i/O005_EXTERNAL.json
READ workspace/campaign/odyssey/TRANSFER_MATRIX.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py on O006 (route map + baseline + fast-Doctor).
If O006 is multimodal, tap the language-MoE router; skip the vision tower.
Then diff against O005:
- route entropy / cold-expert count / top16 mass / transition stability
- organs_bytes_GB and stored_bpw
- doctor battery delta
Write receipts/odyssey-i/O006_TRANSFER.json with `reference=O005`, `delta`, and a `transfer_cells` block
mapping each GRAVITY_RULEBASE rule id to one of TRANSFERRED_UNCHANGED / RETUNED /
ARCHITECTURE_SPECIFIC / PATIENT_SPECIFIC / FAILED / HARMFUL / NOT_TESTED.
Merge those cells into workspace/campaign/odyssey/TRANSFER_MATRIX.json for O006
(do not blank other patients). Refresh workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json transfer + routing + execution.

## ACCEPTANCE
- receipts/odyssey-i/O006_EXTERNAL.json exists (sibling external science) AND receipts/odyssey-i/O006_TRANSFER.json exists with
  reference, delta, transfer_cells. Must pass, exit 0.
- workspace/campaign/odyssey/TRANSFER_MATRIX.json has non-NOT_TESTED cells for O006.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O006/
WRITE workspace/campaign/odyssey/TRANSFER_MATRIX.json
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O006/census.json, workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json, workspace/campaign/odyssey/patients/O005/census.json, workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json, workspace/campaign/odyssey/TRANSFER_MATRIX.json
VERIFY receipts/odyssey-i/O006_TRANSFER.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O006 --weights /Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c --runtime mlx --route-tokens 512 --out receipts/odyssey-i/O006_EXTERNAL.json --packet workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json
Do not touch Genesis state or tools/odyssey/.
