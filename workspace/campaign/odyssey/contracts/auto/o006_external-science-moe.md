# DELEGATION — O006 EXTERNAL SCIENCE (MoE; gate profile: MLX/Metal)

Patient O006 = `Qwen/Qwen3-VL-30B-A3B-Instruct` (moe; Qwen3VLMoeForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`. Repo: `/Users/scammermike/Downloads/hawking`.
This is the O005-style runner: route map + baseline TPS + fast-Doctor.

Native Hawking `load_engine` is not the path. Use `tools/odyssey_patient_runner.py`
(mlx_lm EXTERNAL SPECIMEN). SPECIMEN labels everywhere; this is NOT BASE_TRUE_TPS (§14).

Census (MEASURED): layers=48 experts=128 topk=8
total_params=31070754032 organs: embed=0.63GB, attn=2.1GB, router=0.03GB, expert=57.98GB, mlp_dense=0.54GB, norm=0.0GB, lm_head=0.62GB, other=0.25GB.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O006/census.json
READ workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json
READ workspace/campaign/odyssey/contracts/o005_external_science.md

Call worker_gate.observe()/gate() before load (the runner already does this). Abort on REFUSE.
If REFUSE, convert to 4-bit mlx and LABEL `quant=4bit-mlx`; prefer bf16 if admitted.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py. Do not start from scratch.
If this patient is multimodal, tap the language-MoE router (skip the vision tower).
If the runner assumes Qwen3-MoE config assertions, keep them as recorded pass/fail —
do not fail the receipt solely because a Qwen-specific assertion is N/A; label N/A.

Outputs:
- receipts/odyssey-i/O006_EXTERNAL.json with quant, tps_specimen, ttft, route{entropy_avg,entropy_max,cold_experts,top16_mass_pct,most_popular_share,transition_stability}, doctor{battery,refusals,seal_ref}.
- Refresh workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json routing + execution + doctor from the receipt (schema-valid).

## ACCEPTANCE
- receipts/odyssey-i/O006_EXTERNAL.json exists with route.entropy_avg>0 and doctor.battery. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json still schema-valid after the refresh.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O006/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O006/census.json, workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json
VERIFY receipts/odyssey-i/O006_EXTERNAL.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O006 --weights /Users/scammermike/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c --runtime mlx --route-tokens 512 --out receipts/odyssey-i/O006_EXTERNAL.json --packet workspace/campaign/odyssey/patients/O006/ODYSSEY_PATIENT_O006.json
Do not touch Genesis state or tools/odyssey/.
