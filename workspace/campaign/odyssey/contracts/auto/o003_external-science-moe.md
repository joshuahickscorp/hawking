# DELEGATION — O003 EXTERNAL SCIENCE (MoE; gate profile: MLX/Metal)

Patient O003 = `moonshotai/Kimi-VL-A3B-Instruct` (moe; KimiVLForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-Instruct/snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375`. Repo: `/Users/scammermike/Downloads/hawking`.
This is the O005-style runner: route map + baseline TPS + fast-Doctor.

Native Hawking `load_engine` is not the path. Use `tools/odyssey_patient_runner.py`
(mlx_lm EXTERNAL SPECIMEN). SPECIMEN labels everywhere; this is NOT BASE_TRUE_TPS (§14).

Census (MEASURED): layers=27 experts=64 topk=6
total_params=16407657776 organs: embed=0.68GB, attn=0.96GB, router=0.01GB, expert=28.79GB, shared_expert=0.9GB, mlp_dense=0.67GB, norm=0.0GB, lm_head=0.67GB, other=0.13GB.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O003/census.json
READ workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json
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
- receipts/odyssey-i/O003_EXTERNAL.json with quant, tps_specimen, ttft, route{entropy_avg,entropy_max,cold_experts,top16_mass_pct,most_popular_share,transition_stability}, doctor{battery,refusals,seal_ref}.
- Refresh workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json routing + execution + doctor from the receipt (schema-valid).

## ACCEPTANCE
- receipts/odyssey-i/O003_EXTERNAL.json exists with route.entropy_avg>0 and doctor.battery. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json still schema-valid after the refresh.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O003/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O003/census.json, workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json
VERIFY receipts/odyssey-i/O003_EXTERNAL.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O003 --weights /Users/scammermike/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-Instruct/snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375 --runtime mlx --route-tokens 512 --out receipts/odyssey-i/O003_EXTERNAL.json --packet workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json
Do not touch Genesis state or tools/odyssey/.
