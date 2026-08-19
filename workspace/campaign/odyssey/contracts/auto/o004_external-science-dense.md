# DELEGATION — O004 EXTERNAL SCIENCE (dense/hybrid; gate profile: MLX/Metal)

Patient O004 = `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (dense; Mistral3ForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a`. Repo: `/Users/scammermike/Downloads/hawking`.
Baseline TPS + fast-Doctor. NO route map.

This is a DENSE/HYBRID path. There is no MoE router. The runner already skips
the route tap via `--route-tokens 0` and `--skip-route` (no-ops RouteRecorder
when no layer has gate+switch_mlp, writes `route_skipped=true` instead of
failing). Do not reimplement that flag.

SPECIMEN labels everywhere; mlx TPS is NOT BASE_TRUE_TPS (§14, §60 foreign-runtime).

Census (MEASURED): layers=40 total_params=24011361280
organs: embed=1.34GB, attn=4.4GB, mlp_dense=40.87GB, norm=0.0GB, lm_head=1.34GB, other=0.07GB.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O004/census.json
READ workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE; 4-bit fallback is allowed
and must be labelled.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py `--skip-route`. 
Keep the model loaded once (one load, memory-safe). Do not delete the canonical weights.

Outputs:
- receipts/odyssey-i/O004_EXTERNAL.json with quant, tps_specimen, ttft, doctor{battery,refusals,seal_ref}, route_skipped=true.
- Refresh workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json execution + doctor.

## ACCEPTANCE
- receipts/odyssey-i/O004_EXTERNAL.json exists with tps_specimen, ttft, doctor.battery, route_skipped true. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json still schema-valid.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O004/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O004/census.json, workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json
VERIFY receipts/odyssey-i/O004_EXTERNAL.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O004 --weights /Users/scammermike/.cache/huggingface/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a --runtime mlx --route-tokens 0 --out receipts/odyssey-i/O004_EXTERNAL.json --packet workspace/campaign/odyssey/patients/O004/ODYSSEY_PATIENT_O004.json --skip-route
Do not touch Genesis state or tools/odyssey/.
