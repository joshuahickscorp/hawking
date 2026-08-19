# DELEGATION — O001 EXTERNAL SCIENCE (dense/hybrid; gate profile: MLX/Metal)

Patient O001 = `tiiuae/Falcon-H1-7B-Instruct` (hybrid; FalconH1ForCausalLM),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/41e72f27effbab80cd45b6e884688452253a3686`. Repo: `/Users/scammermike/Downloads/hawking`.
Baseline TPS + fast-Doctor + SSM-state-vs-KV. NO route map.

This is a DENSE/HYBRID path. There is no MoE router. The runner already skips
the route tap via `--route-tokens 0` and `--skip-route` (no-ops RouteRecorder
when no layer has gate+switch_mlp, writes `route_skipped=true` instead of
failing). Do not reimplement that flag.

SPECIMEN labels everywhere; mlx TPS is NOT BASE_TRUE_TPS (§14, §60 foreign-runtime).

Census (MEASURED): layers=44 total_params=7585654880
organs: embed=0.8GB, attn=0.97GB, mlp_dense=9.97GB, norm=0.0GB, lm_head=0.8GB, other=2.64GB.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ workspace/campaign/odyssey/patients/O001/census.json
READ workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE; 4-bit fallback is allowed
and must be labelled.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py `--skip-route`. Also measure hybrid SSM-state-vs-KV byte accounting across ctx (short / moderate / long). Census currently buckets Mamba tensors as `other`; write an `ssm` organ bucket and state-vs-KV bytes into the packet representation + execution. No route map.
Keep the model loaded once (one load, memory-safe). Do not delete the canonical weights.

Outputs:
- receipts/odyssey-i/O001_EXTERNAL.json with quant, tps_specimen, ttft, doctor{battery,refusals,seal_ref}, route_skipped=true, ssm_vs_kv{{ctx,state_bytes,kv_bytes}}.
- Refresh workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json execution + doctor + representation.ssm.

## ACCEPTANCE
- receipts/odyssey-i/O001_EXTERNAL.json exists with tps_specimen, ttft, doctor.battery, route_skipped true. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json still schema-valid.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O001/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, workspace/campaign/odyssey/patients/O001/census.json, workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json
VERIFY receipts/odyssey-i/O001_EXTERNAL.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O001 --weights /Users/scammermike/.cache/huggingface/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/41e72f27effbab80cd45b6e884688452253a3686 --runtime mlx --route-tokens 0 --out receipts/odyssey-i/O001_EXTERNAL.json --packet workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json --skip-route
Do not touch Genesis state or tools/odyssey/.
