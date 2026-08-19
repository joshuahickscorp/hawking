# DELEGATION — O001 MODEST GRAVITY (q4-g64-attn-mlp; gate profile: MLX/Metal)

Patient O001 = `tiiuae/Falcon-H1-7B-Instruct` (hybrid; FalconH1ForCausalLM),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/41e72f27effbab80cd45b6e884688452253a3686`. Repo: `/Users/scammermike/Downloads/hawking`.
One bounded Gravity candidate (steer S002). SPECIMEN-labelled mlx quant; this
is NOT a Hawking NX win (§15).

Protect SSM/conv/norm at full precision; quantize attn+MLP only (`q4-g64-attn-mlp`).

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ tools/doctor_seal.py
READ workspace/campaign/odyssey/patients/O001/census.json
READ workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json
READ receipts/odyssey-i/O001_EXTERNAL.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py `--gravity q4-g64-attn-mlp`. One candidate, do not sweep.
Reload the quantized model, run the SAME fast-Doctor battery + refusal controls.
Measure stored_bytes, stored_bpw (bytes*8/params), active_bytes_per_token + active_bpw
(census active-param split for MoE) and battery/refusal DELTA vs receipts/odyssey-i/O001_EXTERNAL.json.
Write receipts/odyssey-i/O001_GRAVITY_q4-g64-attn-mlp.json (schema odyssey.patient.gravity.v1): spec, stored/active bpw, battery,
delta_hits, doctor_seal, SPECIMEN + quant caveat, verdict CANDIDATE_PASS if
delta_hits>=-1 else DEGRADED. Refresh workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json gravity.wins/kills.
Do not delete canonical weights.

## ACCEPTANCE
- receipts/odyssey-i/O001_GRAVITY_q4-g64-attn-mlp.json exists with stored_bpw < 16, active_bpw, battery, delta vs baseline,
  SPECIMEN label. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json still schema-valid.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O001/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, tools/doctor_seal.py, workspace/campaign/odyssey/patients/O001/census.json, workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json, receipts/odyssey-i/O001_EXTERNAL.json
VERIFY receipts/odyssey-i/O001_GRAVITY_q4-g64-attn-mlp.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O001 --weights /Users/scammermike/.cache/huggingface/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/41e72f27effbab80cd45b6e884688452253a3686 --runtime mlx --out receipts/odyssey-i/O001_GRAVITY_q4-g64-attn-mlp.json --packet workspace/campaign/odyssey/patients/O001/ODYSSEY_PATIENT_O001.json --skip-route --gravity q4-g64-attn-mlp
Do not touch Genesis state or tools/odyssey/.
