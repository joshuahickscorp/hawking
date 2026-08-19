# DELEGATION — O003 AGGRESSIVE GRAVITY (q2-g32-experts; gate profile: MLX/Metal)

Patient O003 = `moonshotai/Kimi-VL-A3B-Instruct` (moe; KimiVLForConditionalGeneration),
on disk at `/Users/scammermike/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-Instruct/snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375`. Repo: `/Users/scammermike/Downloads/hawking`.
One bounded Gravity candidate (steer S004 anti-complacency). SPECIMEN-labelled mlx quant; this
is NOT a Hawking NX win (§15).

AGGRESSIVE candgen spec `q2-g32-experts` (experts low-bit; attention/router protected; norms full). candidate_class from grammar. Complete bpw (payload+scales+biases+metadata); record nominal_bits AND complete_bpw. On Doctor fail: failure_localization naming the organ; do NOT globally retreat.

## Read first
READ tools/odyssey_patient_runner.py
READ tools/worker_gate.py
READ tools/doctor_seal.py
READ workspace/campaign/odyssey/patients/O003/census.json
READ workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json
READ receipts/odyssey-i/O003_EXTERNAL.json

Call worker_gate.observe()/gate() before load. Abort on REFUSE.

## BUILD
The runner ALREADY has this mode — RUN it, do NOT modify tools/odyssey_patient_runner.py.
Reuse tools/odyssey_patient_runner.py `--gravity q2-g32-experts`. One candidate, do not sweep.
Reload the quantized model, run the SAME fast-Doctor battery + refusal controls.
Measure stored_bytes, stored_bpw (bytes*8/params), complete_bpw (payload+scales+biases+
metadata+headers), nominal_bits, active_bytes_per_token + active_bpw
(census active-param split for MoE) and battery/refusal DELTA vs receipts/odyssey-i/O003_EXTERNAL.json.
Write receipts/odyssey-i/O003_GRAVITY_q2-g32-experts.json (schema odyssey.patient.gravity.v1): spec, stored/active bpw, complete_bpw,
nominal_bits, candidate_class, conventionality, battery, delta_hits, doctor_seal,
SPECIMEN + quant caveat, verdict CANDIDATE_PASS if delta_hits>=-1 else DEGRADED
(and failure_localization naming the responsible organ — do not globally retreat).
Refresh workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json gravity.wins/kills + candidate_class/conventionality.
Do not delete canonical weights.

## ACCEPTANCE
- receipts/odyssey-i/O003_GRAVITY_q2-g32-experts.json exists with stored_bpw < 16, complete_bpw, nominal_bits, candidate_class,
  active_bpw, battery, delta vs baseline, SPECIMEN label. Must pass, exit 0.
- workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json still schema-valid.

## SCOPE
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/O003/
READ tools/odyssey_patient_runner.py, tools/worker_gate.py, tools/doctor_seal.py, workspace/campaign/odyssey/patients/O003/census.json, workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json, receipts/odyssey-i/O003_EXTERNAL.json
VERIFY receipts/odyssey-i/O003_GRAVITY_q2-g32-experts.json by running the unfenced command below; must pass, exit 0.
python3 tools/odyssey_patient_runner.py --oxx O003 --weights /Users/scammermike/.cache/huggingface/hub/models--moonshotai--Kimi-VL-A3B-Instruct/snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375 --runtime mlx --out receipts/odyssey-i/O003_GRAVITY_q2-g32-experts.json --packet workspace/campaign/odyssey/patients/O003/ODYSSEY_PATIENT_O003.json --gravity q2-g32-experts
Do not touch Genesis state or tools/odyssey/.
