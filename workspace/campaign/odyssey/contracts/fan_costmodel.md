# DELEGATION — COST MODEL + DETACHMENT METRICS (NEW tools/odyssey_costmodel.py + schema, module-only)
Build the compile-economics + detachment-metrics recorder (steer S003 §7-11, S004 §27/§71). Standalone module.
Repo /Users/scammermike/Downloads/hawking. Branch odyssey-i. Do NOT edit ctl.py/runner.

## BUILD `tools/odyssey_costmodel.py`
- `record(patient, event, wall_s, bytes_scanned=0, bytes_transformed=0, grok_lane=None, opus=False, extra={})` -> appends to `workspace/campaign/odyssey/COMPILE_ECONOMICS.jsonl` (per-patient event log; §7 fields + frontier-depth: time_to_conventional_anchor/first_sub3/first_sub2_5/first_sub2/best_frontier, candidates_to_frontier, cheap_kills, doctor_fast_count, full_doctor_count, grok_calls_to_frontier).
- `derive(patient) -> dict` normalized metrics (§8: gravity GB/s scanned & transformed, s/B-source-param, s/B-active-param, first-NX s/B-param, experiments/valid-NX, grok-lanes/patient, opus/patient, rules reused/retuned/new).
- `detachment_metrics() -> dict` (§71: deterministic_decisions/total from RUN_LOG.jsonl if present, opus_escalations/patient & /cycle, patient_wall/opus, desired-trend flags).
- `fit_cost_model()` -> writes `workspace/campaign/odyssey/ODYSSEY_COST_MODEL.json` with per-input-feature estimates + UNCERTAINTY (never point-estimate as fact); if <2 patients of data, emit "insufficient data" honestly.
Deterministic, stdlib+json. Timestamps passed in (do not call wall-clock in pure fns; the recorder may use `time`).

## Self-check (exit 0): record a couple synthetic events to a temp path, derive() returns finite normalized metrics, fit_cost_model reports insufficient-data gracefully with <2 patients.
## SCOPE
WRITE tools/odyssey_costmodel.py
WRITE workspace/campaign/odyssey/
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json
VERIFY tools/odyssey_costmodel.py by running `python3 tools/odyssey_costmodel.py --self-check` — exit 0.
