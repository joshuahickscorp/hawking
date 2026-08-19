# DELEGATION — LOOP DONE-DETECTION FIX (sandboxed, pure Python)

Bug: `tools/odyssey_ctl.py run` re-selects obligations that are ALREADY COMPLETE
(it launched O001 external-science-dense again after O001 external science was sealed).
Unattended, the driver would re-run finished work forever. Fix the completion check.

Repo `/Users/scammermike/Downloads/hawking`, python
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.

## Read first
- `tools/odyssey_ctl.py` — the `run`/`value`/obligation-selection code + the templates.
- `workspace/campaign/odyssey/patients/*/ODYSSEY_PATIENT_*.json` — the completion markers live here.

## FIX
An obligation is DONE (must NOT be selected/launched) when its completion marker exists in the patient packet:
- `external-science-moe` / `external-science-dense` DONE when `execution.baseline_tps` (or `execution.tps_specimen`) is a real number AND a `receipts/odyssey-i/<OXX>_EXTERNAL.json` exists.
- `sensitivity-map` DONE when `representation.per_organ_sensitivity` is non-null AND `receipts/odyssey-i/<OXX>_SENSITIVITY.json` exists.
- `transfer-control` DONE when a `receipts/odyssey-i/<OXX>_TRANSFER.json` exists.
- SSM/`ssm-accounting` DONE when `representation.ssm` / `execution` ssm fields are filled.
Selection must filter these out. Also treat an obligation as in-flight (skip) if a lane for the same (oxx, template) is RUNNING or sitting in REVIEW_QUEUE.jsonl. Keep the info-value ranking for the REMAINING obligations.

## Constraints
Reuse existing code; small diff; no new deps; do not touch Genesis or `tools/odyssey/`. No git commit/push.

## ACCEPTANCE
- `python3 tools/odyssey_ctl.py run --dry-run` no longer lists O001/O005 external-science or O001 sensitivity (all sealed in their packets); it lists only genuinely-pending obligations (e.g. O003/O006 external, O004 external, O005 sensitivity, transfers).
- `--self-check` still passes; extend it: an obligation whose packet has the completion marker is filtered out of selection.

## SCOPE
WRITE tools/odyssey_ctl.py
READ workspace/campaign/odyssey/patients/
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py run --dry-run` and `python3 tools/odyssey_ctl.py --self-check` — must pass, exit 0.
