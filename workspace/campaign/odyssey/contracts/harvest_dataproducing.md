# DELEGATION — HARVEST COMPLETES DATA-PRODUCING LANES (sandboxed, ctl.py ONLY)

The autonomous loop stalls: gravity/nx/sensitivity lanes RUN the existing runner
modes but grok often makes an incidental 1-5 line runner tweak, so `harvest`
classifies the whole lane CODE → REVIEW → never writes the completion → the mechanism
is never marked done → the cycle cannot advance. Fix the classification. Edit ONLY
`tools/odyssey_ctl.py`. Repo `/Users/scammermike/Downloads/hawking`, python
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`. Branch odyssey-i.

## Principle
Every current template (`external-science-*`, `route-map`, `sensitivity-map`,
`gravity-*`, `nx-*`, `transfer-control`) is DATA-PRODUCING: it RUNS the existing
runner and its deliverable is a RECEIPT (+ packet fields). The runner is already
feature-complete for these. So an incidental edit to `tools/odyssey_patient_runner.py`
by such a lane is NOISE, not a deliverable — harvest must take the receipt + packet
and IGNORE the runner diff, then auto-complete.

## FIX `harvest` classification
Classify by the lane's TEMPLATE/mechanism, not by diff-file inspection:
- If the lane's template is a KNOWN data-producing mechanism AND the lane produced its
  expected receipt (`receipts/odyssey-i/<OXX>_*.json` matching the mechanism's
  RECEIPT_PATTERN), classify **DATA-ONLY** regardless of any `tools/*.py` diff:
  copy the receipt(s) + apply ONLY the patient-packet changes from the worktree, DROP
  any `tools/` diff (the modes already exist on HEAD), write the completion via the
  existing `complete()`/harvest path, and cleanup the worktree.
- Only classify CODE → REVIEW when the lane is NOT a known data-producing template, OR
  it produced NO valid receipt (a real infra/broken lane that needs human eyes).
- Malformed (no report / no receipt) → REFUTED with reason, as before.
Keep it idempotent. Preserve the existing DATA-ONLY auto-merge + completion writing.

## Also
Make the gravity/nx/sensitivity/external/transfer auto-contract templates state
explicitly: "the runner ALREADY has this mode — RUN it, do NOT modify
tools/odyssey_patient_runner.py" and drop the runner from their declared `write_set`
(so these lanes are data-only + parallel-safe, no false runner serialization). If a
lane genuinely needs a runner change it will produce no valid receipt and fall to
REVIEW — which is the correct signal.

## Constraints
Edit ONLY ctl.py. No runner edits. Reuse `complete()`/existing harvest merge code. No
commit/push/launchd. Keep `--self-check` green (extend it: a data-producing fake lane
whose diff includes a tools/*.py tweak still classifies DATA-ONLY, drops the code, and
auto-completes; a non-template or no-receipt fake lane goes to REVIEW).

## ACCEPTANCE
- `python3 tools/odyssey_ctl.py harvest --dry-run` classifies the real
  `odyssey-o005-sensitivity-map-*` and `odyssey-o003-sensitivity-map-*` lanes DATA-ONLY
  (they have receipts + a trivial runner tweak) with action MERGE+COMPLETE, not REVIEW.
- `python3 tools/odyssey_ctl.py harvest` then writes O005:sensitivity-map (and any other
  done data lane) into ODYSSEY_COMPLETIONS.json and drops the runner tweak.
- `python3 tools/odyssey_ctl.py --self-check` passes.

## SCOPE
WRITE tools/odyssey_ctl.py
WRITE workspace/campaign/odyssey/
READ receipts/odyssey-i/, workspace/campaign/odyssey/patients/
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py --self-check` and `python3 tools/odyssey_ctl.py harvest --dry-run` — must pass, exit 0.
