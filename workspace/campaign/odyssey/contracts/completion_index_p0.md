# DELEGATION — P0 CANONICAL COMPLETION INDEX + REPLAY-PROOF SCHEDULER (sandboxed, pure Python)

The autonomous loop can re-select ALREADY-SEALED obligations (it re-launched O001
external-science after it was sealed). A prior fix guessed packet marker fields and
broke self-check — REJECTED. Fix it properly with ONE canonical completion source of
truth, and add write-scope serialization. This is the only thing blocking unattended
operation. Repo `/Users/scammermike/Downloads/hawking`, python
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`. Branch odyssey-i.

## Read first
- `tools/odyssey_ctl.py` — the controller (run/value/harvest/select). You extend it.
- `receipts/odyssey-i/*.json` — the sealed evidence to backfill from.
- `workspace/campaign/odyssey/patients/*/ODYSSEY_PATIENT_*.json` — patient descriptions (NOT the completion DB).

## THREE SEPARATE CONCEPTS (do not overload one JSON — steer §3)
- patient packet = describes the PATIENT.
- completion index = WORKFLOW state (what science is sealed).
- receipts = EVIDENCE.
Flow: experiment → receipt → verification → completion index → packet → scheduler.

## BUILD 1 — `workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json`
A list of entries; each:
`{obligation_id, patient_id, mechanism_id, status, receipt_ref, receipt_sha256, source_revision, completed_at, supersedes, reopen_if}`.
- `status` ∈ {VERIFIED, REFUTED, SUPERSEDED, ARCHIVED} (terminal only).
- `mechanism_id` = the science type: `external-science`, `sensitivity-map`, `route-map`, `transfer-control`, `gravity-<name>`, `nx-<name>`, etc.
- `reopen_if` = a machine-checkable predicate string (e.g. `source_revision != <rev>` or `null`).
- A `complete` API/function writes entries; `completed_at` must be passed IN (do not call Date.now — pass via arg/env, scripts may use shell `date`).

## BUILD 2 — backfill the index from existing receipts (idempotent)
`odyssey_ctl.py completions --rebuild` derives VERIFIED entries from `receipts/odyssey-i/*.json`:
- O001 external-science (O001_EXTERNAL.json), O001 sensitivity-map (O001_SENSITIVITY.json),
  O003 external-science (O003_EXTERNAL.json), O005 external-science + route-map (O005_EXTERNAL.json).
- receipt_sha256 = sha256 of the receipt file; source_revision = `git rev-parse HEAD`.
- Do NOT invent completions for science that has no receipt (O005 sensitivity has NO receipt → it stays PENDING).

## BUILD 3 — scheduler queries the completion index (replay-proof)
Selection/`run` must treat an (patient_id, mechanism_id) as DONE (never launch) when a terminal
entry exists AND its `reopen_if` is NOT mechanically satisfied. VERIFIED/REFUTED/SUPERSEDED all block
re-launch. An obligation with a valid, mechanically-satisfied `reopen_if` is selectable. Remove the old
packet-marker-based done check entirely — completions is the ONLY source.

## BUILD 4 — write-scope serialization (steer §4)
Each obligation/template declares `write_set` (files it may edit) + `exclusive_resources` (e.g. `gpu`).
The launcher REFUSES a concurrent launch whose `write_set` intersects any RUNNING lane's write_set, or
whose exclusive_resources collide. Different-file lanes run in parallel; same-file lanes SERIALIZE.
Templates: `external-science-*` and `sensitivity-map` write `tools/odyssey_patient_runner.py` (+ that
patient's packet + receipts) → SERIALIZE with each other; a RUN-only template that does not edit the
runner writes only its packet+receipts → parallel-safe. Encode write_set per template honestly.

## BUILD 5 — adversarial idempotence battery in `--self-check` (steer §2; watch fail AND pass)
Synthetic completions + queue: A=VERIFIED, B=REFUTED, C=SUPERSEDED, D=pending(no entry),
E=VERIFIED but reopen_if mechanically TRUE, F=same patient different mechanism (no entry).
Assert selection verdicts: A/B/C REFUSE, D/F LAUNCH, E LAUNCH. Also assert two obligations with
intersecting write_sets are never both admitted concurrently. Keep the existing gate/self-check assertions passing.

## Constraints
Reuse existing code; no new deps; do not touch Genesis or `tools/odyssey/`. No git commit/push. No launchd.

## ACCEPTANCE
- `python3 tools/odyssey_ctl.py completions --rebuild` writes ODYSSEY_COMPLETIONS.json with VERIFIED entries for O001 external+sensitivity, O003 external, O005 external+route (and NOT O005 sensitivity).
- `python3 tools/odyssey_ctl.py run --dry-run` (with `ODYSSEY_HEADROOM_ADMIT=1`) does NOT list O001 external-science, O001 sensitivity-map, O003 external-science, or O005 route-map; it DOES list O005 sensitivity-map, O004 external, O006 transfer (genuinely pending).
- `python3 tools/odyssey_ctl.py --self-check` passes including the A-F idempotence battery + write-scope collision assertion, exit 0.

## SCOPE
WRITE tools/odyssey_ctl.py
WRITE workspace/campaign/odyssey/
READ receipts/odyssey-i/, workspace/campaign/odyssey/patients/, tools/worker_gate.py
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py --self-check` and `python3 tools/odyssey_ctl.py completions --rebuild` — must pass, exit 0.
