# DELEGATION A — ODYSSEY-I CONTROL PLANE (sandboxed, pure Python)

You are building the autonomous control plane for the Hawking Odyssey-I compiler
curriculum. Repo: `/Users/scammermike/Downloads/hawking`. Python: use
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` (has hf; no torch/mlx needed here).

## Read first (context, do not duplicate what exists)
- `/Users/scammermike/Downloads/h_odyssey.md` §§12,17,18,20,21,22,23,60,64,97 (doctrine).
- `workspace/campaign/odyssey/ODYSSEY.md` — the active ledger + patient queue (source of truth for patients/states).
- `workspace/campaign/odyssey/patient_packet_schema.json` — the packet schema you MUST conform to.
- `workspace/campaign/odyssey/evidence/reuse_surface.md` — exact reusable governor signatures + file:line. USE these, do not rebuild them.
- `workspace/campaign/odyssey/patients/O005/ODYSSEY_PATIENT_O005.json` and `patients/O001/...O001.json` — hand-seeded examples your packet builder must be able to reproduce/refresh.
- `receipts/ascent-2026-08-18/A3B_RECON.json` — prior route classification (seed for transfer matrix).
- `tools/foundry/NEGATIVE_TRANSFER_ATLAS.json`, `tools/foundry/GRAVITY_METHOD_REGISTRY.json` — rulebase/negative seeds.

## REUSE these governors (import, do NOT reimplement)
- `tools/worker_gate.py` — `observe()` + `gate(obs)->{decision,note}`. Call BEFORE launching any model-loading worker; abort on decision=="REFUSE".
- `tools/reclaim_safe.sh` — disk reclaim; pair a 15 GiB floor; preserve `reports/` and receipts.
- `tools/doctor_seal.py` — `seal(...)` structural seal; reuse for every patient seal.
- `tools/ascent_controller.py` — its loop/value pattern is the QUEUE SKELETON. Reuse the shape; give it a new state path + patient target schema. Do NOT touch the existing Genesis state.
- grok-run launch argv: `~/.claude-grok/bin/grok-run delegate|audit|consult --task <slug> --contract <file> --repo <dir> [--profile gate] --background`. Lanes write `~/.claude-grok/tasks/<id>/grok-report.md` + `status`.

## BUILD
1. `tools/odyssey_ctl.py` with subcommands (argparse), state file `workspace/campaign/odyssey/ODYSSEY_STATE.json`:
   - `status` — print the bible §97 compact status block, computed from ODYSSEY_STATE.json + patient packets + census files. No essays.
   - `queue` — show patients with state in {READY,RUNNING,BLOCKED,LANDED,VERIFYING,VERIFIED,REFUTED,ARCHIVED} and phase; seed the queue from ODYSSEY.md's table on first run.
   - `value` — rank READY work by info-value proxy (§22): expected reusable-compiler-info / (wall+gpu+opus cost). Order the queue; no fake precision.
   - `harvest` — scan `~/.claude-grok/tasks/` for tasks whose slug starts `odyssey-`; parse `grok-report.md`; reject malformed (no structured result); classify evidence (§18 labels); write a receipt under `receipts/odyssey-i/<task>.json`; update the relevant patient packet; nominate anomalies/contradictions for Opus into `workspace/campaign/odyssey/OPUS_ESCALATIONS.jsonl`. (Bible §12.)
   - `packet <Oxx>` — assemble/refresh `workspace/campaign/odyssey/patients/<Oxx>/ODYSSEY_PATIENT_<Oxx>.json` from `census.json` + receipts, conforming to `patient_packet_schema.json`. Idempotent.
   - `admit <slug> <est_gib>` — call worker_gate; print GO/REFUSE + note. Helper for model-loading lanes.
   - `--self-check` — assert state round-trips, status renders on the seed queue, packet builder validates against schema, harvest tolerates a malformed task dir. No network, no model.
2. Seed compiler-frontier files (valid JSON, evidence-labelled):
   - `workspace/campaign/odyssey/GRAVITY_RULEBASE.json` — rules per bible §64 schema (conditions, supporting_patients, negative_patients, confidence, reopen_if, physical_rationale). Seed from GRAVITY_METHOD_REGISTRY.json priors + the A3B classification (e.g. rule: "MoE sparse-active path => candidate native expert-gather"; "near-uniform routing => cold-expert compression N/A, per-expert uniform codec ok").
   - `workspace/campaign/odyssey/TRANSFER_MATRIX.json` — rule × patient status grid (TRANSFERRED_UNCHANGED/RETUNED/ARCHITECTURE_SPECIFIC/PATIENT_SPECIFIC/FAILED/HARMFUL/NOT_TESTED). Seed rows from A3B_RECON.json CLASSIFICATION (uniform_routing=A3B/MoE-specific; sparse_active_path=MoE-universal). Patients = O000..O013.
   - `workspace/campaign/odyssey/NEGATIVE_SCIENCE.json` — killed mechanisms + premise + reopen_if, seeded from NEGATIVE_TRANSFER_ATLAS.json and h_odyssey §19 examples.

## Constraints
- Reuse governors; keep new code small and boring. No new deps.
- Never touch the existing Genesis ascent state or `tools/odyssey/` (that is the training-data odyssey — different program).
- Every emitted number/label carries a §18 evidence class.

## ACCEPTANCE
- `python3 tools/odyssey_ctl.py --self-check` exits 0.
- `python3 tools/odyssey_ctl.py status` prints the §97 block for the seeded queue (O005/O001 on-disk, O000/O002/O004 BLOCKED-auth, rest queued).
- `python3 tools/odyssey_ctl.py packet O005` rewrites a schema-valid packet.
- The three frontier JSONs exist, parse, and are non-empty with seeded rows.

## VERIFY (runnable)
```
python3 tools/odyssey_ctl.py --self-check && \
python3 tools/odyssey_ctl.py status && \
python3 tools/odyssey_ctl.py packet O005 && \
python3 -c "import json;[json.load(open(f)) for f in ['workspace/campaign/odyssey/GRAVITY_RULEBASE.json','workspace/campaign/odyssey/TRANSFER_MATRIX.json','workspace/campaign/odyssey/NEGATIVE_SCIENCE.json']];print('frontier ok')"
```
Report the diff and the VERIFY output.

## SCOPE
WRITE tools/odyssey_ctl.py
WRITE workspace/campaign/odyssey/
WRITE receipts/odyssey-i/
READ h_odyssey.md, tools/worker_gate.py, tools/doctor_seal.py, tools/ascent_controller.py, tools/foundry/, receipts/ascent-2026-08-18/A3B_RECON.json
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py --self-check` — tests must pass, exit 0.
