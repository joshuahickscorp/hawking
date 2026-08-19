# DELEGATION — PATIENT-CYCLING MECHANICS (sandboxed, pure Python, ctl.py ONLY)

Make the loop cycle whole PATIENTS through the canonical ladder unattended, not just
obligations (steer S002). Edit ONLY `tools/odyssey_ctl.py` (+ its state/index JSONs).
Do NOT touch `tools/odyssey_patient_runner.py` (a parallel lane owns it). Repo
`/Users/scammermike/Downloads/hawking`, python
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`. Branch odyssey-i.

## Read first
- `tools/odyssey_ctl.py` — controller with the P0 completion index + write-scope gate.
- `workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json`, `ODYSSEY_STATE.json`, `ODYSSEY.md` (ladder O000..O013).

## Concepts (keep P0's separation)
completion index = workflow state; packet = patient; receipts = evidence.

## BUILD 1 — required-obligation set + retirement
Per patient CLASS define the BOUNDED required obligation set (info budget, steer S002 — do NOT over-deepen):
- MoE (O005/O003/O006): `external-science`, `route-map`, `sensitivity-map`, `gravity-moe`, `nx-gather-moe` (+ `transfer-control` if the patient has a `reference` sibling in ODYSSEY.md).
- dense (O004): `external-science`, `sensitivity-map`, `gravity-dense`, `nx-dense`.
- hybrid (O001): `external-science`, `ssm-accounting`, `sensitivity-map`, `gravity-hybrid`, `nx-state-hybrid`.
A patient is RETIRE-ELIGIBLE when every required mechanism has a terminal (VERIFIED/REFUTED) completion entry.
`odyssey_ctl.py retire <OXX>` (and auto in `cycle`): assert eligibility, write a completion entry `mechanism_id="patient-sealed"` status VERIFIED, set packet phase `SEALED` + state `RETIRED`, and emit `receipts/odyssey-i/<OXX>_PATIENT_SEAL.json` (summary: sealed mechanisms, receipt refs, source_revision). Do NOT delete weights here (a separate disk-gated reclaim step handles bulk; just record `reclaimable: true`).

## BUILD 2 — next-patient acquisition
`odyssey_ctl.py acquire-next` (and auto in `cycle` when the ready frontier is empty):
- Pick the lowest-numbered canonical ladder patient (ODYSSEY.md order) that is NOT on disk, NOT RETIRED, and NOT blocked (skip HF-gated O000/O002 unless a token+license is available — detect by attempting `hf` metadata, else mark BLOCKED and skip).
- Gate on disk: require free >= (patient est size + 45 GiB floor); if not, first try retiring+reclaiming a RETIRED patient's weights (record provenance), else skip acquisition and log `disk-hold`.
- Download via `hf download <repo>` (background is fine), run `tools/odyssey_census.py` when complete, seed `patients/<OXX>/` packet, register its obligations. Do NOT block the cycle on a long download — mark the patient ACQUIRING and move on; the next cycle picks it up when on-disk.

## BUILD 3 — register gravity/nx obligation templates
Add templates that render SG-valid contracts driving the runner's NEW modes (a parallel lane adds these runner flags; agree on the exact flags):
- `gravity-moe` → runner `--gravity q3-g32-experts` (per-expert q3 group32 + attn q4); `gravity-dense` → `--gravity q4-g64`; `gravity-hybrid` → `--gravity q4-g64-attn-mlp` (protect SSM/norm). Each: Doctor battery + stored/active BPW vs baseline → `receipts/odyssey-i/<OXX>_GRAVITY_*.json` (SPECIMEN).
- `nx-gather-moe` → runner `--nx-gather` (theoretical selected-expert bytes/token vs measured); `nx-state-hybrid` → `--nx-state`; `nx-dense` → `--nx-dense`. → `<OXX>_NX_*.json`.
- write_set for these templates = `tools/odyssey_patient_runner.py` + that patient's packet + its receipts (so they serialize with each other + sensitivity, per the write-scope gate).

## BUILD 4 — `cycle` command (the driver tick)
`odyssey_ctl.py cycle [--go] [--max-lanes N]`: harvest finished lanes → rebuild/refresh completions → retire eligible patients → if ready frontier empty, `acquire-next` → rank remaining ready obligations by info-value → admit up to max-lanes honoring completion + write-scope + worker/disk/clean_box gates → print a one-line §97 summary → yield. `--go` launches; default dry-run. Idempotent, event-safe, cheap (no model calls in the controller itself).

## Constraints
Reuse P0 code; edit ONLY ctl.py + odyssey JSONs; NO runner edits; no commit/push/launchd. Every emitted label §18-classed.

## ACCEPTANCE
- `python3 tools/odyssey_ctl.py --self-check` passes (extend: a patient with all required mechanisms VERIFIED is retire-eligible; one missing is not; `cycle --dry-run` renders a plan without launching).
- `python3 tools/odyssey_ctl.py cycle --dry-run` (ODYSSEY_HEADROOM_ADMIT=1) prints: retire-eligible patients (none yet — all need gravity/nx), ready obligations (O005 sensitivity, O006 transfer, O004 external, O003 sensitivity, gravity/nx per patient), with write-scope serialization applied, launches nothing.
- `retire`/`acquire-next` refuse when preconditions unmet, with a clear reason.

## SCOPE
WRITE tools/odyssey_ctl.py
WRITE workspace/campaign/odyssey/
READ workspace/campaign/odyssey/ODYSSEY.md, workspace/campaign/odyssey/patients/, tools/odyssey_census.py
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py --self-check` and `python3 tools/odyssey_ctl.py cycle --dry-run` — must pass, exit 0.
