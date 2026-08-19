# DELEGATION — ODYSSEY-I AUTONOMOUS LOOP (sandboxed, pure Python)

Extend `tools/odyssey_ctl.py` (already built + committed) with a `run` subcommand:
the §21/§22 autonomous queue that selects the highest info-value READY obligation,
generates a concrete grok contract for it, launches ONE gated lane, and lets
`harvest` reap it. Repo `/Users/scammermike/Downloads/hawking`, python
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.

## Read first
- `tools/odyssey_ctl.py` — the controller you are extending (value/harvest/packet/admit already exist). Reuse them.
- `workspace/campaign/odyssey/ODYSSEY_STATE.json` — patient/obligation state you read + update.
- `workspace/campaign/odyssey/contracts/o005_external_science.md` and `control_plane.md` — SG-passing contract shape to mirror (WRITE/VERIFY/ACCEPTANCE sections; python3 commands live in unfenced lines; SG grammar rejects fenced-only commands).
- `tools/worker_gate.py` (observe/gate), `tools/reclaim_safe.sh`, and `machine_state.clean_box_ok` — the gates you MUST honor.
- `~/.claude-grok/bin/grok-run` argv (delegate --task --contract --repo [--profile gate] --background).

## BUILD: `odyssey_ctl.py run`
- `run --dry-run` (DEFAULT — never launches): rank READY obligations via existing `value`; for the top `--max-lanes` (default 2), pick an obligation-type template, render a concrete contract to `workspace/campaign/odyssey/contracts/auto/<oxx>_<type>.md`, and PRINT the plan (obligation, patient, contract path, model-loading? gate verdict). Launch NOTHING.
- `run --go [--max-lanes N]`: same selection, then for each planned lane actually launch `grok-run delegate --task odyssey-<oxx>-<type> --contract <auto> --repo <repo> [--profile gate] --background`, recording it in ODYSSEY_STATE.json (state RUNNING, task id, started, contract). GATES, enforced before each launch:
  - model-loading lane → call worker_gate.observe()/gate(); if REFUSE, SKIP that lane (log why), try next.
  - disk free < 45 GiB → run `reclaim_safe.sh`; if still low, skip download-implying lanes.
  - TIMING / native-measurement lane → require `clean_box_ok()`; if false (any grok worktree live / swap), SKIP (§14 protected-GPU).
  - never exceed `--max-lanes` concurrently-RUNNING odyssey lanes (count grok worktrees + state RUNNING); hard cap 3.
  - each rendered contract MUST pass the SG gate (WRITE + VERIFY unfenced lines). Validate by dry-checking `~/.claude-grok/v2/lint.mjs` before launch; refuse to launch (and log) any contract that would be SG-rejected. Never use SG_OFF.
- After launching, print the §97 status. `run` is idempotent and safe to re-invoke; it does NOT block on lanes (harvest reaps them later).

## Obligation → contract templates (render full patient-specific contracts)
Parameterize each from the patient census.json + packet. Minimum set:
- `external-science-moe` — the O005-style runner (route map + baseline + fast-doctor). For MoE patients missing routing data.
- `external-science-dense` — baseline TPS + fast-doctor + (hybrid) SSM-state-vs-KV byte accounting across ctx; NO route map. For dense/hybrid patients (O001 Falcon, O004 Mistral).
- `sensitivity-map` — per-organ / per-expert Doctor sensitivity: zero/round each organ, re-run the fast battery, record capability delta (§17 per_organ_sensitivity). For any censused patient with a baseline.
- `transfer-control` — run the O005 runner on a sibling and diff route/representation vs a named reference patient (O006 vs O005 §41); write a transfer-matrix delta.
Each template reuses `tools/odyssey_patient_runner.py`; templates that need a non-MoE path must say so and let the runner skip the route tap.

## Obligation source
Read obligations from ODYSSEY_STATE.json (per-patient `next`/phase) + the ODYSSEY.md "Active obligations" list. An obligation is READY if its patient is on_disk and its prerequisites (phase order INGEST→...→SEAL) are met.

## Constraints
- Reuse the gates; keep new code boring; no new deps. Do not touch Genesis state or `tools/odyssey/`.
- Every launch decision logged to `workspace/campaign/odyssey/RUN_LOG.jsonl` (obligation, verdict, gate results, task id or skip-reason), §18-labelled.
- DRY-RUN IS THE DEFAULT. `--go` is required to spawn anything.

## ACCEPTANCE
- `python3 tools/odyssey_ctl.py run --dry-run` prints a ranked plan with rendered contract paths and gate verdicts, launches nothing, exit 0.
- Rendered auto-contracts exist under `workspace/campaign/odyssey/contracts/auto/` and each passes `~/.claude-grok/v2/lint.mjs` (no ERROR).
- `run --go --max-lanes 0` launches nothing (cap respected).
- `--self-check` still passes (extend it: assert the loop selects, renders an SG-valid contract, and honors max-lanes/gate skips with injected REFUSE).

## SCOPE
WRITE tools/odyssey_ctl.py
WRITE workspace/campaign/odyssey/
READ tools/worker_gate.py, tools/reclaim_safe.sh, workspace/campaign/odyssey/contracts/o005_external_science.md
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py run --dry-run` and `python3 tools/odyssey_ctl.py --self-check` — must pass, exit 0.
