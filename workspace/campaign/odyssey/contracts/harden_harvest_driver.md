# DELEGATION — HARDEN HARVEST + HANDS-OFF DRIVER (sandboxed, pure Python)

Make the Odyssey-I loop run hands-off safely. Extend `tools/odyssey_ctl.py` and add a
driver. Repo `/Users/scammermike/Downloads/hawking`, python
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.

## Problem being fixed
Grok lanes finish UNCOMMITTED; new receipts are untracked and lost if the worktree is
cleaned first. And some lanes only write DATA (receipts + packet fields) while others
change CODE (`tools/*.py`) — code must NOT auto-merge (needs human review), data is safe
to merge. The driver must never spawn a same-file-editing lane concurrently.

## Read first
- `tools/odyssey_ctl.py` — harvest/run/evaluate_gates already exist. Extend them.
- `workspace/campaign/odyssey/ODYSSEY_STATE.json`, `RUN_LOG.jsonl`.
- Grok task dirs `~/.claude-grok/tasks/odyssey-*/` hold `status`, `grok-report.md`, `diff.patch`; worktrees `~/.claude-grok/worktrees/odyssey-*/` hold untracked outputs.

## BUILD 1 — harden `odyssey_ctl.py harvest`
For each `~/.claude-grok/tasks/odyssey-*` whose `status`==done and is recorded RUNNING in state:
- Determine its files from `diff.patch` (list of `+++ b/…`) PLUS untracked files in its worktree (`git -C <worktree> status --porcelain`).
- Classify: **DATA-ONLY** if every changed/new path is under `receipts/` or `workspace/campaign/odyssey/patients/*/` (json); **CODE** if any path is under `tools/` or is `*.py` or other source.
- DATA-ONLY: copy the untracked receipt/packet files from the worktree into the main tree, apply the packet-field changes, write a harvest receipt `receipts/odyssey-i/harvest_<task>.json` (§18-labelled), mark the lane VERIFIED in state, then `grok-run cleanup --id <task>`.
- CODE: append `{task, files, report, worktree}` to `workspace/campaign/odyssey/REVIEW_QUEUE.jsonl`, mark the lane REVIEW, DO NOT merge and DO NOT cleanup (leave the worktree for Opus).
- Reject malformed lanes (no report / no receipt) -> mark REFUTED with reason. Idempotent; safe to re-run.
- `harvest --dry-run` prints the classification + planned action for each, changes nothing.

## BUILD 2 — `tools/odyssey_driver.sh`
One tick, self-contained, logs to `workspace/campaign/odyssey/driver.log` with a UTC-free timestamp arg (`date` is fine in a shell script):
1. `python3 tools/odyssey_ctl.py harvest` (reap finished lanes).
2. `ODYSSEY_HEADROOM_ADMIT=1 python3 tools/odyssey_ctl.py run --go --max-lanes 2` (launch next, self-gated).
Never let two RUNNING lanes edit the same file: the loop's launcher must skip an obligation whose template is known to edit `tools/odyssey_patient_runner.py` (sensitivity/dense-generalization) while any lane is RUNNING — those go one-at-a-time; RUN-only templates (external-science-moe/transfer on the existing runner) may fill to max-lanes.

## BUILD 3 — launchd template (do NOT load)
Write `workspace/campaign/odyssey/com.hawking.odyssey.plist` — a LaunchAgent that runs `tools/odyssey_driver.sh` every 1800 s, `RunAtLoad` false, stdout/err to `workspace/campaign/odyssey/driver.log`. Write it to the REPO only; do NOT install or load it. Put the exact `launchctl bootstrap gui/$UID …` / `launchctl bootout …` commands in your report.

## Constraints
- Reuse existing code; keep it boring. No new deps. Do not touch Genesis state or `tools/odyssey/`.
- Never auto-merge CODE. Never `git commit`, `push`, or load launchd.

## ACCEPTANCE
- `python3 tools/odyssey_ctl.py harvest --dry-run` classifies each finished lane DATA-ONLY vs CODE and prints the planned action, exit 0.
- `tools/odyssey_driver.sh` runs harvest then a gated `run --go`, appends to driver.log, exit 0.
- `com.hawking.odyssey.plist` exists, is valid XML, RunAtLoad false, not loaded.
- `--self-check` still passes (extend it: a DATA-ONLY fake lane harvests+would-cleanup; a CODE fake lane goes to REVIEW_QUEUE and is NOT merged).

## SCOPE
WRITE tools/odyssey_ctl.py
WRITE tools/odyssey_driver.sh
WRITE workspace/campaign/odyssey/
READ tools/worker_gate.py, tools/reclaim_safe.sh
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py harvest --dry-run` and `python3 tools/odyssey_ctl.py --self-check` — must pass, exit 0.
