# DELEGATION — DETERMINISTIC DIRECT EXECUTION (ctl.py) — stop wrapping runner commands in Grok

The loop currently launches deterministic science (external/route/sensitivity/gravity/nx/transfer)
as GROK DELEGATE lanes — a full Grok agent reasons for minutes before running ONE deterministic
command (`odyssey_patient_runner.py --gravity q2...`), and a faulty/hung Grok agent DEAD-ENDS the
cycle. Fix per S004 §52/§55 (deterministic software owns execution; Grok only for novelty). Make the
Python orchestrator RUN THE RUNNER DIRECTLY as a subprocess, PID-tracked, timeout-reaped, self-healing.
Reserve Grok delegate for NOVELTY (new-mechanism design) + genuine code-building ONLY. Repo
`/Users/scammermike/Downloads/hawking`, py `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.
Branch odyssey-i. Driver is PAUSED; no concurrent control-plane writer.

## Read first
`tools/odyssey_ctl.py` (cycle/run/evaluate_gates/launch path/harvest/state/write_set), the template →
runner-spec maps, `tools/odyssey_patient_runner.py` argv (`--oxx --weights --gravity <spec> --sensitivity
--nx-gather --route-tokens --out`), `tools/odyssey_memgate.py` (admit), `ODYSSEY_MANIFEST.json` (weights
paths / patient snapshot dirs), `ODYSSEY_POLICY.detachment.memory`.

## BUILD
1. **DETERMINISTIC_TEMPLATES** = {external-science-moe, external-science-dense, route-map, sensitivity-map,
   gravity-moe, gravity-dense, gravity-hybrid, gravity-aggressive-moe, gravity-aggressive-dense,
   gravity-aggressive-hybrid, nx-gather-moe, nx-state-hybrid, nx-dense, transfer-control} — all run the
   existing runner with known args (no code change, no novelty).
2. **`launch_deterministic(oblig)`**: resolve the runner argv from the template + patient weights (from
   manifest/HF cache snapshot) + `--out receipts/odyssey-i/<OXX>_<...>.json`. `memgate.admit(est_gib,
   in_flight)`; if GO, spawn the framework-python runner as a DETACHED subprocess (setsid/`start_new_session`,
   stdout/err to `workspace/campaign/odyssey/lanes/<oxx>-<mech>.log`). Record in ODYSSEY_STATE work[]:
   {oxx, mechanism, template, kind:"subprocess", pid, started_epoch, receipt_path, timeout_s, status:RUNNING}.
   `started_epoch`/timeouts: the caller passes wall time in (no bare Date.now in pure fns; the launcher may
   read time). timeout_s default from policy (`lane_timeout_min`, default 30) * 60.
3. **`reap_lanes(now_epoch)`** (called at the TOP of every cycle tick, before admission): for each RUNNING
   subprocess entry — PID alive & age<timeout → still RUNNING (counts toward cap); PID dead & receipt exists
   → harvest it (write completion via existing `complete()`, mark VERIFIED, drop entry); PID dead & no receipt
   → mark FAILED, retry ONCE (re-queue), else REFUTED + drop; PID alive & age>timeout → kill the process group,
   mark FAILED (retry once). This is the SELF-HEALING: no Grok dependency, a dead/hung lane frees the cap
   automatically. Count RUNNING = live subprocess PIDs + live grok-novelty lanes (by status, also age-capped).
4. **cycle/run routing**: DETERMINISTIC_TEMPLATES → `launch_deterministic` (subprocess). NOVELTY templates
   (novelty-*, and any explicit code-building obligation) → grok delegate as before. Multi-model via memgate
   across BOTH. Keep write-scope + completion gates.
5. **harvest**: also completes subprocess lanes from their receipts (already covered by reap_lanes); keep the
   grok-lane harvest for novelty/code lanes.

## Constraints
Reuse memgate/complete/existing harvest. Deterministic; no Grok for runner runs. The runner re-execs to the
mlx-capable framework python itself — invoke it with that interpreter. No commit/push/launchd.

## ACCEPTANCE
- `python3 tools/odyssey_ctl.py --self-check` passes incl: a deterministic obligation plans a SUBPROCESS
  (not a grok lane); a RUNNING entry with a dead PID + existing receipt is reaped→VERIFIED; a dead PID + no
  receipt is FAILED→retry; an over-timeout live PID is killed→FAILED; cap counts live PIDs.
- `ODYSSEY_HEADROOM_ADMIT=1 python3 tools/odyssey_ctl.py cycle --dry-run` shows deterministic obligations
  planned as subprocess runner invocations (print the argv), memgate-gated, multi-model.
- `python3 tools/odyssey_ctl.py cycle --go --max-lanes 3` (real) spawns >=1 detached runner subprocess that
  loads mlx and writes a receipt (verify a lanes/*.log shows mlx activity), and the NEXT `cycle` reaps+completes it.

## SCOPE
WRITE tools/odyssey_ctl.py
WRITE workspace/campaign/odyssey/
READ tools/odyssey_patient_runner.py, tools/odyssey_memgate.py, workspace/campaign/odyssey/ODYSSEY_MANIFEST.json, workspace/campaign/odyssey/ODYSSEY_POLICY.json
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py --self-check` and `python3 tools/odyssey_ctl.py cycle --dry-run` — must pass, exit 0.
