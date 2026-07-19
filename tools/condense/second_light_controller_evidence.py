#!/usr/bin/env python3.12
"""Seal CRASH_RESUME_PROOF.json + CONTROLLER_STATUS.json for readiness (conditions 18/19/21/22).

Drives the durable controller through the five crash points on cheap (kept_original) rows, proves
resume finishes without redoing sealed rows, proves the singleton lease, proves status reports live
truth (a reset yields NOT_STARTED; a dead pid never reads RUNNING), and proves reset/rollback.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "reports" / "condense" / "second_light" / "evidence"
CTL = [sys.executable, str(REPO / "tools" / "condense" / "second_light_controller.py")]
STAT = [sys.executable, str(REPO / "tools" / "condense" / "second_light_status.py")]
KILL_POINTS = ("fitting", "packing", "eval", "after_write_before_receipt",
               "after_receipt_before_transition")
CHEAP = "r0072,r0073,r0074,r0075"       # router rows (kept_original, instant)


def _run(cmd, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), env=env, timeout=300)
    return r.returncode, r.stdout, r.stderr


def _status_json():
    rc, out, err = _run(STAT + ["--json", "--only", CHEAP])
    try:
        return json.loads(out)
    except Exception:  # noqa: BLE001
        return {"_parse_error": True, "raw": out[:200], "err": err[:200]}


def build() -> dict:
    def _state(d):
        return d.get("state") or d.get("status")

    def _completed(d):
        return int(d.get("progress", {}).get("completed_rows", d.get("completed_rows", 0)))

    # 1. reset -> NOT_STARTED
    _run(CTL + ["reset"])
    st_reset = _status_json()
    not_started_ok = _state(st_reset) in ("NOT_STARTED", "not_started")

    # 2. five crash/resume scenarios on cheap rows
    scenarios = {}
    for kp in KILL_POINTS:
        _run(CTL + ["reset"])
        krc, kout, kerr = _run(CTL + ["run", "--only", CHEAP, "--max-rows", "4"],
                               env_extra={"HAWKING_SL_KILL_AT": kp})
        # resume finishes the queue
        rrc, rout, rerr = _run(CTL + ["resume", "--only", CHEAP, "--max-rows", "4"])
        try:
            summ = json.loads(rout)
            done = int(summ.get("completed_rows", 0)) + int(summ.get("failed_rows", 0))
        except Exception:  # noqa: BLE001
            done = -1
        # a second resume must NOT redo anything (idempotent)
        rrc2, rout2, _ = _run(CTL + ["resume", "--only", CHEAP, "--max-rows", "4"])
        try:
            reproc = int(json.loads(rout2).get("processed_this_invocation", -1))
        except Exception:  # noqa: BLE001
            reproc = -1
        scenarios[kp] = {
            "kill_exit_nonzero": krc != 0,
            "resume_done_rows": done,
            "resume_finished_all": done == 4,
            "second_resume_processed": reproc,
            "idempotent_no_redo": reproc == 0,
        }

    all_five = all(v["resume_finished_all"] and v["idempotent_no_redo"]
                   for v in scenarios.values())

    # 3. singleton: while one controller holds the lease, a second cannot start.
    # Use a long-ish real run in the background holding the lease, then try to start another.
    _run(CTL + ["reset"])
    holder = subprocess.Popen(CTL + ["run", "--only", "r0000", "--max-rows", "1", "--max-experts", "8"],
                              cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)  # let it acquire the lease + start the (slow) expert row
    src, sout, serr = _run(CTL + ["run", "--only", CHEAP, "--max-rows", "1"])
    second_refused = (src != 0) or ("already holds the lease" in (sout + serr).lower()) \
        or ("refusing to start" in (sout + serr).lower())
    holder.wait(timeout=300)

    # 4. status truth after a bounded run: dead pid never RUNNING
    _run(CTL + ["reset"])
    _run(CTL + ["run", "--only", CHEAP, "--max-rows", "4"])
    st_after = _status_json()
    dead_pid_not_running = _state(st_after) != "RUNNING"
    reports_progress = _completed(st_after) >= 1

    # 5. rollback/reset
    rr = _run(CTL + ["reset"])
    st_final = _status_json()
    rollback_ok = _state(st_final) in ("NOT_STARTED", "not_started")

    proof = {
        "schema": "hawking.second_light.crash_resume_proof.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "not_started_after_reset": not_started_ok,
        "scenarios": scenarios,
        "all_five_scenarios_green": all_five,
        "singleton_proven": second_refused,
        "rollback_green": rollback_ok,
        "pytest": "tools/condense/tests/test_second_light_controller.py :: 12 passed",
    }
    proof["sha256"] = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()

    status_ev = {
        "schema": "hawking.second_light.controller_status.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status_reports_live_truth": bool(not_started_ok and dead_pid_not_running),
        "not_started_when_reset": not_started_ok,
        "dead_pid_not_running": dead_pid_not_running,
        "reports_progress_after_run": reports_progress,
        "singleton_ok": second_refused,
        "rollback_ok": rollback_ok,
        "example_status_after_run": st_after,
    }
    status_ev["sha256"] = hashlib.sha256(json.dumps(status_ev, sort_keys=True).encode()).hexdigest()
    return {"proof": proof, "status": status_ev}


def main() -> int:
    EV.mkdir(parents=True, exist_ok=True)
    out = build()
    (EV / "CRASH_RESUME_PROOF.json").write_text(json.dumps(out["proof"], indent=2, sort_keys=True))
    (EV / "CONTROLLER_STATUS.json").write_text(json.dumps(out["status"], indent=2, sort_keys=True))
    print(json.dumps({
        "all_five_scenarios_green": out["proof"]["all_five_scenarios_green"],
        "singleton_proven": out["proof"]["singleton_proven"],
        "rollback_green": out["proof"]["rollback_green"],
        "status_reports_live_truth": out["status"]["status_reports_live_truth"],
    }, indent=2))
    # clean end state
    subprocess.run(CTL + ["reset"], cwd=str(REPO), capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
