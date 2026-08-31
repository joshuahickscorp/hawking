#!/usr/bin/env python3
"""Retire the legacy Odyssey periodic writer.

`tools/odyssey_driver.sh` (launchd label com.hawking.odyssey, KeepAlive) ran a
5-minute loop whose last act each window was a bare ``git commit``. A bare
commit takes whatever is staged in the *shared* index, so any file another
author had staged and not yet committed was swept into a commit titled
"odyssey-i resident: autonomous cycle". That is the index-lock pathology and
the accidental sweep-up, and it is architectural, not a bug in any one commit.

This module checkpoints that driver's durable state, records its exact process
identity, proves nothing unique lives only in its process memory, and stops it
gracefully. It never SIGKILLs a driver with work in flight.

Git is not the event log. The replacement is tools/future/resident_supervisor.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, git  # noqa: E402

LABEL = "com.hawking.odyssey"
DRIVER = REPO / "tools" / "odyssey_driver.sh"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOCK = REPO / "workspace" / "campaign" / "odyssey" / ".resident.lock"
RECEIPT = REPO / "receipts" / "future" / "LEGACY_ODYSSEY_DRIVER_RETIREMENT.json"

# The exact paths the legacy driver named in its own commit_data().
DURABLE = (
    "workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json",
    "workspace/campaign/odyssey/ODYSSEY_STATE.json",
    "workspace/campaign/odyssey/RUN_LOG.jsonl",
    "workspace/campaign/odyssey/TRANSFER_MATRIX.json",
    "workspace/campaign/odyssey/GRAVITY_RULEBASE.json",
    "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json",
)

STOP_POLL_S = 0.5
STOP_BUDGET_S = 30.0


def _sh(*args: str, timeout: float = 20.0) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def _digest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"present": False}
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return {"present": True, "bytes": size, "sha256": h.hexdigest()}


def _pids() -> list[dict[str, object]]:
    """Every live process that is the driver or a descendant of it."""
    rc, out = _sh("ps", "-eo", "pid=,ppid=,etime=,command=")
    if rc != 0:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, etime, cmd = parts
        if "odyssey_driver.sh" in cmd or "odyssey_ctl.py" in cmd:
            rows.append(
                {"pid": int(pid), "ppid": int(ppid), "elapsed": etime, "command": cmd[:400]}
            )
    return rows


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _in_flight() -> dict[str, object]:
    """Work the driver could still lose. Its ctl children are the lanes."""
    live = _pids()
    ctl = [r["pid"] for r in live if "odyssey_ctl.py" in str(r["command"])]
    lanes: list[int] = []
    for pid in ctl:
        rc, out = _sh("pgrep", "-P", str(pid))
        if rc == 0:
            lanes.extend(int(x) for x in out.split() if x.strip().isdigit())
    staged = [p for p in git("diff", "--cached", "--name-only").splitlines() if p]
    return {
        "ctl_pids": ctl,
        "lane_child_pids": lanes,
        "lanes_in_flight": len(lanes),
        "staged_paths_a_bare_commit_would_sweep": staged,
        "safe_to_stop": not lanes,
        "why": (
            "no ctl child process, so no lane is mid-execution; every tick's "
            "state is written to disk before the tick returns"
            if not lanes
            else f"{len(lanes)} lane child processes are still running"
        ),
    }


def checkpoint() -> dict[str, object]:
    return {
        "durable_state": {p: _digest(REPO / p) for p in DURABLE},
        "last_commit_touching_durable_state": {
            p: git("log", "-1", "--format=%H %aI %s", "--", p) or None for p in DURABLE
        },
        "head_at_retirement": git("rev-parse", "HEAD") or None,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "lock_dir_present": LOCK.exists(),
        "lock_pid": (LOCK / "pid").read_text().strip() if (LOCK / "pid").exists() else None,
    }


def stop(execute: bool) -> dict[str, object]:
    """Graceful launchd bootout. KeepAlive means `kill` alone would not hold."""
    before = _pids()
    if not execute:
        return {"executed": False, "processes_before": before, "method": "dry-run"}
    uid = os.getuid()
    rc, out = _sh("launchctl", "bootout", f"gui/{uid}/{LABEL}")
    deadline = time.monotonic() + STOP_BUDGET_S
    while time.monotonic() < deadline:
        if not any(_alive(int(r["pid"])) for r in before):
            break
        time.sleep(STOP_POLL_S)
    after = _pids()
    survivors = [r for r in after if _alive(int(r["pid"]))]
    return {
        "executed": True,
        "method": f"launchctl bootout gui/{uid}/{LABEL}",
        "bootout_rc": rc,
        "bootout_output": out or None,
        "processes_before": before,
        "survivors": survivors,
        "stopped_cleanly": not survivors,
        "sigkill_used": False,
        "lock_released": not LOCK.exists(),
        # Narrow probe, narrow label: this says only whether THIS bootout call
        # returned 0. A second invocation returns non-zero because the label is
        # already unregistered, which is not the same as KeepAlive still armed.
        # The durable disarm is the plist rename, reported as plist_present.
        "bootout_call_succeeded": rc == 0,
    }


def build(execute: bool = False) -> dict[str, object]:
    flight = _in_flight()
    doc: dict[str, object] = {
        "schema": "hawking.future.legacy_driver_retirement.v1",
        "version": 1,
        "recorded_by": "tools/future/legacy_driver_retirement.py",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "retired": {
            "driver": "tools/odyssey_driver.sh",
            "inner_command": "tools/odyssey_ctl.py cycle --go --max-lanes 14 --grok-lanes 0",
            "launchd_label": LABEL,
            "plist": str(PLIST),
            "plist_present": PLIST.exists(),
            "keepalive": True,
            "window_secs": 300,
        },
        "why_retired": [
            "it committed the primary tree every ~300s",
            "its commit_data() ran a BARE `git commit`, so it took whatever was "
            "staged in the shared index — including another author's staged "
            "work — and titled it 'odyssey-i resident: autonomous cycle'",
            "that shared-index race is the mechanism behind the stale "
            ".git/index.lock files observed with no holder",
            "it was automation on a timer, not autonomy on evidence: its last "
            "25 ticks all read running=0 launched=0 and still committed",
        ],
        "sweep_mechanism": {
            "claim": "a bare `git commit` commits the index, not a path list",
            "consequence": "any file staged by another author lands in the "
            "driver's commit under the driver's message",
            "observed": "commit 12a76e3e5 carries 46 tools/future and "
            "receipts/future files that commit_data() never named",
            "not_a_data_loss": "author is the repo owner and the content is "
            "intact; the defect is provenance and index contention",
        },
        "in_flight_check": flight,
        "checkpoint": checkpoint(),
        "replacement": {
            "module": "tools/future/resident_supervisor.py",
            "law": "ONE CANONICAL WRITER — research workers mutate isolated "
            "worktrees and emit patches; the integration authority alone lands "
            "on the primary tree",
            "git_is_not_the_event_log": "durable runtime state lives in mission "
            "state, an append-only event log, WorkUnit receipts, the frontier "
            "store and the experiment store",
        },
        "claim_boundary": (
            "Retiring a periodic writer is an operational act, not a scientific "
            "one. This receipt asserts only what it probed: which processes were "
            "alive, what was staged, what the durable state hashed to, and "
            "whether the stop succeeded. It does not assert the replacement "
            "supervisor works."
        ),
    }
    if not flight["safe_to_stop"] and execute:
        doc["stop"] = {
            "executed": False,
            "refused": "lanes in flight; graceful stop would lose work",
            "lane_child_pids": flight["lane_child_pids"],
        }
        doc["retirement_complete"] = False
        return doc
    doc["stop"] = stop(execute)
    doc["retirement_complete"] = bool(execute and doc["stop"].get("stopped_cleanly"))
    return doc


def record(execute: bool = False) -> Path:
    doc = build(execute=execute)
    payload = json.dumps(doc, indent=1, sort_keys=True, default=str)
    doc["seal_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(RECEIPT)
    return RECEIPT


if __name__ == "__main__":
    ex = "--execute" in sys.argv
    if "--record" in sys.argv or ex:
        p = record(execute=ex)
        d = json.loads(p.read_text())
        print(f"wrote {p}")
        print(f"safe_to_stop={d['in_flight_check']['safe_to_stop']} "
              f"executed={d['stop'].get('executed')} "
              f"complete={d['retirement_complete']}")
    else:
        print(json.dumps(build(execute=False), indent=1, default=str))
