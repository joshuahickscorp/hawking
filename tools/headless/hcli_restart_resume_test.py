#!/usr/bin/env python3
"""Restart, resume and durable steering across a real process boundary.

An in-process "restart" that just rebuilds objects proves nothing about
durability: the state under test is on disk, and the thing that must survive is
the loss of the process holding it. So phase one runs in a CHILD process which
is SIGKILLed mid-mission -- no atexit, no flush, no graceful shutdown -- and
phase two is a separate interpreter that only ever sees the workspace.

What must hold across that boundary:

* the mission id and every WorkUnit id are unchanged
* completed units are not replayed
* a unit left `running` by the killed process is recovered, not lost
* a mutation lock held by the dead process is recoverable, and one held by a
  live process is not
* a steer queued before the kill is still there afterwards
* the resumed mission can still complete further work

    python3 tools/headless/hcli_restart_resume_test.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
HAIDER = REPO_ROOT

RESULTS: List[Dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


# Phase one runs in a child so it can be killed outright.
_PHASE_ONE = r"""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
from hcli.mission import Mission
from hcli.workunit import WorkUnit, transition_status
from hcli.resources import MutationLock

ws = sys.argv[2]

class Engine:
    active = False
    def execute_workunit(self, wu, context):
        return {"kind": "answer", "content": "did " + wu.id,
                "validation": {"ok": True, "verifier": "phase-one-engine"}}

units = {
    "done1": WorkUnit(id="done1", role="implement", description="already finished"),
    "stuck": WorkUnit(id="stuck", role="implement", description="left running by the kill"),
    "later": WorkUnit(id="later", role="implement", description="never started",
                      dependencies=["done1"]),
}
m = Mission(ws, engine=Engine(), units=units, goal="restart durability",
            mission_id="restart-fixed-id", quiet=True, no_progress_threshold=50)

# One unit genuinely completed, through the verifier gate.
wu = m.scheduler.units["done1"]
transition_status(wu, "ready"); transition_status(wu, "running")
m.scheduler.complete("done1", verification={"ok": True, "verifier": "phase-one"})

# One unit left mid-flight, exactly as a crash would leave it.
stuck = m.scheduler.units["stuck"]
transition_status(stuck, "ready"); transition_status(stuck, "running")
stuck.backend_task_id = "external-grok-task-1"
stuck.assigned_backend = "grok"

# A mutation lock this process holds and will never release.
lock = MutationLock(ws)
lock.acquire("stuck")

m.checkpoint()
if m._steering is not None:
    m._steering.enqueue("prefer the smaller diff", kind="knowledge")

json.dump({"mission_id": m.id,
           "units": {u.id: u.status for u in m.scheduler.units.values()},
           "lock_pid": os.getpid()},
          open(os.path.join(ws, "phase_one.json"), "w"))
sys.stdout.write("READY\n"); sys.stdout.flush()
while True:
    time.sleep(0.25)
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as ws:
        # ---- phase one: build durable state, then die hard -------------------
        child = subprocess.Popen(
            [sys.executable, "-c", _PHASE_ONE, str(HAIDER), ws],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        line = child.stdout.readline() if child.stdout else ""
        if "READY" not in line:
            child.kill()
            err = child.stderr.read() if child.stderr else ""
            check("phase one reached a checkpoint", False, err[-600:])
            return 1
        before = json.loads(Path(ws, "phase_one.json").read_text())
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=10)
        check(
            "phase one was SIGKILLed with state on disk",
            child.returncode not in (0, None),
            f"returncode={child.returncode} mission={before['mission_id']}",
        )

        # The lock holder is now a dead pid. That is the state a real crash leaves.
        from hcli.resources import MutationLock

        dead_pid = before["lock_pid"]
        lock = MutationLock(ws)
        rec = lock.read()
        check(
            "the killed process left its mutation lock behind",
            bool(rec) and int(rec.get("pid", -1)) == dead_pid,
            f"record={rec}",
        )
        check(
            "a lock held by a dead pid is recoverable",
            lock.acquire("resumed-unit") is True,
            "acquire after crash",
        )
        lock.release("resumed-unit")

        # ---- phase two: a fresh interpreter that only sees the workspace ------
        from hcli.mission import Mission

        class Engine:
            active = False

            def execute_workunit(self, wu, context):  # noqa: ANN001
                return {
                    "kind": "answer",
                    "content": f"did {wu.id}",
                    "validation": {"ok": True, "verifier": "phase-two-engine"},
                }

        engine = Engine()
        m2 = Mission.from_workspace(ws, engine=engine, quiet=True, runtime_count=2)

        check(
            "mission id survives the kill",
            m2.id == before["mission_id"],
            f"{before['mission_id']} -> {m2.id}",
        )
        after_ids = sorted(m2.scheduler.units)
        check(
            "every WorkUnit id survives the kill",
            after_ids == sorted(before["units"]),
            f"{sorted(before['units'])} -> {after_ids}",
        )
        check(
            "the completed unit is still completed",
            m2.scheduler.units["done1"].status == "completed",
            f"done1={m2.scheduler.units['done1'].status}",
        )
        stuck_after = m2.scheduler.units["stuck"].status
        check(
            "a unit left running by a SIGKILL is recovered as INTERRUPTED",
            # Tightened, not loosened. This used to accept "failed" among four
            # statuses, which let a crash look exactly like a verifier failure
            # and quietly consume one of the unit's retries. A killed process
            # did not fail its verifier. The rule now is terminal-grok ->
            # failed; stale-running, unobservable, or a non-grok crash ->
            # interrupted and still retryable, so this asserts the specific
            # status rather than a set that includes the wrong one.
            stuck_after == "interrupted",
            f"stuck={stuck_after} (external task id "
            f"{m2.scheduler.units['stuck'].backend_task_id!r} preserved: "
            f"{m2.scheduler.units['stuck'].backend_task_id == 'external-grok-task-1'})",
        )
        check(
            "the external backend task id survives the kill",
            m2.scheduler.units["stuck"].backend_task_id == "external-grok-task-1",
            f"{m2.scheduler.units['stuck'].backend_task_id!r}",
        )

        steers = []
        if m2._steering is not None:
            try:
                steers = [getattr(e, "text", "") for e in m2._steering.all()]
            except Exception as exc:
                steers = [f"<error {exc}>"]
        check(
            "a steer queued before the kill survives it",
            any("smaller diff" in s for s in steers),
            f"steers={steers}",
        )

        # ---- continue: further verified work, and no replay ------------------
        ran: List[str] = []
        original = engine.execute_workunit

        def recording(wu, context):  # noqa: ANN001
            ran.append(wu.id)
            return original(wu, context)

        engine.execute_workunit = recording  # type: ignore[assignment]

        # Steering must reach FUTURE work and leave finished work alone.
        if m2._steering is not None:
            m2._steering.enqueue(
                "constraint: every diff must name the file it repairs", kind="constraint"
            )
        future_ctx = m2._unit_context(m2.scheduler.units["later"])
        future_prompt = future_ctx.get("prompt") if isinstance(future_ctx, dict) else str(future_ctx)
        check(
            "a constraint steer reaches a not-yet-run WorkUnit's compiled context",
            "name the file it repairs" in (future_prompt or ""),
            f"steering section present={'STEERING' in (future_prompt or '')}",
        )
        done_before = dict(m2.scheduler.units["done1"].to_dict())

        m2.run()

        done_after = dict(m2.scheduler.units["done1"].to_dict())
        drift = [k for k in done_before if done_before[k] != done_after.get(k)]
        check(
            "steering does not rewrite already-verified history",
            done_before.get("status") == done_after.get("status") == "completed"
            and not [k for k in drift if k not in ("ready_at", "running_at", "finished_at")],
            f"changed fields on the completed unit: {drift}",
        )
        final = {u.id: u.status for u in m2.scheduler.units.values()}
        check(
            "the completed unit was NOT replayed after restart",
            "done1" not in ran,
            f"units executed after restart: {ran}",
        )
        check(
            "the resumed mission completes further verified work",
            final.get("later") == "completed",
            f"final={final}",
        )

    failed = [r for r in RESULTS if not r["ok"]]
    out = REPO_ROOT / "receipts/headless/HCLI_RESTART_RESUME.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gates": ["HCLI_RESTART_RESUME", "AGENTOS_STEERING_DURABLE"],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "method": "phase one runs in a child process and is SIGKILLed mid-mission; phase two "
                "is a separate interpreter that only sees the workspace",
                "results": RESULTS,
                "failed": [r["name"] for r in failed],
                "result": "PASS" if not failed else "FAIL",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    print(f"receipt: {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
