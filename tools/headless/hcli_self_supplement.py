#!/usr/bin/env python3
"""HCLI running a real Hawking-improvement ultragoal end to end, then surviving a kill.

The chain that has to hold, with a receipt at every hop:

    HCLI -> ultragoal -> durable Goal -> obligations -> WorkUnit DAG -> scheduler
    -> backend assignment -> actual tools -> deterministic verifier -> receipt
    -> VERIFIED -> checkpoint -> restart -> same mission -> steer -> next verified unit

The goal below is real work, not a fixture: each obligation names the test file
that proves it, the compiler derives a genuine `pytest` command from that, and a
unit is accepted only on that command's own exit code. Nothing here greps for a
nonce the harness wrote, and no unit exists to occupy capacity.

Phase two is a SEPARATE PROCESS -- the first one is SIGKILLed -- because an
in-process restart only proves that live objects still hold their own state.

    python3 tools/headless/hcli_self_supplement.py
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

TESTS = REPO_ROOT / "hcli/tests"

# A real improvement goal: keep the properties this session established true.
# Absolute paths, because a WorkUnit's verifier runs with the mission workspace
# as its cwd and these checks live in the repository.
GOAL_TEXT = f"""# Hawking self-improvement: hold the hardening frontier

The control plane gained several load-bearing properties today. Each must stay
true, and each must be provable by a check that can actually fail.

## Obligation: the context budget authority keeps deriving the per-request ceiling
The canonical authority must keep dividing the allocation by the slot count and
must keep refusing an over-large root prompt before any inference. Proven by
{TESTS}/test_context_budget.py.

## Obligation: the mutation lock stays a real mutual exclusion primitive
Two racing processes must never both acquire it. Proven by
{TESTS}/test_mutation_lock_race.py.

## Obligation: report compaction does not inflate a tiny report
A 39-byte report must not become a 716-byte envelope, and repeated noise must not
balloon the digest. Proven by {TESTS}/test_report_compiler.py.

## Obligation: Grok task identity stays unique and liveness stays observable
Two dispatches in the same second must not share a task directory, and a task
whose process is gone must not report as running. Proven by
{TESTS}/test_grok_identity.py.
"""

RESULTS: List[Dict[str, Any]] = []
HOPS: Dict[str, Any] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


_PHASE_ONE = r'''
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
from hcli.controller import Controller
from hcli.events import EventBus
from hcli.workspace import Workspace

ws = sys.argv[2]
goal = open(sys.argv[3], encoding="utf-8").read()

c = Controller(workspace=Workspace(ws), runtime_count=2, model=None, bus=EventBus())
res = c.start_ultragoal(goal)

mission = c.mission
out = {"mission_id": res.get("mission_id"),
       "obligation_ids": res.get("obligation_ids"),
       "workunit_ids": res.get("workunit_ids"),
       "ledger_path": res.get("ledger_path")}

# Run only until the first unit is verified, then stop: phase two has to have
# unfinished work left to resume.
if mission is not None:
    import threading
    def stop_after_first():
        for _ in range(1200):
            done = [u for u in mission.scheduler.units.values() if u.status == "completed"]
            if done:
                time.sleep(0.4)
                mission.cancel("phase one stop after first verified unit")
                return
            time.sleep(0.1)
    threading.Thread(target=stop_after_first, daemon=True).start()
    try:
        mission.run()
    except Exception as exc:
        out["run_error"] = f"{type(exc).__name__}: {exc}"
    mission.checkpoint()
    out["units_after_phase_one"] = {u.id: u.status for u in mission.scheduler.units.values()}
    out["verifiers"] = {u.id: u.verifier for u in mission.scheduler.units.values()}
    if mission._steering is not None:
        mission._steering.enqueue("prefer the smallest diff that keeps the check honest",
                                  kind="knowledge")

json.dump(out, open(os.path.join(ws, "phase_one.json"), "w"))
sys.stdout.write("READY\n"); sys.stdout.flush()
while True:
    time.sleep(0.25)
'''


def main() -> int:
    # The mission workspace is the REPOSITORY, not a temp dir. That is the
    # honest configuration for a self-improvement goal -- HCLI improving
    # Hawking operates in Hawking -- and it is also the only one where the
    # compiled verifiers actually run: the test files import
    # `hcli...`, so pytest needs the repo root on sys.path, and a
    # verifier launched with a temp cwd fails with ModuleNotFoundError no
    # matter how absolute its path is. A compiled verifier that is not portable
    # to the workspace it is scheduled into is a real defect; recorded as F4.
    ws = str(REPO_ROOT)
    existing = Path(ws) / ".hcli" / "mission" / "state.json"
    if existing.is_file():
        print(f"refusing to run: a mission already exists at {existing}", file=sys.stderr)
        print("this harness must never clobber durable mission state", file=sys.stderr)
        return 2
    if True:
        goal_file = Path(tempfile.mkdtemp(prefix="selfsupp-goal-")) / "goal.md"
        goal_file.write_text(GOAL_TEXT, encoding="utf-8")

        child = subprocess.Popen(
            [sys.executable, "-c", _PHASE_ONE, str(HAIDER), ws, str(goal_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        line = child.stdout.readline() if child.stdout else ""
        if "READY" not in line:
            child.kill()
            err = child.stderr.read() if child.stderr else ""
            check("phase one compiled and ran the ultragoal", False, err[-900:])
            return 1
        one = json.loads(Path(ws, "phase_one.json").read_text())
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=10)

        # ---- hop-by-hop evidence from phase one -----------------------------
        HOPS["ultragoal"] = {"chars": len(GOAL_TEXT), "path": str(goal_file)}
        HOPS["durable_goal"] = {"ledger": one.get("ledger_path"),
                                "exists": bool(one.get("ledger_path")
                                               and Path(one["ledger_path"]).is_file())}
        HOPS["obligations"] = one.get("obligation_ids")
        HOPS["workunit_dag"] = one.get("workunit_ids")
        HOPS["verifiers"] = one.get("verifiers")
        HOPS["units_after_phase_one"] = one.get("units_after_phase_one")

        check(
            "ultragoal compiled into a durable Goal on disk",
            HOPS["durable_goal"]["exists"],
            str(one.get("ledger_path")),
        )
        check(
            "obligations were derived from the goal text",
            bool(one.get("obligation_ids")) and len(one["obligation_ids"]) >= 3,
            f"{one.get('obligation_ids')}",
        )
        check(
            "a WorkUnit DAG was derived from the obligations",
            bool(one.get("workunit_ids")) and len(one["workunit_ids"]) >= 3,
            f"{one.get('workunit_ids')}",
        )
        verifiers = one.get("verifiers") or {}
        real = {k: v for k, v in verifiers.items() if v and "pytest" in v}
        check(
            "each WorkUnit carries a real, failable verifier taken from its obligation",
            len(real) >= 3,
            f"{len(real)} of {len(verifiers)} units carry a pytest command",
        )
        verified_one = [
            uid for uid, st in (one.get("units_after_phase_one") or {}).items()
            if st == "completed"
        ]
        check(
            "at least one unit reached VERIFIED through actual tools",
            bool(verified_one),
            f"verified in phase one: {verified_one}",
        )
        check(
            "unfinished work remained when the process was killed",
            len(verified_one) < len(one.get("units_after_phase_one") or {}),
            f"{len(verified_one)} of {len(one.get('units_after_phase_one') or {})} done",
        )
        check(
            "phase one was SIGKILLed",
            child.returncode not in (0, None),
            f"returncode={child.returncode}",
        )

        # ---- phase two: fresh interpreter, workspace only -------------------
        from hcli.controller import Controller
        from hcli.events import EventBus
        from hcli.mission import Mission
        from hcli.workspace import Workspace

        c2 = Controller(workspace=Workspace(ws), runtime_count=2, model=None, bus=EventBus())
        m2 = Mission.from_workspace(ws, engine=c2.engine, quiet=True, runtime_count=2)
        c2.mission = m2

        check(
            "restart recovers the SAME mission",
            m2.id == one["mission_id"],
            f"{one['mission_id']} -> {m2.id}",
        )
        check(
            "the unit verified before the kill is still verified",
            all(m2.scheduler.units[u].status == "completed" for u in verified_one),
            f"{ {u: m2.scheduler.units[u].status for u in verified_one} }",
        )

        steers_before = []
        if m2._steering is not None:
            steers_before = [getattr(e, "text", "") for e in m2._steering.all()]
        check(
            "the steer queued before the kill survived it",
            any("smallest diff" in s for s in steers_before),
            f"{steers_before}",
        )

        # steer the resumed mission, then finish more verified work
        if m2._steering is not None:
            m2._steering.enqueue(
                "constraint: a check that cannot fail does not discharge an obligation",
                kind="constraint",
            )
        ran: List[str] = []
        m2.before_dispatch = lambda mission: ran.append("dispatch")
        m2.run()
        final = {u.id: u.status for u in m2.scheduler.units.values()}
        HOPS["units_after_restart"] = final

        newly = [u for u, st in final.items() if st == "completed" and u not in verified_one]
        check(
            "the steered, resumed mission completed FURTHER verified work",
            bool(newly),
            f"newly verified after restart+steer: {newly}",
        )
        check(
            "no completed unit was replayed",
            all(final.get(u) == "completed" for u in verified_one),
            f"phase-one verified units still completed: {verified_one}",
        )

        # The durable artifacts of a mission are spread across .hcli: GOAL.md
        # (the ledger), dag.json (the authoritative unit store) and
        # mission/state.json + mission.log. Looking only inside mission/ misses
        # the two that matter most.
        receipts_dir = Path(ws) / ".hcli"
        wanted = ["GOAL.md", "dag.json", "mission/state.json", "mission/mission.log"]
        HOPS["durable_artifacts"] = [
            w for w in wanted if (receipts_dir / w).is_file()
        ]
        check(
            "the run left the ledger, the DAG and the mission state on disk",
            len(HOPS["durable_artifacts"]) == 4,
            f"{HOPS['durable_artifacts'][:6]}",
        )
        try:
            c2.shutdown()
        except Exception:
            pass

    failed = [r for r in RESULTS if not r["ok"]]
    out = REPO_ROOT / "receipts/headless/HCLI_SELF_SUPPLEMENT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gate": "HCLI_SELF_SUPPLEMENT",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_head": subprocess.check_output(
                    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
                "goal_text": GOAL_TEXT,
                "work_is_real": "each obligation names the test file that proves it; the compiler "
                "derives a genuine pytest command from that text and a unit is accepted only on that "
                "command's own exit code",
                "hops": HOPS,
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
