#!/usr/bin/env python3
"""Deterministic gates for the P0 defects found in the HCLI control plane.

Every gate here was watched FAILING against the tree it was written on. That is
the point: a gate nobody has seen fail is decoration. Each one is a real check
over real behaviour, not a restatement of an intention, and each names the
`file:line` it is defending.

    python3 tools/headless/hcli_p0_gates.py            # run all, print red/green
    python3 tools/headless/hcli_p0_gates.py --gate P0-1
    python3 tools/headless/hcli_p0_gates.py --json --out receipts/headless/P0_GATES.json

Exit 0 only when every selected gate passes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Verify commands that can never fail. A gate that accepted these would be
# defending nothing.
UNFAILABLE = (
    re.compile(r"SystemExit\(\s*0\s*\)"),
    re.compile(r"^\s*true\s*$"),
    re.compile(r"^\s*exit\s+0\s*$"),
    re.compile(r"^\s*:\s*$"),
)


def _controller(workspace: str):
    from hcli.controller import Controller
    from hcli.events import EventBus
    from hcli.workspace import Workspace

    return Controller(
        workspace=Workspace(workspace), runtime_count=1, model=None, bus=EventBus()
    )


def _directive_text() -> str:
    p = Path.home() / ".claude/ultragoal/hawking-headless-v3/ultragoal-directive.md"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    # Fall back to something large and real rather than skipping the gate.
    return (REPO_ROOT / "hcli" / "mission.py").read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------- gates


def gate_p0_1() -> Dict[str, Any]:
    """No compiled obligation may carry a verify command that cannot fail.

    Defends: controller.py:580 `start_ultragoal`, which today writes
    `python3 -c 'raise SystemExit(0)'` as the verify command for every
    obligation it compiles.
    """
    with tempfile.TemporaryDirectory(prefix="p0-1-") as t:
        c = _controller(t)
        try:
            c.start_ultragoal(_directive_text())
            goal_md = (Path(t) / ".hcli" / "GOAL.md").read_text(encoding="utf-8")
        finally:
            c.shutdown()

    verifies = [
        line.strip()[len("verify:") :].strip()
        for line in goal_md.splitlines()
        if line.strip().startswith("verify:")
    ]
    bad = [v for v in verifies if any(rx.search(v) for rx in UNFAILABLE)]
    return {
        "gate": "P0-1",
        "claim": "every compiled obligation carries a verify command that is capable of failing",
        "defends": "tools/haider/hcli/controller.py:580 start_ultragoal",
        "obligations": len(verifies),
        "unfailable": len(bad),
        "distinct_unfailable": sorted(set(bad))[:5],
        "ok": len(verifies) > 0 and not bad,
    }


def gate_p0_12() -> Dict[str, Any]:
    """A many-obligation goal must decompose into more than two WorkUnits.

    Defends: the DAG must be derived from the obligations. Today 57 obligations
    compile to exactly `implement` and `validate`, so nothing can be scheduled
    or verified per obligation.
    """
    with tempfile.TemporaryDirectory(prefix="p0-12-") as t:
        c = _controller(t)
        try:
            res = c.start_ultragoal(_directive_text())
            dag = json.loads((Path(t) / ".hcli" / "dag.json").read_text(encoding="utf-8"))
            goal_md = (Path(t) / ".hcli" / "GOAL.md").read_text(encoding="utf-8")
        finally:
            c.shutdown()

    units = list((dag.get("units") or {}).keys())
    obligations = len([l for l in goal_md.splitlines() if l.strip().startswith("- [")])
    return {
        "gate": "P0-12",
        "claim": "the WorkUnit DAG is derived from the obligations, not a fixed two-step template",
        "defends": "tools/haider/hcli/controller.py start_ultragoal -> dag.json",
        "obligations": obligations,
        "workunits": len(units),
        "workunit_ids": units[:8],
        "returned_workunit_ids": (res or {}).get("workunit_ids"),
        # Not "one unit per obligation" — that would over-specify the design.
        # The bar is that the count responds to the goal at all.
        "ok": obligations > 2 and len(units) > 2,
    }


def gate_p0_3() -> Dict[str, Any]:
    """A Grok task in a failed terminal state may never be accepted.

    Behavioural, not a source grep. An earlier version of this gate looked for
    the word "state" near "fail" in executors.py and reported RED against a
    working guard -- a check that reads code instead of running it can only
    ever measure phrasing.
    """
    from hcli.executors import WorkUnitExecutor

    class _Handle:
        task_id = "p0-3-probe"

    class _Bridge:
        def __init__(self, state):
            self._state = state

        def consult(self, *a, **k):
            return _Handle()

        def wait(self, task_id, timeout=0):
            return {"state": self._state, "exit_code": 1, "task_id": task_id}

        def compact_report(self, task_id):
            return {}

    class _WU:
        id = "p0-3"
        role = "implement"
        description = "probe"
        # A verifier that passes for reasons unrelated to the Grok task.
        verifier = "python3 -c \"raise SystemExit(0)\""
        backend_task_id = None
        verification = None

    out = {}
    with tempfile.TemporaryDirectory(prefix="p0-3-") as t:
        for state in ("failed", "done"):
            ex = WorkUnitExecutor(t, engine=None)
            ex.grok_bridge = lambda s=state: _Bridge(s)  # type: ignore[assignment]
            res = ex._run_grok(_WU(), {"prompt": "probe"})
            validation = res.get("validation") or {}
            out[state] = {
                "ok": bool(validation.get("ok")),
                "reason": validation.get("reason"),
            }

    return {
        "gate": "P0-3",
        "claim": "a failed Grok terminal state blocks acceptance regardless of the verifier",
        "defends": "tools/haider/hcli/executors.py WorkUnitExecutor._run_grok",
        "failed_state": out.get("failed"),
        "done_state": out.get("done"),
        # The failed state must not be accepted. The done state is allowed to be
        # rejected too (its verifier is vacuous), so only the failed direction
        # is asserted here -- test_p0_closeout covers the positive direction.
        "ok": out.get("failed", {}).get("ok") is False,
    }


def gate_p0_4() -> Dict[str, Any]:
    """An executor-run verifier that cannot fail must not accept a WorkUnit.

    `Ledger.run_verify` already rejects vacuous commands (SystemExit(0), bare
    `true`, `exit 0`, `:`) -- but the CPU/Grok executor path runs the verifier
    through `subprocess(..., shell=True)` and accepts exit 0, so the same
    unfailable command that the ledger refuses is accepted here. Found while
    proving P0-3: a Grok task in state `done` was accepted on the strength of
    `python3 -c "raise SystemExit(0)"`.
    """
    from hcli.executors import WorkUnitExecutor

    class _WU:
        id = "p0-4"
        role = "implement"
        description = "probe"
        verifier = "python3 -c \"raise SystemExit(0)\""
        backend_task_id = None
        verification = None

    with tempfile.TemporaryDirectory(prefix="p0-4-") as t:
        ex = WorkUnitExecutor(t, engine=None)
        res = ex._run_cpu(_WU(), {"prompt": "probe"})
    validation = res.get("validation") or {}
    return {
        "gate": "P0-4",
        "claim": "an executor-run verifier that cannot fail is rejected, not accepted",
        "defends": "tools/haider/hcli/executors.py WorkUnitExecutor._run_cpu",
        "validation": {"ok": bool(validation.get("ok")), "reason": validation.get("reason")},
        "ok": validation.get("ok") is not True,
    }


def gate_p0_6() -> Dict[str, Any]:
    """Two Grok dispatches inside the same second must get distinct task ids.

    Defends: grok-run derives its task id by appending a one-second-resolution
    timestamp to the slug HCLI supplies. HCLI used the literal slug "consult"
    for every consult, so two dispatches in the same second landed on one task
    directory (`consult-20260822-224557`) and two WorkUnits were accepted off a
    single Grok execution.

    grok-run is outside this repo's mutation scope, so the check is on what
    HCLI actually hands it. Capture the slug at the launch boundary rather than
    looking for a particular helper name -- an earlier version of this gate
    probed for a `_task_id` attribute and reported RED against a working fix.
    """
    from hcli.grok_bridge import GrokBridge

    with tempfile.TemporaryDirectory(prefix="p0-6-") as t:
        b = GrokBridge(t)
        seen: List[str] = []

        def capture(**kw):
            seen.append(str(kw.get("task")))
            raise RuntimeError("captured")

        b._launch = capture  # type: ignore[method-assign]
        for _ in range(50):
            try:
                b.consult("probe prompt")
            except RuntimeError:
                pass

    return {
        "gate": "P0-6",
        "claim": "HCLI hands grok-run a distinct task slug per dispatch",
        "defends": "tools/haider/hcli/grok_bridge.py consult() -> --task slug",
        "dispatches": len(seen),
        "distinct_slugs": len(set(seen)),
        "example": seen[0] if seen else None,
        "ok": bool(seen) and len(set(seen)) == len(seen),
    }


def gate_p0_11() -> Dict[str, Any]:
    """A failing receipt must record the evidence that was actually inlined.

    Defends: engine.py:788 wipes evidence on the exception path, which is why
    the 887c receipt claims `evidence_files: []` while five files were in fact
    inlined into the 23532-token prompt that failed.
    """
    src = (REPO_ROOT / "tools/haider/hcli/engine.py").read_text(encoding="utf-8")
    # The real shape is `evidence=[]` passed into _write_receipt on the error
    # path. An earlier version of this gate looked for `evidence_files: []` and
    # went green while the defect was sitting there — match the argument, not
    # the receipt field it eventually becomes.
    wipes = []
    for i, line in enumerate(src.splitlines(), 1):
        if re.search(r"^\s*evidence\s*=\s*\[\s*\]\s*,?\s*$", line):
            wipes.append(i)
    return {
        "gate": "P0-11",
        "claim": "the error path preserves the evidence list instead of emitting an empty one",
        "defends": "tools/haider/hcli/engine.py:788",
        "empty_evidence_arguments_at_lines": wipes,
        "ok": not wipes,
    }


def gate_p0_2() -> Dict[str, Any]:
    """The mutation lock must be a real mutual exclusion primitive.

    Defends: resources.py:481-505, an advisory JSON check-then-`os.replace`.
    Measured elsewhere at 79/80 double-acquire under an undelayed race. This
    gate is the cheap structural check; the race itself is the expensive one.
    """
    src = (REPO_ROOT / "tools/haider/hcli/resources.py").read_text(encoding="utf-8")
    real = bool(re.search(r"\bflock\b|O_EXCL|O_CREAT\s*\|\s*os\.O_EXCL", src))
    return {
        "gate": "P0-2",
        "claim": "the mutation lock uses flock or O_EXCL, not check-then-replace",
        "defends": "tools/haider/hcli/resources.py:481-505",
        "uses_real_primitive": real,
        "ok": real,
    }


def gate_p0_8() -> Dict[str, Any]:
    """Scheduler.complete must consult the verifier outcome.

    Defends: scheduler.py:94-102, which transitions a WorkUnit to `completed`
    without ever reading `wu.verifier`.
    """
    src = (REPO_ROOT / "tools/haider/hcli/scheduler.py").read_text(encoding="utf-8")
    m = re.search(r"def complete\(.*?\n(?:.*?\n){0,30}?\n    def ", src, re.DOTALL)
    body = m.group(0) if m else src
    consults = bool(re.search(r"verifier|verified|verdict", body))
    return {
        "gate": "P0-8",
        "claim": "Scheduler.complete consults a verifier outcome before completing a WorkUnit",
        "defends": "tools/haider/hcli/scheduler.py:94-102",
        "body_mentions_verifier": consults,
        "ok": consults,
    }


def gate_p0_7() -> Dict[str, Any]:
    """/status must not report Grok activity it cannot observe.

    Defends: max_policy.grok_pool_snapshot, which counts WorkUnit flags. This
    gate compares its `active` against the real process table.
    """
    from hcli.max_policy import grok_pool_snapshot

    snap = grok_pool_snapshot(str(REPO_ROOT))
    # Count grok-run wrappers specifically. An earlier version of this gate
    # matched '[g]rok-1.0', which returned 0 while four lanes were demonstrably
    # running -- a gate whose own observation was wrong.
    live = subprocess.run(
        ["sh", "-c", "ps -axo command | grep -c '[g]rok-run \\(delegate\\|audit\\|consult\\)'"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    live_n = int(live) if live.isdigit() else 0
    reported = int(snap.get("active") or 0)
    return {
        "gate": "P0-7",
        "claim": "reported Grok activity tracks the real process table",
        "defends": "tools/haider/hcli/max_policy.py grok_pool_snapshot",
        "reported_active": reported,
        "live_grok_processes": live_n,
        # Only meaningful while lanes are actually running; if nothing is live
        # the gate cannot distinguish truth from luck and says so.
        "conclusive": live_n > 0,
        # Inconclusive is not a pass. Reporting ok=True when nothing was
        # running let this gate go green while proving nothing -- the same
        # failure mode it exists to catch.
        "ok": (reported > 0) if live_n > 0 else False,
        "inconclusive": live_n == 0,
        "note": None if live_n > 0 else "no grok processes live; gate is INCONCLUSIVE, not passed",
    }


GATES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "P0-1": gate_p0_1,
    "P0-2": gate_p0_2,
    "P0-3": gate_p0_3,
    "P0-4": gate_p0_4,
    "P0-6": gate_p0_6,
    "P0-7": gate_p0_7,
    "P0-8": gate_p0_8,
    "P0-11": gate_p0_11,
    "P0-12": gate_p0_12,
}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", action="append", help="run only these gates")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    names = args.gate or list(GATES)
    results = []
    for n in names:
        fn = GATES.get(n)
        if fn is None:
            results.append({"gate": n, "ok": False, "error": "unknown gate"})
            continue
        try:
            results.append(fn())
        except Exception as exc:
            results.append(
                {"gate": n, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )

    report = {
        "suite": "hcli_p0_gates",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "results": results,
        "green": [r["gate"] for r in results if r.get("ok")],
        "red": [r["gate"] for r in results if not r.get("ok") and not r.get("inconclusive")],
        "inconclusive": [r["gate"] for r in results if r.get("inconclusive")],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in results:
            mark = "GREEN" if r.get("ok") else ("INCON" if r.get("inconclusive") else "RED  ")
            extra = r.get("error") or r.get("reason") or r.get("note") or ""
            print(f"{mark} {r['gate']:<6} {r.get('claim','')[:78]} {extra}")
        print(
            f"\ngreen={len(report['green'])} red={len(report['red'])} "
            f"inconclusive={len(report['inconclusive'])}"
        )

    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"receipt: {p}")

    return 0 if not report["red"] and not report["inconclusive"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
