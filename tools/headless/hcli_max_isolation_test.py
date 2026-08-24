#!/usr/bin/env python3
"""Dependency ordering and backend failure isolation under concurrent MAX.

Two properties the scheduler must hold when several WorkUnits are in flight:

* **Ordering.** A dependent unit never starts before the unit it depends on has
  been accepted. Not "usually" -- the check records the actual start and finish
  instants of every unit and compares them, so an ordering that only holds
  because the machine happened to be slow would fail here.
* **Isolation.** One backend failing does not stop the others. A Grok bridge
  that raises, a CPU verifier that exits non-zero and an engine that throws must
  each fail their own unit and leave every independent unit free to complete.

Plain python3 + assert, and also importable under pytest.

    python3 tools/headless/hcli_max_isolation_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

from hcli.mission import Mission  # noqa: E402
from hcli.workunit import WorkUnit  # noqa: E402

RESULTS: List[Dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


class RecordingEngine:
    """Records when each unit started and finished, and can be told to fail."""

    def __init__(self, fail_ids: Optional[set] = None, delay: float = 0.05) -> None:
        self.fail_ids = set(fail_ids or ())
        self.delay = delay
        self.lock = threading.Lock()
        self.started: Dict[str, float] = {}
        self.finished: Dict[str, float] = {}
        self.order: List[str] = []
        self.active = False

    def execute_workunit(self, wu: Any, context: Any) -> Dict[str, Any]:
        with self.lock:
            self.started[wu.id] = time.monotonic()
            self.order.append(wu.id)
        time.sleep(self.delay)
        with self.lock:
            self.finished[wu.id] = time.monotonic()
        if wu.id in self.fail_ids:
            raise RuntimeError(f"injected engine failure for {wu.id}")
        return {
            "kind": "answer",
            "content": f"did {wu.id}",
            "validation": {"ok": True, "verifier": "recording-engine"},
        }


def run_with_deadline(mission: Mission, seconds: float = 25.0) -> None:
    """Run a mission but never let a check hang the suite.

    A failing unit spawns a repair unit, which can fail in turn, so a mission
    built around an injected failure does not necessarily settle on its own.
    The deadline is the harness protecting itself, not a weakened assertion:
    every check below asserts on the unit statuses after the mission stops,
    and a unit that never completed is a failure either way.
    """
    done = threading.Event()

    def _watch() -> None:
        if not done.wait(seconds):
            try:
                mission.cancel("harness deadline")
            except Exception:
                pass

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    try:
        mission.run()
    finally:
        done.set()


def _wu(uid: str, deps: Optional[List[str]] = None, **kw: Any) -> WorkUnit:
    return WorkUnit(
        id=uid,
        role=kw.pop("role", "implement"),
        description=kw.pop("description", f"unit {uid}"),
        dependencies=list(deps or []),
        **kw,
    )


def check_dependency_ordering() -> None:
    """A -> B -> C, plus three independent units, all under concurrency."""
    units = {
        "A": _wu("A"),
        "B": _wu("B", ["A"]),
        "C": _wu("C", ["B"]),
        "D": _wu("D"),
        "E": _wu("E"),
        "F": _wu("F"),
    }
    engine = RecordingEngine(delay=0.08)
    with tempfile.TemporaryDirectory() as tmp:
        mission = Mission(
            tmp,
            engine=engine,
            units=units,
            goal="dependency ordering under concurrency",
            runtime_count=4,
            quiet=True,
            no_progress_threshold=25,
        )
        run_with_deadline(mission)
        statuses = {u.id: u.status for u in mission.scheduler.units.values()}

    violations = []
    for uid, deps in (("B", ["A"]), ("C", ["B"])):
        for dep in deps:
            s = engine.started.get(uid)
            f = engine.finished.get(dep)
            if s is None or f is None:
                violations.append(f"{uid} or {dep} never ran")
            elif s < f:
                violations.append(f"{uid} started {f - s:.3f}s BEFORE {dep} finished")
    check(
        "dependency ordering respected under concurrency",
        not violations and statuses.get("C") == "completed",
        f"violations={violations} statuses={statuses} order={engine.order}",
    )
    # An ordering check that never had a chance to overlap proves nothing.
    concurrent = _max_overlap(engine)
    check(
        "the run was actually concurrent (otherwise ordering is vacuous)",
        concurrent >= 2,
        f"max simultaneous units observed={concurrent}",
    )


def _max_overlap(engine: RecordingEngine) -> int:
    events = []
    for uid, s in engine.started.items():
        f = engine.finished.get(uid)
        if f is None:
            continue
        events.append((s, 1))
        events.append((f, -1))
    events.sort()
    cur = best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best


def check_failure_isolation() -> None:
    """One failing unit must not stop the independent ones."""
    units = {
        "boom": _wu("boom"),
        "ok1": _wu("ok1"),
        "ok2": _wu("ok2"),
        "ok3": _wu("ok3"),
    }
    engine = RecordingEngine(fail_ids={"boom"}, delay=0.05)
    with tempfile.TemporaryDirectory() as tmp:
        mission = Mission(
            tmp,
            engine=engine,
            units=units,
            goal="failure isolation",
            runtime_count=4,
            quiet=True,
            no_progress_threshold=25,
        )
        run_with_deadline(mission)
        statuses = {u.id: u.status for u in mission.scheduler.units.values()}

    survivors = [u for u in ("ok1", "ok2", "ok3") if statuses.get(u) == "completed"]
    check(
        "an injected engine failure does not stop independent units",
        len(survivors) == 3 and statuses.get("boom") != "completed",
        f"statuses={statuses}",
    )


def check_cpu_verifier_failure_isolation() -> None:
    """A CPU verifier exiting non-zero fails its unit and nothing else."""
    units = {
        "badver": _wu(
            "badver",
            preferred_backend="cpu",
            resource_class="TEST",
            verifier="python3 -c \"import sys; sys.exit(7)\"",
        ),
        "goodver": _wu(
            "goodver",
            preferred_backend="cpu",
            resource_class="TEST",
            verifier="python3 -c \"import sys; sys.exit(0 if 1+1==2 else 1)\"",
        ),
        "engine_ok": _wu("engine_ok"),
    }
    engine = RecordingEngine(delay=0.02)
    with tempfile.TemporaryDirectory() as tmp:
        mission = Mission(
            tmp,
            engine=engine,
            units=units,
            goal="cpu verifier isolation",
            runtime_count=3,
            quiet=True,
            no_progress_threshold=25,
        )
        run_with_deadline(mission)
        statuses = {u.id: u.status for u in mission.scheduler.units.values()}
    check(
        "a failing CPU verifier fails only its own unit",
        statuses.get("badver") != "completed"
        and statuses.get("goodver") == "completed"
        and statuses.get("engine_ok") == "completed",
        f"statuses={statuses}",
    )


def check_grok_bridge_failure_isolation() -> None:
    """A Grok bridge that raises must not take the mission down."""
    import hcli.executors as executors

    original = executors.WorkUnitExecutor.grok_bridge

    def exploding(self):  # noqa: ANN001
        raise RuntimeError("injected grok-run outage")

    units = {
        "grokfail": _wu("grokfail", preferred_backend="grok", resource_class="GROK"),
        "ok1": _wu("ok1"),
        "ok2": _wu("ok2"),
    }
    engine = RecordingEngine(delay=0.02)
    executors.WorkUnitExecutor.grok_bridge = exploding
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mission = Mission(
                tmp,
                engine=engine,
                units=units,
                goal="grok outage isolation",
                runtime_count=3,
                quiet=True,
                no_progress_threshold=25,
            )
            run_with_deadline(mission)
            statuses = {u.id: u.status for u in mission.scheduler.units.values()}
    finally:
        executors.WorkUnitExecutor.grok_bridge = original
    check(
        "a Grok backend outage fails only its own unit",
        statuses.get("grokfail") != "completed"
        and statuses.get("ok1") == "completed"
        and statuses.get("ok2") == "completed",
        f"statuses={statuses}",
    )


CHECKS = (
    ("dependency ordering", check_dependency_ordering),
    ("engine failure isolation", check_failure_isolation),
    ("cpu verifier failure isolation", check_cpu_verifier_failure_isolation),
    ("grok outage isolation", check_grok_bridge_failure_isolation),
)


def main() -> int:
    for _name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # a raising check is a failing check
            check(_name, False, f"{type(exc).__name__}: {exc}")
    failed = [r for r in RESULTS if not r["ok"]]
    out = REPO_ROOT / "receipts/headless/MAX_DEPENDENCY_AND_ISOLATION.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gates": ["MAX_DEPENDENCY_SCHEDULING", "BACKEND_FAILURE_ISOLATION"],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
