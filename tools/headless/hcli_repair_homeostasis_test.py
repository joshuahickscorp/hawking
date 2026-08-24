#!/usr/bin/env python3
"""Repair and retry homeostasis: the repair tree must not grow without bound.

The defect this defends against was observed, not imagined. With a Grok bridge
raising on every call, one failed unit produced

    grokfail.repair.1
    grokfail.repair.1.repair.1
    grokfail.repair.2
    grokfail.repair.1.repair.1.repair.1
    ...

and kept going. Isolation held -- independent units completed throughout -- but
a permanently unavailable backend manufactured work forever. The retry budget
bounds attempts of ONE unit; nothing bounded the DEPTH of repair-of-a-repair.

Two independent stops now exist, and this harness proves each separately:

* a **depth bound** (`MAX_REPAIR_DEPTH`), for a lineage whose failures keep
  changing
* **cycle detection** on the failure signature, for a lineage that keeps failing
  the same way -- which is the realistic case for a dead backend

Both terminate in an explicit durable `repair_exhausted` state carrying its root,
depth and reason, and an exhausted unit is never re-readied.

    python3 tools/headless/hcli_repair_homeostasis_test.py
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

import hcli.scheduler as scheduler_mod  # noqa: E402
from hcli.mission import Mission  # noqa: E402
from hcli.workunit import WorkUnit, is_ready  # noqa: E402

RESULTS: List[Dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


class AlwaysFails:
    """A backend that is permanently unavailable, the realistic bad case."""

    active = False

    def __init__(self, vary: bool = False) -> None:
        self.vary = vary
        self.n = 0

    def execute_workunit(self, wu: Any, context: Any) -> Dict[str, Any]:
        self.n += 1
        # `vary` makes every failure look different, which defeats cycle
        # detection and forces the DEPTH bound to be the thing that stops it.
        reason = f"injected failure {self.n}" if self.vary else "injected failure"
        raise RuntimeError(reason)


def _unit(uid: str) -> WorkUnit:
    return WorkUnit(id=uid, role="implement", description=f"unit {uid}")


def _run(engine: Any, units: Dict[str, WorkUnit], seconds: float = 25.0) -> Dict[str, WorkUnit]:
    with tempfile.TemporaryDirectory() as tmp:
        mission = Mission(
            tmp,
            engine=engine,
            units=units,
            goal="repair homeostasis",
            runtime_count=2,
            quiet=True,
            no_progress_threshold=50,
        )
        done = threading.Event()

        def watch() -> None:
            if not done.wait(seconds):
                try:
                    mission.cancel("harness deadline")
                except Exception:
                    pass

        threading.Thread(target=watch, daemon=True).start()
        try:
            mission.run()
        finally:
            done.set()
        return dict(mission.scheduler.units)


def check_dead_backend_is_bounded() -> None:
    units = _run(AlwaysFails(), {"dead": _unit("dead"), "fine": _unit("fine")})
    tree = [u for u in units if u.startswith("dead")]
    exhausted = [u for u in units.values() if getattr(u, "repair_exhausted", False)]
    deepest = max((int(getattr(u, "repair_depth", 0) or 0) for u in units.values()), default=0)
    check(
        "a permanently dead backend produces a BOUNDED repair tree",
        len(tree) <= 1 + scheduler_mod.MAX_REPAIR_DEPTH,
        f"{len(tree)} units in the lineage: {sorted(tree)}",
    )
    check(
        "the lineage terminates in an explicit exhausted state",
        bool(exhausted) and all(u.repair_reason for u in exhausted),
        f"exhausted={[(u.id, u.repair_reason) for u in exhausted]}",
    )
    check(
        "repair depth never exceeds the bound",
        deepest <= scheduler_mod.MAX_REPAIR_DEPTH,
        f"deepest={deepest} bound={scheduler_mod.MAX_REPAIR_DEPTH}",
    )


def check_depth_bound_when_failures_all_differ() -> None:
    """Cycle detection cannot fire here, so the depth bound must."""
    units = _run(AlwaysFails(vary=True), {"dead": _unit("dead")})
    deepest = max((int(getattr(u, "repair_depth", 0) or 0) for u in units.values()), default=0)
    exhausted = [u for u in units.values() if getattr(u, "repair_exhausted", False)]
    check(
        "with every failure distinct, the DEPTH bound stops the lineage",
        deepest <= scheduler_mod.MAX_REPAIR_DEPTH and bool(exhausted),
        f"deepest={deepest} bound={scheduler_mod.MAX_REPAIR_DEPTH} "
        f"exhausted={[u.id for u in exhausted]} units={len(units)}",
    )


def check_lineage_is_structured() -> None:
    units = _run(AlwaysFails(vary=True), {"dead": _unit("dead")})
    repairs = [u for u in units.values() if u.repairs]
    ok = bool(repairs) and all(
        u.repair_root == "dead" and int(u.repair_depth or 0) >= 1 for u in repairs
    )
    check(
        "repair lineage is structured, not parsed back out of the id string",
        ok,
        f"{[(u.id, u.repair_root, u.repair_depth) for u in repairs]}",
    )


def check_exhausted_is_not_rereadied() -> None:
    wu = _unit("spent")
    wu.status = "failed"
    wu.repair_exhausted = True
    wu.repair_reason = "budget spent"
    check(
        "an exhausted unit is never re-readied",
        is_ready(wu, {"spent": wu}) is False,
        "is_ready returned False as required",
    )


def check_lineage_survives_restart() -> None:
    from hcli.workunit import WorkUnit as WU

    wu = _unit("root")
    wu.repair_root = "root"
    wu.repair_depth = 2
    wu.repair_reason = "budget spent"
    wu.repair_exhausted = True
    back = WU.from_dict(wu.to_dict())
    check(
        "repair lineage survives serialization",
        back.repair_root == "root"
        and back.repair_depth == 2
        and back.repair_exhausted is True
        and back.repair_reason == "budget spent",
        f"root={back.repair_root} depth={back.repair_depth} "
        f"exhausted={back.repair_exhausted}",
    )


def check_negative_control() -> None:
    """Prove the bound is what stops the tree, not something incidental.

    Raise the depth bound and disable cycle detection, and the lineage must grow
    past where it previously stopped. A bound nobody has seen NOT hold is not a
    demonstrated bound.
    """
    original_depth = scheduler_mod.MAX_REPAIR_DEPTH
    original_count = scheduler_mod.MAX_REPAIRS_PER_ROOT
    original_sig = scheduler_mod.Scheduler._failure_signature
    # BOTH bounds have to be lifted. Lifting only the depth proves nothing once
    # the per-root count cap exists, because the count is then what stops the
    # tree -- which is exactly what the first version of this control missed.
    scheduler_mod.MAX_REPAIR_DEPTH = 12
    scheduler_mod.MAX_REPAIRS_PER_ROOT = 500
    counter = {"n": 0}

    def unique_signature(self, wu, context):  # noqa: ANN001
        counter["n"] += 1
        return f"unique-{counter['n']}"

    scheduler_mod.Scheduler._failure_signature = unique_signature
    try:
        units = _run(AlwaysFails(vary=True), {"dead": _unit("dead")}, seconds=30.0)
        deepest = max((int(getattr(u, "repair_depth", 0) or 0) for u in units.values()), default=0)
    finally:
        scheduler_mod.MAX_REPAIR_DEPTH = original_depth
        scheduler_mod.MAX_REPAIRS_PER_ROOT = original_count
        scheduler_mod.Scheduler._failure_signature = original_sig
    check(
        "NEGATIVE CONTROL: raising the bound really does let the lineage grow",
        deepest > original_depth,
        f"deepest={deepest} with both bounds lifted "
        f"(defaults: depth {original_depth}, per-root {original_count})",
    )


def check_per_root_count_cap() -> None:
    """Depth alone allows an exponential tree; the count cap makes it linear."""
    original_sig = scheduler_mod.Scheduler._failure_signature
    counter = {"n": 0}

    def unique_signature(self, wu, context):  # noqa: ANN001
        counter["n"] += 1
        return f"unique-{counter['n']}"

    scheduler_mod.Scheduler._failure_signature = unique_signature
    try:
        units = _run(AlwaysFails(vary=True), {"dead": _unit("dead")}, seconds=30.0)
    finally:
        scheduler_mod.Scheduler._failure_signature = original_sig
    lineage = [u for u in units if u.startswith("dead.repair")]
    check(
        "with cycle detection defeated, the per-root COUNT cap bounds the tree",
        len(lineage) <= scheduler_mod.MAX_REPAIRS_PER_ROOT,
        f"{len(lineage)} repairs for one root, cap is "
        f"{scheduler_mod.MAX_REPAIRS_PER_ROOT} (depth bound alone allowed "
        f"3+9+27=39 before this cap existed)",
    )


CHECKS = (
    check_dead_backend_is_bounded,
    check_depth_bound_when_failures_all_differ,
    check_lineage_is_structured,
    check_exhausted_is_not_rereadied,
    check_lineage_survives_restart,
    check_per_root_count_cap,
    check_negative_control,
)


def main() -> int:
    for fn in CHECKS:
        try:
            fn()
        except Exception as exc:
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
    failed = [r for r in RESULTS if not r["ok"]]
    out = REPO_ROOT / "receipts/headless/REPAIR_HOMEOSTASIS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gate": "REPAIR_HOMEOSTASIS",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "max_repair_depth": scheduler_mod.MAX_REPAIR_DEPTH,
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
