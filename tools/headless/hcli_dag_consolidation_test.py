#!/usr/bin/env python3
"""Protected DAG-consolidation + governor-gap checks. Plain python3 + assert.

No GPU, no model, no network.

Run:
    python3 tools/headless/hcli_dag_consolidation_test.py
    pytest tools/headless/hcli_dag_consolidation_test.py -q

Each of the six S006 checks was watched FAILING against the pre-consolidation
code (WorkUnitDAG's naive readiness / direct status writes for 1-4; ledger
governor gaps for 5-6). Failure text is recorded in
receipts/headless/DAG_CONSOLIDATION_DECISION.json.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

# Scheduler.complete now refuses to complete a WorkUnit without a passing
# deterministic verifier outcome. These checks are about scheduling, resource
# classes and durability -- not about completing unverified work -- so they
# supply a passing outcome rather than dropping the new gate.
PASSING_VERIFICATION = {"ok": True, "verifier": "headless-test-fixture"}

REPO = Path(__file__).resolve().parents[2]

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILS.append(f"{name}: {detail}")


MIXED_MD = """# GOAL
terminal_blocker: missing GPU firmware

- [ ] G001 — first obligation | status: PENDING | risk: high | tier: V2
      acceptance: a exists
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: (none yet)
- [ ] G002 — blocked obligation | status: BLOCKED | risk: medium | tier: V1
      acceptance: b exists
      verify: python3 -c "import sys; sys.exit(1)"
      evidence: waiting on firmware
- [x] G003 — third obligation | status: VERIFIED | risk: low | tier: V0
      acceptance: c exists
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: already done
"""


def _reload_dag(workspace: str):
    from hcli.dag_store import DagStore
    from hcli.goal import WorkUnitDAG

    if hasattr(WorkUnitDAG, "from_workspace"):
        return WorkUnitDAG.from_workspace(
            workspace, recover_running=False
        )
    dag = WorkUnitDAG()
    dag.units.update(DagStore(workspace).load(recover_running=False))
    return dag


def check_1_persisted_dag_survives_restart():
    """Build via GoalCompiler / WorkUnitDAG, persist via DagStore, reload."""
    from hcli.dag_store import DagStore
    from hcli.goal import GoalCompiler

    compiled = GoalCompiler().compile(
        "implement feature in foo.py. tests must pass."
    )
    dag = compiled["workunits"]
    ready = dag.get_ready_units()
    ready_ids = {wu.id for wu in ready}
    # Canonical identify_ready transitions pending -> ready.
    impl = dag.units["implement"]
    val = dag.units["validate"]
    with tempfile.TemporaryDirectory() as tmp:
        store = DagStore(tmp)
        store.save(dag.units)
        dag2 = _reload_dag(tmp)
        snap1 = dag.to_dict()
        snap2 = dag2.to_dict()
        check(
            "1 persisted DAG survives restart",
            (
                ready_ids == {"implement"}
                and impl.status == "ready"
                and val.status == "pending"
                and val.dependencies == ["implement"]
                and snap1 == snap2
                and dag2.units["implement"].status == "ready"
                and dag2.units["validate"].dependencies == ["implement"]
                and store.path.is_file()
            ),
            f"ready={ready_ids} impl.status={impl.status!r} "
            f"reloaded_impl={dag2.units['implement'].status!r} "
            f"snaps_equal={snap1 == snap2} "
            f"reloaded_deps={dag2.units['validate'].dependencies}",
        )


def check_2_dependencies_survive_restart():
    from hcli.dag_store import DagStore
    from hcli.goal import WorkUnitDAG

    dag = WorkUnitDAG()
    dag.add_unit("a", "root unit")
    dag.add_unit("b", "depends on a", ["a"])
    dag.get_ready_units()
    with tempfile.TemporaryDirectory() as tmp:
        DagStore(tmp).save(dag.units)
        dag2 = _reload_dag(tmp)
        ready_before = {wu.id for wu in dag2.get_ready_units()}
        b_ready_too_soon = "b" in ready_before
        dag2.mark_completed("a")
        ready_after = {wu.id for wu in dag2.get_ready_units()}
        check(
            "2 dependencies survive restart",
            (
                ready_before == {"a"}
                and not b_ready_too_soon
                and dag2.units["a"].status == "completed"
                and ready_after == {"b"}
                and dag2.units["b"].status == "ready"
                and dag2.units["b"].dependencies == ["a"]
            ),
            f"ready_before={ready_before} ready_after={ready_after} "
            f"a.status={dag2.units['a'].status!r} "
            f"b.status={dag2.units['b'].status!r} "
            f"b.deps={dag2.units['b'].dependencies}",
        )


def check_3_completed_workunits_are_not_replayed():
    from hcli.dag_store import DagStore
    from hcli.goal import WorkUnitDAG
    from hcli.workunit import identify_ready

    dag = WorkUnitDAG()
    dag.add_unit("a", "root unit")
    dag.add_unit("b", "depends on a", ["a"])
    dag.get_ready_units()
    dag.mark_completed("a")
    second_raised = False
    second_exc = ""
    try:
        dag.mark_completed("a")
    except Exception as exc:
        second_raised = True
        second_exc = f"{type(exc).__name__}: {exc}"
    with tempfile.TemporaryDirectory() as tmp:
        DagStore(tmp).save(dag.units)
        dag2 = _reload_dag(tmp)
        ready_ids = {wu.id for wu in dag2.get_ready_units()}
        canonical = {
            wu.id
            for wu in identify_ready(
                {uid: wu for uid, wu in dag2.units.items()}
            )
        }
        check(
            "3 completed WorkUnits are not replayed",
            (
                second_raised
                and dag2.units["a"].status == "completed"
                and "a" not in ready_ids
                and "a" not in canonical
                and "b" in ready_ids
            ),
            f"second_raised={second_raised} ({second_exc or 'no exception'}) "
            f"reloaded_a={dag2.units['a'].status!r} "
            f"ready={ready_ids} canonical={canonical}",
        )


def check_4_blocked_workunits_remain_blocked():
    from hcli.dag_store import DagStore
    from hcli.goal import WorkUnitDAG
    from hcli.workunit import identify_ready, WorkUnit

    dag = WorkUnitDAG()
    dag.add_unit("blocked_unit", "cannot proceed")
    dag.units["blocked_unit"].status = "BLOCKED"
    dag.add_unit("orig", "original work")
    dag.units["orig"].status = "failed"
    dag.units["orig"].attempts = 1
    dag.add_unit("orig.repair.1", "repair orig", ["orig"])
    dag.units["orig.repair.1"].repairs = "orig"
    with tempfile.TemporaryDirectory() as tmp:
        DagStore(tmp).save(dag.units)
        dag2 = _reload_dag(tmp)
        copies = {
            uid: WorkUnit.from_dict(wu.to_dict())
            for uid, wu in dag2.units.items()
        }
        canonical = {wu.id for wu in identify_ready(copies)}
        ready_ids = {wu.id for wu in dag2.get_ready_units()}
        check(
            "4 blocked WorkUnits remain blocked",
            (
                dag2.units["blocked_unit"].status == "BLOCKED"
                and "blocked_unit" not in ready_ids
                and dag2.units["orig"].status == "failed"
                and "orig" not in ready_ids
                and ready_ids == canonical
                and "orig.repair.1" in ready_ids
            ),
            f"blocked.status={dag2.units['blocked_unit'].status!r} "
            f"orig.status={dag2.units['orig'].status!r} "
            f"ready={ready_ids} canonical={canonical}",
        )


def check_5_steering_updates_existing_ledger():
    from hcli.ledger import Ledger
    from hcli.steering import SteeringQueue

    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "GOAL.md"
    path.write_text(MIXED_MD, encoding="utf-8")
    led = Ledger.parse(path)
    before_ids = [ob.id for ob in led.obligations()]
    queue = SteeringQueue(tmp, "consolidation-sess")
    event = queue.enqueue(
        "add: extra requirement about widgets",
        kind="constraint",
    )
    apply_ok = True
    apply_exc = ""
    try:
        led.apply_constraint(event, queue)
    except Exception as exc:
        apply_ok = False
        apply_exc = f"{type(exc).__name__}: {exc}"
        # Fall back: in-memory amend only, which is the pre-fix gap
        # (steering mutates memory and does not persist the ledger file).
        try:
            queue.apply_constraint(event, led)
        except Exception:
            pass
    led2 = Ledger.parse(path)
    after_ids = [ob.id for ob in led2.obligations()]
    texts = [ob.text for ob in led2.obligations()]
    amended = any("widgets" in text for text in texts)
    cites = any(event.id in text for text in texts)
    same_g001 = led2.get("G001").text == led.get("G001").text
    check(
        "5 steering updates the correct existing ledger",
        apply_ok and amended and cites and same_g001 and len(after_ids) > len(before_ids),
        f"apply_ok={apply_ok} ({apply_exc or 'ok'}) amended={amended} "
        f"cites={cites} after_ids={after_ids} before_ids={before_ids} "
        f"file_len={len(led2)}",
    )


def check_6_no_semantic_drift_after_reload():
    from hcli.ledger import Ledger

    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "GOAL.md"
    path.write_text(MIXED_MD, encoding="utf-8")
    led = Ledger.parse(path)
    first = led.to_markdown()
    path.write_text(first, encoding="utf-8")
    led2 = Ledger.parse(path)
    second = led2.to_markdown()
    path.write_text(second, encoding="utf-8")
    led3 = Ledger.parse(path)
    third = led3.to_markdown()
    blocker = getattr(led, "terminal_blocker", None)
    blocker3 = getattr(led3, "terminal_blocker", None)
    statuses = [ob.status for ob in led3.obligations()]
    check(
        "6 no semantic drift after reload",
        (
            first == MIXED_MD
            and second == first
            and third == second
            and blocker == "missing GPU firmware"
            and blocker3 == "missing GPU firmware"
            # G003 is marked VERIFIED in the source markdown with no verify
            # receipt beside it, so parse demotes it IN MEMORY to STALE. The
            # FILE is deliberately left alone -- reading a ledger must not
            # rewrite it -- which is why the round-trip above is still exact.
            # The old expectation here (VERIFIED) was the forgeable behaviour:
            # a hand-edited checkbox counted as evidence.
            and statuses == ["PENDING", "BLOCKED", "STALE"]
            and led3.get("G001").checked is False
            and led3.get("G003").checked is False
            and led3.is_goal_met() is False
        ),
        f"first_eq_src={first == MIXED_MD} second_eq_first={second == first} "
        f"third_eq_second={third == second} blocker={blocker!r} "
        f"blocker3={blocker3!r} statuses={statuses}",
    )


CHECKS = [
    ("1 persisted DAG survives restart", check_1_persisted_dag_survives_restart),
    ("2 dependencies survive restart", check_2_dependencies_survive_restart),
    ("3 completed WorkUnits are not replayed", check_3_completed_workunits_are_not_replayed),
    ("4 blocked WorkUnits remain blocked", check_4_blocked_workunits_remain_blocked),
    ("5 steering updates the correct existing ledger", check_5_steering_updates_existing_ledger),
    ("6 no semantic drift after reload", check_6_no_semantic_drift_after_reload),
]


def main() -> int:
    FAILS.clear()
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for item in FAILS:
            print("  " + item)
        return 1
    print("\nall hcli dag consolidation checks passed")
    return 0


def test_hcli_dag_consolidation():
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0, f"{len(FAILS)} dag consolidation checks failed: {FAILS}"


if __name__ == "__main__":
    sys.exit(main())
