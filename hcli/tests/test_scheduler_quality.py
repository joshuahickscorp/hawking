"""Scheduler quality: duplicates, filler, identity, starvation, signals.

Critical-path / remaining_depth ordering was measured and rejected. Dispatch
is FIFO by ready_at, then this-process _ready_seq, then id. These tests
cover the part of the scheduler that was not driven: admission, synthesis,
identity across restart+replan, FIFO under adversarial arrivals, and what
is actually observable as a scheduling signal.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli import scheduler as scheduler_mod
from hcli.resources import ResourceLimits
from hcli.scheduler import Scheduler
from hcli.workunit import (
    MAX_REPAIRS_PER_ROOT,
    IdentityConflict,
    WorkUnit,
    assign_ready,
    content_identity,
    identify_ready,
)


def _wu(uid: str, resource_class: str = "LIGHT_CONTROL", **kwargs) -> WorkUnit:
    return WorkUnit(
        id=uid,
        role=kwargs.pop("role", "work"),
        description=kwargs.pop("description", uid),
        resource_class=resource_class,
        **kwargs,
    )


def _ok(sched: Scheduler, wu_id: str) -> None:
    sched.complete(wu_id, verification={"ok": True})


def _serial_limits() -> ResourceLimits:
    """One-at-a-time so FIFO, not a class cap, is the sequencer."""
    return ResourceLimits(
        gpu_decode=1,
        gpu_decode_source="test",
        gpu_exclusive=1,
        mutation=1,
        cpu_heavy=1,
        compile=1,
        test=1,
        test_authoring=1,
        static_analysis=1,
        memory_heavy=1,
        io_heavy=1,
        tool_wait=1,
        light_control=1,
        grok=1,
    )


class TestDuplicateSubmission(unittest.TestCase):
    def test_same_unit_twice_is_one_unit_and_one_dispatch(self):
        wu = _wu(
            "G001",
            role="impl",
            description="wire the mutation lock",
            verifier="pytest -q",
        )
        sched = Scheduler({}, runtime_count=4, limits=_serial_limits())
        first = sched.submit(wu)
        second = sched.submit(wu)
        self.assertEqual(first.kind, "inserted")
        self.assertEqual(second.kind, "idempotent")
        self.assertIs(second.unit, first.unit)
        self.assertEqual(set(sched.units), {"G001"})
        first_wave = sched.dispatch()
        self.assertEqual([u.id for u, _ in first_wave], ["G001"])
        second_wave = sched.dispatch()
        self.assertEqual(
            second_wave,
            [],
            f"duplicate dispatch of G001: {[(u.id, u.status) for u, _ in second_wave]}",
        )
        self.assertEqual(sched.units["G001"].status, "running")
        self.assertEqual(len(sched.units), 1)

    def test_same_content_different_id_is_idempotent_not_a_second_unit(self):
        a = _wu("G001", role="impl", description="same work", verifier="pytest -q")
        b = _wu("G002", role="impl", description="same work", verifier="pytest -q")
        sched = Scheduler({}, runtime_count=4, limits=_serial_limits())
        first = sched.submit(a)
        second = sched.submit(b)
        self.assertEqual(second.kind, "idempotent")
        self.assertEqual(second.unit.id, first.unit.id)
        self.assertEqual(set(sched.units), {"G001"})
        assigned = sched.dispatch()
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0][0].id, "G001")
        self.assertEqual(sched.dispatch(), [])

    def test_same_id_different_content_is_a_conflict_not_a_silent_overwrite(self):
        v1 = _wu("G001", role="impl", description="first plan: add the lock")
        v2 = _wu("G001", role="impl", description="second plan: rewrite the scheduler")
        sched = Scheduler({}, runtime_count=1)
        sched.submit(v1)
        with self.assertRaises(IdentityConflict) as raised:
            sched.submit(v2)
        self.assertEqual(raised.exception.unit_id, "G001")
        self.assertEqual(sched.units["G001"].description, "first plan: add the lock")
        self.assertEqual(content_identity(sched.units["G001"]), content_identity(v1))
        self.assertNotEqual(content_identity(v1), content_identity(v2))

    def test_replan_conflict_leaves_the_live_graph_unchanged(self):
        keep = _wu("keep", description="untouched")
        v1 = _wu("G001", description="original work")
        sched = Scheduler({"keep": keep, "G001": v1}, runtime_count=1)
        with self.assertRaises(IdentityConflict):
            sched.replan(
                [
                    _wu("keep", description="untouched"),
                    _wu("G001", description="a different plan under the same id"),
                ]
            )
        self.assertEqual(set(sched.units), {"keep", "G001"})
        self.assertEqual(sched.units["G001"].description, "original work")
        self.assertIs(sched.units["keep"], keep)


class TestNoFillerWork(unittest.TestCase):
    def test_dispatch_and_complete_do_not_grow_the_graph(self):
        units = {"g0": _wu("g0", "GPU_DECODE")}
        sched = Scheduler(units, runtime_count=8)
        before = set(sched.units)
        self.assertEqual([u.id for u, _ in sched.dispatch()], ["g0"])
        self.assertEqual(set(sched.units), before)
        _ok(sched, "g0")
        self.assertEqual(sched.dispatch(), [])
        self.assertEqual(set(sched.units), before)
        self.assertTrue(sched.is_done())

    def test_idle_class_does_not_receive_invented_work(self):
        units = {"g0": _wu("g0", "GPU_DECODE")}
        sched = Scheduler(units, runtime_count=8)
        sched.dispatch()
        occupancy = (sched.last_dispatch or {}).get("occupancy") or {}
        self.assertEqual(occupancy.get("GPU_DECODE"), 1)
        self.assertFalse(
            any(k != "GPU_DECODE" and v for k, v in occupancy.items()),
            f"idle class occupied without a unit: {occupancy}",
        )
        self.assertEqual(set(sched.units), {"g0"})

    def test_only_repair_of_an_existing_failure_synthesises_a_unit(self):
        dispatch_src = inspect.getsource(Scheduler.dispatch)
        complete_src = inspect.getsource(Scheduler.complete)
        submit_src = inspect.getsource(Scheduler.submit)
        replan_src = inspect.getsource(Scheduler.replan)
        identify_src = inspect.getsource(identify_ready)
        assign_src = inspect.getsource(assign_ready)
        for name, src in (
            ("dispatch", dispatch_src),
            ("complete", complete_src),
            ("submit", submit_src),
            ("replan", replan_src),
            ("identify_ready", identify_src),
            ("assign_src", assign_src),
        ):
            self.assertNotIn(
                "WorkUnit(",
                src,
                f"{name} constructs a WorkUnit — that is filler",
            )
        emit_src = inspect.getsource(Scheduler._emit_repair)
        self.assertIn("emit_repair", emit_src)

        units = {"orig": _wu("orig", description="the only real work")}
        sched = Scheduler(units, runtime_count=1, limits=_serial_limits())
        sched.dispatch()
        n = len(sched.units)
        repair = sched.fail("orig", context={"error": "boom", "reason": "transient"})
        self.assertIsNotNone(repair)
        self.assertEqual(len(sched.units), n + 1)
        self.assertEqual(repair.repairs, "orig")
        self.assertEqual(repair.repair_root, "orig")
        self.assertTrue(repair.id.startswith("orig.repair."))
        self.assertIn("repair of orig", repair.description)

    def test_empty_scheduler_stays_empty(self):
        sched = Scheduler({}, runtime_count=8)
        self.assertEqual(sched.dispatch(), [])
        self.assertEqual(sched.units, {})
        self.assertIsNone(sched.fail("ghost"))
        self.assertEqual(sched.units, {})


_RESTART_REPLAN = r"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from hcli.scheduler import Scheduler
from hcli.workunit import WorkUnit, content_identity

ws = sys.argv[2]
sched = Scheduler.from_workspace(ws, runtime_count=1)
original = sched.units["G001"]
incoming = WorkUnit(
    id="recomputed-G001",
    role=original.role,
    description=original.description,
    dependencies=list(original.dependencies),
    verifier=original.verifier,
)
outcomes = sched.replan([incoming])
repair_ids = sorted(u.id for u in sched.units.values() if u.repairs)
print(
    json.dumps(
        {
            "ids": sorted(sched.units),
            "g001_status": sched.units["G001"].status,
            "g001_hash": content_identity(sched.units["G001"]),
            "outcome_kind": outcomes[0].kind,
            "outcome_id": outcomes[0].unit.id,
            "repair_ids": repair_ids,
            "repair_root": [
                sched.units[rid].repair_root for rid in repair_ids
            ],
            "has_recomputed": "recomputed-G001" in sched.units,
        }
    )
)
"""


class TestIdentityAcrossRestartAndReplan(unittest.TestCase):
    def test_identity_survives_in_process_restart_and_replan(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = _wu(
                "G001",
                role="impl",
                description="wire the mutation lock",
                verifier="pytest -q hcli/tests/test_scheduler_quality.py",
            )
            child = _wu(
                "G002",
                role="validate",
                description="prove the lock holds",
                dependencies=["G001"],
                verifier="pytest -q",
            )
            sched = Scheduler(
                {"G001": original, "G002": child},
                runtime_count=1,
                workspace=tmp,
                limits=_serial_limits(),
            )
            sched.dispatch()
            self.assertEqual(sched.units["G001"].status, "running")
            fingerprint = content_identity(sched.units["G001"])
            repair = sched.fail(
                "G001", context={"error": "backend down", "reason": "transient"}
            )
            self.assertIsNotNone(repair)
            repair_id = repair.id
            self.assertEqual(repair.repair_root, "G001")

            restarted = Scheduler.from_workspace(
                tmp, runtime_count=1, limits=_serial_limits()
            )
            self.assertEqual(set(restarted.units), set(sched.units))
            self.assertEqual(restarted.units["G001"].id, "G001")
            self.assertEqual(content_identity(restarted.units["G001"]), fingerprint)
            self.assertEqual(restarted.units[repair_id].repair_root, "G001")
            self.assertEqual(restarted.units[repair_id].repairs, "G001")
            self.assertEqual(restarted.units["G002"].dependencies, ["G001"])

            # Replan under a new compiler-assigned id. Same work must keep G001
            # so the repair cap stays anchored on the original root.
            recomputed = _wu(
                "recomputed-G001",
                role="impl",
                description="wire the mutation lock",
                verifier="pytest -q hcli/tests/test_scheduler_quality.py",
            )
            outcomes = restarted.replan([recomputed, child])
            self.assertEqual(outcomes[0].kind, "idempotent")
            self.assertEqual(outcomes[0].unit.id, "G001")
            self.assertNotIn("recomputed-G001", restarted.units)
            self.assertIn("G001", restarted.units)
            self.assertIn(repair_id, restarted.units)
            self.assertEqual(restarted.units[repair_id].repair_root, "G001")
            self.assertEqual(content_identity(restarted.units["G001"]), fingerprint)
            self.assertEqual(
                restarted.units["G001"].status,
                "failed",
                "replan must not clobber runtime state of the surviving unit",
            )

    def test_identity_survives_a_real_process_restart_then_replan(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = _wu(
                "G001",
                role="impl",
                description="wire the mutation lock",
                verifier="pytest -q",
            )
            sched = Scheduler(
                {"G001": original},
                runtime_count=1,
                workspace=tmp,
                limits=_serial_limits(),
            )
            sched.dispatch()
            fingerprint = content_identity(sched.units["G001"])
            repair = sched.fail(
                "G001", context={"error": "e1", "reason": "r1"}
            )
            self.assertIsNotNone(repair)
            env = os.environ.copy()
            env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
            proc = subprocess.run(
                [sys.executable, "-c", _RESTART_REPLAN, str(REPO), tmp],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"restart child failed: stdout={proc.stdout!r} stderr={proc.stderr!r}",
            )
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertIn("G001", payload["ids"])
            self.assertFalse(payload["has_recomputed"])
            self.assertEqual(payload["outcome_kind"], "idempotent")
            self.assertEqual(payload["outcome_id"], "G001")
            self.assertEqual(payload["g001_hash"], fingerprint)
            self.assertEqual(payload["repair_root"], ["G001"])
            self.assertTrue(payload["repair_ids"])

    def test_scheduler_repair_cap_survives_from_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = {"dead": _wu("dead", description="permanently unavailable")}
            sched = Scheduler(units, runtime_count=1, workspace=tmp)
            sched.dispatch()
            for i in range(MAX_REPAIRS_PER_ROOT):
                repair = sched.fail(
                    "dead", context={"error": f"e{i}", "reason": f"r{i}"}
                )
                self.assertIsNotNone(repair, f"repair {i + 1} of cap refused")
            before = sum(1 for u in sched.units.values() if u.repairs)
            self.assertEqual(before, MAX_REPAIRS_PER_ROOT)
            restarted = Scheduler.from_workspace(tmp, runtime_count=1)
            after_load = sum(1 for u in restarted.units.values() if u.repairs)
            self.assertEqual(after_load, before)
            seventh = restarted.fail(
                "dead", context={"error": "post-restart", "reason": "post"}
            )
            self.assertIsNone(seventh)
            self.assertTrue(restarted.units["dead"].repair_exhausted)
            self.assertEqual(
                sum(1 for u in restarted.units.values() if u.repairs),
                MAX_REPAIRS_PER_ROOT,
            )


class TestDependencyStarvation(unittest.TestCase):
    """FIFO by ready_at must not let a cheap stream bury a chain forever.

    Adversarial arrival: a 4-hop chain plus an initial flood of cheap
    unrelated units, then 3 new cheap units injected before every
    dispatch tick. Units already ready before a chain hop becomes ready
    run first (that is FIFO). Units that arrive after the hop is stamped
    ready sort behind it. The chain must complete; the worst wait is
    the size of the ready-ahead flood, not unbounded.
    """

    INITIAL_FLOOD = 20
    INJECT_PER_TICK = 3
    CHAIN = ("c0", "c1", "c2", "c3")

    def test_adversarial_arrivals_do_not_starve_the_chain(self):
        limits = _serial_limits()
        units = {}
        for i, cid in enumerate(self.CHAIN):
            deps = [self.CHAIN[i - 1]] if i else []
            units[cid] = _wu(cid, "GPU_EXCLUSIVE", dependencies=deps)
        sched = Scheduler(units, runtime_count=8, limits=limits)
        for i in range(self.INITIAL_FLOOD):
            sched.submit(_wu(f"flood_{i:02d}", "GPU_EXCLUSIVE"))

        ready_wait = {cid: 0 for cid in self.CHAIN}
        ran_order: list[str] = []
        tick = 0
        max_ticks = 400
        while tick < max_ticks and not all(
            sched.units[c].status == "completed" for c in self.CHAIN
        ):
            for j in range(self.INJECT_PER_TICK):
                sched.submit(_wu(f"adv_{tick:03d}_{j}", "GPU_EXCLUSIVE"))
            assigned = sched.dispatch()
            for cid in self.CHAIN:
                if sched.units[cid].status == "ready":
                    ready_wait[cid] += 1
            running = [u for u in sched.units.values() if u.status == "running"]
            self.assertEqual(
                len(running),
                1,
                f"tick {tick}: expected one running, got {[u.id for u in running]}",
            )
            wu = running[0]
            ran_order.append(wu.id)
            _ok(sched, wu.id)
            tick += 1

        self.assertTrue(
            all(sched.units[c].status == "completed" for c in self.CHAIN),
            f"chain did not finish in {tick} ticks; wait={ready_wait} order={ran_order}",
        )
        # Hop 0 was ready with the flood; it must run first.
        self.assertEqual(ran_order[0], "c0")
        hops_in_order = [uid for uid in ran_order if uid in self.CHAIN]
        self.assertEqual(hops_in_order, list(self.CHAIN))
        worst = max(ready_wait.values())
        # c0 is ready with the flood and runs immediately.
        self.assertEqual(
            ready_wait["c0"],
            0,
            f"c0 must run first tick; per_hop={ready_wait} WORST={worst}",
        )
        # c1 waits for work that became ready while c0 ran: the initial
        # flood plus the inject on the tick that dispatched c0.
        self.assertLessEqual(
            ready_wait["c1"],
            self.INITIAL_FLOOD + self.INJECT_PER_TICK,
            f"c1 wait {ready_wait['c1']} exceeded flood+inject "
            f"{self.INITIAL_FLOOD + self.INJECT_PER_TICK}; FIFO sort is broken. "
            f"wait={ready_wait}",
        )
        # Later hops wait longer: the stream queues behind hop N and is
        # already ready when hop N+1 is stamped. FIFO, finite (the chain
        # finished), not "for free" protection of the tail. Measured:
        # c0=0 c1=23 c2=72 c3=219 over 318 ticks with flood=20 inject=3.
        self.assertLess(worst, max_ticks)
        TestDependencyStarvation.observed_wait = dict(ready_wait)
        TestDependencyStarvation.observed_worst = worst
        TestDependencyStarvation.observed_ticks = tick

    def test_stream_after_ready_does_not_cut_in_line(self):
        limits = _serial_limits()
        sched = Scheduler(
            {
                "head": _wu("head", "GPU_EXCLUSIVE", status="completed"),
                "tail": _wu("tail", "GPU_EXCLUSIVE", dependencies=["head"]),
            },
            runtime_count=8,
            limits=limits,
        )
        ready = identify_ready(sched.units)
        self.assertEqual({u.id for u in ready}, {"tail"})
        self.assertEqual(sched.units["tail"].status, "ready")
        self.assertIsNotNone(sched.units["tail"].ready_at)
        for i in range(50):
            sched.submit(_wu(f"late_{i:02d}", "GPU_EXCLUSIVE"))
        assigned = sched.dispatch()
        self.assertEqual(
            [u.id for u, _ in assigned],
            ["tail"],
            "cheap units arriving after tail was ready cut in line",
        )

    def test_unstamped_ready_unit_sorts_last_not_front(self):
        limits = _serial_limits()
        stamped = _wu("stamped", "GPU_EXCLUSIVE")
        unstamped = _wu("aaa_unstamped", "GPU_EXCLUSIVE")
        stamped.status = "ready"
        stamped.ready_at = 100.0
        unstamped.status = "ready"
        unstamped.ready_at = None
        sched = Scheduler(
            {"stamped": stamped, "aaa_unstamped": unstamped},
            runtime_count=8,
            limits=limits,
        )
        assigned = sched.dispatch()
        self.assertEqual(
            assigned[0][0].id,
            "stamped",
            "unstamped unit floated to the front (ready_at or 0.0 regression)",
        )


class TestSchedulingSignals(unittest.TestCase):
    def test_mutation_lock_congestion_is_recorded_not_a_reorder_key(self):
        units = {
            "m1": _wu("m1", "MUTATION"),
            "m2": _wu("m2", "MUTATION"),
            "m3": _wu("m3", "MUTATION"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            sched = Scheduler(units, runtime_count=8, workspace=tmp)
            assigned = sched.dispatch()
            self.assertEqual(len(assigned), 1)
            rec = sched.last_dispatch
            self.assertIsNotNone(rec)
            self.assertEqual(rec["requested"].get("MUTATION"), 3)
            self.assertEqual(rec["admitted"].get("MUTATION"), 1)
            self.assertEqual(rec["mutation_blocked"], 2)
            dispatch_src = inspect.getsource(Scheduler.dispatch)
            sort_src = dispatch_src[dispatch_src.find("ready.sort") :]
            sort_src = sort_src[: sort_src.find("requested")]
            self.assertNotIn("mutation_blocked", sort_src)
            self.assertNotIn("verifier_backlog", sort_src)

    def test_verifier_backlog_is_not_a_scheduler_input(self):
        src = Path(scheduler_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("verifier_backlog", src)
        self.assertNotIn("unverified", src)
        dispatch_src = inspect.getsource(Scheduler.dispatch)
        self.assertNotIn("ledger", dispatch_src.lower())
        # last_dispatch is occupancy telemetry. It does not grow a backlog
        # field so a missing number cannot be mistaken for zero.
        sched = Scheduler({"a": _wu("a")}, runtime_count=1, limits=_serial_limits())
        sched.dispatch()
        self.assertNotIn("verifier_backlog", sched.last_dispatch or {})


if __name__ == "__main__":
    unittest.main()
