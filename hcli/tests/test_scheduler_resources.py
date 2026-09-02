from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.resources import (
    MutationLock,
    ResourceClass,
    ResourceLimits,
    process_start_token,
    resolve_active_decode_limit,
)
from hcli.scheduler import NO_PROGRESS, Scheduler, UnverifiedCompletion
from hcli.workunit import WorkUnit


def _ok_complete(sched, wu_id, fingerprint=None):
    """Passing verifier outcome is required to reach completed."""
    sched.complete(wu_id, fingerprint=fingerprint, verification={"ok": True})


def _wu(uid: str, resource_class: str = "LIGHT_CONTROL", **kwargs) -> WorkUnit:
    return WorkUnit(
        id=uid,
        role=kwargs.pop("role", "work"),
        description=kwargs.pop("description", uid),
        resource_class=resource_class,
        **kwargs,
    )


class TestActiveDecodeLimit(unittest.TestCase):
    def test_env_override_wins_and_records_source(self):
        saved = os.environ.get("ACTIVE_DECODE_LIMIT")
        os.environ["ACTIVE_DECODE_LIMIT"] = "2"
        try:
            limit, source = resolve_active_decode_limit()
            self.assertEqual(limit, 2)
            self.assertTrue(source.startswith("env:"), source)
            sched = Scheduler({"a": _wu("a", "GPU_DECODE")}, runtime_count=9)
            self.assertEqual(sched.active_decode_limit, 2)
            self.assertEqual(sched.active_decode_limit_source, source)
            self.assertNotEqual(sched.runtime_count, sched.active_decode_limit)
        finally:
            if saved is None:
                os.environ.pop("ACTIVE_DECODE_LIMIT", None)
            else:
                os.environ["ACTIVE_DECODE_LIMIT"] = saved

    def test_delegates_to_resolve_runtime_limits(self):
        from hcli.machine import resolve_runtime_limits

        with tempfile.TemporaryDirectory() as tmp:
            a = resolve_active_decode_limit(repo_root=tmp)
            b = resolve_runtime_limits(repo_root=tmp, start_dir=tmp)
            self.assertEqual(a, (b.active_decode_limit, b.active_source))

    def test_stale_genome_is_not_an_admission_authority(self):
        from unittest.mock import patch

        saved = {
            k: os.environ.pop(k, None)
            for k in (
                "ACTIVE_DECODE_LIMIT",
                "HCLI_ACTIVE_DECODE_LIMIT",
                "HCLI_RESIDENT_RUNTIME_LIMIT",
                "RESIDENT_RUNTIME_LIMIT",
            )
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                home = root / "home"
                genome_dir = home / ".config" / "hcli"
                genome_dir.mkdir(parents=True)
                (genome_dir / "machine_genome.json").write_text(
                    json.dumps(
                        {
                            "schema": "hcli.machine_genome.v1",
                            "generated_at": "2020-01-01T00:00:00Z",
                            "active_decode_limit": 9,
                            "resident_runtime_limit": 9,
                            "machine": {
                                "hw_model": "Mac00,0-not-this-box",
                                "cpu": "not-this-cpu",
                                "ncpu": 1,
                                "mem_bytes": 1,
                            },
                        }
                    )
                )
                with patch.object(Path, "home", return_value=home):
                    limit, source = resolve_active_decode_limit(repo_root=root)
                self.assertNotEqual(limit, 9)
                self.assertNotIn("machine_genome.json", source)
                if source == "fallback":
                    self.assertEqual(limit, 1)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_fallback_is_conservative_one(self):
        saved = os.environ.get("ACTIVE_DECODE_LIMIT")
        os.environ.pop("ACTIVE_DECODE_LIMIT", None)
        os.environ.pop("HCLI_ACTIVE_DECODE_LIMIT", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                limit, source = resolve_active_decode_limit(repo_root=tmp)
                # Home genome may still supply a value; either a file source
                # or the conservative fallback of 1 is acceptable here.
                self.assertGreaterEqual(limit, 1)
                self.assertTrue(isinstance(source, str) and source)
                if source == "fallback":
                    self.assertEqual(limit, 1)
        finally:
            if saved is not None:
                os.environ["ACTIVE_DECODE_LIMIT"] = saved


class TestResourceDispatch(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("ACTIVE_DECODE_LIMIT")
        os.environ["ACTIVE_DECODE_LIMIT"] = "2"

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("ACTIVE_DECODE_LIMIT", None)
        else:
            os.environ["ACTIVE_DECODE_LIMIT"] = self._saved

    def test_decode_cap_is_not_runtime_count(self):
        units = {f"g{i}": _wu(f"g{i}", "GPU_DECODE") for i in range(6)}
        sched = Scheduler(units, runtime_count=6)
        sched.dispatch()
        running = [u for u in units.values() if u.status == "running"]
        self.assertEqual(len(running), 2)
        self.assertEqual(sched.active_decode_limit, 2)
        self.assertEqual(sched.runtime_count, 6)

    def test_cpu_dispatches_while_decode_saturated(self):
        units = {
            "g0": _wu("g0", "GPU_DECODE"),
            "g1": _wu("g1", "GPU_DECODE"),
            "c0": _wu("c0", "COMPILE"),
            "t0": _wu("t0", "TEST"),
        }
        sched = Scheduler(units, runtime_count=2)
        sched.dispatch()
        self.assertEqual(units["g0"].status, "running")
        self.assertEqual(units["g1"].status, "running")
        self.assertEqual(units["c0"].status, "running")
        self.assertEqual(units["t0"].status, "running")

    def test_gpu_exclusive_blocks_decode_then_releases(self):
        units = {
            "ex": _wu("ex", "GPU_EXCLUSIVE"),
            "d0": _wu("d0", "GPU_DECODE"),
        }
        sched = Scheduler(units, runtime_count=4)
        sched.dispatch()
        self.assertEqual(units["ex"].status, "running")
        self.assertNotEqual(units["d0"].status, "running")
        _ok_complete(sched, "ex")
        sched.dispatch()
        self.assertEqual(units["d0"].status, "running")

    def test_mutation_serialized(self):
        units = {"m1": _wu("m1", "MUTATION"), "m2": _wu("m2", "MUTATION")}
        with tempfile.TemporaryDirectory() as tmp:
            sched = Scheduler(units, runtime_count=4, workspace=tmp)
            sched.dispatch()
            running = [u.id for u in units.values() if u.status == "running"]
            self.assertEqual(len(running), 1)
            holder = running[0]
            _ok_complete(sched, holder)
            sched.dispatch()
            other = "m2" if holder == "m1" else "m1"
            self.assertEqual(units[other].status, "running")

    def test_no_barrier_across_independent_chains(self):
        units = {
            "A1": _wu("A1"),
            "A2": _wu("A2", dependencies=["A1"]),
            "B1": _wu("B1"),
            "B2": _wu("B2", dependencies=["B1"]),
        }
        sched = Scheduler(units, runtime_count=4)
        sched.dispatch()
        self.assertEqual(units["A1"].status, "running")
        self.assertEqual(units["B1"].status, "running")
        _ok_complete(sched, "A1")
        assigned = {u.id for u, _ in sched.dispatch()}
        self.assertIn("A2", assigned)
        self.assertEqual(units["A2"].status, "running")
        self.assertEqual(units["B1"].status, "running")
        self.assertNotEqual(units["B2"].status, "running")

    def test_does_not_invent_work_for_idle_class(self):
        units = {"g0": _wu("g0", "GPU_DECODE")}
        sched = Scheduler(units, runtime_count=8)
        sched.dispatch()
        self.assertEqual(set(sched.units), {"g0"})
        _ok_complete(sched, "g0")
        self.assertEqual(sched.dispatch(), [])
        self.assertEqual(set(sched.units), {"g0"})

    def test_already_ready_units_are_reconsidered(self):
        units = {f"g{i}": _wu(f"g{i}", "GPU_DECODE") for i in range(3)}
        sched = Scheduler(units, runtime_count=8)
        sched.dispatch()
        self.assertEqual(sum(1 for u in units.values() if u.status == "running"), 2)
        live = next(u for u in units.values() if u.status == "running")
        _ok_complete(sched, live.id)
        sched.dispatch()
        self.assertEqual(sum(1 for u in units.values() if u.status == "running"), 2)

    def test_cpu_heavy_classes_are_in_the_enum(self):
        names = {c.value for c in ResourceClass}
        self.assertEqual(
            names,
            {
                "GPU_DECODE",
                "GPU_EXCLUSIVE",
                "GPU_DIRTY_OK",
                "CPU_HEAVY",
                "COMPILE",
                "TEST",
                "TEST_AUTHORING",
                "STATIC_ANALYSIS",
                "MEMORY_HEAVY",
                "IO_HEAVY",
                "TOOL_WAIT",
                "LIGHT_CONTROL",
                "MUTATION",
                "GROK",
            },
        )
        limits = ResourceLimits.resolve()
        ncpu = os.cpu_count() or 1
        self.assertEqual(limits.compile, ncpu)
        self.assertEqual(limits.test, ncpu)
        self.assertEqual(limits.cpu_heavy, ncpu)
        self.assertEqual(limits.static_analysis, ncpu)
        self.assertEqual(limits.mutation, 1)
        self.assertGreaterEqual(limits.tool_wait, 64)
        self.assertGreaterEqual(limits.light_control, 64)


class TestMutationLock(unittest.TestCase):
    def test_dead_pid_is_broken_live_pid_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            lock = MutationLock(ws)
            lock.write(
                {
                    "pid": 99999999,
                    "start_time": "ghost",
                    "acquired_at": 0,
                    "unit_id": "ghost",
                }
            )
            self.assertTrue(lock.try_break_stale())
            self.assertIsNone(lock.read())

            live = os.getpid()
            token = process_start_token(live)
            lock.write(
                {
                    "pid": live,
                    "start_time": token,
                    "acquired_at": time.time(),
                    "unit_id": "foreign",
                }
            )
            self.assertFalse(lock.try_break_stale())
            rec = lock.read()
            self.assertIsNotNone(rec)
            self.assertEqual(rec["pid"], live)

    def test_recycled_pid_is_broken(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = MutationLock(tmp)
            lock.write(
                {
                    "pid": os.getpid(),
                    "start_time": "not-this-incarnation",
                    "acquired_at": 0,
                    "unit_id": "stale",
                }
            )
            self.assertTrue(lock.try_break_stale())
            self.assertIsNone(lock.read())


class TestNoProgressAndRepair(unittest.TestCase):
    def test_no_progress_raises_then_stays_quiet_on_change(self):
        units = {f"u{i}": _wu(f"u{i}") for i in range(6)}
        sched = Scheduler(units, runtime_count=6, no_progress_threshold=3)
        sched.dispatch()
        _ok_complete(sched, "u0", fingerprint="same")
        _ok_complete(sched, "u1", fingerprint="same")
        with self.assertRaises(NO_PROGRESS) as raised:
            _ok_complete(sched, "u2", fingerprint="same")
        self.assertEqual(raised.exception.fingerprint, "same")
        self.assertEqual(raised.exception.count, 3)

        quiet = Scheduler(
            {f"u{i}": units[f"u{i}"] for i in range(3, 6)},
            runtime_count=6,
            no_progress_threshold=3,
        )
        quiet.dispatch()
        _ok_complete(quiet, "u3", fingerprint="a")
        _ok_complete(quiet, "u4", fingerprint="b")
        _ok_complete(quiet, "u5", fingerprint="c")

    def test_fail_emits_repair_carrying_context(self):
        units = {"orig": _wu("orig")}
        sched = Scheduler(units, runtime_count=1)
        sched.dispatch()
        attempts = units["orig"].attempts
        repair = sched.fail("orig", context={"error": "boom"})
        self.assertIsNotNone(repair)
        self.assertNotEqual(repair.id, "orig")
        self.assertEqual(repair.repairs, "orig")
        self.assertIn("boom", str(repair.failure_context))
        self.assertEqual(units["orig"].status, "failed")
        self.assertEqual(units["orig"].attempts, attempts)
        self.assertIn(repair.id, sched.units)


class TestVerifierGatedComplete(unittest.TestCase):
    def test_complete_requires_passing_verifier_outcome(self):
        units = {"a": _wu("a")}
        sched = Scheduler(units, runtime_count=1)
        sched.dispatch()
        self.assertEqual(units["a"].status, "running")
        with self.assertRaises(UnverifiedCompletion):
            sched.complete("a")
        self.assertEqual(units["a"].status, "running")
        with self.assertRaises(UnverifiedCompletion):
            sched.complete("a", verification={"ok": False, "reason": "nope"})
        self.assertEqual(units["a"].status, "running")
        sched.complete("a", verification={"ok": True})
        self.assertEqual(units["a"].status, "completed")
        self.assertEqual(units["a"].verification, {"ok": True})
        self.assertIsNotNone(units["a"].finished_at)

    def test_complete_reads_verification_already_on_the_unit(self):
        units = {"a": _wu("a")}
        units["a"].verification = {"ok": True, "source": "attached"}
        sched = Scheduler(units, runtime_count=1)
        sched.dispatch()
        sched.complete("a")
        self.assertEqual(units["a"].status, "completed")

    def test_dispatch_records_requested_admitted_overhead(self):
        # Inject the decode limit instead of letting ResourceLimits.resolve()
        # read it off the host. Ambient resolution returns (1, "fallback") on a
        # box with no decode genome, so the hardcoded 2 below asserted this
        # machine's configuration rather than the bookkeeping under test.
        units = {f"g{i}": _wu(f"g{i}", "GPU_DECODE") for i in range(4)}
        limits = ResourceLimits(gpu_decode=2, gpu_decode_source="test")
        sched = Scheduler(units, runtime_count=8, limits=limits)
        assigned = sched.dispatch()
        self.assertEqual(len(assigned), limits.gpu_decode)
        rec = sched.last_dispatch
        self.assertIsNotNone(rec)
        self.assertEqual(rec["requested"].get("GPU_DECODE"), 4)
        self.assertEqual(rec["admitted"].get("GPU_DECODE"), limits.gpu_decode)
        self.assertIn("overhead_s", rec)
        self.assertGreaterEqual(rec["overhead_s"], 0)
        self.assertEqual(rec["occupancy"].get("GPU_DECODE"), 2)


if __name__ == "__main__":
    unittest.main()
