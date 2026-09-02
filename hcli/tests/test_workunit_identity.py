"""Content identity, interrupt-vs-fail, and durable repair caps.

These tests are written against the public WorkUnit / DagStore surface so
they collect on unmodified code. Missing new names fall through to the
current behaviour and fail on the defect, not on ImportError.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli import dag_store as dag_store_mod
from hcli import workunit as workunit_mod
from hcli.dag_store import DagStore
from hcli.scheduler import Scheduler
from hcli.workunit import WorkUnit, assign_ready, identify_ready, is_ready


def _wu(uid: str, **kwargs) -> WorkUnit:
    role = kwargs.pop("role", "work")
    description = kwargs.pop("description", uid)
    return WorkUnit(id=uid, role=role, description=description, **kwargs)


def _content_hash_fn():
    fn = getattr(workunit_mod, "content_identity", None)
    if callable(fn):
        return fn
    method = getattr(WorkUnit, "content_hash", None)
    if callable(method):
        return lambda wu: wu.content_hash()
    return None


class TestContentIdentity(unittest.TestCase):
    def test_same_content_twice_is_idempotent(self):
        """Two units with the same work are the same work, even under two ids."""
        a = _wu("G001", role="impl", description="wire the mutation lock")
        b = _wu("G002", role="impl", description="wire the mutation lock")
        if hasattr(a, "verifier"):
            a.verifier = "pytest -q hcli/tests/test_workunit_identity.py"
            b.verifier = "pytest -q hcli/tests/test_workunit_identity.py"
        admit = getattr(workunit_mod, "admit_unit", None)
        units = {}
        if admit is None:
            units[a.id] = a
            units[b.id] = b
            self.assertEqual(
                len(units),
                1,
                f"same content admitted twice under ids {sorted(units)}",
            )
            return
        first = admit(units, a)
        second = admit(units, b)
        self.assertEqual(len(units), 1, f"duplicate content survived as {sorted(units)}")
        self.assertEqual(getattr(second, "kind", None), "idempotent")
        self.assertEqual(second.unit.id, first.unit.id)

    def test_same_id_different_content_is_a_conflict_not_a_silent_overwrite(self):
        v1 = _wu("G001", role="impl", description="first plan: add the lock")
        v2 = _wu("G001", role="impl", description="second plan: rewrite the scheduler")
        Conflict = getattr(workunit_mod, "IdentityConflict", None)
        if Conflict is None:
            Conflict = getattr(dag_store_mod, "IdentityConflict", None)
        with tempfile.TemporaryDirectory() as tmp:
            store = DagStore(tmp)
            store.save({"G001": v1})
            raised = None
            try:
                store.save({"G001": v2})
            except Exception as exc:
                raised = exc
            loaded = store.load(recover_running=False)
            if raised is None:
                self.assertEqual(
                    loaded["G001"].description,
                    "first plan: add the lock",
                    "silent overwrite: same id with different content replaced "
                    f"the original with {loaded['G001'].description!r}",
                )
                return
            self.assertIsNotNone(Conflict, "conflict must be a named IdentityConflict")
            self.assertIsInstance(raised, Conflict)
            self.assertEqual(loaded["G001"].description, "first plan: add the lock")

    def test_content_identity_is_a_hash_of_role_description_deps_verifier(self):
        fn = _content_hash_fn()
        if fn is None:
            self.fail(
                "WorkUnit has no content identity hash of "
                "(role, description, dependencies, verifier)"
            )
        a = _wu("id-a", role="impl", description="same work", dependencies=["x"])
        b = _wu("id-b", role="impl", description="same work", dependencies=["x"])
        c = _wu("id-c", role="impl", description="different work", dependencies=["x"])
        if hasattr(a, "verifier"):
            a.verifier = "pytest -q"
            b.verifier = "pytest -q"
            c.verifier = "pytest -q"
        self.assertEqual(fn(a), fn(b))
        self.assertNotEqual(fn(a), fn(c))
        d = _wu("id-d", role="impl", description="same work", dependencies=["y"])
        self.assertNotEqual(fn(a), fn(d))


class TestInterruptedIsNotFailed(unittest.TestCase):
    def test_killed_running_unit_is_interrupted_not_failed(self):
        """A kill at attempt 3 must not look like a red test and must not burn the budget."""
        wu = _wu(
            "verify",
            role="validate",
            description="run the verifier",
            status="running",
            attempts=3,
            assigned_runtime=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = DagStore(tmp)
            store.save({"verify": wu})
            loaded = store.load(recover_running=True)
            got = loaded["verify"]
            self.assertEqual(
                got.attempts,
                3,
                f"crash mutated attempts: before=3 after={got.attempts}",
            )
            self.assertNotEqual(got.status, "completed")
            self.assertNotEqual(
                got.status,
                "failed",
                "process death recovered as failed — a kill is indistinguishable "
                "from a verifier failure",
            )
            classification = getattr(got, "classification", None)
            if classification is None and isinstance(got.failure_context, dict):
                classification = got.failure_context.get("classification")
            self.assertEqual(classification, "INTERRUPTED")
            self.assertTrue(
                is_ready(got, loaded) or got.status in ("interrupted", "ready"),
                "an interrupted unit at attempts=3 must remain re-runnable",
            )

    def test_verifier_failure_stays_failed_and_is_distinct(self):
        failed = _wu(
            "red",
            role="validate",
            description="this one actually failed the verifier",
            status="failed",
            attempts=1,
        )
        self.assertEqual(failed.status, "failed")
        self.assertNotEqual(failed.status, "interrupted")
        self.assertTrue(is_ready(failed, {failed.id: failed}))

    def test_interrupt_rerun_does_not_consume_another_attempt(self):
        wu = _wu(
            "mid",
            role="impl",
            description="inference work",
            status="running",
            attempts=3,
            assigned_runtime=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = DagStore(tmp)
            store.save({"mid": wu})
            units = store.load(recover_running=True)
            got = units["mid"]
            self.assertNotEqual(got.status, "failed")
            self.assertEqual(got.attempts, 3)
            ready = identify_ready(units)
            self.assertTrue(any(u.id == "mid" for u in ready))
            assigned = assign_ready(ready, 1, all_units=units)
            self.assertEqual(len(assigned), 1)
            self.assertEqual(units["mid"].status, "running")
            self.assertEqual(
                units["mid"].attempts,
                3,
                "re-running an interrupted unit must not consume another retry "
                f"(attempts became {units['mid'].attempts})",
            )

    def test_recovery_does_not_claim_mid_token_resume(self):
        policy = getattr(workunit_mod, "RESUME_POLICY", None)
        self.assertEqual(
            policy,
            "rerun",
            "recovery must declare a rerun policy; mid-token continue is not possible",
        )
        wu = _wu("tok", status="running", attempts=1, assigned_runtime=0)
        with tempfile.TemporaryDirectory() as tmp:
            store = DagStore(tmp)
            store.save({"tok": wu})
            got = store.load(recover_running=True)["tok"]
            self.assertNotEqual(got.status, "running")
            ctx = got.failure_context or {}
            blob = json.dumps(ctx, sort_keys=True).lower()
            self.assertNotIn("mid-token", blob)
            self.assertNotIn("continue_from", blob)
            self.assertNotIn("resume_from_token", blob)


class TestRepairCapSurvivesRestart(unittest.TestCase):
    def test_lineage_cannot_exceed_cap_across_a_real_restart(self):
        """The measured hole was 8 repairs against a cap of 6 after process death."""
        MAX = getattr(workunit_mod, "MAX_REPAIRS_PER_ROOT", 6)
        emit = getattr(workunit_mod, "emit_repair", None)
        rebuild = getattr(workunit_mod, "rebuild_repair_budget", None)

        with tempfile.TemporaryDirectory() as tmp:
            if emit is None:
                units = {"dead": _wu("dead", description="permanently unavailable")}
                sched = Scheduler(units, runtime_count=1, workspace=tmp)
                sched.dispatch()
                for i in range(MAX):
                    sched.fail("dead", context={"error": f"e{i}", "reason": f"r{i}"})
                before = sum(1 for u in sched.units.values() if u.repairs)
                restarted = Scheduler.from_workspace(tmp, runtime_count=1)
                after_load = sum(1 for u in restarted.units.values() if u.repairs)
                for i in range(2):
                    restarted.fail(
                        "dead",
                        context={"error": f"post{i}", "reason": f"p{i}"},
                    )
                after = sum(1 for u in restarted.units.values() if u.repairs)
                self.assertEqual(after_load, before)
                self.assertLessEqual(
                    after,
                    MAX,
                    f"measured hole: {after} repairs after restart against cap {MAX} "
                    f"(before={before}, after_load={after_load})",
                )
                return

            units = {"dead": _wu("dead", description="permanently unavailable")}
            units["dead"].status = "failed"
            budget = rebuild(units) if rebuild is not None else None
            for i in range(MAX):
                units["dead"].status = "failed"
                repair = emit(
                    units,
                    units["dead"],
                    context={"error": f"e{i}", "reason": f"r{i}"},
                    budget=budget,
                )
                self.assertIsNotNone(repair, f"repair {i + 1} of {MAX} was refused")
            before = sum(1 for u in units.values() if u.repairs)
            self.assertEqual(before, MAX, f"before restart count={before} cap={MAX}")

            store = DagStore(tmp)
            store.save(units)

            script = (
                "import json, sys\n"
                ""
                "from hcli.dag_store import DagStore\n"
                "from hcli import workunit as wm\n"
                f"store = DagStore({tmp!r})\n"
                "units = store.load(recover_running=False)\n"
                "budget = wm.rebuild_repair_budget(units, store.last_meta.get('repair_budget'))\n"
                "before = budget['counts'].get('dead', 0)\n"
                "units['dead'].status = 'failed'\n"
                "seventh = wm.emit_repair(\n"
                "    units, units['dead'],\n"
                "    context={'error': 'e-post', 'reason': 'post-restart'},\n"
                "    budget=budget,\n"
                ")\n"
                "after = wm.rebuild_repair_budget(units, budget)['counts'].get('dead', 0)\n"
                "print(json.dumps({\n"
                "    'count_before': before,\n"
                "    'seventh': None if seventh is None else seventh.id,\n"
                "    'count_after': after,\n"
                f"    'cap': {MAX},\n"
                "}))\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"restart child failed: stdout={proc.stdout!r} stderr={proc.stderr!r}",
            )
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(
                payload["count_before"],
                MAX,
                f"before={payload['count_before']} cap={MAX} payload={payload}",
            )
            self.assertIsNone(
                payload["seventh"],
                f"seventh repair was admitted after restart: {payload}",
            )
            self.assertEqual(
                payload["count_after"],
                MAX,
                f"after={payload['count_after']} cap={MAX} payload={payload}",
            )

    def test_disk_repair_budget_is_a_floor_across_partial_save(self):
        emit = getattr(workunit_mod, "emit_repair", None)
        rebuild = getattr(workunit_mod, "rebuild_repair_budget", None)
        MAX = getattr(workunit_mod, "MAX_REPAIRS_PER_ROOT", 6)
        if emit is None or rebuild is None:
            self.fail(
                "repair budget is not persisted; a partial save can reset the cap"
            )
        with tempfile.TemporaryDirectory() as tmp:
            units = {"dead": _wu("dead", description="permanently unavailable")}
            units["dead"].status = "failed"
            budget = rebuild(units)
            for i in range(MAX):
                units["dead"].status = "failed"
                self.assertIsNotNone(
                    emit(
                        units,
                        units["dead"],
                        context={"error": f"e{i}", "reason": f"r{i}"},
                        budget=budget,
                    )
                )
            store = DagStore(tmp)
            store.save(units)
            store.save({"dead": units["dead"]})
            loaded = store.load(recover_running=False)
            floor = rebuild(loaded, store.last_meta.get("repair_budget"))
            self.assertEqual(
                floor["counts"].get("dead"),
                MAX,
                f"partial save reset the budget to {floor['counts'].get('dead')}",
            )
            loaded["dead"].status = "failed"
            seventh = emit(
                loaded,
                loaded["dead"],
                context={"error": "after-partial", "reason": "after-partial"},
                budget=floor,
            )
            self.assertIsNone(seventh)
            self.assertEqual(
                rebuild(loaded, floor)["counts"].get("dead"),
                MAX,
            )


if __name__ == "__main__":
    unittest.main()
