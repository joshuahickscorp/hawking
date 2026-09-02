"""Lock the G028 consolidations: one live implementation per aligned concept.

These tests fail if a second limit walker, a second retry-policy literal, a
second atomic writer, or a second TOPOLOGY_KEYS tuple is reintroduced.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli import backends as backends_mod
from hcli import machine as machine_mod
from hcli import mutation as mutation_mod
from hcli import persist as persist_mod
from hcli import resources as resources_mod
from hcli import runtime as runtime_mod
from hcli import scheduler as scheduler_mod
from hcli import verifier_pipeline as verifier_mod
from hcli import workunit as workunit_mod
from hcli.machine import MachineGenome, resolve_runtime_limits
from hcli.persist import atomic_write_json, atomic_write_text
from hcli.resources import resolve_active_decode_limit


class TestRetryPolicyOneLiteral(unittest.TestCase):
    def test_scheduler_reexports_workunit_caps(self):
        self.assertIs(scheduler_mod.MAX_REPAIR_DEPTH, workunit_mod.MAX_REPAIR_DEPTH)
        self.assertIs(
            scheduler_mod.MAX_REPAIRS_PER_ROOT, workunit_mod.MAX_REPAIRS_PER_ROOT
        )

    def test_scheduler_source_does_not_reassign_caps(self):
        src = Path(inspect.getsourcefile(scheduler_mod)).read_text(encoding="utf-8")
        self.assertNotIn("MAX_REPAIR_DEPTH = 3", src)
        self.assertNotIn("MAX_REPAIRS_PER_ROOT = 6", src)

    def test_remaining_depth_is_not_on_scheduler(self):
        self.assertFalse(hasattr(scheduler_mod, "_remaining_depth"))


class TestLimitOneAuthority(unittest.TestCase):
    def test_active_decode_matches_runtime_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = resolve_active_decode_limit(repo_root=tmp)
            b = resolve_runtime_limits(repo_root=tmp, start_dir=tmp)
            self.assertEqual(a[0], b.active_decode_limit)
            self.assertEqual(a[1], b.active_source)

    def test_resources_no_longer_walks_genome_files(self):
        src = Path(inspect.getsourcefile(resources_mod)).read_text(encoding="utf-8")
        self.assertIn("resolve_runtime_limits", src)
        self.assertNotIn("~/.config/hcli/machine_genome.json", src)


class TestAtomicWriteOneAuthority(unittest.TestCase):
    def test_resources_reexports_persist_writer(self):
        self.assertIs(resources_mod._atomic_write_text, persist_mod.atomic_write_text)

    def test_runtime_reexports_persist_writer(self):
        self.assertIs(runtime_mod._atomic_write, persist_mod.atomic_write_text)

    def test_atomic_write_text_survives_as_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.txt"
            calls = []
            real = os.replace

            def spy(src, dst, *args, **kwargs):
                calls.append((str(src), str(dst)))
                return real(src, dst, *args, **kwargs)

            os.replace = spy
            try:
                atomic_write_text(target, "ok\n")
            finally:
                os.replace = real
            self.assertTrue(calls)
            self.assertIn(".tmp", Path(calls[-1][0]).name)
            self.assertEqual(target.read_text(encoding="utf-8"), "ok\n")

    def test_atomic_write_json_uses_text_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.json"
            atomic_write_json(target, {"b": 2, "a": 1})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")), {"a": 1, "b": 2}
            )


class TestMachineGenomeIsNotAdmission(unittest.TestCase):
    def test_save_is_atomic(self):
        src = inspect.getsource(MachineGenome.save)
        self.assertIn("atomic_write_json", src)
        self.assertNotIn("write_text", src)

    def test_save_does_not_clobber_canonical_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            bag = Path(tmp) / "machine-genome.json"
            canonical = Path(tmp) / "machine_genome.json"
            canonical.write_text(json.dumps({"resident_runtime_limit": 4}))
            g = MachineGenome(bag)
            g.data = {"resident_runtime_limit": 1}
            g.save()
            self.assertTrue(bag.is_file())
            self.assertEqual(
                json.loads(canonical.read_text(encoding="utf-8"))[
                    "resident_runtime_limit"
                ],
                4,
            )

    def test_default_path_is_not_the_live_genome(self):
        path = machine_mod.get_machine_genome_path()
        self.assertNotEqual(
            path, Path.home() / ".config" / "hcli" / "machine_genome.json"
        )


class TestTopologyKeysOnce(unittest.TestCase):
    def test_one_tuple(self):
        src = Path(inspect.getsourcefile(runtime_mod)).read_text(encoding="utf-8")
        self.assertEqual(src.count("TOPOLOGY_KEYS = ("), 1)


class TestContextReexportSameObject(unittest.TestCase):
    def test_worker_packet_is_goal_worker_packet(self):
        from hcli.context import WorkerPacket as FromContext
        from hcli.goal import WorkerPacket as FromGoal

        self.assertIs(FromContext, FromGoal)


class TestReceiptExists(unittest.TestCase):
    def test_core_authorities_receipt_covers_every_concept(self):
        path = REPO / "receipts" / "headless" / "CORE_AUTHORITIES.json"
        self.assertTrue(path.is_file(), path)
        data = json.loads(path.read_text(encoding="utf-8"))
        concepts = {
            "Goal",
            "mission",
            "WorkUnit",
            "DAG",
            "scheduler",
            "checkpoint",
            "mutation lock",
            "verifier",
            "backend registry",
            "context budget",
            "runtime registry",
            "receipt",
            "experiment",
            "model identity",
            "machine identity",
            "status",
            "retry policy",
        }
        listed = {row["concept"] for row in data["authorities"]}
        self.assertTrue(concepts <= listed, concepts - listed)
        allowed = {
            "canonical",
            "compatibility",
            "obsolete",
            "test-only",
            "historical",
        }
        for row in data["authorities"]:
            self.assertIn(row["classification"], allowed, row["id"])
        self.assertTrue(data["consolidations"])


class TestEntropyExecuted(unittest.TestCase):
    """G027: DELETE candidates re-proven dead at HEAD stay gone."""

    def test_hcli_index_source_removed_from_this_tree(self):
        # An editable install of another checkout can still satisfy
        # ``import hcli.index``. This tree's source is the identity.
        self.assertFalse((REPO / "hcli" / "index.py").is_file())
        spec = importlib.util.find_spec("hcli.index")
        if spec is not None and spec.origin:
            self.assertNotEqual(
                Path(spec.origin).resolve(),
                (REPO / "hcli" / "index.py").resolve(),
            )

    def test_known_features_is_feature_aliases(self):
        self.assertFalse(hasattr(backends_mod, "KNOWN_FEATURES"))
        self.assertTrue(hasattr(backends_mod, "FEATURE_ALIASES"))

    def test_reserve_gib_constant_gone(self):
        self.assertFalse(hasattr(machine_mod, "DEFAULT_RESERVE_GIB"))

    def test_mutation_has_no_dead_validation_driver(self):
        self.assertFalse(hasattr(mutation_mod, "build_validation_plan"))
        self.assertFalse(hasattr(mutation_mod, "run_validation"))
        self.assertTrue(hasattr(mutation_mod, "discover_tests"))
        self.assertTrue(hasattr(mutation_mod, "validate_python_syntax"))

    def test_health_states_tuple_gone_named_states_remain(self):
        self.assertFalse(hasattr(resources_mod, "HEALTH_STATES"))
        self.assertEqual(resources_mod.STATE_HEALTHY, "healthy")
        self.assertEqual(resources_mod.STATE_DEGRADED, "degraded")
        self.assertEqual(resources_mod.STATE_CIRCUIT_OPEN, "circuit_open")

    def test_workunit_closed_set_is_transition_status(self):
        self.assertFalse(hasattr(workunit_mod, "WORKUNIT_STATUSES"))
        self.assertFalse(hasattr(workunit_mod, "CLASSIFICATION_VERIFIER_FAILURE"))
        self.assertEqual(workunit_mod.CLASSIFICATION_INTERRUPTED, "INTERRUPTED")

    def test_verifier_main_guard_lives_on_engine(self):
        self.assertFalse(hasattr(verifier_mod, "ast_has_main_guard"))
        self.assertFalse(hasattr(verifier_mod, "_assert_line_numbers"))
        self.assertFalse(hasattr(verifier_mod, "_if_is_main_guard"))
        from hcli import engine as engine_mod

        self.assertTrue(hasattr(engine_mod, "_if_is_main_guard"))
        self.assertTrue(hasattr(verifier_mod, "ast_has_tests"))

    def test_genome_path_helper_stays(self):
        path = machine_mod.get_machine_genome_path()
        self.assertNotEqual(
            path, Path.home() / ".config" / "hcli" / "machine_genome.json"
        )


if __name__ == "__main__":
    unittest.main()
