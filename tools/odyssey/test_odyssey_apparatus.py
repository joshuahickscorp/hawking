#!/usr/bin/env python3.12
"""Tests for the Odyssey trainer apparatus on FIXTURE toy models.

These prove the apparatus runs — not that anything real was trained.

  python3.12 -m pytest tools/odyssey/test_odyssey_apparatus.py -q
  python3.12 tools/odyssey/test_odyssey_apparatus.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.odyssey.checkpoints import (  # noqa: E402
    CheckpointError,
    CheckpointStore,
    content_id,
)
from tools.odyssey.objectives import (  # noqa: E402
    UnregisteredObjective,
    get_objective,
    list_objectives,
    require_objective,
)
from tools.odyssey.qat import FakeUniformQuantizer, simulate_qat_step  # noqa: E402
from tools.odyssey.receipts import make_receipt, verify_receipt  # noqa: E402
from tools.odyssey.rollback_exec import execute_rollback, rollback_and_restore_model  # noqa: E402
from tools.odyssey.scheduler import declare_and_admit, declare_resources  # noqa: E402
from tools.odyssey import t0_run  # noqa: E402
from tools.odyssey.toy_model import ToyConfig, ToyMLP  # noqa: E402
from tools.odyssey.tournament import HALO_DIMENSIONS, Scorecard, compare  # noqa: E402
from tools.odyssey.trainer import ToyTrainer, run_uninterrupted_vs_resumed  # noqa: E402
from tools.odyssey.trajectory import (  # noqa: E402
    TEACHER_LEDGER_FACTS,
    TrajectoryTracesMissing,
    require_parent_traces_or_refuse,
    trajectory_loss_shape,
)
from tools.odyssey._paths import FENCE  # noqa: E402


class TestFenceUntouched(unittest.TestCase):
    def test_fence_still_false(self):
        self.assertEqual(FENCE.read_text().strip().lower(), "false")


class TestToyModel(unittest.TestCase):
    def test_param_count_is_small(self):
        m = ToyMLP(ToyConfig())
        self.assertGreater(m.n_params(), 100)
        self.assertLess(m.n_params(), 50_000)

    def test_state_roundtrip_bit_identical(self):
        m = ToyMLP(ToyConfig(seed=11))
        for _ in range(3):
            m.train_step()
        h1 = m.state_sha256()
        m2 = ToyMLP.from_state_dict(m.state_dict())
        self.assertEqual(m2.state_sha256(), h1)
        m2.train_step()
        m.train_step()
        self.assertEqual(m.state_sha256(), m2.state_sha256())


class TestT0UnitCallable(unittest.TestCase):
    def test_run_unit_callable(self):
        result = t0_run.run_unit(include_runtime=True)
        self.assertEqual(result["status"], "PASS", msg=json.dumps(result["summary"], indent=2))
        self.assertEqual(result["mode"], "unit")
        self.assertFalse(result["launch_authorized"])
        self.assertEqual(result["summary"]["substrate_static"], "PASS")
        self.assertEqual(result["summary"]["data"], "PASS")


class TestContentAddressedCheckpoints(unittest.TestCase):
    def test_identical_states_same_id_different_states_different_id(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStore(Path(td))
            m = ToyMLP(ToyConfig(seed=1))
            m.train_step()
            state = m.state_dict()
            a = store.save(
                stage="FIXTURE",
                step=m.step,
                objective="capability_weighted_ce",
                model_state=state,
            )
            b = store.save(
                stage="FIXTURE",
                step=m.step,
                objective="capability_weighted_ce",
                model_state=state,
            )
            self.assertEqual(a["checkpoint_id"], b["checkpoint_id"])
            m.train_step()
            c = store.save(
                stage="FIXTURE",
                step=m.step,
                objective="capability_weighted_ce",
                model_state=m.state_dict(),
            )
            self.assertNotEqual(c["checkpoint_id"], a["checkpoint_id"])
            # Id IS content hash
            payload = store.load_state(a["checkpoint_id"])
            self.assertEqual(content_id(payload), a["checkpoint_id"])


class TestResumeBitIdentity(unittest.TestCase):
    def test_kill_resume_matches_uninterrupted(self):
        """The test that matters: resume is bit-identical, not merely 'continues'."""
        with tempfile.TemporaryDirectory() as td:
            result = run_uninterrupted_vs_resumed(
                Path(td), total_steps=8, kill_after_step=4, cfg=ToyConfig(seed=7)
            )
            self.assertTrue(
                result["bit_identical"],
                msg=json.dumps(result, indent=2),
            )
            self.assertEqual(result["uninterrupted_end_step"], 8)
            self.assertEqual(result["resumed_end_step"], 8)
            self.assertEqual(
                result["uninterrupted_state_sha256"],
                result["resumed_state_sha256"],
            )


class TestRollbackExecutable(unittest.TestCase):
    def test_rollback_records_event_restores_and_reruns_gate(self):
        with tempfile.TemporaryDirectory() as td:
            t = ToyTrainer(Path(td), cfg=ToyConfig(seed=3), checkpoint_every=1)
            ids = {}
            for target in range(1, 6):
                t.run(
                    total_steps=target,
                    stage="FIXTURE",
                    resume=(target > 1),
                    emit_receipt=False,
                )
                ids[t.model.step] = t.last_checkpoint_id
            early = ids[2]
            result = rollback_and_restore_model(t, early, reason="test_regression")
            self.assertEqual(result["status"], "PROVEN", msg=json.dumps(result, indent=2))
            self.assertEqual(t.model.step, 2)
            self.assertEqual(t.store.current_id(), early)
            self.assertTrue(t.store.events_path.is_file())
            events = t.store.events_path.read_text().strip().splitlines()
            self.assertGreaterEqual(len(events), 1)
            ev = json.loads(events[-1])
            self.assertEqual(ev["kind"], "rollback")
            self.assertEqual(ev["restored"], early)
            self.assertFalse(ev["silent_substitution"])
            self.assertTrue(result["entry_gate"]["ok"])


class TestObjectiveRegistry(unittest.TestCase):
    def test_registered_resolves(self):
        obj = require_objective("capability_weighted_ce", stage="FIXTURE")
        self.assertEqual(obj["name"], "capability_weighted_ce")
        self.assertIn("schema", obj)

    def test_unregistered_refused_not_defaulted(self):
        with self.assertRaises(UnregisteredObjective):
            get_objective("definitely_not_registered_objective")
        with self.assertRaises(UnregisteredObjective):
            require_objective("definitely_not_registered_objective")

    def test_list_nonempty(self):
        self.assertIn("trajectory_divergence", list_objectives())


class TestQATFake(unittest.TestCase):
    def test_fake_quantizer_honestly_named(self):
        q = FakeUniformQuantizer()
        self.assertTrue(q.is_fake)
        self.assertIn("Fake", q.name)
        out = simulate_qat_step()
        self.assertEqual(out["kind"], "fake_simulation")
        self.assertIs(out["is_measurement"], False)
        self.assertTrue(out["quantizer_is_fake"])
        self.assertIn("not a qat measurement", out["honesty"].lower())


class TestTrajectoryInterface(unittest.TestCase):
    def test_shape_and_missing_traces(self):
        r = trajectory_loss_shape([1, 2, 3, 4], [1, 2, 9, 4], source="fixture_tokens")
        self.assertEqual(r["first_divergence_index"], 2)
        self.assertEqual(r["teacher_ledger"]["trajectory_traces"], 0)
        self.assertEqual(TEACHER_LEDGER_FACTS["ledger_lines"], 122)
        self.assertEqual(TEACHER_LEDGER_FACTS["per_layer_captures"], 118)
        with self.assertRaises(TrajectoryTracesMissing):
            require_parent_traces_or_refuse()


class TestTournamentRegressions(unittest.TestCase):
    def test_injected_regression_keeps_incumbent(self):
        dims = HALO_DIMENSIONS
        inc = Scorecard("incumbent", 0.7, {d: 0.5 for d in dims})
        bad = {d: 0.6 for d in dims}
        bad["coding"] = 0.1
        ch = Scorecard("regressed", 0.99, bad)
        r = compare(inc, ch)
        self.assertEqual(r["winner"], "incumbent")
        self.assertTrue(r["support_halo"]["regressions"])

    def test_tie_to_incumbent(self):
        dims = HALO_DIMENSIONS
        inc = Scorecard("incumbent", 0.7, {d: 0.5 for d in dims})
        ch = Scorecard("tie", 0.7, {d: 0.5 for d in dims})
        self.assertEqual(compare(inc, ch)["winner"], "incumbent")

    def test_strict_win(self):
        dims = HALO_DIMENSIONS
        inc = Scorecard("incumbent", 0.7, {d: 0.5 for d in dims})
        ch = Scorecard("better", 0.8, {d: 0.6 for d in dims})
        self.assertEqual(compare(inc, ch)["winner"], "better")


class TestFailureInjection(unittest.TestCase):
    def test_corrupt_checkpoint_detected(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStore(Path(td), shard_size=64)
            m = ToyMLP(ToyConfig(seed=2))
            m.train_step()
            meta = store.save(
                stage="FIXTURE",
                step=m.step,
                objective="capability_weighted_ce",
                model_state=m.state_dict(),
            )
            store.inject_corrupt_shard(meta["checkpoint_id"], 0)
            with self.assertRaises(CheckpointError) as cm:
                store.load_state(meta["checkpoint_id"])
            self.assertIn("CORRUPT", str(cm.exception))

    def test_lost_shard_detected(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStore(Path(td), shard_size=64)
            m = ToyMLP(ToyConfig(seed=2))
            for _ in range(3):
                m.train_step()
            meta = store.save(
                stage="FIXTURE",
                step=m.step,
                objective="capability_weighted_ce",
                model_state=m.state_dict(),
            )
            store.inject_lose_shard(meta["checkpoint_id"], 0)
            with self.assertRaises(CheckpointError) as cm:
                store.load_state(meta["checkpoint_id"])
            self.assertIn("MISSING_SHARD", str(cm.exception))

    def test_mid_write_detected(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStore(Path(td))
            m = ToyMLP(ToyConfig(seed=2))
            m.train_step()
            cid = store.inject_kill_mid_write(
                stage="FIXTURE",
                step=1,
                objective="capability_weighted_ce",
                model_state=m.state_dict(),
            )
            with self.assertRaises(CheckpointError) as cm:
                store.load_state(cid)
            msg = str(cm.exception)
            self.assertTrue(
                "INCOMPLETE" in msg or "CORRUPT" in msg or "HASH" in msg or "MISSING" in msg,
                msg=msg,
            )


class TestResourceScheduler(unittest.TestCase):
    def test_declares_wall_rss_threads(self):
        d = declare_resources(
            stage="FIXTURE",
            wall_time_budget_s=30.0,
            rss_budget_bytes=128 * 1024 * 1024,
            threads=2,
        )
        self.assertEqual(d["declared"]["wall_time_budget_s"], 30.0)
        self.assertEqual(d["declared"]["rss_budget_bytes"], 128 * 1024 * 1024)
        self.assertEqual(d["declared"]["threads"], 2)
        r = declare_and_admit(
            stage="FIXTURE",
            wall_time_budget_s=30.0,
            rss_budget_bytes=64 * 1024 * 1024,
            threads=1,
            heavy=False,
            memory_bytes=32 * 1024 * 1024,
        )
        self.assertEqual(r["status"], "ADMITTED")


class TestTrainingReceipts(unittest.TestCase):
    def test_receipt_content_addressed_and_fixture_labelled(self):
        with tempfile.TemporaryDirectory() as td:
            t = ToyTrainer(Path(td), cfg=ToyConfig(seed=5), checkpoint_every=1)
            result = t.run(total_steps=3, stage="FIXTURE")
            rec = result["receipt"]
            self.assertTrue(rec["fixture"])
            self.assertIn("FIXTURE", rec["fixture_label"])
            self.assertEqual(len(rec["receipt_id"]), 64)
            self.assertTrue(verify_receipt(rec))
            # Same body → same id
            rec2 = make_receipt(
                stage=rec["stage"],
                status=rec["status"],
                checkpoint_id=rec["checkpoint_id"],
                objective=rec["objective"],
                steps_completed=rec["steps_completed"],
                state_sha256=rec["state_sha256"],
                fixture=True,
                details=rec["details"],
                parent_receipt_id=rec.get("parent_receipt_id"),
            )
            self.assertEqual(rec2["receipt_id"], rec["receipt_id"])


class TestApparatusEndToEnd(unittest.TestCase):
    def test_apparatus_all_surfaces(self):
        from tools.odyssey.apparatus import run_apparatus

        out = run_apparatus(write_receipts=True)
        self.assertTrue(out["all_pass"], msg=json.dumps(out["pass_map"], indent=2))
        # Deliverables exist
        for name in (
            "ODYSSEY_T0_EXECUTABLE_RECEIPT.json",
            "ODYSSEY_TRAINER_READINESS.json",
            "ODYSSEY_HEAVY_PREREQUISITES.json",
        ):
            path = ROOT / "odyssey" / name
            self.assertTrue(path.is_file(), msg=name)
            doc = json.loads(path.read_text())
            self.assertIn("schema", doc)
        readiness = json.loads((ROOT / "odyssey" / "ODYSSEY_TRAINER_READINESS.json").read_text())
        self.assertTrue(readiness["never_trained_anything_real"])
        self.assertIn("NEVER trained anything real", readiness["plain_statement"])
        # Fence untouched
        self.assertEqual(FENCE.read_text().strip().lower(), "false")


if __name__ == "__main__":
    unittest.main(verbosity=2)
