#!/usr/bin/env python3.12
"""Tests for Odyssey T0 machinery and contract closures.

Run from repo root:
  python3.12 -m pytest tools/odyssey/test_odyssey_t0.py -q
  # or without pytest:
  python3.12 tools/odyssey/test_odyssey_t0.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.odyssey import (  # noqa: E402
    contracts,
    data_verify,
    hidden_memberships,
    known_failures,
    runtime_authority,
    substrate_verify,
    tournament,
)
from tools.odyssey._paths import CHECKPOINTS, FENCE, TRAINING_DIR  # noqa: E402


class TestFenceUntouched(unittest.TestCase):
    def test_fence_is_false(self):
        self.assertTrue(FENCE.is_file())
        self.assertEqual(FENCE.read_text().strip().lower(), "false")


class TestSubstrate(unittest.TestCase):
    def test_static_checks_refuse_the_evicted_substrate(self):
        result = substrate_verify.static_checks()
        # The sealed index/manifest remain on disk, while body shards were
        # deliberately evicted under the 90-GB storage law.  Never allow the
        # historical receipt/progress file to turn this into a current pass.
        self.assertFalse(result["ok"], msg=json.dumps(result["checks"], indent=2))
        self.assertEqual(result["decision_count"], 59585)
        present = next(c for c in result["checks"] if c["check"] == "shards_present_on_disk")
        self.assertFalse(present["ok"])
        self.assertIn("/282", present["detail"])

    def test_verify_one_shard_refuses_before_body_hashing(self):
        result = substrate_verify.verify_shards(max_shards=1, resume=True)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["static"]["ok"])
        self.assertIsNone(result["shard_verification"])


class TestData(unittest.TestCase):
    def test_data_verify_classifies_missing(self):
        result = data_verify.verify_all()
        self.assertEqual(result["status"], "PASS")
        # Training corpora are declared but not collected.
        missing = [
            i for i in result["data"]["items"] if i["status"] == "DECLARED_NOT_PRESENT"
        ]
        self.assertGreaterEqual(len(missing), 1)
        # Teacher ledger should be present on this machine.
        teacher_present = [
            i for i in result["teacher_traces"]["items"] if i["status"] == "PRESENT"
        ]
        self.assertGreaterEqual(len(teacher_present), 1)


class TestRuntime(unittest.TestCase):
    def test_runtime_bit_identical(self):
        result = runtime_authority.verify_runtime()
        self.assertEqual(result["status"], "PASS", msg=json.dumps(result, indent=2, default=str))
        art = result["artifact_single_layer"]
        self.assertTrue(art["bit_identical_two_runs"])
        self.assertEqual(art["logit_sha256_run1"], art["logit_sha256_run2"])


class TestKnownFailures(unittest.TestCase):
    def test_registry(self):
        reg = known_failures.build_registry()
        self.assertEqual(reg["status"], "PASS", msg=json.dumps(reg, indent=2))
        self.assertGreaterEqual(reg["n_entries"], 5)


class TestHiddenMemberships(unittest.TestCase):
    def test_hidden_memberships(self):
        hidden_memberships.write_seed_sets()
        result = hidden_memberships.verify_commitment()
        self.assertEqual(result["status"], "PASS")
        tv = hidden_memberships.load_training_visible()
        self.assertIsNone(tv["hidden_item_ids"])
        self.assertTrue(all(p["set"] == "selection" for p in tv["public_selection"]))


class TestTournament(unittest.TestCase):
    def test_tournament(self):
        dims = tournament.HALO_DIMENSIONS
        inc = tournament.Scorecard("inc", 0.7, {d: 0.5 for d in dims})
        # Tie → incumbent
        ch = tournament.Scorecard("new", 0.7, {d: 0.5 for d in dims})
        r = tournament.compare(inc, ch)
        self.assertEqual(r["winner"], "inc")
        # Strict improvement both axes
        ch2 = tournament.Scorecard("better", 0.8, {d: 0.6 for d in dims})
        r2 = tournament.compare(inc, ch2)
        self.assertEqual(r2["winner"], "better")
        # Math better, coding regresses → incumbent
        bad = {d: 0.6 for d in dims}
        bad["coding"] = 0.1
        ch3 = tournament.Scorecard("regress", 0.99, bad)
        r3 = tournament.compare(inc, ch3)
        self.assertEqual(r3["winner"], "inc")


class TestContracts(unittest.TestCase):
    def test_objective_weights(self):
        w = contracts.objective_weights()
        self.assertEqual(w["status"], "RUNNABLE")
        self.assertAlmostEqual(sum(w["weights"].values()), 1.0, places=6)
        self.assertTrue(contracts.assert_objective_not_using_hidden())

    def test_profile_support_halo(self):
        r = contracts.profile_support_halo_contract()
        self.assertEqual(r["status"], "RUNNABLE", msg=r)

    def test_qat_sim(self):
        r = contracts.qat_simulate_step()
        self.assertEqual(r["status"], "RUNNABLE")
        self.assertGreaterEqual(r["mse"], 0.0)

    def test_trajectory_metrics(self):
        r = contracts.trajectory_divergence([1, 2, 3, 4], [1, 2, 9, 4])
        self.assertEqual(r["status"], "RUNNABLE")
        self.assertEqual(r["first_divergence_index"], 2)

    def test_checkpoint_controller(self):
        ckpt = contracts.create_checkpoint(stage="T0", step=0)
        self.assertTrue(contracts.validate_checkpoint(ckpt))
        self.assertTrue((CHECKPOINTS / "CURRENT").is_file())

    def test_resume(self):
        contracts.create_checkpoint(stage="T0", step=3)
        r = contracts.resume_from_current()
        self.assertEqual(r["status"], "READY")
        self.assertEqual(r["next_step"], 4)

    def test_rollback(self):
        a = contracts.create_checkpoint(stage="T0", step=1)
        b = contracts.create_checkpoint(stage="T0", step=2, parent_sha256=a["checkpoint_id"])
        r = contracts.rollback_to(a["checkpoint_id"])
        self.assertEqual(r["status"], "RUNNABLE")
        self.assertEqual((CHECKPOINTS / "CURRENT").read_text().strip(), a["checkpoint_id"])
        # silence unused
        self.assertTrue(b["checkpoint_id"])

    def test_forge_sovereignty(self):
        f = contracts.forge_gate()
        self.assertEqual(f["status"], "RUNNABLE")
        self.assertEqual(f["F1_F4"]["status"], "NOT_IMPLEMENTABLE_HERE")
        s = contracts.sovereignty_gate()
        self.assertEqual(s["status"], "RUNNABLE")

    def test_resource_scheduler(self):
        r1 = contracts.resource_scheduler_admit({"heavy": True, "memory_bytes": 1 << 30})
        self.assertTrue(r1["admit"])
        r2 = contracts.resource_scheduler_admit(
            {"heavy": True, "memory_bytes": 1 << 30}, state=r1["state"]
        )
        self.assertFalse(r2["admit"])  # max concurrent heavy = 1


class TestRunnerRefusesTraining(unittest.TestCase):
    def test_run_py_refuses(self):
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(TRAINING_DIR / "run.py")],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("ODYSSEY_LAUNCH_AUTHORIZED=false", proc.stderr)



class TestSubstrateCapabilityGate(unittest.TestCase):
    """The second gate. The fence answers 'did a human authorize a run'; this
    answers 'can the artifact generate'. Before it existed the fence was the only
    thing between an agent and training on a substrate that cannot complete
    '2 + 2 ='. These pin it so it is not quietly removed."""

    def test_math_preserve_is_refused(self):
        from tools.odyssey.substrate_capability import SubstrateRefused, assert_trainable
        from tools.odyssey._paths import EXPECTED_INDEX_SHA256

        with self.assertRaises(SubstrateRefused):
            assert_trainable(EXPECTED_INDEX_SHA256)

    def test_unknown_artifact_is_refused_not_permitted(self):
        """Silence is not a pass. An artifact nobody probed is not one known to work."""
        from tools.odyssey.substrate_capability import SubstrateRefused, assert_trainable, verdict_for

        unknown = "0" * 64
        self.assertEqual(verdict_for(unknown)["capability_verdict"], "UNVERIFIED")
        with self.assertRaises(SubstrateRefused):
            assert_trainable(unknown)

    def test_runner_refuses_even_when_the_fence_is_true(self):
        """The whole point: the two gates are independent."""
        import subprocess, sys
        from tools.odyssey._paths import FENCE, ROOT, TRAINING_DIR

        original = FENCE.read_text()
        try:
            FENCE.write_text("true")
            r = subprocess.run(
                [sys.executable, str(TRAINING_DIR / "run.py"), "T1"],
                capture_output=True, text=True, cwd=str(ROOT),
            )
            self.assertEqual(r.returncode, 5, f"expected SUBSTRATE_REFUSED, got {r.returncode}: {r.stderr}")
            self.assertIn("SUBSTRATE_REFUSED", r.stderr)
        finally:
            FENCE.write_text(original)
        self.assertEqual(FENCE.read_text().strip(), "false", "the fence must be left false")


if __name__ == "__main__":
    unittest.main(verbosity=2)
