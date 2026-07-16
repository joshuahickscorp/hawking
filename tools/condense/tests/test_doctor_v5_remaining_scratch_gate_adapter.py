from __future__ import annotations

import copy
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import doctor_v5_remaining_scratch_gate_adapter as gate
import doctor_v5_remaining_scratch_ledger as ledger
from test_doctor_v5_remaining_scratch_ledger import Fixture as LedgerFixture


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic(value: dict, field: str) -> dict:
    value[field] = hashlib.sha256(json.dumps(
        {name: child for name, child in value.items() if name != field},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    return value


class Fixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worker = LedgerFixture(
            self.root, mode="resident", scratch=gate.DECLARED_SCRATCH_BYTES)
        self.projected = 20
        self.plan_path = self.root / "campaign_plan.json"
        self.cell_identity = "8" * 64
        self.plan = semantic({"schema": "plan", "cells": [{
            "cell_id": gate.TARGET_CELL_ID,
            "cell_identity_sha256": self.cell_identity,
            "projected_output_bytes": self.projected,
            "source_deletion_permitted": False,
        }]}, "plan_sha256")
        write_json(self.plan_path, self.plan)
        self.binding = gate.FrozenGateBinding(
            plan_path=self.plan_path, plan_sha256=self.plan["plan_sha256"],
            cell_id=gate.TARGET_CELL_ID,
            cell_identity_sha256=self.cell_identity,
            worker_request_path=self.worker.request_path,
            worker_request_sha256=hashlib.sha256(
                self.worker.request_path.read_bytes()).hexdigest(),
            worker_checkpoint_sha256=hashlib.sha256(
                self.worker.checkpoint_path.read_bytes()).hexdigest(),
            declared_scratch_bytes=gate.DECLARED_SCRATCH_BYTES,
            disk_reserve_bytes=gate.DISK_RESERVE_BYTES,
            projected_packed_output_bytes=self.projected,
        )
        self.receipt = ledger.build_ledger(
            self.worker.request_path,
            projected_packed_output_bytes=self.projected,
            workspace_root=self.root,
        )

    def close(self) -> None:
        self.temporary.cleanup()

    def evaluate(self, receipt=None, **kwargs):
        return gate._evaluate_gate_for_test(
            self.binding, self.receipt if receipt is None else receipt,
            free_bytes=kwargs.pop("free_bytes", 250_000_000_000),
            workspace_root=self.root, **kwargs,
        )


class RemainingScratchGateAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def assert_fallback(self, decision: dict) -> None:
        self.assertEqual("conservative-full-scratch-fallback", decision["mode"])
        self.assertFalse(decision["phase_aware_credit_applied"])
        self.assertEqual(gate.DECLARED_SCRATCH_BYTES,
                         decision["scratch_bytes_charged"])
        self.assertEqual(self.fixture.projected,
                         decision["packed_output_bytes_charged"])
        self.assertEqual(gate.DISK_RESERVE_BYTES + gate.DECLARED_SCRATCH_BYTES
                         + self.fixture.projected,
                         decision["required_free_bytes"])
        self.assertEqual([], gate.validate_decision(decision, self.fixture.binding))

    def test_fresh_recomputed_receipt_is_the_only_phase_aware_path(self) -> None:
        decision = self.fixture.evaluate()
        self.assertEqual("validated-phase-aware-remaining-scratch", decision["mode"])
        self.assertTrue(decision["phase_aware_credit_applied"])
        self.assertTrue(decision["presented_receipt_consumed"])
        self.assertTrue(decision["receipt_recomputed_from_frozen_sources"])
        self.assertEqual(self.fixture.receipt["remaining_scratch_bytes"],
                         decision["scratch_bytes_charged"])
        self.assertEqual(self.fixture.receipt["projected_remaining_packed_output_bytes"],
                         decision["packed_output_bytes_charged"])
        self.assertEqual([], gate.validate_decision(decision, self.fixture.binding))

    def test_absent_or_caller_reduction_has_full_48gb_fallback(self) -> None:
        self.assert_fallback(gate._evaluate_gate_for_test(
            self.fixture.binding, None, free_bytes=250_000_000_000,
            workspace_root=self.fixture.root))
        signature = inspect.signature(gate.evaluate_production_gate)
        self.assertNotIn("remaining_scratch_bytes", signature.parameters)
        self.assertNotIn("scratch_reduction_bytes", signature.parameters)
        self.assertNotIn("free_bytes", signature.parameters)
        self.assertNotIn("ledger_builder", signature.parameters)

    def test_stale_or_resealed_reduced_receipt_falls_back(self) -> None:
        stale = copy.deepcopy(self.fixture.receipt)
        stale["observed_at"] = (dt.datetime.now(dt.timezone.utc)
                                - dt.timedelta(hours=1)).isoformat()
        stale["receipt_sha256"] = ledger._hash_value(
            ledger._without(stale, "receipt_sha256"))
        self.assert_fallback(self.fixture.evaluate(stale))
        reduced = copy.deepcopy(self.fixture.receipt)
        reduced["remaining_scratch_bytes"] -= 1
        reduced["required_free_bytes"] -= 1
        reduced["receipt_sha256"] = ledger._hash_value(
            ledger._without(reduced, "receipt_sha256"))
        self.assert_fallback(self.fixture.evaluate(reduced))

    def test_checkpoint_drift_falls_back_even_if_presented_receipt_is_old_green(self) -> None:
        checkpoint = self.fixture.worker.checkpoint
        checkpoint["updated_at"] = "changed"
        self.fixture.worker.rewrite_checkpoint()
        decision = self.fixture.evaluate()
        self.assert_fallback(decision)
        self.assertTrue(any("checkpoint" in row for row in decision["fallback_reasons"]))

    def test_request_drift_falls_back_even_if_checkpoint_is_rebound(self) -> None:
        self.fixture.worker.request["label"] = "changed"
        self.fixture.worker.rewrite_request()
        decision = self.fixture.evaluate()
        self.assert_fallback(decision)
        self.assertTrue(any("request" in row for row in decision["fallback_reasons"]))

    def test_artifact_path_drift_falls_back(self) -> None:
        checkpoint = self.fixture.worker.checkpoint
        checkpoint["units"]["decode:00000"]["artifact"]["path"] = str(
            self.fixture.worker.source / "model-00000.safetensors")
        self.fixture.worker.rewrite_checkpoint()
        self.assert_fallback(self.fixture.evaluate())

    def test_ordinal_or_checkpoint_plan_drift_falls_back(self) -> None:
        checkpoint = self.fixture.worker.checkpoint
        checkpoint["plan"].append("receipt")
        self.fixture.worker.rewrite_checkpoint()
        self.assert_fallback(self.fixture.evaluate())

    def test_campaign_plan_or_cell_projection_drift_falls_back(self) -> None:
        self.fixture.plan["cells"][0]["projected_output_bytes"] += 1
        semantic(self.fixture.plan, "plan_sha256")
        write_json(self.fixture.plan_path, self.fixture.plan)
        self.assert_fallback(self.fixture.evaluate())

    def test_receipt_from_different_request_falls_back(self) -> None:
        presented = copy.deepcopy(self.fixture.receipt)
        presented["request"]["sha256"] = "a" * 64
        presented["receipt_sha256"] = ledger._hash_value(
            ledger._without(presented, "receipt_sha256"))
        self.assert_fallback(self.fixture.evaluate(presented))

    def test_invalid_free_space_never_admits(self) -> None:
        decision = self.fixture.evaluate(free_bytes=None)
        self.assertFalse(decision["capacity_ok"])
        self.assertEqual([], gate.validate_decision(decision, self.fixture.binding))

    def test_naive_test_clock_falls_back_instead_of_raising(self) -> None:
        decision = self.fixture.evaluate(now=dt.datetime.now())
        self.assert_fallback(decision)
        self.assertTrue(any("naive" in row for row in decision["fallback_reasons"]))

    def test_resealed_unknown_decision_field_is_rejected(self) -> None:
        decision = self.fixture.evaluate()
        decision["unknown"] = True
        decision["decision_sha256"] = gate._hash_value(
            gate._without(decision, "decision_sha256"))
        self.assertTrue(any("keys" in row
                            for row in gate.validate_decision(
                                decision, self.fixture.binding)))


if __name__ == "__main__":
    unittest.main()
