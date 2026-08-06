"""Unit tests for notification event vocabulary and completion-authority rule."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OPS = Path(__file__).resolve().parents[1]
_ROOT = _OPS.parents[2]
for p in (str(_OPS.parent), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ascension.notifications import (  # noqa: E402
    AuthoritySource,
    NotificationBus,
    NotificationKind,
    build_notification,
    may_declare_completion,
)


class TestSection28Kinds(unittest.TestCase):
    def test_required_kinds_exist(self):
        required = {
            "tg_rung_candidate",
            "tg3_review_required",
            "parity_rejection",
            "reviewer_disagreement",
            "repeated_failure",
            "memory_disk_pressure",
            "new_model_admitted",
            "benchmark_complete",
            "human_decision_required",
        }
        values = {k.value for k in NotificationKind}
        self.assertTrue(required.issubset(values))


class TestCompletionAuthority(unittest.TestCase):
    def test_sandbox_model_cannot_complete_benchmark(self):
        ok, reason = may_declare_completion(
            kind=NotificationKind.BENCHMARK_COMPLETE,
            authority=AuthoritySource.SANDBOX_MODEL,
            evidence_paths=["/tmp/fake.json"],
        )
        self.assertFalse(ok)
        self.assertIn("sandbox_model", reason or "")

    def test_sandbox_model_cannot_admit_model(self):
        ok, _ = may_declare_completion(
            kind=NotificationKind.NEW_MODEL_ADMITTED,
            authority=AuthoritySource.SANDBOX_MODEL,
            evidence_paths=["receipt.json"],
        )
        self.assertFalse(ok)

    def test_sealed_receipt_ok_with_evidence(self):
        ok, reason = may_declare_completion(
            kind=NotificationKind.BENCHMARK_COMPLETE,
            authority=AuthoritySource.INDEPENDENT_HARNESS,
            evidence_paths=["/tmp/PROTO_VERIFY.json"],
        )
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_human_ok_without_evidence(self):
        ok, _ = may_declare_completion(
            kind=NotificationKind.TG_RUNG_CANDIDATE,
            authority=AuthoritySource.HUMAN,
            evidence_paths=(),
        )
        self.assertTrue(ok)

    def test_non_completion_kinds_always_ok(self):
        ok, _ = may_declare_completion(
            kind=NotificationKind.PARITY_REJECTION,
            authority=AuthoritySource.SANDBOX_MODEL,
        )
        self.assertTrue(ok)

    def test_build_refuses_sandbox_completion(self):
        ev = build_notification(
            NotificationKind.BENCHMARK_COMPLETE,
            "model claims done",
            authority=AuthoritySource.SANDBOX_MODEL,
            evidence_paths=["x.json"],
        )
        self.assertFalse(ev.may_send)
        self.assertIsNotNone(ev.refuse_reason)

    def test_build_pressure_defaults_warn(self):
        ev = build_notification(
            NotificationKind.MEMORY_DISK_PRESSURE,
            "disk 18G free",
            authority=AuthoritySource.PRESSURE_GOVERNOR,
        )
        self.assertEqual(ev.severity, "warn")
        self.assertTrue(ev.may_send)

    def test_bus_separates_refused(self):
        bus = NotificationBus()
        self.assertFalse(
            bus.publish_built(
                NotificationKind.NEW_MODEL_ADMITTED,
                "sandbox says admitted",
                authority=AuthoritySource.SANDBOX_MODEL,
                evidence_paths=["e.json"],
            )
        )
        self.assertEqual(len(bus.refused), 1)
        self.assertEqual(len(bus.sent), 0)
        self.assertTrue(
            bus.publish_built(
                NotificationKind.REPEATED_FAILURE,
                "same fail ×3",
                authority=AuthoritySource.SUPERVISOR,
                repeated_count=3,
            )
        )
        self.assertEqual(len(bus.sent), 1)

    def test_render_text(self):
        ev = build_notification(
            NotificationKind.TG3_REVIEW_REQUIRED,
            "30B gauntlet candidate",
            authority=AuthoritySource.SUPERVISOR,
            evidence_paths=["rung.json"],
        )
        text = ev.render_text()
        self.assertIn("TG3_REVIEW_REQUIRED", text)
        self.assertIn("30B gauntlet candidate", text)


if __name__ == "__main__":
    unittest.main()
