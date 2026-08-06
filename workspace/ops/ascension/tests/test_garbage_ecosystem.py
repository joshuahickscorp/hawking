"""Unit tests for the garbage ecosystem 4-state classifier and auto-delete gates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow `python -m unittest` from repo root or this directory.
_OPS = Path(__file__).resolve().parents[1]
_ROOT = _OPS.parents[2]
for p in (str(_OPS.parent), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ascension.garbage_ecosystem import (  # noqa: E402
    ObjectClass,
    build_cleanup_receipt,
    classify_object,
    evaluate_auto_delete,
    never_auto_delete_reason,
)


SANDBOX = "/tmp/ascension-sandbox"
EVICTABLE = f"{SANDBOX}/hf-cache/donor-shard"
PINNED_PROTO = "/Users/scammermike/Desktop/hawking-frankenstein/proto-frankenstein"
LEASED = f"{SANDBOX}/active-checkpoint"
UNKNOWN = "/tmp/unclassified-random-dir"


class TestNeverAutoDelete(unittest.TestCase):
    def test_frankenstein_blocked(self):
        reason = never_auto_delete_reason(PINNED_PROTO)
        self.assertIsNotNone(reason)
        self.assertIn("frankenstein", reason.lower())

    def test_rollback_blocked(self):
        reason = never_auto_delete_reason(f"{SANDBOX}/sole-rollback/weights")
        self.assertIsNotNone(reason)

    def test_plain_cache_ok(self):
        self.assertIsNone(never_auto_delete_reason(EVICTABLE))


class TestClassify(unittest.TestCase):
    def test_explicit_pinned(self):
        rec = classify_object(
            f"{SANDBOX}/stable-model",
            sandbox_roots=[SANDBOX],
            pinned_paths=[f"{SANDBOX}/stable-model"],
        )
        self.assertEqual(rec.object_class, ObjectClass.PINNED)
        self.assertTrue(rec.sandbox_owned)

    def test_frankenstein_is_pinned(self):
        rec = classify_object(PINNED_PROTO, sandbox_roots=[SANDBOX])
        self.assertEqual(rec.object_class, ObjectClass.PINNED)

    def test_worktree_is_pinned(self):
        rec = classify_object(
            "/Users/scammermike/.claude-grok/worktrees/some-lane",
            sandbox_roots=[SANDBOX],
        )
        self.assertEqual(rec.object_class, ObjectClass.PINNED)

    def test_leased_by_references(self):
        rec = classify_object(
            f"{SANDBOX}/trace",
            sandbox_roots=[SANDBOX],
            active_references=2,
        )
        self.assertEqual(rec.object_class, ObjectClass.LEASED)

    def test_explicit_leased(self):
        rec = classify_object(
            LEASED,
            sandbox_roots=[SANDBOX],
            leased_paths=[LEASED],
        )
        self.assertEqual(rec.object_class, ObjectClass.LEASED)

    def test_evictable_under_sandbox(self):
        rec = classify_object(
            EVICTABLE,
            sandbox_roots=[SANDBOX],
            evictable_paths=[EVICTABLE],
        )
        self.assertEqual(rec.object_class, ObjectClass.EVICTABLE)
        self.assertTrue(rec.sandbox_owned)

    def test_evictable_outside_sandbox_quarantined(self):
        outside = "/var/tmp/not-sandbox/cache"
        rec = classify_object(
            outside,
            sandbox_roots=[SANDBOX],
            evictable_paths=[outside],
        )
        self.assertEqual(rec.object_class, ObjectClass.QUARANTINED)

    def test_unclassified_is_quarantined(self):
        rec = classify_object(UNKNOWN, sandbox_roots=[SANDBOX])
        self.assertEqual(rec.object_class, ObjectClass.QUARANTINED)
        self.assertTrue(any("unclassified" in r for r in rec.reasons))

    def test_corrupt_is_quarantined(self):
        rec = classify_object(
            f"{SANDBOX}/partial.out",
            sandbox_roots=[SANDBOX],
            known_partial_or_corrupt=True,
        )
        self.assertEqual(rec.object_class, ObjectClass.QUARANTINED)


class TestAutoDeleteGates(unittest.TestCase):
    def _full_evictable(self, **overrides):
        base = dict(
            path=EVICTABLE,
            sandbox_roots=[SANDBOX],
            evictable_paths=[EVICTABLE],
            active_references=0,
            receipt_sealed=True,
            successor_or_rejection_verified=True,
            rollback_preserved=True,
            remote_hash_verified=True,
        )
        base.update(overrides)
        path = base.pop("path")
        return classify_object(path, **base)

    def test_all_gates_pass(self):
        rec = self._full_evictable()
        dec = evaluate_auto_delete(rec, apply=True)
        self.assertTrue(dec.allowed)
        self.assertTrue(dec.would_delete)
        self.assertTrue(all(dec.gates.values()))

    def test_missing_receipt_refuses(self):
        rec = self._full_evictable(receipt_sealed=False)
        dec = evaluate_auto_delete(rec)
        self.assertFalse(dec.allowed)
        self.assertIn("receipt_sealed", " ".join(dec.refuse_reasons))

    def test_active_refs_refuse(self):
        rec = classify_object(
            EVICTABLE,
            sandbox_roots=[SANDBOX],
            evictable_paths=[EVICTABLE],
            active_references=1,
            receipt_sealed=True,
            successor_or_rejection_verified=True,
            rollback_preserved=True,
        )
        # active refs force LEASED, not EVICTABLE
        self.assertEqual(rec.object_class, ObjectClass.LEASED)
        dec = evaluate_auto_delete(rec)
        self.assertFalse(dec.allowed)

    def test_pinned_never_deletes(self):
        rec = classify_object(PINNED_PROTO, sandbox_roots=[SANDBOX])
        dec = evaluate_auto_delete(rec, apply=True)
        self.assertFalse(dec.allowed)
        self.assertFalse(dec.would_delete)

    def test_remote_hash_required_when_false(self):
        rec = self._full_evictable(remote_hash_verified=False)
        dec = evaluate_auto_delete(rec)
        self.assertFalse(dec.allowed)

    def test_remote_hash_none_not_required(self):
        rec = self._full_evictable(remote_hash_verified=None)
        dec = evaluate_auto_delete(rec)
        self.assertTrue(dec.allowed)

    def test_cleanup_receipt_sealed(self):
        rec = self._full_evictable()
        dec = evaluate_auto_delete(rec, apply=False)
        receipt = build_cleanup_receipt([dec], free_bytes_before=10, free_bytes_after=20)
        self.assertEqual(receipt["schema"], "hawking.ascension.cleanup_receipt.v1")
        self.assertIn("receipt_sha256", receipt)
        self.assertEqual(len(receipt["receipt_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
