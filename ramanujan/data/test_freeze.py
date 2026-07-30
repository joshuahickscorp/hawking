#!/usr/bin/env python3.12
"""Tests for sealed train/dev/test membership freeze."""
from __future__ import annotations

import unittest

from ramanujan.data.freeze_memberships import (
    assign_split,
    load_membership,
    verify_membership_seal,
)
from tools.odyssey.contamination import build_barrier
from tools.odyssey.normalize import extract_comparison_text
from tools.odyssey._paths import SUPPORT_HALO_CORPUS
import json


class TestAssignSplit(unittest.TestCase):
    def test_deterministic(self):
        h = "a" * 64
        self.assertEqual(assign_split(h), assign_split(h))

    def test_bucket_boundaries(self):
        # craft hashes with known first-8 hex for buckets 0, 79, 80, 89, 90, 99
        cases = {
            0: "00000000" + "0" * 56,
            79: "0000004f" + "0" * 56,  # 0x4f = 79
            80: "00000050" + "0" * 56,
            89: "00000059" + "0" * 56,
            90: "0000005a" + "0" * 56,
            99: "00000063" + "0" * 56,
        }
        # int(h[:8],16) % 100 — for small values equals the int itself
        self.assertEqual(assign_split(cases[0]), "train")
        self.assertEqual(assign_split(cases[79]), "train")
        self.assertEqual(assign_split(cases[80]), "dev")
        self.assertEqual(assign_split(cases[89]), "dev")
        self.assertEqual(assign_split(cases[90]), "test")
        self.assertEqual(assign_split(cases[99]), "test")


class TestSealedManifest(unittest.TestCase):
    def test_seal_ok(self):
        result = verify_membership_seal()
        self.assertTrue(result["ok"], msg=result)
        m = load_membership()
        self.assertEqual(m["counts"]["total"], 16188)
        self.assertEqual(
            m["counts"]["train"] + m["counts"]["dev"] + m["counts"]["test"],
            m["counts"]["total"],
        )


class TestNegativeControl(unittest.TestCase):
    def test_support_halo_exact_match(self):
        barrier = build_barrier()
        items = [
            json.loads(ln)
            for ln in SUPPORT_HALO_CORPUS.read_text().splitlines()
            if ln.strip()
        ]
        probe = next(x for x in items if x.get("id") == "tl02_bpw")
        hits = barrier.check(extract_comparison_text(probe))
        self.assertTrue(hits)
        self.assertTrue(any(h.reason == "exact_match" for h in hits))


if __name__ == "__main__":
    unittest.main()


class TestFreezeIsReproducible(unittest.TestCase):
    """A freeze nobody can re-derive is a record, not a seal.

    The generation receipt seals each corpus by file sha256, and every record
    carries a wall-clock `provenance.at`, so regenerating the identical corpus
    from the identical pinned Mathlib produces a different file hash. Measured
    on 2026-07-30: D1, D2 and D3 each reproduced all 5000 content hashes in
    identical order while all three file hashes differed. `content_digest`
    exists so the reproducible part can actually be checked.
    """

    def test_content_digest_ignores_the_timestamp(self):
        from ramanujan.data.common import write_jsonl
        import tempfile
        from pathlib import Path

        rows = [
            {"content_hash": "aa", "provenance": {"at": "2026-01-01T00:00:00Z"}},
            {"content_hash": "bb", "provenance": {"at": "2026-01-01T00:00:00Z"}},
        ]
        later = [
            {"content_hash": "aa", "provenance": {"at": "2099-12-31T23:59:59Z"}},
            {"content_hash": "bb", "provenance": {"at": "2099-12-31T23:59:59Z"}},
        ]
        with tempfile.TemporaryDirectory() as d:
            a = write_jsonl(Path(d) / "a.jsonl", rows)
            b = write_jsonl(Path(d) / "b.jsonl", later)
            self.assertNotEqual(a["sha256"], b["sha256"], "file hashes should differ")
            self.assertEqual(a["content_digest"], b["content_digest"])

    def test_content_digest_still_catches_a_real_change(self):
        from ramanujan.data.common import write_jsonl
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            a = write_jsonl(Path(d) / "a.jsonl", [{"content_hash": "aa"}])
            b = write_jsonl(Path(d) / "b.jsonl", [{"content_hash": "ac"}])
            self.assertNotEqual(a["content_digest"], b["content_digest"])

    def test_order_is_part_of_the_seal(self):
        from ramanujan.data.common import write_jsonl
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            a = write_jsonl(Path(d) / "a.jsonl", [{"content_hash": "aa"}, {"content_hash": "bb"}])
            b = write_jsonl(Path(d) / "b.jsonl", [{"content_hash": "bb"}, {"content_hash": "aa"}])
            self.assertNotEqual(a["content_digest"], b["content_digest"])
