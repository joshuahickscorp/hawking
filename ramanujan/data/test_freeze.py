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
