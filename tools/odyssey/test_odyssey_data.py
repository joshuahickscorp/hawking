#!/usr/bin/env python3
"""Tests for Odyssey data inventory, ingest, and contamination barrier.

Includes a deliberate leak: a training item that is a near-duplicate of a
held-out evaluation item must be refused.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey._paths import (  # noqa: E402
    EXPECTED_SUPPORT_HALO_CORPUS_SHA256,
    FIXTURE_DIR,
    HIDDEN_ITEMS,
    SUPPORT_HALO_CORPUS,
)
from tools.odyssey.contamination import (  # noqa: E402
    build_barrier,
    verify_hidden_commitment,
    verify_support_halo_seal,
)
from tools.odyssey.dedup import content_sha256, find_near_duplicates  # noqa: E402
from tools.odyssey.ingest import ingest_corpus  # noqa: E402
from tools.odyssey.inventory import build_inventory, check_declared_corpora  # noqa: E402
from tools.odyssey.membership import item_content_address  # noqa: E402
from tools.odyssey.normalize import extract_comparison_text, normalize_text  # noqa: E402
from tools.odyssey.teacher_assess import assess_teacher_traces  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_normalize_collapse_ws(self):
        self.assertEqual(normalize_text("  Foo   BAR\n"), "foo bar")

    def test_extract_prompt(self):
        self.assertEqual(extract_comparison_text({"prompt": "Hello"}), "Hello")


class TestSupportHaloSeal(unittest.TestCase):
    def test_sealed_corpus_hash(self):
        seal = verify_support_halo_seal()
        self.assertTrue(seal["ok"], msg=seal)
        self.assertEqual(seal["computed_sha256"], EXPECTED_SUPPORT_HALO_CORPUS_SHA256)
        self.assertTrue(SUPPORT_HALO_CORPUS.is_file())


class TestHiddenCommitment(unittest.TestCase):
    def test_t0_hidden_commitment(self):
        self.assertTrue(HIDDEN_ITEMS.is_file(), "T0 hidden items must be present for barrier")
        hc = verify_hidden_commitment()
        self.assertTrue(hc["ok"], msg=hc)
        self.assertGreaterEqual(hc["n_hidden"], 1)


class TestContaminationBarrier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.barrier = build_barrier()

    def test_exact_leak_from_support_halo_rejected(self):
        # Exact text of tl02_bpw from sealed corpus.
        text = (
            'What does BPW stand for in model compression / quantization? '
            'Answer briefly and include the exact phrase "bits per weight".'
        )
        hits = self.barrier.check(text)
        self.assertTrue(hits, "exact eval leak must be rejected")
        self.assertTrue(any(h.reason == "exact_match" for h in hits))
        self.assertTrue(any(h.eval_id == "tl02_bpw" for h in hits))

    def test_near_dup_leak_from_support_halo_rejected(self):
        # Near-duplicate of tl01_idempotent (small wording change).
        text = (
            "In one or two sentences, define what an idempotent HTTP method means. "
            "Include the word \"idempotent\" and the idea that repeating the request "
            "does not change resource state further."
        )
        hits = self.barrier.check(text)
        self.assertTrue(hits, "near-dup eval leak must be rejected")
        self.assertTrue(
            any(h.reason in ("exact_match", "near_duplicate") for h in hits),
            msg=hits,
        )
        self.assertTrue(
            any("tl01" in h.eval_id or h.eval_id == "tl01_idempotent" for h in hits),
            msg=[(h.eval_id, h.reason, h.jaccard) for h in hits],
        )

    def test_near_dup_leak_from_hidden_membership_rejected(self):
        # Near-duplicate of hid_reason_01 (bat and ball).
        text = (
            "A bat and a ball cost $1.10 in total. The bat costs one dollar more "
            "than the ball. How much does the ball cost?"
        )
        hits = self.barrier.check(text)
        self.assertTrue(hits, "hidden membership near-dup must be rejected")
        self.assertTrue(any(h.eval_id == "hid_reason_01" for h in hits), msg=hits)

    def test_clean_fixture_admitted(self):
        text = (
            "FIXTURE: The streaming adapter budget on this host is approximately "
            "two gibibytes for a smoke step."
        )
        hits = self.barrier.check(text)
        self.assertEqual(hits, [], msg=hits)

    def test_naming_convention_is_not_enough(self):
        """Even if the id looks like training data, content overlap rejects."""
        item = {
            "id": "train_math_totally_legit_99",
            "text": (
                'What does BPW stand for in model compression / quantization? '
                'Answer briefly and include the exact phrase "bits per weight".'
            ),
        }
        hits = self.barrier.check(extract_comparison_text(item))
        self.assertTrue(hits)


class TestIngestFixture(unittest.TestCase):
    def test_end_to_end_fixture(self):
        raw = FIXTURE_DIR / "raw_fixture.jsonl"
        self.assertTrue(raw.is_file())
        with tempfile.TemporaryDirectory() as td:
            result = ingest_corpus(
                raw,
                corpus_id="test_fixture",
                role="fixture",
                out_dir=Path(td),
                licence="synthetic-fixture-not-for-training",
                note="unit test",
            )
            self.assertGreaterEqual(result["n_admitted"], 1)
            self.assertGreaterEqual(result["n_rejected_contamination"], 2)
            self.assertGreaterEqual(result["n_rejected_exact_dup"], 1)
            # Deliberate leak ids must appear in contamination rejections.
            rejected_ids = {r.get("source_id") for r in result["contamination_rejections"]}
            self.assertIn("fix_leak_exact_halo_tl02", rejected_ids)
            self.assertTrue(
                "fix_leak_near_halo_tl01" in rejected_ids
                or "fix_leak_near_hidden_batball" in rejected_ids,
                msg=rejected_ids,
            )
            # Admitted items carry content addresses.
            admitted_path = Path(result["admitted_path"])
            lines = [
                json.loads(l) for l in admitted_path.read_text().splitlines() if l.strip()
            ]
            self.assertTrue(all(x.get("content_sha256") for x in lines))
            self.assertTrue(all(x.get("membership_status") == "admitted" for x in lines))


class TestMembershipContract(unittest.TestCase):
    def test_declared_corpora_are_not_present(self):
        declared = check_declared_corpora()
        by_id = {c["id"]: c for c in declared}
        for cid in ("math-core", "support-language", "long-horizon", "sovereignty-corpus"):
            self.assertIn(cid, by_id)
            self.assertEqual(
                by_id[cid]["status"],
                "DECLARED_NOT_PRESENT",
                msg=by_id[cid],
            )

    def test_inventory_schema(self):
        inv = build_inventory()
        self.assertEqual(inv["schema"], "hawking.odyssey.data_inventory.v1")
        self.assertIn("corpora", inv)
        self.assertTrue(any(c["role"] == "eval" for c in inv["corpora"]))


class TestContentAddress(unittest.TestCase):
    def test_stable_address(self):
        a = {"id": "x", "text": "hello"}
        b = {"id": "x", "text": "hello", "membership_status": "admitted"}
        self.assertEqual(item_content_address(a), item_content_address(b))

    def test_exact_hash_identity(self):
        self.assertEqual(content_sha256("Ab C"), content_sha256("ab  c"))


class TestTeacherAssess(unittest.TestCase):
    def test_assess_runs(self):
        report = assess_teacher_traces()
        self.assertIn(report["status"], ("PARTIAL", "LEDGER_NOT_PRESENT", "MANIFEST_MISSING"))
        if report["status"] == "PARTIAL":
            self.assertEqual(report["gap"]["numeric_gap"]["have_trajectory_traces"], 0)
            self.assertGreaterEqual(report["what_exists"]["n_lines"], 1)


class TestNearDupHelper(unittest.TestCase):
    def test_identical_jaccard(self):
        hits = find_near_duplicates(["abcdefghi", "abcdefghi"], threshold=0.5)
        self.assertIn(0, hits)


if __name__ == "__main__":
    unittest.main()
