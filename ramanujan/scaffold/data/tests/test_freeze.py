#!/usr/bin/env python3.12
"""Tests for sealed train/dev/test membership freeze."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from ramanujan.data.freeze_memberships import (
    EXTERNAL_INTAKE_SCHEMA,
    _sealed,
    assign_split,
    external_intake_template,
    freeze_external_sources,
    load_membership,
    freeze_memberships,
    verify_membership_seal,
    verify_membership_sources,
)
from ramanujan.layout import DATA_ROOT
from tools.odyssey.contamination import build_barrier
from tools.odyssey.normalize import extract_comparison_text
from tools.odyssey._paths import SUPPORT_HALO_CORPUS
import json
import hashlib
import os


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

    def test_current_source_drift_refuses_a_valid_but_stale_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "d1.jsonl"
            source.write_text(
                '{"id":"d1:one","content_hash":"00000000' + '0' * 56 + '"}\n',
                encoding="utf-8",
            )
            manifest = root / "membership.json"
            receipt = root / "freeze.json"
            files = {"D1": source}
            freeze_memberships(
                source_files=files,
                manifest_path=manifest,
                receipt_path=receipt,
                run_contamination=False,
            )
            self.assertTrue(verify_membership_seal(manifest, source_files=files)["ok"])

            source.write_text(
                '{"id":"d1:one","content_hash":"ffffffff' + 'f' * 56 + '"}\n',
                encoding="utf-8",
            )
            freshness = verify_membership_sources(manifest, source_files=files)
            self.assertFalse(freshness["ok"])
            self.assertFalse(verify_membership_seal(manifest, source_files=files)["ok"])
            self.assertTrue(freshness["sources"]["D1"]["missing_content_hashes"])


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


class TestExternalQualificationIntake(unittest.TestCase):
    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _fixture(self, root: Path) -> tuple[Path, Path]:
        authority = root / "owner-authority.json"
        authority.write_text('{"authority":"test-owner"}\n')
        licence = root / "LICENSE"
        licence.write_text("Synthetic test license\n")
        adjudicator = root / "adjudicator"
        adjudicator.write_text("#!/bin/sh\nexit 0\n")
        adjudicator.chmod(0o700)
        generator = root / "variant-generator"
        generator.write_text("#!/bin/sh\nexit 0\n")
        generator.chmod(0o700)
        sources = {}
        for sid in ("D5", "D8", "D9"):
            path = root / f"{sid}.jsonl"
            body = {"id": f"{sid.lower()}-one", "text": f"synthetic {sid}"}
            if sid == "D8":
                body["set"] = "hidden"
            path.write_text(json.dumps(body) + "\n")
            path.chmod(0o600)
            sources[sid] = path
        rows = []
        for sid in ("D5", "D8", "D9"):
            variant = None
            if sid == "D9":
                variant = {
                    "actor": "variant-builder",
                    "path": str(generator),
                    "sha256": self._sha(generator),
                    "seed_commitment_sha256": "9" * 64,
                }
            rows.append({
                "id": sid,
                "owner_approved": True,
                "owner_actor": "owner",
                "version": "fixture-v1",
                "source": {"path": str(sources[sid]), "sha256": self._sha(sources[sid])},
                "license": {"spdx": "LicenseRef-Synthetic-Test", "path": str(licence), "sha256": self._sha(licence)},
                "membership_sealer_actor": "membership-sealer",
                "adjudicator": {"actor": "independent-adjudicator", "path": str(adjudicator), "sha256": self._sha(adjudicator)},
                "variant_generator": variant,
            })
        intake = _sealed({
            "schema": EXTERNAL_INTAKE_SCHEMA,
            "status": "OWNER_APPROVED",
            "owner_authority_receipt": {"path": str(authority), "sha256": self._sha(authority)},
            "candidate_launch_started": False,
            "sources": rows,
        })
        intake_path = root / "intake.json"
        intake_path.write_text(json.dumps(intake))
        return intake_path, root / "public-receipt.json"

    def test_template_is_exactly_d5_d8_d9_and_non_authorizing(self):
        template = external_intake_template()
        self.assertEqual([row["id"] for row in template["sources"]], ["D5", "D8", "D9"])
        self.assertEqual(template["status"], "PENDING_OWNER_APPROVAL")
        self.assertTrue(all(row["owner_approved"] is False for row in template["sources"]))
        self.assertIsNone(template["sources"][0]["variant_generator"])
        self.assertIsNone(template["sources"][1]["variant_generator"])
        self.assertIsInstance(template["sources"][2]["variant_generator"], dict)

    def test_owner_bound_sources_freeze_without_hidden_ids_or_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            intake, receipt_path = self._fixture(Path(directory))
            receipt = freeze_external_sources(intake, receipt_path=receipt_path)
            self.assertEqual(receipt["status"], "PASS_INPUTS_FROZEN_RESEARCH_AND_CANDIDATE_AUTHORITY_FALSE")
            self.assertIsNone(receipt["training_visible"]["D8_hidden_item_ids"])
            self.assertFalse(receipt["RAMANUJAN_RESEARCH_AUTHORIZED"])
            self.assertFalse(receipt["candidate_launch_authorized"])
            rendered = receipt_path.read_text()
            source_paths = [row["source"]["path"] for row in json.loads(intake.read_text())["sources"]]
            self.assertTrue(all(path not in rendered for path in source_paths))
            self.assertNotIn("d8-one", rendered)

    def test_owner_false_role_collision_and_public_d8_mode_refuse(self):
        for mutation, expected in (
            ("owner_false", "owner approval"),
            ("role_collision", "roles must be distinct"),
            ("d8_public", "owner-only filesystem mode"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                intake_path, receipt_path = self._fixture(Path(directory))
                intake = json.loads(intake_path.read_text())
                intake.pop("seal_sha256")
                if mutation == "owner_false":
                    intake["status"] = "PENDING"
                elif mutation == "role_collision":
                    intake["sources"][1]["membership_sealer_actor"] = "owner"
                else:
                    Path(intake["sources"][1]["source"]["path"]).chmod(0o644)
                intake_path.write_text(json.dumps(_sealed(intake)))
                with self.assertRaisesRegex(ValueError, expected):
                    freeze_external_sources(intake_path, receipt_path=receipt_path)

    def test_d8_exact_training_leak_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            intake_path, receipt_path = self._fixture(Path(directory))
            intake = json.loads(intake_path.read_text())
            intake.pop("seal_sha256")
            d8 = intake["sources"][1]
            d8_path = Path(d8["source"]["path"])
            current_training = json.loads(next(iter((DATA_ROOT / "corpora").glob("d1_*.jsonl"))).read_text().splitlines()[0])
            d8_path.write_text(json.dumps({"id": "private", "set": "hidden", "text": current_training["text"]}) + "\n")
            d8_path.chmod(0o600)
            d8["source"]["sha256"] = self._sha(d8_path)
            intake_path.write_text(json.dumps(_sealed(intake)))
            with self.assertRaisesRegex(ValueError, "overlaps current frozen training"):
                freeze_external_sources(intake_path, receipt_path=receipt_path)


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
