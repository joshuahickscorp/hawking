#!/usr/bin/env python3.12
"""Adversarial tests for the unbound post-120B acceleration contracts."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))

import doctor_v5_higher_tier_scaffold as higher
import doctor_v5_higher_tier_authority as higher_authority
import doctor_v5_post120_acceleration_scaffold as accel
from higher_tier_authority_fixtures import (
    VERIFIED, fake_operator_seal, make_core_manifest,
)


NOW = "2026-07-14T22:00:00+00:00"


def _rehash(doc: dict, field: str) -> None:
    doc.pop(field, None)
    doc[field] = accel._hash_value(doc)


def _higher_manifest(root: Path) -> dict:
    return make_core_manifest(root)


class Post120AccelerationScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.work = accel._read_json(accel.DEFAULT_WORK_PLAN)
        cls.pending = accel._read_json(accel.DEFAULT_PENDING_WIRING)
        cls.fanout = accel._read_json(accel.DEFAULT_FANOUT_PLAN)
        cls.gptoss = accel.build_gptoss_acceleration_plan(
            cls.work, cls.pending, cls.fanout, created_at=NOW
        )

    def setUp(self) -> None:
        self.signature_patch = mock.patch.object(
            higher_authority, "_sshsig_verify", VERIFIED,
        )
        self.signature_patch.start()

    def tearDown(self) -> None:
        self.signature_patch.stop()

    def test_requirements_and_named_horizons_cover_every_facet_and_cell(self) -> None:
        requirements = accel.build_requirements_packet(created_at=NOW)
        self.assertEqual([], accel.validate_requirements(requirements))
        self.assertEqual(list(accel.FACETS), list(requirements["facets"]))
        horizons = accel.build_named_horizon_scaffold(
            requirements=requirements, created_at=NOW
        )
        self.assertEqual([], accel.validate_named_horizon_scaffold(horizons))
        self.assertEqual(120, horizons["matrix"]["total_cell_templates"])
        self.assertTrue(all(len(row["cells"]) == 40 for row in horizons["models"]))

    def test_gptoss_plan_binds_exact_10x4_and_24600_jobs(self) -> None:
        self.assertEqual([], accel.validate_gptoss_acceleration_plan(
            self.gptoss, self.work, self.pending, self.fanout
        ))
        self.assertEqual(40, self.gptoss["matrix"]["cells"])
        self.assertEqual(24_600, self.gptoss["matrix"]["isolated_jobs"])
        self.assertEqual(0, self.gptoss["activation"]["qualified_facet_count"])
        self.assertFalse(self.gptoss["activation"]["execution_permitted"])

    def test_resealed_profile_selection_cannot_bypass_physical_gate(self) -> None:
        tampered = copy.deepcopy(self.gptoss)
        tampered["profiles"][0]["selected_threads"] = 20
        _rehash(tampered, "acceleration_plan_sha256")
        errors = accel.validate_gptoss_acceleration_plan(
            tampered, self.work, self.pending, self.fanout
        )
        self.assertTrue(any("profiles" in row for row in errors), errors)

    def test_resealed_matrix_substitution_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.gptoss)
        tampered["cells"][0]["cell_identity_sha256"] = "f" * 64
        _rehash(tampered, "acceleration_plan_sha256")
        errors = accel.validate_gptoss_acceleration_plan(
            tampered, self.work, self.pending, self.fanout
        )
        self.assertTrue(any("authority" in row for row in errors), errors)

    def test_resealed_execution_and_facet_claims_remain_closed(self) -> None:
        tampered = copy.deepcopy(self.gptoss)
        tampered["activation"]["execution_permitted"] = True
        tampered["facets"]["metal_preprocess"]["status"] = "qualified"
        tampered["promotion_gate"]["currently_permitted"] = True
        _rehash(tampered, "acceleration_plan_sha256")
        errors = accel.validate_gptoss_acceleration_plan(
            tampered, self.work, self.pending, self.fanout
        )
        self.assertGreaterEqual(len(errors), 3, errors)

    def test_higher_tier_plan_has_same_10x4_and_job_space(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = _higher_manifest(root)
            sealed = fake_operator_seal(root, manifest)
            admission = higher.build_admission_plan(
                manifest, total_memory_bytes=100_000_000_000,
                process_budget_bytes=80_000_000_000,
                control_resident_bytes=5_000_000_000,
                safety_margin_bytes=10_000_000_000,
                logical_cpu_count=20, maximum_lanes=8,
            )
            plan = accel.build_higher_tier_acceleration_plan(
                sealed, admission, created_at=NOW
            )
            self.assertEqual([], accel.validate_higher_tier_acceleration_plan(
                plan, sealed, admission
            ))
            self.assertEqual(40, plan["matrix"]["isolated_jobs"])
            self.assertTrue(plan["activation"]["architecture_adapter_bound"])
            self.assertFalse(
                plan["activation"]["architecture_adapter_physically_qualified"]
            )
            self.assertTrue(all(
                row["adapter_id"] == "fixture-dense-v1" for row in plan["cells"]
            ))

    def test_higher_tier_resealed_adapter_or_source_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = _higher_manifest(root)
            sealed = fake_operator_seal(root, manifest)
            admission = higher.build_admission_plan(
                manifest, total_memory_bytes=100_000_000_000,
                process_budget_bytes=80_000_000_000,
                control_resident_bytes=5_000_000_000,
                safety_margin_bytes=10_000_000_000,
                logical_cpu_count=20, maximum_lanes=8,
            )
            plan = accel.build_higher_tier_acceleration_plan(sealed, admission)
            plan["cells"][0]["adapter_id"] = "unreviewed-adapter"
            plan["cells"][1]["source_manifest_sha256"] = "f" * 64
            _rehash(plan, "acceleration_plan_sha256")
            errors = accel.validate_higher_tier_acceleration_plan(
                plan, sealed, admission
            )
            self.assertTrue(any("10x4" in row for row in errors), errors)

    def test_higher_tier_exact_matrix_refuses_missing_bindings_files_and_byte_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = _higher_manifest(root)

            errors = higher.validate_source_manifest(
                manifest, require_exact_wiring=True, verify_files=True,
            )
            self.assertTrue(any("operator authority" in row for row in errors), errors)

            missing_adapter = copy.deepcopy(manifest)
            missing_adapter.pop("architecture_adapter")
            _rehash(missing_adapter, "manifest_sha256")
            with self.assertRaisesRegex(
                higher_authority.HigherTierAuthorityError,
                "architecture adapter",
            ):
                higher_authority.build_manifest_attestation(missing_adapter)

            admission = higher.build_admission_plan(
                manifest, total_memory_bytes=100_000_000_000,
                process_budget_bytes=80_000_000_000,
                control_resident_bytes=5_000_000_000,
                safety_margin_bytes=10_000_000_000,
                logical_cpu_count=20, maximum_lanes=8,
            )
            sealed = fake_operator_seal(root, manifest)
            valid_plan = accel.build_higher_tier_acceleration_plan(sealed, admission)
            # Validation must return fail-closed errors, never index malformed
            # parent fields after the strict parent gate has already failed.
            errors = accel.validate_higher_tier_acceleration_plan(
                valid_plan, missing_adapter, admission,
            )
            self.assertTrue(any("operator" in row for row in errors), errors)

            gap = copy.deepcopy(manifest)
            end = gap["sources"][0]["bytes"]
            gap["work_units"][0]["source_ranges"][0][
                "absolute_byte_range"
            ] = [1, end]
            _rehash(gap, "manifest_sha256")
            with self.assertRaisesRegex(
                higher_authority.HigherTierAuthorityError,
                "work-unit tensor/source/range differs",
            ):
                higher_authority.build_manifest_attestation(gap)

    def test_higher_tier_remote_ranges_require_semantic_versioned_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = _higher_manifest(root)
            higher_authority.build_manifest_attestation(manifest)
            sealed = fake_operator_seal(root, manifest)
            self.assertEqual([], higher.validate_source_manifest(
                sealed, require_exact_wiring=True, verify_files=True,
            ))

            tampered = copy.deepcopy(manifest)
            tampered["sources"][0]["immutable_version"] = "version-0002"
            _rehash(tampered, "manifest_sha256")
            with self.assertRaisesRegex(
                higher_authority.HigherTierAuthorityError,
                "authority semantics differ",
            ):
                higher_authority.build_manifest_attestation(tampered)

    def test_sealed_handoff_binds_every_parent_and_stays_non_executable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            requirements = accel.build_requirements_packet(created_at=NOW)
            horizons = accel.build_named_horizon_scaffold(
                requirements=requirements, created_at=NOW
            )
            paths = {
                "acceleration_requirements_path": root / "requirements.json",
                "horizons_path": root / "horizons.json",
                "gptoss_acceleration_path": root / "gptoss.json",
            }
            accel._write_json(paths["acceleration_requirements_path"], requirements)
            accel._write_json(paths["horizons_path"], horizons)
            accel._write_json(paths["gptoss_acceleration_path"], self.gptoss)
            handoff = accel.build_handoff_packet(created_at=NOW, **paths)
            self.assertEqual([], accel.validate_handoff_packet(handoff))
            self.assertEqual(13, len(handoff["bindings"]))
            self.assertFalse(handoff["claim_boundary"]["execution_permitted"])
            self.assertEqual(0, handoff["readiness"][
                "physical_acceleration_facets_qualified"
            ])
            self.assertEqual(
                0, handoff["readiness"]["physical_ab_facets_qualified"]
            )
            self.assertFalse(
                handoff["readiness"]["physical_ab_execution_permitted"]
            )
            self.assertEqual(
                list(accel.PHYSICAL_AB_FACETS),
                handoff["coverage"]["physical_ab_facets"],
            )

            tampered = copy.deepcopy(handoff)
            tampered["bindings"]["tokenizer_gate"]["semantic_sha256"] = "f" * 64
            _rehash(tampered, "handoff_sha256")
            errors = accel.validate_handoff_packet(tampered)
            self.assertTrue(any("tokenizer_gate" in row for row in errors), errors)

    def test_physical_plan_is_exact_source_bound_and_default_off(self) -> None:
        physical = accel._read_json(accel.DEFAULT_PHYSICAL_AB_PLAN)
        self.assertEqual([], accel._validate_physical_ab_plan(physical))
        tampered = copy.deepcopy(physical)
        tampered["execution_capability"] = True
        _rehash(tampered, "plan_sha256")
        errors = accel._validate_physical_ab_plan(tampered)
        self.assertTrue(any("default-off" in row or "differs" in row for row in errors), errors)


if __name__ == "__main__":
    unittest.main()
