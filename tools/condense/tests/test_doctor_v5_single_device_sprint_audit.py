from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import doctor_v5_single_device_sprint_audit as audit


class SingleDeviceSprintAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = audit.build_packet()

    def test_current_packet_is_inert_complete_and_valid(self) -> None:
        self.assertEqual([], audit.validate_packet(self.packet))
        self.assertFalse(self.packet["activation_permitted"])
        self.assertFalse(self.packet["production_ready"])
        self.assertTrue(self.packet["all_required_implementations_present"])
        self.assertIn("phase-aware-remaining-scratch", self.packet["components"])
        self.assertIn(
            "phase-aware-remaining-scratch",
            self.packet["required_component_names"],
        )
        scratch_paths = {
            row["path"] for row in self.packet["components"]
            ["phase-aware-remaining-scratch"]["source_artifacts"]
        }
        self.assertIn(
            "tools/condense/doctor_v5_remaining_scratch_gate_adapter.py",
            scratch_paths,
        )
        elastic_paths = {
            row["path"] for row in self.packet["components"]
            ["elastic-phase-admission"]["source_artifacts"]
        }
        self.assertIn(
            "tools/condense/doctor_v5_inert_phase_launcher.py", elastic_paths
        )
        self.assertIn(
            "tools/condense/doctor_v5_fixture_phase_validator.py", elastic_paths
        )
        self.assertIn(
            "tools/condense/doctor_v5_acceleration_eta.py",
            {row["path"] for row in self.packet["control_sources"]},
        )
        self.assertFalse(self.packet["eta_claim_boundary"][
            "component_or_synthetic_speedups_multiplied"
        ])
        self.assertFalse(self.packet["eta_claim_boundary"][
            "unmeasured_120b_or_appendix_speedup_applied"
        ])
        boundary = self.packet["eta_claim_boundary"]
        authority = boundary["production_eta_authority"]
        threshold = boundary["seven_day_threshold"]
        self.assertEqual(audit.benchmark.eta_contract.SCHEMA,
                         authority["schema"])
        self.assertEqual([], audit.benchmark.eta_contract.validate(
            authority, verify_freshness=True
        ))
        self.assertTrue(authority["eta_blocked"])
        self.assertEqual(audit.benchmark.THRESHOLD_SCHEMA, threshold["schema"])
        self.assertEqual(authority["document_sha256"],
                         threshold["production_eta_sha256"])
        self.assertFalse(threshold["available"])
        self.assertTrue(all(
            threshold[field] is None for field in audit.THRESHOLD_NUMERIC_FIELDS
        ))
        self.assertFalse(threshold["gpt_oss_120b_threshold_available"])
        self.assertFalse(threshold["appendix_threshold_available"])
        self.assertFalse(threshold["threshold_equality_is_sufficient"])
        self.assertIsInstance(boundary["stored_snapshot_matches_current"], bool)
        self.assertIn(
            "tools/condense/tests/test_doctor_v5_single_device_sprint_audit.py",
            {row["path"] for row in self.packet["cheap_test_sources"]},
        )
        control_paths = {row["path"] for row in self.packet["control_sources"]}
        self.assertIn("tools/condense/doctor_v5_production_eta.py", control_paths)
        self.assertIn("tools/condense/doctor_v5_single_device_benchmark.py",
                      control_paths)

    def test_component_source_tamper_is_detected_after_resealing(self) -> None:
        damaged = copy.deepcopy(self.packet)
        damaged["components"]["native-pgo-io-profile"][
            "source_artifacts"
        ][0]["sha256"] = "0" * 64
        damaged["audit_sha256"] = audit._hash_value(
            audit._without(damaged, "audit_sha256")
        )
        errors = audit.validate_packet(damaged)
        self.assertTrue(any("source binding changed" in row for row in errors))

    def test_control_source_tamper_is_detected_after_resealing(self) -> None:
        damaged = copy.deepcopy(self.packet)
        damaged["control_sources"][0]["sha256"] = "0" * 64
        damaged["audit_sha256"] = audit._hash_value(
            audit._without(damaged, "audit_sha256")
        )
        errors = audit.validate_packet(damaged)
        self.assertTrue(any("control source binding changed" in row for row in errors))

    def test_activation_or_eta_boundary_tamper_is_detected(self) -> None:
        damaged = copy.deepcopy(self.packet)
        damaged["activation_permitted"] = True
        damaged["eta_claim_boundary"][
            "unmeasured_120b_or_appendix_speedup_applied"
        ] = True
        damaged["audit_sha256"] = audit._hash_value(
            audit._without(damaged, "audit_sha256")
        )
        errors = audit.validate_packet(damaged)
        self.assertTrue(any("isolation/lifecycle" in row for row in errors))
        self.assertTrue(any("ETA claim" in row for row in errors))

    def test_resealed_v1_eta_authority_is_rejected(self) -> None:
        damaged = copy.deepcopy(self.packet)
        authority = damaged["eta_claim_boundary"]["production_eta_authority"]
        authority["schema"] = "hawking.doctor_v5_production_eta.v1"
        authority["document_sha256"] = audit._hash_value(
            audit._without(authority, "document_sha256")
        )
        damaged["audit_sha256"] = audit._hash_value(
            audit._without(damaged, "audit_sha256")
        )
        errors = audit.validate_packet(damaged)
        self.assertIn("single-device production ETA authority is invalid", errors)
        self.assertIn("single-device seven-day threshold authority is invalid", errors)

    def test_resealed_blocked_numeric_threshold_is_rejected(self) -> None:
        damaged = copy.deepcopy(self.packet)
        threshold = damaged["eta_claim_boundary"]["seven_day_threshold"]
        threshold["available"] = True
        threshold["sub_120b_required_additional_speedup_for_entire_range"] = 2.0
        threshold["threshold_sha256"] = audit._hash_value(
            audit._without(threshold, "threshold_sha256")
        )
        damaged["audit_sha256"] = audit._hash_value(
            audit._without(damaged, "audit_sha256")
        )
        errors = audit.validate_packet(damaged)
        self.assertIn("single-device seven-day threshold authority is invalid", errors)
        self.assertIn("blocked production ETA exposes a numeric threshold", errors)


if __name__ == "__main__":
    unittest.main()
