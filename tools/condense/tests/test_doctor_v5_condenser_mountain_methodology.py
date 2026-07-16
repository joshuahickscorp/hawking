#!/usr/bin/env python3.12
"""Tests for the additive empirical condenser mountain methodology."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))

import doctor_v5_condenser_mountain_methodology as methodology


class CondenserMountainMethodologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = methodology.build_report(created_at="2026-07-15T12:00:00+00:00")

    def test_current_report_is_valid_and_fail_closed(self) -> None:
        self.assertEqual([], methodology.validate_report(self.report))
        self.assertTrue(all(
            value is False for value in self.report["claim_boundary"].values()
        ))
        self.assertEqual("observe-only-no-mutation", self.report["live_campaign_action"])

    def test_parameter_weighted_progress_is_below_cell_progress(self) -> None:
        cells = [
            {"model_label": "1B", "model_family": "qwen2.5-dense",
             "exact_stored_parameter_count": 1_000_000_000,
             "status": "complete", "admission": {}},
            {"model_label": "1B", "model_family": "qwen2.5-dense",
             "exact_stored_parameter_count": 1_000_000_000,
             "status": "complete", "admission": {}},
            {"model_label": "100B", "model_family": "fixture-moe",
             "exact_stored_parameter_count": 100_000_000_000,
             "status": "pending", "admission": {"streaming_required": True},
             "parameter_manifest": {"active_moe_parameter_count": 1}},
        ]
        _tiers, progress = methodology._tier_progress(cells)
        self.assertLess(
            progress["parameter_pass_completion_fraction"],
            progress["cell_completion_fraction"],
        )

    def test_32b_72b_and_120b_are_mountains(self) -> None:
        tiers = {row["model_label"]: row for row in self.report["progress"]["tiers"]}
        for label in ("32B", "72B", "120B"):
            self.assertEqual("mountain", tiers[label]["classification"])
        self.assertIn("architecture_discontinuity", tiers["120B"]["mountain_triggers"])
        self.assertIn("streaming_required", tiers["120B"]["mountain_triggers"])

    def test_gptoss_uses_exact_stored_active_and_source_accounting(self) -> None:
        gptoss = self.report["gptoss_120b"]
        self.assertEqual(116_829_156_672, gptoss["stored_parameters"])
        self.assertEqual(5_132_852_352, gptoss["active_moe_parameters"])
        self.assertEqual(615, gptoss["source_units"])
        self.assertEqual(24_600, gptoss["isolated_jobs"])
        self.assertGreater(gptoss["mxfp4_source_physical_bpw"], 4.4)
        self.assertLess(gptoss["mxfp4_source_physical_bpw"], 4.5)

    def test_nominal_bpw_cannot_become_the_promotion_metric(self) -> None:
        self.assertTrue(
            self.report["empirical_quality"][
                "nominal_target_bpw_is_not_a_promotion_metric"
            ]
        )
        self.assertTrue(
            self.report["empirical_quality"]["all_current_quality_is_provisional"]
        )
        self.assertFalse(
            self.report["empirical_quality"]["methodology_quality_claims_permitted"]
        )

    def test_pareto_dominance_uses_bytes_ppl_and_capability(self) -> None:
        better = {
            "actual_model_payload_bpw": 2.0,
            "ppl_relative_delta": 0.1,
            "capability_absolute_delta": 0.0,
        }
        worse = {
            "actual_model_payload_bpw": 2.2,
            "ppl_relative_delta": 0.2,
            "capability_absolute_delta": -0.1,
        }
        self.assertTrue(methodology._dominates(better, worse))
        self.assertFalse(methodology._dominates(worse, better))

    def test_completed_rung_exposes_frontier_repeat_speed_signal(self) -> None:
        rows = []
        values = {
            "codec_control": (3.769, 0.821, -0.05, 2800),
            "doctor_static": (3.909, 1.004, -0.15, 2750),
            "doctor_conditional": (3.652, 1.037, -0.15, 2000),
            "doctor_full": (4.041, 1.449, -0.10, 2580),
        }
        for branch, (bpw, ppl, capability, seconds) in values.items():
            rows.append({
                "model_label": "7B", "rate_id": "3", "branch": branch,
                "target_bpw": 3.0,
                "actual_model_payload_bpw": bpw,
                "ppl_relative_delta": ppl,
                "capability_absolute_delta": capability,
                "receipt_window_wall_seconds": seconds,
            })
        signal = methodology._rung_branch_efficiency(rows)
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(
            ["codec_control", "doctor_conditional"],
            signal["pareto_active_branches"],
        )
        self.assertEqual(
            ["doctor_static", "doctor_full"], signal["dominated_branches"])
        self.assertEqual("bad", signal["promotion_result"])
        self.assertEqual("useful-negative", signal["evidence_value"])
        self.assertEqual(
            ["physical_density", "competitive_quality"], signal["failed_gates"])
        self.assertTrue(signal["exhaustive_track_retains_all_branches"])
        self.assertGreater(signal["observed_dominated_receipt_fraction"], 0.5)

    def test_methodology_has_frontier_and_exhaustive_tracks(self) -> None:
        tracks = self.report["condenser_methodology"]["two_track_completion"]
        self.assertIn("frontier_track", tracks)
        self.assertIn("exhaustive_track", tracks)
        self.assertFalse(tracks["automatic_default_promotion_permitted"])
        quality = self.report["empirical_quality"]
        self.assertTrue(quality["negative_evidence_is_retained"])
        self.assertTrue(
            quality["exhaustive_track_retains_empirically_dominated_branches"])

    def test_resealed_execution_permission_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.report)
        tampered["claim_boundary"]["execution_permitted"] = True
        tampered.pop("report_sha256")
        tampered["report_sha256"] = methodology._hash_value(tampered)
        self.assertTrue(any(
            "claim boundary" in error
            for error in methodology.validate_report(tampered)
        ))

    def test_resealed_120b_reclassification_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.report)
        tampered["gptoss_120b"]["classification"] = "ladder"
        tampered.pop("report_sha256")
        tampered["report_sha256"] = methodology._hash_value(tampered)
        self.assertTrue(any(
            "mountain identity" in error
            for error in methodology.validate_report(tampered)
        ))

    def test_atomic_report_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "report.json"
            methodology._atomic_json(path, self.report)
            loaded = json.loads(path.read_text())
            self.assertEqual([], methodology.validate_report(loaded))


if __name__ == "__main__":
    unittest.main()
