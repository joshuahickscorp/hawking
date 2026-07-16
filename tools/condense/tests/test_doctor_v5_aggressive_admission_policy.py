#!/usr/bin/env python3.12
"""Cheap adversarial gates for the unbound aggressive-admission generation."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import unittest


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))
import doctor_v5_aggressive_admission_policy as policy


class AggressiveAdmissionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = self._plan()
        self.state = self._state(self.plan)

    def test_authenticated_hwm_replaces_whole_parent_projection(self) -> None:
        rows = self._samples("large", rss_bytes=36_000_000_000)
        evidence = policy.authenticated_process_tree_evidence(
            self.plan, self.state, rows
        )
        decision = policy.reservation_for_cell(self.plan["cells"][0], evidence)
        self.assertEqual("authenticated-process-tree-high-water", decision["source"])
        self.assertFalse(decision["exclusive_canary"])
        self.assertEqual(41_400_000_000, decision["deterministic_margin_bytes"]
                         + decision["high_water_bytes"])
        self.assertEqual(41_472_000_000, decision["reservation_bytes"])
        # The measured reservation is independent of the 100GB whole-parent
        # source projection embedded in this synthetic cell.
        self.assertLess(decision["reservation_bytes"],
                        policy.ADMISSION_CEILING_BYTES)

    def test_uncalibrated_profile_is_exclusive_not_optimistic(self) -> None:
        evidence = policy.authenticated_process_tree_evidence(
            self.plan, self.state, self._samples("large", count=2)
        )
        decision = policy.reservation_for_cell(self.plan["cells"][0], evidence)
        self.assertTrue(decision["exclusive_canary"])
        self.assertEqual(policy.ADMISSION_CEILING_BYTES,
                         decision["reservation_bytes"])

    def test_sample_authentication_rejects_request_and_rss_forgery(self) -> None:
        rows = self._samples("large")
        forged_request = copy.deepcopy(rows[0])
        forged_request["request_sha256"] = "f" * 64
        forged_sum = copy.deepcopy(rows[1])
        forged_sum["processes"][1]["rss_bytes"] += 1
        duplicate_pid = copy.deepcopy(rows[2])
        duplicate_pid["processes"][1]["pid"] = duplicate_pid["processes"][0]["pid"]
        evidence = policy.authenticated_process_tree_evidence(
            self.plan, self.state, [*rows, forged_request, forged_sum, duplicate_pid]
        )
        self.assertEqual(len(rows), evidence["accepted_sample_count"])
        self.assertEqual(3, evidence["rejected_sample_count"])
        self.assertIn("request binding mismatch", evidence["rejected_reasons"])
        self.assertIn("process-tree RSS is not the exact member sum",
                      evidence["rejected_reasons"])
        self.assertIn("process membership/RSS is invalid",
                      evidence["rejected_reasons"])

    def test_evidence_tamper_cannot_lower_reservation(self) -> None:
        evidence = policy.authenticated_process_tree_evidence(
            self.plan, self.state, self._samples("large")
        )
        tampered = copy.deepcopy(evidence)
        key = policy.residency_profile_key(self.plan["cells"][0])
        tampered["profiles"][key]["reservation_bytes"] = 1
        with self.assertRaises(policy.PolicyError):
            policy.reservation_for_cell(self.plan["cells"][0], tampered)

    def test_thread_profiles_are_exact_parity_gated(self) -> None:
        large, companion, small = self.plan["cells"]
        policy.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with __import__("tempfile").TemporaryDirectory(dir=policy.STAGE_ROOT) as temp:
            qualification, _, _ = self._qualified_thread_profile(Path(temp))
            # The vendor measurements deliberately select 20 for 14B.  This
            # proves there is no nominal-tier 8/12 override left in admission.
            self.assertEqual(20, policy.selected_thread_profile(
                companion, qualification
            )["threads"])
            self.assertEqual(16, policy.selected_thread_profile(
                large, qualification
            )["threads"])
            self.assertEqual(8, policy.selected_thread_profile(
                small, qualification
            )["threads"])
            selected = policy.selected_thread_profile(companion, qualification)
            self.assertTrue(selected["exclusive_cpu_profile"])
            self.assertEqual([8, 12, 16, 20], [
                row["threads"] for row in selected["candidate_measurements"]
            ])

    def test_exact_packer_selects_heterogeneous_sixteen_plus_eight(self) -> None:
        candidates = [
            {"cell_id": "large", "priority": 0, "reservation_bytes": 38_000_000_000,
             "threads": 16, "exact_parity_approved": True,
             "all_four_candidates_eligible": True,
             "selection_source": "qualified-vendor-thread-profile-contract",
             "projected_wall_seconds": 5.0},
            {"cell_id": "companion", "priority": 1,
             "reservation_bytes": 22_000_000_000, "threads": 8,
             "exact_parity_approved": True, "all_four_candidates_eligible": True,
             "selection_source": "qualified-vendor-thread-profile-contract",
             "projected_wall_seconds": 6.0},
            {"cell_id": "small", "priority": 2, "reservation_bytes": 9_000_000_000,
             "threads": 8, "exact_parity_approved": True,
             "all_four_candidates_eligible": True,
             "selection_source": "qualified-vendor-thread-profile-contract",
             "projected_wall_seconds": 20.0},
        ]
        pack = policy.choose_heterogeneous_pack(candidates)
        self.assertEqual(["large", "companion"], pack["selected_cell_ids"])
        self.assertEqual(24, pack["threads_after"])
        self.assertTrue(pack["heterogeneous_16_plus_8"])
        self.assertLessEqual(pack["reserved_after_bytes"],
                             policy.ADMISSION_CEILING_BYTES)

    def test_packer_rejects_unapproved_profile(self) -> None:
        with self.assertRaises(policy.PolicyError):
            policy.choose_heterogeneous_pack([{
                "cell_id": "x", "priority": 0, "reservation_bytes": 1,
                "threads": 8, "exact_parity_approved": False,
                "all_four_candidates_eligible": True,
                "selection_source": "qualified-vendor-thread-profile-contract",
                "projected_wall_seconds": 1.0,
            }])

    def test_twenty_thread_profile_is_exclusive(self) -> None:
        pack = policy.choose_heterogeneous_pack([
            {"cell_id": "exclusive", "priority": 0,
             "reservation_bytes": 20_000_000_000, "threads": 20,
             "exclusive_cpu_profile": True, "exact_parity_approved": True,
             "all_four_candidates_eligible": True,
             "selection_source": "qualified-vendor-thread-profile-contract",
             "projected_wall_seconds": 2.0},
            {"cell_id": "companion", "priority": 1,
             "reservation_bytes": 10_000_000_000, "threads": 8,
             "exclusive_cpu_profile": False, "exact_parity_approved": True,
             "all_four_candidates_eligible": True,
             "selection_source": "qualified-vendor-thread-profile-contract",
             "projected_wall_seconds": 5.0},
        ])
        self.assertEqual(1, len(pack["selected"]))
        self.assertEqual(20, pack["selected"][0]["threads"])

    def test_slower_twenty_loses_to_measured_sixteen_plus_eight(self) -> None:
        base = {"exact_parity_approved": True,
                "all_four_candidates_eligible": True,
                "selection_source": "qualified-vendor-thread-profile-contract"}
        pack = policy.choose_heterogeneous_pack([
            {**base, "cell_id": "exclusive", "priority": 0,
             "reservation_bytes": 20_000_000_000, "threads": 20,
             "exclusive_cpu_profile": True, "projected_wall_seconds": 3.0},
            {**base, "cell_id": "large", "priority": 1,
             "reservation_bytes": 30_000_000_000, "threads": 16,
             "exclusive_cpu_profile": False, "projected_wall_seconds": 5.0},
            {**base, "cell_id": "companion", "priority": 2,
             "reservation_bytes": 20_000_000_000, "threads": 8,
             "exclusive_cpu_profile": False, "projected_wall_seconds": 5.0},
        ])
        self.assertEqual(["large", "companion"], pack["selected_cell_ids"])
        self.assertAlmostEqual(0.4, pack["projected_throughput_cells_per_second"])

    def test_swap_soft_hard_emergency_are_bounded(self) -> None:
        state = policy.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=0.0
        )
        state, soft = policy.advance_swap_state(
            state, {"pressure_level": 1, "swap_used_mb": 600.0},
            now_epoch=300.0, sealed_baseline_swap_mb=0.0,
        )
        self.assertEqual("soft_throttle", soft["mode"])
        self.assertTrue(soft["allow_launch"])
        self.assertEqual(1, soft["launch_limit"])
        self.assertFalse(soft["shed_one"])
        state, hard = policy.advance_swap_state(
            state, {"pressure_level": 2, "swap_used_mb": 1700.0},
            now_epoch=600.0, sealed_baseline_swap_mb=0.0,
        )
        self.assertEqual("hard_stop", hard["mode"])
        self.assertFalse(hard["allow_launch"])
        self.assertFalse(hard["shed_one"])
        state, emergency = policy.advance_swap_state(
            state, {"pressure_level": 4, "swap_used_mb": 3200.0},
            now_epoch=700.0, sealed_baseline_swap_mb=0.0,
        )
        self.assertEqual("emergency_shed", emergency["mode"])
        self.assertTrue(emergency["shed_one"])
        state, held = policy.advance_swap_state(
            state, {"pressure_level": 4, "swap_used_mb": 3300.0},
            now_epoch=730.0, sealed_baseline_swap_mb=0.0,
        )
        self.assertFalse(held["shed_one"])
        self.assertFalse(held["running_evidence_invalidated"])

    def test_swap_hysteresis_needs_cooldown_and_green_streak(self) -> None:
        state = policy.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=0.0
        )
        state, _ = policy.advance_swap_state(
            state, {"pressure_level": 2, "swap_used_mb": 0.0},
            now_epoch=10.0, sealed_baseline_swap_mb=0.0,
        )
        for when in (100.0, 190.0):
            state, decision = policy.advance_swap_state(
                state, {"pressure_level": 1, "swap_used_mb": 0.0},
                now_epoch=when, sealed_baseline_swap_mb=0.0,
            )
            self.assertEqual("hard_stop", decision["mode"])
        state, decision = policy.advance_swap_state(
            state, {"pressure_level": 1, "swap_used_mb": 0.0},
            now_epoch=191.0, sealed_baseline_swap_mb=0.0,
        )
        self.assertEqual("green", decision["mode"])

    def test_invalid_swap_state_self_heals_without_rebaseline_or_kill(self) -> None:
        state = policy.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 100.0}, now_epoch=0.0
        )
        state["baseline_swap_mb"] = 900.0
        healed, decision = policy.advance_swap_state(
            state, {"pressure_level": 1, "swap_used_mb": 200.0},
            now_epoch=10.0, sealed_baseline_swap_mb=100.0,
        )
        self.assertEqual(100.0, healed["baseline_swap_mb"])
        self.assertEqual("hard_stop", decision["mode"])
        self.assertFalse(decision["shed_one"])
        self.assertTrue(healed["recovered_from_invalid"])

    def test_unknown_probe_stops_launch_without_blind_termination(self) -> None:
        state = policy.initial_swap_state(
            {"pressure_level": 1, "swap_used_mb": 0.0}, now_epoch=0.0
        )
        _, decision = policy.advance_swap_state(
            state, {"pressure_level": None, "swap_used_mb": None},
            now_epoch=10.0, sealed_baseline_swap_mb=0.0,
        )
        self.assertEqual("hard_stop", decision["mode"])
        self.assertFalse(decision["allow_launch"])
        self.assertFalse(decision["shed_one"])

    def test_overlay_and_promotion_are_fail_closed(self) -> None:
        policy.STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with __import__("tempfile").TemporaryDirectory(dir=policy.STAGE_ROOT) as temp:
            _, profile_path, binary_path = self._qualified_thread_profile(Path(temp))
            overlay = policy.build_overlay(
                self.plan, self.state,
                self._samples("large") + self._samples("companion")
                + self._samples("small"),
                baseline_snapshot={"pressure_level": 1, "swap_used_mb": 0.0},
                thread_profile_path=profile_path, thread_binary_path=binary_path,
            )
            self.assertEqual([], policy.validate_overlay(overlay))
            self.assertEqual("qualified",
                             overlay["thread_profile_qualification"]["status"])
            blocked = policy.promotion_gate(
                overlay, self.plan, self.state,
                self._samples("large") + self._samples("companion")
                + self._samples("small"),
                snapshot={"pressure_level": 1, "swap_used_mb": 0.0},
            )
            self.assertFalse(blocked["ready"])
            self.assertIn("queue is not paused/drained", blocked["blockers"])
            quiescent = copy.deepcopy(self.state)
            quiescent["status"] = "drained"
            quiescent["state_sha256"] = policy._hash_value(
                policy._without(quiescent, "state_sha256")
            )
            stale = policy.promotion_gate(
                overlay, self.plan, quiescent,
                self._samples("large") + self._samples("companion")
                + self._samples("small"),
                snapshot={"pressure_level": 1, "swap_used_mb": 0.0},
            )
            self.assertIn("overlay must be re-staged at the exact promotion checkpoint",
                          stale["blockers"])
            # The profile file itself is re-hashed and every receipt is
            # revalidated during promotion, not trusted from the overlay copy.
            profile_path.write_text("{}\n", encoding="utf-8")
            drifted = policy.promotion_gate(
                overlay, self.plan, self.state,
                self._samples("large") + self._samples("companion")
                + self._samples("small"),
                snapshot={"pressure_level": 1, "swap_used_mb": 0.0},
            )
            self.assertTrue(any("thread profile" in row for row in drifted["blockers"]))
            tampered = copy.deepcopy(overlay)
            tampered["resource_policy"]["global_reserve_bytes"] = 1
            self.assertTrue(policy.validate_overlay(tampered))

    @staticmethod
    def _cell(cell_id: str, nominal: float, branch: str, priority: int,
              whole_parent: bool) -> dict:
        return {
            "cell_id": cell_id, "model_family": "qwen2.5-dense",
            "model_label": f"{nominal:g}B",
            "nominal_params_b": nominal, "branch": branch, "priority": priority,
            "rate_id": "q3",
            "admission": {"whole_parent_residency_assumed": whole_parent},
            "parameter_manifest": {"source_weight_bytes": 100_000_000_000,
                                   "largest_source_shard_bytes": 4_000_000_000},
        }

    def _plan(self) -> dict:
        value = {
            "schema": "hawking.doctor_v5_ultra_campaign_plan.v1",
            "cells": [
                self._cell("large", 32.0, "codec_control", 0, False),
                self._cell("companion", 14.0, "codec_control", 1, True),
                self._cell("small", 3.0, "codec_control", 2, True),
            ],
        }
        value["plan_sha256"] = policy._hash_value(value)
        return value

    @staticmethod
    def _state(plan: dict) -> dict:
        cells = {}
        for index, cell in enumerate(plan["cells"], start=1):
            cells[cell["cell_id"]] = {
                "status": "pending", "request_sha256": f"{index:064x}",
            }
        value = {
            "schema": "hawking.doctor_v5_ultra_queue_state.v1",
            "plan_sha256": plan["plan_sha256"], "status": "running",
            "active_children": {}, "cells": cells,
        }
        value["state_sha256"] = policy._hash_value(value)
        return value

    def _qualified_thread_profile(self, directory: Path) \
            -> tuple[dict, Path, Path]:
        contract = policy._load_thread_contract()
        binary_path = directory / "quantizer"
        binary_path.write_bytes(b"exact-test-binary")
        binary_sha = hashlib.sha256(binary_path.read_bytes()).hexdigest()
        receipts = []
        winners = {
            "32B": {8: 10.0, 12: 8.0, 16: 5.0, 20: 6.0},
            "14B": {8: 10.0, 12: 8.0, 16: 7.0, 20: 4.0},
            "3B": {8: 4.0, 12: 5.0, 16: 6.0, 20: 7.0},
        }
        for tier, timings in winners.items():
            for threads, wall in timings.items():
                receipt = {
                    "schema": contract.RECEIPT_SCHEMA, "status": "pass",
                    "scope": "production", "synthetic": False,
                    "tier": tier, "rate": "q3", "threads": threads,
                    "binary_sha256": binary_sha, "source_sha256": "b" * 64,
                    "canonical_output_sha256": "c" * 64,
                    "output_sha256": "c" * 64, "exact_output": True,
                    "wall_seconds": wall, "peak_rss_bytes": 10_000 + threads,
                    "scratch_budget_bytes": 268_435_456, "mode": "block_parallel",
                }
                path = directory / f"{tier}-{threads}.json"
                path.write_text(json.dumps(receipt, sort_keys=True) + "\n",
                                encoding="utf-8")
                receipts.append(path)
        profile = contract.build_profile(
            receipts, expected_binary_sha256=binary_sha,
            rss_limit_bytes=1_000_000,
        )
        profile_path = directory / "thread-profile.json"
        profile_path.write_text(json.dumps(profile, sort_keys=True) + "\n",
                                encoding="utf-8")
        qualification = policy.qualify_thread_profile(
            self.plan["cells"], profile_path=profile_path, binary_path=binary_path,
        )
        self.assertEqual("qualified", qualification["status"],
                         qualification["blockers"])
        return qualification, profile_path, binary_path

    def _samples(self, cell_id: str, *, rss_bytes: int = 36_000_000_000,
                 count: int = policy.MIN_AUTHENTICATED_SAMPLES) -> list[dict]:
        state_row = self.state["cells"][cell_id]
        start = dt.datetime(2026, 7, 14, tzinfo=dt.timezone.utc)
        rows = []
        for index in range(count):
            pgid = 10_000 + 10 * list(self.state["cells"]).index(cell_id)
            root_rss = 1_000_000_000
            row = {
                "cell_id": cell_id, "plan_sha256": self.plan["plan_sha256"],
                "request_sha256": state_row["request_sha256"],
                "process_budget_bytes": policy.PROCESS_BUDGET_BYTES,
                "sampled_at": (start + dt.timedelta(seconds=index * 8)).isoformat(),
                "root_pid": pgid, "pgid": pgid, "process_count": 2,
                "processes": [
                    {"pid": pgid, "ppid": 1, "pgid": pgid,
                     "rss_bytes": root_rss, "state": "S"},
                    {"pid": pgid + 1, "ppid": pgid, "pgid": pgid,
                     "rss_bytes": rss_bytes - root_rss, "state": "R"},
                ],
                "tree_rss_bytes": rss_bytes, "max_tree_rss_bytes": rss_bytes,
                "at_or_over_budget": False,
            }
            rows.append(row)
        return rows


if __name__ == "__main__":
    unittest.main()
