#!/usr/bin/env python3.12
"""Adversarial tests for the inert post-120B mountain ladder."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))

import doctor_v5_mountain_ladder as mountain


NOW = "2026-07-15T12:00:00+00:00"


def hardware(*, free: int = 800 * mountain.GB) -> dict:
    return {
        "profile": "fixture",
        "total_memory_bytes": 96 * mountain.GIB,
        "process_budget_bytes": 78 * mountain.GB,
        "logical_cpu_count": 28,
        "disk_total_bytes": 1_000 * mountain.GB,
        "disk_free_bytes": free,
        "disk_reserve_bytes": 150 * mountain.GB,
        "cache_reserve_bytes": 32 * mountain.GB,
        "stream_workspace_bytes": 32 * mountain.GB,
        "runtime_working_bytes": 20 * mountain.GB,
        "artifact_overhead_ppm": 80_000,
    }


def rehash(plan: dict) -> None:
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = mountain._hash_value(plan)


class MountainLadderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = mountain.build_plan(hardware(), created_at=NOW)

    def test_plan_carries_exact_10x4_to_all_three_mountains(self) -> None:
        self.assertEqual([], mountain.validate_plan(self.plan))
        self.assertEqual(120, self.plan["coverage"]["cell_templates"])
        self.assertTrue(all(len(row["cells"]) == 40 for row in self.plan["models"]))
        self.assertTrue(all(
            cell["execution_permitted"] is False
            for model in self.plan["models"] for cell in model["cells"]
        ))

    def test_storage_frontier_distinguishes_source_and_resident_fit(self) -> None:
        kimi = self.plan["models"][1]["storage"]
        by_rate = {row["rate_id"]: row for row in kimi["rates"]}
        self.assertTrue(kimi["full_source_fits_total_disk_before_output"])
        self.assertFalse(by_rate["0.5"]["fits_process_budget"])
        self.assertTrue(by_rate["0.33"]["fits_process_budget"])

        pro = self.plan["models"][2]
        pro_rates = {row["rate_id"]: row for row in pro["storage"]["rates"]}
        self.assertFalse(pro["full_install_permitted"])
        self.assertFalse(pro_rates["0.33"]["fits_process_budget"])
        self.assertTrue(pro_rates["0.25"]["fits_process_budget"])

    def test_low_current_free_space_blocks_without_changing_lifecycle_fit(self) -> None:
        plan = mountain.build_plan(hardware(free=180 * mountain.GB), created_at=NOW)
        flash = {row["rate_id"]: row for row in plan["models"][0]["storage"]["rates"]}
        self.assertTrue(flash["0.5"]["fits_total_disk_lifecycle"])
        self.assertFalse(flash["0.5"]["fits_current_free_disk"])
        self.assertGreater(flash["0.5"]["additional_free_bytes_required_now"], 0)

    def test_resealed_cell_substitution_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["models"][0]["cells"][0]["rate_id"] = "invented"
        rehash(tampered)
        self.assertTrue(any(
            "10x4" in row for row in mountain.validate_plan(tampered)
        ))

    def test_resealed_cell_authority_substitution_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["models"][0]["cells"][0]["source_manifest_sha256"] = "f" * 64
        rehash(tampered)
        self.assertTrue(any(
            "identity templates" in row for row in mountain.validate_plan(tampered)
        ))

    def test_resealed_storage_optimism_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        pro_rates = {
            row["rate_id"]: row
            for row in tampered["models"][2]["storage"]["rates"]
        }
        pro_rates["0.33"]["fits_process_budget"] = True
        rehash(tampered)
        self.assertTrue(any(
            "storage projection" in row for row in mountain.validate_plan(tampered)
        ))

    def test_resealed_execution_permission_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["models"][1]["phases"][0]["execution_permitted"] = True
        rehash(tampered)
        self.assertTrue(any(
            "executable" in row for row in mountain.validate_plan(tampered)
        ))

    def test_atomic_build_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "plan.json"
            mountain._atomic_json(path, self.plan)
            self.assertEqual([], mountain.validate_plan(mountain._read_json(path)))


if __name__ == "__main__":
    unittest.main()
