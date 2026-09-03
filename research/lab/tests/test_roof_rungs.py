"""S004 §4 rung instrument: place speed claims on the measured roof."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from ascent.roof_rungs import (
    FS_FIELD,
    HONEST_DECODE_CEILING_GB_S,
    QWEN38_CATALOG_ADDR_GB_S,
    QWEN38_CATALOG_FULL_GB_S,
    QWEN38_DEFENDED_GEMV_BYTES,
    QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
    QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S,
    QWEN38_SINGLE_GEMV_ADDR_GB_S,
    REUSE_BAND_GB_S,
    UNIQUE_ONCE_1024MIB_GB_S,
    TODAY,
    flag_fs_latency_language,
    judge_physical_limit_claim,
    load_only_floor_ns,
    markdown_table,
    place,
    place_today,
    roof_tok_s,
    today_table,
    verify_bandwidth_receipt,
    verify_qwen38_roof_receipt,
)


def approx(value: float, expected: float, abs_tol: float) -> None:
    if abs(value - expected) > abs_tol:
        raise AssertionError(f"{value} != {expected} ± {abs_tol}")


class RoofRungsTest(unittest.TestCase):
    def test_bytes_and_ns_alone_emit_every_required_field(self) -> None:
        row = place(2_217_278_160, 1_170_679_064)
        required = (
            "bytes_per_token",
            "measured_memory_bandwidth_gb_s",
            "arithmetic_intensity_flop_per_byte",
            "gpu_occupancy",
            "dispatch_floor",
            "reconstruction_cost",
            "synchronization_floor",
            "roof_tok_s",
            "fraction_of_roof",
            FS_FIELD,
        )
        for key in required:
            self.assertIn(key, row)
        self.assertEqual(row["bytes_per_token"], 2_217_278_160)
        self.assertIsNone(row[FS_FIELD])
        self.assertTrue(row["fs_honesty"]["not_latency"])
        self.assertIn("NOT physical femtosecond latency", row["fs_honesty"]["caveat"])
        self.assertTrue(row["hardware"]["reuse_band_is_not_the_decode_ceiling"])
        self.assertEqual(row["rung"]["current_rung"], "below_A")

    def test_qwen38_today_is_provisional_roof_only_without_current_token_ns(self) -> None:
        row = place_today("qwen38")
        self.assertIsNone(row["current_token_ns"])
        self.assertIsNone(row["current_tps"])
        self.assertIsNone(row["ns_per_token"])
        self.assertIsNone(row["roof_tok_s"])
        self.assertEqual(row["rung"]["current_rung"], "UNMEASURED")
        self.assertFalse(row["may_claim_physical_limit"])
        self.assertEqual(
            row["physical_limit"]["verdict"], "PROVISIONAL_NOT_PHYSICAL_LIMIT"
        )
        self.assertEqual(row["defended_gemv_payload_bytes"], QWEN38_DEFENDED_GEMV_BYTES)
        approx(
            row["hardware"]["single_gemv_addr_gb_s"],
            QWEN38_SINGLE_GEMV_ADDR_GB_S,
            1e-9,
        )
        approx(
            row["hardware"]["sealed_weight_addressing_gb_s"],
            QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S,
            1e-9,
        )
        approx(
            row["hardware"]["sealed_over_single_gemv_addr"],
            QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
            1e-12,
        )
        approx(row["hardware"]["catalog_addr_gb_s"], QWEN38_CATALOG_ADDR_GB_S, 1e-9)
        approx(row["hardware"]["catalog_full_gb_s"], QWEN38_CATALOG_FULL_GB_S, 1e-9)
        self.assertTrue(row["hardware"]["provisional"])
        self.assertTrue(row["hardware"]["cpu_contended"])
        self.assertFalse(row["hardware"]["kernel_and_dispatch_topology_headroom_closed"])
        self.assertIn("Current complete-token TOKEN_NS/TPS was not rerun", row["hardware"]["caveat"])
        self.assertIsNone(row[FS_FIELD])
        self.assertTrue(row["fs_honesty"]["not_latency"])

    def test_q80_mixed_is_off_the_roof_and_can_still_reach_b(self) -> None:
        row = place_today("q80_mixed")
        approx(row["measured_gpu_bandwidth_gb_s"], 2.57, 0.02)
        approx(row["gpu_occupancy"]["fraction_of_honest_decode_ceiling"], 0.0062, 0.0003)
        self.assertFalse(row["may_claim_physical_limit"])
        self.assertEqual(row["rung"]["current_rung"], "below_A")
        approx(row["rung"]["tps"], 0.854, 0.005)
        self.assertEqual(
            row["reachable_at_current_bytes"]["reachable_at_current_bytes"], ["A", "B"]
        )
        approx(row["roof_tok_s"], 185.6, 0.5)
        self.assertGreater(row["reconstruction_cost"]["excess_over_load_only_ns"], 800_000_000)

    def test_dsv4f_can_reach_a_only_and_is_not_at_a_physical_limit(self) -> None:
        row = place_today("dsv4f")
        self.assertEqual(row["rung"]["current_rung"], "below_A")
        approx(row["rung"]["tps"], 0.964, 0.005)
        self.assertEqual(
            row["reachable_at_current_bytes"]["reachable_at_current_bytes"], ["A"]
        )
        approx(row["roof_tok_s"], 70.3, 0.3)
        self.assertFalse(row["may_claim_physical_limit"])
        approx(row["measured_gpu_bandwidth_gb_s"], 14.68, 0.1)

    def test_reuse_band_is_rejected_as_a_decode_ceiling(self) -> None:
        verdict = judge_physical_limit_claim(
            saturated_resource="DRAM",
            evidence="560-647 GB/s probe",
            cites_reuse_band_as_decode_ceiling=True,
        )
        self.assertEqual(verdict["verdict"], "FAIL")
        self.assertTrue(any("cache-resident" in f for f in verdict["failures"]))

    def test_no_further_optimization_is_rejected(self) -> None:
        verdict = judge_physical_limit_claim(
            saturated_resource="something",
            evidence="vibes",
            claim_text="No further optimization is obvious, so this is the physical limit.",
        )
        self.assertEqual(verdict["verdict"], "FAIL")
        self.assertTrue(any("not evidence" in f for f in verdict["failures"]))

    def test_published_819_is_rejected_as_decode_ceiling(self) -> None:
        verdict = judge_physical_limit_claim(
            saturated_resource="819 GB/s peak",
            evidence="datasheet",
            cites_published_819_as_decode_ceiling=True,
        )
        self.assertEqual(verdict["verdict"], "FAIL")
        self.assertTrue(any("411.51" in f for f in verdict["failures"]))

    def test_qwen38_ratio_and_catalog_gap_leave_topology_headroom(self) -> None:
        approx(
            QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
            QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S / QWEN38_SINGLE_GEMV_ADDR_GB_S,
            1e-12,
        )
        approx(QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR, 0.9137740261194911, 1e-12)
        self.assertLess(QWEN38_CATALOG_ADDR_GB_S, QWEN38_SINGLE_GEMV_ADDR_GB_S)
        self.assertLess(QWEN38_CATALOG_FULL_GB_S, QWEN38_CATALOG_ADDR_GB_S)
        row = place_today("qwen38")
        self.assertFalse(row["may_claim_physical_limit"])
        self.assertIn("headroom open", row["physical_limit"]["failures"][0])

    def test_unsaturated_named_resource_fails(self) -> None:
        verdict = judge_physical_limit_claim(
            saturated_resource="decode bandwidth",
            evidence="2.57 GB/s",
            achieved=2.57,
            ceiling=HONEST_DECODE_CEILING_GB_S,
        )
        self.assertEqual(verdict["verdict"], "FAIL")
        self.assertTrue(any("saturation" in f for f in verdict["failures"]))

    def test_fs_metric_is_never_described_as_latency(self) -> None:
        row = place_today("q80_mixed")
        blob = json.dumps(row)
        self.assertIn("NOT physical femtosecond latency", blob)
        self.assertIn(FS_FIELD, row)
        self.assertTrue(row["fs_honesty"]["not_latency"])
        flagged = flag_fs_latency_language(
            "across 60 cores the effective per-FLOP time is femtoseconds. "
            "sub-nanosecond latency is already true."
        )
        self.assertTrue(flagged["flagged"])

    def test_roof_math_matches_sealed_control(self) -> None:
        bytes_tok = 2_217_278_160
        approx(load_only_floor_ns(bytes_tok), bytes_tok / HONEST_DECODE_CEILING_GB_S, 1e-9)
        approx(roof_tok_s(bytes_tok), 1.0e9 / load_only_floor_ns(bytes_tok), 1e-9)
        approx(UNIQUE_ONCE_1024MIB_GB_S, 301.63405407683126, 1e-9)
        self.assertGreater(REUSE_BAND_GB_S[0], HONEST_DECODE_CEILING_GB_S)

    def test_today_table_and_receipt_constants(self) -> None:
        check = verify_bandwidth_receipt()
        self.assertTrue(check["present"])
        self.assertTrue(check["matches_sealed_constants"])
        self.assertTrue(check["reuse_not_decode_ceiling"])
        qwen_check = verify_qwen38_roof_receipt()
        self.assertTrue(qwen_check["present"])
        self.assertTrue(qwen_check["matches_landed_constants"])
        self.assertTrue(qwen_check["cpu_contended"])
        self.assertTrue(qwen_check["sealed_ledger_cited_not_rerun"])
        self.assertFalse(qwen_check["current_token_ns_remeasured"])
        doc = today_table()
        self.assertEqual(set(doc["models"]), {"q80_mixed", "qwen38", "dsv4f"})
        md = markdown_table(doc)
        self.assertIn("q80_mixed", md)
        self.assertIn("qwen38", md)
        self.assertIn("dsv4f", md)
        self.assertIn("below_A", md)
        self.assertIn("Q80 historical unique-once control", md)
        self.assertIn("699.574", md)
        self.assertIn("639.252", md)
        self.assertIn("91.38%", md)
        self.assertIn("530.654/505.810", md)
        self.assertIn("CPU-contended", md)
        self.assertIn("TOKEN_NS/TPS was not rerun", md)
        self.assertNotIn("98.7%", md)
        self.assertNotIn("406.2", md)
        self.assertNotIn("no kernel headroom", md)
        audit = {row["receipt"]: row["verdict"] for row in doc["physical_limit_audit"]}
        self.assertEqual(
            audit["receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json"],
            "PROVISIONAL_NOT_PHYSICAL_LIMIT",
        )
        self.assertEqual(
            audit["receipts/ascent-2026-08-16/TERMINAL_TARGET.json THE_SINGLE_SHARED_BLOCKER"],
            "FAIL",
        )
        self.assertEqual(
            audit["receipts/ascent-2026-08-16/Q80_MIXED_RECONSTRUCTION_WALL.json"],
            "PASS_NO_LIMIT_CLAIMED",
        )
        self.assertTrue(doc["fs_latency_flags"][0]["flag"]["flagged"])
        self.assertIsNone(TODAY["qwen38"]["current_token_ns"])
        self.assertNotIn("ns_per_token", TODAY["qwen38"])
        self.assertEqual(doc["models"]["qwen38"]["rung"]["current_rung"], "UNMEASURED")
        qwen_blob = json.dumps(
            {
                "model": doc["models"]["qwen38"],
                "roof": doc["hardware_roof"]["qwen38_provisional"],
                "audit": doc["physical_limit_audit"][0],
            }
        )
        self.assertNotIn("411.51", qwen_blob)
        self.assertNotIn("98.7", qwen_blob)
        self.assertNotIn("406.2", qwen_blob)

    def test_rejects_negative_inputs(self) -> None:
        with self.assertRaises(ValueError):
            place(-1, 1000)
        with self.assertRaises(ValueError):
            place(1000, math.nan)


if __name__ == "__main__":
    unittest.main()
