from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from thread_profile_contract import (  # noqa: E402
    ContractError,
    build_profile,
    verify_selection,
)


class ThreadProfileContractTests(unittest.TestCase):
    binary_sha = "a" * 64
    source_sha = "b" * 64
    output_sha = "c" * 64

    def receipt(self, threads: int, wall: float, **updates: object) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "hawking.strand.tier-rate-thread-canary.v1",
            "status": "pass",
            "scope": "production",
            "synthetic": False,
            "tier": "72B",
            "rate": "q3",
            "threads": threads,
            "binary_sha256": self.binary_sha,
            "source_sha256": self.source_sha,
            "canonical_output_sha256": self.output_sha,
            "output_sha256": self.output_sha,
            "exact_output": True,
            "wall_seconds": wall,
            "peak_rss_bytes": 10_000 + threads,
            "scratch_budget_bytes": 268_435_456,
            "mode": "block_parallel",
        }
        value.update(updates)
        return value

    def write_receipts(
        self, directory: Path, receipts: list[dict[str, object]]
    ) -> list[Path]:
        paths = []
        for index, receipt in enumerate(receipts):
            path = directory / f"receipt-{index}.json"
            path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
            paths.append(path)
        return paths

    def qualified_fixture(self, directory: Path) -> tuple[dict[str, object], list[Path]]:
        paths = self.write_receipts(
            directory,
            [
                self.receipt(8, 10.0),
                self.receipt(12, 7.0),
                self.receipt(16, 5.0),
                self.receipt(20, 6.0),
            ],
        )
        profile = build_profile(
            paths,
            expected_binary_sha256=self.binary_sha,
            rss_limit_bytes=1_000_000,
        )
        return profile, paths

    def test_complete_exact_matrix_selects_fastest_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            profile, _ = self.qualified_fixture(Path(temp))
            self.assertEqual(profile["status"], "qualified")
            selection = verify_selection(
                profile,
                tier="72B",
                rate="q3",
                binary_sha256=self.binary_sha,
            )
            self.assertTrue(selection["eligible"])
            self.assertEqual(selection["threads"], 16)
            self.assertEqual(selection["scratch_budget_bytes"], 268_435_456)

    def test_receipt_tamper_after_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            profile, paths = self.qualified_fixture(Path(temp))
            paths[0].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "missing or changed"):
                verify_selection(
                    profile,
                    tier="72B",
                    rate="q3",
                    binary_sha256=self.binary_sha,
                )

    def test_profile_winner_tamper_is_recomputed_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            profile, _ = self.qualified_fixture(Path(temp))
            entry = next(iter(profile["entries"].values()))
            entry["selected_threads"] = 20
            with self.assertRaisesRegex(ContractError, "does not match bound receipts"):
                verify_selection(
                    profile,
                    tier="72B",
                    rate="q3",
                    binary_sha256=self.binary_sha,
                )

    def test_missing_candidate_remains_partial_and_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.write_receipts(
                Path(temp),
                [self.receipt(8, 10.0), self.receipt(12, 7.0), self.receipt(16, 5.0)],
            )
            profile = build_profile(
                paths,
                expected_binary_sha256=self.binary_sha,
                rss_limit_bytes=1_000_000,
            )
            self.assertEqual(profile["status"], "partial")
            entry = next(iter(profile["entries"].values()))
            self.assertFalse(entry["qualified"])
            self.assertIn("20", entry["blockers"][0])
            with self.assertRaisesRegex(ContractError, "not qualified"):
                verify_selection(
                    profile,
                    tier="72B",
                    rate="q3",
                    binary_sha256=self.binary_sha,
                )

    def test_synthetic_or_non_exact_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            synthetic = self.receipt(8, 10.0, scope="synthetic", synthetic=True)
            paths = self.write_receipts(Path(temp), [synthetic])
            with self.assertRaisesRegex(ContractError, "production"):
                build_profile(
                    paths,
                    expected_binary_sha256=self.binary_sha,
                    rss_limit_bytes=1_000_000,
                )
            non_exact = self.receipt(8, 10.0, output_sha256="d" * 64)
            paths = self.write_receipts(Path(temp), [non_exact])
            with self.assertRaisesRegex(ContractError, "exact canonical"):
                build_profile(
                    paths,
                    expected_binary_sha256=self.binary_sha,
                    rss_limit_bytes=1_000_000,
                )

    def test_no_nearest_tier_or_rate_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            profile, _ = self.qualified_fixture(Path(temp))
            with self.assertRaisesRegex(ContractError, "fallback is forbidden"):
                verify_selection(
                    profile,
                    tier="120B",
                    rate="q3",
                    binary_sha256=self.binary_sha,
                )

    def test_fastest_over_rss_limit_is_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.write_receipts(
                Path(temp),
                [
                    self.receipt(8, 10.0),
                    self.receipt(12, 7.0),
                    self.receipt(16, 5.0, peak_rss_bytes=2_000_000),
                    self.receipt(20, 6.0),
                ],
            )
            profile = build_profile(
                paths,
                expected_binary_sha256=self.binary_sha,
                rss_limit_bytes=1_000_000,
            )
            selection = verify_selection(
                profile,
                tier="72B",
                rate="q3",
                binary_sha256=self.binary_sha,
            )
            self.assertEqual(selection["threads"], 20)

    def test_ordered_pipeline_candidates_bind_a_production_pipeline_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            pipeline_path = directory / "pipeline.json"
            pipeline_path.write_text(
                json.dumps(
                    {
                        "schema": "hawking.strand.quantize-model-ordered-pipeline-parity.v1",
                        "status": "pass",
                        "scope": "production",
                        "exact_output": True,
                        "canonical_order": True,
                        "dense_exact_match": True,
                        "sidecar_exact_match": True,
                        "packed_v2_exact_match": True,
                        "production_promotion_allowed": True,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            pipeline_sha = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
            paths = self.write_receipts(
                directory,
                [
                    self.receipt(
                        threads,
                        wall,
                        mode="ordered_pipeline",
                        pipeline_receipt_path=str(pipeline_path),
                        pipeline_receipt_sha256=pipeline_sha,
                    )
                    for threads, wall in [(8, 10.0), (12, 7.0), (16, 5.0), (20, 6.0)]
                ],
            )
            profile = build_profile(
                paths,
                expected_binary_sha256=self.binary_sha,
                rss_limit_bytes=1_000_000,
            )
            self.assertEqual(profile["status"], "qualified")
            pipeline_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "pipeline receipt changed"):
                verify_selection(
                    profile,
                    tier="72B",
                    rate="q3",
                    binary_sha256=self.binary_sha,
                )


if __name__ == "__main__":
    unittest.main()
