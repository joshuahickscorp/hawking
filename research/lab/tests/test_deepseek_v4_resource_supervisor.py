from __future__ import annotations

import unittest
from pathlib import Path

from lab.operators.deepseek_v4_resource_supervisor import (
    RSS_BUDGET_BYTES,
    ResourceSample,
    SupervisorConfig,
    SupervisorPolicy,
    evaluate_sample,
    plan_dry_run,
)


class DeepSeekV4ResourceSupervisorTests(unittest.TestCase):
    def _config(self) -> SupervisorConfig:
        root = Path(__file__).resolve().parents[2]
        return SupervisorConfig(
            artifact_dir=root / "workspace/campaign/records/runs/deepseek-v4/test.gravity",
            workspace_root=root,
            xet_root=root / "workspace/campaign/records/runs/deepseek-v4/test-xet",
            protected_floor_bytes=25 * 1024**3,
            worker_ramp=(4, 8, 12, 16),
            dry_run=True,
        )

    def test_dry_run_emits_a_bounded_resumable_ramp(self) -> None:
        receipt = plan_dry_run(self._config())
        self.assertEqual(receipt["status"], "DRY_RUN")
        self.assertEqual(receipt["recommended_workers"], 4)
        self.assertEqual([row["workers"] for row in receipt["candidates"]], [4, 8, 12, 16])
        self.assertEqual(receipt["policy"]["rss_budget_bytes"], RSS_BUDGET_BYTES)
        self.assertFalse(receipt["target"]["downloads_model_objects_directly"])

    def test_budget_evaluation_fails_closed_on_rss_disk_or_swap(self) -> None:
        policy = SupervisorPolicy(
            rss_budget_bytes=RSS_BUDGET_BYTES,
            protected_floor_bytes=25 * 1024**3,
        )
        sample = ResourceSample(
            monotonic_s=1.0,
            process_rss_bytes=RSS_BUDGET_BYTES + 1,
            cpu_percent=50.0,
            free_disk_bytes=25 * 1024**3 - 1,
            swap_used_bytes=2,
            transfer_bytes=0,
            pid=123,
        )
        self.assertEqual(
            evaluate_sample(sample, baseline_swap_used_bytes=1, policy=policy),
            ["RSS_BUDGET_BREACH", "DISK_FLOOR_BREACH", "SWAP_GROWTH"],
        )


if __name__ == "__main__":
    unittest.main()
