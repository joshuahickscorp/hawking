#!/usr/bin/env python3.12
"""Tests for the light-only governor's decision logic.

Both cases here are real defects that fired during the campaign, not hypotheticals.

    python3.12 -m tools.campaign.test_light_governor
"""
from __future__ import annotations

import unittest

from tools.campaign.light_governor import decide, envelope


def s(**kw) -> dict:
    base = {
        "at": "2026-01-01T00:00:00Z", "ncpu": 28, "load1": 0.5, "load5": 0.5, "load15": 0.5,
        "load_per_core": 0.018, "mem_total_gib": 96.0, "mem_available_gib": 60.0,
        "mem_available_frac": 0.62, "swap_used_mb": 100.0, "disk_free_gib": 190.0,
        "thermal_green": True, "foreign_cpu_pct": 0.0, "foreign_cpu_cores": 0.0,
        "foreign_procs": 0,
    }
    base.update(kw)
    return base


class TestHeavyWindowVeto(unittest.TestCase):
    """`ps -eo pcpu` averages CPU over a process's LIFETIME, so a freshly restarted MOP
    worker reports near-zero while pinning a core. This actually happened: the governor
    reported 'foreign 0.94 cores' against a load average of 20.7, which without the veto
    would have declared a heavy window in the middle of MOP restarting."""

    def test_low_pcpu_with_high_load_is_not_a_heavy_window(self) -> None:
        samples = [s(foreign_cpu_cores=0.0, foreign_procs=0, load1=20.7, load_per_core=0.74)
                   for _ in range(3)]
        d = decide(samples)
        self.assertEqual(d["mode"], "LIGHT_ONLY")
        self.assertIn("load average", d["why"][0])

    def test_genuinely_idle_machine_marks_the_window(self) -> None:
        samples = [s(foreign_cpu_cores=0.0, foreign_procs=0, load1=0.4, load_per_core=0.014)
                   for _ in range(3)]
        d = decide(samples)
        self.assertEqual(d["mode"], "HEAVY_WINDOW_AVAILABLE")
        self.assertIn("MARKED ONLY", d["law"])

    def test_heavy_window_is_never_an_instruction_to_launch(self) -> None:
        samples = [s(load1=0.2, load_per_core=0.007) for _ in range(3)]
        self.assertIn("a human decides", decide(samples)["law"])


class TestSwapCalibration(unittest.TestCase):
    """The first swap guard was an absolute 512 MB. macOS keeps swap allocated after a
    peak, so the machine sat at 13 GB with 56 GiB available and BACKOFF fired on every
    single sample. A backoff that always fires trains its reader to ignore it."""

    def test_high_but_stable_swap_is_not_backoff(self) -> None:
        samples = [s(swap_used_mb=13_000.0, load1=20.0, load_per_core=0.71,
                     foreign_cpu_cores=15.0, foreign_procs=16) for _ in range(3)]
        self.assertEqual(decide(samples)["mode"], "LIGHT_ONLY")

    def test_growing_swap_is_backoff(self) -> None:
        samples = [
            s(swap_used_mb=1_000.0, foreign_cpu_cores=15.0),
            s(swap_used_mb=2_500.0, foreign_cpu_cores=15.0),
            s(swap_used_mb=5_000.0, foreign_cpu_cores=15.0),
        ]
        d = decide(samples)
        self.assertEqual(d["mode"], "BACKOFF")
        self.assertTrue(any("GREW" in w for w in d["why"]))

    def test_absolute_ceiling_still_exists(self) -> None:
        samples = [s(swap_used_mb=30_000.0, foreign_cpu_cores=15.0) for _ in range(3)]
        self.assertEqual(decide(samples)["mode"], "BACKOFF")


class TestBackoffTriggers(unittest.TestCase):
    def test_memory_below_guard(self) -> None:
        samples = [s(mem_available_frac=0.05, foreign_cpu_cores=15.0) for _ in range(3)]
        self.assertEqual(decide(samples)["mode"], "BACKOFF")

    def test_thermal(self) -> None:
        samples = [s(thermal_green=False, foreign_cpu_cores=15.0) for _ in range(3)]
        self.assertEqual(decide(samples)["mode"], "BACKOFF")

    def test_disk_reserve(self) -> None:
        samples = [s(disk_free_gib=10.0, foreign_cpu_cores=15.0) for _ in range(3)]
        self.assertEqual(decide(samples)["mode"], "BACKOFF")

    def test_high_load_alone_is_deliberately_not_backoff(self) -> None:
        """MOP's normal state is load/core ~1.66. Backing off on load alone is paralysis,
        not caution -- and it is not a condition a light job can make meaningfully worse."""
        samples = [s(load1=46.0, load_per_core=1.66, foreign_cpu_cores=21.0, foreign_procs=24)
                   for _ in range(3)]
        self.assertEqual(decide(samples)["mode"], "LIGHT_ONLY")
        self.assertNotIn("load_per_core_backoff", envelope(samples[0]))


class TestRelief(unittest.TestCase):
    def test_relief_needs_sustained_green(self) -> None:
        samples = [s(load_per_core=0.5, foreign_cpu_cores=5.0, load1=14.0) for _ in range(3)]
        self.assertEqual(decide(samples)["mode"], "RELIEF_WINDOW")

    def test_one_green_sample_is_not_enough(self) -> None:
        samples = [s(load_per_core=0.5, foreign_cpu_cores=5.0, load1=14.0)]
        self.assertEqual(decide(samples)["mode"], "LIGHT_ONLY")

    def test_relief_demands_declarations(self) -> None:
        samples = [s(load_per_core=0.5, foreign_cpu_cores=5.0, load1=14.0) for _ in range(3)]
        self.assertIn("BEFORE launch", decide(samples)["law"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
