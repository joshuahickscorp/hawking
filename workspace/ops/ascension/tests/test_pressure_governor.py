"""Unit tests for the pressure governor state machine and signal parsers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_OPS = Path(__file__).resolve().parents[1]
_ROOT = _OPS.parents[2]
for p in (str(_OPS.parent), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ascension.pressure_governor import (  # noqa: E402
    GovernorThresholds,
    PressureGovernor,
    PressureLevel,
    evaluate_pressure,
)
from ascension.signals import (  # noqa: E402
    HostSignals,
    collect_host_signals,
    parse_gpu_util_ioreg,
    parse_swapusage,
    parse_thermal_pmset,
    parse_vm_stat,
)


def _sig(
    *,
    free_disk_gib: float = 100.0,
    available_ram_gib: float = 30.0,
    total_ram_gib: float = 36.0,
    swap_used_gib: float = 0.0,
    swap_total_gib: float = 0.0,
    gpu_util_pct: int | None = 10,
    thermal_throttling: bool | None = False,
    foreground_user_active: bool | None = False,
) -> HostSignals:
    g = 1024**3
    return HostSignals(
        free_disk_bytes=int(free_disk_gib * g),
        total_disk_bytes=int(500 * g),
        available_ram_bytes=int(available_ram_gib * g),
        total_ram_bytes=int(total_ram_gib * g),
        swap_used_bytes=int(swap_used_gib * g),
        swap_total_bytes=int(swap_total_gib * g),
        gpu_util_pct=gpu_util_pct,
        thermal_throttling=thermal_throttling,
        foreground_user_active=foreground_user_active,
        source="test",
    )


class TestSignalParsers(unittest.TestCase):
    def test_parse_vm_stat(self):
        text = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                               1000.\n"
            "Pages active:                             2000.\n"
            "Pages inactive:                           3000.\n"
            "Pages speculative:                         500.\n"
            "Pages purgeable:                           100.\n"
        )
        avail, total = parse_vm_stat(text, total_ram_bytes=64 * 1024**3)
        # (1000+3000+500+100)*16384
        self.assertEqual(avail, 4600 * 16384)
        self.assertEqual(total, 64 * 1024**3)

    def test_parse_swapusage(self):
        used, total = parse_swapusage("total = 2.00G  used = 512.0M  free = 1.50G")
        self.assertEqual(total, int(2.0 * 1024**3))
        self.assertEqual(used, int(512.0 * 1024**2))

    def test_parse_gpu_ioreg(self):
        text = '  "Device Utilization %"=42\n  "Device Utilization %"=88\n'
        self.assertEqual(parse_gpu_util_ioreg(text), 88)

    def test_parse_thermal_speed_limit(self):
        self.assertTrue(parse_thermal_pmset("CPU_Speed_Limit = 75"))
        self.assertFalse(parse_thermal_pmset("CPU_Speed_Limit = 100"))

    def test_collect_injected(self):
        g = 1024**3
        sig = collect_host_signals(
            disk_probe=lambda _p: (50 * g, 500 * g),
            runners={
                "hw_memsize": lambda: str(36 * g),
                "vm_stat": lambda: (
                    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                    "Pages free: 100000.\n"
                    "Pages inactive: 200000.\n"
                    "Pages speculative: 10000.\n"
                    "Pages purgeable: 0.\n"
                ),
                "swapusage": lambda: "total = 0.00G  used = 0.00G  free = 0.00G",
                "ioreg_gpu": lambda: '"Device Utilization %"=12',
                "pmset_therm": lambda: "CPU_Speed_Limit = 100",
            },
            foreground_probe=lambda: False,
        )
        self.assertEqual(sig.free_disk_bytes, 50 * g)
        self.assertEqual(sig.gpu_util_pct, 12)
        self.assertFalse(sig.thermal_throttling)
        self.assertEqual(sig.source, "injected")


class TestEvaluatePressure(unittest.TestCase):
    def test_green(self):
        sample = evaluate_pressure(_sig())
        self.assertEqual(sample.level, PressureLevel.GREEN)
        self.assertIn("full_campaign", sample.actions)

    def test_disk_yellow(self):
        sample = evaluate_pressure(_sig(free_disk_gib=35))
        self.assertEqual(sample.level, PressureLevel.YELLOW)
        self.assertIn("pause_new_large_downloads", sample.actions)

    def test_disk_red(self):
        sample = evaluate_pressure(_sig(free_disk_gib=20))
        self.assertEqual(sample.level, PressureLevel.RED)
        self.assertIn("evict_leased_cache", sample.actions)

    def test_disk_critical(self):
        sample = evaluate_pressure(_sig(free_disk_gib=5))
        self.assertEqual(sample.level, PressureLevel.CRITICAL)
        self.assertIn("unload_target_model", sample.actions)
        self.assertIn("return_resources_to_user", sample.actions)

    def test_ram_critical(self):
        sample = evaluate_pressure(_sig(available_ram_gib=1.0, total_ram_gib=36.0))
        self.assertEqual(sample.level, PressureLevel.CRITICAL)

    def test_swap_red(self):
        sample = evaluate_pressure(_sig(swap_used_gib=10.0, swap_total_gib=20.0))
        self.assertEqual(sample.level, PressureLevel.RED)

    def test_thermal_red(self):
        sample = evaluate_pressure(_sig(thermal_throttling=True))
        self.assertEqual(sample.level, PressureLevel.RED)

    def test_gpu_yellow(self):
        sample = evaluate_pressure(_sig(gpu_util_pct=90))
        self.assertEqual(sample.level, PressureLevel.YELLOW)

    def test_foreground_yellow(self):
        sample = evaluate_pressure(_sig(foreground_user_active=True))
        self.assertEqual(sample.level, PressureLevel.YELLOW)

    def test_worst_of_wins(self):
        # yellow disk + critical ram → CRITICAL
        sample = evaluate_pressure(
            _sig(free_disk_gib=35, available_ram_gib=1.0, total_ram_gib=36.0)
        )
        self.assertEqual(sample.level, PressureLevel.CRITICAL)


class TestGovernorHysteresis(unittest.TestCase):
    def test_escalate_immediate(self):
        gov = PressureGovernor(thresholds=GovernorThresholds(deescalate_dwell=2))
        a1 = gov.step(_sig())
        self.assertEqual(a1.level, PressureLevel.GREEN)
        a2 = gov.step(_sig(free_disk_gib=5))
        self.assertEqual(a2.level, PressureLevel.CRITICAL)
        self.assertTrue(a2.changed)

    def test_deescalate_requires_dwell(self):
        gov = PressureGovernor(thresholds=GovernorThresholds(deescalate_dwell=2))
        gov.step(_sig(free_disk_gib=5))  # CRITICAL
        self.assertEqual(gov.level, PressureLevel.CRITICAL)
        a1 = gov.step(_sig())  # still CRITICAL (dwell 1)
        self.assertEqual(a1.level, PressureLevel.CRITICAL)
        self.assertFalse(a1.changed)
        a2 = gov.step(_sig())  # dwell 2 → GREEN
        self.assertEqual(a2.level, PressureLevel.GREEN)
        self.assertTrue(a2.changed)

    def test_history_records_levels(self):
        gov = PressureGovernor()
        gov.step(_sig())
        gov.step(_sig(free_disk_gib=30))
        self.assertEqual(len(gov.history), 2)
        self.assertEqual(gov.history[-1], PressureLevel.YELLOW)


if __name__ == "__main__":
    unittest.main()
