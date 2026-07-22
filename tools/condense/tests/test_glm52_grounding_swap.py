"""Darwin swap parsing: rounded display values must not fail, and must fail safe.

``sysctl -n vm.swapusage`` prints two-decimal display values, so the exact byte count is
unrecoverable.  Requiring an integral byte count rejected almost every real reading -- the
sampler failed as a function of live swap rather than of anything being wrong -- while
guessing a midpoint would let admission be granted on a number the machine never proved.
These tests pin the resolution: consumption rounds up, headroom rounds down.
"""
from __future__ import annotations

import pytest

from tools.condense import glm52_grounding as grounding

HW_MEMSIZE = "103079215104"
VM_STAT = (
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    "Pages free:                                  100.\n"
    "Pages inactive:                               50.\n"
    "Pages speculative:                            10.\n"
)


def sample(swapusage: str):
    return grounding.parse_darwin_memory(
        hw_memsize=HW_MEMSIZE, vm_stat=VM_STAT, swapusage=swapusage)


@pytest.mark.parametrize(
    "swapusage,used,total",
    [
        ("total = 0.00M  used = 0.00M  free = 0.00M", 0, 0),
        # 220.56M is 231,273,922.56 bytes: the reading that used to raise
        ("total = 1024.00M  used = 220.56M  free = 803.44M", 231_273_923, 1_073_741_824),
        # 317.75M lands on a 1/256 MB boundary and is already exact
        ("total = 2048.00M  used = 317.75M  free = 1730.25M", 333_185_024, 2_147_483_648),
        ("total = 4.00G  used = 1.23G  free = 2.77G", 1_320_702_444, 4_294_967_296),
    ],
)
def test_rounded_display_values_parse(swapusage: str, used: int, total: int) -> None:
    got = sample(swapusage)
    assert got.used_swap_bytes == used
    assert got.total_swap_bytes == total


def test_used_rounds_up_and_total_rounds_down() -> None:
    """Ambiguity is resolved against us, never in our favour."""
    got = sample("total = 1024.56M  used = 220.56M  free = 803.44M")
    assert got.used_swap_bytes == 231_273_923      # ceil of ...922.56
    assert got.total_swap_bytes == 1_074_329_026   # floor of ...026.56


def test_full_swap_does_not_invert_the_invariant() -> None:
    """Opposite rounding at the cap must clamp, not raise."""
    got = sample("total = 1024.00M  used = 1024.00M  free = 0.00M")
    assert got.used_swap_bytes == got.total_swap_bytes


@pytest.mark.parametrize("swapusage", [
    "garbage",
    "total = 1024.00M",                                  # missing used
    "used = 1.00M  free = 1.00M",                        # missing total
    "total = 1024.00X  used = 1.00X  free = 1.00X",      # unknown unit
    "",
])
def test_malformed_or_missing_is_refused(swapusage: str) -> None:
    with pytest.raises(grounding.GroundingError):
        sample(swapusage)


def test_live_swap_parses_whatever_the_machine_currently_reports() -> None:
    """The regression this fixes only appeared against real live values."""
    import subprocess

    raw = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True, check=True).stdout
    got = sample(raw)
    assert got.used_swap_bytes >= 0
    assert got.used_swap_bytes <= got.total_swap_bytes
