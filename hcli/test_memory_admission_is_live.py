"""The memory admission gate must read live pressure, not a boot high-water mark.

Regression for the latch that held the sovereign resident at cycles=0 for 23
minutes on 2026-09-01 with 42.6 GB free RAM and 73% system-wide memory free:
``vm.swapusage used=`` reported 12.3 GB because the swapfile had grown once and
macOS never shrinks it, so ``swap > 2 GiB`` refused every admission for the rest
of the boot.
"""
from __future__ import annotations

import hcli.machine as machine
from hcli.agentos.resident import memory_decision


def test_flat_swapouts_report_no_pressure():
    """A grown swapfile with no paging activity is not pressure."""
    machine._LAST_SWAPOUTS = None
    counts = {"Swapouts": 135_341_729}
    assert machine.swap_pressure_bytes(counts) == 0  # no baseline yet
    assert machine.swap_pressure_bytes(counts) == 0  # flat -> still zero


def test_active_swapouts_report_pressure():
    machine._LAST_SWAPOUTS = None
    machine.swap_pressure_bytes({"Swapouts": 1_000})
    paged = machine.swap_pressure_bytes({"Swapouts": 1_000 + 65_536})
    assert paged == 65_536 * machine._get_page_size()


def test_counter_reset_does_not_underflow():
    machine._LAST_SWAPOUTS = None
    machine.swap_pressure_bytes({"Swapouts": 5_000})
    assert machine.swap_pressure_bytes({"Swapouts": 10}) == 0


def test_pressure_label_reads_the_free_percentage():
    """The real ``memory_pressure -Q`` output carries no low/normal/high word."""
    live = (
        "The system has 103079215104 (6291456 pages with a page size of 16384).\n"
        "System-wide memory free percentage: 73%"
    )
    assert machine._pressure_label(live) == "normal"
    assert machine._pressure_label(live.replace("73%", "22%")) == "warn"
    assert machine._pressure_label(live.replace("73%", "9%")) == "high"
    assert machine._pressure_label(None) == "unknown"


# Pin the ceiling rather than inherit HCLI_SWAP_CEILING_GIB from whatever ran
# first: these two cases are about which NUMBER admission reads, so they must not
# also depend on ambient environment.
CEILING = 2 * 1024**3


def test_idle_host_with_a_grown_swapfile_admits():
    """The exact live snapshot that refused admission must now admit."""
    decision = memory_decision(
        {
            "total_bytes": 103_079_215_104,
            "free_bytes": 42_600_464_384,
            "swap_used_bytes": 0,  # flat swapouts, not the 12.9 GB high-water
            "pressure": "normal",
        },
        swap_ceiling_bytes=CEILING,
    )
    assert decision["safe"] is True, decision["reasons"]


def test_a_genuinely_thrashing_host_still_refuses():
    decision = memory_decision(
        {
            "total_bytes": 103_079_215_104,
            "free_bytes": 42_600_464_384,
            "swap_used_bytes": 3 * 1024**3,
            "pressure": "normal",
        },
        swap_ceiling_bytes=CEILING,
    )
    assert decision["safe"] is False
    assert "swap" in decision["reasons"][0]


def test_snapshot_keeps_the_highwater_under_an_honest_name():
    snap = machine.host_snapshot()
    assert "swap_highwater_bytes" in snap
    assert snap["swap_used_bytes"] >= 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all green")
