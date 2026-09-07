"""The campaign swap ceiling is 30 GB, and the guard measured swap by FILE COUNT.

S008 3 authorizes swap up to 30 GB and makes 30 GB the hard campaign ceiling, with bands
at 24 and 27. S008 4 is explicit that the existing guard stays the SINGLE RESOURCE
AUTHORITY -- update its ceiling, do not build a second governor.

The guard had no byte-denominated swap axis at all. It counted SWAPFILES (warn 40, stop
60, cap 100), which is a real signal and stays, but it cannot answer "are we under 30 GB"
because macOS swapfiles are not a fixed size.

The obvious source is `sysctl vm.swapusage used`, and it is the wrong one. That field is a
BOOT HIGH-WATER MARK, not a live reading -- a scar this campaign already paid for: comparing
it to a ceiling latched the gate off permanently and held the resident at cycles=0 with 42.6
GB free and swapouts flat. Summing the actual swapfile sizes under the real Darwin swap store
is live, auditable, and cannot latch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "future"))
import campaign_memory_guard as g  # noqa: E402


def test_the_ceiling_is_thirty_and_the_bands_are_named():
    assert g.SWAP_GB_CEILING == 30.0
    assert g.SWAP_GB_WARN == 24.0
    assert g.SWAP_GB_PROTECT == 27.0
    assert g.SWAP_GB_WARN < g.SWAP_GB_PROTECT < g.SWAP_GB_CEILING


def test_each_band_actually_changes_the_verdict():
    """A band nobody can reach is decoration."""
    assert g.classify_swap_gb(1.0) == "OK"
    assert g.classify_swap_gb(23.9) == "OK"
    assert g.classify_swap_gb(24.0) == "WARN"
    assert g.classify_swap_gb(26.9) == "WARN"
    assert g.classify_swap_gb(27.0) == "WARN"
    assert g.classify_swap_gb(30.0) == "STOP"
    assert g.classify_swap_gb(45.0) == "STOP"


def test_the_ceiling_reaches_the_real_verdict():
    """Not a constant nothing consults: classify() must escalate on swap GB alone."""
    st, why = g.classify(60.0, 1.0, 3, swap_gb=31.0)
    assert st == "STOP", (st, why)
    assert any("30" in w or "swap" in w.lower() for w in why), why
    st_ok, _ = g.classify(60.0, 1.0, 3, swap_gb=1.0)
    assert st_ok == "OK"


def test_swap_gb_is_NOT_read_from_the_boot_high_water_mark():
    """vm.swapusage `used` is a boot high-water mark; comparing it to a ceiling latches."""
    src = (Path(__file__).resolve().parents[2] / "tools" / "future"
           / "campaign_memory_guard.py").read_text()
    body = src.split("def _swap_gb")[1].split("\ndef ")[0]
    # Skip the docstring: it NAMES vm.swapusage while explaining why not to use it, and
    # matching that made this assertion fail on its own explanation. Second time a source
    # test in this repo has matched prose instead of code.
    code = body.split('"""')[2]
    assert "swapusage" not in code, (
        "swap GB is being read from the boot high-water mark, which cannot go down")
    assert "SWAP_DIRS" in code and "swapfile" in code, code


def test_the_snapshot_carries_the_MEASURED_number_not_the_default():
    """swap_gb defaults to 0.0, so `isinstance(float)` passes on an unwired field.

    Removing the assignment in sample() left an earlier version of this test green, which
    is the defaulted-field trap: the snapshot reported a plausible zero forever.
    """
    snap = g.sample()
    assert "swap_gb" in snap.as_dict()
    assert snap.swap_gb == pytest.approx(g._swap_gb(), abs=0.01), (
        f"snapshot says {snap.swap_gb} but the measurement says {g._swap_gb()}")
    # That comparison is VACUOUS on a host with no swap -- 0.0 == 0.0 whether the field is
    # wired or defaulted, and removing the assignment left it green. Inject a value the
    # default cannot produce.
    real = g._swap_gb
    g._swap_gb = lambda: 7.5
    try:
        injected = g.sample()
    finally:
        g._swap_gb = real
    assert injected.swap_gb == pytest.approx(7.5), (
        f"sample() does not carry the measurement: reported {injected.swap_gb}")


def test_the_measurement_is_not_hardcoded_zero():
    """Point it at a directory holding a known-size swapfile and it must see the bytes."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "swapfile0").write_bytes(b"\0" * (4 << 20))
        (Path(d) / "not_a_swapfile").write_bytes(b"\0" * (8 << 20))
        real = g.SWAP_DIRS
        g.SWAP_DIRS = (d,)
        try:
            seen = g._swap_gb()
        finally:
            g.SWAP_DIRS = real
    assert seen == pytest.approx((4 << 20) / (1 << 30), rel=1e-6), seen
