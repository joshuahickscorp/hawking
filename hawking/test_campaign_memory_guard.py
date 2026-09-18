"""The guard's swapfile axis was structurally dead and nothing here noticed.

_swapfiles() read only /private/var/vm, which is EMPTY on Darwin 27 while the
real store is /System/Volumes/VM. It returned 0 unconditionally, so SWAPFILE_WARN
40 and SWAPFILE_STOP 60 were unreachable -- in the guard written because a panic
hit exactly SWAPFILE_CAP 100. There was no test file at all.
"""
import sys, os, glob, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "future"))

import pytest
import campaign_memory_guard as g


def test_swapfile_store_is_actually_read():
    """The count must match what is really on disk, whichever store holds it."""
    on_disk = sum(len(glob.glob(d + "/swapfile*")) for d in g.SWAP_DIRS
                  if os.path.isdir(d))
    assert g._swapfiles() == on_disk
    if not any(os.path.isdir(d) for d in g.SWAP_DIRS):
        pytest.skip("no swap store on this host -- THIS ASSERTION DID NOT RUN")


def test_this_host_has_swapfiles_and_the_guard_sees_them():
    """A darwin host with an active swap store must not report zero."""
    if not glob.glob("/System/Volumes/VM/swapfile*"):
        pytest.skip("host has no swapfiles right now -- THIS ASSERTION DID NOT RUN")
    assert g._swapfiles() > 0, (
        "swapfiles exist on disk but the guard counts none: the axis is blind again")


def test_unreadable_store_is_not_encoded_as_zero(monkeypatch):
    """Absence of a reading must never look like a reading of zero."""
    monkeypatch.setattr(g, "SWAP_DIRS", ("/nonexistent/a", "/nonexistent/b"))
    assert g._swapfiles() == -1


def test_blind_guard_refuses_to_report_ok():
    state, why = g.classify(free_gb=64.0, compressor_gb=1.0, swapfiles=-1)
    assert state == "STOP"
    assert "BLIND" in " ".join(why)


def test_thresholds_are_reachable():
    """The whole point: 40 and 60 must actually be crossable."""
    assert g.classify(64.0, 1.0, g.SWAPFILE_WARN)[0] == "WARN"
    assert g.classify(64.0, 1.0, g.SWAPFILE_STOP)[0] == "STOP"
    assert g.classify(64.0, 1.0, 0)[0] == "OK"


def test_sample_reports_a_real_swapfile_count():
    s = g.sample()
    assert s.swapfiles == g._swapfiles()
    assert s.swapfiles >= 0, "sample() is blind to swap"


def test_resource_cost_reports_unknown_peak_rather_than_the_entry_reading():
    """A run shorter than one interval has an UNKNOWN peak, not a starting value."""
    import time
    with g.resource_cost(interval_s=5.0, label="short") as c:
        time.sleep(0.05)
    assert c["samples"] == 0
    assert c["resident_rss_gb_peak"] is None
    assert c["compressor_gb_peak"] is None
    assert c["free_gb_low"] is None
    assert "UNKNOWN" in c["peak_unknown_reason"]


def test_resource_cost_measures_a_real_run():
    import time
    with g.resource_cost(interval_s=0.3, label="real") as c:
        blocks = [bytearray(40 * 1024 * 1024) for _ in range(6)]
        time.sleep(1.2)
        del blocks
    assert c["samples"] > 0, "the sampler never ran"
    assert c["peak_unknown_reason"] is None
    assert c["wall_s"] >= 1.0
    assert c["free_gb_low"] is not None and c["free_gb_low"] <= c["free_gb_start"]
    assert isinstance(c["swapfiles_delta"], int)


def test_resource_cost_reports_swapfile_movement():
    """Swap growth during an experiment is the signal S010 cares about."""
    with g.resource_cost(interval_s=0.3) as c:
        pass
    assert c["swapfiles_start"] == g._swapfiles()
    assert c["swapfiles_delta"] == c["swapfiles_end"] - c["swapfiles_start"]


def test_the_guard_reads_AVAILABLE_not_merely_free():
    """"Pages free" is not available memory, and the gap blocked real experiments.

    Measured on this host mid-campaign: Pages free 0.44 GB while 17.34 GB sat
    inactive and 16.64 GB was file-backed page cache from streaming the lake --
    clean pages the kernel hands to the next allocation. The guard read the 0.44
    and refused the authoritative experiment three times.

    Same error class as reading vm.swapusage's boot high-water mark as live swap:
    a number whose NAME is not what it MEASURES.
    """
    s = g.sample()
    assert s.free_gb >= s.strictly_free_gb, (s.free_gb, s.strictly_free_gb)
    assert s.reclaimable_gb >= 0.0
    assert abs(s.free_gb - (s.strictly_free_gb + s.reclaimable_gb)) < 0.05, (
        "available must be strictly-free plus reclaimable, and both stay visible")


def test_dirty_anonymous_inactive_is_NOT_counted_as_available(monkeypatch):
    """The opposite mistake: claiming inactive pages that must be swapped first.

    Only the FILE-BACKED part of inactive is clean and evictable. With 1000
    inactive pages but only 100 file-backed, at most 100 may be counted.
    """
    INACTIVE, FILE_BACKED, FREE = 1_000_000, 100_000, 200_000   # pages, not toys
    monkeypatch.setattr(g, "_vm_stat", lambda: {
        "Pages free": FREE, "Pages speculative": 0, "Pages purgeable": 0,
        "Pages inactive": INACTIVE, "File-backed pages": FILE_BACKED,
        "Pages occupied by compressor": 0, "Pages wired down": 0,
    })
    s = g.sample()
    assert s.reclaimable_gb == round(FILE_BACKED * g.PAGE / g.GB, 2), s.reclaimable_gb
    assert s.strictly_free_gb == round(FREE * g.PAGE / g.GB, 2)
    # the whole point: 1M inactive pages exist, only 100k are claimable
    assert s.reclaimable_gb < round(INACTIVE * g.PAGE / g.GB, 2)


def test_a_machine_with_no_reclaimable_cache_is_unchanged(monkeypatch):
    """With nothing cached, available must equal strictly free -- no free lunch."""
    monkeypatch.setattr(g, "_vm_stat", lambda: {
        "Pages free": 500_000, "Pages speculative": 0, "Pages purgeable": 0,
        "Pages inactive": 0, "File-backed pages": 0,
        "Pages occupied by compressor": 0, "Pages wired down": 0,
    })
    s = g.sample()
    assert s.free_gb == s.strictly_free_gb
    assert s.reclaimable_gb == 0.0
