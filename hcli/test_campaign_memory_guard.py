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
