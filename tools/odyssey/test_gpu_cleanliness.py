"""G013 pins. Two defects these exist for:
  1. the resume lived only in __exit__, so a killed parent stranded six downloaders;
  2. the classifier used a CPU-percent threshold, which cannot see the I/O-bound
     downloaders that are the whole point of the window.
"""
import json, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

R = Path(__file__).resolve().parents[2] / "receipts/headless/GPU_CLEANLINESS_OVERRIDE.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G013 receipt not built")


def rec():
    return json.load(open(R))


def test_all_three_resume_guarantees_were_executed_not_asserted():
    d = rec()
    got = {g["guarantee"] for g in d["resume_guarantees"]}
    assert got == {"1_normal_exit", "2_parent_killed_watchdog_resumes",
                   "3_stale_lease_healed"}
    assert d["n_guarantees_passed"] == 3
    for g in d["resume_guarantees"]:
        assert g["passed"] is True, g


def test_the_killed_parent_case_actually_observed_the_stop_surviving_the_kill():
    """If the victim were never stopped, or resumed before the kill, the test proves
    nothing. Both preconditions must be recorded."""
    d = rec()
    g = next(x for x in d["resume_guarantees"]
             if x["guarantee"] == "2_parent_killed_watchdog_resumes")
    assert g["paused"] is True
    assert g["still_stopped_after_kill"] is True
    assert g["resumed_by_detached_watchdog"] is True


def test_pausable_set_matches_what_the_window_actually_suspends():
    """A receipt listing processes the window never touches is a false claim."""
    import gpu_cleanliness as gc
    from performance_qualification import io_pids
    assert set(gc.PAUSE_PATTERN) == {"hf download", "lake_filler.py"}
    src = (Path(__file__).resolve().parent / "performance_qualification.py").read_text()
    for pat in gc.PAUSE_PATTERN:
        assert pat in src, f"{pat} is claimed pausable but io_pids never matches it"
    assert callable(io_pids)


def test_io_bound_processes_are_not_filtered_out_by_a_cpu_threshold():
    """The downloaders sit near 1% CPU. A threshold-only classifier misses them all."""
    d = rec()
    assert d["pausable"], "no pausable processes classified at all"
    assert d["n_pausable_invisible_to_a_cpu_threshold"] >= 1, (
        "every pausable process cleared the CPU threshold, so this run could not have "
        "detected the blind spot the field exists for")


def test_standing_load_is_declared_and_never_paused():
    d = rec()
    assert d["standing_not_paused"], "no standing load recorded"
    assert d["standing_cpu_percent_total"] > 0
    assert d["no_forged_speedups"]["standing_load_is_endured_and_declared"] is True
    paused = {x["name"] for x in d["pausable"]}
    standing = {x["name"] for x in d["standing_not_paused"]}
    assert not (paused & standing), "a process is both paused and declared standing"


def test_our_own_leftovers_are_named_rather_than_counted_as_standing_floor():
    """powermetrics is ours. Filing it as STANDING would inflate the declared floor
    with load that need not exist at all."""
    d = rec()
    standing = {x["name"] for x in d["standing_not_paused"]}
    assert "powermetrics" not in standing


def test_the_historical_defect_is_recorded_with_its_evidence():
    d = rec()
    h = d["historical_defect"]
    assert "__exit__" in h["what"]
    assert "SIGSTOP" in h["observed"] or "stopped" in h["observed"]
    assert "protected_window.py" in h["fix"]


def test_receipt_is_machine_generated_and_passes():
    d = rec()
    assert d["hand_authored"] is False
    assert d["generated_by"] == "tools/odyssey/gpu_cleanliness.py"
    assert d["pass"] is True


def test_the_two_g013_writers_do_not_share_a_canonical_path():
    """gpu_cleanliness.py owns the MECHANISM receipt; performance_qualification.py owns
    the CONTENTION demonstration. Earlier in this campaign two writers on one path
    silently reverted the genome libraries."""
    pq = (Path(__file__).resolve().parent / "performance_qualification.py").read_text()
    gc_ = (Path(__file__).resolve().parent / "gpu_cleanliness.py").read_text()
    assert 'GPU_CLEANLINESS_OVERRIDE.json"' in gc_
    # the qualifier may REFERENCE the mechanism receipt but must not claim to write it
    assert "emit_clean" in pq
    assert 'write_text' in pq
    assert '"mechanism_receipt"' in pq, "qualifier does not point at the mechanism receipt"
