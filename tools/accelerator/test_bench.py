"""Quiescence: the check that was a shell habit rather than an instrument.

Every "the machine is quiet" claim in this program came from an ad-hoc
`pgrep modellake` in a shell command whose silence was then written into a
receipt as a fact. It was never a recorded field and it matched NAMES, so it
could only ever find the one contender this program had already met.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench  # noqa: E402

# --- quiescence: the check that was a shell habit -----------------------------

def test_the_enumerator_actually_enumerates():
    """ANTI-VACUITY. Every other test here is worthless if the enumerator can
    only ever return an empty list. At threshold 0 every process qualifies, so a
    machine with any process at all must report contenders."""
    q = bench.machine_quiescence(cpu_pct=0.0, rss_gib=0.0)
    assert q["quiet"] is False
    assert q["n_contenders"] > 0, "ps returned no process at threshold zero"
    assert q["method"] == "enumerate"


def test_a_quiet_verdict_is_reachable():
    """The other direction. A check that can only ever say BUSY is as useless as
    one that can only ever say QUIET, and an unreachable verdict would make every
    non-quiet result meaningless."""
    q = bench.machine_quiescence(cpu_pct=1e9, rss_gib=1e9)
    assert q["quiet"] is True and q["contenders"] == []


def test_the_name_filter_and_the_enumerator_DISAGREE():
    """THE DEFECT, EXECUTABLE. TOKEN_GRAPH_REDUCTION_TIMED recorded 'no lake fill
    running (pgrep modellake = 0)' and that was read as MACHINE IS QUIET. Here the
    two checks run against the SAME machine at the SAME moment and reach opposite
    verdicts, because one asks what is there and the other asks whether one named
    thing is there."""
    enumerated = bench.machine_quiescence(cpu_pct=0.0, rss_gib=0.0)
    by_name = bench.name_filter_quiescence(("a-process-name-that-cannot-exist-xyzzy",))
    assert enumerated["quiet"] is False
    assert by_name["quiet"] is True
    assert enumerated["quiet"] != by_name["quiet"]


def test_machine_quiescence_CANNOT_BE_GIVEN_A_NAME_LIST():
    """The blind spot was a FILTER, so the fix must not ship a filter. If someone
    later adds a names= parameter this test fails, because an instrument whose
    blind spot is configurable is one somebody will configure blind."""
    import inspect
    params = set(inspect.signature(bench.machine_quiescence).parameters)
    assert params == {"cpu_pct", "rss_gib"}, params


def test_a_FAILED_enumeration_is_not_a_quiet_machine(monkeypatch):
    """0 of 0 reads identically to 0 of many. If ps cannot run, quiet is None --
    never True -- because 'I could not look' and 'I looked and found nothing' are
    different facts and this program has already confused them four times."""
    class R:
        returncode, stdout, stderr = 1, "", "boom"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    q = bench.machine_quiescence()
    assert q["quiet"] is None and "refused" in q


def test_the_caller_is_not_exempt_from_its_own_check():
    """The name filter exempted everything it did not name, including the harness.
    Enumeration must report this very process if it crosses a threshold."""
    import os
    q = bench.machine_quiescence(cpu_pct=0.0, rss_gib=0.0)
    assert os.getpid() in [c["pid"] for c in q["contenders"]]


# --------------------------------------------------------------------------
# S032 §16: latency is first class. A median alone hides the tail a caller waits
# on -- and a tail quoted without its sample count is worse than no tail at all.
# --------------------------------------------------------------------------

def test_time_arm_reports_a_tail_not_only_a_median():
    r = bench.time_arm(lambda: sum(range(200)), reps=40, warmup=2)
    assert r["p50_s"] == r["median_s"]
    assert r["p95_s"] >= r["p50_s"] and r["p99_s"] >= r["p95_s"]


def test_the_tail_is_ORDERED_and_inside_the_sample():
    r = bench.time_arm(lambda: sum(range(200)), reps=40, warmup=2)
    assert r["q1_s"] <= r["p50_s"] <= r["q3_s"] <= r["p99_s"]


def test_A_TWO_SAMPLE_TAIL_SAYS_SO():
    """At 40 reps only three samples sit at or above p95. Quoting that as a
    percentile without saying how many samples stand behind it is the shape of a
    fabricated tail."""
    r = bench.time_arm(lambda: sum(range(50)), reps=40, warmup=2)
    assert r["samples_at_or_above_p95"] < bench.TAIL_SAMPLES_FOR_A_STABLE_P95
    assert r["tail_resolution"] and "ORDER STATISTIC" in r["tail_resolution"]


def test_A_LARGE_SAMPLE_DROPS_THE_CAVEAT():
    r = bench.time_arm(lambda: sum(range(50)), reps=200, warmup=2)
    assert r["samples_at_or_above_p95"] >= bench.TAIL_SAMPLES_FOR_A_STABLE_P95
    assert r["tail_resolution"] is None


def test_the_tail_ratio_is_reported_so_spread_is_visible_without_arithmetic():
    r = bench.time_arm(lambda: sum(range(200)), reps=40, warmup=2)
    assert r["p95_over_p50"] is None or r["p95_over_p50"] >= 1.0
