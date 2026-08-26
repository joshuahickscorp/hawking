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
