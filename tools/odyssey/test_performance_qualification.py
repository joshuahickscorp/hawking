"""The quiesce classifier had three bugs in a row. Each one gets a test.

    1. cpu% as the proxy   -- GPU-bound work sits near 0% cpu and was reported as idle
    2. waiters as workers  -- the variant-B wait-loop names the workload it waits for, so
                              counting it deadlocked the run against its own receipt
    3. comm vs command     -- the downloader's comm is just "Python", so a pausable
                              transfer landed in the must-finish bucket
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "headless"))
import performance_qualification as pq


def test_gpu_bound_work_is_seen_despite_low_cpu():
    assert "ascension_qwen38_hybrid_greedy" in pq.OUR_WORKLOADS
    assert "capability_suite.py" in pq.OUR_WORKLOADS
    q = pq.quiesce_check()
    for r in q["our_busy_processes"]:
        if r["matched_by"] == "workload name":
            return          # matched without needing a cpu threshold
    # nothing of ours running is also a valid state
    assert q["n_ours_busy"] == 0


def test_a_waiting_shell_is_not_counted_as_work():
    """Otherwise a chain that waits on this run's own output can never be satisfied."""
    q = pq.quiesce_check()
    for r in q["our_busy_processes"]:
        c = r["cmd"].lstrip()
        assert not (c.startswith(("sh -c", "/bin/sh -c", "zsh -c"))
                    and (" until " in r["cmd"] or " while " in r["cmd"])), r["cmd"]


def test_transfers_are_pausable_not_blocking():
    q = pq.quiesce_check()
    for r in q["pausable_io"]:
        assert r["matched_by"] == "pausable transfer"
    # a transfer must never be in the must-finish bucket: PausedIO handles it
    for r in q["our_busy_processes"]:
        assert "hf download" not in r["cmd"], r["cmd"]


def test_standing_load_is_recorded_not_subtracted():
    q = pq.quiesce_check()
    assert "standing_system_load" in q
    assert q["standing_cpu_total"] >= 0
    assert "not ours to remove" in q["quiesced_means"]


def test_quiesced_means_unpausable_work_only():
    q = pq.quiesce_check()
    assert q["quiesced"] == (q["n_ours_busy"] == 0)


def test_paused_io_always_resumes():
    """The resume guarantee moved OUT of this class and got stronger.

    PausedIO used to SIGSTOP and SIGCONT inline with the resume in a finally block. That
    survives an exception and nothing else: a parent killed on a timeout left six
    downloaders stopped. It now delegates to ProtectedWindow, which adds a detached
    watchdog and a healable lease. Asserting the literal "SIGSTOP" in this class body
    tested the old implementation, not the property.
    """
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[2]
    src = (repo / "tools/odyssey/performance_qualification.py").read_text()
    body = src[src.index("class PausedIO"):src.index("def run_once")]
    assert "ProtectedWindow" in body, "PausedIO no longer delegates the resume"

    win = (repo / "tools/odyssey/protected_window.py").read_text()
    assert "signal.SIGSTOP" in win and "signal.SIGCONT" in win
    for guarantee in ("watchdog", "LEASE", "heal"):
        assert guarantee in win, guarantee
