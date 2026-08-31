"""A blocking invoke must not be the only thing the loop is doing.

no_wait_orchestration counts a forcing interval when a workunit_launched is
followed by its result_ingested with no other launch in between. The 30m run had
twelve - 1 s, 3 s, 11 s - each naming the safe work that was runnable while the
loop sat on _invoke_bounded.

G037's repair one guarded the wrong door: emit_idle_justified refuses an IDLE
EVENT taken while work is runnable, and a blocking subprocess wait never passes
through that function. The guard watched a door the driver does not use.
"""
from __future__ import annotations

import re

from tools.future import autonomy_run as ar


def _src() -> str:
    return open(ar.__file__, encoding="utf-8").read()


def test_the_top_up_runs_before_the_blocking_invoke():
    src = _src()
    top = src.index("_top_up_detached(doc, queue)")
    invoke = src.index("res = _invoke_bounded(cap, unit_budget)")
    assert top < invoke, "the pool is topped up AFTER the block, which overlaps nothing"
    between = src[top:invoke]
    assert "\n" in between and len(between) < 600, "the two must stay adjacent"


def test_the_top_up_refuses_rather_than_pretending():
    """No scheduler, or a full pool, must emit nothing at all."""
    src = _src()
    i = src.index("def _top_up_detached")
    body = src[i:src.index("\n    def ", i + 10)]
    assert "if sched is None:\n            return doc_now" in body
    assert "if len(live) >= want_live:\n            return doc_now" in body
    assert "already_detached" in body, "a job must not be launched twice"


def test_rank_detachable_puts_priority_zero_first():
    """Priority 0 is a REAL priority, not a missing one.

    m2 found the kickoff ranking with `_detach_priority(j) or 99`, so the long
    detached jobs - the ones most worth having in flight - sorted last and never
    started. Priority comes from the job's shape: long_subprocess or an explicit
    detached launch is 0, a shell is 1, a plain capability is 2.
    """
    queue = [
        {"capability": "x.py", "id": "c"},
        {"long_subprocess": True, "id": "a"},
        {"shell": ["echo"], "id": "b"},
    ]
    got = [j["id"] for j in ar.rank_detachable(queue)]
    assert got == ["a", "b", "c"], got
    assert ar._detach_priority(queue[1]) == 0, "0 must be a value, never None"


def test_a_job_that_cannot_be_detached_is_not_ranked():
    assert ar.rank_detachable([{"id": "x"}]) == []
    assert ar.rank_detachable([{"generate": {}, "id": "g"}]) == []
    assert ar.rank_detachable([{"capability": "x.py", "already_detached": True}]) == []
    assert ar.rank_detachable([{"capability": "x.py", "launch": "parked"}]) == []


def test_the_overlap_launch_says_why_it_exists():
    src = _src()
    assert "started so the next blocking invoke overlaps rather than idles" in src
