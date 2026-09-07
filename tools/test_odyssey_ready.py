"""odyssey_ready printed READY=13/13 and the verdict was vacuous.

Its callers() ran `git grep -l --fixed-strings <module-stem>` and called every
matching file a caller. That counts the WORD, not a call: a roadmap catalog
naming the path as data, a doc listing it, the module's own usage docstring.
It reported 257 "callers" for inventory (2 real production import sites) and
240 for tournament (1). Nothing in the required graph could ever score zero, so
the gate could not fail, so READY=13/13 measured nothing.

These two tests pin the two ways the old count lied. Both go through classify(),
whose signature is unchanged, so they run against either implementation.
"""
from __future__ import annotations

import pathlib
import subprocess

from tools import odyssey_ready

REPO = pathlib.Path(__file__).resolve().parents[1]


def _word_mentions(stem: str) -> list[str]:
    """Exactly what the old callers() counted."""
    out = subprocess.run(
        ["git", "grep", "-l", "--fixed-strings", stem, "--", "."],
        cwd=REPO, capture_output=True, text=True,
    ).stdout.split()
    return [o for o in out
            if not o.startswith(("receipts/", "workspace/campaign/", "research/receipts/"))]


def test_word_mentions_are_not_call_sites():
    """tools/gravity_verify_source.py is named as DATA by tools/roadmap/catalog.py,
    listed in civilization/CAPABILITY_GRAPH.json and in its own usage docstring.
    Nothing imports it and nothing launches it. It is a hand-run CLI, not wired
    capability, and a readiness gate must say so.

    If it ever acquires a real production caller this test should be updated to
    another zero-caller module, not deleted.
    """
    mentions = _word_mentions("gravity_verify_source")
    assert len(mentions) >= 5, f"precondition: the word is widespread, got {mentions}"

    state, why = odyssey_ready.classify("tools/gravity_verify_source.py", None)
    assert state == "PARTIAL", (
        f"{len(mentions)} files mention the word and none call it, "
        f"yet the gate says {state}: {why}"
    )


def test_test_only_callers_are_not_production_wiring():
    """tools/odyssey/performance_qualification.py is imported by two of its own
    test modules and by nothing in production. Test-only reachability is reported,
    never counted as wiring -- otherwise a capability's own tests certify it.
    """
    state, why = odyssey_ready.classify("tools/odyssey/performance_qualification.py", None)
    assert state == "PARTIAL", f"reachable only from tests, yet the gate says {state}: {why}"
    assert "test" in why, f"the test-only reachability is not reported: {why}"
