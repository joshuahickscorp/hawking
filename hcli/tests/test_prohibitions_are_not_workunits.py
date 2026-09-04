"""A prohibition is a constraint on every unit. It is not a unit of work.

Measured, `.hcli/mission/state.json` for the tools/hcli_metric.py goal: FIFTEEN
work units for a one-file, two-function change, and the mission log grinding on
`G003.work.repair.1.repair.1` -- repair depth two -- with `accepted=0`.

`G003`'s compiled OBJECTIVE was, verbatim:

    OBJECTIVE: obligation=G003 Do not weaken or edit that test file.

There is no mutation that discharges that. It is satisfied by doing nothing, so
it can never produce the "observable evidence ... is discharged" its own
acceptance criterion demands, and the repair machinery retries it forever. Each
retry is resident calls, which is the metric directive XI is actually about.

`_extract_obligations` builds an obligation from every sentence matching
`_ACCEPTANCE_MARKERS + _INVARIANT_MARKERS`, and `_INVARIANT_MARKERS` is
("do not", "don't", "must", "must not", "never", "only", "without", "preserve",
"required"). The same sentences ALREADY flow into the packet's INVARIANTS
section, so a prohibition is compiled twice: once as the constraint it is, and
once as a work unit it can never be.

Note the trap: "Do not weaken or edit that test file" also contains "test",
which is an ACCEPTANCE marker. So the rule cannot be "has no invariant marker" --
it has to be that a sentence which OPENS with a prohibition is a constraint,
whatever else it contains.
"""
from __future__ import annotations

from hcli.goal import GoalCompiler


GOAL = (
    "In tools/hcli_metric.py, decode_seconds returns None even when every call is "
    "instrumented. Fix tools/hcli_metric.py so that all four tests in "
    "tools/test_hcli_metric_never_zeroes_unknown.py pass. "
    "Do not edit tools/test_hcli_metric_never_zeroes_unknown.py. "
    "Do not touch any file under crates/ or shaders/. "
    "Never weaken the verifier."
)


def _obligation_texts(compiled) -> list:
    out = []
    for ob in compiled.get("obligations") or []:
        text = ob.get("text") or ob.get("statement") or ob.get("description") or ""
        out.append(str(text))
    return out


def test_a_prohibition_never_becomes_an_obligation():
    compiled = GoalCompiler().compile(GOAL)
    texts = _obligation_texts(compiled)
    offenders = [
        t for t in texts
        if t.strip().lower().startswith(("do not", "don't", "never", "must not", "no "))
    ]
    assert not offenders, (
        "these prohibitions were compiled into work units, each of which can only "
        f"be discharged by doing nothing and so can never pass: {offenders}"
    )


def test_the_prohibition_survives_as_an_invariant():
    """Dropping it from obligations must not drop it from the contract."""
    compiled = GoalCompiler().compile(GOAL)
    inv = " ".join(str(x) for x in (compiled.get("invariants") or []))
    assert "Do not edit tools/test_hcli_metric_never_zeroes_unknown.py" in inv
    assert "Do not touch any file under crates/ or shaders/" in inv
    assert "Never weaken the verifier" in inv


def test_the_real_work_still_becomes_an_obligation():
    """The guard must not eat the actual task."""
    compiled = GoalCompiler().compile(GOAL)
    texts = " ".join(_obligation_texts(compiled))
    assert "pass" in texts.lower(), (
        "the one sentence that names a falsifiable outcome was dropped along with "
        f"the prohibitions: {texts!r}"
    )


def test_a_goal_that_is_only_prohibitions_still_compiles():
    """Degenerate input must not produce an empty, unworkable plan."""
    compiled = GoalCompiler().compile(
        "Do not touch crates/. Do not edit the tests. Never weaken the verifier."
    )
    assert compiled.get("invariants"), "the constraints vanished entirely"
