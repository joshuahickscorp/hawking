"""A file the packet NAMES must become a file the packet can SHOW.

Measured, receipt `.hcli/receipts/16c237d7-d2e0-40b4-9b62-742e6edf2824.json`:
a mutation WorkUnit was compiled with `tools/test_hcli_metric_never_zeroes_unknown.py`
sitting in plain text in its own INVARIANTS and ACCEPTANCE sections, and the same
packet rendered

    EVIDENCE_PATHS:
    (none)

The run was under `HCLI_NO_TOOLS=1`, where evidence is the ONLY channel to a
file's bytes. So the model was shown a filename, given no bytes, no tool, and
asked for an exact `old_text` anchor. It guessed a DIFFERENT file and invented
lines: "your old_text for hcli/engine.py matches nothing in the file -- not one
line of it", twice, then exhausted the completion budget mid-JSON on the third.
Four resident calls, 155 s of prompt wall, zero operations.

That is a PRE-COGNITION defect, not a reasoning failure, and no larger model
fixes it. `_mentioned_and_known_files` scans `wu.description` and the failure
context and nothing else, while `invariants` and `acceptance` are already
computed and in scope at the call site.
"""
from __future__ import annotations

from hcli.goal import compile_worker_context
from hcli.workunit import WorkUnit


def _unit() -> WorkUnit:
    # Deliberately names NO file. The real compiler produced exactly this: the
    # clause naming the target was dropped from the OBJECTIVE.
    return WorkUnit(
        id="G001.work",
        role="implementation",
        description="Add two module-level functions that return None for an "
        "uninstrumented call instead of substituting zero.",
    )


def _packet(compiled: dict):
    return compile_worker_context(
        _unit(),
        compiled,
        phase="running",
        units={},
        steering=[],
        goal_ref="/tmp/state.json#goal",
        root_goal=str(compiled.get("goal") or ""),
    )


def test_a_path_named_only_in_invariants_becomes_evidence():
    packet = _packet(
        {
            "goal": "make the dashboard honest",
            "invariants": [
                "Change it so that tools/test_hcli_metric_never_zeroes_unknown.py passes."
            ],
            "acceptance_criteria": [],
            "referenced_files": [],
        }
    )
    assert "tools/test_hcli_metric_never_zeroes_unknown.py" in packet.evidence_paths, (
        "the packet names the file in its own INVARIANTS and still renders "
        f"EVIDENCE_PATHS {packet.evidence_paths!r}"
    )


def test_a_path_named_only_in_acceptance_becomes_evidence():
    packet = _packet(
        {
            "goal": "make the dashboard honest",
            "invariants": [],
            "acceptance_criteria": [
                "the named tests collect at least one case: tools/hcli_metric.py"
            ],
            "referenced_files": ["tools/hcli_metric.py"],
        }
    )
    assert "tools/hcli_metric.py" in packet.evidence_paths, (
        "the path is in referenced_files AND in the rendered ACCEPTANCE and "
        f"still did not reach EVIDENCE_PATHS {packet.evidence_paths!r}"
    )


def test_the_rendered_prompt_does_not_say_none_when_it_names_a_file():
    packet = _packet(
        {
            "goal": "make the dashboard honest",
            "invariants": ["Do not weaken tools/hcli_metric.py."],
            "acceptance_criteria": [],
            "referenced_files": [],
        }
    )
    assert "EVIDENCE_PATHS:\n(none)" not in packet.prompt, (
        "a packet that names a file rendered EVIDENCE_PATHS (none); under "
        "HCLI_NO_TOOLS that leaves the model a filename and no bytes"
    )


def test_a_packet_that_names_no_file_at_all_still_renders_none():
    # The guard must not invent evidence. Nothing named -> nothing shown.
    packet = _packet(
        {
            "goal": "think about something",
            "invariants": ["Be brief."],
            "acceptance_criteria": ["it is brief"],
            "referenced_files": [],
        }
    )
    assert packet.evidence_paths == ()


def test_a_referenced_file_the_unit_never_re_mentions_is_still_shown():
    """The exact shape that produced the 4-call zero-operation failure.

    The root goal named `tools/hcli_metric.py`; the compiler captured it in
    referenced_files; the OBJECTIVE it derived started one sentence later, so
    the path survived nowhere in the unit's own text. The packet was then
    compiled ABOUT a file it could not show.
    """
    packet = _packet(
        {
            "goal": "make the dashboard honest",
            "invariants": [
                "Change it so that tools/test_hcli_metric_never_zeroes_unknown.py passes."
            ],
            "acceptance_criteria": [],
            "referenced_files": [
                "tools/hcli_metric.py",
                "tools/test_hcli_metric_never_zeroes_unknown.py",
            ],
        }
    )
    assert "tools/hcli_metric.py" in packet.evidence_paths, (
        "the goal's own referenced_files named the target and the packet still "
        f"could not show it: {packet.evidence_paths!r}"
    )


def test_implementation_source_is_first_when_engine_and_test_are_declared():
    unit = WorkUnit(
        id="implement.followup.2",
        role="implementation",
        description=(
            "Use hcli/test_engine_tool_loop.py to prove the next executable "
            "step in hcli/engine.py."
        ),
    )
    packet = compile_worker_context(
        unit,
        {
            "goal": "build FrontierEngine",
            "invariants": [],
            "acceptance_criteria": [],
            "referenced_files": [
                "hcli/engine.py",
                "hcli/test_engine_tool_loop.py",
            ],
        },
        phase="running",
        units={},
        steering=[],
        root_goal="build FrontierEngine",
    )
    assert packet.evidence_paths[0] == "hcli/engine.py"
