"""The compiler's own scaffolding must not look like a body quoting the goal.

A single-obligation mission compiles to one WorkUnit whose description IS the
goal, and excising the root there removes the entire instruction. The guard
against that measured what remained after removing the root -- but the compiler
appends `obligations=G001 ` before the description and `Relevant files: a.py,
b.py` after it, and on a one-sentence goal that scaffolding ALONE cleared the
80-character threshold.

So the guard concluded the objective merely quoted the root, excised it, and the
worker received

    OBJECTIVE: obligations=G001 [ROOT_GOAL_OMITTED] Relevant files: ...

with both evidence files correctly attached and no instruction. It answered with
an empty operation, which is the only correct response to an empty instruction,
and was recorded as a model failure.
"""
from __future__ import annotations

from hcli.goal import ROOT_GOAL_OMITTED, _excise_root_goal, root_is_the_whole_objective

ROOT = (
    "Add a total_lines key to the whole-file return dict of _read_file in "
    "hcli/tool_registry.py, giving the line count of the whole file, so that "
    "hcli/tests/test_read_file_reports_total_lines.py passes."
)
BOILERPLATE = (
    " obligations=G001 " + ROOT + " Relevant files: hcli/tool_registry.py, "
    "hcli/tests/test_read_file_reports_total_lines.py"
)


def test_scaffolding_alone_does_not_make_the_root_a_quotation():
    assert root_is_the_whole_objective(BOILERPLATE, ROOT) is True


def test_a_real_body_quoting_the_goal_is_still_a_quotation():
    """The guard must keep doing its actual job."""
    body = (
        "Background follows. " + ROOT + " Also consider the scheduler, the "
        "cache, the retry path and the receipt writer, each in detail, before "
        "proposing any change at all."
    )
    assert root_is_the_whole_objective(body, ROOT) is False


def test_the_objective_line_keeps_its_instruction():
    prompt = "PHASE: running\nOBJECTIVE:" + BOILERPLATE + "\nEVIDENCE_PATHS:\n- a.py\n"
    out = _excise_root_goal(prompt, ROOT)
    assert ROOT_GOAL_OMITTED not in out, out
    assert ROOT in out


def test_a_duplicated_dump_elsewhere_is_still_excised():
    """Only the OBJECTIVE line is protected, not a re-dump further down."""
    prompt = (
        "OBJECTIVE:" + BOILERPLATE + "\n\nNEIGHBORHOOD:\nsome unit restated the "
        "whole thing: " + ROOT + " and then went on at length about other "
        "matters entirely, which is the duplication this excision exists for.\n"
    )
    out = _excise_root_goal(prompt, ROOT)
    assert out.count(ROOT) == 1
    assert ROOT_GOAL_OMITTED in out
