"""A whole-file replace must SAY how much of the file it destroyed.

Receipt ecf6d616 was ACCEPTED -- kind mutation, status completed, py_compile exit
0, 4 of 4 tests green, red_before_green True -- for a `replace_file` that cut
tools/hcli_metric.py from 202 lines to 72, deleted the entire dashboard, and left
two names read but never bound. Nothing in the receipt said so. The recorded
operation carried `op`, `path` and `new_text`, and a reader looking for "what did
this change actually do to the file" had only the model's own prose summary,
which said "add two helpers".

Strengthening the test file fixed that ONE file. This is the producer: any
replace_file, on any file, now records what it did to the line count, so the next
narrow test cannot license the same destruction invisibly.

This does NOT refuse the operation. A legitimate rewrite exists and a guard that
blocks it would be a worse defect than the one it closes. It makes the blast
radius VISIBLE, which is the thing that was missing.
"""
from __future__ import annotations

from hcli.engine import blast_radius


def test_a_whole_file_deletion_is_reported_as_such():
    before = "\n".join(f"line {i}" for i in range(202)) + "\n"
    after = "\n".join(f"line {i}" for i in range(72)) + "\n"
    br = blast_radius(before, after)
    assert br["lines_before"] == 202
    assert br["lines_after"] == 72
    assert br["lines_removed"] == 130
    assert br["fraction_removed"] > 0.64
    assert br["mostly_deleted"] is True, (
        "a replace_file that removes 64% of a file must be flagged; this is the "
        "exact shape that passed py_compile and 4/4 tests while deleting a dashboard"
    )


def test_an_ordinary_edit_is_not_flagged():
    """The guard must not cry wolf on real work."""
    before = "\n".join(f"line {i}" for i in range(200)) + "\n"
    after = before + "line 200\nline 201\n"
    br = blast_radius(before, after)
    assert br["lines_removed"] == 0
    assert br["mostly_deleted"] is False


def test_a_deliberate_shrink_that_keeps_most_of_the_file_is_not_flagged():
    before = "\n".join(f"line {i}" for i in range(100)) + "\n"
    after = "\n".join(f"line {i}" for i in range(90)) + "\n"
    br = blast_radius(before, after)
    assert br["fraction_removed"] < 0.2
    assert br["mostly_deleted"] is False


def test_an_empty_before_does_not_divide_by_zero():
    br = blast_radius("", "x = 1\n")
    assert br["lines_before"] == 0
    assert br["mostly_deleted"] is False


def test_the_applier_attaches_it_to_the_operation_the_receipt_echoes():
    """The number has to reach the RECEIPT, not merely exist as a function.

    A helper nothing calls is the disease this repo keeps rediscovering, so this
    pins the call site: after replace_file runs, the operation dict the receipt
    serialises must carry the measurement.
    """
    import inspect
    from hcli import engine as eng

    src = inspect.getsource(eng.Engine._apply_operations)
    idx = src.index('if op == "replace_file"')
    tail = src[idx: idx + 2000]
    assert "blast_radius" in tail, (
        "replace_file writes the file without recording what it did to it; "
        "the receipt then shows only the model's own description of the change"
    )
