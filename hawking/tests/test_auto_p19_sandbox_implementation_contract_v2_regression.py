"""Focused regression for the P19 sandbox world identifier-list boundary.

Exercises the production behavior added to :mod:`hawking.sandbox_world`:
``_bounded_identifier_list`` must fail closed on non-list, empty, over-long,
and malformed rows, and must return an immutable tuple of validated
identifiers for a well-formed collection.
"""
from __future__ import annotations

import pytest

from hawking.sandbox_world import SandboxWorldError, _bounded_identifier_list


BRANCH_PATTERN = r"[a-z0-9][a-z0-9._-]{0,63}"


def test_identifier_list_accepts_well_formed_rows_as_immutable_tuple() -> None:
    result = _bounded_identifier_list(
        ["branch-1", "branch.2", "b3"], "branch_ids", BRANCH_PATTERN
    )
    assert result == ("branch-1", "branch.2", "b3")
    assert isinstance(result, tuple)


def test_identifier_list_rejects_non_list_scalar() -> None:
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_identifier_list("branch-1", "branch_ids", BRANCH_PATTERN)
    assert excinfo.value.status == 400
    assert "must be a list of identifiers" in str(excinfo.value)


def test_identifier_list_rejects_empty_collection() -> None:
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_identifier_list([], "branch_ids", BRANCH_PATTERN)
    assert excinfo.value.status == 400
    assert "must be non-empty" in str(excinfo.value)


def test_identifier_list_rejects_over_long_collection() -> None:
    rows = [f"branch-{index}" for index in range(65)]
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_identifier_list(rows, "branch_ids", BRANCH_PATTERN)
    assert excinfo.value.status == 400
    assert "exceeds 64 identifiers" in str(excinfo.value)


def test_identifier_list_rejects_malformed_row_with_indexed_field() -> None:
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_identifier_list(
            ["branch-1", "Branch 2"], "branch_ids", BRANCH_PATTERN
        )
    assert excinfo.value.status == 400
    assert "branch_ids[1]" in str(excinfo.value)
    assert "is not a valid identifier" in str(excinfo.value)