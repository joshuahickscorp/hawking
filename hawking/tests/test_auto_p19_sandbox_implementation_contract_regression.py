"""Focused regression for the P19 sandbox world identifier boundary.

Exercises the production behavior added to :mod:`hawking.sandbox_world`:
``_bounded_identifier`` must accept only values matching the declared
identifier pattern and must fail closed on malformed or over-long values.

Boundary: this is a unit-level check of the S0 record boundary only. It makes
no release claim and performs no hardware qualification.
"""
from __future__ import annotations

import pytest

from hawking.sandbox_world import (
    BRANCH_ID_RE,
    CANDIDATE_ID_RE,
    FINDING_ID_RE,
    WORLD_ID_RE,
    SandboxWorldError,
    _bounded_identifier,
)


def test_bounded_identifier_accepts_declared_shapes() -> None:
    assert _bounded_identifier("S0-abcd1234", "world_id", WORLD_ID_RE) == "S0-abcd1234"
    assert (
        _bounded_identifier("branch-alpha_1", "branch_id", BRANCH_ID_RE)
        == "branch-alpha_1"
    )
    assert (
        _bounded_identifier("finding-f-01", "finding_id", FINDING_ID_RE)
        == "finding-f-01"
    )
    assert (
        _bounded_identifier("candidate-c-01", "candidate_id", CANDIDATE_ID_RE)
        == "candidate-c-01"
    )


@pytest.mark.parametrize(
    "value",
    [
        "S0-ab",
        "world-abcd1234",
        "S0-abcd 1234",
        "S0-abcd/1234",
        "S0-" + "a" * 90,
    ],
)
def test_bounded_identifier_rejects_malformed_world_ids(value: str) -> None:
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_identifier(value, "world_id", WORLD_ID_RE)
    assert excinfo.value.status == 400


def test_bounded_identifier_rejects_non_text() -> None:
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_identifier(1234, "world_id", WORLD_ID_RE)
    assert excinfo.value.status == 400


def test_bounded_identifier_rejects_empty() -> None:
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_identifier("", "world_id", WORLD_ID_RE)
    assert excinfo.value.status == 400