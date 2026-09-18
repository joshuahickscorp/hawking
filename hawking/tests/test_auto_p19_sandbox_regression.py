"""Focused regression for the bounded S0 world field coercion helper.

Exercises the production behavior added to :mod:`hawking.sandbox_world`:
``_bounded_text`` must accept in-range text unchanged and fail closed on
non-strings, empty strings, and over-long strings.
"""
from __future__ import annotations

import pytest

from hawking.sandbox_world import SandboxWorldError, _bounded_text


def test_bounded_text_accepts_in_range_text() -> None:
    assert _bounded_text("branch-abc", "branch_id", 16) == "branch-abc"


def test_bounded_text_rejects_non_string() -> None:
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_text(7, "branch_id", 16)
    assert excinfo.value.status == 400


def test_bounded_text_rejects_empty() -> None:
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_text("", "branch_id", 16)
    assert excinfo.value.status == 400


def test_bounded_text_rejects_over_long() -> None:
    with pytest.raises(SandboxWorldError) as excinfo:
        _bounded_text("x" * 17, "branch_id", 16)
    assert excinfo.value.status == 400