#!/usr/bin/env python3.12
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_runtime_parity_gate as gate  # noqa: E402


def test_runtime_agreement_requires_exact_discrete_and_continuous_bounds():
    reference = np.array([0.5, -1.0, 3.0, 2.0, 1.5, -0.25], dtype=np.float32)
    identical = gate.compare_logits(reference, reference.copy())
    assert identical["pass"]
    assert identical["argmax_exact"]
    assert identical["ordered_top5_exact"]

    wrong_argmax = reference.copy()
    wrong_argmax[3] = 4.0
    rejected = gate.compare_logits(reference, wrong_argmax)
    assert not rejected["pass"]
    assert not rejected["argmax_exact"]

    drift = reference * np.float32(1.001)
    continuous = gate.compare_logits(reference, drift)
    assert not continuous["pass"]
    assert continuous["relative_l2"] > gate.MAX_RELATIVE_L2


def test_runtime_agreement_rejects_shape_and_nonfinite():
    with pytest.raises(gate.RuntimeParityError, match="shapes"):
        gate.compare_logits(np.ones(3, dtype=np.float32), np.ones(2, dtype=np.float32))
    nonfinite = gate.compare_logits(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1.0, np.nan], dtype=np.float32),
    )
    assert not nonfinite["pass"]
    assert not nonfinite["all_finite"]


def test_capability_suite_is_shared_with_capability_gate():
    cases = gate.prompt_cases("capability", None)
    assert [name for name, _ in cases] == ["math", "capital", "python"]
    assert cases[0][1] == gate.G_MATH_TOKENS
    assert [tokens for _, tokens in cases[1:]] == [
        tokens for _, tokens in gate.G_LIVE_PROMPTS
    ]


def test_capability_suite_refuses_ambiguous_custom_tokens():
    with pytest.raises(gate.RuntimeParityError, match="cannot be combined"):
        gate.prompt_cases("capability", [1, 2, 3])
