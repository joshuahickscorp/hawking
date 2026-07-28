#!/usr/bin/env python3.12
from __future__ import annotations

import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_long_context_gate as gate  # noqa: E402


def _rows(passed: bool = True):
    return [
        {
            "rung": "2K",
            "partition": partition,
            "passed": passed,
        }
        for partition in gate.corpus.PARTITIONS
    ]


def test_exact_completion_contract():
    assert gate.score_completion("ORBIT-77\n", "ORBIT-77")
    assert gate.score_completion("orbit-77", "ORBIT-77")
    assert not gate.score_completion("The answer is ORBIT-77", "ORBIT-77")


def test_summary_requires_every_partition_and_every_answer():
    passed = gate.summarize(_rows(), ["2K"])
    assert passed["verdict"] == "PASS"
    assert passed["rungs"][0]["passes"] == len(gate.corpus.PARTITIONS)

    rows = _rows()
    rows[-1]["passed"] = False
    failed = gate.summarize(rows, ["2K"])
    assert failed["verdict"] == "FAIL"


def test_summary_refuses_incomplete_rung():
    with pytest.raises(gate.LongContextGateError, match="expected"):
        gate.summarize(_rows()[:-1], ["2K"])
