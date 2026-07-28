#!/usr/bin/env python3.12
from __future__ import annotations

import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_capability_gate as gate  # noqa: E402


def test_plan_covers_math_and_both_live_prompts_without_execution(tmp_path):
    rows = gate.planned_commands(tmp_path)
    assert [row["name"] for row in rows] == ["math", "capital", "python"]
    assert rows[0]["tokens"] == gate.G_MATH_TOKENS
    assert [row["tokens"] for row in rows[1:]] == [
        tokens for _, tokens in gate.G_LIVE_PROMPTS
    ]
    assert all("glm52_flagship_oracle.py" in row["command"][1] for row in rows)
