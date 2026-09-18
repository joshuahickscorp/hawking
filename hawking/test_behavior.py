"""Behavior lab: real BHV fixtures and a local five-axis score kernel."""
from __future__ import annotations

import ast
from pathlib import Path

from hawking.behavior import (
    FIXTURE_IDS,
    ScoreVector,
    evaluate,
    run_fixture,
    run_matrix,
)


def _call_sites(path: Path, module: str, symbol: str) -> list[int]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    binds: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                binds[alias.asname or alias.name] = (node.module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                binds[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target = None
        if isinstance(func, ast.Name) and func.id in binds:
            mod, name = binds[func.id]
            target = f"{mod}.{name or func.id}"
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in binds:
                mod, name = binds[func.value.id]
                target = f"{mod}.{func.attr}" if name is None else f"{mod}.{name}.{func.attr}"
        if target == f"{module}.{symbol}":
            lines.append(node.lineno)
    return lines


def test_fixture_ids_are_bhv_01_through_23():
    assert list(FIXTURE_IDS) == [f"BHV-{i:02d}" for i in range(1, 24)]


def test_bhv09_refuses_dangerous_command_without_executing():
    row = run_fixture("BHV-09")
    assert row["ok"] is True
    assert row["empty_success"] is False
    assert row["instruction_ok"] is True
    assert row["tool_receipt_ok"] is True
    assert row["initial_hash"]
    assert Path("/").is_dir()


def test_bhv04_goes_red_then_green():
    row = run_fixture("BHV-04")
    assert row["ok"] is True
    assert row["tests"]["red_exit"] != 0
    assert row["tests"]["green_exit"] == 0
    assert row.get("red") is True
    assert row.get("green") is True


def test_bhv21_rejects_empty_success():
    row = run_fixture("BHV-21")
    assert row["ok"] is True
    assert row["empty_success"] is False
    assert row["reasoning_ok"] is True


def test_run_matrix_all_twenty_three_uses_the_hawking_score_kernel():
    src = Path(__file__).resolve().parent / "behavior.py"
    source = src.read_text(encoding="utf-8")
    assert "tools.future.tabula" not in source
    assert "vec = scores_from_behavior_lab(rows)" in source
    assert "verdict = evaluate(vec)" in source
    assert _call_sites(src, "hawking.perception.file_eye", "observe")
    assert _call_sites(src, "hawking.perception.doctor", "profile")
    matrix = run_matrix()
    assert matrix["n"] == 23
    assert matrix["n_ok"] == 23, matrix.get("residuals")
    assert matrix["laboratory_profile_used"] is False
    assert matrix["empty_success"] is False
    assert matrix["verdict"]["outcome"] == "PASS"
    assert matrix["network_used"] is False
    for row in matrix["fixtures"]:
        assert row["ran"] is True
        assert row["empty_success"] is False
        assert row["initial_hash"]
        assert row["tool_receipt_ok"] is True or row["id"] in {"BHV-15", "BHV-21"}


def test_score_kernel_rejects_a_behavioral_only_pass():
    verdict = evaluate(
        ScoreVector(
            behavioral=0.9,
            capability=0.1,
            tool_use=-0.1,
            reasoning=0.1,
            instruction_following=0.1,
        )
    )
    assert verdict.outcome == "FAILURE"
    assert verdict.regressions == ("tool_use",)
