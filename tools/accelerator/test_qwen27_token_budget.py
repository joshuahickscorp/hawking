from __future__ import annotations

from pathlib import Path

import qwen27_token_budget as budget


REPO = Path(__file__).resolve().parents[2]


def test_qwen27_budget_separates_static_bytes_from_missing_physical_metrics():
    body = budget.build_budget(repo_root=REPO)
    assert body["schema"] == budget.SCHEMA
    assert body["status"] == "PLANNED_UNTIL_NATIVE_PROTECTED_EXECUTION"
    assert body["source_byte_denominator"]["active_weight_bytes_per_token"] == 9_878_901_136
    assert len(body["source_byte_denominator"]["regions"]) == 14
    assert all(value is None for value in body["system_ledger"].values())
    assert all(
        value is None
        for organ in body["organs"]
        for value in organ["actual"].values()
    )
    assert body["promotion_allowed"] is False
    assert body["control_observation"]["status"] == "PROTECTED_CONTROL_NOT_FOR_PROMOTION"
