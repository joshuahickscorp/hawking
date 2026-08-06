"""Offline contracts for the family-agnostic HCLI product-test harness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.condense import hcli_product_test_harness as harness  # noqa: E402
from lab.receipts import verify  # noqa: E402


def test_bible_case_count_and_order() -> None:
    assert len(harness.BIBLE_CASE_IDS) == 16
    assert harness.BIBLE_CASE_IDS[0] == "chat"
    assert harness.BIBLE_CASE_IDS[-1] == "document_perception"
    assert "read_safe_swarm" in harness.BIBLE_CASE_IDS
    assert "isolated_write_agent" in harness.BIBLE_CASE_IDS


def test_catalog_file_validates() -> None:
    report = harness.validate_catalog()
    assert report["ok"], report["errors"]
    assert report["case_count"] == 16


def test_master_schedule_seeds_not_started() -> None:
    report = harness.validate_master_schedule()
    assert report["ok"], report["errors"]
    assert report["step_count"] == 34
    assert report["step_ids"][0] == 0
    assert report["step_ids"][-1] == 33


def test_completion_states_seed_candidate() -> None:
    report = harness.validate_completion_states()
    assert report["ok"], report["errors"]
    assert report["state_count"] == 12
    assert "HAWKING_APPLE_PRODUCTION_RELEASE_READY" in report["state_ids"]
    assert "TG3_REVIEW_REQUIRED" in report["state_ids"]


def test_schedule_states_cross_check() -> None:
    report = harness.cross_check_schedule_and_states()
    assert report["ok"], report["errors"]


def test_validate_all() -> None:
    report = harness.validate_all()
    assert report["ok"], report["errors"]


def test_deepseek_map_covers_pattern_proven_cases() -> None:
    mapping = harness.scaffold_deepseek_diagnostic_mapping()
    assert mapping["receipt_schema"] == "hawking.gravity.deepseek_v4.hcli_live_suite.v1"
    # Pattern-proven cores must list evidence; Agent-OS-only cases may be empty.
    for case_id in (
        "chat",
        "repo_context",
        "coding",
        "planner_act_verify",
        "structured_json",
        "session_restart",
        "endpoint_restart",
        "context_compaction",
        "read_safe_swarm",
        "isolated_write_agent",
        "continuous_batching",
    ):
        assert mapping["cases"][case_id], f"{case_id} should map to DeepSeek evidence"
    for case_id in ("search_retrieval", "memory_ops", "skill_execution"):
        assert mapping["cases"][case_id] == []


def test_product_suite_receipt_scaffold_is_sealed_and_private(tmp_path: Path) -> None:
    out = tmp_path / "product-suite.json"
    report = harness.product_suite_receipt(
        family="deepseek_v4_layer4_diagnostic",
        artifact_seal_sha256="a" * 64,
        case_results=[
            {
                "id": "chat",
                "status": "PATTERN_PROVEN_ON_DEEPSEEK_DIAGNOSTIC",
                "evidence_sha256": "b" * 64,
                "prompt_hashes": [{"sha256": "c" * 64}],
                "prompt_text_disclosed": False,
            }
        ],
        out=out,
        live=False,
    )
    assert verify(report, label="HCLI product suite") == report
    assert report["schema"] == harness.PRODUCT_SUITE_SCHEMA
    assert report["status"] == harness.PRODUCT_SUITE_STATUS_SCAFFOLD
    assert report["claim_boundary"]["product_promotion"] is False
    assert report["metrics"]["primary_metric"] == harness.PRIMARY_METRIC
    assert report["cuda_preserve"] == list(harness.CUDA_PRESERVE)
    encoded = json.dumps(report)
    assert "secret prompt text" not in encoded
    assert json.loads(out.read_text(encoding="utf-8"))["seal_sha256"] == report["seal_sha256"]


def test_product_suite_receipt_rejects_prompt_disclosure() -> None:
    with pytest.raises(harness.HcliProductTestHarnessError, match="disclose prompt"):
        harness.product_suite_receipt(
            family="qwen3_coder_30b",
            artifact_seal_sha256="d" * 64,
            case_results=[
                {
                    "id": "chat",
                    "status": "NOT_RUN",
                    "prompt_text_disclosed": True,
                }
            ],
        )


def test_product_suite_receipt_rejects_unknown_case() -> None:
    with pytest.raises(harness.HcliProductTestHarnessError, match="unknown product case"):
        harness.product_suite_receipt(
            family="qwen3_coder_30b",
            artifact_seal_sha256="e" * 64,
            case_results=[{"id": "not_a_bible_case", "status": "NOT_RUN"}],
        )


def test_existing_deepseek_live_suite_contract_still_importable() -> None:
    """Regression anchor: family harness does not replace the DeepSeek packer."""
    from lab.operators import deepseek_v4_gravity as gravity

    assert callable(gravity.hcli_live_suite_receipt)
    assert hasattr(gravity, "hcli_live_suite_receipt")
