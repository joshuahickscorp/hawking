"""Focused unit tests for the unrun raw-final-logit retention successor plan."""
from __future__ import annotations

import pytest

from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_contract as retention


def test_six_vector_plan_is_full_vocab_and_separates_source_from_native() -> None:
    plan = retention.raw_vector_plan()
    assert plan["required_payload_count"] == 6
    assert plan["bytes_per_vector"] == 607_744
    assert plan["required_total_payload_bytes"] == 3_646_464
    assert plan["source_teacher_payloads"] == [
        "source_bf16_exact_prefix_logits.f32le",
        "source_bf16_forced_shared_continuation_logits.f32le",
    ]
    assert len(plan["native_successor_payloads"]) == 4
    assert set(plan["required_payloads"]) == set(plan["source_teacher_payloads"] + plan["native_successor_payloads"])


def test_six_vector_plan_rejects_non_positive_vocab() -> None:
    with pytest.raises(retention.RawFinalLogitRetentionContractError, match="positive"):
        retention.raw_vector_plan(vocab_rows=0)


def test_native_mode_is_explicitly_non_serving_retention_mode() -> None:
    assert retention.NATIVE_MODE == "metal-diagnostic-retain-raw-final-logits"
    assert retention.NATIVE_RAW_VECTOR_MODELS == ("scalar_control", "hq30gr2_candidate")
