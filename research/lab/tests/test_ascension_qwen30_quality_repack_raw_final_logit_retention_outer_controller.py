"""Focused no-child tests for Q30's raw-final-logit outer controller."""
from __future__ import annotations

from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_outer_controller as controller
from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_contract as contract


def _native_payloads() -> dict[str, object]:
    return {
        model: {
            endpoint: {
                "path": f"/tmp/{model}_{endpoint}_logits.f32le",
                "dtype": "f32le",
                "vocab_rows": 151_936,
                "bytes": 607_744,
                "sha256": "a" * 64,
                "all_values_finite": True,
            }
            for endpoint in contract.ENDPOINTS
        }
        for model in contract.NATIVE_RAW_VECTOR_MODELS
    }


def _source_payloads() -> dict[str, object]:
    return {
        endpoint: {
            "path": f"/tmp/source_bf16_{endpoint}_logits.f32le",
            "dtype": "f32le",
            "vocab_rows": 151_936,
            "bytes": 607_744,
            "sha256": "d" * 64,
            "all_values_finite": True,
        }
        for endpoint in contract.ENDPOINTS
    }


def test_native_payload_replay_rejects_hash_that_does_not_match_98db() -> None:
    minimal_contract = {
        "six_vector_retention_contract": contract.raw_vector_plan(),
        "replay_binding": {
            "native_raw_hashes_must_replay_prior_98db_witness": {
                endpoint: {"scalar_control": "a" * 64, "hq30gr2_candidate": "a" * 64}
                for endpoint in contract.ENDPOINTS
            }
        },
    }
    payloads = _native_payloads()
    assert controller.validate_native_payload_replay(contract=minimal_contract, native_payloads=payloads)
    payloads["scalar_control"]["exact_prefix"]["sha256"] = "b" * 64  # type: ignore[index]
    with pytest.raises(controller.RawFinalLogitOuterControllerError, match="98db"):
        controller.validate_native_payload_replay(contract=minimal_contract, native_payloads=payloads)


def test_native_payload_replay_rejects_wrong_full_vector_bytes() -> None:
    minimal_contract = {
        "six_vector_retention_contract": contract.raw_vector_plan(),
        "replay_binding": {
            "native_raw_hashes_must_replay_prior_98db_witness": {
                endpoint: {"scalar_control": "a" * 64, "hq30gr2_candidate": "a" * 64}
                for endpoint in contract.ENDPOINTS
            }
        },
    }
    payloads = _native_payloads()
    payloads["hq30gr2_candidate"]["forced_shared_continuation"]["bytes"] = 1  # type: ignore[index]
    with pytest.raises(controller.RawFinalLogitOuterControllerError, match="byte count"):
        controller.validate_native_payload_replay(contract=minimal_contract, native_payloads=payloads)


def test_six_vector_set_requires_two_source_and_four_native_distinct_paths() -> None:
    minimal_contract = {
        "six_vector_retention_contract": contract.raw_vector_plan(),
        "replay_binding": {
            "native_raw_hashes_must_replay_prior_98db_witness": {
                endpoint: {"scalar_control": "a" * 64, "hq30gr2_candidate": "a" * 64}
                for endpoint in contract.ENDPOINTS
            }
        },
    }
    source = _source_payloads()
    native = _native_payloads()
    checked = controller.validate_six_vector_terminal_set(
        contract=minimal_contract,
        source_payloads=source,
        native_payloads=native,
    )
    assert checked["payload_count"] == 6
    assert checked["total_bytes"] == 3_646_464
    source["exact_prefix"]["path"] = native["scalar_control"]["exact_prefix"]["path"]  # type: ignore[index]
    with pytest.raises(controller.RawFinalLogitOuterControllerError, match="distinct"):
        controller.validate_six_vector_terminal_set(
            contract=minimal_contract,
            source_payloads=source,
            native_payloads=native,
        )


def test_regular_rejects_relative_path() -> None:
    with pytest.raises(controller.RawFinalLogitOuterControllerError, match="absolute"):
        controller._regular(Path("relative"), label="relative")


def test_source_eviction_must_precede_a_fresh_native_lease() -> None:
    source_terminal = {
        "schema": controller.SOURCE_TERMINAL_SCHEMA,
        "status": controller.SOURCE_TERMINAL_STATUS,
        "seal_sha256": "c" * 64,
    }
    eviction = {
        "schema": controller.SOURCE_EVICTION_SCHEMA,
        "status": controller.SOURCE_EVICTION_STATUS,
        "source_teacher_terminal": {"seal_sha256": "c" * 64},
        "eviction": {
            "source_weights_evicted": True,
            "source_backend_shutdown": True,
            "source_model_residency_released": True,
            "swap_remained_zero": True,
            "pre_native_lease_process_tree_checked": True,
        },
    }
    lease = {
        "schema": controller.NATIVE_LEASE_SCHEMA,
        "status": controller.NATIVE_LEASE_STATUS,
        "one_shot_lifecycle": {
            "fresh_for_this_exact_launch": True,
            "prior_terminal_receipt": None,
            "automatic_retry_allowed": False,
        },
    }
    controller.validate_source_eviction_before_native(
        source_terminal=source_terminal,
        source_eviction=eviction,
        native_lease=lease,
    )
    eviction["eviction"]["source_model_residency_released"] = False
    with pytest.raises(controller.RawFinalLogitOuterControllerError, match="source_model_residency_released"):
        controller.validate_source_eviction_before_native(
            source_terminal=source_terminal,
            source_eviction=eviction,
            native_lease=lease,
        )
