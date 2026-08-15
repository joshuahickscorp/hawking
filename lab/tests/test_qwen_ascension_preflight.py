"""Metadata-only Qwen Ascension preflight tests (no network, no Hub, no weights).

These tests exercise fail-closed admission for prospective 30B / 80B lanes.
They never download, never load a model, and never invent public model pins.
"""
from __future__ import annotations

import pytest

from lab.operators.qwen_ascension_preflight import (
    FAMILY_TO_LANE,
    LANE_RECORDS,
    PREFLIGHT_SCHEMA,
    PREFLIGHT_STATUS_ADMITTED,
    PREFLIGHT_STATUS_BLOCKED,
    PreflightInputError,
    QwenLane,
    REQUIRED_GATES,
    default_blocked_decision,
    distinguish_lane,
    evaluate_qwen_preflight,
    lane_record,
    required_gates,
)
from lab.receipts import verify


# Synthetic identity â deliberately not a real public Hub pin. Tests only.
_SYNTH_SOURCE_30B = "synthetic/test-qwen-30b-lane"
_SYNTH_SOURCE_80B = "synthetic/test-qwen-80b-lane"
_SYNTH_REVISION = "a" * 40
_SYNTH_DIGEST = "b" * 64


def _identity_30b(**overrides: object) -> dict:
    base = {
        "scale_label": "30B",
        "family_key": "QWEN3_MOE",
        "source": _SYNTH_SOURCE_30B,
        "revision": _SYNTH_REVISION,
        "config_digest": _SYNTH_DIGEST,
        "license": "test-license",
        "architecture": {
            "model_type": "test_qwen3_moe",
            "architectures": ["TestQwen3MoeForCausalLM"],
        },
    }
    base.update(overrides)
    return base


def _identity_80b(**overrides: object) -> dict:
    base = {
        "scale_label": "80B",
        "family_key": "QWEN3_NEXT",
        "source": _SYNTH_SOURCE_80B,
        "revision": _SYNTH_REVISION,
        "config_digest": _SYNTH_DIGEST,
        "license": "test-license",
        "architecture": {
            "model_type": "test_qwen3_next",
            "architectures": ["TestQwen3NextForCausalLM"],
        },
    }
    base.update(overrides)
    return base


def _all_gates_green() -> dict:
    return {name: {"status": "PASS", "receipt_id": f"rx-{name}"} for name in REQUIRED_GATES}


def _full_evidence(*, family: str = "QWEN3_MOE", gates: dict | None = None) -> dict:
    return {
        "gates": gates if gates is not None else _all_gates_green(),
        "runtime_capability": {
            "supported_families": [family],
            "loader_support": "PASS",
            "forward_support": "PASS",
        },
    }


# ---------------------------------------------------------------------------
# Defaults / inventory
# ---------------------------------------------------------------------------


def test_default_decision_is_blocked_and_download_forbidden() -> None:
    decision = default_blocked_decision()
    verify(decision, label="default-preflight")
    assert decision["schema"] == PREFLIGHT_SCHEMA
    assert decision["status"] == PREFLIGHT_STATUS_BLOCKED
    assert decision["download_permitted"] is False
    assert decision["claim_boundary"]["metadata_only"] is True
    assert decision["claim_boundary"]["permission_to_fetch_model_bodies"] is False
    assert decision["claim_boundary"]["network_calls"] is False
    assert decision["honesty"]["fail_closed"] is True
    assert decision["reasons"]


def test_empty_mappings_block() -> None:
    decision = evaluate_qwen_preflight({}, {})
    assert decision["status"] == PREFLIGHT_STATUS_BLOCKED
    assert decision["download_permitted"] is False
    assert decision["all_gates_green"] is False


def test_required_gates_checklist_is_complete() -> None:
    gates = required_gates()
    assert gates == REQUIRED_GATES
    assert "source_admission" in gates
    assert "pinned_identity" in gates
    assert "runtime_loader_forward_support" in gates
    assert "resource_supervisor_green" in gates
    assert "actual_artifact_receipt" in gates
    assert "profiler_parity_capability" in gates
    assert "controller_approval" in gates
    assert len(gates) == 7


def test_lane_records_distinguish_30b_and_80b() -> None:
    r30 = lane_record("30B")
    r80 = lane_record("80B")
    assert r30["lane_id"] != r80["lane_id"]
    assert r30["family_key"] == "QWEN3_MOE"
    assert r80["family_key"] == "QWEN3_NEXT"
    assert r30["role"] == "executor"
    assert r80["role"] == "reviewer"
    assert r30["scale_label"] == QwenLane.QWEN_30B.value
    assert r80["scale_label"] == QwenLane.QWEN_80B.value
    # Scaffold must not invent a concrete public Hub identity.
    assert "hf_id" not in r30
    assert "hf_id" not in r80
    assert FAMILY_TO_LANE["QWEN3_MOE"] == "30B"
    assert FAMILY_TO_LANE["QWEN3_NEXT"] == "80B"
    assert set(LANE_RECORDS) == {"30B", "80B"}


def test_unknown_lane_raises() -> None:
    with pytest.raises(PreflightInputError):
        lane_record("70B")


# ---------------------------------------------------------------------------
# Lane distinction
# ---------------------------------------------------------------------------


def test_distinguish_30b_via_scale_and_family() -> None:
    info = distinguish_lane({"scale_label": "30B", "family_key": "QWEN3_MOE"})
    assert info["resolved"] is True
    assert info["scale_label"] == "30B"
    assert info["family_key"] == "QWEN3_MOE"
    assert info["lane_id"] == "qwen_30b"


def test_distinguish_80b_via_family_only() -> None:
    info = distinguish_lane({"family_key": "QWEN3_NEXT"})
    assert info["resolved"] is True
    assert info["scale_label"] == "80B"
    assert info["role"] == "reviewer"


def test_mismatched_family_and_scale_refused() -> None:
    info = distinguish_lane({"scale_label": "30B", "family_key": "QWEN3_NEXT"})
    assert info["resolved"] is False
    assert any("does not match" in r for r in info["reasons"])


def test_preflight_keeps_30b_and_80b_records_distinct() -> None:
    d30 = evaluate_qwen_preflight(_identity_30b(), _full_evidence(family="QWEN3_MOE"))
    d80 = evaluate_qwen_preflight(_identity_80b(), _full_evidence(family="QWEN3_NEXT"))
    assert d30["lane"]["scale_label"] == "30B"
    assert d80["lane"]["scale_label"] == "80B"
    assert d30["lane"]["family_key"] != d80["lane"]["family_key"]
    assert d30["identity"]["source"] == _SYNTH_SOURCE_30B
    assert d80["identity"]["source"] == _SYNTH_SOURCE_80B
    assert d30["identity"]["source"] != d80["identity"]["source"]


# ---------------------------------------------------------------------------
# Gate fail-closed behaviour
# ---------------------------------------------------------------------------


def test_missing_any_gate_blocks_download() -> None:
    for missing in REQUIRED_GATES:
        gates = _all_gates_green()
        del gates[missing]
        decision = evaluate_qwen_preflight(
            _identity_30b(),
            _full_evidence(gates=gates),
        )
        assert decision["download_permitted"] is False, missing
        assert decision["status"] == PREFLIGHT_STATUS_BLOCKED, missing
        assert any(missing in r for r in decision["reasons"]), missing


def test_failed_gate_blocks_download() -> None:
    gates = _all_gates_green()
    gates["controller_approval"] = {"status": "DENIED"}
    decision = evaluate_qwen_preflight(_identity_30b(), _full_evidence(gates=gates))
    assert decision["download_permitted"] is False
    assert decision["gates"]["controller_approval"]["green"] is False
    assert decision["status"] == PREFLIGHT_STATUS_BLOCKED


def test_claimed_download_permitted_rejected_until_gates_pass() -> None:
    decision = evaluate_qwen_preflight(
        _identity_30b(),
        {"gates": {}, "runtime_capability": {"supported_families": ["QWEN3_MOE"]}},
        claimed_download_permitted=True,
    )
    assert decision["download_permitted"] is False
    assert decision["claimed_download_permitted"] is True
    assert decision["claimed_download_permitted_rejected"] is True
    assert any("download_permitted=true" in r for r in decision["reasons"])


def test_all_gates_green_with_identity_and_family_admits_metadata_only() -> None:
    decision = evaluate_qwen_preflight(
        _identity_30b(),
        _full_evidence(family="QWEN3_MOE"),
        claimed_download_permitted=True,
    )
    verify(decision, label="admitted-preflight")
    assert decision["status"] == PREFLIGHT_STATUS_ADMITTED
    assert decision["download_permitted"] is True
    assert decision["claimed_download_permitted_rejected"] is False
    assert decision["all_gates_green"] is True
    assert decision["identity_complete"] is True
    assert decision["family_supported_by_runtime"] is True
    # Still not permission to fetch bodies â claim boundary stays honest.
    assert decision["claim_boundary"]["permission_to_fetch_model_bodies"] is False
    assert decision["claim_boundary"]["scaffold_not_live_acquisition"] is True


# ---------------------------------------------------------------------------
# Identity capture + runtime family support
# ---------------------------------------------------------------------------


def test_identity_fields_captured_without_invention() -> None:
    decision = evaluate_qwen_preflight(_identity_80b(), _full_evidence(family="QWEN3_NEXT"))
    identity = decision["identity"]
    assert identity["source"] == _SYNTH_SOURCE_80B
    assert identity["revision"] == _SYNTH_REVISION
    assert identity["config_digest"] == _SYNTH_DIGEST
    assert identity["license"] == "test-license"
    assert identity["architecture_identity"]["model_type"] == "test_qwen3_next"
    assert identity["architecture_identity"]["architectures"] == [
        "TestQwen3NextForCausalLM"
    ]


def test_incomplete_identity_blocks_even_if_gates_green() -> None:
    manifest = _identity_30b()
    del manifest["revision"]
    decision = evaluate_qwen_preflight(manifest, _full_evidence())
    assert decision["download_permitted"] is False
    assert decision["identity_complete"] is False
    assert any("revision" in r for r in decision["reasons"])


def test_non_hex_revision_and_digest_rejected() -> None:
    decision = evaluate_qwen_preflight(
        _identity_30b(revision="main", config_digest="not-a-hash"),
        _full_evidence(),
    )
    assert decision["download_permitted"] is False
    assert any("40-character" in r for r in decision["reasons"])
    assert any("config_digest" in r for r in decision["reasons"])


def test_family_not_in_runtime_capability_blocks() -> None:
    evidence = _full_evidence(family="DEEPSEEK_V4")  # wrong family list
    decision = evaluate_qwen_preflight(_identity_30b(), evidence)
    assert decision["download_permitted"] is False
    assert decision["family_supported_by_runtime"] is False
    assert any("not explicitly listed" in r for r in decision["reasons"])


def test_missing_runtime_capability_blocks() -> None:
    decision = evaluate_qwen_preflight(
        _identity_30b(),
        {"gates": _all_gates_green()},
    )
    assert decision["download_permitted"] is False
    assert decision["runtime_capability"]["present"] is False
    assert any("runtime-capability" in r for r in decision["reasons"])


def test_structured_reasons_are_list_of_strings() -> None:
    decision = evaluate_qwen_preflight(None, None)
    assert isinstance(decision["reasons"], list)
    assert all(isinstance(r, str) and r for r in decision["reasons"])
    assert decision["gates"]
    for name in REQUIRED_GATES:
        assert name in decision["gates"]
        assert decision["gates"][name]["green"] is False
