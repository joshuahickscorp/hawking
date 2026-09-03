"""Adversarial tests for CO_RESIDENT_TEACHER vs STREAMED_TEACHER mode split.

Proves the two execution-mode contracts evaluate independently: a blocked
co-resident snapshot must not poison a healthy streamed assessment, and the
streamed gate refuses on its own physical requirements.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_guarded_streamed_source_oracle_outer_controller as controller
from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_contract as retention
from lab.operators import ascension_qwen30_quality_repack_source_bf16_memory_lease_preflight as co_resident
from lab.operators import ascension_qwen30_streamed_teacher_resource_preflight as streamed
from lab.receipts import seal


SOURCE_BYTES = 61_066_575_656  # 56.9 GiB sealed source weight lower bound
CO_RESIDENT_REQUIRED = SOURCE_BYTES + 8 * 1024**3  # 64.9 GiB
MEASURED_BLOCKED_RECLAIMABLE = 45_615_546_368  # 42.5 GiB live measurement
PIN_A = "a" * 64
PIN_B = "b" * 64
EXECUTOR_PIN = streamed.EXPECTED_EXECUTOR_SHA256
GIB = 1024**3


def _snapshot(
    *,
    reclaimable: int,
    swap: int = 0,
    swapouts: int = 0,
    physical: int = 96 * GIB,
) -> dict[str, object]:
    return {
        "physical_memory_bytes": physical,
        "vm_stat": {
            "reclaimable_bytes": reclaimable,
            "swapouts_pages": swapouts,
        },
        "swap": {"used_bytes": swap},
    }


def _healthy_streamed_kwargs() -> dict[str, object]:
    return {
        "source_weight_bytes": SOURCE_BYTES,
        "expected_source_pin_sha256": PIN_A,
        "observed_source_pin_sha256": PIN_A,
        "expected_executor_sha256": EXECUTOR_PIN,
        "observed_executor_sha256": EXECUTOR_PIN,
        "bounded_stream_window_bytes": streamed.MAX_STREAM_WINDOW_BYTES,
        "live_stream_windows": streamed.MAX_LIVE_STREAM_WINDOWS,
    }


# ---------------------------------------------------------------------------
# CO_RESIDENT_TEACHER — exact arithmetic unchanged
# ---------------------------------------------------------------------------


def test_co_resident_blocks_at_measured_42_5_gib_against_64_9_gib_required() -> None:
    result = co_resident.assess_headroom(
        _snapshot(reclaimable=MEASURED_BLOCKED_RECLAIMABLE),
        source_weight_bytes=SOURCE_BYTES,
    )
    assert result["status"] == co_resident.BLOCKED_STATUS
    assert result["minimum_reclaimable_bytes_required_before_source_load"] == CO_RESIDENT_REQUIRED
    assert result["measured_reclaimable_bytes"] == MEASURED_BLOCKED_RECLAIMABLE
    assert result["measured_reclaimable_deficit_bytes"] == CO_RESIDENT_REQUIRED - MEASURED_BLOCKED_RECLAIMABLE
    assert result["lease_granted"] is False


def test_co_resident_passes_only_when_full_requirement_met() -> None:
    blocked = co_resident.assess_headroom(
        _snapshot(reclaimable=CO_RESIDENT_REQUIRED - 1),
        source_weight_bytes=SOURCE_BYTES,
    )
    assert blocked["status"] == co_resident.BLOCKED_STATUS

    ready = co_resident.assess_headroom(
        _snapshot(reclaimable=CO_RESIDENT_REQUIRED),
        source_weight_bytes=SOURCE_BYTES,
    )
    assert ready["status"] == co_resident.READY_STATUS
    assert ready["lease_granted"] is False
    assert ready["measured_reclaimable_deficit_bytes"] == 0


def test_co_resident_still_blocks_on_any_swap() -> None:
    result = co_resident.assess_headroom(
        _snapshot(reclaimable=CO_RESIDENT_REQUIRED + GIB, swap=1),
        source_weight_bytes=SOURCE_BYTES,
    )
    assert result["status"] == co_resident.BLOCKED_STATUS


# ---------------------------------------------------------------------------
# STREAMED_TEACHER — independent resource contract
# ---------------------------------------------------------------------------


def test_streamed_passes_at_streamed_floor_with_zero_swap() -> None:
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    assert floor == 1_295_402_056
    result = streamed.assess_streamed_resources(
        _snapshot(reclaimable=floor),
        **_healthy_streamed_kwargs(),
    )
    assert result["status"] == streamed.READY_STATUS
    assert result["verdict"] == "READY"
    assert result["lease_granted"] is False
    assert result["does_not_inherit_co_resident_memory_gate_verdict"] is True
    assert result["blockers"] == []


def test_streamed_refuses_nonzero_swap() -> None:
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    result = streamed.assess_streamed_resources(
        _snapshot(reclaimable=floor + GIB, swap=1),
        **_healthy_streamed_kwargs(),
    )
    assert result["verdict"] == "BLOCKED"
    assert "nonzero_swap_used_bytes" in result["blockers"]


def test_streamed_refuses_nonzero_swapouts() -> None:
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    result = streamed.assess_streamed_resources(
        _snapshot(reclaimable=floor + GIB, swapouts=1),
        **_healthy_streamed_kwargs(),
    )
    assert result["verdict"] == "BLOCKED"
    assert "nonzero_swapouts_pages" in result["blockers"]


def test_streamed_refuses_reclaimable_below_streamed_floor() -> None:
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    result = streamed.assess_streamed_resources(
        _snapshot(reclaimable=floor - 1),
        **_healthy_streamed_kwargs(),
    )
    assert result["verdict"] == "BLOCKED"
    assert "reclaimable_bytes_below_streamed_floor" in result["blockers"]


def test_streamed_refuses_stream_window_above_ceiling() -> None:
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    kwargs = _healthy_streamed_kwargs()
    kwargs["bounded_stream_window_bytes"] = streamed.MAX_STREAM_WINDOW_BYTES + 1
    result = streamed.assess_streamed_resources(_snapshot(reclaimable=floor + GIB), **kwargs)
    assert result["verdict"] == "BLOCKED"
    assert "stream_window_exceeds_ceiling" in result["blockers"]


def test_streamed_refuses_more_than_one_live_window() -> None:
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    kwargs = _healthy_streamed_kwargs()
    kwargs["live_stream_windows"] = 2
    result = streamed.assess_streamed_resources(_snapshot(reclaimable=floor + GIB), **kwargs)
    assert result["verdict"] == "BLOCKED"
    assert "live_stream_windows_not_exactly_one" in result["blockers"]


def test_streamed_refuses_source_pin_mismatch() -> None:
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    kwargs = _healthy_streamed_kwargs()
    kwargs["observed_source_pin_sha256"] = PIN_B
    result = streamed.assess_streamed_resources(_snapshot(reclaimable=floor + GIB), **kwargs)
    assert result["verdict"] == "BLOCKED"
    assert "source_pin_mismatch" in result["blockers"]


def test_streamed_refuses_executor_sha_mismatch() -> None:
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    kwargs = _healthy_streamed_kwargs()
    kwargs["observed_executor_sha256"] = PIN_B
    result = streamed.assess_streamed_resources(_snapshot(reclaimable=floor + GIB), **kwargs)
    assert result["verdict"] == "BLOCKED"
    assert "executor_sha_mismatch" in result["blockers"]


def test_streamed_refuses_when_non_residency_ceiling_exceeded() -> None:
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    ceiling = plan["maximum_child_rss_bytes_non_residency_ceiling"]
    assert ceiling < SOURCE_BYTES
    kwargs = _healthy_streamed_kwargs()
    kwargs["measured_or_declared_child_rss_bytes"] = ceiling + 1
    result = streamed.assess_streamed_resources(_snapshot(reclaimable=floor + GIB), **kwargs)
    assert result["verdict"] == "BLOCKED"
    assert "non_residency_rss_ceiling_exceeded" in result["blockers"]
    assert (
        result["non_residency_proof"]["field_that_catches_covert_full_model_residency"]
        == "maximum_child_rss_bytes_non_residency_ceiling"
    )


def test_full_model_rss_would_trip_non_residency_ceiling() -> None:
    """Covert full-model residency is caught by the RSS ceiling field."""
    plan = streamed.streamed_working_set(source_weight_bytes=SOURCE_BYTES)
    floor = plan["minimum_reclaimable_bytes_required_immediately_before_source_child"]
    kwargs = _healthy_streamed_kwargs()
    # A silent full BF16 load would report ~source weight RSS.
    kwargs["measured_or_declared_child_rss_bytes"] = SOURCE_BYTES
    result = streamed.assess_streamed_resources(_snapshot(reclaimable=floor + GIB), **kwargs)
    assert result["verdict"] == "BLOCKED"
    assert "non_residency_rss_ceiling_exceeded" in result["blockers"]
    assert result["maximum_child_rss_bytes_non_residency_ceiling"] < SOURCE_BYTES


# ---------------------------------------------------------------------------
# Independence: STREAMED is not derived from CO_RESIDENT status
# ---------------------------------------------------------------------------


def test_streamed_verdict_independent_of_blocked_co_resident_snapshot() -> None:
    """Construct BLOCKED co-resident + healthy streamed; streamed still READY."""
    # Measured live mismatch: 42.5 GiB reclaimable fails co-resident (needs 64.9)
    # but clears the streamed floor (~1.3 GiB) with zero swap.
    blocked_co = co_resident.assess_headroom(
        _snapshot(reclaimable=MEASURED_BLOCKED_RECLAIMABLE),
        source_weight_bytes=SOURCE_BYTES,
    )
    assert blocked_co["status"] == co_resident.BLOCKED_STATUS

    streamed_result = streamed.assess_streamed_resources(
        _snapshot(reclaimable=MEASURED_BLOCKED_RECLAIMABLE),
        **_healthy_streamed_kwargs(),
    )
    assert streamed_result["verdict"] == "READY"
    assert streamed_result["status"] == streamed.READY_STATUS
    assert streamed_result["does_not_inherit_co_resident_memory_gate_verdict"] is True

    # Same independence through the six-vector gate builder.
    memory_doc = {
        "status": co_resident.BLOCKED_STATUS,
        "headroom_assessment": blocked_co,
        "measured_system_snapshot": _snapshot(reclaimable=MEASURED_BLOCKED_RECLAIMABLE),
    }
    three_way = {"seal_sha256": PIN_A}
    gates = retention.build_execution_mode_gates(
        memory=memory_doc,
        three_way=three_way,
        source_weight_bytes=SOURCE_BYTES,
        observed_executor_sha256=EXECUTOR_PIN,
    )
    assert gates["CO_RESIDENT_TEACHER"]["verdict"] == "BLOCKED"
    assert gates["CO_RESIDENT_TEACHER"]["source_teacher_capture_is_currently_blocked"] is True
    assert gates["STREAMED_TEACHER"]["verdict"] == "READY"
    assert gates["STREAMED_TEACHER"]["streamed_teacher_capture_is_currently_blocked"] is False
    assert gates["STREAMED_TEACHER"]["does_not_inherit_co_resident_verdict"] is True
    assert gates["modes_are_independent"] is True
    assert gates["streamed_authority_must_read_STREAMED_TEACHER_not_co_resident_flag"] is True


def test_executor_pin_matches_controller_frozen_pin() -> None:
    assert streamed.EXPECTED_EXECUTOR_SHA256 == controller.CURRENT_STREAMED_TEACHER_CHILD_SHA256


def test_streamed_preflight_write_new_refuses_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "preflight.json"
    # Parent exists; first write must use absolute path.
    abs_out = out.resolve()
    document = seal({"schema": streamed.SCHEMA, "status": streamed.BLOCKED_STATUS, "probe": True})
    streamed._write_new(abs_out, document)
    with pytest.raises(streamed.StreamedTeacherResourcePreflightError, match="new absolute path"):
        streamed._write_new(abs_out, document)


def test_legacy_boolean_scoped_to_co_resident_only() -> None:
    memory_doc = {
        "status": co_resident.BLOCKED_STATUS,
        "headroom_assessment": co_resident.assess_headroom(
            _snapshot(reclaimable=MEASURED_BLOCKED_RECLAIMABLE),
            source_weight_bytes=SOURCE_BYTES,
        ),
        "measured_system_snapshot": _snapshot(reclaimable=MEASURED_BLOCKED_RECLAIMABLE),
    }
    gates = retention.build_execution_mode_gates(
        memory=memory_doc,
        three_way={"seal_sha256": PIN_A},
        source_weight_bytes=SOURCE_BYTES,
        observed_executor_sha256=EXECUTOR_PIN,
    )
    # Legacy co-resident flag may be true while STREAMED is ready.
    assert gates["CO_RESIDENT_TEACHER"]["source_teacher_capture_is_currently_blocked"] is True
    assert gates["STREAMED_TEACHER"]["streamed_teacher_capture_is_currently_blocked"] is False
    # Streamed numbers are the streamed floor, not the co-resident 64.9 GiB.
    assert gates["STREAMED_TEACHER"]["minimum_reclaimable_bytes_required"] == 1_295_402_056
    assert gates["CO_RESIDENT_TEACHER"]["minimum_reclaimable_bytes_required"] == CO_RESIDENT_REQUIRED
