"""Contract tests for the storage-first, breadth-first TG ladder."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import tg_ladder as ladder  # noqa: E402


@pytest.fixture()
def plan() -> dict:
    value = ladder.load_json(ladder.default_plan_path())
    ladder.validate_plan(value)
    return value


def test_plan_is_a_complete_breadth_rotation(plan: dict):
    rungs = plan["rungs"]
    assert [row["rung"] for row in rungs] == list(ladder.RUNG_ORDER)
    assert [row["family"] for row in rungs] == list(ladder.FAMILY_ROTATION)
    assert set(row["family"] for row in rungs) == {"qwen", "mistral", "llama", "deepseek"}
    assert max(row["source_budget_gib"] for row in rungs) <= plan["storage_contract"]["max_active_parent_gib"]
    assert max(row["gravity_artifact_budget_gib"] for row in rungs) <= plan["storage_contract"]["max_retained_gravity_artifact_gib"]


def test_current_deepseek_adapter_cannot_receive_a_tg4_claim(plan: dict):
    state = ladder.empty_state()
    state["rungs"]["TG4"] = {
        "claim": "MEASURED",
        "source_materialized": True,
        "gates": {
            "SOURCE_SEALED": "PASS",
            "RAW_PARENT_FORWARD": "FAIL",
            "STATE_ADVANCE": "FAIL",
            "ADAPTER_ACCOUNTING": "PASS",
            "AUTOTUNE_PROFILE": "PASS",
            "GRAVITY_ADAPTER": "FAIL",
            "GRAVITY_ORACLE": "FAIL",
            "RUNG_MEASUREMENT": "PASS",
            "EVICTION_RECEIPT": "FAIL",
        },
    }
    with pytest.raises(ladder.LadderError, match="claims MEASURED"):
        ladder.validate_state(plan, state)


def test_state_allows_only_one_materialized_parent(plan: dict):
    state = ladder.empty_state()
    state["rungs"]["TG20"] = {"source_materialized": True, "gates": {"SOURCE_SEALED": "PASS"}}
    state["rungs"]["TG10"] = {"source_materialized": True, "gates": {"SOURCE_SEALED": "PASS"}}
    with pytest.raises(ladder.LadderError, match="one materialized parent"):
        ladder.validate_state(plan, state)


def test_next_actions_prioritize_adapter_and_gravity_gates_not_kernel_work(plan: dict):
    state = ladder.empty_state()
    state["rungs"]["TG20"] = {
        "source_materialized": True,
        "gates": {
            "SOURCE_SEALED": "PASS",
            "RAW_PARENT_FORWARD": "PASS",
            "STATE_ADVANCE": "PASS",
            "ADAPTER_ACCOUNTING": "PASS",
            "AUTOTUNE_PROFILE": "PASS",
        },
    }
    actions = ladder.next_actions(plan, state)
    assert actions[0].rung == "TG20"
    assert actions[0].gate == "GRAVITY_ADAPTER"
    assert "header, tensor map" in actions[0].action


def test_correctness_failure_is_sealed_and_evicted_instead_of_becoming_kernel_work(plan: dict):
    state = ladder.empty_state()
    state["rungs"]["TG4"] = {
        "source_materialized": True,
        "gates": {"SOURCE_SEALED": "PASS", "RAW_PARENT_FORWARD": "FAIL"},
    }
    actions = ladder.next_actions(plan, state)
    deepseek = next(action for action in actions if action.rung == "TG4")
    assert deepseek.gate == "EVICTION_RECEIPT"
    assert "RAW_PARENT_FORWARD=FAIL" in deepseek.action


def test_plan_rejects_non_breadth_rotation(plan: dict):
    invalid = copy.deepcopy(plan)
    invalid["rungs"][1]["family"] = "qwen"
    with pytest.raises(ladder.LadderError, match="breadth rotation"):
        ladder.validate_plan(invalid)


def test_source_bytes_cannot_overflow_its_stage_budget(plan: dict):
    state = ladder.empty_state()
    state["rungs"]["TG20"] = {
        "source_materialized": True,
        "source_bytes": 11 * ladder.GIB,
        "gates": {"SOURCE_SEALED": "PASS"},
    }
    with pytest.raises(ladder.LadderError, match="exceeds its 10 GiB storage budget"):
        ladder.validate_state(plan, state)


def test_unified_lane_admits_a_72b_q4_with_full_scratch_reserve():
    # Official Qwen2.5-72B Q4_K_M split payload; no second reserve is added.
    result = ladder.unified_lane_admission(
        free_bytes=418_015_854_592,
        model_bytes=44_010_465_472,
        scratch_bytes=25 * ladder.GIB,
    )
    assert result["status"] == "ADMITTED"
    assert result["residual_above_floor_bytes"] == 218_009_965_036
    assert result["incremental_lane_bytes"] <= ladder.MODEL_LANE_CEILING_BYTES


def test_unified_lane_refuses_second_model_and_over_envelope_total():
    occupied = ladder.unified_lane_admission(
        free_bytes=418_015_854_592,
        model_bytes=44_010_465_472,
        scratch_bytes=15 * ladder.GIB,
        active_model_count=1,
    )
    assert occupied["status"] == "DENIED"
    assert "an active model/artifact is already present" in occupied["failures"]

    over_envelope = ladder.unified_lane_admission(
        free_bytes=418_015_854_592,
        model_bytes=82_000_000_000,
        scratch_bytes=25 * ladder.GIB,
    )
    assert over_envelope["status"] == "DENIED"
    assert "model plus scratch/metadata exceeds the 102-GB decimal lane ceiling" in over_envelope["failures"]


def test_executable_working_set_rejects_the_observed_qwen72_whole_source_path():
    # The direct Qwen-72B Q4_K_M run reached swap before first dispatch.  A
    # retry cannot turn that source into an executable resident body merely
    # because disk admission succeeds: it must fit a separately measured
    # runtime budget or be changed into a dependency-complete window plan.
    result = ladder.executable_working_set_admission(
        source_bytes=47_415_715_488,
        execution_mode="direct_full_source",
        resident_weight_bytes=47_415_715_488,
        resident_kv_bytes=1_073_741_824,
        resident_activation_bytes=1_073_741_824,
        runtime_scratch_bytes=2_147_483_648,
        resident_budget_bytes=32_000_000_000,
    )
    assert result["status"] == "DENIED"
    assert "charged executable working set exceeds the freshly measured resident budget" in result["failures"]


def test_executable_working_set_admits_only_a_dependency_complete_bounded_window():
    result = ladder.executable_working_set_admission(
        source_bytes=47_415_715_488,
        execution_mode="dependency_complete_window",
        resident_weight_bytes=8_000_000_000,
        resident_kv_bytes=1_073_741_824,
        resident_activation_bytes=1_073_741_824,
        runtime_scratch_bytes=2_147_483_648,
        bounded_next_range_bytes=67_108_864,
        resident_budget_bytes=32_000_000_000,
        dependency_complete_window=True,
    )
    assert result["status"] == "ADMITTED"
    assert result["required_resident_bytes"] < result["resident_budget_bytes"]

    incomplete = ladder.executable_working_set_admission(
        source_bytes=47_415_715_488,
        execution_mode="dependency_complete_window",
        resident_weight_bytes=8_000_000_000,
        resident_kv_bytes=0,
        resident_activation_bytes=0,
        runtime_scratch_bytes=0,
        resident_budget_bytes=32_000_000_000,
    )
    assert incomplete["status"] == "DENIED"
    assert "windowed execution requires a dependency-complete current window" in incomplete["failures"]


def test_split_gguf_streaming_merge_fits_where_a_retained_double_copy_cannot():
    # Qwen2.5-72B-Instruct's official Q4_K_M set: twelve independent GGUF
    # containers.  It may only be transformed by a receipt-gated streamer
    # that evicts each durable source part, never by retaining source+target.
    parts = [
        3_986_869_120,
        3_877_864_544,
        3_963_622_720,
        3_941_463_040,
        3_963_688_384,
        3_941_463_040,
        3_826_849_152,
        3_983_901_792,
        3_921_183_968,
        3_995_005_888,
        3_586_683_584,
        1_021_870_240,
    ]
    source = sum(parts)
    largest_part = max(parts)
    result = ladder.split_gguf_merge_admission(
        free_bytes=418_015_854_592,
        part_bytes=parts,
        merged_bytes=source,
        scratch_bytes=25 * ladder.GIB,
    )
    assert result["status"] == "ADMITTED"
    assert result["streaming_peak_model_bytes"] == source + largest_part
    assert result["streaming"]["status"] == "ADMITTED"
    assert result["traditional_copy"]["status"] == "DENIED"
    assert "model artifact exceeds the normal 82-GB decimal cap" in result["traditional_copy"]["failures"]
