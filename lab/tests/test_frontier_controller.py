from __future__ import annotations

import json

import pytest

from lab.operators.frontier_controller import PlanningRefusal, plan_wave, write_wave


def state(**changes):
    value = {
        "model_id": "Mixtral-8x7B-Instruct-v0.1-Q4_K_M",
        "model_selection_frozen": True,
        "clean_lease": True,
        "storage": {"authority": "SUBSTRATE_SANDBOX_DYNAMIC_STORAGE_PLAN", "available_model_lane_bytes": 12_000},
        "roofline": {"bytes": 0.8, "operations": 0.6, "depth": 0.9, "synchronization": 1.0, "state": 0.2},
        "sealed_negatives": {"container_only": {"reopen_condition_met": False}},
    }
    value.update(changes)
    return value


def candidate(id, families, reduces, **extra):
    return {"id": id, "families": families, "reduces": reduces, "artifact_bytes": 100, "temporary_bytes": 100, "falsifier": f"{id} complete-token p99 does not improve", **extra}


def test_combines_distinct_methods_against_the_measured_bottlenecks():
    wave = plan_wave(
        state(),
        [
            candidate("format_native", ["gravity_format"], {"bytes": 0.3}),
            candidate("persistent_graph", ["command_topology"], {"depth": 0.3, "synchronization": 0.5}),
            candidate("kv_codec", ["state_codec"], {"state": 0.9}),
        ],
        max_candidates=2,
    )
    assert [row["id"] for row in wave["selected"]] == ["format_native", "persistent_graph"]
    assert wave["combined_predicted_reduction"]["synchronization"] == 0.5
    assert wave["status"] == "PREREGISTRATION_REQUIRED_NOT_EVIDENCE"
    assert len(wave["plan_sha256"]) == 64


def test_sealed_negative_cannot_be_retried_without_its_specific_reopen_condition():
    blocked = candidate("container_retry", ["gravity_format"], {"bytes": 0.2}, blocked_by=["container_only"])
    wave = plan_wave(state(), [blocked, candidate("kernel", ["kernel"], {"operations": 0.2})])
    assert wave["selected"][0]["id"] == "kernel"
    assert wave["rejected"] == [{"id": "container_retry", "reasons": ["sealed negative 'container_only' has no satisfied reopen condition"]}]

    reopened = plan_wave(
        state(sealed_negatives={"container_only": {"reopen_condition_met": True}}),
        [candidate("container_retry", ["gravity_format"], {"bytes": 0.2}, blocked_by=["container_only"], reopens=["container_only"])],
    )
    assert reopened["selected"][0]["id"] == "container_retry"


def test_authoritative_model_lane_rejects_an_oversized_combination():
    too_large = candidate("student", ["distillation"], {"operations": 0.5}, artifact_bytes=11_000, temporary_bytes=1_001)
    with pytest.raises(PlanningRefusal, match="no candidate is admissible"):
        plan_wave(state(), [too_large])


@pytest.mark.parametrize("changes, message", [
    ({"model_selection_frozen": False}, "selection is not frozen"),
    ({"clean_lease": False}, "clean lease"),
    ({"roofline": {}}, "no measured remaining"),
])
def test_nonnegotiable_fences_fail_closed(changes, message):
    with pytest.raises(PlanningRefusal, match=message):
        plan_wave(state(**changes), [candidate("kernel", ["kernel"], {"operations": 0.2})])


def test_incompatible_methods_are_not_combined_and_written_plan_is_stable(tmp_path):
    wave = plan_wave(
        state(),
        [
            candidate("a", ["a"], {"synchronization": 0.8}, incompatible_with=["b"]),
            candidate("b", ["b"], {"depth": 0.8}),
        ],
        max_candidates=2,
    )
    assert len(wave["selected"]) == 1
    output = write_wave(tmp_path / "HAWKING_DYNAMIC_WAVE_PREREGISTRATION.json", wave)
    written = json.loads(output.read_text())
    assert written["plan_sha256"] == wave["plan_sha256"]


def test_shared_load_mode_allows_only_a_paired_differential_claim():
    shared = state(
        clean_lease=False,
        measurement_mode="shared_load_paired",
        shared_load_invariant=True,
        paired_control_runs=3,
        absolute_tps_claim=False,
    )
    wave = plan_wave(shared, [candidate("kernel", ["kernel"], {"operations": 0.2})])
    assert wave["measurement"]["absolute_tps_eligible"] is False
    assert "differential result" in wave["promotion_rule"]


@pytest.mark.parametrize("changes, message", [
    ({"shared_load_invariant": False}, "same-contemporaneous-load invariant"),
    ({"paired_control_runs": 1}, "at least two"),
    ({"absolute_tps_claim": True}, "forbid an absolute"),
])
def test_shared_load_mode_fails_closed_when_pairing_is_not_bound(changes, message):
    shared = state(
        clean_lease=False,
        measurement_mode="shared_load_paired",
        shared_load_invariant=True,
        paired_control_runs=2,
        absolute_tps_claim=False,
    )
    shared.update(changes)
    with pytest.raises(PlanningRefusal, match=message):
        plan_wave(shared, [candidate("kernel", ["kernel"], {"operations": 0.2})])
