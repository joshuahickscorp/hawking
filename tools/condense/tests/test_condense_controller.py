"""Focused contract tests for the offline N-1/N/N+1 rotation controller."""
from __future__ import annotations

import pytest

from lab.operators.condense_controller import CondenseController, CondenseTask, ControllerError


LEASE = "clean-heavy-gpu-lease"
SEAL_A = "a" * 64


def _tasks() -> list[CondenseTask]:
    return [
        CondenseTask("qwen", source_bytes=100, metadata_bytes=5, artifact_bytes={"gravity": 40, "condense": 20}),
        CondenseTask("llama", source_bytes=90, metadata_bytes=7, artifact_bytes={"gravity": 30, "condense": 10}),
    ]


def _ready_first(controller: CondenseController) -> None:
    assert controller.claim_metadata("fetcher-a") == "qwen"
    controller.complete_metadata("qwen", "fetcher-a")


def test_rotates_n_minus_one_n_and_n_plus_one_under_one_budget() -> None:
    controller = CondenseController(_tasks(), byte_budget_bytes=172, heavy_lease_token=LEASE)
    _ready_first(controller)
    controller.begin_processing("qwen", LEASE)

    assert controller.claim_metadata("fetcher-a") == "llama"
    controller.steal_metadata("llama", "fetcher-b")
    controller.complete_metadata("llama", "fetcher-b")
    controller.record_profile_sample("qwen", rate_id="1bpw", profile_id="m3ultra", metrics={"fit_s": 3.0, "pack_s": 2.0})
    controller.finish_one_pass("qwen", LEASE, {"gravity": 39, "condense": 19})
    controller.seal_and_evict("qwen", LEASE, SEAL_A)

    controller.begin_processing("llama", LEASE)
    controller.finish_one_pass("llama", LEASE, {"gravity": 30, "condense": 10})
    controller.seal_and_evict("llama", LEASE, "b" * 64)

    snapshot = controller.snapshot()
    assert snapshot["resident_bytes"] == 0
    assert snapshot["active_task_id"] is None
    assert [task["phase"] for task in snapshot["tasks"]] == ["EVICTED", "EVICTED"]
    assert snapshot["shared_rate_profile_stats"]["1bpw/m3ultra"]["means"] == {"fit_s": 3.0, "pack_s": 2.0}
    assert [event["kind"] for event in snapshot["events"]] == [
        "metadata_claimed", "metadata_ready", "processing_started", "metadata_claimed",
        "metadata_stolen", "metadata_ready", "one_pass_multi_artifact_packed",
        "sealed_and_evicted", "processing_started", "one_pass_multi_artifact_packed", "sealed_and_evicted",
    ]


def test_capacity_is_refused_before_a_second_source_or_artifact_can_exist() -> None:
    controller = CondenseController(_tasks(), byte_budget_bytes=164, heavy_lease_token=LEASE)
    _ready_first(controller)
    with pytest.raises(ControllerError, match="capacity violation"):
        controller.begin_processing("qwen", LEASE)
    snapshot = controller.snapshot()
    assert snapshot["resident_bytes"] == 5
    assert snapshot["active_task_id"] is None
    assert snapshot["tasks"][0]["phase"] == "METADATA_READY"


def test_heavy_lease_and_rotation_order_fail_closed() -> None:
    controller = CondenseController(_tasks(), byte_budget_bytes=300, heavy_lease_token=LEASE)
    _ready_first(controller)
    with pytest.raises(ControllerError, match="expected lease token"):
        controller.begin_processing("qwen", "some-other-lease")
    controller.begin_processing("qwen", LEASE)
    assert controller.claim_metadata("fetcher") == "llama"
    controller.complete_metadata("llama", "fetcher")
    with pytest.raises(ControllerError, match="another task is already processing"):
        controller.begin_processing("llama", LEASE)
    controller.finish_one_pass("qwen", LEASE, {"gravity": 40, "condense": 20})
    with pytest.raises(ControllerError, match="another task is already processing"):
        controller.begin_processing("llama", LEASE)
    controller.seal_and_evict("qwen", LEASE, SEAL_A)


def test_one_pass_and_artifact_budget_are_exact() -> None:
    controller = CondenseController(_tasks(), byte_budget_bytes=300, heavy_lease_token=LEASE)
    _ready_first(controller)
    controller.begin_processing("qwen", LEASE)
    with pytest.raises(ControllerError, match="exactly match"):
        controller.finish_one_pass("qwen", LEASE, {"gravity": 40})
    with pytest.raises(ControllerError, match="exceed"):
        controller.finish_one_pass("qwen", LEASE, {"gravity": 41, "condense": 20})
    controller.finish_one_pass("qwen", LEASE, {"gravity": 40, "condense": 20})
    with pytest.raises(ControllerError, match="exactly once"):
        controller.finish_one_pass("qwen", LEASE, {"gravity": 40, "condense": 20})
    controller.seal_and_evict("qwen", LEASE, SEAL_A)


def test_only_the_admissible_n_plus_one_job_can_be_stolen() -> None:
    controller = CondenseController(_tasks(), byte_budget_bytes=300, heavy_lease_token=LEASE)
    _ready_first(controller)
    with pytest.raises(ControllerError, match="current unfinished metadata"):
        controller.steal_metadata("llama", "worker-b")
    controller.begin_processing("qwen", LEASE)
    assert controller.claim_metadata("worker-a") == "llama"
    controller.steal_metadata("llama", "worker-b")
    with pytest.raises(ControllerError, match="only the worker"):
        controller.complete_metadata("llama", "worker-a")
    controller.complete_metadata("llama", "worker-b")
    controller.finish_one_pass("qwen", LEASE, {"gravity": 40, "condense": 20})
    controller.seal_and_evict("qwen", LEASE, SEAL_A)


def test_scratch_is_part_of_the_incremental_envelope() -> None:
    task = CondenseTask(
        "qwen",
        source_bytes=100,
        metadata_bytes=5,
        artifact_bytes={"gravity": 40},
        scratch_bytes=20,
    )
    controller = CondenseController([task], byte_budget_bytes=164, heavy_lease_token=LEASE)
    _ready_first(controller)
    with pytest.raises(ControllerError, match="capacity violation"):
        controller.begin_processing("qwen", LEASE)


def test_process_wide_heavy_slot_refuses_a_second_controller_until_seal_and_evict() -> None:
    first = CondenseController(_tasks(), byte_budget_bytes=300, heavy_lease_token="first-lease")
    second = CondenseController(_tasks(), byte_budget_bytes=300, heavy_lease_token="second-lease")
    _ready_first(first)
    _ready_first(second)

    first.begin_processing("qwen", "first-lease")
    with pytest.raises(ControllerError, match="already active in another controller"):
        second.begin_processing("qwen", "second-lease")

    first.finish_one_pass("qwen", "first-lease", {"gravity": 40, "condense": 20})
    first.seal_and_evict("qwen", "first-lease", SEAL_A)
    second.begin_processing("qwen", "second-lease")
    second.finish_one_pass("qwen", "second-lease", {"gravity": 40, "condense": 20})
    second.seal_and_evict("qwen", "second-lease", "b" * 64)
