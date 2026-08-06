#!/usr/bin/env python3.12
"""Focused contracts for conservative rotating-model range scheduling."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators.glm52_common import seal
from lab.operators.gravity_range_scheduler import (
    RANGE_ALIGNMENT_BYTES,
    REQUIRES_LIVE_ALLOCATION_AND_CAPABILITY,
    RangeScheduleError,
    load_glm52_ranges,
    plan_candidate,
    plan_glm52_organ_windows,
    plan_windowed_candidate,
)


def test_explicit_ranges_are_exact_once_and_charge_64k_scratch() -> None:
    candidate = plan_candidate(
        "Qwen2.5-7B",
        ranges=[
            {"range_id": "first", "shard": "model-01", "start": 0, "end": 1},
            {"range_id": "second", "shard": "model-01", "start": 1, "end": 65_537},
        ],
        artifact_bytes=1,
        metadata_bytes=2,
        scratch_multiplier=1.5,
    )

    assert candidate["status"] == REQUIRES_LIVE_ALLOCATION_AND_CAPABILITY
    assert candidate["authoritative"] is False
    assert candidate["capability_claim"] is None
    assert candidate["offline_admission"] == "ADMITTED"
    assert candidate["exact_once_range_accounting"] is True
    assert [row["range_id"] for row in candidate["ranges"]] == ["first", "second"]
    assert candidate["totals"] == {
        "range_payload_bytes": 65_537,
        "range_rounded_bytes": 2 * RANGE_ALIGNMENT_BYTES,
        "scratch_bytes": 2 * RANGE_ALIGNMENT_BYTES,
        "artifact_bytes": 1,
        "artifact_rounded_bytes": RANGE_ALIGNMENT_BYTES,
        "metadata_bytes": 2,
        "metadata_rounded_bytes": RANGE_ALIGNMENT_BYTES,
        "charged_total_bytes": 6 * RANGE_ALIGNMENT_BYTES,
        "active_total_bytes": 0,
        "combined_total_bytes": 6 * RANGE_ALIGNMENT_BYTES,
    }


@pytest.mark.parametrize(
    "ranges",
    [
        [
            {"range_id": "a", "shard": "s", "start": 0, "end": 10},
            {"range_id": "b", "shard": "s", "start": 0, "end": 10},
        ],
        [
            {"range_id": "a", "shard": "s", "start": 0, "end": 10},
            {"range_id": "b", "shard": "s", "start": 9, "end": 20},
        ],
    ],
)
def test_duplicate_or_overlapping_intervals_are_rejected(ranges: list[dict[str, object]]) -> None:
    with pytest.raises(RangeScheduleError):
        plan_candidate("Qwen2.5-7B", ranges=ranges)


def test_one_active_model_is_default_and_parallel_needs_combined_envelope() -> None:
    active = plan_candidate(
        "Qwen2.5-7B",
        ranges=[{"shard": "qwen", "start": 0, "end": 20_000_000_000}],
        artifact_bytes=1_000_000_000,
        metadata_bytes=1_000_000_000,
    )
    default_blocked = plan_candidate(
        "Llama-8B",
        ranges=[{"shard": "llama", "start": 0, "end": 1}],
        active_candidates=[active],
    )
    assert default_blocked["offline_admission"] == "BLOCKED"
    assert default_blocked["offline_block_reasons"] == ["one_active_model_default"]

    parallel = plan_candidate(
        "Llama-8B",
        ranges=[{"shard": "llama", "start": 0, "end": 50_000_000_000}],
        artifact_bytes=4_000_000_000,
        metadata_bytes=3_000_000_000,
        scratch_multiplier=1.2,
        active_candidates=[active],
        allow_parallel=True,
    )
    assert parallel["offline_admission"] == "ADMITTED"
    assert parallel["totals"]["combined_total_bytes"] <= parallel["envelope_bytes"]

    over_envelope = plan_candidate(
        "Mistral-7B",
        ranges=[{"shard": "mistral", "start": 0, "end": 52_000_000_000}],
        artifact_bytes=4_000_000_000,
        metadata_bytes=3_000_000_000,
        scratch_multiplier=1.2,
        active_candidates=[active],
        allow_parallel=True,
    )
    assert over_envelope["offline_admission"] == "BLOCKED"
    assert "combined_parallel_charge_exceeds_90gb_envelope" in over_envelope["offline_block_reasons"]
    assert over_envelope["totals"]["combined_total_bytes"] > over_envelope["envelope_bytes"]


def _write_sealed(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(seal(value), sort_keys=True), encoding="utf-8")


def test_sealed_glm52_manifest_and_graph_select_and_bound_ranges(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    graph_path = tmp_path / "graph.json"
    _write_sealed(
        manifest_path,
        {
            "schema": "hawking.glm52.official_manifest.v1",
            "repo": "zai-org/GLM-5.2",
            "revision": "abc",
            "files": [
                {"path": "model-01.safetensors", "is_weight": True, "logical_bytes": 512},
                {"path": "config.json", "is_weight": False, "logical_bytes": 20},
            ],
        },
    )
    _write_sealed(
        graph_path,
        {
            "schema": "hawking.glm52.shard_dependency_graph.v1",
            "repo": "zai-org/GLM-5.2",
            "revision": "abc",
            "tensors": [
                {
                    "name": "embed.weight",
                    "shard": "model-01.safetensors",
                    "absolute_start": 8,
                    "absolute_end": 128,
                    "payload_bytes": 120,
                },
                {
                    "name": "head.weight",
                    "shard": "model-01.safetensors",
                    "absolute_start": 128,
                    "absolute_end": 256,
                    "payload_bytes": 128,
                },
            ],
        },
    )

    ranges = load_glm52_ranges(manifest_path, graph_path, tensor_names=["head.weight"])
    assert ranges == [
        {
            "range_id": "tensor:model-01.safetensors:head.weight:128:256",
            "shard": "model-01.safetensors",
            "name": "head.weight",
            "start": 128,
            "end": 256,
            "payload_bytes": 128,
        }
    ]
    candidate = plan_candidate(
        "GLM-5.2-slice",
        manifest_path=manifest_path,
        graph_path=graph_path,
        shard_names=["model-01.safetensors"],
    )
    assert candidate["source"] == "sealed_glm52_manifest_and_graph"
    assert len(candidate["ranges"]) == 2


def test_scratch_multiplier_cannot_understate_planning_charge() -> None:
    with pytest.raises(RangeScheduleError, match="scratch_multiplier"):
        plan_candidate(
            "Qwen2.5-7B",
            ranges=[{"shard": "qwen", "start": 0, "end": 1}],
            scratch_multiplier=0.99,
        )


def test_windowed_candidate_charges_peak_and_exactly_once_source_ranges() -> None:
    candidate = plan_windowed_candidate(
        "GLM-5.2-stream",
        windows=[
            {
                "window_id": "W0",
                "ranges": [{"range_id": "a", "shard": "s", "start": 0, "end": 100}],
                "artifact_bytes": 200,
                "metadata_bytes": 300,
                "prefetch_bytes": 400,
            },
            {
                "window_id": "W1",
                "ranges": [{"range_id": "b", "shard": "s", "start": 100, "end": 300}],
                "artifact_bytes": 200,
                "metadata_bytes": 300,
                "carry_bytes": 128,
            },
        ],
    )

    assert candidate["offline_admission"] == "ADMITTED"
    assert candidate["streamed"] is True
    assert candidate["exact_once_range_accounting"] is True
    assert candidate["totals"]["range_payload_bytes"] == 300
    assert candidate["totals"]["charged_total_bytes"] == max(
        window["resident_bytes"] for window in candidate["windows"]
    )
    assert candidate["totals"]["charged_total_bytes"] < sum(
        window["resident_bytes"] for window in candidate["windows"]
    )


def test_windowed_candidate_rejects_cross_window_overlap_and_parallel_overage() -> None:
    with pytest.raises(RangeScheduleError, match="overlaps"):
        plan_windowed_candidate(
            "GLM-5.2-stream",
            windows=[
                {"window_id": "W0", "ranges": [{"shard": "s", "start": 0, "end": 100}]},
                {"window_id": "W1", "ranges": [{"shard": "s", "start": 99, "end": 200}]},
            ],
        )

    active = plan_windowed_candidate(
        "Qwen-stream",
        windows=[{"window_id": "W0", "ranges": [{"shard": "q", "start": 0, "end": 20_000_000_000}]}],
    )
    parallel = plan_windowed_candidate(
        "Llama-stream",
        windows=[{"window_id": "W0", "ranges": [{"shard": "l", "start": 0, "end": 75_000_000_000}]}],
        active_candidates=[active],
        allow_parallel=True,
    )
    assert parallel["offline_admission"] == "BLOCKED"
    assert "combined_parallel_peak_exceeds_90gb_envelope" in parallel["offline_block_reasons"]


def test_real_glm52_organ_candidate_replays_the_blocked_artifact_retention_plan() -> None:
    candidate = plan_glm52_organ_windows(
        "GLM-5.2",
        manifest_path="evidence/glm52/GLM52_OFFICIAL_MANIFEST.json",
        graph_path="evidence/glm52/GLM52_SHARD_DEPENDENCY_GRAPH.json",
        artifact_bytes_per_window=2_147_483_648,
        metadata_bytes_per_window=67_108_864,
        scratch_multiplier=1.05,
    )
    assert candidate["status"] == REQUIRES_LIVE_ALLOCATION_AND_CAPABILITY
    assert candidate["authoritative"] is False
    assert candidate["offline_admission"] == "BLOCKED"
    assert candidate["offline_block_reasons"] == ["candidate_peak_exceeds_90gb_envelope"]
    assert candidate["partition"] == {
        "algorithm": "organ_execution_order.v1",
        "window_partition_sha256": "ecff939ad471cbe324c4dc7a8ee99beea0e29082abfb22ed0ee4c38107f76772",
        "window_count": 81,
        "range_count": 59_585,
        "prefetch_contract": "next_organ_source_payload_bytes",
        "physical_range_fetch_implemented": False,
    }
    assert candidate["totals"]["peak_resident_bytes"] == 208_071_098_368
    assert candidate["totals"]["retained_artifact_rounded_bytes"] == 173_946_175_488


def test_windowed_candidate_carries_emitted_artifacts_until_a_separate_evicting_candidate() -> None:
    candidate = plan_windowed_candidate(
        "retained-artifact-reproducer",
        windows=[
            {
                "window_id": "W0",
                "ranges": [{"shard": "first", "start": 0, "end": 1}],
                "artifact_bytes": 50_000_000_000,
            },
            {
                "window_id": "W1",
                "ranges": [{"shard": "second", "start": 0, "end": 1}],
                "artifact_bytes": 50_000_000_000,
            },
        ],
    )

    assert candidate["offline_admission"] == "BLOCKED"
    assert "candidate_peak_exceeds_90gb_envelope" in candidate["offline_block_reasons"]
    assert candidate["windows"][1]["prior_retained_artifact_rounded_bytes"] >= 50_000_000_000
    assert candidate["totals"]["peak_resident_bytes"] > candidate["envelope_bytes"]
