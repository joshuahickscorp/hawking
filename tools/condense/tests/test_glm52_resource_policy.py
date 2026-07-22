#!/usr/bin/env python3.12
from __future__ import annotations

import copy
import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
REPO = CONDENSE.parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_resource_policy as resource  # noqa: E402
from glm52_common import read_sealed_json, seal  # noqa: E402


CREATED_AT = "2026-07-21T22:07:16Z"


def test_derives_exact_conservative_official_policy() -> None:
    value = resource.derive_policy(REPO, created_at=CREATED_AT)
    restored = resource.validate_policy(value, root=REPO)
    assert restored["policy"] == {
        "emergency_floor_bytes": 5_368_709_120,
        "largest_atomic_source_write_bytes": 5_368_361_544,
        "largest_compact_shard_write_bytes": 92_282_917_708,
        "next_checkpoint_write_bytes": 1_073_741_824,
        "xet_reconstruction_scratch_bytes": 10_734_792_512,
        "two_largest_official_source_shards_bytes": 10_734_792_512,
        "projected_remaining_compact_bytes": 209_990_720_908,
        "projected_teacher_evidence_bytes": 21_474_836_480,
        "active_scratch_bytes": 0,
        "current_best_artifact_bytes": 92_282_917_708,
        "rollback_capsule_bytes": 5_001_815,
        "minimum_available_ram_bytes": 17_179_869_184,
        "maximum_swap_used_bytes": 8_589_934_592,
    }
    assert restored["derived"]["required_free_disk_bytes"] == 416_036_394_619
    assert restored["derived"]["preregistered_usable_raw_window_bytes"] == 225_426_172_293
    assert restored["derived"]["maximum_scheduled_resident_window_bytes"] == 134_336_513_960
    assert restored["derived"]["required_window_plus_prefetch_headroom_bytes"] == \
        236_190_533_120
    assert restored["derived"]["full_window_plus_prefetch_headroom_deficit_bytes"] == \
        10_764_360_827
    prefetch = restored["prefetch_control"]
    assert prefetch["mode"] == resource.SERIALIZED_OR_PARTIAL_PREFETCH
    assert prefetch["full_two_complete_window_pipeline_preregistered"] is False
    assert prefetch["largest_adjacent_transition"] == ["W017", "W018"]
    assert prefetch[
        "largest_adjacent_active_plus_prefetch_union_remote_logical_bytes"
    ] == 236_190_533_120
    assert prefetch["largest_adjacent_full_prefetch_deficit_bytes"] == 10_764_360_827
    assert prefetch["maximum_materialized_raw_allocated_bytes"] == 225_426_172_293
    assert prefetch["maximum_additional_prefetch_allocated_bytes_formula"] == (
        "max(0, min(maximum_materialized_raw_allocated_bytes "
        "- live_materialized_raw_allocated_bytes_before_prefetch, "
        "live_disk_free_bytes - required_free_disk_bytes))"
    )
    assert prefetch[
        "live_allocated_byte_measurement_required_before_materialized_xet_body_acquisition"
    ] is True
    assert prefetch["remote_logical_bytes_are_allocated_upper_bounds"] is False
    assert prefetch["transition_count"] == 19
    serialized = [
        (item["active_window_id"], item["prefetch_window_id"])
        for item in prefetch["transitions"]
        if item["preregistered_mode"] == resource.SERIALIZED_OR_PARTIAL_PREFETCH
    ]
    assert serialized == [("W014", "W015"), ("W015", "W016"), ("W017", "W018")]
    largest_transition = prefetch["transitions"][17]
    assert largest_transition[
        "preregistered_additional_prefetch_logical_proxy_ceiling_bytes"
    ] == 107_186_793_789
    assert largest_transition["requested_prefetch_remote_logical_bytes"] == \
        117_951_154_616
    provisional = restored["provisional_control_limits"]
    assert provisional == {
        "status": resource.PROVISIONAL_CONTROL_LIMIT_STATUS,
        "minimum_available_ram_bytes": 17_179_869_184,
        "maximum_swap_used_bytes": 8_589_934_592,
        "next_checkpoint_write_bytes": 1_073_741_824,
        "derived_from_sealed_input_evidence": False,
        "live_measurement_required": True,
    }
    assert restored["activation"]["eviction_must_remain_disabled_until_all_are_true"] is True
    assert restored["activation"][
        "remote_logical_bytes_authorize_materialized_body_acquisition"
    ] is False


def test_derivation_is_deterministic_and_seal_detects_tamper() -> None:
    first = resource.derive_policy(REPO, created_at=CREATED_AT)
    second = resource.derive_policy(REPO, created_at=CREATED_AT)
    assert first == second
    changed = copy.deepcopy(first)
    changed["policy"]["emergency_floor_bytes"] -= 1
    with pytest.raises(resource.ResourcePolicyError, match="seal"):
        resource.validate_policy(changed, root=REPO)


def test_validate_rebuilds_and_rejects_resealed_substitution() -> None:
    original = resource.derive_policy(REPO, created_at=CREATED_AT)
    zeroed = copy.deepcopy(original)
    zeroed.pop("seal_sha256")
    for key in zeroed["policy"]:
        zeroed["policy"][key] = 0
    zeroed["derived"]["operational_reserve_floor_bytes"] = 0
    zeroed["derived"]["additional_reserved_bytes"] = 0
    zeroed["derived"]["required_free_disk_bytes"] = 0
    with pytest.raises(resource.ResourcePolicyError, match="deterministic derivation"):
        resource.validate_policy(seal(zeroed), root=REPO)

    retimestamped = copy.deepcopy(original)
    retimestamped.pop("seal_sha256")
    retimestamped["created_at"] = "2026-07-21T22:07:17Z"
    with pytest.raises(resource.ResourcePolicyError, match="frozen policy timestamp"):
        resource.validate_policy(seal(retimestamped), root=REPO)


@pytest.mark.parametrize("section", ("derived", "prefetch_control", "activation", "notes"))
def test_validate_rejects_every_resealed_nested_shape_or_value_drift(
    section: str,
) -> None:
    changed = copy.deepcopy(resource.derive_policy(REPO, created_at=CREATED_AT))
    changed.pop("seal_sha256")
    if section == "derived":
        changed[section]["preregistered_free_disk_bytes"] = 0
    elif section == "prefetch_control":
        changed[section]["mode"] = "FULL_PREFETCH"
    elif section == "activation":
        changed[section] = {}
    else:
        changed[section] = []
    with pytest.raises(resource.ResourcePolicyError, match="deterministic derivation"):
        resource.validate_policy(seal(changed), root=REPO)


def test_committed_policy_is_the_exact_deterministic_build() -> None:
    committed = read_sealed_json(REPO / "GLM52_RESOURCE_RESERVE_POLICY.json")
    expected = resource.derive_policy(REPO, created_at=CREATED_AT)
    assert committed == expected
    assert resource.validate_policy(committed, root=REPO) == expected


@pytest.mark.parametrize(
    "created_at",
    (
        "",
        "now",
        "2026-07-21T22:07:16+00:00",
        "2026-99-99T99:99:99Z",
        "2026-07-21T22:07:17Z",
    ),
)
def test_created_at_is_explicit_real_utc(created_at: str) -> None:
    with pytest.raises(resource.ResourcePolicyError, match="created_at"):
        resource.derive_policy(REPO, created_at=created_at)
