from __future__ import annotations

import json

import pytest

from lab.operators.glm52_common import verify_sealed
from lab.operators.glm52_restream_contract import build_contract, live_window_admission
from ramanujan.restream_guard import validate_bounded_restream


@pytest.fixture(scope="module")
def contract() -> tuple[dict[str, object], dict[str, object]]:
    return build_contract(
        manifest_path="evidence/glm52/GLM52_OFFICIAL_MANIFEST.json",
        graph_path="evidence/glm52/GLM52_SHARD_DEPENDENCY_GRAPH.json",
    )


def test_real_parent_range_contract_is_sealed_exact_and_below_90gb(contract) -> None:
    schedule, policy = contract
    verify_sealed(schedule)
    verify_sealed(policy)
    result = validate_bounded_restream(schedule, policy)
    assert result == {"peak_incremental_bytes": 58_885_799_936, "window_count": 81}
    assert schedule["source_accounting"] == {
        "logical_payload_bytes": 1_506_659_919_872,
        "rounded_payload_bytes": 1_506_686_271_488,
        "range_count": 59_585,
        "window_count": 81,
        "exact_once": True,
    }
    assert schedule["authoritative"] is False
    assert schedule["live_execution_authorized"] is False


def test_each_next_window_requires_sealed_predecessor_eviction(contract) -> None:
    schedule, _policy = contract
    windows = schedule["windows"]
    assert windows[0]["predecessor_gate"] == "GENESIS"
    for previous, current in zip(windows, windows[1:]):
        assert current["predecessor_window_id"] == previous["window_id"]
        assert current["predecessor_gate"] == "SEALED_ARTIFACT_HASH_AND_EXACT_SOURCE_ARTIFACT_TEMP_EVICTION_RECEIPT"
    assert schedule["lifecycle"]["artifact_retention"] == "CURRENT_WINDOW_ONLY"


def test_live_admission_uses_fresh_free_bytes_and_one_unified_floor(contract) -> None:
    schedule, policy = contract
    peak_window = max(schedule["windows"], key=lambda row: row["incremental_accounting"]["resident_incremental_bytes"])
    required = peak_window["incremental_accounting"]["resident_incremental_bytes"]
    floor = policy["policy"]["protected_filesystem_floor_bytes"]
    assert live_window_admission(schedule, policy, window_id=peak_window["window_id"], free_bytes=floor + required)["admitted"] is True
    assert live_window_admission(schedule, policy, window_id=peak_window["window_id"], free_bytes=floor + required - 1)["admitted"] is False


def test_schedule_has_no_body_bytes_or_credential_material(contract) -> None:
    rendered = json.dumps(contract)
    assert "accessToken" not in rendered
    assert "Authorization" not in rendered
    assert "live_execution_authorized\": true" not in rendered
