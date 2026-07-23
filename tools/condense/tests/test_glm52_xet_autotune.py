#!/usr/bin/env python3.12
"""Adversarial offline tests for the GLM-5.2 Xet autotune planner."""
from __future__ import annotations

import copy
import json
import os
import pathlib
import socket
import sys

import pytest


CONDENSE = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = CONDENSE.parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_xet_autotune as autotune  # noqa: E402
import glm52_state as state  # noqa: E402
from glm52_common import Glm52Error, atomic_json, seal, verify_sealed  # noqa: E402


@pytest.fixture(scope="module")
def runtime_receipt() -> dict:
    return autotune.runtime_compatibility()


@pytest.fixture(scope="module")
def plan(runtime_receipt: dict) -> dict:
    return autotune.build_plan(runtime=runtime_receipt)


def test_runtime_compatibility_is_exact_and_legacy_knob_is_inert(runtime_receipt: dict) -> None:
    assert runtime_receipt["status"] == "PASS_WITH_PINNED_RUNTIME_LIMITATIONS"
    assert runtime_receipt["versions"] == autotune.PINNED_VERSIONS
    assert runtime_receipt["gates"] == {
        "pinned_versions_exact": True,
        "default_bounds_1_4_64": True,
        "legacy_range_get_variable_ignored": True,
        "fixed_download_alias_effective": True,
        "high_performance_profile_effective": True,
        "chunk_cache_size_configuration_parsed": True,
    }
    assert runtime_receipt["effective_fixed_16"]["client.ac_min_download_concurrency"] == 16
    assert runtime_receipt["effective_high_performance"]["reconstruction.download_buffer_limit"] == 64_000_000_000
    assert "inert" in runtime_receipt["semantics"]["chunk_cache"]


def test_runtime_classifier_rejects_version_or_effective_semantics(runtime_receipt: dict) -> None:
    snapshots = {
        "default": runtime_receipt["effective_default"],
        "legacy64": runtime_receipt["effective_default"],
        "fixed16": runtime_receipt["effective_fixed_16"],
        "hp": runtime_receipt["effective_high_performance"],
        "cache1g": runtime_receipt["effective_cache_1g"],
    }
    with pytest.raises(Glm52Error, match="compatibility failed"):
        autotune.classify_runtime_snapshots(
            {"huggingface_hub": "0.0.0", "hf_xet": "1.5.2"}, snapshots
        )
    broken = json.loads(json.dumps(snapshots))
    broken["fixed16"]["client.ac_max_download_concurrency"] = 64
    with pytest.raises(Glm52Error, match="compatibility failed"):
        autotune.classify_runtime_snapshots(autotune.PINNED_VERSIONS, broken)


def test_plan_is_sealed_deterministic_and_body_free(plan: dict, runtime_receipt: dict) -> None:
    verify_sealed(plan)
    rebuilt = autotune.build_plan(runtime=runtime_receipt)
    assert rebuilt == plan
    assert plan["status"] == "PASS_OFFLINE_PLAN_BODY_NOT_READ"
    assert plan["body_read_boundary"] == {
        "planner_network_access": False,
        "planner_model_body_bytes_read": 0,
        "planner_model_body_files_created": 0,
        "planner_cli_live_execution_implemented": False,
        "separate_live_executor_implemented": True,
        "separate_live_executor_path": "tools/condense/glm52_xet_live.py",
        "offline_plan_alone_authorizes_execution": False,
        "planner_live_run_default": "REFUSE",
    }
    assert plan["claims"]["source_shards_fetched"] == 0
    assert plan["claims"]["xet_autotune_complete"] is False


def test_plan_binds_exact_resource_policy_and_rejects_substitution(plan: dict) -> None:
    policy = plan["resource_reserve_policy"]
    assert policy == autotune.resource_policy_binding(
        autotune.load_and_validate_inputs()[autotune.RESOURCE_POLICY_PATH]
    )
    assert policy["seal_sha256"] == autotune.RESOURCE_POLICY_SEAL_SHA256
    assert policy["required_free_disk_bytes"] == 416_036_394_619
    assert policy["minimum_available_ram_bytes"] == 17_179_869_184
    assert policy["maximum_swap_used_bytes"] == 8_589_934_592
    assert policy["maximum_swap_growth_bytes"] == 0
    assert policy["maximum_materialized_raw_allocated_bytes"] == 225_426_172_293
    assert policy["live_allocated_byte_measurement_required"] is True
    assert any(
        item["path"] == autotune.RESOURCE_POLICY_PATH
        and item["seal_sha256"] == autotune.RESOURCE_POLICY_SEAL_SHA256
        for item in plan["inputs"]
    )

    substituted = copy.deepcopy(plan)
    substituted.pop("seal_sha256")
    substituted["resource_reserve_policy"]["required_free_disk_bytes"] -= 1
    with pytest.raises(Glm52Error, match="resource reserve policy binding"):
        autotune.verify_plan(seal(substituted), rebuild=False)


def test_committed_plan_is_rebuilt_against_current_sealed_inputs() -> None:
    committed = json.loads(
        (REPO_ROOT / "GLM52_XET_AUTOTUNE_PLAN.json").read_text(encoding="utf-8")
    )
    verified = autotune.verify_plan(committed, root=REPO_ROOT, rebuild=True)
    assert verified["seal_sha256"] == committed["seal_sha256"]


def test_plan_binds_generator_common_requirements_and_runtime_probe(plan: dict) -> None:
    binding = plan["toolchain_binding"]
    assert binding["schema"] == autotune.TOOLCHAIN_BINDING_SCHEMA
    assert {row["role"] for row in binding["files"]} == {
        "planner_generator", "shared_common", "controller_state", "live_executor",
        "requirements_lock",
    }
    for row in binding["files"]:
        path = REPO_ROOT / row["path"]
        assert row["sha256"] == autotune.sha256_file(path)
    program_hash = autotune.hashlib.sha256(
        autotune._runtime_probe_program().encode("utf-8")
    ).hexdigest()
    assert binding["runtime_probe_program_sha256"] == program_hash


def test_planning_surface_contains_no_download_primitive() -> None:
    source = (CONDENSE / "glm52_xet_autotune.py").read_text(encoding="utf-8")
    forbidden = (
        "hf_hub_download",
        "snapshot_download",
        ".download_stream(",
        ".start_download_file(",
        "requests.get(",
        "httpx.get(",
    )
    assert all(token not in source for token in forbidden)


def test_preflight_does_not_need_a_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network connection attempted")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    receipt = autotune.preflight_receipt()
    verify_sealed(receipt)
    assert receipt["body_boundary"]["network_access"] is False
    assert receipt["body_boundary"]["model_body_bytes_read"] == 0
    assert receipt["body_boundary"]["live_controller_state_created"] is False


def test_ranges_are_unique_payload_bounded_and_deterministic(plan: dict) -> None:
    ranges = plan["range_strategy"]["body_ranges"]
    expected_indices = [1 + (index * 280) // 47 for index in range(48)]
    assert [row["shard_index"] for row in ranges] == expected_indices
    assert len({row["path"] for row in ranges}) == 48
    assert ranges[0]["path"] == "model-00001-of-00282.safetensors"
    assert ranges[-1]["path"] == "model-00281-of-00282.safetensors"
    for row in ranges:
        assert row["start"] >= row["data_start"]
        assert row["end"] <= row["file_bytes"]
        assert row["end"] - row["start"] == autotune.RANGE_BYTES
        assert (row["start"] - row["data_start"]) % autotune.RANGE_ALIGNMENT == 0
        assert row["manifest_identity_matches"] is True


def test_matrix_covers_every_requested_setting_and_stays_under_two_percent(plan: dict) -> None:
    matrix = plan["trial_matrix"]
    assert len(matrix) == 12
    assert sum(row["range_count"] for row in matrix) == 184
    assert {row["caller_concurrent_shard_streams"] for row in matrix} >= {8, 16, 24, 32, 48}
    fixed = {
        int(row["environment"]["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"])
        for row in matrix
        if "HF_XET_FIXED_DOWNLOAD_CONCURRENCY" in row["environment"]
    }
    assert fixed == {16, 32, 64}
    assert any(row["high_performance"] for row in matrix)
    assert {row["trial_id"] for row in matrix if row["chunk_cache_policy"].startswith("ONE_GIB")} == {
        "CACHE_1G_COLD", "CACHE_1G_REPLAY"
    }
    budget = plan["network_budget"]
    assert budget["bounded_range_payload_bytes"] == 12_348_030_976
    assert budget["largest_shard_validation_bytes"] == 10_736_723_088
    assert budget["planned_maximum_bytes"] == 23_084_754_064
    assert budget["hard_cap_bytes"] == 30_133_860_738
    assert budget["planned_maximum_bytes"] < budget["hard_cap_bytes"]
    assert plan["cache_policy"]["conditional_10_gib_status"] == "SKIPPED_CONDITION_FALSE"
    required_rows = {
        row["caller_concurrent_shard_streams"]: row
        for row in matrix
        if row["trial_id"].startswith("FILES_")
    }
    assert set(required_rows) == set(autotune.REQUIRED_FILE_SETTINGS)
    for row in required_rows.values():
        assert row["selectable_after_schedule_refreeze"] is True
        assert row["diagnostic_only_reason"] is None
    range_ids = [row["range_id_sha256"] for row in plan["range_strategy"]["body_ranges"]]
    for row in matrix:
        assert row["ordered_range_ids"] == range_ids[:row["range_count"]]
        assert row["ordered_range_ids_sha256"] == autotune._canonical_sha256(
            row["ordered_range_ids"]
        )
    by_count: dict[int, list[list[str]]] = {}
    for row in matrix:
        by_count.setdefault(row["range_count"], []).append(row["ordered_range_ids"])
    assert all(all(ids == groups[0] for ids in groups) for groups in by_count.values())


def test_plan_refuses_unaligned_range_or_cap_above_two_percent(runtime_receipt: dict) -> None:
    with pytest.raises(Glm52Error, match="aligned"):
        autotune.build_plan(range_bytes=autotune.RANGE_BYTES + 1, runtime=runtime_receipt)
    with pytest.raises(Glm52Error, match="above 2%"):
        autotune.build_plan(
            network_cap_bytes=autotune.NETWORK_CAP_BYTES + 1,
            runtime=runtime_receipt,
        )


def test_schedule_reconciliation_is_exact(plan: dict) -> None:
    row = plan["schedule_reconciliation"]
    assert row["source_shards_exactly_once"] == 282
    assert row["planned_refetches"] == 0
    assert row["maximum_new_fetch_shards"] == 23
    assert row["maximum_resident_shards_one_window"] == 26
    assert row["maximum_actual_adjacent_active_prefetch_union"] == 45
    assert row["conservative_active_prefetch_upper_bound"] == 52
    assert row["required_file_settings_tested"] == [8, 16, 24, 32, 48]
    assert row["largest_useful_tested_file_setting"] == 48
    assert row["preliminary_schedule_maximum_caller_dispatch"] == 23
    assert row["diagnostic_file_settings_not_selectable"] == []


def test_plan_requires_controller_v2_intent_and_bot_api_receipt_v3(plan: dict) -> None:
    authority = plan["execution_authority"]
    assert authority["live_executor_implemented"] is True
    assert authority["live_executor_path"] == "tools/condense/glm52_xet_live.py"
    assert authority["current_plan_authorizes_execution"] is False
    assert authority["offline_plan_alone_authorizes_execution"] is False
    assert authority["required_authority_schema"] == (
        "hawking.glm52.xet_execution_authority.v3"
    )
    assert authority["required_controller_epoch"] == "glm52-controller-v2"
    assert authority["required_prepared_transition_intent_schema"] == (
        state.TRANSITION_INTENT_SCHEMA
    )
    assert authority["authenticated_telegram_receipt_schema"] == (
        state.TELEGRAM_RECEIPT_SCHEMA
    )
    assert authority["exact_expected_campaign_contract_binding_required"] is True
    assert authority["exact_plan_seal_in_prepared_terminal_evidence_required"] is True
    assert authority["self_reported_controller_booleans_are_authority"] is False


def test_plan_markdown_states_selectability_and_separate_executor_honestly(plan: dict) -> None:
    markdown = autotune.render_plan_markdown(plan)
    assert "`8`, `16`, `24`, `32`, and `48` are all selectable" in markdown
    assert "informational, not a selection cap" in markdown
    assert "separate live executor exists" in markdown
    assert "offline plan alone does not authorize execution" in markdown
    assert "are diagnostic" not in markdown
    assert "Live execution is not implemented" not in markdown


def _sample(**updates: object) -> dict:
    value = {
        "swap_used_bytes": 100,
        "swapouts": 7,
        "thermal_warning": False,
        "free_disk_bytes": 1_000,
        "available_ram_bytes": 500,
        "cpu_percent": 25.0,
        "disk_write_bytes_per_second": 100.0,
        "reconstruction_latency_seconds": 1.5,
        "retry_rate": 0.01,
        "temporary_amplification_ratio": 1.1,
        "actual_network_bytes": 0,
        "materialized_raw_allocated_bytes": 10,
    }
    value.update(updates)
    return value


def _resource(**kwargs: object) -> dict:
    return autotune.evaluate_resource_trial(
        _sample(),
        kwargs.pop("samples", [_sample()]),
        kwargs.pop("after", _sample(actual_network_bytes=100)),
        required_free_bytes=800,
        required_available_ram_bytes=400,
        trial_network_cap_bytes=1_000,
        **kwargs,
    )


def test_resource_verdict_accepts_exact_five_percent_but_rejects_sustained_overage() -> None:
    passed = _resource(heavy_lane_regressions=[0.05, 0.05, 0.05])
    assert passed["status"] == "PASS"
    assert set(autotune.SELECTABLE_TRIAL_MEASUREMENTS) <= set(passed["measured"])
    assert passed["measured"]["actual_network_bytes"] == 100
    two_bins = _resource(heavy_lane_regressions=[0.051, 0.051, 0.0])
    assert two_bins["status"] == "PASS"
    failed = _resource(heavy_lane_regressions=[0.051, 0.052, 0.053])
    assert failed["status"] == "FAIL"
    assert "SUSTAINED_HEAVY_LANE_REGRESSION_GT_5_PERCENT" in failed["reasons"]


@pytest.mark.parametrize(
    "field",
    [
        "cpu_percent",
        "disk_write_bytes_per_second",
        "reconstruction_latency_seconds",
        "retry_rate",
        "temporary_amplification_ratio",
        "actual_network_bytes",
    ],
)
def test_resource_verdict_requires_complete_live_observation_schema(field: str) -> None:
    before = _sample()
    before.pop(field)
    verdict = autotune.evaluate_resource_trial(
        before,
        [_sample()],
        _sample(actual_network_bytes=100),
        required_free_bytes=800,
        required_available_ram_bytes=400,
        trial_network_cap_bytes=1_000,
    )
    assert verdict["status"] == "FAIL"
    assert any("MISSING" in reason and field.upper() in reason for reason in verdict["reasons"])


@pytest.mark.parametrize(
    ("samples", "after", "reason"),
    [
        ([_sample(actual_network_bytes=100)], _sample(actual_network_bytes=50),
         "ACTUAL_NETWORK_COUNTER_REGRESSION"),
        ([_sample(actual_network_bytes=500)], _sample(actual_network_bytes=1_001),
         "TRIAL_NETWORK_CAP_EXCEEDED"),
        ([_sample(retry_rate=float("nan"))], _sample(actual_network_bytes=100),
         "RETRY_RATE_INVALID"),
    ],
)
def test_resource_verdict_rejects_invalid_network_or_retry_observations(
    samples: list[dict], after: dict, reason: str
) -> None:
    verdict = _resource(samples=samples, after=after)
    assert verdict["status"] == "FAIL"
    assert any(reason in item for item in verdict["reasons"])


@pytest.mark.parametrize(
    ("samples", "after", "views", "reason"),
    [
        ([_sample(swap_used_bytes=101)], _sample(), 0, "SWAP_GROWTH"),
        ([_sample(swapouts=8)], _sample(), 0, "NEW_SWAPOUTS"),
        ([_sample(thermal_warning=True)], _sample(), 0, "THERMAL_WARNING"),
        ([_sample(free_disk_bytes=799)], _sample(), 0, "DISK_FLOOR_RISK"),
        ([_sample(available_ram_bytes=399)], _sample(), 0, "RAM_FLOOR_RISK"),
        ([_sample()], _sample(), 2, "DUPLICATED_COMPLETE_SOURCE_VIEW"),
    ],
)
def test_resource_verdict_rejects_each_safety_boundary(
    samples: list[dict], after: dict, views: int, reason: str
) -> None:
    verdict = _resource(samples=samples, after=after, complete_source_views=views)
    assert verdict["status"] == "FAIL"
    assert reason in verdict["reasons"]


def test_resource_verdict_enforces_absolute_swap_and_live_allocation_ceilings() -> None:
    absolute_swap = _resource(maximum_swap_used_bytes=99)
    assert "ABSOLUTE_SWAP_CEILING" in absolute_swap["reasons"]

    absolute_allocation = _resource(
        maximum_materialized_raw_allocated_bytes=9,
    )
    assert "MATERIALIZED_RAW_ALLOCATION_CEILING" in absolute_allocation["reasons"]

    live_growth = _resource(samples=[_sample(materialized_raw_allocated_bytes=211)])
    assert "LIVE_ALLOCATION_GROWTH_CEILING" in live_growth["reasons"]


def _trial(
    trial_id: str,
    throughput: float,
    *,
    rss: int,
    transfer: int,
    streams: int,
    low: float | None = None,
    high: float | None = None,
    verdict: str = "PASS",
    steady: bool = False,
) -> dict:
    return {
        "trial_id": trial_id,
        "eligible_lanes": ["acquisition", "steady"],
        "resource_verdict": verdict,
        "caller_concurrent_shard_streams": streams,
        "throughput_bytes_per_second": throughput,
        "throughput_ci95_low": throughput if low is None else low,
        "throughput_ci95_high": throughput if high is None else high,
        "peak_rss_bytes": rss,
        "effective_transfer_concurrency": transfer,
        "sustained_heavy_lane_regression": not steady,
        "peak_cpu_percent": 40.0,
        "peak_disk_write_bytes_per_second": 100.0,
        "maximum_reconstruction_latency_seconds": 1.0,
        "maximum_retry_rate": 0.01,
        "maximum_temporary_amplification_ratio": 1.1,
        "actual_network_bytes": 100,
        "trial_network_cap_bytes": 1_000,
    }


def test_selection_uses_confidence_then_resource_tie_break_and_allows_32_and_48() -> None:
    trials = [
        _trial("fast-heavy", 105.0, rss=900, transfer=64, streams=24, low=100, high=110, steady=True),
        _trial("lean-overlap", 102.0, rss=400, transfer=16, streams=16, low=99, high=106, steady=True),
        _trial("files-32", 120.0, rss=1, transfer=1, streams=32, steady=True),
        _trial("failed", 200.0, rss=1, transfer=1, streams=8, verdict="FAIL", steady=True),
    ]
    selected = autotune.select_profile(trials, lane="steady")
    assert selected["trial_id"] == "files-32"
    assert selected["selected_caller_concurrent_shard_streams"] == 32
    assert selected["preliminary_schedule_maximum_dispatch"] == 23
    assert selected["post_autotune_schedule_refreeze_required"] is True
    assert selected["selection_pool"] == ["files-32"]

    files_48 = _trial("files-48", 130.0, rss=1, transfer=1, streams=48, steady=True)
    assert autotune.select_profile([files_48], lane="steady")[
        "selected_caller_concurrent_shard_streams"
    ] == 48


def test_selection_requires_steady_heavy_lane_pass() -> None:
    trials = [_trial("acquisition-only", 10, rss=1, transfer=4, streams=8, steady=False)]
    assert autotune.select_profile(trials, lane="acquisition")["trial_id"] == "acquisition-only"
    with pytest.raises(Glm52Error, match="no safe selectable steady"):
        autotune.select_profile(trials, lane="steady")


@pytest.mark.parametrize("field", sorted(autotune.SELECTABLE_TRIAL_MEASUREMENTS))
def test_selection_rejects_trials_missing_any_recorded_live_metric(field: str) -> None:
    trial = _trial("incomplete", 10, rss=1, transfer=4, streams=8, steady=True)
    trial.pop(field)
    with pytest.raises(Glm52Error, match="no safe selectable"):
        autotune.select_profile([trial], lane="steady")


def test_selection_uses_recorded_retry_metric_before_rss_tie_break() -> None:
    low_retry = _trial("low-retry", 100, rss=900, transfer=16, streams=16, steady=True)
    high_retry = _trial("high-retry", 100, rss=100, transfer=16, streams=16, steady=True)
    low_retry["maximum_retry_rate"] = 0.0
    high_retry["maximum_retry_rate"] = 0.1
    assert autotune.select_profile([high_retry, low_retry], lane="steady")["trial_id"] == "low-retry"


def _gc_entry(path: pathlib.Path, *, inode: int = 1, size: int = 0) -> dict:
    return {
        "path": str(path),
        "expected_size": size,
        "expected_inode": inode,
        "expected_sha256": "0" * 64,
    }


def test_gc_validation_is_exact_and_never_deletes(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "trial" / "partial.bin"
    target.parent.mkdir()
    target.write_bytes(b"body")
    metadata = target.stat()
    entry = {
        "path": str(target),
        "expected_size": 4,
        "expected_inode": metadata.st_ino,
        "expected_sha256": autotune.hashlib.sha256(b"body").hexdigest(),
    }
    receipt = autotune.validate_gc_manifest(tmp_path, [entry], inspect_existing=True)
    assert receipt[0]["deletion_performed"] is False
    assert target.read_bytes() == b"body"


def test_gc_validation_rejects_globs_escape_duplicate_and_symlink(tmp_path: pathlib.Path) -> None:
    inside = tmp_path / "a.bin"
    outside = tmp_path.parent / "outside.bin"
    with pytest.raises(Glm52Error, match="exact path"):
        autotune.validate_gc_manifest(tmp_path, [_gc_entry(tmp_path / "*.bin")])
    with pytest.raises(Glm52Error, match="escapes"):
        autotune.validate_gc_manifest(tmp_path, [_gc_entry(outside)])
    with pytest.raises(Glm52Error, match="duplicates"):
        autotune.validate_gc_manifest(tmp_path, [_gc_entry(inside), _gc_entry(inside)])
    link = tmp_path / "link"
    link.symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(Glm52Error, match="outside|symlink"):
        autotune.validate_gc_manifest(tmp_path, [_gc_entry(link / "victim.bin")])


def test_plan_verifier_rejects_tamper_and_stale_identity(plan: dict) -> None:
    corrupted = json.loads(json.dumps(plan))
    corrupted["claims"]["source_shards_fetched"] = 1
    with pytest.raises(Glm52Error, match="seal mismatch"):
        autotune.verify_plan(corrupted, rebuild=False)
    resealed = seal({key: value for key, value in corrupted.items() if key != "seal_sha256"})
    with pytest.raises(Glm52Error, match="deterministic rebuild"):
        autotune.verify_plan(resealed)


def test_plan_verifier_rejects_resealed_range_order_or_toolchain_tamper(plan: dict) -> None:
    reordered = json.loads(json.dumps(plan))
    ids = reordered["trial_matrix"][0]["ordered_range_ids"]
    ids[0], ids[1] = ids[1], ids[0]
    reordered["trial_matrix"][0]["ordered_range_ids_sha256"] = autotune._canonical_sha256(ids)
    resealed = seal({key: value for key, value in reordered.items() if key != "seal_sha256"})
    with pytest.raises(Glm52Error, match="range identities/order"):
        autotune.verify_plan(resealed, rebuild=False)

    tool_tamper = json.loads(json.dumps(plan))
    tool_tamper["toolchain_binding"]["files"][0]["sha256"] = "f" * 64
    resealed = seal({key: value for key, value in tool_tamper.items() if key != "seal_sha256"})
    with pytest.raises(Glm52Error, match="toolchain binding"):
        autotune.verify_plan(resealed, rebuild=False)


def test_cli_plan_verify_and_preflight_are_offline(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "plan.json"
    markdown = tmp_path / "plan.md"
    assert autotune.main(["preflight", "--no-body"]) == 0
    assert autotune.main([
        "plan", "--output", str(output), "--markdown-output", str(markdown)
    ]) == 0
    assert output.exists() and markdown.exists()
    assert autotune.main(["verify", "--offline", "--plan", str(output)]) == 0
    monkeypatch.delenv("HAWKING_GLM52_XET_EXECUTE", raising=False)
    assert autotune.main(["run", "--plan", str(output)]) == 2


_AUTH_CAMPAIGN = "glm52-bf16-xet-authority-test"
_AUTH_CONTRACT = "a" * 64
_AUTH_CHAT_ID = -100424242
_AUTH_KEY = b"glm52-offline-authority-test-key-material!!"


def _auth_config() -> state.TelegramAuthConfig:
    return state.TelegramAuthConfig(
        hmac_key=_AUTH_KEY,
        expected_chat_identity_digest=state.telegram_chat_identity_digest(_AUTH_CHAT_ID),
    )


def _prepared_autotune_intent(
    plan: dict, expected_contract_sha256: str = _AUTH_CONTRACT
) -> dict:
    anchor_body = {
        "schema": state.CONTROLLER_ANCHOR_SCHEMA,
        "campaign_id": _AUTH_CAMPAIGN,
        "source_revision": autotune.REVISION,
        "controller_epoch": autotune.CONTROLLER_EPOCH,
        "expected_contract_sha256": expected_contract_sha256,
        "from_state": "BUILD_DEPENDENCY_GRAPH",
        "checkpoint": {
            "event_count": 6,
            "event_head_hash": "1" * 64,
            "window_event_count": 0,
            "window_event_head_hash": state.GENESIS_HASH,
            "checkpoint_seal_sha256": "2" * 64,
        },
    }
    anchor = {
        **anchor_body,
        "anchor_sha256": autotune._canonical_sha256(anchor_body),
    }
    terminal = seal({
        "schema": state.TERMINAL_EVIDENCE_SCHEMA,
        "state": "AUTOTUNE_XET",
        "expected_contract_sha256": expected_contract_sha256,
        "controller_anchor_sha256": anchor["anchor_sha256"],
        "artifact_seals": {
            "xet_autotune_plan": {
                "path": "GLM52_XET_AUTOTUNE_PLAN.json",
                "file_sha256": "3" * 64,
                "seal_sha256": plan["seal_sha256"],
                "schema": autotune.PLAN_SCHEMA,
                "status": "PASS_OFFLINE_PLAN_BODY_NOT_READ",
            },
        },
        "checklist": {},
        "phone_status": None,
        "created_at": "2026-07-21T12:00:00Z",
    })
    requested_payload: dict = {}
    state_payload = {"terminal_evidence": terminal}
    claim_id = "test:xet-autotune:authority:0001"
    request_sha256 = autotune._canonical_sha256({
        "schema": "hawking.glm52.state_transition_request.v2",
        "campaign_id": _AUTH_CAMPAIGN,
        "source_revision": autotune.REVISION,
        "controller_epoch": autotune.CONTROLLER_EPOCH,
        "expected_contract_sha256": expected_contract_sha256,
        "claim_id": claim_id,
        "to_state": "AUTOTUNE_XET",
        "state_payload": state_payload,
    })
    status = {
        "state": "AUTOTUNE_XET",
        "source_coverage_percent": 0.0,
        "shards": {"fetched": 0, "verified": 0, "evicted": 0, "total": 282},
        "network_bytes": 0,
        "throughput_bytes_per_second": 0.0,
        "eta_seconds": None,
        "current": {"window": None, "layer": None},
        "candidate_rates": [],
        "best_metrics": {},
        "resources": {
            "disk_free_bytes": 640_000_000_000,
            "ram_available_bytes": 90_000_000_000,
            "swap_used_bytes": 0,
        },
        "process": {"pid": 4242, "lease_held": True, "lease_owner": "glm52-worker"},
    }
    status_sha256 = state._campaign_status_hash(status)
    dedupe_key = autotune._canonical_sha256({
        "schema": "hawking.glm52.transition_notification_identity.v2",
        "campaign_id": _AUTH_CAMPAIGN,
        "source_revision": autotune.REVISION,
        "controller_epoch": autotune.CONTROLLER_EPOCH,
        "expected_contract_sha256": expected_contract_sha256,
        "event_kind": state.TRANSITION_EVENT_KINDS["AUTOTUNE_XET"],
        "claim_id": claim_id,
        "from_state": "BUILD_DEPENDENCY_GRAPH",
        "to_state": "AUTOTUNE_XET",
        "requested_payload_sha256": autotune._canonical_sha256(requested_payload),
        "controller_anchor_sha256": anchor["anchor_sha256"],
        "canonical_status_sha256": status_sha256,
    })
    rendered = state.render_campaign_status_message(
        state.TRANSITION_EVENT_KINDS["AUTOTUNE_XET"],
        dedupe_key,
        status,
        anchor,
        claim_id=claim_id,
        from_state="BUILD_DEPENDENCY_GRAPH",
        to_state="AUTOTUNE_XET",
    )
    sealed_intent = seal({
        "schema": state.TRANSITION_INTENT_SCHEMA,
        "campaign_id": _AUTH_CAMPAIGN,
        "source_revision": autotune.REVISION,
        "controller_epoch": autotune.CONTROLLER_EPOCH,
        "expected_contract_sha256": expected_contract_sha256,
        "event_kind": state.TRANSITION_EVENT_KINDS["AUTOTUNE_XET"],
        "from_state": "BUILD_DEPENDENCY_GRAPH",
        "to_state": "AUTOTUNE_XET",
        "claim_id": claim_id,
        "requested_payload": requested_payload,
        "state_payload": state_payload,
        "request_sha256": request_sha256,
        "dedupe_key": dedupe_key,
        "controller_anchor": anchor,
        "canonical_status": status,
        "canonical_status_sha256": status_sha256,
        "rendered_message": rendered,
        "rendered_message_sha256": autotune.hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest(),
        "prepared_at": "2026-07-21T12:00:00Z",
    })
    intent = {
        **sealed_intent,
        "controller_hmac_sha256": _auth_config().authenticate({
            "schema": "hawking.glm52.state_transition_intent_auth.v1",
            "intent": sealed_intent,
        }),
    }
    state.validate_transition_intent(intent, _auth_config())
    return intent


def _authority(plan: dict, expected_contract_sha256: str = _AUTH_CONTRACT) -> dict:
    intent = _prepared_autotune_intent(plan, expected_contract_sha256)
    receipt = state.make_telegram_delivery_receipt(
        intent,
        auth=_auth_config(),
        bot_api_response={
            "ok": True,
            "result": {
                "message_id": 9,
                "chat": {"id": _AUTH_CHAT_ID, "type": "private"},
                "text": intent["rendered_message"],
            },
        },
        http_status=200,
        delivered_at="2026-07-21T12:00:01Z",
    )
    anchor_checkpoint = intent["controller_anchor"]["checkpoint"]
    checkpoint = {
        "schema": autotune.COMMITTED_CHECKPOINT_REF_SCHEMA,
        "checkpoint_schema": state.CHECKPOINT_SCHEMA,
        "campaign_id": _AUTH_CAMPAIGN,
        "source_revision": autotune.REVISION,
        "controller_epoch": autotune.CONTROLLER_EPOCH,
        "expected_contract_sha256": expected_contract_sha256,
        "state": "AUTOTUNE_XET",
        "last_claim_id": intent["claim_id"],
        "transition_intent_sha256": intent["seal_sha256"],
        "telegram_receipt_hmac_sha256": receipt["hmac_sha256"],
        "checkpoint_seal_sha256": "8" * 64,
        "event_count": anchor_checkpoint["event_count"] + 1,
        "event_head_hash": "9" * 64,
        "window_event_count": anchor_checkpoint["window_event_count"],
        "window_event_head_hash": anchor_checkpoint["window_event_head_hash"],
    }
    return seal({
        "schema": autotune.EXECUTION_AUTHORITY_SCHEMA,
        "status": "AUTHORIZED_BY_COMMITTED_AUTOTUNE_XET_TRANSITION",
        "repo": autotune.REPO_ID,
        "revision": autotune.REVISION,
        "campaign_id": _AUTH_CAMPAIGN,
        "controller_epoch": autotune.CONTROLLER_EPOCH,
        "expected_contract_sha256": expected_contract_sha256,
        "plan_seal_sha256": plan["seal_sha256"],
        "transition_intent": intent,
        "telegram_delivery_receipt": receipt,
        "committed_controller_checkpoint": checkpoint,
        "credentials_serialized": False,
    })


class CurrentStateAuthorityVerifier:
    """Test double with independent auth, checkpoint, and lease observations."""

    def __init__(
        self,
        authority: dict,
        *,
        fail: str | None = None,
        lease_held: bool = True,
    ) -> None:
        self.fail = fail
        self.lease_held = lease_held
        self.plan_seal = authority["plan_seal_sha256"]
        self.contract_seal = authority["expected_contract_sha256"]
        self.live_checkpoint = copy.deepcopy(authority["committed_controller_checkpoint"])
        self.calls: list[str] = []

    def _result(self, name: str) -> bool:
        self.calls.append(name)
        return self.fail != name

    def _trusted_args(self, plan_seal: str, expected_contract_sha256: str) -> bool:
        return plan_seal == self.plan_seal and expected_contract_sha256 == self.contract_seal

    def verify_prepared_transition_intent_hmac(
        self,
        transition_intent: dict,
        *,
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        try:
            state.validate_transition_intent(transition_intent, _auth_config())
        except state.StateError:
            return False
        return self._trusted_args(plan_seal, expected_contract_sha256) \
            and self._result("intent")

    def verify_telegram_delivery_receipt(
        self,
        receipt: dict,
        *,
        transition_intent: dict,
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        try:
            state.validate_telegram_delivery_receipt(
                receipt, transition_intent, _auth_config()
            )
        except state.StateError:
            return False
        return self._trusted_args(plan_seal, expected_contract_sha256) \
            and self._result("telegram")

    def verify_committed_controller_checkpoint(
        self,
        checkpoint: dict,
        *,
        transition_intent: dict,
        telegram_receipt: dict,
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        bound = checkpoint == self.live_checkpoint \
            and checkpoint["transition_intent_sha256"] == transition_intent["seal_sha256"] \
            and checkpoint["telegram_receipt_hmac_sha256"] == telegram_receipt["hmac_sha256"]
        return bound and self._trusted_args(plan_seal, expected_contract_sha256) \
            and self._result("checkpoint")

    def verify_live_singleton_lease(
        self,
        checkpoint: dict,
        *,
        transition_intent: dict,
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        bound = checkpoint == self.live_checkpoint \
            and checkpoint["controller_epoch"] == autotune.CONTROLLER_EPOCH \
            and transition_intent["canonical_status"]["process"]["lease_held"] is True
        return bound and self.lease_held \
            and self._trusted_args(plan_seal, expected_contract_sha256) \
            and self._result("lease")


def _validate_authority(authority: dict, verifier: CurrentStateAuthorityVerifier) -> dict:
    return autotune.validate_execution_authority(
        authority,
        plan_seal=authority["plan_seal_sha256"],
        expected_contract_sha256=authority["expected_contract_sha256"],
        verifier=verifier,
    )


def test_authority_uses_current_prepared_intent_and_bot_api_receipt_v3(plan: dict) -> None:
    authority = _authority(plan)
    assert authority["transition_intent"]["controller_epoch"] == "glm52-controller-v2"
    assert authority["telegram_delivery_receipt"]["schema"] == (
        "hawking.glm52.telegram_delivery_receipt.v3"
    )
    assert authority["telegram_delivery_receipt"]["transition_intent"] == (
        authority["transition_intent"]
    )
    with pytest.raises(Glm52Error, match="independent.*verifier"):
        autotune.validate_execution_authority(
            authority,
            plan_seal=plan["seal_sha256"],
            expected_contract_sha256=_AUTH_CONTRACT,
        )
    verifier = CurrentStateAuthorityVerifier(authority)
    assert _validate_authority(authority, verifier) == authority
    assert verifier.calls == ["intent", "telegram", "checkpoint", "lease"]


@pytest.mark.parametrize("failed_check", ["intent", "telegram", "checkpoint", "lease"])
def test_authority_refuses_each_independent_verification_failure(
    plan: dict, failed_check: str
) -> None:
    authority = _authority(plan)
    with pytest.raises(Glm52Error, match="independent"):
        _validate_authority(
            authority, CurrentStateAuthorityVerifier(authority, fail=failed_check)
        )


def test_authority_refuses_stale_live_checkpoint_or_absent_live_lease(plan: dict) -> None:
    authority = _authority(plan)
    stale = CurrentStateAuthorityVerifier(authority)
    stale.live_checkpoint["checkpoint_seal_sha256"] = "f" * 64
    with pytest.raises(Glm52Error, match="checkpoint"):
        _validate_authority(authority, stale)
    with pytest.raises(Glm52Error, match="lease"):
        _validate_authority(
            authority, CurrentStateAuthorityVerifier(authority, lease_held=False)
        )


@pytest.mark.parametrize(
    ("target", "field", "replacement", "message"),
    [
        ("receipt", "rendered_message", "forged", "exact prepared intent"),
        ("receipt", "controller_anchor_sha256", "f" * 64, "exact prepared intent"),
        ("receipt", "canonical_status_sha256", "e" * 64, "exact prepared intent"),
        ("checkpoint", "event_count", 999, "committed checkpoint"),
        ("checkpoint", "telegram_receipt_hmac_sha256", "d" * 64, "committed checkpoint"),
    ],
)
def test_resealed_authority_rejects_receipt_or_checkpoint_binding_tamper(
    plan: dict, target: str, field: str, replacement: object, message: str
) -> None:
    original = _authority(plan)
    body = copy.deepcopy({key: value for key, value in original.items() if key != "seal_sha256"})
    nested = (
        body["telegram_delivery_receipt"]
        if target == "receipt"
        else body["committed_controller_checkpoint"]
    )
    nested[field] = replacement
    tampered = seal(body)
    with pytest.raises(Glm52Error, match=message):
        _validate_authority(tampered, CurrentStateAuthorityVerifier(original))


def test_resealed_authority_cannot_forge_intent_or_receipt_hmac(plan: dict) -> None:
    original = _authority(plan)
    body = copy.deepcopy({key: value for key, value in original.items() if key != "seal_sha256"})
    body["transition_intent"]["controller_hmac_sha256"] = "f" * 64
    body["telegram_delivery_receipt"]["transition_intent"] = copy.deepcopy(
        body["transition_intent"]
    )
    forged_intent = seal(body)
    with pytest.raises(Glm52Error, match="independent prepared transition intent HMAC"):
        _validate_authority(
            forged_intent, CurrentStateAuthorityVerifier(forged_intent)
        )

    body = copy.deepcopy({key: value for key, value in original.items() if key != "seal_sha256"})
    body["telegram_delivery_receipt"]["hmac_sha256"] = "e" * 64
    body["committed_controller_checkpoint"]["telegram_receipt_hmac_sha256"] = "e" * 64
    forged_receipt = seal(body)
    with pytest.raises(Glm52Error, match="independent Telegram Bot API v3 receipt"):
        _validate_authority(
            forged_receipt, CurrentStateAuthorityVerifier(forged_receipt)
        )


def test_prepared_intent_must_bind_external_plan_and_contract(plan: dict) -> None:
    original = _authority(plan)
    body = copy.deepcopy({key: value for key, value in original.items() if key != "seal_sha256"})
    intent = body["transition_intent"]
    terminal_body = {
        key: value for key, value in intent["state_payload"]["terminal_evidence"].items()
        if key != "seal_sha256"
    }
    terminal_body["artifact_seals"]["xet_autotune_plan"]["seal_sha256"] = "f" * 64
    intent["state_payload"]["terminal_evidence"] = seal(terminal_body)
    sealed_intent = seal({
        key: value for key, value in intent.items()
        if key not in {"seal_sha256", "controller_hmac_sha256"}
    })
    body["transition_intent"] = {
        **sealed_intent,
        "controller_hmac_sha256": intent["controller_hmac_sha256"],
    }
    tampered = seal(body)
    with pytest.raises(Glm52Error, match="exact Xet plan seal"):
        _validate_authority(tampered, CurrentStateAuthorityVerifier(original))

    with pytest.raises(Glm52Error, match="identity mismatch"):
        autotune.validate_execution_authority(
            original,
            plan_seal=plan["seal_sha256"],
            expected_contract_sha256="b" * 64,
            verifier=CurrentStateAuthorityVerifier(original),
        )


def test_v2_or_operator_confirmed_receipt_cannot_authorize_execution(plan: dict) -> None:
    original = _authority(plan)
    body = copy.deepcopy({key: value for key, value in original.items() if key != "seal_sha256"})
    body["telegram_delivery_receipt"]["schema"] = (
        "hawking.glm52.telegram_delivery_receipt.v2"
    )
    with pytest.raises(Glm52Error, match="Bot API v3"):
        _validate_authority(seal(body), CurrentStateAuthorityVerifier(original))

    body = copy.deepcopy({key: value for key, value in original.items() if key != "seal_sha256"})
    body["telegram_delivery_receipt"] = {
        "schema": "hawking.glm52.telegram_operator_confirmation.v1",
        "status": "OPERATOR_CONFIRMED_DELIVERED",
        "transition_intent": original["transition_intent"],
        "message_id": 10,
        "hmac_sha256": "f" * 64,
    }
    with pytest.raises(Glm52Error, match="Bot API v3"):
        _validate_authority(seal(body), CurrentStateAuthorityVerifier(original))


def test_self_reported_authority_booleans_cannot_replace_live_checks(plan: dict) -> None:
    original = _authority(plan)
    body = copy.deepcopy({key: value for key, value in original.items() if key != "seal_sha256"})
    body["committed_controller_checkpoint"]["singleton_lease_held"] = True
    body["committed_controller_checkpoint"]["authenticated_checkpoint"] = True
    claimed = seal(body)
    with pytest.raises(Glm52Error, match="committed checkpoint"):
        _validate_authority(claimed, CurrentStateAuthorityVerifier(original))


def test_run_remains_refused_even_with_structurally_valid_authority(
    tmp_path: pathlib.Path,
    plan: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.json"
    atomic_json(plan_path, plan)
    contract_path = tmp_path / "expected-contract.json"
    contract = seal({
        "schema": state.EXPECTED_CONTRACT_SCHEMA,
        "campaign_id": _AUTH_CAMPAIGN,
        "source_revision": autotune.REVISION,
    })
    atomic_json(contract_path, contract)
    authority = _authority(plan, contract["seal_sha256"])
    authority_path = tmp_path / "authority.json"
    atomic_json(authority_path, authority)
    monkeypatch.setenv("HAWKING_GLM52_XET_EXECUTE", "1")
    assert autotune.main([
        "run", "--plan", str(plan_path),
        "--expected-contract", str(contract_path),
        "--authority", str(authority_path),
    ]) == 2
    assert "independent intent/checkpoint/lease/Telegram" in capsys.readouterr().err
