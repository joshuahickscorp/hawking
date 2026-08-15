from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from lab.operators import ascension_qwen80_multi_layer_capture_lifecycle as lifecycle
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPLETE_MAIN = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "ascension-sandbox/physical/qwen80/complete-runtime"
)
FROZEN_HOST_PREFLIGHT = (
    COMPLETE_MAIN / "QWEN80_MULTI_LAYER_HOST_PREFLIGHT_L0_L2_20260809T192633Z/preflight.json"
)
FROZEN_SCHEDULE = COMPLETE_MAIN / "QWEN80_48_LAYER_EXECUTION_SCHEDULE_AUTHORITY_20260809T192559Z.json"
FROZEN_ORACLE = COMPLETE_MAIN / "QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_L0_L2_20260809T192600Z.json"
FROZEN_ASSESSMENT = COMPLETE_MAIN / "QWEN80_L1_FULL_LAYER_COMPLETION_ASSESSMENT_20260809T185418Z.json"
FROZEN_JOINT_ASSESSMENT = COMPLETE_MAIN / "QWEN80_L0_L1_JOINT_POST_CAPTURE_ASSESSMENT_20260809T115059Z.json"
FROZEN_HOST = Path(
    "/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/debug/examples/"
    "ascension_qwen80_source_token_multi_layer_same_runtime_device"
)


def _sha(character: str) -> str:
    return character * 64


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def _evidence(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    document = verify(json.loads(raw.decode("utf-8")))
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "document_sha256": document["seal_sha256"],
        "document_seal_sha256": document["seal_sha256"],
    }


def _green_snapshot() -> dict[str, Any]:
    return {
        "memory_free_percent": 91,
        "swap_used_bytes": 0,
        "q80_watcher_parent_pids": [22035],
        "watcher_hold_active": True,
        "q80_multi_layer_capture_children": [],
        "q30_capture_children": [],
    }


def _kernel_names() -> list[str]:
    one = [
        "qwen_next_direct_packed_input_rmsnorm",
        "qwen_binary_sign_scale_matvec",
        "qwen_binary_sign_scale_matvec",
        "qwen_next_qkvz_rearrange_conv_l2",
        "qwen_next_ba_to_decay_beta",
        "qwen_next_gated_delta_decode_single",
        "qwen_next_deltanet_gated_rmsnorm",
        "qwen_binary_sign_scale_matvec",
        "qwen_next_add_residual",
        "qwen80_postnorm_router_top10_rmsnorm",
        "qwen80_postnorm_router_top10_matvec",
        "qwen80_postnorm_router_top10_select",
        "qwen80_all_ten_routed_wave_route_guard",
        "qwen80_all_ten_routed_wave_gate_up",
        "qwen80_all_ten_routed_wave_swiglu",
        "qwen80_all_ten_routed_wave_down_weighted",
        "qwen80_shared_expert_wave_gate_up",
        "qwen80_shared_expert_wave_swiglu",
        "qwen80_shared_expert_wave_down",
        "qwen80_shared_expert_wave_scalar_gate",
        "qwen80_shared_expert_wave_apply_sigmoid_gate",
        "qwen80_moe_wave_aggregate_second_residual_route_sum",
        "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
    ]
    return one * lifecycle.LAYER_COUNT


def _synthetic_context(
    tmp_path: Path, *, capture_body_wired: bool = True
) -> tuple[lifecycle.AuthorityContext, lifecycle.AuthorityPins, Path, Path, Path]:
    host_binary = tmp_path / "host"
    host_binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    host_binary.chmod(0o700)
    host_raw = host_binary.read_bytes()
    host_evidence = {
        "path": str(host_binary.resolve()),
        "bytes": len(host_raw),
        "sha256": hashlib.sha256(host_raw).hexdigest(),
    }
    kernel_names = _kernel_names()
    one_layer = kernel_names[:23]

    schedule = _write(
        tmp_path / "schedule.json",
        seal(
            {
                "schema": lifecycle.SCHEDULE_SCHEMA,
                "status": lifecycle.SCHEDULE_STATUS,
                "layers": [
                    {
                        "layer": index,
                        "mixer": "delta_net",
                        "full_layer_dispatch_count": 23,
                        "full_layer_kernel_names": list(one_layer),
                    }
                    for index in range(3)
                ],
                "source_authority": {"source_revision": "deadbeef" * 5},
            }
        ),
    )
    oracle = _write(
        tmp_path / "oracle.json",
        seal(
            {
                "schema": lifecycle.CHAIN_ORACLE_SCHEMA,
                "status": lifecycle.CHAIN_ORACLE_STRUCTURE_STATUS,
                "layer_count": 3,
                "includes_unready_gqa": False,
                "total_dispatches_physical_capture": 69,
                "numeric_layer_outputs_composed": False,
            }
        ),
    )
    assessment = _write(
        tmp_path / "assessment.json",
        seal(
            {
                "schema": lifecycle.L1_ASSESSMENT_SCHEMA,
                "status": lifecycle.L1_ASSESSMENT_STATUS,
                "earned_complete_l1_component_only": True,
                "blockers": [],
            }
        ),
    )
    joint = _write(
        tmp_path / "joint-assessment.json",
        seal(
            {
                "schema": lifecycle.JOINT_ASSESSMENT_SCHEMA,
                "status": lifecycle.JOINT_ASSESSMENT_STATUS,
                "earned_component_only": True,
                "blockers": [],
            }
        ),
    )
    schedule_ev = _evidence(schedule)
    oracle_ev = _evidence(oracle)
    assessment_ev = _evidence(assessment)
    joint_ev = _evidence(joint)

    host = _write(
        tmp_path / "host-preflight.json",
        seal(
            {
                "schema": lifecycle.HOST_PREFLIGHT_SCHEMA,
                "status": "COMPILED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED",
                "source_token_id": 1,
                "layer_count": 3,
                "layers_inclusive_range": {"first": 0, "last": 2},
                "host_binary": {"present": True, **host_evidence},
                "execution_schedule_authority": schedule_ev,
                "chain_cpu_oracle": oracle_ev,
                "l1_full_layer_assessment_provenance": assessment_ev,
                "joint_assessment": joint_ev,
                "execution_policy": {
                    "one_runtime": True,
                    "one_command_buffer": True,
                    "single_fence_after_all_dispatches": True,
                    "fence_count": 1,
                    "non_timed": True,
                    "structural_kernel_trace_required": True,
                    "receipt_written_last": True,
                    "caller_owned_per_layer_state_slots": True,
                    "total_dispatches": 69,
                    "per_layer_dispatch_count": 23,
                },
                "structural_kernel_trace": {
                    "exact_order": True,
                    "kernel_names": kernel_names,
                },
                "future_capture_schemas": {
                    "inner": lifecycle.INNER_SCHEMA,
                    "inner_status": lifecycle.INNER_SUCCESS_STATUS,
                    "outer": lifecycle.OUTER_TERMINAL_SCHEMA,
                    "outer_status": lifecycle.OUTER_TERMINAL_STATUS,
                    "release": lifecycle.RELEASE_SCHEMA,
                    "release_status": lifecycle.RELEASE_STATUS,
                },
                "metal_path": {
                    "preflight_only": True,
                    "metal_context_or_dispatch_performed": False,
                    "physical_capture_requires_owner_lease_and_admission": True,
                    "capture_body_wired": capture_body_wired,
                    "mode_metal_available": capture_body_wired,
                    "future_metal_entrypoint": {
                        "explicit_mode_required": True,
                        "default_execution_disabled": True,
                        "requires_new_multi_layer_lease": True,
                        "requires_sealed_outer_launch_authority": True,
                        "requires_fresh_outer_and_inner_capture_directories": True,
                        "capture_body_wired": capture_body_wired,
                    },
                },
                "claim_boundary": {
                    "host_preflight_only": True,
                    "multi_layer_device_parity": False,
                    "component_only": True,
                    "token_generated": False,
                    "decoder_started": False,
                    "server_or_watcher_started": False,
                    "tps_or_tg_measured": False,
                    "tournament_started": False,
                    "test_only_fake_child": False,
                    "fixture_or_synthetic": False,
                },
            }
        ),
    )
    host_preflight_evidence = _evidence(host)
    outer = _write(
        tmp_path / "outer-preflight.json",
        seal(
            {
                "schema": lifecycle.OUTER_PREFLIGHT_SCHEMA,
                "status": "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_OUTER_CPU_ONLY_NOT_LEASED_OR_EXECUTED",
                "host_preflight": host_preflight_evidence,
                "host_binary": {"present": True, **host_evidence},
                "execution_schedule_authority": schedule_ev,
                "chain_cpu_oracle": oracle_ev,
                "l1_full_layer_assessment": assessment_ev,
                "joint_assessment": joint_ev,
                "l0_source_outer_preflight": {
                    "path": str(tmp_path / "l0-outer.json"),
                    "present": True,
                    "bytes": 1,
                    "sha256": _sha("e"),
                    "document_sha256": _sha("f"),
                    "document_seal_sha256": _sha("f"),
                },
                "original_l1_route_authority": {
                    "path": str(tmp_path / "route.json"),
                    "present": True,
                    "bytes": 1,
                    "sha256": _sha("g"),
                    "document_sha256": _sha("h"),
                    "document_seal_sha256": _sha("h"),
                },
                "exact_component_scope": {
                    "source_token_id": 1,
                    "layer_count": 3,
                    "layers_first": 0,
                    "layers_last": 2,
                    "per_layer_dispatches": 23,
                    "total_dispatches": 69,
                    "one_fence_required": True,
                    "non_timed_exact_trace_required": True,
                    "kernel_names": kernel_names,
                },
                "future_metal_entrypoint": {
                    "explicit_mode_required": True,
                    "default_execution_disabled": True,
                    "requires_new_multi_layer_lease": True,
                    "requires_sealed_outer_launch_authority": True,
                    "requires_fresh_outer_and_inner_capture_directories": True,
                    "self_hashes_current_executable": True,
                    "no_device_execution_in_this_cpu_preflight": True,
                    "capture_body_wired": capture_body_wired,
                },
                "lifecycle": {
                    "replay_guard_required": True,
                    "one_child_process_required": True,
                    "outer_reaped_terminal_required": True,
                    "automatic_retry_authorized": False,
                    "fake_child_reaper_test_only": True,
                    "real_host_metal_cli_available": capture_body_wired,
                },
            }
        ),
    )
    pins = lifecycle.AuthorityPins(
        host_binary_sha256=host_evidence["sha256"],
        host_preflight_seal_sha256=host_preflight_evidence["document_seal_sha256"],
        outer_preflight_seal_sha256=_evidence(outer)["document_seal_sha256"],
        execution_schedule_seal_sha256=schedule_ev["document_seal_sha256"],
        chain_cpu_oracle_seal_sha256=oracle_ev["document_seal_sha256"],
        l1_full_layer_assessment_seal_sha256=assessment_ev["document_seal_sha256"],
        joint_assessment_seal_sha256=joint_ev["document_seal_sha256"],
    )
    context = lifecycle.load_authority_context(
        host_preflight=host,
        outer_preflight=outer,
        host_binary=host_binary,
        pins=pins,
    )
    return context, pins, host, outer, host_binary


def _resource_path(
    tmp_path: Path, context: lifecycle.AuthorityContext, *, snapshot: dict[str, Any] | None = None
) -> Path:
    path = tmp_path / "resource.json"
    document = lifecycle.build_resource_admission(
        context=context,
        snapshot=snapshot or _green_snapshot(),
        recorded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    _write(path, document)
    return path


def _config(
    tmp_path: Path,
    *,
    context: lifecycle.AuthorityContext,
    pins: lifecycle.AuthorityPins,
    host: Path,
    outer: Path,
    binary: Path,
) -> lifecycle.ExecuteConfig:
    return lifecycle.ExecuteConfig(
        host_preflight=host,
        outer_preflight=outer,
        host_binary=binary,
        resource_admission=_resource_path(tmp_path, context),
        launch_dir=tmp_path / "new-launch",
        replay_dir=tmp_path / "new-replay",
        outer_capture_dir=tmp_path / "new-outer-capture",
        workers=1,
        timeout_seconds=5.0,
        pins=pins,
    )


def _fake_child(path: Path) -> Path:
    path.write_text(
        """
import argparse
import hashlib
import json
from pathlib import Path
from lab.receipts import seal

def evidence(path):
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode('utf-8'))
    return {
        'path': str(Path(path).resolve()), 'present': True, 'bytes': len(raw),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'document_sha256': doc['seal_sha256'], 'document_seal_sha256': doc['seal_sha256'],
    }

parser = argparse.ArgumentParser()
parser.add_argument('--layer-count', required=True)
parser.add_argument('--outer-preflight', required=True)
parser.add_argument('--lease-receipt', required=True)
parser.add_argument('--outer-launch-authority', required=True)
parser.add_argument('--outer-capture-dir', required=True)
parser.add_argument('--capture-dir', required=True)
parser.add_argument('--workers', required=True)
args = parser.parse_args()
outer = json.loads(Path(args.outer_preflight).read_text())
lease = json.loads(Path(args.lease_receipt).read_text())
capture = Path(args.capture_dir)
capture.mkdir(parents=False, exist_ok=False)
receipt = seal({
    'schema': 'hawking.ascension.qwen80_source_token_multi_layer_same_runtime_capture.v1',
    'status': 'CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_COMPONENT_ONLY',
    'fixture_or_synthetic': True,
    'lease_id': lease['lease_id'],
    'multi_layer_lease_binding': {'lease_id': lease['lease_id'], 'receipt': evidence(args.lease_receipt)},
    'outer_preflight_binding': evidence(args.outer_preflight),
    'outer_launch_authority_binding': evidence(args.outer_launch_authority),
    'durable_capture': {
        'capture_directory': str(capture.resolve()),
        'receipt_written_last_is_completion_marker': True,
    },
    'fresh_same_runtime_execution': {
        'fresh_runtime': True, 'same_runtime': True, 'same_tcb': True,
        'single_fence_after_all_dispatches': True,
        'layer_count': 3, 'total_dispatches': 69, 'fence_count': 1,
        'per_layer_dispatches': 23,
    },
    'structural_kernel_trace': {
        'exact_order': True,
        'kernel_names': outer['exact_component_scope']['kernel_names'],
    },
    'claim_boundary': {
        'multi_layer_component_only': True, 'token_generated': False, 'decoder_started': False,
        'tps_or_tg_measured': False, 'tournament_started': False,
    },
})
(capture / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True), encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )
    return path


def test_stale_host_preflight_pin_refuses() -> None:
    if not FROZEN_HOST_PREFLIGHT.is_file() or not FROZEN_HOST.is_file():
        pytest.skip("frozen multi-layer host preflight/binary not available on this machine")
    # A synthetic outer is not required: the frozen pin check fires first on host seal.
    with pytest.raises(
        lifecycle.MultiLayerLifecycleError,
        match="host preflight seal does not match the explicit frozen pin|current host binary SHA",
    ):
        lifecycle.load_authority_context(
            host_preflight=FROZEN_HOST_PREFLIGHT,
            outer_preflight=FROZEN_HOST_PREFLIGHT,  # wrong document; refuse either way
            host_binary=FROZEN_HOST,
            pins=lifecycle.AuthorityPins(),  # frozen pins
        )


def test_resource_admission_requires_zero_swap_80_percent_watcher_and_no_children(
    tmp_path: Path,
) -> None:
    context, _pins, _host, _outer, _binary = _synthetic_context(tmp_path)
    blocked = _green_snapshot()
    blocked["memory_free_percent"] = 79
    blocked["swap_used_bytes"] = 1
    blocked["q80_watcher_parent_pids"] = []
    blocked["q30_capture_children"] = [{"pid": 10}]
    blocked["q80_multi_layer_capture_children"] = [{"pid": 11}]
    result = lifecycle.build_resource_admission(context=context, snapshot=blocked)
    assert verify(result) == result
    assert result["status"] == lifecycle.RESOURCE_REFUSED_STATUS
    assert result["prepared"] is False
    assert "memory free percentage is below 80" in result["blockers"]
    assert "swap must be exactly zero" in result["blockers"]
    assert "exactly one held Q80 watcher parent is required" in result["blockers"]
    assert "Q30 capture child is already active" in result["blockers"]
    assert "Q80 multi-layer capture child is already active" in result["blockers"]


def test_resource_admission_refuses_wrong_watcher_count(tmp_path: Path) -> None:
    context, _pins, _host, _outer, _binary = _synthetic_context(tmp_path)
    two = _green_snapshot()
    # Matching the module alone can catch both runtime and tg3 watchers.
    two["q80_watcher_parent_pids"] = [100, 101]
    result = lifecycle.build_resource_admission(context=context, snapshot=two)
    assert result["prepared"] is False
    assert "exactly one held Q80 watcher parent is required" in result["blockers"]


def test_resource_admission_refuses_nonzero_swap(tmp_path: Path) -> None:
    context, _pins, _host, _outer, _binary = _synthetic_context(tmp_path)
    swap = _green_snapshot()
    swap["swap_used_bytes"] = 4096
    result = lifecycle.build_resource_admission(context=context, snapshot=swap)
    assert result["prepared"] is False
    assert "swap must be exactly zero" in result["blockers"]


def test_fake_child_lifecycle_writes_terminal_then_exactly_one_release_and_refuses_replay(
    tmp_path: Path,
) -> None:
    context, pins, host, outer, binary = _synthetic_context(tmp_path)
    config = _config(
        tmp_path, context=context, pins=pins, host=host, outer=outer, binary=binary
    )
    fake = _fake_child(tmp_path / "fake-child.py")
    result = lifecycle.run_one_shot_for_test(
        config, fake_child_command=(sys.executable, str(fake))
    )
    terminal = verify(result["outer_terminal"])
    release = verify(result["release"])
    lease = verify(result["lease"])
    launch = verify(result["outer_launch_authority"])
    assert terminal["status"] == lifecycle.OUTER_TERMINAL_TEST_STATUS
    assert terminal["test_only_fake_child"] is True
    assert terminal["fixture_or_synthetic"] is True
    assert terminal["self_asserted"] is False
    assert isinstance(terminal["lease_id"], str) and len(terminal["lease_id"]) == 64
    assert terminal["child_terminal"]["reaped"] is True
    assert terminal["child_terminal"]["terminal_receipt_written_last"] is True
    assert lease["host_binary"]["present"] is True
    assert launch["host_binary"]["present"] is True
    assert launch["lease_receipt"]["document_seal_sha256"] == lease["seal_sha256"]
    assert (config.outer_capture_dir / lifecycle.TERMINAL_FILENAME).is_file()
    assert (config.launch_dir / lifecycle.RELEASE_FILENAME).is_file()
    assert release["status"] == lifecycle.RELEASE_STATUS
    assert release["capture_succeeded"] is False
    assert release["exactly_one_release_for_this_lease"] is True
    assert release["actual_release_performed"] is True
    assert release["released_after_outer_terminal"] is True
    assert release["lease_released"] is True
    assert release["automatic_retry_prohibited"] is True
    assert release["fresh_lease_required_for_any_future_gpu_work"] is True
    assert release["watcher_restart_or_transition_authorized"] is False
    with pytest.raises(lifecycle.MultiLayerLifecycleError, match="must be a new"):
        lifecycle.run_one_shot_for_test(config, fake_child_command=(sys.executable, str(fake)))
    assert len(list(config.launch_dir.glob(lifecycle.RELEASE_FILENAME))) == 1


def test_unwired_or_wrong_mode_refuses_before_creating_any_lease_or_directory(tmp_path: Path) -> None:
    context, pins, host, outer, binary = _synthetic_context(tmp_path, capture_body_wired=False)
    config = _config(
        tmp_path, context=context, pins=pins, host=host, outer=outer, binary=binary
    )
    with pytest.raises(lifecycle.MultiLayerLifecycleError, match="capture_body_wired=false"):
        lifecycle.execute_one_shot(config)
    assert not config.launch_dir.exists()
    assert not config.replay_dir.exists()
    assert not config.outer_capture_dir.exists()
    result = lifecycle.main(
        [
            "--mode", "execute",
            "--host-preflight", str(host),
            "--outer-preflight", str(outer),
            "--host-binary", str(binary),
            "--resource-admission", str(config.resource_admission),
            "--launch-dir", str(config.launch_dir),
            "--replay-dir", str(config.replay_dir),
            "--outer-capture-dir", str(config.outer_capture_dir),
            "--out", str(tmp_path / "wrong-mode-out.json"),
            "--expected-host-binary-sha256", pins.host_binary_sha256,
            "--expected-host-preflight-seal-sha256", pins.host_preflight_seal_sha256,
            "--expected-outer-preflight-seal-sha256", pins.outer_preflight_seal_sha256,
            "--expected-execution-schedule-seal-sha256", pins.execution_schedule_seal_sha256,
            "--expected-chain-cpu-oracle-seal-sha256", pins.chain_cpu_oracle_seal_sha256,
            "--expected-l1-full-layer-assessment-seal-sha256", pins.l1_full_layer_assessment_seal_sha256,
            "--expected-joint-assessment-seal-sha256", pins.joint_assessment_seal_sha256,
        ]
    )
    assert result == 2
    assert not (tmp_path / "wrong-mode-out.json").exists()
    assert not config.launch_dir.exists()


def test_lifecycle_authority_context_binds_both_assessments(tmp_path: Path) -> None:
    context, pins, _host, _outer, _binary = _synthetic_context(tmp_path)
    assert context.l1_full_layer_assessment.seal_sha256 == pins.l1_full_layer_assessment_seal_sha256
    assert context.joint_assessment.seal_sha256 == pins.joint_assessment_seal_sha256
    assert context.l1_full_layer_assessment.document["schema"] == lifecycle.L1_ASSESSMENT_SCHEMA
    assert context.joint_assessment.document["schema"] == lifecycle.JOINT_ASSESSMENT_SCHEMA
    # Distinct documents: seals must not collapse to one pin.
    assert (
        context.l1_full_layer_assessment.seal_sha256 != context.joint_assessment.seal_sha256
    )
    resource = lifecycle.build_resource_admission(context=context, snapshot=_green_snapshot())
    assert resource["l1_full_layer_assessment"]["document_seal_sha256"] == pins.l1_full_layer_assessment_seal_sha256
    assert resource["joint_assessment"]["document_seal_sha256"] == pins.joint_assessment_seal_sha256


def test_outer_missing_joint_assessment_refuses(tmp_path: Path) -> None:
    context, pins, host, outer, binary = _synthetic_context(tmp_path)
    # Rewrite outer without joint_assessment so lifecycle refuses before any lease.
    stripped = dict(context.outer_preflight.document)
    stripped.pop("joint_assessment", None)
    stripped.pop("seal_sha256", None)
    bad_outer = _write(tmp_path / "outer-no-joint.json", seal(stripped))
    bad_pins = lifecycle.AuthorityPins(
        host_binary_sha256=pins.host_binary_sha256,
        host_preflight_seal_sha256=pins.host_preflight_seal_sha256,
        outer_preflight_seal_sha256=_evidence(bad_outer)["document_seal_sha256"],
        execution_schedule_seal_sha256=pins.execution_schedule_seal_sha256,
        chain_cpu_oracle_seal_sha256=pins.chain_cpu_oracle_seal_sha256,
        l1_full_layer_assessment_seal_sha256=pins.l1_full_layer_assessment_seal_sha256,
        joint_assessment_seal_sha256=pins.joint_assessment_seal_sha256,
    )
    with pytest.raises(
        lifecycle.MultiLayerLifecycleError,
        match="outer joint assessment must be an object|outer preflight missing joint_assessment",
    ):
        lifecycle.load_authority_context(
            host_preflight=host,
            outer_preflight=bad_outer,
            host_binary=binary,
            pins=bad_pins,
        )


def test_frozen_pin_constants_match_task_prefixes() -> None:
    assert lifecycle.EXECUTION_SCHEDULE_SEAL_SHA256.startswith("54084ddf")
    assert lifecycle.CHAIN_CPU_ORACLE_SEAL_SHA256.startswith("a217fc80")
    assert lifecycle.CURRENT_HOST_PREFLIGHT_SEAL_SHA256.startswith("bf40d5e0")
    assert lifecycle.L1_FULL_LAYER_ASSESSMENT_SEAL_SHA256.startswith("47a4f33f")
    assert lifecycle.JOINT_ASSESSMENT_SEAL_SHA256.startswith("d1b28931")
    assert lifecycle.TOTAL_DISPATCHES == 69
    assert lifecycle.LAYER_COUNT == 3
