from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from lab.operators import ascension_qwen80_l1_full_layer_capture_lifecycle as lifecycle
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPLETE = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime"
FROZEN_HOST = REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen80_source_token_l0_l1_full_layer_same_runtime_device"
FROZEN_HOST_PREFLIGHT = (
    COMPLETE / "QWEN80_L1_FULL_LAYER_HOST_CPU_PREFLIGHT_WIRED_RftkDr/l1-full-layer-host-preflight.json"
)
FROZEN_OUTER_PREFLIGHT = (
    COMPLETE / "QWEN80_L1_FULL_LAYER_OUTER_GATE_CPU_PREFLIGHT_WIRED_yM9XIZ/l1-full-layer-outer-preflight.json"
)
RAW_ROUTE_AUTHORITY = (
    COMPLETE
    / "QWEN80_SOURCE_TOKEN_L1_ROUTE_AUTHORITY_CPU_SCAN_20260809T130548Z/inner/l1-source-token-route-authority.json"
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


def _descriptor(role: str, character: str) -> dict[str, Any]:
    return {
        "role": role,
        "tensor_name": f"model.layers.1.{role}",
        "artifact_sha256": _sha(character),
        "direct_packed_payload_sha256": _sha("b"),
        "header_sha256": _sha("c"),
    }


def _green_snapshot() -> dict[str, Any]:
    return {
        "memory_free_percent": 91,
        "swap_used_bytes": 0,
        "q80_watcher_parent_pids": [22035],
        "watcher_hold_active": True,
        "q80_full_l1_capture_children": [],
        "q30_capture_children": [],
    }


def _synthetic_context(
    tmp_path: Path, *, capture_body_wired: bool = True
) -> tuple[lifecycle.AuthorityContext, lifecycle.AuthorityPins, Path, Path, Path, Path]:
    host_binary = tmp_path / "host"
    host_binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    host_binary.chmod(0o700)

    ids = list(range(10))
    weights = [0.1] * 10
    route = _write(
        tmp_path / "route.json",
        seal(
            {
                "schema": lifecycle.ROUTE_AUTHORITY_SCHEMA,
                "status": lifecycle.ROUTE_AUTHORITY_STATUS,
                "source_token_router_evidence": {
                    "source_stable_route_ids": ids,
                    "source_stable_normalized_weights": weights,
                },
                "fixed_l1_payloads": [_descriptor(f"fixed-{index}", "a") for index in range(6)],
                "deterministic_waves": [
                    {
                        "wave_index": index,
                        "layer": 1,
                        "expert_id": ids[index],
                        "gate": _descriptor("gate", "a"),
                        "up": _descriptor("up", "b"),
                        "down": _descriptor("down", "c"),
                    }
                    for index in range(10)
                ],
            }
        ),
    )
    host_raw = host_binary.read_bytes()
    host_evidence = {
        "path": str(host_binary.resolve()),
        "bytes": len(host_raw),
        "sha256": hashlib.sha256(host_raw).hexdigest(),
    }
    route_evidence = _evidence(route)
    kernel_names = [f"kernel-{index}" for index in range(lifecycle.TOTAL_DISPATCHES)]
    host = _write(
        tmp_path / "host-preflight.json",
        seal(
            {
                "schema": lifecycle.HOST_PREFLIGHT_SCHEMA,
                "status": "COMPILED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED",
                "host_binary": {"present": True, **host_evidence},
                "l1_route_payload_authority": {
                    "binding": {
                        key: route_evidence[key]
                        for key in (
                            "path",
                            "present",
                            "bytes",
                            "sha256",
                            "document_sha256",
                            "document_seal_sha256",
                        )
                    }
                },
                "future_joint_command_graph": {
                    "source_token_id": 1,
                    "l0_reencode_dispatches": 23,
                    "l1_prefix_dispatches": 9,
                    "l1_moe_suffix_dispatches": 14,
                    "total_dispatches": 46,
                    "single_fence_after_all_dispatches_required": True,
                    "non_timed_structural_trace_required": True,
                    "exact_kernel_trace": [
                        {"ordinal": index, "kernel": name}
                        for index, name in enumerate(kernel_names)
                    ],
                },
                "future_metal_entrypoint": {
                    "explicit_mode_required": True,
                    "default_execution_disabled": True,
                    "requires_new_full_l1_lease": True,
                    "requires_sealed_outer_launch_authority": True,
                    "requires_fresh_outer_and_inner_capture_directories": True,
                    "self_hashes_current_executable": True,
                    "no_device_execution_in_this_cpu_preflight": True,
                    "capture_body_wired": capture_body_wired,
                },
                "future_inner_receipt_contract": {
                    "schema": lifecycle.INNER_SCHEMA,
                    "status": lifecycle.INNER_SUCCESS_STATUS,
                },
                "claim_boundary": {
                    "catalog_or_payload_scan_performed": False,
                    "metal_context_or_dispatch_performed": False,
                    "lease_issued_or_consumed": False,
                    "watcher_server_hcli_or_runtime_changed": False,
                    "complete_layer_or_token_decoder_claim_earned": False,
                    "tps_tg_or_tournament_claim_earned": False,
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
                "status": "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_OUTER_CPU_ONLY_NOT_LEASED_OR_EXECUTED",
                "host_preflight": host_preflight_evidence,
                "host_binary": {"present": True, **host_evidence},
                "original_l1_route_authority": route_evidence,
                "exact_component_scope": {
                    "source_token_id": 1,
                    "l0_reencode_dispatches": 23,
                    "l1_prefix_dispatches": 9,
                    "l1_moe_suffix_dispatches": 14,
                    "total_dispatches": 46,
                    "one_fence_required": True,
                    "non_timed_exact_trace_required": True,
                    "kernel_names": kernel_names,
                },
                "future_metal_entrypoint": {
                    "explicit_mode_required": True,
                    "default_execution_disabled": True,
                    "requires_new_full_l1_lease": True,
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
        raw_l1_route_authority_seal_sha256=route_evidence["document_seal_sha256"],
    )
    context = lifecycle.load_authority_context(
        host_preflight=host,
        outer_preflight=outer,
        raw_l1_route_authority=route,
        host_binary=host_binary,
        pins=pins,
    )
    return context, pins, host, outer, route, host_binary


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
    route: Path,
    binary: Path,
) -> lifecycle.ExecuteConfig:
    return lifecycle.ExecuteConfig(
        host_preflight=host,
        outer_preflight=outer,
        raw_l1_route_authority=route,
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
    'schema': 'hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_capture.v1',
    'status': 'CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_COMPONENT_ONLY',
    'fixture_or_synthetic': True,
    'outer_preflight_binding': evidence(args.outer_preflight),
    'full_l1_lease_binding': {'lease_id': lease['lease_id'], 'receipt': evidence(args.lease_receipt)},
    'outer_launch_authority_binding': evidence(args.outer_launch_authority),
    'durable_capture': {
        'capture_directory': str(capture.resolve()),
        'receipt_written_last_is_completion_marker': True,
    },
    'fresh_same_runtime_execution': {
        'fresh_runtime': True, 'fresh_session': True, 'same_runtime': True, 'same_tcb': True,
        'l0_reencoded_in_this_capture': True, 'l1_prefix_and_moe_suffix_in_this_capture': True,
        'route_guard_enforced_before_l1_moe_suffix': True, 'source_token_id': 1,
        'l0_dispatches': 23, 'l1_prefix_dispatches': 9, 'l1_moe_suffix_dispatches': 14,
        'total_dispatches': 46, 'fence_count': 1,
    },
    'structural_kernel_trace': {
        'non_timed': True, 'exact_order': True,
        'kernel_names': outer['exact_component_scope']['kernel_names'],
    },
    'single_fence': {
        'only_command_buffer_consumed': True, 'fence_succeeded': True,
        'readbacks_after_fence': True, 'append_after_fence_possible': False, 'fence_count': 1,
    },
    'l1_completion_readbacks': {
        key: {} for key in ('input', 'prefix_first_residual', 'postnorm', 'router_logits',
        'shared_output', 'routed_sum', 'second_residual_output', 'active_conv',
        'active_recurrent', 'rollback_conv', 'rollback_recurrent')
    },
    'claim_boundary': {
        'complete_l1_component_only': True, 'token_generated': False, 'decoder_started': False,
        'server_or_watcher_started': False, 'tps_or_tg_measured': False,
        'tournament_started': False, 'next_layer_executed': False,
    },
})
(capture / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True), encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )
    return path


def test_superseded_host_preflight_refuses_against_the_current_frozen_pins() -> None:
    # This historical preflight predates the canonical route-authority byte
    # binding.  The controller must reject it before a lease can be issued.
    with pytest.raises(lifecycle.FullL1LifecycleError, match="host preflight seal does not match the explicit frozen pin"):
        lifecycle.load_authority_context(
            host_preflight=FROZEN_HOST_PREFLIGHT,
            outer_preflight=FROZEN_OUTER_PREFLIGHT,
            raw_l1_route_authority=RAW_ROUTE_AUTHORITY,
            host_binary=FROZEN_HOST,
        )


def test_resource_admission_requires_zero_swap_80_percent_watcher_and_no_q30_q80_child(
    tmp_path: Path,
) -> None:
    context, _pins, _host, _outer, _route, _binary = _synthetic_context(tmp_path)
    blocked = _green_snapshot()
    blocked["memory_free_percent"] = 79
    blocked["swap_used_bytes"] = 1
    blocked["q80_watcher_parent_pids"] = []
    blocked["q30_capture_children"] = [{"pid": 10}]
    blocked["q80_full_l1_capture_children"] = [{"pid": 11}]
    result = lifecycle.build_resource_admission(context=context, snapshot=blocked)
    assert verify(result) == result
    assert result["status"] == lifecycle.RESOURCE_REFUSED_STATUS
    assert result["prepared"] is False
    assert "memory free percentage is below 80" in result["blockers"]
    assert "swap must be exactly zero" in result["blockers"]
    assert "exactly one held Q80 watcher parent is required" in result["blockers"]
    assert "Q30 capture child is already active" in result["blockers"]
    assert "Q80 full-L1 capture child is already active" in result["blockers"]


def test_fake_child_lifecycle_writes_terminal_then_exactly_one_release_and_refuses_replay(
    tmp_path: Path,
) -> None:
    context, pins, host, outer, route, binary = _synthetic_context(tmp_path)
    config = _config(
        tmp_path, context=context, pins=pins, host=host, outer=outer, route=route, binary=binary
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
    assert terminal["child_terminal"]["reaped"] is True
    assert terminal["child_terminal"]["terminal_receipt_written_last"] is True
    # This is the exact Rust host ABI that rejected the first terminally
    # released attempt when the presence bit was omitted.
    assert lease["host_binary"]["present"] is True
    assert launch["host_binary"]["present"] is True
    assert launch["lease_receipt"]["document_seal_sha256"] == lease["seal_sha256"]
    assert (config.outer_capture_dir / lifecycle.TERMINAL_FILENAME).is_file()
    assert (config.launch_dir / lifecycle.RELEASE_FILENAME).is_file()
    assert release["status"] == lifecycle.RELEASE_STATUS
    assert release["capture_succeeded"] is False
    assert release["exactly_one_release_for_this_lease"] is True
    with pytest.raises(FullL1LifecycleError := lifecycle.FullL1LifecycleError, match="must be a new"):
        lifecycle.run_one_shot_for_test(config, fake_child_command=(sys.executable, str(fake)))
    assert len(list(config.launch_dir.glob(lifecycle.RELEASE_FILENAME))) == 1


def test_unwired_or_wrong_mode_refuses_before_creating_any_lease_or_directory(tmp_path: Path) -> None:
    context, pins, host, outer, route, binary = _synthetic_context(tmp_path, capture_body_wired=False)
    config = _config(
        tmp_path, context=context, pins=pins, host=host, outer=outer, route=route, binary=binary
    )
    with pytest.raises(lifecycle.FullL1LifecycleError, match="capture_body_wired=false"):
        lifecycle.execute_one_shot(config)
    assert not config.launch_dir.exists()
    assert not config.replay_dir.exists()
    assert not config.outer_capture_dir.exists()
    result = lifecycle.main(
        [
            "--mode", "execute",
            "--host-preflight", str(host),
            "--outer-preflight", str(outer),
            "--raw-l1-route-authority", str(route),
            "--host-binary", str(binary),
            "--resource-admission", str(config.resource_admission),
            "--launch-dir", str(config.launch_dir),
            "--replay-dir", str(config.replay_dir),
            "--outer-capture-dir", str(config.outer_capture_dir),
            "--out", str(tmp_path / "wrong-mode-out.json"),
            "--expected-host-binary-sha256", pins.host_binary_sha256,
            "--expected-host-preflight-seal-sha256", pins.host_preflight_seal_sha256,
            "--expected-outer-preflight-seal-sha256", pins.outer_preflight_seal_sha256,
            "--expected-raw-l1-route-authority-seal-sha256", pins.raw_l1_route_authority_seal_sha256,
        ]
    )
    assert result == 2
    assert not (tmp_path / "wrong-mode-out.json").exists()
    assert not config.launch_dir.exists()
