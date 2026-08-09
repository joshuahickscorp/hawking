from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from lab.operators import ascension_qwen80_l1_full_layer_same_runtime_outer as outer
from lab.receipts import seal, verify


def _sha(character: str) -> str:
    return character * 64


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def _descriptor(role: str, character: str) -> dict[str, Any]:
    return {
        "role": role,
        "tensor_name": f"model.layers.1.{role}.{character}",
        "artifact_sha256": _sha(character),
        "direct_packed_payload_sha256": _sha("b"),
        "header_sha256": _sha("c"),
    }


def _route_authority() -> dict[str, Any]:
    ids = list(range(10))
    weights = [0.01 * (index + 1) for index in range(10)]
    return seal(
        {
            "schema": outer.L1_ROUTE_AUTHORITY_SCHEMA,
            "status": outer.L1_ROUTE_AUTHORITY_STATUS,
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
                    "normalized_weight": weights[index],
                    "gate": _descriptor("gate", "a"),
                    "up": _descriptor("up", "b"),
                    "down": _descriptor("down", "c"),
                }
                for index in range(10)
            ],
        }
    )


def _plain(schema: str, status: str) -> dict[str, Any]:
    return seal({"schema": schema, "status": status})


def _binding(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "present": True,
        "bytes": len(raw),
        "sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "document_sha256": document["seal_sha256"],
        "document_seal_sha256": document["seal_sha256"],
    }


def _inputs(tmp_path: Path) -> tuple[outer.OuterInputs, Path]:
    host_binary = tmp_path / "host"
    host_binary.write_bytes(b"fake-current-host")
    host_binary.chmod(0o700)

    joint = _write(
        tmp_path / "joint.json",
        _plain(outer.JOINT_ASSESSMENT_SCHEMA, outer.JOINT_ASSESSMENT_STATUS),
    )
    route = _write(tmp_path / "route.json", _route_authority())
    route_bound = _binding(route)
    recovery = _write(
        tmp_path / "route-recovery.json",
        seal(
            {
                "schema": outer.L1_ROUTE_AUTHORITY_RECOVERY_SCHEMA,
                "status": outer.L1_ROUTE_AUTHORITY_RECOVERY_STATUS,
                "historical_inner_authority": {
                    **route_bound,
                    "schema": outer.L1_ROUTE_AUTHORITY_SCHEMA,
                    "status": outer.L1_ROUTE_AUTHORITY_STATUS,
                },
                "downstream_authority": {
                    "consume_historical_inner_directly": True,
                    "recovery_wrapper_is_not_a_dynamic_route_authority_substitute": True,
                    "authority_path": route_bound["path"],
                    "authority_document_sha256": route_bound["document_sha256"],
                    "authority_seal_sha256": route_bound["document_seal_sha256"],
                    "authority_schema": outer.L1_ROUTE_AUTHORITY_SCHEMA,
                    "authority_status": outer.L1_ROUTE_AUTHORITY_STATUS,
                },
                "canonicalization": {
                    "historical_inner_validated_against_reaped_identity_chain": True,
                    "historical_outer_remains_refused": True,
                    "historical_outer_status_relabelled": False,
                    "no_new_scan_or_child": True,
                    "static_downstream_contract_valid": True,
                    "downstream_authority_is_historical_inner": True,
                },
            }
        ),
    )
    completion = _write(
        tmp_path / "completion.json",
        _plain(outer.COMPLETION_PREFLIGHT_SCHEMA, outer.COMPLETION_PREFLIGHT_STATUS),
    )
    l0_outer = _write(
        tmp_path / "l0.json",
        _plain(outer.L0_OUTER_PREFLIGHT_SCHEMA, outer.L0_OUTER_PREFLIGHT_STATUS),
    )
    route_doc = json.loads(route.read_text(encoding="utf-8"))
    route_evidence = route_doc["source_token_router_evidence"]
    host = seal(
        {
            "schema": outer.HOST_PREFLIGHT_SCHEMA,
            "status": outer.HOST_PREFLIGHT_STATUS,
            "host_binary": {
                "path": str(host_binary.resolve()),
                "present": True,
                "bytes": host_binary.stat().st_size,
                "sha256": __import__("hashlib").sha256(host_binary.read_bytes()).hexdigest(),
            },
            "joint_assessment": _binding(joint),
            "completion_preflight": _binding(completion),
            "l0_source_outer_preflight": _binding(l0_outer),
            "l1_route_payload_authority": {
                "schema": outer.L1_ROUTE_AUTHORITY_SCHEMA,
                "status": outer.L1_ROUTE_AUTHORITY_STATUS,
                "binding": _binding(route),
                "source_stable_route_ids": route_evidence["source_stable_route_ids"],
                "source_stable_normalized_weights": route_evidence[
                    "source_stable_normalized_weights"
                ],
                "distinct_payload_bindings": 36,
                "route_guard_required_value": 1,
                "six_fixed_payloads": route_doc["fixed_l1_payloads"],
                "ten_ordered_waves": route_doc["deterministic_waves"],
            },
            "future_same_runtime_host_interface": {
                "consuming_finalizer": "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::"
                "finalize_after_exact_l1_moe_completion_fence_with_readbacks",
                "receipt_last_required": True,
                "fresh_runtime_required": True,
                "same_runtime_required": True,
                "same_token_command_buffer_required": True,
                "single_fence_required": True,
                "readbacks_after_fence_required": True,
                "cross_process_pinned_buffer_or_state_import_allowed": False,
            },
            "future_metal_entrypoint": {
                "explicit_mode_required": True,
                "default_execution_disabled": True,
                "requires_new_full_l1_lease": True,
                "requires_sealed_outer_launch_authority": True,
                "requires_fresh_outer_and_inner_capture_directories": True,
                "self_hashes_current_executable": True,
                "no_device_execution_in_this_cpu_preflight": True,
                "capture_body_wired": True,
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
                    {"ordinal": index, "kernel": kernel}
                    for index, kernel in enumerate(outer.EXACT_KERNELS)
                ],
            },
            "future_inner_receipt_contract": {
                "schema": outer.INNER_SCHEMA,
                "status": outer.INNER_STATUS,
                "outer_schema": outer.OUTER_CAPTURE_SCHEMA,
                "outer_status": outer.OUTER_CAPTURE_STATUS,
                "requires_distinct_cpu_and_device_hashes_with_bounded_numeric_parity": True,
                "requires_l1_route_guard_all_ten_shared_routed_sum_and_second_residual_readbacks": True,
                "requires_l0_and_l1_active_rollback_state_witnesses": True,
            },
            "claim_boundary": {
                "cpu_build_preflight_only": True,
                "catalog_or_payload_scan_performed": False,
                "metal_context_or_dispatch_performed": False,
                "lease_issued_or_consumed": False,
                "watcher_server_hcli_or_runtime_changed": False,
                "complete_layer_or_token_decoder_claim_earned": False,
                "tps_tg_or_tournament_claim_earned": False,
            },
        }
    )
    host_path = _write(tmp_path / "host-preflight.json", host)
    return (
        outer.OuterInputs(
            host_preflight=host_path,
            host_binary=host_binary,
            joint_assessment=joint,
            l1_route_authority=route,
            l1_route_authority_recovery_provenance=recovery,
            completion_preflight=completion,
            l0_source_outer_preflight=l0_outer,
        ),
        host_binary,
    )


def _fake_inner(outer_preflight: dict[str, Any]) -> dict[str, Any]:
    ids = outer_preflight["exact_component_scope"]["route_ids"]
    weights = outer_preflight["exact_component_scope"]["normalized_route_weights"]
    return seal(
        {
            "schema": outer.INNER_SCHEMA,
            "status": outer.INNER_STATUS,
            "historical_component_provenance": {
                "present": True,
                "document_sha256": outer_preflight["joint_assessment"]["document_sha256"],
                "document_seal_sha256": outer_preflight["joint_assessment"][
                    "document_seal_sha256"
                ],
            },
            "fresh_same_runtime_execution": {
                "fresh_runtime": True,
                "fresh_session": True,
                "same_runtime": True,
                "same_tcb": True,
                "l0_reencoded_in_this_capture": True,
                "l1_prefix_and_moe_suffix_in_this_capture": True,
                "route_guard_enforced_before_l1_moe_suffix": True,
                "source_token_id": 1,
                "l0_dispatches": 23,
                "l1_prefix_dispatches": 9,
                "l1_moe_suffix_dispatches": 14,
                "total_dispatches": 46,
                "fence_count": 1,
                "runtime_identity_sha256": _sha("a"),
                "tcb_identity_sha256": _sha("b"),
            },
            "structural_kernel_trace": {
                "non_timed": True,
                "exact_order": True,
                "kernel_names": list(outer.EXACT_KERNELS),
            },
            "single_fence": {
                "only_command_buffer_consumed": True,
                "fence_succeeded": True,
                "readbacks_after_fence": True,
                "append_after_fence_possible": False,
                "fence_count": 1,
            },
            "l1_route_payload_authority": {
                "route_guard": {"passed": True, "value": 1},
                "route_payloads": [
                    {"route_index": route, "payload_kind": kind}
                    for route in range(10)
                    for kind in ("gate", "up", "down")
                ],
            },
            "l1_completion_readbacks": {
                "layer": 1,
                "slot": 1,
                "output_elements": 2048,
                "output_bytes": 8192,
                **{field: {} for field in (
                    "input", "prefix_first_residual", "postnorm", "router_logits", "shared_output",
                    "routed_sum", "second_residual_output", "active_conv", "active_recurrent",
                    "rollback_conv", "rollback_recurrent",
                )},
            },
            "claim_boundary": {
                "complete_l1_component_only": True,
                "token_generated": False,
                "decoder_started": False,
                "server_or_watcher_started": False,
                "tps_or_tg_measured": False,
                "tournament_started": False,
                "next_layer_executed": False,
            },
        }
    )


def test_outer_preflight_is_exact_host_bound_and_cpu_file_only(tmp_path: Path) -> None:
    inputs, _ = _inputs(tmp_path)
    document = outer.build_outer_preflight(inputs)
    assert verify(document) == document
    assert document["status"] == outer.OUTER_PREFLIGHT_STATUS
    assert document["exact_component_scope"]["total_dispatches"] == 46
    assert len(document["exact_component_scope"]["kernel_names"]) == 46
    assert document["future_metal_entrypoint"]["capture_body_wired"] is True
    assert document["lifecycle"]["real_host_metal_cli_available"] is True
    assert document["claim_boundary"]["metal_context_or_dispatch_performed"] is False


def test_outer_refuses_binary_drift(tmp_path: Path) -> None:
    inputs, host_binary = _inputs(tmp_path)
    host_binary.write_bytes(b"mutated")
    with pytest.raises(outer.FullL1OuterError, match="current binary"):
        outer.build_outer_preflight(inputs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-route-bytes", "route authority binding drifted"),
        ("mismatched-route-bytes", "route authority binding drifted"),
        ("missing-host-present", "current binary"),
    ],
)
def test_outer_refuses_incomplete_host_file_evidence(
    tmp_path: Path, mutation: str, message: str
) -> None:
    inputs, _ = _inputs(tmp_path)
    host = json.loads(inputs.host_preflight.read_text(encoding="utf-8"))
    host.pop("seal_sha256")
    if mutation == "missing-route-bytes":
        host["l1_route_payload_authority"]["binding"].pop("bytes")
    elif mutation == "mismatched-route-bytes":
        host["l1_route_payload_authority"]["binding"]["bytes"] += 1
    elif mutation == "missing-host-present":
        host["host_binary"].pop("present")
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)
    _write(inputs.host_preflight, seal(host))
    with pytest.raises(outer.FullL1OuterError, match=message):
        outer.build_outer_preflight(inputs)


def test_outer_refuses_recovery_wrapper_that_substitutes_a_different_inner(tmp_path: Path) -> None:
    inputs, _ = _inputs(tmp_path)
    recovery = json.loads(inputs.l1_route_authority_recovery_provenance.read_text(encoding="utf-8"))
    recovery["historical_inner_authority"]["document_sha256"] = _sha("f")
    recovery = seal({key: value for key, value in recovery.items() if key != "seal_sha256"})
    inputs.l1_route_authority_recovery_provenance.write_text(
        json.dumps(recovery, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(outer.FullL1OuterError, match="recovery historical inner.document_sha256"):
        outer.build_outer_preflight(inputs)


def test_fake_reaper_requires_exact_receipt_and_writes_terminal_last(tmp_path: Path) -> None:
    inputs, _ = _inputs(tmp_path)
    preflight = outer.build_outer_preflight(inputs)
    preflight_path = tmp_path / "outer-preflight.json"
    _write(preflight_path, preflight)
    inner = _fake_inner(preflight)
    child = tmp_path / "fake-child.py"
    child.write_text(
        "import argparse, json\n"
        "p=argparse.ArgumentParser(); p.add_argument('--out', required=True); a=p.parse_args()\n"
        f"open(a.out, 'w').write({json.dumps(json.dumps(inner))})\n",
        encoding="utf-8",
    )
    terminal = outer.reap_fake_child_for_test(
        outer_preflight=preflight_path,
        fake_child_command=(sys.executable, str(child)),
        capture_dir=tmp_path / "capture",
    )
    assert verify(terminal) == terminal
    assert terminal["child_terminal"]["reaped"] is True
    assert (tmp_path / "capture" / "outer-terminal-receipt.json").exists()


def test_fake_reaper_refuses_trace_drift(tmp_path: Path) -> None:
    inputs, _ = _inputs(tmp_path)
    preflight = outer.build_outer_preflight(inputs)
    preflight_path = tmp_path / "outer-preflight.json"
    _write(preflight_path, preflight)
    inner = _fake_inner(preflight)
    inner["structural_kernel_trace"]["kernel_names"][-1] = "wrong"
    inner = seal({key: value for key, value in inner.items() if key != "seal_sha256"})
    child = tmp_path / "fake-child.py"
    child.write_text(
        "import argparse, json\n"
        "p=argparse.ArgumentParser(); p.add_argument('--out', required=True); a=p.parse_args()\n"
        f"open(a.out, 'w').write({json.dumps(json.dumps(inner))})\n",
        encoding="utf-8",
    )
    with pytest.raises(outer.FullL1OuterError, match="trace"):
        outer.reap_fake_child_for_test(
            outer_preflight=preflight_path,
            fake_child_command=(sys.executable, str(child)),
            capture_dir=tmp_path / "capture",
        )
    assert not (tmp_path / "capture" / "outer-terminal-receipt.json").exists()
