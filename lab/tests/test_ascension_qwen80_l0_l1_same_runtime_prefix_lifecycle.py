from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from lab.operators import ascension_qwen80_l0_l1_same_runtime_prefix_lifecycle as lifecycle
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPLETE = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime"
GRAVITY = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-gravity"
WATCHER_HOLD = COMPLETE / "QWEN80_WATCHER_GPU_COORDINATION_HOLD_20260808T220751Z.json"


def _sha(character: str) -> str:
    return character * 64


def _paths() -> lifecycle.PreflightPaths:
    return lifecycle.PreflightPaths(
        continuation_readiness=COMPLETE / "QWEN80_L1_SOURCE_TOKEN_CONTINUATION_READINESS_WITH_SEALED_SCHEDULE_20260809T084000Z.json",
        l0_outer_terminal=COMPLETE / "QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_CAPTURE_20260809T081620Z/outer-terminal-receipt.json",
        l0_inner_capture=COMPLETE / "QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_CAPTURE_20260809T081620Z/inner/receipt.json",
        assessor_binding=COMPLETE / "QWEN80_L0_STATE_HANDOFF_POST_CAPTURE_ASSESSOR_BINDING_20260809T085000Z.json",
        post_capture_assessment=COMPLETE / "QWEN80_L0_STATE_HANDOFF_POST_CAPTURE_ASSESSMENT_20260809T083200Z.json",
        prior_lease_release=COMPLETE / "QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_QUIET_METAL_LEASE_RELEASE_20260809T081925Z.json",
        manifest=GRAVITY / "QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json",
        admission_receipt=GRAVITY / "complete-admission/receipts/QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT_14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b.json",
        schedule=COMPLETE / "QWEN80_48_LAYER_PAYLOAD_SCHEDULE_SEALED_WRAPPER_20260809T083400Z.json",
        joint_static_plan=COMPLETE / "QWEN80_L0_L1_STRICT_HOST_INTERFACE_STATIC_PLAN_20260809T114151Z/joint-child-static-plan.json",
        l0_source_outer_preflight=COMPLETE / "QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_CPU_PREFLIGHT_20260809T060500Z/outer-preflight.json",
        joint_host_preflight=COMPLETE / "QWEN80_L0_L1_STRICT_HOST_INTERFACE_CPU_PREFLIGHT_20260809T114204Z/host-preflight.json",
    )


@pytest.fixture(scope="module")
def live_preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return lifecycle.build_cpu_preflight(_paths())


def _parity(character: str) -> dict[str, Any]:
    return {
        "passed": True,
        "cpu_f32le_sha256": _sha(character),
        "device_f32le_sha256": _sha(character),
        "max_abs_error": 0.0,
    }


def _state(character: str, *, slot: int, offset: int, capacity: int) -> dict[str, Any]:
    return {
        "passed": True,
        "slot": slot,
        "offset_bytes": offset,
        "capacity_bytes": capacity,
        "device_buffer_identity_sha256": _sha(character),
        "f32le_sha256": _sha("f" if character != "f" else "e"),
        "max_abs_error": 0.0,
    }


def _inner_document(outer: dict[str, Any], lease: lifecycle.LeaseContext) -> dict[str, Any]:
    chain = outer["authority_chain"]
    scope = outer["exact_joint_scope"]
    ids = scope["route_ids"]
    weights = scope["normalized_route_weights"]
    routes = [
        {
            "wave_index": index,
            "expert_id": expert_id,
            "passed": True,
            "f32le_sha256": _sha("abcdef"[index % 6]),
            "cpu_f32le_sha256": _sha("fedcba"[index % 6]),
            "max_abs_error": 0.0,
        }
        for index, expert_id in enumerate(ids)
    ]
    return seal(
        {
            "schema": lifecycle.INNER_SCHEMA,
            "status": lifecycle.INNER_STATUS,
            "fixture_or_synthetic": False,
            "self_asserted": False,
            "issuer": {
                "role": "joint_component_capture_child",
                "issuer_identity_sha256": _sha("a"),
            },
            "upstream_authorities": {
                "schedule_wrapper": {
                    "present": True,
                    "document_sha256": chain["schedule"]["document_sha256"],
                    "document_seal_sha256": chain["schedule"]["document_seal_sha256"],
                },
                "continuation": {
                    "present": True,
                    "document_sha256": chain["continuation_readiness"]["document_sha256"],
                    "document_seal_sha256": chain["continuation_readiness"]["document_seal_sha256"],
                },
                "assessor_binding": {
                    "present": True,
                    "document_sha256": chain["l0_post_capture_assessor_binding"]["document_sha256"],
                    "document_seal_sha256": chain["l0_post_capture_assessor_binding"]["document_seal_sha256"],
                },
            },
            "opaque_l0_continuation": {
                "factory": lifecycle.CAPABILITY_FACTORY,
                "l1_encoder": lifecycle.L1_ENCODER,
                "consuming_finalizer": lifecycle.FINALIZER,
                "opaque": True,
                "freshly_derived_from_l0_23_dispatch_graph": True,
                "same_runtime_state_arena_bound": True,
                "same_command_buffer_bound": True,
                "non_transferable_across_processes": True,
                "raw_pinned_buffer_or_dispatch_count_input_accepted": False,
                "capability_identity_sha256": _sha("b"),
                "runtime_identity_sha256": _sha("c"),
                "runtime_state_arena_identity_sha256": _sha("d"),
                "command_buffer_identity_sha256": _sha("e"),
            },
            "fresh_joint_execution": {
                "fresh_runtime": True,
                "fresh_session": True,
                "same_runtime": True,
                "same_tcb": True,
                "structural_trace_non_timed": True,
                "route_guard_enforced_before_l1": True,
                "runtime_identity_sha256": _sha("c"),
                "session_identity_sha256": _sha("f"),
                "tcb_identity_sha256": _sha("e"),
                "source_token_id": lifecycle.SOURCE_TOKEN_ID,
                "l0_dispatches": lifecycle.L0_DISPATCHES,
                "l1_prefix_dispatches": lifecycle.L1_DISPATCHES,
                "total_dispatches": lifecycle.TOTAL_DISPATCHES,
                "fence_count": 1,
            },
            "structural_kernel_trace": {
                "non_timed": True,
                "exact_order": True,
                "kernel_names": list(lifecycle.STRUCTURAL_KERNELS),
            },
            "single_fence": {
                "consuming_finalizer": lifecycle.FINALIZER,
                "only_command_buffer_consumed": True,
                "fence_succeeded": True,
                "readbacks_after_fence": True,
                "append_after_fence_possible": False,
                "fence_count": 1,
            },
            "fresh_readbacks": {
                "l0_suffix": {
                    "route_guard": {
                        "passed": True,
                        "value": 1,
                        "expected_route_ids": ids,
                        "observed_route_ids": ids,
                        "expected_route_weights": weights,
                        "observed_route_weights": weights,
                        "weights_max_abs_error": 0.0,
                    },
                    "postnorm": _parity("a"),
                    "router_logits": _parity("b"),
                    "all_ten_weighted_route_witnesses": routes,
                    "shared_output": _parity("c"),
                    "routed_sum": _parity("d"),
                    "second_residual": _parity("e"),
                },
                "fresh_l0_state": {
                    "active_conv": _state("a", slot=0, offset=0, capacity=lifecycle.L0_CONV_BYTES),
                    "active_recurrent": _state("b", slot=0, offset=0, capacity=lifecycle.L0_RECURRENT_BYTES),
                    "rollback_conv": _state("c", slot=0, offset=0, capacity=lifecycle.L0_CONV_BYTES),
                    "rollback_recurrent": _state("d", slot=0, offset=0, capacity=lifecycle.L0_RECURRENT_BYTES),
                },
                "fresh_l1_slot1": {
                    "layer": 1,
                    "linear_state_slot": 1,
                    "output_elements": lifecycle.HIDDEN_ELEMENTS,
                    "output_bytes": lifecycle.HIDDEN_BYTES,
                    "input": _parity("e"),
                    "first_residual_output": _parity("f"),
                    "active_conv": _state("a", slot=1, offset=lifecycle.L0_CONV_BYTES, capacity=lifecycle.L1_CONV_CAPACITY_BYTES),
                    "active_recurrent": _state("b", slot=1, offset=lifecycle.L0_RECURRENT_BYTES, capacity=lifecycle.L1_RECURRENT_CAPACITY_BYTES),
                    "rollback_conv": _state("c", slot=1, offset=lifecycle.L0_CONV_BYTES, capacity=lifecycle.L1_CONV_CAPACITY_BYTES),
                    "rollback_recurrent": _state("d", slot=1, offset=lifecycle.L0_RECURRENT_BYTES, capacity=lifecycle.L1_RECURRENT_CAPACITY_BYTES),
                },
            },
            "outer_launch_authority_binding": {
                "path": "/tmp/fake-joint-outer-launch-authority.json",
                "bytes": 1,
                "sha256": _sha("a"),
                "document_sha256": _sha("b"),
                "document_seal_sha256": _sha("c"),
            },
            "joint_outer_preflight_binding": {
                "path": str(lease.outer_preflight.path),
                "bytes": lease.outer_preflight.bytes,
                "sha256": lease.outer_preflight.raw_sha256,
                "document_sha256": lease.outer_preflight.document_sha256,
                "document_seal_sha256": lease.outer_preflight.document_seal_sha256,
            },
            "joint_lease_binding": {
                "lease_id": lease.lease_id,
                "receipt": {
                    "path": str(lease.lease.path),
                    "bytes": lease.lease.bytes,
                    "sha256": lease.lease.raw_sha256,
                    "document_sha256": lease.lease.document_sha256,
                    "document_seal_sha256": lease.lease.document_seal_sha256,
                },
            },
            "execution_phase": {
                "strict_artifact_admission_started": True,
                "strict_artifact_admission_succeeded": True,
                "metal_context_construction_attempted": True,
                "metal_context_constructed": True,
                "structural_kernel_trace_enabled": True,
                "dispatches_encoded": lifecycle.TOTAL_DISPATCHES,
                "encoded_kernel_names": list(lifecycle.STRUCTURAL_KERNELS),
                "command_commit_may_have_been_attempted": True,
                "command_fence_succeeded": True,
                "readback_started": True,
                "device_dispatch_may_have_occurred": True,
            },
            "durable_capture": {
                "capture_directory": "/tmp/fake-joint-inner",
                "receipt_written_last_is_completion_marker": True,
                "outer_reaped_capture_required": True,
                "replay_guarded": True,
            },
            "claim_boundary": {
                "component_only": True,
                "l1_suffix_or_moe_executed": False,
                "complete_layer_executed": False,
                "token_generated": False,
                "decoder_started": False,
                "server_or_watcher_started": False,
            },
        }
    )


def _execution_binding(outer: dict[str, Any]) -> dict[str, Any]:
    host = REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen80_source_token_l0_l1_same_runtime_prefix_device"
    raw = host.read_bytes()
    return seal(
        {
            "schema": lifecycle.EXECUTION_BINDING_SCHEMA,
            "status": lifecycle.EXECUTION_BINDING_STATUS,
            "metal_entrypoint_available": True,
            "writes_assessor_compatible_inner_receipt": True,
            "outer_reaped_receipt_last_required": True,
            "non_timed_exact_32_dispatches_required": True,
            "host_binary": {
                "path": str(host),
                "bytes": len(raw),
                "sha256": outer["exact_joint_scope"]["host_binary_sha256"],
            },
            "outer_preflight": {
                "document_sha256": lifecycle.independent._sha256(outer),
                "document_seal_sha256": outer["seal_sha256"],
            },
            "execution_policy": {
                "source_token_id": lifecycle.SOURCE_TOKEN_ID,
                "l0_dispatches": lifecycle.L0_DISPATCHES,
                "l1_prefix_dispatches": lifecycle.L1_DISPATCHES,
                "total_dispatches": lifecycle.TOTAL_DISPATCHES,
                "strict_math": True,
                "non_timed": True,
                "single_fence_required": True,
                "tcb_trace_mode": "off",
                "l1_suffix_or_moe_authorized": False,
                "complete_layer_or_token_authorized": False,
                "server_hcli_tps_or_tournament_authorized": False,
            },
            "receipt_contract": {
                "schema": lifecycle.INNER_SCHEMA,
                "status": lifecycle.INNER_STATUS,
                "receipt_written_last": True,
                "phase_accurate_terminal_refusal_required": True,
                "opaque_same_runtime_continuation_required": True,
            },
            "claim_boundary": {"test_fixture_only": True, "no_device_action_performed_by_binding": True},
        }
    )


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


def test_live_chain_cpu_preflight_binds_exact_current_host_and_authorities(
    live_preflight: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    request, result, outer = live_preflight
    assert verify(request) == request
    assert verify(result) == result
    assert verify(outer) == outer
    assert result["prepared"] is True
    assert result["blockers"] == []
    assert len(result["seal_sha256"]) == 64
    assert outer["status"] == lifecycle.OUTER_PREFLIGHT_STATUS
    assert outer["authority_chain"]["joint_l0_l1_child_preflight"]["document_seal_sha256"] == "c03b548ecb16935273957902554ab50427b80c79781c9f2e57ed28577f810eb3"
    assert outer["authority_chain"]["joint_l0_l1_host_preflight"]["document_seal_sha256"] == "890f71f94ba48db5fd6d06242d691f88700e062620c654dfcdb98e8d99a4f20c"
    assert outer["host_execution_interface"]["metal_entrypoint_available"] is True
    assert outer["exact_joint_scope"]["total_dispatches"] == 32


def test_prepare_execution_binding_is_cpu_file_only_and_exact_host_bound(
    tmp_path: Path, live_preflight: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    _, _, outer = live_preflight
    outer_path = tmp_path / "outer-preflight.json"
    _write_json(outer_path, outer)
    host = REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen80_source_token_l0_l1_same_runtime_prefix_device"
    binding = lifecycle.prepare_execution_binding(
        outer_preflight=outer_path,
        host_binary=host,
        out=tmp_path / "execution-binding.json",
    )
    assert verify(binding.document) == binding.document
    assert binding.document["host_binary"]["sha256"] == outer["exact_joint_scope"]["host_binary_sha256"]
    assert binding.document["claim_boundary"]["metal_or_gpu_activity_performed"] is False
    assert binding.document["execution_policy"]["total_dispatches"] == 32


def test_fake_child_reaper_serializer_replay_release_and_cross_language_assessor(
    tmp_path: Path, live_preflight: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    _, _, outer = live_preflight
    outer_path = tmp_path / "outer-preflight.json"
    _write_json(outer_path, outer)
    execution = _execution_binding(outer)
    execution_path = tmp_path / "execution-binding.json"
    _write_json(execution_path, execution)
    lease_path = tmp_path / "lease.json"
    context = lifecycle.issue_lease(
        outer_preflight=outer_path,
        execution_binding=execution_path,
        watcher_hold=WATCHER_HOLD,
        out=lease_path,
    )
    assert verify(context.lease.document) == context.lease.document
    capture_dir = tmp_path / "capture"
    inner = _inner_document(outer, context)
    inner_template = tmp_path / "inner-template.json"
    _write_json(inner_template, inner)
    fake_child = tmp_path / "fake_child.py"
    fake_child.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "capture=Path(sys.argv[1]); template=Path(sys.argv[2])\n"
        "dest=capture/'inner'; dest.mkdir(parents=True,exist_ok=True)\n"
        "(dest/'receipt.json').write_text(template.read_text(),encoding='utf-8')\n"
        "print('fake joint child wrote sealed receipt')\n",
        encoding="utf-8",
    )
    receipt = lifecycle.run_one_shot_for_test(
        lifecycle.CaptureConfig(
            lease_receipt=lease_path,
            outer_preflight=outer_path,
            execution_binding=execution_path,
            watcher_hold=WATCHER_HOLD,
            capture_dir=capture_dir,
            timeout_seconds=5.0,
            workers=1,
            child_command=(sys.executable, str(fake_child), str(capture_dir), str(inner_template)),
        )
    )
    assert verify(receipt) == receipt
    assert receipt["status"] == lifecycle.OUTER_STATUS
    assert receipt["child_terminal"]["reaped"] is True
    assert receipt["child_terminal"]["terminal_receipt_written_last"] is True
    assert lifecycle.run_one_shot_for_test(
        lifecycle.CaptureConfig(
            lease_receipt=lease_path,
            outer_preflight=outer_path,
            execution_binding=execution_path,
            watcher_hold=WATCHER_HOLD,
            capture_dir=capture_dir,
            timeout_seconds=5.0,
            workers=1,
            child_command=(sys.executable, str(fake_child), str(capture_dir), str(inner_template)),
        )
    ) == receipt
    release_path = tmp_path / "release.json"
    release = lifecycle.release_after_terminal_for_test(
        outer_terminal=capture_dir / lifecycle.OUTER_TERMINAL_FILENAME,
        lease_receipt=lease_path,
        out=release_path,
        release_issuer_identity_sha256=_sha("d"),
    )
    assert verify(release) == release
    assert release["lease_id"] == context.lease_id

    assessor_binary = REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen80_l0_l1_joint_post_capture_assessor"
    if not assessor_binary.is_file():
        pytest.skip("build the joint assessor binary to run the cross-language seal check")
    schedule = json.loads(_paths().schedule.read_text(encoding="utf-8"))
    continuation = json.loads(_paths().continuation_readiness.read_text(encoding="utf-8"))
    binding = json.loads(_paths().assessor_binding.read_text(encoding="utf-8"))
    def bound(document: dict[str, Any]) -> dict[str, Any]:
        return {
            "document": document,
            "document_sha256": lifecycle.independent._sha256(document),
            "document_seal_sha256": document["seal_sha256"],
        }
    assessor_input = seal(
        {
            "schema": "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessor_input.v1",
            "schedule_wrapper": bound(schedule),
            "continuation": bound(continuation),
            "assessor_binding": bound(binding),
            "joint_inner_capture": bound(inner),
            "joint_outer_terminal": bound(receipt),
            "joint_lease_release": bound(release),
        }
    )
    assessor_input_path = tmp_path / "assessor-input.json"
    assessor_output_path = tmp_path / "assessor-output.json"
    _write_json(assessor_input_path, assessor_input)
    completed = subprocess.run(
        [str(assessor_binary), "--input", str(assessor_input_path), "--out", str(assessor_output_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    output = json.loads(assessor_output_path.read_text(encoding="utf-8"))
    assert output["status"] == "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER"


def test_inner_refuses_runtime_identity_mismatch(
    tmp_path: Path, live_preflight: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    _, _, outer = live_preflight
    outer_path = tmp_path / "outer-preflight.json"
    _write_json(outer_path, outer)
    execution = _execution_binding(outer)
    execution_path = tmp_path / "execution-binding.json"
    _write_json(execution_path, execution)
    lease = lifecycle.issue_lease(
        outer_preflight=outer_path,
        execution_binding=execution_path,
        watcher_hold=WATCHER_HOLD,
        out=tmp_path / "lease.json",
    )
    inner = _inner_document(outer, lease)
    inner["opaque_l0_continuation"]["runtime_identity_sha256"] = _sha("9")
    inner.pop("seal_sha256")
    inner = seal(inner)
    with pytest.raises(lifecycle.JointLifecycleError, match="runtime/TCB identity mismatch"):
        lifecycle.validate_inner_receipt(inner, lease.outer_preflight, lease)


def test_inner_accepts_bounded_metal_route_weight_rounding_but_refuses_drift(
    tmp_path: Path, live_preflight: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    _, _, outer = live_preflight
    outer_path = tmp_path / "outer-preflight.json"
    _write_json(outer_path, outer)
    execution_path = tmp_path / "execution-binding.json"
    _write_json(execution_path, _execution_binding(outer))
    lease = lifecycle.issue_lease(
        outer_preflight=outer_path,
        execution_binding=execution_path,
        watcher_hold=WATCHER_HOLD,
        out=tmp_path / "lease.json",
    )
    inner = _inner_document(outer, lease)
    guard = inner["fresh_readbacks"]["l0_suffix"]["route_guard"]
    observed = list(guard["observed_route_weights"])
    observed[0] += 5.0e-7
    guard["observed_route_weights"] = observed
    guard["weights_max_abs_error"] = 5.0e-7
    inner.pop("seal_sha256")
    bounded = seal(inner)
    lifecycle.validate_inner_receipt(bounded, lease.outer_preflight, lease)

    guard = bounded["fresh_readbacks"]["l0_suffix"]["route_guard"]
    observed = list(guard["observed_route_weights"])
    observed[0] += 2.0e-6
    guard["observed_route_weights"] = observed
    guard["weights_max_abs_error"] = 2.5e-6
    bounded.pop("seal_sha256")
    drifted = seal(bounded)
    with pytest.raises(lifecycle.JointLifecycleError, match="route guard weights drifted"):
        lifecycle.validate_inner_receipt(drifted, lease.outer_preflight, lease)
