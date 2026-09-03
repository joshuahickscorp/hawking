from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from lab.operators import ascension_qwen80_l1_source_token_prefix_launcher as launcher
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime"
    / "QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_CAPTURE_20260809T081620Z"
)


def _sha(number: int) -> str:
    return f"{number:064x}"


def _bound(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "document": document,
        "document_sha256": launcher._sha256(document),
        "document_seal_sha256": document["seal_sha256"],
    }


def _identity(document: dict[str, Any]) -> dict[str, str]:
    return {
        "document_sha256": launcher._sha256(document),
        "document_seal_sha256": document["seal_sha256"],
    }


def _l0_state_record(name: str, capacity: int, hash_field: str, seed: int) -> dict[str, Any]:
    return {
        "allocation_id": name,
        "slot": launcher.L0_SLOT,
        "offset_bytes": 0,
        "capacity_bytes": capacity,
        "device_buffer_id": _sha(seed),
        hash_field: _sha(seed + 1),
    }


def _l1_binding_state_record(name: str, offset: int, capacity: int, seed: int) -> dict[str, Any]:
    identity = _sha(seed)
    return {
        "allocation_id": name,
        "slot": launcher.L1_SLOT,
        "offset_bytes": offset,
        "capacity_bytes": capacity,
        "device_buffer_id": identity,
        "device_buffer_identity_sha256": identity,
    }


def _manifest() -> dict[str, Any]:
    return seal(
        {
            "schema": launcher.MANIFEST_SCHEMA,
            "status": "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
            "source": {"repository": "Qwen/Qwen3-Coder-Next"},
        }
    )


def _admission(manifest: dict[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema": launcher.ADMISSION_RECEIPT_SCHEMA,
            "status": launcher.ADMISSION_RECEIPT_STATUS,
            "model": {
                "id": "Qwen3-Coder-Next-80B",
                "key": launcher.MODEL_KEY,
                "revision": "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
            },
            "complete_manifest": {
                "schema": launcher.MANIFEST_SCHEMA,
                "seal_sha256": manifest["seal_sha256"],
                "document_sha256": launcher._sha256(manifest),
            },
        }
    )


def _schedule() -> dict[str, Any]:
    return seal(
        {
            "schema": launcher.SCHEDULE_SCHEMA,
            "status": launcher.SCHEDULE_STATUS,
            "claim_boundary": {
                "wrapper_is_read_only": True,
                "future_joint_l0_to_l1_capture_authorized": False,
            },
            "raw_schedule_authority": {
                "present": True,
                "raw_schedule_is_static_and_unmodified": True,
                "schema": launcher.RAW_SCHEDULE_SCHEMA,
                "status": launcher.RAW_SCHEDULE_STATUS,
                "sha256": _sha(70),
                "raw_schedule_seal_sha256": None,
            },
            "schedule_facts": {
                "all_48_layers_scheduled": True,
                "layer_count": 48,
                "delta_net_layer_count": 36,
                "gqa_layer_count": 12,
                "layer_1": {
                    "layer": launcher.L1_LAYER,
                    "mixer": "delta_net",
                    "state_slot": launcher.L1_SLOT,
                    "state_domain": "delta_net_conv_and_recurrent",
                },
            },
        }
    )


def _l0_inner(
    manifest: dict[str, Any],
    admission: dict[str, Any],
    *,
    retained_bytes: int = launcher.HIDDEN_BYTES,
) -> dict[str, Any]:
    session_id = "qwen80-source-token-l0-next-layer"
    output_hash = _sha(10)
    output_buffer = _sha(11)
    return seal(
        {
            "schema": launcher.L0_INNER_SCHEMA,
            "status": launcher.L0_INNER_STATUS,
            "mode": "metal",
            "component_only": True,
            "complete_layer_or_token_performed": False,
            "l1_binding_not_executed": True,
            "l1_prefix_dispatches": 0,
            "artifact_binding": {
                "manifest_seal_sha256": manifest["seal_sha256"],
                "admission_receipt_seal_sha256": admission["seal_sha256"],
                "source_revision": "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
            },
            "same_command_graph": {
                "source_token_id": launcher.SOURCE_TOKEN_ID,
                "prefix_dispatches": launcher.L0_PREFIX_DISPATCHES,
                "suffix_dispatches": launcher.L0_SUFFIX_DISPATCHES,
                "total_dispatches": launcher.L0_TOTAL_DISPATCHES,
                "same_command_graph_retained": True,
                "fenced_once_after_prefix_and_suffix": True,
            },
            "l0_state_handoff": {
                "schema": launcher.L0_INNER_SCHEMA,
                "status": launcher.L0_INNER_STATUS,
                "session_id": session_id,
                "source_token_id": launcher.SOURCE_TOKEN_ID,
                "same_command_graph_retained": True,
                "l1_binding_not_executed": True,
                "l1_prefix_dispatches": 0,
                "retained_l0_second_residual": {
                    "elements": launcher.HIDDEN_ELEMENTS,
                    "bytes": retained_bytes,
                    "f32le_sha256": output_hash,
                    "device_buffer_id": output_buffer,
                    "retained_for_future_layer1_encode": True,
                },
                "l0_post_state_commit": {
                    "layer": launcher.L0_LAYER,
                    "linear_state_slot": launcher.L0_SLOT,
                    "checkpoint_before_mutation": True,
                    "active_conv": _l0_state_record(
                        "l0-active-conv", launcher.L0_CONV_BYTES, "post_state_f32le_sha256", 20
                    ),
                    "active_recurrent": _l0_state_record(
                        "l0-active-recurrent",
                        launcher.L0_RECURRENT_BYTES,
                        "post_state_f32le_sha256",
                        22,
                    ),
                    "rollback_conv": _l0_state_record(
                        "l0-rollback-conv", launcher.L0_CONV_BYTES, "checkpoint_f32le_sha256", 24
                    ),
                    "rollback_recurrent": _l0_state_record(
                        "l0-rollback-recurrent",
                        launcher.L0_RECURRENT_BYTES,
                        "checkpoint_f32le_sha256",
                        26,
                    ),
                },
                "layer1_input_binding": {
                    "session_id": session_id,
                    "layer": launcher.L1_LAYER,
                    "linear_state_slot": launcher.L1_SLOT,
                    "input_device_buffer_id": output_buffer,
                    "input_f32le_sha256": output_hash,
                    "same_command_graph_retained": True,
                    "l1_binding_executed": False,
                    "active_conv": _l1_binding_state_record(
                        "l1-active-conv",
                        launcher.L1_CONV_OFFSET_BYTES,
                        launcher.L1_CONV_CAPACITY_BYTES,
                        40,
                    ),
                    "active_recurrent": _l1_binding_state_record(
                        "l1-active-recurrent",
                        launcher.L1_RECURRENT_OFFSET_BYTES,
                        launcher.L1_RECURRENT_CAPACITY_BYTES,
                        42,
                    ),
                },
                "claim_boundary": {
                    "component_only": True,
                    "layer1_not_encoded": True,
                    "retention_binding_is_not_a_layer1_execution_claim": True,
                    "may_not_satisfy_next_layer_execution_dependency": True,
                },
            },
        }
    )


def _l0_outer(inner: dict[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema": launcher.L0_OUTER_SCHEMA,
            "status": launcher.L0_OUTER_STATUS,
            "inner_probe_capture": {
                "binding_valid": True,
                "present": True,
                "schema": launcher.L0_INNER_SCHEMA,
                "status": launcher.L0_INNER_STATUS,
                "receipt": {
                    "present": True,
                    "sha256": _sha(50),
                    "seal_sha256": inner["seal_sha256"],
                },
            },
            "source_binding": {
                "handoff_contract": {
                    "source_token_id": launcher.SOURCE_TOKEN_ID,
                    "prefix_dispatches": launcher.L0_PREFIX_DISPATCHES,
                    "suffix_dispatches": launcher.L0_SUFFIX_DISPATCHES,
                    "total_dispatches": launcher.L0_TOTAL_DISPATCHES,
                    "same_tcb_fence_required": True,
                    "l1_binding_not_executed": True,
                    "l1_prefix_dispatches": 0,
                }
            },
            "one_shot": {
                "automatic_retry_disabled": True,
                "lease_reuse_prohibited_after_terminal": True,
                "outer_reaped_child": True,
                "same_capture_dir_never_starts_a_second_child": True,
                "terminal_receipt_written_last": True,
            },
            "claim_boundary": {
                "l1_binding_not_executed": True,
                "l1_prefix_executed": False,
                "watcher_or_server_transition_not_authorized": True,
            },
            "child": {"terminal": {"exit_code": 0, "reaped": True, "timed_out": False}},
        }
    )


def _lease_release(outer: dict[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema": launcher.L0_RELEASE_SCHEMA,
            "status": launcher.L0_RELEASE_STATUS,
            "coordination": {
                "quiet_qwen80_component_lease_released": True,
                "watcher_hold_remains_active": True,
                "automatic_retry_prohibited": True,
            },
            "outer_terminal": {
                "seal_sha256": outer["seal_sha256"],
                "status": launcher.L0_OUTER_STATUS,
            },
        }
    )


def _assessment(
    outer: dict[str, Any], inner: dict[str, Any], release: dict[str, Any]
) -> dict[str, Any]:
    return seal(
        {
            "schema": launcher.L0_ASSESSMENT_SCHEMA,
            "status": launcher.L0_ASSESSMENT_STATUS,
            "earned_l0_state_handoff_component": True,
            "l0_handoff_is_evidence_baseline_only": True,
            "l1_binding_not_executed": True,
            "l1_prefix_dispatches": 0,
            "l1_continuation_prepared": False,
            "l1_continuation_remains_non_executing": True,
            "may_not_satisfy_next_layer_execution_dependency": True,
            "future_l1_requires_fresh_same_runtime_same_tcb_joint_l0_to_l1_capture": True,
            "cross_process_or_prior_capture_pinned_buffer_reuse_authorized": False,
            "l0_outer_terminal": _identity(outer),
            "l0_inner_receipt": _identity(inner),
            "lease_release_receipt": _identity(release),
        }
    )


def _readiness(inner: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema": launcher.READINESS_SCHEMA,
            "status": launcher.READINESS_STATUS,
            "prepared": True,
            "l1_execution_performed_by_this_contract": False,
            "l1_prefix_dispatches_executed_by_this_contract": 0,
            "l0_state_handoff_receipt": _identity(inner),
            "schedule_authority": _identity(schedule),
            "future_l1_slot1_deltanet_prefix_scope": {
                "layer": launcher.L1_LAYER,
                "mixer": "delta_net",
                "linear_state_slot": launcher.L1_SLOT,
                "exact_prefix_dispatch_count": launcher.L1_PREFIX_DISPATCHES,
                "exact_prefix_dispatches": list(launcher.L1_PREFIX),
                "no_l1_suffix_or_moe_dispatch_authorized": True,
            },
            "authority_boundary": {
                "new_physical_model_processes_authorized": 0,
                "server_starts_authorized": 0,
                "port_binds_authorized": 0,
                "gpu_leases_authorized": 0,
                "watcher_changes_authorized": 0,
                "tournament_state_mutations_authorized": 0,
            },
        }
    )


def _assessor_binding(
    outer: dict[str, Any],
    inner: dict[str, Any],
    assessment: dict[str, Any],
    release: dict[str, Any],
    manifest: dict[str, Any],
    admission: dict[str, Any],
) -> dict[str, Any]:
    return seal(
        {
            "schema": launcher.L0_ASSESSOR_BINDING_SCHEMA,
            "status": launcher.L0_ASSESSOR_BINDING_STATUS,
            "l0_outer_terminal": _identity(outer),
            "l0_inner_capture": _identity(inner),
            "post_capture_assessment": _identity(assessment),
            "lease_release_receipt": _identity(release),
            "required_assessment": {
                "schema": launcher.L0_ASSESSMENT_SCHEMA,
                "earned_status": launcher.L0_ASSESSMENT_STATUS,
                "assessment_document_sha256": launcher._sha256(assessment),
                "assessment_document_seal_sha256": assessment["seal_sha256"],
                "must_be_sealed": True,
                "must_bind_actual_release": True,
                "must_bind_l0_outer_and_inner": True,
                "must_remain_l1_not_executed": True,
            },
            "assessment_result_bound": True,
            "assessment_required_before_joint_child_launch": True,
            "baseline_l0_evidence_is_provenance_only": True,
            "cross_process_pinned_buffer_transfer_allowed": False,
            "joint_l0_reencode_required": True,
            "future_l1_requires_fresh_same_runtime_same_tcb_joint_l0_to_l1_capture": True,
            "joint_child_execution_authorized_by_this_wrapper": False,
            "l1_execution_authorized_by_this_wrapper": False,
            "retained_l0_state_handoff": {
                "source_token_id": launcher.SOURCE_TOKEN_ID,
                "l1_binding_not_executed": True,
                "l1_prefix_dispatches": 0,
                "retained_l0_second_residual": {
                    "elements": launcher.HIDDEN_ELEMENTS,
                    "bytes": launcher.HIDDEN_BYTES,
                    "f32le_sha256": inner["l0_state_handoff"]["retained_l0_second_residual"]["f32le_sha256"],
                    "device_buffer_id": inner["l0_state_handoff"]["retained_l0_second_residual"]["device_buffer_id"],
                },
                "reserved_l1_slot": {
                    "layer": launcher.L1_LAYER,
                    "linear_state_slot": launcher.L1_SLOT,
                    "active_conv_offset_bytes": launcher.L1_CONV_OFFSET_BYTES,
                    "active_recurrent_offset_bytes": launcher.L1_RECURRENT_OFFSET_BYTES,
                    "active_conv_capacity_bytes": launcher.L1_CONV_CAPACITY_BYTES,
                    "active_recurrent_capacity_bytes": launcher.L1_RECURRENT_CAPACITY_BYTES,
                },
            },
            "future_joint_capture_requirement": {
                "fresh_l0_reencode_dispatches": launcher.L0_TOTAL_DISPATCHES,
                "future_l1_slot1_prefix_dispatches": launcher.L1_PREFIX_DISPATCHES,
                "future_joint_total_dispatches": launcher.JOINT_TOTAL_DISPATCHES,
                "historical_pinned_buffer_or_state_import_allowed": False,
                "historical_receipts_are_provenance_only": True,
                "same_runtime_required": True,
                "same_session_required": True,
                "same_tcb_required": True,
            },
            "immutable_authority_chain": {
                "versioned_manifest_and_admission": {
                    "model_key": launcher.MODEL_KEY,
                    "manifest_seal_sha256": manifest["seal_sha256"],
                    "admission_seal_sha256": admission["seal_sha256"],
                    "manifest_file_sha256": admission["complete_manifest"]["document_sha256"],
                    "admission_file_sha256": _sha(69),
                }
            },
        }
    )


def _joint_child(
    *,
    upstream: str,
    child_sha: str,
    old_process_buffer_claim: bool = False,
    joint_total: int = launcher.JOINT_TOTAL_DISPATCHES,
    l1_dispatches: list[dict[str, Any]] | None = None,
    concrete_host: bool = True,
) -> dict[str, Any]:
    child: dict[str, Any] = {
        "schema": launcher.JOINT_CHILD_PREFLIGHT_SCHEMA,
        "status": launcher.JOINT_CHILD_PREFLIGHT_STATUS,
        "future_joint_l0_l1_child_sha256": child_sha,
        "preflight_identity_sha256": upstream,
        "child_started": False,
        "metal_or_gpu_activity_performed": False,
        "component_only": True,
        "same_runtime_required": True,
        "same_session_required": True,
        "same_tcb_required": True,
        "baseline_l0_receipts_provenance_only": True,
        "cross_process_pinned_buffer_transfer_allowed": False,
        "external_l0_buffer_or_state_import_allowed": False,
        "opaque_canonical_l0_continuation_required": True,
        "raw_pinned_buffer_or_dispatch_count_input_allowed": False,
        "opaque_capability_must_bind_runtime_state_arena_identity": True,
        "future_joint_host_binary_bound": concrete_host,
        "future_joint_host_binary_role": (
            "strict_joint_l0_l1_same_runtime_host"
            if concrete_host
            else "static_preflight_authority_only_not_joint_host"
        ),
        "l0_reencode": {
            "source_token_id": launcher.SOURCE_TOKEN_ID,
            "prefix_dispatches": launcher.L0_PREFIX_DISPATCHES,
            "suffix_dispatches": launcher.L0_SUFFIX_DISPATCHES,
            "total_dispatches": launcher.L0_TOTAL_DISPATCHES,
            "same_tcb_fence_required": True,
        },
        "l1_prefix": {
            "layer": launcher.L1_LAYER,
            "mixer": "delta_net",
            "linear_state_slot": launcher.L1_SLOT,
            "prefix_dispatches": launcher.L1_PREFIX_DISPATCHES,
            "exact_prefix_dispatches": list(launcher.L1_PREFIX)
            if l1_dispatches is None
            else l1_dispatches,
            "no_l1_suffix_or_moe_dispatch_authorized": True,
        },
        "joint_command_graph": {
            "l0_dispatches": launcher.L0_TOTAL_DISPATCHES,
            "l1_prefix_dispatches": launcher.L1_PREFIX_DISPATCHES,
            "total_dispatches": joint_total,
            "same_runtime_same_tcb_required": True,
            "same_session_required": True,
            "single_fence_after_l0_and_l1_prefix_required": True,
            "non_timed_token_command_buffer_required": True,
            "tcb_trace_mode": "off",
            "opaque_capability_factory": launcher.CANONICAL_L0_CAPABILITY_FACTORY,
            "runtime_api": launcher.CANONICAL_L1_PREFIX_ENCODER,
            "consuming_finalizer": launcher.CANONICAL_L0_L1_FINALIZER,
            "structural_kernel_trace_required": True,
            "exact_l0_kernel_trace": list(launcher.L0_TRUE_MOE_KERNEL_TRACE),
            "exact_joint_kernel_trace": list(launcher.JOINT_L0_L1_KERNEL_TRACE),
            "finalizer_must_consume_the_only_command_buffer_before_fence": True,
            "fresh_l0_suffix_readbacks_required": {
                "route_guard": True,
                "postnorm": True,
                "router_logits": True,
                "all_ten_weighted_route_witnesses": 10,
                "shared_output": True,
                "routed_sum": True,
                "second_residual": True,
            },
            "fresh_l0_and_l1_state_output_rollback_witnesses_required": True,
        },
        "claim_boundary": {
            "complete_layer_or_token_allowed": False,
            "decoder_generation_hcli_tps_tg_or_tournament_allowed": False,
            "l1_suffix_or_moe_allowed": False,
            "automatic_retry_allowed": False,
        },
    }
    if old_process_buffer_claim:
        child["input_device_buffer_id"] = _sha(91)
    return seal(child)


def _source_l0_outer_preflight() -> dict[str, Any]:
    return seal(
        {
            "schema": launcher.L0_SOURCE_OUTER_PREFLIGHT_SCHEMA,
            "status": launcher.L0_SOURCE_OUTER_PREFLIGHT_STATUS,
            "source_token_route": {
                "token_id": launcher.SOURCE_TOKEN_ID,
                "layer": launcher.L0_LAYER,
                "same_command_graph_required": True,
                "zero_l0_state_required": True,
                "all_ten_unique": True,
            },
            "claim_boundary": {
                "lease_issued": False,
                "metal_device_or_dispatch_performed": False,
            },
            "next_child_contract": {
                "requires_same_tcb_prefix_lineage": True,
                "requires_source_token_authority_and_typed_bridge": True,
            },
        }
    )


def _joint_host_preflight(
    child: dict[str, Any], source_l0_outer: dict[str, Any]
) -> dict[str, Any]:
    child_sha = child["future_joint_l0_l1_child_sha256"]
    return seal(
        {
            "schema": launcher.JOINT_HOST_PREFLIGHT_SCHEMA,
            "status": launcher.JOINT_HOST_PREFLIGHT_STATUS,
            "child_started": False,
            "metal_or_gpu_activity_performed": False,
            "lease_issued_or_consumed": False,
            "component_only": True,
            "host_binary": {"sha256": child_sha},
            "joint_static_plan": {
                "document_sha256": launcher._sha256(child),
                "seal_sha256": child["seal_sha256"],
            },
            "l0_source_outer_preflight": {
                "document_sha256": launcher._sha256(source_l0_outer),
                "seal_sha256": source_l0_outer["seal_sha256"],
            },
            "same_runtime_same_session_same_tcb_required": True,
            "joint_command_graph": {
                "source_token_id": launcher.SOURCE_TOKEN_ID,
                "l0_dispatches": launcher.L0_TOTAL_DISPATCHES,
                "l1_prefix_dispatches": launcher.L1_PREFIX_DISPATCHES,
                "total_dispatches": launcher.JOINT_TOTAL_DISPATCHES,
                "single_fence_required": True,
                "non_timed_token_command_buffer_required": True,
                "tcb_trace_mode": "off",
                "exact_l0_kernel_trace": list(launcher.L0_TRUE_MOE_KERNEL_TRACE),
                "exact_joint_kernel_trace": list(launcher.JOINT_L0_L1_KERNEL_TRACE),
                "opaque_capability_factory": launcher.CANONICAL_L0_CAPABILITY_FACTORY,
                "runtime_api": launcher.CANONICAL_L1_PREFIX_ENCODER,
                "consuming_finalizer": launcher.CANONICAL_L0_L1_FINALIZER,
            },
            "host_body": {
                "strict_joint_entrypoint_compiled": True,
                "future_outer_reaper_and_fresh_lease_required_before_entrypoint_may_run": True,
                "historical_l0_receipt_or_pinned_buffer_import_allowed": False,
                "l1_suffix_or_moe_authorized": False,
            },
            "claim_boundary": {
                "complete_layer_or_token_decoder_hcli_tps_tg_or_tournament_allowed": False,
                "watcher_server_or_runtime_transition_allowed": False,
                "automatic_retry_allowed": False,
            },
        }
    )


def _future_input(
    *,
    retained_bytes: int = launcher.HIDDEN_BYTES,
    include_assessor: bool = True,
    old_process_buffer_claim: bool = False,
    joint_total: int = launcher.JOINT_TOTAL_DISPATCHES,
    l1_dispatches: list[dict[str, Any]] | None = None,
    concrete_host: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _manifest()
    admission = _admission(manifest)
    schedule = _schedule()
    inner = _l0_inner(manifest, admission, retained_bytes=retained_bytes)
    outer = _l0_outer(inner)
    readiness = _readiness(inner, schedule)
    release = _lease_release(outer)
    assessment = _assessment(outer, inner, release)
    assessor = _assessor_binding(outer, inner, assessment, release, manifest, admission)
    child_sha = _sha(80)
    upstream = launcher._upstream_authority_identity(
        readiness=launcher.BoundDocument(readiness, launcher._sha256(readiness), readiness["seal_sha256"]),
        l0_outer=launcher.BoundDocument(outer, launcher._sha256(outer), outer["seal_sha256"]),
        l0_inner=launcher.BoundDocument(inner, launcher._sha256(inner), inner["seal_sha256"]),
        assessor_binding=launcher.BoundDocument(assessor, launcher._sha256(assessor), assessor["seal_sha256"]),
        assessment=launcher.BoundDocument(assessment, launcher._sha256(assessment), assessment["seal_sha256"]),
        lease_release=launcher.BoundDocument(release, launcher._sha256(release), release["seal_sha256"]),
        manifest=launcher.BoundDocument(manifest, launcher._sha256(manifest), manifest["seal_sha256"]),
        admission_receipt=launcher.BoundDocument(admission, launcher._sha256(admission), admission["seal_sha256"]),
        schedule=launcher.BoundDocument(schedule, launcher._sha256(schedule), schedule["seal_sha256"]),
        child_sha256=child_sha,
    )
    child = _joint_child(
        upstream=upstream,
        child_sha=child_sha,
        old_process_buffer_claim=old_process_buffer_claim,
        joint_total=joint_total,
        l1_dispatches=l1_dispatches,
        concrete_host=concrete_host,
    )
    source_l0_outer = _source_l0_outer_preflight()
    host = _joint_host_preflight(child, source_l0_outer)
    source: dict[str, Any] = {
        "schema": launcher.INPUT_SCHEMA,
        "status": launcher.INPUT_STATUS,
        "joint_capture_requested": False,
        "continuation_readiness": _bound(readiness),
        "l0_outer_terminal": _bound(outer),
        "l0_inner_capture": _bound(inner),
        "post_capture_assessment": _bound(assessment),
        "lease_release_receipt": _bound(release),
        "manifest": _bound(manifest),
        "admission_receipt": _bound(admission),
        "schedule": _bound(schedule),
        "joint_l0_l1_child_preflight": _bound(child),
        "l0_source_outer_preflight": _bound(source_l0_outer),
        "joint_l0_l1_host_preflight": _bound(host),
        "future_joint_l0_l1_child_sha256": child_sha,
    }
    if include_assessor:
        source["l0_post_capture_assessor_binding"] = _bound(assessor)
    return seal(
        source
    ), {
        "outer": outer,
        "inner": inner,
        "assessment": assessment,
        "release": release,
        "assessor": assessor,
        "child": child,
        "source_l0_outer": source_l0_outer,
        "host": host,
    }


def test_absent_evidence_refuses_without_child_or_gpu() -> None:
    result = launcher.preflight()

    assert verify(result) == result
    assert result["status"] == launcher.REFUSED_STATUS
    assert result["prepared"] is False
    assert result["claim_boundary"]["child_spawned"] is False
    assert result["claim_boundary"]["metal_or_gpu_activity_performed"] is False


def test_full_chain_preflights_only_a_32_dispatch_same_runtime_component() -> None:
    source, _ = _future_input()
    result = launcher.preflight(source)

    assert verify(result) == result
    assert result["status"] == launcher.PREPARED_STATUS
    assert result["prepared"] is True
    scope = result["authorized_future_component_scope"]
    assert scope["l0_reencode_dispatches"] == 23
    assert scope["l1_prefix_dispatches"] == 9
    assert scope["joint_total_dispatches"] == 32
    assert scope["same_runtime_required"] is True
    assert scope["same_tcb_required"] is True
    assert result["baseline_l0_capture"]["baseline_evidence_only"] is True
    assert result["baseline_l0_capture"]["may_not_be_imported_by_future_child"] is True
    assert result["required_assessment"]["assessment_result_bound"] is True
    assert result["claim_boundary"]["joint_child_execution_authorized_by_this_launcher"] is False


def test_missing_sealed_assessor_placeholder_refuses() -> None:
    source, _ = _future_input(include_assessor=False)
    result = launcher.preflight(source)

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("l0_post_capture_assessor_binding" in blocker for blocker in result["blockers"])


def test_actual_081620_outer_and_nested_handoff_abi_is_recognized() -> None:
    outer = json.loads((CAPTURE_DIR / "outer-terminal-receipt.json").read_text(encoding="utf-8"))
    inner = json.loads((CAPTURE_DIR / "inner/receipt.json").read_text(encoding="utf-8"))

    assert verify(outer) == outer
    assert verify(inner) == inner
    assert outer["schema"] == launcher.L0_OUTER_SCHEMA
    assert outer["status"] == launcher.L0_OUTER_STATUS
    assert outer["inner_probe_capture"]["receipt"]["seal_sha256"] == inner["seal_sha256"]
    handoff = inner["l0_state_handoff"]
    assert handoff["schema"] == launcher.L0_INNER_SCHEMA
    assert handoff["status"] == launcher.L0_INNER_STATUS
    assert handoff["retained_l0_second_residual"]["bytes"] == launcher.HIDDEN_BYTES
    assert handoff["l1_binding_not_executed"] is True
    assert handoff["l1_prefix_dispatches"] == 0


def test_actual_081620_l0_capture_can_be_bound_only_as_joint_child_provenance() -> None:
    outer = json.loads((CAPTURE_DIR / "outer-terminal-receipt.json").read_text(encoding="utf-8"))
    inner = json.loads((CAPTURE_DIR / "inner/receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (
            REPO_ROOT
            / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-gravity"
            / "QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
        ).read_text(encoding="utf-8")
    )
    admission = json.loads(
        (
            REPO_ROOT
            / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-gravity/complete-admission/receipts"
            / "QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT_14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b.json"
        ).read_text(encoding="utf-8")
    )
    schedule = _schedule()
    readiness = _readiness(inner, schedule)
    release = _lease_release(outer)
    assessment = _assessment(outer, inner, release)
    assessor = _assessor_binding(outer, inner, assessment, release, manifest, admission)
    child_sha = _sha(80)
    upstream = launcher._upstream_authority_identity(
        readiness=launcher.BoundDocument(readiness, launcher._sha256(readiness), readiness["seal_sha256"]),
        l0_outer=launcher.BoundDocument(outer, launcher._sha256(outer), outer["seal_sha256"]),
        l0_inner=launcher.BoundDocument(inner, launcher._sha256(inner), inner["seal_sha256"]),
        assessor_binding=launcher.BoundDocument(assessor, launcher._sha256(assessor), assessor["seal_sha256"]),
        assessment=launcher.BoundDocument(assessment, launcher._sha256(assessment), assessment["seal_sha256"]),
        lease_release=launcher.BoundDocument(release, launcher._sha256(release), release["seal_sha256"]),
        manifest=launcher.BoundDocument(manifest, launcher._sha256(manifest), manifest["seal_sha256"]),
        admission_receipt=launcher.BoundDocument(admission, launcher._sha256(admission), admission["seal_sha256"]),
        schedule=launcher.BoundDocument(schedule, launcher._sha256(schedule), schedule["seal_sha256"]),
        child_sha256=child_sha,
    )
    child = _joint_child(upstream=upstream, child_sha=child_sha)
    source_l0_outer = _source_l0_outer_preflight()
    host = _joint_host_preflight(child, source_l0_outer)
    source = seal(
        {
            "schema": launcher.INPUT_SCHEMA,
            "status": launcher.INPUT_STATUS,
            "joint_capture_requested": False,
            "continuation_readiness": _bound(readiness),
            "l0_outer_terminal": _bound(outer),
            "l0_inner_capture": _bound(inner),
            "l0_post_capture_assessor_binding": _bound(assessor),
            "post_capture_assessment": _bound(assessment),
            "lease_release_receipt": _bound(release),
            "manifest": _bound(manifest),
            "admission_receipt": _bound(admission),
            "schedule": _bound(schedule),
            "joint_l0_l1_child_preflight": _bound(child),
            "l0_source_outer_preflight": _bound(source_l0_outer),
            "joint_l0_l1_host_preflight": _bound(host),
            "future_joint_l0_l1_child_sha256": child_sha,
        }
    )

    result = launcher.preflight(source)

    assert result["status"] == launcher.PREPARED_STATUS
    assert result["baseline_l0_capture"]["baseline_evidence_only"] is True
    assert result["baseline_l0_capture"]["may_not_satisfy_same_runtime_or_same_tcb_for_future_child"] is True
    assert result["claim_boundary"]["joint_child_execution_authorized_by_this_launcher"] is False


def test_l0_handoff_must_retain_exact_8192_byte_baseline_output() -> None:
    source, _ = _future_input(retained_bytes=8_196)
    result = launcher.preflight(source)

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("retained_l0_second_residual.bytes" in blocker for blocker in result["blockers"])


def test_child_refuses_old_process_buffer_claim_even_when_it_matches_baseline() -> None:
    source, _ = _future_input(old_process_buffer_claim=True)
    result = launcher.preflight(source)

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("may not import historical input_device_buffer_id" in blocker for blocker in result["blockers"])
    assert result["claim_boundary"]["cross_process_pinned_buffer_transfer_authorized"] is False


def test_child_must_preserve_exact_l0_plus_l1_total_and_prefix() -> None:
    source, _ = _future_input(joint_total=31)
    result = launcher.preflight(source)
    assert result["status"] == launcher.REFUSED_STATUS
    assert any("joint_command_graph.total_dispatches" in blocker for blocker in result["blockers"])

    widened = [*launcher.L1_PREFIX, {"ordinal": 10, "stage": "suffix", "kernel": "forbidden"}]
    source, _ = _future_input(l1_dispatches=widened)
    result = launcher.preflight(source)
    assert result["status"] == launcher.REFUSED_STATUS
    assert any("l1_prefix.exact_prefix_dispatches" in blocker for blocker in result["blockers"])


def test_child_refuses_raw_buffer_or_non_consuming_finalizer_contracts() -> None:
    source, values = _future_input()
    child = dict(values["child"])
    child["raw_pinned_buffer_or_dispatch_count_input_allowed"] = True
    graph = dict(child["joint_command_graph"])
    graph["consuming_finalizer"] = "caller_owned_fence"
    child["joint_command_graph"] = graph
    child.pop("seal_sha256")
    child = seal(child)

    source = dict(source)
    source["joint_l0_l1_child_preflight"] = _bound(child)
    source.pop("seal_sha256")
    source = seal(source)
    result = launcher.preflight(source)

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("raw_pinned_buffer_or_dispatch_count_input_allowed" in blocker for blocker in result["blockers"])
    assert any("consuming_finalizer" in blocker for blocker in result["blockers"])


def test_static_plan_cannot_stand_in_for_a_concrete_joint_host() -> None:
    source, _ = _future_input(concrete_host=False)

    result = launcher.preflight(source)

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("does not bind a concrete strict joint L0+L1 host binary" in blocker for blocker in result["blockers"])


def test_concrete_host_must_bind_static_plan_and_exact_source_l0_outer() -> None:
    source, values = _future_input()
    host = deepcopy(values["host"])
    host["joint_static_plan"]["document_sha256"] = _sha(97)
    host.pop("seal_sha256")
    host = seal(host)
    source = dict(source)
    source["joint_l0_l1_host_preflight"] = _bound(host)
    source.pop("seal_sha256")
    result = launcher.preflight(seal(source))

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("joint_static_plan.document_sha256" in blocker for blocker in result["blockers"])

    source, values = _future_input()
    source_l0_outer = deepcopy(values["source_l0_outer"])
    source_l0_outer["source_token_route"]["all_ten_unique"] = False
    source_l0_outer.pop("seal_sha256")
    source_l0_outer = seal(source_l0_outer)
    source = dict(source)
    source["l0_source_outer_preflight"] = _bound(source_l0_outer)
    source.pop("seal_sha256")
    result = launcher.preflight(seal(source))

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("l0_source_outer_preflight.source_token_route.all_ten_unique" in blocker for blocker in result["blockers"])


def test_assessor_binding_requires_earned_assessment_and_exact_release() -> None:
    source, values = _future_input()
    assessor = dict(values["assessor"])
    assessor["assessment_result_bound"] = False
    assessor.pop("seal_sha256")
    assessor = seal(assessor)

    source = dict(source)
    source["l0_post_capture_assessor_binding"] = _bound(assessor)
    source.pop("seal_sha256")
    result = launcher.preflight(seal(source))

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("assessment_result_bound" in blocker for blocker in result["blockers"])

    source, values = _future_input()
    assessor = dict(values["assessor"])
    release = dict(assessor["lease_release_receipt"])
    release["document_seal_sha256"] = _sha(99)
    assessor["lease_release_receipt"] = release
    assessor.pop("seal_sha256")
    assessor = seal(assessor)
    source = dict(source)
    source["l0_post_capture_assessor_binding"] = _bound(assessor)
    source.pop("seal_sha256")
    result = launcher.preflight(seal(source))

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("lease_release_receipt" in blocker for blocker in result["blockers"])


def test_child_requires_exact_non_timed_full_kernel_trace() -> None:
    source, values = _future_input()
    child = dict(values["child"])
    graph = dict(child["joint_command_graph"])
    graph["tcb_trace_mode"] = "gpu"
    graph["exact_joint_kernel_trace"] = graph["exact_joint_kernel_trace"][:-1]
    child["joint_command_graph"] = graph
    child.pop("seal_sha256")
    child = seal(child)
    source = dict(source)
    source["joint_l0_l1_child_preflight"] = _bound(child)
    source.pop("seal_sha256")

    result = launcher.preflight(seal(source))

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("tcb_trace_mode" in blocker for blocker in result["blockers"])
    assert any("exact_joint_kernel_trace" in blocker for blocker in result["blockers"])


def test_outer_must_bind_the_actual_inner_capture_seal() -> None:
    source, values = _future_input()
    outer = dict(values["outer"])
    probe = dict(outer["inner_probe_capture"])
    receipt = dict(probe["receipt"])
    receipt["seal_sha256"] = _sha(98)
    probe["receipt"] = receipt
    outer["inner_probe_capture"] = probe
    outer.pop("seal_sha256")
    outer = seal(outer)

    source = dict(source)
    source["l0_outer_terminal"] = _bound(outer)
    source.pop("seal_sha256")
    source = seal(source)
    result = launcher.preflight(source)

    assert result["status"] == launcher.REFUSED_STATUS
    assert any("inner_probe_capture.receipt.seal_sha256" in blocker for blocker in result["blockers"])


def test_cli_has_no_execution_or_lease_flag_and_only_prints_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert launcher.main(["--lease"]) == 2
    assert "forbidden" in capsys.readouterr().out

    input_path = tmp_path / "in.json"
    input_path.write_text(
        json.dumps(
            seal(
                {
                    "schema": launcher.INPUT_SCHEMA,
                    "status": launcher.INPUT_STATUS,
                    "joint_capture_requested": False,
                }
            )
        ),
        encoding="utf-8",
    )
    assert launcher.main(["--input", str(input_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == launcher.REFUSED_STATUS
    assert output["claim_boundary"]["child_spawned"] is False
