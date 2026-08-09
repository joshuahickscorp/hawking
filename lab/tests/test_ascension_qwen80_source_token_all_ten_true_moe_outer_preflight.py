from __future__ import annotations

from lab.operators import ascension_qwen80_source_token_all_ten_true_moe_outer_preflight as preflight


def _static_plan() -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema": preflight.FIXED_ABI_SCHEMA,
        "status": preflight.FIXED_ABI_STATUS,
        "source_binding": {
            "manifest_schema": "hawking.ascension.qwen80_complete_binary_gravity.v1",
            "manifest_document_sha256": digest,
            "manifest_seal_sha256": digest,
            "admission_receipt_seal_sha256": digest,
        },
        "external_authority": {
            "route_plan_schema": preflight.SOURCE_AUTHORITY_SCHEMA,
            "route_plan_status": preflight.SOURCE_AUTHORITY_STATUS,
            "first_residual_schema": "hawking.ascension.qwen80_first_residual_outer_capture.v1",
            "first_residual_status": "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY",
            "typed_bridge_schema": preflight.SOURCE_TYPED_BRIDGE_SCHEMA,
            "typed_bridge_status": preflight.SOURCE_TYPED_BRIDGE_STATUS,
            "route_payloads_materialized_here": False,
            "first_residual_materialized_here": False,
            "expected_topk_witness_materialized_here": False,
            "route_tensor_sha256s_materialized_here": False,
        },
        "fixed_payloads": [
            {"tensor_name": name, "tensor_artifact_sha256": sha}
            for name, sha in preflight.EXPECTED_FIXED_TENSORS.items()
        ],
        "fixed_14_dispatch_abi": [
            {"ordinal": index, "kernel": kernel}
            for index, kernel in enumerate(preflight.EXPECTED_KERNELS, start=1)
        ],
        "claim_boundary": {
            "artifact_scan_or_payload_open_performed": False,
            "metal_context_or_dispatch_performed": False,
            "runtime_watcher_server_registry_or_hcli_changed": False,
            "token_or_tps_claim": False,
            "execution_status": "PREPARED_NOT_EXECUTED",
        },
    }


def test_source_token_fixed_suffix_accepts_only_source_token_authority_family() -> None:
    digest = "a" * 64
    preflight.validate_source_token_fixed_suffix(
        _static_plan(),
        manifest={"sha256": digest},
        manifest_seal=digest,
        admission_receipt_seal=digest,
    )


def test_source_token_fixed_suffix_rejects_legacy_fixture_authority() -> None:
    digest = "a" * 64
    plan = _static_plan()
    authority = dict(plan["external_authority"])
    authority["route_plan_schema"] = "hawking.ascension.qwen80_all_ten_routed_expert_binding_plan.v1"
    plan["external_authority"] = authority
    try:
        preflight.validate_source_token_fixed_suffix(
            plan,
            manifest={"sha256": digest},
            manifest_seal=digest,
            admission_receipt_seal=digest,
        )
    except preflight.SourceTokenOuterPreflightError as exc:
        assert "authority family" in str(exc)
    else:
        raise AssertionError("legacy fixture authority was accepted")


def test_typed_bridge_binding_uses_authority_document_for_route_data() -> None:
    assert "source_authority_document" in preflight._bind_source_typed_bridge.__code__.co_varnames


def test_source_token_fixed_suffix_rejects_any_static_execution_promotion() -> None:
    digest = "a" * 64
    plan = _static_plan()
    boundary = dict(plan["claim_boundary"])
    boundary["metal_context_or_dispatch_performed"] = True
    plan["claim_boundary"] = boundary
    try:
        preflight.validate_source_token_fixed_suffix(
            plan,
            manifest={"sha256": digest},
            manifest_seal=digest,
            admission_receipt_seal=digest,
        )
    except preflight.SourceTokenOuterPreflightError as exc:
        assert "promoted" in str(exc)
    else:
        raise AssertionError("promoted static suffix was accepted")
