"""Fail-closed, CPU-only preflight for Qwen80's source-token L0 true-MoE graph.

This does not create a Metal context, issue a quiet lease, re-open the model,
or start a child.  It joins only the already sealed source-token route
authority, strict-Metal prefix receipt, sealed source-token bridge, and the
separate unsealed static suffix ABI.  The historical fixture route plan and
legacy true-MoE launcher are deliberately not inputs to this authority.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_source_token_all_ten_true_moe_bridge as bridge
from lab.operators import ascension_qwen80_source_token_l0_route_plan as route_plan
from lab.receipts import seal


SCHEMA = "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1"
STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_"
    "OUTER_READY_FOR_SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED"
)
SOURCE_TYPED_BRIDGE_SCHEMA = bridge.BRIDGE_SCHEMA
SOURCE_TYPED_BRIDGE_STATUS = bridge.BRIDGE_STATUS
SOURCE_AUTHORITY_SCHEMA = bridge.SOURCE_AUTHORITY_SCHEMA
SOURCE_AUTHORITY_STATUS = bridge.SOURCE_AUTHORITY_STATUS
FIXED_ABI_SCHEMA = "hawking.ascension.qwen80_l0_true_moe_fixed_payload_contract.v1"
FIXED_ABI_STATUS = "PREPARED_QWEN80_L0_TRUE_MOE_FIXED_SUFFIX_PAYLOAD_PLAN_NOT_EXECUTED"
HIDDEN = 2_048
TOP_K = 10
EXPECTED_FIXED_TENSORS = {
    "model.layers.0.post_attention_layernorm.weight": "a00ba60c88bd0d5dcf77e4c1fad05d83ddb6feec844ee3bbc65480fffd5a1fa7",
    "model.layers.0.mlp.gate.weight": "582725c1fa47c62b0f109216e8c2c40533b2931a583f4a41dfa34477deda45f4",
    "model.layers.0.mlp.shared_expert.gate_proj.weight": "92172dc4463a3a0610460ecf768427f6c9c8da04b43a73e904ca1fa36bc79aa6",
    "model.layers.0.mlp.shared_expert.up_proj.weight": "9d76293fa8abf4ccc2611d77386060671107e83dfd4458b5fddd5e345f24b4c4",
    "model.layers.0.mlp.shared_expert.down_proj.weight": "acf137a00b364f9c490e1282f18632465f05323b89903a5617162437b1ff500b",
    "model.layers.0.mlp.shared_expert_gate.weight": "a40ff8a3f4e4b7e990a4672470cbd028b0c96b1cb15acd40aa3b8b2e2215096c",
}
EXPECTED_KERNELS = (
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
)


class SourceTokenOuterPreflightError(RuntimeError):
    """The source-token component graph lacks one exact antecedent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceTokenOuterPreflightError(f"{label} must be an object")
    return dict(value)


def _route_ids(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != TOP_K:
        raise SourceTokenOuterPreflightError(f"{label} must contain exactly ten IDs")
    if any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 512 for item in value):
        raise SourceTokenOuterPreflightError(f"{label} contains invalid expert IDs")
    if len(set(value)) != TOP_K:
        raise SourceTokenOuterPreflightError(f"{label} contains duplicate expert IDs")
    return list(value)


def _evidence_matches(value: object, expected: Mapping[str, Any], label: str) -> None:
    try:
        route_plan._evidence_matches(value, expected, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenOuterPreflightError(str(exc)) from exc


def _file_evidence(path: Path, label: str) -> dict[str, Any]:
    try:
        return route_plan._file_evidence(path, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenOuterPreflightError(str(exc)) from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return route_plan._read_json(path, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenOuterPreflightError(str(exc)) from exc


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        return route_plan._sealed_json(path, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenOuterPreflightError(str(exc)) from exc


def _require_sha(value: object, label: str) -> str:
    try:
        return route_plan._require_sha256(value, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenOuterPreflightError(str(exc)) from exc


def _bind_source_typed_bridge(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission: Mapping[str, Any],
    admission_receipt: Mapping[str, Any],
    admission_receipt_seal: str,
    source_authority_evidence: Mapping[str, Any],
    source_authority_document: Mapping[str, Any],
    source_authority_seal: str,
    prefix: Mapping[str, Any],
    prefix_seal: str,
) -> tuple[dict[str, Any], str, list[int], list[float]]:
    evidence = _file_evidence(path, "--typed-bridge-receipt")
    document, document_seal = _sealed_json(path, "source-token typed bridge")
    if document.get("schema") != SOURCE_TYPED_BRIDGE_SCHEMA or document.get("status") != SOURCE_TYPED_BRIDGE_STATUS:
        raise SourceTokenOuterPreflightError("source-token typed bridge schema/status drifted")
    source = _mapping(document.get("source_binding"), "source-token typed bridge source_binding")
    _evidence_matches(source.get("manifest"), manifest, "source-token typed bridge manifest")
    historical_pointer = _mapping(source.get("admission_current"), "source-token typed bridge admission")
    if historical_pointer.get("present") is not True or historical_pointer.get("path") != admission.get("path"):
        raise SourceTokenOuterPreflightError("source-token typed bridge admission pointer path drifted")
    _require_sha(historical_pointer.get("sha256"), "source-token typed bridge historical admission SHA")
    _evidence_matches(source.get("admission_receipt"), admission_receipt, "source-token typed bridge admission")
    _evidence_matches(
        source.get("source_token_route_authority"),
        source_authority_evidence,
        "source-token typed bridge route authority",
    )
    _evidence_matches(source.get("first_residual_receipt"), prefix, "source-token typed bridge prefix")
    if (
        source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
        or source.get("source_token_route_authority_seal_sha256") != source_authority_seal
        or source.get("first_residual_receipt_seal_sha256") != prefix_seal
    ):
        raise SourceTokenOuterPreflightError("source-token typed bridge immutable identity drifted")
    typed = _mapping(document.get("typed_bridge"), "source-token typed bridge payload")
    if (
        typed.get("layer") != 0
        or typed.get("source_token_id") != 1
        or typed.get("route_count") != TOP_K
        or typed.get("first_residual_elements") != HIDDEN
        or typed.get("same_command_graph_required") is not True
        or typed.get("first_residual_receipt_seal_sha256") != prefix_seal
        or typed.get("source_token_route_authority_seal_sha256") != source_authority_seal
    ):
        raise SourceTokenOuterPreflightError("source-token typed bridge payload geometry/lineage drifted")
    _require_sha(typed.get("first_residual_output_sha256"), "source-token typed bridge first residual SHA")
    sections = _mapping(typed.get("compact_section_sha256"), "source-token typed bridge compact sections")
    if set(sections) != {"gate_scales", "gate_signs", "up_scales", "up_signs", "down_scales", "down_signs"}:
        raise SourceTokenOuterPreflightError("source-token typed bridge compact sections drifted")
    for label, digest in sections.items():
        _require_sha(digest, f"source-token typed bridge compact section {label}")
    route = _mapping(document.get("route_authority"), "source-token typed bridge route authority")
    ids = _route_ids(route.get("ids"), "source-token typed bridge route IDs")
    weights = route.get("normalized_weights")
    source_plan = _mapping(source_authority_document.get("source_token_plan"), "source-token authority plan")
    router = _mapping(source_plan.get("source_token_router_evidence"), "source-token authority router")
    expected_ids = _route_ids(router.get("source_stable_route_ids"), "source-token authority route IDs")
    expected_weights = router.get("source_stable_normalized_weights")
    if (
        ids != expected_ids
        or not bridge._same_route_weights(weights, expected_weights)
        or route.get("wave_count") != TOP_K
        or route.get("all_thirty_wave_payloads_use_admission_verified_immutable_snapshots") is not True
    ):
        raise SourceTokenOuterPreflightError("source-token typed bridge route authority drifted")
    if not isinstance(weights, list):
        raise SourceTokenOuterPreflightError("source-token typed bridge route weights must be a list")
    return evidence, document_seal, ids, [float(value) for value in weights]


def validate_source_token_fixed_suffix(
    document: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_receipt_seal: str,
) -> None:
    """Validate the static suffix grammar without treating it as execution evidence."""
    if document.get("schema") != FIXED_ABI_SCHEMA or document.get("status") != FIXED_ABI_STATUS:
        raise SourceTokenOuterPreflightError("source-token fixed suffix schema/status drifted")
    if document.get("seal_sha256") is not None:
        raise SourceTokenOuterPreflightError("source-token fixed suffix must remain an unsealed raw static plan")
    source = _mapping(document.get("source_binding"), "source-token fixed suffix source_binding")
    if (
        source.get("manifest_schema") != "hawking.ascension.qwen80_complete_binary_gravity.v1"
        or source.get("manifest_document_sha256") != manifest.get("sha256")
        or source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise SourceTokenOuterPreflightError("source-token fixed suffix artifact identity drifted")
    authority = _mapping(document.get("external_authority"), "source-token fixed suffix external authority")
    if (
        authority.get("route_plan_schema") != SOURCE_AUTHORITY_SCHEMA
        or authority.get("route_plan_status") != SOURCE_AUTHORITY_STATUS
        or authority.get("first_residual_schema") != "hawking.ascension.qwen80_first_residual_outer_capture.v1"
        or authority.get("first_residual_status") != "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY"
        or authority.get("typed_bridge_schema") != SOURCE_TYPED_BRIDGE_SCHEMA
        or authority.get("typed_bridge_status") != SOURCE_TYPED_BRIDGE_STATUS
        or authority.get("route_payloads_materialized_here") is not False
        or authority.get("first_residual_materialized_here") is not False
        or authority.get("expected_topk_witness_materialized_here") is not False
        or authority.get("route_tensor_sha256s_materialized_here") is not False
    ):
        raise SourceTokenOuterPreflightError("source-token fixed suffix authority family drifted")
    payloads = document.get("fixed_payloads")
    if not isinstance(payloads, list) or len(payloads) != len(EXPECTED_FIXED_TENSORS):
        raise SourceTokenOuterPreflightError("source-token fixed suffix fixed-payload inventory drifted")
    observed_payloads = {
        item.get("tensor_name"): item.get("tensor_artifact_sha256")
        for item in payloads
        if isinstance(item, Mapping)
    }
    if observed_payloads != EXPECTED_FIXED_TENSORS:
        raise SourceTokenOuterPreflightError("source-token fixed suffix tensor identity drifted")
    dispatches = document.get("fixed_14_dispatch_abi")
    if not isinstance(dispatches, list) or tuple(
        item.get("kernel") if isinstance(item, Mapping) else None for item in dispatches
    ) != EXPECTED_KERNELS or any(
        not isinstance(item, Mapping) or item.get("ordinal") != index
        for index, item in enumerate(dispatches, start=1)
    ):
        raise SourceTokenOuterPreflightError("source-token fixed suffix 14-dispatch ABI drifted")
    boundary = _mapping(document.get("claim_boundary"), "source-token fixed suffix claim boundary")
    if (
        boundary.get("artifact_scan_or_payload_open_performed") is not False
        or boundary.get("metal_context_or_dispatch_performed") is not False
        or boundary.get("runtime_watcher_server_registry_or_hcli_changed") is not False
        or boundary.get("token_or_tps_claim") is not False
        or boundary.get("execution_status") != "PREPARED_NOT_EXECUTED"
    ):
        raise SourceTokenOuterPreflightError("source-token fixed suffix was promoted beyond static preparation")


def build_preflight(
    *,
    manifest_path: Path,
    admission_path: Path,
    source_authority_path: Path,
    first_residual_path: Path,
    typed_bridge_path: Path,
    fixed_suffix_path: Path,
) -> dict[str, Any]:
    try:
        manifest, manifest_seal = route_plan._bind_manifest(manifest_path)
        admission, pointer_seal, admission_receipt, admission_receipt_seal = route_plan._bind_admission(
            admission_path, manifest, manifest_seal
        )
        prefix, prefix_seal, _ = route_plan._bind_prefix(
            first_residual_path,
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission_receipt_seal=admission_receipt_seal,
        )
        source_authority, source_authority_seal, source_authority_document = bridge._bind_source_authority(
            source_authority_path,
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission=admission,
            admission_receipt=admission_receipt,
            admission_receipt_seal=admission_receipt_seal,
            prefix=prefix,
            prefix_seal=prefix_seal,
        )
    except (route_plan.SourceTokenRoutePlanError, bridge.SourceTokenBridgeSealError) as exc:
        raise SourceTokenOuterPreflightError(str(exc)) from exc
    typed_evidence, typed_seal, route_ids, route_weights = _bind_source_typed_bridge(
        typed_bridge_path,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt=admission_receipt,
        admission_receipt_seal=admission_receipt_seal,
        source_authority_evidence=source_authority,
        source_authority_document=source_authority_document,
        source_authority_seal=source_authority_seal,
        prefix=prefix,
        prefix_seal=prefix_seal,
    )
    fixed_evidence = _file_evidence(fixed_suffix_path, "--fixed-suffix-contract")
    fixed_document = _read_json(fixed_suffix_path, "--fixed-suffix-contract")
    validate_source_token_fixed_suffix(
        fixed_document,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_receipt_seal=admission_receipt_seal,
    )
    divergence = _mapping(source_authority_document.get("fixture_divergence"), "source-token authority fixture divergence")
    if (
        divergence.get("same_route_ids") is not False
        or divergence.get("conclusion")
        != "the fixture-derived route plan is prohibited from driving the source-token true-MoE graph"
    ):
        raise SourceTokenOuterPreflightError("historical fixture route mismatch was not retained as negative science")
    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "recorded_at": _utc_now(),
            "source_binding": {
                "manifest": manifest,
                "admission_current": admission,
                "admission_receipt": admission_receipt,
                "source_token_route_authority": source_authority,
                "first_residual_receipt": prefix,
                "typed_bridge_receipt": typed_evidence,
                "fixed_suffix_contract": fixed_evidence,
                "manifest_seal_sha256": manifest_seal,
                "admission_pointer_seal_sha256": pointer_seal,
                "admission_receipt_seal_sha256": admission_receipt_seal,
                "source_token_route_authority_seal_sha256": source_authority_seal,
                "first_residual_receipt_seal_sha256": prefix_seal,
                "typed_bridge_receipt_seal_sha256": typed_seal,
            },
            "source_token_route": {
                "layer": 0,
                "token_id": 1,
                "zero_l0_state_required": True,
                "same_command_graph_required": True,
                "route_ids": route_ids,
                "normalized_weights": route_weights,
                "all_ten_unique": len(set(route_ids)) == TOP_K,
            },
            "fixed_suffix": {
                "schema": FIXED_ABI_SCHEMA,
                "status": FIXED_ABI_STATUS,
                "dispatch_count": len(EXPECTED_KERNELS),
                "fixed_tensor_count": len(EXPECTED_FIXED_TENSORS),
                "route_authority_family": "source_token_only",
            },
            "legacy_fixture_negative": {
                "historical_fixture_plan_is_authority": False,
                "same_route_ids": False,
                "preserved_conclusion": divergence["conclusion"],
            },
            "next_child_contract": {
                "legacy_router_receipt_or_fixture_plan_accepted": False,
                "requires_source_token_authority_and_typed_bridge": True,
                "requires_same_tcb_prefix_lineage": True,
                "requires_fresh_component_only_quiet_lease": True,
                "requires_outer_reaped_receipt_last_capture": True,
                "source_token_device_child_implementation_required_before_lease": True,
            },
            "claim_boundary": {
                "artifact_scan_performed_by_preflight": False,
                "metal_device_or_dispatch_performed": False,
                "lease_issued": False,
                "watcher_server_registry_or_hcli_changed": False,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
            },
        }
    )


def write_new(path: Path, document: Mapping[str, Any]) -> None:
    try:
        route_plan.write_new(path, document)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenOuterPreflightError(str(exc)) from exc


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--source-token-route-authority", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--typed-bridge-receipt", type=Path, required=True)
    parser.add_argument("--fixed-suffix-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if arguments is None else arguments)
    try:
        document = build_preflight(
            manifest_path=args.manifest,
            admission_path=args.admission_current,
            source_authority_path=args.source_token_route_authority,
            first_residual_path=args.first_residual_receipt,
            typed_bridge_path=args.typed_bridge_receipt,
            fixed_suffix_path=args.fixed_suffix_contract,
        )
        write_new(args.out, document)
    except SourceTokenOuterPreflightError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_PREFLIGHT", "error": str(exc)}))
        return 2
    print(json.dumps({"status": document["status"], "out": str(args.out), "seal_sha256": document["seal_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
