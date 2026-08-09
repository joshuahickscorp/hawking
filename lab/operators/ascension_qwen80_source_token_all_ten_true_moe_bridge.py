"""Seal CPU-only source-token Qwen80 all-ten bridge material.

This wrapper preserves the historical fixture-plan mismatch as negative
science and permits only the distinct sealed token-1 route authority.  It
does not issue a lease or create a device context.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_source_token_l0_route_plan as route_plan
from lab.receipts import seal


MATERIAL_SCHEMA = "hawking.ascension.qwen80_source_token_all_ten_true_moe_source_bridge_material.v1"
MATERIAL_STATUS = (
    "CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_MATERIAL_READY_FOR_SEAL"
)
BRIDGE_SCHEMA = "hawking.ascension.qwen80_source_token_all_ten_true_moe_source_bridge.v1"
BRIDGE_STATUS = "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_READY_FOR_OUTER_PREFLIGHT"
SOURCE_AUTHORITY_SCHEMA = route_plan.AUTHORITY_SCHEMA
SOURCE_AUTHORITY_STATUS = route_plan.AUTHORITY_STATUS
HIDDEN = 2048
TOP_K = 10


class SourceTokenBridgeSealError(RuntimeError):
    """The source-token all-ten bridge cannot safely be sealed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceTokenBridgeSealError(f"{label} must be an object")
    return dict(value)


def _require_sha256(value: object, label: str) -> str:
    try:
        return route_plan._require_sha256(value, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenBridgeSealError(str(exc)) from exc


def _require_evidence(value: object, expected: Mapping[str, Any], label: str) -> None:
    try:
        route_plan._evidence_matches(value, expected, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenBridgeSealError(str(exc)) from exc


def _same_route_weights(observed: object, expected: object) -> bool:
    if not isinstance(observed, list) or not isinstance(expected, list) or len(observed) != TOP_K or len(expected) != TOP_K:
        return False
    for left, right in zip(observed, expected):
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, (int, float))
            or not isinstance(right, (int, float))
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
            or abs(float(left) - float(right)) > 1.0e-6
        ):
            return False
    return True


def _bind_source_authority(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission: Mapping[str, Any],
    admission_receipt: Mapping[str, Any],
    admission_receipt_seal: str,
    prefix: Mapping[str, Any],
    prefix_seal: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    try:
        evidence = route_plan._file_evidence(path, "--source-token-route-authority")
        document, document_seal = route_plan._sealed_json(path, "source-token route authority")
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenBridgeSealError(str(exc)) from exc
    if document.get("schema") != SOURCE_AUTHORITY_SCHEMA or document.get("status") != SOURCE_AUTHORITY_STATUS:
        raise SourceTokenBridgeSealError("source-token route authority schema/status drifted")
    source = _mapping(document.get("source_binding"), "source-token route authority source_binding")
    _require_evidence(source.get("manifest"), manifest, "source-token route authority manifest")
    # Versioned current pointers may be harmlessly resealed while a CPU child
    # runs.  The path, manifest, and immutable receipt must remain exact.
    historical_pointer = _mapping(source.get("admission_current"), "source-token route authority admission")
    if historical_pointer.get("present") is not True or historical_pointer.get("path") != admission.get("path"):
        raise SourceTokenBridgeSealError("source-token route authority admission pointer path drifted")
    _require_sha256(historical_pointer.get("sha256"), "source-token route authority historical admission SHA")
    _require_evidence(source.get("admission_receipt"), admission_receipt, "source-token route authority immutable admission")
    _require_evidence(source.get("first_residual_outer_receipt"), prefix, "source-token route authority prefix")
    if (
        source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
        or source.get("first_residual_outer_seal_sha256") != prefix_seal
    ):
        raise SourceTokenBridgeSealError("source-token route authority immutable identity drifted")
    plan = _mapping(document.get("source_token_plan"), "source-token route authority source_token_plan")
    if plan.get("schema") != route_plan.SOURCE_PLAN_SCHEMA or plan.get("status") != route_plan.SOURCE_PLAN_STATUS:
        raise SourceTokenBridgeSealError("source-token route authority nested plan drifted")
    return evidence, document_seal, document


def _validate_material(
    material: Mapping[str, Any],
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    if material.get("schema") != MATERIAL_SCHEMA or material.get("status") != MATERIAL_STATUS:
        raise SourceTokenBridgeSealError("source-token bridge material schema/status drifted")
    source = _mapping(material.get("source_binding"), "source-token bridge material source_binding")
    _require_evidence(source.get("manifest"), manifest, "source-token bridge material manifest")
    historical_pointer = _mapping(source.get("admission_current"), "source-token bridge material admission")
    if historical_pointer.get("present") is not True or historical_pointer.get("path") != admission.get("path"):
        raise SourceTokenBridgeSealError("source-token bridge material admission pointer path drifted")
    _require_sha256(historical_pointer.get("sha256"), "source-token bridge material historical admission SHA")
    _require_evidence(source.get("admission_receipt"), admission_receipt, "source-token bridge material admission")
    _require_evidence(
        source.get("source_token_route_authority"),
        source_authority_evidence,
        "source-token bridge material route authority",
    )
    _require_evidence(source.get("first_residual_receipt"), prefix, "source-token bridge material prefix")
    if (
        source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
        or source.get("source_token_route_authority_seal_sha256") != source_authority_seal
        or source.get("first_residual_receipt_seal_sha256") != prefix_seal
    ):
        raise SourceTokenBridgeSealError("source-token bridge material immutable identity drifted")
    for field in ("source_audit_seal_sha256",):
        _require_sha256(source.get(field), f"source-token bridge material {field}")

    bridge = _mapping(material.get("typed_bridge"), "source-token bridge material typed_bridge")
    if (
        bridge.get("layer") != 0
        or bridge.get("source_token_id") != 1
        or bridge.get("route_count") != TOP_K
        or bridge.get("first_residual_elements") != HIDDEN
        or bridge.get("same_command_graph_required") is not True
        or bridge.get("first_residual_receipt_seal_sha256") != prefix_seal
        or bridge.get("source_token_route_authority_seal_sha256") != source_authority_seal
    ):
        raise SourceTokenBridgeSealError("source-token bridge geometry/authority drifted")
    _require_sha256(bridge.get("first_residual_output_sha256"), "source-token bridge first residual SHA")
    sections = _mapping(bridge.get("compact_section_sha256"), "source-token compact section hashes")
    if set(sections) != {"gate_scales", "gate_signs", "up_scales", "up_signs", "down_scales", "down_signs"}:
        raise SourceTokenBridgeSealError("source-token bridge compact section inventory drifted")
    for name, digest in sections.items():
        _require_sha256(digest, f"source-token bridge compact section {name}")

    source_plan = _mapping(source_authority_document.get("source_token_plan"), "source-token route authority plan")
    expected_router = _mapping(source_plan.get("source_token_router_evidence"), "source-token route authority router")
    expected_ids = expected_router.get("source_stable_route_ids")
    expected_weights = expected_router.get("source_stable_normalized_weights")
    route = _mapping(material.get("route_authority"), "source-token bridge route authority")
    if (
        route.get("ids") != expected_ids
        or not _same_route_weights(route.get("normalized_weights"), expected_weights)
        or route.get("wave_count") != TOP_K
        or route.get("all_thirty_wave_payloads_use_admission_verified_immutable_snapshots") is not True
    ):
        raise SourceTokenBridgeSealError("source-token bridge route authority drifted")
    scan = _mapping(material.get("artifact_scan"), "source-token bridge artifact scan")
    boundary = _mapping(material.get("claim_boundary"), "source-token bridge claim boundary")
    if (
        scan.get("complete_artifact_admission_performed_once") is not True
        or scan.get("catalog_reused_for_source_token_all_ten_bridge") is not True
        or scan.get("raw_bf16_or_safetensors_opened") is not False
        or boundary.get("cpu_source_token_bridge_material_only") is not True
        or boundary.get("metal_device_or_dispatch_performed") is not False
        or boundary.get("no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim") is not True
    ):
        raise SourceTokenBridgeSealError("source-token bridge scan/claim boundary drifted")
    return bridge, route


def build_receipt(
    *,
    manifest_path: Path,
    admission_path: Path,
    source_authority_path: Path,
    first_residual_path: Path,
    material_path: Path,
) -> dict[str, Any]:
    try:
        manifest, manifest_seal = route_plan._bind_manifest(manifest_path)
        admission, pointer_seal, immutable_admission, immutable_admission_seal = route_plan._bind_admission(
            admission_path, manifest, manifest_seal
        )
        prefix, prefix_seal, _ = route_plan._bind_prefix(
            first_residual_path,
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission_receipt_seal=immutable_admission_seal,
        )
        material_evidence = route_plan._file_evidence(material_path, "--material")
        material = route_plan._read_json(material_path, "--material")
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenBridgeSealError(str(exc)) from exc
    source_authority_evidence, source_authority_seal, source_authority_document = _bind_source_authority(
        source_authority_path,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt=immutable_admission,
        admission_receipt_seal=immutable_admission_seal,
        prefix=prefix,
        prefix_seal=prefix_seal,
    )
    bridge, route = _validate_material(
        material,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt=immutable_admission,
        admission_receipt_seal=immutable_admission_seal,
        source_authority_evidence=source_authority_evidence,
        source_authority_document=source_authority_document,
        source_authority_seal=source_authority_seal,
        prefix=prefix,
        prefix_seal=prefix_seal,
    )
    return seal(
        {
            "schema": BRIDGE_SCHEMA,
            "status": BRIDGE_STATUS,
            "recorded_at": _utc_now(),
            "source_binding": {
                "manifest": manifest,
                "admission_current": admission,
                "admission_receipt": immutable_admission,
                "source_token_route_authority": source_authority_evidence,
                "first_residual_receipt": prefix,
                "bridge_material": material_evidence,
                "manifest_seal_sha256": manifest_seal,
                "admission_pointer_seal_sha256": pointer_seal,
                "admission_receipt_seal_sha256": immutable_admission_seal,
                "source_token_route_authority_seal_sha256": source_authority_seal,
                "first_residual_receipt_seal_sha256": prefix_seal,
            },
            "typed_bridge": bridge,
            "route_authority": route,
            "artifact_scan": {
                "complete_artifact_admission_performed_once_in_material": True,
                "new_scan_performed_by_sealer": False,
                "metal_device_or_dispatch_performed": False,
            },
            "claim_boundary": {
                "sealed_source_token_cpu_bridge_only": True,
                "not_a_device_component_lease_or_layer_token": True,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
                "requires_source_token_outer_preflight_and_fresh_component_lease": True,
            },
        }
    )


def write_new(path: Path, document: Mapping[str, Any]) -> None:
    try:
        route_plan.write_new(path, document)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenBridgeSealError(str(exc)) from exc


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--source-token-route-authority", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if arguments is None else arguments)
    try:
        document = build_receipt(
            manifest_path=args.manifest,
            admission_path=args.admission_current,
            source_authority_path=args.source_token_route_authority,
            first_residual_path=args.first_residual_receipt,
            material_path=args.material,
        )
        write_new(args.out, document)
    except SourceTokenBridgeSealError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_BRIDGE", "error": str(exc)}))
        return 2
    print(json.dumps({"status": document["status"], "out": str(args.out), "seal_sha256": document["seal_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
