"""Seal a current admitted Qwen80 all-ten source-bridge authority.

The Rust material producer is the only code that opens the complete compact
artifact for this bridge.  This wrapper deliberately does not run a device
stage: it binds that immutable material to the current admission, the sealed
strict-Metal first-residual outer capture, and the sealed router authority,
then writes a receipt which a later outer-reaped component child can consume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_true_input_all_ten_moe_graph_launcher as launcher
from lab.receipts import seal


MATERIAL_SCHEMA = "hawking.ascension.qwen80_all_ten_true_moe_source_bridge_material.v1"
MATERIAL_STATUS = (
    "CURRENT_ADMITTED_QWEN80_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_MATERIAL_READY_FOR_SEAL"
)


class SourceBridgeSealError(RuntimeError):
    """The typed bridge cannot safely bind the supplied authority."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceBridgeSealError(f"{label} must be an object")
    return dict(value)


def _sha256(value: object, label: str) -> str:
    try:
        return launcher._require_sha256(value, label)
    except launcher.TrueInputAllTenMoeGraphLauncherError as exc:
        raise SourceBridgeSealError(str(exc)) from exc


def _require_evidence(value: object, expected: Mapping[str, Any], label: str) -> None:
    evidence = _mapping(value, label)
    if evidence.get("present") is not True:
        raise SourceBridgeSealError(f"{label} does not attest a present file")
    if (
        evidence.get("path") != expected.get("path")
        or evidence.get("bytes") != expected.get("bytes")
        or evidence.get("sha256") != expected.get("sha256")
    ):
        raise SourceBridgeSealError(f"{label} byte/path identity drifted")


def _validate_material(
    material: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission: Mapping[str, Any],
    admission_pointer_seal: str,
    admission_receipt_seal: str,
    router: Mapping[str, Any],
    router_outer: Mapping[str, Any],
    router_outer_seal: str,
    route_plan: Mapping[str, Any],
    route_ids: Sequence[int],
    route_weights: Sequence[float],
    first_residual: Mapping[str, Any],
    first_residual_seal: str,
    first_residual_output_sha256: str,
) -> dict[str, Any]:
    if material.get("schema") != MATERIAL_SCHEMA or material.get("status") != MATERIAL_STATUS:
        raise SourceBridgeSealError("source bridge material schema/status drifted")
    source = _mapping(material.get("source_binding"), "source bridge material source_binding")
    _require_evidence(source.get("manifest"), manifest, "source bridge material manifest")
    # The material child can run long enough for the versioned current pointer
    # to be harmlessly resealed.  Treat its raw pointer evidence as historical
    # while binding this wrapper to a freshly validated *current* pointer path
    # plus the immutable manifest/admission-receipt authority.
    historical_admission = _mapping(
        source.get("admission_current"), "source bridge material admission"
    )
    if (
        historical_admission.get("present") is not True
        or historical_admission.get("path") != admission.get("path")
    ):
        raise SourceBridgeSealError("source bridge material admission pointer path drifted")
    _sha256(historical_admission.get("sha256"), "source bridge historical admission SHA")
    _require_evidence(source.get("router_receipt"), router, "source bridge material router")
    _require_evidence(
        source.get("router_outer_receipt"), router_outer, "source bridge material router outer"
    )
    _require_evidence(source.get("route_plan"), route_plan, "source bridge material route plan")
    _require_evidence(
        source.get("first_residual_receipt"),
        first_residual,
        "source bridge material first residual",
    )
    if (
        source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
        or source.get("router_outer_receipt_seal_sha256") != router_outer_seal
    ):
        raise SourceBridgeSealError("source bridge material authority seal drifted")
    _sha256(
        source.get("admission_pointer_seal_sha256"),
        "source bridge historical admission pointer seal",
    )
    baseline = _mapping(
        source.get("first_residual_cpu_baseline"), "source bridge material CPU baseline"
    )
    _sha256(baseline.get("sha256"), "source bridge material CPU baseline SHA")

    bridge = _mapping(material.get("typed_bridge"), "source bridge material typed_bridge")
    if (
        bridge.get("layer") != 0
        or bridge.get("route_count") != launcher.TOP_K
        or bridge.get("first_residual_elements") != launcher.HIDDEN
        or bridge.get("same_command_graph_required") is not True
        or bridge.get("first_residual_output_sha256") != first_residual_output_sha256
        or bridge.get("first_residual_receipt_seal_sha256") != first_residual_seal
    ):
        raise SourceBridgeSealError("source bridge material first-residual identity drifted")
    compact = _mapping(bridge.get("compact_section_sha256"), "source bridge compact sections")
    expected_sections = {
        "gate_scales",
        "gate_signs",
        "up_scales",
        "up_signs",
        "down_scales",
        "down_signs",
    }
    if set(compact) != expected_sections:
        raise SourceBridgeSealError("source bridge compact-section inventory drifted")
    for name in sorted(expected_sections):
        _sha256(compact.get(name), f"source bridge compact section {name}")

    route = _mapping(material.get("route_authority"), "source bridge route authority")
    if route.get("ids") != list(route_ids) or route.get("wave_count") != launcher.TOP_K:
        raise SourceBridgeSealError("source bridge route ID/wave authority drifted")
    observed_weights = route.get("normalized_weights")
    if not isinstance(observed_weights, list) or len(observed_weights) != launcher.TOP_K:
        raise SourceBridgeSealError("source bridge route weights missing")
    for index, (observed, expected) in enumerate(zip(observed_weights, route_weights)):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise SourceBridgeSealError(f"source bridge route weight {index} is invalid")
        if abs(float(observed) - float(expected)) > launcher.ROUTER_WEIGHT_TOLERANCE:
            raise SourceBridgeSealError(f"source bridge route weight {index} drifted")
    if route.get("all_thirty_wave_payloads_use_admission_verified_immutable_snapshots") is not True:
        raise SourceBridgeSealError("source bridge lacks immutable all-ten payload authority")

    scan = _mapping(material.get("artifact_scan"), "source bridge artifact scan")
    if (
        scan.get("complete_artifact_admission_performed_once") is not True
        or scan.get("catalog_reused_for_all_ten_source_bridge") is not True
        or scan.get("raw_bf16_or_safetensors_opened") is not False
    ):
        raise SourceBridgeSealError("source bridge material did not preserve one strict packed scan")
    boundary = _mapping(material.get("claim_boundary"), "source bridge claim boundary")
    if (
        boundary.get("metal_device_or_dispatch_performed") is not False
        or boundary.get("no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim")
        is not True
        or boundary.get("material_requires_separate_sealed_component_lease_and_outer_reaped_capture")
        is not True
    ):
        raise SourceBridgeSealError("source bridge material claim boundary drifted")
    return bridge


def build_receipt(
    *,
    manifest_path: Path,
    admission_path: Path,
    router_path: Path,
    router_outer_path: Path,
    route_plan_path: Path,
    first_residual_path: Path,
    material_path: Path,
) -> dict[str, Any]:
    try:
        manifest, manifest_seal = launcher._bind_manifest(manifest_path)
        admission, admission_pointer_seal, admission_receipt_seal = launcher._bind_admission(
            admission_path, manifest, manifest_seal
        )
        router, router_outer, router_outer_seal = launcher._bind_router(
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission=admission,
            admission_receipt_seal=admission_receipt_seal,
            router_path=router_path,
            router_outer_path=router_outer_path,
        )
        route_plan = launcher._file_evidence(route_plan_path, "--route-plan")
        route_ids, route_weights = launcher._route_ids_and_weights(
            launcher._read_json(route_plan_path, "--route-plan")
        )
        first_residual, first_residual_seal, output_sha = launcher._bind_first_residual(
            first_residual_path,
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission=admission,
            admission_receipt_seal=admission_receipt_seal,
        )
        material_evidence = launcher._file_evidence(material_path, "--material")
        material = launcher._read_json(material_path, "--material")
    except launcher.TrueInputAllTenMoeGraphLauncherError as exc:
        raise SourceBridgeSealError(str(exc)) from exc
    bridge = _validate_material(
        material,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_pointer_seal=admission_pointer_seal,
        admission_receipt_seal=admission_receipt_seal,
        router=router,
        router_outer=router_outer,
        router_outer_seal=router_outer_seal,
        route_plan=route_plan,
        route_ids=route_ids,
        route_weights=route_weights,
        first_residual=first_residual,
        first_residual_seal=first_residual_seal,
        first_residual_output_sha256=output_sha,
    )
    return seal(
        {
            "schema": launcher.BRIDGE_SCHEMA,
            "status": launcher.BRIDGE_STATUS,
            "recorded_at": _utc_now(),
            "source_binding": {
                "manifest": manifest,
                "admission_current": admission,
                "route_plan": route_plan,
                "first_residual_receipt": first_residual,
                "router_receipt": router,
                "router_outer_receipt": router_outer,
                "bridge_material": material_evidence,
                "manifest_seal_sha256": manifest_seal,
                "admission_pointer_seal_sha256": admission_pointer_seal,
                "admission_receipt_seal_sha256": admission_receipt_seal,
                "router_outer_receipt_seal_sha256": router_outer_seal,
            },
            "typed_bridge": bridge,
            "bridge_material": {
                "schema": MATERIAL_SCHEMA,
                "status": MATERIAL_STATUS,
                "complete_artifact_admission_performed_once": True,
                "catalog_reused_for_all_ten_source_bridge": True,
                "raw_bf16_or_safetensors_opened": False,
            },
            "claim_boundary": {
                "sealed_source_authority_only": True,
                "no_metal_device_or_dispatch_performed_by_this_wrapper": True,
                "does_not_execute_a_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament": True,
                "requires_fresh_component_only_quiet_lease_and_outer_reaped_capture": True,
            },
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise SourceBridgeSealError("--out must be absolute")
    if path.exists():
        raise SourceBridgeSealError(f"refusing to overwrite immutable --out {path}")
    if not path.parent.is_dir():
        raise SourceBridgeSealError("--out parent must already exist")
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--router-receipt", type=Path, required=True)
    parser.add_argument("--router-outer-receipt", type=Path, required=True)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if arguments is None else arguments)
    try:
        receipt = build_receipt(
            manifest_path=args.manifest,
            admission_path=args.admission_current,
            router_path=args.router_receipt,
            router_outer_path=args.router_outer_receipt,
            route_plan_path=args.route_plan,
            first_residual_path=args.first_residual_receipt,
            material_path=args.material,
        )
        _write_new(args.out, receipt)
    except SourceBridgeSealError as exc:
        print(f"Qwen80 all-ten source bridge seal refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"receipt": str(args.out), "seal_sha256": receipt["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
