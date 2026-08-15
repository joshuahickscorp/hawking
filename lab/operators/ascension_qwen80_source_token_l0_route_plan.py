"""Seal a source-token-specific Qwen80 L0 all-ten route-plan authority.

The older all-ten plan deliberately describes a synthetic post-attention
fixture.  This module accepts only raw material emitted by
``ascension_qwen80_source_token_l0_router_discriminator``: one strict
complete-binary scan which replays the source token retained by the sealed
first-residual prefix baseline.  It is CPU-only and cannot issue a lease,
create a Metal context, or promote a component result into a layer/token
result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import seal, verify


MATERIAL_SCHEMA = "hawking.ascension.qwen80_source_token_l0_router_discriminator_material.v1"
MATERIAL_STATUS = (
    "CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_ROUTER_DISCRIMINATOR_MATERIAL_READY_FOR_SEAL"
)
SOURCE_PLAN_SCHEMA = "hawking.ascension.qwen80_source_token_l0_all_ten_routed_expert_binding_plan.v1"
SOURCE_PLAN_STATUS = "SOURCE_TOKEN_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED"
AUTHORITY_SCHEMA = "hawking.ascension.qwen80_source_token_l0_all_ten_route_plan_authority.v1"
AUTHORITY_STATUS = (
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_ALL_TEN_ROUTE_PLAN_READY_FOR_NEW_TYPED_BRIDGE"
)
MANIFEST_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
ADMISSION_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
ADMISSION_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
ADMISSION_RECEIPT_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
ADMISSION_RECEIPT_STATUS = (
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
)
PREFIX_SCHEMA = "hawking.ascension.qwen80_first_residual_outer_capture.v1"
PREFIX_STATUS = "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY"
OLD_PLAN_SCHEMA = "hawking.ascension.qwen80_all_ten_routed_expert_binding_plan.v1"
OLD_PLAN_STATUS = "SOURCE_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED"

HIDDEN = 2048
TOP_K = 10
EXPERT_COUNT = 512
GROUP_SIZE = 128
WEIGHT_SUM_TOLERANCE = 2.0e-6


class SourceTokenRoutePlanError(RuntimeError):
    """A source-token route authority cannot safely be sealed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise SourceTokenRoutePlanError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceTokenRoutePlanError(f"{label} must be an object")
    return dict(value)


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise SourceTokenRoutePlanError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SourceTokenRoutePlanError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise SourceTokenRoutePlanError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise SourceTokenRoutePlanError(f"{label} must be executable")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise SourceTokenRoutePlanError(f"cannot canonicalize {label}: {exc}") from exc


def _file_evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    clean = _canonical_regular(path, label, executable=executable)
    digest = hashlib.sha256()
    with clean.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return {
        "path": str(clean),
        "present": True,
        "bytes": clean.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    clean = _canonical_regular(path, label)
    try:
        value = json.loads(clean.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceTokenRoutePlanError(f"cannot read {label}: {exc}") from exc
    return _mapping(value, label)


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    value = _read_json(path, label)
    try:
        verify(value, label=str(path))
    except ValueError as exc:
        raise SourceTokenRoutePlanError(f"{label} is not sealed: {exc}") from exc
    return value, _require_sha256(value.get("seal_sha256"), f"{label}.seal_sha256")


def _evidence_matches(value: object, expected: Mapping[str, Any], label: str) -> None:
    observed = _mapping(value, label)
    if observed.get("present") is not True:
        raise SourceTokenRoutePlanError(f"{label} does not attest a present file")
    if (
        observed.get("path") != expected.get("path")
        or observed.get("bytes") != expected.get("bytes")
        or observed.get("sha256") != expected.get("sha256")
    ):
        raise SourceTokenRoutePlanError(f"{label} byte/path identity drifted")


def _require_string(value: Mapping[str, Any], field: str, label: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise SourceTokenRoutePlanError(f"{label}.{field} must be a non-empty string")
    return item


def _bind_manifest(path: Path) -> tuple[dict[str, Any], str]:
    evidence = _file_evidence(path, "--manifest")
    document = _read_json(path, "--manifest")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise SourceTokenRoutePlanError("manifest schema drifted")
    return evidence, _require_sha256(document.get("seal_sha256"), "manifest seal_sha256")


def _bind_admission(
    path: Path, manifest: Mapping[str, Any], manifest_seal: str
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    evidence = _file_evidence(path, "--admission-current")
    document = _read_json(path, "--admission-current")
    if document.get("schema") != ADMISSION_SCHEMA or document.get("status") != ADMISSION_STATUS:
        raise SourceTokenRoutePlanError("admission-current schema/status drifted")
    selected_manifest = _mapping(document.get("complete_manifest"), "admission current complete_manifest")
    if (
        selected_manifest.get("document_sha256") != manifest.get("sha256")
        or selected_manifest.get("seal_sha256") != manifest_seal
    ):
        raise SourceTokenRoutePlanError("admission-current manifest identity drifted")
    selected_receipt = _mapping(document.get("admission_receipt"), "admission current receipt")
    receipt_path = _canonical_regular(
        Path(_require_string(selected_receipt, "path", "admission current receipt")),
        "immutable admission receipt",
    )
    receipt_evidence = _file_evidence(receipt_path, "immutable admission receipt")
    receipt, receipt_seal = _sealed_json(receipt_path, "immutable admission receipt")
    if (
        receipt.get("schema") != ADMISSION_RECEIPT_SCHEMA
        or receipt.get("status") != ADMISSION_RECEIPT_STATUS
        or selected_receipt.get("seal_sha256") != receipt_seal
    ):
        raise SourceTokenRoutePlanError("immutable admission receipt schema/status/seal drifted")
    return (
        evidence,
        _require_sha256(document.get("seal_sha256"), "admission current pointer seal"),
        receipt_evidence,
        receipt_seal,
    )


def _bind_prefix(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_receipt_seal: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    evidence = _file_evidence(path, "--first-residual-receipt")
    document, receipt_seal = _sealed_json(path, "first-residual prefix receipt")
    if document.get("schema") != PREFIX_SCHEMA or document.get("status") != PREFIX_STATUS:
        raise SourceTokenRoutePlanError("first-residual prefix schema/status drifted")
    source = _mapping(document.get("source_binding"), "first-residual prefix source_binding")
    _evidence_matches(source.get("manifest"), manifest, "first-residual prefix manifest")
    if (
        source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise SourceTokenRoutePlanError("first-residual prefix immutable identity drifted")
    output = _mapping(document.get("first_residual_output"), "first-residual prefix output")
    if (
        output.get("layer") != 0
        or output.get("linear_state_slot") != 0
        or output.get("elements") != HIDDEN
        or output.get("same_command_graph_required") is not True
    ):
        raise SourceTokenRoutePlanError("first-residual prefix output geometry/state drifted")
    _require_sha256(output.get("sha256"), "first-residual prefix output SHA")
    return evidence, receipt_seal, document


def _bind_old_fixture_plan(path: Path) -> dict[str, Any]:
    evidence = _file_evidence(path, "--old-route-plan")
    document = _read_json(path, "--old-route-plan")
    if document.get("schema") != OLD_PLAN_SCHEMA or document.get("status") != OLD_PLAN_STATUS:
        raise SourceTokenRoutePlanError("historical fixture plan schema/status drifted")
    router = _mapping(document.get("router_evidence"), "historical fixture plan router_evidence")
    ids = router.get("source_stable_route_ids")
    if not isinstance(ids, list) or len(ids) != TOP_K:
        raise SourceTokenRoutePlanError("historical fixture plan must retain ten IDs")
    return evidence


def _numeric_list(value: object, label: str, *, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SourceTokenRoutePlanError(f"{label} must contain exactly {length} entries")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise SourceTokenRoutePlanError(f"{label}[{index}] must be finite numeric")
        result.append(float(item))
    return result


def _route_ids(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != TOP_K:
        raise SourceTokenRoutePlanError(f"{label} must contain exactly ten IDs")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < EXPERT_COUNT:
            raise SourceTokenRoutePlanError(f"{label}[{index}] must be an expert ID in [0,{EXPERT_COUNT})")
        result.append(item)
    if len(set(result)) != TOP_K:
        raise SourceTokenRoutePlanError(f"{label} contains duplicate experts")
    return result


def _validate_projection(
    projection: object, *, expected_name: str, expected_shape: list[int], label: str
) -> str:
    row = _mapping(projection, label)
    if row.get("tensor_name") != expected_name or row.get("shape") != expected_shape:
        raise SourceTokenRoutePlanError(f"{label} tensor name/shape drifted")
    if row.get("payload_opened_by_this_plan") is not False:
        raise SourceTokenRoutePlanError(f"{label} must not claim payload opened by plan")
    for field in ("artifact_path", "artifact_bytes", "source_dtype", "source_shard", "source_shard_sha256"):
        if field not in row:
            raise SourceTokenRoutePlanError(f"{label}.{field} missing")
    if not isinstance(row["artifact_path"], str) or not row["artifact_path"]:
        raise SourceTokenRoutePlanError(f"{label}.artifact_path invalid")
    if not isinstance(row["artifact_bytes"], int) or row["artifact_bytes"] <= 0:
        raise SourceTokenRoutePlanError(f"{label}.artifact_bytes invalid")
    digest = _require_sha256(row.get("artifact_sha256"), f"{label}.artifact_sha256")
    _require_sha256(row.get("source_shard_sha256"), f"{label}.source_shard_sha256")
    layout = _mapping(row.get("layout"), f"{label}.layout")
    if (
        layout.get("magic") != "HQ30G1B1"
        or layout.get("version") != 1
        or layout.get("group_size") != GROUP_SIZE
        or layout.get("scale_dtype") != "float16"
        or layout.get("sign_bit_order") != "little"
    ):
        raise SourceTokenRoutePlanError(f"{label}.layout drifted")
    return digest


def _validate_material(
    material: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission: Mapping[str, Any],
    admission_receipt: Mapping[str, Any],
    admission_receipt_seal: str,
    prefix: Mapping[str, Any],
    prefix_seal: str,
    prefix_document: Mapping[str, Any],
    old_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if material.get("schema") != MATERIAL_SCHEMA or material.get("status") != MATERIAL_STATUS:
        raise SourceTokenRoutePlanError("source-token router material schema/status drifted")
    source = _mapping(material.get("source_binding"), "source-token router material source_binding")
    _evidence_matches(source.get("manifest"), manifest, "source-token material manifest")
    # The mutable current pointer may be resealed while the child scans.  Its
    # canonical path and the immutable receipt must remain continuous; raw
    # pointer bytes are preserved as historical evidence rather than treated
    # as a false mismatch.
    historical_pointer = _mapping(source.get("admission_current"), "source-token material admission pointer")
    if historical_pointer.get("present") is not True or historical_pointer.get("path") != admission.get("path"):
        raise SourceTokenRoutePlanError("source-token material admission pointer path drifted")
    _require_sha256(historical_pointer.get("sha256"), "source-token material historical admission SHA")
    _evidence_matches(source.get("admission_receipt"), admission_receipt, "source-token material admission receipt")
    _evidence_matches(source.get("first_residual_outer_receipt"), prefix, "source-token material prefix receipt")
    _evidence_matches(source.get("historical_fixture_route_plan"), old_plan, "source-token material old fixture plan")
    if (
        source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise SourceTokenRoutePlanError("source-token material immutable artifact identity drifted")
    for field in ("source_audit_seal_sha256",):
        _require_sha256(source.get(field), f"source-token material {field}")
    if not isinstance(source.get("source_revision"), str) or not source["source_revision"]:
        raise SourceTokenRoutePlanError("source-token material source revision missing")

    plan = _mapping(material.get("source_token_plan"), "source-token route plan")
    if plan.get("schema") != SOURCE_PLAN_SCHEMA or plan.get("status") != SOURCE_PLAN_STATUS:
        raise SourceTokenRoutePlanError("source-token route plan schema/status drifted")
    if plan.get("layer") != 0:
        raise SourceTokenRoutePlanError("source-token route plan must bind layer 0")
    provenance = _mapping(plan.get("source_input_provenance"), "source-token input provenance")
    if provenance.get("source_token_id") != 1 or provenance.get("same_input_state_identity_required") is not True:
        raise SourceTokenRoutePlanError("source-token input/state identity drifted")
    _evidence_matches(provenance.get("prefix_outer_receipt"), prefix, "source-token plan prefix receipt")
    if provenance.get("prefix_outer_receipt_seal_sha256") != prefix_seal:
        raise SourceTokenRoutePlanError("source-token plan prefix seal drifted")
    prefix_output = _mapping(prefix_document.get("first_residual_output"), "first-residual prefix output")
    if provenance.get("strict_metal_prefix_first_residual_sha256") != prefix_output.get("sha256"):
        raise SourceTokenRoutePlanError("source-token plan strict prefix output drifted")
    prefix_source = _mapping(prefix_document.get("source_binding"), "first-residual prefix source_binding")
    _evidence_matches(
        provenance.get("cpu_baseline_receipt"),
        _mapping(prefix_source.get("cpu_baseline_receipt"), "first-residual prefix CPU baseline"),
        "source-token plan CPU baseline",
    )
    for field in (
        "input_hidden_f32le_sha256",
        "cpu_first_residual_f32le_sha256",
        "zero_conv_state_f32le_sha256",
        "zero_recurrent_state_f32le_sha256",
    ):
        _require_sha256(provenance.get(field), f"source-token plan {field}")

    router = _mapping(plan.get("source_token_router_evidence"), "source-token router evidence")
    if router.get("derived_from_direct_packed_source_token_l0_cpu_oracle") is not True or router.get("router_component_only") is not True:
        raise SourceTokenRoutePlanError("source-token router provenance drifted")
    _require_sha256(router.get("post_attention_normalized_hidden_f32le_sha256"), "postnorm SHA")
    _require_sha256(router.get("router_logits_f32le_sha256"), "router logits SHA")
    ids = _route_ids(router.get("source_stable_route_ids"), "source-token route IDs")
    weights = _numeric_list(router.get("source_stable_normalized_weights"), "source-token route weights", length=TOP_K)
    if any(weight < 0.0 for weight in weights) or abs(sum(weights) - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise SourceTokenRoutePlanError("source-token route weights are not normalized")
    inventory = _mapping(plan.get("manifest_descriptor_inventory"), "source-token descriptor inventory")
    if (
        inventory.get("inventory_document_sha256") != manifest.get("sha256")
        or inventory.get("manifest_schema") != MANIFEST_SCHEMA
        or inventory.get("manifest_seal_sha256") != manifest_seal
        or inventory.get("resolved_route_tensor_count") != TOP_K * 3
        or inventory.get("payload_opened_by_this_plan") is not False
    ):
        raise SourceTokenRoutePlanError("source-token descriptor inventory drifted")
    waves = plan.get("deterministic_waves")
    if not isinstance(waves, list) or len(waves) != TOP_K:
        raise SourceTokenRoutePlanError("source-token plan requires exactly ten waves")
    artifact_hashes: set[str] = set()
    for index, wave_value in enumerate(waves):
        wave = _mapping(wave_value, f"source-token wave {index}")
        expert = ids[index]
        if (
            wave.get("wave_index") != index
            or wave.get("layer") != 0
            or wave.get("expert_id") != expert
            or wave.get("route_execution_status") != "NOT_EXECUTED_SOURCE_TOKEN_BOUND_PLAN_ONLY"
            or wave.get("route_delta_materialized") is not False
            or wave.get("route_weight_applied") is not False
        ):
            raise SourceTokenRoutePlanError(f"source-token wave {index} identity/status drifted")
        value = wave.get("normalized_weight")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or abs(float(value) - weights[index]) > 0.0:
            raise SourceTokenRoutePlanError(f"source-token wave {index} weight drifted")
        stem = f"model.layers.0.mlp.experts.{expert}"
        for field, name, shape in (
            ("gate", f"{stem}.gate_proj.weight", [512, 2048]),
            ("up", f"{stem}.up_proj.weight", [512, 2048]),
            ("down", f"{stem}.down_proj.weight", [2048, 512]),
        ):
            digest = _validate_projection(wave.get(field), expected_name=name, expected_shape=shape, label=f"wave {index} {field}")
            if digest in artifact_hashes:
                raise SourceTokenRoutePlanError("source-token plan reuses a projection payload")
            artifact_hashes.add(digest)
    if len(artifact_hashes) != TOP_K * 3:
        raise SourceTokenRoutePlanError("source-token plan lacks thirty unique projection bindings")
    gate = _mapping(plan.get("rawls_real_all_ten_provenance_gate"), "source-token rawls gate")
    if (
        gate.get("all_ten_source_bindings_complete") is not True
        or gate.get("expected_layer") != 0
        or gate.get("route_order") != ids
        or gate.get("normalized_weights") != [float(weight) for weight in weights]
        or gate.get("execution_receipt_required_for_each_wave") is not True
        or gate.get("rejects_tensor_substitution") is not True
        or gate.get("rejects_route_reorder") is not True
        or gate.get("rejects_duplicate_experts") is not True
        or gate.get("rejects_missing_tensor_or_weight") is not True
    ):
        raise SourceTokenRoutePlanError("source-token all-ten provenance gate drifted")
    if any(plan.get(field) is not False for field in (
        "route_execution_performed", "route_combine_performed", "shared_expert_performed",
        "residual_combine_performed", "metal_device_or_dispatch_performed", "model_execution_performed",
        "hcli_execution_performed", "tps_or_tg_measurement_performed", "complete_layer_or_decoder_claim_earned",
    )):
        raise SourceTokenRoutePlanError("source-token plan claim boundary drifted")
    divergence = _mapping(material.get("fixture_divergence"), "fixture divergence")
    _evidence_matches(divergence.get("old_route_plan"), old_plan, "fixture divergence old plan")
    if divergence.get("conclusion") != "the fixture-derived route plan is prohibited from driving the source-token true-MoE graph":
        raise SourceTokenRoutePlanError("fixture divergence conclusion weakened")
    scan = _mapping(material.get("artifact_scan"), "source-token material artifact scan")
    boundary = _mapping(material.get("claim_boundary"), "source-token material claim boundary")
    if (
        scan.get("complete_artifact_admission_performed_once") is not True
        or scan.get("catalog_reused_for_embedding_mixer_router_and_all_thirty_descriptors") is not True
        or scan.get("raw_bf16_or_safetensors_opened") is not False
        or boundary.get("cpu_discriminator_and_descriptor_plan_only") is not True
        or boundary.get("metal_device_or_dispatch_performed") is not False
        or boundary.get("no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim") is not True
    ):
        raise SourceTokenRoutePlanError("source-token material scan/claim boundary drifted")
    return plan


def build_authority(
    *,
    manifest_path: Path,
    admission_path: Path,
    first_residual_path: Path,
    old_plan_path: Path,
    material_path: Path,
) -> dict[str, Any]:
    manifest, manifest_seal = _bind_manifest(manifest_path)
    admission, admission_pointer_seal, admission_receipt, admission_receipt_seal = _bind_admission(
        admission_path, manifest, manifest_seal
    )
    prefix, prefix_seal, prefix_document = _bind_prefix(
        first_residual_path,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_receipt_seal=admission_receipt_seal,
    )
    old_plan = _bind_old_fixture_plan(old_plan_path)
    material_evidence = _file_evidence(material_path, "--material")
    material = _read_json(material_path, "--material")
    plan = _validate_material(
        material,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt=admission_receipt,
        admission_receipt_seal=admission_receipt_seal,
        prefix=prefix,
        prefix_seal=prefix_seal,
        prefix_document=prefix_document,
        old_plan=old_plan,
    )
    return seal(
        {
            "schema": AUTHORITY_SCHEMA,
            "status": AUTHORITY_STATUS,
            "recorded_at": _utc_now(),
            "source_binding": {
                "manifest": manifest,
                "admission_current": admission,
                "admission_receipt": admission_receipt,
                "first_residual_outer_receipt": prefix,
                "historical_fixture_route_plan": old_plan,
                "raw_material": material_evidence,
                "manifest_seal_sha256": manifest_seal,
                "admission_pointer_seal_sha256": admission_pointer_seal,
                "admission_receipt_seal_sha256": admission_receipt_seal,
                "first_residual_outer_seal_sha256": prefix_seal,
            },
            "source_token_plan": plan,
            "fixture_divergence": material["fixture_divergence"],
            "artifact_scan": {
                "complete_artifact_admission_performed_once_in_material": True,
                "new_scan_performed_by_sealer": False,
                "metal_device_or_dispatch_performed": False,
            },
            "claim_boundary": {
                "sealed_cpu_only_route_authority": True,
                "not_a_typed_device_bridge_or_component_lease": True,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
                "requires_new_source_token_typed_bridge_outer_preflight_and_fresh_component_lease": True,
            },
        }
    )


def write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise SourceTokenRoutePlanError("--out must be absolute with an existing parent")
    if path.exists():
        raise SourceTokenRoutePlanError(f"refusing to overwrite {path}")
    raw = json.dumps(dict(document), indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise SourceTokenRoutePlanError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--old-route-plan", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if arguments is None else arguments)
    try:
        authority = build_authority(
            manifest_path=args.manifest,
            admission_path=args.admission_current,
            first_residual_path=args.first_residual_receipt,
            old_plan_path=args.old_route_plan,
            material_path=args.material,
        )
        write_new(args.out, authority)
    except SourceTokenRoutePlanError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_SOURCE_TOKEN_ROUTE_PLAN_AUTHORITY", "error": str(exc)}))
        return 2
    print(json.dumps({"status": authority["status"], "out": str(args.out), "seal_sha256": authority["seal_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
