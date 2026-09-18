#!/usr/bin/env python3
"""Compare admitted Flash source/native token-0 boundary captures in order.

The comparison is CPU-only and fail-closed.  It re-admits the machine-owner
teacher and both sealed capture identities before reserving output, then stops
at the first exact payload or ordered-route difference.  Continuous metrics
are diagnostic because no numerical acceptance bound has yet been earned.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey import flash_external_greedy_reference_mlx as oracle  # noqa: E402
from tools.odyssey import flash_repeated_accepted_decode as gate  # noqa: E402
from tools.odyssey import flash_source_boundary_capture_mlx as source_capture  # noqa: E402
from tools.odyssey import flash_source_boundary_native as native  # noqa: E402
from tools.odyssey import flash_source_boundary_preflight as preflight  # noqa: E402


SCHEMA = "hawking.flash.source_boundary_comparison.v1"
DIFFERENCE_STATUS = "FIRST_OBSERVED_PAYLOAD_DIFFERENCE_LOCALIZED__NO_SEMANTIC_VERDICT"
EXACT_STATUS = "ALL_PREDECLARED_PAYLOADS_EXACT__NATIVE_PLE_STILL_ABSENT"
RUST_SCHEMA = "hawking.gravity.flash_boundary_payload_compare.v1"
RUST_STATUS = "COMPARED_ADMITTED_PAYLOAD_PAIR__NO_SEMANTIC_VERDICT"


def _metrics(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    if not reference or len(reference) != len(candidate):
        raise ValueError("continuous boundary vectors must have equal nonzero extent")
    if any(not math.isfinite(value) for value in [*reference, *candidate]):
        raise ValueError("continuous boundary vectors contain non-finite values")
    errors = [observed - expected for expected, observed in zip(reference, candidate, strict=True)]
    max_abs = max(abs(value) for value in errors)
    error_l2 = math.sqrt(math.fsum(value * value for value in errors))
    reference_l2 = math.sqrt(math.fsum(value * value for value in reference))
    candidate_l2 = math.sqrt(math.fsum(value * value for value in candidate))
    relative_l2 = error_l2 / reference_l2 if reference_l2 else (0.0 if error_l2 == 0.0 else None)
    denominator = reference_l2 * candidate_l2
    cosine = (
        math.fsum(left * right for left, right in zip(reference, candidate, strict=True)) / denominator
        if denominator
        else None
    )
    return {
        "finite": True,
        "elements": len(reference),
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "reference_l2_zero": reference_l2 == 0.0,
        "candidate_l2_zero": candidate_l2 == 0.0,
    }


def _payload_values(
    binding: object,
    *,
    base: Path,
    label: str,
    dtype: str,
) -> tuple[dict[str, Any], list[float] | list[int]]:
    verified = source_capture._payload(binding, out_dir=base, label=label, dtype=dtype)
    raw = Path(verified["path"]).read_bytes()
    if dtype == "F32_LE":
        values: list[float] | list[int] = [value[0] for value in struct.iter_unpack("<f", raw)]
    elif dtype == "I32_LE":
        values = [value[0] for value in struct.iter_unpack("<i", raw)]
    else:  # The caller owns the closed dtype set.
        raise ValueError(f"unsupported boundary payload dtype: {dtype}")
    if len(values) != verified["elements"]:
        raise ValueError(f"{label} decoded extent differs")
    return verified, values


def _rust_compare(
    binary: Path,
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    dtype: str,
) -> dict[str, Any]:
    if reference["elements"] != candidate["elements"]:
        raise ValueError("Rust boundary comparison inputs have different extents")
    completed = subprocess.run(
        [
            str(binary),
            "gravity",
            "boundary-payload-compare",
            "--reference",
            reference["path"],
            "--candidate",
            candidate["path"],
            "--dtype",
            dtype,
            "--expected-elements",
            str(reference["elements"]),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(completed.stdout)
    result = document.get("comparison")
    if (
        document.get("schema") != RUST_SCHEMA
        or document.get("status") != RUST_STATUS
        or document.get("model_loaded") is not False
        or document.get("gpu_or_metal_started") is not False
        or document.get("promotion_allowed") is not False
        or not isinstance(result, dict)
        or result.get("schema") != "hawking.flash.boundary_payload_comparison.v1"
        or result.get("dtype") != dtype
        or result.get("elements") != reference["elements"]
        or result.get("reference_sha256") != reference["sha256"]
        or result.get("candidate_sha256") != candidate["sha256"]
        or result.get("numerical_acceptance_bound") is not None
        or result.get("semantic_verdict_allowed") is not False
    ):
        raise ValueError("Rust boundary metric owner returned an incompatible identity or scope")
    return result


def _same_optional_float(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(
        float(left), float(right), rel_tol=1e-12, abs_tol=1e-15
    )


def _continuous_record(
    source_record: dict[str, Any],
    native_record: dict[str, Any],
    *,
    source_dir: Path,
    native_dir: Path,
    label: str,
    rust_binary: Path | None = None,
) -> dict[str, Any]:
    source_binding, source_values = _payload_values(
        source_record.get("payload"), base=source_dir, label=f"source {label}", dtype="F32_LE"
    )
    native_binding, native_values = _payload_values(
        native_record.get("payload"), base=native_dir, label=f"native {label}", dtype="F32_LE"
    )
    if source_binding["elements"] != native_binding["elements"]:
        raise ValueError(f"source/native {label} payload extents differ")
    python_metrics = _metrics(
        [float(value) for value in source_values],
        [float(value) for value in native_values],
    )
    result = {
        "source_payload": source_binding,
        "native_payload": native_binding,
        "exact_sha256": source_binding["sha256"] == native_binding["sha256"],
        "metrics": python_metrics,
    }
    if rust_binary is not None:
        rust = _rust_compare(
            rust_binary,
            reference=source_binding,
            candidate=native_binding,
            dtype="F32_LE",
        )
        rust_metrics = rust.get("continuous")
        if (
            rust.get("exact_sha256") != result["exact_sha256"]
            or not isinstance(rust_metrics, dict)
            or any(
                not _same_optional_float(python_metrics[field], rust_metrics.get(field))
                for field in ("max_abs", "relative_l2", "cosine")
            )
            or any(
                python_metrics[field] != rust_metrics.get(field)
                for field in ("finite", "elements", "reference_l2_zero", "candidate_l2_zero")
            )
        ):
            raise ValueError("Rust/Python continuous boundary metric owners disagree")
        result["rust_owner"] = rust
        result["independent_python_crosscheck_exact"] = True
    return result


def _route_record(
    source_record: dict[str, Any],
    native_record: dict[str, Any],
    *,
    source_dir: Path,
    native_dir: Path,
    rust_binary: Path | None = None,
) -> dict[str, Any]:
    source_ids_binding, source_ids = _payload_values(
        source_record.get("ids_payload"), base=source_dir, label="source route ids", dtype="I32_LE"
    )
    native_ids_binding, native_ids = _payload_values(
        native_record.get("ids_payload"), base=native_dir, label="native route ids", dtype="I32_LE"
    )
    source_weights_binding, source_weights = _payload_values(
        source_record.get("weights_payload"),
        base=source_dir,
        label="source route weights",
        dtype="F32_LE",
    )
    native_weights_binding, native_weights = _payload_values(
        native_record.get("weights_payload"),
        base=native_dir,
        label="native route weights",
        dtype="F32_LE",
    )
    if len(source_ids) != 10 or len(native_ids) != 10:
        raise ValueError("source/native route id payloads must each contain exactly ten entries")
    weight_metrics = _metrics(
        [float(value) for value in source_weights],
        [float(value) for value in native_weights],
    )
    result = {
        "source_ids_payload": source_ids_binding,
        "native_ids_payload": native_ids_binding,
        "source_weights_payload": source_weights_binding,
        "native_weights_payload": native_weights_binding,
        "ordered_route_ids_exact": source_ids == native_ids,
        "route_weights_exact_sha256": (
            source_weights_binding["sha256"] == native_weights_binding["sha256"]
        ),
        "route_weight_metrics": weight_metrics,
    }
    if rust_binary is not None:
        rust_ids = _rust_compare(
            rust_binary,
            reference=source_ids_binding,
            candidate=native_ids_binding,
            dtype="I32_LE",
        )
        rust_weights = _rust_compare(
            rust_binary,
            reference=source_weights_binding,
            candidate=native_weights_binding,
            dtype="F32_LE",
        )
        rust_weight_metrics = rust_weights.get("continuous")
        if (
            rust_ids.get("ordered_i32_exact") != result["ordered_route_ids_exact"]
            or rust_weights.get("exact_sha256") != result["route_weights_exact_sha256"]
            or not isinstance(rust_weight_metrics, dict)
            or any(
                not _same_optional_float(weight_metrics[field], rust_weight_metrics.get(field))
                for field in ("max_abs", "relative_l2", "cosine")
            )
        ):
            raise ValueError("Rust/Python route boundary metric owners disagree")
        result["rust_id_owner"] = rust_ids
        result["rust_weight_owner"] = rust_weights
        result["independent_python_crosscheck_exact"] = True
    return result


def _compare_seams(
    source_seams: list[dict[str, Any]],
    native_seams: list[dict[str, Any]],
    *,
    source_dir: Path,
    native_dir: Path,
    rust_binary: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    declarations = preflight._required_seams()
    if len(source_seams) != len(declarations) or len(native_seams) != len(declarations):
        raise ValueError("source/native capture omitted a predeclared seam")
    compared: list[dict[str, Any]] = []
    first_difference: dict[str, Any] | None = None
    for declaration, source_record, native_record in zip(
        declarations, source_seams, native_seams, strict=True
    ):
        if any(
            not isinstance(record, dict)
            or any(record.get(field) != declaration[field] for field in ("ordinal", "layer", "stage", "shape"))
            for record in (source_record, native_record)
        ):
            raise ValueError("source/native seam identity differs from the canonical declaration")
        ordinal = declaration["ordinal"]
        result: dict[str, Any] = dict(declaration)
        if ordinal == 11:
            result["comparison"] = _route_record(
                source_record,
                native_record,
                source_dir=source_dir,
                native_dir=native_dir,
                rust_binary=rust_binary,
            )
            differs = not (
                result["comparison"]["ordered_route_ids_exact"]
                and result["comparison"]["route_weights_exact_sha256"]
            )
            difference_kind = "ORDERED_ROUTE_IDS_OR_WEIGHT_BYTES"
        else:
            result["comparison"] = _continuous_record(
                source_record,
                native_record,
                source_dir=source_dir,
                native_dir=native_dir,
                label=f"seam {ordinal}/{declaration['stage']}",
                rust_binary=rust_binary,
            )
            differs = not result["comparison"]["exact_sha256"]
            difference_kind = "F32_PAYLOAD_BYTES"
        if ordinal == 3:
            result["interpretation_boundary"] = (
                "control injection equality only; native PLE is absent and this seam cannot evidence native PLE parity"
            )
        compared.append(result)
        if differs:
            first_difference = {
                "ordinal": ordinal,
                "layer": declaration["layer"],
                "stage": declaration["stage"],
                "observed_difference_kind": difference_kind,
                "semantic_attribution": "WITHHELD_NO_PREDECLARED_NUMERICAL_ACCEPTANCE_BOUND",
            }
            break
    return compared, first_difference


def _compare_layer0_trace(
    source_trace: object,
    native_trace: object,
    *,
    source_dir: Path,
    native_dir: Path,
    rust_binary: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Compare the fine-grained layer-0 trace before the coarse seam list."""
    declarations = preflight._required_layer0_trace()
    if (
        not isinstance(source_trace, list)
        or not isinstance(native_trace, list)
        or len(source_trace) != len(declarations)
        or len(native_trace) != len(declarations)
    ):
        raise ValueError("source/native capture omitted a predeclared layer-0 trace payload")
    compared: list[dict[str, Any]] = []
    for declaration, source_record, native_record in zip(
        declarations, source_trace, native_trace, strict=True
    ):
        if any(
            not isinstance(record, dict)
            or any(record.get(field) != declaration[field] for field in ("ordinal", "layer", "stage", "shape"))
            for record in (source_record, native_record)
        ):
            raise ValueError("source/native layer-0 trace identity differs from the canonical declaration")
        result: dict[str, Any] = dict(declaration)
        result["comparison"] = _continuous_record(
            source_record,
            native_record,
            source_dir=source_dir,
            native_dir=native_dir,
            label=f"layer-0 trace {declaration['ordinal']}/{declaration['stage']}",
            rust_binary=rust_binary,
        )
        compared.append(result)
        if not result["comparison"]["exact_sha256"]:
            return compared, {
                "comparison_scope": "layer0_trace",
                "ordinal": declaration["ordinal"],
                "layer": declaration["layer"],
                "stage": declaration["stage"],
                "observed_difference_kind": "F32_PAYLOAD_BYTES",
                "semantic_attribution": "WITHHELD_NO_PREDECLARED_NUMERICAL_ACCEPTANCE_BOUND",
            }
    return compared, None


def _compare_cache(
    source_cache: dict[str, Any],
    native_cache: dict[str, Any],
    *,
    source_dir: Path,
    native_dir: Path,
    rust_binary: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    compared: list[dict[str, Any]] = []
    for boundary in ("pre_attention", "post_attention"):
        source_records = source_cache.get(boundary)
        native_records = native_cache.get(boundary)
        if not isinstance(source_records, list) or not isinstance(native_records, list):
            raise ValueError(f"source/native layer-4 {boundary} cache arrays are absent")
        if len(source_records) != len(native_records):
            raise ValueError(f"source/native layer-4 {boundary} cache array counts differ")
        for index, (source_record, native_record) in enumerate(
            zip(source_records, native_records, strict=True)
        ):
            if not isinstance(source_record, dict) or not isinstance(native_record, dict):
                raise ValueError("source/native layer-4 cache record is malformed")
            source_role = source_record.get("state_role")
            expected_native_name = (
                source_role.replace("_", "-") if isinstance(source_role, str) else None
            )
            if native_record.get("name") != expected_native_name:
                raise ValueError("source/native layer-4 cache state-role mapping differs")
            result = {
                "boundary": boundary,
                "array_index": index,
                "state_role": source_role,
                "native_state_name": native_record.get("name"),
                "source_shape": source_record.get("shape"),
                "native_shape": native_record.get("shape"),
                "comparison": _continuous_record(
                    source_record,
                    native_record,
                    source_dir=source_dir,
                    native_dir=native_dir,
                    label=f"layer-4 {boundary} cache {index}",
                    rust_binary=rust_binary,
                ),
            }
            result["shape_exact"] = result["source_shape"] == result["native_shape"]
            result["layout_boundary"] = (
                "source shape retained; native diagnostic is explicitly flattened; equal extent and ordering are required"
            )
            compared.append(result)
            if not result["comparison"]["exact_sha256"]:
                return compared, {
                    "boundary": boundary,
                    "array_index": index,
                    "observed_difference_kind": "F32_CACHE_PAYLOAD_BYTES",
                    "semantic_attribution": "WITHHELD_NO_PREDECLARED_NUMERICAL_ACCEPTANCE_BOUND",
                }
    return compared, None


def compare(
    *,
    model_root: Path,
    reference: Path,
    owner_authorization: Path | None,
    preflight_path: Path,
    source_capture_path: Path,
    native_capture_path: Path,
    hawking_binary: Path,
    out: Path,
    diagnostic_attention_hc_source_bf16_norm: bool = False,
    diagnostic_attention_hc_norm_threadgroup: int | None = None,
    diagnostic_attention_hc_norm_full_pairwise: bool = False,
    diagnostic_attention_hc_source_bf16_stages: bool = False,
    layer0_hc_trace_only: bool = False,
) -> dict[str, Any]:
    diagnostic_threadgroup = native._diagnostic_attention_hc_norm_threadgroup(
        diagnostic_attention_hc_source_bf16_norm,
        diagnostic_attention_hc_norm_threadgroup,
        diagnostic_attention_hc_norm_full_pairwise,
        diagnostic_attention_hc_source_bf16_stages,
    )
    model_root = model_root.expanduser().resolve(strict=True)
    reference = oracle._regular(reference, "admitted V2 reference")
    if owner_authorization is not None:
        owner_authorization = oracle._regular(owner_authorization, "owner authorization")
    reference_sha = oracle._sha256(reference)
    # Hard authority gate: no capture consumption or output reservation first.
    authority = gate.load_admitted_reference_for_boundary(
        reference, model_root, owner_authorization_path=owner_authorization
    )
    authorization_sha = authority["authorization_binding_sha256"]
    _, preflight_binding = source_capture._preflight(preflight_path, reference_sha)
    source_document, source_binding = native._source_capture(
        source_capture_path,
        reference_sha256=reference_sha,
        authorization_sha256=authorization_sha,
        preflight_sha256=preflight_binding["sha256"],
        require_post_ple_state=not layer0_hc_trace_only,
    )
    native_binding = native._verify_native_receipt(
        native_capture_path,
        model_root=model_root,
        source_capture_sha256=source_binding["capture"]["sha256"],
        preflight_sha256=preflight_binding["sha256"],
        post_ple_sha256=(
            None if layer0_hc_trace_only else source_binding["post_ple_state"]["sha256"]
        ),
        diagnostic_attention_hc_source_bf16_norm=diagnostic_attention_hc_source_bf16_norm,
        # Keep the raw override distinct from the profile's effective width;
        # native receipt validation owns that normalization.
        diagnostic_attention_hc_norm_threadgroup=diagnostic_attention_hc_norm_threadgroup,
        diagnostic_attention_hc_norm_full_pairwise=diagnostic_attention_hc_norm_full_pairwise,
        diagnostic_attention_hc_source_bf16_stages=diagnostic_attention_hc_source_bf16_stages,
        layer0_hc_trace_only=layer0_hc_trace_only,
    )
    out = out.expanduser().resolve()
    if os.path.lexists(out):
        raise ValueError(f"boundary comparison output already exists; refusing overwrite: {out}")
    selected_hawking, resolved_hawking = oracle._executable(
        hawking_binary, "prebuilt hawking boundary metric owner"
    )
    metric_owner = {
        "selected_path": str(selected_hawking),
        "resolved_path": str(resolved_hawking),
        "sha256": oracle._sha256(resolved_hawking),
    }
    native_document = gate._json(native_capture_path.expanduser().resolve(strict=True))
    source_payloads = source_document["source_payloads"]
    source_dir = source_capture_path.expanduser().resolve(strict=True).parent
    native_dir = native._native_artifact_dir(native_capture_path.expanduser().resolve(strict=True))
    trace_results, first_difference = _compare_layer0_trace(
        source_payloads.get("layer0_trace"),
        native_document.get("layer0_trace"),
        source_dir=source_dir,
        native_dir=native_dir,
        rust_binary=selected_hawking,
    )
    seam_results: list[dict[str, Any]] = []
    if first_difference is None:
        source_seams = source_payloads["seams"][:1] if layer0_hc_trace_only else source_payloads["seams"]
        native_seams = native_document["seams"]
        seam_results, first_difference = _compare_seams(
            source_seams,
            native_seams,
            source_dir=source_dir,
            native_dir=native_dir,
            rust_binary=selected_hawking,
        )
    cache_results: list[dict[str, Any]] = []
    if first_difference is None and not layer0_hc_trace_only:
        cache_results, first_difference = _compare_cache(
            source_payloads["layer4_cache"],
            native_document["layer4_cache"],
            source_dir=source_dir,
            native_dir=native_dir,
            rust_binary=selected_hawking,
        )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": DIFFERENCE_STATUS if first_difference is not None else EXACT_STATUS,
        "source_reference": {
            "path": str(reference),
            "sha256": reference_sha,
            "seal_sha256": authority["seal_sha256"],
            "authorization_binding_sha256": authorization_sha,
            "admission_class": authority.get("admission_class"),
            "owner_authorization": authority["owner_authorization"],
        },
        "science_status": "UNEARNED",
        "science_unearned": True,
        "preflight": preflight_binding,
        "source_capture": source_binding["capture"],
        "native_capture": native_binding,
        "diagnostic_attention_hc_source_bf16_norm": diagnostic_attention_hc_source_bf16_norm,
        "diagnostic_attention_hc_source_bf16_stages": diagnostic_attention_hc_source_bf16_stages,
        "diagnostic_attention_hc_norm_threadgroup": diagnostic_threadgroup,
        "diagnostic_attention_hc_norm_full_pairwise": diagnostic_attention_hc_norm_full_pairwise,
        "layer0_hc_trace_only": layer0_hc_trace_only,
        "metric_owner": {
            **metric_owner,
            "command": "hawking gravity boundary-payload-compare",
            "independent_python_crosscheck": True,
        },
        "comparison_policy": {
            "ordering": (
                "layer-0 trace ordinal, then canonical seam ordinal, then layer-4 pre/post cache array order"
            ),
            "stop_condition": "first exact payload-byte or ordered-route difference",
            "continuous_metrics": ["finite", "max_abs", "relative_l2", "cosine"],
            "numerical_acceptance_bound": None,
            "semantic_attribution_allowed": False,
            "cache_hashes_are_not_hidden_states": True,
        },
        "layer0_trace_compared": trace_results,
        "seams_compared": seam_results,
        "cache_arrays_compared": cache_results,
        "first_observed_difference": first_difference,
        "native_ple_implemented": False,
        "comparison_performed": True,
        "promotion_allowed": False,
        "claim_boundary": (
            "Admitted (owner-authorized or local-canonical) ordered payload localization only. Byte equality/"
            "difference and diagnostic metrics do not supply an unearned numerical bound or semantic verdict. The "
            "native path injects source post-PLE state and does not implement native PLE. This is not complete "
            "inference, NR, EBPW/TPS/capability, deployment, Pulsar promotion, or Kimi retirement. "
            "science_status=UNEARNED."
        ),
    }
    result["seal_sha256"] = gate._compact_utf8_sorted_seal(result)
    gate._write_new_runner_sidecar(out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=gate.DEFAULT_MODEL_ROOT)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--owner-authorization", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path)
    parser.add_argument("--native-capture", type=Path)
    parser.add_argument("--hawking-binary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostic-attention-hc-source-bf16-norm", action="store_true")
    parser.add_argument("--diagnostic-attention-hc-source-bf16-stages", action="store_true")
    parser.add_argument("--diagnostic-attention-hc-norm-threadgroup", type=int, choices=(128, 256))
    parser.add_argument("--diagnostic-attention-hc-norm-full-pairwise", action="store_true")
    parser.add_argument("--layer0-hc-trace-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    diagnostic_threadgroup = native._diagnostic_attention_hc_norm_threadgroup(
        args.diagnostic_attention_hc_source_bf16_norm,
        args.diagnostic_attention_hc_norm_threadgroup,
        args.diagnostic_attention_hc_norm_full_pairwise,
        args.diagnostic_attention_hc_source_bf16_stages,
    )
    reference = oracle._regular(args.reference_contract, "V2 reference candidate")
    reference_sha = oracle._sha256(reference)
    _, preflight_binding = source_capture._preflight(args.preflight, reference_sha)
    if args.dry_run:
        print(json.dumps({
            "status": "WITHHELD_PENDING_ADMITTED_SOURCE_AND_NATIVE_CAPTURES",
            "reference_contract": {"path": str(reference), "sha256": reference_sha},
            "preflight": preflight_binding,
            "owner_authorization_supplied": args.owner_authorization is not None,
            "source_capture_supplied": args.source_capture is not None,
            "native_capture_supplied": args.native_capture is not None,
            "hawking_binary_supplied": args.hawking_binary is not None,
            "hawking_binary_exists": (
                args.hawking_binary.expanduser().exists() if args.hawking_binary is not None else False
            ),
            "diagnostic_attention_hc_source_bf16_norm": args.diagnostic_attention_hc_source_bf16_norm,
            "diagnostic_attention_hc_source_bf16_stages": args.diagnostic_attention_hc_source_bf16_stages,
            "diagnostic_attention_hc_norm_threadgroup": diagnostic_threadgroup,
            "diagnostic_attention_hc_norm_full_pairwise": args.diagnostic_attention_hc_norm_full_pairwise,
            "layer0_hc_trace_only": args.layer0_hc_trace_only,
            "output_fresh": not os.path.lexists(args.out.expanduser().resolve()),
            "capture_payloads_read": False,
            "model_or_metal_started": False,
            "claim_boundary": "dry preflight only; no admission, payload comparison, or execution",
        }, indent=2, sort_keys=True))
        return 0
    if not args.execute:
        raise ValueError("source boundary comparison requires explicit --execute")
    if (
        args.source_capture is None
        or args.native_capture is None
        or args.hawking_binary is None
    ):
        raise ValueError(
            "source boundary comparison requires source/native captures and a prebuilt hawking binary "
            "(--owner-authorization optional for local-canonical admission)"
        )
    result = compare(
        model_root=args.root,
        reference=reference,
        owner_authorization=args.owner_authorization,
        preflight_path=args.preflight,
        source_capture_path=args.source_capture,
        native_capture_path=args.native_capture,
        hawking_binary=args.hawking_binary,
        out=args.out,
        diagnostic_attention_hc_source_bf16_norm=args.diagnostic_attention_hc_source_bf16_norm,
        diagnostic_attention_hc_norm_threadgroup=args.diagnostic_attention_hc_norm_threadgroup,
        diagnostic_attention_hc_norm_full_pairwise=args.diagnostic_attention_hc_norm_full_pairwise,
        diagnostic_attention_hc_source_bf16_stages=args.diagnostic_attention_hc_source_bf16_stages,
        layer0_hc_trace_only=args.layer0_hc_trace_only,
    )
    print(json.dumps({
        "status": result["status"],
        "first_observed_difference": result["first_observed_difference"],
        "seal_sha256": result["seal_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
