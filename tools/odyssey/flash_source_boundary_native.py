#!/usr/bin/env python3
"""Run the native half of Flash's owner-authorized token-0 boundary control.

This wrapper admits the original V2 teacher, verifies the bounded source
capture, and only then launches a Rust/Metal diagnostic.  The default captures
layers 0 through 4 with one source post-PLE injection because native PLE does
not yet exist; that diagnostic dependency is forbidden from ordinary
inference.  ``--layer0-hc-trace-only`` is a separate, source-lineage-bound
causal localizer that never loads that PLE payload, layer-0 MLP weights, or
layers 1 through 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from tools.odyssey import flash_source_boundary_preflight as preflight  # noqa: E402


SCHEMA = "hawking.flash.native_source_boundary_runner.v1"
STATUS = "CAPTURED_NATIVE_LAYERS_0_THROUGH_4_TOKEN_0_WITH_OWNER_BOUND_SOURCE_PLE_INJECTION"
NATIVE_SCHEMA = "hawking.flash.native_source_boundary_capture.v1"
NATIVE_STATUS = "CAPTURED_NATIVE_LAYERS_0_THROUGH_4_TOKEN_0_WITH_SOURCE_PLE_INJECTION"
LAYER0_TRACE_STATUS = "CAPTURED_NATIVE_LAYER0_ATTENTION_HC_TRACE_ONLY_WITH_OWNER_BOUND_SOURCE_LINEAGE"
NATIVE_LAYER0_TRACE_SCHEMA = "hawking.flash.native_layer0_attention_hc_trace_capture.v1"
NATIVE_LAYER0_TRACE_STATUS = "CAPTURED_NATIVE_LAYER0_ATTENTION_HC_TRACE_ONLY_TOKEN_0"


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _native_artifact_dir(out: Path) -> Path:
    return gate._native_session_artifact_dir(out)


def _diagnostic_attention_hc_norm_threadgroup(
    diagnostic_attention_hc_source_bf16_norm: bool,
    requested_threadgroup: int | None,
    diagnostic_attention_hc_norm_full_pairwise: bool = False,
    diagnostic_attention_hc_source_bf16_stages: bool = False,
) -> int | None:
    source_bf16_diagnostic = (
        diagnostic_attention_hc_source_bf16_norm or diagnostic_attention_hc_source_bf16_stages
    )
    if requested_threadgroup is not None and requested_threadgroup not in (128, 256):
        raise ValueError("attention-HC diagnostic threadgroup must be 128 or 256")
    if diagnostic_attention_hc_norm_full_pairwise and not source_bf16_diagnostic:
        raise ValueError(
            "attention-HC full-pairwise diagnostic requires --diagnostic-attention-hc-source-bf16-norm "
            "or --diagnostic-attention-hc-source-bf16-stages"
        )
    if diagnostic_attention_hc_norm_full_pairwise and requested_threadgroup is not None:
        raise ValueError(
            "attention-HC full-pairwise diagnostic may not specify a threadgroup override"
        )
    if requested_threadgroup is not None and not source_bf16_diagnostic:
        raise ValueError(
            "attention-HC diagnostic threadgroup requires --diagnostic-attention-hc-source-bf16-norm "
            "or --diagnostic-attention-hc-source-bf16-stages"
        )
    if diagnostic_attention_hc_norm_full_pairwise:
        return 256
    return requested_threadgroup if source_bf16_diagnostic else None


def _fresh_outputs(out: Path) -> Path:
    out = out.expanduser().resolve()
    sidecar = out.with_name(f"{out.stem}.runner.json")
    targets = (out, _native_artifact_dir(out), sidecar)
    existing = [str(path) for path in targets if _path_exists(path)]
    if existing:
        raise ValueError("native boundary outputs already exist; refusing reuse: " + ", ".join(existing))
    return sidecar


def _authorization_binding_sha256(source_reference: dict[str, Any]) -> str | None:
    """Read a local-canonical binding or the legacy signed-file binding."""
    bound = source_reference.get("authorization_binding_sha256")
    if isinstance(bound, str) and len(bound) == 64:
        return bound
    owner = source_reference.get("owner_authorization") or {}
    if isinstance(owner, dict):
        for key in ("authorization_binding_sha256", "sha256"):
            value = owner.get(key)
            if isinstance(value, str) and len(value) == 64:
                return value
    return None


def _verify_layer0_trace(
    records: object,
    *,
    out_dir: Path,
    label_prefix: str,
    source_runtime_dtype_required: bool,
) -> list[dict[str, Any]]:
    declarations = preflight._required_layer0_trace()
    if not isinstance(records, list) or len(records) != len(declarations):
        raise ValueError(f"{label_prefix} omitted a predeclared layer-0 trace payload")
    verified: list[dict[str, Any]] = []
    for declaration, record in zip(declarations, records, strict=True):
        if not isinstance(record, dict) or any(
            record.get(field) != declaration[field] for field in ("ordinal", "layer", "stage", "shape")
        ):
            raise ValueError(f"{label_prefix} layer-0 trace order, name, layer, or shape differs")
        if source_runtime_dtype_required and (
            not isinstance(record.get("original_dtype"), str) or not record["original_dtype"]
        ):
            raise ValueError(f"{label_prefix} layer-0 trace omitted the source runtime dtype")
        payload = source_capture._payload(
            record.get("payload"),
            out_dir=out_dir,
            label=f"{label_prefix} layer-0 trace {declaration['ordinal']}",
            dtype="F32_LE",
        )
        if payload["elements"] != preflight._shape_elements(declaration["shape"]):
            raise ValueError(f"{label_prefix} layer-0 trace payload extent differs from its declared shape")
        verified.append({**record, "payload": payload})
    return verified


def _source_capture(
    path: Path,
    *,
    reference_sha256: str,
    authorization_sha256: str,
    preflight_sha256: str,
    require_post_ple_state: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = oracle._regular(path, "owner-authorized source boundary capture")
    document = gate._json(path)
    seal = gate._verify_compact_seal(document, "owner-authorized source boundary capture")
    source_reference = document.get("source_reference")
    payloads = document.get("source_payloads")
    if (
        document.get("schema") != source_capture.SCHEMA
        or document.get("status") != source_capture.STATUS
        or not isinstance(source_reference, dict)
        or source_reference.get("sha256") != reference_sha256
        or _authorization_binding_sha256(source_reference) != authorization_sha256
        or not isinstance(payloads, dict)
        or payloads.get("layer0_trace_contract_version") != 5
        or document.get("full_body_executed") is not False
        or document.get("native_body_executed") is not False
        or document.get("parity_compared") is not False
    ):
        raise ValueError("source boundary capture identity, authorization, or claim scope differs")
    preflight_binding = document.get("preflight")
    if not isinstance(preflight_binding, dict) or preflight_binding.get("sha256") != preflight_sha256:
        raise ValueError("source boundary capture preflight binding differs")
    seams = payloads.get("seams")
    if not isinstance(seams, list) or len(seams) != 14:
        raise ValueError("source boundary capture omitted a predeclared seam")
    binding: dict[str, Any] = {
        "capture": {**oracle._artifact_binding(path), "seal_sha256": seal},
    }
    if require_post_ple_state:
        post_ple = seams[3]
        if (
            not isinstance(post_ple, dict)
            or post_ple.get("ordinal") != 3
            or post_ple.get("layer") != 1
            or post_ple.get("stage") != "ple_additive_output"
            or post_ple.get("shape") != [1, 1, 10240]
        ):
            raise ValueError("source boundary capture omitted the exact post-PLE token-0 seam")
        payload = source_capture._payload(
            post_ple.get("payload"),
            out_dir=path.parent,
            label="owner-bound source post-PLE token-0 state",
            dtype="F32_LE",
        )
        if payload["elements"] != 10_240:
            raise ValueError("source post-PLE token-0 state width differs from hc_count=4")
        binding["post_ple_state"] = payload
    binding["layer0_trace"] = _verify_layer0_trace(
        payloads.get("layer0_trace"),
        out_dir=path.parent,
        label_prefix="source boundary capture",
        source_runtime_dtype_required=True,
    )
    return document, binding


def _verify_native_receipt(
    path: Path,
    *,
    model_root: Path,
    source_capture_sha256: str,
    preflight_sha256: str,
    post_ple_sha256: str | None,
    diagnostic_attention_hc_source_bf16_norm: bool,
    diagnostic_attention_hc_norm_threadgroup: int | None = None,
    diagnostic_attention_hc_norm_full_pairwise: bool = False,
    diagnostic_attention_hc_source_bf16_stages: bool = False,
    layer0_hc_trace_only: bool = False,
) -> dict[str, Any]:
    path = oracle._regular(path, "native source boundary receipt")
    document = gate._json(path)
    seal = gate._verify_compact_seal(document, "native source boundary receipt")
    ple = document.get("ple")
    source_capture_lineage = document.get("source_capture_lineage")
    execution = document.get("execution")
    precision = execution.get("attention_hc_norm_precision") if isinstance(execution, dict) else None
    expected_threadgroup = _diagnostic_attention_hc_norm_threadgroup(
        diagnostic_attention_hc_source_bf16_norm,
        diagnostic_attention_hc_norm_threadgroup,
        diagnostic_attention_hc_norm_full_pairwise,
        diagnostic_attention_hc_source_bf16_stages,
    )
    expected_precision_mode = (
        (
            (
                "SOURCE_BF16_ROUNDTRIP_AFTER_EACH_ATTENTION_HC_STAGE_FULL_PAIRWISE_F32_REDUCTION"
                if diagnostic_attention_hc_norm_full_pairwise
                else "SOURCE_BF16_ROUNDTRIP_AFTER_EACH_ATTENTION_HC_STAGE"
            )
            if diagnostic_attention_hc_source_bf16_stages
            else (
                "SOURCE_BF16_ROUNDTRIP_AFTER_ATTENTION_HC_RMSNORM_FULL_PAIRWISE_F32_REDUCTION"
                if diagnostic_attention_hc_norm_full_pairwise
                else "SOURCE_BF16_ROUNDTRIP_AFTER_ATTENTION_HC_RMSNORM"
            )
        )
        if diagnostic_attention_hc_source_bf16_norm or diagnostic_attention_hc_source_bf16_stages
        else "NATIVE_F32_NO_DIAGNOSTIC_CAST"
    )
    if layer0_hc_trace_only:
        if (
            document.get("schema") != NATIVE_LAYER0_TRACE_SCHEMA
            or document.get("status") != NATIVE_LAYER0_TRACE_STATUS
            or document.get("capture_scope") != "LAYER0_ATTENTION_HC_TRACE_ONLY"
            or document.get("root") != str(model_root)
            or document.get("token_index") != 0
            or document.get("token_id") != gate.PROMPT_IDS[0]
            or document.get("layers_executed") != [0]
            or document.get("layer0_trace_contract_version") != 5
            or not isinstance(source_capture_lineage, dict)
            or source_capture_lineage.get("source_capture_sha256") != source_capture_sha256
            or source_capture_lineage.get("preflight_sha256") != preflight_sha256
            or source_capture_lineage.get("source_ple_state_consumed") is not False
            or ple is not None
            or document.get("layer4_cache") is not None
            or not isinstance(execution, dict)
            or not isinstance(precision, dict)
            or precision.get("mode") != expected_precision_mode
            or execution.get("mlp_executed") is not False
            or execution.get("full_body_executed") is not False
            or execution.get("lm_head_executed") is not False
            or document.get("promotion_allowed") is not False
        ):
            raise ValueError("native layer-0 trace receipt identity, source lineage, or scope differs")
    else:
        if (
            document.get("schema") != NATIVE_SCHEMA
            or document.get("status") != NATIVE_STATUS
            or document.get("root") != str(model_root)
            or document.get("token_index") != 0
            or document.get("token_id") != gate.PROMPT_IDS[0]
            or document.get("layers_executed") != [0, 1, 2, 3, 4]
            or document.get("layer0_trace_contract_version") != 5
            or not isinstance(ple, dict)
            or ple.get("native_implementation_present") is not False
            or ple.get("injected_at_layer") != 1
            or ple.get("source_capture_sha256") != source_capture_sha256
            or ple.get("preflight_sha256") != preflight_sha256
            or ((ple.get("source_post_ple_state") or {}).get("sha256")) != post_ple_sha256
            or not isinstance(execution, dict)
            or not isinstance(precision, dict)
            or precision.get("mode") != expected_precision_mode
            or execution.get("full_body_executed") is not False
            or execution.get("lm_head_executed") is not False
            or document.get("promotion_allowed") is not False
        ):
            raise ValueError("native source boundary receipt identity, PLE injection, or scope differs")
    actual_threadgroup = precision.get("reduction_threadgroup_size")
    if diagnostic_attention_hc_source_bf16_norm or diagnostic_attention_hc_source_bf16_stages:
        if expected_threadgroup is not None and actual_threadgroup != expected_threadgroup:
            raise ValueError("native source boundary precision threadgroup differs from the requested diagnostic")
        if expected_threadgroup is None and actual_threadgroup not in (None, 256):
            raise ValueError("native source boundary precision threadgroup is not the default diagnostic")
    if (
        diagnostic_attention_hc_norm_full_pairwise
        and precision.get("reduction_topology") != "CONTIGUOUS_FULL_PAIRWISE_F32"
    ):
        raise ValueError("native source boundary precision reduction topology differs from the diagnostic")
    artifact_dir = _native_artifact_dir(path)
    seams = document.get("seams")
    declarations = (
        preflight._required_seams()[:1]
        if layer0_hc_trace_only
        else preflight._required_seams()
    )
    if not isinstance(seams, list) or len(seams) != len(declarations):
        raise ValueError("native source boundary receipt omitted a predeclared seam")
    verified_seams = []
    for declaration, record in zip(declarations, seams, strict=True):
        if not isinstance(record, dict) or any(
            record.get(field) != declaration[field] for field in ("ordinal", "layer", "stage", "shape")
        ):
            raise ValueError("native source boundary seam order, name, layer, or shape differs")
        if not layer0_hc_trace_only and declaration["ordinal"] == 11:
            ids = record.get("expert_ids")
            weights = record.get("weights")
            if not isinstance(ids, list) or len(ids) != 10 or not isinstance(weights, list) or len(weights) != 10:
                raise ValueError("native layer-4 route is malformed")
            source_capture._payload(
                record.get("ids_payload"), out_dir=artifact_dir,
                label="native layer-4 route ids", dtype="I32_LE",
            )
            source_capture._payload(
                record.get("weights_payload"), out_dir=artifact_dir,
                label="native layer-4 route weights", dtype="F32_LE",
            )
        else:
            source_capture._payload(
                record.get("payload"), out_dir=artifact_dir,
                label=f"native boundary seam {declaration['ordinal']}", dtype="F32_LE",
            )
        verified_seams.append(record)
    verified_trace = _verify_layer0_trace(
        document.get("layer0_trace"),
        out_dir=artifact_dir,
        label_prefix="native source boundary receipt",
        source_runtime_dtype_required=False,
    )
    if not layer0_hc_trace_only:
        caches = document.get("layer4_cache")
        if not isinstance(caches, dict) or set(caches) != {"pre_attention", "post_attention"}:
            raise ValueError("native source boundary receipt omitted layer-4 cache boundaries")
        cache_declarations = preflight._required_layer4_cache()
        for boundary in ("pre_attention", "post_attention"):
            records = caches[boundary]
            declarations = cache_declarations[boundary]
            if not isinstance(records, list) or len(records) != len(declarations):
                raise ValueError(f"native layer-4 {boundary} cache must contain conv and recurrent state")
            native_boundary = boundary.replace("_", "-")
            for index, (record, declaration) in enumerate(zip(records, declarations, strict=True)):
                if (
                    not isinstance(record, dict)
                    or record.get("name") != declaration["state_role"].replace("_", "-")
                    or record.get("boundary") != native_boundary
                ):
                    raise ValueError("native layer-4 cache record is malformed")
                payload = source_capture._payload(
                    record.get("payload"), out_dir=artifact_dir,
                    label=f"native layer-4 {boundary} cache {index}", dtype="F32_LE",
                )
                if payload["elements"] != preflight._shape_elements(declaration["shape"]):
                    raise ValueError(f"native layer-4 {boundary} cache payload extent differs from the source contract")
    return {
        "path": str(path),
        "sha256": oracle._sha256(path),
        "seal_sha256": seal,
        "schema": document["schema"],
        "status": document["status"],
        "capture_scope": document.get("capture_scope"),
        "seam_count": len(verified_seams),
        "layer0_trace_count": len(verified_trace),
        "attention_hc_norm_precision": execution["attention_hc_norm_precision"],
    }


def _command(
    binary: Path,
    *,
    model_root: Path,
    out: Path,
    source_capture_binding: dict[str, Any],
    preflight_sha256: str,
    diagnostic_attention_hc_source_bf16_norm: bool = False,
    diagnostic_attention_hc_norm_threadgroup: int | None = None,
    diagnostic_attention_hc_norm_full_pairwise: bool = False,
    diagnostic_attention_hc_source_bf16_stages: bool = False,
    layer0_hc_trace_only: bool = False,
) -> list[str]:
    diagnostic_threadgroup = _diagnostic_attention_hc_norm_threadgroup(
        diagnostic_attention_hc_source_bf16_norm,
        diagnostic_attention_hc_norm_threadgroup,
        diagnostic_attention_hc_norm_full_pairwise,
        diagnostic_attention_hc_source_bf16_stages,
    )
    capture_binding = source_capture_binding["capture"]
    command = [
        str(binary),
        "--root", str(model_root),
        "--token-ids", str(gate.PROMPT_IDS[0]),
        "--out", str(out),
        (
            "--capture-layer0-hc-trace-only"
            if layer0_hc_trace_only
            else "--capture-source-boundary"
        ),
    ]
    if not layer0_hc_trace_only:
        post_ple = source_capture_binding.get("post_ple_state")
        if not isinstance(post_ple, dict):
            raise ValueError("full native source boundary capture requires an admitted source post-PLE state")
        command.extend((
            "--source-post-ple-state", post_ple["path"],
            "--source-post-ple-state-sha256", post_ple["sha256"],
        ))
    command.extend((
        "--source-boundary-capture-sha256", capture_binding["sha256"],
        "--source-boundary-preflight-sha256", preflight_sha256,
        "--wrapper-admitted-source-control",
        "--supervising-launcher-pid", str(os.getpid()),
    ))
    if diagnostic_attention_hc_source_bf16_norm:
        command.append("--diagnostic-attention-hc-source-bf16-norm")
    if diagnostic_attention_hc_source_bf16_stages:
        command.append("--diagnostic-attention-hc-source-bf16-stages")
    if diagnostic_attention_hc_norm_full_pairwise:
        command.append("--diagnostic-attention-hc-norm-full-pairwise")
    if diagnostic_threadgroup is not None and not diagnostic_attention_hc_norm_full_pairwise:
        command.extend(("--diagnostic-attention-hc-norm-threadgroup", str(diagnostic_threadgroup)))
    return command


def run(
    *,
    model_root: Path,
    reference: Path,
    owner_authorization: Path | None,
    preflight_path: Path,
    source_capture_path: Path,
    native_binary: Path,
    out: Path,
    diagnostic_attention_hc_source_bf16_norm: bool = False,
    diagnostic_attention_hc_norm_threadgroup: int | None = None,
    diagnostic_attention_hc_norm_full_pairwise: bool = False,
    diagnostic_attention_hc_source_bf16_stages: bool = False,
    layer0_hc_trace_only: bool = False,
) -> dict[str, Any]:
    diagnostic_threadgroup = _diagnostic_attention_hc_norm_threadgroup(
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
    # Hard authority gate: no output reservation, executable inspection, or
    # native/Metal action may precede this call.
    authority = gate.load_admitted_reference_for_boundary(
        reference, model_root, owner_authorization_path=owner_authorization
    )
    authorization_sha = authority["authorization_binding_sha256"]
    _, preflight_binding = source_capture._preflight(preflight_path, reference_sha)
    source_document, source_binding = _source_capture(
        source_capture_path,
        reference_sha256=reference_sha,
        authorization_sha256=authorization_sha,
        preflight_sha256=preflight_binding["sha256"],
        require_post_ple_state=not layer0_hc_trace_only,
    )
    sidecar = _fresh_outputs(out)
    lane = gate._protected_native_lane()
    if not lane["clean"]:
        raise ValueError(f"protected native/provider lane is not clean: {lane['matches']}")
    binary = oracle._regular(native_binary, "prebuilt native boundary executable")
    if not os.access(binary, os.X_OK):
        raise ValueError("prebuilt native boundary executable is not executable")
    build_before = gate._preexecution_build_binding(binary)
    command = _command(
        binary,
        model_root=model_root,
        out=out.expanduser().resolve(),
        source_capture_binding=source_binding,
        preflight_sha256=preflight_binding["sha256"],
        diagnostic_attention_hc_source_bf16_norm=diagnostic_attention_hc_source_bf16_norm,
        # Preserve the caller's raw choice here.  The full-pairwise profile has
        # an effective 256-lane implementation but deliberately forbids a
        # user-supplied threadgroup override; feeding the resolved value back
        # through validation would falsely turn it into such an override.
        diagnostic_attention_hc_norm_threadgroup=diagnostic_attention_hc_norm_threadgroup,
        diagnostic_attention_hc_norm_full_pairwise=diagnostic_attention_hc_norm_full_pairwise,
        diagnostic_attention_hc_source_bf16_stages=diagnostic_attention_hc_source_bf16_stages,
        layer0_hc_trace_only=layer0_hc_trace_only,
    )
    subprocess.run(command, cwd=ROOT, check=True)
    build_after = gate._preexecution_build_binding(binary)
    if not gate._same_executable_closure(build_before, build_after):
        raise ValueError("native boundary executable/source closure changed during execution")
    native = _verify_native_receipt(
        out.expanduser().resolve(),
        model_root=model_root,
        source_capture_sha256=source_binding["capture"]["sha256"],
        preflight_sha256=preflight_binding["sha256"],
        post_ple_sha256=(
            None if layer0_hc_trace_only else source_binding["post_ple_state"]["sha256"]
        ),
        diagnostic_attention_hc_source_bf16_norm=diagnostic_attention_hc_source_bf16_norm,
        diagnostic_attention_hc_norm_threadgroup=diagnostic_attention_hc_norm_threadgroup,
        diagnostic_attention_hc_norm_full_pairwise=diagnostic_attention_hc_norm_full_pairwise,
        diagnostic_attention_hc_source_bf16_stages=diagnostic_attention_hc_source_bf16_stages,
        layer0_hc_trace_only=layer0_hc_trace_only,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": LAYER0_TRACE_STATUS if layer0_hc_trace_only else STATUS,
        "source_reference": {
            "path": str(reference), "sha256": reference_sha,
            "seal_sha256": authority["seal_sha256"],
            "authorization_binding_sha256": authorization_sha,
            "admission_class": authority.get("admission_class"),
            "owner_authorization": authority["owner_authorization"],
        },
        "science_status": "UNEARNED",
        "science_unearned": True,
        "preflight": preflight_binding,
        "source_capture": source_binding["capture"],
        "native_capture": native,
        "build": build_after,
        "command": command,
        "diagnostic_attention_hc_source_bf16_norm": diagnostic_attention_hc_source_bf16_norm,
        "diagnostic_attention_hc_source_bf16_stages": diagnostic_attention_hc_source_bf16_stages,
        "diagnostic_attention_hc_norm_threadgroup": diagnostic_threadgroup,
        "diagnostic_attention_hc_norm_full_pairwise": diagnostic_attention_hc_norm_full_pairwise,
        "comparison_performed": False,
        "promotion_allowed": False,
    }
    if layer0_hc_trace_only:
        result.update({
            "capture_scope": "LAYER0_ATTENTION_HC_TRACE_ONLY",
            "source_layer0_trace": source_binding["layer0_trace"],
            "source_ple_state": {
                "consumed": False,
                "ordinary_inference_dependency_allowed": False,
            },
            "claim_boundary": (
                "Bounded admitted (owner-authorized or local-canonical) native layer-0 attention-HyperConnection "
                "causal capture only. It does not load or inject source PLE payload state, load layer-0 MLP weights "
                "or execute layer-0 MLP operators, execute layers 1 through 4 or the complete body, establish source parity by itself, "
                "qualify an NR, measure EBPW/TPS/capability, "
                "deploy, promote Pulsar, or retire Kimi. science_status=UNEARNED."
            ),
        })
    else:
        result.update({
            "source_post_ple_state": source_binding["post_ple_state"],
            "teacher_injection": {
                "layer": 1,
                "native_ple_implemented": False,
                "ordinary_inference_dependency_allowed": False,
                "purpose": "diagnostic localization beyond the missing native PLE seam",
            },
            "claim_boundary": (
                "Bounded admitted (owner-authorized or local-canonical) native payload capture only. It injects one "
                "source PLE state and cannot establish native PLE, complete source parity, complete inference, "
                "EBPW/TPS/capability, deployment, Pulsar promotion, or Kimi retirement. science_status=UNEARNED."
            ),
        })
    result["seal_sha256"] = gate._compact_utf8_sorted_seal(result)
    gate._write_new_runner_sidecar(sidecar, result)
    del source_document
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=gate.DEFAULT_MODEL_ROOT)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--owner-authorization", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path)
    parser.add_argument("--native-binary", type=Path, default=gate._native_binary_path())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostic-attention-hc-source-bf16-norm", action="store_true")
    parser.add_argument("--diagnostic-attention-hc-source-bf16-stages", action="store_true")
    parser.add_argument("--diagnostic-attention-hc-norm-threadgroup", type=int, choices=(128, 256))
    parser.add_argument("--diagnostic-attention-hc-norm-full-pairwise", action="store_true")
    parser.add_argument("--layer0-hc-trace-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    diagnostic_threadgroup = _diagnostic_attention_hc_norm_threadgroup(
        args.diagnostic_attention_hc_source_bf16_norm,
        args.diagnostic_attention_hc_norm_threadgroup,
        args.diagnostic_attention_hc_norm_full_pairwise,
        args.diagnostic_attention_hc_source_bf16_stages,
    )
    reference = oracle._regular(args.reference_contract, "V2 reference candidate")
    reference_sha = oracle._sha256(reference)
    _, preflight_binding = source_capture._preflight(args.preflight, reference_sha)
    lane = gate._protected_native_lane()
    if args.dry_run:
        native_binary = args.native_binary.expanduser().resolve()
        native_binary_exists = native_binary.is_file()
        print(json.dumps({
            "status": (
                "WITHHELD_PENDING_ADMITTED_SOURCE_CAPTURE"
                if native_binary_exists
                else "WITHHELD_PENDING_ADMITTED_SOURCE_CAPTURE_AND_NATIVE_BUILD"
            ),
            "reference_contract": {"path": str(reference), "sha256": reference_sha},
            "preflight": preflight_binding,
            "owner_authorization_supplied": args.owner_authorization is not None,
            "source_capture_supplied": args.source_capture is not None,
            "native_binary": str(native_binary),
            "native_binary_exists": native_binary_exists,
            "diagnostic_attention_hc_source_bf16_norm": args.diagnostic_attention_hc_source_bf16_norm,
            "diagnostic_attention_hc_source_bf16_stages": args.diagnostic_attention_hc_source_bf16_stages,
            "diagnostic_attention_hc_norm_threadgroup": diagnostic_threadgroup,
            "diagnostic_attention_hc_norm_full_pairwise": args.diagnostic_attention_hc_norm_full_pairwise,
            "capture_scope": (
                "LAYER0_ATTENTION_HC_TRACE_ONLY"
                if args.layer0_hc_trace_only
                else "FULL_SOURCE_PLE_BOUNDARY"
            ),
            "protected_native_lane": lane,
            "outputs_fresh": not any((
                _path_exists(args.out.expanduser().resolve()),
                _path_exists(_native_artifact_dir(args.out.expanduser().resolve())),
                _path_exists(args.out.expanduser().resolve().with_name(f"{args.out.stem}.runner.json")),
            )),
            "model_or_metal_started": False,
            "claim_boundary": "dry preflight only; no admission, source capture consumption, build, or execution",
        }, indent=2, sort_keys=True))
        return 0
    if not args.execute:
        raise ValueError("native boundary capture requires explicit --execute")
    if args.source_capture is None:
        raise ValueError("native source boundary capture requires --source-capture")
    result = run(
        model_root=args.root,
        reference=reference,
        owner_authorization=args.owner_authorization,
        preflight_path=args.preflight,
        source_capture_path=args.source_capture,
        native_binary=args.native_binary,
        out=args.out,
        diagnostic_attention_hc_source_bf16_norm=args.diagnostic_attention_hc_source_bf16_norm,
        diagnostic_attention_hc_source_bf16_stages=args.diagnostic_attention_hc_source_bf16_stages,
        diagnostic_attention_hc_norm_threadgroup=args.diagnostic_attention_hc_norm_threadgroup,
        diagnostic_attention_hc_norm_full_pairwise=args.diagnostic_attention_hc_norm_full_pairwise,
        layer0_hc_trace_only=args.layer0_hc_trace_only,
    )
    print(json.dumps({"status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
