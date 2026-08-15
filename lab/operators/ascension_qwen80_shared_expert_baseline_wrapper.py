#!/usr/bin/env python3
"""Seal an immutable Qwen80 shared-expert CPU baseline wrapper.

The direct-packed shared-expert CPU component intentionally writes an unsigned
inner receipt: it is a component oracle, not a promotion authority.  This
small tool attests that exact receipt, its durable capture files, and the
current complete-artifact/admission binding in a new sealed record.  It never
opens tensor payloads, creates a Metal context, or mutates the inner receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.receipts import SealIntegrityError, seal, verify


WRAPPER_SCHEMA = "hawking.ascension.qwen80_shared_expert_cpu_baseline_wrapper.v1"
WRAPPER_STATUS = "SEALED_CURRENT_ADMITTED_QWEN80_SHARED_EXPERT_CPU_ORACLE_BASELINE"
INNER_SCHEMA = "hawking.ascension.qwen80_direct_packed_shared_expert_wave.v1"
INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_SHARED_EXPERT_CPU_ORACLE_READY_METAL_LEASE_REQUIRED"
)
MANIFEST_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
ADMISSION_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
MODEL_ID = "Qwen3-Coder-Next-80B"
MODEL_KEY = "qwen80"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
SOURCE_CONFIG_SHA256 = "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8"
SOURCE_BODY_AUDIT_SEAL = "c572b2270b623b8677c374b43c89ddd729de135c25721488bb874b184ff8c3d4"
SOURCE_REVALIDATION_SEAL = "541b16fca1d4805ecba356face97b4e8de1accdeb21e98ee0c13b70ab0746c45"
MANIFEST_SEAL = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b"
ADMISSION_RECEIPT_SEAL = "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628"

EXPECTED_TENSORS = {
    "post_attention_norm": (
        "model.layers.0.post_attention_layernorm.weight",
        "a00ba60c88bd0d5dcf77e4c1fad05d83ddb6feec844ee3bbc65480fffd5a1fa7",
        [2048],
    ),
    "shared_gate_proj": (
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "92172dc4463a3a0610460ecf768427f6c9c8da04b43a73e904ca1fa36bc79aa6",
        [512, 2048],
    ),
    "shared_up_proj": (
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        "9d76293fa8abf4ccc2611d77386060671107e83dfd4458b5fddd5e345f24b4c4",
        [512, 2048],
    ),
    "shared_down_proj": (
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "acf137a00b364f9c490e1282f18632465f05323b89903a5617162437b1ff500b",
        [2048, 512],
    ),
    "shared_expert_gate": (
        "model.layers.0.mlp.shared_expert_gate.weight",
        "a40ff8a3f4e4b7e990a4672470cbd028b0c96b1cb15acd40aa3b8b2e2215096c",
        [1, 2048],
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_path(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} cannot be resolved: {path}: {exc}") from exc


def _regular_bytes(path: Path, label: str) -> bytes:
    clean = _canonical_path(path, label)
    try:
        info = os.lstat(clean)
    except OSError as exc:
        raise ValueError(f"{label} stat failed: {clean}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {clean}")
    return clean.read_bytes()


def _file_evidence(path: Path, label: str) -> dict[str, Any]:
    clean = _canonical_path(path, label)
    raw = _regular_bytes(clean, label)
    return {
        "path": str(clean),
        "present": True,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _json_object(path: Path, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    clean = _canonical_path(path, label)
    raw = _regular_bytes(clean, label)
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return clean, raw, document


def _require_string(document: Mapping[str, Any], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} missing non-empty string {key!r}")
    return value


def _require_object(document: Mapping[str, Any], key: str, label: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label} missing object {key!r}")
    return value


def _expect(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, observed {value!r}")


def _verify_sealed(document: Mapping[str, Any], label: str) -> None:
    try:
        verify(document, label=label)
    except SealIntegrityError as exc:
        raise ValueError(str(exc)) from exc


def _validate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    clean, raw, document = _json_object(path, "manifest")
    _verify_sealed(document, "manifest")
    _expect(document.get("schema"), MANIFEST_SCHEMA, "manifest schema")
    _expect(document.get("seal_sha256"), MANIFEST_SEAL, "manifest seal")
    _expect(
        document.get("status"),
        "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
        "manifest status",
    )
    model = _require_object(document, "source", "manifest")
    _expect(model.get("repository"), SOURCE_REPOSITORY, "manifest source repository")
    if len(raw) < 64 * 1024 * 1024:
        raise ValueError("manifest unexpectedly below Qwen80 complete-manifest envelope")
    return document, {
        "path": str(clean),
        "present": True,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _validate_admission(path: Path, manifest_evidence: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    clean, raw, document = _json_object(path, "admission current pointer")
    _verify_sealed(document, "admission current pointer")
    _expect(document.get("schema"), ADMISSION_SCHEMA, "admission current schema")
    _expect(document.get("status"), "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED", "admission current status")
    model = _require_object(document, "model", "admission current pointer")
    _expect(model.get("id"), MODEL_ID, "admission model id")
    _expect(model.get("key"), MODEL_KEY, "admission model key")
    _expect(model.get("repository"), SOURCE_REPOSITORY, "admission source repository")
    _expect(model.get("revision"), SOURCE_REVISION, "admission source revision")
    complete = _require_object(document, "complete_manifest", "admission current pointer")
    _expect(complete.get("path"), manifest_evidence["path"], "admission manifest path")
    _expect(complete.get("document_sha256"), manifest_evidence["sha256"], "admission manifest digest")
    _expect(complete.get("seal_sha256"), MANIFEST_SEAL, "admission manifest seal")
    receipt = _require_object(document, "admission_receipt", "admission current pointer")
    _expect(receipt.get("seal_sha256"), ADMISSION_RECEIPT_SEAL, "admission receipt seal")
    return document, {
        "path": str(clean),
        "present": True,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _validate_inner(
    path: Path,
    manifest_evidence: Mapping[str, Any],
    admission_evidence: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    clean, raw, document = _json_object(path, "shared-expert CPU inner receipt")
    _expect(document.get("schema"), INNER_SCHEMA, "shared-expert CPU inner schema")
    _expect(document.get("status"), INNER_STATUS, "shared-expert CPU inner status")
    _expect(document.get("mode"), "cpu-oracle", "shared-expert CPU inner mode")
    _expect(
        document.get("metal_device_or_dispatch_performed"),
        False,
        "shared-expert CPU inner Metal execution",
    )
    _expect(document.get("shared_expert_only"), True, "shared-expert CPU inner component scope")
    _expect(document.get("raw_bf16_or_safetensors_opened"), False, "shared-expert CPU inner raw source access")
    _expect(
        document.get("complete_artifact_scan_performed"),
        False,
        "shared-expert CPU inner full artifact scan",
    )
    binding = _require_object(document, "artifact_binding", "shared-expert CPU inner receipt")
    _expect(binding.get("manifest_path"), manifest_evidence["path"], "inner manifest path")
    _expect(binding.get("manifest_document_sha256"), manifest_evidence["sha256"], "inner manifest digest")
    _expect(binding.get("manifest_seal_sha256"), MANIFEST_SEAL, "inner manifest seal")
    _expect(binding.get("admission_current_path"), admission_evidence["path"], "inner admission current path")
    _expect(
        binding.get("admission_receipt_seal_sha256"),
        admission["admission_receipt"]["seal_sha256"],
        "inner admission receipt seal",
    )
    _expect(binding.get("source_repository"), SOURCE_REPOSITORY, "inner source repository")
    _expect(binding.get("source_revision"), SOURCE_REVISION, "inner source revision")
    _expect(binding.get("source_config_sha256"), SOURCE_CONFIG_SHA256, "inner source config")
    _expect(binding.get("source_body_audit_seal_sha256"), SOURCE_BODY_AUDIT_SEAL, "inner source audit")
    _expect(binding.get("source_revalidation_seal_sha256"), SOURCE_REVALIDATION_SEAL, "inner source revalidation")
    _expect(binding.get("layer"), 0, "inner layer")
    _expect(binding.get("layer_kind"), "linear_attention", "inner layer kind")
    _expect(binding.get("hidden"), 2048, "inner hidden geometry")
    _expect(binding.get("shared_expert_intermediate"), 512, "inner shared intermediate geometry")
    _expect(binding.get("experts"), 512, "inner expert count")
    _expect(binding.get("experts_per_token"), 10, "inner top-k geometry")
    for field, (name, digest, shape) in EXPECTED_TENSORS.items():
        tensor = _require_object(binding, field, f"inner tensor {field}")
        _expect(tensor.get("name"), name, f"inner tensor {field} name")
        _expect(tensor.get("artifact_sha256"), digest, f"inner tensor {field} digest")
        header = _require_object(tensor, "header", f"inner tensor {field}")
        _expect(header.get("shape"), shape, f"inner tensor {field} shape")
        _expect(header.get("magic"), "HQ30G1B1", f"inner tensor {field} format")
        _expect(header.get("group_size"), 128, f"inner tensor {field} group size")
    capture = _require_object(document, "durable_capture", "shared-expert CPU inner receipt")
    _expect(capture.get("receipt_written_last_is_completion_marker"), True, "inner receipt-last contract")
    capture_dir = _canonical_path(Path(_require_string(capture, "directory", "inner durable capture")), "inner capture directory")
    _expect(capture_dir, clean.parent, "inner capture directory")
    capture_evidence = {
        name: _file_evidence(capture_dir / filename, f"inner capture {name}")
        for name, filename in {
            "invocation": _require_string(capture, "invocation_file", "inner durable capture"),
            "stdout": _require_string(capture, "stdout_file", "inner durable capture"),
            "stderr": _require_string(capture, "stderr_file", "inner durable capture"),
            "receipt": _require_string(capture, "receipt_file", "inner durable capture"),
        }.items()
    }
    _expect(capture_evidence["receipt"]["path"], str(clean), "inner capture receipt path")
    oracle = _require_object(document, "cpu_oracle", "shared-expert CPU inner receipt")
    gated = _require_object(oracle, "gated_shared_output", "shared-expert CPU oracle")
    _expect(gated.get("all_2048_values_finite"), True, "shared-expert CPU gated output finite")
    _require_string(gated, "candidate_gated_shared_sha256", "shared-expert CPU gated output")
    return document, {
        "path": str(clean),
        "present": True,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }, capture_evidence


def build_wrapper(
    *, baseline_receipt: Path, manifest: Path, admission_current: Path
) -> dict[str, Any]:
    """Validate the current binding and return an unsigned wrapper body."""
    _manifest, manifest_evidence = _validate_manifest(manifest)
    admission, admission_evidence = _validate_admission(admission_current, manifest_evidence)
    inner, inner_evidence, capture_evidence = _validate_inner(
        baseline_receipt, manifest_evidence, admission_evidence, admission
    )
    binding = inner["artifact_binding"]
    oracle = inner["cpu_oracle"]
    return {
        "schema": WRAPPER_SCHEMA,
        "status": WRAPPER_STATUS,
        "recorded_at": _utc_now(),
        "source_binding": {
            "model": {"id": MODEL_ID, "key": MODEL_KEY},
            "manifest": manifest_evidence,
            "manifest_seal_sha256": MANIFEST_SEAL,
            "admission_current": admission_evidence,
            "admission_current_seal_sha256": admission["seal_sha256"],
            "admission_receipt_seal_sha256": admission["admission_receipt"]["seal_sha256"],
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_body_audit_seal_sha256": SOURCE_BODY_AUDIT_SEAL,
            "source_revalidation_seal_sha256": SOURCE_REVALIDATION_SEAL,
            "layer": binding["layer"],
            "layer_kind": binding["layer_kind"],
            "hidden": binding["hidden"],
            "shared_expert_intermediate": binding["shared_expert_intermediate"],
            "tensor_payload_sha256": {
                field: binding[field]["artifact_sha256"] for field in EXPECTED_TENSORS
            },
        },
        "cpu_inner_receipt": inner_evidence,
        "durable_cpu_capture": {
            "receipt_written_last_is_completion_marker": True,
            "files": capture_evidence,
        },
        "cpu_component_output": {
            "postnorm_hidden_sha256": oracle["postnorm_hidden_sha256"],
            "shared_gate_sha256": oracle["shared_gate_up"]["candidate_gate_sha256"],
            "shared_up_sha256": oracle["shared_gate_up"]["candidate_up_sha256"],
            "shared_swiglu_sha256": oracle["swiglu"]["candidate_activated_sha256"],
            "shared_down_sha256": oracle["shared_down"]["candidate_down_sha256"],
            "gated_shared_sha256": oracle["gated_shared_output"]["candidate_gated_shared_sha256"],
            "gated_shared_f32_vs_source_f64_max_abs": oracle["gated_shared_output"][
                "candidate_f32_vs_full_source_f64_chain_max_abs"
            ],
        },
        "claim_boundary": {
            "sealed_wrapper_attests_an_unsigned_cpu_component_baseline_only": True,
            "does_not_reseal_or_modify_the_inner_receipt": True,
            "does_not_perform_a_complete_artifact_scan": True,
            "does_not_open_raw_bf16_or_safetensors": True,
            "does_not_perform_metal_device_execution": True,
            "does_not_combine_routed_experts_or_apply_second_residual": True,
            "does_not_claim_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament": True,
        },
    }


def _write_new_sealed(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    clean = path if path.is_absolute() else (_ for _ in ()).throw(ValueError("output must be absolute"))
    if clean.exists():
        raise FileExistsError(f"refusing to overwrite existing immutable wrapper: {clean}")
    if not clean.parent.is_dir():
        raise ValueError(f"output parent must already exist: {clean.parent}")
    sealed = seal(body)
    raw = json.dumps(sealed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(clean, flags, 0o644)
    try:
        total = 0
        while total < len(raw):
            written = os.write(fd, raw[total:])
            if written <= 0:
                raise OSError("short immutable wrapper write")
            total += written
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(clean.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    _, observed_raw, observed = _json_object(clean, "sealed shared-expert baseline wrapper")
    _verify_sealed(observed, "sealed shared-expert baseline wrapper")
    if observed != sealed:
        raise ValueError("sealed shared-expert baseline wrapper changed after durable write")
    if _sha256(observed_raw) != _sha256(raw):
        raise ValueError("sealed shared-expert baseline wrapper raw bytes changed after durable write")
    return observed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--admission-current", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        body = build_wrapper(
            baseline_receipt=args.baseline_receipt,
            manifest=args.manifest,
            admission_current=args.admission_current,
        )
        sealed = _write_new_sealed(args.output, body)
    except (OSError, ValueError, SealIntegrityError) as exc:
        print(f"ascension_qwen80_shared_expert_baseline_wrapper: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(sealed, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
