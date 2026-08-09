#!/usr/bin/env python3
"""Seal a Qwen80 MoE-combine CPU baseline without promoting it.

The MoE-combine component intentionally produces an unsigned CPU receipt.
This tool attests the exact receipt bytes, its receipt-last capture, current
complete-artifact/admission identity, and the sealed postnorm/router source
top-10 chain in one new immutable wrapper.  It does not open a tensor payload,
create a Metal context, modify a watcher, or issue a device lease.
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

# The detached campaign invokes some operators by absolute script path.  Keep
# this small receipt tool importable both that way and as ``python -m``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify


WRAPPER_SCHEMA = "hawking.ascension.qwen80_moe_combine_cpu_baseline_wrapper.v1"
WRAPPER_STATUS = (
    "SEALED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_"
    "CPU_ORACLE_BASELINE"
)
CPU_INNER_SCHEMA = "hawking.ascension.qwen80_direct_packed_moe_combine.v1"
CPU_INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_"
    "CPU_ORACLE_READY_METAL_LEASE_REQUIRED"
)
MANIFEST_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
ADMISSION_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
ROUTER_INNER_SCHEMA = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1"
ROUTER_INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_"
    "STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)
ROUTER_OUTER_SCHEMA = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1"
ROUTER_OUTER_STATUS = "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY"

MODEL_ID = "Qwen3-Coder-Next-80B"
MODEL_KEY = "qwen80"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
MANIFEST_SHA256 = "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10"
MANIFEST_SEAL = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b"
ADMISSION_RECEIPT_SEAL = "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628"
SOURCE_BODY_AUDIT_SEAL = "c572b2270b623b8677c374b43c89ddd729de135c25721488bb874b184ff8c3d4"
SOURCE_REVALIDATION_SEAL = "541b16fca1d4805ecba356face97b4e8de1accdeb21e98ee0c13b70ab0746c45"
SOURCE_TOP10_IDS = [65, 245, 227, 35, 189, 440, 298, 405, 109, 494]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_regular(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"cannot canonicalize {label} {path}: {exc}") from exc


def _canonical_dir(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory, not a symlink: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"cannot canonicalize {label} {path}: {exc}") from exc


def _read_json(path: Path, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    clean = _canonical_regular(path, label)
    try:
        raw = clean.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {label} {clean}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} root must be an object")
    return clean, raw, document


def _file_evidence(path: Path, label: str) -> dict[str, Any]:
    clean = _canonical_regular(path, label)
    raw = clean.read_bytes()
    return {"path": str(clean), "present": True, "bytes": len(raw), "sha256": _sha256(raw)}


def _required_string(document: Mapping[str, Any], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} missing non-empty string {key!r}")
    return value


def _required_object(document: Mapping[str, Any], key: str, label: str) -> dict[str, Any]:
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


def _path(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path string")
    return _canonical_regular(Path(value), label)


def _assert_evidence(value: object, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be file evidence")
    if (
        value.get("present") is not True
        or _path(value.get("path"), f"{label}.path") != Path(str(expected["path"]))
        or value.get("bytes") != expected["bytes"]
        or value.get("sha256") != expected["sha256"]
    ):
        raise ValueError(f"{label} immutable file evidence drifted")


def _validate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    clean, raw, document = _read_json(path, "manifest")
    _verify_sealed(document, "manifest")
    _expect(_sha256(raw), MANIFEST_SHA256, "manifest raw SHA-256")
    _expect(document.get("schema"), MANIFEST_SCHEMA, "manifest schema")
    _expect(document.get("status"), "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED", "manifest status")
    _expect(document.get("seal_sha256"), MANIFEST_SEAL, "manifest seal")
    _expect(document.get("source_body_audit_seal_sha256"), SOURCE_BODY_AUDIT_SEAL, "manifest source audit")
    _expect(
        document.get("source_revalidation_receipt_seal_sha256"),
        SOURCE_REVALIDATION_SEAL,
        "manifest source revalidation",
    )
    source = _required_object(document, "source", "manifest")
    _expect(source.get("repository"), SOURCE_REPOSITORY, "manifest source repository")
    _expect(source.get("tensor_count"), 74_391, "manifest source tensor count")
    return document, {"path": str(clean), "present": True, "bytes": len(raw), "sha256": _sha256(raw)}


def _validate_admission(path: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    clean, raw, document = _read_json(path, "admission current pointer")
    _verify_sealed(document, "admission current pointer")
    _expect(document.get("schema"), ADMISSION_SCHEMA, "admission schema")
    _expect(document.get("status"), "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED", "admission status")
    model = _required_object(document, "model", "admission")
    _expect(model.get("id"), MODEL_ID, "admission model id")
    _expect(model.get("key"), MODEL_KEY, "admission model key")
    _expect(model.get("repository"), SOURCE_REPOSITORY, "admission source repository")
    _expect(model.get("revision"), SOURCE_REVISION, "admission source revision")
    complete = _required_object(document, "complete_manifest", "admission")
    _expect(_path(complete.get("path"), "admission manifest path"), Path(str(manifest["path"])), "admission manifest path")
    _expect(complete.get("document_sha256"), manifest["sha256"], "admission manifest SHA-256")
    _expect(complete.get("seal_sha256"), MANIFEST_SEAL, "admission manifest seal")
    receipt = _required_object(document, "admission_receipt", "admission")
    _expect(receipt.get("seal_sha256"), ADMISSION_RECEIPT_SEAL, "admission receipt seal")
    return document, {"path": str(clean), "present": True, "bytes": len(raw), "sha256": _sha256(raw)}


def _validate_top10_binding(
    value: object,
    router: Mapping[str, Any],
    router_outer: Mapping[str, Any],
    router_outer_seal: str,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if (
        _path(value.get("router_receipt_path"), f"{label}.router_receipt_path")
        != Path(str(router["path"]))
        or value.get("router_receipt_sha256") != router["sha256"]
        or _path(value.get("router_outer_receipt_path"), f"{label}.router_outer_receipt_path")
        != Path(str(router_outer["path"]))
        or value.get("router_outer_receipt_sha256") != router_outer["sha256"]
        or value.get("router_outer_receipt_seal_sha256") != router_outer_seal
        or value.get("ids") != SOURCE_TOP10_IDS
    ):
        raise ValueError(f"{label} source top-10 binding drifted")


def _validate_router(
    inner_path: Path,
    outer_path: Path,
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    inner_clean, inner_raw, inner = _read_json(inner_path, "router inner receipt")
    _expect(inner.get("schema"), ROUTER_INNER_SCHEMA, "router inner schema")
    _expect(inner.get("status"), ROUTER_INNER_STATUS, "router inner status")
    _expect(inner.get("mode"), "metal", "router inner mode")
    _expect(inner.get("component_only"), True, "router inner component boundary")
    _expect(inner.get("metal_device_or_dispatch_performed"), True, "router inner Metal execution")
    artifact = _required_object(inner, "artifact_binding", "router inner")
    _expect(_path(artifact.get("manifest_path"), "router inner manifest path"), Path(str(manifest["path"])), "router inner manifest path")
    _expect(artifact.get("manifest_document_sha256"), manifest["sha256"], "router inner manifest SHA")
    _expect(artifact.get("manifest_seal_sha256"), MANIFEST_SEAL, "router inner manifest seal")
    _expect(_path(artifact.get("admission_current_path"), "router inner admission path"), Path(str(admission["path"])), "router inner admission path")
    _expect(artifact.get("admission_receipt_seal_sha256"), ADMISSION_RECEIPT_SEAL, "router inner admission receipt seal")
    _expect(artifact.get("layer"), 0, "router inner layer")
    _expect(artifact.get("hidden"), 2_048, "router inner hidden")
    _expect(artifact.get("experts_per_token"), 10, "router inner top-k")
    route = _required_object(inner, "source_stable_top10_router", "router inner")
    _expect(route.get("ids"), SOURCE_TOP10_IDS, "router source IDs")
    _expect(route.get("device_ids"), SOURCE_TOP10_IDS, "router device IDs")
    _expect(route.get("device_ids_exact_match"), True, "router device ID parity")
    _expect(route.get("ids_unique_and_in_range"), True, "router source ID geometry")
    weights = route.get("renormalized_weights")
    if not isinstance(weights, list) or len(weights) != 10 or not all(isinstance(v, (int, float)) and float(v) > 0.0 for v in weights):
        raise ValueError("router renormalized source weights are malformed")
    if abs(sum(float(v) for v in weights) - 1.0) > 2.0e-6:
        raise ValueError("router renormalized source weights do not sum to one")
    router = {"path": str(inner_clean), "present": True, "bytes": len(inner_raw), "sha256": _sha256(inner_raw)}

    outer_clean, outer_raw, outer = _read_json(outer_path, "router outer terminal")
    _verify_sealed(outer, "router outer terminal")
    _expect(outer.get("schema"), ROUTER_OUTER_SCHEMA, "router outer schema")
    _expect(outer.get("status"), ROUTER_OUTER_STATUS, "router outer status")
    outer_source = _required_object(outer, "source_binding", "router outer")
    _assert_evidence(_required_object(outer_source, "manifest", "router outer"), manifest, "router outer manifest")
    outer_admission = _required_object(outer_source, "admission_current", "router outer")
    _expect(_path(outer_admission.get("path"), "router outer admission path"), Path(str(admission["path"])), "router outer admission path")
    captured = _required_object(outer, "inner_probe_capture", "router outer")
    _expect(_path(captured.get("path"), "router outer inner path"), Path(str(router["path"])), "router outer inner path")
    _expect(captured.get("sha256"), router["sha256"], "router outer inner SHA")
    _expect(captured.get("schema"), ROUTER_INNER_SCHEMA, "router outer inner schema")
    _expect(captured.get("status"), ROUTER_INNER_STATUS, "router outer inner status")
    _expect(captured.get("mode"), "metal", "router outer inner mode")
    _expect(captured.get("metal_performed"), True, "router outer Metal execution")
    outer_evidence = {"path": str(outer_clean), "present": True, "bytes": len(outer_raw), "sha256": _sha256(outer_raw)}
    return router, outer_evidence, _required_string(outer, "seal_sha256", "router outer terminal")


def _capture_files(inner: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    durable = _required_object(inner, "durable_capture", label)
    _expect(durable.get("receipt_written_last_is_completion_marker"), True, f"{label} receipt-last")
    raw_directory = durable.get("directory")
    if not isinstance(raw_directory, str):
        raise ValueError(f"{label} capture directory must be an absolute path string")
    directory = _canonical_dir(Path(raw_directory), f"{label} capture directory")
    files: list[dict[str, Any]] = []
    for field in ("invocation_file", "stdout_file", "stderr_file", "receipt_file"):
        name = _required_string(durable, field, label)
        files.append(_file_evidence(directory / name, f"{label} {field}"))
    return files


def _validate_cpu_inner(
    path: Path,
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
    router: Mapping[str, Any],
    router_outer: Mapping[str, Any],
    router_outer_seal: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    clean, raw, inner = _read_json(path, "MoE-combine CPU inner receipt")
    _expect(inner.get("schema"), CPU_INNER_SCHEMA, "MoE-combine CPU inner schema")
    _expect(inner.get("status"), CPU_INNER_STATUS, "MoE-combine CPU inner status")
    _expect(inner.get("mode"), "cpu-oracle", "MoE-combine CPU inner mode")
    for field, expected in (
        ("metal_device_or_dispatch_performed", False),
        ("component_only", True),
        ("routed_expert_aggregation_performed", True),
        ("shared_expert_add_performed", True),
        ("second_residual_performed", True),
        ("complete_layer_or_token_performed", False),
        ("materialized_source_route_shaped_fixture_only", True),
    ):
        _expect(inner.get(field), expected, f"MoE-combine CPU inner {field}")
    artifact = _required_object(inner, "artifact_binding", "MoE-combine CPU inner")
    _expect(_path(artifact.get("manifest_path"), "MoE-combine CPU manifest path"), Path(str(manifest["path"])), "MoE-combine CPU manifest path")
    _expect(artifact.get("manifest_document_sha256"), manifest["sha256"], "MoE-combine CPU manifest SHA")
    _expect(artifact.get("manifest_seal_sha256"), MANIFEST_SEAL, "MoE-combine CPU manifest seal")
    _expect(_path(artifact.get("admission_current_path"), "MoE-combine CPU admission path"), Path(str(admission["path"])), "MoE-combine CPU admission path")
    _expect(artifact.get("admission_receipt_seal_sha256"), ADMISSION_RECEIPT_SEAL, "MoE-combine CPU admission receipt seal")
    _validate_top10_binding(
        _required_object(inner, "source_top10_binding", "MoE-combine CPU inner"),
        router,
        router_outer,
        router_outer_seal,
        "MoE-combine CPU top-10",
    )
    files = _capture_files(inner, "MoE-combine CPU inner")
    return inner, {"path": str(clean), "present": True, "bytes": len(raw), "sha256": _sha256(raw)}, files


def build_wrapper(
    *,
    cpu_inner_receipt: Path,
    manifest_path: Path,
    admission_current: Path,
    router_receipt: Path,
    router_outer_receipt: Path,
) -> dict[str, Any]:
    _manifest, manifest = _validate_manifest(manifest_path)
    admission_document, admission = _validate_admission(admission_current, manifest)
    router, router_outer, router_outer_seal = _validate_router(
        router_receipt, router_outer_receipt, manifest, admission
    )
    inner, inner_evidence, capture_files = _validate_cpu_inner(
        cpu_inner_receipt, manifest, admission, router, router_outer, router_outer_seal
    )
    return {
        "schema": WRAPPER_SCHEMA,
        "status": WRAPPER_STATUS,
        "recorded_at": _utc_now(),
        "source_binding": {
            "model": {"id": MODEL_ID, "key": MODEL_KEY},
            "manifest": manifest,
            "manifest_seal_sha256": MANIFEST_SEAL,
            "admission_current": admission,
            "admission_current_seal_sha256": admission_document["seal_sha256"],
            "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_body_audit_seal_sha256": SOURCE_BODY_AUDIT_SEAL,
            "source_revalidation_seal_sha256": SOURCE_REVALIDATION_SEAL,
            "source_top10_binding": {
                "router_receipt_path": router["path"],
                "router_receipt_sha256": router["sha256"],
                "router_outer_receipt_path": router_outer["path"],
                "router_outer_receipt_sha256": router_outer["sha256"],
                "router_outer_receipt_seal_sha256": router_outer_seal,
                "ids": SOURCE_TOP10_IDS,
            },
        },
        "cpu_inner_receipt": inner_evidence,
        "durable_cpu_capture": {
            "receipt_written_last_is_completion_marker": True,
            "files": capture_files,
        },
        "cpu_component_output": {
            "materialized_source_route_shaped_fixture_only": True,
            "routed_sum": _required_object(inner, "cpu_oracle", "MoE-combine CPU inner").get("routed_sum"),
            "second_residual": _required_object(inner, "cpu_oracle", "MoE-combine CPU inner").get("second_residual"),
            "routed_sum_f32_vs_f64_max_abs": _required_object(inner, "cpu_oracle", "MoE-combine CPU inner").get("routed_sum_f32_vs_f64_max_abs"),
            "second_residual_f32_vs_f64_max_abs": _required_object(inner, "cpu_oracle", "MoE-combine CPU inner").get("second_residual_f32_vs_f64_max_abs"),
        },
        "claim_boundary": {
            "sealed_wrapper_attests_an_unsigned_cpu_combine_component_only": True,
            "does_not_reseal_or_modify_the_inner_receipt": True,
            "does_not_open_tensor_payloads_or_raw_bf16_or_safetensors": True,
            "does_not_perform_metal_device_execution_or_issue_a_lease": True,
            "does_not_establish_ten_physical_routed_experts_or_a_complete_layer": True,
            "does_not_claim_token_decoder_generation_hcli_tps_tg_or_tournament": True,
        },
    }


def _write_new_sealed(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("--output must be absolute")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable wrapper: {path}")
    if not path.parent.is_dir():
        raise ValueError(f"--output parent must exist: {path.parent}")
    sealed = seal(body)
    raw = json.dumps(sealed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        written = 0
        while written < len(raw):
            count = os.write(fd, raw[written:])
            if count <= 0:
                raise OSError("short immutable wrapper write")
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    _, observed_raw, observed = _read_json(path, "sealed MoE-combine CPU baseline wrapper")
    _verify_sealed(observed, "sealed MoE-combine CPU baseline wrapper")
    if observed != sealed or _sha256(observed_raw) != _sha256(raw):
        raise ValueError("sealed MoE-combine CPU baseline wrapper changed after durable write")
    return observed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-inner-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--admission-current", required=True, type=Path)
    parser.add_argument("--router-receipt", required=True, type=Path)
    parser.add_argument("--router-outer-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        body = build_wrapper(
            cpu_inner_receipt=args.cpu_inner_receipt,
            manifest_path=args.manifest,
            admission_current=args.admission_current,
            router_receipt=args.router_receipt,
            router_outer_receipt=args.router_outer_receipt,
        )
        sealed = _write_new_sealed(args.output, body)
    except (OSError, ValueError, SealIntegrityError) as exc:
        print(f"ascension_qwen80_moe_combine_baseline_wrapper: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(sealed, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
