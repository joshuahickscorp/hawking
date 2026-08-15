"""Prepare the exact CPU-oracle input for Qwen30's HQ30GR2 sparse pair.

This is deliberately a CPU-only receipt constructor.  It binds one already
captured device-produced L0 router-input vector (the ``literal_hawking``
position where expert 0 was selected) to the separately admitted HQ30GR2
candidate and to the non-serving sparse gate/up parity executable.  It does
not open a Metal context, load a model, start a watcher, or make a coherence,
HCLI, TPS, capability, or tournament claim.

The resulting request gives an outer process controller the exact immutable
arguments for its required two-stage sequence:

1. `cpu-oracle`, with no device work; then
2. a fresh, separately leased `device-parity` invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_ADMISSION_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_NATIVE_ADMISSION_CURRENT.json"
DEFAULT_ROUTE_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_L0_ROUTE_CAPTURE.json"
DEFAULT_PREPARATION_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_ALL_LAYER_CURRENT_TRACE_PREPARATION_CURRENT.json"
DEFAULT_PROBE_BINARY = (
    REPO_ROOT
    / "workspace/ops/build/rust/debug/examples"
    / "ascension_qwen30_hq30gr2_sparse_gate_up_device_parity"
)

SCHEMA = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_component_prepare.v1"
CURRENT_SCHEMA = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_component_prepare_current.v1"
STATUS = "PREPARED_HQ30GR2_LITERAL_HAWKING_L0_E0_CPU_ORACLE_INPUT_NOT_RUN"
CURRENT_STATUS = "CURRENT_PREPARED_HQ30GR2_LITERAL_HAWKING_L0_E0_CPU_ORACLE_INPUT_SELECTED"

ADMISSION_CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_current_pointer.v1"
ADMISSION_CURRENT_STATUS = "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED"
ADMISSION_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_receipt.v1"
ADMISSION_STATUS = "EARNED_QUALITY_REPACK_COMPLETE_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
ROUTE_CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture_current.v1"
ROUTE_CURRENT_STATUS = "CURRENT_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_CAPTURE_SELECTED"
ROUTE_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture.v1"
ROUTE_STATUS = "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED"
PREPARATION_CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_prepare_current.v1"
PREPARATION_CURRENT_STATUS = "CURRENT_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_PREPARATION_SELECTED"
PREPARATION_SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_prepare.v1"
PREPARATION_STATUS = "PREPARED_CURRENT_TRACE_TYPED_HQ30GR2_ALL_LAYER_DIAGNOSTIC_NOT_RUN"

EXPECTED_BINARY_BASENAME = "ascension_qwen30_hq30gr2_sparse_gate_up_device_parity"
TARGET_PROBE = "literal_hawking"
TARGET_POSITION = 337
TARGET_TOKEN_COUNT = 369
TARGET_HIDDEN_RELATIVE = "hidden/literal_hawking/000337.f32le"
TARGET_HIDDEN_BYTES = 2048 * 4


class ComponentPreparationError(RuntimeError):
    """An upstream binding is missing or differs from the typed component scope."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComponentPreparationError(f"{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComponentPreparationError(f"{label} must be an object")
    return dict(value)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise ComponentPreparationError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ComponentPreparationError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ComponentPreparationError(f"{label} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ComponentPreparationError(f"{label} must be executable: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ComponentPreparationError(f"cannot canonicalize {label} {path}: {exc}") from exc


def _sealed(path: Path, label: str) -> tuple[dict[str, Any], Path]:
    canonical = _canonical_regular(path, label)
    try:
        result = verify(json.loads(canonical.read_text(encoding="utf-8")), label=str(canonical))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError, ValueError) as exc:
        raise ComponentPreparationError(f"{label} is absent or has an invalid seal: {exc}") from exc
    if not isinstance(result, Mapping):
        raise ComponentPreparationError(f"{label} is not an object")
    if not _is_sha256(result.get("seal_sha256")):
        raise ComponentPreparationError(f"{label} lacks a lowercase SHA-256 seal")
    return dict(result), canonical


def _file_evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    canonical = _canonical_regular(path, label, executable=executable)
    return {
        "path": str(canonical),
        "present": True,
        "bytes": canonical.stat().st_size,
        "sha256": _sha256_file(canonical),
    }


def _atomic_json_new(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise ComponentPreparationError(f"refusing to overwrite immutable receipt {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise ComponentPreparationError(f"refusing to overwrite immutable receipt {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json_replace(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _selected(
    current_path: Path,
    *,
    current_schema: str,
    current_status: str,
    field: str,
    receipt_schema: str,
    receipt_status: str,
    label: str,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    current, canonical_current = _sealed(current_path, f"current {label} pointer")
    if current.get("schema") != current_schema or current.get("status") != current_status:
        raise ComponentPreparationError(f"current {label} pointer schema/status drifted")
    selected = _mapping(current.get(field), f"current {label} pointer {field}")
    receipt_path = _canonical_regular(Path(_text(selected.get("path"), f"current {label} receipt path")), f"current {label} receipt")
    receipt, canonical_receipt = _sealed(receipt_path, f"current {label} receipt")
    if selected.get("seal_sha256") != receipt.get("seal_sha256"):
        raise ComponentPreparationError(f"current {label} receipt seal differs from pointer")
    if receipt.get("schema") != receipt_schema or receipt.get("status") != receipt_status:
        raise ComponentPreparationError(f"current {label} receipt schema/status drifted")
    return current, canonical_current, receipt, canonical_receipt


def _reference_path(row: object, label: str) -> tuple[Path, str]:
    value = _mapping(row, label)
    path = _canonical_regular(Path(_text(value.get("path"), f"{label}.path")), f"{label}.path")
    receipt, _ = _sealed(path, label)
    observed = receipt.get("seal_sha256")
    if value.get("seal_sha256") != observed:
        raise ComponentPreparationError(f"{label} seal differs from its file")
    return path, str(observed)


def _admission_context(
    current_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    current, canonical_current, receipt, canonical_receipt = _selected(
        current_path,
        current_schema=ADMISSION_CURRENT_SCHEMA,
        current_status=ADMISSION_CURRENT_STATUS,
        field="admission_receipt",
        receipt_schema=ADMISSION_SCHEMA,
        receipt_status=ADMISSION_STATUS,
        label="candidate admission",
    )
    manifest = _mapping(current.get("complete_manifest"), "candidate admission complete_manifest")
    native_loader = _mapping(receipt.get("native_loader"), "candidate admission native_loader")
    if (
        manifest.get("seal_sha256") != native_loader.get("manifest_seal_sha256")
        or manifest.get("path") != native_loader.get("manifest_path")
        or native_loader.get("tensor_count") != 18_867
        or native_loader.get("selected_residual_organs")
        != [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
        ]
    ):
        raise ComponentPreparationError("candidate admission does not bind the exact HQ30GR2 L0/E0 pair")
    fields = {
        "manifest": _file_evidence(
            _canonical_regular(Path(_text(manifest.get("path"), "candidate manifest path")), "candidate manifest"),
            "candidate manifest",
        ),
        "manifest_seal_sha256": _text(manifest.get("seal_sha256"), "candidate manifest seal"),
        "source_audit_seal_sha256": _text(native_loader.get("source_audit_seal_sha256"), "candidate source audit seal"),
        "source_revision": _text(native_loader.get("source_revision"), "candidate source revision"),
    }
    for name, key in (
        ("revalidation", "immutable_source_revalidation"),
        ("selection", "selection_receipt"),
        ("source_snapshot", "source_binding_snapshot"),
        ("terminal", "terminal"),
    ):
        row = _mapping(receipt.get(key), f"candidate admission {key}")
        path, seal_sha256 = _reference_path(row, f"candidate admission {key}")
        fields[name] = {"path": str(path), "seal_sha256": seal_sha256, "document_sha256": _sha256_file(path)}
    fields["admission_current"] = {
        **_file_evidence(canonical_current, "candidate admission current"),
        "seal_sha256": current.get("seal_sha256"),
    }
    fields["admission_receipt"] = {
        **_file_evidence(canonical_receipt, "candidate admission receipt"),
        "seal_sha256": receipt.get("seal_sha256"),
    }
    return current, canonical_current, receipt, canonical_receipt, fields


def _component_input(
    route_receipt: Mapping[str, Any],
    preparation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    planned = _mapping(preparation_receipt.get("planned_bounded_input"), "all-layer preparation input")
    hidden = _mapping(planned.get("l0_e0_router_input_hidden"), "all-layer preparation hidden input")
    if (
        planned.get("probe_id") != TARGET_PROBE
        or planned.get("l0_e0_selected_position") != TARGET_POSITION
        or planned.get("source_template_token_count") != TARGET_TOKEN_COUNT
        or hidden.get("relative_path") != TARGET_HIDDEN_RELATIVE
        or not _is_sha256(hidden.get("sha256"))
    ):
        raise ComponentPreparationError("all-layer preparation does not select literal_hawking L0/E0 position 337")
    route_binding = _mapping(route_receipt.get("binding"), "route capture binding")
    output_root = Path(_text(route_binding.get("capture_output_root"), "route capture output root"))
    if not output_root.is_absolute():
        raise ComponentPreparationError("route capture output root is not absolute")
    relative = PurePath(_text(hidden.get("relative_path"), "selected hidden relative path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ComponentPreparationError("selected hidden path escapes the immutable route-capture root")
    input_path = _canonical_regular(output_root / relative, "selected L0 router input F32LE")
    if input_path.stat().st_size != TARGET_HIDDEN_BYTES:
        raise ComponentPreparationError("selected L0 router input does not have 2048 F32LE values")
    if _sha256_file(input_path) != hidden.get("sha256"):
        raise ComponentPreparationError("selected L0 router input SHA-256 differs from preparation binding")
    summary_rows = route_receipt.get("probe_summary")
    if not isinstance(summary_rows, list):
        raise ComponentPreparationError("route capture has no probe summary")
    summary = next(
        (row for row in summary_rows if isinstance(row, Mapping) and row.get("probe_id") == TARGET_PROBE),
        None,
    )
    if not isinstance(summary, Mapping):
        raise ComponentPreparationError("route capture does not contain literal_hawking")
    if (
        summary.get("source_template_token_count") != TARGET_TOKEN_COUNT
        or summary.get("l0_expert0_selected_positions") != [TARGET_POSITION]
    ):
        raise ComponentPreparationError("route capture does not prove the selected literal_hawking E0 membership")
    payloads = summary.get("hidden_payloads")
    if not isinstance(payloads, list) or not any(
        isinstance(row, Mapping)
        and row.get("relative_path") == TARGET_HIDDEN_RELATIVE
        and row.get("sha256") == hidden.get("sha256")
        for row in payloads
    ):
        raise ComponentPreparationError("route capture hidden payload catalog does not contain the selected vector")
    return {
        "probe_id": TARGET_PROBE,
        "source_template_token_count": TARGET_TOKEN_COUNT,
        "l0_e0_selected_position": TARGET_POSITION,
        "device_produced_router_input_f32le": {
            "path": str(input_path),
            "sha256": _sha256_file(input_path),
            "bytes": TARGET_HIDDEN_BYTES,
            "elements": 2048,
            "relative_to_route_capture_root": TARGET_HIDDEN_RELATIVE,
        },
    }


def run_once(
    *,
    root: Path,
    admission_current_path: Path,
    route_current_path: Path,
    preparation_current_path: Path,
    probe_binary: Path,
) -> dict[str, Any]:
    probe = _file_evidence(probe_binary, "HQ30GR2 sparse parity binary", executable=True)
    if Path(probe["path"]).name != EXPECTED_BINARY_BASENAME:
        raise ComponentPreparationError("probe binary basename is not the typed sparse gate/up parity executable")
    _admission_current, admission_current, admission_receipt, admission_receipt_path, admission = _admission_context(
        admission_current_path
    )
    route_current, route_current_canonical, route_receipt, route_receipt_path = _selected(
        route_current_path,
        current_schema=ROUTE_CURRENT_SCHEMA,
        current_status=ROUTE_CURRENT_STATUS,
        field="route_capture_receipt",
        receipt_schema=ROUTE_SCHEMA,
        receipt_status=ROUTE_STATUS,
        label="current HCLI L0 route capture",
    )
    preparation_current, preparation_current_canonical, preparation_receipt, preparation_receipt_path = _selected(
        preparation_current_path,
        current_schema=PREPARATION_CURRENT_SCHEMA,
        current_status=PREPARATION_CURRENT_STATUS,
        field="preparation_receipt",
        receipt_schema=PREPARATION_SCHEMA,
        receipt_status=PREPARATION_STATUS,
        label="all-layer preparation",
    )
    preparation_binding = _mapping(preparation_receipt.get("binding"), "all-layer preparation binding")
    if preparation_binding.get("candidate_manifest_seal_sha256") != admission["manifest_seal_sha256"]:
        raise ComponentPreparationError("all-layer preparation candidate manifest seal differs from current admission")
    admission_ref = _mapping(
        preparation_binding.get("candidate_admission_current_pointer"), "all-layer preparation admission pointer"
    )
    if (
        admission_ref.get("path") != str(admission_current)
        or admission_ref.get("seal_sha256") != admission["admission_current"]["seal_sha256"]
    ):
        raise ComponentPreparationError("all-layer preparation candidate admission pointer differs from current admission")
    route_ref = _mapping(preparation_binding.get("route_capture_receipt"), "all-layer preparation route receipt")
    if route_ref.get("path") != str(route_receipt_path) or route_ref.get("seal_sha256") != route_receipt.get("seal_sha256"):
        raise ComponentPreparationError("all-layer preparation route receipt differs from current route capture")
    component_input = _component_input(route_receipt, preparation_receipt)
    command_common = [
        str(probe["path"]),
        "--manifest",
        str(admission["manifest"]["path"]),
        "--expected-manifest-seal-sha256",
        str(admission["manifest_seal_sha256"]),
        "--expected-source-audit-seal-sha256",
        str(admission["source_audit_seal_sha256"]),
        "--expected-source-revision",
        str(admission["source_revision"]),
        "--expected-revalidation-path",
        str(admission["revalidation"]["path"]),
        "--expected-revalidation-seal-sha256",
        str(admission["revalidation"]["seal_sha256"]),
        "--expected-selection-path",
        str(admission["selection"]["path"]),
        "--expected-selection-seal-sha256",
        str(admission["selection"]["seal_sha256"]),
        "--expected-source-snapshot-path",
        str(admission["source_snapshot"]["path"]),
        "--expected-source-snapshot-seal-sha256",
        str(admission["source_snapshot"]["seal_sha256"]),
        "--expected-terminal-path",
        str(admission["terminal"]["path"]),
        "--expected-terminal-seal-sha256",
        str(admission["terminal"]["seal_sha256"]),
        "--input-f32le",
        str(component_input["device_produced_router_input_f32le"]["path"]),
        "--expected-input-sha256",
        str(component_input["device_produced_router_input_f32le"]["sha256"]),
        "--max-seq-len",
        "512",
    ]
    binding = {
        "candidate_admission_current": admission["admission_current"],
        "candidate_admission_receipt": admission["admission_receipt"],
        "candidate_manifest": admission["manifest"],
        "candidate_manifest_seal_sha256": admission["manifest_seal_sha256"],
        "route_capture_current": {
            **_file_evidence(route_current_canonical, "route capture current"),
            "seal_sha256": route_current.get("seal_sha256"),
        },
        "route_capture_receipt": {
            **_file_evidence(route_receipt_path, "route capture receipt"),
            "seal_sha256": route_receipt.get("seal_sha256"),
        },
        "all_layer_preparation_current": {
            **_file_evidence(preparation_current_canonical, "all-layer preparation current"),
            "seal_sha256": preparation_current.get("seal_sha256"),
        },
        "all_layer_preparation_receipt": {
            **_file_evidence(preparation_receipt_path, "all-layer preparation receipt"),
            "seal_sha256": preparation_receipt.get("seal_sha256"),
        },
        "probe_binary": probe,
        "component_input": component_input,
    }
    source_digest = _canonical_json_sha256(binding)
    receipt = seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "recorded_at": _utc_now(),
            "binding": binding,
            "cpu_oracle_invocation": {
                "mode": "cpu-oracle",
                "command_without_output_dir": [*command_common[:1], "--mode", "cpu-oracle", *command_common[1:]],
                "outer_controller_must_create_fresh_output_dir": True,
                "metal_context_or_dispatch_performed": False,
            },
            "future_device_parity_contract": {
                "mode": "device-parity",
                "requires_cpu_oracle_activation_f64le_from_this_exact_binding": True,
                "requires_fresh_explicit_quiet_gpu_lease": True,
                "one_device_process_one_lease_one_terminal_receipt_or_refusal": True,
                "automatic_retry_forbidden": True,
            },
            "claim_boundary": {
                "preparation_only": True,
                "input_is_device_produced_l0_router_state_but_not_a_full_model_result": True,
                "no_metal_context_no_model_forward_no_server_or_watcher_change": True,
                "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": True,
            },
        }
    )
    receipt_path = (
        root
        / "sparse-gate-up-component-preparation/receipts"
        / f"QWEN30_HQ30GR2_SPARSE_GATE_UP_COMPONENT_PREPARATION_{admission['manifest_seal_sha256']}_{route_receipt['seal_sha256'][:16]}_{source_digest[:16]}.json"
    )
    if receipt_path.exists():
        existing, _ = _sealed(receipt_path, "existing sparse gate/up component preparation")
        if existing.get("binding") != receipt.get("binding") or existing.get("status") != STATUS:
            raise ComponentPreparationError("refusing to overwrite a distinct component preparation receipt")
        selected_receipt = existing
        reused = True
    else:
        _atomic_json_new(receipt_path, receipt)
        selected_receipt = receipt
        reused = False
    current_path = root / "QWEN30_HQ30GR2_SPARSE_GATE_UP_COMPONENT_PREPARATION_CURRENT.json"
    current = seal(
        {
            "schema": CURRENT_SCHEMA,
            "status": CURRENT_STATUS,
            "recorded_at": _utc_now(),
            "component_preparation_receipt": {
                "path": str(receipt_path.resolve()),
                "seal_sha256": selected_receipt.get("seal_sha256"),
            },
            "claim_boundary": selected_receipt.get("claim_boundary"),
        }
    )
    _atomic_json_replace(current_path, current)
    return {
        "status": selected_receipt.get("status"),
        "receipt_path": str(receipt_path),
        "receipt_seal_sha256": selected_receipt.get("seal_sha256"),
        "current_path": str(current_path),
        "current_seal_sha256": current.get("seal_sha256"),
        "reused": reused,
        "cpu_oracle_command_without_output_dir": selected_receipt["cpu_oracle_invocation"]["command_without_output_dir"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--admission-current", type=Path, default=DEFAULT_ADMISSION_CURRENT)
    parser.add_argument("--route-current", type=Path, default=DEFAULT_ROUTE_CURRENT)
    parser.add_argument("--preparation-current", type=Path, default=DEFAULT_PREPARATION_CURRENT)
    parser.add_argument("--probe-binary", type=Path, default=DEFAULT_PROBE_BINARY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_once(
            root=args.root.expanduser().resolve(),
            admission_current_path=args.admission_current.expanduser().resolve(),
            route_current_path=args.route_current.expanduser().resolve(),
            preparation_current_path=args.preparation_current.expanduser().resolve(),
            probe_binary=args.probe_binary.expanduser().resolve(),
        )
    except ComponentPreparationError as exc:
        print(json.dumps({"status": "BLOCKED_HQ30GR2_SPARSE_GATE_UP_COMPONENT_PREPARATION_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
