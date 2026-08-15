"""Prepare, but never execute, the bounded HQ30GR2 all-layer diagnostic.

The current L0/E0 captured-vector result shows a local improvement, not a
coherence repair.  This operator records the only safe next experiment:
one existing ``literal_hawking`` source-template trace plus one forced shared
continuation forward, executed in a future *typed* HQ30GR2 Metal diagnostic
runtime.  It performs no model admission, no Metal creation, no server call,
and no HCLI invocation.

The prepared contract deliberately fails closed if the current source no
longer contains a typed candidate admission snapshot/reader.  That prevents a
future experiment from routing selected HQ30GR2 tensors through a direct-only
loader, decoded BF16 shadow, or file read on a token path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_EFFECT_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_L0_E0_CHAIN_EFFECT.json"
DEFAULT_CAPTURE_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_L0_ROUTE_CAPTURE.json"
DEFAULT_ADMISSION_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_NATIVE_ADMISSION_CURRENT.json"
DEFAULT_COMPLETE_BINARY_SOURCE = REPO_ROOT / "crates/hawking-core/src/model/qwen_complete_binary.rs"
DEFAULT_RUNTIME_SOURCE = REPO_ROOT / "crates/hawking-core/src/model/qwen30_complete_runtime.rs"
DEFAULT_DIAGNOSTIC_SOURCE = REPO_ROOT / "crates/hawking-core/src/model/qwen30_quality_repack_diagnostic.rs"
DEFAULT_SPARSE_GATE_UP_SHADER = REPO_ROOT / "crates/hawking-core/shaders/qwen30_quality_repack_sparse_gate_up.metal"

SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_prepare.v1"
CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_prepare_current.v1"
STATUS = "PREPARED_CURRENT_TRACE_TYPED_HQ30GR2_ALL_LAYER_DIAGNOSTIC_NOT_RUN"
TARGET_PROBE = "literal_hawking"
TARGET_POSITION = 337
TARGET_TOKEN_COUNT = 369


class AllLayerPreparationError(RuntimeError):
    """The exact candidate/current-trace build contract cannot safely be prepared."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = verify(json.loads(path.read_text(encoding="utf-8")), label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise AllLayerPreparationError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AllLayerPreparationError(f"{label} is not an object")
    return dict(value)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AllLayerPreparationError(f"{label} must be a non-empty string")
    return value


def _current_receipt(pointer_path: Path, *, status: str, field: str, label: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    pointer = _sealed(pointer_path, label=f"current {label} pointer")
    selected = pointer.get(field)
    if not isinstance(selected, Mapping):
        raise AllLayerPreparationError(f"current {label} pointer lacks {field}")
    receipt_path = Path(_text(selected.get("path"), label=f"current {label} receipt path"))
    receipt = _sealed(receipt_path, label=f"current {label} receipt")
    if receipt.get("seal_sha256") != selected.get("seal_sha256"):
        raise AllLayerPreparationError(f"current {label} receipt seal differs from pointer")
    if receipt.get("status") != status:
        raise AllLayerPreparationError(f"current {label} receipt status is not {status}")
    return pointer, receipt_path, receipt


def _target_from_capture(receipt: Mapping[str, Any]) -> dict[str, Any]:
    rows = receipt.get("probe_summary")
    if not isinstance(rows, list):
        raise AllLayerPreparationError("route capture receipt has no probe summary")
    row = next((item for item in rows if isinstance(item, Mapping) and item.get("probe_id") == TARGET_PROBE), None)
    if not isinstance(row, Mapping):
        raise AllLayerPreparationError("route capture receipt lacks literal_hawking")
    if row.get("source_template_token_count") != TARGET_TOKEN_COUNT:
        raise AllLayerPreparationError("literal_hawking token count differs from sealed current trace")
    if row.get("l0_expert0_selected_position_count") != 1 or row.get("l0_expert0_selected_positions") != [TARGET_POSITION]:
        raise AllLayerPreparationError("literal_hawking no longer has the one sealed L0/E0 target position")
    payloads = row.get("hidden_payloads")
    if not isinstance(payloads, list):
        raise AllLayerPreparationError("literal_hawking route capture has no hidden payload catalog")
    selected = next((item for item in payloads if isinstance(item, Mapping) and item.get("relative_path") == f"hidden/{TARGET_PROBE}/{TARGET_POSITION:06d}.f32le"), None)
    if not isinstance(selected, Mapping):
        raise AllLayerPreparationError("literal_hawking selected L0/E0 hidden payload is absent")
    return {
        "probe_id": TARGET_PROBE,
        "source_template_token_count": TARGET_TOKEN_COUNT,
        "l0_e0_selected_position": TARGET_POSITION,
        "l0_e0_router_input_hidden": dict(selected),
    }


def _source_readiness(
    complete_binary_source: Path,
    runtime_source: Path,
    diagnostic_source: Path,
    sparse_gate_up_shader: Path,
) -> dict[str, Any]:
    try:
        complete_text = complete_binary_source.read_text(encoding="utf-8")
        runtime_text = runtime_source.read_text(encoding="utf-8")
        diagnostic_text = diagnostic_source.read_text(encoding="utf-8")
        shader_text = sparse_gate_up_shader.read_text(encoding="utf-8")
    except OSError as exc:
        raise AllLayerPreparationError(f"typed candidate source is unreadable: {exc}") from exc
    required_complete_tokens = (
        "pub enum Qwen30QualityRepackVerifiedTensor",
        "pub fn verified_tensor_payload(&self, tensor_name: &str)",
        "pub fn verified_typed_tensor(",
        "pub(crate) verified_payloads: BTreeMap<String, Arc<[u8]>>",
        "Qwen30QualityRepackTensorLayout::SparseResidual",
    )
    missing = [token for token in required_complete_tokens if token not in complete_text]
    if missing:
        raise AllLayerPreparationError(
            "typed HQ30GR2 admission reader is not build-ready; missing source markers: " + ", ".join(missing)
        )
    if "Qwen30QualityRepackArtifact" in runtime_text or "verified_typed_tensor" in runtime_text:
        raise AllLayerPreparationError(
            "live Qwen30 runtime source already references the candidate type; prepare-only contract refuses to infer its safety"
        )
    required_diagnostic_tokens = (
        "pub struct Qwen30QualityRepackDiagnosticCatalog",
        "pub struct Qwen30QualityRepackSparseGateUpDevicePair",
        "pub fn sparse_gate_up_dispatch(&self)",
        "pub fn encode(",
        "QWEN30_QUALITY_REPACK_SPARSE_GATE_UP_KERNEL",
    )
    missing_diagnostic = [token for token in required_diagnostic_tokens if token not in diagnostic_text]
    if missing_diagnostic:
        raise AllLayerPreparationError(
            "typed HQ30GR2 sparse gate/up host dispatch is not build-ready; missing source markers: "
            + ", ".join(missing_diagnostic)
        )
    required_shader_tokens = (
        "kernel void qwen30_quality_repack_sparse_gate_up_swiglu(",
        "#pragma clang fp contract(off)",
        "#pragma clang fp reassociate(off)",
        "qwen30_quality_repack_add_sparse_row",
    )
    missing_shader = [token for token in required_shader_tokens if token not in shader_text]
    if missing_shader or "fma(" in shader_text:
        raise AllLayerPreparationError(
            "typed HQ30GR2 sparse gate/up shader is not build-ready or uses forbidden FMA; missing="
            + ", ".join(missing_shader)
        )
    return {
        "complete_binary_source": {
            "path": str(complete_binary_source.resolve()),
            "sha256": _sha256_file(complete_binary_source),
            "typed_candidate_admission_snapshot_and_reader_present": True,
            "required_markers": list(required_complete_tokens),
        },
        "live_qwen30_runtime_source": {
            "path": str(runtime_source.resolve()),
            "sha256": _sha256_file(runtime_source),
            "candidate_type_absent_from_live_runtime": True,
            "meaning": "live direct-packed control runtime remains unmodified and cannot accidentally load HQ30GR2",
        },
        "typed_candidate_diagnostic_source": {
            "path": str(diagnostic_source.resolve()),
            "sha256": _sha256_file(diagnostic_source),
            "typed_catalog_and_sparse_gate_up_host_dispatch_build_ready": True,
            "required_markers": list(required_diagnostic_tokens),
        },
        "typed_sparse_gate_up_shader": {
            "path": str(sparse_gate_up_shader.resolve()),
            "sha256": _sha256_file(sparse_gate_up_shader),
            "kernel_build_source_present": True,
            "device_compilation_and_cpu_device_parity": "NOT_RUN_REQUIRES_EXPLICIT_FUTURE_GPU_LEASE",
            "required_markers": list(required_shader_tokens),
        },
    }


def run_once(
    *,
    root: Path,
    effect_current_path: Path,
    capture_current_path: Path,
    admission_current_path: Path,
    complete_binary_source: Path,
    runtime_source: Path,
    diagnostic_source: Path,
    sparse_gate_up_shader: Path,
) -> dict[str, Any]:
    effect_pointer, effect_path, effect = _current_receipt(
        effect_current_path,
        status="EARNED_CURRENT_CAPTURED_L0_E0_CHAIN_IMPROVEMENT_UNQUALIFIED",
        field="chain_receipt",
        label="L0/E0 local-chain effect",
    )
    capture_pointer, capture_path, capture = _current_receipt(
        capture_current_path,
        status="EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED",
        field="route_capture_receipt",
        label="L0 route capture",
    )
    admission = _sealed(admission_current_path, label="HQ30GR2 candidate native admission current pointer")
    if admission.get("status") != "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED":
        raise AllLayerPreparationError("HQ30GR2 candidate native admission is not selected")
    assessment = effect.get("assessment")
    if not isinstance(assessment, Mapping) or assessment.get("all_three_selected_positions_complete_mlp_down_output_improved_vs_source") is not True:
        raise AllLayerPreparationError("current local-chain effect does not earn all-three-position improvement")
    effect_binding = effect.get("binding")
    if not isinstance(effect_binding, Mapping):
        raise AllLayerPreparationError("current local-chain effect has no binding")
    if effect_binding.get("capture_receipt_seal_sha256") != capture.get("seal_sha256"):
        raise AllLayerPreparationError("local-chain effect does not bind the selected route capture")
    target = _target_from_capture(capture)
    sources = _source_readiness(
        complete_binary_source,
        runtime_source,
        diagnostic_source,
        sparse_gate_up_shader,
    )
    binding = {
        "local_chain_effect_current_pointer": {
            "path": str(effect_current_path.resolve()),
            "seal_sha256": effect_pointer.get("seal_sha256"),
        },
        "local_chain_effect_receipt": {
            "path": str(effect_path.resolve()),
            "seal_sha256": effect.get("seal_sha256"),
        },
        "route_capture_current_pointer": {
            "path": str(capture_current_path.resolve()),
            "seal_sha256": capture_pointer.get("seal_sha256"),
        },
        "route_capture_receipt": {
            "path": str(capture_path.resolve()),
            "seal_sha256": capture.get("seal_sha256"),
        },
        "candidate_admission_current_pointer": {
            "path": str(admission_current_path.resolve()),
            "seal_sha256": admission.get("seal_sha256"),
        },
        "candidate_manifest_seal_sha256": effect_binding.get("candidate_manifest_seal_sha256"),
        "source_readiness": sources,
    }
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at": _utc_now(),
        "binding": binding,
        "planned_bounded_input": target,
        "candidate_runtime_contract": {
            "new_runtime_type_required": "Qwen30QualityRepackNativeDiagnosticRuntime",
            "live_control_runtime_reuse_forbidden": "Qwen30CompleteNativeRuntime only accepts CompleteBinaryArtifact/direct headers",
            "candidate_admission_requirement": "admit_qwen30_quality_repack_artifact with full 18867 immutable verified payload snapshots",
            "typed_tensor_requirement": {
                "direct": "HQ30G1B1 -> GpuBinaryTensor",
                "selected_sparse_residual": "HQ30GR2 -> embedded direct GpuBinaryTensor + validated index/value buffers + exact residual header",
                "no_bf16_shadow_or_dense_weight_materialization": True,
                "no_file_read_or_sha256_on_token_path": True,
                "no_direct_reader_fallback_for_sparse_residual": True,
            },
            "metal_parity_preconditions": [
                "CPU qwen30_quality_residual_matvec_f64 parity on the exact captured literal_hawking L0/E0 input",
                "device base-plus-sparse residual gate/up parity before candidate layer execution",
                "candidate route-major gate/up/down offsets must preserve the validated scalar control order",
                "fallback_count remains zero; any missing typed dispatch refuses the diagnostic",
            ],
        },
        "planned_execution_not_run": {
            "one_existing_trace_only": TARGET_PROBE,
            "prefill": f"exact {TARGET_TOKEN_COUNT} source-template IDs through all 48 layers for baseline and candidate separately",
            "target_observation": f"record L0/E0 local chain at position {TARGET_POSITION} and final logits after the full exact prefix",
            "one_bounded_continuation": "derive baseline deterministic argmax after the exact prefix; force that same one token into both paths for one additional 48-layer forward",
            "captures": [
                "per-layer completion count",
                "candidate/base route membership after the modified L0 contribution",
                "final-logit F32LE hashes and bounded top-k deltas at prefix and forced continuation",
                "typed tensor/kernel identities and fallback_count",
            ],
            "explicitly_not_run_or_claimed": [
                "HCLI endpoint",
                "server or watcher reload",
                "chat coherence",
                "unbounded generation or sampling loop",
                "TPS, TG, capability, manager, or tournament",
            ],
        },
        "next_window_gate": {
            "requires_explicit_gpu_lease_after_qwen80_shared_expert_component_terminalizes": True,
            "must_build_and_pass_cpu_static_parity_before_any_metal_context": True,
            "one_diagnostic_process_one_lease_one_terminal_receipt_or_refusal": True,
            "no_retry_without_new_parent_authorization": True,
        },
        "current_blockers": [
            "Qwen30QualityRepackNativeDiagnosticRuntime is not implemented",
            "typed HQ30GR2 sparse gate/up device compilation and CPU/device parity are not yet run",
            "all-layer candidate validation and bounded current-trace executable do not exist",
        ],
        "claim_boundary": {
            "preparation_only": True,
            "cpu_and_static_source_validation_only": True,
            "no_candidate_artifact_admission_or_metal_runtime_execution": True,
            "no_live_server_runtime_watcher_or_adapter_change": True,
            "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": True,
        },
    }
    sealed = seal(body)
    candidate_seal = _text(effect_binding.get("candidate_manifest_seal_sha256"), label="candidate manifest seal")
    capture_seal = _text(capture.get("seal_sha256"), label="route capture seal")
    source_binding_sha = _canonical_sha256(sources)
    # The complete source-binding digest remains inside the sealed document.
    # Keep only a short deterministic suffix in the filename so the atomic
    # temporary-file prefix stays below macOS NAME_MAX.
    receipt_path = root / "all-layer-current-trace-preparation/receipts" / f"QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_PREPARATION_{candidate_seal}_{capture_seal[:16]}_{source_binding_sha[:16]}.json"
    if receipt_path.exists():
        existing = _sealed(receipt_path, label="existing all-layer preparation")
        if existing.get("binding") != sealed.get("binding") or existing.get("status") != STATUS:
            raise AllLayerPreparationError("refusing to overwrite a distinct all-layer preparation receipt")
        result = existing
        reused = True
    else:
        _atomic_json(receipt_path, sealed)
        result = sealed
        reused = False
    current_path = root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_ALL_LAYER_CURRENT_TRACE_PREPARATION_CURRENT.json"
    pointer = seal(
        {
            "schema": CURRENT_SCHEMA,
            "status": "CURRENT_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_PREPARATION_SELECTED",
            "recorded_at": _utc_now(),
            "preparation_receipt": {
                "path": str(receipt_path.resolve()),
                "seal_sha256": result.get("seal_sha256"),
            },
            "current_blockers": result.get("current_blockers"),
            "claim_boundary": result.get("claim_boundary"),
        }
    )
    _atomic_json(current_path, pointer)
    return {
        "status": result.get("status"),
        "receipt_path": str(receipt_path),
        "receipt_seal_sha256": result.get("seal_sha256"),
        "current_path": str(current_path),
        "current_seal_sha256": pointer.get("seal_sha256"),
        "reused": reused,
        "current_blockers": result.get("current_blockers"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--effect-current", type=Path, default=DEFAULT_EFFECT_CURRENT)
    parser.add_argument("--capture-current", type=Path, default=DEFAULT_CAPTURE_CURRENT)
    parser.add_argument("--admission-current", type=Path, default=DEFAULT_ADMISSION_CURRENT)
    parser.add_argument("--complete-binary-source", type=Path, default=DEFAULT_COMPLETE_BINARY_SOURCE)
    parser.add_argument("--runtime-source", type=Path, default=DEFAULT_RUNTIME_SOURCE)
    parser.add_argument("--diagnostic-source", type=Path, default=DEFAULT_DIAGNOSTIC_SOURCE)
    parser.add_argument("--sparse-gate-up-shader", type=Path, default=DEFAULT_SPARSE_GATE_UP_SHADER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_once(
            root=args.root.expanduser().resolve(),
            effect_current_path=args.effect_current.expanduser().resolve(),
            capture_current_path=args.capture_current.expanduser().resolve(),
            admission_current_path=args.admission_current.expanduser().resolve(),
            complete_binary_source=args.complete_binary_source.expanduser().resolve(),
            runtime_source=args.runtime_source.expanduser().resolve(),
            diagnostic_source=args.diagnostic_source.expanduser().resolve(),
            sparse_gate_up_shader=args.sparse_gate_up_shader.expanduser().resolve(),
        )
    except AllLayerPreparationError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_PREPARATION_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
