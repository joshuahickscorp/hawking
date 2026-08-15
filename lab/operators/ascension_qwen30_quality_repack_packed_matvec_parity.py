"""Candidate-only direct-packed HQ30GR2 matvec parity receipt.

This target deliberately starts after the separate candidate native-admission
and CPU scalar-parity receipts.  It binds those immutable selections plus the
source snapshot, then invokes a native CPU probe which computes
``HQ30G1B1 matvec + sparse FP16 residual matvec`` without dense weight
materialization.  It cannot activate the candidate runtime or any baseline
pointer, and it explicitly documents the fresh gates still required before a
later full-model integration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lab.operators import ascension_qwen_complete_binary_admission as shared
from lab.operators.ascension_qwen30_quality_repack import ARTIFACT_PREFIX
from lab.operators.ascension_qwen30_quality_repack_scalar_parity import (
    CURRENT_SCHEMA as SCALAR_CURRENT_SCHEMA,
    RECEIPT_SCHEMA as SCALAR_RECEIPT_SCHEMA,
    RESULT_STATUS as SCALAR_RESULT_STATUS,
    ScalarParityError,
    ScalarParityTarget,
    _file_binding,
    _native_digest,
    _read_sealed,
    _require_int,
    _require_mapping,
    _require_number,
    _require_sha256,
    _require_string,
    _same_path,
    _stable_evidence,
    _validate_bindings,
    _verify_binding,
)
from lab.receipts import seal


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA = "hawking.ascension.qwen30_quality_repack_packed_matvec_parity_result.v1"
RESULT_STATUS = "EARNED_HQ30GR2_CPU_DIRECT_PACKED_MATVEC_PARITY_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
RECEIPT_SCHEMA = "hawking.ascension.qwen30_quality_repack_packed_matvec_parity_receipt.v1"
CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_packed_matvec_parity_current_pointer.v1"
HARNESS_VERSION = "v1-production-py312"


class PackedMatvecParityError(ScalarParityError):
    """The target must fail closed instead of treating HQ30GR2 as direct."""


NativeRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
DEFAULT_TARGET = ScalarParityTarget(
    root=REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/gate-up-residual-v1",
    baseline_root=REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fail(message: str) -> PackedMatvecParityError:
    return PackedMatvecParityError(message)


def _scalar_current_path(target: ScalarParityTarget) -> Path:
    return target.root / f"{ARTIFACT_PREFIX}_CPU_SCALAR_PARITY_CURRENT.json"


def _matvec_receipts_root(target: ScalarParityTarget) -> Path:
    return target.root / "cpu-packed-matvec-parity" / "receipts"


def _matvec_current_path(target: ScalarParityTarget) -> Path:
    return target.root / f"{ARTIFACT_PREFIX}_CPU_PACKED_MATVEC_PARITY_CURRENT.json"


def _validate_scalar_current(target: ScalarParityTarget, base_evidence: Mapping[str, Any]) -> dict[str, Any]:
    pointer_path = _scalar_current_path(target)
    pointer, pointer_meta = _read_sealed(pointer_path, "quality scalar parity current pointer")
    if (
        pointer.get("schema") != SCALAR_CURRENT_SCHEMA
        or pointer.get("status") != "CURRENT_QWEN30_QUALITY_REPACK_CPU_SCALAR_PARITY_RECEIPT_SELECTED"
        or pointer.get("candidate_root") != str(target.root.resolve())
    ):
        raise _fail("scalar parity current pointer is not selected for this isolated candidate")
    manifest = _require_mapping(base_evidence.get("candidate_manifest"), "packed matvec candidate manifest")
    _verify_binding(pointer.get("candidate_manifest"), target.manifest_path, {"seal_sha256": manifest["seal_sha256"]}, {"document_sha256": manifest["document_sha256"], "file_identity": manifest["file_identity"]}, "scalar parity current manifest")
    receipt_binding = _require_mapping(pointer.get("scalar_parity_receipt"), "scalar parity current receipt")
    receipt_path = Path(_require_string(receipt_binding.get("path"), "scalar parity receipt path"))
    if not receipt_path.is_absolute() or receipt_path.parent.resolve(strict=False) != (target.root / "cpu-scalar-parity" / "receipts").resolve():
        raise _fail("scalar parity receipt leaves the candidate root")
    receipt, receipt_meta = _read_sealed(receipt_path, "quality scalar parity receipt")
    if receipt.get("schema") != SCALAR_RECEIPT_SCHEMA or receipt.get("status") != SCALAR_RESULT_STATUS:
        raise _fail("scalar parity receipt did not earn the strict CPU-only scalar contract")
    _verify_binding(receipt_binding, receipt_path, receipt, receipt_meta, "scalar parity current receipt binding")
    for field, evidence_field in (
        ("candidate_native_admission_receipt", "candidate_native_admission_receipt"),
        ("candidate_manifest", "candidate_manifest"),
        ("candidate_source_binding_snapshot", "candidate_source_binding_snapshot"),
        ("immutable_source_revalidation", "immutable_source_revalidation"),
    ):
        if _require_mapping(receipt.get(field), f"scalar receipt {field}") != _require_mapping(base_evidence.get(evidence_field), f"packed matvec {evidence_field}"):
            raise _fail(f"scalar parity receipt {field} differs from current source/candidate authority")
    native = _require_mapping(receipt.get("native_cpu_scalar_probe"), "scalar parity native probe")
    boundary = _require_mapping(native.get("claim_boundary"), "scalar parity native boundary")
    if boundary.get("cpu_only") is not True or boundary.get("metal_not_opened") is not True:
        raise _fail("scalar parity receipt is not strictly CPU-only")
    return {
        "scalar_parity_current_pointer": _file_binding(pointer_path, pointer, pointer_meta),
        "scalar_parity_receipt": _file_binding(receipt_path, receipt, receipt_meta),
    }


def _validate_evidence(target: ScalarParityTarget) -> dict[str, Any]:
    try:
        base = _validate_bindings(target)
    except ScalarParityError as exc:
        raise _fail(str(exc)) from exc
    scalar = _validate_scalar_current(target, base)
    return {**base, **scalar}


def _default_runner(command: Sequence[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=timeout_seconds)


def _invoke_native(*, evidence: Mapping[str, Any], native_probe: Path, timeout_seconds: float, runner: NativeRunner) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise _fail("packed matvec parity timeout must be positive")
    before = _native_digest(native_probe)
    command = [str(native_probe.resolve()), *list(evidence["pair_arguments"])]
    try:
        completed = runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise _fail(f"packed matvec parity timed out after {timeout_seconds:g} seconds") from exc
    except OSError as exc:
        raise _fail(f"cannot execute packed matvec parity: {exc}") from exc
    if _native_digest(native_probe) != before:
        raise _fail("packed matvec parity executable changed while it ran")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "packed matvec parity returned no detail").strip()
        raise _fail(f"packed matvec parity refused candidate (exit={completed.returncode}): {detail[:1000]}")
    try:
        result = shared._parse_json((completed.stdout or "").encode("utf-8"), "packed matvec parity result")
    except shared.CompleteBinaryAdmissionError as exc:
        raise _fail(str(exc)) from exc
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != RESULT_STATUS
        or result.get("mode") != "cpu_only_direct_packed_base_plus_sparse_residual_matvec_v1"
    ):
        raise _fail("packed matvec parity did not declare the strict direct-packed CPU contract")
    expected_pairs = list(evidence["pair_bindings"])
    pairs = result.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(expected_pairs):
        raise _fail("packed matvec parity did not return exactly the sealed two-organ set")
    for observed, expected in zip(pairs, expected_pairs, strict=True):
        row = _require_mapping(observed, "packed matvec pair")
        if row.get("organ") != expected["organ"]:
            raise _fail("packed matvec organ order differs from sealed selection")
        for result_key, evidence_key in (("candidate_payload", "candidate"), ("admitted_scalar_control_payload", "admitted_scalar_control")):
            actual = _require_mapping(row.get(result_key), f"packed matvec {result_key}")
            expected_binding = _require_mapping(expected[evidence_key], f"packed matvec expected {evidence_key}")
            if actual.get("path") != expected_binding["path"] or actual.get("sha256") != expected_binding["sha256"] or actual.get("bytes") != expected_binding["bytes"]:
                raise _fail(f"packed matvec {result_key} differs from sealed binding")
        hq30gr2 = _require_mapping(row.get("hq30gr2"), "packed matvec HQ30GR2")
        if hq30gr2.get("magic") != "HQ30GR2\\u0000" or _require_int(hq30gr2.get("residual_count"), "packed matvec residual count", positive=True) != expected["residual_count"]:
            raise _fail("packed matvec HQ30GR2 grammar differs from scalar receipt")
        refusal = _require_mapping(row.get("exact_format_refusal"), "packed matvec format refusal")
        if (
            row.get("embedded_base_exactly_matches_admitted_control") is not True
            or refusal.get("direct_packed_control_operator_refuses_hq30gr2") is not True
            or refusal.get("hq30gr2_packed_operator_refuses_direct_control") is not True
        ):
            raise _fail("packed matvec did not refuse unsafe direct/residual format fallback")
        direct = _require_mapping(row.get("direct_packed_matvec"), "packed direct matvec")
        if (
            direct.get("no_dense_weight_materialization") is not True
            or _require_int(direct.get("deterministic_input_count"), "packed matvec input count", positive=True) != 8
            or _require_number(direct.get("max_abs_candidate_minus_control_minus_sparse_residual"), "packed matvec parity error") > 1e-12
        ):
            raise _fail("packed matvec did not earn exact direct-packed base-plus-residual parity")
    boundary = _require_mapping(result.get("claim_boundary"), "packed matvec boundary")
    required_true = (
        "cpu_only",
        "metal_not_opened",
        "direct_packed_matvec_operator_only",
        "not_a_full_qwen_layer_decoder_generation_hcli_or_tps_result",
        "not_a_capability_tg_agent_os_or_tournament_qualification",
        "later_candidate_full_model_integration_requires_fresh_layer_model_and_runtime_gates",
    )
    if any(boundary.get(key) is not True for key in required_true):
        raise _fail("packed matvec result overclaims its CPU-only adapter boundary")
    return {"executable_path": str(native_probe.resolve()), "executable_sha256": before, **result}


def _receipt_path(target: ScalarParityTarget, evidence: Mapping[str, Any]) -> Path:
    manifest = _require_mapping(evidence.get("candidate_manifest"), "packed matvec candidate manifest")
    return _matvec_receipts_root(target) / f"{ARTIFACT_PREFIX}_CPU_PACKED_MATVEC_PARITY_{HARNESS_VERSION}_{_require_sha256(manifest.get('seal_sha256'), 'packed matvec manifest seal')}.json"


def _receipt(target: ScalarParityTarget, evidence: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": RESULT_STATUS,
            "recorded_at": _utc_now(),
            "candidate_root": str(target.root.resolve()),
            "candidate_admission_current_pointer": dict(_require_mapping(evidence.get("candidate_current_pointer"), "packed evidence admission current")),
            "candidate_native_admission_receipt": dict(_require_mapping(evidence.get("candidate_native_admission_receipt"), "packed evidence admission receipt")),
            "candidate_manifest": dict(_require_mapping(evidence.get("candidate_manifest"), "packed evidence manifest")),
            "candidate_source_binding_snapshot": dict(_require_mapping(evidence.get("candidate_source_binding_snapshot"), "packed evidence source snapshot")),
            "immutable_source_revalidation": dict(_require_mapping(evidence.get("immutable_source_revalidation"), "packed evidence source revalidation")),
            "scalar_parity_current_pointer": dict(_require_mapping(evidence.get("scalar_parity_current_pointer"), "packed evidence scalar current")),
            "scalar_parity_receipt": dict(_require_mapping(evidence.get("scalar_parity_receipt"), "packed evidence scalar receipt")),
            "admitted_scalar_control": {
                "manifest": dict(_require_mapping(evidence.get("admitted_control_manifest"), "packed evidence control manifest")),
                "admission_receipt": dict(_require_mapping(evidence.get("admitted_control_admission_receipt"), "packed evidence control admission")),
            },
            "selected_organs": list(evidence["pair_bindings"]),
            "native_cpu_direct_packed_matvec_probe": dict(native),
            "integration_contract": {
                "candidate_scalar_adapter_must_bind_this_exact_receipt_before_using_hq30gr2": True,
                "no_direct_hq30g1b1_fallback_for_selected_organs": True,
                "candidate_manifest_current_pointer_source_snapshot_and_baseline_control_must_be_rechecked_at_integration": True,
                "required_before_full_candidate_model_claim": [
                    "fresh_complete_layer_parity_with_the_candidate_representation",
                    "fresh_full_model_native_decoder_and_autoregressive_generation",
                    "fresh_hcli_context_kv_restart_agent_os_and_clean_complete_token_tps_gates",
                    "candidate_remains_separate_from_admitted_baseline_until_all_required_gates_pass",
                ],
            },
            "isolation": {
                "candidate_root_only": True,
                "baseline_runtime_server_tournament_and_current_pointers_untouched": True,
                "metal_and_full_candidate_runtime_not_started": True,
            },
            "claim_boundary": {
                "direct_packed_matvec_adapter_compatibility_only": True,
                "not_a_full_layer_or_model_runtime": True,
                "not_generation_hcli_capability_tps_tg_agent_os_or_tournament_qualification": True,
            },
        }
    )


def _stable_matvec_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_stable_evidence(evidence),
        "scalar_parity_receipt": evidence["scalar_parity_receipt"],
    }


def _validate_existing_receipt(target: ScalarParityTarget, evidence: Mapping[str, Any], receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, metadata = _read_sealed(receipt_path, "existing packed matvec parity receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != RESULT_STATUS or receipt.get("candidate_root") != str(target.root.resolve()):
        raise _fail("existing packed matvec receipt does not belong to this candidate-only target")
    for field in (
        "candidate_native_admission_receipt",
        "candidate_manifest",
        "candidate_source_binding_snapshot",
        "immutable_source_revalidation",
        "scalar_parity_receipt",
    ):
        if _require_mapping(receipt.get(field), f"existing packed receipt {field}") != _require_mapping(evidence.get(field), f"current packed evidence {field}"):
            raise _fail(f"existing packed matvec receipt {field} differs from current authority")
    if receipt.get("selected_organs") != evidence.get("pair_bindings"):
        raise _fail("existing packed matvec selected organs differ")
    return receipt, metadata


def _publish_current(target: ScalarParityTarget, evidence: Mapping[str, Any], receipt_path: Path, receipt: Mapping[str, Any], metadata: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Avoid timestamp churn when the exact CPU-only receipt is unchanged."""

    expected_manifest = dict(_require_mapping(evidence.get("candidate_manifest"), "packed evidence manifest"))
    expected_isolation = {
        "candidate_root_only": True,
        "baseline_runtime_server_tournament_and_current_pointers_untouched": True,
        "metal_and_full_candidate_runtime_not_started": True,
    }
    current_path = _matvec_current_path(target)
    if current_path.exists():
        try:
            existing, _existing_meta = _read_sealed(current_path, "existing packed matvec parity current pointer")
            existing_receipt = _require_mapping(existing.get("packed_matvec_parity_receipt"), "existing packed matvec current receipt")
            if (
                existing.get("schema") == CURRENT_SCHEMA
                and existing.get("status") == "CURRENT_QWEN30_QUALITY_REPACK_CPU_PACKED_MATVEC_PARITY_RECEIPT_SELECTED"
                and existing.get("candidate_root") == str(target.root.resolve())
                and _require_mapping(existing.get("candidate_manifest"), "existing packed matvec current manifest") == expected_manifest
                and _require_string(existing_receipt.get("path"), "existing packed matvec current receipt path") == str(receipt_path.resolve())
                and _require_sha256(existing_receipt.get("document_sha256"), "existing packed matvec current receipt document") == metadata["document_sha256"]
                and _require_sha256(existing_receipt.get("seal_sha256"), "existing packed matvec current receipt seal") == receipt["seal_sha256"]
                and _require_mapping(existing.get("isolation"), "existing packed matvec current isolation") == expected_isolation
            ):
                return existing
        except ScalarParityError:
            # An unexpected selector is replaced only with a sealed,
            # candidate-local pointer for the same exact immutable receipt.
            pass
    pointer = seal(
        {
            "schema": CURRENT_SCHEMA,
            "status": "CURRENT_QWEN30_QUALITY_REPACK_CPU_PACKED_MATVEC_PARITY_RECEIPT_SELECTED",
            "recorded_at": _utc_now(),
            "candidate_root": str(target.root.resolve()),
            "candidate_manifest": expected_manifest,
            "packed_matvec_parity_receipt": {
                "path": str(receipt_path.resolve()),
                "document_sha256": metadata["document_sha256"],
                "seal_sha256": receipt["seal_sha256"],
                "selection_source": source,
            },
            "isolation": expected_isolation,
        }
    )
    shared._atomic_json(current_path, pointer)
    return pointer


def run_once(target: ScalarParityTarget, *, native_probe: Path, timeout_seconds: float = 300.0, runner: NativeRunner = _default_runner) -> dict[str, Any]:
    evidence = _validate_evidence(target)
    receipt_path = _receipt_path(target, evidence)
    if receipt_path.exists():
        receipt, metadata = _validate_existing_receipt(target, evidence, receipt_path)
        pointer = _publish_current(target, evidence, receipt_path, receipt, metadata, "VERSIONED_CURRENT_MANIFEST")
        return {"status": RESULT_STATUS, "receipt_path": str(receipt_path), "receipt_seal_sha256": receipt["seal_sha256"], "current_path": str(_matvec_current_path(target)), "current_seal_sha256": pointer["seal_sha256"], "reused": True}
    native = _invoke_native(evidence=evidence, native_probe=native_probe, timeout_seconds=timeout_seconds, runner=runner)
    if _stable_matvec_evidence(_validate_evidence(target)) != _stable_matvec_evidence(evidence):
        raise _fail("source/candidate/scalar authority changed during packed matvec parity")
    receipt = _receipt(target, evidence, native)
    try:
        shared._write_immutable_json(receipt_path, receipt, "quality packed matvec parity receipt")
    except shared.CompleteBinaryAdmissionError as exc:
        raise _fail(str(exc)) from exc
    verified, metadata = _validate_existing_receipt(target, evidence, receipt_path)
    pointer = _publish_current(target, evidence, receipt_path, verified, metadata, "VERSIONED_NEW_CPU_DIRECT_PACKED_MATVEC")
    return {"status": RESULT_STATUS, "receipt_path": str(receipt_path), "receipt_seal_sha256": verified["seal_sha256"], "current_path": str(_matvec_current_path(target)), "current_seal_sha256": pointer["seal_sha256"], "reused": False, "native": native}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("once",), nargs="?", default="once")
    parser.add_argument("--root", type=Path, default=DEFAULT_TARGET.root)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_TARGET.baseline_root)
    parser.add_argument("--native-probe", type=Path, default=REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen30_quality_repack_packed_matvec_parity")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    target = ScalarParityTarget(root=args.root.expanduser().resolve(), baseline_root=args.baseline_root.expanduser().resolve())
    try:
        result = run_once(target, native_probe=args.native_probe.expanduser().resolve(), timeout_seconds=args.timeout_seconds)
    except PackedMatvecParityError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_QUALITY_REPACK_CPU_PACKED_MATVEC_PARITY_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
