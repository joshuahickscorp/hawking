"""Candidate-only CPU parity harness for Qwen30's two HQ30GR2 organs.

This is deliberately not a Qwen runtime.  It binds the separately admitted
quality-candidate current pointer, its immutable native-admission receipt, and
the preserved admitted Qwen30 direct-binary control before invoking a tiny
native CPU scalar probe.  The probe proves only this narrow adapter contract:

``HQ30GR2 == the exact admitted HQ30G1B1 control payload + sorted FP16 residual``.

It never selects the candidate as a baseline, starts Metal, touches a server,
or makes generation/capability/TPS/tournament claims.  Its receipt and mutable
current selector remain under the quality candidate root so a later scalar
adapter implementation has concrete, source-bound parity evidence without
getting an implicit promotion path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lab.operators import ascension_qwen_complete_binary_admission as shared
from lab.operators.ascension_qwen30_quality_repack import (
    ARTIFACT_PREFIX,
    BRANCH_ID,
    SOURCE_SNAPSHOT_SCHEMA,
)
from lab.receipts import seal


REPO_ROOT = Path(__file__).resolve().parents[2]
QUALITY_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/gate-up-residual-v1"
)
BASELINE_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity"

CURRENT_POINTER_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_current_pointer.v1"
ADMISSION_RECEIPT_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_receipt.v1"
ADMISSION_RECEIPT_STATUS = "EARNED_QUALITY_REPACK_COMPLETE_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
MANIFEST_SCHEMA = "hawking.ascension.qwen30_quality_repack_candidate.v1"
MANIFEST_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
BASELINE_MANIFEST_SCHEMA = "hawking.ascension.qwen30_complete_binary_gravity.v1"
BASELINE_ADMISSION_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
BASELINE_ADMISSION_STATUS = "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
RESULT_SCHEMA = "hawking.ascension.qwen30_quality_repack_scalar_parity_result.v1"
RESULT_STATUS = "EARNED_HQ30GR2_CPU_SCALAR_COMPATIBILITY_PARITY_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
RECEIPT_SCHEMA = "hawking.ascension.qwen30_quality_repack_scalar_parity_receipt.v1"
CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_scalar_parity_current_pointer.v1"
# Receipts are produced only through tools/condense/.venv/bin/python (3.12)
# in production.  Preserve prior interpreter-produced evidence as historical
# support, but make the active receipt family unambiguous and re-runnable under
# the campaign's canonical sealing interpreter.
HARNESS_VERSION = "v3-production-py312-source-and-selection-bound"
SELECTED_ORGANS = (
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
)


class ScalarParityError(RuntimeError):
    """No scalar-adapter evidence may be published for mixed authorities."""


NativeRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ScalarParityTarget:
    root: Path
    baseline_root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / f"{ARTIFACT_PREFIX}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"

    @property
    def admission_current_path(self) -> Path:
        return self.root / f"{ARTIFACT_PREFIX}_NATIVE_ADMISSION_CURRENT.json"

    @property
    def admission_receipts_root(self) -> Path:
        return self.root / "complete-admission" / "receipts"

    @property
    def baseline_manifest_path(self) -> Path:
        return self.baseline_root / "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"

    @property
    def baseline_admission_path(self) -> Path:
        return self.baseline_root / "QWEN30_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"

    @property
    def receipts_root(self) -> Path:
        return self.root / "cpu-scalar-parity" / "receipts"

    @property
    def current_path(self) -> Path:
        return self.root / f"{ARTIFACT_PREFIX}_CPU_SCALAR_PARITY_CURRENT.json"


DEFAULT_TARGET = ScalarParityTarget(root=QUALITY_ROOT, baseline_root=BASELINE_ROOT)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fail(message: str) -> ScalarParityError:
    return ScalarParityError(message)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail(f"{label} must be an array")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise _fail(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _fail(f"{label} must be an integer")
    if positive and value <= 0:
        raise _fail(f"{label} must be positive")
    return value


def _require_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise _fail(f"{label} must be a finite number")
    return float(value)


def _same_path(value: object, expected: Path, label: str) -> None:
    observed = Path(_require_string(value, label))
    if not observed.is_absolute():
        raise _fail(f"{label} must be absolute")
    try:
        if observed.resolve(strict=True) != expected.resolve(strict=True):
            raise _fail(f"{label} does not bind {expected}")
    except OSError as exc:
        raise _fail(f"cannot resolve {label}: {exc}") from exc


def _read_sealed(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return shared._read_document(path, label, sealed=True)
    except shared.CompleteBinaryAdmissionError as exc:
        raise _fail(str(exc)) from exc


def _file_binding(path: Path, document: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "document_sha256": str(metadata["document_sha256"]),
        "seal_sha256": _require_sha256(document.get("seal_sha256"), f"{path.name} seal"),
        "file_identity": dict(_require_mapping(metadata["file_identity"], f"{path.name} identity")),
    }


def _verify_binding(binding: object, expected_path: Path, document: Mapping[str, Any], metadata: Mapping[str, Any], label: str) -> None:
    row = _require_mapping(binding, label)
    _same_path(row.get("path"), expected_path, f"{label}.path")
    if _require_sha256(row.get("document_sha256"), f"{label}.document_sha256") != metadata["document_sha256"]:
        raise _fail(f"{label} raw document SHA-256 differs")
    if _require_sha256(row.get("seal_sha256"), f"{label}.seal_sha256") != document["seal_sha256"]:
        raise _fail(f"{label} seal differs")
    if "file_identity" in row and row.get("file_identity") != metadata["file_identity"]:
        raise _fail(f"{label} file identity differs")


def _row_by_name(rows: Sequence[object], name: str, label: str) -> Mapping[str, Any]:
    matches = [
        _require_mapping(row, f"{label} tensor row")
        for row in rows
        if isinstance(row, Mapping) and row.get("tensor_name") == name
    ]
    if len(matches) != 1:
        raise _fail(f"{label} must contain exactly one {name}")
    return matches[0]


def _regular_payload(path_value: object, expected_root: Path, expected_sha256: str, expected_bytes: int, label: str) -> Path:
    path = Path(_require_string(path_value, f"{label}.artifact_path"))
    if not path.is_absolute():
        raise _fail(f"{label}.artifact_path must be absolute")
    try:
        original = os.lstat(path)
    except OSError as exc:
        raise _fail(f"cannot stat {label}.artifact_path: {exc}") from exc
    if os.path.islink(path) or not os.path.isfile(path):
        raise _fail(f"{label}.artifact_path must be a non-symlink regular file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"cannot resolve {label}.artifact_path: {exc}") from exc
    if resolved.parent != expected_root.resolve():
        raise _fail(f"{label}.artifact_path leaves its expected tensor root")
    if original.st_size != expected_bytes:
        raise _fail(f"{label}.artifact bytes differ from sealed row")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise _fail(f"{label}.artifact SHA-256 differs from sealed row")
    return resolved


def _validate_bindings(target: ScalarParityTarget) -> dict[str, Any]:
    root = target.root.resolve()
    manifest, manifest_meta = _read_sealed(target.manifest_path, "quality candidate manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != MANIFEST_STATUS:
        raise _fail("quality candidate manifest has an unsupported schema/status")
    current, current_meta = _read_sealed(target.admission_current_path, "quality admission current pointer")
    if (
        current.get("schema") != CURRENT_POINTER_SCHEMA
        or current.get("status") != "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED"
        or current.get("candidate_root") != str(root)
    ):
        raise _fail("quality admission current pointer is not the selected isolated candidate")
    _verify_binding(current.get("complete_manifest"), target.manifest_path, manifest, manifest_meta, "quality current manifest")

    current_receipt = _require_mapping(current.get("admission_receipt"), "quality current admission receipt")
    receipt_path = Path(_require_string(current_receipt.get("path"), "quality current admission receipt.path"))
    if not receipt_path.is_absolute() or receipt_path.parent.resolve(strict=False) != target.admission_receipts_root.resolve():
        raise _fail("quality current receipt leaves the candidate admission receipt root")
    receipt, receipt_meta = _read_sealed(receipt_path, "quality native admission receipt")
    if receipt.get("schema") != ADMISSION_RECEIPT_SCHEMA or receipt.get("status") != ADMISSION_RECEIPT_STATUS:
        raise _fail("quality current receipt is not an earned candidate-only native admission")
    _verify_binding(current_receipt, receipt_path, receipt, receipt_meta, "quality current receipt binding")
    _verify_binding(receipt.get("complete_manifest"), target.manifest_path, manifest, manifest_meta, "quality receipt manifest")
    isolation = _require_mapping(receipt.get("isolation"), "quality admission isolation")
    if (
        isolation.get("candidate_root") != str(root)
        or isolation.get("baseline_current_pointer_untouched") is not True
        or isolation.get("runtime_server_tournament_promotion_forbidden") is not True
    ):
        raise _fail("quality admission isolation is insufficient for scalar parity")
    native = _require_mapping(receipt.get("native_loader"), "quality native admission loader")
    if (
        native.get("status") != ADMISSION_RECEIPT_STATUS
        or native.get("model") != "qwen30-quality-repack"
        or _require_int(native.get("tensor_count"), "quality native tensor_count", positive=True) != 18_867
        or native.get("selected_residual_organs") != list(SELECTED_ORGANS)
        or native.get("selected_residual_discriminators_verified") is not True
    ):
        raise _fail("quality native admission did not earn the exact two-organ catalog")
    verifier = _require_mapping(native.get("payload_verification"), "quality native payload verifier")
    if (
        verifier.get("mode") != "bounded_parallel_source_shard_lanes_ordered_reconciliation_v1"
        or _require_int(verifier.get("workers_used"), "quality native verifier workers", positive=True) > 4
        or _require_int(verifier.get("manifest_rows"), "quality native verifier rows", positive=True) != 18_867
        or verifier.get("candidate_only_read_path") is not True
    ):
        raise _fail("quality native admission payload verification evidence differs")

    baseline_manifest, baseline_manifest_meta = _read_sealed(target.baseline_manifest_path, "admitted control manifest")
    baseline_admission, baseline_admission_meta = _read_sealed(target.baseline_admission_path, "admitted control admission")
    if baseline_manifest.get("schema") != BASELINE_MANIFEST_SCHEMA:
        raise _fail("admitted control manifest schema differs")
    if baseline_admission.get("schema") != BASELINE_ADMISSION_SCHEMA or baseline_admission.get("status") != BASELINE_ADMISSION_STATUS:
        raise _fail("admitted control admission is not an artifact-only admitted control")
    _verify_binding(
        baseline_admission.get("complete_manifest"),
        target.baseline_manifest_path,
        baseline_manifest,
        baseline_manifest_meta,
        "admitted control admission manifest",
    )
    branch = _require_mapping(manifest.get("quality_repack_branch"), "quality candidate branch")
    if branch.get("branch_id") != BRANCH_ID or branch.get("changed_organs") != list(SELECTED_ORGANS):
        raise _fail("quality candidate branch does not declare exactly the sealed gate/up organs")
    baseline_control = _require_mapping(branch.get("baseline_rollback_control"), "quality candidate baseline rollback")
    _verify_binding(
        baseline_control.get("manifest"),
        target.baseline_manifest_path,
        baseline_manifest,
        baseline_manifest_meta,
        "quality candidate baseline manifest binding",
    )
    if baseline_control.get("preserve_as_rollback_control") is not True:
        raise _fail("quality candidate did not preserve its admitted control rollback")
    snapshot_binding = _require_mapping(receipt.get("source_binding_snapshot"), "quality admission source snapshot")
    snapshot_path = Path(_require_string(snapshot_binding.get("path"), "quality admission source snapshot path"))
    if not snapshot_path.is_absolute() or snapshot_path.parent.resolve(strict=False) != root:
        raise _fail("quality admission source snapshot leaves the candidate root")
    snapshot, snapshot_meta = _read_sealed(snapshot_path, "quality immutable source binding snapshot")
    if snapshot.get("schema") != SOURCE_SNAPSHOT_SCHEMA or snapshot.get("status") != "EARNED_IMMUTABLE_SOURCE_AND_ROLLBACK_BINDING":
        raise _fail("quality admission source snapshot is not an earned immutable binding")
    _verify_binding(snapshot_binding, snapshot_path, snapshot, snapshot_meta, "quality admission source snapshot binding")
    _verify_binding(
        branch.get("source_binding_snapshot"), snapshot_path, snapshot, snapshot_meta,
        "quality candidate manifest source snapshot binding",
    )
    source_binding = _require_mapping(snapshot.get("binding"), "quality immutable source snapshot binding")
    if source_binding.get("branch_id") != BRANCH_ID or source_binding.get("selected_organs") != list(SELECTED_ORGANS):
        raise _fail("quality immutable source snapshot branch/organ binding differs")
    source_revalidation = _require_mapping(source_binding.get("immutable_source_revalidation"), "quality immutable source revalidation")
    _same_path(
        source_revalidation.get("path"),
        target.baseline_root / "QWEN30_CURRENT_SOURCE_SHARD_REVALIDATION.json",
        "quality immutable source revalidation path",
    )
    if not _require_sha256(source_revalidation.get("seal_sha256"), "quality immutable source revalidation seal"):
        raise _fail("quality immutable source revalidation seal is absent")

    candidate_rows = _require_list(manifest.get("tensors"), "quality candidate tensors")
    baseline_rows = _require_list(baseline_manifest.get("tensors"), "admitted control tensors")
    pair_arguments: list[str] = []
    pair_bindings: list[dict[str, Any]] = []
    for short_name, organ in zip(("gate", "up"), SELECTED_ORGANS, strict=True):
        candidate = _row_by_name(candidate_rows, organ, "quality candidate")
        control = _row_by_name(baseline_rows, organ, "admitted control")
        mutation = _require_mapping(candidate.get("candidate_mutation"), f"quality candidate {organ} mutation")
        layout = _require_mapping(candidate.get("layout"), f"quality candidate {organ} layout")
        if mutation.get("changed_from_admitted_control") is not True or layout.get("magic") != "HQ30GR2\x00":
            raise _fail(f"quality candidate {organ} is not the explicit HQ30GR2 mutation")
        rollback = _require_mapping(mutation.get("baseline_rollback"), f"quality candidate {organ} rollback")
        if rollback.get("rollback_action") != "use the separately admitted baseline tensor; this candidate never overwrites it":
            raise _fail(f"quality candidate {organ} lacks exact baseline rollback refusal")
        if rollback.get("baseline_artifact_sha256") != control.get("artifact_sha256") or rollback.get("baseline_artifact_bytes") != control.get("artifact_bytes"):
            raise _fail(f"quality candidate {organ} rollback does not bind the admitted scalar control row")
        _same_path(rollback.get("baseline_artifact_path"), Path(_require_string(control.get("artifact_path"), f"control {organ} artifact path")), f"quality candidate {organ} rollback artifact")
        discriminator = _require_mapping(mutation.get("source_to_packed_discriminator"), f"quality candidate {organ} discriminator")
        candidate_sha = _require_sha256(candidate.get("artifact_sha256"), f"quality candidate {organ} SHA")
        if _require_sha256(discriminator.get("payload_sha256"), f"quality candidate {organ} discriminator SHA") != candidate_sha:
            raise _fail(f"quality candidate {organ} discriminator payload differs")
        candidate_path = _regular_payload(
            candidate.get("artifact_path"), root / "tensors", candidate_sha,
            _require_int(candidate.get("artifact_bytes"), f"quality candidate {organ} bytes", positive=True), f"quality candidate {organ}",
        )
        control_sha = _require_sha256(control.get("artifact_sha256"), f"admitted control {organ} SHA")
        control_path = _regular_payload(
            control.get("artifact_path"), target.baseline_root.resolve() / "tensors", control_sha,
            _require_int(control.get("artifact_bytes"), f"admitted control {organ} bytes", positive=True), f"admitted control {organ}",
        )
        pair_arguments.extend(
            [
                f"--candidate-{short_name}", str(candidate_path),
                f"--candidate-{short_name}-sha256", candidate_sha,
                f"--control-{short_name}", str(control_path),
                f"--control-{short_name}-sha256", control_sha,
            ]
        )
        pair_bindings.append(
            {
                "organ": organ,
                "candidate": {"path": str(candidate_path), "sha256": candidate_sha, "bytes": candidate["artifact_bytes"]},
                "admitted_scalar_control": {"path": str(control_path), "sha256": control_sha, "bytes": control["artifact_bytes"]},
                "residual_count": _require_int(
                    _require_mapping(layout.get("residual"), f"quality candidate {organ} residual").get("selected_count"),
                    f"quality candidate {organ} residual count", positive=True,
                ),
            }
        )
    return {
        "candidate_current_pointer": _file_binding(target.admission_current_path, current, current_meta),
        "candidate_native_admission_receipt": _file_binding(receipt_path, receipt, receipt_meta),
        "candidate_manifest": _file_binding(target.manifest_path, manifest, manifest_meta),
        "candidate_source_binding_snapshot": _file_binding(snapshot_path, snapshot, snapshot_meta),
        "immutable_source_revalidation": dict(source_revalidation),
        "admitted_control_manifest": _file_binding(target.baseline_manifest_path, baseline_manifest, baseline_manifest_meta),
        "admitted_control_admission_receipt": _file_binding(target.baseline_admission_path, baseline_admission, baseline_admission_meta),
        "pair_arguments": pair_arguments,
        "pair_bindings": pair_bindings,
    }


def _native_digest(path: Path) -> str:
    try:
        return shared._native_loader_digest(path)
    except shared.CompleteBinaryAdmissionError as exc:
        raise _fail(str(exc)) from exc


def _default_runner(command: Sequence[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=timeout_seconds)


def _invoke_native(
    *, target: ScalarParityTarget, evidence: Mapping[str, Any], native_probe: Path, timeout_seconds: float, runner: NativeRunner
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise _fail("scalar parity timeout must be positive")
    before = _native_digest(native_probe)
    command = [str(native_probe.resolve()), *list(evidence["pair_arguments"])]
    try:
        completed = runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise _fail(f"native scalar parity timed out after {timeout_seconds:g} seconds") from exc
    except OSError as exc:
        raise _fail(f"cannot execute native scalar parity: {exc}") from exc
    if _native_digest(native_probe) != before:
        raise _fail("native scalar parity executable changed while it ran")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "native scalar parity returned no detail").strip()
        raise _fail(f"native scalar parity refused candidate (exit={completed.returncode}): {detail[:1000]}")
    try:
        result = shared._parse_json((completed.stdout or "").encode("utf-8"), "native scalar parity result")
    except shared.CompleteBinaryAdmissionError as exc:
        raise _fail(str(exc)) from exc
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != RESULT_STATUS or result.get("mode") != "cpu_only_scalar_adapter_compatibility_parity_v1":
        raise _fail("native scalar parity result does not declare the strict CPU-only contract")
    pairs = _require_list(result.get("pairs"), "native scalar parity pairs")
    if len(pairs) != len(SELECTED_ORGANS):
        raise _fail("native scalar parity did not return exactly two selected organs")
    expected_pairs = list(evidence["pair_bindings"])
    for observed, expected in zip(pairs, expected_pairs, strict=True):
        row = _require_mapping(observed, "native scalar parity pair")
        if row.get("organ") != expected["organ"]:
            raise _fail("native scalar parity organ order differs from sealed candidate selection")
        for key, expected_key in (("candidate_payload", "candidate"), ("admitted_scalar_control_payload", "admitted_scalar_control")):
            payload = _require_mapping(row.get(key), f"native scalar parity {key}")
            binding = _require_mapping(expected[expected_key], f"expected scalar parity {expected_key}")
            if payload.get("path") != binding["path"] or payload.get("sha256") != binding["sha256"] or payload.get("bytes") != binding["bytes"]:
                raise _fail(f"native scalar parity {key} differs from sealed candidate/control binding")
        hq30gr2 = _require_mapping(row.get("hq30gr2"), "native scalar parity HQ30GR2")
        if hq30gr2.get("magic") != "HQ30GR2\\u0000" or _require_int(hq30gr2.get("residual_count"), "native scalar parity residual count", positive=True) != expected["residual_count"]:
            raise _fail("native scalar parity residual grammar differs from sealed candidate row")
        refusal = _require_mapping(row.get("exact_fallback_refusal"), "native scalar parity fallback refusal")
        identity = _require_mapping(row.get("scalar_identity"), "native scalar parity scalar identity")
        if (
            row.get("embedded_base_exactly_matches_admitted_control") is not True
            or refusal.get("direct_decoder_refuses_hq30gr2") is not True
            or refusal.get("hq30gr2_decoder_refuses_direct_control") is not True
            or identity.get("candidate_equals_admitted_control_plus_exact_sparse_fp16_residual") is not True
            or _require_int(identity.get("changed_scalar_count"), "native scalar parity changed scalar count", positive=True) != expected["residual_count"]
        ):
            raise _fail("native scalar parity did not prove exact residual semantics/fallback refusal")
        projection = _require_mapping(row.get("projection_parity"), "native scalar parity projection")
        if _require_number(projection.get("max_abs_candidate_minus_control_minus_residual"), "native scalar parity projection error") > 1e-10:
            raise _fail("native scalar parity projection identity exceeds its exact CPU tolerance")
    boundary = _require_mapping(result.get("claim_boundary"), "native scalar parity claim boundary")
    if (
        boundary.get("cpu_only") is not True
        or boundary.get("metal_not_opened") is not True
        or boundary.get("not_a_full_qwen_layer_decoder_generation_hcli_or_tps_result") is not True
        or boundary.get("not_a_capability_tg_agent_os_or_tournament_qualification") is not True
    ):
        raise _fail("native scalar parity result overclaims beyond its CPU-only boundary")
    return {"executable_path": str(native_probe.resolve()), "executable_sha256": before, **result}


def _receipt_path(target: ScalarParityTarget, evidence: Mapping[str, Any]) -> Path:
    manifest = _require_mapping(evidence.get("candidate_manifest"), "scalar parity candidate manifest")
    return target.receipts_root / f"{ARTIFACT_PREFIX}_CPU_SCALAR_PARITY_{HARNESS_VERSION}_{_require_sha256(manifest.get('seal_sha256'), 'scalar parity manifest seal')}.json"


def _receipt(target: ScalarParityTarget, evidence: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": RESULT_STATUS,
            "recorded_at": _utc_now(),
            "candidate_root": str(target.root.resolve()),
            "candidate_admission_current_pointer": dict(_require_mapping(evidence.get("candidate_current_pointer"), "scalar evidence current pointer")),
            "candidate_native_admission_receipt": dict(_require_mapping(evidence.get("candidate_native_admission_receipt"), "scalar evidence candidate receipt")),
            "candidate_manifest": dict(_require_mapping(evidence.get("candidate_manifest"), "scalar evidence candidate manifest")),
            "candidate_source_binding_snapshot": dict(_require_mapping(evidence.get("candidate_source_binding_snapshot"), "scalar evidence source snapshot")),
            "immutable_source_revalidation": dict(_require_mapping(evidence.get("immutable_source_revalidation"), "scalar evidence source revalidation")),
            "admitted_scalar_control": {
                "manifest": dict(_require_mapping(evidence.get("admitted_control_manifest"), "scalar evidence control manifest")),
                "admission_receipt": dict(_require_mapping(evidence.get("admitted_control_admission_receipt"), "scalar evidence control admission")),
            },
            "selected_organs": list(evidence["pair_bindings"]),
            "native_cpu_scalar_probe": dict(native),
            "isolation": {
                "candidate_root_only": True,
                "baseline_admission_current_runtime_server_and_tournament_pointers_untouched": True,
                "full_metal_execution_forbidden": True,
            },
            "claim_boundary": {
                "hq30gr2_scalar_compatibility_and_control_parity_only": True,
                "not_a_full_layer_or_model_runtime": True,
                "not_generation_hcli_capability_tps_tg_agent_os_or_tournament_qualification": True,
            },
        }
    )


def _validate_existing_receipt(target: ScalarParityTarget, evidence: Mapping[str, Any], receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, metadata = _read_sealed(receipt_path, "existing scalar parity receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != RESULT_STATUS or receipt.get("candidate_root") != str(target.root.resolve()):
        raise _fail("existing scalar parity receipt does not belong to this candidate-only harness")
    # The native-admission current pointer is deliberately mutable (it may be
    # reselected with the same immutable receipt), so reuse binds its *selected
    # immutable receipt* below rather than requiring a timestamp/identity byte
    # match on the selector file itself.
    for field, evidence_field in (
        ("candidate_native_admission_receipt", "candidate_native_admission_receipt"),
        ("candidate_manifest", "candidate_manifest"),
        ("candidate_source_binding_snapshot", "candidate_source_binding_snapshot"),
        ("immutable_source_revalidation", "immutable_source_revalidation"),
    ):
        if _require_mapping(receipt.get(field), f"existing scalar receipt {field}") != _require_mapping(evidence.get(evidence_field), f"current scalar evidence {evidence_field}"):
            raise _fail(f"existing scalar parity receipt {field} differs from current candidate authority")
    control = _require_mapping(receipt.get("admitted_scalar_control"), "existing scalar receipt admitted control")
    if _require_mapping(control.get("manifest"), "existing scalar receipt control manifest") != _require_mapping(evidence.get("admitted_control_manifest"), "current scalar evidence control manifest"):
        raise _fail("existing scalar parity receipt admitted control manifest differs")
    if _require_mapping(control.get("admission_receipt"), "existing scalar receipt control admission") != _require_mapping(evidence.get("admitted_control_admission_receipt"), "current scalar evidence control admission"):
        raise _fail("existing scalar parity receipt admitted control admission differs")
    if receipt.get("selected_organs") != evidence.get("pair_bindings"):
        raise _fail("existing scalar parity receipt selected organ binding differs")
    return receipt, metadata


def _stable_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Authority that cannot change merely because a mutable selector refreshes."""

    return {
        key: evidence[key]
        for key in (
            "candidate_native_admission_receipt",
            "candidate_manifest",
            "candidate_source_binding_snapshot",
            "immutable_source_revalidation",
            "admitted_control_manifest",
            "admitted_control_admission_receipt",
            "pair_arguments",
            "pair_bindings",
        )
    }


def _publish_current(target: ScalarParityTarget, evidence: Mapping[str, Any], receipt_path: Path, receipt: Mapping[str, Any], metadata: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Keep the selected immutable CPU receipt stable across no-op reruns."""

    expected_manifest = dict(_require_mapping(evidence.get("candidate_manifest"), "scalar evidence candidate manifest"))
    expected_isolation = {
        "candidate_root_only": True,
        "baseline_admission_current_runtime_server_and_tournament_pointers_untouched": True,
        "full_metal_execution_forbidden": True,
    }
    if target.current_path.exists():
        try:
            existing, _existing_meta = _read_sealed(target.current_path, "existing scalar parity current pointer")
            existing_receipt = _require_mapping(existing.get("scalar_parity_receipt"), "existing scalar parity current receipt")
            if (
                existing.get("schema") == CURRENT_SCHEMA
                and existing.get("status") == "CURRENT_QWEN30_QUALITY_REPACK_CPU_SCALAR_PARITY_RECEIPT_SELECTED"
                and existing.get("candidate_root") == str(target.root.resolve())
                and _require_mapping(existing.get("candidate_manifest"), "existing scalar parity current manifest") == expected_manifest
                and _require_string(existing_receipt.get("path"), "existing scalar parity current receipt path") == str(receipt_path.resolve())
                and _require_sha256(existing_receipt.get("document_sha256"), "existing scalar parity current receipt document") == metadata["document_sha256"]
                and _require_sha256(existing_receipt.get("seal_sha256"), "existing scalar parity current receipt seal") == receipt["seal_sha256"]
                and _require_mapping(existing.get("isolation"), "existing scalar parity current isolation") == expected_isolation
            ):
                return existing
        except ScalarParityError:
            # Only an exact, sealed candidate-local selection is reusable.
            pass
    pointer = seal(
        {
            "schema": CURRENT_SCHEMA,
            "status": "CURRENT_QWEN30_QUALITY_REPACK_CPU_SCALAR_PARITY_RECEIPT_SELECTED",
            "recorded_at": _utc_now(),
            "candidate_root": str(target.root.resolve()),
            "candidate_manifest": expected_manifest,
            "scalar_parity_receipt": {
                "path": str(receipt_path.resolve()),
                "document_sha256": metadata["document_sha256"],
                "seal_sha256": receipt["seal_sha256"],
                "selection_source": source,
            },
            "isolation": expected_isolation,
        }
    )
    shared._atomic_json(target.current_path, pointer)
    return pointer


def run_once(
    target: ScalarParityTarget,
    *,
    native_probe: Path,
    timeout_seconds: float = 300.0,
    runner: NativeRunner = _default_runner,
) -> dict[str, Any]:
    """Create/reuse only a candidate-root CPU scalar-parity receipt."""

    evidence = _validate_bindings(target)
    receipt_path = _receipt_path(target, evidence)
    if receipt_path.exists():
        receipt, metadata = _validate_existing_receipt(target, evidence, receipt_path)
        pointer = _publish_current(target, evidence, receipt_path, receipt, metadata, "VERSIONED_CURRENT_MANIFEST")
        return {"status": RESULT_STATUS, "receipt_path": str(receipt_path), "receipt_seal_sha256": receipt["seal_sha256"], "current_path": str(target.current_path), "current_seal_sha256": pointer["seal_sha256"], "reused": True}
    native = _invoke_native(target=target, evidence=evidence, native_probe=native_probe, timeout_seconds=timeout_seconds, runner=runner)
    # Re-read every authority after the CPU scan.  A receipt is forbidden if a
    # candidate pointer, manifest, or control binding changed meanwhile.
    if _stable_evidence(_validate_bindings(target)) != _stable_evidence(evidence):
        raise _fail("candidate/current/control authority changed during CPU scalar parity")
    receipt = _receipt(target, evidence, native)
    try:
        shared._write_immutable_json(receipt_path, receipt, "quality scalar parity receipt")
    except shared.CompleteBinaryAdmissionError as exc:
        raise _fail(str(exc)) from exc
    verified, metadata = _validate_existing_receipt(target, evidence, receipt_path)
    pointer = _publish_current(target, evidence, receipt_path, verified, metadata, "VERSIONED_NEW_CPU_SCALAR_PROBE")
    return {"status": RESULT_STATUS, "receipt_path": str(receipt_path), "receipt_seal_sha256": verified["seal_sha256"], "current_path": str(target.current_path), "current_seal_sha256": pointer["seal_sha256"], "reused": False, "native": native}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("once",), nargs="?", default="once")
    parser.add_argument("--root", type=Path, default=DEFAULT_TARGET.root)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_TARGET.baseline_root)
    parser.add_argument("--native-probe", type=Path, default=REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen30_quality_repack_scalar_parity")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    target = ScalarParityTarget(root=args.root.expanduser().resolve(), baseline_root=args.baseline_root.expanduser().resolve())
    try:
        result = run_once(target, native_probe=args.native_probe.expanduser().resolve(), timeout_seconds=args.timeout_seconds)
    except ScalarParityError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_QUALITY_REPACK_CPU_SCALAR_PARITY_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
