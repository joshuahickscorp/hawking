"""CPU-only frozen-activation cross-depth discriminator for HQ30GR2.

The isolated Qwen30 quality branch changes only the layer-0/expert-0 gate/up
pair.  This diagnostic deliberately does *not* run a candidate model.  It
loads the exact source BF16 organs and the candidate's packed payloads for
layer 0, 24, and 47; then applies a deterministic, source-bound frozen
activation panel to each pair.  The result distinguishes a local packed
operator improvement from any unearned claim about all-layer hidden-state
propagation or prompt coherence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators import ascension_qwen30_complete_gravity as complete
from lab.operators import ascension_qwen30_quality_repack as quality
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_BASELINE = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity"
    / "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
)
DEFAULT_CANDIDATE = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
DEFAULT_SELECTION = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_SELECTION_RECEIPT.json"
DEFAULT_SOURCE_SNAPSHOT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_SOURCE_BINDING_SNAPSHOT.json"
DEFAULT_ADMISSION_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_NATIVE_ADMISSION_CURRENT.json"

SCHEMA = "hawking.ascension.qwen30_quality_repack_cross_depth_swiglu.v1"
STATUS = "EARNED_FROZEN_ACTIVATION_CROSS_DEPTH_DIAGNOSTIC_UNQUALIFIED"
BRANCH_ID = "qwen30-gate-up-sparse-fp16-residual-v1"
DEPTHS = (0, 24, 47)
EXPERT = 0


class CrossDepthError(RuntimeError):
    """The source or candidate binding is insufficient for a safe diagnostic."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


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


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise CrossDepthError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise CrossDepthError(f"{label} is not an object")
    return dict(checked)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CrossDepthError(f"{label} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tensor_rows(document: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    rows = document.get("tensors")
    if not isinstance(rows, list):
        raise CrossDepthError(f"{label} has no tensor list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CrossDepthError(f"{label} tensor row is not an object")
        name = _text(row.get("tensor_name"), f"{label} tensor name")
        if name in result:
            raise CrossDepthError(f"{label} repeats tensor {name}")
        result[name] = dict(row)
    return result


def _organ_name(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{EXPERT}.{projection}_proj.weight"


def _current_admission_binding(pointer_path: Path, *, candidate_seal: str) -> dict[str, Any]:
    pointer = _sealed(pointer_path, label="candidate native admission current pointer")
    if pointer.get("status") != "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED":
        raise CrossDepthError("candidate native admission pointer has an unexpected status")
    manifest = pointer.get("complete_manifest")
    receipt = pointer.get("admission_receipt")
    if not isinstance(manifest, Mapping) or not isinstance(receipt, Mapping):
        raise CrossDepthError("candidate native admission pointer lacks manifest or receipt")
    if manifest.get("seal_sha256") != candidate_seal:
        raise CrossDepthError("candidate admission pointer does not bind the selected candidate manifest")
    receipt_path = Path(_text(receipt.get("path"), "candidate admission receipt path"))
    checked = _sealed(receipt_path, label="candidate native admission receipt")
    if checked.get("seal_sha256") != receipt.get("seal_sha256"):
        raise CrossDepthError("candidate native admission receipt seal no longer matches pointer")
    return {
        "current_pointer_path": str(pointer_path.resolve()),
        "current_pointer_seal_sha256": pointer.get("seal_sha256"),
        "admission_receipt_path": str(receipt_path.resolve()),
        "admission_receipt_seal_sha256": checked.get("seal_sha256"),
    }


def _source_revalidation(snapshot: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    binding = snapshot.get("binding")
    if not isinstance(binding, Mapping):
        raise CrossDepthError("source snapshot has no binding")
    source = binding.get("immutable_source_revalidation")
    if not isinstance(source, Mapping):
        raise CrossDepthError("source snapshot has no immutable revalidation binding")
    path = Path(_text(source.get("path"), "source revalidation path"))
    checked = _sealed(path, label="immutable source revalidation")
    if checked.get("seal_sha256") != source.get("seal_sha256"):
        raise CrossDepthError("source snapshot revalidation seal no longer matches")
    if checked.get("status") != "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED":
        raise CrossDepthError("source revalidation is not current")
    return path, checked


def _read_source_tensor(
    *, model_dir: Path, row: Mapping[str, Any], revalidation: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    tensor_name = _text(row.get("tensor_name"), "source tensor name")
    shard = _text(row.get("source_shard"), f"{tensor_name} source shard")
    expected_shard_hash = _text(row.get("source_shard_sha256"), f"{tensor_name} source shard hash")
    source_path = model_dir / shard
    shards = revalidation.get("shards")
    if not isinstance(shards, Mapping) or not isinstance(shards.get(shard), Mapping):
        raise CrossDepthError(f"source revalidation lacks shard {shard}")
    shard_record = dict(shards[shard])
    sealed_shard_hash = shard_record.get("observed_sha256") or shard_record.get("expected_sha256")
    if sealed_shard_hash != expected_shard_hash:
        raise CrossDepthError(f"source revalidation shard hash differs for {shard}")
    expected_identity = shard_record.get("file_identity")
    if complete._file_identity(source_path, label=f"source shard {shard}") != expected_identity:
        raise CrossDepthError(f"source shard identity diverged from immutable revalidation: {shard}")
    header = complete.CompleteBinaryGravity._header(source_path)
    info = header.get(tensor_name)
    if not isinstance(info, Mapping):
        raise CrossDepthError(f"source shard header lacks {tensor_name}")
    dtype = _text(info.get("dtype"), f"{tensor_name} source dtype")
    shape = info.get("shape")
    offsets = info.get("data_offsets")
    if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
        raise CrossDepthError(f"source shard metadata is invalid for {tensor_name}")
    dimensions = [int(value) for value in shape]
    if dimensions != [int(value) for value in row.get("shape", [])]:
        raise CrossDepthError(f"source tensor shape differs from candidate row for {tensor_name}")
    begin, end = (int(value) for value in offsets)
    if begin < 0 or end < begin:
        raise CrossDepthError(f"source data offsets are invalid for {tensor_name}")
    with source_path.open("rb") as handle:
        header_bytes = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_bytes + begin)
        raw = handle.read(end - begin)
    if complete._file_identity(source_path, label=f"source shard {shard}") != expected_identity:
        raise CrossDepthError(f"source shard changed while reading {tensor_name}")
    values = complete._values_from_raw(raw, dtype, dimensions)
    source_hash = hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()
    return values, {
        "tensor_name": tensor_name,
        "source_shard": shard,
        "source_shard_sha256": expected_shard_hash,
        "source_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source_value_sha256": source_hash,
        "source_dtype": dtype,
        "shape": dimensions,
    }


def _packed_tensor(row: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(_text(row.get("artifact_path"), "packed artifact path"))
    expected_sha = _text(row.get("artifact_sha256"), "packed artifact sha256")
    payload = path.read_bytes()
    observed_sha = hashlib.sha256(payload).hexdigest()
    if observed_sha != expected_sha:
        raise CrossDepthError(f"packed artifact hash diverged: {path}")
    magic = payload[:8]
    if magic == complete.MAGIC:
        shape, values = quality._unpack_binary(payload)
        codec = "HQ30G1B1"
        residual = None
    elif magic == quality.RESIDUAL_MAGIC:
        residual, values = quality._unpack_sparse_residual(payload)
        shape = tuple(int(value) for value in residual["shape"])
        codec = "HQ30GR2"
    else:
        raise CrossDepthError(f"unsupported direct-packed payload magic for {path}")
    expected_shape = tuple(int(value) for value in row.get("shape", []))
    if shape != expected_shape:
        raise CrossDepthError(f"packed artifact shape diverged from manifest for {path}")
    return values, {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": observed_sha,
        "artifact_bytes": len(payload),
        "codec": codec,
        "residual": residual,
    }


def _activation_panel(width: int, *, layer: int) -> np.ndarray:
    # A depth-qualified, frozen source-bound panel makes the diagnostic replayable
    # without claiming to be a hidden-state capture from a full model execution.
    return quality._activation_controls(
        width, label=f"{BRANCH_ID}:cross-depth-frozen-swiglu:L{layer}:E{EXPERT}:v1"
    )


def _pair_output(gate: np.ndarray, up: np.ndarray, activations: np.ndarray) -> np.ndarray:
    return quality._silu(activations @ gate.T) * (activations @ up.T)


def compare_depth(
    *,
    layer: int,
    source_gate: np.ndarray,
    source_up: np.ndarray,
    baseline_gate: np.ndarray,
    baseline_up: np.ndarray,
    candidate_gate: np.ndarray,
    candidate_up: np.ndarray,
) -> dict[str, Any]:
    """Return deterministic, operator-local metrics with no model inference claim."""

    if source_gate.shape != source_up.shape or source_gate.shape != candidate_gate.shape:
        raise CrossDepthError(f"L{layer} gate/up shape mismatch")
    activations = _activation_panel(int(source_gate.shape[1]), layer=layer)
    source = _pair_output(source_gate, source_up, activations)
    baseline = _pair_output(baseline_gate, baseline_up, activations)
    candidate = _pair_output(candidate_gate, candidate_up, activations)
    baseline_metric = quality._metric(source, baseline)
    candidate_metric = quality._metric(source, candidate)
    comparison = quality._metric(baseline, candidate)
    baseline_relative = float(baseline_metric["relative_l2"])
    candidate_relative = float(candidate_metric["relative_l2"])
    return {
        "layer": layer,
        "expert": EXPERT,
        "frozen_activation_panel": {
            "kind": "deterministic_source_bound_non_runtime_control",
            "count": int(activations.shape[0]),
            "width": int(activations.shape[1]),
            "f32le_sha256": hashlib.sha256(activations.astype("<f4").tobytes()).hexdigest(),
        },
        "source_swiglu_f32le_sha256": hashlib.sha256(source.astype("<f4").tobytes()).hexdigest(),
        "baseline_swiglu_f32le_sha256": hashlib.sha256(baseline.astype("<f4").tobytes()).hexdigest(),
        "candidate_swiglu_f32le_sha256": hashlib.sha256(candidate.astype("<f4").tobytes()).hexdigest(),
        "source_to_baseline_metrics": baseline_metric,
        "source_to_candidate_metrics": candidate_metric,
        "baseline_to_candidate_metrics": comparison,
        "candidate_relative_l2_improvement_vs_baseline": (
            (baseline_relative - candidate_relative) / baseline_relative
            if baseline_relative > 0.0
            else 0.0
        ),
    }


def _depth_binding(row: Mapping[str, Any], source: Mapping[str, Any], packed: Mapping[str, Any]) -> dict[str, Any]:
    mutation = row.get("candidate_mutation")
    return {
        "source": dict(source),
        "packed": dict(packed),
        "candidate_mutation": dict(mutation) if isinstance(mutation, Mapping) else None,
    }


def _assess(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    l0 = next(row for row in rows if row.get("layer") == 0)
    later = [row for row in rows if row.get("layer") in {24, 47}]
    l0_improvement = float(l0["candidate_relative_l2_improvement_vs_baseline"])
    later_unchanged = all(
        float(row["baseline_to_candidate_metrics"]["max_abs"]) == 0.0 for row in later
    )
    return {
        "layer0_selected_pair_frozen_panel_improvement_fraction": l0_improvement,
        "middle_and_late_candidate_payload_effect_on_frozen_panel": (
            "EXACTLY_ZERO_CONTROL_MATCH" if later_unchanged else "UNEXPECTED_NONZERO_REFUSE_PROMOTION"
        ),
        "global_coherence_causal_reach": "NOT_EARNED",
        "reason": (
            "this panel evaluates only source-to-packed gate/up operators on deterministic frozen "
            "inputs; it does not measure actual MoE route membership, full-layer hidden-state "
            "propagation, logits, or HCLI trajectories"
        ),
        "next_required_discriminator": (
            "bind actual failed-HCLI token trajectories to layer-0 top-k routes, then require "
            "cross-depth hidden-state/logit evidence before a candidate runtime is eligible"
        ),
    }


def run_once(
    *,
    root: Path,
    baseline_path: Path,
    candidate_path: Path,
    selection_path: Path,
    source_snapshot_path: Path,
    admission_current_path: Path,
) -> dict[str, Any]:
    baseline = _sealed(baseline_path, label="admitted baseline manifest")
    candidate = _sealed(candidate_path, label="isolated HQ30GR2 candidate manifest")
    selection = _sealed(selection_path, label="HQ30GR2 selection receipt")
    snapshot = _sealed(source_snapshot_path, label="immutable quality source snapshot")
    if baseline.get("seal_sha256") != "3321a99d719e70499663b7bfebe14dd6c732bfc533bb05b9277eb398e44d6357":
        raise CrossDepthError("baseline is not the preserved admitted Qwen30 control")
    if candidate.get("schema") != "hawking.ascension.qwen30_quality_repack_candidate.v1":
        raise CrossDepthError("candidate does not use the isolated HQ30GR2 manifest grammar")
    candidate_seal = _text(candidate.get("seal_sha256"), "candidate manifest seal")
    selection_binding = selection.get("binding")
    if not isinstance(selection_binding, Mapping) or selection_binding.get("selected_organs") != [
        _organ_name(0, "gate"),
        _organ_name(0, "up"),
    ]:
        raise CrossDepthError("selection receipt is not exactly the L0/E0 HQ30GR2 pair")
    source_root = candidate.get("source")
    if not isinstance(source_root, Mapping):
        raise CrossDepthError("candidate has no source binding")
    model_dir = Path(_text(source_root.get("model_dir"), "candidate model directory"))
    revalidation_path, revalidation = _source_revalidation(snapshot)
    admission = _current_admission_binding(admission_current_path, candidate_seal=candidate_seal)
    baseline_rows = _tensor_rows(baseline, label="baseline manifest")
    candidate_rows = _tensor_rows(candidate, label="candidate manifest")
    if set(baseline_rows) != set(candidate_rows):
        raise CrossDepthError("candidate and baseline catalogs differ")

    depth_results: list[dict[str, Any]] = []
    for layer in DEPTHS:
        names = {projection: _organ_name(layer, projection) for projection in ("gate", "up")}
        if any(name not in baseline_rows or name not in candidate_rows for name in names.values()):
            raise CrossDepthError(f"candidate catalog lacks L{layer}/E{EXPERT} gate/up")
        source_gate, source_gate_binding = _read_source_tensor(
            model_dir=model_dir, row=candidate_rows[names["gate"]], revalidation=revalidation
        )
        source_up, source_up_binding = _read_source_tensor(
            model_dir=model_dir, row=candidate_rows[names["up"]], revalidation=revalidation
        )
        baseline_gate, baseline_gate_binding = _packed_tensor(baseline_rows[names["gate"]])
        baseline_up, baseline_up_binding = _packed_tensor(baseline_rows[names["up"]])
        candidate_gate, candidate_gate_binding = _packed_tensor(candidate_rows[names["gate"]])
        candidate_up, candidate_up_binding = _packed_tensor(candidate_rows[names["up"]])
        result = compare_depth(
            layer=layer,
            source_gate=source_gate,
            source_up=source_up,
            baseline_gate=baseline_gate,
            baseline_up=baseline_up,
            candidate_gate=candidate_gate,
            candidate_up=candidate_up,
        )
        result["gate"] = {
            "baseline": _depth_binding(baseline_rows[names["gate"]], source_gate_binding, baseline_gate_binding),
            "candidate": _depth_binding(candidate_rows[names["gate"]], source_gate_binding, candidate_gate_binding),
        }
        result["up"] = {
            "baseline": _depth_binding(baseline_rows[names["up"]], source_up_binding, baseline_up_binding),
            "candidate": _depth_binding(candidate_rows[names["up"]], source_up_binding, candidate_up_binding),
        }
        depth_results.append(result)

    binding = {
        "baseline_manifest_path": str(baseline_path.resolve()),
        "baseline_manifest_seal_sha256": baseline.get("seal_sha256"),
        "candidate_manifest_path": str(candidate_path.resolve()),
        "candidate_manifest_seal_sha256": candidate_seal,
        "selection_receipt_path": str(selection_path.resolve()),
        "selection_receipt_seal_sha256": selection.get("seal_sha256"),
        "source_snapshot_path": str(source_snapshot_path.resolve()),
        "source_snapshot_seal_sha256": snapshot.get("seal_sha256"),
        "source_revalidation_path": str(revalidation_path.resolve()),
        "source_revalidation_seal_sha256": revalidation.get("seal_sha256"),
        "candidate_native_admission": admission,
    }
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at": _utc_now(),
        "binding": binding,
        "depth_results": depth_results,
        "assessment": _assess(depth_results),
        "claim_boundary": {
            "cpu_only": True,
            "uses_source_bf16_as_reference_oracle_only": True,
            "uses_deterministic_frozen_activations_not_captured_model_hidden_states": True,
            "does_not_execute_candidate_runtime_metal_or_server": True,
            "does_not_claim_route_membership_hidden_state_propagation_logits_generation_hcli_tps_tg_capability_or_tournament": True,
            "baseline_runtime_server_and_candidate_artifact_are_untouched": True,
        },
    }
    sealed = seal(body)
    receipt_root = root / "frozen-swiglu-cross-depth" / "receipts"
    receipt_path = receipt_root / f"QWEN30_HQ30GR2_FROZEN_SWIGLU_CROSS_DEPTH_{candidate_seal}.json"
    if receipt_path.exists():
        existing = _sealed(receipt_path, label="existing cross-depth receipt")
        if existing.get("binding") != sealed.get("binding") or existing.get("status") != STATUS:
            raise CrossDepthError("refusing to overwrite a historical cross-depth receipt")
        result = existing
        reused = True
    else:
        _atomic_json(receipt_path, sealed)
        result = sealed
        reused = False
    pointer = seal(
        {
            "schema": "hawking.ascension.qwen30_quality_repack_cross_depth_swiglu_current.v1",
            "status": "CURRENT_QWEN30_HQ30GR2_FROZEN_SWIGLU_CROSS_DEPTH_RECEIPT_SELECTED",
            "recorded_at": _utc_now(),
            "candidate_root": str(root.resolve()),
            "cross_depth_receipt": {
                "path": str(receipt_path.resolve()),
                "seal_sha256": result.get("seal_sha256"),
            },
            "claim_boundary": body["claim_boundary"],
        }
    )
    _atomic_json(root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_FROZEN_SWIGLU_CROSS_DEPTH_CURRENT.json", pointer)
    return {
        "status": result.get("status"),
        "receipt_path": str(receipt_path),
        "receipt_seal_sha256": result.get("seal_sha256"),
        "current_path": str(root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_FROZEN_SWIGLU_CROSS_DEPTH_CURRENT.json"),
        "current_seal_sha256": pointer.get("seal_sha256"),
        "reused": reused,
        "assessment": result.get("assessment"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--admission-current", type=Path, default=DEFAULT_ADMISSION_CURRENT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_once(
            root=args.root.expanduser().resolve(),
            baseline_path=args.baseline.expanduser().resolve(),
            candidate_path=args.candidate.expanduser().resolve(),
            selection_path=args.selection.expanduser().resolve(),
            source_snapshot_path=args.source_snapshot.expanduser().resolve(),
            admission_current_path=args.admission_current.expanduser().resolve(),
        )
    except CrossDepthError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_HQ30GR2_FROZEN_SWIGLU_CROSS_DEPTH_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
