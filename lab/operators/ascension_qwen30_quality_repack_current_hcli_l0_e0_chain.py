"""Evaluate HQ30GR2's exact L0/E0 local chain on captured current-trace inputs.

This is deliberately narrower than a candidate-runtime experiment.  It consumes
the sealed, device-produced L0 router-input F32LE buffers from the three
current compiler traces *only at positions where expert 0 was actually
selected*.  It compares the source BF16 oracle, the admitted HQ30G1B1 control,
and the separately admitted HQ30GR2 candidate through:

    gate projection -> up projection -> SwiGLU -> unchanged down projection

The calculation is CPU-only and operator-local.  It cannot establish hidden
propagation after L0, candidate logits, HCLI coherence, or performance.  It
does establish whether the sparse gate/up correction improves the complete
selected-expert MLP contribution on all three actually-reachable inputs before
any all-layer candidate diagnostic is even prepared.
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

import numpy as np

from lab.operators import ascension_qwen30_quality_repack as quality
from lab.operators import ascension_qwen30_quality_repack_cross_depth_swiglu as cross_depth
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
DEFAULT_CAPTURE_CURRENT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_L0_ROUTE_CAPTURE.json"

SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_l0_e0_chain.v1"
CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_l0_e0_chain_current.v1"
CAPTURE_STATUS = "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED"
SUCCESS_STATUS = "EARNED_CURRENT_CAPTURED_L0_E0_CHAIN_IMPROVEMENT_UNQUALIFIED"
INSUFFICIENT_STATUS = "EARNED_CURRENT_CAPTURED_L0_E0_CHAIN_INSUFFICIENT_UNQUALIFIED"
EXPECTED_PROBES = ("literal_hawking", "json_status", "python_add")
LAYER = 0
EXPERT = 0


class CurrentL0E0ChainError(RuntimeError):
    """A bound input is missing or invalid, so the diagnostic must refuse."""


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


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        document = verify(json.loads(path.read_text(encoding="utf-8")), label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise CurrentL0E0ChainError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(document, Mapping):
        raise CurrentL0E0ChainError(f"{label} is not an object")
    return dict(document)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CurrentL0E0ChainError(f"{label} must be a non-empty string")
    return value


def _relative_under(root: Path, relative: str, *, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CurrentL0E0ChainError(f"{label} escapes the sealed capture root") from exc
    return candidate


def _metric(source: np.ndarray, reconstructed: np.ndarray) -> dict[str, float | bool]:
    try:
        return dict(quality._metric(source, reconstructed))
    except quality.QualityRepackError as exc:
        raise CurrentL0E0ChainError(str(exc)) from exc


def _name(projection: str) -> str:
    return f"model.layers.{LAYER}.mlp.experts.{EXPERT}.{projection}_proj.weight"


def _load_capture(
    *, current_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, list[dict[str, Any]]]:
    pointer = _sealed(current_path, label="current L0 route capture pointer")
    selected = pointer.get("route_capture_receipt")
    if pointer.get("status") != "CURRENT_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_CAPTURE_SELECTED":
        raise CurrentL0E0ChainError("current L0 route capture pointer is not selected")
    if not isinstance(selected, Mapping):
        raise CurrentL0E0ChainError("current L0 route capture pointer lacks receipt binding")
    receipt_path = Path(_text(selected.get("path"), label="route capture receipt path"))
    receipt = _sealed(receipt_path, label="current L0 route capture receipt")
    if receipt.get("seal_sha256") != selected.get("seal_sha256"):
        raise CurrentL0E0ChainError("current L0 route capture receipt seal differs from pointer")
    if receipt.get("status") != CAPTURE_STATUS:
        raise CurrentL0E0ChainError("current L0 route capture has an unexpected status")
    binding = receipt.get("binding")
    if not isinstance(binding, Mapping):
        raise CurrentL0E0ChainError("current L0 route capture has no binding")
    result_path = Path(_text(binding.get("capture_result_path"), label="capture result path"))
    if _sha256_file(result_path) != binding.get("capture_result_sha256"):
        raise CurrentL0E0ChainError("capture result hash differs from sealed route capture receipt")
    output_root = Path(_text(binding.get("capture_output_root"), label="capture output root"))
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurrentL0E0ChainError(f"capture result is unreadable: {exc}") from exc
    if not isinstance(result, Mapping):
        raise CurrentL0E0ChainError("capture result is not an object")
    if result.get("status") != CAPTURE_STATUS:
        raise CurrentL0E0ChainError("capture result is not the strict new diagnostic output")
    if result.get("capture_protocol_revision") != "l0-route-hidden-capture-output-parent-v2":
        raise CurrentL0E0ChainError("capture result protocol revision is not the sealed successor")
    probe_rows = result.get("probes")
    receipt_summary = receipt.get("probe_summary")
    if not isinstance(probe_rows, list) or not isinstance(receipt_summary, list):
        raise CurrentL0E0ChainError("capture result/receipt lacks probe rows")
    if tuple(row.get("probe_id") for row in probe_rows if isinstance(row, Mapping)) != EXPECTED_PROBES:
        raise CurrentL0E0ChainError("capture result probe order differs from the three protected probes")
    summary_by_probe = {
        row.get("probe_id"): row for row in receipt_summary if isinstance(row, Mapping) and isinstance(row.get("probe_id"), str)
    }
    active: list[dict[str, Any]] = []
    for probe in probe_rows:
        if not isinstance(probe, Mapping):
            raise CurrentL0E0ChainError("capture probe is malformed")
        probe_id = _text(probe.get("probe_id"), label="capture probe id")
        steps = probe.get("steps")
        summary = summary_by_probe.get(probe_id)
        if not isinstance(steps, list) or not isinstance(summary, Mapping):
            raise CurrentL0E0ChainError(f"{probe_id} lacks steps or sealed summary")
        expected_positions = summary.get("l0_expert0_selected_positions")
        if not isinstance(expected_positions, list) or not all(isinstance(item, int) for item in expected_positions):
            raise CurrentL0E0ChainError(f"{probe_id} has invalid sealed expert-0 positions")
        actual_positions: list[int] = []
        for position, step in enumerate(steps):
            if not isinstance(step, Mapping) or step.get("position") != position:
                raise CurrentL0E0ChainError(f"{probe_id} step ordering is malformed")
            routes = step.get("selected_expert_ids")
            weights = step.get("normalized_route_weights")
            hidden = step.get("router_input_hidden_f32le")
            if not isinstance(routes, list) or len(routes) != 8 or not all(isinstance(value, int) for value in routes):
                raise CurrentL0E0ChainError(f"{probe_id} step {position} route IDs are invalid")
            if not isinstance(weights, list) or len(weights) != 8 or not all(isinstance(value, (int, float)) for value in weights):
                raise CurrentL0E0ChainError(f"{probe_id} step {position} route weights are invalid")
            if EXPERT not in routes:
                continue
            actual_positions.append(position)
            if not isinstance(hidden, Mapping):
                raise CurrentL0E0ChainError(f"{probe_id} selected E0 step {position} lacks hidden payload")
            relative = _text(hidden.get("relative_path"), label=f"{probe_id} selected hidden relative path")
            hidden_path = _relative_under(output_root, relative, label=f"{probe_id} selected hidden path")
            if not hidden_path.is_file() or _sha256_file(hidden_path) != hidden.get("sha256"):
                raise CurrentL0E0ChainError(f"{probe_id} selected E0 hidden payload hash differs")
            if hidden.get("bytes") != 2048 * 4 or hidden.get("elements") != 2048:
                raise CurrentL0E0ChainError(f"{probe_id} selected E0 hidden payload geometry is invalid")
            values = np.frombuffer(hidden_path.read_bytes(), dtype="<f4")
            if values.size != 2048 or not np.isfinite(values).all():
                raise CurrentL0E0ChainError(f"{probe_id} selected E0 hidden payload is non-finite or wrong width")
            route_rank = routes.index(EXPERT)
            active.append(
                {
                    "probe_id": probe_id,
                    "position": position,
                    "input_token_id": step.get("input_token_id"),
                    "route_rank": route_rank,
                    "normalized_route_weight": float(weights[route_rank]),
                    "selected_expert_ids": list(routes),
                    "hidden_path": str(hidden_path.resolve()),
                    "hidden_sha256": hidden.get("sha256"),
                    "hidden_bytes": hidden.get("bytes"),
                    "hidden": np.ascontiguousarray(values, dtype=np.float32).copy(),
                }
            )
        if actual_positions != expected_positions or len(actual_positions) != summary.get("l0_expert0_selected_position_count"):
            raise CurrentL0E0ChainError(f"{probe_id} E0 route positions differ from sealed route receipt")
        if len(actual_positions) != 1:
            raise CurrentL0E0ChainError(
                f"{probe_id} does not have exactly one actual E0 selected position required by this bounded discriminator"
            )
    if len(active) != len(EXPECTED_PROBES):
        raise CurrentL0E0ChainError("route capture does not provide exactly one active E0 input per protected probe")
    return pointer, receipt, output_root, active


def _load_weights(
    *, baseline_path: Path, candidate_path: Path, selection_path: Path, source_snapshot_path: Path, admission_current_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    baseline = cross_depth._sealed(baseline_path, label="admitted baseline manifest")
    candidate = cross_depth._sealed(candidate_path, label="HQ30GR2 candidate manifest")
    selection = cross_depth._sealed(selection_path, label="HQ30GR2 selection receipt")
    snapshot = cross_depth._sealed(source_snapshot_path, label="quality source snapshot")
    if baseline.get("seal_sha256") != "3321a99d719e70499663b7bfebe14dd6c732bfc533bb05b9277eb398e44d6357":
        raise CurrentL0E0ChainError("baseline is not the preserved admitted Qwen30 control")
    candidate_seal = _text(candidate.get("seal_sha256"), label="candidate manifest seal")
    if candidate.get("schema") != "hawking.ascension.qwen30_quality_repack_candidate.v1":
        raise CurrentL0E0ChainError("candidate is not an isolated HQ30GR2 manifest")
    selection_binding = selection.get("binding")
    selected_organs = [_name("gate"), _name("up")]
    if not isinstance(selection_binding, Mapping) or selection_binding.get("selected_organs") != selected_organs:
        raise CurrentL0E0ChainError("selection does not bind exactly L0/E0 gate/up")
    source = candidate.get("source")
    if not isinstance(source, Mapping):
        raise CurrentL0E0ChainError("candidate lacks source binding")
    model_dir = Path(_text(source.get("model_dir"), label="candidate source model dir"))
    revalidation_path, revalidation = cross_depth._source_revalidation(snapshot)
    admission = cross_depth._current_admission_binding(admission_current_path, candidate_seal=candidate_seal)
    baseline_rows = cross_depth._tensor_rows(baseline, label="baseline manifest")
    candidate_rows = cross_depth._tensor_rows(candidate, label="candidate manifest")
    weights: dict[str, np.ndarray] = {}
    tensor_binding: dict[str, Any] = {}
    for projection in ("gate", "up", "down"):
        name = _name(projection)
        if name not in baseline_rows or name not in candidate_rows:
            raise CurrentL0E0ChainError(f"candidate/baseline lacks {name}")
        source_values, source_binding = cross_depth._read_source_tensor(
            model_dir=model_dir, row=candidate_rows[name], revalidation=revalidation
        )
        baseline_values, baseline_binding = cross_depth._packed_tensor(baseline_rows[name])
        candidate_values, candidate_binding = cross_depth._packed_tensor(candidate_rows[name])
        if source_values.shape != baseline_values.shape or source_values.shape != candidate_values.shape:
            raise CurrentL0E0ChainError(f"{name} source/control/candidate geometry differs")
        if projection == "down":
            if candidate_binding.get("artifact_sha256") != baseline_binding.get("artifact_sha256"):
                raise CurrentL0E0ChainError("HQ30GR2 candidate illegally changes L0/E0 down projection")
            if not np.array_equal(candidate_values, baseline_values):
                raise CurrentL0E0ChainError("candidate L0/E0 down payload differs from admitted control")
        weights[f"source_{projection}"] = np.ascontiguousarray(source_values, dtype=np.float32)
        weights[f"baseline_{projection}"] = np.ascontiguousarray(baseline_values, dtype=np.float32)
        weights[f"candidate_{projection}"] = np.ascontiguousarray(candidate_values, dtype=np.float32)
        tensor_binding[projection] = {
            "source": source_binding,
            "baseline": {
                "manifest_row": dict(baseline_rows[name]),
                "packed": baseline_binding,
            },
            "candidate": {
                "manifest_row": dict(candidate_rows[name]),
                "packed": candidate_binding,
            },
        }
    if weights["source_gate"].shape != (768, 2048) or weights["source_up"].shape != (768, 2048):
        raise CurrentL0E0ChainError("L0/E0 gate/up geometry differs from Qwen30 768x2048")
    if weights["source_down"].shape != (2048, 768):
        raise CurrentL0E0ChainError("L0/E0 down geometry differs from Qwen30 2048x768")
    return weights, {
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
        "tensors": tensor_binding,
    }


def _chain(hidden: np.ndarray, weights: Mapping[str, np.ndarray], *, prefix: str) -> dict[str, np.ndarray]:
    gate = np.ascontiguousarray(hidden @ weights[f"{prefix}_gate"].T, dtype=np.float32)
    up = np.ascontiguousarray(hidden @ weights[f"{prefix}_up"].T, dtype=np.float32)
    swiglu = np.ascontiguousarray(quality._silu(gate) * up, dtype=np.float32)
    down = np.ascontiguousarray(swiglu @ weights[f"{prefix}_down"].T, dtype=np.float32)
    if not all(np.isfinite(value).all() for value in (gate, up, swiglu, down)):
        raise CurrentL0E0ChainError(f"{prefix} local chain produced a non-finite value")
    return {"gate": gate, "up": up, "swiglu": swiglu, "down": down}


def _stage_metrics(source: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in ("gate", "up", "swiglu", "down"):
        control_metric = _metric(source[stage], baseline[stage])
        candidate_metric = _metric(source[stage], candidate[stage])
        control_relative = float(control_metric["relative_l2"])
        candidate_relative = float(candidate_metric["relative_l2"])
        result[stage] = {
            "source_f32le_sha256": hashlib.sha256(np.ascontiguousarray(source[stage], dtype="<f4").tobytes()).hexdigest(),
            "baseline_f32le_sha256": hashlib.sha256(np.ascontiguousarray(baseline[stage], dtype="<f4").tobytes()).hexdigest(),
            "candidate_f32le_sha256": hashlib.sha256(np.ascontiguousarray(candidate[stage], dtype="<f4").tobytes()).hexdigest(),
            "source_to_baseline_metrics": control_metric,
            "source_to_candidate_metrics": candidate_metric,
            "baseline_to_candidate_metrics": _metric(baseline[stage], candidate[stage]),
            "candidate_relative_l2_improvement_vs_baseline": (
                (control_relative - candidate_relative) / control_relative if control_relative > 0.0 else 0.0
            ),
        }
    return result


def _assessment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_down_improved = all(
        float(row["chain_errors"]["down"]["candidate_relative_l2_improvement_vs_baseline"]) > 0.0 for row in rows
    )
    status = SUCCESS_STATUS if all_down_improved else INSUFFICIENT_STATUS
    return {
        "status": status,
        "actual_route_reach": {
            "layer": LAYER,
            "expert": EXPERT,
            "selected_positions": {
                row["probe_id"]: row["position"] for row in rows
            },
            "one_actual_e0_position_per_protected_probe": True,
        },
        "all_three_selected_positions_complete_mlp_down_output_improved_vs_source": all_down_improved,
        "candidate_scope": "only L0/E0 gate/up changes; down/control routing are unchanged",
        "next_action": (
            "PREPARE_ONLY_BOUNDED_CURRENT_TRACE_ALL_LAYER_CANDIDATE_DIAGNOSTIC_REQUIRING_TYPED_HQ30GR2_RESIDUAL_RUNTIME"
            if all_down_improved
            else "REJECT_L0_E0_AS_INSUFFICIENT_FOR_THIS_CURRENT_TRACE_AND_TARGET_BROADER_SOURCE_BOUND_REPRESENTATION"
        ),
        "global_coherence_causal_reach": "NOT_EARNED",
        "reason": (
            "the measurement ends at the local selected-expert down contribution; it does not run residual add, "
            "later layers, logits, generation, endpoint transport, or a candidate model"
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
    capture_current_path: Path,
) -> dict[str, Any]:
    capture_pointer, capture_receipt, _capture_root, active = _load_capture(current_path=capture_current_path)
    weights, weight_binding = _load_weights(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        selection_path=selection_path,
        source_snapshot_path=source_snapshot_path,
        admission_current_path=admission_current_path,
    )
    capture_binding = capture_receipt.get("binding")
    if not isinstance(capture_binding, Mapping):
        raise CurrentL0E0ChainError("route capture receipt has no binding")
    selection_binding = capture_binding.get("candidate_selection")
    if not isinstance(selection_binding, Mapping) or selection_binding.get("seal_sha256") != weight_binding["selection_receipt_seal_sha256"]:
        raise CurrentL0E0ChainError("route capture selection binding differs from current candidate selection")
    if capture_binding.get("baseline_direct_packed_control", {}).get("manifest_seal_sha256") != weight_binding["baseline_manifest_seal_sha256"]:
        raise CurrentL0E0ChainError("route capture baseline differs from current admitted baseline")

    rows: list[dict[str, Any]] = []
    for entry in active:
        hidden = entry.pop("hidden")
        source = _chain(hidden, weights, prefix="source")
        baseline = _chain(hidden, weights, prefix="baseline")
        candidate = _chain(hidden, weights, prefix="candidate")
        rows.append(
            {
                "probe_id": entry["probe_id"],
                "position": entry["position"],
                "input_token_id": entry["input_token_id"],
                "route_rank": entry["route_rank"],
                "normalized_route_weight": entry["normalized_route_weight"],
                "selected_expert_ids": entry["selected_expert_ids"],
                "router_input_hidden_f32le": {
                    "path": entry["hidden_path"],
                    "sha256": entry["hidden_sha256"],
                    "bytes": entry["hidden_bytes"],
                    "elements": 2048,
                },
                "chain_errors": _stage_metrics(source, baseline, candidate),
            }
        )
    assessment = _assessment(rows)
    binding = {
        "capture_current_pointer_path": str(capture_current_path.resolve()),
        "capture_current_pointer_seal_sha256": capture_pointer.get("seal_sha256"),
        "capture_receipt_path": str(Path(_text(capture_pointer["route_capture_receipt"].get("path"), label="capture pointer receipt path")).resolve()),
        "capture_receipt_seal_sha256": capture_receipt.get("seal_sha256"),
        **weight_binding,
    }
    status = assessment["status"]
    sealed = seal(
        {
            "schema": SCHEMA,
            "status": status,
            "recorded_at": _utc_now(),
            "binding": binding,
            "actual_selected_expert_chain_results": rows,
            "assessment": assessment,
            "claim_boundary": {
                "new_diagnostic_not_historical": True,
                "cpu_only": True,
                "source_bf16_is_reference_oracle_only": True,
                "uses_exact_device_produced_baseline_l0_router_input_f32le_at_actual_e0_selected_positions": True,
                "candidate_runtime_metal_or_server_not_executed": True,
                "does_not_execute_residual_add_later_layers_logits_sampler_autoregressive_feedback_or_generation": True,
                "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": True,
                "baseline_runtime_server_watcher_and_candidate_artifact_are_untouched": True,
            },
        }
    )
    candidate_seal = weight_binding["candidate_manifest_seal_sha256"]
    capture_seal = capture_receipt["seal_sha256"]
    receipt_path = root / "current-hcli-route-effect/receipts" / f"QWEN30_HQ30GR2_CURRENT_HCLI_L0_E0_CHAIN_{candidate_seal}_{capture_seal}.json"
    if receipt_path.exists():
        existing = _sealed(receipt_path, label="existing current HCLI L0/E0 chain receipt")
        if existing.get("binding") != sealed.get("binding") or existing.get("status") != status:
            raise CurrentL0E0ChainError("refusing to overwrite a distinct current HCLI L0/E0 chain receipt")
        result = existing
        reused = True
    else:
        _atomic_json(receipt_path, sealed)
        result = sealed
        reused = False
    pointer = seal(
        {
            "schema": CURRENT_SCHEMA,
            "status": "CURRENT_QWEN30_HQ30GR2_CURRENT_HCLI_L0_E0_CHAIN_RECEIPT_SELECTED",
            "recorded_at": _utc_now(),
            "candidate_root": str(root.resolve()),
            "chain_receipt": {
                "path": str(receipt_path.resolve()),
                "seal_sha256": result.get("seal_sha256"),
            },
            "assessment": result.get("assessment"),
            "claim_boundary": result.get("claim_boundary"),
        }
    )
    current_path = root / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_CURRENT_HCLI_L0_E0_CHAIN_EFFECT.json"
    _atomic_json(current_path, pointer)
    return {
        "status": result.get("status"),
        "receipt_path": str(receipt_path),
        "receipt_seal_sha256": result.get("seal_sha256"),
        "current_path": str(current_path),
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
    parser.add_argument("--capture-current", type=Path, default=DEFAULT_CAPTURE_CURRENT)
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
            capture_current_path=args.capture_current.expanduser().resolve(),
        )
    except CurrentL0E0ChainError as exc:
        print(json.dumps({"status": "BLOCKED_QWEN30_HQ30GR2_CURRENT_HCLI_L0_E0_CHAIN_FAIL_CLOSED", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
