#!/usr/bin/env python3
"""Screen a shared vector-code family on a real stacked MoE expert bank.

The screen is a reusable Gravity discriminator for stacked expert tensors.  It
fits one BF16 product-codebook family across disjoint train rows from the
selected experts, evaluates disjoint rows, and evaluates a sealed source
activation/control when one is supplied.  It is intentionally not a model
compressor: the source matrices are materialized only for this bounded
measurement, and the receipt makes no direct-runtime, whole-NR, capability, or
TPS claim.

The useful question here is whether expert-family structure survives a
vector-code description at the extreme rates needed by the Flash 0.1 EBPW
program.  The codebook implementation is shared with the canonical lookup-PQ
owner so a new model supplies geometry and source bindings rather than another
model-specific quantization framework.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey.sharded_lookup_pq_screen import (  # noqa: E402
    encode_decode_product_quantizer,
    fit_product_quantizer,
    row_metrics,
)
from tools.odyssey.stacked_expert_lowrank_sparse_repair import (  # noqa: E402
    load_expert,
    load_source_activation_bridge,
)

SCHEMA = "hawking.odyssey.expert_vector_code_screen.v1"
DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
DEFAULT_TENSOR = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DEFAULT_BRIDGE = ROOT / "receipts/headless/FLASH_NOETIC_LAYER0_SOURCE_MOE_BRIDGE.json"
METADATA_BYTES = 256
DECODER_SUPPORT_BYTES = 65536
EXPERT_DIRECTORY_ENTRY_BYTES = 4


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(document: dict[str, Any]) -> str:
    body = dict(document)
    body.pop("seal_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _parse_ints(raw: str, *, label: str) -> list[int]:
    values: list[int] = []
    for field in str(raw).split(","):
        field = field.strip()
        if not field:
            continue
        value = int(field)
        if value <= 0:
            raise ValueError(f"{label} must contain positive integers")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{label} must not be empty")
    return values


def _bf16(values: np.ndarray) -> np.ndarray:
    """Round a float32 factor to the BF16 payload billed by the screen."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def _route_ids_from_bridge(path: Path) -> tuple[list[int], list[float]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != "hawking.flash_source_moe_bridge.v1":
        raise ValueError("source activation bridge is not a Flash source-MoE bridge")
    if document.get("status") != "PASSED":
        raise ValueError("source activation bridge is not passed")
    selection = document.get("route_selection")
    if not isinstance(selection, dict):
        raise ValueError("source activation bridge has no route selection")
    raw_ids = selection.get("expert_ids")
    raw_weights = selection.get("selected_weights")
    if not isinstance(raw_ids, list) or not isinstance(raw_weights, list):
        raise ValueError("source activation bridge route selection is malformed")
    if len(raw_ids) != len(raw_weights) or not raw_ids:
        raise ValueError("source activation bridge route selection is empty or mismatched")
    expert_ids = [int(value) for value in raw_ids]
    weights = [float(value) for value in raw_weights]
    if len(set(expert_ids)) != len(expert_ids) or not all(math.isfinite(value) for value in weights):
        raise ValueError("source activation bridge route selection is not finite and unique")
    if abs(sum(weights) - 1.0) > 1e-5:
        raise ValueError("source activation bridge route weights do not sum to one")
    return expert_ids, weights


def _split_rows(rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, disjoint train/held-out row positions."""
    rows = int(rows)
    if rows < 4:
        raise ValueError("expert matrix needs at least four rows for a split")
    train = np.arange(0, rows, 2, dtype=np.int64)
    heldout = np.arange(1, rows, 2, dtype=np.int64)
    if train.size < 2 or heldout.size < 2 or np.intersect1d(train, heldout).size:
        raise ValueError("could not form disjoint train and held-out rows")
    return train, heldout


def _vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("source output vectors have different shapes")
    ref_norm = float(np.linalg.norm(reference))
    cand_norm = float(np.linalg.norm(candidate))
    error = reference - candidate
    return {
        "elements": int(reference.size),
        "relative_l2": float(np.linalg.norm(error) / max(ref_norm, 1e-30)),
        "rmse": float(np.sqrt(np.mean(np.square(error, dtype=np.float64)))),
        "max_abs_error": float(np.max(np.abs(error), initial=0.0)),
        "cosine": (
            float(np.dot(reference, candidate) / (max(ref_norm, 1e-30) * cand_norm))
            if cand_norm > 0.0
            else None
        ),
        "finite": bool(np.isfinite(candidate).all()),
        "reference_sha256": _sha256_bytes(reference.astype("<f4", copy=False).tobytes()),
        "candidate_sha256": _sha256_bytes(candidate.astype("<f4", copy=False).tobytes()),
    }


def _complete_accounting(
    *,
    expert_count: int,
    rows: int,
    width: int,
    subdimension: int,
    cardinality: int,
) -> dict[str, Any]:
    if width % int(subdimension):
        raise ValueError("vector-code subdimension must divide the expert width")
    groups = width // int(subdimension)
    bits = int(math.log2(int(cardinality)))
    entries = int(expert_count) * int(rows) * int(width)
    code_bits = int(expert_count) * int(rows) * groups * bits
    code_bytes = (code_bits + 7) // 8
    codebook_bytes = groups * int(cardinality) * int(subdimension) * 2
    directory_bytes = int(expert_count) * EXPERT_DIRECTORY_ENTRY_BYTES
    complete_bytes = code_bytes + codebook_bytes + directory_bytes + METADATA_BYTES + DECODER_SUPPORT_BYTES
    return {
        "logical_experts": int(expert_count),
        "rows_per_expert": int(rows),
        "source_width": int(width),
        "subdimension": int(subdimension),
        "subspaces": int(groups),
        "cardinality": int(cardinality),
        "bits_per_vector_code": bits,
        "code_payload_bits": code_bits,
        "code_payload_bytes": code_bytes,
        "shared_bf16_codebook_bytes": codebook_bytes,
        "expert_directory_bytes": directory_bytes,
        "metadata_bytes": METADATA_BYTES,
        "decoder_support_bytes": DECODER_SUPPORT_BYTES,
        "complete_bytes": complete_bytes,
        "complete_scoped_ebpw": complete_bytes * 8.0 / max(entries, 1),
        "accounting_denominator_source_weights": entries,
    }


def _screen_candidate(
    weights: Sequence[np.ndarray],
    *,
    expert_ids: Sequence[int],
    route_weights: Sequence[float],
    activation: np.ndarray,
    subdimension: int,
    cardinality: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    matrices = [np.asarray(value, dtype=np.float32) for value in weights]
    if not matrices or len(matrices) != len(expert_ids) or len(expert_ids) != len(route_weights):
        raise ValueError("expert, route ID, and route weight counts must match")
    shape = matrices[0].shape
    if len(shape) != 2 or any(matrix.shape != shape for matrix in matrices):
        raise ValueError("all selected expert matrices must share one 2D shape")
    rows, width = (int(shape[0]), int(shape[1]))
    activation = np.asarray(activation, dtype=np.float32).reshape(-1)
    if activation.size != width or not np.isfinite(activation).all():
        raise ValueError("source activation must match the expert input width and be finite")
    train_ids, heldout_ids = _split_rows(rows)
    train = np.concatenate([matrix[train_ids] for matrix in matrices], axis=0)
    heldout = np.concatenate([matrix[heldout_ids] for matrix in matrices], axis=0)

    fit_started_ns = time.perf_counter_ns()
    quantizer = fit_product_quantizer(
        train,
        subdimension=int(subdimension),
        cardinality=int(cardinality),
        iterations=int(iterations),
        seed=int(seed),
    )
    fit_ns = time.perf_counter_ns() - fit_started_ns

    heldout_started_ns = time.perf_counter_ns()
    heldout_candidate = encode_decode_product_quantizer(heldout, quantizer)
    heldout_ns = time.perf_counter_ns() - heldout_started_ns
    heldout_metrics = row_metrics(heldout, heldout_candidate)

    # This is deliberately an evaluation materialization.  The receipt must
    # not confuse it with a production direct decoder or a dense-parent-free
    # runtime path.
    output_started_ns = time.perf_counter_ns()
    candidate_matrices = [encode_decode_product_quantizer(matrix, quantizer) for matrix in matrices]
    source_outputs = [matrix @ activation for matrix in matrices]
    candidate_outputs = [matrix @ activation for matrix in candidate_matrices]
    source_route = sum(float(weight) * output for weight, output in zip(route_weights, source_outputs))
    candidate_route = sum(float(weight) * output for weight, output in zip(route_weights, candidate_outputs))
    output_ns = time.perf_counter_ns() - output_started_ns
    per_expert_output = [
        {
            "expert": int(expert),
            "route_weight": float(route_weight),
            "metrics": _vector_metrics(source, candidate),
        }
        for expert, route_weight, source, candidate in zip(
            expert_ids, route_weights, source_outputs, candidate_outputs
        )
    ]
    accounting = _complete_accounting(
        expert_count=len(matrices),
        rows=rows,
        width=width,
        subdimension=int(subdimension),
        cardinality=int(cardinality),
    )
    return {
        "variant": {
            "subdimension": int(subdimension),
            "cardinality": int(cardinality),
            "iterations": int(iterations),
            "seed": int(seed),
        },
        "status": "SOURCE_OUTPUT_VECTOR_CODE_SCREENED",
        "accounting": accounting,
        "heldout_weight_metrics": heldout_metrics,
        "source_activation_output": {
            "per_expert": per_expert_output,
            "route_weighted_sum": _vector_metrics(source_route, candidate_route),
            "route_weight_sum": float(sum(float(value) for value in route_weights)),
        },
        "direct_execution": {
            "status": "NOT_ESTABLISHED",
            "screen_materialized_candidate_rows": True,
            "dense_parent_required_by_screen": True,
            "production_kernel": "NOT_VALIDATED",
        },
        "timing": {
            "fit_ns": int(fit_ns),
            "heldout_decode_ns": int(heldout_ns),
            "source_output_decode_and_matvec_ns": int(output_ns),
            "total_candidate_ns": int(time.perf_counter_ns() - started_ns),
        },
    }


def _shared_basis_accounting(
    *,
    expert_count: int,
    rows: int,
    width: int,
    rank: int,
) -> dict[str, Any]:
    basis_bytes = int(rank) * int(width) * 2
    coefficient_bytes = int(expert_count) * int(rows) * int(rank) * 2
    directory_bytes = int(expert_count) * EXPERT_DIRECTORY_ENTRY_BYTES
    complete_bytes = basis_bytes + coefficient_bytes + directory_bytes + METADATA_BYTES + DECODER_SUPPORT_BYTES
    return {
        "logical_experts": int(expert_count),
        "rows_per_expert": int(rows),
        "source_width": int(width),
        "shared_basis_rank": int(rank),
        "shared_bf16_basis_bytes": basis_bytes,
        "bf16_row_coefficient_bytes": coefficient_bytes,
        "expert_directory_bytes": directory_bytes,
        "metadata_bytes": METADATA_BYTES,
        "decoder_support_bytes": DECODER_SUPPORT_BYTES,
        "complete_bytes": complete_bytes,
        "complete_scoped_ebpw": complete_bytes * 8.0 / max(int(expert_count) * int(rows) * int(width), 1),
        "accounting_denominator_source_weights": int(expert_count) * int(rows) * int(width),
    }


def _shared_basis_candidate(
    weights: Sequence[np.ndarray],
    *,
    expert_ids: Sequence[int],
    route_weights: Sequence[float],
    activation: np.ndarray,
    rank: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Fit a shared input basis on train rows and encode every source row."""
    started_ns = time.perf_counter_ns()
    matrices = [np.asarray(value, dtype=np.float32) for value in weights]
    if not matrices or len(matrices) != len(expert_ids) or len(expert_ids) != len(route_weights):
        raise ValueError("expert, route ID, and route weight counts must match")
    shape = matrices[0].shape
    if len(shape) != 2 or any(matrix.shape != shape for matrix in matrices):
        raise ValueError("all selected expert matrices must share one 2D shape")
    rows, width = (int(shape[0]), int(shape[1]))
    rank = int(rank)
    if not 1 <= rank <= min(rows, width):
        raise ValueError("shared basis rank is outside the matrix geometry")
    activation = np.asarray(activation, dtype=np.float32).reshape(-1)
    if activation.size != width or not np.isfinite(activation).all():
        raise ValueError("source activation must match the expert input width and be finite")
    train_ids, heldout_ids = _split_rows(rows)
    train = np.concatenate([matrix[train_ids] for matrix in matrices], axis=0)

    fit_started_ns = time.perf_counter_ns()
    oversample = min(8, max(1, width - rank))
    rng = np.random.default_rng(int(seed))
    omega = rng.standard_normal((width, rank + oversample)).astype(np.float32)
    projected = train @ omega
    q, _ = np.linalg.qr(projected, mode="reduced")
    compressed = q.T @ train
    _u, _s, right = np.linalg.svd(compressed, full_matrices=False)
    basis = _bf16(right[:rank])
    coefficients = [_bf16(matrix @ basis.T) for matrix in matrices]
    fit_ns = time.perf_counter_ns() - fit_started_ns

    heldout_started_ns = time.perf_counter_ns()
    heldout_candidate = np.concatenate(
        [coefficient[heldout_ids] @ basis for coefficient in coefficients], axis=0
    )
    heldout_ns = time.perf_counter_ns() - heldout_started_ns
    heldout_reference = np.concatenate([matrix[heldout_ids] for matrix in matrices], axis=0)
    heldout_metrics = row_metrics(heldout_reference, heldout_candidate)

    # Evaluate the executable factor formula directly for the source control.
    # Full candidate matrices below are diagnostic materializations for the
    # row metrics; they are not a production dense-parent dependency.
    output_started_ns = time.perf_counter_ns()
    basis_activation = basis @ activation
    source_outputs = [matrix @ activation for matrix in matrices]
    candidate_outputs = [coefficient @ basis_activation for coefficient in coefficients]
    source_route = sum(float(weight) * output for weight, output in zip(route_weights, source_outputs))
    candidate_route = sum(float(weight) * output for weight, output in zip(route_weights, candidate_outputs))
    output_ns = time.perf_counter_ns() - output_started_ns
    per_expert_output = [
        {
            "expert": int(expert),
            "route_weight": float(route_weight),
            "metrics": _vector_metrics(source, candidate),
        }
        for expert, route_weight, source, candidate in zip(
            expert_ids, route_weights, source_outputs, candidate_outputs
        )
    ]
    accounting = _shared_basis_accounting(
        expert_count=len(matrices),
        rows=rows,
        width=width,
        rank=rank,
    )
    return {
        "variant": {
            "family": "shared_input_basis",
            "rank": rank,
            "iterations": int(iterations),
            "seed": int(seed),
        },
        "status": "SOURCE_OUTPUT_SHARED_BASIS_SCREENED",
        "accounting": accounting,
        "heldout_weight_metrics": heldout_metrics,
        "source_activation_output": {
            "per_expert": per_expert_output,
            "route_weighted_sum": _vector_metrics(source_route, candidate_route),
            "route_weight_sum": float(sum(float(value) for value in route_weights)),
        },
        "direct_execution": {
            "status": "NOT_ESTABLISHED",
            "screen_materialized_candidate_rows": False,
            "dense_parent_required_by_formula": False,
            "candidate_formula": "BF16 row coefficients @ (BF16 shared basis @ activation)",
            "production_kernel": "NOT_VALIDATED",
        },
        "timing": {
            "fit_and_encode_ns": int(fit_ns),
            "heldout_decode_ns": int(heldout_ns),
            "source_output_factor_formula_and_matvec_ns": int(output_ns),
            "total_candidate_ns": int(time.perf_counter_ns() - started_ns),
        },
    }


def screen_shared_basis_matrices(
    weights: Sequence[np.ndarray],
    *,
    expert_ids: Sequence[int],
    route_weights: Sequence[float],
    activation: np.ndarray,
    ranks: Sequence[int],
    iterations: int = 1,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Run the shared-basis family on already loaded source matrices."""
    matrices = [np.asarray(value, dtype=np.float32) for value in weights]
    if not matrices:
        raise ValueError("at least one expert matrix is required")
    return [
        _shared_basis_candidate(
            matrices,
            expert_ids=expert_ids,
            route_weights=route_weights,
            activation=activation,
            rank=int(rank),
            iterations=iterations,
            seed=seed,
        )
        for rank in sorted(set(int(value) for value in ranks))
    ]


def screen_matrices(
    weights: Sequence[np.ndarray],
    *,
    expert_ids: Sequence[int],
    route_weights: Sequence[float],
    activation: np.ndarray,
    subdimensions: Sequence[int],
    cardinalities: Sequence[int],
    iterations: int = 4,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Run the bounded vector-code fanout on already loaded matrices."""
    matrices = [np.asarray(value, dtype=np.float32) for value in weights]
    if not matrices:
        raise ValueError("at least one expert matrix is required")
    width = int(matrices[0].shape[1])
    candidates: list[dict[str, Any]] = []
    for subdimension in sorted(set(int(value) for value in subdimensions)):
        if subdimension <= 0 or width % subdimension:
            raise ValueError(f"subdimension {subdimension} does not divide width {width}")
        for cardinality in sorted(set(int(value) for value in cardinalities)):
            if cardinality <= 1 or cardinality & (cardinality - 1):
                raise ValueError("cardinalities must be powers of two greater than one")
            candidates.append(
                _screen_candidate(
                    matrices,
                    expert_ids=expert_ids,
                    route_weights=route_weights,
                    activation=activation,
                    subdimension=subdimension,
                    cardinality=cardinality,
                    iterations=iterations,
                    seed=seed,
                )
            )
    return candidates


def _load_bound_inputs(
    spec: Path,
    tensor: str,
    bridge: Path,
    expert_ids: Sequence[int] | None,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[int], list[float], np.ndarray, dict[str, Any]]:
    route_ids, route_weights = _route_ids_from_bridge(bridge)
    if expert_ids is None:
        selected_ids = route_ids
    else:
        selected_ids = [int(value) for value in expert_ids]
        if not selected_ids:
            raise ValueError("expert selection must not be empty")
        if any(value not in route_ids for value in selected_ids):
            raise ValueError("every selected expert must be in the sealed source route")
    weight_by_id = {expert: weight for expert, weight in zip(route_ids, route_weights)}
    selected_weights = [float(weight_by_id[expert]) for expert in selected_ids]
    source_activation: np.ndarray | None = None
    bridge_controls: list[dict[str, Any]] = []
    loaded: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []
    for expert in selected_ids:
        weight, source = load_expert(spec, tensor, int(expert))
        loaded.append(weight)
        source_records.append(source)
        current_activation, control = load_source_activation_bridge(
            bridge,
            expected_columns=int(weight.shape[1]),
            expert=int(expert),
        )
        if source_activation is None:
            source_activation = current_activation
        elif not np.array_equal(source_activation, current_activation):
            raise ValueError("source activation bridge changed across expert validations")
        bridge_controls.append(control)
    if source_activation is None:
        raise ValueError("no source activation was validated")
    bridge_binding = {
        "path": str(bridge.expanduser().resolve()),
        "sha256": _sha256_bytes(bridge.expanduser().resolve().read_bytes()),
        "route_expert_ids": route_ids,
        "route_weights": route_weights,
        "selected_expert_ids": selected_ids,
        "selected_route_weights": selected_weights,
        "activation_sha256": bridge_controls[0]["activation_sha256"],
        "source_layer_receipt_sha256": bridge_controls[0]["source_layer_receipt_sha256"],
        "teacher_only": True,
        "standalone_nr_dependency": False,
    }
    return loaded, source_records, list(selected_ids), selected_weights, source_activation, bridge_binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--source-activation-bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument(
        "--experts",
        default=None,
        help="comma-separated subset of the sealed route experts; default is all sealed route experts",
    )
    parser.add_argument("--subdimensions", default="32,64,128")
    parser.add_argument("--cardinalities", default="2,4,8")
    parser.add_argument(
        "--family",
        choices=("vector-code", "shared-basis", "both"),
        default="vector-code",
        help="representation family to screen; both composes the same canonical owner",
    )
    parser.add_argument("--basis-ranks", default="4,8,16,32")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        raise SystemExit("iterations must be positive")
    subdimensions = _parse_ints(args.subdimensions, label="subdimensions")
    cardinalities = _parse_ints(args.cardinalities, label="cardinalities")
    selected_experts = (
        _parse_ints(args.experts, label="experts") if args.experts is not None else None
    )
    spec = args.spec.expanduser().resolve()
    bridge = args.source_activation_bridge.expanduser().resolve()
    started = datetime.now(timezone.utc)
    started_ns = time.perf_counter_ns()
    weights, source_records, expert_ids, route_weights, activation, bridge_binding = _load_bound_inputs(
        spec, args.tensor, bridge, selected_experts
    )
    candidates: list[dict[str, Any]] = []
    if args.family in {"vector-code", "both"}:
        candidates.extend(
            screen_matrices(
                weights,
                expert_ids=expert_ids,
                route_weights=route_weights,
                activation=activation,
                subdimensions=subdimensions,
                cardinalities=cardinalities,
                iterations=args.iterations,
                seed=args.seed,
            )
        )
    if args.family in {"shared-basis", "both"}:
        candidates.extend(
            screen_shared_basis_matrices(
                weights,
                expert_ids=expert_ids,
                route_weights=route_weights,
                activation=activation,
                ranks=_parse_ints(args.basis_ranks, label="basis-ranks"),
                iterations=args.iterations,
                seed=args.seed,
            )
        )
    index_path = spec / "model.safetensors.index.json"
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": started.isoformat(),
        "status": "SOURCE_BOUND_EXPERT_VECTOR_CODE_FANOUT_COMPLETE",
        "source": {
            "specimen": str(spec),
            "index_path": str(index_path),
            "index_sha256": _sha256_bytes(index_path.read_bytes()),
            "tensor": args.tensor,
            "expert_ids": expert_ids,
            "records": source_records,
            "matrix_shape": list(weights[0].shape),
            "source_payload_bytes_read": int(sum(int(record["payload_bytes_read"]) for record in source_records)),
            "activation_control": bridge_binding,
        },
        "split": {
            "rule": "even source rows train; odd source rows held out, independently for every selected expert",
            "train_rows_per_expert": int(weights[0].shape[0] // 2 + weights[0].shape[0] % 2),
            "heldout_rows_per_expert": int(weights[0].shape[0] // 2),
            "disjoint": True,
        },
        "representation": {
            "family": (
                "shared_bf16_product_vector_code"
                if args.family == "vector-code"
                else "shared_input_basis_and_product_vector_code"
                if args.family == "both"
                else "shared_input_basis"
            ),
            "requested_family": args.family,
            "codebooks": "one shared BF16 codebook per input subspace across selected experts",
            "codes": "one packed code per expert row and input subspace",
            "screen_only": True,
            "all_complete_parts_billed": [
                "packed expert vector codes",
                "shared BF16 codebooks",
                "expert directory",
                "metadata",
                "decoder support",
            ],
        },
        "candidates": candidates,
        "claim_boundary": (
            "This is one sealed Flash source-activation/control, selected-expert-family vector-code "
            "discriminator. Codebooks are trained on disjoint source rows and held-out weights plus "
            "route-weighted source matvec outputs are measured. The screen materializes candidate "
            "rows from source matrices for evaluation; it therefore proves no production direct "
            "decoder, no dense-parent-free execution, no full-body EBPW, no PLE closure, no TPS, "
            "no capability preservation, and no promotion."
        ),
        "next_gate": (
            "Only a held-out/output survivor with a materially useful slope earns a serialized "
            "expert-family representation and a native direct decoder; otherwise close this fixed "
            "shared-vector-code family for this source control and move to a different functional "
            "expert hypothesis."
        ),
        "elapsed_ns": int(time.perf_counter_ns() - started_ns),
    }
    document["seal_sha256"] = _canonical_sha256(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": document["status"],
        "experts": expert_ids,
        "candidates": len(candidates),
        "best_density": min(
            (candidate["accounting"]["complete_scoped_ebpw"] for candidate in candidates),
            default=None,
        ),
        "best_route_output_relative_l2": min(
            (
                candidate["source_activation_output"]["route_weighted_sum"]["relative_l2"]
                for candidate in candidates
            ),
            default=None,
        ),
        "elapsed_ns": document["elapsed_ns"],
        "output": str(args.output),
        "seal": document["seal_sha256"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
