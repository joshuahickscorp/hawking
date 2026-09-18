#!/usr/bin/env python3
"""Screen lawful nonlinear atom sharing across two routed MoE experts.

This is the canonical E02 Gravity discriminator for a gated-MoE expert bank.
It shares only complete neuron triples under operations that preserve the
gated function: a common neuron permutation and reciprocal scaling between an
up row and its down column.  The screen uses exact source gate/up/down tensors,
binds one sealed source activation and route control, compares a matched atom
assignment with a byte-identical shuffled null, and bills every scoped byte.

The experiment is deliberately local.  It replaces only a deterministic set
of selected neuron triples in two source experts while leaving the other
neurons exact in the diagnostic control.  Therefore its EBPW is scoped to the
selected triples and is not a complete expert, model, NR, capability, TPS, or
promotion result.  The implementation materializes source arrays for scoring;
the receipt describes the compact gated formula but does not claim a
production direct decoder.
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
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey.expert_vector_code_screen import _route_ids_from_bridge  # noqa: E402
from tools.odyssey.flash_repeated_accepted_decode import (  # noqa: E402
    SOURCE_SHARD_LEDGER_SCHEMA,
    SOURCE_SHARD_LEDGER_STATUS,
    _canonical_model_lake_manifest,
    _verify_shard_ledger,
)
from tools.odyssey.stacked_expert_lowrank_sparse_repair import (  # noqa: E402
    load_expert,
    load_source_activation_bridge,
)

SCHEMA = "hawking.odyssey.expert_atom_sharing_screen.v1"
REPLICATION_SCHEMA = "hawking.odyssey.expert_atom_sharing_replication_bank.v1"
DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
DEFAULT_GATE_UP_TENSOR = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DEFAULT_DOWN_TENSOR = "model.language_model.layers.0.mlp.experts.down_proj"
DEFAULT_BRIDGE = ROOT / "receipts/headless/FLASH_NOETIC_LAYER0_SOURCE_MOE_BRIDGE.json"
METADATA_BYTES = 256
DECODER_SUPPORT_BYTES = 65536
EXPERT_DIRECTORY_ENTRY_BYTES = 8
NEURON_INDEX_BYTES = 2
REPLICATION_EXPERT_COUNT = 8
REPLICATION_PAIR_COUNT = 4
DEFAULT_SOURCE_SHARD_LEDGER = ROOT / "receipts/headless/FLASH_SOURCE_SHARD_LEDGER_20260912.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(document: dict[str, Any]) -> str:
    body = dict(document)
    body.pop("seal_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to the BF16 payload billed by the candidate."""
    values = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def _silu(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -80.0, 80.0)
    return np.asarray(clipped / (1.0 + np.exp(-clipped)), dtype=np.float32)


def _vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes differ")
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    error = reference_flat - candidate_flat
    ref_norm = float(np.linalg.norm(reference_flat))
    candidate_norm = float(np.linalg.norm(candidate_flat))
    return {
        "elements": int(reference_flat.size),
        "relative_l2": float(np.linalg.norm(error) / max(ref_norm, 1e-30)),
        "rmse": float(np.sqrt(np.mean(np.square(error, dtype=np.float64)))),
        "max_abs_error": float(np.max(np.abs(error), initial=0.0)),
        "cosine": (
            float(np.dot(reference_flat, candidate_flat) / (max(ref_norm, 1e-30) * candidate_norm))
            if candidate_norm > 0.0
            else None
        ),
        "finite": bool(np.isfinite(candidate_flat).all()),
        "reference_sha256": _sha256_bytes(reference_flat.astype("<f4", copy=False).tobytes()),
        "candidate_sha256": _sha256_bytes(candidate_flat.astype("<f4", copy=False).tobytes()),
    }


def _parse_ints(raw: str, *, label: str) -> list[int]:
    values: list[int] = []
    for field in str(raw).split(","):
        field = field.strip()
        if not field:
            continue
        value = int(field)
        if value < 0 or value in values:
            if value < 0:
                raise ValueError(f"{label} must contain non-negative integers")
            continue
        values.append(value)
    if not values:
        raise ValueError(f"{label} must not be empty")
    return values


def _selected_neurons(intermediate: int, count: int) -> np.ndarray:
    intermediate = int(intermediate)
    count = int(count)
    if intermediate < 2 or count < 2 or count > intermediate:
        raise ValueError("selected neuron count must be between two and the intermediate width")
    # Evenly spaced, deterministic positions; the unselected complement stays
    # exact in the diagnostic full-expert control.
    return np.floor(np.arange(count, dtype=np.float64) * intermediate / count).astype(np.int64)


def _canonicalize_triples(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply only the exact reciprocal up/down rescaling convention.

    For each neuron, `up = scale * up_unit` and `down_scaled = scale * down`,
    so `SiLU(gate*x) * (up*x) * down` is unchanged before BF16 rounding.  The
    gate row is never arbitrarily scaled because SiLU does not preserve that
    transformation.
    """
    gate = np.asarray(gate, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    down = np.asarray(down, dtype=np.float32)
    if gate.shape != up.shape or gate.ndim != 2 or down.shape != gate.shape:
        raise ValueError("gate, up, and down triple arrays must share a 2D shape")
    scales = np.linalg.norm(up, axis=1).astype(np.float32)
    safe_scales = np.where(scales > 1e-30, scales, 1.0).astype(np.float32)
    up_unit = up / safe_scales[:, None]
    down_scaled = down * scales[:, None]
    down_scaled[scales <= 1e-30] = 0.0
    return gate, up_unit, down_scaled, scales


def _triples_from_source(
    gate_up: np.ndarray,
    down: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate_up = np.asarray(gate_up, dtype=np.float32)
    down = np.asarray(down, dtype=np.float32)
    if gate_up.ndim != 2 or down.ndim != 2 or gate_up.shape[0] % 2:
        raise ValueError("source gate_up must be 2D with an even row count")
    intermediate = gate_up.shape[0] // 2
    if down.shape != (gate_up.shape[1], intermediate):
        raise ValueError(
            "source down tensor must have shape [hidden, intermediate] matching gate_up"
        )
    indices = np.asarray(indices, dtype=np.int64)
    return gate_up[:intermediate][indices], gate_up[intermediate:][indices], down[:, indices].T


def _triple_outputs(
    triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    inputs: np.ndarray,
) -> np.ndarray:
    gate, up, down = triples
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    scalar = _silu(gate @ inputs.T) * (up @ inputs.T)
    return scalar.T @ down


def _full_outputs(gate_up: np.ndarray, down: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    gate_up = np.asarray(gate_up, dtype=np.float32)
    down = np.asarray(down, dtype=np.float32)
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    intermediate = gate_up.shape[0] // 2
    gate = gate_up[:intermediate]
    up = gate_up[intermediate:]
    scalar = _silu(gate @ inputs.T) * (up @ inputs.T)
    return scalar.T @ down.T


def _build_probes(
    source_activation: np.ndarray,
    *,
    probe_count: int,
    hidden: int,
    output_probe_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if probe_count < 2 or output_probe_dim < 2:
        raise ValueError("probe_count and output_probe_dim must be at least two")
    source_activation = np.asarray(source_activation, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(int(seed))
    inputs = rng.standard_normal((source_activation.size, probe_count)).astype(np.float32)
    inputs[:, 0] = source_activation
    output_probes = rng.standard_normal((int(hidden), int(output_probe_dim))).astype(np.float32)
    output_probes /= np.sqrt(float(hidden))
    return inputs, output_probes


def _function_signature(
    triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    inputs: np.ndarray,
    output_probes: np.ndarray,
) -> np.ndarray:
    gate, up, down = triples
    gate_response = gate @ inputs
    up_response = up @ inputs
    scalar = _silu(gate_response) * up_response
    down_projection = down @ output_probes
    signature = (scalar[:, :, None] * down_projection[:, None, :]).reshape(gate.shape[0], -1)
    norms = np.linalg.norm(signature, axis=1, keepdims=True)
    return signature / np.maximum(norms, 1e-30)


def _accounting(*, expert_count: int, atom_count: int, width: int) -> dict[str, Any]:
    if expert_count != 2:
        raise ValueError("E02 accounting is intentionally bounded to two experts")
    atom_values = int(atom_count) * 3 * int(width)
    source_values = int(expert_count) * atom_values
    atom_bytes = atom_values * 2
    source_exact_bytes = source_values * 2
    bits_per_assignment = max(1, int(math.ceil(math.log2(max(atom_count, 2)))))
    assignment_bits = (expert_count - 1) * int(atom_count) * bits_per_assignment
    assignment_bytes = (assignment_bits + 7) // 8
    neuron_index_bytes = expert_count * int(atom_count) * NEURON_INDEX_BYTES
    directory_bytes = expert_count * EXPERT_DIRECTORY_ENTRY_BYTES
    complete_bytes = (
        atom_bytes
        + assignment_bytes
        + neuron_index_bytes
        + directory_bytes
        + METADATA_BYTES
        + DECODER_SUPPORT_BYTES
    )
    return {
        "logical_experts": int(expert_count),
        "selected_neuron_triples_per_expert": int(atom_count),
        "triple_components": ["gate_row", "up_row_after_reciprocal_normalization", "down_column_after_reciprocal_scaling"],
        "source_width": int(width),
        "source_selected_logical_weights": source_values,
        "source_exact_bf16_scope_bytes": source_exact_bytes,
        "shared_bf16_atom_dictionary_bytes": atom_bytes,
        "atom_assignment_bits_per_selected_non_dictionary_expert_neuron": bits_per_assignment,
        "packed_atom_assignment_bits": assignment_bits,
        "packed_atom_assignment_bytes": assignment_bytes,
        "selected_neuron_index_bytes": neuron_index_bytes,
        "expert_directory_bytes": directory_bytes,
        "metadata_bytes": METADATA_BYTES,
        "decoder_support_bytes": DECODER_SUPPORT_BYTES,
        "reciprocal_scaling_descriptor_bytes": 0,
        "reciprocal_scaling_descriptor_rule": "up norm is folded into the down column; no free scale is omitted",
        "complete_bytes": complete_bytes,
        "complete_scoped_ebpw": complete_bytes * 8.0 / max(source_values, 1),
        "byte_advantage_vs_two_exact_bf16_selected_scopes": 1.0 - complete_bytes / max(source_exact_bytes, 1),
        "byte_advantage_percent": 100.0 * (1.0 - complete_bytes / max(source_exact_bytes, 1)),
        "accounting_denominator_source_weights": source_values,
        "scope_boundary": "two experts x selected neuron triples only; unselected neurons remain exact in diagnostic full-output control",
    }


def _screen_pair(
    gate_ups: Sequence[np.ndarray],
    downs: Sequence[np.ndarray],
    *,
    expert_ids: Sequence[int],
    route_weights: Sequence[float],
    activation: np.ndarray,
    selected_count: int = 128,
    probe_count: int = 8,
    output_probe_dim: int = 16,
    seed: int = 1337,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    if len(gate_ups) != 2 or len(downs) != 2 or len(expert_ids) != 2 or len(route_weights) != 2:
        raise ValueError("E02 requires exactly two selected experts")
    if len(set(int(value) for value in expert_ids)) != 2:
        raise ValueError("E02 experts must be distinct")
    if not all(math.isfinite(float(value)) for value in route_weights):
        raise ValueError("route weights must be finite")
    gate_ups = [np.asarray(value, dtype=np.float32) for value in gate_ups]
    downs = [np.asarray(value, dtype=np.float32) for value in downs]
    if any(value.ndim != 2 for value in gate_ups + downs):
        raise ValueError("source tensors must be 2D after expert extraction")
    if any(value.shape != gate_ups[0].shape for value in gate_ups):
        raise ValueError("selected gate_up experts have different shapes")
    if any(value.shape != downs[0].shape for value in downs):
        raise ValueError("selected down experts have different shapes")
    rows, width = (int(gate_ups[0].shape[0]), int(gate_ups[0].shape[1]))
    if rows % 2 or downs[0].shape != (width, rows // 2):
        raise ValueError("gate_up/down geometry is not a gated-MoE triple bank")
    intermediate = rows // 2
    activation = np.asarray(activation, dtype=np.float32).reshape(-1)
    if activation.size != width or not np.isfinite(activation).all():
        raise ValueError("source activation must match gate_up width and be finite")
    selected = _selected_neurons(intermediate, selected_count)
    source_triples = [_triples_from_source(gate_up, down, selected) for gate_up, down in zip(gate_ups, downs)]
    canonical = [_canonicalize_triples(*triples)[:3] for triples in source_triples]
    canonical_a = canonical[0]
    canonical_b = canonical[1]
    atom_gate = _bf16(canonical_a[0])
    atom_up = _bf16(canonical_a[1])
    atom_down = _bf16(canonical_a[2])
    atoms = (atom_gate, atom_up, atom_down)
    inputs, output_probes = _build_probes(
        activation,
        probe_count=probe_count,
        hidden=width,
        output_probe_dim=output_probe_dim,
        seed=seed,
    )

    match_started_ns = time.perf_counter_ns()
    signature_a = _function_signature(canonical_a, inputs, output_probes)
    signature_b = _function_signature(canonical_b, inputs, output_probes)
    similarity = signature_b @ signature_a.T
    assignments = np.argmax(similarity, axis=1).astype(np.int64)
    matched_similarity = similarity[np.arange(selected.size), assignments]
    # A permutation of complete triples would be algebraically invisible:
    # the gated-MoE sum is commutative.  The matched null therefore draws
    # independent dictionary IDs with the same packed ID width and selected
    # neuron count.  It preserves the byte budget while breaking the
    # source-function matching relation.
    shuffled_assignments = np.random.default_rng(int(seed) + 1).integers(
        0, selected.size, size=selected.size, dtype=np.int64
    )
    match_ns = time.perf_counter_ns() - match_started_ns

    identity = np.arange(selected.size, dtype=np.int64)
    candidate_triples_a = tuple(value[identity] for value in atoms)
    candidate_triples_b = tuple(value[assignments] for value in atoms)
    shuffled_triples_b = tuple(value[shuffled_assignments] for value in atoms)
    source_selected_outputs = [_triple_outputs(triples, activation) for triples in source_triples]
    candidate_selected_outputs = [
        _triple_outputs(candidate, activation) for candidate in (candidate_triples_a, candidate_triples_b)
    ]
    shuffled_selected_outputs = [
        _triple_outputs(candidate, activation) for candidate in (candidate_triples_a, shuffled_triples_b)
    ]
    source_probe_outputs = [_triple_outputs(triples, inputs.T) for triples in source_triples]
    candidate_probe_outputs = [
        _triple_outputs(candidate, inputs.T) for candidate in (candidate_triples_a, candidate_triples_b)
    ]
    shuffled_probe_outputs = [
        _triple_outputs(candidate, inputs.T) for candidate in (candidate_triples_a, shuffled_triples_b)
    ]

    source_full_outputs = [_full_outputs(gate_up, down, activation) for gate_up, down in zip(gate_ups, downs)]
    candidate_full_outputs = []
    shuffled_full_outputs = []
    for source_full, source_selected, candidate_selected, shuffled_selected, gate_up, down in zip(
        source_full_outputs,
        source_selected_outputs,
        candidate_selected_outputs,
        shuffled_selected_outputs,
        gate_ups,
        downs,
    ):
        # The unselected 512 neurons stay exact, so the full-output diagnostic
        # isolates the effect of the shared selected-triple replacement.
        candidate_full_outputs.append(source_full - source_selected + candidate_selected)
        shuffled_full_outputs.append(source_full - source_selected + shuffled_selected)

    per_expert = []
    per_expert_probe = []
    for position, (expert, weight) in enumerate(zip(expert_ids, route_weights)):
        per_expert.append(
            {
                "expert": int(expert),
                "route_weight": float(weight),
                "selected_triple_output": {
                    "source": _vector_metrics(source_selected_outputs[position][0], candidate_selected_outputs[position][0]),
                    "shuffled_null": _vector_metrics(source_selected_outputs[position][0], shuffled_selected_outputs[position][0]),
                },
                "full_expert_output": {
                    "source": _vector_metrics(source_full_outputs[position][0], candidate_full_outputs[position][0]),
                    "shuffled_null": _vector_metrics(source_full_outputs[position][0], shuffled_full_outputs[position][0]),
                },
            }
        )
        per_expert_probe.append(
            {
                "expert": int(expert),
                "selected_triple_function_probes": {
                    "source": _vector_metrics(source_probe_outputs[position], candidate_probe_outputs[position]),
                    "shuffled_null": _vector_metrics(source_probe_outputs[position], shuffled_probe_outputs[position]),
                },
            }
        )
    route_source_selected = sum(
        float(weight) * output for weight, output in zip(route_weights, source_selected_outputs)
    )
    route_candidate_selected = sum(
        float(weight) * output for weight, output in zip(route_weights, candidate_selected_outputs)
    )
    route_shuffled_selected = sum(
        float(weight) * output for weight, output in zip(route_weights, shuffled_selected_outputs)
    )
    route_source_full = sum(float(weight) * output for weight, output in zip(route_weights, source_full_outputs))
    route_candidate_full = sum(float(weight) * output for weight, output in zip(route_weights, candidate_full_outputs))
    route_shuffled_full = sum(float(weight) * output for weight, output in zip(route_weights, shuffled_full_outputs))
    route_source_probe = sum(float(weight) * output for weight, output in zip(route_weights, source_probe_outputs))
    route_candidate_probe = sum(float(weight) * output for weight, output in zip(route_weights, candidate_probe_outputs))
    route_shuffled_probe = sum(float(weight) * output for weight, output in zip(route_weights, shuffled_probe_outputs))

    accounting = _accounting(expert_count=2, atom_count=int(selected.size), width=width)
    matched_b_selected = per_expert[1]["selected_triple_output"]["source"]
    null_b_selected = per_expert[1]["selected_triple_output"]["shuffled_null"]
    matched_b_full = per_expert[1]["full_expert_output"]["source"]
    null_b_full = per_expert[1]["full_expert_output"]["shuffled_null"]
    matched_b_probe = per_expert_probe[1]["selected_triple_function_probes"]["source"]
    null_b_probe = per_expert_probe[1]["selected_triple_function_probes"]["shuffled_null"]
    source_output_null_advantage = bool(
        float(matched_b_selected["relative_l2"]) < float(null_b_selected["relative_l2"])
        and float(matched_b_full["relative_l2"]) < float(null_b_full["relative_l2"])
        and float(_vector_metrics(route_source_selected, route_candidate_selected)["relative_l2"])
        < float(_vector_metrics(route_source_selected, route_shuffled_selected)["relative_l2"])
        and float(_vector_metrics(route_source_full, route_candidate_full)["relative_l2"])
        < float(_vector_metrics(route_source_full, route_shuffled_full)["relative_l2"])
    )
    synthetic_probe_null_advantage = bool(
        float(matched_b_probe["relative_l2"]) < float(null_b_probe["relative_l2"])
    )
    byte_advantage = float(accounting["byte_advantage_vs_two_exact_bf16_selected_scopes"])
    survives = bool(source_output_null_advantage and byte_advantage >= 0.20)
    return {
        "variant": {
            "family": "lawful_nonlinear_gated_moe_atom_sharing",
            "selected_experts": [int(value) for value in expert_ids],
            "selected_neuron_count": int(selected.size),
            "seed": int(seed),
            "probe_count": int(probe_count),
            "output_probe_dim": int(output_probe_dim),
        },
        "outcome": {
            "status": "SOURCE_OUTPUT_DISCRIMINATOR_SURVIVES" if survives else "FIXED_FAMILY_NEGATIVE",
            "source_output_matched_null_advantage": source_output_null_advantage,
            "synthetic_probe_matched_null_advantage": synthetic_probe_null_advantage,
            "matched_null_advantage": source_output_null_advantage,
            "byte_advantage_threshold": 0.20,
            "byte_advantage_achieved": byte_advantage >= 0.20,
            "reopen_rule": (
                "Expand only to a second source layer or a materially broader source-output bank after independent replication."
                if survives
                else "Close this fixed two-expert atom-sharing form for this source control; do not sweep atom count or matching seed without a new causal representation hypothesis."
            ),
        },
        "accounting": accounting,
        "source": {
            "expert_ids": [int(value) for value in expert_ids],
            "route_weights": [float(value) for value in route_weights],
            "selected_neuron_indices": selected.tolist(),
            "gate_up_shape": list(gate_ups[0].shape),
            "down_shape": list(downs[0].shape),
        },
        "matching": {
            "lawful_operations": [
                "common neuron permutation represented by packed atom assignments",
                "matched-null definition: independent dictionary IDs with the same packed descriptor width",
                "reciprocal up-row/down-column scaling with scale folded into down",
            ],
            "gate_scaling": "not used; SiLU gate scaling is not function-preserving",
            "dictionary_owner": "expert_ids[0] selected canonical BF16 triples",
            "non_dictionary_assignment_owner": "expert_ids[1] selected neurons",
            "assignments": assignments.tolist(),
            "shuffled_assignments": shuffled_assignments.tolist(),
            "assignment_similarity_mean": float(np.mean(matched_similarity)),
            "assignment_similarity_min": float(np.min(matched_similarity)),
            "assignment_similarity_max": float(np.max(matched_similarity)),
            "matching_ns": int(match_ns),
            "synthetic_probe_note": "probe columns after column zero are deterministic synthetic function probes; only column zero is the sealed source activation",
        },
        "source_activation_output": {
            "per_expert": per_expert,
            "route_weight_sum": float(sum(float(value) for value in route_weights)),
            "selected_triple_route": {
                "matched": _vector_metrics(route_source_selected, route_candidate_selected),
                "shuffled_null": _vector_metrics(route_source_selected, route_shuffled_selected),
            },
            "full_route": {
                "matched": _vector_metrics(route_source_full, route_candidate_full),
                "shuffled_null": _vector_metrics(route_source_full, route_shuffled_full),
            },
        },
        "function_probe_output": {
            "per_expert": per_expert_probe,
            "route_weighted": {
                "matched": _vector_metrics(route_source_probe, route_candidate_probe),
                "shuffled_null": _vector_metrics(route_source_probe, route_shuffled_probe),
            },
        },
        "direct_execution": {
            "status": "NOT_ESTABLISHED",
            "candidate_formula": "sum_j SiLU(gate_atom_j @ x) * (up_unit_atom_j @ x) * down_scaled_atom_j, with packed B-expert atom IDs",
            "dense_parent_required_by_formula": False,
            "screen_materializes_source_and_candidate_arrays": True,
            "production_kernel": "NOT_VALIDATED",
        },
        "timing": {
            "total_ns": int(time.perf_counter_ns() - started_ns),
        },
    }


def screen_matrices(
    gate_ups: Sequence[np.ndarray],
    downs: Sequence[np.ndarray],
    *,
    expert_ids: Sequence[int],
    route_weights: Sequence[float],
    activation: np.ndarray,
    selected_count: int = 128,
    probe_count: int = 8,
    output_probe_dim: int = 16,
    seed: int = 1337,
) -> dict[str, Any]:
    """Run one deterministic E02 pair screen on already loaded arrays."""
    return _screen_pair(
        gate_ups,
        downs,
        expert_ids=expert_ids,
        route_weights=route_weights,
        activation=activation,
        selected_count=selected_count,
        probe_count=probe_count,
        output_probe_dim=output_probe_dim,
        seed=seed,
    )


def _aggregate_replication_bank(
    pair_results: Sequence[Mapping[str, Any]], *, expert_ids: Sequence[int], seed: int,
) -> dict[str, Any]:
    """Apply the preregistered all-three-new-pairs replication rule."""
    if len(pair_results) != REPLICATION_PAIR_COUNT or len(expert_ids) != REPLICATION_EXPERT_COUNT:
        raise ValueError("E02 replication requires exactly four disjoint pairs from eight experts")
    pairs = [list(map(int, expert_ids[index:index + 2])) for index in range(0, 8, 2)]
    if len(set(map(int, expert_ids))) != REPLICATION_EXPERT_COUNT:
        raise ValueError("E02 replication experts must be distinct")
    summaries: list[dict[str, Any]] = []
    for index, (pair, result) in enumerate(zip(pairs, pair_results)):
        outcome = result.get("outcome")
        accounting = result.get("accounting")
        source_output = result.get("source_activation_output")
        if not isinstance(outcome, Mapping) or not isinstance(accounting, Mapping) or not isinstance(source_output, Mapping):
            raise ValueError("E02 replication pair result is malformed")
        matched = source_output["full_route"]["matched"]["relative_l2"]
        shuffled = source_output["full_route"]["shuffled_null"]["relative_l2"]
        summaries.append(
            {
                "pair_index": index,
                "expert_ids": pair,
                "seed": int(seed) + index,
                "is_original_calibration_pair": index == 0,
                "status": outcome["status"],
                "source_output_matched_null_advantage": outcome[
                    "source_output_matched_null_advantage"
                ],
                "full_route_matched_relative_l2": matched,
                "full_route_shuffled_relative_l2": shuffled,
                "complete_scoped_ebpw": accounting["complete_scoped_ebpw"],
                "complete_bytes": accounting["complete_bytes"],
                "source_exact_bf16_scope_bytes": accounting[
                    "source_exact_bf16_scope_bytes"
                ],
            }
        )
    independent = summaries[1:]
    survivors = sum(row["status"] == "SOURCE_OUTPUT_DISCRIMINATOR_SURVIVES" for row in independent)
    replicated = survivors == len(independent)
    total_complete_bytes = sum(int(row["complete_bytes"]) for row in summaries)
    total_exact_bytes = sum(int(row["source_exact_bf16_scope_bytes"]) for row in summaries)
    return {
        "status": (
            "REPLICATED_ACROSS_ALL_THREE_NEW_TOP8_ROUTE_PAIRS"
            if replicated
            else "NOT_REPLICATED_ACROSS_PREDECLARED_TOP8_ROUTE_BANK"
        ),
        "pairing_rule": "adjacent_disjoint_pairs_in_descending_sealed_route_order.v1",
        "null_seed_rule": "base_seed_plus_pair_index.v1",
        "base_seed": int(seed),
        "pairs": summaries,
        "original_calibration_pair_reproduced": summaries[0]["status"]
        == "SOURCE_OUTPUT_DISCRIMINATOR_SURVIVES",
        "independent_replication_pair_count": len(independent),
        "independent_survivor_count": survivors,
        "replication_rule": "all three new disjoint pairs must clear the unchanged source-output matched-null and 20-percent byte-advantage gate",
        "replicated": replicated,
        "aggregate_accounting": {
            "four_independent_pair_dictionaries_complete_bytes": total_complete_bytes,
            "four_pair_exact_bf16_selected_scope_bytes": total_exact_bytes,
            "byte_advantage_percent": 100.0
            * (1.0 - total_complete_bytes / max(total_exact_bytes, 1)),
            "not_a_single_eight_expert_dictionary": True,
        },
        "disposition": (
            "retain only as a replicated bounded signal; still not a usable representation"
            if replicated
            else "close this fixed top8-pair replication branch; preserve the original pair as a bounded non-generalized signal"
        ),
    }


def _verified_source_shard_binding(spec: Path, index_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Bind the replication to the existing exact-byte source ledger."""
    canonical_spec = DEFAULT_SPEC.expanduser().resolve()
    if spec != canonical_spec:
        raise ValueError("E02 replication requires the pinned canonical Flash specimen")
    manifest_path = _canonical_model_lake_manifest(spec)
    manifest_sha256 = _sha256_bytes(manifest_path.read_bytes())
    index_sha256 = _sha256_bytes(index_path.read_bytes())
    ledger_path = DEFAULT_SOURCE_SHARD_LEDGER.resolve()
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    verified = _verify_shard_ledger(
        {
            "path": str(ledger_path),
            "sha256": _sha256_bytes(ledger_bytes),
            "schema": SOURCE_SHARD_LEDGER_SCHEMA,
            "status": SOURCE_SHARD_LEDGER_STATUS,
            "seal_sha256": ledger.get("seal_sha256"),
        },
        contract_dir=ROOT,
        model_root=spec,
        manifest_sha256=manifest_sha256,
        index_path=index_path,
        index_sha256=index_sha256,
    )
    needed = {"model-00002-of-00131.safetensors", "model-00003-of-00131.safetensors"}
    shard_hashes = {
        str(row["path"]): str(row["sha256"])
        for row in ledger.get("shards", [])
        if isinstance(row, Mapping) and row.get("path") in needed
    }
    if set(shard_hashes) != needed:
        raise ValueError("verified source ledger lacks an E02 replication shard")
    return verified, shard_hashes


def verify_replication_receipt(document: Mapping[str, Any]) -> dict[str, Any]:
    """Independently verify a sealed E02 top-eight replication receipt."""
    if document.get("schema") != REPLICATION_SCHEMA:
        raise ValueError("E02 replication receipt schema is invalid")
    if document.get("status") != "SOURCE_BOUND_NONLINEAR_ATOM_SHARING_REPLICATION_COMPLETE":
        raise ValueError("E02 replication receipt status is invalid")
    if document.get("seal_sha256") != _canonical_sha256(dict(document)):
        raise ValueError("E02 replication receipt seal mismatch")
    if document.get("model_loaded") is not False or document.get("gpu_session_started") is not False:
        raise ValueError("E02 replication receipt unexpectedly claims model or GPU execution")
    producer = document.get("producer")
    current_path = Path(__file__).resolve()
    if not isinstance(producer, Mapping) or (
        Path(str(producer.get("path") or "")).resolve() != current_path
        or producer.get("sha256") != _sha256_bytes(current_path.read_bytes())
    ):
        raise ValueError("E02 replication producer identity does not match current source")
    source = document.get("source")
    if not isinstance(source, Mapping) or Path(str(source.get("specimen") or "")).resolve() != DEFAULT_SPEC.resolve():
        raise ValueError("E02 replication source is not the pinned Flash specimen")
    ledger = source.get("verified_source_shard_ledger")
    if not isinstance(ledger, Mapping) or ledger.get("verification") != (
        "sha256_exact_file_bytes (attested by sealed ledger)"
    ):
        raise ValueError("E02 replication lacks exact source-shard ledger identity")
    experts = source.get("sealed_route_top8_expert_ids")
    if not isinstance(experts, list) or len(experts) != 8 or len(set(experts)) != 8:
        raise ValueError("E02 replication source route does not contain eight distinct experts")
    replication = document.get("replication")
    if not isinstance(replication, Mapping):
        raise ValueError("E02 replication outcome is absent")
    pairs = replication.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 4:
        raise ValueError("E02 replication does not contain four pair results")
    if [row.get("expert_ids") for row in pairs] != [experts[index:index + 2] for index in range(0, 8, 2)]:
        raise ValueError("E02 replication pairs do not match the sealed route partition")
    if [row.get("seed") for row in pairs] != [1337, 1338, 1339, 1340]:
        raise ValueError("E02 replication null seeds do not match the fixed rule")
    independent_survivors = sum(
        row.get("status") == "SOURCE_OUTPUT_DISCRIMINATOR_SURVIVES" for row in pairs[1:]
    )
    if replication.get("independent_replication_pair_count") != 3 or (
        replication.get("independent_survivor_count") != independent_survivors
        or replication.get("replicated") is not (independent_survivors == 3)
    ):
        raise ValueError("E02 replication aggregate contradicts its pair outcomes")
    return dict(document)


def _load_bound_inputs(
    spec: Path,
    gate_up_tensor: str,
    down_tensor: str,
    bridge: Path,
    expert_ids: Sequence[int] | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]], list[int], list[float], np.ndarray, dict[str, Any]]:
    route_ids, route_weights = _route_ids_from_bridge(bridge)
    selected = list(route_ids[:2] if expert_ids is None else [int(value) for value in expert_ids])
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError("E02 requires two distinct experts")
    if any(value not in route_ids for value in selected):
        raise ValueError("every E02 expert must be present in the sealed route")
    weights_by_id = {expert: weight for expert, weight in zip(route_ids, route_weights)}
    selected_route_weights = [float(weights_by_id[expert]) for expert in selected]
    gate_ups: list[np.ndarray] = []
    downs: list[np.ndarray] = []
    gate_records: list[dict[str, Any]] = []
    down_records: list[dict[str, Any]] = []
    activation: np.ndarray | None = None
    controls: list[dict[str, Any]] = []
    for expert in selected:
        gate_up, gate_record = load_expert(spec, gate_up_tensor, int(expert))
        down, down_record = load_expert(spec, down_tensor, int(expert))
        current_activation, control = load_source_activation_bridge(
            bridge,
            expected_columns=int(gate_up.shape[1]),
            expert=int(expert),
        )
        if activation is None:
            activation = current_activation
        elif not np.array_equal(activation, current_activation):
            raise ValueError("source activation changed between expert validations")
        gate_ups.append(gate_up)
        downs.append(down)
        gate_records.append(gate_record)
        down_records.append(down_record)
        controls.append(control)
    if activation is None:
        raise ValueError("no source activation was validated")
    index_path = spec / "model.safetensors.index.json"
    bridge_path = bridge.expanduser().resolve()
    binding = {
        "bridge_path": str(bridge_path),
        "bridge_sha256": _sha256_bytes(bridge_path.read_bytes()),
        "route_expert_ids": route_ids,
        "route_weights": route_weights,
        "selected_expert_ids": selected,
        "selected_route_weights": selected_route_weights,
        "activation_sha256": controls[0]["activation_sha256"],
        "source_layer_receipt_sha256": controls[0]["source_layer_receipt_sha256"],
        "teacher_only": True,
        "standalone_nr_dependency": False,
    }
    for record in gate_records:
        record["tensor"] = gate_up_tensor
    for record in down_records:
        record["tensor"] = down_tensor
    return gate_ups, downs, gate_records, down_records, selected, selected_route_weights, activation, {
        "index_path": str(index_path),
        "index_sha256": _sha256_bytes(index_path.read_bytes()),
        "activation_control": binding,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--gate-up-tensor", default=DEFAULT_GATE_UP_TENSOR)
    parser.add_argument("--down-tensor", default=DEFAULT_DOWN_TENSOR)
    parser.add_argument("--source-activation-bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--experts", default=None, help="two comma-separated experts from the sealed route; default is the first two")
    parser.add_argument("--selected-count", type=int, default=128)
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--output-probe-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--replicate-top8",
        action="store_true",
        help="run the fixed four-pair top-eight routed-expert replication bank",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.selected_count < 2 or args.probe_count < 2 or args.output_probe_dim < 2:
        raise SystemExit("selected-count, probe-count, and output-probe-dim must be at least two")
    selected_experts = _parse_ints(args.experts, label="experts") if args.experts is not None else None
    spec = args.spec.expanduser().resolve()
    bridge = args.source_activation_bridge.expanduser().resolve()
    started = datetime.now(timezone.utc)
    started_ns = time.perf_counter_ns()
    if args.replicate_top8:
        if (
            selected_experts is not None
            or spec != DEFAULT_SPEC.expanduser().resolve()
            or bridge != DEFAULT_BRIDGE.expanduser().resolve()
            or args.gate_up_tensor != DEFAULT_GATE_UP_TENSOR
            or args.down_tensor != DEFAULT_DOWN_TENSOR
            or args.selected_count != 128
            or args.probe_count != 8
            or args.output_probe_dim != 16
            or args.seed != 1337
        ):
            raise SystemExit(
                "E02 top-eight replication is preregistered; source, tensors, bridge, geometry, and seeds accept no deviations"
            )
        route_ids, route_weights = _route_ids_from_bridge(bridge)
        top_experts = [int(value) for value in route_ids[:REPLICATION_EXPERT_COUNT]]
        if len(top_experts) != REPLICATION_EXPERT_COUNT or len(set(top_experts)) != REPLICATION_EXPERT_COUNT:
            raise SystemExit("sealed route does not provide eight distinct experts")
        pair_results: list[dict[str, Any]] = []
        gate_records: list[dict[str, Any]] = []
        down_records: list[dict[str, Any]] = []
        activation_sha256: str | None = None
        activation_control: dict[str, Any] | None = None
        index_path = spec / "model.safetensors.index.json"
        for pair_index in range(REPLICATION_PAIR_COUNT):
            pair = top_experts[pair_index * 2:pair_index * 2 + 2]
            (
                gate_ups,
                downs,
                current_gate_records,
                current_down_records,
                expert_ids,
                pair_route_weights,
                activation,
                binding,
            ) = _load_bound_inputs(
                spec,
                args.gate_up_tensor,
                args.down_tensor,
                bridge,
                pair,
            )
            current_activation_sha256 = binding["activation_control"]["activation_sha256"]
            if activation_sha256 is None:
                activation_sha256 = current_activation_sha256
                activation_control = binding["activation_control"]
            elif current_activation_sha256 != activation_sha256:
                raise ValueError("source activation changed across E02 replication pairs")
            pair_results.append(
                screen_matrices(
                    gate_ups,
                    downs,
                    expert_ids=expert_ids,
                    route_weights=pair_route_weights,
                    activation=activation,
                    selected_count=args.selected_count,
                    probe_count=args.probe_count,
                    output_probe_dim=args.output_probe_dim,
                    seed=args.seed + pair_index,
                )
            )
            gate_records.extend(current_gate_records)
            down_records.extend(current_down_records)
        ledger_binding, shard_hashes = _verified_source_shard_binding(spec, index_path)
        for record in gate_records + down_records:
            record["source_shard_sha256"] = shard_hashes[Path(record["source_file"]).name]
        replication = _aggregate_replication_bank(
            pair_results,
            expert_ids=top_experts,
            seed=args.seed,
        )
        document = {
            "schema": REPLICATION_SCHEMA,
            "generated_at": started.isoformat(),
            "status": "SOURCE_BOUND_NONLINEAR_ATOM_SHARING_REPLICATION_COMPLETE",
            "producer": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_bytes(Path(__file__).read_bytes()),
            },
            "source": {
                "specimen": str(spec),
                "index_path": str(index_path),
                "index_sha256": _sha256_bytes(index_path.read_bytes()),
                "verified_source_shard_ledger": ledger_binding,
                "source_shard_sha256s": shard_hashes,
                "gate_up_tensor": args.gate_up_tensor,
                "down_tensor": args.down_tensor,
                "gate_up_records": gate_records,
                "down_records": down_records,
                "source_payload_bytes_read": int(
                    sum(int(record["payload_bytes_read"]) for record in gate_records + down_records)
                ),
                "activation_control": activation_control,
                "sealed_route_top8_expert_ids": top_experts,
                "sealed_route_top8_weights": [float(value) for value in route_weights[:8]],
            },
            "replication": replication,
            "pair_results": pair_results,
            "model_loaded": False,
            "gpu_session_started": False,
            "claim_boundary": (
                "Fixed E02 replication over four disjoint adjacent pairs from the sealed layer-0 top-eight route. "
                "The original pair is a calibration; the other three are the independent replication bank. "
                "All pairs share one sealed source activation and retain exact unselected neurons in diagnostic "
                "outputs. This is not cross-layer or cross-context evidence, a complete expert/model NR, whole-body "
                "EBPW, capability, production execution, TPS, or promotion."
            ),
            "elapsed_ns": int(time.perf_counter_ns() - started_ns),
        }
        document["seal_sha256"] = _canonical_sha256(document)
        verify_replication_receipt(document)
        if args.output.exists():
            raise SystemExit(f"refusing to replace existing E02 replication receipt: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": document["status"],
                    "replication": replication,
                    "source_payload_bytes_read": document["source"]["source_payload_bytes_read"],
                    "elapsed_ns": document["elapsed_ns"],
                    "output": str(args.output),
                    "seal": document["seal_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    gate_ups, downs, gate_records, down_records, expert_ids, route_weights, activation, binding = _load_bound_inputs(
        spec,
        args.gate_up_tensor,
        args.down_tensor,
        bridge,
        selected_experts,
    )
    result = screen_matrices(
        gate_ups,
        downs,
        expert_ids=expert_ids,
        route_weights=route_weights,
        activation=activation,
        selected_count=args.selected_count,
        probe_count=args.probe_count,
        output_probe_dim=args.output_probe_dim,
        seed=args.seed,
    )
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": started.isoformat(),
        "status": "SOURCE_BOUND_NONLINEAR_ATOM_SHARING_SCREEN_COMPLETE",
        "source": {
            "specimen": str(spec),
            "index_path": binding["index_path"],
            "index_sha256": binding["index_sha256"],
            "gate_up_tensor": args.gate_up_tensor,
            "down_tensor": args.down_tensor,
            "gate_up_records": gate_records,
            "down_records": down_records,
            "matrix_shapes": {
                "gate_up": list(gate_ups[0].shape),
                "down": list(downs[0].shape),
            },
            "source_payload_bytes_read": int(
                sum(int(record["payload_bytes_read"]) for record in gate_records + down_records)
            ),
            "activation_control": binding["activation_control"],
        },
        "representation": {
            "family": "lawful_nonlinear_gated_moe_atom_sharing",
            "dictionary": "one BF16 canonical gate/up/down triple dictionary from the first selected expert",
            "assignment": "packed atom IDs for the second selected expert; common neuron permutation only",
            "reciprocal_scaling": "up row is normalized and its norm is folded into the paired down column; gate rows are not scaled",
            "all_complete_parts_billed": [
                "shared BF16 atom dictionary",
                "packed non-dictionary atom assignments",
                "selected neuron indices",
                "expert directory",
                "metadata",
                "decoder support",
            ],
            "screen_only": True,
        },
        "candidate": result,
        "claim_boundary": (
            "This is a source-bound two-expert Flash layer-0 gated-MoE atom-sharing discriminator. "
            "It evaluates exact gate/up/down triples on one sealed source activation plus deterministic "
            "synthetic function probes, compares a matched assignment to a byte-identical shuffled null, "
            "and bills the selected scope completely. The remaining neurons are exact only in the diagnostic "
            "full-output control. It proves no complete expert/model NR, whole-body EBPW, capability, "
            "production direct decoder, TPS, or promotion result."
        ),
        "elapsed_ns": int(time.perf_counter_ns() - started_ns),
    }
    document["seal_sha256"] = _canonical_sha256(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": document["status"],
                "outcome": result["outcome"],
                "complete_scoped_ebpw": result["accounting"]["complete_scoped_ebpw"],
                "byte_advantage_percent": result["accounting"]["byte_advantage_percent"],
                "matched_b_source_relative_l2": result["source_activation_output"]["per_expert"][1]["selected_triple_output"]["source"]["relative_l2"],
                "shuffled_b_source_relative_l2": result["source_activation_output"]["per_expert"][1]["selected_triple_output"]["shuffled_null"]["relative_l2"],
                "elapsed_ns": document["elapsed_ns"],
                "output": str(args.output),
                "seal": document["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
