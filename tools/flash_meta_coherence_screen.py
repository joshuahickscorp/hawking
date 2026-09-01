#!/usr/bin/env python3
"""Screen a teacher-trace-backed Flash functional meta representation.

This is an offline organ probe, not a packer.  It fits a shared input latent
basis and an expert-local output readout to the exact layer-4 routed expert
functions observed on teacher hidden states.  The program is priced as a
diagnostic factor equivalent, but no runtime artifact or physical EBPW claim
is emitted.

The default path refuses a small capture.  ``--unsafe-small-probe`` exists
only to make the failure mode visible while developing the screen; its output
is permanently non-promotable.  A successful held-out surface fit still needs
generated-token capability, router-margin protection, a serialized loader, a
native direct consumer, and a protected complete-token measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
DEFAULT_PARITY = ROOT / "receipts/headless/FLASH_FAST_COMPACT_L0_L7_PARITY.json"
DEFAULT_META_BUDGET = ROOT / "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json"
DEFAULT_STATE = ROOT / "receipts/headless/FLASH_META_TEACHER_L4_MLP_INPUT.f32"
DEFAULT_OUT = ROOT / "receipts/headless/FLASH_META_COHERENCE_SCREEN_L4.json"

SCHEMA = "hawking.flash.meta_coherence_screen.v1"
MODEL = "Qwen/Qwen3.8-Flash-Next"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
HIDDEN = 2560
MIN_TEACHER_ROWS = 256
DEFAULT_RANKS = (4, 8, 16, 32, 64)
MAX_HELDOUT_REL_RMSE = 0.05
MIN_HELDOUT_COSINE = 0.999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_teacher_states(paths: Iterable[Path], *, width: int = HIDDEN) -> tuple[np.ndarray, dict[str, Any]]:
    unique_rows: list[np.ndarray] = []
    files: list[dict[str, Any]] = []
    row_hashes: set[str] = set()
    raw_rows = 0
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"teacher state not found: {path}")
        values = np.fromfile(path, dtype="<f4")
        if values.size == 0 or values.size % width:
            raise ValueError(f"teacher state has invalid width: {path} ({values.size})")
        rows = values.reshape(-1, width)
        if not np.isfinite(rows).all():
            raise ValueError(f"teacher state contains non-finite values: {path}")
        raw_rows += int(rows.shape[0])
        for row in rows:
            digest = hashlib.sha256(row.astype("<f4", copy=False).tobytes()).hexdigest()
            if digest not in row_hashes:
                row_hashes.add(digest)
                unique_rows.append(row.copy())
        files.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "rows": int(rows.shape[0]),
                "width": int(rows.shape[1]),
            }
        )
    if not unique_rows:
        raise ValueError("at least one teacher state is required")
    # Exact duplicate rows are removed before the fit/holdout split.  This
    # prevents replayed captures from leaking the same teacher point into both
    # surfaces while preserving raw per-file counts for auditability.
    states = np.stack(unique_rows, axis=0).astype(np.float32, copy=False)
    return states, {
        "files": files,
        "rows": int(states.shape[0]),
        "raw_rows": raw_rows,
        "unique_row_hashes": len(row_hashes),
        "width": int(states.shape[1]),
    }


def load_teacher_capture_binding(
    receipt_path: Path,
    state_paths: list[Path],
    *,
    expected_layer: int = 4,
) -> dict[str, Any]:
    """Validate that raw rows are the requested layer's MLP-input surface."""
    if expected_layer < 0:
        raise ValueError(f"expected teacher layer must be non-negative: {expected_layer}")
    receipt_path = receipt_path.resolve()
    if not receipt_path.is_file():
        raise ValueError(f"teacher capture receipt not found: {receipt_path}")
    document = load_json(receipt_path)
    if document.get("schema") != "hawking.flash.meta_teacher_trace.v1":
        raise ValueError(f"{receipt_path}: unexpected teacher capture schema")
    if document.get("status") != "CAPTURED_SOURCE_MLP_INPUT_NOT_CAPABILITY_PROVEN":
        raise ValueError(f"{receipt_path}: teacher capture is not a source MLP-input capture")
    if document.get("model") != MODEL or document.get("pinned_revision") != PINNED_REVISION:
        raise ValueError(f"{receipt_path}: teacher capture is for a different Flash revision")
    source_identity = document.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError(f"{receipt_path}: exact source specimen identity is missing")
    if (
        source_identity.get("model") != MODEL
        or source_identity.get("pinned_revision") != PINNED_REVISION
    ):
        raise ValueError(f"{receipt_path}: source specimen identity is for a different revision")
    artifact_root = source_identity.get("artifact_root")
    if not isinstance(artifact_root, str) or not artifact_root:
        raise ValueError(f"{receipt_path}: source specimen artifact_root is missing")
    index_identity = source_identity.get("safetensors_index")
    if not isinstance(index_identity, Mapping):
        raise ValueError(f"{receipt_path}: source specimen safetensors index identity is missing")
    index_path = index_identity.get("path")
    index_sha256 = index_identity.get("sha256")
    index_tensor_count = index_identity.get("bf16_tensor_count")
    if (
        not isinstance(index_path, str)
        or not index_path
        or not is_sha256(index_sha256)
        or isinstance(index_tensor_count, bool)
        or not isinstance(index_tensor_count, int)
        or index_tensor_count < 1
    ):
        raise ValueError(f"{receipt_path}: source specimen safetensors index identity is invalid")
    config_identity = source_identity.get("config")
    if not isinstance(config_identity, Mapping):
        raise ValueError(f"{receipt_path}: source specimen config identity is missing")
    if not isinstance(config_identity.get("path"), str) or not config_identity.get("path"):
        raise ValueError(f"{receipt_path}: source specimen config path is missing")
    if not is_sha256(config_identity.get("sha256")):
        raise ValueError(f"{receipt_path}: source specimen config sha256 is invalid")
    trace = document.get("teacher_trace")
    if not isinstance(trace, Mapping):
        raise ValueError(f"{receipt_path}: teacher_trace is missing")
    if trace.get("layer") != expected_layer:
        raise ValueError(
            f"{receipt_path}: expected teacher layer {expected_layer} "
            f"(layers.{expected_layer}), got {trace.get('layer')!r}"
        )
    expected_surface = f"model.language_model.layers.{expected_layer}.mlp_input"
    if trace.get("surface") != expected_surface:
        raise ValueError(
            f"{receipt_path}: expected {expected_surface!r}, got {trace.get('surface')!r}"
        )
    expected_organ = f"layer_{expected_layer}.routed_experts.gate_up_proj"
    if trace.get("organ") != expected_organ:
        raise ValueError(
            f"{receipt_path}: teacher capture organ is not {expected_organ!r}"
        )
    if trace.get("dtype") != "F32_LE" or trace.get("width") != HIDDEN:
        raise ValueError(f"{receipt_path}: teacher capture geometry is not F32 width {HIDDEN}")
    if len(state_paths) != 1:
        raise ValueError("a bound Flash meta teacher capture must provide exactly one state file")
    declared_path = trace.get("state_path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError(f"{receipt_path}: teacher capture state_path is missing")
    declared = Path(declared_path)
    candidates = [declared]
    if not declared.is_absolute():
        candidates.extend((receipt_path.parent / declared, ROOT / declared))
    state_path = state_paths[0].resolve()
    if not any(candidate.resolve() == state_path for candidate in candidates):
        raise ValueError(
            f"{receipt_path}: bound state path does not match the requested state: "
            f"declared={declared_path!r} requested={str(state_path)!r}"
        )
    if not state_path.is_file():
        raise ValueError(f"teacher state not found: {state_path}")
    actual_sha256 = sha256(state_path)
    if trace.get("state_sha256") != actual_sha256:
        raise ValueError(f"{receipt_path}: teacher state sha256 does not match {state_path}")
    raw_rows = trace.get("raw_rows")
    rows = trace.get("rows")
    if (
        not isinstance(raw_rows, int)
        or raw_rows < 1
        or not isinstance(rows, int)
        or rows < 1
        or rows != raw_rows
    ):
        raise ValueError(f"{receipt_path}: teacher row counts are invalid")
    row_records = document.get("rows")
    if not isinstance(row_records, list) or len(row_records) != raw_rows:
        raise ValueError(f"{receipt_path}: per-row route records do not match raw_rows")
    route_union: set[int] = set()
    for index, row in enumerate(row_records):
        if not isinstance(row, Mapping):
            raise ValueError(f"{receipt_path}: teacher row {index} is not an object")
        route_ids = row.get("route_ids")
        if (
            not isinstance(route_ids, list)
            or len(route_ids) != 10
            or any(isinstance(value, bool) or not isinstance(value, int) for value in route_ids)
            or any(value < 0 or value >= 512 for value in route_ids)
            or len(set(route_ids)) != len(route_ids)
        ):
            raise ValueError(f"{receipt_path}: teacher row {index} has invalid top-K route IDs")
        route_union.update(route_ids)
    if not route_union:
        raise ValueError(f"{receipt_path}: teacher capture has no routed experts")
    if state_path.stat().st_size != raw_rows * HIDDEN * 4:
        raise ValueError(f"{receipt_path}: teacher state byte length does not match its receipt")
    state_values = np.fromfile(state_path, dtype="<f4")
    state_rows = state_values.reshape(raw_rows, HIDDEN)
    state_row_hashes: list[str] = []
    for index, row in enumerate(state_rows):
        digest = hashlib.sha256(row.astype("<f4", copy=False).tobytes()).hexdigest()
        state_row_hashes.append(digest)
        record = row_records[index]
        if record.get("row") != index:
            raise ValueError(f"{receipt_path}: teacher row {index} has an invalid row index")
        token_id = record.get("token_id")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{receipt_path}: teacher row {index} has an invalid token ID")
        input_hash = record.get("layer4_mlp_input_sha256")
        output_hash = record.get("layer4_output_sha256")
        layer3_hash = record.get("layer3_state_sha256")
        if input_hash != digest:
            raise ValueError(f"{receipt_path}: teacher row {index} input hash does not match state")
        for label, value in (
            ("layer3_state_sha256", layer3_hash),
            ("layer4_output_sha256", output_hash),
        ):
            if not is_sha256(value):
                raise ValueError(f"{receipt_path}: teacher row {index} lacks {label}")
        if record.get("layer4_output_surface") != "layer_4.final_state":
            raise ValueError(
                f"{receipt_path}: teacher row {index} lacks the layer-4 final-state output surface"
            )
    declared_unique = trace.get("unique_rows")
    if declared_unique != len(set(state_row_hashes)):
        raise ValueError(
            f"{receipt_path}: declared unique_rows does not match the captured state rows"
        )
    token_ids = document.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != raw_rows
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in token_ids)
        or len(set(token_ids)) != len(token_ids)
    ):
        raise ValueError(f"{receipt_path}: token IDs are not a unique per-row capture sequence")
    for index, record in enumerate(row_records):
        if record.get("token_id") != token_ids[index]:
            raise ValueError(f"{receipt_path}: token ID order does not match row records")
    route_audit = document.get("route_audit")
    if not isinstance(route_audit, Mapping):
        raise ValueError(f"{receipt_path}: route audit is missing")
    execution = document.get("execution")
    source_authority = isinstance(execution, Mapping) and execution.get("source_bf16_authority") is True
    if (
        not isinstance(execution, Mapping)
        or not source_authority
        or execution.get("dense_prefix") is not True
        or execution.get("dense_layer4") is not True
    ):
        raise ValueError(
            f"{receipt_path}: teacher capture is not a dense source-BF16 authority path"
        )
    if execution.get("source_index_sha256") != index_sha256:
        raise ValueError(
            f"{receipt_path}: execution source index sha256 does not match source identity"
        )
    declared_route_union = route_audit.get("route_union")
    if not isinstance(declared_route_union, list):
        raise ValueError(f"{receipt_path}: route audit has no explicit route union")
    if sorted(set(declared_route_union)) != sorted(route_union):
        raise ValueError(f"{receipt_path}: declared route union does not match per-row routes")
    declared_route_rows = route_audit.get("rows")
    if declared_route_rows != raw_rows:
        raise ValueError(f"{receipt_path}: route audit row count does not match raw_rows")
    route_set_count = route_audit.get("unique_ordered_topk_sets")
    actual_route_set_count = len({tuple(row.get("route_ids", ())) for row in row_records})
    if route_set_count != actual_route_set_count or actual_route_set_count < 2:
        raise ValueError(f"{receipt_path}: teacher capture lacks route-set diversity")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "state_path": str(state_path),
        "state_sha256": actual_sha256,
        "surface": expected_surface,
        "organ": str(trace["organ"]),
        "rows_declared": rows,
        "raw_rows_declared": raw_rows,
        "unique_rows_declared": trace.get("unique_rows"),
        "unique_rows_actual": len(set(state_row_hashes)),
        "source_pipeline": trace.get("source_pipeline"),
        "source_authority": source_authority,
        "source_identity": dict(source_identity),
        "route_union": sorted(route_union),
    }


def load_meta_budget_binding(path: Path) -> dict[str, Any]:
    """Bind a normal coherence screen to the prospective detached budget."""
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"meta budget receipt not found: {path}")
    document = load_json(path)
    if document.get("schema") != "hawking.flash.meta_representation.v1":
        raise ValueError(f"{path}: unexpected Flash meta budget schema")
    if document.get("status") != "PROSPECTIVE_META_ONLY":
        raise ValueError(f"{path}: meta budget is not prospective-only")
    if document.get("model") != MODEL or document.get("pinned_revision") != PINNED_REVISION:
        raise ValueError(f"{path}: meta budget is for a different Flash revision")
    metric = document.get("metric")
    if not isinstance(metric, Mapping) or metric.get("name") != "meta_bpw":
        raise ValueError(f"{path}: meta_bpw metric is missing")
    target = metric.get("prospective_target")
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise ValueError(f"{path}: prospective meta_bpw target is not numeric")
    target = float(target)
    if not math.isfinite(target) or target <= 0.0 or target >= 1.0:
        raise ValueError(f"{path}: prospective meta_bpw target is not below one")
    if metric.get("physical_ebpw") is not None:
        raise ValueError(f"{path}: meta budget must keep physical_ebpw null")
    state = document.get("measurement_state")
    if not isinstance(state, Mapping) or state.get("promotion_allowed") is not False:
        raise ValueError(f"{path}: meta budget is not fail-closed")
    return {
        "receipt_path": str(path),
        "receipt_sha256": sha256(path),
        "metric": "meta_bpw",
        "prospective_target": target,
        "below_one_target": True,
        "physical_ebpw": None,
    }


def tensor_locations(root: Path) -> dict[str, tuple[Path, int, int, tuple[int, ...], str]]:
    index_path = root / "model.safetensors.index.json"
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError("Flash source index has no weight_map")
    locations: dict[str, tuple[Path, int, int, tuple[int, ...], str]] = {}
    for shard_name in sorted(set(str(value) for value in weight_map.values())):
        shard = root / shard_name
        with shard.open("rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
        for name, metadata in header.items():
            if not isinstance(metadata, Mapping) or "data_offsets" not in metadata:
                continue
            begin, end = metadata["data_offsets"]
            locations[str(name)] = (
                shard,
                8 + header_length + int(begin),
                int(end - begin),
                tuple(int(item) for item in metadata["shape"]),
                str(metadata["dtype"]),
            )
    return locations


def read_bf16_experts(
    location: tuple[Path, int, int, tuple[int, ...], str], experts: list[int]
) -> np.ndarray:
    shard, offset, total_bytes, shape, dtype = location
    if dtype != "BF16" or len(shape) != 3 or max(experts) >= shape[0]:
        raise ValueError(f"unsupported expert tensor {shape} {dtype}")
    per_expert = int(np.prod(shape[1:])) * 2
    if total_bytes < (max(experts) + 1) * per_expert:
        raise ValueError("expert tensor byte range is truncated")
    raw = bytearray()
    with shard.open("rb") as handle:
        for expert in experts:
            handle.seek(offset + expert * per_expert)
            payload = handle.read(per_expert)
            if len(payload) != per_expert:
                raise IOError(f"short expert read: {shard} expert={expert}")
            raw.extend(payload)
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (words << 16).view("<f4").reshape(len(experts), shape[1], shape[2])


def heldout_split(rows: int) -> tuple[np.ndarray, np.ndarray]:
    if rows < 2:
        raise ValueError("at least two teacher rows are required for a holdout")
    heldout = np.zeros(rows, dtype=bool)
    heldout[np.arange(rows) % 5 == 0] = True
    if not heldout.any():
        heldout[-1] = True
    if heldout.all():
        heldout[-1] = False
    return ~heldout, heldout


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1).astype(np.float64, copy=False)
    b = right.reshape(-1).astype(np.float64, copy=False)
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
    return float(np.dot(a, b) / denominator)


def symmetric_group_q4(weights: np.ndarray, group: int = 64) -> np.ndarray:
    experts, output, input_width = weights.shape
    if input_width % group:
        raise ValueError(f"Q4 group {group} does not divide input width {input_width}")
    grouped = weights.reshape(experts, output, input_width // group, group)
    scale = np.maximum(np.max(np.abs(grouped), axis=3, keepdims=True) / 7.0, 1e-30)
    code = np.clip(np.rint(grouped / scale), -8, 7).astype(np.int8)
    return (code.astype(np.float32) * scale).reshape(weights.shape)


def projected_outputs(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("ni,eoi->eno", states, weights, optimize=True)


def fit_shared_latent_program(
    states: np.ndarray,
    weights: np.ndarray,
    *,
    rank: int,
    fit_rows: np.ndarray,
    heldout_rows: np.ndarray,
) -> dict[str, Any]:
    if rank < 1 or rank > min(int(fit_rows.sum()), states.shape[1]):
        raise ValueError(f"rank {rank} is not available for {int(fit_rows.sum())} fit rows")
    fit_states = states[fit_rows]
    # The basis is learned from teacher inputs only. It is shared by every
    # expert, while each expert gets its own output readout. No dense expert
    # tensor is emitted by this screen.
    _, _, vh = np.linalg.svd(fit_states, full_matrices=False)
    basis = vh[:rank].T.astype(np.float32, copy=False)
    fit_features = fit_states @ basis
    eval_features = states[heldout_rows] @ basis
    teacher = projected_outputs(states, weights)
    readouts: list[np.ndarray] = []
    fit_prediction = np.empty((weights.shape[0], int(fit_rows.sum()), weights.shape[1]), dtype=np.float32)
    heldout_prediction = np.empty(
        (weights.shape[0], int(heldout_rows.sum()), weights.shape[1]), dtype=np.float32
    )
    for expert in range(weights.shape[0]):
        readout, _, _, _ = np.linalg.lstsq(
            fit_features,
            teacher[expert, fit_rows],
            rcond=None,
        )
        readout = readout.astype(np.float32, copy=False)
        readouts.append(readout)
        fit_prediction[expert] = fit_features @ readout
        heldout_prediction[expert] = eval_features @ readout
    readout_stack = np.stack(readouts, axis=0)
    fit_teacher = teacher[:, fit_rows]
    heldout_teacher = teacher[:, heldout_rows]
    fit_error = float(
        np.linalg.norm(fit_prediction.astype(np.float64) - fit_teacher.astype(np.float64))
        / max(float(np.linalg.norm(fit_teacher)), 1e-30)
    )
    heldout_error = float(
        np.linalg.norm(heldout_prediction.astype(np.float64) - heldout_teacher.astype(np.float64))
        / max(float(np.linalg.norm(heldout_teacher)), 1e-30)
    )
    # ``weights`` is decoded to float32 for the CPU screen, but the source
    # tensor is BF16.  Price the diagnostic against the source parameter
    # count, and expose both byte domains so a loader cannot mistake the
    # temporary float32 working set for representation bytes.
    source_parameters = int(weights.size)
    source_bytes = source_parameters * 2
    factor_bytes = int((basis.size + readout_stack.size) * 2)
    baseline = symmetric_group_q4(weights)
    baseline_heldout = projected_outputs(states[heldout_rows], baseline)
    baseline_error = float(
        np.linalg.norm(baseline_heldout.astype(np.float64) - heldout_teacher.astype(np.float64))
        / max(float(np.linalg.norm(heldout_teacher)), 1e-30)
    )
    heldout_cosine = cosine(heldout_prediction, heldout_teacher)
    gate_pass = (
        heldout_error <= MAX_HELDOUT_REL_RMSE
        and heldout_cosine >= MIN_HELDOUT_COSINE
        and heldout_error < baseline_error
    )
    failure_gates: list[str] = []
    if heldout_error > MAX_HELDOUT_REL_RMSE:
        failure_gates.append("held-out function error")
    if heldout_cosine < MIN_HELDOUT_COSINE:
        failure_gates.append("held-out function cosine")
    if heldout_error >= baseline_error:
        failure_gates.append("does not beat per-expert Q4")
    return {
        "rank": rank,
        "fit_rows": int(fit_rows.sum()),
        "heldout_rows": int(heldout_rows.sum()),
        "fit_relative_fro_error": fit_error,
        "heldout_relative_fro_error": heldout_error,
        "heldout_cosine": heldout_cosine,
        "per_expert_q4_heldout_relative_fro_error": baseline_error,
        "beats_per_expert_q4_on_heldout": heldout_error < baseline_error,
        "diagnostic_factor_equivalent_bpw": factor_bytes * 8.0 / max(source_parameters, 1),
        "diagnostic_factor_bytes": factor_bytes,
        "selected_dense_source_bytes": source_bytes,
        "selected_dense_loaded_f32_bytes": int(weights.nbytes),
        "surface_failure_gates": failure_gates,
        "first_surface_failure": failure_gates[0] if failure_gates else None,
        "surface_gate_pass": gate_pass,
    }


def route_ids_for_layer(parity: Mapping[str, Any], layer: int) -> list[int]:
    rows = parity.get("comparisons")
    if not isinstance(rows, list):
        raise ValueError("parity receipt has no comparisons")
    row = next((item for item in rows if isinstance(item, Mapping) and item.get("layer") == layer), None)
    if row is None:
        raise ValueError(f"parity receipt has no layer {layer}")
    values = row.get("dense_route_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(f"parity receipt has no route ids for layer {layer}")
    experts = [int(value) for value in values]
    if len(set(experts)) != len(experts):
        raise ValueError("route ids must be distinct")
    return experts


def build_receipt(
    *,
    model_root: Path,
    parity_path: Path,
    state_paths: list[Path],
    output_path: Path,
    layer: int = 4,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    min_teacher_rows: int = MIN_TEACHER_ROWS,
    unsafe_small_probe: bool = False,
    teacher_receipt_path: Path | None = None,
    meta_budget_path: Path = DEFAULT_META_BUDGET,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    states, teacher = load_teacher_states(state_paths)
    meta_budget = load_meta_budget_binding(meta_budget_path)
    teacher_binding = None
    if teacher_receipt_path is not None:
        teacher_binding = load_teacher_capture_binding(
            teacher_receipt_path,
            state_paths,
            expected_layer=layer,
        )
    elif not unsafe_small_probe:
        raise ValueError(
            "a semantically aligned Flash meta screen requires --teacher-receipt; "
            "use --unsafe-small-probe only for an explicitly non-promotable development probe"
        )
    parity = load_json(parity_path)
    experts = (
        list(teacher_binding["route_union"])
        if teacher_binding is not None
        else route_ids_for_layer(parity, layer)
    )
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "model": MODEL,
        "pinned_revision": PINNED_REVISION,
        "status": "REFUSED_INSUFFICIENT_TEACHER_COVERAGE",
        "representation": {
            "kind": "shared_input_latent_plus_expert_local_output_readout",
            "organ": "layer_4.routed_experts.gate_up_proj",
            "dense_rematerialization": False,
            "runtime_artifact_emitted": False,
            "meta_bpw_target": meta_budget["prospective_target"],
            "physical_ebpw": None,
        },
        "teacher_trace": {
            **teacher,
            "layer_input": layer - 1,
            "route_layer": layer,
            "min_rows_required": min_teacher_rows,
            "unsafe_small_probe": unsafe_small_probe,
            "capture_binding": teacher_binding
            or {
                "status": "NOT_BOUND_UNSAFE_DEVELOPMENT_PROBE",
                "surface": "UNVERIFIED_RAW_STATE",
            },
        },
        "source": {
            "root": str(model_root.resolve()),
            "index_sha256": sha256(model_root / "model.safetensors.index.json"),
            "meta_budget_receipt": meta_budget,
            "parity_receipt": str(parity_path.resolve()),
            "parity_sha256": sha256(parity_path),
            "route_ids": experts,
        },
        "coherence_contract": {
            "fit_holdout_required": True,
            "min_heldout_cosine": MIN_HELDOUT_COSINE,
            "max_heldout_relative_fro_error": MAX_HELDOUT_REL_RMSE,
            "must_beat_per_expert_q4": True,
            "router_topk_membership_and_order": "protected exact; not replaced by this organ screen",
            "recurrent_and_kv_state": "protected exact; not measured here",
            "generated_token_capability": "required later",
        },
        "measurement_state": {
            "teacher_surface": "NOT_RUN",
            "serialized_artifact": "NOT_BUILT",
            "native_kernel": "NOT_BUILT",
            "physical_ebpw": "NULL_BY_RULE",
            "complete_token": "NOT_MEASURED",
            "promotion_allowed": False,
        },
        "claim_boundary": (
            "This screen can only evaluate a bounded teacher hidden-state surface. "
            "It cannot prove a whole-model functional representation, serialized bytes, "
            "physical EBPW, route stability, capability, residency, GPU latency, or TPS."
        ),
    }
    if states.shape[0] < min_teacher_rows and not unsafe_small_probe:
        base["next_gate"] = (
            f"collect at least {min_teacher_rows} non-duplicate teacher rows across held-out prompts "
            "before fitting any sub-1 candidate"
        )
        base["bench"] = {
            "state": "UNKNOWN",
            "measurement_state": "REFUSED_INSUFFICIENT_TEACHER_COVERAGE",
            "recorded_by": "tools/flash_meta_coherence_screen.py",
            "elapsed_ns": time.perf_counter_ns() - started,
        }
        base["seal_sha256"] = hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        return base

    locations = tensor_locations(model_root.resolve())
    tensor_name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
    location = locations.get(tensor_name)
    if location is None:
        raise ValueError(f"source tensor not found: {tensor_name}")
    weights = read_bf16_experts(location, experts)
    fit_rows, heldout_rows = heldout_split(states.shape[0])
    max_rank = min(int(fit_rows.sum()), states.shape[1], weights.shape[1])
    rows: list[dict[str, Any]] = []
    for rank in ranks:
        if rank <= max_rank:
            rows.append(
                fit_shared_latent_program(
                    states,
                    weights,
                    rank=rank,
                    fit_rows=fit_rows,
                    heldout_rows=heldout_rows,
                )
            )
    if not rows:
        raise ValueError(f"none of the requested ranks are available; max rank is {max_rank}")
    passing = [row for row in rows if row["surface_gate_pass"]]
    if unsafe_small_probe:
        status = "UNSAFE_SMALL_PROBE_NOT_PROMOTABLE"
    elif passing:
        status = "OFFLINE_META_SURFACE_GATE_PASS_RUNTIME_REQUIRED"
    else:
        status = "OFFLINE_META_SURFACE_GATE_FAILED"
    base.update(
        {
            "status": status,
            "source": {
                **base["source"],
                "tensor": tensor_name,
                "tensor_shape": list(location[3]),
                "tensor_dtype": location[4],
                "tensor_shard": str(location[0]),
                "selected_expert_count": len(experts),
                "selected_dense_source_bytes": int(weights.size) * 2,
                "selected_dense_loaded_f32_bytes": int(weights.nbytes),
            },
            "surface": {
                "organ": "gate_up_proj",
                "rows": rows,
                "frontier": passing,
                "fit_rows": int(fit_rows.sum()),
                "heldout_rows": int(heldout_rows.sum()),
            },
            "measurement_state": {
                **base["measurement_state"],
                "teacher_surface": "MEASURED_OFFLINE" if not unsafe_small_probe else "UNSAFE_SMALL_PROBE",
            },
            "next_gate": (
                "collect broader teacher traces, distill router/hidden/routed-output/terminal-logit surfaces, "
                "then build a serializer and direct generated-tile consumer"
            ),
            "bench": {
                "state": "UNKNOWN",
                "measurement_state": "MEASURED_OFFLINE_META_SURFACE",
                "recorded_by": "tools/flash_meta_coherence_screen.py",
                "machine": "Apple host CPU; selected Flash source expert ranges",
                "rule": "S032 §3 -- offline surface screen is not native execution",
                "elapsed_ns": time.perf_counter_ns() - started,
            },
        }
    )
    base["seal_sha256"] = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--state", type=Path, action="append")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--rank", type=int, action="append", dest="ranks")
    parser.add_argument("--min-teacher-rows", type=int, default=MIN_TEACHER_ROWS)
    parser.add_argument("--unsafe-small-probe", action="store_true")
    parser.add_argument(
        "--meta-budget",
        type=Path,
        default=DEFAULT_META_BUDGET,
        help="prospective detached meta_bpw budget receipt to hash-bind",
    )
    parser.add_argument(
        "--teacher-receipt",
        type=Path,
        help="source-authority receipt that binds the state file to layer-4 mlp_input",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.min_teacher_rows < 2:
        parser.error("--min-teacher-rows must be at least 2")
    state_paths = args.state or [DEFAULT_STATE]
    ranks = tuple(args.ranks or DEFAULT_RANKS)
    receipt = build_receipt(
        model_root=args.root,
        parity_path=args.parity,
        state_paths=state_paths,
        output_path=args.out,
        layer=args.layer,
        ranks=ranks,
        min_teacher_rows=args.min_teacher_rows,
        unsafe_small_probe=args.unsafe_small_probe,
        teacher_receipt_path=args.teacher_receipt,
        meta_budget_path=args.meta_budget,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "teacher_rows": receipt["teacher_trace"]["rows"],
                "unique_teacher_rows": receipt["teacher_trace"]["unique_row_hashes"],
                "frontier_rows": len(receipt.get("surface", {}).get("frontier", [])),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 2 if receipt["status"] == "REFUSED_INSUFFICIENT_TEACHER_COVERAGE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
