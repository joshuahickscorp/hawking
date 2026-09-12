#!/usr/bin/env python3
"""Low-rank plus sparse-repair discriminator for one stacked expert.

This is a bounded Flash-Next representation experiment.  It reads one expert
from one stacked safetensors tensor, stores the SVD factors as BF16, restores a
small set of largest residual entries as BF16 values plus uint32 indices, and
bills every scoped representation byte.  A caller may additionally bind one
sealed source activation control.  That produces a *single-control* executed
matvec error alongside weight error; it does not turn this bounded screen into
a model compressor, held-out evaluation, direct-execution result, or runtime
claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.representational_anatomy import _read_header_ex

DEFAULT_SPEC = Path("/Volumes/corpdrive/hawking-modellake/specimens/"
                    "Qwen--Qwen3.8-Flash-Next@34567a4712bc")
DEFAULT_TENSOR = "model.language_model.layers.1.mlp.experts.gate_up_proj"
FACTOR_BYTES = 2             # BF16 left/right factors
RESIDUAL_VALUE_BYTES = 2     # BF16 correction value
RESIDUAL_INDEX_BYTES = 4     # uint32 flattened matrix index
METADATA_BYTES = 128
DECODER_SUPPORT_BYTES = 65536
SOURCE_ACTIVATION_BRIDGE_SCHEMA = "hawking.flash_source_moe_bridge.v1"


def _decode(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").copy()
    raise ValueError(f"unsupported dtype {dtype}")


def _bf16(value: np.ndarray) -> np.ndarray:
    """Round float32 values to the BF16 values billed by the artifact."""
    values = np.asarray(value, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def load_expert(spec: Path, tensor: str, expert: int) -> tuple[np.ndarray, dict]:
    widths = {"BF16": 2, "F16": 2, "F32": 4}
    for path in sorted(spec.glob("*.safetensors")):
        header, header_len = _read_header_ex(str(path))
        if tensor not in header:
            continue
        entry = header[tensor]
        dtype, shape = entry["dtype"], [int(x) for x in entry["shape"]]
        if dtype not in widths or len(shape) != 3:
            raise ValueError(f"expected stacked 3D float tensor, got {dtype} {shape}")
        n_experts, rows, cols = shape
        if not 0 <= expert < n_experts:
            raise ValueError(f"expert {expert} outside [0,{n_experts})")
        nbytes = rows * cols * widths[dtype]
        offset = 8 + header_len + int(entry["data_offsets"][0]) + expert * nbytes
        with path.open("rb") as fh:
            fh.seek(offset)
            raw = fh.read(nbytes)
        if len(raw) != nbytes:
            raise ValueError("short expert payload read")
        return _decode(raw, dtype).reshape(rows, cols), {
            "source_file": str(path),
            "dtype": dtype,
            "stacked_shape": shape,
            "expert": expert,
            "payload_bytes_read": nbytes,
        }
    raise KeyError(f"tensor not found: {tensor}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_source_activation_bridge(
    path: Path,
    *,
    expected_columns: int,
    expert: int,
) -> tuple[np.ndarray, dict]:
    """Load one sealed source MLP activation without treating it as an NR part.

    The bridge is deliberately a teacher control.  Its referenced source-layer
    receipt and activation bytes are integrity-checked so a representation
    experiment cannot accidentally substitute an arbitrary local vector and
    report it as source-bound evidence.
    """
    bridge_path = path.expanduser().resolve()
    try:
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read source activation bridge {bridge_path}: {exc}") from exc
    if not isinstance(bridge, dict):
        raise ValueError("source activation bridge must be a JSON object")
    if bridge.get("schema") != SOURCE_ACTIVATION_BRIDGE_SCHEMA:
        raise ValueError("source activation bridge schema is not accepted")
    if bridge.get("status") != "PASSED":
        raise ValueError("source activation bridge is not passed")

    source_layer = bridge.get("source_layer")
    if not isinstance(source_layer, dict):
        raise ValueError("source activation bridge has no source layer binding")
    source_receipt = source_layer.get("source_layer_receipt")
    if not isinstance(source_receipt, dict):
        raise ValueError("source activation bridge has no source layer receipt binding")
    source_receipt_path = Path(str(source_receipt.get("path") or "")).expanduser().resolve()
    source_receipt_sha256 = str(source_receipt.get("sha256") or "")
    if len(source_receipt_sha256) != 64 or not source_receipt_path.is_file():
        raise ValueError("source activation bridge has an invalid source receipt binding")
    observed_source_receipt_sha256 = _sha256_bytes(source_receipt_path.read_bytes())
    if observed_source_receipt_sha256 != source_receipt_sha256:
        raise ValueError("source activation bridge source receipt hash mismatch")
    try:
        source_receipt_doc = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("source activation bridge source receipt is unreadable") from exc
    if not isinstance(source_receipt_doc, dict) or source_receipt_doc.get("status") != "PASSED":
        raise ValueError("source activation bridge source receipt is not passed")

    activation = bridge.get("mlp_input")
    if not isinstance(activation, dict):
        raise ValueError("source activation bridge has no MLP input")
    if activation.get("dtype") != "F32_LE":
        raise ValueError("source activation bridge MLP input must be F32_LE")
    elements = int(activation.get("elements") or 0)
    if elements != expected_columns:
        raise ValueError(
            f"source activation has {elements} values; expert requires {expected_columns}"
        )
    activation_path = Path(str(activation.get("path") or "")).expanduser().resolve()
    expected_sha256 = str(activation.get("sha256") or "")
    expected_bytes = elements * np.dtype("<f4").itemsize
    if int(activation.get("bytes") or -1) != expected_bytes:
        raise ValueError("source activation bridge MLP input byte count is invalid")
    if len(expected_sha256) != 64 or not activation_path.is_file():
        raise ValueError("source activation bridge MLP input binding is invalid")
    raw = activation_path.read_bytes()
    if len(raw) != expected_bytes or _sha256_bytes(raw) != expected_sha256:
        raise ValueError("source activation bridge MLP input hash mismatch")
    values = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("source activation bridge MLP input is non-finite")

    selection = bridge.get("route_selection")
    if not isinstance(selection, dict):
        raise ValueError("source activation bridge has no route selection")
    expert_ids = selection.get("expert_ids")
    selected_weights = selection.get("selected_weights")
    if not isinstance(expert_ids, list) or not isinstance(selected_weights, list):
        raise ValueError("source activation bridge route selection is malformed")
    if len(expert_ids) != len(selected_weights) or expert not in expert_ids:
        raise ValueError("selected expert is not present in source activation route control")
    route_position = [int(item) for item in expert_ids].index(int(expert))
    route_weight = float(selected_weights[route_position])
    if not math.isfinite(route_weight):
        raise ValueError("source activation bridge route weight is non-finite")
    return values, {
        "kind": "sealed_source_layer_mlp_input",
        "bridge_path": str(bridge_path),
        "bridge_sha256": _sha256_bytes(bridge_path.read_bytes()),
        "source_layer_receipt_path": str(source_receipt_path),
        "source_layer_receipt_sha256": source_receipt_sha256,
        "activation_path": str(activation_path),
        "activation_sha256": expected_sha256,
        "activation_elements": elements,
        "route_expert": int(expert),
        "route_position": route_position,
        "route_weight": route_weight,
        "teacher_only": True,
        "standalone_nr_dependency": False,
    }


def _source_output_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("source output vectors have different shapes")
    error = reference - candidate
    reference_norm = max(float(np.linalg.norm(reference)), 1e-30)
    candidate_norm = float(np.linalg.norm(candidate))
    return {
        "count": int(reference.size),
        "relative_l2": float(np.linalg.norm(error) / reference_norm),
        "rmse": float(np.sqrt(np.mean(np.square(error, dtype=np.float64)))),
        "max_abs_error": float(np.max(np.abs(error), initial=0.0)),
        "cosine": (
            float(np.dot(reference, candidate) / (reference_norm * candidate_norm))
            if candidate_norm > 0.0
            else None
        ),
        "finite": bool(np.isfinite(candidate).all()),
    }


def _repaired_rows(
    weight: np.ndarray,
    approximation: np.ndarray,
    fractions: list[float],
    *,
    source_activation: np.ndarray | None = None,
    repair_selection: str = "weight_largest",
) -> list[dict]:
    flat_residual = (weight - approximation).reshape(-1)
    n = int(flat_residual.size)
    denom = max(float(np.linalg.norm(weight.reshape(-1))), 1e-30)
    source_output = (
        weight @ source_activation if source_activation is not None else None
    )
    if repair_selection == "weight_largest":
        selection_scores = np.abs(flat_residual)
    elif repair_selection == "activation_weighted":
        if source_activation is None:
            raise ValueError("activation-weighted sparse repair requires a source activation")
        selection_scores = np.abs(
            (weight - approximation) * source_activation.reshape(1, -1)
        ).reshape(-1)
    else:
        raise ValueError(f"unsupported sparse repair selection {repair_selection!r}")
    rows: list[dict] = []
    for fraction in fractions:
        count = min(n, max(0, int(math.ceil(n * float(fraction)))))
        if count:
            ids = np.argpartition(selection_scores, -count)[-count:]
            correction = np.zeros(n, dtype=np.float32)
            correction[ids] = _bf16(flat_residual[ids])
            error = float(np.linalg.norm(flat_residual - correction) / denom)
        else:
            correction = np.zeros(n, dtype=np.float32)
            error = float(np.linalg.norm(flat_residual) / denom)
        row = {
            "residual_fraction": float(fraction),
            "residual_entries": count,
            "residual_value_bytes": count * RESIDUAL_VALUE_BYTES,
            "residual_index_bytes": count * RESIDUAL_INDEX_BYTES,
            "relative_l2": error,
        }
        if source_output is not None:
            candidate = approximation.reshape(-1) + correction
            row["source_activation_output"] = _source_output_metrics(
                source_output,
                candidate.reshape(weight.shape) @ source_activation,
            )
        rows.append(row)
    return rows


def screen(
    weight: np.ndarray,
    ranks: list[int],
    fractions: list[float],
    *,
    source_activation: np.ndarray | None = None,
    repair_selection: str = "weight_largest",
) -> list[dict]:
    if weight.ndim != 2:
        raise ValueError(f"expected matrix, got {weight.shape}")
    rows, cols = (int(weight.shape[0]), int(weight.shape[1]))
    if source_activation is not None:
        source_activation = np.asarray(source_activation, dtype=np.float32).reshape(-1)
        if source_activation.size != cols or not np.isfinite(source_activation).all():
            raise ValueError("source activation must be finite and match the matrix columns")
    elif repair_selection == "activation_weighted":
        raise ValueError("activation-weighted sparse repair requires a source activation")
    entries = rows * cols
    _u, singular, vt = np.linalg.svd(weight, full_matrices=False)
    result: list[dict] = []
    for rank in sorted(set(ranks)):
        if not 1 <= rank <= min(rows, cols):
            raise ValueError(f"rank {rank} outside matrix range")
        # Absorb singular values into the left factor, then round both factors
        # before reconstruction.  This makes the billed BF16 factor format part
        # of the measured distortion rather than a free float32 assumption.
        left = _bf16(_u[:, :rank] * singular[:rank])
        right = _bf16(vt[:rank, :])
        approximation = left @ right
        factor_bytes = rank * (rows + cols) * FACTOR_BYTES
        factor_error = float(
            np.linalg.norm((approximation - weight).reshape(-1))
            / max(float(np.linalg.norm(weight.reshape(-1))), 1e-30)
        )
        repaired = _repaired_rows(
            weight,
            approximation,
            fractions,
            source_activation=source_activation,
            repair_selection=repair_selection,
        )
        for row in repaired:
            complete_bytes = (
                factor_bytes
                + row["residual_value_bytes"]
                + row["residual_index_bytes"]
                + METADATA_BYTES
                + DECODER_SUPPORT_BYTES
            )
            row.update({
                "rank": rank,
                "factor_bytes": factor_bytes,
                "metadata_bytes": METADATA_BYTES,
                "decoder_support_bytes": DECODER_SUPPORT_BYTES,
                "complete_bytes": complete_bytes,
                "complete_scoped_ebpw": complete_bytes * 8.0 / entries,
                "factor_only_relative_l2": factor_error,
            })
            result.append(row)
        del left, right, approximation
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--ranks", default="16,32,64,128,256")
    parser.add_argument("--fractions", default="0,0.001,0.005,0.01,0.02")
    parser.add_argument(
        "--source-activation-bridge",
        type=Path,
        help=(
            "passed hawking.flash_source_moe_bridge.v1 control to measure one sealed "
            "source MLP activation; it remains a teacher-only dependency"
        ),
    )
    parser.add_argument(
        "--repair-selection",
        choices=("weight_largest", "activation_weighted"),
        default="weight_largest",
        help="selection objective for sparse repair; activation_weighted requires a bridge",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    fractions = [float(value) for value in args.fractions.split(",") if value.strip()]
    if not ranks or any(value <= 0 for value in ranks):
        raise SystemExit("ranks must contain positive integers")
    if not fractions or any(value < 0 or value > 1 for value in fractions):
        raise SystemExit("residual fractions must lie in [0, 1]")
    weight, source = load_expert(args.spec, args.tensor, args.expert)
    source_activation = None
    source_activation_control = None
    if args.source_activation_bridge is not None:
        source_activation, source_activation_control = load_source_activation_bridge(
            args.source_activation_bridge,
            expected_columns=int(weight.shape[1]),
            expert=int(args.expert),
        )
    result = {
        "schema": "hawking.odyssey.stacked_expert_lowrank_sparse_repair.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "SOURCE_ACTIVATION_RATE_DISTORTION_ONLY"
            if source_activation_control is not None
            else "STATIC_RATE_DISTORTION_ONLY"
        ),
        "specimen": str(args.spec),
        "tensor": args.tensor,
        "source": source,
        "matrix_shape": list(weight.shape),
        "representation": {
            "base": "BF16 left/right low-rank factors with singular values absorbed into the left factor",
            "repair": "BF16 residual values plus uint32 flattened indexes",
            "repair_selection": args.repair_selection,
            "billed": {
                "factor_bytes_per_value": FACTOR_BYTES,
                "residual_value_bytes": RESIDUAL_VALUE_BYTES,
                "residual_index_bytes": RESIDUAL_INDEX_BYTES,
                "metadata_bytes": METADATA_BYTES,
                "decoder_support_bytes": DECODER_SUPPORT_BYTES,
            },
            "factor_quantization": "BF16 rounded before reconstruction",
        },
        "source_activation_control": source_activation_control,
        "rows": screen(
            weight,
            ranks,
            fractions,
            source_activation=source_activation,
            repair_selection=args.repair_selection,
        ),
        "claim_boundary": (
            "One selected source expert and complete scoped byte accounting with BF16 factor "
            "rounding. When present, the source activation is one sealed teacher control and "
            "the reported matvec error is not held-out output or capability evidence. No "
            "routing, direct execution, full-body EBPW, runtime, or TPS claim."
        ),
        "next_gate": (
            "A Pareto survivor earns a held-out routed-output discriminator only after the "
            "resource/runtime gate remains satisfied; factor and residual decode must then "
            "be parity-tested before any direct-execution claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
