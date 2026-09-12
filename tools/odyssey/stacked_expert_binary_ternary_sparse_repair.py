#!/usr/bin/env python3
"""Static binary/ternary substrates plus sparse repair for one stacked expert.

This is a bounded Flash-Next rate--distortion discriminator.  It reads one
expert slice from a stacked safetensors tensor, quantizes each row to a
scale-plus-packed binary or ternary code, restores selected residual entries,
and bills every scoped representation byte.  It is not a model compressor:
there is no activation, routing, capability, direct-execution, or runtime
claim here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey.stacked_expert_lowrank_sparse_repair import load_expert


DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
DEFAULT_TENSOR = "model.language_model.layers.1.mlp.experts.gate_up_proj"
ROW_SCALE_BYTES = 2  # BF16 scale per output row
RESIDUAL_VALUE_BYTES = 2  # BF16 correction value
RESIDUAL_INDEX_BYTES = 4  # uint32 flattened matrix index
METADATA_BYTES = 128
DECODER_SUPPORT_BYTES = 65536


def _bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to the billed BF16 values."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (
        (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    ).astype(np.uint16)
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def _packed_bytes(entries: int, bits: int) -> int:
    return (int(entries) * int(bits) + 7) // 8


def _substrate(weight: np.ndarray, base: str) -> np.ndarray:
    if weight.ndim != 2:
        raise ValueError(f"expected a matrix, got {weight.shape}")
    if base == "binary":
        scale = np.mean(np.abs(weight), axis=1, keepdims=True)
        code = np.where(weight >= 0.0, 1.0, -1.0).astype(np.float32)
    elif base == "ternary":
        scale = np.max(np.abs(weight), axis=1, keepdims=True)
        code = np.where(
            weight > scale * 0.5,
            1.0,
            np.where(weight < -scale * 0.5, -1.0, 0.0),
        ).astype(np.float32)
    else:
        raise ValueError(f"unsupported base {base!r}")
    scale = _bf16(np.maximum(scale, np.finfo(np.float32).tiny))
    return code * scale


def _repair(weight: np.ndarray, base: np.ndarray, fraction: float) -> tuple[np.ndarray, int]:
    residual = (weight - base).reshape(-1)
    count = min(residual.size, max(0, int(math.ceil(residual.size * fraction))))
    if count == 0:
        return base, 0
    ids = np.argpartition(np.abs(residual), -count)[-count:]
    # Sort the selected indexes so equal-magnitude ties have a stable encoded
    # order without paying for a full sort of the residual vector.
    ids = np.sort(ids)
    correction = np.zeros(residual.size, dtype=np.float32)
    correction[ids] = _bf16(residual[ids])
    return (base.reshape(-1) + correction).reshape(weight.shape), count


def screen(weight: np.ndarray, base: str, fractions: list[float]) -> list[dict]:
    rows, cols = (int(weight.shape[0]), int(weight.shape[1]))
    entries = rows * cols
    substrate = _substrate(weight, base)
    denominator = max(float(np.linalg.norm(weight.reshape(-1))), 1e-30)
    bits = 1 if base == "binary" else 2
    rows_out: list[dict] = []
    for fraction in fractions:
        candidate, repaired_entries = _repair(weight, substrate, fraction)
        complete_bytes = (
            _packed_bytes(entries, bits)
            + rows * ROW_SCALE_BYTES
            + repaired_entries * (RESIDUAL_VALUE_BYTES + RESIDUAL_INDEX_BYTES)
            + METADATA_BYTES
            + DECODER_SUPPORT_BYTES
        )
        rows_out.append(
            {
                "residual_fraction": float(fraction),
                "residual_entries": repaired_entries,
                "packed_code_bytes": _packed_bytes(entries, bits),
                "row_scale_bytes": rows * ROW_SCALE_BYTES,
                "residual_value_bytes": repaired_entries * RESIDUAL_VALUE_BYTES,
                "residual_index_bytes": repaired_entries * RESIDUAL_INDEX_BYTES,
                "metadata_bytes": METADATA_BYTES,
                "decoder_support_bytes": DECODER_SUPPORT_BYTES,
                "complete_bytes": complete_bytes,
                "complete_scoped_ebpw": complete_bytes * 8.0 / entries,
                "relative_l2": float(
                    np.linalg.norm((candidate - weight).reshape(-1)) / denominator
                ),
            }
        )
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--bases", default="binary,ternary")
    parser.add_argument("--fractions", default="0,0.001,0.005,0.01,0.02")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bases = [value.strip() for value in args.bases.split(",") if value.strip()]
    fractions = [float(value) for value in args.fractions.split(",") if value.strip()]
    if not bases or any(value not in {"binary", "ternary"} for value in bases):
        raise SystemExit("bases must contain binary and/or ternary")
    if not fractions or any(value < 0.0 or value > 1.0 for value in fractions):
        raise SystemExit("residual fractions must lie in [0, 1]")
    weight, source = load_expert(args.spec, args.tensor, args.expert)
    result = {
        "schema": "hawking.odyssey.stacked_expert_binary_ternary_sparse_repair.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC_RATE_DISTORTION_ONLY",
        "specimen": str(args.spec),
        "tensor": args.tensor,
        "source": source,
        "matrix_shape": list(weight.shape),
        "representation": {
            "bases": bases,
            "binary": "per-row BF16 mean-abs scale plus packed 1-bit sign code",
            "ternary": "per-row BF16 absmax scale plus packed 2-bit {-1,0,+1} code",
            "repair": "largest residual entries, BF16 value plus uint32 flattened index",
            "billed": {
                "row_scale_bytes": ROW_SCALE_BYTES,
                "residual_value_bytes": RESIDUAL_VALUE_BYTES,
                "residual_index_bytes": RESIDUAL_INDEX_BYTES,
                "metadata_bytes": METADATA_BYTES,
                "decoder_support_bytes": DECODER_SUPPORT_BYTES,
            },
        },
        "curves": {
            base: screen(weight, base, fractions) for base in bases
        },
        "claim_boundary": (
            "One selected source expert and complete scoped byte accounting with BF16 row "
            "scales. No held-out output objective, routing, activation/capability, direct "
            "execution, full-body EBPW, runtime, or TPS claim."
        ),
        "next_gate": (
            "Only a Pareto survivor with a valid runtime/resource path earns a held-out "
            "routed-output discriminator."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
