#!/usr/bin/env python3
"""Static low-rank base plus dense binary residual for one stacked expert.

This bounded Flash-Next experiment tests a composition rather than another
pure rank curve.  One expert is represented by BF16 SVD factors and a dense
per-row scaled one-bit sign stream for the residual.  Factors, residual
codes/scales, metadata, and decoder support are all billed.  It is static
rate--distortion evidence only; no activation, routing, capability, direct
execution, or runtime claim is made.
"""
from __future__ import annotations

import argparse
import json
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
FACTOR_BYTES = 2  # BF16 left/right factor values
RESIDUAL_CODE_BITS = 1
RESIDUAL_SCALE_BYTES = 2  # BF16 scale per output row
METADATA_BYTES = 128
DECODER_SUPPORT_BYTES = 65536


def _bf16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (
        (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    ).astype(np.uint16)
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def _packed_bytes(entries: int, bits: int) -> int:
    return (int(entries) * int(bits) + 7) // 8


def _binary_residual(approximation: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, int]:
    residual = weight - approximation
    scale = _bf16(
        np.maximum(np.mean(np.abs(residual), axis=1, keepdims=True), np.finfo(np.float32).tiny)
    )
    sign = np.where(residual >= 0.0, 1.0, -1.0).astype(np.float32)
    return approximation + sign * scale, int(scale.shape[0])


def screen(weight: np.ndarray, ranks: list[int]) -> list[dict]:
    if weight.ndim != 2:
        raise ValueError(f"expected a matrix, got {weight.shape}")
    rows, cols = (int(weight.shape[0]), int(weight.shape[1]))
    entries = rows * cols
    denominator = max(float(np.linalg.norm(weight.reshape(-1))), 1e-30)
    left_basis, singular, right_basis = np.linalg.svd(weight, full_matrices=False)
    results: list[dict] = []
    for rank in sorted(set(ranks)):
        if not 1 <= rank <= min(rows, cols):
            raise ValueError(f"rank {rank} outside matrix range")
        left = _bf16(left_basis[:, :rank] * singular[:rank])
        right = _bf16(right_basis[:rank, :])
        approximation = left @ right
        candidate, residual_rows = _binary_residual(approximation, weight)
        factor_bytes = rank * (rows + cols) * FACTOR_BYTES
        residual_code_bytes = _packed_bytes(entries, RESIDUAL_CODE_BITS)
        residual_scale_bytes = residual_rows * RESIDUAL_SCALE_BYTES
        complete_bytes = (
            factor_bytes
            + residual_code_bytes
            + residual_scale_bytes
            + METADATA_BYTES
            + DECODER_SUPPORT_BYTES
        )
        results.append(
            {
                "rank": rank,
                "factor_bytes": factor_bytes,
                "residual_code_bits": RESIDUAL_CODE_BITS,
                "residual_code_bytes": residual_code_bytes,
                "residual_scale_bytes": residual_scale_bytes,
                "metadata_bytes": METADATA_BYTES,
                "decoder_support_bytes": DECODER_SUPPORT_BYTES,
                "complete_bytes": complete_bytes,
                "complete_scoped_ebpw": complete_bytes * 8.0 / entries,
                "relative_l2": float(
                    np.linalg.norm((candidate - weight).reshape(-1)) / denominator
                ),
                "lowrank_base_relative_l2": float(
                    np.linalg.norm((approximation - weight).reshape(-1)) / denominator
                ),
            }
        )
        del left, right, approximation, candidate
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--ranks", default="1,2,4,8,16,32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    if not ranks or any(value <= 0 for value in ranks):
        raise SystemExit("ranks must contain positive integers")
    weight, source = load_expert(args.spec, args.tensor, args.expert)
    result = {
        "schema": "hawking.odyssey.stacked_expert_lowrank_binary_residual.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC_RATE_DISTORTION_ONLY",
        "specimen": str(args.spec),
        "tensor": args.tensor,
        "source": source,
        "matrix_shape": list(weight.shape),
        "representation": {
            "base": "BF16 left/right low-rank factors with singular values absorbed into left",
            "residual": "dense per-row BF16 mean-abs scale plus packed 1-bit sign stream",
            "billed": {
                "factor_bytes_per_value": FACTOR_BYTES,
                "residual_code_bits": RESIDUAL_CODE_BITS,
                "residual_scale_bytes": RESIDUAL_SCALE_BYTES,
                "metadata_bytes": METADATA_BYTES,
                "decoder_support_bytes": DECODER_SUPPORT_BYTES,
            },
        },
        "rows": screen(weight, ranks),
        "claim_boundary": (
            "One selected source expert and complete scoped byte accounting with BF16 factor "
            "and residual-scale rounding. No held-out output objective, routing, activation/"
            "capability, direct execution, full-body EBPW, runtime, or TPS claim."
        ),
        "next_gate": (
            "A Pareto survivor earns a held-out routed-output discriminator only after the "
            "resource/runtime gate remains satisfied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
