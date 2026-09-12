#!/usr/bin/env python3
"""Static ternary substrate plus sparse-repair discriminator for one MoE tensor.

This is deliberately a rate--distortion filter, not a model compressor.  It
loads one named float safetensors matrix, uses per-row symmetric ternary
quantisation, restores the largest residual entries, and bills every scoped
representation byte.  A positive row only earns activation-aware testing.
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

from tools.future.representational_anatomy import _load_float_tensor, _read_header_ex

DEFAULT_SPEC = Path("/Volumes/corpdrive/hawking-modellake/specimens/"
                    "LiquidAI--LFM2-24B-A2B@a3bbacd91a67")
DEFAULT_TENSOR = "model.layers.2.feed_forward.experts.0.w1.weight"
CODE_BITS = 2                 # packed ternary code, one state intentionally unused
ROW_SCALE_BYTES = 2           # fp16 scale per output row
RESIDUAL_VALUE_BYTES = 2      # bf16 residual value
RESIDUAL_INDEX_BYTES = 4      # uint32 flattened matrix index
METADATA_BYTES = 128
DECODER_SUPPORT_BYTES = 65536


def packed_bytes(n: int, bits: int) -> int:
    return (int(n) * int(bits) + 7) // 8


def ternary_reconstruct(weight: np.ndarray) -> np.ndarray:
    """Per-row absmax ternary reconstruction, with deterministic zero rows."""
    if weight.ndim != 2:
        raise ValueError(f"expected matrix, got {weight.shape}")
    scale = np.max(np.abs(weight), axis=1, keepdims=True) / 1.0
    scale = np.maximum(scale, np.finfo(np.float32).tiny)
    # Nearest member of {-scale, 0, +scale}; thresholds are scale / 2.
    code = np.where(weight > scale * 0.5, 1.0,
                    np.where(weight < -scale * 0.5, -1.0, 0.0))
    return (code * scale).astype(np.float32, copy=False)


def binary_reconstruct(weight: np.ndarray) -> np.ndarray:
    """Per-row least-squares scale for a sign matrix."""
    sign = np.where(weight >= 0, 1.0, -1.0).astype(np.float32)
    scale = np.mean(np.abs(weight), axis=1, keepdims=True)
    return (sign * scale).astype(np.float32, copy=False)


def repaired(weight: np.ndarray, base: np.ndarray, fraction: float) -> tuple[np.ndarray, int]:
    residual = weight - base
    k = min(residual.size, max(0, int(math.ceil(residual.size * float(fraction)))))
    if not k:
        return base, 0
    flat = residual.reshape(-1)
    ids = np.argpartition(np.abs(flat), -k)[-k:]
    out = np.array(base, copy=True).reshape(-1)
    # The declared BF16 persistent value is rounded before restoration.
    values = flat[ids].astype(np.float32)
    bits = values.view(np.uint32)
    bf16 = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    rounded = (bf16.astype(np.uint32) << 16).view(np.float32)
    out[ids] += rounded
    return out.reshape(base.shape), k


def screen(weight: np.ndarray, fractions: list[float], *, base_kind: str = "ternary") -> list[dict]:
    base = ternary_reconstruct(weight) if base_kind == "ternary" else binary_reconstruct(weight)
    code_bits = CODE_BITS if base_kind == "ternary" else 1
    n = int(weight.size)
    rows = int(weight.shape[0])
    denom = max(float(np.linalg.norm(weight.reshape(-1))), 1e-12)
    rows_out = []
    for fraction in fractions:
        candidate, k = repaired(weight, base, fraction)
        persistent = (packed_bytes(n, code_bits) + rows * ROW_SCALE_BYTES +
                      k * (RESIDUAL_VALUE_BYTES + RESIDUAL_INDEX_BYTES) +
                      METADATA_BYTES + DECODER_SUPPORT_BYTES)
        rows_out.append({
            "residual_fraction": float(fraction),
            "residual_entries": k,
            "complete_bytes": persistent,
            "complete_ebpw": persistent * 8.0 / n,
            "relative_l2": float(np.linalg.norm((candidate - weight).reshape(-1)) / denom),
        })
    return rows_out


def load(spec: Path, tensor: str) -> np.ndarray:
    for shard in sorted(spec.glob("*.safetensors")):
        header, header_len = _read_header_ex(str(shard))
        if tensor in header:
            return _load_float_tensor(str(shard), header_len, header[tensor])
    raise KeyError(f"tensor not found: {tensor}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    p.add_argument("--tensor", default=DEFAULT_TENSOR)
    p.add_argument("--fractions", default="0,0.001,0.005,0.01,0.02")
    p.add_argument("--base", choices=("ternary", "binary"), default="ternary")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fractions = [float(x) for x in a.fractions.split(",")]
    if any(x < 0 or x > 1 for x in fractions):
        raise SystemExit("residual fractions must lie in [0, 1]")
    weight = load(a.spec, a.tensor)
    result = {
        "schema": "hawking.odyssey.layerwise_sparse_repair.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC_RATE_DISTORTION_ONLY",
        "specimen": str(a.spec),
        "tensor": a.tensor,
        "shape": list(weight.shape),
        "representation": {
            "base": ("per-row symmetric ternary; packed 2-bit codes with one unused state"
                     if a.base == "ternary" else "per-row scaled binary; packed 1-bit sign codes"),
            "billed": {"row_scales_bytes": ROW_SCALE_BYTES, "residual_value_bytes": RESIDUAL_VALUE_BYTES,
                       "residual_index_bytes": RESIDUAL_INDEX_BYTES, "metadata_bytes": METADATA_BYTES,
                       "decoder_support_bytes": DECODER_SUPPORT_BYTES},
        },
        "rows": screen(weight, fractions, base_kind=a.base),
        "claim_boundary": "One frozen tensor, static relative-L2 only. No activation-aware objective, capability, direct execution, or full-model EBPW claim.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
