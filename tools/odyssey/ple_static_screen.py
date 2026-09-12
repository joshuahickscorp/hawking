#!/usr/bin/env python3
"""Bounded static discriminator for Qwen4-Exp PLE n-gram tables.

PLE is a large vocabulary-indexed table, not an MoE matrix.  Loading the
entire organ just to discover that a cheap codec is incoherent would consume
nearly the whole machine.  This tool reads fixed, evenly spaced contiguous
windows directly from one safetensors payload, then reports only static
rate-distortion and covariance signals.  It cannot establish capability,
complete EBPW, direct execution, or a resident body.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
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
DEFAULT_TENSOR = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"


def _dtype_bytes(dtype: str) -> int:
    return {"BF16": 2, "F16": 2, "F32": 4}[dtype]


def _decode(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").copy()
    raise ValueError(f"unsupported PLE dtype {dtype}")


def sampled_windows(rows: int, windows: int, rows_per_window: int) -> list[tuple[int, int]]:
    if rows <= 0 or windows <= 0 or rows_per_window <= 0:
        raise ValueError("rows, windows, and rows_per_window must be positive")
    if rows_per_window > rows:
        raise ValueError("rows_per_window exceeds tensor rows")
    max_start = rows - rows_per_window
    starts = np.linspace(0, max_start, num=windows, dtype=np.int64)
    return [(int(s), int(rows_per_window)) for s in starts]


def load_windows(spec: Path, tensor: str, windows: int, rows_per_window: int) -> tuple[np.ndarray, dict]:
    for path in sorted(spec.glob("*.safetensors")):
        header, header_len = _read_header_ex(str(path))
        if tensor not in header:
            continue
        entry = header[tensor]
        dtype, shape = entry["dtype"], [int(x) for x in entry["shape"]]
        if dtype not in {"BF16", "F16", "F32"} or len(shape) != 2:
            raise ValueError(f"expected 2D float PLE tensor, got {dtype} {shape}")
        rows, cols = shape
        windows_at = sampled_windows(rows, windows, rows_per_window)
        element_bytes = _dtype_bytes(dtype)
        start = int(entry["data_offsets"][0])
        payload_base = 8 + header_len + start
        parts: list[np.ndarray] = []
        with path.open("rb") as fh:
            for row_start, nrows in windows_at:
                fh.seek(payload_base + row_start * cols * element_bytes)
                raw = fh.read(nrows * cols * element_bytes)
                if len(raw) != nrows * cols * element_bytes:
                    raise ValueError("short payload read")
                parts.append(_decode(raw, dtype).reshape(nrows, cols))
        return np.concatenate(parts, axis=0), {
            "source_file": str(path), "dtype": dtype, "shape": shape,
            "windows": [{"row_start": s, "rows": n} for s, n in windows_at],
        }
    raise KeyError(f"tensor not found: {tensor}")


def _effective_rank(x: np.ndarray) -> dict[str, float | int]:
    centered = x.astype(np.float32, copy=True)
    centered -= centered.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, centered.shape[0])
    eig = np.maximum(np.linalg.eigvalsh(cov.astype(np.float64)), 0.0)[::-1]
    total = float(eig.sum())
    p = eig / max(total, 1e-30)
    positive = p > 0
    participation = float(math.exp(-(p[positive] * np.log(p[positive])).sum()))
    return {
        "columns": int(x.shape[1]),
        "participation": participation,
        "ratio": participation / x.shape[1],
        "rank_90": int((np.cumsum(eig) / max(total, 1e-30) < 0.9).sum()) + 1,
    }


def _relative_l2(weight: np.ndarray, kind: str) -> float:
    if kind == "binary":
        scale = np.mean(np.abs(weight), axis=1, keepdims=True)
        recon = np.where(weight >= 0, 1.0, -1.0).astype(np.float32) * scale
    elif kind == "ternary":
        scale = np.maximum(np.max(np.abs(weight), axis=1, keepdims=True), np.finfo(np.float32).tiny)
        recon = np.where(weight > scale * 0.5, 1.0, np.where(weight < -scale * 0.5, -1.0, 0.0)) * scale
    else:
        raise ValueError(kind)
    return float(np.linalg.norm((recon - weight).reshape(-1)) / max(np.linalg.norm(weight.reshape(-1)), 1e-30))


def screen(sample: np.ndarray) -> dict:
    norms = np.linalg.norm(sample, axis=1)
    rng = np.random.default_rng(17)
    null = rng.standard_normal(sample.shape, dtype=np.float32)
    null *= norms[:, None] / np.maximum(np.linalg.norm(null, axis=1, keepdims=True), 1e-30)
    observed, null_rank = _effective_rank(sample), _effective_rank(null)
    return {
        "sample_rows": int(sample.shape[0]),
        "columns": int(sample.shape[1]),
        "row_norm": {"mean": float(norms.mean()), "std": float(norms.std()), "min": float(norms.min()), "max": float(norms.max())},
        "within_table_covariance": observed,
        "norm_matched_gaussian_null": null_rank,
        "covariance_deficit_pct": 100.0 * (float(null_rank["participation"]) - float(observed["participation"])) / max(float(null_rank["participation"]), 1e-30),
        "static_relative_l2": {"binary_per_row": _relative_l2(sample, "binary"), "ternary_per_row": _relative_l2(sample, "ternary")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--rows-per-window", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample, source = load_windows(args.spec, args.tensor, args.windows, args.rows_per_window)
    result = {
        "schema": "hawking.odyssey.ple_static_screen.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC_SAMPLE_ONLY",
        "specimen": str(args.spec),
        "tensor": args.tensor,
        "sampling": source,
        "result": screen(sample),
        "claim_boundary": "Deterministic sampled PLE windows only. No full-table result, codec artifact, complete EBPW, activation/capability, direct execution, runtime, or TPS claim.",
        "next_gate": "A positive static signal earns a held-out PLE codec discriminator; a negative signal rules out only the tested linear/per-row base families.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
