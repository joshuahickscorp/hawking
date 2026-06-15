#!/usr/bin/env python3
"""Build an importance artifact for Q2_K / IQ2-style rebakes.

This consumes the per-site activation statistics produced by
``colab/mega_calibrate.py`` and emits a compact ``.npz`` with normalized
importance vectors per layer/site. It does not rewrite GGUF weights by itself;
the output is the scoring input a quantization bake can use to protect the
activation-sensitive channels while pushing weights below the Q4_K wall.

The schema is intentionally simple:

  layer_{L}_{site}_importance      float32[in_dim]
  layer_{L}_{site}_mean_abs        float32[in_dim]
  layer_{L}_{site}_max_abs         float32[in_dim]
  metadata_json                    UTF-8 JSON bytes

Importance score:

  score = normalize(mean_abs) * mean_weight + normalize(max_abs) * max_weight

The max channel term keeps rare outliers visible; the mean term keeps broad
high-energy channels protected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


SITE_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _parse_key(key: str, suffix: str):
    if not key.startswith("layer_") or not key.endswith(suffix):
        return None
    body = key[len("layer_") : -len(suffix)]
    head, _, site = body.partition("_")
    if not head.isdigit() or site not in SITE_NAMES:
        return None
    return int(head), site


def _robust_norm(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p99 = float(np.percentile(x, 99.0)) if x.size else 0.0
    denom = max(p99, float(x.max()) if x.size else 0.0, eps)
    y = np.clip(x / denom, 0.0, 1.0)
    return y.astype(np.float32)


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stats", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--mean-weight", type=float, default=0.70)
    p.add_argument("--max-weight", type=float, default=0.30)
    p.add_argument("--epsilon", type=float, default=1e-8)
    args = p.parse_args()

    if not args.stats.exists():
        print(f"[q2k-importance] missing stats file: {args.stats}", file=sys.stderr)
        return 1
    if args.mean_weight < 0.0 or args.max_weight < 0.0:
        print("[q2k-importance] weights must be non-negative", file=sys.stderr)
        return 2
    total_w = args.mean_weight + args.max_weight
    if total_w <= 0.0:
        print("[q2k-importance] at least one weight must be positive", file=sys.stderr)
        return 2

    mean_w = args.mean_weight / total_w
    max_w = args.max_weight / total_w

    payload: dict[str, np.ndarray] = {}
    rows = []
    with np.load(args.stats) as z:
        n_layers = int(np.asarray(z["n_layers"]).item()) if "n_layers" in z.files else 0
        seqs = (
            int(np.asarray(z["sequences_written"]).item())
            if "sequences_written" in z.files
            else 0
        )
        means = {}
        maxes = {}
        for key in z.files:
            parsed = _parse_key(key, "_mean_abs")
            if parsed is not None:
                means[parsed] = np.asarray(z[key], dtype=np.float32)
                continue
            parsed = _parse_key(key, "_max_abs")
            if parsed is not None:
                maxes[parsed] = np.asarray(z[key], dtype=np.float32)

    for layer, site in sorted(means.keys(), key=lambda x: (x[0], SITE_NAMES.index(x[1]))):
        mean_abs = means[(layer, site)]
        max_abs = maxes.get((layer, site))
        if max_abs is None:
            max_abs = np.zeros_like(mean_abs)
        if max_abs.shape != mean_abs.shape:
            print(
                f"[q2k-importance] shape mismatch layer={layer} site={site}: "
                f"mean={mean_abs.shape} max={max_abs.shape}",
                file=sys.stderr,
            )
            return 1
        mean_norm = _robust_norm(mean_abs, args.epsilon)
        max_norm = _robust_norm(max_abs, args.epsilon)
        importance = (mean_w * mean_norm + max_w * max_norm).astype(np.float32)
        stem = f"layer_{layer}_{site}"
        payload[f"{stem}_importance"] = importance
        payload[f"{stem}_mean_abs"] = mean_abs.astype(np.float32)
        payload[f"{stem}_max_abs"] = max_abs.astype(np.float32)
        rows.append(
            {
                "layer": layer,
                "site": site,
                "channels": int(importance.size),
                "p50": float(np.percentile(importance, 50.0)),
                "p90": float(np.percentile(importance, 90.0)),
                "p99": float(np.percentile(importance, 99.0)),
                "max": float(importance.max()) if importance.size else 0.0,
            }
        )

    if not rows:
        print("[q2k-importance] no per-site stats found", file=sys.stderr)
        return 1

    metadata = {
        "schema": "q2k-importance-v1",
        "model": args.model,
        "stats": str(args.stats),
        "n_layers": n_layers,
        "sequences_calibrated": seqs,
        "mean_weight": mean_w,
        "max_weight": max_w,
        "rows": rows,
    }
    payload["metadata_json"] = np.frombuffer(
        json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
        dtype=np.uint8,
    )
    _write_npz_atomic(args.out, payload)
    print(
        f"[q2k-importance] wrote {args.out} with {len(rows)} layer/site vectors "
        f"from {seqs} sequences",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
