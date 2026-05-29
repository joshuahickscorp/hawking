#!/usr/bin/env python3
"""AWQ smoothing-factor calculator (per-channel adaptive path).

Sibling to ``tools/training/awq_calibrate.py``. Same input contract (the
``per_site_activation_stats.npz`` produced by ``colab/mega_calibrate.py``),
same output schema (``awq-smoothing-v1``), but uses a per-channel adaptive
strategy instead of a fixed ``alpha=0.5`` everywhere.

Why a separate script
---------------------
The original calibrate uses ``f = mean_abs ** alpha`` with one global ``alpha``
for every channel of every site. That's the AWQ-paper heuristic, but on Qwen-
class models the outlier channels are heavily clustered in the first ~10
layers (see ``memory/composition_decision_matrix_2026_05_26.md``). Channels
with very large mean_abs benefit from more aggressive smoothing (higher
alpha); channels at the noise floor regress when over-smoothed.

This script computes per-channel alpha based on outlier severity, so the
loudest channels get pushed harder onto the weight side than the quiet
channels. The output JSON is wire-compatible with the existing runtime
loader.

Strategies
----------
``--mode heuristic-fixed`` (default fallback)
    Identical to the original ``tools/training/awq_calibrate.py``:
    ``f = mean_abs ** alpha`` with one global alpha.

``--mode adaptive-alpha``
    Per-channel alpha: ``alpha_c = alpha_floor + (alpha_ceil - alpha_floor) *
    clip(z_c, 0, 1)``, where ``z_c`` is the channel's rank in the
    site's mean_abs distribution (0 = quietest, 1 = loudest).
    Defaults: ``alpha_floor=0.3``, ``alpha_ceil=0.7``.

``--mode outlier-clipped``
    Like ``heuristic-fixed`` but clamps the smoothing factor at the 99th
    percentile of the site's mean_abs distribution before pow, preventing a
    single mega-outlier channel from getting a runaway factor.

CLI
---
    python colab/awq_per_channel_calibrate.py \\
        --stats artifacts/calibration/qwen3b_corpus/per_site_activation_stats.npz \\
        --out   profiles/qwen3b_awq_per_channel.json \\
        --mode  adaptive-alpha

CPU only. numpy + stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


SITE_NAMES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _parse_layer_site(key: str) -> Tuple[int, str] | None:
    if not key.startswith("layer_") or not key.endswith("_mean_abs"):
        return None
    body = key[len("layer_"):-len("_mean_abs")]
    head, _, tail = body.partition("_")
    if not head.isdigit() or not tail or tail not in SITE_NAMES:
        return None
    return int(head), tail


def _factor_fixed(mean_abs: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
    x = np.maximum(np.asarray(mean_abs, dtype=np.float64), 0.0)
    f = np.ones_like(x) if alpha == 0.0 else np.power(x, alpha)
    return np.maximum(f, epsilon).astype(np.float32)


def _factor_outlier_clipped(
    mean_abs: np.ndarray, alpha: float, epsilon: float, clip_percentile: float
) -> np.ndarray:
    x = np.maximum(np.asarray(mean_abs, dtype=np.float64), 0.0)
    if x.size > 0:
        cap = np.percentile(x, clip_percentile)
        x = np.minimum(x, cap)
    f = np.ones_like(x) if alpha == 0.0 else np.power(x, alpha)
    return np.maximum(f, epsilon).astype(np.float32)


def _factor_adaptive_alpha(
    mean_abs: np.ndarray, alpha_floor: float, alpha_ceil: float, epsilon: float
) -> np.ndarray:
    x = np.maximum(np.asarray(mean_abs, dtype=np.float64), 0.0)
    if x.size == 0:
        return x.astype(np.float32)
    # Per-channel rank in [0, 1]: 0 = quietest, 1 = loudest.
    order = np.argsort(x)
    ranks = np.empty_like(x)
    ranks[order] = np.linspace(0.0, 1.0, num=x.size)
    # Per-channel alpha bias toward higher when channel is loud.
    alpha_per = alpha_floor + (alpha_ceil - alpha_floor) * ranks
    # Vectorized: f_c = x_c ** alpha_c. Avoid 0**0 == 1 for quiet channels.
    safe_x = np.where(x > 0.0, x, 1.0)
    f = np.where(x > 0.0, np.power(safe_x, alpha_per), 1.0)
    return np.maximum(f, epsilon).astype(np.float32)


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=("heuristic-fixed", "adaptive-alpha", "outlier-clipped"),
        default="adaptive-alpha",
    )
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Used by heuristic-fixed and outlier-clipped modes.")
    p.add_argument("--alpha-floor", type=float, default=0.3,
                   help="adaptive-alpha: alpha for the quietest channel.")
    p.add_argument("--alpha-ceil", type=float, default=0.7,
                   help="adaptive-alpha: alpha for the loudest channel.")
    p.add_argument("--clip-percentile", type=float, default=99.0,
                   help="outlier-clipped: percentile cap on mean_abs.")
    p.add_argument("--epsilon", type=float, default=1e-5)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    args = p.parse_args()

    if not args.stats.exists():
        print(f"[awq-per-channel] ERROR: stats not found: {args.stats}",
              file=sys.stderr)
        return 1

    print(f"[awq-per-channel] loading {args.stats}", file=sys.stderr)
    factors_out: dict = {}
    sequences_calibrated = 0
    n_layers_seen = 0
    with np.load(args.stats) as z:
        for key in z.files:
            if key == "sequences_written":
                sequences_calibrated = int(np.asarray(z[key]).item())
                continue
            parsed = _parse_layer_site(key)
            if parsed is None:
                continue
            layer, site = parsed
            n_layers_seen = max(n_layers_seen, layer + 1)
            mean_abs = np.asarray(z[key])
            if args.mode == "heuristic-fixed":
                f = _factor_fixed(mean_abs, args.alpha, args.epsilon)
            elif args.mode == "outlier-clipped":
                f = _factor_outlier_clipped(
                    mean_abs, args.alpha, args.epsilon, args.clip_percentile
                )
            else:  # adaptive-alpha
                f = _factor_adaptive_alpha(
                    mean_abs, args.alpha_floor, args.alpha_ceil, args.epsilon
                )
            factors_out[f"layer_{layer}_{site}"] = f.tolist()

    # Aggregate stats for downstream visibility (matches awq_calibrate.py).
    all_vals = np.concatenate([np.asarray(v, dtype=np.float64) for v in factors_out.values()]) \
        if factors_out else np.array([], dtype=np.float64)
    outliers_per_layer: list[int] = []
    if factors_out and all_vals.size > 0:
        ceiling = np.percentile(all_vals, 99.0)
        for layer in range(n_layers_seen):
            count = 0
            for site in SITE_NAMES:
                k = f"layer_{layer}_{site}"
                if k in factors_out:
                    arr = np.asarray(factors_out[k], dtype=np.float64)
                    count += int((arr > ceiling).sum())
            outliers_per_layer.append(count)

    payload = {
        "schema": "awq-smoothing-v1",
        "model": args.model,
        "method": f"per-channel:{args.mode}",
        "alpha": args.alpha if args.mode != "adaptive-alpha"
                 else 0.5 * (args.alpha_floor + args.alpha_ceil),
        "alpha_floor": args.alpha_floor if args.mode == "adaptive-alpha" else None,
        "alpha_ceil": args.alpha_ceil if args.mode == "adaptive-alpha" else None,
        "clip_percentile": args.clip_percentile if args.mode == "outlier-clipped" else None,
        "epsilon": args.epsilon,
        "n_layers": n_layers_seen,
        "sequences_calibrated": sequences_calibrated,
        "stats": {
            "min_factor": float(all_vals.min()) if all_vals.size else None,
            "max_factor": float(all_vals.max()) if all_vals.size else None,
            "mean_factor": float(all_vals.mean()) if all_vals.size else None,
            "outlier_channels_per_layer": outliers_per_layer,
        },
        "smoothing_factors": factors_out,
    }
    _write_json_atomic(args.out, payload)
    print(
        f"[awq-per-channel] wrote {args.out} mode={args.mode} "
        f"n_factors={len(factors_out)} sequences={sequences_calibrated}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
