#!/usr/bin/env python3
"""AWQ smoothing-factor calculator (heuristic path).

Consumes the per-site activation aggregates produced by
``colab/mega_calibrate.py`` (``per_site_activation_stats.npz``) and emits a JSON
artifact of per-channel smoothing factors for use at model load time by
dismantle's Rust loader. The output schema is ``awq-smoothing-v1``.

Reference: Lin et al. 2023, "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration" (arXiv:2306.00978).

For each linear with input activation ``X`` and weight ``W``:

  1. ``s_x = mean(|X|, dim=(0,1))``  — per-channel activation magnitude.
     (This is the ``layer_{L}_{site}_mean_abs`` array we already captured.)
  2. Find a per-channel exponent ``s ∈ [0, 1]`` minimizing
     ``|| W @ X − (W * s_x^s) @ (X / s_x^s) ||`` under quantization of the
     smoothed weight.
  3. Smoothing factor ``f = s_x ** s``. At runtime: ``X' = X / f``,
     ``W' = W * f``. Math is invariant in fp; the trick is that ``W'`` is now
     easier to int-quantize because outliers moved off ``X`` and onto ``W``.

THIS SCRIPT TAKES THE HEURISTIC PATH:
  We do NOT perform the per-channel grid search over ``s`` because that
  requires the actual GGUF-quantized weights and a quantize+matmul loop. That
  step is heavy and model-aware. Instead we use the AWQ paper's typical
  default ``f = max(s_x ** alpha, epsilon)`` with ``alpha = 0.5``. This is the
  "AWQ-lite" / heuristic regime documented in the paper's ablation. Quality is
  typically within ~0.1-0.3 perplexity of full-search AWQ on Qwen-class models.

  TODO (future work): port the model-aware search step. It would need to:
    * Load Qwen-3B fp16 weights from GGUF (or HF).
    * For each (layer, site), grid-search ``s ∈ linspace(0, 1, 20)``.
    * For each candidate ``s``: build ``W' = W * s_x^s``, quantize ``W'`` to
      int8 (or Q4_K group-wise), recompute ``W'_q @ (X_sample / s_x^s)``
      against ``W @ X_sample`` on a held-out calibration batch, pick the
      ``s`` minimizing reconstruction MSE.
    * That requires either a Python forward pass with the real weights or a
      Rust-side calibration entry point. Too heavy for this scaffold.

CLI:
  python tools/training/awq_calibrate.py \\
      --stats artifacts/calibration/qwen3b_corpus/per_site_activation_stats.npz \\
      --out   profiles/qwen3b_awq_smoothing.json \\
      --alpha 0.5

CPU only. numpy + stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    """Parse ``layer_{L}_{site}_mean_abs`` → (L, site).

    Returns ``None`` if the key doesn't fit the expected pattern.
    """
    if not key.startswith("layer_") or not key.endswith("_mean_abs"):
        return None
    # Strip prefix/suffix: layer_{L}_{site}_mean_abs
    body = key[len("layer_"):-len("_mean_abs")]
    # body looks like "{L}_{site}", but site may itself contain underscores
    # (e.g. q_proj, gate_proj, down_proj).
    head, _, tail = body.partition("_")
    if not head.isdigit() or not tail:
        return None
    try:
        layer = int(head)
    except ValueError:
        return None
    if tail not in SITE_NAMES:
        return None
    return layer, tail


def compute_smoothing_factor(
    mean_abs: np.ndarray,
    alpha: float,
    epsilon: float,
) -> np.ndarray:
    """Heuristic AWQ smoothing factor.

    ``f = max(mean_abs ** alpha, epsilon)`` clamped to be strictly positive.
    Output dtype is float32. Shape matches input (per-channel, length=in_dim).
    """
    if mean_abs.ndim != 1:
        raise ValueError(
            f"expected 1-D per-channel mean_abs, got shape {mean_abs.shape}"
        )
    x = np.asarray(mean_abs, dtype=np.float64)
    # Clip pre-pow so a never-active channel (mean_abs ≈ 0) doesn't collapse
    # to 0 (which would div-by-zero at runtime) — keep it at epsilon^(1/alpha)
    # equivalent, but simpler to clamp after.
    x = np.maximum(x, 0.0)
    if alpha == 0.0:
        # f = 1.0 everywhere — degenerate to "no smoothing".
        f = np.ones_like(x, dtype=np.float64)
    else:
        f = np.power(x, alpha)
    f = np.maximum(f, epsilon)
    return f.astype(np.float32)


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute heuristic AWQ smoothing factors from "
                    "mega_calibrate.py per-site activation stats."
    )
    p.add_argument(
        "--stats",
        type=Path,
        default=Path("artifacts/calibration/qwen3b_corpus/"
                     "per_site_activation_stats.npz"),
        help="Input .npz from mega_calibrate.py.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("profiles/qwen3b_awq_smoothing.json"),
        help="Output JSON path.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="AWQ smoothing exponent in [0.0, 1.0]. Paper default 0.5.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model id (metadata only; not loaded).",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=1e-5,
        help="Floor on smoothing factor to avoid div-by-zero at runtime on "
             "never-active channels.",
    )
    args = p.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print(
            f"[awq] ERROR: --alpha {args.alpha} out of range [0.0, 1.0]",
            file=sys.stderr,
        )
        return 2
    if args.epsilon <= 0.0:
        print(
            f"[awq] ERROR: --epsilon {args.epsilon} must be > 0",
            file=sys.stderr,
        )
        return 2

    if not args.stats.exists():
        print(
            f"[awq] ERROR: stats file not found: {args.stats}",
            file=sys.stderr,
        )
        return 1

    print(
        "[awq] WARNING: this is the HEURISTIC AWQ path (f = mean_abs ** "
        f"alpha, alpha={args.alpha}). Full per-channel search over the "
        "quantization-error landscape is NOT performed — see the script "
        "docstring TODO.",
        file=sys.stderr,
    )

    print(f"[awq] loading {args.stats}", file=sys.stderr)
    with np.load(args.stats) as z:
        keys = list(z.files)
        n_layers = (
            int(np.asarray(z["n_layers"]).item())
            if "n_layers" in z.files else 0
        )
        sequences_written = (
            int(np.asarray(z["sequences_written"]).item())
            if "sequences_written" in z.files else 0
        )
        # Pull only the per-site mean_abs arrays. We deliberately ignore
        # _max_abs / _sum_abs / _count here — heuristic AWQ only needs mean.
        per_site_mean: Dict[Tuple[int, str], np.ndarray] = {}
        for key in keys:
            parsed = _parse_layer_site(key)
            if parsed is None:
                continue
            layer, site = parsed
            arr = np.asarray(z[key], dtype=np.float32)
            per_site_mean[(layer, site)] = arr

    if not per_site_mean:
        print(
            f"[awq] ERROR: no layer_*_{{site}}_mean_abs arrays found in "
            f"{args.stats}; keys present: {keys[:10]}{'...' if len(keys) > 10 else ''}",
            file=sys.stderr,
        )
        return 1

    # Sanity: every (layer, site) pair should have the same in_dim across
    # the 4 attention sites (q/k/v/o all act on hidden_dim). MLP sites act on
    # hidden or intermediate; just record what we see, don't reject.
    smoothing_factors: Dict[str, List[float]] = {}
    outlier_per_layer: Dict[int, int] = {}
    factor_min = float("inf")
    factor_max = float("-inf")
    factor_sum = 0.0
    factor_count = 0

    # Iterate in a stable order (layer asc, then site in SITE_NAMES order)
    layers_present = sorted({l for (l, _) in per_site_mean.keys()})
    for layer in layers_present:
        outlier_per_layer[layer] = 0
        # Collect all factors in this layer for median-based outlier count
        layer_factors: List[np.ndarray] = []
        for site in SITE_NAMES:
            key = (layer, site)
            if key not in per_site_mean:
                continue
            mean_abs = per_site_mean[key]
            f = compute_smoothing_factor(mean_abs, args.alpha, args.epsilon)
            smoothing_factors[f"layer_{layer}_{site}"] = f.tolist()
            layer_factors.append(f)
            factor_min = min(factor_min, float(f.min()))
            factor_max = max(factor_max, float(f.max()))
            factor_sum += float(f.sum())
            factor_count += int(f.size)
        if layer_factors:
            all_f = np.concatenate(layer_factors)
            med = float(np.median(all_f))
            if med > 0.0:
                outlier_per_layer[layer] = int(np.sum(all_f > 10.0 * med))

    if factor_count == 0:
        print(
            "[awq] ERROR: parsed mean_abs arrays but produced zero factors",
            file=sys.stderr,
        )
        return 1

    mean_factor = factor_sum / float(factor_count)
    inferred_n_layers = max(n_layers, max(layers_present) + 1)

    payload = {
        "schema": "awq-smoothing-v1",
        "model": args.model,
        "n_layers": int(inferred_n_layers),
        "alpha": float(args.alpha),
        "method": "heuristic",
        "sequences_calibrated": int(sequences_written),
        "smoothing_factors": smoothing_factors,
        "stats": {
            "max_factor": factor_max,
            "min_factor": factor_min,
            "mean_factor": mean_factor,
            "outlier_channels_per_layer": [
                outlier_per_layer.get(l, 0) for l in range(inferred_n_layers)
            ],
        },
    }

    _write_json_atomic(args.out, payload)

    total_sites = len(smoothing_factors)
    print(
        f"[awq] computed {total_sites} (layer, site) factors across "
        f"{len(layers_present)} layers; mean factor={mean_factor:.4f} "
        f"min={factor_min:.4g} max={factor_max:.4g}; "
        f"outliers (>10× layer-median): "
        f"total={sum(outlier_per_layer.values())} "
        f"per_layer={[outlier_per_layer.get(l, 0) for l in range(inferred_n_layers)]}",
        file=sys.stderr,
    )
    print(
        f"[awq] sequences_calibrated={sequences_written} "
        f"n_layers={inferred_n_layers} alpha={args.alpha} → {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
