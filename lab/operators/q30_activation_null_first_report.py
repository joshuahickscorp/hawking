#!/usr/bin/env python3
"""Report constant-mean null on a Q30 L0 route capture BEFORE any family result.

This is the headline instrument check for capture quality: if mean null stays
near ~0.94, the capture still cannot price activation-aware surplus honestly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.q30_activation_aware_family_probe import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    collect_expert_activations,
    constant_mean_null,
    holdout_split,
    load_capture,
    matvec_rows,
    sha256_file,
    silu,
    utc_now,
    SEED,
)
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map  # noqa: E402


def report_null(
    *,
    capture_run: Path,
    model_dir: Path,
    label: str,
    min_tokens: int = 32,
    components: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
    top_experts: int = 6,
) -> dict:
    capture = load_capture(capture_run)
    by_expert, prov = collect_expert_activations(capture_run, capture)
    ranked = sorted(by_expert.items(), key=lambda kv: -kv[1].shape[0])
    chosen = [(e, X) for e, X in ranked if X.shape[0] >= min_tokens][:top_experts]
    if not chosen:
        raise RuntimeError(f"no experts with >= {min_tokens} hits")

    weight_map = load_weight_map(model_dir)
    rows = []
    for expert, X_all in chosen:
        for component in components:
            name = f"model.layers.0.mlp.experts.{expert}.{component}.weight"
            W = load_tensor(model_dir, weight_map, name).astype(np.float32, copy=False)
            if component in ("gate_proj", "up_proj"):
                X_use = X_all
            else:
                Wg = load_tensor(
                    model_dir,
                    weight_map,
                    f"model.layers.0.mlp.experts.{expert}.gate_proj.weight",
                ).astype(np.float32, copy=False)
                Wu = load_tensor(
                    model_dir,
                    weight_map,
                    f"model.layers.0.mlp.experts.{expert}.up_proj.weight",
                ).astype(np.float32, copy=False)
                X_use = silu(matvec_rows(Wg, X_all)) * matvec_rows(Wu, X_all)
            comp_seed = int(__import__("hashlib").sha256(component.encode()).hexdigest()[:8], 16)
            _X_fit, X_hold = holdout_split(X_use, seed=SEED ^ (expert * 1009) ^ (comp_seed & 0xFFFF))
            y = matvec_rows(W, X_hold)
            null = constant_mean_null(y)
            # also report input-space diversity proxies
            x_mu = X_hold.mean(axis=0)
            x_centered = X_hold - x_mu
            # mean pairwise cosine of outputs vs mean — null is that; also var of y norms
            rows.append(
                {
                    "expert": int(expert),
                    "component": component,
                    "n_routed": int(X_all.shape[0]),
                    "n_hold": int(X_hold.shape[0]),
                    "null_baseline": float(null),
                    "y_row_norm_cv": float(
                        np.std(np.linalg.norm(y, axis=1))
                        / (np.mean(np.linalg.norm(y, axis=1)) + 1e-12)
                    ),
                    "x_mean_norm": float(np.linalg.norm(x_mu)),
                    "x_centered_rms": float(np.sqrt(np.mean(x_centered**2))),
                }
            )

    nulls = [r["null_baseline"] for r in rows]
    high = [r for r in rows if r["n_routed"] >= 200]
    high_nulls = [r["null_baseline"] for r in high] if high else nulls
    return {
        "schema": "hawking.ascension.qwen30_activation_null_first.v1",
        "label": label,
        "reported_at": utc_now(),
        "capture_run": str(capture_run),
        "capture_result_sha256": sha256_file(capture_run / "capture-result.json"),
        "activation_provenance": prov,
        "headline": {
            "mean_null_all_scored": float(np.mean(nulls)),
            "min_null_all_scored": float(np.min(nulls)),
            "max_null_all_scored": float(np.max(nulls)),
            "mean_null_high_hit_ge_200": float(np.mean(high_nulls)),
            "n_rows": len(rows),
            "n_high_hit_rows": len(high),
            "null_trap_threshold": 0.942,
            "materially_below_prior_null": bool(float(np.mean(high_nulls)) < 0.942 - 0.05),
            "verdict": (
                "NULL_FELL_MATERIALLY"
                if float(np.mean(high_nulls)) < 0.892
                else (
                    "NULL_STILL_HIGH_CAPTURE_STRATEGY_STILL_WRONG"
                    if float(np.mean(high_nulls)) >= 0.942 - 0.05
                    else "NULL_IMPROVED_BUT_STILL_ELEVATED"
                )
            ),
        },
        "rows": rows,
        "claim_boundary": {
            "null_only_no_family_result": True,
            "diagnostic_not_capability": True,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-run", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--min-tokens", type=int, default=32)
    args = p.parse_args()
    doc = report_null(
        capture_run=args.capture_run.expanduser().resolve(),
        model_dir=args.model_dir.expanduser().resolve(),
        label=args.label,
        min_tokens=args.min_tokens,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(doc["headline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
