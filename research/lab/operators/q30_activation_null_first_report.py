#!/usr/bin/env python3
"""Report constant-mean null on a Q30 route capture BEFORE any family result.

Works for:
  - L0-only broad / HCLI captures (historical)
  - All-layer broad captures (`...all_layer_route_capture_result.v1`)

Per-layer null is the headline for all-layer runs: layer 0 previously showed
~0.94 on the three-prompt set and ~0.41–0.59 on the broad set; deeper layers
may have a different null structure and must be stated first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (  # noqa: E402
    capture_is_all_layer,
    collect_expert_activations,
    organ_activations,
)
from lab.operators.q30_activation_aware_family_probe import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    constant_mean_null,
    holdout_split,
    load_capture,
    matvec_rows,
    sha256_file,
    utc_now,
    SEED,
)
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map  # noqa: E402


def _score_layer(
    *,
    layer: int,
    by_layer_expert: Mapping[tuple[int, int], np.ndarray],
    model_dir: Path,
    weight_map: Mapping[str, str],
    min_tokens: int,
    components: tuple[str, ...],
    top_experts: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        ((e, X) for (L, e), X in by_layer_expert.items() if L == layer),
        key=lambda kv: -kv[1].shape[0],
    )
    chosen = [(e, X) for e, X in ranked if X.shape[0] >= min_tokens][:top_experts]
    rows: list[dict[str, Any]] = []
    for expert, X_all in chosen:
        for component in components:
            name = f"model.layers.{layer}.mlp.experts.{expert}.{component}.weight"
            W = load_tensor(model_dir, weight_map, name).astype(np.float32, copy=False)
            X_use = organ_activations(
                layer=layer,
                expert=expert,
                component=component,
                X_hidden=X_all,
                model_dir=model_dir,
                weight_map=weight_map,
            )
            comp_seed = int(hashlib.sha256(component.encode()).hexdigest()[:8], 16)
            _X_fit, X_hold = holdout_split(
                X_use, seed=SEED ^ (layer * 9176) ^ (expert * 1009) ^ (comp_seed & 0xFFFF)
            )
            y = matvec_rows(W, X_hold)
            null = constant_mean_null(y)
            x_mu = X_hold.mean(axis=0)
            x_centered = X_hold - x_mu
            rows.append(
                {
                    "layer": int(layer),
                    "expert": int(expert),
                    "component": component,
                    "tensor_name": name,
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
    return rows


def _headline(rows: list[dict[str, Any]], *, prior_null: float = 0.942) -> dict[str, Any]:
    if not rows:
        return {
            "mean_null_all_scored": None,
            "min_null_all_scored": None,
            "max_null_all_scored": None,
            "mean_null_high_hit_ge_200": None,
            "n_rows": 0,
            "n_high_hit_rows": 0,
            "null_trap_threshold": prior_null,
            "materially_below_prior_null": False,
            "verdict": "NO_ELIGIBLE_ORGANS",
        }
    nulls = [r["null_baseline"] for r in rows]
    high = [r for r in rows if r["n_routed"] >= 200]
    high_nulls = [r["null_baseline"] for r in high] if high else nulls
    mean_high = float(np.mean(high_nulls))
    return {
        "mean_null_all_scored": float(np.mean(nulls)),
        "min_null_all_scored": float(np.min(nulls)),
        "max_null_all_scored": float(np.max(nulls)),
        "mean_null_high_hit_ge_200": mean_high,
        "n_rows": len(rows),
        "n_high_hit_rows": len(high),
        "null_trap_threshold": prior_null,
        "materially_below_prior_null": bool(mean_high < prior_null - 0.05),
        "verdict": (
            "NULL_FELL_MATERIALLY"
            if mean_high < 0.892
            else (
                "NULL_STILL_HIGH_CAPTURE_STRATEGY_STILL_WRONG"
                if mean_high >= prior_null - 0.05
                else "NULL_IMPROVED_BUT_STILL_ELEVATED"
            )
        ),
    }


def report_null(
    *,
    capture_run: Path,
    model_dir: Path,
    label: str,
    min_tokens: int = 32,
    components: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
    top_experts: int = 6,
    max_layers: int | None = None,
) -> dict:
    capture = load_capture(capture_run)
    by_layer_expert, prov = collect_expert_activations(capture_run, capture)
    all_layer = capture_is_all_layer(capture)
    layers = sorted({layer for layer, _ in by_layer_expert})
    if max_layers is not None:
        layers = [L for L in layers if L < int(max_layers)]
    if not layers:
        raise RuntimeError("capture has no layers with retained hidden hits")

    weight_map = load_weight_map(model_dir)
    rows: list[dict[str, Any]] = []
    per_layer: list[dict[str, Any]] = []
    for layer in layers:
        layer_rows = _score_layer(
            layer=layer,
            by_layer_expert=by_layer_expert,
            model_dir=model_dir,
            weight_map=weight_map,
            min_tokens=min_tokens,
            components=components,
            top_experts=top_experts,
        )
        rows.extend(layer_rows)
        per_layer.append(
            {
                "layer": int(layer),
                "headline": _headline(layer_rows),
                "n_experts_scored": len({r["expert"] for r in layer_rows}),
                "n_rows": len(layer_rows),
            }
        )

    overall = _headline(rows)
    return {
        "schema": "hawking.ascension.qwen30_activation_null_first.v1",
        "label": label,
        "reported_at": utc_now(),
        "capture_run": str(capture_run),
        "capture_result_sha256": sha256_file(capture_run / "capture-result.json"),
        "all_layer_capture": all_layer,
        "layers_scored": layers,
        "activation_provenance": prov,
        "headline": overall,
        "per_layer": per_layer,
        "rows": rows,
        "claim_boundary": {
            "null_only_no_family_result": True,
            "diagnostic_not_capability": True,
            "per_layer_null_precedes_any_family_result": True,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-run", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--min-tokens", type=int, default=32)
    p.add_argument("--top-experts", type=int, default=6)
    p.add_argument("--max-layers", type=int, default=None)
    args = p.parse_args()
    doc = report_null(
        capture_run=args.capture_run.expanduser().resolve(),
        model_dir=args.model_dir.expanduser().resolve(),
        label=args.label,
        min_tokens=args.min_tokens,
        top_experts=args.top_experts,
        max_layers=args.max_layers,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"headline": doc["headline"], "per_layer_summary": [
        {"layer": r["layer"], **r["headline"]} for r in doc["per_layer"]
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
