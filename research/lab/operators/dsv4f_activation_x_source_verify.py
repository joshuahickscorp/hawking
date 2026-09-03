"""Verify a DSV4F source capture with the unchanged doctor6 collector."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    collect_expert_activations,
)
from lab.operators.dsv4f_activation_capture import load_capture_result


def n_fit_from_result(cap: dict[str, Any]) -> dict[str, Any]:
    return dict(cap.get("bounded_storage", {}).get("n_fit_distribution") or {})


def verify_doctor6(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cap = load_capture_result(run_dir)
    stacked, prov = collect_expert_activations(run_dir, cap)
    if not stacked:
        raise SystemExit("doctor6 yielded zero (layer, expert) arrays")
    hidden = int(cap["runtime_binding"]["hidden"])
    finite = True
    sample_key = None
    sample_shape = None
    for key, arr in sorted(stacked.items()):
        if arr.dtype != np.float32:
            raise SystemExit(f"organ {key} dtype {arr.dtype} != float32")
        if arr.ndim != 2 or arr.shape[1] != hidden:
            raise SystemExit(f"organ {key} shape {arr.shape} expected (?, {hidden})")
        if not np.isfinite(arr).all():
            finite = False
        if sample_key is None:
            sample_key = key
            sample_shape = [int(arr.shape[0]), int(arr.shape[1])]
    n_fit = n_fit_from_result(cap)
    source = cap.get("source_run") or {}
    report = {
        "run_dir": str(run_dir),
        "doctor6_key_count": len(stacked),
        "doctor6_layers_with_hidden_hits": prov.get("layers_with_hidden_hits"),
        "doctor6_token_expert_pairs": prov.get("token_expert_pairs"),
        "sample_organ": {
            "layer": int(sample_key[0]) if sample_key else None,
            "expert": int(sample_key[1]) if sample_key else None,
            "X_shape": sample_shape,
            "all_finite": finite,
        },
        "all_organs_finite": finite,
        "n_fit_distribution": n_fit,
        "tokens": source.get("tokens") or cap.get("capture_summary", {}).get("total_tokens"),
        "layers": source.get("layers") or cap.get("runtime_binding", {}).get("layers"),
        "execution_path": source.get("execution_path")
        or cap.get("runtime_binding", {}).get("execution_path"),
        "peak_rss_bytes": source.get("peak_rss_bytes"),
        "wall_ms": source.get("wall_ms"),
        "schema": cap.get("schema"),
        "all_layer": bool(cap.get("capture_summary", {}).get("all_layer_activation_capture")),
    }
    return report


def disk_bytes(run_dir: Path) -> int:
    total = 0
    for path in Path(run_dir).rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = verify_doctor6(args.run_dir)
    report["disk_bytes"] = disk_bytes(args.run_dir)
    n_fit = report.get("n_fit_distribution") or {}
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        "n_fit: mean={mean} min={min} p10={p10} p50={p50} p90={p90} max={max} "
        "zeros={zeros} ({pct}%) frac_ge_64={frac}".format(
            mean=n_fit.get("mean"),
            min=n_fit.get("min"),
            p10=n_fit.get("p10"),
            p50=n_fit.get("p50"),
            p90=n_fit.get("p90"),
            max=n_fit.get("max"),
            zeros=n_fit.get("count_zero"),
            pct=n_fit.get("pct_zero"),
            frac=n_fit.get("frac_at_or_above_64"),
        )
    )
    if not report["all_organs_finite"]:
        raise SystemExit("non-finite values in doctor6 stacks")
    if not math.isfinite(float(n_fit.get("mean") or 0.0)):
        raise SystemExit("n_fit mean is not finite")


if __name__ == "__main__":
    main()
