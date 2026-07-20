#!/usr/bin/env python3.12
"""F1-tier falsification of the function-aware codec on REAL Qwen3-235B expert tensors.

Runs the exact test SUBBIT_CLOSURE_PROGRAM.json prescribes for M01/M02, on real weights streamed
from the 438 GiB sharded checkpoint - no synthetic stand-in:

    "Measure codeword occupancy per gate/up tensor before and after. If the single-codeword share
     does not fall from 94 percent below 20 percent, or if tensor reconstruction error does not
     improve at the identical exact rate, the lever is dead and goes to the atlas."

Every arm carries the IDENTICAL index rate. The scale-invariant arms additionally ship one bf16
scale per row, reported as `extra_bpw_row_scale` and folded into the whole-model ledger by
qwen_subbit_ledger, so no arm can hide a rate advantage.

This is F1/F2 evidence (tensor and expert tier). It selects nothing. Only a real parent-vs-packed
forward may select a frontier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import qwen_function_aware_codec as FAC  # noqa: E402
from qwen_real_forward import SafetensorsIndexReader  # noqa: E402

SCHEMA = "hawking.gravity.function_aware_probe.v1"
DEFAULT_SOURCE = "models/qwen3-235b-a22b"

# The F1 candidate that collapsed 6/6 at 0.4930 complete BPW: shared grammar, gate/up d16 k1024
# stages 1, down d64 k1024 stages 1. The probe re-fits exactly these geometries.
SPECS = {
    "gate": {"dim": 16, "k": 1024, "stages": 1},
    "up": {"dim": 16, "k": 1024, "stages": 1},
    "down": {"dim": 64, "k": 1024, "stages": 1},
}
_SUFFIX = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def load_cluster(reader: SafetensorsIndexReader, layer: int, organ: str,
                 experts: list[int]) -> list[np.ndarray]:
    suffix = _SUFFIX[organ]
    return [reader.bf16(f"model.layers.{layer}.mlp.experts.{e}.{suffix}.weight").astype(np.float32)
            for e in experts]


def row_norm_span(mats: list[np.ndarray]) -> dict[str, float]:
    """The claimed geometry: gate/up row norms span 1e-5 .. 0.91. Verified, not assumed."""
    n = np.concatenate([np.linalg.norm(m, axis=1) for m in mats])
    nz = n[n > 0]
    return {"min": float(n.min()), "max": float(n.max()),
            "p50": float(np.percentile(n, 50)),
            "decades": float(np.log10(nz.max() / nz.min())) if nz.size else 0.0}


def probe(source: str, layers: list[int], organs: list[str], n_experts: int,
          seed: int, iters: int) -> dict[str, Any]:
    reader = SafetensorsIndexReader(source)
    if not reader.source_present():
        raise SystemExit(f"source shards absent under {source}; probe refuses to run on metadata")
    experts = list(range(n_experts))
    cells: list[dict[str, Any]] = []
    for layer in layers:
        for organ in organs:
            t0 = time.time()
            mats = load_cluster(reader, layer, organ, experts)
            spec = SPECS[organ]
            res = FAC.compare(mats, dim=spec["dim"], k=spec["k"], stages=spec["stages"],
                              seed=seed, iters=iters)
            res["cell"] = {"layer": layer, "organ": organ, "n_experts": n_experts,
                           "shape": list(mats[0].shape)}
            res["row_norm_span"] = row_norm_span(mats)
            res["load_and_fit_seconds"] = round(time.time() - t0, 1)
            cells.append(res)
            del mats
            FAC.gf._torch()  # keep the device handle warm; frees nothing by itself
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
    base = [c["arms"]["baseline_raw_lloyd"] for c in cells]
    inv = [c["arms"]["scale_invariant"] for c in cells]
    agg = {
        "mean_single_codeword_share_baseline": round(float(np.mean(
            [a["mean_single_codeword_share"] for a in base])), 6),
        "mean_single_codeword_share_scale_invariant": round(float(np.mean(
            [a["mean_single_codeword_share"] for a in inv])), 6),
        "mean_rel_error_baseline": round(float(np.mean([a["mean_rel_error"] for a in base])), 6),
        "mean_rel_error_scale_invariant": round(float(np.mean(
            [a["mean_rel_error"] for a in inv])), 6),
    }
    agg["gate_single_codeword_below_0.20"] = bool(
        agg["mean_single_codeword_share_scale_invariant"] < 0.20)
    agg["gate_error_improves_at_identical_rate"] = bool(
        agg["mean_rel_error_scale_invariant"] < agg["mean_rel_error_baseline"])
    agg["error_reduction_fraction"] = round(
        1.0 - agg["mean_rel_error_scale_invariant"] / max(agg["mean_rel_error_baseline"], 1e-12), 6)
    agg["verdict"] = ("LEVER_ALIVE" if agg["gate_single_codeword_below_0.20"]
                      and agg["gate_error_improves_at_identical_rate"] else "LEVER_DEAD")
    return {"schema": SCHEMA, "evidence_class": "F1_TENSOR_F2_EXPERT",
            "parent": "qwen3-235b-a22b-instruct-2507", "source": source,
            "claim": "weight-space and occupancy evidence only; no capability is claimed",
            "aggregate": agg, "cells": cells}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Function-aware codec probe on real Qwen tensors.")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--layers", default="0,46,93")
    ap.add_argument("--organs", default="gate,up,down")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    out = probe(args.source, [int(x) for x in args.layers.split(",")],
                args.organs.split(","), args.experts, args.seed, args.iters)
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(json.dumps(out["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
