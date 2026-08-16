#!/usr/bin/env python3
"""Verify captured Q80 post-SwiGLU intermediates against sealed BF16 weights.

For each (layer, expert) pair: load captured router-input X (2048) and captured
post-SwiGLU H (512), recompute

    h_expected = silu(X @ W_gate.T) * (X @ W_up.T)

from the sealed source tensors, and report max-abs and cosine. Config
``hidden_act`` is ``silu``; the runtime uses ``silu_mul`` = silu(gate)*up.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab.operators.ascension_dual_gravity_worker import _mean_row_cosine
from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    collect_expert_activations,
    silu,
)
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

DEFAULT_MODEL = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/"
    "qwen-80b/Qwen3-Coder-Next"
)


def _pairs_from_swiglu(by_h, n: int = 3) -> list[tuple[int, int]]:
    ranked = sorted(
        ((k, int(v.shape[0])) for k, v in by_h.items()),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )
    picked: list[tuple[int, int]] = []
    used: set[int] = set()
    for (layer, expert), _rows in ranked:
        if layer in used:
            continue
        picked.append((layer, expert))
        used.add(layer)
        if len(picked) >= n:
            return picked
    for (layer, expert), _rows in ranked:
        if (layer, expert) not in picked:
            picked.append((layer, expert))
        if len(picked) >= n:
            break
    return picked


def verify_pairs(
    capture: Path,
    model_dir: Path,
    pairs: list[tuple[int, int]] | None = None,
) -> dict:
    wanted = set(pairs) if pairs else None
    by_hidden, hid_prov = collect_expert_activations(
        capture, wanted_keys=wanted, x_kind="router_input"
    )
    by_swiglu, sw_prov = collect_expert_activations(
        capture, wanted_keys=wanted, x_kind="swiglu_hidden_routed"
    )
    if pairs is None:
        pairs = _pairs_from_swiglu(by_swiglu, n=3)
    wmap = load_weight_map(model_dir)
    rows = []
    for layer, expert in pairs:
        X = np.asarray(by_hidden[(layer, expert)], dtype=np.float32)
        H = np.asarray(by_swiglu[(layer, expert)], dtype=np.float32)
        if X.shape[0] != H.shape[0]:
            raise RuntimeError(
                f"L{layer}.E{expert}: hidden rows {X.shape[0]} != swiglu rows {H.shape[0]}"
            )
        if H.shape[1] != 512:
            raise RuntimeError(f"L{layer}.E{expert}: swiglu width {H.shape[1]} != 512")
        Wg = np.asarray(
            load_tensor(
                model_dir,
                wmap,
                f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
            ),
            dtype=np.float32,
        )
        Wu = np.asarray(
            load_tensor(
                model_dir,
                wmap,
                f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
            ),
            dtype=np.float32,
        )
        h_expected = silu(X @ Wg.T) * (X @ Wu.T)
        diff = np.abs(H - h_expected)
        cos = float(_mean_row_cosine(H, h_expected))
        rec = {
            "layer": layer,
            "expert": expert,
            "rows": int(H.shape[0]),
            "max_abs_diff": float(diff.max()) if diff.size else None,
            "mean_abs_diff": float(diff.mean()) if diff.size else None,
            "cosine": cos,
            "formula": "silu(X @ W_gate.T) * (X @ W_up.T)",
            "hidden_act": "silu",
        }
        rows.append(rec)
        print(
            f"L{layer}.E{expert}  rows={H.shape[0]}  "
            f"max_abs={rec['max_abs_diff']:.6g}  "
            f"mean_abs={rec['mean_abs_diff']:.6g}  cos={cos:.8f}",
            flush=True,
        )
    return {
        "schema": "hawking.ascension.qwen80_swiglu_intermediate_verify.v1",
        "capture": str(capture),
        "model_dir": str(model_dir),
        "hidden_act": "silu",
        "formula": "silu(x @ gate_proj.T) * (x @ up_proj.T)",
        "note": (
            "Recompute uses f32 numpy matmul on BF16-widened source weights. "
            "The capture used M=1 f32 GEMV after the same BF16 widen, so "
            "differences are rounding, not a different activation."
        ),
        "pairs": rows,
        "hidden_provenance": {
            k: hid_prov.get(k) for k in ("token_expert_pairs", "x_kind")
        },
        "swiglu_provenance": {
            k: sw_prov.get(k)
            for k in ("token_expert_pairs", "x_kind", "packed_swiglu", "swiglu_width")
        },
    }


def main() -> int:
    capture = Path(os.environ["CAPTURE"])
    model_dir = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL))
    dest = Path(os.environ.get("OUT", "/tmp/q80_swiglu_intermediate_verify.json"))
    pairs = None
    if os.environ.get("LE_PAIRS"):
        pairs = []
        for tok in os.environ["LE_PAIRS"].split(","):
            a, b = tok.split(":")
            pairs.append((int(a), int(b)))
    out = verify_pairs(capture, model_dir, pairs)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
