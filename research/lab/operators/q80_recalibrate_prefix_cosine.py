#!/usr/bin/env python3
"""Holdout organ cosine of prefix-truncated packed r160 down_proj factors.

Scores the runtime rank-cap (zero mid[k:]) against BF16 holdout X, so generation
points labeled down_prefix_rK carry a measured cosine, not the fresh-SVD rK
number from the 588 grid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab.operators.ascension_dual_gravity_worker import (  # noqa: E402
    MAGIC_ACT_SVD,
    _decode_uniform_body,
    _parse_container,
)
from lab.operators.q80_mixed_representation_pack import (  # noqa: E402
    CaptureHiddens,
    post_swiglu,
    read_catalog,
)
from lab.operators.q80_subbit_capability_curve import (  # noqa: E402
    ARTIFACT_1P44,
    CAPTURE,
    HOLD_FRAC,
    MIN_HOLDOUT_ROWS,
    MODEL_DIR,
    ROW_SEED,
    collect_x_for_pairs,
    holdout_split,
    output_cosine,
)
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map  # noqa: E402

RANKS = (160, 80, 40, 20, 16, 8)


def decode_factors(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    header, body = _parse_container(payload, expected_magic=MAGIC_ACT_SVD)
    left_bytes = int(header["left_body_bytes"])
    left = _decode_uniform_body(dict(header["left"]), body[:left_bytes])
    right = _decode_uniform_body(dict(header["right"]), body[left_bytes:])
    return np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curve", type=Path, default=REPO / "receipts/ascent-2026-08-16/q80-subbit-capability-curve.json")
    p.add_argument("--artifact", type=Path, default=ARTIFACT_1P44)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--capture", type=Path, default=CAPTURE)
    p.add_argument("--max-pairs", type=int, default=24)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO / "receipts/ascent-2026-08-16/q80-recalibrate-prefix-cosines.json",
    )
    args = p.parse_args()

    curve = json.loads(args.curve.read_text())
    pairs = [r for r in curve["pair_rows"] if r.get("has_holdout")]
    # Prefer screen organs, then busiest, then the rest.
    rank_role = {"screen_down_busy": 0, "screen_gate_up_busy": 1, "busiest": 2, "median": 3}
    pairs.sort(key=lambda r: (rank_role.get(r.get("role"), 9), r["layer"], r["expert"]))
    pairs = pairs[: int(args.max_pairs)]
    print(f"[prefix] scoring {len(pairs)} holdout pairs", flush=True)

    catalog = read_catalog(args.artifact / "catalog.hq80m15")
    rec_by_name = {r["name"]: r for r in catalog["records"]}
    seg_by_id = {s["id"]: s for s in catalog["segments"]}
    wmap = load_weight_map(args.model_dir)
    caps = CaptureHiddens(args.capture)
    # collect_x wants the same pair dicts
    by_x = collect_x_for_pairs(args.capture, pairs)

    acc: dict[int, list[float]] = {k: [] for k in RANKS}
    used = []
    for i, pair in enumerate(pairs, 1):
        layer, expert = int(pair["layer"]), int(pair["expert"])
        name = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
        rec = rec_by_name.get(name)
        X = by_x.get((layer, expert))
        if rec is None or X is None or int(X.shape[0]) < MIN_HOLDOUT_ROWS:
            print(f"[prefix] skip L{layer}.E{expert} rec={rec is not None} X={None if X is None else X.shape}", flush=True)
            continue
        fit_idx, hold_idx, has = holdout_split(int(X.shape[0]), hold_frac=HOLD_FRAC, seed=ROW_SEED ^ (layer * 1009 + expert))
        if not has:
            continue
        seg = seg_by_id[rec["segment_id"]]
        payload = (args.artifact / "segments" / seg["filename"]).read_bytes()[
            rec["offset"] : rec["offset"] + rec["nbytes"]
        ]
        left, right = decode_factors(payload)
        w_gate = np.asarray(
            load_tensor(args.model_dir, wmap, f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"),
            dtype=np.float32,
        )
        w_up = np.asarray(
            load_tensor(args.model_dir, wmap, f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"),
            dtype=np.float32,
        )
        w_down = np.asarray(load_tensor(args.model_dir, wmap, name), dtype=np.float32)
        X_hold = np.ascontiguousarray(X[hold_idx], dtype=np.float32)
        X_sw = post_swiglu(X_hold, w_gate, w_up)
        for k in RANKS:
            hat = left[:, :k] @ right[:k, :]
            acc[k].append(output_cosine(w_down, hat, X_sw))
        used.append({"layer": layer, "expert": expert, "role": pair.get("role"), "n_hold": int(X_hold.shape[0])})
        print(f"[prefix] {i}/{len(pairs)} L{layer}.E{expert} r160={acc[160][-1]:.4f} r8={acc[8][-1]:.4f}", flush=True)

    summary = {
        str(k): {
            "n": len(vals),
            "mean": None if not vals else float(np.mean(vals)),
            "p10": None if not vals else float(np.percentile(vals, 10)),
            "p50": None if not vals else float(np.percentile(vals, 50)),
            "min": None if not vals else float(min(vals)),
        }
        for k, vals in acc.items()
    }
    payload = {
        "schema": "hawking.ascent.q80_recalibrate_prefix_cosine.v1",
        "n_pairs": len(used),
        "pairs": used,
        "ranks": summary,
        "mean_by_rank": {int(k): summary[k]["mean"] for k in summary},
        "note": "prefix of packed quantized r160 factors; runtime rank-cap does this",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["mean_by_rank"], indent=2), flush=True)
    print(f"[prefix] {args.out}", flush=True)
    # silence unused
    _ = caps
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
