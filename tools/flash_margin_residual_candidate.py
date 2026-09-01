#!/usr/bin/env python3
"""Construct and score a bounded margin-aware router residual NR.

The residual is intentionally oracle-derived from the already captured dense
and compact layer-3 states.  This proves the accounting and the seam gate,
not generalization: a future implementation must learn a state-dependent
residual and pass native layer-3/4 organ parity before promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc")


def load_router(root: Path, layer: int) -> tuple[np.ndarray, str]:
    name = f"model.language_model.layers.{layer}.mlp.gate.weight"
    idx = json.loads((root / "model.safetensors.index.json").read_text())
    shard = root / idx["weight_map"][name]
    with shard.open("rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(hlen))
        meta = header[name]
        begin, end = meta["data_offsets"]
        fh.seek(8 + hlen + begin)
        raw = fh.read(end - begin)
    vals = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (vals << 16).view("<f4").reshape(meta["shape"]), str(shard)


def topk(x: np.ndarray) -> np.ndarray:
    return np.argsort(-x, axis=1, kind="stable")[:, :10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=MODEL)
    ap.add_argument("--dense", type=Path, default=ROOT / "receipts/headless/FLASH_DENSE_L0_L7_V1/layer-3/state.f32")
    ap.add_argument("--compact", type=Path, default=ROOT / "receipts/headless/FLASH_COMPACT_L0_L7_V1/layer-3/state.f32")
    ap.add_argument("--out", type=Path, default=ROOT / "receipts/headless/FLASH_MARGIN_RESIDUAL_CANDIDATE_L3_L4.json")
    ap.add_argument("--state-out", type=Path, default=None, help="optional oracle conditional-repair state artifact for native seam probing")
    args = ap.parse_args()
    started = time.perf_counter_ns()
    dense = np.fromfile(args.dense, dtype="<f4").reshape(-1, 2560)
    compact = np.fromfile(args.compact, dtype="<f4").reshape(-1, 2560)
    router, shard = load_router(args.root.resolve(), 4)
    delta = dense - compact
    dense_logits = dense @ router.T
    compact_logits = compact @ router.T
    dense_ids = topk(dense_logits)
    dense_sorted = np.sort(dense_logits, axis=1)[:, ::-1]
    margins = dense_sorted[:, 9] - dense_sorted[:, 10]
    # Contribution of each hidden coordinate to the observed router-logit repair.
    salience = np.abs(delta[:, :, None] * router.T[None, :, :]).sum(axis=(0, 2))
    order = np.argsort(-salience)
    rows = []
    for fraction in (0.001, 0.0025, 0.005, 0.01, 0.02):
        dimensions = max(1, int(round(2560 * fraction)))
        keep = order[:dimensions]
        repaired = compact.copy()
        repaired[:, keep] += delta[:, keep]
        logits = repaired @ router.T
        ids = topk(logits)
        diffs = [len(set(a.tolist()) ^ set(b.tolist())) for a, b in zip(dense_ids, ids)]
        order_diffs = [int(np.sum(a != b)) for a, b in zip(dense_ids, ids)]
        rows.append({
            "policy": "oracle_salience_fixed_coordinate_residual",
            "fraction": fraction,
            "dimensions": dimensions,
            "residual_bytes_fp16": int(dense.shape[0] * dimensions * 2),
            "residual_bytes_fp32": int(dense.shape[0] * dimensions * 4),
            "rows_with_top10_membership_change": int(sum(v > 0 for v in diffs)),
            "mean_top10_membership_symmetric_difference": float(np.mean(diffs)),
            "rows_with_top10_order_change": int(sum(v > 0 for v in order_diffs)),
            "mean_top10_order_difference": float(np.mean(order_diffs)),
            "router_logit_rmse": float(np.sqrt(np.mean((dense_logits - logits) ** 2))),
            "hidden_rmse": float(np.sqrt(np.mean((dense - repaired) ** 2))),
            "min_margin_rows_below_1e-5": int(np.sum(margins < 1e-5)),
            "route_stable_on_observed_seam": not any(diffs),
        })
    # A conditional policy spends the larger residual only on a low-margin row.
    small = order[: max(1, int(round(2560 * 0.005)))]
    large = order[: max(1, int(round(2560 * 0.02)))]
    conditional = compact.copy()
    for row, margin in enumerate(margins):
        keep = large if margin < 1e-5 else small
        conditional[row, keep] += delta[row, keep]
    cond_ids = topk(conditional @ router.T)
    cond_diffs = [len(set(a.tolist()) ^ set(b.tolist())) for a, b in zip(dense_ids, cond_ids)]
    doc = {
        "schema": "hawking.flash.margin_residual_candidate.v1",
        "status": "ORACLE_NR_CANDIDATE_UNPROMOTED",
        "artifact_kind": "NR",
        "model": "Qwen3.8-Flash-Next",
        "seam": {"source_layer": 3, "router_layer": 4, "positions": int(dense.shape[0]), "hidden": 2560, "dense": str(args.dense), "compact": str(args.compact)},
        "router": {"tensor": "model.language_model.layers.4.mlp.gate.weight", "shard": shard, "sha256": hashlib.sha256(Path(shard).read_bytes()).hexdigest(), "dense_min_top10_top11_margin": float(np.min(margins)), "margin_threshold": 1e-5},
        "candidate_rows": rows,
        "conditional_policy": {"policy": "0.5% salience residual normally; 2% when dense top10/top11 margin < 1e-5", "residual_bytes_fp16_upper_bound": int(dense.shape[0] * len(large) * 2), "rows_with_membership_change": int(sum(v > 0 for v in cond_diffs)), "mean_membership_symmetric_difference": float(np.mean(cond_diffs))},
        "claim_boundary": "oracle-derived seam candidate only; residual is not learned or generalized, and no native organ parity, complete-token, capability, TPS, EBPW, or residency claim is made",
        "promotion_allowed": False,
        "next_gate": "learn/compile state-dependent residual, then run exact layer-3/4 native organ parity and route-stability tests",
        "bench": {"state": "UNKNOWN", "measurement_state": "MEASURED_ORACLE_CANDIDATE", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": "Apple host CPU; existing state artifacts plus BF16 router", "rule": "S032 §3 -- oracle residual diagnostic, not execution timing", "elapsed_ns": time.perf_counter_ns() - started},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    if args.state_out is not None:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        conditional.astype("<f4").tofile(args.state_out)
        doc["conditional_policy"]["state_artifact"] = str(args.state_out)
        doc["conditional_policy"]["state_sha256"] = hashlib.sha256(args.state_out.read_bytes()).hexdigest()
        args.out.write_text(json.dumps(doc, indent=2) + "\n")
    nr = {"nr_version": "1.0.0", "nr_kind": "hawking.nos.noetic_representation", "artifact_kind": "NR", "semantic_provenance": {"parent_model": "Qwen3.8-Flash-Next", "source_state_sha256": {"dense": hashlib.sha256(args.dense.read_bytes()).hexdigest(), "compact": hashlib.sha256(args.compact.read_bytes()).hexdigest()}}, "representation": {"scope": "layer-3 output residual targeting layer-4 router", "candidates": rows, "conditional_policy": doc["conditional_policy"], "runtime_required": False}, "dependencies": [], "kernel_requirements": [{"requires": "margin_aware_residual_apply", "parameters": "state-dependent coordinate/codebook selection"}], "verifier": "exact layer-3/4 native organ parity and route stability", "seal": {"status": "UNSEALED_ORACLE_CANDIDATE", "promotion_allowed": False}, "bench": {"state": "UNKNOWN", "rule": "NR diagnostic only"}}
    args.out.with_suffix(".nr.json").write_text(json.dumps(nr, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "rows": rows, "conditional": doc["conditional_policy"], "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
