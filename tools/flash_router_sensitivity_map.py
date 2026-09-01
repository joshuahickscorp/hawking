#!/usr/bin/env python3
"""Bounded layer-3 -> layer-4 router sensitivity diagnostic.

This is the narrow experiment required by the second-order reduction brief:
use the dense and compact layer-3 states already on disk, apply the real
layer-4 router, and quantify which state directions control top-k membership
and margins.  It is diagnostic evidence for a future router-sensitive NR; it
does not claim that masking or residual repair is already a valid executable.
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


def read_bf16(root: Path, name: str) -> tuple[np.ndarray, str, int]:
    idx = json.loads((root / "model.safetensors.index.json").read_text())
    shard = root / idx["weight_map"][name]
    with shard.open("rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(hlen))
        meta = header[name]
        begin, end = meta["data_offsets"]
        fh.seek(8 + hlen + begin)
        payload = fh.read(end - begin)
    if meta["dtype"] != "BF16":
        raise ValueError((name, meta["dtype"]))
    vals = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
    return (vals << 16).view("<f4").reshape(meta["shape"]), str(shard), int(end - begin)


def topk(logits: np.ndarray, k: int = 10) -> np.ndarray:
    # Stable descending order makes ties explicit rather than platform-dependent.
    return np.argsort(-logits, axis=1, kind="stable")[:, :k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=MODEL)
    ap.add_argument("--dense", type=Path, default=ROOT / "receipts/headless/FLASH_DENSE_L0_L7_V1/layer-3/state.f32")
    ap.add_argument("--compact", type=Path, default=ROOT / "receipts/headless/FLASH_COMPACT_L0_L7_V1/layer-3/state.f32")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "receipts/headless/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json")
    args = ap.parse_args()
    started = time.perf_counter_ns()
    dense = np.fromfile(args.dense, dtype="<f4")
    compact = np.fromfile(args.compact, dtype="<f4")
    if dense.size != compact.size or dense.size % 2560:
        raise ValueError((dense.size, compact.size))
    dense = dense.reshape(-1, 2560)
    compact = compact.reshape(-1, 2560)
    name = f"model.language_model.layers.{args.layer}.mlp.gate.weight"
    router, shard, source_bytes = read_bf16(args.root.resolve(), name)
    delta = compact - dense
    dense_logits = dense @ router.T
    compact_logits = compact @ router.T
    dense_ids = topk(dense_logits)
    compact_ids = topk(compact_logits)
    dense_set = [set(row.tolist()) for row in dense_ids]
    compact_set = [set(row.tolist()) for row in compact_ids]
    membership_changed = [len(a ^ b) for a, b in zip(dense_set, compact_set)]
    dense_sorted = np.sort(dense_logits, axis=1)[:, ::-1]
    compact_sorted = np.sort(compact_logits, axis=1)[:, ::-1]
    dense_margin = dense_sorted[:, 9] - dense_sorted[:, 10]
    compact_margin = compact_sorted[:, 9] - compact_sorted[:, 10]
    # The router's right singular vectors are the state directions it can see.
    _, singular, vt = np.linalg.svd(router.astype(np.float32), full_matrices=False)
    total_delta = max(float(np.sum(delta * delta)), 1e-30)
    subspace = []
    for rank in (8, 16, 32, 64, 128, 256, 512):
        basis = vt[:rank]
        projected = (delta @ basis.T) @ basis
        repaired = compact - projected
        repaired_ids = topk(repaired @ router.T)
        repaired_changed = [len(set(a.tolist()) ^ set(b.tolist())) for a, b in zip(dense_ids, repaired_ids)]
        subspace.append({"rank": rank, "delta_energy_fraction": float(np.sum(projected * projected) / total_delta), "projected_rel_norm": float(np.linalg.norm(projected) / np.linalg.norm(delta)), "oracle_projected_repair_bytes_fp16": int(dense.shape[0] * rank * 2), "oracle_repaired_rows_with_membership_change": int(sum(v > 0 for v in repaired_changed)), "oracle_repaired_mean_topk_symmetric_difference": float(np.mean(repaired_changed))})
    # Coordinate salience is a cheap residual-budget proxy: preserve dimensions
    # that contribute most to the observed router-logit delta.
    contribution = np.abs(delta[:, :, None] * router.T[None, :, :]).sum(axis=(0, 2))
    order = np.argsort(-contribution)
    coordinate = []
    for fraction in (0.001, 0.005, 0.01, 0.02, 0.05):
        count = max(1, int(round(2560 * fraction)))
        keep = order[:count]
        masked = np.zeros_like(delta)
        masked[:, keep] = delta[:, keep]
        # Repair starts from the compact state; the selected slice adds back
        # the observed dense-minus-compact delta.
        masked_logits = compact_logits + masked @ router.T
        ids = topk(masked_logits)
        changed = [len(set(a.tolist()) ^ set(b.tolist())) for a, b in zip(dense_ids, ids)]
        coordinate.append({"fraction": fraction, "dimensions": count, "residual_bytes_f32": int(count * dense.shape[0] * 4), "logit_delta_l2_fraction": float(np.linalg.norm(masked @ router.T) / max(np.linalg.norm((delta @ router.T)), 1e-30)), "mean_topk_symmetric_difference": float(np.mean(changed)), "rows_with_membership_change": int(sum(v > 0 for v in changed))})
    doc = {
        "schema": "hawking.flash.router_sensitivity_map.v1",
        "status": "MEASURED_SEAM_DIAGNOSTIC",
        "artifact_kind": "NR_DIAGNOSTIC",
        "model": "Qwen3.8-Flash-Next",
        "seam": {"preceding_layer": 3, "router_layer": args.layer, "dense_state": str(args.dense), "compact_state": str(args.compact), "positions": int(dense.shape[0]), "hidden": 2560},
        "router_source": {"tensor": name, "shard": shard, "tensor_bytes": source_bytes, "sha256": hashlib.sha256(Path(shard).read_bytes()).hexdigest()},
        "routing": {"dense_top10": dense_ids.tolist(), "compact_top10": compact_ids.tolist(), "top10_membership_symmetric_difference": membership_changed, "rows_with_membership_change": int(sum(v > 0 for v in membership_changed)), "dense_top10_top11_margin": dense_margin.tolist(), "compact_top10_top11_margin": compact_margin.tolist(), "dense_margin_min": float(np.min(dense_margin)), "compact_margin_min": float(np.min(compact_margin))},
        "delta": {"l2": float(np.linalg.norm(delta)), "max_abs": float(np.max(np.abs(delta))), "router_logit_l2": float(np.linalg.norm(delta @ router.T)), "router_logit_max_abs": float(np.max(np.abs(delta @ router.T))), "router_singular_values_head": singular[:32].tolist(), "router_visible_subspace": subspace},
        "coordinate_salience": coordinate,
        "claim_boundary": "bounded diagnostic of the observed layer-3 compact error and layer-4 router; no residual-repair parity, native execution, complete-token, capability, TPS, EBPW, or promotion claim",
        "promotion_allowed": False,
        "next_gate": "fit a margin-aware router-sensitive residual NR and test exact layer-3/4 organ parity; retain dense router at high fidelity",
        "bench": {"state": "UNKNOWN", "measurement_state": "MEASURED_SEAM_DIAGNOSTIC", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": "Apple host CPU; existing state artifacts plus BF16 router", "rule": "S032 §3 -- seam diagnostic, not execution timing", "elapsed_ns": time.perf_counter_ns() - started},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "positions": dense.shape[0], "rows_with_membership_change": doc["routing"]["rows_with_membership_change"], "dense_margin_min": doc["routing"]["dense_margin_min"], "compact_margin_min": doc["routing"]["compact_margin_min"], "router_logit_l2": doc["delta"]["router_logit_l2"], "visible_subspace": subspace, "coordinate": coordinate, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
