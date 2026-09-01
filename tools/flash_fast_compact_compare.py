#!/usr/bin/env python3
"""Seal parity and source-read reduction for grouped Flash fast-chain runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def load_rows(path: Path) -> tuple[dict, dict[int, dict]]:
    doc = json.loads(path.read_text())
    rows: dict[int, dict] = {}
    for group in doc.get("groups", []):
        for row in group.get("layers", []):
            if isinstance(row, dict) and "layer" in row:
                rows[int(row["layer"])] = row
        if group.get("kind") == "full_attention" and group.get("receipt"):
            receipt_path = Path(group["receipt"])
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                layer = int(receipt.get("layer", group.get("start_layer", -1)))
                final = (receipt.get("parity") or {}).get("final_hyperconnection") or {}
                routes = (receipt.get("parity") or {}).get("route_ids") or {}
                output = (receipt.get("source") or {}).get("output_state") or {}
                rows[layer] = {
                    "layer": layer,
                    "full_attention": True,
                    "output_sha256": output.get("sha256"),
                    "final_within_tolerance": final.get("within_tolerance") is True,
                    "route_ids_observed": routes.get("observed"),
                    "route_ids_match": routes.get("match") is True,
                    "source_bytes_read": sum(int(t.get("bytes") or 0) for t in (receipt.get("source") or {}).get("tensors", [])),
                    "gpu_ns": [int(receipt.get("execution", {}).get("gpu_ns") or 0)],
                }
    return doc, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dense", type=Path)
    ap.add_argument("compact", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args()
    dense_doc, dense = load_rows(ns.dense)
    compact_doc, compact = load_rows(ns.compact)
    layers = sorted(set(dense) & set(compact))
    comparisons = []
    for layer in layers:
        left, right = dense[layer], compact[layer]
        comparisons.append({
            "layer": layer,
            "dense_output_sha256": left.get("output_sha256"),
            "compact_output_sha256": right.get("output_sha256"),
            "exact_fingerprint_match": left.get("output_sha256") == right.get("output_sha256"),
            "full_attention_tolerance_match": left.get("final_within_tolerance") and right.get("final_within_tolerance"),
            "dense_route_ids": left.get("route_ids_observed"),
            "compact_route_ids": right.get("route_ids_observed"),
            "route_ids_match": left.get("route_ids_observed") == right.get("route_ids_observed"),
        })
    dense_bytes = sum(int(row.get("source_bytes_read") or 0) for row in dense.values())
    compact_bytes = sum(int(row.get("source_bytes_read") or 0) for row in compact.values())
    dense_wall = int(dense_doc.get("elapsed_wall_ns") or 0)
    compact_wall = int(compact_doc.get("elapsed_wall_ns") or 0)
    dense_gpu = sum(sum(int(x or 0) for x in (row.get("gpu_ns") or [])) for row in dense.values())
    compact_gpu = sum(sum(int(x or 0) for x in (row.get("gpu_ns") or [])) for row in compact.values())
    span = f"layers {min(layers)}..{max(layers)}" if layers else "no layers"
    passed = bool(layers) and len(layers) == len(dense) == len(compact) and all(
        (c["full_attention_tolerance_match"] if dense[c["layer"]].get("full_attention")
         else c["exact_fingerprint_match"])
        and c["route_ids_match"] for c in comparisons
    )
    first_route_divergence = next((c["layer"] for c in comparisons if not c["route_ids_match"]), None)
    first_exact_divergence = next((c["layer"] for c in comparisons if not c["exact_fingerprint_match"]), None)
    report = {
        "schema": "hawking.flash.fast_compact_parity.v1",
        "status": "PASSED_FAST_COMPACT_PARITY" if passed else "BLOCKED_FAST_COMPACT_PARITY",
        "dense_receipt": str(ns.dense),
        "dense_receipt_sha256": hashlib.sha256(ns.dense.read_bytes()).hexdigest(),
        "compact_receipt": str(ns.compact),
        "compact_receipt_sha256": hashlib.sha256(ns.compact.read_bytes()).hexdigest(),
        "layers": layers,
        "comparisons": comparisons,
        "divergence": {
            "first_route_id_mismatch_layer": first_route_divergence,
            "first_exact_fingerprint_mismatch_layer": first_exact_divergence,
            "interpretation": "A route mismatch after a tolerance-only seam is a promotion blocker: compact mode must preserve route decisions or carry a capability-sensitive residual/fallback."
            if first_route_divergence is not None else "No route mismatch in the bounded comparison.",
        },
        "source_payload": {
            "dense_bytes": dense_bytes,
            "compact_bytes": compact_bytes,
            "bytes_saved": dense_bytes - compact_bytes,
            "reduction_fraction": (dense_bytes - compact_bytes) / dense_bytes if dense_bytes else None,
        },
        "timing": {
            "dense_wall_ns": dense_wall,
            "compact_wall_ns": compact_wall,
            "wall_reduction_fraction": (dense_wall - compact_wall) / dense_wall if dense_wall else None,
            "dense_gpu_ns": dense_gpu,
            "compact_gpu_ns": compact_gpu,
            "gpu_reduction_fraction": (dense_gpu - compact_gpu) / dense_gpu if dense_gpu else None,
            "interpretation": f"bounded device-resident cross-species prefix ({span}); full-attention seam uses source-parity tolerance, no complete-token/TPS claim",
        },
        "claim_boundary": (
            f"Per-layer compact route-before-payload execution matched the dense device-resident control where routes remained stable ({span}); full-attention seams were checked within established source-parity tolerance. This is bounded cross-species prefix evidence, not complete-model TPS, EBPW, or residency promotion."
            if passed else
            "Compact and dense fast-chain fingerprints or route IDs diverged; promotion is prohibited."
        ),
        "next": "If route IDs diverge, retain the negative control and test capability-sensitive residuals or route-conditioned representations before extending the range.",
        "bench": {
            "state": "UNKNOWN",
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "recorded_by": "tools/flash_fast_compact_compare.py",
            "machine": "Apple M3 Ultra",
            "rule": "bounded parity/source-read comparison only; no TPS or residency claim",
        },
    }
    report["seal_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "source_payload": report["source_payload"], "timing": report["timing"], "layers": len(layers)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
