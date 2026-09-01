#!/usr/bin/env python3
"""Seal dense-vs-route-union parity and source-read reduction."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def rows(path: Path) -> dict[tuple[int, int], dict]:
    doc = json.loads(path.read_text())
    default_layer = int(doc.get("layer", -1))
    return {
        (int(row.get("layer", default_layer)), int(row["step"])): row
        for row in doc.get("steps", [])
        if isinstance(row, dict) and "step" in row and ("layer" in row or default_layer >= 0)
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dense", type=Path)
    ap.add_argument("union", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args()
    dense_doc = json.loads(ns.dense.read_text())
    union_doc = json.loads(ns.union.read_text())
    dense_rows, union_rows = rows(ns.dense), rows(ns.union)
    keys = sorted(set(dense_rows) & set(union_rows))
    comparisons = []
    for key in keys:
        left, right = dense_rows[key], union_rows[key]
        comparisons.append({
            "layer": key[0], "step": key[1], "token_id": left.get("token_id"),
            "dense_final_state_sha256": left.get("final_state_sha256"),
            "union_final_state_sha256": right.get("final_state_sha256"),
            "exact_fingerprint_match": left.get("final_state_sha256") == right.get("final_state_sha256"),
        })
    dense_bytes = int((dense_doc.get("execution") or {}).get("source_payload_bytes_read") or 0)
    union_bytes = int((union_doc.get("execution") or {}).get("source_payload_bytes_read") or 0)
    reduction = (dense_bytes - union_bytes) / dense_bytes if dense_bytes else None
    dense_wall = sum(int(row.get("wall_ns") or 0) for row in dense_rows.values())
    union_wall = sum(int(row.get("wall_ns") or 0) for row in union_rows.values())
    dense_gpu = sum(int(row.get("gpu_ns") or 0) for row in dense_rows.values())
    union_gpu = sum(int(row.get("gpu_ns") or 0) for row in union_rows.values())
    passed = bool(comparisons) and len(comparisons) == len(dense_rows) == len(union_rows) and all(c["exact_fingerprint_match"] for c in comparisons)
    report = {
        "schema": "hawking.flash.route_union_parity.v1",
        "status": "PASSED_ROUTE_UNION_PARITY" if passed else "BLOCKED_ROUTE_UNION_PARITY",
        "dense_receipt": str(ns.dense),
        "dense_receipt_sha256": hashlib.sha256(ns.dense.read_bytes()).hexdigest(),
        "union_receipt": str(ns.union),
        "union_receipt_sha256": hashlib.sha256(ns.union.read_bytes()).hexdigest(),
        "comparisons": comparisons,
        "source_payload": {
            "dense_bytes": dense_bytes,
            "route_union_bytes": union_bytes,
            "bytes_saved": dense_bytes - union_bytes,
            "reduction_fraction": reduction,
        },
        "timing": {
            "dense_wall_ns": dense_wall,
            "route_union_wall_ns": union_wall,
            "wall_reduction_fraction": (dense_wall - union_wall) / dense_wall if dense_wall else None,
            "dense_gpu_ns": dense_gpu,
            "route_union_gpu_ns": union_gpu,
            "gpu_reduction_fraction": (dense_gpu - union_gpu) / dense_gpu if dense_gpu else None,
            "interpretation": "bounded two-token prefix comparison; includes source preparation and execution, not a complete-token benchmark",
        },
        "claim_boundary": (
            "The dense and route-union compact executors produced identical final-state fingerprints for every audited layer/token pair. The route union is exact for this bounded token window; this is not complete-model TPS, EBPW, or residency evidence."
            if passed else
            "Route-union fingerprints did not match the dense audit; compact promotion is prohibited."
        ),
        "next": "Extend route-union planning across full Flash species seams, then compare complete accepted-token wall time.",
        "bench": {"state": "UNKNOWN", "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "recorded_by": "tools/flash_route_union_compare.py", "machine": "Apple M3 Ultra", "rule": "parity and source-read comparison only; no TPS or residency claim"},
    }
    report["seal_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "source_payload": report["source_payload"], "timing": report["timing"], "comparisons": len(comparisons)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
