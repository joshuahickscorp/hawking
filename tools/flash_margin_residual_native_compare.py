#!/usr/bin/env python3
"""Compare an oracle residual native continuation with dense and compact controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def rows(path: Path) -> tuple[dict, dict[int, dict]]:
    doc = json.loads(path.read_text())
    out: dict[int, dict] = {}
    for group in doc.get("groups", []):
        for row in group.get("layers", []):
            if isinstance(row, dict) and "layer" in row:
                out[int(row["layer"])] = row
    return doc, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dense", type=Path, default=Path("receipts/headless/FLASH_DENSE_L0_L7_V1/FAST_CHAIN_SUMMARY.json"), nargs="?")
    ap.add_argument("compact", type=Path, default=Path("receipts/headless/FLASH_COMPACT_L0_L7_V1/FAST_CHAIN_SUMMARY.json"), nargs="?")
    ap.add_argument("candidate", type=Path, default=Path("receipts/headless/FLASH_MARGIN_RESIDUAL_NATIVE_L4_L6/FAST_CHAIN_SUMMARY.json"), nargs="?")
    ap.add_argument("--residual", type=Path, default=Path("receipts/headless/FLASH_MARGIN_RESIDUAL_CANDIDATE_L3_L4.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dense_doc, dense = rows(args.dense)
    compact_doc, compact = rows(args.compact)
    candidate_doc, candidate = rows(args.candidate)
    layers = sorted(set(dense) & set(compact) & set(candidate))
    comparisons = []
    for layer in layers:
        d, c, n = dense[layer], compact[layer], candidate[layer]
        comparisons.append({
            "layer": layer,
            "dense_output_sha256": d.get("output_sha256"),
            "compact_output_sha256": c.get("output_sha256"),
            "candidate_output_sha256": n.get("output_sha256"),
            "candidate_matches_dense_exact": n.get("output_sha256") == d.get("output_sha256"),
            "candidate_matches_compact_exact": n.get("output_sha256") == c.get("output_sha256"),
            "dense_route_ids": d.get("route_ids_expected") or d.get("route_ids_observed"),
            "compact_route_ids": c.get("route_ids_expected") or c.get("route_ids_observed"),
            "candidate_route_ids": n.get("route_ids_expected") or n.get("route_ids_observed"),
            "candidate_routes_match_dense": (n.get("route_ids_expected") or n.get("route_ids_observed")) == (d.get("route_ids_expected") or d.get("route_ids_observed")),
            "candidate_routes_match_compact": (n.get("route_ids_expected") or n.get("route_ids_observed")) == (c.get("route_ids_expected") or c.get("route_ids_observed")),
            "candidate_final_within_tolerance": (n.get("final_state") or {}).get("within_tolerance") is True,
        })
    first_route = next((r["layer"] for r in comparisons if not r["candidate_routes_match_dense"]), None)
    first_exact = next((r["layer"] for r in comparisons if not r["candidate_matches_dense_exact"]), None)
    report = {
        "schema": "hawking.flash.margin_residual_native_compare.v1",
        "status": "PASSED_MARGIN_RESIDUAL_NATIVE_PARITY" if first_route is None and first_exact is None else "BLOCKED_MARGIN_RESIDUAL_NATIVE_PARITY",
        "dense_receipt": str(args.dense), "compact_receipt": str(args.compact), "candidate_receipt": str(args.candidate),
        "residual_receipt": str(args.residual), "residual_receipt_sha256": hashlib.sha256(args.residual.read_bytes()).hexdigest(),
        "layers": layers, "comparisons": comparisons,
        "divergence": {"first_route_id_mismatch_layer": first_route, "first_exact_fingerprint_mismatch_layer": first_exact, "interpretation": "The oracle residual did not recover dense routing; the candidate remains an unpromoted diagnostic." if first_route is not None else "Candidate matches dense controls in the bounded native range."},
        "source_payload": {"dense_bytes": sum(int(dense[x].get("source_bytes_read") or 0) for x in layers), "compact_bytes": sum(int(compact[x].get("source_bytes_read") or 0) for x in layers), "candidate_bytes": sum(int(candidate[x].get("source_bytes_read") or 0) for x in layers)},
        "timing": {"dense_wall_ns": int(dense_doc.get("elapsed_wall_ns") or 0), "compact_wall_ns": int(compact_doc.get("elapsed_wall_ns") or 0), "candidate_wall_ns": int(candidate_doc.get("elapsed_wall_ns") or 0), "candidate_gpu_ns": sum(sum(int(v or 0) for v in (candidate[x].get("gpu_ns") or [])) for x in layers)},
        "claim_boundary": "Native compact continuation from an oracle-derived conditional residual was measured only for layers 4-6. It is not a learned/generalized residual, complete-token, capability, TPS, EBPW, or residency claim.",
        "promotion_allowed": False,
        "bench": {"state": "UNKNOWN", "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "recorded_by": "tools/flash_margin_residual_native_compare.py", "machine": "Apple M3 Ultra", "rule": "bounded native seam comparison only; no complete-token promotion"},
    }
    report["seal_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "layers": layers, "first_route_id_mismatch_layer": first_route, "first_exact_fingerprint_mismatch_layer": first_exact, "out": str(args.out)}, indent=2))
    return 0 if report["status"].startswith("PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
