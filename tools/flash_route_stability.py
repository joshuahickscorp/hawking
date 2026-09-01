#!/usr/bin/env python3
"""Audit whether a compact Flash expert bank remains exact across token steps.

The compact loader is route-before-payload, but a bank selected for one token
is only reusable when later router selections stay inside that bank. This
receipt makes that condition explicit instead of silently treating unmapped
experts as valid execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-experts", type=int, default=512)
    ns = ap.parse_args()
    doc = json.loads(ns.receipt.read_text())
    steps = [row for row in doc.get("steps", []) if isinstance(row, dict) and isinstance(row.get("route_ids"), list)]
    by_layer: dict[int, list[dict]] = {}
    default_layer = doc.get("layer", 0)
    for row in steps:
        by_layer.setdefault(int(row.get("layer", default_layer)), []).append(row)
    layers = []
    missing_total = 0
    for layer, rows in sorted(by_layer.items()):
        loaded = set(int(v) for v in rows[0]["route_ids"])
        missing_by_step = []
        for row in rows[1:]:
            missing = sorted(set(int(v) for v in row["route_ids"]) - loaded)
            missing_total += len(missing)
            missing_by_step.append({"step": row.get("step"), "token_id": row.get("token_id"), "missing_experts": missing})
        layers.append({
            "layer": layer,
            "initial_loaded_experts": sorted(loaded),
            "compact_experts": len(loaded),
            "dense_experts": ns.source_experts,
            "compact_bank_fraction": len(loaded) / ns.source_experts if ns.source_experts else None,
            "later_steps": missing_by_step,
            "route_stable": all(not item["missing_experts"] for item in missing_by_step),
        })
    status = "ROUTE_STABLE" if layers and missing_total == 0 else "ROUTE_UNSAFE_FOR_COMPACT_REUSE"
    report = {
        "schema": "hawking.flash.route_stability_audit.v1",
        "status": status,
        "source_receipt": str(ns.receipt),
        "source_receipt_sha256": hashlib.sha256(ns.receipt.read_bytes()).hexdigest(),
        "layers": layers,
        "summary": {
            "layers_audited": len(layers),
            "steps_audited": len(steps),
            "unmapped_route_count": missing_total,
            "compact_reuse_allowed": bool(layers) and missing_total == 0,
        },
        "claim_boundary": (
            "Every audited later token stays within the first-token compact bank. Compact reuse is route-safe for this bounded receipt."
            if layers and missing_total == 0
            else "Later token router selections escape the first-token compact bank. Compact reuse would omit required expert payloads and is not exact; use a route union or dense fallback."
        ),
        "next": "Build a route-union compact loader across a verified token window; retain dense execution until union parity is proven.",
        "bench": {"state": "UNKNOWN", "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "recorded_by": "tools/flash_route_stability.py", "machine": "Apple M3 Ultra", "rule": "route safety evidence only; no TPS or residency claim"},
    }
    report["seal_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0 if status == "ROUTE_STABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
