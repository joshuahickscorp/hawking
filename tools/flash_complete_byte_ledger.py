#!/usr/bin/env python3
"""Close the first complete Flash byte ledger without inventing compression."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, default=Path("receipts/headless/FLASH_ORGAN_CENSUS.json"))
    ap.add_argument("--profile", type=Path, default=Path("receipts/headless/FLASH_HOT_CHAIN_PROFILE_DEVICE_L0_L47_COMPLETE_V1.json"))
    ap.add_argument("--nr", type=Path, default=Path("receipts/headless/FLASH_COMPLETE_V0.nr.json"))
    ap.add_argument("--out", type=Path, default=Path("receipts/headless/FLASH_COMPLETE_V0.BYTE_LEDGER.json"))
    a = ap.parse_args()
    census = json.loads(a.census.read_text())
    profile = json.loads(a.profile.read_text())
    nr = json.loads(a.nr.read_text())
    source_bytes = int(census["source_parameter_bytes_indexed"])
    denominator = source_bytes // 2
    profile_bytes = int(profile.get("source_bytes_read") or profile.get("source_payload_bytes_read") or 0)
    routed = next(row["bytes"] for row in census["family_summary"] if row["family"] == "routed_experts")
    active_ten = routed * 10 / 512
    doc = {
        "schema": "hawking.flash.complete_byte_ledger.v1",
        "status": "MEASURED_EXACT_CONTROL_WITH_ROUTE_IO_PROFILE",
        "model": census["model"],
        "source_index_sha256": census["source_index_sha256"],
        "nr_sha256": hashlib.sha256(a.nr.read_bytes()).hexdigest(),
        "denominator": {"source_parameter_equivalent": denominator, "rule": "BF16 source bytes / 2"},
        "complete_exact_control": {
            "runtime_required_bytes": source_bytes,
            "complete_ebpw": source_bytes * 8 / denominator,
            "representation": "source_bf16_exact fallback for every family",
        },
        "measured_fastpath_profile": {
            "receipt": str(a.profile),
            "source_bytes_read": profile_bytes,
            "active_bytes_per_token_profile": profile_bytes,
            "active_fraction_of_indexed_source": profile_bytes / source_bytes if source_bytes else None,
            "complete_ebpw": None,
            "note": "physical profile read volume, not a compact NX storage claim",
        },
        "routed_expert_sensitivity": {
            "full_family_bytes": routed,
            "ten_of_512_route_fraction_bytes": active_ten,
            "complete_storage_bytes": None,
            "note": "active route estimate only; dynamic all-expert representation remains open",
        },
        "bench": {
            "state": "UNKNOWN",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recorded_by": "tools/flash_complete_byte_ledger.py",
            "machine": "Apple M3 Ultra (profile receipt binding)",
            "rule": "S032 §3 -- ledger records exact bytes and profiled reads; no unmeasured compact EBPW",
        },
        "promotion_allowed": False,
        "claim_boundary": "Exact complete-control byte ledger is closed at 16.0 EBPW; the route profile is measured I/O only and does not establish a smaller complete NX. Compact EBPW remains open until dynamic routing, storage, and accepted-token execution are qualified.",
        "next": "compile and execute a complete candidate, then replace one family at a time using measured complete-token Pareto results",
    }
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "complete_ebpw": doc["complete_exact_control"]["complete_ebpw"], "profile_source_bytes": profile_bytes, "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
