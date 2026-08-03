#!/usr/bin/env python3.12
"""Run Odyssey T0: four reproductions + contract closure + feasibility.

Does not flip ODYSSEY_LAUNCH_AUTHORIZED. Does not start training.
Does not write into the Math-Preserve artifact.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.odyssey._paths import FENCE, RECORDS_DIR, ROOT
from tools.odyssey import (
    contracts,
    data_verify,
    feasibility,
    hidden_memberships,
    known_failures,
    runtime_authority,
    substrate_verify,
)

SCHEMA = "hawking.odyssey.t0.v1"
RECEIPT = RECORDS_DIR / "ODYSSEY_T0_RECEIPT.json"


def _fence_still_false() -> bool:
    return FENCE.is_file() and FENCE.read_text().strip().lower() == "false"


def run_unit(*, include_runtime: bool = True) -> dict:
    """Callable T0 unit path for tests and apparatus proof.

    Exercises the four T0 reproduction legs without a multi-shard hash sweep:
      - substrate static checks (hashes/counts; no 92 GB scan)
      - data classification (DECLARED_NOT_PRESENT is success)
      - runtime authority (optional; bit-identical single-layer)
      - known-failure registry

    Does not flip the fence. Does not start training. Does not write launch/.
    """
    if not _fence_still_false():
        fence_val = FENCE.read_text().strip() if FENCE.is_file() else "missing"
        fence_note = f"fence reads {fence_val!r}; T0 unit does not write the fence"
    else:
        fence_note = "ODYSSEY_LAUNCH_AUTHORIZED remains false"

    hidden_memberships.write_seed_sets()
    static = substrate_verify.static_checks()
    data = data_verify.verify_all()
    runtime = runtime_authority.verify_runtime() if include_runtime else {
        "status": "SKIPPED",
        "note": "runtime skipped by caller",
    }
    failures = known_failures.build_registry()

    summary = {
        "substrate_static": "PASS" if static.get("ok") else "FAIL",
        "data": data["status"],
        "runtime": runtime["status"],
        "known_failures": failures["status"],
    }
    bad = [
        k
        for k, v in summary.items()
        if v not in ("PASS", "PARTIAL", "SKIPPED")
    ]
    return {
        "schema": "hawking.odyssey.t0.unit.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fence": fence_note,
        "launch_authorized": False,
        "mode": "unit",
        "summary": summary,
        "status": "PASS" if not bad else "FAIL",
        "detail": {
            "static": static,
            "data": data,
            "runtime": runtime,
            "known_failures": {
                "status": failures["status"],
                "n_entries": failures["n_entries"],
            },
        },
        "note": "Unit path; full shard verification is tools/odyssey/t0_run.py:run",
    }


def run(*, max_shards: int = 8, max_bytes: int | None = 512 * 1024 * 1024) -> dict:
    if not _fence_still_false():
        # Refuse to continue if someone flipped the fence in this session by accident
        # from another process — we still do not flip it ourselves.
        fence_val = FENCE.read_text().strip() if FENCE.is_file() else "missing"
        # T0 harness is allowed even if fence is true (reproduction only), but we record it.
        fence_note = f"fence reads {fence_val!r}; T0 harness does not write the fence"
    else:
        fence_note = "ODYSSEY_LAUNCH_AUTHORIZED remains false"

    hidden_memberships.write_seed_sets()

    substrate = substrate_verify.verify_shards(max_shards=max_shards, max_bytes=max_bytes)
    data = data_verify.verify_all()
    runtime = runtime_authority.verify_runtime()
    failures = known_failures.write_registry()
    closure = contracts.closure_report()
    feas = feasibility.estimate()

    (RECORDS_DIR / "ODYSSEY_CONTRACT_CLOSURE.json").write_text(
        json.dumps(closure, indent=2, sort_keys=True, default=str) + "\n"
    )
    (RECORDS_DIR / "ODYSSEY_FEASIBILITY.json").write_text(
        json.dumps(feas, indent=2, sort_keys=True) + "\n"
    )

    receipt = {
        "schema": SCHEMA,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fence": fence_note,
        "launch_authorized": False,
        "reproductions": {
            "substrate": {
                "status": substrate["status"],
                "detail": substrate,
            },
            "data": {
                "status": data["status"],
                "detail": data,
            },
            "runtime": {
                "status": runtime["status"],
                "detail": runtime,
            },
            "known_failures": {
                "status": failures["status"],
                "detail": {
                    "n_entries": failures["n_entries"],
                    "entries": [
                        {"id": e["id"], "status": e["status"], "source": e.get("source")}
                        for e in failures["entries"]
                    ],
                    "what_was_checked": failures["what_was_checked"],
                    "what_was_skipped": failures["what_was_skipped"],
                },
            },
        },
        "contract_closure_path": "odyssey/ODYSSEY_CONTRACT_CLOSURE.json",
        "feasibility_path": "odyssey/ODYSSEY_FEASIBILITY.json",
        "summary": {
            "substrate": substrate["status"],
            "data": data["status"],
            "runtime": runtime["status"],
            "known_failures": failures["status"],
            "contracts_runnable": closure["summary"]["RUNNABLE"],
            "contracts_declared": closure["summary"]["DECLARED"],
            "t1_t5_feasible": feas["verdict"]["t1_t5_full_training_feasible_on_this_hardware_now"],
        },
    }
    # Overall T0 status: substrate may be PARTIAL (resumable); that is acceptable for T0 smoke.
    bad = [
        k
        for k, v in receipt["summary"].items()
        if k in ("substrate", "data", "runtime", "known_failures") and v not in ("PASS", "PARTIAL")
    ]
    receipt["status"] = "PASS" if not bad else "FAIL"
    if receipt["summary"]["substrate"] == "PARTIAL":
        receipt["status_note"] = (
            "substrate verification is PARTIAL (resumable); static hash/count checks must PASS"
        )
        if not substrate.get("static", {}).get("ok"):
            receipt["status"] = "FAIL"

    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-shards", type=int, default=8)
    p.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024)
    p.add_argument("--full-substrate", action="store_true", help="hash all 282 shards (slow/IO-heavy)")
    args = p.parse_args(argv)
    if args.full_substrate:
        receipt = run(max_shards=None, max_bytes=None)
    else:
        receipt = run(max_shards=args.max_shards, max_bytes=args.max_bytes)
    print(json.dumps({"status": receipt["status"], "summary": receipt["summary"]}, indent=2))
    print("wrote", RECEIPT)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
