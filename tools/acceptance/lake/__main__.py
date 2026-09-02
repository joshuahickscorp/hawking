"""Run ModelLake + Qwen27 acceptance gates.

    python3 -m tools.acceptance.lake
    python3 -m tools.acceptance.lake --gate MODELLAKE_IDENTITY_RESOLVED
"""
from __future__ import annotations

import argparse
import json
import sys

from tools.acceptance.lake.common import GATES, ensure_tools_path, receipts_dir
from tools.acceptance.lake.hash_verify import GATE as HASH_GATE
from tools.acceptance.lake.hash_verify import run_hash_gate
from tools.acceptance.lake.identity import GATE as IDENTITY_GATE
from tools.acceptance.lake.identity import run_identity_gate
from tools.acceptance.lake.promotion import GATE as PROMO_GATE
from tools.acceptance.lake.promotion import run_promotion_gate
from tools.acceptance.lake.qwen27 import (
    BASELINE_GATE,
    IDENTITY_GATE as QWEN_ID_GATE,
    REGRESSION_GATE,
    run_protected_baseline_gate,
    run_regression_gate,
    run_runtime_identity_gate,
)

ORDER = (
    IDENTITY_GATE,
    HASH_GATE,
    PROMO_GATE,
    QWEN_ID_GATE,
    BASELINE_GATE,
    REGRESSION_GATE,
)

RUNNERS = {
    IDENTITY_GATE: lambda: run_identity_gate(live=True, run_census=True),
    HASH_GATE: lambda: run_hash_gate(live=True, run_canonical_oid_hash=True),
    PROMO_GATE: lambda: run_promotion_gate(live=True),
    QWEN_ID_GATE: run_runtime_identity_gate,
    BASELINE_GATE: lambda: run_protected_baseline_gate(ready_timeout_s=2.0),
    REGRESSION_GATE: lambda: run_regression_gate(invoke_live_ab=False),
}


def main(argv: list[str] | None = None) -> int:
    ensure_tools_path()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=list(GATES), help="run one gate (default: all)")
    parser.add_argument(
        "--ready-timeout-s",
        type=float,
        default=2.0,
        help="protected-baseline quiescence wait (default 2s; never hours)",
    )
    args = parser.parse_args(argv)
    gates = (args.gate,) if args.gate else ORDER
    if BASELINE_GATE in gates:
        RUNNERS[BASELINE_GATE] = lambda: run_protected_baseline_gate(
            ready_timeout_s=float(args.ready_timeout_s)
        )
    results = []
    failed = 0
    for gate in gates:
        print(f"== {gate} ==", flush=True)
        receipt = RUNNERS[gate]()
        verdict = receipt.get("verdict")
        print(
            f"{gate}: {verdict}  symbol_invoked={receipt.get('symbol_invoked')}  "
            f"tier={receipt.get('evidence_tier')}  elapsed_s={receipt.get('elapsed_s')}",
            flush=True,
        )
        print((receipt.get("output") or {}).get("summary") or "", flush=True)
        if receipt.get("blocker"):
            print(f"  blocker: {receipt['blocker'].get('missing_input')}", flush=True)
        results.append(
            {
                "gate": gate,
                "verdict": verdict,
                "receipt_path": receipt.get("receipt_path"),
                "symbol_invoked": receipt.get("symbol_invoked"),
            }
        )
    summary = {
        "schema": "hawking.acceptance.lake.summary.v1",
        "gates": results,
        "accepted": sum(1 for r in results if r["verdict"] == "ACCEPTED"),
        "blocked": sum(1 for r in results if r["verdict"] == "BLOCKED"),
        "criterion_altered": False,
        "receipts_dir": str(receipts_dir()),
    }
    dest = receipts_dir() / "SUMMARY.json"
    dest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary: accepted={summary['accepted']} blocked={summary['blocked']} -> {dest}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
