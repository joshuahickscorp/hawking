"""Run VMCP/AgentOS acceptance gates and write receipts/acceptance/*.json.

    python3 -m tools.acceptance.vmcp
    python3 -m tools.acceptance.vmcp --gate VMCP_DEEP_DIGEST
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from tools.acceptance.vmcp import common as C
from tools.acceptance.vmcp.gates import RUNNERS


def run_one(gate: str) -> dict[str, Any]:
    fn = RUNNERS[gate]
    return fn()


def run_all(gates: list[str] | None = None) -> dict[str, dict[str, Any]]:
    selected = list(gates or C.GATES)
    out: dict[str, dict[str, Any]] = {}
    for name in selected:
        if name not in RUNNERS:
            raise KeyError(name)
        out[name] = run_one(name)
    return out


def write_receipts(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    C.RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for gate, doc in results.items():
        path = C.write_json(C.RECEIPT_DIR / f"{gate}.json", doc)
        written.append(str(path.relative_to(C.REPO)))
    accepted = [g for g, d in results.items() if d.get("verdict") == "ACCEPTED"]
    blocked = [g for g, d in results.items() if d.get("verdict") == "BLOCKED"]
    index = {
        "schema": C.INDEX_SCHEMA,
        "lane": "acc3-vmcp-acceptance",
        "gates": [
            {
                "gate": g,
                "verdict": results[g]["verdict"],
                "evidence_tier": results[g]["evidence_tier"],
                "receipt": f"receipts/acceptance/{g}.json",
                "blocker_missing": (results[g].get("blocker") or {}).get("missing"),
                "invoked_call_count": len(results[g].get("invoked_symbols") or []),
            }
            for g in results
        ],
        "accepted": accepted,
        "blocked": blocked,
        "accepted_count": len(accepted),
        "blocked_count": len(blocked),
        "assigned_count": len(results),
        "criterion_weakened": False,
        "gpu_authority": False,
        "auditor_note": (
            "These receipts record a real run of each gate's own acceptance "
            "span. The catalog currently only auto-accepts numeric specs; this "
            "lane does not edit tools/roadmap or tools/audit. Consume "
            "receipts/acceptance/<GATE>.json as accepted evidence: verdict, "
            "invoked_symbols (kind=call), checks, measured, output."
        ),
        "recorded_by": "tools/acceptance/vmcp/run.py",
        "generated_at": C.utc_now(),
    }
    C.write_json(C.RECEIPT_DIR / "INDEX.json", index)
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="append", dest="gates")
    ap.add_argument("--print", action="store_true", dest="dump")
    args = ap.parse_args(argv)
    results = run_all(args.gates)
    index = write_receipts(results)
    summary = {
        "accepted_count": index["accepted_count"],
        "blocked_count": index["blocked_count"],
        "accepted": index["accepted"],
        "blocked": [
            {"gate": g, "missing": (results[g].get("blocker") or {}).get("missing")}
            for g in index["blocked"]
        ],
        "criterion_weakened": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dump:
        print(json.dumps({g: results[g]["verdict"] for g in results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
