"""python3 -m tools.doctor --build

    python3 -m tools.doctor --diagnose receipts/QWEN80_BIT_BUDGET_LEDGER.json
    python3 -m tools.doctor --diagnose Qwen--Qwen3.8-Flash-Next@34567a4712bc
    python3 -m tools.doctor --zeros-selftest
"""
from __future__ import annotations

import argparse
import json
import sys

from tools.doctor.engine import build, diagnose, zeros_controls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools.doctor")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--zeros-selftest", action="store_true")
    ap.add_argument("--diagnose", metavar="TARGET")
    args = ap.parse_args(argv)
    if args.zeros_selftest:
        print(json.dumps(zeros_controls(), indent=2, sort_keys=True))
        return 0
    if args.diagnose:
        doc = diagnose(args.diagnose)
        print(json.dumps(doc, indent=2, sort_keys=True, default=str))
        io = doc.get("io") or {}
        if io.get("weight_bytes_loaded"):
            print("weight bytes loaded — diagnosis is invalid", file=sys.stderr)
            return 2
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
