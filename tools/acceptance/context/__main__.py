"""python3 -m tools.acceptance.context [--gate GATE]"""

from __future__ import annotations

import argparse
import json
import sys

from tools.acceptance.context.gates import GATE_IDS, run_all, run_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="append", choices=list(GATE_IDS))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    bundle = run_all(args.gate)
    summary = bundle["summary"]
    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(
            f"accepted {summary['accepted_count']}/{len(summary['gates'])}  "
            f"blocked {summary['blocked_count']}  "
            f"criterion_altered={summary['criterion_altered']}"
        )
        for gate in summary["gates"]:
            row = summary["results"][gate]
            flag = "ACCEPTED" if row["verdict"] == "ACCEPTED" else "BLOCKED "
            extra = ""
            if row.get("blocker"):
                extra = "  " + str(row["blocker"])[:180]
            print(f"  {flag}  {gate}{extra}")
    failed = [
        gate
        for gate, row in bundle["results"].items()
        if row.get("verdict") not in {"ACCEPTED", "BLOCKED"}
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
