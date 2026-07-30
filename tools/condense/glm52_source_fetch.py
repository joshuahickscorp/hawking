#!/usr/bin/env python3.12
"""Installed-path source-fetch CLI (launchd Program). Real argparse; delegates to lab.operators."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path for lab package imports when invoked as a script path.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lab.operators import glm52_source_fetch as body  # noqa: E402

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glm52_source_fetch", description="GLM-5.2 BF16 source fetch controller")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run fetch/pack windows under disk floor")
    sub.add_parser("status", help="print controller status JSON")
    sub.add_parser("rollup", help="rollup window ledger")
    sub.add_parser("safe-to-leave", help="host/controller safe-to-leave check")
    sub.add_parser("reconcile", help="reconcile packed shards")
    sub.add_parser("selftest", help="offline schedule/eviction selftest")
    return p

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Back-compat: bare "run" etc. without subparser flags
    if argv and argv[0] in {"run", "status", "rollup", "safe-to-leave", "reconcile", "selftest"} and "--help" not in argv:
        cmd = argv[0]
    else:
        args = build_parser().parse_args(argv)
        cmd = args.command
    dispatch = {
        "run": body.run,
        "status": body.status,
        "rollup": body.rollup,
        "safe-to-leave": body.safe_to_leave,
        "reconcile": body.reconcile,
        "selftest": body.selftest,
    }
    return int(dispatch[cmd]())

if __name__ == "__main__":
    raise SystemExit(main())
