#!/usr/bin/env python3
# strand_eval.cli — the one CLI over the canon eval + ledger.
#
#   strand-eval run    --model DIR --tag TAG [--ctx 2048] [--chunks 64]
#                      [--device auto|cpu|mps] [--dtype bfloat16]
#                      [--out-dir DIR] [--ce-slice N] [--no-ledger]
#   strand-eval ledger check  [--ledger PATH]
#   strand-eval ledger ingest DIR [--ledger PATH] [--quiet]
#   strand-eval where         (self-location proof: prints module + repo root)
#
# Ledger subcommands are torch-free (safe from conductor every poll).

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from strand_eval import HARNESS_VERSION, default_ledger_path, locate_repo_root


def main(argv=None):
    p = argparse.ArgumentParser(prog="strand-eval",
                                description="STRAND canonical PPL eval + results ledger")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the canon WikiText-2 PPL eval")
    r.add_argument("--model", required=True, help="model dir (HF layout or recon dir)")
    r.add_argument("--tag", required=True, help="config tag (e.g. q2_l12_out1, baseline)")
    r.add_argument("--ctx", type=int, default=2048)
    r.add_argument("--chunks", type=int, default=64,
                   help="window cap: 64=screening, 146=anchor, 0=all")
    r.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "mps"])
    r.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    r.add_argument("--out-dir", default=None,
                   help="result dir (default: the model dir); NAME is always derived")
    r.add_argument("--ce-slice", type=int, default=None,
                   help="CE row-slice size (default: 0 on cpu, 512 on mps)")
    r.add_argument("--ledger", default=None, help="ledger path override")
    r.add_argument("--no-ledger", action="store_true")

    lg = sub.add_parser("ledger", help="results-ledger operations (torch-free)")
    lsub = lg.add_subparsers(dest="lcmd", required=True)
    c = lsub.add_parser("check", help="run the 15-digit tell + harness_key checks")
    c.add_argument("--ledger", default=None)
    g = lsub.add_parser("ingest", help="append new ppl_*.json discoveries from a dir")
    g.add_argument("src_dir")
    g.add_argument("--ledger", default=None)
    g.add_argument("--quiet", action="store_true")

    sub.add_parser("where", help="print self-location (module path, repo root, ledger)")

    a = p.parse_args(argv)

    if a.cmd == "where":
        import strand_eval
        print(f"module     : {os.path.realpath(strand_eval.__file__)}")
        print(f"version    : {HARNESS_VERSION}")
        print(f"repo_root  : {locate_repo_root()}")
        print(f"ledger     : {default_ledger_path()}")
        return 0

    if a.cmd == "run":
        from strand_eval.core import run_eval
        run_eval(a.model, a.tag, ctx=a.ctx, limit_chunks=a.chunks, device=a.device,
                 dtype=a.dtype, out_dir=a.out_dir, ledger_path=a.ledger,
                 ce_slice=a.ce_slice, no_ledger=a.no_ledger)
        return 0

    if a.cmd == "ledger":
        from strand_eval.ledger import check, ingest
        lp = a.ledger or default_ledger_path()
        if a.lcmd == "ingest":
            ingest(lp, a.src_dir, quiet=a.quiet)
            return 0
        errors, warnings = check(lp)
        for w in warnings:
            print(f"WARN  {w}")
        for e in errors:
            print(f"ERROR {e}")
        n = len(read_count(lp))
        print(f"[ledger] {n} record(s), {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1 if errors else 0
    return 2


def read_count(lp):
    from strand_eval.ledger import read_ledger
    return read_ledger(lp)


if __name__ == "__main__":
    sys.exit(main())
