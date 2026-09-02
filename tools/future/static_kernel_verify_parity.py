#!/usr/bin/env python3
"""Parity gate: Rust static_kernel_verify vs Python on real hawking-core sources.

A fast path that silently changes a verdict is worse than a slow one.
This harness forces the Python analyzer and the Rust binary to scan the same
repo and compares canonical JSON (head/branch stripped — those are git
bookkeeping, not findings).

    HAWKING_SKV_FORCE_PYTHON=1 python3 tools/future/static_kernel_verify_parity.py
    python3 tools/future/static_kernel_verify_parity.py --rust-bin PATH

Exit 0 only when the documents match. Does not write receipts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.future import static_kernel_verify as skv

BOOKKEEPING = frozenset({"head", "branch"})


def canonical(doc: dict[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k not in BOOKKEEPING}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def python_scan(repo: Path) -> tuple[dict[str, Any], float]:
    metal, rust, membership = skv.load_repo_sources(repo)
    t0 = time.perf_counter()
    raw = skv.analyze(metal, rust, library_membership=membership)
    doc = skv.report_from_analyze(raw)
    return doc, time.perf_counter() - t0


def rust_scan(bin_path: Path, repo: Path) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(bin_path), "--repo", str(repo), "--json"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(
            f"rust binary exited {proc.returncode}\nstdout={proc.stdout[:2000]}\nstderr={proc.stderr[:2000]}"
        )
    doc = json.loads(proc.stdout)
    if not isinstance(doc, dict) or doc.get("schema") != skv.SCHEMA:
        raise SystemExit(f"rust output is not a {skv.SCHEMA} document")
    return doc, dt


def _diff_keys(a: dict, b: dict, prefix: str = "") -> list[str]:
    out = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        pa, pb = f"{prefix}.{k}" if prefix else k, None
        path = f"{prefix}.{k}" if prefix else k
        if k not in a:
            out.append(f"MISSING_IN_PYTHON {path}")
            continue
        if k not in b:
            out.append(f"MISSING_IN_RUST {path}")
            continue
        va, vb = a[k], b[k]
        if type(va) is not type(vb) and not (
            isinstance(va, (int, float)) and isinstance(vb, (int, float))
        ):
            out.append(f"TYPE {path}: python={type(va).__name__} rust={type(vb).__name__}")
            continue
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(_diff_keys(va, vb, path))
        elif va != vb:
            if path == "findings" and isinstance(va, list) and isinstance(vb, list):
                out.append(f"LEN {path}: python={len(va)} rust={len(vb)}")
                n = min(len(va), len(vb), 8)
                for i in range(n):
                    if va[i] != vb[i]:
                        out.append(f"FINDING[{i}] python={va[i]!r:.400} rust={vb[i]!r:.400}")
                continue
            sa = json.dumps(va, sort_keys=True, default=str)
            sb = json.dumps(vb, sort_keys=True, default=str)
            out.append(f"VALUE {path}: python={sa[:300]} rust={sb[:300]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--rust-bin", type=Path, default=None)
    args = ap.parse_args()
    repo = args.repo.resolve()

    print(f"python scan {repo} …", flush=True)
    py_doc, py_s = python_scan(repo)
    print(
        f"python {py_s:.3f}s  kernels={py_doc['coverage']['metal_kernels']}  "
        f"ERROR={py_doc['counts']['ERROR']} WARNING={py_doc['counts']['WARNING']}  "
        f"UNVERIFIABLE={py_doc['counts']['UNVERIFIABLE']}",
        flush=True,
    )

    bin_path = args.rust_bin or skv._rust_binary()
    if bin_path is None:
        print("PARITY FAIL: rust binary absent (set --rust-bin or build hawking-static-kernel-verify)")
        return 2
    print(f"rust scan {bin_path} …", flush=True)
    rs_doc, rs_s = rust_scan(Path(bin_path), repo)
    print(
        f"rust   {rs_s:.3f}s  kernels={rs_doc['coverage']['metal_kernels']}  "
        f"ERROR={rs_doc['counts']['ERROR']} WARNING={rs_doc['counts']['WARNING']}  "
        f"UNVERIFIABLE={rs_doc['counts']['UNVERIFIABLE']}",
        flush=True,
    )

    py_c = canonical(py_doc)
    rs_c = canonical(rs_doc)
    if py_c == rs_c:
        print(f"PARITY PASS  python={py_s:.3f}s rust={rs_s:.3f}s  bytes={len(py_c)}")
        if rs_s > 0:
            print(f"speedup python/rust = {py_s / rs_s:.2f}x (measured, not estimated)")
        return 0

    py_b = {k: v for k, v in py_doc.items() if k not in BOOKKEEPING}
    rs_b = {k: v for k, v in rs_doc.items() if k not in BOOKKEEPING}
    diffs = _diff_keys(py_b, rs_b)
    print("PARITY FAIL")
    for line in diffs[:40]:
        print(" ", line)
    print(f"  ({len(diffs)} diffs, showing up to 40)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
