#!/usr/bin/env python3.12
"""Reachability + pack sealing for Stage B (CLEAN SLATE Sections 13, 16, 21).

Sharpens the census kernel estimate with a real import-reachability graph: which tools/condense
Python modules are reachable from the CLI entrypoints (the true controller/forge/doctor kernel) vs
orphaned (laboratory / retired-campaign surface). Splits the Rust crates into the hawking kernel vs
the extractable hide product. Then SEALS a hawking-lab pack manifest for the orphaned modules
(content-addressed, source-commit-bound) WITHOUT moving or deleting anything - the destructive move
is a later checkpoint with per-file test/rollback.

Read-only over the working tree. No file relocated, no import rewritten.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

OUT = Path("reports/condense/gravity_forge/condensation")
ENTRYPOINTS = ["succ_cli.py", "eco_cli.py"]   # the real user-facing CLIs (Section 18)


def _loc(f: str) -> int:
    try:
        return sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
    except Exception:
        return 0


def _sha_file(f: str) -> str:
    return hashlib.sha256(open(f, "rb").read()).hexdigest()


def _imports(f: str) -> set[str]:
    out: set[str] = set()
    try:
        tree = ast.parse(open(f, encoding="utf-8", errors="ignore").read())
    except Exception:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                out.add(node.module.split(".")[0])
    return out


def build() -> dict[str, Any]:
    tracked = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n")
               if f and os.path.isfile(f)]

    rust = Counter()
    for f in tracked:
        if f.startswith("crates/") and f.endswith(".rs"):
            crate = f.split("/")[1]
            rust["hide_product" if crate.startswith("hide") else "hawking_kernel"] += _loc(f)

    pyfiles = {f for f in tracked if f.startswith("tools/condense/") and f.endswith(".py")}
    mod2file = {os.path.splitext(os.path.basename(f))[0]: f for f in pyfiles}
    entry = [f"tools/condense/{e}" for e in ENTRYPOINTS if f"tools/condense/{e}" in pyfiles]

    reach, q = set(entry), deque(entry)
    while q:
        for m in _imports(q.popleft()):
            tf = mod2file.get(m)
            if tf and tf not in reach:
                reach.add(tf); q.append(tf)
    orphan = sorted(pyfiles - reach)
    reach_loc = sum(_loc(f) for f in reach)
    orphan_loc = sum(_loc(f) for f in orphan)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)

    reachdoc = {
        "schema": "hawking.condense_reachability.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entrypoints": entry,
        "rust": {"hawking_kernel_loc": rust["hawking_kernel"], "hide_product_loc": rust["hide_product"]},
        "python_tools_condense": {
            "reachable_modules": len(reach), "reachable_loc": reach_loc,
            "orphaned_modules": len(orphan), "orphaned_loc": orphan_loc,
            "orphaned_fraction": round(orphan_loc / max(1, reach_loc + orphan_loc), 3),
        },
        "kernel_floor_estimate": {
            "rust_hawking": rust["hawking_kernel"], "python_reachable": reach_loc,
            "combined": rust["hawking_kernel"] + reach_loc,
            "note": "true CLI-reachable kernel; the Rust engine dominates, the Python spine is small",
        },
        "caveat": "orphaned = not reachable from the two CLIs. Some orphaned modules are the live/retired "
                  "doctor_v5 campaign (their own entrypoints + launchd), not dead code; they are the "
                  "hawking-lab pack surface, sealed below, NOT deleted.",
    }
    reachdoc["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in reachdoc.items() if k != "sha256"},
                                                   sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    (OUT / "REACHABILITY.json").write_text(json.dumps(reachdoc, indent=2, sort_keys=True, default=str))

    # SEAL the hawking-lab pack manifest (Section 21) - content-addressed, no move
    entries = [{"path": f, "loc": _loc(f), "sha256": _sha_file(f)} for f in orphan]
    pack = {
        "schema": "hawking.pack.manifest.v1", "pack": "hawking-lab", "version": "0.1.0",
        "source_commit": commit, "generated_at": reachdoc["generated_at"],
        "reason": "orphaned-from-CLI laboratory / retired-campaign Python surface (Section 21)",
        "content_count": len(entries), "content_loc": orphan_loc,
        "offline_hydratable": True, "rollback": "restore from source_commit",
        "status": "SEALED_MANIFEST_ONLY (no files moved or deleted this checkpoint)",
        "contents": entries,
    }
    pack["manifest_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in pack.items() if k != "manifest_sha256"},
                                                        sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    (OUT / "PACK_hawking-lab.json").write_text(json.dumps(pack, indent=2, sort_keys=True, default=str))
    return reachdoc


def main(argv=None) -> int:
    d = build()
    r = d["python_tools_condense"]
    print(f"Rust: hawking kernel {d['rust']['hawking_kernel_loc']:,} | hide product {d['rust']['hide_product_loc']:,}")
    print(f"Python tools/condense: reachable {r['reachable_modules']} ({r['reachable_loc']:,} LOC) | "
          f"orphaned {r['orphaned_modules']} ({r['orphaned_loc']:,} LOC = {r['orphaned_fraction']*100:.0f}%)")
    print(f"true kernel floor estimate: {d['kernel_floor_estimate']['combined']:,} "
          f"(rust {d['kernel_floor_estimate']['rust_hawking']:,} + py {d['kernel_floor_estimate']['python_reachable']:,})")
    print("sealed hawking-lab pack manifest (no files moved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
