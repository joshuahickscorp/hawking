"""CODEX_MUTATION_SURFACE — who is allowed to write where.

Codex owns the live physical Accelerator frontier. Claude/Grok must not mutate
files Codex is holding. This module records the ownership map with mtime
evidence and provides a checker that FAILS when a proposed write path lands on
the Codex surface.

    python3 tools/future/mutation_surface.py --build
    python3 tools/future/mutation_surface.py --check-disjoint tools/future receipts/future
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path
from typing import Any

from tools.future._common import REPO, git, newest_mtime, write_receipt

RECEIPT = "CODEX_MUTATION_SURFACE.json"

# Paths Codex actively mutates during an Accelerator campaign. Ordered most
# specific first so SIDECAR_OWNED can carve out of a broader Codex glob.
SIDECAR_OWNED = (
    "tools/future/*",
    "receipts/future/*",
)

CODEX_OWNED = (
    "crates/*",
    "hcli/*",
    "tools/*",
    "civilization/*",
    "docs/*",
    "receipts/headless/*",
    "receipts/odyssey-i/*",
    "docs/spec/*",
    "docs/contracts/*",
)

# Quiescence threshold: source untouched for this long means the integration
# window in directive section 77 is open. It never means Codex cannot resume.
QUIESCENT_SECONDS = 6 * 3600


def owner(rel: str) -> str:
    """Which authority owns this repo-relative path."""
    for pat in SIDECAR_OWNED:
        if fnmatch.fnmatch(rel, pat) or rel == pat.rstrip("/*"):
            return "SIDECAR"
    for pat in CODEX_OWNED:
        if fnmatch.fnmatch(rel, pat):
            return "CODEX"
    return "SHARED"


def intersects_codex(rel: str) -> bool:
    return owner(rel) == "CODEX"


def _surface_evidence() -> list[dict[str, Any]]:
    now = time.time()
    rows = []
    for pat in CODEX_OWNED:
        root = REPO / pat.rstrip("/*")
        if not root.exists():
            continue
        mtime, who = newest_mtime(root, skip=("/future/",))
        rows.append(
            {
                "glob": pat,
                "exists": True,
                "newest_mtime_epoch": round(mtime, 3),
                "newest_file": who,
                "age_seconds": round(now - mtime, 1) if mtime else None,
                "quiescent": bool(mtime and (now - mtime) > QUIESCENT_SECONDS),
            }
        )
    return rows


def build() -> Path:
    evidence = _surface_evidence()
    doc = {
        "schema": "hawking.future.codex_mutation_surface.v1",
        "version": 1,
        "purpose": (
            "Ownership map preventing Claude/Grok sidecar lanes from mutating "
            "files the live Codex Accelerator campaign is holding."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_paths": len([l for l in git("status", "--porcelain").splitlines() if l]),
        "codex_owned": list(CODEX_OWNED),
        "sidecar_owned": list(SIDECAR_OWNED),
        "carve_out_rule": (
            "SIDECAR globs are evaluated first, so tools/future/* is sidecar-owned "
            "even though tools/* is listed as Codex-owned."
        ),
        "quiescence_threshold_seconds": QUIESCENT_SECONDS,
        "surface_evidence": evidence,
        "integration_window_open": all(r["quiescent"] for r in evidence if r["exists"]),
        "policy": {
            "sidecar_writes": "only into SIDECAR globs",
            "codex_writes": "unconstrained; Codex is the live frontier",
            "on_conflict": "sidecar yields; prepare an integration bundle instead",
            "quiescence_is_not_permission": (
                "A quiescent surface opens an integration window. It never grants "
                "the sidecar authority to edit a Codex file."
            ),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/mutation_surface.py")


def check_disjoint(paths: list[str]) -> int:
    bad = []
    for p in paths:
        rel = os.path.relpath(os.path.abspath(p), REPO)
        if intersects_codex(rel):
            bad.append(rel)
        # Also check every file underneath a directory argument.
        full = REPO / rel
        if full.is_dir():
            for dirpath, dirnames, filenames in os.walk(full):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for fn in filenames:
                    child = os.path.relpath(os.path.join(dirpath, fn), REPO)
                    if intersects_codex(child):
                        bad.append(child)
    if bad:
        print("CODEX SURFACE COLLISION:", file=sys.stderr)
        for b in sorted(set(bad))[:40]:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"disjoint: {len(paths)} path(s) clear of the Codex mutation surface")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check-disjoint", nargs="+", metavar="PATH")
    a = ap.parse_args()
    if a.check_disjoint:
        return check_disjoint(a.check_disjoint)
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
