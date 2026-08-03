#!/usr/bin/env python3.12
"""Topology authority for the semantic recomposition arc.

The prior arc measured LOC and hit a floor. This arc's binding order is folders, then
crates, then files, then APIs, then functions, with LOC as the consequence -- so LOC alone
cannot tell us whether we are succeeding. A tree that loses 40% of its lines while keeping
every directory has not been recomposed; it has been compressed.

This counts the structural dimensions, so a rung can be judged on all of them at once.

    python3.12 tools/loc/hawking_topology.py
    python3.12 tools/loc/hawking_topology.py --rev arc-430k-green --json
    python3.12 tools/loc/hawking_topology.py --diff BASE.json NEW.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCHEMA = "hawking.topology.v1"
REPO = Path(__file__).resolve().parents[2]
VENDORED_ROOTS = ("vendor/", "workspace/vendor/")

CODE_EXT = {".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".metal", ".lean", ".sh"}

# Public surface. Rust `pub` at item level, Python module-level def/class without underscore,
# TypeScript `export`. Approximate by design -- the point is a comparable trend, not a compiler.
RS_PUB = re.compile(r"^\s*pub(?:\(crate\))?\s+(?:async\s+)?(?:fn|struct|enum|trait|type|const|static)\s+(\w+)", re.M)
RS_FN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)", re.M)
PY_PUB = re.compile(r"^(?:def|class)\s+([a-zA-Z]\w*)", re.M)
PY_FN = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.M)
TS_PUB = re.compile(r"^\s*export\s+(?:async\s+)?(?:function|const|class|interface|type|enum)\s+(\w+)", re.M)


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout


def is_active(p: str) -> bool:
    return not p.startswith(VENDORED_ROOTS) and "/generated/" not in p


def read(rev: str | None, path: str) -> str:
    if rev is None:
        f = REPO / path
        try:
            return f.read_text(errors="ignore") if f.exists() else ""
        except OSError:
            return ""
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def measure(rev: str | None, *, include_untracked: bool = False) -> dict:
    listing = git("ls-files") if rev is None else git("ls-tree", "-r", "--name-only", rev)
    files = listing.splitlines()
    if include_untracked:
        if rev is not None:
            raise ValueError("include_untracked is only defined for the working tree")
        files = sorted(set(files) | set(git("ls-files", "--others", "--exclude-standard").splitlines()))
    src = [p for p in files if Path(p).suffix in CODE_EXT and is_active(p)]

    leaf_dirs = {str(Path(p).parent) for p in src}
    all_dirs: set[str] = set()
    for d in leaf_dirs:
        parts = d.split("/")
        for i in range(1, len(parts) + 1):
            all_dirs.add("/".join(parts[:i]))
    all_dirs.discard(".")

    per_dir = Counter(str(Path(p).parent) for p in src)
    single_file_dirs = sorted(d for d, n in per_dir.items() if n == 1)

    crates = sorted(p for p in files if p.endswith("Cargo.toml") and p.startswith("crates/"))
    hide_crates = [c for c in crates if c.startswith("crates/hide-")]

    pub = fn = 0
    big_files: list[dict] = []
    tiny_forwarders: list[str] = []
    for p in src:
        s = read(rev, p)
        if not s:
            continue
        n = len(s.split("\n")) - 1
        ext = Path(p).suffix
        if ext == ".rs":
            pub += len(RS_PUB.findall(s)); fn += len(RS_FN.findall(s))
        elif ext == ".py":
            pub += len(PY_PUB.findall(s)); fn += len(PY_FN.findall(s))
        elif ext in {".ts", ".tsx", ".js", ".jsx"}:
            pub += len(TS_PUB.findall(s))
        if n > 1500:
            big_files.append({"file": p, "lines": n})
        if n < 25 and ("pub use" in s or re.search(r"^from .+ import", s, re.M)):
            tiny_forwarders.append(p)

    by_area = Counter(p.split("/")[0] for p in src)
    hide_files = [p for p in src if p.startswith("crates/hide-") or p.startswith("app/")]
    hide_dirs = {str(Path(p).parent) for p in hide_files}

    return {
        "schema": SCHEMA,
        "include_untracked_active_source": include_untracked,
        "rev": rev or "WORKING_TREE",
        "commit": git("rev-parse", rev or "HEAD").strip(),
        "directories_all": len(all_dirs),
        "directories_leaf": len(leaf_dirs),
        "single_file_directories": len(single_file_dirs),
        "source_files": len(src),
        "rust_crates": len(crates),
        "hide_crates": len(hide_crates),
        "hide_files": len(hide_files),
        "hide_directories": len(hide_dirs),
        "public_symbols": pub,
        "functions": fn,
        "files_over_1500_lines": len(big_files),
        "tiny_forwarders": len(tiny_forwarders),
        "by_area": dict(by_area.most_common()),
        "detail": {
            "single_file_directories": single_file_dirs,
            "files_over_1500_lines": sorted(big_files, key=lambda b: -b["lines"])[:40],
            "tiny_forwarders": tiny_forwarders[:40],
        },
    }


DIMS = ["directories_all", "directories_leaf", "source_files", "rust_crates",
        "hide_crates", "hide_files", "hide_directories", "public_symbols", "functions"]


def diff(base: Path, new: Path) -> int:
    a, b = json.loads(base.read_text()), json.loads(new.read_text())
    worse = 0
    print(f"{'dimension':<24}{'base':>10}{'new':>10}{'delta':>10}{'pct':>8}")
    for k in DIMS:
        x, y = a.get(k, 0), b.get(k, 0)
        d = y - x
        pct = (100.0 * d / x) if x else 0.0
        flag = "  WORSE" if d > 0 else ""
        if d > 0:
            worse += 1
        print(f"{k:<24}{x:>10,}{y:>10,}{d:>+10,}{pct:>7.1f}%{flag}")
    print("all dimensions improved or held" if not worse else f"{worse} dimension(s) regressed")
    return 1 if worse else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--diff", nargs=2, metavar=("BASE", "NEW"))
    ap.add_argument("--include-untracked", action="store_true")
    args = ap.parse_args()

    if args.diff:
        return diff(Path(args.diff[0]), Path(args.diff[1]))

    r = measure(args.rev, include_untracked=args.include_untracked)
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(r, indent=2, sort_keys=True))
        return 0
    print(f"rev {r['rev']} ({r['commit'][:12]})")
    for k in DIMS:
        print(f"  {k:<24}{r[k]:>8,}")
    print(f"  {'files >1500 lines':<24}{r['files_over_1500_lines']:>8,}")
    print(f"  {'tiny forwarders':<24}{r['tiny_forwarders']:>8,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
