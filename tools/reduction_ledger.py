#!/usr/bin/env python3
"""Measure Hawking's implementation surface. No reduction claim without this.

Counts only what is TRACKED, because the working tree carries gigabytes of
untracked model artifacts that are not the codebase.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=REPO,
                          capture_output=True, text=True).stdout.split()


def loc(paths: list[str]) -> int:
    total = 0
    for p in paths:
        f = REPO / p
        try:
            with f.open("rb") as handle:
                total += sum(1 for _ in handle)
        except OSError:
            pass
    return total


def main() -> int:
    files = tracked()
    by_ext = Counter(Path(f).suffix for f in files)
    py = [f for f in files if f.endswith(".py")]
    rs = [f for f in files if f.endswith(".rs")]
    md = [f for f in files if f.endswith(".md")]
    tests = [f for f in py if "test" in Path(f).name] + \
            [f for f in rs if "/tests/" in f or Path(f).name.startswith("test")]

    cargo = (REPO / "Cargo.toml").read_text()
    members = re.findall(r'"([^"]+)"', re.search(
        r'^members\s*=\s*\[(.*?)^\]', cargo, re.S | re.M).group(1))
    dm = re.search(r'^default-members\s*=\s*\[(.*?)^\]', cargo, re.S | re.M)
    default = re.findall(r'"([^"]+)"', dm.group(1)) if dm else []

    deps = set()
    for c in (REPO / "crates").glob("*/Cargo.toml"):
        body = c.read_text()
        sect = re.search(r'^\[dependencies\](.*?)(?=^\[|\Z)', body, re.S | re.M)
        if sect:
            deps |= set(re.findall(r'^([a-zA-Z0-9_-]+)\s*=', sect.group(1), re.M))

    size = 0
    for f in files:
        try:
            size += (REPO / f).stat().st_size
        except OSError:
            pass

    out = {
        "tracked_files": len(files),
        "top_level_dirs": len({f.split("/")[0] for f in files if "/" in f}),
        "tracked_mb": round(size / 1048576, 1),
        "python_files": len(py),
        "python_loc": loc(py),
        "rust_files": len(rs),
        "rust_loc": loc(rs),
        "markdown_files": len(md),
        "markdown_loc": loc(md),
        "test_files": len(tests),
        "test_loc": loc(tests),
        "total_loc": loc(py) + loc(rs),
        "workspace_crates": len(members),
        "default_crates": len(default),
        "crate_dirs": len(list((REPO / "crates").glob("*/Cargo.toml"))),
        "rust_direct_deps": len(deps),
        "by_ext": dict(by_ext.most_common(10)),
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
