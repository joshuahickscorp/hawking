#!/usr/bin/env python3
"""Reject new public product claims that reintroduce retired Hawking identities.

Historical evidence and compatibility declarations remain legal.  The default
mode examines added lines relative to the merge base with ``main`` so existing
receipts and old prose are never rewritten to make the check pass.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


RULES = (
    re.compile(r"\bDeep Gravity\b", re.IGNORECASE),
    re.compile(r"\bSingularity Profile\b", re.IGNORECASE),
    re.compile(r"\bNoetic (?:Compiler|Program|IR)\b", re.IGNORECASE),
    re.compile(r"\bHCLI\s+(?:is|—|-)\s+", re.IGNORECASE),
    re.compile(r"\bHIDE\s+(?:is|—|-)\s+", re.IGNORECASE),
)
ALLOWED_CONTEXT = (
    "alias",
    "compatibility",
    "deprecated",
    "historical",
    "history",
    "migration",
    "retired",
    "source name",
)
EXEMPT_PREFIXES = (
    "receipts/",
    "docs/roadmap-lineage/",
    "research/",
    "docs/HAWKING_NOMENCLATURE.md",
    "hcli/nomenclature.py",
    "tools/verify/hawking_terminology.py",
)


def added_lines(diff: str):
    path = ""
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            path = raw[6:]
        elif raw.startswith("+") and not raw.startswith("+++"):
            yield path, raw[1:]


def violations(diff: str):
    found = []
    for path, line in added_lines(diff):
        if path.startswith(EXEMPT_PREFIXES):
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in ALLOWED_CONTEXT):
            continue
        for rule in RULES:
            if rule.search(line):
                found.append((path, line.strip(), rule.pattern))
    return found


def git_diff(repo: Path, base: str) -> str:
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", base],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return subprocess.run(
        ["git", "diff", "--unified=0", merge_base],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def selfcheck() -> None:
    bad = "+++ b/docs/new.md\n+Deep Gravity is a separate product\n"
    good = "+++ b/docs/new.md\n+Deep Gravity is a historical compatibility alias\n"
    assert len(violations(bad)) == 1
    assert violations(good) == []
    print("hawking terminology selfcheck: PASS")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="main")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        selfcheck()
        return 0
    found = violations(git_diff(args.repo.resolve(), args.base))
    for path, line, _rule in found:
        print(f"{path}: retired public identity in added line: {line}", file=sys.stderr)
    if found:
        print(
            "Use canonical Hawking/Gravity/NR/NX vocabulary or mark a real "
            "history/compatibility boundary on the same line.",
            file=sys.stderr,
        )
        return 1
    print("hawking terminology additions: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
