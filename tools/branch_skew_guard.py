#!/usr/bin/env python3
"""Branch-skew guard: find the files a lane and trunk both touched.

Lanes are cut from the HEAD at launch time and land hours later, by which point
trunk has moved. The failure mode this prevents is a wholesale file copy from a
lane branch silently reverting trunk work that landed in the meantime — which has
happened on this repo before.

The rule it enforces: where a lane and trunk touched the SAME file, there is no
wholesale copy. Rebase, minimal graft, or manual composition only.

    python3 tools/branch_skew_guard.py                       # every grok/* lane
    python3 tools/branch_skew_guard.py grok/q80-pack-2026...  # one lane
    python3 tools/branch_skew_guard.py --trunk main --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def lane_branches() -> list[str]:
    out = git("for-each-ref", "--format=%(refname:short)", "refs/heads/grok/")
    return [b for b in out.splitlines() if b]


def files(rev_range: str) -> set[str]:
    out = git("diff", "--name-only", rev_range)
    return {f for f in out.splitlines() if f}


def _verdict(ahead: int, has_overlap: bool, behind: int) -> str:
    if ahead == 0:
        return "EMPTY"          # nothing to integrate
    if has_overlap:
        return "SKEWED"         # manual composition required
    if behind:
        return "STALE_CLEAN"    # behind trunk but no shared files
    return "CLEAN"


def audit(branch: str, trunk: str) -> dict:
    base = git("merge-base", trunk, branch)
    if not base:
        return {"branch": branch, "error": "no merge-base with trunk"}

    lane_files = files(f"{base}...{branch}")
    trunk_files = files(f"{base}...{trunk}")
    overlap = sorted(lane_files & trunk_files)
    behind = len([c for c in git("log", "--oneline", f"{branch}..{trunk}").splitlines() if c])
    ahead = len([c for c in git("log", "--oneline", f"{trunk}..{branch}").splitlines() if c])

    verdict = _verdict(ahead, bool(overlap), behind)

    return {
        "branch": branch,
        "base": base[:8],
        "commits_ahead": ahead,
        "commits_behind_trunk": behind,
        "lane_files": len(lane_files),
        "overlap_files": overlap,
        "verdict": verdict,
    }


VERDICT_NOTE = {
    "SKEWED": "NO WHOLESALE FILE COPY - rebase, minimal graft, or hand-compose these files",
    "STALE_CLEAN": "behind trunk but no shared files; rebase then re-run protected markers",
    "CLEAN": "safe to integrate; still re-run protected markers after",
    "EMPTY": "no commits on this lane - check for uncommitted work before reaping",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("branches", nargs="*")
    ap.add_argument("--trunk", default="main")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return 0

    targets = args.branches or lane_branches()
    reports = [audit(b, args.trunk) for b in targets]

    if args.json:
        json.dump(reports, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    worst = 0
    for r in sorted(reports, key=lambda r: r.get("verdict", "")):
        if "error" in r:
            print(f"  [ERROR       ] {r['branch']}: {r['error']}")
            continue
        v = r["verdict"]
        if v == "EMPTY":
            continue
        print(f"  [{v:<12}] {r['branch']}")
        print(f"      base={r['base']} ahead={r['commits_ahead']} "
              f"behind={r['commits_behind_trunk']} lane_files={r['lane_files']}")
        print(f"      {VERDICT_NOTE[v]}")
        for f in r["overlap_files"]:
            print(f"        ! both touched: {f}")
        worst = max(worst, 2 if v == "SKEWED" else 1)
    empty = sum(1 for r in reports if r.get("verdict") == "EMPTY")
    if empty:
        print(f"  ({empty} lane branch(es) with no commits - not shown)")
    return worst


def _selfcheck() -> None:
    """The one thing that must never happen: a shared-file lane called safe."""
    assert _verdict(ahead=1, has_overlap=True, behind=3) == "SKEWED"
    assert _verdict(ahead=1, has_overlap=True, behind=0) == "SKEWED", (
        "overlap outranks being up to date"
    )
    assert _verdict(ahead=1, has_overlap=False, behind=3) == "STALE_CLEAN"
    assert _verdict(ahead=1, has_overlap=False, behind=0) == "CLEAN"
    assert _verdict(ahead=0, has_overlap=True, behind=3) == "EMPTY"
    assert VERDICT_NOTE.keys() >= {"SKEWED", "STALE_CLEAN", "CLEAN", "EMPTY"}
    print("selfcheck ok")


if __name__ == "__main__":
    sys.exit(main())
