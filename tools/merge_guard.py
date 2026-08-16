#!/usr/bin/env python3
"""Refuse to merge a lane whose measured baseline no longer exists.

The failure this prevents happened three times on 2026-08-16, and it is
structural to running lanes in parallel rather than anyone's mistake:

  gk_*_simd     claimed 2-45x against the serial extract that the reconstruction
                fix had already deleted. Measured properly: 2.5x SLOWER.
  rice penalty  a 5.9x reconstruction cost transferred between models, which
                turned out to be an artifact of a kernel choice.
  DeltaNet port measured host recurrent at 43.26 ms against a 1376 ms token,
                on a branch predating the fix that took the token to 301 ms -
                and touching the same file.

A lane result is valid only against the HEAD it was measured on. So before
merging, ask which landed commits the lane never saw, and whether any of them
touched the files the lane also touches. Overlap means the lane's baseline is
stale and the number must be re-measured, not merged.

    merge_guard.py <lane-branch> [--base main]

Exit codes: 0 safe, 1 STALE (overlap found), 2 usage/git error.
"""
from __future__ import annotations

import subprocess
import sys


def sh(cmd: str) -> tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def files_of(rev_range: str) -> set[str]:
    code, out = sh(f"git diff --name-only {rev_range}")
    if code != 0:
        return set()
    # source only; receipts and artifacts colliding are not a correctness risk
    return {f for f in out.splitlines() if f.endswith((".rs", ".metal", ".py", ".toml"))}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    lane = args[0]
    base = "main"
    if "--base" in sys.argv:
        base = sys.argv[sys.argv.index("--base") + 1]

    code, _ = sh(f"git rev-parse --verify {lane}")
    if code != 0:
        print(f"no such branch: {lane}", file=sys.stderr)
        return 2

    fork = sh(f"git merge-base {base} {lane}")[1]
    lane_files = files_of(f"{fork}...{lane}")
    if not lane_files:
        # Two very different cases print the same way, so distinguish them.
        # A live lane has its work UNCOMMITTED in its worktree - lanes finish
        # uncommitted here - so the branch legitimately shows no diff yet. This
        # guard runs at MERGE time, after the work is preserved, not in flight.
        wt = subprocess.run(
            f"git worktree list --porcelain | grep -A2 -F '{lane}' || true",
            shell=True, capture_output=True, text=True).stdout
        if wt.strip():
            print(f"{lane}: no committed source diff, but a worktree is CHECKED OUT.")
            print("  Work is probably still uncommitted. Preserve it first, then re-run.")
            return 2
        print(f"{lane}: touches no source files - nothing to guard")
        return 0

    # commits on base the lane never saw
    code, missed = sh(f"git log --format=%H {fork}..{base}")
    missed_shas = [s for s in missed.splitlines() if s]
    if not missed_shas:
        print(f"{lane}: up to date with {base} - SAFE")
        return 0

    overlapping = []
    for sha in missed_shas:
        touched = files_of(f"{sha}^!")
        common = touched & lane_files
        if common:
            subj = sh(f"git log -1 --format=%s {sha}")[1]
            overlapping.append((sha[:9], subj, sorted(common)))

    print(f"{lane}")
    print(f"  forked at {fork[:9]}, missing {len(missed_shas)} commit(s) from {base}")
    print(f"  lane touches {len(lane_files)} source file(s)")
    if not overlapping:
        print("  no overlap with anything it missed - SAFE to merge")
        return 0

    print(f"\n  STALE: {len(overlapping)} missed commit(s) touch the SAME files:")
    for sha, subj, common in overlapping:
        print(f"    {sha}  {subj[:70]}")
        for f in common[:4]:
            print(f"        {f}")
    print("\n  This lane's baseline no longer exists. Rebase and RE-MEASURE.")
    print("  Do not merge a number whose baseline was deleted underneath it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
