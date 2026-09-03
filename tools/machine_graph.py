#!/usr/bin/env python3
"""Classify every tracked source region of Hawking into one of nine categories.

Deterministic. Built from git, not from the working tree, because the tree
carries gigabytes of untracked model artifacts that are not the codebase.

Three laws are encoded here, each learned by getting it wrong first:

  INVENTORY RECEIPTS ARE NOT CALLERS. A census document that names every path
  in the repository hides every dead module behind it, so receipts/ and
  workspace/campaign/ are evidence, never callers.

  A FORWARDING SHIM IS NOT DEAD CODE. Sixteen lines that re-export main can be
  the installed entrypoint -- 28 of them are named by launchd plists -- so the
  entrypoint graph is consulted before anything is called dead.

  TEST GREEN IS NOT TOPOLOGY SAFE. Reachability is computed over dotted imports
  as well as path strings, because a move can rewrite every `lab/` and leave
  1,247 `from lab.operators import` broken with the suite still green.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")

#: Evidence, not callers.
INVENTORY = ("receipts/", "workspace/campaign/", "research/receipts/")
CALLER_EXT = {".py", ".rs", ".toml", ".yml", ".yaml", ".sh", ".md", ".json",
              ".plist", ".txt", ""}

CATEGORIES = (
    "CURRENT_MACHINE", "VERIFIER_ORACLE", "ODYSSEY_REQUIRED", "HARDWARE_SEAM",
    "HISTORICAL_SCIENCE", "SUPERSEDED_IMPLEMENTATION", "GENERATED_STATE",
    "DEAD", "UNKNOWN",
)


def tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=REPO,
                          capture_output=True, text=True).stdout.split()


def loc(path: str) -> int:
    try:
        with (REPO / path).open("rb") as h:
            return sum(1 for _ in h)
    except OSError:
        return 0


def entrypoints(files: list[str]) -> set[str]:
    """Basenames named by a plist, shell script, CI workflow or cargo bin."""
    named: set[str] = set()
    for f in files:
        if not f.endswith((".plist", ".sh", ".yml", ".yaml")):
            continue
        try:
            text = (REPO / f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        named |= {Path(m).name for m in re.findall(r"[A-Za-z0-9_/.-]+\.(?:py|rs|sh)", text)}
    for c in (REPO / "crates").glob("*/Cargo.toml"):
        try:
            body = c.read_text()
        except OSError:
            continue
        if "[[bin]]" in body:
            named.add(c.parent.name)
    return named


def _operator_cli(f: str) -> bool:
    """A top-level tools/ script with a __main__ guard is an operator surface.

    Nothing imports it and no plist names it, because a human runs it. The
    graph cannot see that from references alone, and calling it dead would
    delete the very instruments this campaign is measured with.
    """
    if not f.startswith("tools/") or f.count("/") != 1 or not f.endswith(".py"):
        return False
    try:
        return "__main__" in (REPO / f).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def build(files: list[str]) -> dict:
    caller_text: list[tuple[str, str]] = []
    inventory_text: list[tuple[str, str]] = []
    for f in files:
        p = REPO / f
        if p.suffix not in CALLER_EXT:
            continue
        try:
            if p.stat().st_size > 8_000_000:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        (inventory_text if f.startswith(INVENTORY) else caller_text).append((f, text))

    caller_blob = "\n".join(t for _, t in caller_text)
    inventory_blob = "\n".join(t for _, t in inventory_text)
    eps = entrypoints(files)
    # Whole tokens only. A substring test against the roadmap blob matched any
    # short stem and classified 683 files as roadmap-required, which is not a
    # classification, it is a coincidence.
    roadmap = set()
    for f, t in caller_text:
        if f.startswith("civilization/"):
            roadmap |= set(TOK.findall(t))

    out: dict[str, dict] = {}
    for f in files:
        p = Path(f)
        if p.suffix not in (".py", ".rs", ".metal"):
            continue
        stem, name = p.stem, p.name
        n = loc(f)

        if f.startswith(INVENTORY):
            cat, why = "GENERATED_STATE", "lives under an evidence tree"
        elif name in eps or stem in eps:
            cat, why = "CURRENT_MACHINE", "named by an entrypoint (plist/script/CI/bin)"
        elif _operator_cli(f):
            cat, why = "CURRENT_MACHINE", "operator CLI: runnable, invoked by hand"
        elif p.name.startswith("test_") or "/tests/" in f:
            cat, why = "VERIFIER_ORACLE", "test"
        elif stem in roadmap:
            cat, why = "ODYSSEY_REQUIRED", "named in civilization roadmap state"
        else:
            # Dotted imports AND Rust `mod x;` declarations count. The trailing
            # character class must include `;` or every module declared with
            # `mod foo;` reads as DEAD -- which is how a vendored crate's whole
            # source tree was reported dead on the first pass. `from lab.operators.x import y` never
            # contains "from x", so a check for the bare form reports a module
            # with a live test as DEAD -- the dangerous direction to be wrong in.
            called = (
                name in caller_blob
                or re.search(rf"(?:^|[\s.:]){re.escape(stem)}(?:[\s.(,;:}}]|$)", caller_blob, re.M) is not None
            )
            if called:
                cat, why = "CURRENT_MACHINE", "referenced by a caller"
            elif name in inventory_blob or stem in inventory_blob:
                cat, why = "HISTORICAL_SCIENCE", "named only by evidence: producer, science kept"
            else:
                cat, why = "DEAD", "no caller, no evidence, no entrypoint"
        out[f] = {"loc": n, "category": cat, "why": why}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files = tracked()
    graph = build(files)
    Path(REPO / ".hcli" / "machine_graph.json").parent.mkdir(parents=True, exist_ok=True)
    (REPO / ".hcli" / "machine_graph.json").write_text(json.dumps(graph, indent=1))

    by = collections.Counter()
    locs = collections.Counter()
    for f, r in graph.items():
        by[r["category"]] += 1
        locs[r["category"]] += r["loc"]
    total_loc = sum(locs.values())

    if args.json:
        print(json.dumps({"files": dict(by), "loc": dict(locs)}, indent=1))
        return 0

    print(f"{'category':28} {'files':>6} {'LOC':>10}  share")
    for c in CATEGORIES:
        if not by[c]:
            continue
        print(f"{c:28} {by[c]:6} {locs[c]:10,}  {100*locs[c]/total_loc:5.1f}%")
    print(f"{'TOTAL':28} {sum(by.values()):6} {total_loc:10,}")
    classified = total_loc - locs["UNKNOWN"]
    print(f"\nclassified: {100*classified/total_loc:.1f}% of source LOC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
