#!/usr/bin/env python3.12
"""The single LOC authority for the Hawking 300k semantic-density arc.

One tool, one policy. Every checkpoint in the ladder is measured with this and
nothing else, so a number is comparable to the number before it.

What counts as *active* LOC: every physical line, including blanks and comments,
of a tracked source file that is not archived, not generated, and not vendored.
Physical lines are deliberate. Stripping comments or packing lines would change
the number without changing the repository, and the campaign forbids exactly that.

Run against the working tree:

    python3.12 tools/loc/hawking_loc.py

or against any committed tree:

    python3.12 tools/loc/hawking_loc.py --rev hawking-loc-r250-v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCHEMA = "hawking.loc.authority.v1"
REPO = Path(__file__).resolve().parents[2]

# Keep historical revisions measurable while recognizing the current workspace
# shelf.  The old roots remain here intentionally: ``--rev`` can still inspect a
# commit from before the physical-layout change.
VENDORED_ROOTS = ("vendor/", "workspace/vendor/")
DOC_ARCHIVE_ROOTS = ("docs/archive/", "workspace/docs/archive/")
TEST_ROOTS = ("tests/", "workspace/quality/tests/")

# Extension -> language bucket. Anything not listed is not code and is not counted.
LANGS = {
    ".rs": "rust",
    ".py": "python",
    ".md": "markdown",
    ".sh": "shell",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".metal": "shader",
    ".lean": "lean",
}

# Classification is ordered: the first rule that matches a path wins.
# (bucket, predicate) — buckets other than "active" are reported but excluded
# from the headline combined number.
def classify(path: str) -> str:
    p = path
    if p.startswith(VENDORED_ROOTS):
        return "vendored"
    if "/generated/" in p or p.endswith(".generated.rs") or p.endswith(".generated.ts"):
        return "generated"
    # Only documentation archives are excluded. An earlier version excluded any path
    # containing "/archive/", and a lane moved 102,159 lines of still-runnable Python --
    # 161 modules with live __main__ entrypoints -- into tools/condense/archive/. The
    # headline dropped 86,211 without a line being eliminated. Renaming a directory is
    # not archiving, and the campaign counts neither packs, archives nor relocation as
    # condensation. Executable code is active wherever it sits.
    if p.startswith(DOC_ARCHIVE_ROOTS) and not p.endswith((".py", ".rs", ".sh", ".ts", ".tsx")):
        return "archived"
    if "/target/" in p or p.startswith("target/"):
        return "build"
    if "/node_modules/" in p:
        return "build"
    return "active"


def subsystem(path: str) -> str:
    if path.startswith("crates/hide-") or path.startswith("app/"):
        return "hide"
    if path.startswith("crates/"):
        return "hawking"
    if path.startswith("tools/") or path.startswith("ramanujan/"):
        return "laboratory"
    return "shared"


def is_test(path: str) -> bool:
    return (
        "/tests/" in path
        or path.startswith(TEST_ROOTS)
        or Path(path).name.startswith("test_")
        or Path(path).name.endswith("_test.rs")
        or "/benches/" in path
    )


def git(args: list[str]) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def line_count(rev: str | None, path: str) -> int:
    if rev is None:
        f = REPO / path
        try:
            return len(f.read_bytes().split(b"\n")) - 1 if f.exists() else 0
        except OSError:
            return 0
    try:
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{rev}:{path}"],
            capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return 0
    return len(blob.split(b"\n")) - 1


def measure(rev: str | None, *, include_untracked: bool = False) -> dict:
    if rev is None:
        files = git(["ls-files"]).splitlines()
        if include_untracked:
            files = sorted(set(files) | set(git(["ls-files", "--others", "--exclude-standard"]).splitlines()))
    else:
        if include_untracked:
            raise ValueError("include_untracked is only defined for the working tree")
        files = git(["ls-tree", "-r", "--name-only", rev]).splitlines()

    buckets: dict[str, int] = {}
    langs: dict[str, int] = {}
    subs: dict[str, int] = {}
    test_loc = 0
    runtime_loc = 0
    n_active = 0

    for path in files:
        ext = Path(path).suffix
        lang = LANGS.get(ext)
        if lang is None:
            continue
        n = line_count(rev, path)
        if n == 0:
            continue
        bucket = classify(path)
        buckets[bucket] = buckets.get(bucket, 0) + n
        if bucket != "active":
            continue
        n_active += 1
        langs[lang] = langs.get(lang, 0) + n
        sub = subsystem(path)
        subs[sub] = subs.get(sub, 0) + n
        if is_test(path):
            test_loc += n
        else:
            runtime_loc += n

    combined = sum(langs.values())
    return {
        "schema": SCHEMA,
        "rev": rev or "WORKING_TREE",
        "commit": git(["rev-parse", rev or "HEAD"]).strip(),
        "combined_active_monorepo_LOC": combined,
        "by_language": dict(sorted(langs.items())),
        "by_subsystem": dict(sorted(subs.items())),
        "hawking_active_LOC": subs.get("hawking", 0),
        "hide_active_LOC": subs.get("hide", 0),
        "shared_contract_LOC": subs.get("shared", 0),
        "laboratory_LOC": subs.get("laboratory", 0),
        "test_LOC": test_loc,
        "runtime_LOC": runtime_loc,
        "generated_LOC": buckets.get("generated", 0),
        "archived_LOC": buckets.get("archived", 0),
        "vendored_LOC": buckets.get("vendored", 0),
        "active_files": n_active,
        "policy": {
            "counts": (
                "physical lines of tracked and non-ignored untracked source files"
                if include_untracked else "physical lines of tracked source files"
            ),
            "include_untracked_active_source": include_untracked,
            "languages": sorted(set(LANGS.values())),
            "excluded_buckets": ["vendored", "generated", "archived", "build"],
            "no_gaming": (
                "physical lines only; comment stripping, line packing, minification, "
                "extension renaming and moving code out of the tree change nothing"
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default=None, help="git rev to measure; default working tree")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--ledger", default=None, help="append the result to this jsonl ledger")
    ap.add_argument("--note", default="", help="note recorded in the ledger row")
    ap.add_argument("--include-untracked", action="store_true", help="include non-ignored untracked source in a working-tree audit")
    args = ap.parse_args()

    result = measure(args.rev, include_untracked=args.include_untracked)
    if args.ledger:
        row = dict(result)
        row["note"] = args.note
        with open(args.ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    r = result
    print(f"rev {r['rev']}  ({r['commit'][:12]})")
    print(f"combined active LOC: {r['combined_active_monorepo_LOC']:,}  in {r['active_files']:,} files")
    for k, v in r["by_language"].items():
        print(f"  {k:<12} {v:>9,}")
    print("subsystem:")
    for k, v in r["by_subsystem"].items():
        print(f"  {k:<12} {v:>9,}")
    print(f"  {'test':<12} {r['test_LOC']:>9,}")
    print(f"  {'runtime':<12} {r['runtime_LOC']:>9,}")
    print(f"excluded: generated {r['generated_LOC']:,}  archived {r['archived_LOC']:,}  vendored {r['vendored_LOC']:,}")
    return 0


def _selfcheck() -> None:
    """Smallest check that fails if the policy drifts."""
    assert classify("workspace/vendor/strand-quant/src/lib.rs") == "vendored"
    assert classify("vendor/strand-quant/src/lib.rs") == "vendored"
    assert classify("crates/hawking-adapters/generated/abi.rs") == "generated"
    assert classify("workspace/docs/archive/old.md") == "archived"
    assert classify("docs/archive/old.md") == "archived"
    assert classify("crates/hawking-core/src/lib.rs") == "active"
    assert subsystem("crates/hide-core/src/lib.rs") == "hide"
    assert subsystem("crates/hawking-core/src/lib.rs") == "hawking"
    assert subsystem("tools/condense/glm52_state.py") == "laboratory"
    assert is_test("crates/hawking-core/tests/parity.rs")
    assert is_test("workspace/quality/tests/fixtures/test_x.py")
    assert is_test("tools/condense/tests/test_x.py")
    assert not is_test("crates/hawking-core/src/lib.rs")
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
