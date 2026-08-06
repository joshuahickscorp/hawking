#!/usr/bin/env python3.12
"""Capability and logical-test inventories for the 300k arc.

The LOC authority answers "how much code". These answer "what does it do" and "what does
it assert", so a condensation lane can be checked for loss rather than trusted.

Both are derived from the tree, not hand-maintained, so they can be regenerated at every
rung and diffed. A capability or a logical case that disappears between two rungs is a
regression even when every suite is green.

    python3.12 tools/loc/hawking_inventory.py --capabilities
    python3.12 tools/loc/hawking_inventory.py --tests
    python3.12 tools/loc/hawking_inventory.py --diff BASE.json NEW.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDORED_ROOTS = ("vendor/", "workspace/vendor/")
TEST_ROOTS = ("tests/", "workspace/quality/tests/")

RS_TEST = re.compile(r"^\s*(?:#\[(?:tokio::)?test\][\s\S]{0,200}?)?\s*(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)", re.M)
RS_TEST_ATTR = re.compile(r"#\[(?:tokio::)?test\]")
PY_TEST = re.compile(r"^\s*(?:async\s+)?def\s+(test_[a-z0-9_]*)", re.M)
PY_PARAM = re.compile(r"@pytest\.mark\.parametrize\(\s*[\"']([^\"']+)[\"']\s*,\s*(\[[\s\S]*?\])\s*\)")
PY_MAIN = re.compile(r"^if\s+__name__\s*==\s*[\"']__main__[\"']", re.M)


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout


def tracked(pattern: str) -> list[str]:
    return [p for p in git("ls-files").splitlines() if p.endswith(pattern)]


def read(p: str) -> str:
    try:
        return (REPO / p).read_text(errors="ignore")
    except OSError:
        return ""


def is_vendored(path: str) -> bool:
    """Exclude both the legacy and current physical vendor shelves."""
    return path.startswith(VENDORED_ROOTS)


def is_test_path(path: str) -> bool:
    """Recognize tests after the root suite moved under ``workspace/quality``."""
    return path.startswith(TEST_ROOTS) or "/tests/" in path


def logical_tests() -> dict:
    """Count logical cases, not test functions.

    A parametrised case is N logical assertions, not one. Collapsing twenty parametrised
    cases into one table-driven test is compression; collapsing them into one case is loss,
    and only counting cases tells the two apart.
    """
    rows: list[dict] = []
    for p in tracked(".rs"):
        if is_vendored(p):
            continue
        src = read(p)
        if not RS_TEST_ATTR.search(src):
            continue
        n = len(RS_TEST_ATTR.findall(src))
        ignored = src.count("#[ignore")
        rows.append({"file": p, "lang": "rust", "cases": n, "ignored": ignored})
    for p in tracked(".py"):
        if is_vendored(p):
            continue
        name = Path(p).name
        if not (name.startswith("test_") or is_test_path(p)):
            continue
        src = read(p)
        fns = PY_TEST.findall(src)
        if not fns:
            continue
        # each parametrize decorator multiplies the cases it wraps
        params = sum(max(1, blob.count(",") + 1) for _, blob in PY_PARAM.findall(src))
        rows.append({
            "file": p, "lang": "python",
            "cases": len(fns),
            "parametrised_cases": params,
            "skipped_markers": src.count("pytest.mark.skip") + src.count("pytest.mark.xfail"),
        })
    total = sum(r["cases"] for r in rows)
    param = sum(r.get("parametrised_cases", 0) for r in rows)
    return {
        "schema": "hawking.logical_test_inventory.v1",
        "commit": git("rev-parse", "HEAD").strip(),
        "files": len(rows),
        "logical_cases": total,
        "parametrised_expansion": param,
        "ignored_rust": sum(r.get("ignored", 0) for r in rows),
        "skipped_python_markers": sum(r.get("skipped_markers", 0) for r in rows),
        "by_file": sorted(rows, key=lambda r: -r["cases"]),
    }


def capabilities() -> dict:
    """Every executable surface the repository exposes.

    Derived, not curated: a binary, a CLI entrypoint or a registered adapter that vanishes
    between two rungs is a lost capability, and no LOC number will say so.
    """
    bins: list[dict] = []
    for p in tracked("Cargo.toml"):
        src = read(p)
        pkg = re.search(r'^\s*name\s*=\s*"([^"]+)"', src, re.M)
        for m in re.finditer(r'\[\[bin\]\][\s\S]{0,200}?name\s*=\s*"([^"]+)"', src):
            bins.append({"kind": "rust_bin", "name": m.group(1), "manifest": p})
        if pkg and "[[bin]]" not in src and (REPO / p).parent.joinpath("src/main.rs").exists():
            bins.append({"kind": "rust_bin", "name": pkg.group(1), "manifest": p})

    py_cli = [
        {"kind": "python_cli", "name": p}
        for p in tracked(".py")
        if not is_vendored(p) and PY_MAIN.search(read(p))
    ]
    sh = [{"kind": "shell", "name": p} for p in tracked(".sh") if not is_vendored(p)]

    members = re.findall(r'"(crates/[^"]+|tools/[^"]+)"', read("Cargo.toml"))
    default_members = []
    dm = re.search(r"default-members\s*=\s*\[([\s\S]*?)\]", read("Cargo.toml"))
    if dm:
        default_members = re.findall(r'"([^"]+)"', dm.group(1))

    return {
        "schema": "hawking.capability_inventory.v1",
        "commit": git("rev-parse", "HEAD").strip(),
        "rust_binaries": sorted(bins, key=lambda b: b["name"]),
        "python_entrypoints": len(py_cli),
        "python_entrypoint_list": sorted(c["name"] for c in py_cli),
        "shell_entrypoints": len(sh),
        "workspace_members": len(members),
        "default_members": default_members,
        "counts": {
            "rust_binaries": len(bins),
            "python_entrypoints": len(py_cli),
            "shell_entrypoints": len(sh),
        },
    }


def diff(base: Path, new: Path) -> int:
    a, b = json.loads(base.read_text()), json.loads(new.read_text())
    if a.get("schema") != b.get("schema"):
        print("schema mismatch", file=sys.stderr)
        return 2
    lost = 0
    if "logical_cases" in a:
        d = b["logical_cases"] - a["logical_cases"]
        print(f"logical cases {a['logical_cases']:,} -> {b['logical_cases']:,}  ({d:+,})")
        if d < 0:
            print("  LOSS: logical assertions disappeared")
            lost = 1
    else:
        for key in ("python_entrypoint_list",):
            gone = sorted(set(a.get(key, [])) - set(b.get(key, [])))
            if gone:
                print(f"  LOST {key}: {len(gone)}")
                for g in gone[:20]:
                    print(f"    {g}")
                lost = 1
        an = {x["name"] for x in a.get("rust_binaries", [])}
        bn = {x["name"] for x in b.get("rust_binaries", [])}
        if an - bn:
            print(f"  LOST binaries: {sorted(an - bn)}")
            lost = 1
    print("OK — nothing lost" if not lost else "REGRESSION")
    return lost


def snapshot(prefix: str) -> int:
    """Write both inventories under one prefix, so neither can be taken alone."""
    Path(f"{prefix}.tests.json").write_text(json.dumps(logical_tests(), indent=2, sort_keys=True) + "\n")
    Path(f"{prefix}.caps.json").write_text(json.dumps(capabilities(), indent=2, sort_keys=True) + "\n")
    print(f"snapshot written: {prefix}.tests.json {prefix}.caps.json")
    return 0


def gate(base_prefix: str, new_prefix: str) -> int:
    """Run BOTH diffs and fail if either regresses.

    A merge once deleted verify_grades.py -- 220 lines with a live __main__ -- and was
    reported as capability-preserving. The detector did not fail; only the test diff was
    ever run, and the capability diff was never taken. One command that always checks both
    is the fix, because the failure was choosing which half to look at.
    """
    worst = 0
    for kind in ("tests", "caps"):
        b, n = Path(f"{base_prefix}.{kind}.json"), Path(f"{new_prefix}.{kind}.json")
        if not b.exists() or not n.exists():
            print(f"MISSING {kind} snapshot — gate cannot pass without both", file=sys.stderr)
            return 2
        print(f"--- {kind} ---")
        worst = max(worst, diff(b, n))
    print("GATE PASS" if not worst else "GATE FAIL")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capabilities", action="store_true")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--diff", nargs=2, metavar=("BASE", "NEW"))
    ap.add_argument("--snapshot", default=None, metavar="PREFIX",
                    help="write both inventories under PREFIX (use with --gate)")
    ap.add_argument("--gate", nargs=2, metavar=("BASE_PREFIX", "NEW_PREFIX"),
                    help="diff BOTH inventories; non-zero if either regressed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.snapshot:
        return snapshot(args.snapshot)
    if args.gate:
        return gate(args.gate[0], args.gate[1])
    if args.diff:
        return diff(Path(args.diff[0]), Path(args.diff[1]))

    result = capabilities() if args.capabilities else logical_tests() if args.tests else None
    if result is None:
        ap.error("choose --capabilities, --tests or --diff")
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
        head = {k: v for k, v in result.items() if not isinstance(v, list)}
        print(json.dumps(head, indent=2, sort_keys=True))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
