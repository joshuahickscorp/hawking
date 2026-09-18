#!/usr/bin/env python3.12
"""The single LOC authority for the Hawking 300k semantic-density arc.

One tool, one policy. Every checkpoint in the ladder is measured with this and
nothing else, so a number is comparable to the number before it. The report
keeps the broad total active LOC separate from the narrower minimum-product
boundary required by Phase IV.

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
import re
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
    if path.startswith("tools/") or path.startswith("research/ramanujan/"):
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


def is_product(path: str) -> bool:
    """Return whether an active source file belongs to the minimum product.

    Product LOC is deliberately conservative and mechanical: HCLI Python,
    Rust/shader implementation under crate ``src``/``shaders`` trees, and the
    retained VMCP boundary adapters. Research, experiments, acceptance/oracle
    harnesses, examples, tests, documentation, and receipts remain in total
    active LOC but are not counted as product implementation.
    """
    if path.startswith("hcli/"):
        return not is_test(path)
    if path.startswith("crates/"):
        return (
            ("/src/" in path or "/shaders/" in path)
            and "/tests/" not in path
            and "/examples/" not in path
            and "/benches/" not in path
            and not is_test(path)
        )
    if path.startswith("hcli/agentos/vmcp/"):
        return not is_test(path)
    return False


def git(args: list[str]) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def line_count(rev: str | None, path: str) -> int:
    return len(file_bytes(rev, path).split(b"\n")) - 1


def file_bytes(rev: str | None, path: str) -> bytes:
    if rev is None:
        f = REPO / path
        try:
            return f.read_bytes() if f.exists() else b""
        except OSError:
            return b""
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "show", f"{rev}:{path}"],
            capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return b""


def rust_test_excluding_count(source: bytes) -> int:
    """Count physical Rust lines outside items guarded by ``cfg(test)``.

    This is a deliberately stable syntactic census rather than a build-profile
    estimate. It excludes the cfg attribute, adjacent attributes and the whole
    following item. Braces inside ordinary strings and comments are ignored so
    an assertion message cannot alter the measured item boundary.
    """
    lines = source.split(b"\n")[:-1]
    text = [line.decode("utf-8", "replace") for line in lines]
    shape: list[str] = []
    block_comment_depth = 0
    raw_terminator = ""
    for line in text:
        out: list[str] = []
        index = 0
        in_string = False
        escaped = False
        while index < len(line):
            pair = line[index:index + 2]
            char = line[index]
            if raw_terminator:
                if line.startswith(raw_terminator, index):
                    index += len(raw_terminator)
                    raw_terminator = ""
                else:
                    index += 1
                out.append(" ")
                continue
            if block_comment_depth:
                if pair == "/*":
                    block_comment_depth += 1
                    index += 2
                elif pair == "*/":
                    block_comment_depth -= 1
                    index += 2
                else:
                    index += 1
                continue
            if in_string:
                out.append(" ")
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                index += 1
                continue
            raw = re.match(r'(?:br|r)(?P<hashes>#{0,255})"', line[index:])
            if raw:
                raw_terminator = '"' + raw.group("hashes")
                width = raw.end()
                out.extend(" " * width)
                index += width
                continue
            char_literal = re.match(r"'(?:\\.|[^\\'])'", line[index:])
            if char_literal:
                width = char_literal.end()
                out.extend(" " * width)
                index += width
                continue
            if pair == "//":
                break
            if pair == "/*":
                block_comment_depth = 1
                index += 2
                continue
            if char == '"':
                in_string = True
                out.append(" ")
            else:
                out.append(char)
            index += 1
        shape.append("".join(out))

    excluded: set[int] = set()
    index = 0
    while index < len(shape):
        stripped = shape[index].strip()
        if not stripped.startswith("#[cfg"):
            index += 1
            continue
        attribute_end = index
        attribute = stripped
        while "]" not in attribute and attribute_end + 1 < len(shape):
            attribute_end += 1
            attribute += shape[attribute_end].strip()
        if "test" not in attribute:
            index = attribute_end + 1
            continue

        item_start = attribute_end + 1
        while item_start < len(shape) and (
            not shape[item_start].strip()
            or shape[item_start].lstrip().startswith("#")
        ):
            item_start += 1
        end = item_start
        depth = 0
        opened = False
        while end < len(shape):
            code = shape[end]
            if not opened and ";" in code and "{" not in code:
                break
            opens = code.count("{")
            closes = code.count("}")
            if opens:
                opened = True
            depth += opens - closes
            if opened and depth <= 0:
                break
            end += 1
        excluded.update(range(index, min(end + 1, len(shape))))
        index = max(end + 1, index + 1)
    return len(lines) - len(excluded)


def strict_runtime_count(rev: str | None, path: str, *, test_file: bool) -> int:
    """Physical runtime lines with path tests and inline Rust tests removed."""
    if test_file:
        return 0
    source = file_bytes(rev, path)
    if path.endswith(".rs"):
        return rust_test_excluding_count(source)
    return len(source.split(b"\n")) - 1


def rust_python_composition(langs: dict[str, int], scope: str) -> dict:
    """Report the Rust mandate without changing the established LOC policy."""
    rust = langs.get("rust", 0)
    python = langs.get("python", 0)
    denominator = rust + python
    return {
        "scope": scope,
        "rust_LOC": rust,
        "python_LOC": python,
        "denominator_LOC": denominator,
        "rust_percent": round((100.0 * rust / denominator), 6) if denominator else None,
    }


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
    product_loc = 0
    product_files = 0
    product_langs: dict[str, int] = {}
    strict_runtime_langs: dict[str, int] = {}
    strict_product_langs: dict[str, int] = {}
    strict_runtime_loc = 0
    strict_product_loc = 0
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
        test_file = is_test(path)
        if test_file:
            test_loc += n
        else:
            runtime_loc += n
        strict_n = strict_runtime_count(rev, path, test_file=test_file)
        strict_runtime_loc += strict_n
        strict_runtime_langs[lang] = strict_runtime_langs.get(lang, 0) + strict_n
        if is_product(path):
            product_loc += n
            product_files += 1
            product_langs[lang] = product_langs.get(lang, 0) + n
            strict_product_loc += strict_n
            strict_product_langs[lang] = strict_product_langs.get(lang, 0) + strict_n

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
        "test_excluding_runtime_LOC": strict_runtime_loc,
        "inline_or_path_test_LOC": combined - strict_runtime_loc,
        "product_LOC": product_loc,
        "test_excluding_product_LOC": strict_product_loc,
        "product_files": product_files,
        "product_by_language": dict(sorted(product_langs.items())),
        "test_excluding_runtime_by_language": dict(sorted(strict_runtime_langs.items())),
        "test_excluding_product_by_language": dict(sorted(strict_product_langs.items())),
        "rust_python_composition": rust_python_composition(
            langs, "all active tracked first-party Rust and Python physical lines"
        ),
        "product_rust_python_composition": rust_python_composition(
            product_langs,
            "minimum-product Rust and Python physical lines selected by is_product",
        ),
        "test_excluding_runtime_rust_python_composition": rust_python_composition(
            strict_runtime_langs,
            "active first-party runtime lines after path tests and inline cfg(test) Rust items are excluded",
        ),
        "test_excluding_product_rust_python_composition": rust_python_composition(
            strict_product_langs,
            "minimum-product runtime lines after path tests and inline cfg(test) Rust items are excluded",
        ),
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
            "included_paths": "tracked active source extensions in LANGS",
            "test_excluding_rule": "exclude test/bench paths and complete Rust items carrying cfg(test)",
            "minimum_product_paths": "hcli non-test Python plus crates/*/src and crates/*/shaders, excluding examples, benches and tests",
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
    print(f"minimum product LOC: {r['product_LOC']:,}  in {r['product_files']:,} files")
    for k, v in r["by_language"].items():
        print(f"  {k:<12} {v:>9,}")
    for label, key in (
        ("active Rust/Python", "rust_python_composition"),
        ("product Rust/Python", "product_rust_python_composition"),
        ("strict runtime Rust/Python", "test_excluding_runtime_rust_python_composition"),
        ("strict product Rust/Python", "test_excluding_product_rust_python_composition"),
    ):
        row = r[key]
        print(
            f"{label}: {row['rust_LOC']:,} / {row['denominator_LOC']:,} "
            f"Rust ({row['rust_percent']:.2f}%)"
        )
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
    assert is_product("crates/hawking-core/src/lib.rs")
    assert is_product("hcli/engine.py")
    assert is_product("hcli/agentos/vmcp/file_eye.py")
    assert rust_python_composition({"rust": 3, "python": 1}, "test") == {
        "scope": "test",
        "rust_LOC": 3,
        "python_LOC": 1,
        "denominator_LOC": 4,
        "rust_percent": 75.0,
    }
    assert not is_product("crates/hawking-core/examples/flash_fast_chain.rs")
    assert not is_product("research/lab/runtime.py")
    fixture = b"pub fn live() {}\n#[cfg(test)]\nmod tests {\n  #[test]\n  fn check() { assert_eq!(\"}\", \"}\"); }\n}\npub fn after() {}\n"
    assert rust_test_excluding_count(fixture) == 2
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
