#!/usr/bin/env python3
"""Measure the repository against the extreme condensation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "condensation.json"
PRIMARY_SUFFIXES = {
    ".py",
    ".rs",
    ".sh",
    ".md",
    ".ts",
    ".tsx",
    ".css",
    ".metal",
}
RUST_TEST = re.compile(rb"^[ \t]*#\[(?:tokio::)?test[^\]]*\]", re.MULTILINE)
RUST_IGNORE = re.compile(rb"^[ \t]*#\[ignore[^\]]*\]", re.MULTILINE)


def _repository_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    )
    return [
        relative
        for row in output.split(b"\0")
        if row
        for relative in [row.decode("utf-8")]
        if (ROOT / relative).is_file()
    ]


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def measure() -> dict[str, int]:
    files = _repository_files()
    directories: set[str] = set()
    primary_lines = 0
    markdown_lines = 0
    condense_python_lines = 0
    markdown_files = 0
    condense_python_files = 0
    rust_test_attributes = 0
    rust_ignore_attributes = 0
    for relative in files:
        pure = PurePosixPath(relative)
        parent = pure.parent
        while str(parent) != ".":
            directories.add(str(parent))
            parent = parent.parent
        suffix = pure.suffix.lower()
        lines = None
        if suffix in PRIMARY_SUFFIXES:
            lines = _line_count(ROOT / relative)
            primary_lines += lines
        if suffix == ".md":
            markdown_files += 1
            markdown_lines += lines if lines is not None else _line_count(ROOT / relative)
        if relative.startswith("tools/condense/") and suffix == ".py":
            condense_python_files += 1
            condense_python_lines += (
                lines if lines is not None else _line_count(ROOT / relative)
            )
        if suffix == ".rs":
            source = (ROOT / relative).read_bytes()
            rust_test_attributes += len(RUST_TEST.findall(source))
            rust_ignore_attributes += len(RUST_IGNORE.findall(source))
    return {
        "tracked_files": len(files),
        "tracked_directories": len(directories),
        "primary_lines": primary_lines,
        "markdown_files": markdown_files,
        "markdown_lines": markdown_lines,
        "condense_python_files": condense_python_files,
        "condense_python_lines": condense_python_lines,
        "rust_test_attributes": rust_test_attributes,
        "rust_ignore_attributes": rust_ignore_attributes,
    }


def frozen_status(contract: dict[str, object]) -> dict[str, object]:
    baseline = str(contract["baseline_commit"])
    active_root = Path(str(contract["active_worktree"]))
    external = contract.get("external_artifacts", {})
    rows: list[dict[str, object]] = []
    for relative in contract["live_bound_paths"]:  # type: ignore[index]
        path = ROOT / str(relative)
        baseline_result = subprocess.run(
            ["git", "show", f"{baseline}:{relative}"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if baseline_result.returncode != 0:
            binding = external.get(relative) if isinstance(external, dict) else None
            active = active_root / str(relative)
            if isinstance(binding, dict) and active.is_file():
                raw = active.read_bytes()
                expected = binding.get("sha256")
                actual = hashlib.sha256(raw).hexdigest()
                expected_bytes = binding.get("bytes")
                rows.append(
                    {
                        "path": relative,
                        "tracked_at_baseline": False,
                        "external_artifact": True,
                        "expected_sha256": expected,
                        "actual_sha256": actual,
                        "expected_bytes": expected_bytes,
                        "actual_bytes": len(raw),
                        "unchanged": actual == expected and len(raw) == expected_bytes,
                    }
                )
            else:
                rows.append(
                    {
                        "path": relative,
                        "tracked_at_baseline": False,
                        "external_artifact": bool(binding),
                        "unchanged": False,
                    }
                )
            continue
        current = path.read_bytes() if path.is_file() else None
        expected = hashlib.sha256(baseline_result.stdout).hexdigest()
        actual = hashlib.sha256(current).hexdigest() if current is not None else None
        rows.append(
            {
                "path": relative,
                "tracked_at_baseline": True,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "unchanged": actual == expected,
            }
        )
    changed = [
        row["path"]
        for row in rows
        if not row.get("unchanged")
    ]
    return {
        "all_live_bindings_unchanged": not changed,
        "all_tracked_sources_unchanged": not changed,
        "changed": changed,
        "paths": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-frozen", action="store_true")
    args = parser.parse_args(argv)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    current = measure()
    target = contract["extreme_target"]
    comparisons = {
        key.removesuffix("_max"): {
            "current": current[key.removesuffix("_max")],
            "maximum": maximum,
            "remaining_reduction": max(
                0, current[key.removesuffix("_max")] - maximum
            ),
            "met": current[key.removesuffix("_max")] <= maximum,
        }
        for key, maximum in target.items()
    }
    test_surface = {
        key: {
            "current": current[key],
            "expected": expected,
            "met": current[key] == expected,
        }
        for key, expected in contract["test_surface"].items()
    }
    result = {
        "baseline_commit": contract["baseline_commit"],
        "recommended_target": contract["recommended_target"],
        "current": current,
        "extreme_target": target,
        "comparisons": comparisons,
        "target_met": all(row["met"] for row in comparisons.values()),
        "test_surface": test_surface,
        "test_surface_preserved": all(row["met"] for row in test_surface.values()),
        "live_bound": frozen_status(contract),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    failed_shape = args.check and (
        not result["target_met"] or not result["test_surface_preserved"]
    )
    failed_frozen = (
        args.check_frozen
        and not result["live_bound"]["all_live_bindings_unchanged"]
    )
    return 1 if failed_shape or failed_frozen else 0


if __name__ == "__main__":
    sys.exit(main())
