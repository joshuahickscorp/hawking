#!/usr/bin/env python3
"""Reject a Rust test source that is absent from the fast compile topology.

The fast lane deliberately keeps its aggregate and legacy targets out of Cargo's
default test selection.  This checker makes that optimization fail closed: every
top-level integration source must be represented once in an aggregate checker and
once as a separately selectable legacy target with ``test = false``.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any


CORE_TQ_CASES = frozenset(
    {
        "qwen_tq_serve_parity.rs",
        "rwkv7_tq_bench.rs",
        "rwkv7_tq_loader.rs",
        "rwkv7_tq_parity.rs",
        "tq_llama_probe.rs",
        "tq_output_space_quality.rs",
        "tq_trellis_parity.rs",
    }
)

# Keys are package directories relative to the workspace root.  The runner
# filenames are intentionally conventional so a reviewer can identify the
# source-check target from the package manifest alone.
RUNNERS: dict[Path, tuple[str, ...]] = {
    Path("crates/hawking-core"): ("compile_default.rs", "compile_tq.rs"),
    Path("crates/hawking-serve"): ("compile_all.rs",),
    Path("crates/hawking-adapters"): ("compile_all.rs",),
    Path("crates/hawking-events"): ("compile_all.rs",),
    Path("crates/hide-backend"): ("compile_all.rs",),
    Path("crates/hide-core"): ("compile_all.rs",),
    Path("crates/hide-fleet"): ("compile_all.rs",),
    Path("crates/hide-kernel"): ("compile_all.rs",),
    Path("crates/hide-protocol"): ("compile_all.rs",),
    Path("tools/tq_bake"): ("compile_all.rs",),
}

PATH_MODULE = re.compile(
    r'^\s*#\[path\s*=\s*"([^"\\]+)"\]\s*\n\s*mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;',
    re.MULTILINE,
)
TQ_CFG = re.compile(r'#!\[cfg\([^\n]*feature\s*=\s*"tq"')


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def error(errors: list[str], message: str) -> None:
    errors.append(message)


def direct_sources(package: Path, runner_names: tuple[str, ...]) -> list[Path]:
    tests = package / "tests"
    return sorted(
        path
        for path in tests.glob("*.rs")
        if path.name not in runner_names and path.name != "mod.rs"
    )


def runner_modules(runner: Path, tests_dir: Path, errors: list[str]) -> list[Path]:
    text = runner.read_text(encoding="utf-8")
    modules: list[Path] = []
    for raw_path, module in PATH_MODULE.findall(text):
        path = (runner.parent / raw_path).resolve()
        if path.parent != tests_dir.resolve() or path.suffix != ".rs":
            error(
                errors,
                f"{runner}: module {module!r} must point at a direct tests/*.rs source, got {raw_path!r}",
            )
            continue
        if path.stem != module:
            error(
                errors,
                f"{runner}: module name {module!r} does not match source stem {path.stem!r}",
            )
        modules.append(path)
    return modules


def target_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("test", [])
    return records if isinstance(records, list) else []


def require_target(
    records: list[dict[str, Any]],
    *,
    name: str,
    path: str,
    role: str,
    errors: list[str],
    require_tq: bool = False,
) -> None:
    matches = [record for record in records if record.get("name") == name]
    if len(matches) != 1:
        error(errors, f"{role}: expected one [[test]] named {name!r}, found {len(matches)}")
        return
    record = matches[0]
    if record.get("path") != path:
        error(errors, f"{role}: {name!r} path is {record.get('path')!r}, expected {path!r}")
    if record.get("test") is not False:
        error(errors, f"{role}: {name!r} must set test = false for explicit selection")
    if role.startswith("aggregate") and record.get("harness") is False:
        error(errors, f"{role}: {name!r} must keep the normal test harness so #[test] bodies typecheck")
    if require_tq and record.get("required-features") != ["tq"]:
        error(errors, f"{role}: {name!r} must require exactly [\"tq\"]")


def check_package(root: Path, package_rel: Path, runner_names: tuple[str, ...], runner_only: bool) -> list[str]:
    errors: list[str] = []
    package = root / package_rel
    tests = package / "tests"
    manifest_path = package / "Cargo.toml"
    if not tests.is_dir() or not manifest_path.is_file():
        return [f"{package_rel}: expected tests/ and Cargo.toml"]

    sources = direct_sources(package, runner_names)
    source_set = {source.resolve() for source in sources}
    aggregate_paths: list[Path] = []
    for runner_name in runner_names:
        runner = tests / runner_name
        if not runner.is_file():
            error(errors, f"{package_rel}: missing aggregate runner tests/{runner_name}")
            continue
        aggregate_paths.extend(runner_modules(runner, tests, errors))

    aggregate_count = Counter(aggregate_paths)
    for source in sorted(source_set):
        count = aggregate_count[source]
        if count != 1:
            error(errors, f"{package_rel}: {source.name} appears {count} times across aggregate runners")
    for source, count in sorted(aggregate_count.items(), key=lambda item: str(item[0])):
        if source not in source_set:
            error(errors, f"{package_rel}: aggregate source {source} is not a direct integration source")
        elif count != 1:
            error(errors, f"{package_rel}: aggregate source {source.name} appears {count} times")

    core = package_rel == Path("crates/hawking-core")
    if core:
        detected_tq = {
            source.name for source in sources if TQ_CFG.search(source.read_text(encoding="utf-8"))
        }
        if detected_tq != CORE_TQ_CASES:
            error(
                errors,
                "crates/hawking-core: TQ source set changed; update CORE_TQ_CASES and the two aggregate runners",
            )
        default_runner = tests / "compile_default.rs"
        tq_runner = tests / "compile_tq.rs"
        default_modules = set(runner_modules(default_runner, tests, errors)) if default_runner.is_file() else set()
        tq_modules = set(runner_modules(tq_runner, tests, errors)) if tq_runner.is_file() else set()
        expected_tq = {tests.joinpath(name).resolve() for name in CORE_TQ_CASES}
        if default_modules & expected_tq:
            error(errors, "crates/hawking-core: compile_default.rs must not include TQ-gated sources")
        if tq_modules != expected_tq:
            error(errors, "crates/hawking-core: compile_tq.rs must include exactly the TQ-gated sources")

    if runner_only:
        return errors

    manifest = load_toml(manifest_path)
    if manifest.get("package", {}).get("autotests") is not False:
        error(errors, f"{package_rel}/Cargo.toml: package.autotests must be false")
    records = target_records(manifest)
    expected_target_names = {source.stem for source in sources}
    expected_target_names.update(Path(runner).stem for runner in runner_names)
    for record in records:
        path = record.get("path")
        if isinstance(path, str) and path.startswith("tests/") and record.get("name") not in expected_target_names:
            error(errors, f"{package_rel}/Cargo.toml: unexpected tests target {record.get('name')!r} at {path!r}")

    for source in sources:
        require_target(
            records,
            name=source.stem,
            path=f"tests/{source.name}",
            role=f"legacy {package_rel}",
            errors=errors,
        )
    for runner_name in runner_names:
        stem = Path(runner_name).stem
        require_target(
            records,
            name=stem,
            path=f"tests/{runner_name}",
            role=f"aggregate {package_rel}",
            errors=errors,
            require_tq=core and stem == "compile_tq",
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="workspace root (defaults to this script's repository)",
    )
    parser.add_argument(
        "--runner-only",
        action="store_true",
        help="check source-to-runner coverage without requiring manifest entries",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    source_count = 0
    for package_rel, runners in RUNNERS.items():
        source_count += len(direct_sources(root / package_rel, runners))
        errors.extend(check_package(root, package_rel, runners, args.runner_only))
    if errors:
        for message in errors:
            print(f"error: {message}", file=sys.stderr)
        return 1
    print(
        f"rust test topology valid: {source_count} direct integration sources across {len(RUNNERS)} packages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
