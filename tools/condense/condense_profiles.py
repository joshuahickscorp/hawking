#!/usr/bin/env python3.12
"""Git-backed profiles and compatibility dispatch for retired condense tools."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType
from typing import Any

import condense_common as common


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE_COMMIT = "1c525380204d61beb8b570516576c5a683d73595"
SCHEMA = "hawking.condense_profiles.v1"
SCHEMA_RE = re.compile(rb"hawking\.[A-Za-z0-9_.-]+\.v[0-9]+")
PATH_RE = re.compile(rb"(?:reports|scratch|receipts)/[A-Za-z0-9_./-]+")
SUBCOMMAND_RE = re.compile(rb"add_parser\([\"']([^\"']+)")

RETAINED = frozenset({
    "__main__.py",
    "adapter_contract.py",
    "appendix_physical_counter_authority.py",
    "appendix_physical_evidence_gate.py",
    "appendix_runtime.py",
    "audit_ladder.py",
    "condense_common.py",
    "condense_profiles.py",
    "doctor.py",
    "frontier_ops.py",
    "frontier_runtime.py",
    "ladder.py",
    "multi_eval.py",
    "physical_counter_attestation.py",
    "preflight.py",
    "quality_battery_v5.py",
    "ram_scheduler.py",
    "studio_environment.py",
    "studio_manifest.py",
    "sweep.py",
    "training_ladder_v5.py",
    "tripwire_gate.py",
})


def _archive_paths() -> dict[str, str]:
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", ARCHIVE_COMMIT, "--", "tools/condense"],
        cwd=ROOT,
        text=True,
    )
    return {
        Path(path).stem: path
        for path in output.splitlines()
        if Path(path).parent.as_posix() == "tools/condense"
        and path.endswith(".py")
        and not Path(path).name.startswith("doctor_v5_")
    }


ARCHIVE_PATHS = _archive_paths()


def normalize(name: str) -> str:
    return Path(name).name.removesuffix(".py").replace("-", "_")


def family(name: str) -> str:
    value = normalize(name)
    if value.startswith(("appendix_", "tq_", "spec_", "physical_counter_")):
        return "appendix"
    if value.startswith(("frontier_", "doctor_frontier", "terminal_frontier")):
        return "frontier"
    if value.startswith(("doctor", "healer")):
        return "doctor_legacy"
    if value.startswith(("studio", "processing", "download", "procure")):
        return "studio"
    if value.startswith(("quality", "eval", "ppl", "score", "bench")):
        return "quality"
    return "core"


def archive_source(name: str) -> bytes:
    module = normalize(name)
    relative = ARCHIVE_PATHS.get(module)
    if relative is None:
        raise KeyError(f"unknown archived condense module: {name}")
    return subprocess.check_output(
        ["git", "show", f"{ARCHIVE_COMMIT}:{relative}"], cwd=ROOT,
    )


def legacy_record(name: str, *, include_source: bool = False) -> dict[str, Any]:
    module = normalize(name)
    raw = archive_source(module)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "module": module,
        "path": ARCHIVE_PATHS[module],
        "family": family(module),
        "archive_commit": ARCHIVE_COMMIT,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "schemas": sorted(row.decode() for row in set(SCHEMA_RE.findall(raw))),
        "artifact_paths": sorted(row.decode().rstrip("./") for row in set(PATH_RE.findall(raw))),
        "subcommands": sorted(row.decode() for row in set(SUBCOMMAND_RE.findall(raw))),
        "replacement": {
            "appendix": "appendix.runtime",
            "frontier": "frontier.runtime",
        }.get(family(module), "core.profile"),
    }
    if include_source:
        record["source"] = raw.decode("utf-8")
    return record


class _ArchiveLoader(importlib.abc.Loader):
    def __init__(self, module: str) -> None:
        self.module = module

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        relative = ARCHIVE_PATHS[self.module]
        filename = str(ROOT / relative)
        module.__file__ = filename
        module.__package__ = ""
        exec(compile(archive_source(self.module), filename, "exec"), module.__dict__)


class _ArchiveFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if "." in fullname or fullname not in ARCHIVE_PATHS:
            return None
        if (HERE / f"{fullname}.py").is_file():
            return None
        return importlib.util.spec_from_loader(fullname, _ArchiveLoader(fullname))


_FINDER = _ArchiveFinder()


def install_archive_importer() -> None:
    if not any(isinstance(item, _ArchiveFinder) for item in sys.meta_path):
        sys.meta_path.insert(0, _FINDER)


def archived_module(name: str) -> ModuleType:
    module = normalize(name)
    if module not in ARCHIVE_PATHS:
        raise KeyError(f"unknown archived condense module: {name}")
    install_archive_importer()
    return importlib.import_module(module)


def invoke(name: str, argv: list[str]) -> int:
    module_name = normalize(name)
    module = archived_module(module_name)
    entry = getattr(module, "main", None)
    prior = sys.argv
    sys.argv = [str(ROOT / ARCHIVE_PATHS[module_name]), *argv]
    try:
        if callable(entry):
            parameters = inspect.signature(entry).parameters
            result = entry() if not parameters else entry(argv)
            return int(result or 0)
        namespace = {
            "__name__": "__main__",
            "__file__": str(ROOT / ARCHIVE_PATHS[module_name]),
            "__package__": "",
        }
        try:
            exec(
                compile(
                    archive_source(module_name),
                    namespace["__file__"],
                    "exec",
                ),
                namespace,
            )
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0
    finally:
        sys.argv = prior


def profile_document() -> dict[str, Any]:
    rows: dict[str, list[str]] = {}
    current = {path.name for path in HERE.glob("*.py")}
    for module in sorted(ARCHIVE_PATHS):
        filename = f"{module}.py"
        if filename not in current:
            rows.setdefault(family(module), []).append(module)
    document = {
        "schema": SCHEMA,
        "archive_commit": ARCHIVE_COMMIT,
        "retained": sorted(RETAINED),
        "retired_by_family": rows,
        "retired_count": sum(map(len, rows.values())),
    }
    document["profile_sha256"] = common.canonical_sha256(document)
    return document


def validate_layout() -> list[str]:
    current = {
        path.name for path in HERE.glob("*.py")
        if not (
            path.name.startswith("doctor_v5_")
            or path.name.startswith("test_doctor_v5")
        )
    }
    errors = []
    if current != set(RETAINED):
        errors.append(
            f"non-Doctor layout differs: missing={sorted(RETAINED - current)} "
            f"extra={sorted(current - RETAINED)}"
        )
    if not ARCHIVE_PATHS:
        errors.append("Git archive inventory is empty")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("module")
    show.add_argument("--source", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("module")
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    sub.add_parser("validate")
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(profile_document(), indent=2, sort_keys=True))
        return 0
    if args.command == "show":
        print(json.dumps(
            legacy_record(args.module, include_source=args.source),
            indent=2, sort_keys=True,
        ))
        return 0
    if args.command == "run":
        return invoke(args.module, args.arguments)
    if args.command == "validate":
        errors = validate_layout()
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1
    assert archive_source("appendix_catalog").startswith(b"#!/")
    assert legacy_record("frontier_claims")["family"] == "frontier"
    if args.command == "selftest":
        assert not validate_layout()
        print("condense_profiles.py selftest OK")
    return 0


install_archive_importer()


if __name__ == "__main__":
    raise SystemExit(main())
