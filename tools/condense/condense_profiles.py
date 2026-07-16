#!/usr/bin/env python3.12
"""Static profiles and sealed-pack dispatch for retired condense tools."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any

import compat_catalog
import condense_common as common


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE_COMMIT = "1c525380204d61beb8b570516576c5a683d73595"
SCHEMA = "hawking.condense_profiles.v1"
PACK_ID = "hawking-runtime-validation-r225"
PACK_SOURCE = Path("support/compat/condense")
MAX_SOURCE_BYTES = 64 * 1024 * 1024
CONDENSE_METADATA = compat_catalog.CONDENSE_METADATA
EXECUTABLE_MODULES = compat_catalog.EXECUTABLE_MODULES
SOURCE_CLOSURE_EXCLUSIONS = EXECUTABLE_MODULES | {"compat_catalog"}
ARCHIVE_PATHS = {
    module: f"tools/condense/{module}.py" for module in CONDENSE_METADATA
}
MODULE_MARKER_ATTRIBUTE = "__hawking_compat_loader_marker__"
MODULE_NAME_ATTRIBUTE = "__hawking_compat_module__"
MODULE_SHA_ATTRIBUTE = "__hawking_compat_source_sha256__"
MODULE_ORIGIN_ATTRIBUTE = "__hawking_compat_archive_origin__"
MODULE_PATH_ATTRIBUTE = "__hawking_compat_materialized_path__"
MODULE_MODE_ATTRIBUTE = "__hawking_compat_materialized_mode__"
MODULE_STATE_ATTRIBUTE = "__hawking_compat_state__"
_LOADER_MARKER = object()

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
PROFILE_PRESENT = RETAINED | {"test_doctor_v5_strand_ladder_runtime.py"}


class CompatibilitySourceError(RuntimeError):
    """A retired source body is unavailable or fails its immutable binding."""


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


def _record(module: str) -> dict[str, Any]:
    digest, schemas, artifact_paths, subcommands = CONDENSE_METADATA[module]
    module_family = family(module)
    return {
        "schema": SCHEMA,
        "module": module,
        "path": ARCHIVE_PATHS[module],
        "family": module_family,
        "archive_commit": ARCHIVE_COMMIT,
        "source_sha256": digest,
        "schemas": list(schemas),
        "artifact_paths": list(artifact_paths),
        "subcommands": list(subcommands),
        "replacement": {
            "appendix": "appendix.runtime",
            "frontier": "frontier.runtime",
        }.get(module_family, "core.profile"),
    }


def legacy_record(name: str, *, include_source: bool = False) -> dict[str, Any]:
    module = normalize(name)
    if module not in CONDENSE_METADATA:
        raise KeyError(f"unknown archived condense module: {name}")
    record = _record(module)
    if include_source and module in EXECUTABLE_MODULES:
        record["source"] = archive_source(module).decode("utf-8")
    elif include_source:
        record.update({
            "source_available": False,
            "source_status": "superseded-unavailable",
        })
    return record


def _pack_root() -> Path:
    try:
        lock = common.read_json(ROOT / "packs.lock.json")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompatibilitySourceError(f"{PACK_ID} lock is unavailable: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("hydrate_root") != ".hawking/packs":
        raise CompatibilitySourceError("pack lock hydrate_root must be .hawking/packs")
    packs = lock.get("packs")
    if not isinstance(packs, list):
        raise CompatibilitySourceError("pack lock packs must be a list")
    rows = [
        row for row in packs
        if isinstance(row, dict) and row.get("id") == PACK_ID
    ]
    if len(rows) != 1:
        raise CompatibilitySourceError(
            f"pack lock must contain exactly one {PACK_ID} row"
        )
    hydrate_name = rows[0].get("hydrate_name")
    if not isinstance(hydrate_name, str) or not hydrate_name \
            or hydrate_name in {".", ".."} or "/" in hydrate_name \
            or "\\" in hydrate_name:
        raise CompatibilitySourceError(f"{PACK_ID} hydrate_name is unsafe")
    root = ROOT / ".hawking/packs" / hydrate_name / PACK_SOURCE
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise CompatibilitySourceError(f"{PACK_ID} is not hydrated: {exc}") from exc
    if resolved != root or not resolved.is_dir():
        raise CompatibilitySourceError(f"{PACK_ID} compatibility root is unsafe")
    return resolved


def _read_bound_source(root: Path, path: Path) -> tuple[bytes, int]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CompatibilitySourceError(f"compatibility source is missing: {path}") from exc
    if resolved != path or resolved.parent != root:
        raise CompatibilitySourceError(f"compatibility source path is unsafe: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise CompatibilitySourceError(
            f"compatibility source cannot be opened: {path}: {exc}"
        ) from exc
    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                    or before.st_size > MAX_SOURCE_BYTES:
                raise CompatibilitySourceError(f"compatibility source is invalid: {path}")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise CompatibilitySourceError(
                        f"compatibility source short read: {path}"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise CompatibilitySourceError(
                    f"compatibility source grew during read: {path}"
                )
            after = os.fstat(descriptor)
        except OSError as exc:
            raise CompatibilitySourceError(
                f"compatibility source read failed: {path}: {exc}"
            ) from exc
    finally:
        os.close(descriptor)
    identity = lambda row: (
        row.st_dev, row.st_ino, row.st_mode, row.st_nlink, row.st_size,
        row.st_mtime_ns, row.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise CompatibilitySourceError(f"compatibility source changed during read: {path}")
    return b"".join(chunks), stat.S_IMODE(after.st_mode)


def _source_binding(module: str) -> tuple[bytes, Path, int]:
    root = _pack_root()
    raw, source_mode = _read_bound_source(root, root / f"{module}.py")
    expected = CONDENSE_METADATA[module][0]
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise CompatibilitySourceError(
            f"{module} source SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    materialized = ROOT / ARCHIVE_PATHS[module]
    copy, copy_mode = _read_bound_source(materialized.parent, materialized)
    if copy != raw:
        raise CompatibilitySourceError(
            f"{module} materialized source bytes differ from the authoritative pack"
        )
    if copy_mode != source_mode:
        raise CompatibilitySourceError(
            f"{module} materialized mode {copy_mode:#o} differs from "
            f"the authoritative pack mode {source_mode:#o}"
        )
    return raw, materialized, source_mode


def archive_source(name: str) -> bytes:
    module = normalize(name)
    if module not in CONDENSE_METADATA:
        raise KeyError(f"unknown archived condense module: {name}")
    if module not in EXECUTABLE_MODULES:
        raise CompatibilitySourceError(
            f"{module} is superseded; its raw source is intentionally unavailable"
        )
    return _source_binding(module)[0]


def _archive_identity(module: str) -> str:
    return f"git:{ARCHIVE_COMMIT}:{ARCHIVE_PATHS[module]}"


class _PackLoader(importlib.abc.Loader):
    def __init__(self, module: str) -> None:
        self.module = module
        self.origin = _archive_identity(module)
        self.source_sha256 = CONDENSE_METADATA[module][0]
        self.materialized_path: str | None = None
        self.materialized_mode: int | None = None
        self._marker = _LOADER_MARKER

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        if spec.name != self.module or spec.origin != self.origin:
            raise CompatibilitySourceError(
                f"{self.module} loader specification identity differs"
            )
        module = ModuleType(spec.name)
        setattr(module, MODULE_MARKER_ATTRIBUTE, _LOADER_MARKER)
        setattr(module, MODULE_NAME_ATTRIBUTE, self.module)
        setattr(module, MODULE_SHA_ATTRIBUTE, self.source_sha256)
        setattr(module, MODULE_ORIGIN_ATTRIBUTE, self.origin)
        setattr(module, MODULE_STATE_ATTRIBUTE, "created")
        return module

    def exec_module(self, module: ModuleType) -> None:
        if not _trusted_loaded_module(self.module, module):
            raise CompatibilitySourceError(
                f"{self.module} module identity was poisoned before execution"
            )
        raw, materialized, mode = _source_binding(self.module)
        filename = str(materialized)
        self.materialized_path = filename
        self.materialized_mode = mode
        setattr(module, MODULE_PATH_ATTRIBUTE, filename)
        setattr(module, MODULE_MODE_ATTRIBUTE, mode)
        setattr(module, MODULE_STATE_ATTRIBUTE, "loading")
        module.__file__ = filename
        module.__package__ = ""
        module.__archive_identity__ = self.origin
        exec(compile(raw, filename, "exec"), module.__dict__)
        setattr(module, MODULE_MARKER_ATTRIBUTE, _LOADER_MARKER)
        setattr(module, MODULE_NAME_ATTRIBUTE, self.module)
        setattr(module, MODULE_SHA_ATTRIBUTE, self.source_sha256)
        setattr(module, MODULE_ORIGIN_ATTRIBUTE, self.origin)
        setattr(module, MODULE_PATH_ATTRIBUTE, filename)
        setattr(module, MODULE_MODE_ATTRIBUTE, mode)
        setattr(module, MODULE_STATE_ATTRIBUTE, "verified")
        module.__file__ = filename
        module.__archive_identity__ = self.origin
        if not _trusted_loaded_module(self.module, module):
            raise CompatibilitySourceError(
                f"{self.module} changed its sealed loader identity during execution"
            )


def _trusted_loaded_module(name: str, module: Any) -> bool:
    loader = getattr(module, "__loader__", None)
    spec = getattr(module, "__spec__", None)
    expected_origin = _archive_identity(name)
    expected_sha = CONDENSE_METADATA[name][0]
    if type(module) is not ModuleType or type(loader) is not _PackLoader \
            or getattr(loader, "_marker", None) is not _LOADER_MARKER \
            or loader.module != name or loader.origin != expected_origin \
            or loader.source_sha256 != expected_sha \
            or spec is None or spec.loader is not loader \
            or spec.name != name or spec.origin != expected_origin \
            or getattr(module, "__name__", None) != name \
            or getattr(module, MODULE_MARKER_ATTRIBUTE, None) is not _LOADER_MARKER \
            or getattr(module, MODULE_NAME_ATTRIBUTE, None) != name \
            or getattr(module, MODULE_SHA_ATTRIBUTE, None) != expected_sha \
            or getattr(module, MODULE_ORIGIN_ATTRIBUTE, None) != expected_origin:
        return False
    state = getattr(module, MODULE_STATE_ATTRIBUTE, None)
    if state in {"created", "loading"}:
        return bool(getattr(spec, "_initializing", False))
    return state == "verified" \
        and isinstance(loader.materialized_path, str) \
        and getattr(module, "__file__", None) == loader.materialized_path \
        and getattr(module, "__archive_identity__", None) == expected_origin \
        and getattr(module, MODULE_PATH_ATTRIBUTE, None) == loader.materialized_path \
        and loader.materialized_mode is not None \
        and getattr(module, MODULE_MODE_ATTRIBUTE, None) == loader.materialized_mode


def _reject_poisoned_modules() -> None:
    poisoned = [
        name for name in sorted(EXECUTABLE_MODULES)
        if name in sys.modules and not _trusted_loaded_module(name, sys.modules[name])
    ]
    if poisoned:
        raise CompatibilitySourceError(
            "untrusted preexisting compatibility module(s): " + ", ".join(poisoned)
        )


class _PackFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if "." in fullname or fullname not in EXECUTABLE_MODULES:
            return None
        _reject_poisoned_modules()
        return importlib.util.spec_from_loader(
            fullname, _PackLoader(fullname), origin=_archive_identity(fullname),
        )


_FINDER = _PackFinder()


def install_archive_importer() -> None:
    _reject_poisoned_modules()
    sys.meta_path[:] = [item for item in sys.meta_path if item is not _FINDER]
    sys.meta_path.insert(0, _FINDER)


def archived_module(name: str) -> ModuleType:
    module = normalize(name)
    if module not in CONDENSE_METADATA:
        raise KeyError(f"unknown archived condense module: {name}")
    if module not in EXECUTABLE_MODULES:
        raise CompatibilitySourceError(
            f"{module} is superseded and is not executable"
        )
    install_archive_importer()
    loaded = importlib.import_module(module)
    if not _trusted_loaded_module(module, loaded):
        raise CompatibilitySourceError(f"{module} import did not use the sealed pack loader")
    return loaded


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _superseded(module: str, argv: list[str]) -> int:
    record = legacy_record(module)
    record.update({
        "status": "superseded",
        "source_available": False,
        "compatibility_arguments": argv,
        "mutates_campaign": False,
    })
    _print(record)
    informational = not argv or argv[0] in {
        "status", "plan", "show", "list", "validate", "dry-run", "--dry-run",
        "selftest", "--selftest",
    }
    return 0 if informational else 64


def _pack_unavailable(module: str, argv: list[str], exc: Exception) -> int:
    record = legacy_record(module)
    record.update({
        "status": "compatibility-source-unavailable",
        "source_available": False,
        "compatibility_arguments": argv,
        "mutates_campaign": False,
        "error": str(exc),
    })
    _print(record)
    return 69


def invoke(name: str, argv: list[str]) -> int:
    module_name = normalize(name)
    if module_name not in CONDENSE_METADATA:
        raise KeyError(f"unknown archived condense module: {name}")
    if module_name not in EXECUTABLE_MODULES:
        return _superseded(module_name, argv)
    try:
        module = archived_module(module_name)
    except CompatibilitySourceError as exc:
        return _pack_unavailable(module_name, argv, exc)
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
    for module in sorted(ARCHIVE_PATHS):
        if f"{module}.py" not in PROFILE_PRESENT:
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
            path.name == "compat_catalog.py"
            or path.stem in EXECUTABLE_MODULES
            or path.name.startswith("doctor_v5_")
            or path.name.startswith("test_doctor_v5")
        )
    }
    errors = []
    if current != set(RETAINED):
        errors.append(
            f"non-Doctor layout differs: missing={sorted(RETAINED - current)} "
            f"extra={sorted(current - RETAINED)}"
        )
    if len(CONDENSE_METADATA) != 101:
        errors.append("static archive inventory must contain 101 modules")
    if len(EXECUTABLE_MODULES) != 41 or not EXECUTABLE_MODULES <= CONDENSE_METADATA.keys():
        errors.append("executable compatibility inventory must contain 41 known modules")
    records = {module: _record(module) for module in sorted(CONDENSE_METADATA)}
    if common.canonical_sha256(records) != compat_catalog.CONDENSE_RECORDS_SHA256:
        errors.append("static compatibility metadata differs from the golden capture")
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
        _print(profile_document())
        return 0
    if args.command == "show":
        try:
            _print(legacy_record(args.module, include_source=args.source))
        except CompatibilitySourceError as exc:
            return _pack_unavailable(normalize(args.module), [], exc)
        return 0
    if args.command == "run":
        return invoke(args.module, args.arguments)
    if args.command == "validate":
        errors = validate_layout()
        _print({"ok": not errors, "errors": errors})
        return 0 if not errors else 1
    assert legacy_record("appendix_catalog")["family"] == "appendix"
    assert not legacy_record("appendix_catalog", include_source=True)["source_available"]
    if args.command == "selftest":
        assert not validate_layout()
        print("condense_profiles.py selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
