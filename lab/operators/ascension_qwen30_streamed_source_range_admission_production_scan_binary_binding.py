#!/usr/bin/env python3
"""Create a sealed CPU-only binding for the Q30 production hash-scan child.

This is a file-integrity binder, not a launcher.  It reads the existing Q30
bootstrap preflight/binary/resource records only as ancestry, hashes a supplied
compiled production executable and Rust source, and writes a create-new sealed
record.  It has no source-root, child, model, GPU, server, HCLI, or lease
surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_bootstrap_outer_preflight as bootstrap,
)
from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_outer_reaper as outer,
)
from lab.receipts import seal

SCHEMA = outer.PRODUCTION_BINARY_SCHEMA
STATUS = outer.PRODUCTION_BINARY_STATUS
INTERFACE_SCHEMA = outer.INTERFACE_SCHEMA
INTERFACE_STATUS = outer.INTERFACE_STATUS
MAX_BINDING_FILE_BYTES = 2 * 1024**3


class ProductionBinaryBindingError(RuntimeError):
    """The production child or its exact Q30 ancestry is not bindable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str, executable: bool) -> Path:
    if not path.is_absolute():
        raise ProductionBinaryBindingError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionBinaryBindingError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProductionBinaryBindingError(f"{label} must be a regular non-symlink file")
    if executable and not metadata.st_mode & stat.S_IXUSR:
        raise ProductionBinaryBindingError(f"{label} must be executable by its owner")
    if metadata.st_size <= 0 or metadata.st_size > MAX_BINDING_FILE_BYTES:
        raise ProductionBinaryBindingError(f"{label} has invalid size")
    return path.resolve(strict=True)


def _evidence(document: bootstrap.Document) -> dict[str, str]:
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _command_hash(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def build_binary_binding(
    *,
    bootstrap_preflight_path: Path,
    bootstrap_binary_path: Path,
    bootstrap_resource_path: Path,
    executable_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    """Return a sealed non-executing binary binding for production outer input."""
    try:
        preflight = bootstrap._sealed(bootstrap_preflight_path, label="bootstrap preflight")
        binary = bootstrap._sealed(bootstrap_binary_path, label="bootstrap binary")
        resource = bootstrap._sealed(bootstrap_resource_path, label="bootstrap resource")
        bootstrap._validate_preflight(preflight)
        binary_sha = bootstrap._validate_binary(binary, preflight=preflight)
        bootstrap._validate_resource(resource, preflight=preflight, binary_sha=binary_sha)
    except bootstrap.BootstrapOuterError as exc:
        raise ProductionBinaryBindingError(f"bootstrap ancestry is invalid: {exc}") from exc
    executable = _regular_file(executable_path, label="production executable", executable=True)
    source = _regular_file(source_path, label="production Rust source", executable=False)
    executable_sha = _sha256_file(executable)
    source_sha = _sha256_file(source)
    command = [
        "cargo",
        "build",
        "-p",
        "hawking-core",
        "--example",
        "ascension_qwen30_streamed_source_range_admission_production_scan_interface",
    ]
    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "recorded_at": _utc_now(),
            "cpu_only": True,
            "production_hash_scan_backend_compiled": True,
            "production_hash_scan_executed": False,
            "source_root_opened_or_statted": False,
            "source_payload_opened": False,
            "source_teacher_or_logits_executed": False,
            "model_gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
            "binary_sha256": executable_sha,
            "source_sha256": source_sha,
            "executable": {
                "path": str(executable),
                "bytes": executable.stat().st_size,
                "sha256": executable_sha,
            },
            "source": {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": source_sha,
            },
            "compiled_command": command,
            "compiled_command_sha256": _command_hash(command),
            "production_interface_expected": {
                "schema": INTERFACE_SCHEMA,
                "status": INTERFACE_STATUS,
                "future_execution_not_authorized_by_binding": True,
            },
            "bootstrap_preflight": _evidence(preflight),
            "bootstrap_binary": _evidence(binary),
            "bootstrap_resource": _evidence(resource),
            "legacy_bootstrap_resource_is_ancestry_only": True,
            "execution_boundary": {
                "binary_invoked_by_this_binder": False,
                "source_root_argument_or_stat_performed": False,
                "source_payload_opened": False,
                "source_model_loaded": False,
                "capture_child_spawned": False,
                "gpu_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed_or_released": False,
            },
            "claim_boundary": "CPU-only production-child binding. It does not authorize or execute a source hash scan, source teacher, source model, GPU, server, HCLI, lease, TPS, TG, or tournament action.",
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ProductionBinaryBindingError("--out must be a new absolute path below an existing parent")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-preflight", type=Path, required=True)
    parser.add_argument("--bootstrap-binary", type=Path, required=True)
    parser.add_argument("--bootstrap-resource", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = build_binary_binding(
            bootstrap_preflight_path=args.bootstrap_preflight,
            bootstrap_binary_path=args.bootstrap_binary,
            bootstrap_resource_path=args.bootstrap_resource,
            executable_path=args.executable,
            source_path=args.source,
        )
        _write_new(args.out, document)
    except ProductionBinaryBindingError as exc:
        print(f"Q30 production hash-scan binary binding refused: {exc}")
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": document["status"],
                "seal_sha256": document["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
