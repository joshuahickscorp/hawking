#!/usr/bin/env python3
"""Create a sealed CPU-only binding for the Q30 range-bootstrap binary.

The binding consumes an already sealed metadata-only bootstrap preflight and
hashes only the explicitly supplied compiled executable plus its Rust source.
It does not invoke either file, accept a source root, issue a lease, or start
any Q30/Q80 process.  Its schema/status are consumed unchanged by the existing
receipt-last bootstrap outer preflight.
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
    ascension_qwen30_streamed_source_range_admission_bootstrap_outer_preflight as outer,
)
from lab.receipts import seal

SCHEMA = outer.BINARY_SCHEMA
STATUS = outer.BINARY_STATUS
MAX_BINDING_FILE_BYTES = 2 * 1024**3


class BootstrapBinaryBindingError(RuntimeError):
    """A build artifact or its metadata-only preflight is not bindable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise BootstrapBinaryBindingError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BootstrapBinaryBindingError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BootstrapBinaryBindingError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_BINDING_FILE_BYTES:
        raise BootstrapBinaryBindingError(f"{label} has invalid binding size")
    return path.resolve(strict=True)


def _preflight_pointer(document: outer.Document) -> dict[str, str]:
    return {
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def build_binary_binding(
    *, preflight_path: Path, binary_path: Path, source_path: Path
) -> dict[str, Any]:
    """Return a sealed non-executing binary binding for outer preflight input."""
    try:
        preflight = outer._sealed(preflight_path, label="bootstrap preflight")
        outer._validate_preflight(preflight)
    except outer.BootstrapOuterError as exc:
        raise BootstrapBinaryBindingError(f"bootstrap preflight is invalid: {exc}") from exc
    binary = _regular_file(binary_path, label="compiled bootstrap binary")
    source = _regular_file(source_path, label="bootstrap Rust source")
    binary_sha256 = _sha256_file(binary)
    source_sha256 = _sha256_file(source)
    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "recorded_at": _utc_now(),
            "cpu_only": True,
            "scan_or_runtime_executed": False,
            "binary_sha256": binary_sha256,
            "source_sha256": source_sha256,
            "bootstrap_preflight": _preflight_pointer(preflight),
            "binary_artifact": {
                "path": str(binary),
                "bytes": binary.stat().st_size,
                "sha256": binary_sha256,
                "executed_by_this_binder": False,
            },
            "source_artifact": {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": source_sha256,
                "source_payload_or_model_data": False,
            },
            "execution_boundary": {
                "bootstrap_binary_invoked_by_this_binder": False,
                "source_root_argument_or_stat_performed": False,
                "source_payload_opened": False,
                "source_model_loaded": False,
                "capture_child_spawned": False,
                "gpu_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed_or_released": False,
            },
            "claim_boundary": "CPU-only build binding. It does not authorize or execute a bootstrap scan, source teacher, source model, GPU, server, HCLI, TPS, TG, or tournament action.",
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise BootstrapBinaryBindingError("--out must be a new absolute path below an existing parent")
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
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = build_binary_binding(
            preflight_path=args.bootstrap_preflight,
            binary_path=args.binary,
            source_path=args.source,
        )
        _write_new(args.out, document)
    except BootstrapBinaryBindingError as exc:
        print(f"Q30 range-bootstrap binary binding refused: {exc}")
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
