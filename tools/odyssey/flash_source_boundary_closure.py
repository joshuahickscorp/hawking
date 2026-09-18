#!/usr/bin/env python3
"""Seal a fresh, non-executing Flash source-boundary executable closure.

The source-boundary transaction accepts only a receipt that rechecks its exact
release binaries, their declared source leaves, and the two Python wrappers it
will invoke.  This producer deliberately performs no model, Metal, daemon, or
resident action; it merely records an observed release candidate after a build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey import flash_repeated_accepted_decode as gate  # noqa: E402


SCHEMA = "hawking.flash.source_boundary_executable_closure.v1"
RECEIPT_ROOT = ROOT / "receipts" / "future"
NATIVE_SOURCE_PATHS = (
    Path("tools/odyssey/flash_repeated_accepted_decode.py"),
    Path("crates/hawking-core/src/lib.rs"),
    Path("crates/hawking-core/examples/flash_stateful_complete_token_session.rs"),
    Path("crates/hawking-core/examples/flash_full_attention_layer3.rs"),
    Path("crates/hawking-core/examples/flash_noetic_complete_layer0.rs"),
    # The stateful diagnostic routes the HC precision profile through these
    # owners.  Bind them explicitly so a closure cannot certify a binary while
    # its selected Metal kernel or Rust launch policy has silently drifted.
    Path("crates/hawking-core/src/kernels/mod.rs"),
    Path("crates/hawking-core/src/metal/mod.rs"),
    Path("crates/hawking-core/shaders/matmul.metal"),
    Path("crates/hawking-core/shaders/qwen_next.metal"),
    Path("crates/hawking-core/examples/flash_source_bf16_terminal.rs"),
    Path("crates/hawking-core/src/flash_ple.rs"),
    Path("crates/hawking-core/src/model/source_safetensors.rs"),
    Path("crates/hawking-core/src/model/qwen80_source_bf16_layer_major.rs"),
    Path("Cargo.lock"),
)
HAWKING_SOURCE_PATHS = (
    Path("crates/hawking-core/src/flash_boundary_compare.rs"),
    Path("crates/hawking/src/main.rs"),
    Path("Cargo.lock"),
)
NATIVE_WRAPPER = Path("tools/odyssey/flash_source_boundary_native.py")
COMPARISON_WRAPPER = Path("tools/odyssey/flash_source_boundary_compare.py")
# These Python leaves execute inside the one leased source/native transaction.
# They are bound separately from the two Rust binaries so a queued run cannot
# silently reinterpret a source boundary because a wrapper or source oracle
# drifted after the closure was sealed.
TRANSACTION_WRAPPER_PATHS = (
    Path("tools/odyssey/flash_source_boundary_transaction.py"),
    Path("tools/odyssey/flash_source_boundary_capture_mlx.py"),
    Path("tools/odyssey/flash_source_boundary_mlx_worker.py"),
    Path("tools/odyssey/flash_external_greedy_reference_mlx.py"),
    Path("tools/odyssey/flash_external_greedy_reference_mlx_worker.py"),
    NATIVE_WRAPPER,
    COMPARISON_WRAPPER,
    Path("tools/odyssey/flash_source_boundary_preflight.py"),
)


def _regular(path: Path, label: str) -> Path:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {requested}")
    resolved = requested.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay beneath {root.resolve()}: {resolved}") from exc
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _file_format(path: Path) -> str:
    completed = subprocess.run(["file", "-b", str(path)], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _record(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    resolved = _regular(path, label)
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} is not executable: {resolved}")
    record: dict[str, Any] = {
        "path": _display(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if executable:
        record["format"] = _file_format(resolved)
    return record


def _source_records(paths: tuple[Path, ...], label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        resolved = _inside(ROOT / path, ROOT, label)
        records.append(_record(resolved, label))
    return records


def _command_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _predecessor_binding(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    predecessor = _regular(path, "predecessor closure")
    document = gate._json(predecessor)
    seal = gate._verify_compact_seal(document, "predecessor closure")
    return {
        **_record(predecessor, "predecessor closure"),
        "seal_sha256": seal,
        "status": document.get("status"),
    }


def build_closure(
    *,
    native_binary: Path,
    hawking_binary: Path,
    out: Path,
    target_dir: Path,
    build_commands: list[str],
    predecessor: Path | None,
) -> dict[str, Any]:
    """Create a fresh, exclusive closure receipt for already-built binaries."""
    receipt_root = RECEIPT_ROOT.resolve()
    out = _inside(out.expanduser().resolve(), receipt_root, "closure output")
    if os.path.lexists(out):
        raise ValueError(f"closure output already exists; refusing to overwrite: {out}")
    target_root = (ROOT / "target").resolve()
    native_binary = _inside(_regular(native_binary, "native diagnostic executable"), target_root, "native diagnostic executable")
    hawking_binary = _inside(_regular(hawking_binary, "hawking metric executable"), target_root, "hawking metric executable")
    if not os.access(native_binary, os.X_OK) or not os.access(hawking_binary, os.X_OK):
        raise ValueError("source-boundary executable candidate is not executable")
    if not build_commands:
        raise ValueError("at least one exact build command is required for a closure receipt")

    native_record = _record(native_binary, "native diagnostic executable", executable=True)
    native_record.update({
        "source_closure": _source_records(NATIVE_SOURCE_PATHS, "native source closure"),
        "wrapper": _record(ROOT / NATIVE_WRAPPER, "native capture wrapper"),
        "preexecution_identity_rechecked_after_build": True,
        "executed": False,
    })
    hawking_record = _record(hawking_binary, "hawking metric executable", executable=True)
    hawking_record.update({
        "source": _source_records(HAWKING_SOURCE_PATHS, "hawking metric source closure"),
        "orchestrator": _record(ROOT / COMPARISON_WRAPPER, "ordered comparison wrapper"),
        "version": _command_text([str(hawking_binary), "--version"]),
    })

    git_status = _command_text(["git", "status", "--porcelain=v1"])
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "TASK029_RECONCILED_RELEASE_EXECUTABLES_OBSERVED__LOCAL_CANONICAL_ADMISSION",
        "task_id": "029",
        "admission_class": gate.LOCAL_CANONICAL_ADMISSION_STATUS,
        "science_status": "UNEARNED",
        "science_unearned": True,
        "observed_unix_ns": time.time_ns(),
        "git_head": _command_text(["git", "rev-parse", "HEAD"]),
        "worktree_dirty": bool(git_status),
        "build": {
            "commands": build_commands,
            "target_dir": _display(_inside(target_dir.expanduser().resolve(), target_root, "build target directory")),
            "rustc_version": _command_text(["rustc", "-Vv"]),
            "compiler_provenance_attested": False,
            "reproducible_build_claimed": False,
        },
        "native_diagnostic": native_record,
        "hawking_metric_owner": hawking_record,
        "transaction_wrappers": _source_records(
            TRANSACTION_WRAPPER_PATHS,
            "leased source-boundary transaction wrapper",
        ),
        "supersession": {
            "predecessor": _predecessor_binding(predecessor),
            "reason": (
                "A bound release binary or transaction wrapper changed. The predecessor remains immutable and "
                "must not be reused after source drift."
            ),
            "scientific_evidence_changed": False,
        },
        "runtime_actions": {
            "source_model_loaded": False,
            "native_model_loaded": False,
            "gpu_or_metal_started": False,
            "daemon_restarted": False,
            "kimi_loaded": False,
            "deployment_performed": False,
        },
        "owner_gate": {
            "detached_ed25519_required_for_local_canonical": False,
            "local_canonical_admission_accepted": True,
            "source_boundary_transaction_executed": False,
        },
        "next_safe_action": (
            "Use this closure only with the independently admitted local-canonical reference for one bounded "
            "source/native transaction; stop at the first exact payload or route difference. "
            "science_status remains UNEARNED."
        ),
        "claim_boundary": (
            "This is a static executable/source closure for bounded localization only. It is not a source/native "
            "comparison, source-parity result, complete NR, EBPW, TPS, capability, deployment, Pulsar promotion, "
            "or Kimi retirement."
        ),
    }
    document["seal_sha256"] = gate._compact_utf8_sorted_seal(document)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--hawking-binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--build-command", action="append", default=[])
    parser.add_argument("--predecessor", type=Path)
    args = parser.parse_args(argv)
    document = build_closure(
        native_binary=args.native_binary,
        hawking_binary=args.hawking_binary,
        out=args.out,
        target_dir=args.target_dir,
        build_commands=args.build_command,
        predecessor=args.predecessor,
    )
    print(json.dumps({
        "status": document["status"],
        "out": str(args.out.expanduser().resolve()),
        "seal_sha256": document["seal_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
