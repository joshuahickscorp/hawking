#!/usr/bin/env python3.12
"""Run and receiptize Appendix gates that do not open a model or use the GPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any

import appendix_contract
import spec_reentry_scaffold


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "hawking.appendix_cheap_gate_report.v3"
RELEASE_PACKET_SCHEMA = "hawking.appendix_release_packet_cheap_gate_report.v3"
SOURCE_CAPSULE_SCHEMA = "hawking.appendix_cheap_gate_source_capsule.v1"
EXECUTION_AUTHORITY_SCHEMA = "hawking.appendix_cheap_gate_execution_authority.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

PYTHON_BIN_DIR = "/Library/Frameworks/Python.framework/Versions/3.12/bin"
CARGO_BIN_DIR = "/opt/homebrew/bin"
SYSTEM_BIN_DIRS = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")
TOOL_NAMES = ("python3.12", "cargo", "rustc", "nice", "git")
TOOL_VERSION_ARGV = {
    "python3.12": ("--version",),
    "cargo": ("-Vv",),
    "rustc": ("-vV",),
    "git": ("--version",),
}

# This is an intentionally source-only selection.  It includes the dirty-tree
# bytes consumed by the Python and Rust gates (including tests and build
# scripts), but never walks reports, model artifacts, the active corpus, target,
# or build output.  Adding a matching source file changes the capsule just as
# editing an existing one does.
SOURCE_CAPSULE_GLOBS = (
    "Cargo.toml",
    "Cargo.lock",
    ".cargo/*.toml",
    ".cargo/config*",
    "rust-toolchain*",
    "crates/**/Cargo.toml",
    "crates/**/Cargo.lock",
    "crates/**/.cargo/config*",
    "crates/**/*.rs",
    "crates/**/*.metal",
    "tools/condense/*.py",
    "tools/condense/tests/*.py",
    "vendor/**/Cargo.toml",
    "vendor/**/Cargo.lock",
    "vendor/**/.cargo/config*",
    "vendor/**/*.rs",
    "vendor/**/*.metal",
    "docs/plans/appendix_counter_authority_allowed_signers",
    "docs/plans/appendix_counter_authority_registry.json",
)
SOURCE_CAPSULE_EXCLUDED_DIRECTORY_NAMES = frozenset({
    ".git", "__pycache__", "build", "reports", "target",
})
GATES = (
    ("appendix_catalog_selftest", ["python3.12", "tools/condense/appendix_catalog.py", "--selftest"]),
    ("appendix_cheap_gate_selftest", ["python3.12", "tools/condense/appendix_cheap_gates.py", "--selftest"]),
    ("appendix_contract_selftest", ["python3.12", "tools/condense/appendix_contract.py", "--selftest"]),
    ("appendix_corpus_selftest", ["python3.12", "tools/condense/appendix_corpus.py", "--selftest"]),
    ("appendix_probe_selftest", ["python3.12", "tools/condense/tq_runtime_probe.py", "--selftest"]),
    ("tq_runtime_matrix_selftest", ["python3.12", "tools/condense/tq_runtime_matrix.py", "--selftest"]),
    ("appendix_postrun_selftest", ["python3.12", "tools/condense/appendix_postrun.py", "--selftest"]),
    ("appendix_physical_counter_collector_selftest", ["python3.12", "tools/condense/appendix_physical_counter_collector.py", "--selftest"]),
    ("appendix_physical_evidence_gate_selftest", ["python3.12", "tools/condense/appendix_physical_evidence_gate.py", "--selftest"]),
    ("appendix_physical_release_state_selftest", ["python3.12", "tools/condense/appendix_physical_release_state.py", "--selftest"]),
    ("appendix_device_runner_selftest", ["python3.12", "tools/condense/appendix_device_runner.py", "--selftest"]),
    ("tq_receipt_contract_selftest", ["python3.12", "tools/condense/tq_receipt_contract.py", "--selftest"]),
    ("appendix_ledger_selftest", ["python3.12", "tools/condense/appendix_ledger.py", "--selftest"]),
    ("spec_reentry_selftest", ["python3.12", "tools/condense/spec_reentry_scaffold.py", "--selftest"]),
    ("spec_receipt_contract_selftest", ["python3.12", "tools/condense/spec_receipt_contract.py", "--selftest"]),
    ("spec_tq_runner_selftest", ["python3.12", "tools/condense/spec_tq_runner.py", "--selftest"]),
    ("appendix_master_selftest", ["python3.12", "tools/condense/appendix_scaffold.py", "--selftest"]),
    ("appendix_handoff_audit", ["python3.12", "tools/condense/appendix_handoff.py", "--audit"]),
    (
        "appendix_pytests",
        [
            "python3.12", "-m", "pytest", "-q",
            "tools/condense/tests/test_appendix_catalog.py",
            "tools/condense/tests/test_appendix_cheap_gates.py",
            "tools/condense/tests/test_appendix_contract.py",
            "tools/condense/tests/test_appendix_corpus.py",
            "tools/condense/tests/test_appendix_device_runner.py",
            "tools/condense/tests/test_appendix_handoff.py",
            "tools/condense/tests/test_appendix_ledger.py",
            "tools/condense/tests/test_appendix_postrun.py",
            "tools/condense/tests/test_appendix_physical_counter_collector.py",
            "tools/condense/tests/test_appendix_physical_evidence_gate.py",
            "tools/condense/tests/test_appendix_physical_release_state.py",
            "tools/condense/tests/test_appendix_scaffold.py",
            "tools/condense/tests/test_spec_reentry_scaffold.py",
            "tools/condense/tests/test_spec_receipt_contract.py",
            "tools/condense/tests/test_spec_tq_runner.py",
            "tools/condense/tests/test_tq_runtime_probe.py",
            "tools/condense/tests/test_tq_runtime_matrix.py",
            "tools/condense/tests/test_tq_receipt_contract.py",
        ],
    ),
    (
        "vendor_tq_gate_compile",
        [
            "nice", "-n", "15", "cargo", "check", "--manifest-path",
            "vendor/strand-decode-kernel/Cargo.toml",
            "--bin", "gate-bitslice", "--bin", "gate-tablecompact",
            "--bin", "gate-coopwindow", "--bin", "gate-token-buffer",
            "--bin", "gate-bitslice-staged",
        ],
    ),
    (
        "tq_runtime_accounting_tests",
        ["nice", "-n", "15", "cargo", "test", "-p", "hawking-core", "--features", "tq", "--lib", "tq_gpu::tests::runtime_", "--", "--test-threads=1"],
    ),
    (
        "tq_gpu_host_invariant_tests",
        ["nice", "-n", "15", "cargo", "test", "-p", "hawking-core", "--features", "tq", "--lib", "tq_gpu::tests::gpu_", "--", "--test-threads=1"],
    ),
    (
        "tq_runtime_kernel_source_contract",
        ["nice", "-n", "15", "cargo", "test", "-p", "hawking-core", "--features", "tq", "--lib", "tq_gpu::tests::every_runtime_path_has_decode_and_fused_kernel_sources", "--", "--test-threads=1"],
    ),
    (
        "spec_router_tests",
        ["nice", "-n", "15", "cargo", "test", "-p", "hawking-core", "--features", "tq", "--lib", "speculate::router::tests", "--", "--test-threads=1"],
    ),
    (
        "spec_governor_tests",
        ["nice", "-n", "15", "cargo", "test", "-p", "hawking-core", "--features", "tq", "--lib", "speculate::governor::tests", "--", "--test-threads=1"],
    ),
    (
        "computed_codebook_contract",
        ["nice", "-n", "15", "cargo", "test", "--manifest-path", "vendor/strand-quant/Cargo.toml", "--lib", "codebook::tests::computed_", "--", "--test-threads=1"],
    ),
    (
        "scalar_decode_source_identity",
        ["nice", "-n", "15", "cargo", "test", "--manifest-path", "vendor/strand-quant/Cargo.toml", "--lib", "tests::scalar_decode_sources_are_bit_identical", "--", "--test-threads=1"],
    ),
    (
        "absent_sub_scale_unity_contract",
        ["nice", "-n", "15", "cargo", "test", "--manifest-path", "vendor/strand-decode-kernel/Cargo.toml", "--lib", "block_walk::tests::absent_sub_scale_stream_is_canonical_unity", "--", "--test-threads=1"],
    ),
    (
        "hawking_tq_check",
        ["nice", "-n", "15", "cargo", "check", "-p", "hawking-core", "--features", "tq"],
    ),
    (
        "hawking_appendix_probe_checks",
        ["nice", "-n", "15", "cargo", "check", "-p", "hawking", "--features", "tq", "--bin", "hawking-tq-device-probe", "--bin", "hawking-tq-spec-probe"],
    ),
    ("diff_check", ["git", "diff", "--check"]),
)

# This narrow extension is intentionally separate from the main Appendix
# report.  It can be regenerated while Doctor owns the machine: no command
# builds Rust, opens a model/corpus file, or uses the GPU.  Keeping a separate
# exact manifest avoids retroactively claiming that another report executed
# these tests.
RELEASE_PACKET_GATES = (
    (
        "appendix_release_packet_selftest",
        ["python3.12", "tools/condense/appendix_physical_release_packet.py", "--selftest"],
    ),
    (
        "appendix_counter_authority_selftest",
        ["python3.12", "tools/condense/appendix_physical_counter_authority.py", "selftest"],
    ),
    (
        "appendix_counter_executor_selftest",
        ["python3.12", "tools/condense/appendix_physical_counter_executor.py", "--selftest"],
    ),
    (
        "appendix_counter_request_selftest",
        ["python3.12", "tools/condense/appendix_physical_counter_request.py", "selftest"],
    ),
    (
        "appendix_release_packet_pycompile",
        [
            "python3.12", "-m", "py_compile",
            "tools/condense/appendix_corpus.py",
            "tools/condense/appendix_postrun.py",
            "tools/condense/appendix_physical_evidence_gate.py",
            "tools/condense/appendix_physical_release_packet.py",
            "tools/condense/appendix_physical_counter_authority.py",
            "tools/condense/appendix_physical_counter_executor.py",
            "tools/condense/appendix_physical_counter_request.py",
        ],
    ),
    (
        "appendix_release_packet_pytests",
        [
            "python3.12", "-m", "pytest", "-q",
            "tools/condense/tests/test_appendix_corpus.py",
            "tools/condense/tests/test_appendix_postrun.py",
            "tools/condense/tests/test_appendix_physical_evidence_gate.py",
            "tools/condense/tests/test_appendix_physical_release_packet.py",
            "tools/condense/tests/test_appendix_physical_release_state.py",
            "tools/condense/tests/test_appendix_physical_counter_authority.py",
            "tools/condense/tests/test_appendix_physical_counter_executor.py",
            "tools/condense/tests/test_appendix_physical_counter_request.py",
        ],
    ),
)


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _command_manifest(gates: tuple[tuple[str, list[str]], ...]) -> list[dict[str, Any]]:
    return [
        {"id": gate_id, "argv": list(command)}
        for gate_id, command in gates
    ]


def gate_environment() -> dict[str, str]:
    """Return the complete, minimal environment admitted for every gate.

    The report is not allowed to inherit Python path/plugin injection, Cargo
    wrappers/flags, a caller-selected Rust toolchain, locale, or an alternate
    PATH.  Rustup/Cargo still need their on-host homes, so those exact paths
    are explicit evidence rather than invisible ambient inputs.
    """
    home = str(pathlib.Path.home().resolve())
    return {
        "PATH": ":".join((PYTHON_BIN_DIR, CARGO_BIN_DIR, *SYSTEM_BIN_DIRS)),
        "HOME": home,
        "CARGO_HOME": str(pathlib.Path(home, ".cargo")),
        "RUSTUP_HOME": str(pathlib.Path(home, ".rustup")),
        "CARGO_BUILD_JOBS": "2",
        "CARGO_INCREMENTAL": "0",
        "RUST_TEST_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "TMPDIR": "/tmp",
        "RUSTFLAGS": "",
        "CARGO_ENCODED_RUSTFLAGS": "",
        "RUSTC_WRAPPER": "",
        "RUSTC_WORKSPACE_WRAPPER": "",
        "CARGO_BUILD_RUSTC_WRAPPER": "",
    }


def _tool_binding(name: str, environment: dict[str, str]) -> dict[str, Any]:
    invocation = shutil.which(name, path=environment["PATH"])
    if invocation is None:
        raise RuntimeError(f"cheap-gate tool is absent from the fixed PATH: {name}")
    invocation_path = pathlib.Path(invocation).absolute()
    resolved = invocation_path.resolve(strict=True)
    entry = _stable_source_entry_external(resolved)
    binding = {
        "name": name,
        "invocation_path": str(invocation_path),
        "resolved_path": str(resolved),
        "size_bytes": entry["size_bytes"],
        "sha256": entry["sha256"],
    }
    version_args = TOOL_VERSION_ARGV.get(name)
    if version_args is not None:
        process = subprocess.run(
            [str(invocation_path), *version_args], cwd=ROOT,
            env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False,
        )
        version_text = (process.stdout + process.stderr).strip()
        if process.returncode != 0 or not version_text:
            raise RuntimeError(f"cheap-gate tool version probe failed: {name}")
        binding["version_argv"] = [str(invocation_path), *version_args]
        binding["version_text"] = version_text
        binding["version_sha256"] = hashlib.sha256(
            version_text.encode("utf-8")
        ).hexdigest()
    return binding


def _stable_source_entry_external(path: pathlib.Path) -> dict[str, Any]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"cheap-gate tool is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"cheap-gate tool changed before open: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current) \
            or size != before.st_size:
        raise RuntimeError(f"cheap-gate tool changed during capture: {path}")
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def execution_authority() -> dict[str, Any]:
    environment = gate_environment()
    tools = [_tool_binding(name, environment) for name in TOOL_NAMES]
    authority = {
        "schema": EXECUTION_AUTHORITY_SCHEMA,
        "environment": environment,
        "environment_sha256": appendix_contract.canonical_sha256(environment),
        "tools": tools,
        "tools_sha256": appendix_contract.canonical_sha256(tools),
        "ambient_environment_inherited": False,
        "pytest_plugin_autoload": False,
    }
    authority["authority_sha256"] = appendix_contract.canonical_sha256(authority)
    return authority


def _selected_source_paths() -> list[pathlib.Path]:
    selected: set[pathlib.Path] = set()
    for pattern in SOURCE_CAPSULE_GLOBS:
        for path in ROOT.glob(pattern):
            relative = path.relative_to(ROOT)
            if any(
                part in SOURCE_CAPSULE_EXCLUDED_DIRECTORY_NAMES
                for part in relative.parts[:-1]
            ):
                continue
            if path.is_file() or path.is_symlink():
                selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(ROOT).as_posix())


def _stable_source_entry(path: pathlib.Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"cheap-gate source capsule rejects non-regular path: {relative}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_after != identity_before or size != before.st_size:
        raise RuntimeError(f"cheap-gate source changed during capture: {relative}")
    return {"path": relative, "size_bytes": size, "sha256": digest.hexdigest()}


def current_source_capsule() -> dict[str, Any]:
    """Hash the exact current dirty-tree bytes in the cheap-gate source scope."""
    paths = _selected_source_paths()
    if not paths:
        raise RuntimeError("cheap-gate source capsule selection is empty")
    entries = [_stable_source_entry(path) for path in paths]
    if _selected_source_paths() != paths:
        raise RuntimeError("cheap-gate source path set changed during capture")
    capsule = {
        "schema": SOURCE_CAPSULE_SCHEMA,
        "scope": "dirty-tree-source-bytes-no-reports-models-corpus-target-or-build",
        "selection_globs": list(SOURCE_CAPSULE_GLOBS),
        "selection_globs_sha256": appendix_contract.canonical_sha256(
            list(SOURCE_CAPSULE_GLOBS)
        ),
        "excluded_directory_names": sorted(SOURCE_CAPSULE_EXCLUDED_DIRECTORY_NAMES),
        "entry_count": len(entries),
        "entries": entries,
        "entries_sha256": appendix_contract.canonical_sha256(entries),
    }
    capsule["capsule_sha256"] = appendix_contract.canonical_sha256(capsule)
    return capsule


def gate_contract() -> dict[str, Any]:
    """Return the current command/source authority used by release planning."""
    source = current_source_capsule()
    main = _command_manifest(GATES)
    release = _command_manifest(RELEASE_PACKET_GATES)
    execution = execution_authority()
    return {
        "main_gate_manifest_sha256": appendix_contract.canonical_sha256(main),
        "release_packet_gate_manifest_sha256": appendix_contract.canonical_sha256(release),
        "source_capsule_sha256": source["capsule_sha256"],
        "execution_authority_sha256": execution["authority_sha256"],
    }


def _stamp(report: dict) -> dict:
    stamped = copy.deepcopy(report)
    stamped.pop("report_sha256", None)
    stamped["report_sha256"] = appendix_contract.canonical_sha256(stamped)
    return stamped


def run_gates() -> dict:
    source_before = current_source_capsule()
    before = spec_reentry_scaffold.active_heavy_owners()
    rows = []
    execution = execution_authority()
    env = dict(execution["environment"])
    for gate_id, command in GATES:
        start = time.monotonic_ns()
        try:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                check=False,
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            timed_out = True
        duration_ns = time.monotonic_ns() - start
        rows.append({
            "id": gate_id,
            "command": command,
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "timed_out": timed_out,
            "duration_ns": duration_ns,
            "stdout_sha256": _digest_text(stdout),
            "stderr_sha256": _digest_text(stderr),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        })
    after = spec_reentry_scaffold.active_heavy_owners()
    source_after = current_source_capsule()
    execution_after = execution_authority()
    command_manifest = _command_manifest(GATES)
    return _stamp({
        "schema": SCHEMA,
        "source_commit": _source_commit(),
        "source_commit_role": "repository-base-only-not-byte-authority",
        "gate_manifest": command_manifest,
        "gate_manifest_sha256": appendix_contract.canonical_sha256(command_manifest),
        "source_capsule": source_before,
        "source_capsule_sha256": source_before["capsule_sha256"],
        "source_capsule_after_sha256": source_after["capsule_sha256"],
        "source_capsule_stable_during_run": source_after == source_before,
        "execution_authority": execution,
        "execution_authority_sha256": execution["authority_sha256"],
        "execution_authority_after_sha256": execution_after["authority_sha256"],
        "execution_authority_stable_during_run": execution_after == execution,
        "uses_gpu": False,
        "reads_model_artifacts": False,
        "mutates_active_corpus": False,
        "cargo_build_jobs": env["CARGO_BUILD_JOBS"],
        "active_heavy_owner_count_before": len(before),
        "active_heavy_owner_count_after": len(after),
        "gate_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "failed_count": sum(not row["passed"] for row in rows),
        "gates": rows,
    })


def run_release_packet_gates() -> dict:
    """Run only the no-Cargo/no-model release-packet extension manifest."""
    source_before = current_source_capsule()
    before = spec_reentry_scaffold.active_heavy_owners()
    rows = []
    execution = execution_authority()
    env = dict(execution["environment"])
    for gate_id, command in RELEASE_PACKET_GATES:
        start = time.monotonic_ns()
        try:
            proc = subprocess.run(
                command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=120, check=False,
            )
            exit_code = proc.returncode
            stdout, stderr = proc.stdout, proc.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout, stderr = exc.stdout or "", exc.stderr or ""
            timed_out = True
        rows.append({
            "id": gate_id,
            "command": command,
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "timed_out": timed_out,
            "duration_ns": time.monotonic_ns() - start,
            "stdout_sha256": _digest_text(stdout),
            "stderr_sha256": _digest_text(stderr),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        })
    after = spec_reentry_scaffold.active_heavy_owners()
    source_after = current_source_capsule()
    execution_after = execution_authority()
    command_manifest = _command_manifest(RELEASE_PACKET_GATES)
    return _stamp({
        "schema": RELEASE_PACKET_SCHEMA,
        "source_base_commit": _source_commit(),
        "source_base_commit_role": "repository-base-only-not-byte-authority",
        "gate_manifest": command_manifest,
        "gate_manifest_sha256": appendix_contract.canonical_sha256(command_manifest),
        "source_capsule": source_before,
        "source_capsule_sha256": source_before["capsule_sha256"],
        "source_capsule_after_sha256": source_after["capsule_sha256"],
        "source_capsule_stable_during_run": source_after == source_before,
        "execution_authority": execution,
        "execution_authority_sha256": execution["authority_sha256"],
        "execution_authority_after_sha256": execution_after["authority_sha256"],
        "execution_authority_stable_during_run": execution_after == execution,
        "uses_gpu": False,
        "reads_model_artifacts": False,
        "opens_or_hashes_active_corpus": False,
        "runs_cargo": False,
        "mutates_active_corpus": False,
        "mutates_runtime_defaults": False,
        "active_heavy_owner_count_before": len(before),
        "active_heavy_owner_count_after": len(after),
        "gate_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "failed_count": sum(not row["passed"] for row in rows),
        "gates": rows,
    })


def _authority_errors(
    report: dict[str, Any], gates: tuple[tuple[str, list[str]], ...],
) -> list[str]:
    errors: list[str] = []
    expected_manifest = _command_manifest(gates)
    expected_manifest_sha256 = appendix_contract.canonical_sha256(expected_manifest)
    if report.get("gate_manifest") != expected_manifest:
        errors.append("gate command manifest differs from the exact current manifest")
    if report.get("gate_manifest_sha256") != expected_manifest_sha256:
        errors.append("gate command manifest hash differs from the exact current manifest")

    execution = report.get("execution_authority")
    if not isinstance(execution, dict):
        errors.append("gate execution authority is missing")
    else:
        unstamped_execution = copy.deepcopy(execution)
        claimed_execution = unstamped_execution.pop("authority_sha256", None)
        if claimed_execution != appendix_contract.canonical_sha256(unstamped_execution):
            errors.append("gate execution authority self-hash mismatch")
        if report.get("execution_authority_sha256") != claimed_execution:
            errors.append("report is not bound to its execution authority")
        if report.get("execution_authority_after_sha256") != claimed_execution \
                or report.get("execution_authority_stable_during_run") is not True:
            errors.append("gate execution authority was not stable for the complete run")
        if execution.get("schema") != EXECUTION_AUTHORITY_SCHEMA \
                or execution.get("ambient_environment_inherited") is not False \
                or execution.get("pytest_plugin_autoload") is not False:
            errors.append("gate execution authority is unsafe or has the wrong schema")
        environment = execution.get("environment")
        tools = execution.get("tools")
        if not isinstance(environment, dict) or (
            execution.get("environment_sha256")
            != appendix_contract.canonical_sha256(environment)
        ):
            errors.append("gate environment binding is malformed")
        if not isinstance(tools, list) or (
            execution.get("tools_sha256") != appendix_contract.canonical_sha256(tools)
        ):
            errors.append("gate tool binding is malformed")
        try:
            current_execution = execution_authority()
        except (OSError, RuntimeError) as exc:
            errors.append(f"cannot capture current gate execution authority: {exc}")
        else:
            if current_execution != execution:
                errors.append("gate interpreter/tool/environment authority drifted")

    capsule = report.get("source_capsule")
    if not isinstance(capsule, dict):
        errors.append("source capsule is missing")
        return errors
    if capsule.get("schema") != SOURCE_CAPSULE_SCHEMA:
        errors.append("source capsule schema is invalid")
    unstamped = copy.deepcopy(capsule)
    claimed = unstamped.pop("capsule_sha256", None)
    if claimed != appendix_contract.canonical_sha256(unstamped):
        errors.append("source capsule self-hash mismatch")
    entries = capsule.get("entries")
    if not isinstance(entries, list) or capsule.get("entry_count") != len(entries):
        errors.append("source capsule entries/count are malformed")
    elif (
        any(
            not isinstance(row, dict)
            or set(row) != {"path", "size_bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("size_bytes"), int)
            or isinstance(row.get("size_bytes"), bool)
            or row.get("size_bytes", -1) < 0
            or not isinstance(row.get("sha256"), str)
            or HEX64.fullmatch(row.get("sha256", "")) is None
            for row in entries
        )
        or [row["path"] for row in entries] != sorted({row["path"] for row in entries})
    ):
        errors.append("source capsule entries are invalid, duplicated, or unordered")
    if capsule.get("selection_globs") != list(SOURCE_CAPSULE_GLOBS) or (
        capsule.get("selection_globs_sha256")
        != appendix_contract.canonical_sha256(list(SOURCE_CAPSULE_GLOBS))
    ):
        errors.append("source capsule selection differs from current source scope")
    if capsule.get("excluded_directory_names") != sorted(
        SOURCE_CAPSULE_EXCLUDED_DIRECTORY_NAMES
    ):
        errors.append("source capsule exclusions differ from current source scope")
    if isinstance(entries, list) and (
        capsule.get("entries_sha256") != appendix_contract.canonical_sha256(entries)
    ):
        errors.append("source capsule entry hash mismatch")
    if report.get("source_capsule_sha256") != claimed:
        errors.append("report is not bound to its source capsule")
    if report.get("source_capsule_after_sha256") != claimed or (
        report.get("source_capsule_stable_during_run") is not True
    ):
        errors.append("source capsule was not stable for the complete gate run")
    try:
        current = current_source_capsule()
    except (OSError, RuntimeError) as exc:
        errors.append(f"cannot capture current source capsule: {exc}")
    else:
        if current != capsule:
            errors.append("source bytes drifted since the gate report was produced")
    return errors


def verify_release_packet_report(report: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["release-packet cheap-gate report must be an object"]
    if report.get("schema") != RELEASE_PACKET_SCHEMA:
        errors.append(f"schema must be {RELEASE_PACKET_SCHEMA}")
    unstamped = copy.deepcopy(report)
    claimed = unstamped.pop("report_sha256", None)
    if claimed != appendix_contract.canonical_sha256(unstamped):
        errors.append("report_sha256 mismatch")
    if (
        report.get("source_base_commit_role")
        != "repository-base-only-not-byte-authority"
    ):
        errors.append("release-packet report overclaims base-commit byte authority")
    errors.extend(_authority_errors(report, RELEASE_PACKET_GATES))
    for field in (
        "uses_gpu", "reads_model_artifacts", "opens_or_hashes_active_corpus",
        "runs_cargo", "mutates_active_corpus", "mutates_runtime_defaults",
    ):
        if report.get(field) is not False:
            errors.append(f"release-packet report safety field {field} must be false")
    gates = report.get("gates")
    expected_ids = [gate_id for gate_id, _command in RELEASE_PACKET_GATES]
    if not isinstance(gates, list) or [
        row.get("id") for row in gates if isinstance(row, dict)
    ] != expected_ids:
        errors.append("release-packet gate IDs/order differ from exact manifest")
        return errors
    for row, (_gate_id, command) in zip(gates, RELEASE_PACKET_GATES):
        if not isinstance(row, dict) or row.get("command") != command:
            errors.append("release-packet gate command differs from exact manifest")
        elif row.get("passed") is not True or row.get("exit_code") != 0:
            errors.append(f"release-packet gate failed: {row.get('id')}")
    if report.get("gate_count") != len(RELEASE_PACKET_GATES) \
            or report.get("passed_count") != len(RELEASE_PACKET_GATES) \
            or report.get("failed_count") != 0:
        errors.append("release-packet gate summary is not all-pass")
    return errors


def verify_report(report: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["gate report must be an object"]
    if report.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    unstamped = copy.deepcopy(report)
    claimed = unstamped.pop("report_sha256", None)
    if claimed != appendix_contract.canonical_sha256(unstamped):
        errors.append("report_sha256 mismatch")
    if report.get("uses_gpu") is not False or report.get("reads_model_artifacts") is not False:
        errors.append("cheap gate report must not use GPU or model artifacts")
    if report.get("mutates_active_corpus") is not False:
        errors.append("cheap gate report weakened active-corpus safety")
    if report.get("source_commit_role") != "repository-base-only-not-byte-authority":
        errors.append("cheap gate report overclaims base-commit byte authority")
    errors.extend(_authority_errors(report, GATES))
    gates = report.get("gates")
    if not isinstance(gates, list) or len(gates) != len(GATES):
        errors.append("gate list is missing or incomplete")
        return errors
    expected_ids = [gate_id for gate_id, _command in GATES]
    if [gate.get("id") for gate in gates if isinstance(gate, dict)] != expected_ids:
        errors.append("gate IDs/order differ from the current manifest")
    for gate, (_gate_id, command) in zip(gates, GATES):
        if not isinstance(gate, dict) or gate.get("command") != command:
            errors.append(
                f"gate command differs from exact manifest: "
                f"{gate.get('id') if isinstance(gate, dict) else '<malformed>'}"
            )
        if not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("exit_code") != 0:
            errors.append(f"gate failed: {gate.get('id') if isinstance(gate, dict) else '<malformed>'}")
    if report.get("passed_count") != len(GATES) or report.get("failed_count") != 0:
        errors.append("gate summary does not report an all-pass run")
    return errors


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selftest() -> int:
    source = current_source_capsule()
    execution = execution_authority()
    command_manifest = _command_manifest(GATES)
    rows = [
        {
            "id": gate_id,
            "command": command,
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "duration_ns": 1,
            "stdout_sha256": _digest_text(""),
            "stderr_sha256": _digest_text(""),
            "stdout_tail": "",
            "stderr_tail": "",
        }
        for gate_id, command in GATES
    ]
    fake = _stamp({
        "schema": SCHEMA,
        "source_commit": "0123456789abcdef",
        "source_commit_role": "repository-base-only-not-byte-authority",
        "gate_manifest": command_manifest,
        "gate_manifest_sha256": appendix_contract.canonical_sha256(command_manifest),
        "source_capsule": source,
        "source_capsule_sha256": source["capsule_sha256"],
        "source_capsule_after_sha256": source["capsule_sha256"],
        "source_capsule_stable_during_run": True,
        "execution_authority": execution,
        "execution_authority_sha256": execution["authority_sha256"],
        "execution_authority_after_sha256": execution["authority_sha256"],
        "execution_authority_stable_during_run": True,
        "uses_gpu": False,
        "reads_model_artifacts": False,
        "mutates_active_corpus": False,
        "cargo_build_jobs": "2",
        "active_heavy_owner_count_before": 1,
        "active_heavy_owner_count_after": 1,
        "gate_count": len(rows),
        "passed_count": len(rows),
        "failed_count": 0,
        "gates": rows,
    })
    assert verify_report(fake) == []
    broken = copy.deepcopy(fake)
    broken["gates"][0]["passed"] = False
    broken = _stamp(broken)
    assert any("gate failed" in error for error in verify_report(broken))
    print("appendix_cheap_gates.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--verify", type=pathlib.Path)
    parser.add_argument("--run-release-packet", action="store_true")
    parser.add_argument("--verify-release-packet", type=pathlib.Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.run_release_packet:
        report = run_release_packet_gates()
        if args.output is None:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _atomic_json(args.output, report)
        return 0 if not verify_release_packet_report(report) else 1
    if args.verify_release_packet is not None:
        errors = verify_release_packet_report(_load(args.verify_release_packet))
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    if args.run:
        report = run_gates()
        if args.output is None:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _atomic_json(args.output, report)
        return 0 if not verify_report(report) else 1
    if args.verify is not None:
        errors = verify_report(_load(args.verify))
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    parser.error("choose --run, --verify, or --selftest")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
