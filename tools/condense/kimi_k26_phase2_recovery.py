#!/usr/bin/env python3.12
"""Fail-closed reconstruction of the frozen Kimi-K2.6 rollback capsule.

This is deliberately a narrow Phase-2 tool.  It can verify authority, export
six specifically vetted Git blobs into a fresh private staging directory, run
the two historical generators without network access, and install two exact
binary artifacts.  It has no source-release or cleanup command.

``preflight`` and ``verify`` never run a scientific generator.  Only the
explicit ``generate`` command can do that.  Historical repository state is
read with ``git cat-file blob`` one allowlisted object at a time; this module
never checks out or archives the historical tree.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import os
import platform
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ContextManager, Protocol

if __package__:
    from . import kimi_k26_download_supervisor as download_supervisor
    from . import kimi_k26_release_cycle as phase1
else:  # direct module import or direct script execution
    import kimi_k26_download_supervisor as download_supervisor  # type: ignore[no-redef]
    import kimi_k26_release_cycle as phase1  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_COMMIT = "0210e5aa05f0e3c69d6f2022c539c9dc90cce322"
HISTORICAL_TREE = "f2ca03a67a0f11423e34badcac2586325c2652e1"
HISTORICAL_PARENT = "543147b4076e0eccdf00a678612fd8febc196f46"
CORPUS_SEAL_SHA256 = "91e6cdd09bfcc293f67dcdf7d7c675c4a615aeccac1ea4147411e61636ffd5f6"

GIT = Path("/usr/bin/git")
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12")
PYTHON_FRAMEWORK = Path("/Library/Frameworks/Python.framework/Versions/3.12/Python")
PYTHON_ROOT = Path("/Library/Frameworks/Python.framework/Versions/3.12")
SITE_PACKAGES = PYTHON_ROOT / "lib/python3.12/site-packages"

SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)"
BOOTSTRAP = (
    "import runpy,sys;"
    "script=sys.argv[1];"
    "sys.argv=sys.argv[1:];"
    "sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)];"
    "sys.pycache_prefix=sys.argv.pop(1);"
    "runpy.run_path(script,run_name='__main__')"
)
MAX_CHILD_OUTPUT_BYTES = 2 * 1024 * 1024
PHASE2_ROOT = PurePosixPath("phase2-recovery")
FINAL_RECEIPT = PurePosixPath("KIMI_K26_PHASE2_RECOVERY.json")
DOWNLOAD_LEASE_NAME = ".kimi-k26-download-supervisor.lease"

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_TOKENISH = ("TOKEN", "SECRET", "CREDENTIAL", "AUTH", "API_KEY")


class Phase2RecoveryError(RuntimeError):
    """Raised when exact recovery authority cannot be established."""


def _fail(message: str) -> None:
    raise Phase2RecoveryError(message)


@dataclass(frozen=True)
class BlobSpec:
    relative_path: str
    git_blob_sha1: str
    logical_bytes: int
    sha256: str
    git_mode: str = "100644"


HISTORICAL_BLOBS: tuple[BlobSpec, ...] = (
    BlobSpec(
        "KIMI_K26_CORPUS_INTEGRITY.json",
        "cf8c8f6d61fc7d76bea81d049abdba2cd2552c5c",
        9_628,
        "1b75504059b5e172da1cb92831e6392f850397f43dcab525b9f07102d0585dbe",
    ),
    BlobSpec(
        "tools/condense/kimi_k26_f1_bracket.py",
        "b1454905f7aad3f5afeb1bdd1845ab8e73de3429",
        39_959,
        "91d3ba0fe90013ce58b0ebd0f7945e48dc668bc3560b94c907d495b2b4419a36",
    ),
    BlobSpec(
        "tools/condense/kimi_k26_f1_doctor_auction.py",
        "1c55aa1a646034817f8e2f4c947e037de41fd418",
        18_473,
        "fc0117dcd1cec48fa4242739be6c0970ccf1190401a73a18c7cd923a699f8176",
    ),
    BlobSpec(
        "tools/condense/gravity_forge.py",
        "9fb5492e29a7e3dab02a37fc55903795a267de60",
        59_768,
        "253cf68aad0cb64b19c8cdb8bbfc34adced420a8ff0e4a95105d9f97f4134681",
    ),
    BlobSpec(
        "tools/condense/kimi_k26_adapter.py",
        "8033e3ad42576724362e1f5053fc8a14cc712475",
        16_429,
        "1219485109e2a980f3bbe0c6e66e7442c775553f8b61a60a0eb5556241e6aec0",
    ),
    BlobSpec(
        "tools/condense/kimi_k26_reference.py",
        "b47f788beaaa2ecbf00d289e15794f46abe86831",
        39_297,
        "43bda2bf42a24646aff9203793c95b5ee16320db41a4dc20e1fcf049d1edf18e",
    ),
)


@dataclass(frozen=True)
class RuntimeFileSpec:
    path: Path
    logical_bytes: int
    sha256: str
    uid: int
    mode: int
    hard_links: int = 1


RUNTIME_FILES: tuple[RuntimeFileSpec, ...] = (
    RuntimeFileSpec(
        PYTHON,
        152_096,
        "69e6adfac00978215c4e5f3eaf63d262a5fb646b6dca1897b29f78872c4c5a26",
        0,
        0o775,
    ),
    RuntimeFileSpec(
        PYTHON_FRAMEWORK,
        15_922_304,
        "963042bd702141c472aa58eb31001361fb4a4f27a75a5ce16df94d08c46f1bfc",
        0,
        0o775,
    ),
    RuntimeFileSpec(
        GIT,
        118_640,
        "31ec19f3253cc0044133c892bf65183a9fdb0cca36dfe04074a46d7201417da5",
        0,
        0o755,
        78,
    ),
    RuntimeFileSpec(
        SANDBOX_EXEC,
        53_216,
        "2d0d4cb4c8eab07c7261195798388d93c640c7f8db1bece63372946e4b00e91a",
        0,
        0o755,
    ),
)


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    version: str
    dist_info: str
    record_sha256: str


DISTRIBUTIONS: tuple[DistributionSpec, ...] = (
    DistributionSpec(
        "numpy", "2.2.6", "numpy-2.2.6.dist-info",
        "04c180a4b5a2da0e3cfe8b7b70dc230ee011c51199be05dc28a2044573843c2d",
    ),
    DistributionSpec(
        "mlx", "0.31.2", "mlx-0.31.2.dist-info",
        "26c574bc04b5feccb934032c0522daeb1b17968deda5dd5b1a063fdada4a20a3",
    ),
    DistributionSpec(
        "mlx-metal", "0.31.2", "mlx_metal-0.31.2.dist-info",
        "a413ca2a049142d2f0c649772ee6265d51d092ab8d4cc9a3d1afee6222831379",
    ),
    DistributionSpec(
        "ml_dtypes", "0.5.4", "ml_dtypes-0.5.4.dist-info",
        "7ff2e9e9bc5ff376e2d2b2bd177be5bbf10485f13b32114c0b180d1236466b0c",
    ),
    DistributionSpec(
        "torch", "2.6.0", "torch-2.6.0.dist-info",
        "4f56769c53cacc930cfd9bf20b7d31da5ec7ca2e0e50f5fad5f10bff7ec6f6e8",
    ),
    DistributionSpec(
        "tiktoken", "0.7.0", "tiktoken-0.7.0.dist-info",
        "91eace0a6e7482daea7103d017d2c1e636d8c540bff5075d2ad9f7e6b2af6939",
    ),
    DistributionSpec(
        "regex", "2026.4.4", "regex-2026.4.4.dist-info",
        "3fff86f1e8ce91b47a7820be118ebf536520af837bac0a2c1e79f5cfe9704fb7",
    ),
)


@dataclass(frozen=True)
class FrozenRecordSpec:
    relative_path: str
    logical_bytes: int
    sha256: str
    seal_sha256: str


FROZEN_RECORDS: tuple[FrozenRecordSpec, ...] = (
    FrozenRecordSpec(
        phase1.BEST_RESULT_RELATIVE,
        *phase1.RECOVERY_ENTRY_ALLOWLIST[phase1.BEST_RESULT_RELATIVE],
        phase1.BEST_RESULT_SEAL_SHA256,
    ),
    FrozenRecordSpec(
        phase1.TEACHER_CAPTURE_RECORD_RELATIVE,
        *phase1.RECOVERY_ENTRY_ALLOWLIST[phase1.TEACHER_CAPTURE_RECORD_RELATIVE],
        phase1.TEACHER_CAPTURE_RECORD_SEAL_SHA256,
    ),
    FrozenRecordSpec(
        phase1.GRAVITY_FINAL_RELATIVE,
        *phase1.RECOVERY_ENTRY_ALLOWLIST[phase1.GRAVITY_FINAL_RELATIVE],
        phase1.GRAVITY_FINAL_SEAL_SHA256,
    ),
)


@dataclass(frozen=True)
class BinarySpec:
    filename: str
    logical_bytes: int
    sha256: str


BRACKET_BINARIES: tuple[BinarySpec, ...] = (
    BinarySpec(
        "teacher_capture.npz", 6_036_848,
        "ffac463e8c28fc7df1490c511ee74c602ac2b0f520856110c63d35a5ac8df981",
    ),
    BinarySpec(
        "P1_sentinel_expert.k26f1", 5_248_756,
        "26792048fc5e63f7924ac005cd121757872e8879c0c4c6dafab8ef1f868f741a",
    ),
    BinarySpec(
        "P5_sentinel_expert.k26f1", 2_630_526,
        "c6aa80701c11cc851f84992acedd48ca71da3d9bea12343c2127aa9db71c700e",
    ),
)

DOCTOR_BINARIES: tuple[BinarySpec, ...] = (
    BinarySpec(
        "P1_BASE_OUTPUT_RECOVERY_R31.k26f1", 4_958_445,
        "958a869c988db4f3d13f4c5a144dcd28af644ff58ebc948859329ee59651658c",
    ),
    BinarySpec(
        phase1.BEST_PAYLOAD_BASENAME, phase1.BEST_PAYLOAD_BYTES,
        phase1.BEST_PAYLOAD_SHA256,
    ),
    BinarySpec(
        "P5_BASE_OUTPUT_RECOVERY_R31.k26f1", 2_340_216,
        "2d425f9eaa943546bc40ace426543fc23f51aa0747fbc6ee34f641185e6aa856",
    ),
    BinarySpec(
        "P5_DUAL_PATH_RECOVERY_R16X2.k26f1", 2_383_586,
        "dd56220801569e238cc646217e30860e8a6805323f4fe7d072f10bbbcd1754a7",
    ),
)

CAPSULE_BINARIES: tuple[BinarySpec, ...] = (
    BinarySpec(
        phase1.BEST_PAYLOAD_BASENAME,
        phase1.BEST_PAYLOAD_BYTES,
        phase1.BEST_PAYLOAD_SHA256,
    ),
    BinarySpec(
        phase1.TEACHER_CAPTURE_BASENAME,
        phase1.TEACHER_CAPTURE_BYTES,
        phase1.TEACHER_CAPTURE_SHA256,
    ),
)

BRACKET_EXPECTED_FILES = frozenset(
    {
        "teacher_capture.npz",
        "teacher_capture.json",
        "P1_sentinel_expert.k26f1",
        "P1_F1_RESULT.json",
        "P5_sentinel_expert.k26f1",
        "P5_F1_RESULT.json",
        "KIMI_K26_F1_REPRESENTATION_BRACKET.json",
        "KIMI_K26_SCIENTIFIC_STATUS.json",
        "KIMI_K26_F1_PROGRESS.json",
    }
)
DOCTOR_EXPECTED_FILES = frozenset(
    {spec.filename for spec in DOCTOR_BINARIES}
    | {
        spec.filename.removesuffix(".k26f1") + "_RESULT.json"
        for spec in DOCTOR_BINARIES
    }
    | {"KIMI_K26_F1_DOCTOR_AUCTION.json"}
)

SOURCE_DEPENDENCIES = (
    "config.json",
    "model-00001-of-000064.safetensors",
    "model-00002-of-000064.safetensors",
    "model-00062-of-000064.safetensors",
)


class ProcessRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path,
        pass_fds: Sequence[int],
    ) -> subprocess.CompletedProcess[bytes]: ...


@dataclass(frozen=True)
class Phase2Hooks:
    verify_source: Callable[[phase1.SessionLayout], dict[str, Any]]
    verify_archive: Callable[[], dict[str, Any]]
    verify_runtime: Callable[[], dict[str, Any]]
    load_historical_sources: Callable[[], Mapping[str, bytes]]
    extract_records: Callable[[phase1.SessionLayout, set[str]], dict[str, Any]]
    verify_capsule: Callable[[phase1.SessionLayout], dict[str, Any]]
    exclusive_lease: Callable[[phase1.SessionLayout], ContextManager[int]]
    source_guard: Callable[
        [phase1.SessionLayout, dict[str, Any]], ContextManager[dict[str, Any]]
    ]
    run_process: ProcessRunner

    @classmethod
    def live(
        cls,
        *,
        manifest_path: Path = phase1.OFFICIAL_MANIFEST,
        archive_path: Path = phase1.SANITIZED_ARCHIVE,
        mop_root: Path = phase1.MOP_ROOT,
        shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
    ) -> "Phase2Hooks":
        return cls(
            verify_source=lambda layout: phase1.verify_source(
                layout,
                manifest_path=manifest_path,
                mop_root=mop_root,
                shared_xet=shared_xet,
            ),
            verify_archive=lambda: phase1.verify_recovery_archive(archive_path),
            verify_runtime=verify_generation_runtime,
            load_historical_sources=lambda: load_historical_sources(REPO_ROOT),
            extract_records=lambda layout, names: phase1.extract_sanitized_recovery(
                layout,
                archive_path=archive_path,
                entries=names,
                mop_root=mop_root,
                shared_xet=shared_xet,
            ),
            verify_capsule=lambda layout: phase1.verify_payload_result_capture(
                layout, mop_root=mop_root, shared_xet=shared_xet
            ),
            exclusive_lease=_live_exclusive_lease,
            source_guard=_live_source_guard,
            run_process=_system_run_process,
        )


def _hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _verify_blob_mapping(values: Mapping[str, bytes]) -> dict[str, Any]:
    expected = {spec.relative_path: spec for spec in HISTORICAL_BLOBS}
    if set(values) != set(expected):
        missing = sorted(set(expected) - set(values))
        extra = sorted(set(values) - set(expected))
        _fail(f"historical export allowlist mismatch: missing={missing} extra={extra}")
    rows: list[dict[str, Any]] = []
    for name in sorted(expected):
        raw = values[name]
        if not isinstance(raw, bytes):
            _fail(f"historical blob is not immutable bytes: {name}")
        spec = expected[name]
        if len(raw) != spec.logical_bytes or _hash_bytes(raw) != spec.sha256:
            _fail(f"historical blob bytes changed: {name}")
        rows.append(
            {
                "relative_path": name,
                "git_mode": spec.git_mode,
                "git_blob_sha1": spec.git_blob_sha1,
                "logical_bytes": len(raw),
                "sha256": _hash_bytes(raw),
            }
        )
    corpus = phase1.strict_json_bytes(
        values["KIMI_K26_CORPUS_INTEGRITY.json"], label="frozen historical corpus"
    )
    phase1.verify_sealed_document(corpus, label="frozen historical corpus")
    if corpus.get("seal_sha256") != CORPUS_SEAL_SHA256:
        _fail("historical corpus canonical seal changed")
    if corpus.get("status") != "PASS" or corpus.get("source") != {
        "repo": phase1.KIMI_REPO,
        "revision": phase1.KIMI_REVISION,
    }:
        _fail("historical corpus source authority changed")
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.historical_export.v1",
            "status": "PASS_EXACT_SIX_BLOB_ALLOWLIST",
            "commit": HISTORICAL_COMMIT,
            "tree": HISTORICAL_TREE,
            "parent": HISTORICAL_PARENT,
            "blobs": rows,
            "corpus_seal_sha256": corpus["seal_sha256"],
            "checkout_used": False,
            "archive_used": False,
            "network_accessed": False,
        }
    )


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _hash_system_binary(spec: RuntimeFileSpec, *, label: str) -> dict[str, Any]:
    """Bind a root-owned SSV binary without imposing a single-link policy.

    macOS Signed System Volume files can legitimately share one inode across
    many names.  The exact link count is therefore evidence, not a reason to
    reject the otherwise immutable root-owned executable.
    """
    if spec.uid != 0 or spec.mode & 0o022:
        _fail(f"{label} specification is not root-owned and non-writable")
    digest = hashlib.sha256()
    with phase1._open_regular_fd(  # noqa: SLF001
        spec.path,
        label=label,
        expected_uid=0,
        require_single_link=False,
    ) as (descriptor, metadata):
        if (
            metadata.st_size != spec.logical_bytes
            or metadata.st_nlink != spec.hard_links
            or stat.S_IMODE(metadata.st_mode) != spec.mode
        ):
            _fail(f"{label} size, mode, or SSV link binding changed")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_post = os.fstat(descriptor)
        if (
            not phase1._identity_equal(metadata, opened_post)  # noqa: SLF001
            or stat.S_IMODE(opened_post.st_mode) != spec.mode
        ):
            _fail(f"{label} changed while hashing")
    if digest.hexdigest() != spec.sha256:
        _fail(f"{label} SHA-256 changed")
    return {
        "logical_bytes": int(metadata.st_size),
        "allocated_bytes": int(metadata.st_blocks) * 512,
        "sha256": digest.hexdigest(),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "hard_links": int(metadata.st_nlink),
        "uid": int(metadata.st_uid),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "ssv_multi_link_allowed": metadata.st_nlink > 1,
    }


def _git_read(repo_root: Path, *arguments: str) -> bytes:
    git_dir = repo_root / ".git"
    try:
        descriptor = phase1._open_absolute_directory(git_dir)  # noqa: SLF001
    except (OSError, phase1.ReleaseCycleError) as exc:
        raise Phase2RecoveryError(f"repository Git directory is unsafe: {exc}") from exc
    else:
        os.close(descriptor)
    completed = subprocess.run(
        [os.fspath(GIT), f"--git-dir={git_dir}", "--no-replace-objects", *arguments],
        env=_git_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        close_fds=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr[:1_000].decode("utf-8", "replace")
        _fail(f"pinned local git read failed: {detail}")
    return completed.stdout


def load_historical_sources(repo_root: Path = REPO_ROOT) -> dict[str, bytes]:
    """Read exactly six vetted historical blobs; never materialize the old tree."""
    git_spec = next(spec for spec in RUNTIME_FILES if spec.path == GIT)
    _hash_system_binary(git_spec, label="pinned SSV Git executable")
    commit_type = _git_read(repo_root, "cat-file", "-t", HISTORICAL_COMMIT).strip()
    commit_raw = _git_read(repo_root, "cat-file", "-p", HISTORICAL_COMMIT)
    if commit_type != b"commit":
        _fail("historical authority is not a commit")
    headers = commit_raw.split(b"\n\n", 1)[0].splitlines()
    if headers[:2] != [
        f"tree {HISTORICAL_TREE}".encode("ascii"),
        f"parent {HISTORICAL_PARENT}".encode("ascii"),
    ]:
        _fail("historical commit tree or parent changed")
    values: dict[str, bytes] = {}
    for spec in HISTORICAL_BLOBS:
        object_name = f"{HISTORICAL_COMMIT}:{spec.relative_path}"
        resolved = _git_read(repo_root, "rev-parse", "--verify", object_name).strip()
        if resolved != spec.git_blob_sha1.encode("ascii"):
            _fail(f"historical path resolves to a different blob: {spec.relative_path}")
        tree_row = _git_read(
            repo_root, "ls-tree", "-z", HISTORICAL_COMMIT, "--", spec.relative_path
        )
        expected_row = (
            f"{spec.git_mode} blob {spec.git_blob_sha1}\t{spec.relative_path}".encode("utf-8")
            + b"\x00"
        )
        if tree_row != expected_row:
            _fail(f"historical path mode/type changed: {spec.relative_path}")
        if _git_read(repo_root, "cat-file", "-t", spec.git_blob_sha1).strip() != b"blob":
            _fail(f"historical object is not a blob: {spec.relative_path}")
        size_raw = _git_read(repo_root, "cat-file", "-s", spec.git_blob_sha1).strip()
        if size_raw != str(spec.logical_bytes).encode("ascii"):
            _fail(f"historical blob size changed: {spec.relative_path}")
        values[spec.relative_path] = _git_read(
            repo_root, "cat-file", "blob", spec.git_blob_sha1
        )
    _verify_blob_mapping(values)
    return values


def _record_target(record_path: str) -> Path:
    if (
        not record_path
        or "\x00" in record_path
        or "\\" in record_path
        or record_path.startswith("/")
    ):
        _fail(f"unsafe runtime RECORD path: {record_path!r}")
    target = Path(os.path.normpath(os.path.join(SITE_PACKAGES, record_path)))
    if not phase1._within(target, PYTHON_ROOT):  # noqa: SLF001
        _fail(f"runtime RECORD path escapes Python root: {record_path}")
    return target


def _record_digest(encoded: str, *, label: str) -> str:
    if not encoded.startswith("sha256="):
        _fail(f"{label} does not use SHA-256")
    payload = encoded.removeprefix("sha256=")
    try:
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    except (ValueError, base64.binascii.Error) as exc:
        raise Phase2RecoveryError(f"{label} has invalid base64 digest") from exc
    if len(decoded) != 32:
        _fail(f"{label} has wrong digest length")
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if payload != canonical:
        _fail(f"{label} digest is not canonical")
    return decoded.hex()


def _verify_distribution(spec: DistributionSpec) -> dict[str, Any]:
    record_path = SITE_PACKAGES / spec.dist_info / "RECORD"
    record_raw = phase1._read_regular_bytes(  # noqa: SLF001
        record_path,
        label=f"{spec.name} RECORD",
        maximum_bytes=2_000_000,
        expected_uid=os.getuid(),
    )
    if _hash_bytes(record_raw) != spec.record_sha256:
        _fail(f"{spec.name} RECORD hash changed")
    metadata_raw = phase1._read_regular_bytes(  # noqa: SLF001
        SITE_PACKAGES / spec.dist_info / "METADATA",
        label=f"{spec.name} METADATA",
        maximum_bytes=2_000_000,
        expected_uid=os.getuid(),
    )
    name, version = phase1._metadata_identity(metadata_raw, label=spec.name)  # noqa: SLF001
    if (name, version) != (spec.name, spec.version):
        _fail(f"{spec.name} installed identity changed")
    try:
        rows = list(csv.reader(io.StringIO(record_raw.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise Phase2RecoveryError(f"invalid {spec.name} RECORD: {exc}") from exc
    seen: set[str] = set()
    hashed = 0
    unhashed_pyc = 0
    total_bytes = 0
    for index, row in enumerate(rows, 1):
        if len(row) != 3:
            _fail(f"{spec.name} RECORD row {index} does not have three fields")
        relative, digest, size_text = row
        if relative in seen:
            _fail(f"{spec.name} RECORD repeats {relative}")
        seen.add(relative)
        target = _record_target(relative)
        if not digest:
            is_record = relative == f"{spec.dist_info}/RECORD"
            is_pyc = "/__pycache__/" in f"/{relative}" and relative.endswith(".pyc")
            if size_text or not (is_record or is_pyc):
                _fail(f"{spec.name} has an unauthorized unhashed RECORD row: {relative}")
            unhashed_pyc += int(is_pyc)
            continue
        if not size_text.isdecimal() or (len(size_text) > 1 and size_text.startswith("0")):
            _fail(f"{spec.name} RECORD has invalid size: {relative}")
        expected_size = int(size_text)
        expected_sha = _record_digest(digest, label=f"{spec.name}:{relative}")
        facts = phase1._hash_regular(  # noqa: SLF001
            target, label=f"{spec.name} runtime file {relative}", expected_uid=os.getuid()
        )
        if facts["logical_bytes"] != expected_size or facts["sha256"] != expected_sha:
            _fail(f"{spec.name} installed file changed: {relative}")
        if int(facts["mode"], 8) & 0o022:
            _fail(f"{spec.name} installed file is group/world writable: {relative}")
        hashed += 1
        total_bytes += expected_size
    return {
        "name": spec.name,
        "version": spec.version,
        "dist_info": spec.dist_info,
        "record_sha256": spec.record_sha256,
        "record_rows": len(rows),
        "hashed_files": hashed,
        "unhashed_pyc_ignored_by_fresh_pycache_prefix": unhashed_pyc,
        "hashed_logical_bytes": total_bytes,
    }


def verify_generation_runtime() -> dict[str, Any]:
    """Hash every pinned runtime artifact without importing installed packages."""
    runtime_files: list[dict[str, Any]] = []
    for spec in RUNTIME_FILES:
        label = f"generation runtime {spec.path.name}"
        if spec.path in {GIT, SANDBOX_EXEC}:
            facts = _hash_system_binary(spec, label=label)
        else:
            facts = phase1._hash_regular(  # noqa: SLF001
                spec.path, label=label, expected_uid=spec.uid
            )
            phase1._expect_file_facts(  # noqa: SLF001
                facts,
                label=label,
                logical_bytes=spec.logical_bytes,
                sha256=spec.sha256,
                mode=spec.mode,
                uid=spec.uid,
            )
        runtime_files.append({"path": os.fspath(spec.path), **facts})
    distributions = [_verify_distribution(spec) for spec in DISTRIBUTIONS]
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        _fail("generation runtime is not Darwin arm64")
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.generation_runtime.v1",
            "status": "PASS_EXACT_PINNED_RUNTIME",
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "runtime_files": runtime_files,
            "distributions": distributions,
            "site_packages": os.fspath(SITE_PACKAGES),
            "python_flags": ["-I", "-S", "-B"],
            "bootstrap_sha256": _hash_bytes(BOOTSTRAP.encode("utf-8")),
            "sandbox_profile": SANDBOX_PROFILE,
            "sandbox_profile_sha256": _hash_bytes(SANDBOX_PROFILE.encode("utf-8")),
            "package_imports_executed_by_verifier": False,
            "network_accessed": False,
        }
    )


def _require_source_verification(value: dict[str, Any]) -> dict[str, Any]:
    phase1.verify_sealed_document(value, label="Phase-1 source verification")
    expected = {
        "schema": "hawking.kimi_k26.release_cycle.source_verification.v1",
        "status": "PASS_EXACT_IMMUTABLE_SOURCE",
        "repo": phase1.KIMI_REPO,
        "revision": phase1.KIMI_REVISION,
        "manifest_seal_sha256": phase1.KIMI_MANIFEST_SEAL_SHA256,
        "file_count": phase1.KIMI_FILE_COUNT,
        "weight_shards": phase1.KIMI_WEIGHT_SHARDS,
        "logical_bytes": phase1.KIMI_TOTAL_BYTES,
        "weight_bytes": phase1.KIMI_WEIGHT_BYTES,
        "shared_xet_used": False,
        "mop_touched": False,
        "network_accessed_by_verifier": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            _fail(f"source verification {key} changed")
    return value


def _require_archive_verification(value: dict[str, Any]) -> dict[str, Any]:
    phase1.verify_sealed_document(value, label="sanitized recovery archive")
    expected = {
        "schema": "hawking.kimi_k26.release_cycle.recovery_archive.v1",
        "status": "PASS_SANITIZED_EXACT_ARCHIVE",
        "archive_sha256": phase1.ACCEPTED_ARCHIVE_SHA256,
        "archive_bytes": phase1.ACCEPTED_ARCHIVE_BYTES,
        "entry_count": phase1.ACCEPTED_ARCHIVE_ENTRIES,
        "credential_entry_present": False,
        "old_898063_byte_archive_rejected": True,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            _fail(f"recovery archive verification {key} changed")
    return value


def _require_runtime_verification(value: dict[str, Any]) -> dict[str, Any]:
    phase1.verify_sealed_document(value, label="Phase-2 generation runtime")
    if value.get("schema") != "hawking.kimi_k26.phase2.generation_runtime.v1":
        _fail("generation runtime schema changed")
    if value.get("status") != "PASS_EXACT_PINNED_RUNTIME":
        _fail("generation runtime did not pass")
    if value.get("network_accessed") is not False:
        _fail("generation runtime verifier used network")
    return value


def _scan_no_incomplete(root: Path) -> int:
    root_fd = phase1._open_absolute_directory(root)  # noqa: SLF001
    count = 0

    def walk(descriptor: int, prefix: PurePosixPath) -> None:
        nonlocal count
        for name in sorted(os.listdir(descriptor)):
            if name in {".", ".."} or "/" in name or "\x00" in name:
                _fail("unsafe cache directory entry")
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = prefix / name
            count += 1
            if ".incomplete" in name:
                _fail(f"stale or active incomplete transfer blocks recovery: {root}/{relative}")
            if stat.S_ISDIR(named.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                    dir_fd=descriptor,
                )
                try:
                    opened = os.fstat(child)
                    if not phase1._identity_equal(named, opened):  # noqa: SLF001
                        _fail(f"cache directory changed while scanning: {relative}")
                    walk(child, relative)
                finally:
                    os.close(child)
            elif not (stat.S_ISREG(named.st_mode) or stat.S_ISLNK(named.st_mode)):
                _fail(f"special cache entry blocks recovery: {relative}")

    try:
        walk(root_fd, PurePosixPath())
    finally:
        os.close(root_fd)
    return count


def preflight(
    layout: phase1.SessionLayout,
    *,
    hooks: Phase2Hooks | None = None,
    mop_root: Path = phase1.MOP_ROOT,
    shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    """Read-only authority verification.  It performs no extraction/generation."""
    selected = hooks or Phase2Hooks.live(mop_root=mop_root, shared_xet=shared_xet)
    phase1.validate_layout(layout, mop_root=mop_root, shared_xet=shared_xet)
    source = _require_source_verification(selected.verify_source(layout))
    scanned_entries = _scan_no_incomplete(layout.hub) + _scan_no_incomplete(layout.xet)
    archive = _require_archive_verification(selected.verify_archive())
    runtime = _require_runtime_verification(selected.verify_runtime())
    values = dict(selected.load_historical_sources())
    historical = _verify_blob_mapping(values)
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.preflight.v1",
            "status": "PASS_READ_ONLY_READY_FOR_EXPLICIT_GENERATE",
            "session": os.fspath(layout.session),
            "source_verification": source,
            "recovery_archive_verification": archive,
            "generation_runtime_verification": runtime,
            "historical_export_verification": historical,
            "cache_entries_scanned_for_incomplete": scanned_entries,
            "incomplete_transfer_objects": 0,
            "generator_executed": False,
            "files_written": False,
            "network_accessed": False,
            "delete_capability_present": False,
        }
    )


def _mkdir_exclusive(root_fd: int, relative: PurePosixPath) -> int:
    parts = phase1._relative_parts(relative, label="private Phase-2 directory")  # noqa: SLF001
    parent = phase1._open_relative_directory(root_fd, parts[:-1], create=True)  # noqa: SLF001
    try:
        try:
            os.mkdir(parts[-1], 0o700, dir_fd=parent)
        except FileExistsError as exc:
            raise Phase2RecoveryError(
                f"refusing to reuse generation directory {relative}"
            ) from exc
        os.fsync(parent)
        child = os.open(
            parts[-1], os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=parent
        )
        metadata = os.fstat(child)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            os.close(child)
            _fail(f"new Phase-2 directory is not private: {relative}")
        return child
    finally:
        os.close(parent)


def _fresh_stage(layout: phase1.SessionLayout) -> Path:
    build_fd = phase1._open_absolute_directory(layout.build)  # noqa: SLF001
    try:
        root_fd = phase1._open_relative_directory(  # noqa: SLF001
            build_fd, PHASE2_ROOT.parts, create=True
        )
    finally:
        os.close(build_fd)
    try:
        name = f"run-{os.getpid()}-{time.monotonic_ns()}"
        run_fd = _mkdir_exclusive(root_fd, PurePosixPath(name))
        os.close(run_fd)
    finally:
        os.close(root_fd)
    stage = layout.build / PHASE2_ROOT / name
    stage_fd = phase1._open_absolute_directory(stage)  # noqa: SLF001
    try:
        for relative in (
            PurePosixPath("export/tools/condense"),
            PurePosixPath("f1"),
            PurePosixPath("doctor"),
            PurePosixPath("home"),
            PurePosixPath("tmp"),
            PurePosixPath("cache"),
            PurePosixPath("cache/hf/hub"),
            PurePosixPath("cache/hf/xet"),
            PurePosixPath("cache/torch"),
            PurePosixPath("cache/mlx"),
            PurePosixPath("pycache"),
        ):
            descriptor = phase1._open_relative_directory(  # noqa: SLF001
                stage_fd, relative.parts, create=True
            )
            os.close(descriptor)
    finally:
        os.close(stage_fd)
    return stage


def _write_export(stage: Path, values: Mapping[str, bytes]) -> dict[str, Any]:
    verified = _verify_blob_mapping(values)
    stage_fd = phase1._open_absolute_directory(stage)  # noqa: SLF001
    try:
        for spec in HISTORICAL_BLOBS:
            phase1._write_new_private_file(  # noqa: SLF001
                stage_fd,
                PurePosixPath("export") / PurePosixPath(spec.relative_path),
                values[spec.relative_path],
            )
    finally:
        os.close(stage_fd)
    for spec in HISTORICAL_BLOBS:
        path = stage / "export" / spec.relative_path
        facts = phase1._hash_regular(  # noqa: SLF001
            path, label=f"private historical export {spec.relative_path}", expected_uid=os.getuid()
        )
        phase1._expect_file_facts(  # noqa: SLF001
            facts,
            label=f"private historical export {spec.relative_path}",
            logical_bytes=spec.logical_bytes,
            sha256=spec.sha256,
            mode=0o600,
            uid=os.getuid(),
        )
    return verified


def _path_exists_nofollow(path: Path) -> bool:
    try:
        with phase1._open_parent(path) as (parent_fd, leaf):  # noqa: SLF001
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _verify_frozen_record(layout: phase1.SessionLayout, spec: FrozenRecordSpec) -> dict[str, Any]:
    path = layout.recovery / spec.relative_path
    raw = phase1._read_regular_bytes(  # noqa: SLF001
        path,
        label=f"frozen recovery record {spec.relative_path}",
        maximum_bytes=2_000_000,
        expected_uid=os.getuid(),
    )
    facts = phase1._hash_regular(  # noqa: SLF001
        path, label=f"frozen recovery record {spec.relative_path}", expected_uid=os.getuid()
    )
    phase1._expect_file_facts(  # noqa: SLF001
        facts,
        label=f"frozen recovery record {spec.relative_path}",
        logical_bytes=spec.logical_bytes,
        sha256=spec.sha256,
        mode=0o600,
        uid=os.getuid(),
    )
    value = phase1.strict_json_bytes(raw, label=f"frozen record {spec.relative_path}")
    phase1.verify_sealed_document(value, label=f"frozen record {spec.relative_path}")
    if value.get("seal_sha256") != spec.seal_sha256:
        _fail(f"frozen recovery record seal changed: {spec.relative_path}")
    return {
        "relative_path": spec.relative_path,
        "logical_bytes": spec.logical_bytes,
        "sha256": spec.sha256,
        "seal_sha256": spec.seal_sha256,
        "mode": "0600",
    }


def _ensure_frozen_records(
    layout: phase1.SessionLayout, hooks: Phase2Hooks
) -> dict[str, Any]:
    missing: set[str] = set()
    for spec in FROZEN_RECORDS:
        if not _path_exists_nofollow(layout.recovery / spec.relative_path):
            missing.add(spec.relative_path)
        else:
            _verify_frozen_record(layout, spec)
    if missing:
        extraction = hooks.extract_records(layout, set(missing))
        phase1.verify_sealed_document(extraction, label="frozen record extraction")
        if extraction.get("status") != "PASS_ALLOWLISTED_TEXT_ONLY":
            _fail("frozen record extraction did not pass")
        rows = extraction.get("extracted")
        if not isinstance(rows, list) or {
            row.get("relative_path") for row in rows if isinstance(row, dict)
        } != missing:
            _fail("frozen record extractor returned the wrong allowlist")
    verified = [_verify_frozen_record(layout, spec) for spec in FROZEN_RECORDS]
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.frozen_records.v1",
            "status": "PASS_EXACT_THREE_SANITIZED_RECORDS",
            "records": verified,
            "idempotent_existing_exact_records_accepted": True,
            "binary_extracted_from_archive": False,
            "network_accessed": False,
        }
    )


def _generation_environment(stage: Path) -> dict[str, str]:
    values = {
        "HOME": os.fspath(stage / "home"),
        "TMPDIR": os.fspath(stage / "tmp"),
        "TMP": os.fspath(stage / "tmp"),
        "TEMP": os.fspath(stage / "tmp"),
        "XDG_CACHE_HOME": os.fspath(stage / "cache"),
        "HF_HOME": os.fspath(stage / "cache" / "hf"),
        "HF_HUB_CACHE": os.fspath(stage / "cache" / "hf" / "hub"),
        "HF_XET_CACHE": os.fspath(stage / "cache" / "hf" / "xet"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TORCH_HOME": os.fspath(stage / "cache" / "torch"),
        "MLX_METAL_CACHE_DIR": os.fspath(stage / "cache" / "mlx"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPYCACHEPREFIX": os.fspath(stage / "pycache"),
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
    }
    if any(marker in key.upper() for key in values for marker in _TOKENISH):
        _fail("generation environment contains a token/secret-bearing key")
    if any("\x00" in key or "\x00" in value for key, value in values.items()):
        _fail("generation environment contains NUL")
    return values


def _python_argv(stage: Path, script: str, arguments: Sequence[str]) -> list[str]:
    condense = stage / "export/tools/condense"
    script_path = condense / script
    return [
        os.fspath(SANDBOX_EXEC),
        "-p",
        SANDBOX_PROFILE,
        os.fspath(PYTHON),
        "-I",
        "-S",
        "-B",
        "-c",
        BOOTSTRAP,
        os.fspath(script_path),
        os.fspath(condense),
        os.fspath(SITE_PACKAGES),
        os.fspath(stage / "pycache"),
        *arguments,
    ]


def _system_run_process(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    pass_fds: Sequence[int],
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        list(argv),
        env=dict(env),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        close_fds=True,
        pass_fds=tuple(pass_fds),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    return subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)


def _run_generator(
    hooks: Phase2Hooks,
    *,
    argv: Sequence[str],
    environment: Mapping[str, str],
    cwd: Path,
    lease_descriptor: int,
    label: str,
) -> dict[str, Any]:
    if list(argv[:3]) != [os.fspath(SANDBOX_EXEC), "-p", SANDBOX_PROFILE]:
        _fail(f"{label} lost its OS network-denial sandbox")
    if any(item in {"checkout", "archive"} for item in argv):
        _fail(f"{label} contains forbidden historical tree operation")
    if any(marker in key.upper() for key in environment for marker in _TOKENISH):
        _fail(f"{label} environment contains credential-shaped key")
    completed = hooks.run_process(
        tuple(argv), env=dict(environment), cwd=cwd, pass_fds=(lease_descriptor,)
    )
    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
        _fail(f"{label} runner did not return byte output")
    if len(stdout) > MAX_CHILD_OUTPUT_BYTES or len(stderr) > MAX_CHILD_OUTPUT_BYTES:
        _fail(f"{label} output exceeded the evidence bound")
    if completed.returncode != 0:
        detail = stderr[:1_000].decode("utf-8", "replace")
        _fail(f"{label} exited {completed.returncode}: {detail}")
    return {
        "label": label,
        "argv": list(argv),
        "argv_sha256": _hash_bytes(phase1.canonical_json(list(argv))),
        "environment_keys": sorted(environment),
        "environment_sha256": _hash_bytes(phase1.canonical_json(dict(environment))),
        "returncode": completed.returncode,
        "stdout_sha256": _hash_bytes(stdout),
        "stderr_sha256": _hash_bytes(stderr),
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "lease_inherited": True,
        "shell": False,
        "stdin": "DEVNULL",
        "start_new_session": True,
        "network_denied_by_os_sandbox": True,
    }


def _verify_private_inventory(directory: Path, expected: frozenset[str], *, label: str) -> None:
    descriptor = phase1._open_absolute_directory(directory)  # noqa: SLF001
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            _fail(f"{label} directory is not private")
        names = set(os.listdir(descriptor))
        if names != set(expected):
            _fail(
                f"{label} inventory changed: missing={sorted(set(expected)-names)} "
                f"extra={sorted(names-set(expected))}"
            )
        for name in sorted(names):
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(named.st_mode)
                or named.st_uid != os.getuid()
                or named.st_nlink != 1
                or stat.S_IMODE(named.st_mode) != 0o600
            ):
                _fail(f"{label} output is not a private single-link file: {name}")
    finally:
        os.close(descriptor)


def _verify_binary(directory: Path, spec: BinarySpec, *, label: str) -> dict[str, Any]:
    path = directory / spec.filename
    facts = phase1._hash_regular(path, label=label, expected_uid=os.getuid())  # noqa: SLF001
    phase1._expect_file_facts(  # noqa: SLF001
        facts,
        label=label,
        logical_bytes=spec.logical_bytes,
        sha256=spec.sha256,
        mode=0o600,
        uid=os.getuid(),
    )
    return {"filename": spec.filename, **facts}


def _strict_self_sealed(path: Path, *, label: str) -> dict[str, Any]:
    raw = phase1._read_regular_bytes(  # noqa: SLF001
        path, label=label, maximum_bytes=2_000_000, expected_uid=os.getuid()
    )
    value = phase1.strict_json_bytes(raw, label=label)
    return phase1.verify_sealed_document(value, label=label)


def _require_subset(value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for key, item in expected.items():
        if value.get(key) != item:
            _fail(f"{label} field {key} changed")


def _verify_bracket_outputs(stage: Path) -> dict[str, Any]:
    directory = stage / "f1"
    _verify_private_inventory(directory, BRACKET_EXPECTED_FILES, label="bracket output")
    binaries = [
        _verify_binary(directory, spec, label=f"bracket binary {spec.filename}")
        for spec in BRACKET_BINARIES
    ]
    capture = _strict_self_sealed(directory / "teacher_capture.json", label="generated capture")
    _require_subset(
        capture,
        {
            "schema": "hawking.kimi_k26.f1_teacher_capture.v1",
            "status": "PASS",
            "revision": phase1.KIMI_REVISION,
            "layer": 1,
            "token_overlap": 0,
            "sentinel_expert": 0,
            "capture_bytes": phase1.TEACHER_CAPTURE_BYTES,
            "capture_sha256": phase1.TEACHER_CAPTURE_SHA256,
        },
        label="generated capture",
    )
    expected_candidates: dict[str, dict[str, Any]] = {
        "P1": {
            "candidate_verdict": "DEGRADED_F1",
            "target_complete_bpw": "49/50",
            "complete_ceiling_bytes": 5_394_923,
            "base_ceiling_bytes": 4_046_192,
            "doctor_ceiling_bytes": 1_240_832,
            "overhead_ceiling_bytes": 107_899,
            "logical_weights_represented": 44_040_192,
            "payload": BRACKET_BINARIES[1],
            "base_component_bytes": 4_022_298,
            "doctor_component_bytes": 1_220_627,
            "header_overhead_bytes": 5_831,
        },
        "P5": {
            "candidate_verdict": "COLLAPSE_F1",
            "target_complete_bpw": "1/2",
            "complete_ceiling_bytes": 2_752_512,
            "base_ceiling_bytes": 1_431_306,
            "doctor_ceiling_bytes": 1_238_630,
            "overhead_ceiling_bytes": 82_576,
            "logical_weights_represented": 44_040_192,
            "payload": BRACKET_BINARIES[2],
            "base_component_bytes": 1_404_937,
            "doctor_component_bytes": 1_220_627,
            "header_overhead_bytes": 4_962,
        },
    }
    results: list[dict[str, Any]] = []
    for candidate, expected in expected_candidates.items():
        result = _strict_self_sealed(
            directory / f"{candidate}_F1_RESULT.json",
            label=f"generated {candidate} bracket result",
        )
        _require_subset(
            result,
            {
                "schema": "hawking.kimi_k26.f1_candidate_result.v1",
                "status": "PASS",
                "candidate": candidate,
                "source": {"repo": phase1.KIMI_REPO, "revision": phase1.KIMI_REVISION},
                "layer": 1,
                "sentinel_expert": 0,
                "candidate_verdict": expected["candidate_verdict"],
            },
            label=f"generated {candidate} bracket result",
        )
        budget = result.get("physical_budget")
        payload = result.get("payload")
        if not isinstance(budget, dict) or not isinstance(payload, dict):
            _fail(f"generated {candidate} bracket accounting is malformed")
        _require_subset(
            budget,
            {
                key: expected[key]
                for key in (
                    "target_complete_bpw",
                    "complete_ceiling_bytes",
                    "base_ceiling_bytes",
                    "doctor_ceiling_bytes",
                    "overhead_ceiling_bytes",
                    "logical_weights_represented",
                )
            }
            | {"all_payload_bytes_counted": True},
            label=f"generated {candidate} bracket budget",
        )
        payload_spec = expected["payload"]
        assert isinstance(payload_spec, BinarySpec)
        _require_subset(
            payload,
            {
                "bytes": payload_spec.logical_bytes,
                "sha256": payload_spec.sha256,
                "base_component_bytes": expected["base_component_bytes"],
                "doctor_component_bytes": expected["doctor_component_bytes"],
                "header_overhead_bytes": expected["header_overhead_bytes"],
            },
            label=f"generated {candidate} bracket payload",
        )
        results.append({"candidate": candidate, "seal_sha256": result["seal_sha256"]})
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.bracket_outputs.v1",
            "status": "PASS_EXACT_BRACKET_BINARIES_AND_SEMANTICS",
            "binaries": binaries,
            "capture_generated_seal_sha256": capture["seal_sha256"],
            "candidate_results": results,
        }
    )


def _verify_doctor_outputs(stage: Path) -> dict[str, Any]:
    directory = stage / "doctor"
    _verify_private_inventory(directory, DOCTOR_EXPECTED_FILES, label="Doctor output")
    binaries = [
        _verify_binary(directory, spec, label=f"Doctor binary {spec.filename}")
        for spec in DOCTOR_BINARIES
    ]
    best = _strict_self_sealed(
        directory / "P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json",
        label="generated best Doctor result",
    )
    _require_subset(
        best,
        {
            "schema": "hawking.kimi_k26.f1_hidden_doctor_result.v1",
            "status": "PASS",
            "candidate": "P1",
            "architecture": "DUAL_PATH_RECOVERY_R16X2",
            "source": {"repo": phase1.KIMI_REPO, "revision": phase1.KIMI_REVISION},
            "layer": 1,
            "sentinel_expert": 0,
            "candidate_verdict": "SURVIVES_F1",
        },
        label="generated best Doctor result",
    )
    payload = best.get("payload")
    budget = best.get("physical_budget")
    if not isinstance(payload, dict) or not isinstance(budget, dict):
        _fail("generated best Doctor accounting is malformed")
    _require_subset(
        payload,
        {
            "bytes": phase1.BEST_PAYLOAD_BYTES,
            "sha256": phase1.BEST_PAYLOAD_SHA256,
            "base_component_bytes": 4_022_298,
            "doctor_component_bytes": 974_848,
            "header_overhead_bytes": 4_669,
        },
        label="generated best Doctor payload",
    )
    _require_subset(
        budget,
        {
            "target_complete_bpw": "49/50",
            "complete_ceiling_bytes": 5_394_923,
            "base_ceiling_bytes": 4_046_192,
            "doctor_ceiling_bytes": 1_240_832,
            "overhead_ceiling_bytes": 107_899,
            "logical_weights_represented": 44_040_192,
            "all_payload_bytes_counted": True,
        },
        label="generated best Doctor budget",
    )
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.doctor_outputs.v1",
            "status": "PASS_ALL_FOUR_EXACT_DOCTOR_BINARIES",
            "binaries": binaries,
            "best_generated_result_seal_sha256": best["seal_sha256"],
            "frozen_result_remains_authoritative": True,
        }
    )


def _capsule_shape(layout: phase1.SessionLayout) -> dict[str, Any]:
    descriptor = phase1._open_absolute_directory(layout.capsule)  # noqa: SLF001
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            _fail("capsule directory must be owned private mode 0700")
        names = set(os.listdir(descriptor))
        expected = {spec.filename for spec in CAPSULE_BINARIES}
        if names != expected:
            _fail(
                f"capsule inventory changed: missing={sorted(expected-names)} "
                f"extra={sorted(names-expected)}"
            )
    finally:
        os.close(descriptor)
    binaries = [
        _verify_binary(layout.capsule, spec, label=f"capsule {spec.filename}")
        for spec in CAPSULE_BINARIES
    ]
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.capsule_shape.v1",
            "status": "PASS_EXACT_TWO_PRIVATE_FILES",
            "capsule": os.fspath(layout.capsule),
            "directory_mode": "0700",
            "binaries": binaries,
            "extra_entries": 0,
        }
    )


def _capsule_partial_state(layout: phase1.SessionLayout) -> dict[str, Any]:
    """Validate a crash-left capsule without accepting it as complete."""
    descriptor = phase1._open_absolute_directory(layout.capsule)  # noqa: SLF001
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            _fail("partial capsule directory must be owned private mode 0700")
        names = set(os.listdir(descriptor))
        expected = {spec.filename for spec in CAPSULE_BINARIES}
        if not names <= expected:
            _fail(f"partial capsule contains unauthorized entries: {sorted(names-expected)}")
    finally:
        os.close(descriptor)
    by_name = {spec.filename: spec for spec in CAPSULE_BINARIES}
    verified = [
        _verify_binary(layout.capsule, by_name[name], label=f"partial capsule {name}")
        for name in sorted(names)
    ]
    return {
        "present": sorted(names),
        "missing": sorted(expected - names),
        "complete": names == expected,
        "verified": verified,
    }


def _stage_capsule(layout: phase1.SessionLayout, stage: Path) -> dict[str, Any]:
    if _path_exists_nofollow(layout.capsule):
        partial = _capsule_partial_state(layout)
        if partial["complete"]:
            _fail("complete capsule unexpectedly reached the installation path")
        capsule_fd = phase1._open_absolute_directory(layout.capsule)  # noqa: SLF001
    else:
        partial = {"present": [], "missing": sorted(spec.filename for spec in CAPSULE_BINARIES)}
        build_fd = phase1._open_absolute_directory(layout.build)  # noqa: SLF001
        try:
            capsule_fd = _mkdir_exclusive(build_fd, PurePosixPath("capsule"))
        finally:
            os.close(build_fd)
    try:
        sources = {
            phase1.BEST_PAYLOAD_BASENAME: stage / "doctor" / phase1.BEST_PAYLOAD_BASENAME,
            phase1.TEACHER_CAPTURE_BASENAME: stage / "f1" / "teacher_capture.npz",
        }
        for spec in CAPSULE_BINARIES:
            if spec.filename in partial["present"]:
                continue
            raw = phase1._read_regular_bytes(  # noqa: SLF001
                sources[spec.filename],
                label=f"verified staging source {spec.filename}",
                maximum_bytes=spec.logical_bytes,
                expected_uid=os.getuid(),
            )
            if len(raw) != spec.logical_bytes or _hash_bytes(raw) != spec.sha256:
                _fail(f"staging source changed before capsule install: {spec.filename}")
            phase1._write_new_private_file(  # noqa: SLF001
                capsule_fd, PurePosixPath(spec.filename), raw
            )
        os.fsync(capsule_fd)
    finally:
        os.close(capsule_fd)
    return _capsule_shape(layout)


def _verify_capsule_authority(
    layout: phase1.SessionLayout, hooks: Phase2Hooks
) -> tuple[dict[str, Any], dict[str, Any]]:
    shape = _capsule_shape(layout)
    verification = hooks.verify_capsule(layout)
    phase1.verify_sealed_document(verification, label="Phase-1 capsule verification")
    if verification.get("schema") != "hawking.kimi_k26.release_cycle.rollback_capsule.v1":
        _fail("Phase-1 capsule verification schema changed")
    if verification.get("status") != "PASS_EXACT_PAYLOAD_RESULT_CAPTURE":
        _fail("Phase-1 capsule verification did not pass")
    return shape, verification


@contextlib.contextmanager
def _live_exclusive_lease(layout: phase1.SessionLayout) -> Iterator[int]:
    """Acquire the exact download-supervisor lease and validate its journal."""
    with download_supervisor._exclusive_lease(layout) as descriptor:  # noqa: SLF001
        raw = download_supervisor._read_evidence_leaf(  # noqa: SLF001
            layout,
            download_supervisor._JOURNAL_NAME,  # noqa: SLF001
            maximum_bytes=download_supervisor._MAX_JOURNAL_BYTES,  # noqa: SLF001
        )
        if raw is None:
            entries: list[dict[str, Any]] = []
        else:
            entries = download_supervisor._verify_journal_bytes(raw)  # noqa: SLF001
        download_supervisor._assert_no_unfinished_live_child(entries)  # noqa: SLF001
        yield descriptor


@dataclass
class _PinnedSource:
    relative_path: str
    target: Path
    expected_content_id: str
    sha256_expected: bool
    descriptor: int
    opened: os.stat_result
    link_target: str
    link_metadata: os.stat_result


def _hash_descriptor(descriptor: int, size: int) -> tuple[str, str]:
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {size}\0".encode("ascii"))  # noqa: S324
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            _fail("pinned source dependency truncated")
        sha.update(chunk)
        git.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        _fail("pinned source dependency grew")
    return sha.hexdigest(), git.hexdigest()


@contextlib.contextmanager
def _live_source_guard(
    layout: phase1.SessionLayout, source_verification: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    rows = source_verification.get("files")
    if not isinstance(rows, list):
        _fail("source verification omitted its file binding rows")
    by_name = {
        row.get("path"): row for row in rows
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    pins: list[_PinnedSource] = []
    try:
        for relative in SOURCE_DEPENDENCIES:
            row = by_name.get(relative)
            if not isinstance(row, dict):
                _fail(f"source verification omitted generator dependency {relative}")
            link_target, link_metadata = phase1._read_relative_symlink(  # noqa: SLF001
                layout.snapshot, relative, label=f"generator source dependency {relative}"
            )
            entry = layout.snapshot / relative
            target = Path(os.path.normpath(os.path.join(entry.parent, link_target)))
            expected_content_id = row.get("content_id")
            if (
                not isinstance(expected_content_id, str)
                or target.parent != layout.blobs
                or target.name != expected_content_id
            ):
                _fail(f"source dependency target changed: {relative}")
            descriptor = os.open(target, os.O_RDONLY | _NOFOLLOW | _CLOEXEC)
            opened = os.fstat(descriptor)
            named = os.stat(target, follow_symlinks=False)
            if not phase1._identity_equal(opened, named):  # noqa: SLF001
                os.close(descriptor)
                _fail(f"source dependency changed while pinning: {relative}")
            pins.append(
                _PinnedSource(
                    relative,
                    target,
                    expected_content_id,
                    bool(row.get("sha256_verified")),
                    descriptor,
                    opened,
                    link_target,
                    link_metadata,
                )
            )
        yield phase1.seal_document(
            {
                "schema": "hawking.kimi_k26.phase2.source_guard.v1",
                "status": "PINNED_OPEN_EXACT_GENERATOR_DEPENDENCIES",
                "dependencies": [pin.relative_path for pin in pins],
                "source_files_copied": False,
                "source_hardlinks_created": False,
            }
        )
        verified: list[dict[str, Any]] = []
        for pin in pins:
            current_link, current_link_metadata = phase1._read_relative_symlink(  # noqa: SLF001
                layout.snapshot,
                pin.relative_path,
                label=f"post-generation source dependency {pin.relative_path}",
            )
            if current_link != pin.link_target or not phase1._identity_equal(  # noqa: SLF001
                pin.link_metadata, current_link_metadata
            ):
                _fail(f"source dependency link changed during generation: {pin.relative_path}")
            opened_post = os.fstat(pin.descriptor)
            named_post = os.stat(pin.target, follow_symlinks=False)
            if not phase1._identity_equal(pin.opened, opened_post) or not phase1._identity_equal(  # noqa: SLF001
                pin.opened, named_post
            ):
                _fail(f"source dependency identity changed during generation: {pin.relative_path}")
            sha, git = _hash_descriptor(pin.descriptor, int(pin.opened.st_size))
            actual = sha if pin.sha256_expected else git
            if actual != pin.expected_content_id:
                _fail(f"source dependency content changed during generation: {pin.relative_path}")
            verified.append(
                {
                    "relative_path": pin.relative_path,
                    "logical_bytes": int(pin.opened.st_size),
                    "content_id": pin.expected_content_id,
                }
            )
    finally:
        for pin in pins:
            os.close(pin.descriptor)


def _receipt_document(
    *,
    layout: phase1.SessionLayout,
    preflight_record: dict[str, Any],
    frozen_records: dict[str, Any],
    capsule_shape: dict[str, Any],
    capsule_verification: dict[str, Any],
) -> dict[str, Any]:
    return phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.recovery.v1",
            "status": "PASS_EXACT_RECOVERED_CAPSULE",
            "session": os.fspath(layout.session),
            "historical_commit": HISTORICAL_COMMIT,
            "preflight_seal_sha256": preflight_record["seal_sha256"],
            "authority": {
                "source_verification": preflight_record["source_verification"],
                "recovery_archive_verification": preflight_record[
                    "recovery_archive_verification"
                ],
                "generation_runtime_verification": preflight_record[
                    "generation_runtime_verification"
                ],
                "historical_export_verification": preflight_record[
                    "historical_export_verification"
                ],
            },
            "source_verification_seal_sha256": preflight_record["source_verification"][
                "seal_sha256"
            ],
            "runtime_verification_seal_sha256": preflight_record[
                "generation_runtime_verification"
            ]["seal_sha256"],
            "historical_export_seal_sha256": preflight_record[
                "historical_export_verification"
            ]["seal_sha256"],
            "frozen_records_seal_sha256": frozen_records["seal_sha256"],
            "capsule_shape": capsule_shape,
            "phase1_capsule_verification": capsule_verification,
            "capsule_files": sorted(spec.filename for spec in CAPSULE_BINARIES),
            "capsule_extra_entries": 0,
            "source_files_copied": False,
            "historical_checkout_used": False,
            "historical_archive_used": False,
            "network_accessed_by_generators": False,
            "delete_capability_present": False,
        }
    )


def _write_receipt(layout: phase1.SessionLayout, value: dict[str, Any]) -> None:
    raw = phase1.canonical_json(value) + b"\n"
    evidence_fd = phase1._open_absolute_directory(layout.evidence)  # noqa: SLF001
    try:
        try:
            phase1._write_new_private_file(evidence_fd, FINAL_RECEIPT, raw)  # noqa: SLF001
        except phase1.ReleaseCycleError as exc:
            if not _path_exists_nofollow(layout.evidence / FINAL_RECEIPT):
                raise Phase2RecoveryError(str(exc)) from exc
            existing = phase1._read_regular_bytes(  # noqa: SLF001
                layout.evidence / FINAL_RECEIPT,
                label="existing Phase-2 receipt",
                maximum_bytes=4_000_000,
                expected_uid=os.getuid(),
            )
            if existing != raw:
                _fail("existing Phase-2 receipt differs; refusing overwrite")
    finally:
        os.close(evidence_fd)


def generate(
    layout: phase1.SessionLayout,
    *,
    hooks: Phase2Hooks | None = None,
    mop_root: Path = phase1.MOP_ROOT,
    shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    """Run the two generators only after the caller explicitly selects this action."""
    selected = hooks or Phase2Hooks.live(mop_root=mop_root, shared_xet=shared_xet)
    initial = preflight(
        layout, hooks=selected, mop_root=mop_root, shared_xet=shared_xet
    )
    previous_umask = os.umask(0o077)
    try:
        with selected.exclusive_lease(layout) as lease_descriptor:
            guarded = preflight(
                layout, hooks=selected, mop_root=mop_root, shared_xet=shared_xet
            )
            if guarded["source_verification"]["seal_sha256"] != initial[
                "source_verification"
            ]["seal_sha256"]:
                _fail("source authority changed before lease acquisition")
            frozen = _ensure_frozen_records(layout, selected)
            if _path_exists_nofollow(layout.capsule):
                partial = _capsule_partial_state(layout)
                if partial["complete"]:
                    shape, verified = _verify_capsule_authority(layout, selected)
                    receipt = _receipt_document(
                        layout=layout,
                        preflight_record=guarded,
                        frozen_records=frozen,
                        capsule_shape=shape,
                        capsule_verification=verified,
                    )
                    _write_receipt(layout, receipt)
                    return receipt
            values = dict(selected.load_historical_sources())
            historical = _verify_blob_mapping(values)
            if historical["seal_sha256"] != guarded[
                "historical_export_verification"
            ]["seal_sha256"]:
                _fail("historical source binding changed under lease")
            stage = _fresh_stage(layout)
            _write_export(stage, values)
            environment = _generation_environment(stage)
            bracket_argv = _python_argv(
                stage,
                "kimi_k26_f1_bracket.py",
                [
                    "--source",
                    os.fspath(layout.snapshot),
                    "--corpus",
                    os.fspath(stage / "export/KIMI_K26_CORPUS_INTEGRITY.json"),
                    "--output-dir",
                    os.fspath(stage / "f1"),
                ],
            )
            doctor_argv = _python_argv(
                stage,
                "kimi_k26_f1_doctor_auction.py",
                [
                    "--source-dir",
                    os.fspath(stage / "f1"),
                    "--output-dir",
                    os.fspath(stage / "doctor"),
                ],
            )
            with selected.source_guard(layout, guarded["source_verification"]):
                _run_generator(
                    selected,
                    argv=bracket_argv,
                    environment=environment,
                    cwd=stage,
                    lease_descriptor=lease_descriptor,
                    label="historical F1 bracket",
                )
                _verify_bracket_outputs(stage)
                _run_generator(
                    selected,
                    argv=doctor_argv,
                    environment=environment,
                    cwd=stage,
                    lease_descriptor=lease_descriptor,
                    label="historical Doctor auction",
                )
                _verify_doctor_outputs(stage)
            _scan_no_incomplete(layout.hub)
            _scan_no_incomplete(layout.xet)
            shape = _stage_capsule(layout, stage)
            final_shape, verified = _verify_capsule_authority(layout, selected)
            if final_shape["seal_sha256"] != shape["seal_sha256"]:
                _fail("capsule changed between installation and final verification")
            receipt = _receipt_document(
                layout=layout,
                preflight_record=guarded,
                frozen_records=frozen,
                capsule_shape=final_shape,
                capsule_verification=verified,
            )
            _write_receipt(layout, receipt)
            return receipt
    finally:
        os.umask(previous_umask)


def verify(
    layout: phase1.SessionLayout,
    *,
    hooks: Phase2Hooks | None = None,
    mop_root: Path = phase1.MOP_ROOT,
    shared_xet: Path = phase1.SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    """Read-only verification of an already installed exact capsule."""
    selected = hooks or Phase2Hooks.live(mop_root=mop_root, shared_xet=shared_xet)
    authority = preflight(
        layout, hooks=selected, mop_root=mop_root, shared_xet=shared_xet
    )
    frozen = phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.phase2.frozen_records.v1",
            "status": "PASS_EXACT_THREE_SANITIZED_RECORDS",
            "records": [_verify_frozen_record(layout, spec) for spec in FROZEN_RECORDS],
            "idempotent_existing_exact_records_accepted": True,
            "binary_extracted_from_archive": False,
            "network_accessed": False,
        }
    )
    shape, capsule = _verify_capsule_authority(layout, selected)
    return _receipt_document(
        layout=layout,
        preflight_record=authority,
        frozen_records=frozen,
        capsule_shape=shape,
        capsule_verification=capsule,
    )


def _layout_from_cli(value: str) -> phase1.SessionLayout:
    session = Path(value)
    return phase1.layout_for(session, parent=phase1.SESSION_PARENT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "generate", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--session", required=True, help="exact dedicated Phase-1 session")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        layout = _layout_from_cli(args.session)
        if args.command == "preflight":
            result = preflight(layout)
        elif args.command == "generate":
            result = generate(layout)
        elif args.command == "verify":
            result = verify(layout)
        else:  # argparse makes this unreachable
            _fail("unsupported Phase-2 command")
        print(json.dumps(result, sort_keys=True))
        return 0
    except (Phase2RecoveryError, phase1.ReleaseCycleError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
