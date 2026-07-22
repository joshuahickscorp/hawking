#!/usr/bin/env python3.12
"""Fail-closed preparation for a Kimi-K2.6 source rehydration/release cycle.

This module implements *phase 1* only.  It can create a private, dedicated
session, describe (but never execute) the immutable Hugging Face download,
verify the source and the sanitized recovery records, inventory exact local
objects, and perform read-only pre-release reader/process/queue audits.

There is intentionally no download, delete, unlink, rename, quarantine, or
release command in this file.  A later phase must consume the sealed evidence
and implement those capabilities under separate authority.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

KIMI_REPO = "moonshotai/Kimi-K2.6"
KIMI_REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
KIMI_FILE_COUNT = 96
KIMI_WEIGHT_SHARDS = 64
KIMI_TOTAL_BYTES = 595_204_999_341
KIMI_WEIGHT_BYTES = 595_177_988_208
KIMI_LARGEST_SHARD_BYTES = 9_809_050_936
KIMI_MANIFEST_SEAL_SHA256 = (
    "22123f6ed9ae2688da383b5de8c671e70071802e6fc1ca0a16df90c607885c60"
)

SESSION_PARENT = Path(
    "/Users/scammermike/Library/Application Support/Hawking/KimiK26ReleaseCycle"
)
MOP_ROOT = Path("/Users/scammermike/Downloads/mop")
SHARED_HF_XET_ROOT = Path("/Users/scammermike/.cache/huggingface/xet")
LEGACY_RUNTIME_ROOT = Path(
    "/Users/scammermike/Library/Application Support/Hawking/KimiK26"
)

OFFICIAL_MANIFEST = (
    REPO_ROOT / "reports/condense/kimi_k26/KIMI_K26_OFFICIAL_MANIFEST.json"
)
SANITIZED_ARCHIVE = (
    REPO_ROOT
    / "reports/condense/kimi_k26/KIMI_K26_RUNTIME_PROGRESS_ARCHIVE.zip"
)
HF_CLI = REPO_ROOT / ".venv/glm52/bin/hf"
TRANSFER_VENV_ROOT = REPO_ROOT / ".venv/glm52"
TRANSFER_VENV_BIN = TRANSFER_VENV_ROOT / "bin"
TRANSFER_PYTHON_LAUNCHER = TRANSFER_VENV_BIN / "python"
TRANSFER_PYTHON_LINK = TRANSFER_VENV_BIN / "python3.12"
TRANSFER_INTERPRETER = Path(
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
)
TRANSFER_SITE_PACKAGES = (
    TRANSFER_VENV_ROOT / "lib/python3.12/site-packages"
)
HF_CLI_SHEBANG = f"#!{TRANSFER_PYTHON_LAUNCHER}\n"
HF_CLI_BYTES = 264
HF_CLI_SHA256 = "67b3b3a09cbf187c9eb32c960f8f816df80b47a06dd05bb3adcea3d50eb39b6d"
PYVENV_CFG_BYTES = 328
PYVENV_CFG_SHA256 = "400022a23a9bfd34ed9581783032b06bd9e2db5a68c9f27421bfe8ebf16086f8"
TRANSFER_INTERPRETER_BYTES = 152_096
TRANSFER_INTERPRETER_SHA256 = (
    "69e6adfac00978215c4e5f3eaf63d262a5fb646b6dca1897b29f78872c4c5a26"
)
HUGGINGFACE_HUB_VERSION = "1.24.0"
HF_XET_VERSION = "1.5.2"
HUGGINGFACE_HUB_RECORD_SHA256 = (
    "a10369736f28efdd11bbec1a26d1cf7f6597fe0fe62aa02a3f9dba9e3b92adb4"
)
HF_XET_RECORD_SHA256 = (
    "9485591f71714b94ee2fd7f15bba48ea803492247395838dc2dd9c00ab4a1bcc"
)
TRANSFER_PTH_FILES = {
    "distutils-precedence.pth": (
        151,
        "2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224",
    )
}
TRANSFER_RUNTIME_ARTIFACTS: dict[str, tuple[int, str, int]] = {
    "huggingface_hub/__init__.py": (
        60_255, "f92d0d3c4e99e2725e910b3f21f5c3633bf7d7894d31b6da0c312109397c0b77", 0o644
    ),
    "huggingface_hub/cli/hf.py": (
        5_208, "1b647e320a36017d3d6918806fed777996cee38d66f720d3c0c0ad4947f589a7", 0o644
    ),
    "huggingface_hub/cli/download.py": (
        9_669, "a3a7cc289a0d40cc429b54505275f76e58a1275ee66a7b7191fd05ebcad9231b", 0o644
    ),
    "huggingface_hub/file_download.py": (
        87_253, "8d2c2be997bd093d2149c020119851c772c5dc146065264ba29b58c28b50c62b", 0o644
    ),
    "huggingface_hub/constants.py": (
        13_182, "788dfe243cba8e41b512e5c08d4755746ebb91b8fa83118d96bf489f6fb900c7", 0o644
    ),
    "huggingface_hub/utils/_xet.py": (
        6_305, "62d13386f5e2db5541e03b6b9ea1a24a1dbf3b1efb5e15732e50c8c1c9d8c63d", 0o644
    ),
    "huggingface_hub-1.24.0.dist-info/METADATA": (
        16_311, "bfd35ce43e19acbde437cd5c36f77aea227524f77db8e904ed378b8fad5c8cc4", 0o644
    ),
    "huggingface_hub-1.24.0.dist-info/RECORD": (
        31_965, HUGGINGFACE_HUB_RECORD_SHA256, 0o644
    ),
    "huggingface_hub-1.24.0.dist-info/WHEEL": (
        91, "2b6eb4118ce7cd7b09601406aa623c553c4476265836f0d9c16f5c061f7efcc0", 0o644
    ),
    "huggingface_hub-1.24.0.dist-info/entry_points.txt": (
        212, "ccfec5fdb052762adc3d0172b07419f45dcb727e7f74248e119c559460ac1b4c", 0o644
    ),
    "hf_xet/__init__.py": (
        107, "13c50377243c82567f9ef7bd84711fdb66cf6a783cf912b1e15b80a5761e414a", 0o644
    ),
    "hf_xet/hf_xet.abi3.so": (
        7_780_000, "33d75018f1ac8680a58ff40668d7ef9375315d41d3b9a0ea08a8cbdd4cd4a0c9", 0o755
    ),
    "hf_xet-1.5.2.dist-info/METADATA": (
        4_882, "56388269c9d1fb62d668bd240aa759ccb0176f52f665137921d4c8dba26c02f6", 0o644
    ),
    "hf_xet-1.5.2.dist-info/RECORD": (
        793, HF_XET_RECORD_SHA256, 0o644
    ),
    "hf_xet-1.5.2.dist-info/WHEEL": (
        103, "0f1e18b10c1ea0c5fe6d116fa3552e31a4234475237a3c6017e7b91f096ee1d5", 0o644
    ),
    "hf_xet-1.5.2.dist-info/sboms/hf_xet.cyclonedx.json": (
        290_351, "7f5a4c60aa06f70c9854e7316ded6b2c41a7241625ce4f17c7aba803cedb8649", 0o644
    ),
}

ACCEPTED_ARCHIVE_BYTES = 897_859
ACCEPTED_ARCHIVE_SHA256 = (
    "f77fecfa45c3f31fbbe12cdcdaa7ccc1975c4f9bbfe5d35c86ec2591dada1dab"
)
ACCEPTED_ARCHIVE_ENTRIES = 101
ACCEPTED_ARCHIVE_UNCOMPRESSED_BYTES = 5_831_435
REJECTED_OLD_ARCHIVE_BYTES = 898_063
REJECTED_OLD_ARCHIVE_SHA256 = (
    "7e48036324cf2cfa307eb839f542e87ba2998f4777c2e39aa589d2b71c4fda7d"
)

BEST_PAYLOAD_BASENAME = "P1_DUAL_PATH_RECOVERY_R16X2.k26f1"
BEST_PAYLOAD_BYTES = 5_001_815
BEST_PAYLOAD_SHA256 = (
    "3546c9b17f720d6d5197c8a8d1dae80e5994e053a808de708aef6bb5e97561bb"
)
BEST_RESULT_RELATIVE = (
    "f1_representation_bracket/doctor_auction/"
    "P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json"
)
BEST_RESULT_SEAL_SHA256 = (
    "dc87eb2c850268f148b80df7c21e4192f69701876cf6f3ad24a708458514a21d"
)
TEACHER_CAPTURE_BASENAME = "P1_TEACHER_CAPTURE.npz"
TEACHER_CAPTURE_BYTES = 6_036_848
TEACHER_CAPTURE_SHA256 = (
    "ffac463e8c28fc7df1490c511ee74c602ac2b0f520856110c63d35a5ac8df981"
)
TEACHER_CAPTURE_RECORD_RELATIVE = "f1_representation_bracket/teacher_capture.json"
TEACHER_CAPTURE_RECORD_SEAL_SHA256 = (
    "a195efcfef2c253c7fc3b908db318e5746e4a4c23fabf0d4008b5d83e941c800"
)
GRAVITY_FINAL_RELATIVE = "KIMI_K26_GRAVITY_FINAL.json"
GRAVITY_FINAL_SEAL_SHA256 = (
    "63e478a1b24da9604b18cb3388fa478bac1b2fa1a24f2953cc601d8aa445823a"
)

# Only these already-sanitized, UTF-8 scientific records may leave the ZIP.
# The exact byte length and hash make the allowlist independent of ZIP metadata.
RECOVERY_ENTRY_ALLOWLIST: dict[str, tuple[int, str]] = {
    "KIMI_K26_ADAPTER_TWIN.json": (
        2624,
        "723546c650bc064d65c876b61e93984dcd9f036261c0c92703cd47b915cb771f",
    ),
    "KIMI_K26_FINAL_BYTE_AUCTION.json": (
        172479,
        "6734fc90817efdb1a8b7f954fdc3bcee905e59d881babfe8503c63cf4b413681",
    ),
    GRAVITY_FINAL_RELATIVE: (
        177249,
        "fbb2fbf0486a7edfe30015d7f1e6d623c776d98dda5c676ef0f2957a92cdf80f",
    ),
    "KIMI_K26_LONG_RUN_FINAL.json": (
        201777,
        "a1d77aad964afefd1830f2b6f54adefd488c77e51c7fa850915c777c2bbc7136",
    ),
    "KIMI_K26_LOGICAL_WEIGHT_LEDGER.json": (
        2826,
        "8025db9206e13ae8fcb4cd54a98681d4a3ea3517a97a3656610f7e5b4b24af55",
    ),
    "KIMI_K26_OFFICIAL_MANIFEST.json": (
        20721,
        "a4584e22df830b040d87e3ce1b3d17fe9e13221bc3849c8f791073d9fa8c07fe",
    ),
    "KIMI_K26_ONE_COPY_RECEIPT.json": (
        1133,
        "d2ee03334d0ceaea494bfb10f7c88e5a937b155ee58b5a5f6013ed9bf38bebde",
    ),
    "KIMI_K26_PARENT_FORWARD_VALIDATION.json": (
        5026,
        "51551f8b237ba37dab96947230736d13b2960d03196d5d2d8be50ea822fd817b",
    ),
    "KIMI_K26_REFERENCE_FORWARD.json": (
        1504,
        "5c3fdb4dc2b08962c006ebea2e43afb83603e8e77ab4f48299177d3fdd7d083b",
    ),
    "KIMI_K26_SOURCE_VERIFICATION.json": (
        757,
        "84de24e651f9e1a17d7b4d86b5142ebc86d37a6ae4987d4e018f80128d958a89",
    ),
    BEST_RESULT_RELATIVE: (
        3736,
        "6823316702811fe185dbe2756778be738b606d6c87f240606cf7f2e0b6e82f37",
    ),
    TEACHER_CAPTURE_RECORD_RELATIVE: (
        1665,
        "039068696c11de87a77f1b32713b794ff36cf5656c2aecd3936d00e3ffea1279",
    ),
    "reference_run/KIMI_K26_PARENT_FORWARD_VALIDATION.json": (
        5026,
        "51551f8b237ba37dab96947230736d13b2960d03196d5d2d8be50ea822fd817b",
    ),
}

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SESSION_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_SHARD = re.compile(r"model-(\d{5})-of-(\d{6})\.safetensors\Z")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class ReleaseCycleError(RuntimeError):
    """Raised when phase-1 evidence cannot be established exactly."""


@dataclass(frozen=True)
class SessionLayout:
    parent: Path
    session: Path
    hub: Path
    xet: Path
    build: Path
    tmp: Path
    hf_home: Path
    recovery: Path
    evidence: Path

    @property
    def model_cache(self) -> Path:
        return self.hub / "models--moonshotai--Kimi-K2.6"

    @property
    def snapshot(self) -> Path:
        return self.model_cache / "snapshots" / KIMI_REVISION

    @property
    def blobs(self) -> Path:
        return self.model_cache / "blobs"

    @property
    def capsule(self) -> Path:
        return self.build / "capsule"


def _fail(message: str) -> None:
    raise ReleaseCycleError(message)


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseCycleError(f"value is not canonical JSON: {exc}") from exc


def seal_document(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("sealed document root must be an object")
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    return {
        **unsigned,
        "seal_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }


def verify_sealed_document(
    value: dict[str, Any], *, label: str = "document"
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} root is not an object")
    recorded = value.get("seal_sha256")
    if not isinstance(recorded, str) or _HEX64.fullmatch(recorded) is None:
        _fail(f"{label} has no valid canonical seal")
    expected = seal_document(value)["seal_sha256"]
    if recorded != expected:
        _fail(f"{label} seal mismatch")
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseCycleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReleaseCycleError(f"non-finite JSON constant {value!r}")


def strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseCycleError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ReleaseCycleError) as exc:
        raise ReleaseCycleError(f"invalid strict JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} JSON root is not an object")
    return value


def _require_absolute_clean(path: Path, *, label: str) -> Path:
    raw = os.fspath(path)
    if "\x00" in raw:
        _fail(f"{label} contains NUL")
    candidate = Path(raw)
    if not candidate.is_absolute():
        _fail(f"{label} must be absolute: {raw!r}")
    if ".." in candidate.parts or "." in candidate.parts:
        _fail(f"{label} contains a traversal component: {raw!r}")
    normalized = Path(os.path.normpath(raw))
    if os.fspath(normalized) != raw.rstrip("/") and raw != "/":
        _fail(f"{label} is not lexically normalized: {raw!r}")
    return normalized


def _relative_parts(relative: PurePosixPath | str, *, label: str) -> tuple[str, ...]:
    raw = str(relative)
    if not raw or "\x00" in raw or "\\" in raw:
        _fail(f"{label} is empty or non-POSIX")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"{label} is not a safe relative POSIX path: {raw!r}")
    if str(pure) != raw:
        _fail(f"{label} is not normalized: {raw!r}")
    return pure.parts


def _within(path: Path, root: Path) -> bool:
    try:
        os.path.commonpath((os.fspath(path), os.fspath(root))) == os.fspath(root)
    except ValueError:
        return False
    return os.path.commonpath((os.fspath(path), os.fspath(root))) == os.fspath(root)


def _require_platform_guards() -> None:
    if not _NOFOLLOW or not _DIRECTORY:
        _fail("O_NOFOLLOW and O_DIRECTORY are required")


def _open_absolute_directory(path: Path) -> int:
    """Open an absolute directory while refusing every symlink component."""
    _require_platform_guards()
    clean = _require_absolute_clean(path, label="directory path")
    descriptor = os.open("/", os.O_RDONLY | _DIRECTORY | _CLOEXEC)
    try:
        for component in clean.parts[1:]:
            child = os.open(
                component,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextlib.contextmanager
def _open_parent(path: Path) -> Iterator[tuple[int, str]]:
    clean = _require_absolute_clean(path, label="file path")
    if clean == Path("/"):
        _fail("root has no leaf")
    descriptor = _open_absolute_directory(clean.parent)
    try:
        yield descriptor, clean.name
    finally:
        os.close(descriptor)


def _identity_equal(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_uid == right.st_uid
        and left.st_nlink == right.st_nlink
        and left.st_size == right.st_size
    )


@contextlib.contextmanager
def _open_regular_fd(
    path: Path,
    *,
    label: str,
    expected_uid: int | None = None,
    require_single_link: bool = True,
) -> Iterator[tuple[int, os.stat_result]]:
    with _open_parent(path) as (parent_fd, leaf):
        try:
            named_pre = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise ReleaseCycleError(f"cannot stat {label}: {exc}") from exc
        if not stat.S_ISREG(named_pre.st_mode):
            _fail(f"{label} is not a no-follow regular file")
        if expected_uid is not None and named_pre.st_uid != expected_uid:
            _fail(f"{label} is not owned by uid {expected_uid}")
        if require_single_link and named_pre.st_nlink != 1:
            _fail(f"{label} has unsafe hard-link count {named_pre.st_nlink}")
        try:
            descriptor = os.open(
                leaf, os.O_RDONLY | _NOFOLLOW | _CLOEXEC, dir_fd=parent_fd
            )
        except OSError as exc:
            raise ReleaseCycleError(f"cannot open {label} without following links: {exc}") from exc
        try:
            opened = os.fstat(descriptor)
            if not _identity_equal(named_pre, opened):
                _fail(f"{label} changed while opening")
            yield descriptor, opened
            named_post = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if not _identity_equal(opened, named_post):
                _fail(f"{label} changed while reading")
        finally:
            os.close(descriptor)


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    expected_uid: int | None = None,
) -> bytes:
    with _open_regular_fd(path, label=label, expected_uid=expected_uid) as (
        descriptor,
        metadata,
    ):
        if metadata.st_size > maximum_bytes:
            _fail(f"{label} exceeds {maximum_bytes} bytes")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"{label} truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"{label} grew while reading")
        return b"".join(chunks)


def _hash_regular(
    path: Path,
    *,
    label: str,
    expected_uid: int | None = None,
) -> dict[str, Any]:
    sha256 = hashlib.sha256()
    with _open_regular_fd(path, label=label, expected_uid=expected_uid) as (
        descriptor,
        metadata,
    ):
        git_sha1 = hashlib.sha1(  # noqa: S324 - official Git blob identity
            f"blob {metadata.st_size}\0".encode("ascii")
        )
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
            git_sha1.update(chunk)
    return {
        "logical_bytes": int(metadata.st_size),
        "allocated_bytes": int(metadata.st_blocks) * 512,
        "sha256": sha256.hexdigest(),
        "git_blob_sha1": git_sha1.hexdigest(),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "hard_links": int(metadata.st_nlink),
        "uid": int(metadata.st_uid),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
    }


def _private_directory_metadata(path: Path, *, exact_mode: bool = True) -> dict[str, Any]:
    descriptor = _open_absolute_directory(path)
    try:
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid != os.getuid():
        _fail(f"private directory is not owned by current uid: {path}")
    if exact_mode and mode != 0o700:
        _fail(f"private directory mode is {mode:o}, expected 700: {path}")
    return {
        "path": os.fspath(path),
        "realpath": os.path.realpath(path),
        "uid": int(metadata.st_uid),
        "mode": f"{mode:04o}",
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
    }


def layout_for(session: Path, *, parent: Path = SESSION_PARENT) -> SessionLayout:
    clean_parent = _require_absolute_clean(parent, label="session parent")
    clean_session = _require_absolute_clean(session, label="session")
    if clean_session.parent != clean_parent or not _SESSION_ID.fullmatch(clean_session.name):
        _fail("session must be one normalized, safe child of the fixed session parent")
    return SessionLayout(
        parent=clean_parent,
        session=clean_session,
        hub=clean_session / "hub",
        xet=clean_session / "xet",
        build=clean_session / "build",
        tmp=clean_session / "build" / "tmp",
        hf_home=clean_session / "build" / "hf-home",
        recovery=clean_session / "recovery",
        evidence=clean_session / "evidence",
    )


def _mkdir_private_leaf(path: Path, *, must_not_exist: bool) -> None:
    with _open_parent(path) as (parent_fd, leaf):
        if must_not_exist:
            try:
                os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                _fail(f"refusing to reuse existing path: {path}")
        try:
            os.mkdir(leaf, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            if must_not_exist:
                _fail(f"refusing to reuse existing path: {path}")
        except OSError as exc:
            raise ReleaseCycleError(f"cannot create private directory {path}: {exc}") from exc
    _private_directory_metadata(path)


def init_session(
    session_id: str, *, parent: Path = SESSION_PARENT
) -> dict[str, Any]:
    """Create only the private session directories; never start a download."""
    if not _SESSION_ID.fullmatch(session_id) or session_id in {".", ".."}:
        _fail("session id must match [a-z0-9][a-z0-9._-]{0,63}")
    clean_parent = _require_absolute_clean(parent, label="session parent")
    if not clean_parent.parent.exists():
        _fail(f"session parent ancestor must already exist: {clean_parent.parent}")
    try:
        _private_directory_metadata(clean_parent)
    except FileNotFoundError:
        _mkdir_private_leaf(clean_parent, must_not_exist=False)
    layout = layout_for(clean_parent / session_id, parent=clean_parent)
    _mkdir_private_leaf(layout.session, must_not_exist=True)
    for path in (layout.hub, layout.xet, layout.build):
        _mkdir_private_leaf(path, must_not_exist=True)
    for path in (layout.tmp, layout.hf_home):
        _mkdir_private_leaf(path, must_not_exist=True)
    for path in (layout.recovery, layout.evidence):
        _mkdir_private_leaf(path, must_not_exist=True)
    validated = validate_layout(layout, mop_root=MOP_ROOT, shared_xet=SHARED_HF_XET_ROOT)
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.session.v1",
            "status": "PRIVATE_SESSION_CREATED_NO_LIVE_ACTION",
            "layout": validated,
            "network_accessed": False,
            "download_executed": False,
            "delete_capability_present": False,
        }
    )


def validate_layout(
    layout: SessionLayout,
    *,
    mop_root: Path = MOP_ROOT,
    shared_xet: Path = SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    expected = layout_for(layout.session, parent=layout.parent)
    if layout != expected:
        _fail("session layout fields are not the exact dedicated layout")
    rows = [
        _private_directory_metadata(path)
        for path in (
            layout.parent,
            layout.session,
            layout.hub,
            layout.xet,
            layout.build,
            layout.tmp,
            layout.hf_home,
            layout.recovery,
            layout.evidence,
        )
    ]
    for row in rows:
        if row["path"] != row["realpath"]:
            _fail(f"session path resolves through a link: {row['path']}")
    session_real = Path(rows[1]["realpath"])
    mop = _require_absolute_clean(mop_root, label="MOP root")
    shared = _require_absolute_clean(shared_xet, label="shared Xet root")
    mop_boundary = _private_directory_metadata(mop, exact_mode=False)
    shared_boundary = _private_directory_metadata(shared, exact_mode=False)
    mop_real = Path(mop_boundary["realpath"])
    shared_real = Path(shared_boundary["realpath"])
    if mop_real != mop:
        _fail("MOP root itself resolves through a symlink")
    if shared_real != shared:
        _fail("shared Xet root itself resolves through a symlink")
    for forbidden, label in ((mop, "MOP"), (mop_real, "resolved MOP"),
                             (shared, "shared Xet"), (shared_real, "resolved shared Xet")):
        if _within(session_real, forbidden) or _within(forbidden, session_real):
            _fail(f"session overlaps {label}: {forbidden}")
    if layout.xet == shared or Path(os.path.realpath(layout.xet)) == shared_real:
        _fail("dedicated Xet directory aliases the shared Hugging Face Xet cache")
    return {
        "parent": os.fspath(layout.parent),
        "session": os.fspath(layout.session),
        "hub": os.fspath(layout.hub),
        "xet": os.fspath(layout.xet),
        "build": os.fspath(layout.build),
        "tmp": os.fspath(layout.tmp),
        "hf_home": os.fspath(layout.hf_home),
        "recovery": os.fspath(layout.recovery),
        "evidence": os.fspath(layout.evidence),
        "directories": rows,
        "mop_root": os.fspath(mop),
        "mop_realpath": os.fspath(mop_real),
        "mop_boundary": mop_boundary,
        "shared_xet_excluded": os.fspath(shared),
        "shared_xet_boundary": shared_boundary,
        "all_private_mode": "0700",
        "owner_uid": os.getuid(),
    }


def _manifest_from_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    value = strict_json_bytes(raw, label=label)
    verify_sealed_document(value, label=label)
    expected_top = {
        "file_count", "files", "largest_shard", "last_modified", "library_name",
        "license_api", "pipeline_tag", "repo", "resolved_at", "schema",
        "seal_sha256", "sha", "total_bytes", "weight_bytes", "weight_shards",
    }
    if set(value) != expected_top:
        _fail(f"{label} top-level shape changed")
    exact = {
        "schema": "hawking.kimi_k26.official_manifest.v1",
        "repo": KIMI_REPO,
        "sha": KIMI_REVISION,
        "file_count": KIMI_FILE_COUNT,
        "weight_shards": KIMI_WEIGHT_SHARDS,
        "total_bytes": KIMI_TOTAL_BYTES,
        "weight_bytes": KIMI_WEIGHT_BYTES,
        "largest_shard": KIMI_LARGEST_SHARD_BYTES,
        "seal_sha256": KIMI_MANIFEST_SEAL_SHA256,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            _fail(f"{label} {key} is not the frozen value")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != KIMI_FILE_COUNT:
        _fail(f"{label} does not contain exactly {KIMI_FILE_COUNT} file rows")
    paths: list[str] = []
    shards: dict[int, dict[str, Any]] = {}
    total = 0
    for index, row in enumerate(files):
        if not isinstance(row, dict) or set(row) != {"blob_id", "path", "sha256", "size"}:
            _fail(f"{label} file row {index} has an unexpected shape")
        path = row.get("path")
        blob_id = row.get("blob_id")
        digest = row.get("sha256")
        size = row.get("size")
        if not isinstance(path, str):
            _fail(f"{label} file row {index} has no path")
        _relative_parts(path, label=f"manifest path {index}")
        if not isinstance(blob_id, str) or _HEX40.fullmatch(blob_id) is None:
            _fail(f"{label} file row {index} has an invalid blob id")
        if digest is not None and (
            not isinstance(digest, str) or _HEX64.fullmatch(digest) is None
        ):
            _fail(f"{label} file row {index} has an invalid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail(f"{label} file row {index} has an invalid size")
        paths.append(path)
        total += size
        match = _SHARD.fullmatch(path)
        if match:
            number, denominator = int(match.group(1)), int(match.group(2))
            if denominator != KIMI_WEIGHT_SHARDS or number in shards:
                _fail(f"{label} has an invalid or duplicate weight shard {path}")
            if digest is None:
                _fail(f"{label} weight shard lacks SHA-256: {path}")
            shards[number] = row
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _fail(f"{label} file paths are not unique and sorted")
    if total != KIMI_TOTAL_BYTES:
        _fail(f"{label} file sizes do not sum to the frozen total")
    if set(shards) != set(range(1, KIMI_WEIGHT_SHARDS + 1)):
        _fail(f"{label} weight shard sequence is incomplete")
    if sum(row["size"] for row in shards.values()) != KIMI_WEIGHT_BYTES:
        _fail(f"{label} weight bytes do not sum to the frozen value")
    if max(row["size"] for row in shards.values()) != KIMI_LARGEST_SHARD_BYTES:
        _fail(f"{label} largest shard changed")
    return value


def verify_manifest(path: Path = OFFICIAL_MANIFEST) -> dict[str, Any]:
    raw = _read_regular_bytes(
        _require_absolute_clean(path, label="manifest path"),
        label="official manifest",
        maximum_bytes=1_000_000,
        expected_uid=os.getuid(),
    )
    value = _manifest_from_bytes(raw, label="official manifest")
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.manifest_verification.v1",
            "status": "PASS",
            "path": os.fspath(path),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest_seal_sha256": value["seal_sha256"],
            "repo": value["repo"],
            "revision": value["sha"],
            "file_count": value["file_count"],
            "weight_shards": value["weight_shards"],
            "total_bytes": value["total_bytes"],
            "weight_bytes": value["weight_bytes"],
        }
    )


def _expect_file_facts(
    facts: dict[str, Any],
    *,
    label: str,
    logical_bytes: int,
    sha256: str,
    mode: int,
    uid: int,
) -> None:
    expected = {
        "logical_bytes": logical_bytes,
        "sha256": sha256,
        "mode": f"{mode:04o}",
        "uid": uid,
        "hard_links": 1,
    }
    for key, value in expected.items():
        if facts.get(key) != value:
            _fail(f"{label} {key} is not the frozen value")


def _metadata_identity(raw: bytes, *, label: str) -> tuple[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseCycleError(f"{label} METADATA is not UTF-8") from exc
    names = [line[6:] for line in text.splitlines() if line.startswith("Name: ")]
    versions = [line[9:] for line in text.splitlines() if line.startswith("Version: ")]
    if len(names) != 1 or len(versions) != 1:
        _fail(f"{label} METADATA has ambiguous name/version")
    return names[0], versions[0]


def _record_target(record_path: str) -> Path:
    if not record_path or "\x00" in record_path or "\\" in record_path \
            or record_path.startswith("/"):
        _fail(f"unsafe distribution RECORD path: {record_path!r}")
    target = Path(os.path.normpath(os.path.join(TRANSFER_SITE_PACKAGES, record_path)))
    if not _within(target, TRANSFER_VENV_ROOT):
        _fail(f"distribution RECORD path escapes transfer venv: {record_path}")
    return target


def _verify_distribution_record(
    *,
    distribution: str,
    version: str,
    dist_info_name: str,
    expected_record_sha256: str,
) -> dict[str, Any]:
    record_path = TRANSFER_SITE_PACKAGES / dist_info_name / "RECORD"
    record_raw = _read_regular_bytes(
        record_path,
        label=f"{distribution} RECORD",
        maximum_bytes=1_000_000,
        expected_uid=os.getuid(),
    )
    if hashlib.sha256(record_raw).hexdigest() != expected_record_sha256:
        _fail(f"{distribution} RECORD hash changed")
    try:
        rows = list(csv.reader(io.StringIO(record_raw.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ReleaseCycleError(f"invalid {distribution} RECORD: {exc}") from exc
    verified: list[dict[str, Any]] = []
    unhashed: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3 or row[0] in seen:
            _fail(f"{distribution} RECORD shape/uniqueness changed")
        path, encoded_hash, encoded_size = row
        seen.add(path)
        target = _record_target(path)
        if not encoded_hash:
            allowed_unhashed = path == f"{dist_info_name}/RECORD" or (
                "/__pycache__/" in path and path.endswith(".pyc")
            )
            if not allowed_unhashed or encoded_size:
                _fail(f"unexpected unhashed {distribution} RECORD row: {path}")
            unhashed.append(path)
            continue
        if not encoded_hash.startswith("sha256=") or not encoded_size.isdigit():
            _fail(f"invalid hash/size in {distribution} RECORD: {path}")
        payload = encoded_hash.removeprefix("sha256=")
        try:
            expected_digest = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).hex()
        except (ValueError, TypeError) as exc:
            raise ReleaseCycleError(
                f"invalid base64 hash in {distribution} RECORD: {path}"
            ) from exc
        facts = _hash_regular(
            target, label=f"{distribution} installed file {path}", expected_uid=os.getuid()
        )
        if facts["logical_bytes"] != int(encoded_size) or facts["sha256"] != expected_digest:
            _fail(f"installed {distribution} file differs from frozen RECORD: {path}")
        verified.append(
            {
                "path": path,
                "logical_bytes": facts["logical_bytes"],
                "sha256": facts["sha256"],
            }
        )
    metadata_rel = f"{dist_info_name}/METADATA"
    metadata_path = TRANSFER_SITE_PACKAGES / metadata_rel
    metadata = _read_regular_bytes(
        metadata_path,
        label=f"{distribution} METADATA",
        maximum_bytes=1_000_000,
        expected_uid=os.getuid(),
    )
    actual_name, actual_version = _metadata_identity(metadata, label=distribution)
    if actual_name != distribution or actual_version != version:
        _fail(f"{distribution} installed name/version changed")
    return {
        "name": actual_name,
        "version": actual_version,
        "dist_info_path": os.fspath(TRANSFER_SITE_PACKAGES / dist_info_name),
        "record_path": os.fspath(record_path),
        "record_sha256": expected_record_sha256,
        "record_row_count": len(rows),
        "verified_hashed_file_count": len(verified),
        "verified_hashed_logical_bytes": sum(row["logical_bytes"] for row in verified),
        "verified_content_manifest_sha256": hashlib.sha256(
            canonical_json(verified)
        ).hexdigest(),
        "unhashed_pyc_rows_ignored_because_pycache_is_redirected": len(unhashed) - 1,
    }


def build_transfer_runtime_binding() -> dict[str, Any]:
    """Ground the exact no-follow venv and native Xet bytes without importing it."""
    cli_raw = _read_regular_bytes(
        HF_CLI,
        label="hf transfer launcher",
        maximum_bytes=10_000,
        expected_uid=os.getuid(),
    )
    cli = _hash_regular(HF_CLI, label="hf transfer launcher", expected_uid=os.getuid())
    _expect_file_facts(
        cli,
        label="hf transfer launcher",
        logical_bytes=HF_CLI_BYTES,
        sha256=HF_CLI_SHA256,
        mode=0o755,
        uid=os.getuid(),
    )
    first_line = cli_raw.splitlines(keepends=True)[0].decode("utf-8", "strict")
    if first_line != HF_CLI_SHEBANG:
        _fail("hf transfer launcher shebang changed")

    launcher_target, launcher_link = _read_relative_symlink(
        TRANSFER_VENV_BIN, "python", label="venv python launcher"
    )
    python_target, python_link = _read_relative_symlink(
        TRANSFER_VENV_BIN, "python3.12", label="venv python3.12 link"
    )
    if launcher_target != "python3.12" or python_target != os.fspath(TRANSFER_INTERPRETER):
        _fail("transfer interpreter symlink chain changed")
    for label, metadata in (
        ("venv python launcher", launcher_link),
        ("venv python3.12 link", python_link),
    ):
        if metadata.st_uid != os.getuid() or metadata.st_nlink != 1 \
                or stat.S_IMODE(metadata.st_mode) != 0o755:
            _fail(f"{label} owner/link/mode changed")

    interpreter = _hash_regular(
        TRANSFER_INTERPRETER,
        label="resolved transfer interpreter",
        expected_uid=0,
    )
    _expect_file_facts(
        interpreter,
        label="resolved transfer interpreter",
        logical_bytes=TRANSFER_INTERPRETER_BYTES,
        sha256=TRANSFER_INTERPRETER_SHA256,
        mode=0o775,
        uid=0,
    )
    pyvenv = _hash_regular(
        TRANSFER_VENV_ROOT / "pyvenv.cfg",
        label="transfer pyvenv.cfg",
        expected_uid=os.getuid(),
    )
    _expect_file_facts(
        pyvenv,
        label="transfer pyvenv.cfg",
        logical_bytes=PYVENV_CFG_BYTES,
        sha256=PYVENV_CFG_SHA256,
        mode=0o644,
        uid=os.getuid(),
    )

    site_fd = _open_absolute_directory(TRANSFER_SITE_PACKAGES)
    try:
        pth_names = sorted(name for name in os.listdir(site_fd) if name.endswith(".pth"))
        dist_infos = sorted(name for name in os.listdir(site_fd) if name.endswith(".dist-info"))
        for forbidden in ("sitecustomize.py", "usercustomize.py"):
            try:
                os.stat(forbidden, dir_fd=site_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            _fail(f"unbound Python startup customization is present: {forbidden}")
    finally:
        os.close(site_fd)
    if pth_names != sorted(TRANSFER_PTH_FILES):
        _fail("transfer venv .pth startup file set changed")
    required_dist_infos = {
        f"huggingface_hub-{HUGGINGFACE_HUB_VERSION}.dist-info",
        f"hf_xet-{HF_XET_VERSION}.dist-info",
    }
    observed_target_dist_infos = {
        name for name in dist_infos
        if name.startswith("huggingface_hub-") or name.startswith("hf_xet-")
    }
    if observed_target_dist_infos != required_dist_infos:
        _fail("transfer venv contains a substituted/ambiguous hub or Xet distribution")
    pth_rows: list[dict[str, Any]] = []
    for name, (expected_bytes, expected_sha) in sorted(TRANSFER_PTH_FILES.items()):
        facts = _hash_regular(
            TRANSFER_SITE_PACKAGES / name,
            label=f"transfer startup file {name}",
            expected_uid=os.getuid(),
        )
        _expect_file_facts(
            facts,
            label=f"transfer startup file {name}",
            logical_bytes=expected_bytes,
            sha256=expected_sha,
            mode=0o644,
            uid=os.getuid(),
        )
        pth_rows.append({"path": name, **facts})

    distributions = {
        "huggingface_hub": _verify_distribution_record(
            distribution="huggingface_hub",
            version=HUGGINGFACE_HUB_VERSION,
            dist_info_name=f"huggingface_hub-{HUGGINGFACE_HUB_VERSION}.dist-info",
            expected_record_sha256=HUGGINGFACE_HUB_RECORD_SHA256,
        ),
        "hf_xet": _verify_distribution_record(
            distribution="hf-xet",
            version=HF_XET_VERSION,
            dist_info_name=f"hf_xet-{HF_XET_VERSION}.dist-info",
            expected_record_sha256=HF_XET_RECORD_SHA256,
        ),
    }
    artifacts: list[dict[str, Any]] = []
    for relative, (expected_bytes, expected_sha, expected_mode) in sorted(
        TRANSFER_RUNTIME_ARTIFACTS.items()
    ):
        facts = _hash_regular(
            TRANSFER_SITE_PACKAGES / relative,
            label=f"transfer runtime artifact {relative}",
            expected_uid=os.getuid(),
        )
        _expect_file_facts(
            facts,
            label=f"transfer runtime artifact {relative}",
            logical_bytes=expected_bytes,
            sha256=expected_sha,
            mode=expected_mode,
            uid=os.getuid(),
        )
        artifacts.append({"relative_path": relative, **facts})
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.transfer_runtime.v1",
            "status": "PASS_EXACT_NOFOLLOW_RUNTIME",
            "venv_root": os.fspath(TRANSFER_VENV_ROOT),
            "cli": {
                "path": os.fspath(HF_CLI),
                "exact_shebang": HF_CLI_SHEBANG.rstrip("\n"),
                **cli,
            },
            "interpreter_chain": [
                {
                    "path": os.fspath(TRANSFER_PYTHON_LAUNCHER),
                    "type": "symlink",
                    "target": launcher_target,
                    "uid": int(launcher_link.st_uid),
                    "hard_links": int(launcher_link.st_nlink),
                    "mode": f"{stat.S_IMODE(launcher_link.st_mode):04o}",
                },
                {
                    "path": os.fspath(TRANSFER_PYTHON_LINK),
                    "type": "symlink",
                    "target": python_target,
                    "uid": int(python_link.st_uid),
                    "hard_links": int(python_link.st_nlink),
                    "mode": f"{stat.S_IMODE(python_link.st_mode):04o}",
                },
            ],
            "resolved_interpreter": {
                "path": os.fspath(TRANSFER_INTERPRETER),
                **interpreter,
            },
            "pyvenv_cfg": {
                "path": os.fspath(TRANSFER_VENV_ROOT / "pyvenv.cfg"),
                **pyvenv,
            },
            "site_packages": os.fspath(TRANSFER_SITE_PACKAGES),
            "startup_pth_files": pth_rows,
            "sitecustomize_absent": True,
            "usercustomize_absent": True,
            "distributions": distributions,
            "relevant_dist_info_and_native_artifacts": artifacts,
            "runtime_imported_by_verifier": False,
            "subprocess_used_by_verifier": False,
            "network_accessed_by_verifier": False,
        }
    )


def verify_transfer_runtime_binding(value: dict[str, Any]) -> dict[str, Any]:
    verify_sealed_document(value, label="Kimi transfer runtime binding")
    expected = build_transfer_runtime_binding()
    if canonical_json(value) != canonical_json(expected):
        _fail("Kimi transfer runtime binding is not the exact deterministic runtime")
    return value


def build_download_plan(
    layout: SessionLayout,
    *,
    manifest_path: Path = OFFICIAL_MANIFEST,
    mop_root: Path = MOP_ROOT,
    shared_xet: Path = SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    layout_evidence = validate_layout(layout, mop_root=mop_root, shared_xet=shared_xet)
    manifest = verify_manifest(manifest_path)
    runtime = build_transfer_runtime_binding()
    environment = {
        "HF_HOME": os.fspath(layout.hf_home),
        "HF_HUB_CACHE": os.fspath(layout.hub),
        "HF_XET_CACHE": os.fspath(layout.xet),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "HF_HUB_OFFLINE": "0",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "8",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": os.fspath(layout.tmp / "pycache"),
        "PYTHONSAFEPATH": "1",
        "TEMP": os.fspath(layout.tmp),
        "TMP": os.fspath(layout.tmp),
        "TMPDIR": os.fspath(layout.tmp),
    }
    if Path(environment["HF_XET_CACHE"]) == shared_xet:
        _fail("download plan selected the shared Hugging Face Xet cache")
    command = [
        os.fspath(HF_CLI),
        "download",
        KIMI_REPO,
        "--revision",
        KIMI_REVISION,
        "--repo-type",
        "model",
        "--cache-dir",
        os.fspath(layout.hub),
        "--max-workers",
        "8",
    ]
    restart_16_command = list(command)
    restart_16_command[-1] = "16"
    restart_16_environment = dict(environment)
    restart_16_environment["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] = "16"
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.download_plan.v1",
            "status": "PLANNED_NOT_EXECUTED",
            "repo": KIMI_REPO,
            "revision": KIMI_REVISION,
            "command_argv": command,
            "environment_mode": "REPLACE_NOT_MERGE",
            "environment": environment,
            "transfer_runtime": runtime,
            "transfer_runtime_seal_sha256": runtime["seal_sha256"],
            "expected_snapshot": os.fspath(layout.snapshot),
            "transfer_profile": {
                "hardware_target": "OBSERVED_10_GBIT_ETHERNET_LINK",
                "end_to_end_throughput_status": "NOT_YET_MEASURED",
                "maximum_file_download_workers": 8,
                "xet_high_performance": True,
                "xet_chunk_cache_size_bytes": 0,
                "xet_data_max_concurrent_file_downloads": 8,
                "reason": (
                    "eight simultaneous approximately 9 GB shard downloads plus "
                    "Xet internal range concurrency, without a second chunk-cache copy"
                ),
                "live_measurement_and_ramp_authority": "LIVE_SUPERVISOR_ONLY",
                "phase1_claims_saturation": False,
            },
            "restart_profiles": [
                {
                    "profile_id": "PRIMARY_8",
                    "command_argv": command,
                    "environment": environment,
                    "activation": "INITIAL_TARGET",
                },
                {
                    "profile_id": "CONDITIONAL_RESTART_16",
                    "command_argv": restart_16_command,
                    "environment": restart_16_environment,
                    "activation": "LIVE_SUPERVISOR_ONLY_AFTER_SUSTAINED_LOW_MEASURED_TRANSFER",
                    "prior_transfer_process_must_be_fully_exited": True,
                    "concurrent_with_primary_forbidden": True,
                    "same_hub_xet_tmp_hf_home_required": True,
                    "new_source_copy_forbidden": True,
                },
            ],
            "temporary_directory_policy": {
                "exact_tmpdir": os.fspath(layout.tmp),
                "tmp_and_temp_alias_exact_tmpdir": True,
                "directory_precreated_uid_owned_mode_0700": True,
                "fallback_to_system_or_shared_tmp_forbidden": True,
                "pycache_redirect": os.fspath(layout.tmp / "pycache"),
            },
            "one_copy_law": {
                "source_payload_location": os.fspath(layout.hub),
                "snapshot_view": os.fspath(layout.snapshot),
                "snapshot_uses_content_addressed_blob_links": True,
                "local_dir_copy_forbidden": True,
                "xet_chunk_cache_disabled": True,
                "shared_xet_forbidden": os.fspath(shared_xet),
                "build_directory_may_contain_source_copy": False,
                "recovery_directory_may_contain_source_copy": False,
            },
            "manifest_verification_seal_sha256": manifest["seal_sha256"],
            "layout": layout_evidence,
            "shared_xet_explicitly_excluded": os.fspath(shared_xet),
            "network_accessed": False,
            "download_executed": False,
            "executor_present_in_this_phase": False,
        }
    )


def verify_download_plan(
    value: dict[str, Any],
    layout: SessionLayout,
    *,
    manifest_path: Path = OFFICIAL_MANIFEST,
    mop_root: Path = MOP_ROOT,
    shared_xet: Path = SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    """Reject even validly resealed substitutions of a transfer-control pin."""
    verify_sealed_document(value, label="Kimi download plan")
    expected = build_download_plan(
        layout,
        manifest_path=manifest_path,
        mop_root=mop_root,
        shared_xet=shared_xet,
    )
    if canonical_json(value) != canonical_json(expected):
        _fail("Kimi download plan is not the exact deterministic frozen plan")
    return value


def _safe_zip_name(name: str) -> PurePosixPath:
    parts = _relative_parts(name, label="ZIP member")
    if name.endswith("/"):
        _fail(f"ZIP directory members are not accepted: {name!r}")
    return PurePosixPath(*parts)


def _archive_bytes(path: Path) -> bytes:
    clean = _require_absolute_clean(path, label="archive path")
    raw = _read_regular_bytes(
        clean,
        label="sanitized recovery archive",
        maximum_bytes=2_000_000,
        expected_uid=os.getuid(),
    )
    if len(raw) == REJECTED_OLD_ARCHIVE_BYTES:
        _fail("the rejected 898,063-byte historical archive is never accepted")
    digest = hashlib.sha256(raw).hexdigest()
    if digest == REJECTED_OLD_ARCHIVE_SHA256:
        _fail("the historical credential-bearing archive is never accepted")
    if len(raw) != ACCEPTED_ARCHIVE_BYTES or digest != ACCEPTED_ARCHIVE_SHA256:
        _fail("archive is not the one frozen sanitized f77fecfa... object")
    return raw


def _verify_archive_raw(raw: bytes) -> dict[str, Any]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ReleaseCycleError(f"sanitized archive is not a valid ZIP: {exc}") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(infos) != ACCEPTED_ARCHIVE_ENTRIES or len(names) != len(set(names)):
            _fail("sanitized archive entry count or uniqueness changed")
        if sum(info.file_size for info in infos) != ACCEPTED_ARCHIVE_UNCOMPRESSED_BYTES:
            _fail("sanitized archive uncompressed byte total changed")
        for info in infos:
            _safe_zip_name(info.filename)
            if info.flag_bits & 0x1:
                _fail(f"encrypted ZIP member is forbidden: {info.filename}")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and stat.S_IFMT(unix_mode) not in {0, stat.S_IFREG}:
                _fail(f"non-regular ZIP member is forbidden: {info.filename}")
            if info.file_size > 2_000_000:
                _fail(f"oversized ZIP member is forbidden: {info.filename}")
        if ".telegram_creds.json" in names:
            _fail("credential-bearing archive member remains present")
        corrupt = archive.testzip()
        if corrupt is not None:
            _fail(f"ZIP integrity failed at {corrupt}")
        for name, (expected_bytes, expected_sha) in RECOVERY_ENTRY_ALLOWLIST.items():
            if name not in names:
                _fail(f"allowlisted recovery member is absent: {name}")
            value = archive.read(name)
            if len(value) != expected_bytes or hashlib.sha256(value).hexdigest() != expected_sha:
                _fail(f"allowlisted recovery member changed: {name}")
            if b"\x00" in value:
                _fail(f"allowlisted recovery member is not text: {name}")
            try:
                value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ReleaseCycleError(
                    f"allowlisted recovery member is not UTF-8: {name}"
                ) from exc
    return {
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_bytes": len(raw),
        "entry_count": len(infos),
        "uncompressed_bytes": sum(info.file_size for info in infos),
        "allowlisted_text_entries": sorted(RECOVERY_ENTRY_ALLOWLIST),
        "credential_entry_present": False,
    }


def verify_recovery_archive(path: Path = SANITIZED_ARCHIVE) -> dict[str, Any]:
    raw = _archive_bytes(path)
    facts = _verify_archive_raw(raw)
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.recovery_archive.v1",
            "status": "PASS_SANITIZED_EXACT_ARCHIVE",
            "path": os.fspath(path),
            **facts,
            "old_898063_byte_archive_rejected": True,
        }
    )


def _open_relative_directory(
    root_fd: int,
    parts: Sequence[str],
    *,
    create: bool,
    require_private: bool = True,
) -> int:
    descriptor = os.dup(root_fd)
    try:
        for component in parts:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(
                    component,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                    dir_fd=descriptor,
                )
            metadata = os.fstat(child)
            if metadata.st_uid != os.getuid() or (
                require_private and stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                os.close(child)
                _fail(f"recovery subdirectory is not private: {component}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_relative_symlink(
    root: Path, relative: PurePosixPath | str, *, label: str
) -> tuple[str, os.stat_result]:
    parts = _relative_parts(relative, label=label)
    root_fd = _open_absolute_directory(root)
    try:
        parent_fd = _open_relative_directory(
            root_fd, parts[:-1], create=False, require_private=False
        )
    finally:
        os.close(root_fd)
    try:
        named_pre = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISLNK(named_pre.st_mode):
            _fail(f"{label} is not a symlink")
        target = os.readlink(parts[-1], dir_fd=parent_fd)
        named_post = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if not _identity_equal(named_pre, named_post):
            _fail(f"{label} changed while reading its link target")
        return target, named_post
    finally:
        os.close(parent_fd)


def _write_new_private_file(root_fd: int, relative: PurePosixPath, raw: bytes) -> None:
    parts = _relative_parts(relative, label="recovery output")
    parent_fd = _open_relative_directory(root_fd, parts[:-1], create=True)
    temporary_fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC
        try:
            temporary_fd = os.open(parts[-1], flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            _fail(f"recovery extraction refuses to overwrite {relative}")
        opened = os.fstat(temporary_fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid() \
                or opened.st_nlink != 1 or stat.S_IMODE(opened.st_mode) != 0o600:
            _fail(f"new recovery output is not a private single-link file: {relative}")
        view = memoryview(raw)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                _fail(f"short write extracting {relative}")
            view = view[written:]
        os.fsync(temporary_fd)
        opened_post = os.fstat(temporary_fd)
        named = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if not _identity_equal(opened_post, named):
            _fail(f"recovery output changed during extraction: {relative}")
        os.fsync(parent_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        os.close(parent_fd)


def extract_sanitized_recovery(
    layout: SessionLayout,
    *,
    archive_path: Path = SANITIZED_ARCHIVE,
    entries: Iterable[str] | None = None,
    mop_root: Path = MOP_ROOT,
    shared_xet: Path = SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    """Extract only frozen UTF-8 records, with no ``ZipFile.extract`` use."""
    validate_layout(layout, mop_root=mop_root, shared_xet=shared_xet)
    requested = sorted(RECOVERY_ENTRY_ALLOWLIST if entries is None else set(entries))
    if not requested or any(name not in RECOVERY_ENTRY_ALLOWLIST for name in requested):
        _fail("recovery extraction request is outside the exact text allowlist")
    raw = _archive_bytes(archive_path)
    archive_facts = _verify_archive_raw(raw)
    extracted: list[dict[str, Any]] = []
    recovery_fd = _open_absolute_directory(layout.recovery)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for name in requested:
                relative = _safe_zip_name(name)
                value = archive.read(name)
                if name.endswith(".json"):
                    strict_json_bytes(value, label=f"recovery member {name}")
                elif name.endswith(".jsonl"):
                    for line_number, line in enumerate(value.splitlines(), 1):
                        strict_json_bytes(line, label=f"recovery member {name}:{line_number}")
                _write_new_private_file(recovery_fd, relative, value)
                extracted.append(
                    {
                        "relative_path": name,
                        "logical_bytes": len(value),
                        "sha256": hashlib.sha256(value).hexdigest(),
                        "mode": "0600",
                    }
                )
    finally:
        os.close(recovery_fd)
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.recovery_extraction.v1",
            "status": "PASS_ALLOWLISTED_TEXT_ONLY",
            "session": os.fspath(layout.session),
            "archive_sha256": archive_facts["archive_sha256"],
            "extracted": extracted,
            "binary_payload_extracted_from_archive": False,
            "network_accessed": False,
        }
    )


def _require_exact_sealed_record(
    path: Path,
    *,
    label: str,
    expected_seal: str,
    maximum_bytes: int = 2_000_000,
) -> dict[str, Any]:
    raw = _read_regular_bytes(
        path, label=label, maximum_bytes=maximum_bytes, expected_uid=os.getuid()
    )
    value = strict_json_bytes(raw, label=label)
    verify_sealed_document(value, label=label)
    if value["seal_sha256"] != expected_seal:
        _fail(f"{label} is not the frozen record")
    return value


def verify_payload_result_capture(
    layout: SessionLayout,
    *,
    payload_path: Path | None = None,
    capture_path: Path | None = None,
    mop_root: Path = MOP_ROOT,
    shared_xet: Path = SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    validate_layout(layout, mop_root=mop_root, shared_xet=shared_xet)
    expected_payload = layout.capsule / BEST_PAYLOAD_BASENAME
    expected_capture = layout.capsule / TEACHER_CAPTURE_BASENAME
    payload = expected_payload if payload_path is None else _require_absolute_clean(
        payload_path, label="payload path"
    )
    capture = expected_capture if capture_path is None else _require_absolute_clean(
        capture_path, label="capture path"
    )
    if payload != expected_payload or capture != expected_capture:
        _fail("payload and capture must occupy their exact dedicated capsule paths")
    if _within(payload, mop_root) or _within(capture, mop_root):
        _fail("payload or capture overlaps MOP")
    payload_facts = _hash_regular(payload, label="best P1 payload", expected_uid=os.getuid())
    capture_facts = _hash_regular(capture, label="P1 teacher capture", expected_uid=os.getuid())
    if payload_facts["logical_bytes"] != BEST_PAYLOAD_BYTES \
            or payload_facts["sha256"] != BEST_PAYLOAD_SHA256:
        _fail("best P1 payload size or hash changed")
    if capture_facts["logical_bytes"] != TEACHER_CAPTURE_BYTES \
            or capture_facts["sha256"] != TEACHER_CAPTURE_SHA256:
        _fail("P1 teacher capture size or hash changed")
    result = _require_exact_sealed_record(
        layout.recovery / BEST_RESULT_RELATIVE,
        label="best P1 result",
        expected_seal=BEST_RESULT_SEAL_SHA256,
    )
    capture_record = _require_exact_sealed_record(
        layout.recovery / TEACHER_CAPTURE_RECORD_RELATIVE,
        label="teacher capture record",
        expected_seal=TEACHER_CAPTURE_RECORD_SEAL_SHA256,
    )
    final = _require_exact_sealed_record(
        layout.recovery / GRAVITY_FINAL_RELATIVE,
        label="Kimi Gravity final",
        expected_seal=GRAVITY_FINAL_SEAL_SHA256,
    )
    source = result.get("source")
    result_payload = result.get("payload")
    best = final.get("best_deployable_candidate")
    if source != {"repo": KIMI_REPO, "revision": KIMI_REVISION}:
        _fail("best P1 result source binding changed")
    if result.get("status") != "PASS" or result.get("candidate") != "P1":
        _fail("best P1 result status/candidate changed")
    if not isinstance(result_payload, dict) or any(
        result_payload.get(key) != expected
        for key, expected in (
            ("bytes", BEST_PAYLOAD_BYTES),
            ("sha256", BEST_PAYLOAD_SHA256),
            ("base_component_bytes", 4_022_298),
            ("doctor_component_bytes", 974_848),
            ("header_overhead_bytes", 4_669),
        )
    ):
        _fail("best P1 result payload accounting changed")
    if capture_record.get("revision") != KIMI_REVISION \
            or capture_record.get("capture_bytes") != TEACHER_CAPTURE_BYTES \
            or capture_record.get("capture_sha256") != TEACHER_CAPTURE_SHA256:
        _fail("teacher capture record binding changed")
    if result.get("teacher_capture_seal_sha256") not in {None, TEACHER_CAPTURE_RECORD_SEAL_SHA256}:
        _fail("best P1 result points at a substituted capture record")
    if final.get("status") != "CLOSED" or final.get("terminal_outcome") != "OUTCOME_C":
        _fail("Kimi Gravity final outcome changed")
    if not isinstance(best, dict) or any(
        best.get(key) != expected
        for key, expected in (
            ("candidate", "P1_DUAL_PATH_RECOVERY_R16X2"),
            ("complete_physical_bytes", BEST_PAYLOAD_BYTES),
            ("payload_sha256", BEST_PAYLOAD_SHA256),
        )
    ):
        _fail("Kimi Gravity final no longer selects the exact best payload")
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.rollback_capsule.v1",
            "status": "PASS_EXACT_PAYLOAD_RESULT_CAPTURE",
            "session": os.fspath(layout.session),
            "payload": {"path": os.fspath(payload), **payload_facts},
            "result": {
                "path": os.fspath(layout.recovery / BEST_RESULT_RELATIVE),
                "seal_sha256": result["seal_sha256"],
            },
            "capture": {"path": os.fspath(capture), **capture_facts},
            "capture_record": {
                "path": os.fspath(layout.recovery / TEACHER_CAPTURE_RECORD_RELATIVE),
                "seal_sha256": capture_record["seal_sha256"],
            },
            "gravity_final_seal_sha256": final["seal_sha256"],
            "legacy_paths_are_metadata_only": True,
            "mop_touched": False,
        }
    )


def _list_tree_files(root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Return files/symlinks and directory metadata using descriptor recursion."""
    files: list[str] = []
    directories: list[dict[str, Any]] = []
    root_fd = _open_absolute_directory(root)

    def walk(descriptor: int, prefix: PurePosixPath) -> None:
        metadata = os.fstat(descriptor)
        directories.append(
            {
                "relative_path": str(prefix) if str(prefix) != "." else "",
                "uid": int(metadata.st_uid),
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            }
        )
        for name in sorted(os.listdir(descriptor)):
            if name in {".", ".."} or "/" in name or "\x00" in name:
                _fail("unsafe directory entry encountered")
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = prefix / name
            if stat.S_ISDIR(named.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                    dir_fd=descriptor,
                )
                try:
                    if not _identity_equal(named, os.fstat(child)):
                        _fail(f"source directory changed while opening: {relative}")
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(named.st_mode) or stat.S_ISLNK(named.st_mode):
                files.append(str(relative))
            else:
                _fail(f"special source entry is forbidden: {relative}")

    try:
        walk(root_fd, PurePosixPath())
    finally:
        os.close(root_fd)
    return sorted(files), directories


def verify_source(
    layout: SessionLayout,
    *,
    manifest_path: Path = OFFICIAL_MANIFEST,
    mop_root: Path = MOP_ROOT,
    shared_xet: Path = SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    validate_layout(layout, mop_root=mop_root, shared_xet=shared_xet)
    raw_manifest = _read_regular_bytes(
        manifest_path,
        label="official manifest",
        maximum_bytes=1_000_000,
        expected_uid=os.getuid(),
    )
    manifest = _manifest_from_bytes(raw_manifest, label="official manifest")
    snapshot = layout.snapshot
    if os.path.realpath(snapshot) != os.fspath(snapshot):
        _fail("snapshot root itself resolves through a symlink")
    actual, directories = _list_tree_files(snapshot)
    expected_rows = {row["path"]: row for row in manifest["files"]}
    if actual != sorted(expected_rows):
        missing = sorted(set(expected_rows) - set(actual))
        extra = sorted(set(actual) - set(expected_rows))
        _fail(f"snapshot inventory mismatch: missing={missing[:3]} extra={extra[:3]}")
    for row in directories:
        if row["uid"] != os.getuid():
            _fail(f"snapshot directory ownership changed: {row['relative_path']}")
    verified: list[dict[str, Any]] = []
    unique_blobs: dict[tuple[int, int], dict[str, Any]] = {}
    snapshot_link_allocated = 0
    for relative in actual:
        row = expected_rows[relative]
        entry = snapshot.joinpath(*_relative_parts(relative, label="snapshot entry"))
        link, named = _read_relative_symlink(
            snapshot, relative, label=f"snapshot entry {relative}"
        )
        if named.st_uid != os.getuid():
            _fail(f"snapshot symlink has wrong owner: {relative}")
        snapshot_link_allocated += int(named.st_blocks) * 512
        if os.path.isabs(link) or "\x00" in link:
            _fail(f"snapshot symlink is absolute or invalid: {relative}")
        target = Path(os.path.normpath(os.path.join(entry.parent, link)))
        if not _within(target, layout.blobs) or target.parent != layout.blobs:
            _fail(f"snapshot symlink escapes the dedicated blob directory: {relative}")
        expected_content_id = row["sha256"] or row["blob_id"]
        if target.name != expected_content_id:
            _fail(f"snapshot symlink points at a substituted blob: {relative}")
        facts = _hash_regular(target, label=f"source blob for {relative}", expected_uid=os.getuid())
        if facts["logical_bytes"] != row["size"]:
            _fail(f"source blob size mismatch: {relative}")
        if row["sha256"] is not None:
            if facts["sha256"] != row["sha256"]:
                _fail(f"source blob SHA-256 mismatch: {relative}")
        elif facts["git_blob_sha1"] != row["blob_id"]:
            _fail(f"source Git blob identity mismatch: {relative}")
        key = (facts["device"], facts["inode"])
        if key in unique_blobs and unique_blobs[key]["content_id"] != expected_content_id:
            _fail("different manifest identities alias one physical blob")
        unique_blobs[key] = {
            "content_id": expected_content_id,
            "logical_bytes": facts["logical_bytes"],
            "allocated_bytes": facts["allocated_bytes"],
        }
        verified.append(
            {
                "path": relative,
                "content_id": expected_content_id,
                "logical_bytes": facts["logical_bytes"],
                "allocated_bytes": facts["allocated_bytes"],
                "sha256_verified": row["sha256"] is not None,
                "git_blob_sha1_verified": row["sha256"] is None,
            }
        )
    if sum(row["logical_bytes"] for row in verified) != KIMI_TOTAL_BYTES:
        _fail("verified snapshot logical byte total changed")
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.source_verification.v1",
            "status": "PASS_EXACT_IMMUTABLE_SOURCE",
            "repo": KIMI_REPO,
            "revision": KIMI_REVISION,
            "snapshot": os.fspath(snapshot),
            "manifest_seal_sha256": manifest["seal_sha256"],
            "file_count": len(verified),
            "weight_shards": KIMI_WEIGHT_SHARDS,
            "logical_bytes": sum(row["logical_bytes"] for row in verified),
            "weight_bytes": KIMI_WEIGHT_BYTES,
            "snapshot_symlink_allocated_bytes": snapshot_link_allocated,
            "unique_blob_allocated_bytes": sum(
                row["allocated_bytes"] for row in unique_blobs.values()
            ),
            "files": verified,
            "shared_xet_used": False,
            "mop_touched": False,
            "network_accessed_by_verifier": False,
        }
    )


def _entry_type(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "special"


def build_inventory(
    roots: Sequence[Path],
    *,
    session_root: Path,
    mop_root: Path = MOP_ROOT,
    protected_roots: Sequence[Path] = (),
    include_file_hashes: bool = False,
) -> dict[str, Any]:
    """Build a physical-inode inventory; hard links are counted exactly once."""
    if not roots:
        _fail("inventory requires at least one release root")
    clean_session = _require_absolute_clean(session_root, label="inventory session")
    clean_roots = [_require_absolute_clean(path, label="inventory root") for path in roots]
    if len(clean_roots) != len(set(clean_roots)):
        _fail("inventory roots are duplicated")
    mop = _require_absolute_clean(mop_root, label="MOP root")
    protected = [
        _require_absolute_clean(path, label="protected root") for path in protected_roots
    ]
    for root in clean_roots:
        if not _within(root, clean_session) or root == clean_session:
            _fail(f"inventory root is not a strict child of its session: {root}")
        if _within(root, mop) or _within(mop, root):
            _fail(f"inventory root overlaps MOP: {root}")
        for keep in protected:
            if _within(root, keep) or _within(keep, root):
                _fail(f"inventory root overlaps protected root: {root} / {keep}")

    objects: dict[tuple[int, int], dict[str, Any]] = {}
    violations: list[str] = []

    def add(path: Path, named: os.stat_result, *, link_target: str | None = None) -> None:
        kind = _entry_type(named.st_mode)
        key = (int(named.st_dev), int(named.st_ino))
        row = objects.get(key)
        if row is None:
            row = {
                "device": key[0],
                "inode": key[1],
                "type": kind,
                "aliases": [],
                "logical_bytes": int(named.st_size),
                "allocated_bytes": int(named.st_blocks) * 512,
                "uid": int(named.st_uid),
                "mode": f"{stat.S_IMODE(named.st_mode):04o}",
                "hard_links": int(named.st_nlink),
            }
            if link_target is not None:
                row["link_target"] = link_target
                row["resolved_link_target"] = os.path.normpath(
                    os.path.join(path.parent, link_target)
                )
            if include_file_hashes and kind == "regular":
                row["sha256"] = _hash_regular(
                    path, label=f"inventory file {path}", expected_uid=os.getuid()
                )["sha256"]
            objects[key] = row
        elif row["type"] != kind:
            violations.append(f"INODE_TYPE_CHANGED:{path}")
        row["aliases"].append(os.fspath(path))
        if named.st_uid != os.getuid():
            violations.append(f"OWNER_UID_MISMATCH:{path}")
        if kind == "special":
            violations.append(f"SPECIAL_NODE:{path}")
        if kind == "regular" and named.st_nlink != 1:
            violations.append(f"HARDLINK_COUNT_{named.st_nlink}:{path}")
        if _within(path, mop):
            violations.append(f"MOP_PATH:{path}")
        if kind == "symlink":
            assert link_target is not None
            if os.path.isabs(link_target) or "\x00" in link_target:
                violations.append(f"ABSOLUTE_OR_INVALID_SYMLINK:{path}")
            resolved = Path(os.path.normpath(os.path.join(path.parent, link_target)))
            if _within(resolved, mop):
                violations.append(f"MOP_SYMLINK_TARGET:{path}")
            if not any(_within(resolved, root) for root in clean_roots):
                violations.append(f"SYMLINK_ESCAPES_RELEASE_ROOTS:{path}")

    def walk(path: Path, descriptor: int) -> None:
        root_meta = os.fstat(descriptor)
        add(path, root_meta)
        for name in sorted(os.listdir(descriptor)):
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child_path = path / name
            if stat.S_ISDIR(named.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                    dir_fd=descriptor,
                )
                try:
                    if not _identity_equal(named, os.fstat(child)):
                        _fail(f"inventory directory changed while opening: {child_path}")
                    walk(child_path, child)
                finally:
                    os.close(child)
            elif stat.S_ISLNK(named.st_mode):
                add(child_path, named, link_target=os.readlink(name, dir_fd=descriptor))
            else:
                add(child_path, named)

    root_identities: set[tuple[int, int]] = set()
    for root in sorted(clean_roots, key=os.fspath):
        if os.path.realpath(root) != os.fspath(root):
            _fail(f"inventory root resolves through a symlink: {root}")
        descriptor = _open_absolute_directory(root)
        try:
            metadata = os.fstat(descriptor)
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            if identity in root_identities:
                _fail("inventory roots alias the same directory")
            root_identities.add(identity)
            walk(root, descriptor)
        finally:
            os.close(descriptor)

    rows = []
    for row in objects.values():
        row["aliases"] = sorted(set(row["aliases"]))
        rows.append(row)
    rows.sort(key=lambda row: row["aliases"][0])
    violations = sorted(set(violations))
    entry_count = sum(len(row["aliases"]) for row in rows)
    document = {
        "schema": "hawking.kimi_k26.release_cycle.inventory.v1",
        "status": "PASS" if not violations else "BLOCKED",
        "session_root": os.fspath(clean_session),
        "release_roots": [os.fspath(path) for path in sorted(clean_roots, key=os.fspath)],
        "protected_roots": [os.fspath(path) for path in sorted(protected, key=os.fspath)],
        "mop_root": os.fspath(mop),
        "entry_count": entry_count,
        "unique_physical_object_count": len(rows),
        "logical_bytes_deduplicated": sum(row["logical_bytes"] for row in rows),
        "allocated_bytes_deduplicated": sum(row["allocated_bytes"] for row in rows),
        "inventory_rows": rows,
        "violations": violations,
        "hardlinks_counted_once": True,
        "deletion_performed": False,
    }
    return seal_document(document)


def verify_inventory(value: dict[str, Any]) -> dict[str, Any]:
    verify_sealed_document(value, label="release inventory")
    if value.get("schema") != "hawking.kimi_k26.release_cycle.inventory.v1":
        _fail("release inventory schema changed")
    rows = value.get("inventory_rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        _fail("release inventory rows are malformed")
    if value.get("unique_physical_object_count") != len(rows):
        _fail("release inventory unique object count changed")
    if value.get("entry_count") != sum(len(row.get("aliases", [])) for row in rows):
        _fail("release inventory entry count changed")
    if value.get("logical_bytes_deduplicated") != sum(
        row.get("logical_bytes", -1) for row in rows
    ):
        _fail("release inventory logical byte sum changed")
    if value.get("allocated_bytes_deduplicated") != sum(
        row.get("allocated_bytes", -1) for row in rows
    ):
        _fail("release inventory allocated byte sum changed")
    status = "PASS" if not value.get("violations") else "BLOCKED"
    if value.get("status") != status:
        _fail("release inventory status disagrees with violations")
    return value


def build_source_release_inventory(
    layout: SessionLayout,
    *,
    mop_root: Path = MOP_ROOT,
    shared_xet: Path = SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    validate_layout(layout, mop_root=mop_root, shared_xet=shared_xet)
    return build_inventory(
        (layout.hub, layout.xet),
        session_root=layout.session,
        mop_root=mop_root,
        protected_roots=(layout.build, layout.recovery, layout.evidence, shared_xet),
        include_file_hashes=False,
    )


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C"},
    )


def _audit_lsof(roots: Sequence[Path], runner: Runner) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    failures: list[str] = []
    for root in roots:
        argv = ("/usr/sbin/lsof", "-nP", "+D", os.fspath(root))
        result = runner(argv)
        output = result.stdout or ""
        lines = [line for line in output.splitlines() if line.strip()]
        data_lines = lines[1:] if lines and lines[0].startswith("COMMAND") else lines
        if result.returncode not in {0, 1}:
            failures.append(f"LSOF_EXIT_{result.returncode}:{root}")
        if data_lines:
            matches.append(
                {
                    "root": os.fspath(root),
                    "match_count": len(data_lines),
                    "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                }
            )
    return {"matches": matches, "failures": failures, "reader_count": sum(
        row["match_count"] for row in matches
    )}


def _audit_processes(roots: Sequence[Path], runner: Runner) -> dict[str, Any]:
    result = runner(("/bin/ps", "-axo", "pid=,ppid=,command="))
    if result.returncode != 0:
        return {"matches": [], "failures": [f"PS_EXIT_{result.returncode}"]}
    markers = [os.fspath(root) for root in roots]
    matches: list[dict[str, Any]] = []
    for line in (result.stdout or "").splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 3 or not fields[0].isdigit():
            continue
        pid, ppid, command = int(fields[0]), int(fields[1]), fields[2]
        if pid == os.getpid():
            continue
        if any(marker in command for marker in markers) or (
            KIMI_REPO in command and KIMI_REVISION in command
        ):
            matches.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                }
            )
    return {"matches": matches, "failures": [], "matching_process_count": len(matches)}


def _audit_queues(queue_roots: Sequence[Path]) -> dict[str, Any]:
    pending: list[str] = []
    failures: list[str] = []
    for raw in queue_roots:
        path = _require_absolute_clean(raw, label="queue root")
        try:
            named = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
            failures.append(f"QUEUE_ROOT_NOT_REGULAR_DIRECTORY:{path}")
            continue
        try:
            descriptor = _open_absolute_directory(path)
        except (OSError, ReleaseCycleError):
            failures.append(f"QUEUE_ROOT_UNSAFE:{path}")
            continue
        try:
            for name in sorted(os.listdir(descriptor)):
                pending.append(os.fspath(path / name))
        finally:
            os.close(descriptor)
    return {"pending_entries": pending, "failures": failures, "pending_count": len(pending)}


def build_pre_release_audit(
    inventory: dict[str, Any],
    *,
    queue_roots: Sequence[Path] | None = None,
    runner: Runner = _default_runner,
) -> dict[str, Any]:
    verified = verify_inventory(inventory)
    roots = [Path(value) for value in verified["release_roots"]]
    queues = list(queue_roots) if queue_roots is not None else [
        LEGACY_RUNTIME_ROOT / "queue",
        LEGACY_RUNTIME_ROOT / "outbox",
        Path(verified["session_root"]) / "queue",
        Path(verified["session_root"]) / "outbox",
    ]
    lsof = _audit_lsof(roots, runner)
    processes = _audit_processes(roots, runner)
    queue = _audit_queues(queues)
    blockers: list[str] = []
    if verified["status"] != "PASS":
        blockers.append("RELEASE_INVENTORY_NOT_SAFE")
    if lsof["reader_count"]:
        blockers.append("OPEN_FILE_READERS_PRESENT")
    if lsof["failures"]:
        blockers.append("LSOF_AUDIT_FAILED")
    if processes.get("matching_process_count", len(processes["matches"])):
        blockers.append("MATCHING_PROCESS_PRESENT")
    if processes["failures"]:
        blockers.append("PROCESS_AUDIT_FAILED")
    if queue["pending_count"]:
        blockers.append("QUEUE_OR_OUTBOX_NOT_EMPTY")
    if queue["failures"]:
        blockers.append("QUEUE_AUDIT_FAILED")
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.pre_release_audit.v1",
            "status": "PASS" if not blockers else "BLOCKED",
            "inventory_seal_sha256": verified["seal_sha256"],
            "exact_release_allocated_bytes": verified["allocated_bytes_deduplicated"],
            "release_roots": verified["release_roots"],
            "lsof": lsof,
            "processes": processes,
            "queues": queue,
            "blockers": sorted(blockers),
            "deletion_authorized": False,
            "deletion_performed": False,
        }
    )


def build_preflight(
    layout: SessionLayout,
    *,
    manifest_path: Path = OFFICIAL_MANIFEST,
    archive_path: Path = SANITIZED_ARCHIVE,
    mop_root: Path = MOP_ROOT,
    shared_xet: Path = SHARED_HF_XET_ROOT,
) -> dict[str, Any]:
    layout_evidence = validate_layout(layout, mop_root=mop_root, shared_xet=shared_xet)
    manifest = verify_manifest(manifest_path)
    archive = verify_recovery_archive(archive_path)
    plan = build_download_plan(
        layout, manifest_path=manifest_path, mop_root=mop_root, shared_xet=shared_xet
    )
    return seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.preflight.v1",
            "status": "PASS_PHASE1_NO_LIVE_ACTION",
            "layout": layout_evidence,
            "manifest_verification_seal_sha256": manifest["seal_sha256"],
            "recovery_archive_seal_sha256": archive["seal_sha256"],
            "download_plan_seal_sha256": plan["seal_sha256"],
            "source_present": layout.snapshot.exists(),
            "network_accessed": False,
            "download_executed": False,
            "delete_capability_present": False,
        }
    )


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-session", help="create private directories only")
    init.add_argument("session_id")
    for name in ("preflight", "plan-download", "verify-source", "verify-recovery"):
        child = sub.add_parser(name)
        child.add_argument("--session", type=Path, required=True)
    sub.choices["preflight"].add_argument("--manifest", type=Path, default=OFFICIAL_MANIFEST)
    sub.choices["preflight"].add_argument("--archive", type=Path, default=SANITIZED_ARCHIVE)
    sub.choices["plan-download"].add_argument(
        "--manifest", type=Path, default=OFFICIAL_MANIFEST
    )
    sub.choices["verify-source"].add_argument(
        "--manifest", type=Path, default=OFFICIAL_MANIFEST
    )
    sub.choices["verify-recovery"].add_argument(
        "--archive", type=Path, default=SANITIZED_ARCHIVE
    )
    sub.choices["verify-recovery"].add_argument(
        "--verify-capsule", action="store_true",
        help="also verify the exact payload/result/capture at fixed session paths",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init-session":
            value = init_session(args.session_id)
        else:
            layout = layout_for(args.session)
            if args.command == "preflight":
                value = build_preflight(
                    layout, manifest_path=args.manifest, archive_path=args.archive
                )
            elif args.command == "plan-download":
                value = build_download_plan(layout, manifest_path=args.manifest)
            elif args.command == "verify-source":
                value = verify_source(layout, manifest_path=args.manifest)
            elif args.command == "verify-recovery":
                archive = verify_recovery_archive(args.archive)
                if args.verify_capsule:
                    capsule = verify_payload_result_capture(layout)
                    value = seal_document(
                        {
                            "schema": "hawking.kimi_k26.release_cycle.recovery_verification.v1",
                            "status": "PASS",
                            "archive_seal_sha256": archive["seal_sha256"],
                            "capsule_seal_sha256": capsule["seal_sha256"],
                        }
                    )
                else:
                    value = archive
            else:  # pragma: no cover - argparse constrains this branch
                _fail("unsupported phase-1 command")
    except (OSError, ReleaseCycleError, zipfile.BadZipFile) as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "error": str(exc)},
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    _print_json(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
