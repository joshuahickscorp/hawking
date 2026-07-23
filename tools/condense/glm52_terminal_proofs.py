#!/usr/bin/env python3.12
"""Pure semantic proofs for the GLM-5.2 offline-ready stop conditions.

These proofs are inputs to, not substitutes for, authenticated controller stop
receipts.  This module performs no writes and deliberately proves only the nine
conditions whose evidence is complete before model-body streaming.  Every proof
binds the exact frozen artifact seals, raw file bytes, campaign identity, and the
SHA-256 of this validator source file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

try:  # Package import under pytest/library use.
    from .glm52_common import Glm52Error, canonical, verify_sealed
except ImportError:  # Direct script/tool import with tools/condense on sys.path.
    from glm52_common import Glm52Error, canonical, verify_sealed


PROOF_SCHEMA = "hawking.glm52.terminal_semantic_proof.v1"
CAMPAIGN_ID = "glm52-bf16-xet-gravity"
OFFICIAL_REPO = "zai-org/GLM-5.2"
OFFICIAL_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
OFFICIAL_LICENSE = "MIT"
OFFICIAL_LICENSE_SHA256 = (
    "f4a18c6ae40b0a8e7d2b7667f52f6e1994e54a46430d2e172b73cb8c9b5eb0d7"
)
OFFICIAL_FILE_COUNT = 295
OFFICIAL_WEIGHT_SHARDS = 282
OFFICIAL_SOURCE_LOGICAL_BYTES = 1_506_693_036_946
OFFICIAL_WEIGHT_CONTAINER_BYTES = 1_506_667_387_408
OFFICIAL_NONWEIGHT_BYTES = 25_649_538
OFFICIAL_TENSOR_PAYLOAD_BYTES = 1_506_659_919_872
OFFICIAL_TENSOR_COUNT = 59_585
OFFICIAL_LOGICAL_WEIGHTS = 753_329_940_480
OFFICIAL_BF16_TENSOR_COUNT = 59_509
OFFICIAL_BF16_LOGICAL_WEIGHTS = 753_329_921_024
OFFICIAL_BF16_PAYLOAD_BYTES = 1_506_659_842_048
OFFICIAL_F32_TENSOR_COUNT = 76
OFFICIAL_F32_LOGICAL_WEIGHTS = 19_456
OFFICIAL_F32_PAYLOAD_BYTES = 77_824

KIMI_REPO = "moonshotai/Kimi-K2.6"
KIMI_REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
KIMI_SOURCE_LOGICAL_BYTES = 595_204_999_341
KIMI_WEIGHT_BYTES = 595_177_988_208
KIMI_WEIGHT_SHARDS = 64
KIMI_FILE_COUNT = 96
KIMI_SOURCE_ALLOCATED_BYTES = 595_205_144_576
KIMI_TOTAL_CLEANUP_DELTA_BYTES = 597_515_915_264

READY_STOP_CONDITIONS: tuple[str, ...] = (
    "kimi_final_evidence_verified",
    "kimi_raw_source_safely_released",
    "official_glm52_immutable_revision_sealed",
    "bf16_source_manifest_complete",
    "exact_logical_weight_ledger_sealed",
    "gravity_pre_audit_complete",
    "external_baseline_matrix_complete",
    "adapter_twin_green",
    "corpus_integrity_green",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class TerminalProofError(Glm52Error):
    """Raised when a stop-condition claim is not exactly supported."""


FROZEN_ARTIFACT_SEALS: dict[str, str] = {
    "GLM52_HANDOFF_PRECHECK.json":
        "7d2892d5e5c156d9537d334d7a91ddf023d6f7835bd444a32f71f734f229331e",
    "KIMI_K26_GRAVITY_FINAL.json":
        "63e478a1b24da9604b18cb3388fa478bac1b2fa1a24f2953cc601d8aa445823a",
    "KIMI_K26_SOURCE_RELEASE_FOR_GLM52.json":
        "580c923ef86cfdb2a7aa16d84fa80139f5d49177509958edee87f8792b488ee2",
    "KIMI_K26_DEVICE_CLEANSE_FINAL.json":
        "b08ce29c12bbc3c2d37bc9f3154013df6316c261599d22843fef4289b48891c9",
    "KIMI_K26_CREDENTIAL_REMEDIATION_FOR_GLM52.json":
        "fd7fb566795030c1a0a21876c1d1302b780f0c5ac677bd30d9567867be72f2bd",
    "reports/condense/kimi_k26/KIMI_K26_OFFICIAL_MANIFEST.json":
        "22123f6ed9ae2688da383b5de8c671e70071802e6fc1ca0a16df90c607885c60",
    "KIMI_K26_GRAVITY_M1_ORACLE_BANDWIDTH.json":
        "8ff3234bf77ca96bc3bbb6858745fa37ec937e0d4ce5829283480e29e4a52de2",
    "KIMI_K26_GRAVITY_M2_CONDITIONAL_GATE.json":
        "c1a001c4bedb40ffc0b6a590514259fe7a198aaac402ffb5b691d5010c7881a1",
    "KIMI_K26_GRAVITY_RATE_LADDER.json":
        "b03e1c75c591f43ba3ba8beeb88c6a21537c673e746e947f8a5e7098f9ee45b8",
    "KIMI_K26_GRAVITY_M7_ORACLE_GAP.json":
        "5f82435a0c655b2384a1bfffb90e3d0106f7aa448d7f9c0e2687bc3adef4434e",
    "KIMI_K26_FINAL_BYTE_AUCTION.json":
        "d8b8f02b2e19c4bee39cf8a5ceb91f05e1878ee18e889c2bcf5f88ccc1b42719",
    "KIMI_K26_DISK_POLICY.json":
        "e44219c1f448055c49f27e17bc410d107ff7a45b7bd9d6eab9109a0affe5c4f2",
    "KIMI_K26_GRAVITY_NONLINEAR_TOURNAMENT.json":
        "8c201bbf527c66daf99f3c87de3fb26978c55f66635dd414a7ec641e721f3b6e",
    "KIMI_K26_GRAVITY_ONEOFF_CLOSURE.json":
        "6776748c50fbcdf45a3441144dda55986592530e60e0e39d5307745cb8248ce7",
    "KIMI_K26_LONG_RUN_FINAL.json":
        "9c31d2a22e488d4234f4acad443beb564ab263a32ebee06f01c5ba4e76a18bff",
    "GLM52_SOURCE_ADMISSION.json":
        "f6d80dcfd32ce77d591958c70771a41ce4ce0c00d167caa6a35a46267c42ddbe",
    "GLM52_ARCHITECTURE_CONTRACT.json":
        "dd1fea044872734b74572f57b3bd7d4966110d8f9001c115e0d989994189948f",
    "GLM52_SHARD_DEPENDENCY_GRAPH.json":
        "a08d1ad1b040856a3ed0649d615faf37f3304000c7e00d3f22561d51519b40b9",
    "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json":
        "ca765d5197d25a888ebc1e31ca5d03cae4492235d3470dd6c93c53f98682af89",
    "GLM52_OFFICIAL_MANIFEST.json":
        "ef4e3cbf6b4b6ddc52aa7278d657a9f7f63254fa72d35591cc8f6c5c706c4791",
    "GLM52_LOGICAL_WEIGHT_LEDGER.json":
        "d479791a7f3c279f0d1cc56a6203e214e4e2610da04010ba020347603dacb0db",
    "GLM52_SOURCE_FORMAT_LEDGER.json":
        "14d1e124fdbb9286e0122736a2cf3ff857c375a7abf9204043d8b356f29f6d8e",
    "GRAVITY_COMPLETENESS_AUDIT_GLM52_PRE.json":
        "d298c265b3ecede35fcb3f0809aa004ea2a41bd2c11c1da51b741949e60648bd",
    "GRAVITY_EXTERNAL_BASELINE_MATRIX.json":
        "c69b5d636944fbbfcdc76c72ce3441e323add538b94a3566b25f251f1ff49206",
    "GLM52_ADAPTER_TWIN.json":
        "8e2f0c5dfbc9647c240373ad207d6572ba6aad85d8fa1d521c9f7f1b93eb11fe",
    "GLM52_REFERENCE_PARITY.json":
        "6b7d9764f8af6d1d795bada3966d7b73807941a092c781d0fca1a9daeb2dafa1",
    "GLM52_CORPUS_INTEGRITY.json":
        "7f5eff81a10c01b231e829d5870d6e8290a745a4bacdf4dfe74c4d0e5080e767",
}

FROZEN_RAW_DOCUMENT_SHA256: dict[str, str] = {
    "KIMI_K26_GRAVITY_FINAL.md":
        "04370b55d1073923877989874cbf336869c949cd4892dbe3e1a845c5e2fc0752",
    "KIMI_K26_NEXT_PARENT_TRANSFER.md":
        "8c34b679524327e4a3ff61bc82fe7d451f62c42f3d200e7aafe0423a10a01a34",
}

_KIMI_CHAIN: dict[str, tuple[str, str, str | None]] = {
    "M1": (
        "KIMI_K26_GRAVITY_M1_ORACLE_BANDWIDTH.json",
        "hawking.kimi_k26.gravity.m1_oracle_bandwidth.v1",
        "PASS",
    ),
    "M2": (
        "KIMI_K26_GRAVITY_M2_CONDITIONAL_GATE.json",
        "hawking.kimi_k26.gravity.m2_conditional_gate.v1",
        "PASS",
    ),
    "M5": (
        "KIMI_K26_GRAVITY_RATE_LADDER.json",
        "hawking.kimi_k26.gravity_rate_ladder.v1.artifact",
        "PASS",
    ),
    "M7": (
        "KIMI_K26_GRAVITY_M7_ORACLE_GAP.json",
        "hawking.kimi_k26.gravity.m7_oracle_gap.v1",
        "PARTIAL_WAITING_PREREQUISITE",
    ),
    "byte_auction": (
        "KIMI_K26_FINAL_BYTE_AUCTION.json",
        "hawking.kimi_k26.final_byte_auction.v1",
        "PASS",
    ),
    "disk_policy": (
        "KIMI_K26_DISK_POLICY.json",
        "hawking.kimi_k26.disk_policy.v1",
        "PASS",
    ),
    "nonlinear_tournament": (
        "KIMI_K26_GRAVITY_NONLINEAR_TOURNAMENT.json",
        "hawking.kimi_k26.gravity_nonlinear.v1.tournament",
        "PASS",
    ),
    "oneoff_closure": (
        "KIMI_K26_GRAVITY_ONEOFF_CLOSURE.json",
        "hawking.kimi_k26.gravity_oneoff_closure.v1",
        "PASS",
    ),
    "prior_long_run": (
        "KIMI_K26_LONG_RUN_FINAL.json",
        "hawking.kimi_k26.long_run_final.v1",
        None,
    ),
}

_PRE_AUDIT_AXES: tuple[str, ...] = (
    "source_authority", "source_precision", "logical_weight_accounting",
    "physical_artifact_accounting", "adapter_fidelity",
    "teacher_forward_fidelity", "streaming_completeness", "resume_recovery",
    "data_integrity", "causal_diagnosis", "doctor_breadth",
    "native_studentization", "rate_exploration", "full_model_artifact",
    "capability_evaluation", "direct_runtime", "metal_execution",
    "speed_efficiency", "resource_utilization", "scientific_transfer",
    "reproducibility",
)


@dataclass(frozen=True, slots=True)
class _Loaded:
    path: str
    raw: bytes
    file_sha256: str
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SemanticResult:
    artifact_paths: tuple[str, ...]
    document_paths: tuple[str, ...]
    facts: dict[str, Any]
    scope: dict[str, Any]
    proven: bool = True
    blockers: tuple[str, ...] = ()


def _fail(message: str) -> None:
    raise TerminalProofError(message)


def _expect(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        _fail(f"{label} mismatch: observed={actual!r} expected={expected!r}")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be a list")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        _fail(f"{label} contains non-finite number {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalProofError(f"cannot decode strict JSON {label}: {exc}") from exc
    return _require_object(value, label)


class _Reader:
    """Stable no-follow reader for fixed repository-relative evidence files."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        root_text = os.fspath(root)
        if not isinstance(root_text, str) or not os.path.isabs(root_text):
            _fail("terminal proof root must be absolute")
        if root_text.startswith("//") or os.path.normpath(root_text) != root_text:
            _fail("terminal proof root must be normalized")
        if not _NOFOLLOW or not getattr(os, "O_DIRECTORY", 0):
            _fail("terminal proof reads require O_NOFOLLOW and O_DIRECTORY")
        self.root = Path(root_text)
        self._root_path = root_text
        self._loaded: dict[str, _Loaded] = {}
        self._raw: dict[str, bytes] = {}
        descriptors, _, root_stat = self._open_absolute_root()
        self._root_identity = self._identity(root_stat)
        for descriptor in reversed(descriptors):
            os.close(descriptor)

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, int, int]:
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            stat.S_IFMT(metadata.st_mode),
        )

    @classmethod
    def _fingerprint(cls, metadata: os.stat_result) -> tuple[int, ...]:
        return (
            *cls._identity(metadata),
            int(metadata.st_size),
            int(metadata.st_nlink),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )

    @staticmethod
    def _lstat_at(component: str, directory_fd: int) -> os.stat_result:
        return os.stat(component, dir_fd=directory_fd, follow_symlinks=False)

    def _open_absolute_root(
        self,
    ) -> tuple[
        list[int],
        list[tuple[str, tuple[int, int, int]]],
        os.stat_result,
    ]:
        flags = (
            os.O_RDONLY | _CLOEXEC | _NOFOLLOW | getattr(os, "O_DIRECTORY", 0)
        )
        descriptors: list[int] = []
        links: list[tuple[str, tuple[int, int, int]]] = []
        try:
            named_root = os.stat("/", follow_symlinks=False)
            filesystem_root = os.open("/", flags)
            descriptors.append(filesystem_root)
            if self._identity(named_root) != self._identity(os.fstat(filesystem_root)):
                _fail("filesystem root changed while opening terminal-proof root")
            for component in (
                item for item in self._root_path.split("/") if item
            ):
                named = self._lstat_at(component, descriptors[-1])
                if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
                    _fail(
                        f"terminal-proof root component is not a real directory: {component!r}"
                    )
                child = os.open(component, flags, dir_fd=descriptors[-1])
                try:
                    opened = os.fstat(child)
                except OSError:
                    os.close(child)
                    raise
                if self._identity(named) != self._identity(opened):
                    os.close(child)
                    _fail(
                        f"terminal-proof root component changed while opening: {component!r}"
                    )
                links.append((component, self._identity(opened)))
                descriptors.append(child)
            return descriptors, links, os.fstat(descriptors[-1])
        except BaseException as exc:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
            if isinstance(exc, OSError):
                raise TerminalProofError(
                    f"cannot open terminal-proof root: {exc}"
                ) from exc
            raise

    def _open_relative_directories(
        self, root_fd: int, components: Sequence[str]
    ) -> tuple[list[int], list[tuple[str, tuple[int, int, int]]]]:
        flags = (
            os.O_RDONLY | _CLOEXEC | _NOFOLLOW | getattr(os, "O_DIRECTORY", 0)
        )
        descriptors = [os.dup(root_fd)]
        links: list[tuple[str, tuple[int, int, int]]] = []
        try:
            for component in components:
                named = self._lstat_at(component, descriptors[-1])
                if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
                    _fail(
                        f"evidence directory is not a real directory: {component!r}"
                    )
                child = os.open(component, flags, dir_fd=descriptors[-1])
                try:
                    opened = os.fstat(child)
                except OSError:
                    os.close(child)
                    raise
                if self._identity(named) != self._identity(opened):
                    os.close(child)
                    _fail(f"evidence directory changed while opening: {component!r}")
                links.append((component, self._identity(opened)))
                descriptors.append(child)
            return descriptors, links
        except BaseException as exc:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
            if isinstance(exc, OSError):
                raise TerminalProofError(
                    f"cannot open terminal-proof evidence directory: {exc}"
                ) from exc
            raise

    def _verify_chain(
        self,
        descriptors: Sequence[int],
        links: Sequence[tuple[str, tuple[int, int, int]]],
        *,
        label: str,
    ) -> None:
        if len(descriptors) != len(links) + 1:
            _fail(f"{label} descriptor chain is malformed")
        for index, (component, expected) in enumerate(links):
            named = self._lstat_at(component, descriptors[index])
            opened = os.fstat(descriptors[index + 1])
            if stat.S_ISLNK(named.st_mode) or self._identity(named) != expected \
                    or self._identity(opened) != expected:
                _fail(f"{label} component identity changed: {component!r}")

    @staticmethod
    def _parts(relative: str) -> tuple[str, ...]:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or not pure.parts or any(
            part in {"", ".", ".."} for part in pure.parts
        ) or "/".join(pure.parts) != relative:
            _fail(f"unsafe evidence path: {relative!r}")
        return tuple(pure.parts)

    def raw(self, relative: str) -> bytes:
        if relative in self._raw:
            return self._raw[relative]
        parts = self._parts(relative)
        root_descriptors, root_links, root_stat = self._open_absolute_root()
        if self._identity(root_stat) != self._root_identity:
            for descriptor in reversed(root_descriptors):
                os.close(descriptor)
            _fail("terminal-proof root identity changed after initialization")
        directory_descriptors: list[int] = []
        fd: int | None = None
        try:
            directory_descriptors, directory_links = self._open_relative_directories(
                root_descriptors[-1], parts[:-1]
            )
            parent_fd = directory_descriptors[-1]
            leaf = parts[-1]
            named_pre = self._lstat_at(leaf, parent_fd)
            if stat.S_ISLNK(named_pre.st_mode) or not stat.S_ISREG(named_pre.st_mode):
                _fail(f"evidence path is not a regular non-symlink file: {relative}")
            if int(named_pre.st_nlink) != 1:
                _fail(f"evidence file has multiple hard links: {relative}")
            fd = os.open(
                leaf,
                os.O_RDONLY | _CLOEXEC | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            before = os.fstat(fd)
            if self._fingerprint(before) != self._fingerprint(named_pre):
                _fail(f"evidence changed while opening: {relative}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            after = os.fstat(fd)
            if self._fingerprint(after) != self._fingerprint(before) \
                    or total != int(after.st_size):
                _fail(f"evidence changed while reading: {relative}")
            named_post = self._lstat_at(leaf, parent_fd)
            if self._fingerprint(named_post) != self._fingerprint(after):
                _fail(f"evidence path changed after reading: {relative}")
            self._verify_chain(
                directory_descriptors,
                directory_links,
                label="terminal-proof evidence directory",
            )
            self._verify_chain(
                root_descriptors,
                root_links,
                label="terminal-proof root",
            )
            if self._identity(os.fstat(root_descriptors[-1])) != self._root_identity:
                _fail("terminal-proof root identity changed during read")
        except OSError as exc:
            raise TerminalProofError(f"cannot open evidence {relative}: {exc}") from exc
        finally:
            if fd is not None:
                os.close(fd)
            for descriptor in reversed(directory_descriptors):
                os.close(descriptor)
            for descriptor in reversed(root_descriptors):
                os.close(descriptor)
        raw = b"".join(chunks)
        self._raw[relative] = raw
        return raw

    def sealed(
        self,
        relative: str,
        *,
        schema: str | None = None,
        status: str | None | object = ...,
    ) -> dict[str, Any]:
        if relative in self._loaded:
            value = self._loaded[relative].value
        else:
            raw = self.raw(relative)
            value = _strict_json(raw, label=relative)
            try:
                verify_sealed(value, label=relative)
            except Glm52Error as exc:
                raise TerminalProofError(str(exc)) from exc
            frozen = FROZEN_ARTIFACT_SEALS.get(relative)
            if frozen is None:
                _fail(f"artifact has no frozen seal policy: {relative}")
            _expect(value.get("seal_sha256"), frozen, f"{relative} frozen seal")
            self._loaded[relative] = _Loaded(
                path=relative,
                raw=raw,
                file_sha256=hashlib.sha256(raw).hexdigest(),
                value=value,
            )
        if schema is not None:
            _expect(value.get("schema"), schema, f"{relative} schema")
        if status is not ...:
            _expect(value.get("status"), status, f"{relative} status")
        return value

    def artifact_binding(self, relative: str) -> dict[str, Any]:
        loaded = self._loaded.get(relative)
        if loaded is None:
            _fail(f"internal validator omitted artifact load: {relative}")
        return {
            "path": relative,
            "file_sha256": loaded.file_sha256,
            "bytes": len(loaded.raw),
            "schema": loaded.value.get("schema"),
            "status": loaded.value.get("status"),
            "seal_sha256": loaded.value.get("seal_sha256"),
        }

    def document_binding(self, relative: str) -> dict[str, Any]:
        raw = self.raw(relative)
        return {
            "path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }


def validator_source_binding() -> dict[str, Any]:
    path = Path(__file__).resolve()
    named_pre = path.lstat()
    if stat.S_ISLNK(named_pre.st_mode) or not stat.S_ISREG(named_pre.st_mode) \
            or int(named_pre.st_nlink) != 1:
        _fail("terminal-proof validator source is not a single-link regular file")
    try:
        descriptor = os.open(path, os.O_RDONLY | _CLOEXEC | _NOFOLLOW)
    except OSError as exc:
        raise TerminalProofError(f"cannot open validator source: {exc}") from exc
    try:
        opened_pre = os.fstat(descriptor)
        before = (
            int(opened_pre.st_dev), int(opened_pre.st_ino), int(opened_pre.st_size),
            int(opened_pre.st_nlink), int(opened_pre.st_mtime_ns),
            int(opened_pre.st_ctime_ns),
        )
        named = (
            int(named_pre.st_dev), int(named_pre.st_ino), int(named_pre.st_size),
            int(named_pre.st_nlink), int(named_pre.st_mtime_ns),
            int(named_pre.st_ctime_ns),
        )
        if before != named:
            _fail("terminal-proof validator source changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_post = os.fstat(descriptor)
        after = (
            int(opened_post.st_dev), int(opened_post.st_ino), int(opened_post.st_size),
            int(opened_post.st_nlink), int(opened_post.st_mtime_ns),
            int(opened_post.st_ctime_ns),
        )
        if after != before:
            _fail("terminal-proof validator source changed while reading")
    finally:
        os.close(descriptor)
    named_post = path.lstat()
    if (
        int(named_post.st_dev), int(named_post.st_ino), int(named_post.st_size),
        int(named_post.st_nlink), int(named_post.st_mtime_ns),
        int(named_post.st_ctime_ns),
    ) != after:
        _fail("terminal-proof validator source name changed after reading")
    raw = b"".join(chunks)
    if len(raw) != int(opened_post.st_size):
        _fail("terminal-proof validator source read length mismatch")
    return {
        "path": "tools/condense/glm52_terminal_proofs.py",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _stable_absolute_absence(path_value: Any) -> dict[str, Any]:
    if not isinstance(path_value, str) or not os.path.isabs(path_value):
        _fail("Kimi former source root must be an absolute path")
    path = Path(path_value)
    parent = path.parent
    try:
        parent_pre = parent.stat()
    except OSError as exc:
        raise TerminalProofError(f"cannot inspect Kimi source parent: {exc}") from exc
    if os.path.lexists(path_value):
        _fail("Kimi raw source root is present; release is not complete")
    if os.path.lexists(path_value):
        _fail("Kimi raw source root appeared during absence observation")
    parent_post = parent.stat()
    before = (
        int(parent_pre.st_dev), int(parent_pre.st_ino), int(parent_pre.st_mtime_ns),
        int(parent_pre.st_ctime_ns),
    )
    after = (
        int(parent_post.st_dev), int(parent_post.st_ino), int(parent_post.st_mtime_ns),
        int(parent_post.st_ctime_ns),
    )
    if before != after:
        _fail("Kimi source parent changed during absence observation")
    return {"path": path_value, "absent": True}


def _validate_kimi_final(reader: _Reader) -> _SemanticResult:
    handoff_path = "GLM52_HANDOFF_PRECHECK.json"
    final_path = "KIMI_K26_GRAVITY_FINAL.json"
    handoff = reader.sealed(
        handoff_path,
        schema="hawking.glm52.handoff_precheck.v1",
        status="PASS_WITH_SECURITY_AND_ROLLBACK_EXCEPTIONS",
    )
    final = reader.sealed(
        final_path,
        schema="hawking.kimi_k26.gravity_final.v1",
        status="CLOSED",
    )
    _expect(final.get("terminal_outcome"), "OUTCOME_C", "Kimi terminal outcome")
    candidate = _require_object(final.get("best_deployable_candidate"), "Kimi candidate")
    _expect(candidate.get("candidate"), "P1_DUAL_PATH_RECOVERY_R16X2", "Kimi candidate")
    _expect(candidate.get("f2_promotable"), False, "Kimi F2 promotion")
    bpw = candidate.get("complete_bpw")
    if not isinstance(bpw, (int, float)) or not math.isclose(
        float(bpw), 0.9085909525553385, rel_tol=0.0, abs_tol=1e-15
    ):
        _fail("Kimi complete BPW mismatch")
    diagnosis = _require_object(final.get("causal_diagnosis"), "Kimi diagnosis")
    _expect(
        diagnosis.get("diagnosis"),
        "UPSTREAM_STATE_PRIMARY_ROUTE_SECONDARY_CONDITIONAL_AMPLIFIER",
        "Kimi causal diagnosis",
    )
    fidelity = _require_object(final.get("fidelity"), "Kimi fidelity")
    _expect(_require_object(fidelity.get("F1"), "Kimi F1").get("status"), "COMPLETE", "Kimi F1")
    _expect(
        _require_object(fidelity.get("F2"), "Kimi F2").get("promotable"),
        False,
        "Kimi F2 promotable",
    )
    chain = _require_object(final.get("evidence_chain"), "Kimi evidence chain")
    _expect(set(chain), set(_KIMI_CHAIN), "Kimi evidence-chain inventory")
    handoff_science = _require_object(handoff.get("kimi_science"), "handoff Kimi science")
    _expect(handoff_science.get("terminal_outcome"), "OUTCOME_C", "handoff outcome")
    _expect(
        handoff_science.get("primary_diagnosis"),
        "UPSTREAM_STATE_PRIMARY_ROUTE_SECONDARY_CONDITIONAL_AMPLIFIER",
        "handoff diagnosis",
    )
    handoff_final = _require_object(handoff_science.get("final"), "handoff final")
    _expect(handoff_final.get("seal_sha256"), final["seal_sha256"], "handoff final seal")
    _expect(
        handoff_final.get("file_sha256"),
        hashlib.sha256(reader.raw(final_path)).hexdigest(),
        "handoff final file hash",
    )
    artifact_paths = [handoff_path, final_path]
    observed_chain_rows: list[dict[str, Any]] = []
    for evidence_id, (path, schema, status) in _KIMI_CHAIN.items():
        entry = _require_object(chain.get(evidence_id), f"Kimi chain {evidence_id}")
        _expect(Path(str(entry.get("path"))).name, path, f"Kimi chain {evidence_id} path")
        _expect(
            entry.get("seal_sha256"),
            FROZEN_ARTIFACT_SEALS[path],
            f"Kimi chain {evidence_id} seal",
        )
        reader.sealed(path, schema=schema, status=status)
        artifact_paths.append(path)
        observed_chain_rows.append(
            {"evidence_id": evidence_id, "path": path, "seal_sha256": entry["seal_sha256"]}
        )
    _expect(
        handoff_science.get("evidence_chain_verified"),
        observed_chain_rows,
        "handoff evidence chain",
    )
    for path, expected_hash in FROZEN_RAW_DOCUMENT_SHA256.items():
        raw = reader.raw(path)
        _expect(hashlib.sha256(raw).hexdigest(), expected_hash, f"{path} raw hash")
    final_markdown = reader.raw("KIMI_K26_GRAVITY_FINAL.md").decode("utf-8")
    transfer = reader.raw("KIMI_K26_NEXT_PARENT_TRANSFER.md").decode("utf-8")
    for needle in ("OUTCOME_C", "P1_DUAL_PATH_RECOVERY_R16X2", "0.908590952555"):
        if needle not in final_markdown:
            _fail(f"Kimi final Markdown omits {needle!r}")
    if "native functional student" not in transfer.lower() \
            or "teacher-hidden" not in transfer.lower():
        _fail("Kimi next-parent transfer omits required native/teacher-hidden action")
    handoff_final_md = _require_object(
        handoff_science.get("final_markdown"), "handoff final Markdown"
    )
    handoff_transfer = _require_object(
        handoff_science.get("next_parent_transfer"), "handoff next-parent transfer"
    )
    _expect(
        handoff_final_md.get("sha256"),
        FROZEN_RAW_DOCUMENT_SHA256["KIMI_K26_GRAVITY_FINAL.md"],
        "handoff final Markdown hash",
    )
    _expect(
        handoff_transfer.get("sha256"),
        FROZEN_RAW_DOCUMENT_SHA256["KIMI_K26_NEXT_PARENT_TRANSFER.md"],
        "handoff transfer hash",
    )
    _expect(
        handoff_final_md.get("integrity_authority"),
        "0210e5aa05f0e3c69d6f2022c539c9dc90cce322",
        "handoff closure commit",
    )
    _expect(
        handoff_transfer.get("integrity_authority"),
        "0210e5aa05f0e3c69d6f2022c539c9dc90cce322",
        "handoff transfer commit",
    )
    rollback = _require_object(handoff.get("rollback_exception"), "handoff rollback exception")
    _expect(rollback.get("best_local_payload_preserved"), False, "handoff payload exception")
    _expect(rollback.get("runtime_preserved"), False, "handoff runtime exception")
    _expect(
        rollback.get("reproducibility_capsule_status"),
        "DEGRADED_BY_PRIOR_BROAD_CLEANSE",
        "handoff capsule exception",
    )
    return _SemanticResult(
        artifact_paths=tuple(artifact_paths),
        document_paths=tuple(FROZEN_RAW_DOCUMENT_SHA256),
        facts={
            "terminal_outcome": "OUTCOME_C",
            "best_candidate": "P1_DUAL_PATH_RECOVERY_R16X2",
            "best_complete_bpw": float(bpw),
            "f2_promotable": False,
            "causal_diagnosis": diagnosis["diagnosis"],
            "evidence_artifact_count": len(_KIMI_CHAIN),
            "partial_prerequisite_evidence_preserved": True,
            "next_parent_action": "NATIVE_STUDENT_TEACHER_HIDDEN_FIRST",
        },
        scope={
            "proves": "sealed Kimi scientific closure and transfer evidence",
            "does_not_prove": [
                "retained Kimi runtime or payload",
                "GLM-5.2 body execution",
                "GLM-5.2 capability",
            ],
        },
    )


def _validate_kimi_release(reader: _Reader) -> _SemanticResult:
    release_path = "KIMI_K26_SOURCE_RELEASE_FOR_GLM52.json"
    final_path = "KIMI_K26_GRAVITY_FINAL.json"
    manifest_path = "reports/condense/kimi_k26/KIMI_K26_OFFICIAL_MANIFEST.json"
    remediation_path = "KIMI_K26_CREDENTIAL_REMEDIATION_FOR_GLM52.json"
    cleanse_path = "KIMI_K26_DEVICE_CLEANSE_FINAL.json"
    release = reader.sealed(
        release_path,
        schema="hawking.kimi_k26.source_release_for_glm52.v1",
        status="RECONCILED_ALREADY_RELEASED",
    )
    final = reader.sealed(
        final_path, schema="hawking.kimi_k26.gravity_final.v1", status="CLOSED"
    )
    manifest = reader.sealed(
        manifest_path, schema="hawking.kimi_k26.official_manifest.v1", status=None
    )
    remediation = reader.sealed(
        remediation_path,
        schema="hawking.kimi_k26.credential_remediation_for_glm52.v1",
        status="SANITIZED_CURRENT_TREE_ROTATION_PENDING",
    )
    cleanse = reader.sealed(
        cleanse_path,
        schema="hawking.kimi_k26.device_cleanse.final.v1",
        status=None,
    )
    _expect(cleanse.get("state"), "CLEANSE_COMPLETE", "Kimi cleanse state")
    closure = _require_object(release.get("closure"), "Kimi release closure")
    _expect(closure.get("terminal_outcome"), "OUTCOME_C", "Kimi release outcome")
    _expect(closure.get("final_seal_sha256"), final["seal_sha256"], "Kimi final binding")
    _expect(
        closure.get("byte_auction_seal_sha256"),
        FROZEN_ARTIFACT_SEALS["KIMI_K26_FINAL_BYTE_AUCTION.json"],
        "Kimi byte-auction binding",
    )
    source = _require_object(release.get("source"), "Kimi release source")
    for key, expected in (
        ("repo", KIMI_REPO), ("revision", KIMI_REVISION),
        ("manifest_files", KIMI_FILE_COUNT), ("weight_shards", KIMI_WEIGHT_SHARDS),
        ("logical_bytes", KIMI_SOURCE_LOGICAL_BYTES), ("weight_bytes", KIMI_WEIGHT_BYTES),
        ("exists_now", False),
    ):
        _expect(source.get(key), expected, f"Kimi release source {key}")
    _expect(source.get("manifest_seal_sha256"), manifest["seal_sha256"], "Kimi manifest binding")
    _expect(manifest.get("repo"), KIMI_REPO, "Kimi manifest repo")
    _expect(manifest.get("sha"), KIMI_REVISION, "Kimi manifest revision")
    _expect(manifest.get("file_count"), KIMI_FILE_COUNT, "Kimi manifest files")
    _expect(manifest.get("total_bytes"), KIMI_SOURCE_LOGICAL_BYTES, "Kimi manifest bytes")
    _expect(manifest.get("weight_shards"), KIMI_WEIGHT_SHARDS, "Kimi manifest shards")
    _expect(manifest.get("weight_bytes"), KIMI_WEIGHT_BYTES, "Kimi manifest weight bytes")
    _expect(
        release.get("credential_remediation_seal_sha256"),
        remediation["seal_sha256"],
        "Kimi credential remediation binding",
    )
    _expect(
        release.get("release_timing"),
        "COMPLETED_BEFORE_THIS_GLM52_GOAL",
        "Kimi release timing",
    )
    _expect(release.get("new_deletion_performed_by_this_receipt"), False, "Kimi new deletion")
    live = _require_object(release.get("live_absence"), "Kimi live absence")
    for key in ("source_absent", "runtime_absent", "installed_plist_absent"):
        _expect(live.get(key), True, f"Kimi live absence {key}")
    _expect(live.get("queue_or_outbox_possible"), False, "Kimi queue/outbox")
    process = _require_object(live.get("process_audit"), "Kimi process audit")
    _expect(process.get("matching_process_count"), 0, "Kimi matching processes")
    live_observation = _stable_absolute_absence(source.get("former_root"))
    runtime_observation = _stable_absolute_absence(
        "/Users/scammermike/Library/Application Support/Hawking/KimiK26"
    )
    plist_observation = _stable_absolute_absence(
        "/Users/scammermike/Library/LaunchAgents/"
        "com.hawking.kimi-k26-doctor-prime.plist"
    )
    allocated = _require_object(
        release.get("allocated_byte_reconciliation"), "Kimi allocated reconciliation"
    )
    _expect(
        allocated.get("exact_kimi_k2_6_source_root_allocated_bytes"),
        KIMI_SOURCE_ALLOCATED_BYTES,
        "Kimi exact allocated bytes",
    )
    _expect(
        allocated.get("total_cleanup_free_space_delta_bytes"),
        KIMI_TOTAL_CLEANUP_DELTA_BYTES,
        "Kimi cleanup delta",
    )
    _expect(
        allocated.get("total_delta_includes_runtime_and_other_targets"),
        True,
        "Kimi delta scope",
    )
    rollback = _require_object(release.get("rollback_boundary"), "Kimi rollback boundary")
    expected_rollback = {
        "best_local_payload_preserved": False,
        "runtime_preserved": False,
        "scientific_reports_preserved": True,
        "implementation_recoverable_from_git_history": True,
        "reproducibility_capsule_status": "DEGRADED_BY_PRIOR_BROAD_CLEANSE",
    }
    for key, expected in expected_rollback.items():
        _expect(rollback.get(key), expected, f"Kimi rollback {key}")
    archive = _require_object(remediation.get("archive"), "Kimi credential archive")
    rotation = _require_object(archive.get("rotation"), "Kimi credential rotation")
    _expect(rotation.get("confirmed"), False, "Kimi credential rotation")
    _expect(rotation.get("status"), "REQUIRED_EXTERNAL_ACTION", "Kimi rotation status")
    publication = _require_object(
        remediation.get("publication_gate"), "Kimi publication gate"
    )
    _expect(publication.get("telegram_delivery_allowed"), False, "Kimi Telegram gate")
    cleanse_verification = _require_object(
        cleanse.get("verification"), "Kimi cleanse verification"
    )
    _expect(cleanse_verification.get("mop_touched"), False, "Kimi MOP boundary")
    cleanse_deleted = _require_object(cleanse.get("deleted"), "Kimi cleanse deleted")
    _expect(
        cleanse_deleted.get("kimi_k2_6_source_allocated_bytes"),
        KIMI_SOURCE_ALLOCATED_BYTES,
        "Kimi cleanse source allocation",
    )
    _expect(
        cleanse_deleted.get("kimi_runtime_allocated_bytes"),
        2_310_508_544,
        "Kimi cleanse runtime allocation",
    )
    _expect(
        cleanse_deleted.get("runtime_credentials_logs_captures_payloads_and_checkpoints"),
        True,
        "Kimi broad runtime deletion",
    )
    return _SemanticResult(
        artifact_paths=(
            release_path, final_path, manifest_path, remediation_path, cleanse_path
        ),
        document_paths=(),
        facts={
            "source_repo": KIMI_REPO,
            "source_revision": KIMI_REVISION,
            "source_logical_bytes": KIMI_SOURCE_LOGICAL_BYTES,
            "source_allocated_bytes": KIMI_SOURCE_ALLOCATED_BYTES,
            "total_cleanup_delta_bytes": KIMI_TOTAL_CLEANUP_DELTA_BYTES,
            "source_absence": live_observation,
            "runtime_absence": runtime_observation,
            "launchd_plist_absence": plist_observation,
            "release_preceded_glm_goal": True,
            "new_deletion_by_release_receipt": False,
            "rollback_payload_preserved": False,
            "rollback_runtime_preserved": False,
            "rollback_capsule_status": "DEGRADED_BY_PRIOR_BROAD_CLEANSE",
            "scientific_reports_preserved": True,
            "credential_rotation_pending": True,
            "git_history_purge_pending": True,
            "telegram_delivery_allowed": False,
            "mop_touched": False,
            "runtime_allocated_bytes_deleted": 2_310_508_544,
            "broad_runtime_payload_capture_checkpoint_deletion": True,
        },
        scope={
            "proves": "current Kimi raw-source absence and retrospective accounting",
            "exceptions_carried": [
                "best local payload was not preserved",
                "runtime was not preserved",
                "rollback capsule is degraded",
                "credential rotation and history coordination remain pending",
            ],
            "does_not_prove": [
                "Part-I-compliant safe release",
                "pre-delete exact-path/realpath/MOP gates",
                "pre-delete mapped-reader and queue absence",
                "healthy Kimi rollback",
                "Telegram readiness",
            ],
            "remediation_required": [
                "rehydrate the exact immutable Kimi source",
                "reconstruct and preserve the best payload and runtime capsule",
                "repeat exact-path, resolved-realpath, outside-MOP, no-reader, "
                "no-queue, and official-rehydration gates",
                "release only the exact dependency-closed source inventory",
                "ground the exact allocated bytes recovered by that release",
            ],
        },
        proven=False,
        blockers=(
            "MISSING_GROUNDED_PREDELETE_EXACT_PATH_REALPATH_MOP_AUDIT",
            "MISSING_GROUNDED_PREDELETE_READER_AND_QUEUE_AUDIT",
            "REPRODUCIBILITY_CAPSULE_DEGRADED_PAYLOAD_AND_RUNTIME_ABSENT",
            "RETROSPECTIVE_AGGREGATE_DELTA_NOT_EXACT_RELEASE_RECOVERY",
        ),
    )


def _validate_revision(reader: _Reader) -> _SemanticResult:
    admission_path = "GLM52_SOURCE_ADMISSION.json"
    manifest_path = "GLM52_OFFICIAL_MANIFEST.json"
    architecture_path = "GLM52_ARCHITECTURE_CONTRACT.json"
    ledger_path = "GLM52_LOGICAL_WEIGHT_LEDGER.json"
    format_path = "GLM52_SOURCE_FORMAT_LEDGER.json"
    admission = reader.sealed(
        admission_path,
        schema="hawking.glm52.source_admission.v1",
        status="ADMITTED_CONTROL_PLANE_HEADERS_AND_PLAN_BODY_PENDING",
    )
    manifest = reader.sealed(
        manifest_path,
        schema="hawking.glm52.official_manifest.v1",
        status="PASS_CONTROL_PLANE_AND_HEADERS_BODY_PENDING",
    )
    architecture = reader.sealed(
        architecture_path,
        schema="hawking.glm52.architecture_contract.v1",
        status="PASS_CONFIG_INDEX_AND_HEADERS",
    )
    ledger = reader.sealed(
        ledger_path,
        schema="hawking.glm52.logical_weight_ledger.v1",
        status="PASS_HEADER_DERIVED",
    )
    formats = reader.sealed(
        format_path,
        schema="hawking.glm52.source_format_ledger.v1",
        status="PASS_HEADER_DERIVED_BODY_PENDING",
    )
    for artifact, label in (
        (admission, "admission"), (manifest, "manifest"),
        (architecture, "architecture"), (ledger, "weight ledger"),
        (formats, "format ledger"),
    ):
        _expect(artifact.get("repo"), OFFICIAL_REPO, f"{label} repo")
        _expect(artifact.get("revision"), OFFICIAL_REVISION, f"{label} revision")
    gates = _require_object(admission.get("admission_gates"), "admission gates")
    _expect(gates.get("immutable_revision"), True, "immutable revision gate")
    _expect(admission.get("main_resolved_live"), OFFICIAL_REVISION, "resolved main")
    _expect(
        admission.get("main_matches_pinned_revision_at_admission"),
        True,
        "main revision binding",
    )
    _expect(admission.get("experiment_identity_uses_main"), False, "experiment identity")
    license_value = _require_object(admission.get("license"), "admission license")
    _expect(license_value.get("spdx"), OFFICIAL_LICENSE, "official license")
    _expect(license_value.get("sha256"), OFFICIAL_LICENSE_SHA256, "license hash")
    tree_url = manifest.get("immutable_tree_url")
    if not isinstance(tree_url, str) or not tree_url.endswith(f"/tree/{OFFICIAL_REVISION}"):
        _fail("manifest immutable tree URL is not revision pinned")
    evidence = _require_object(admission.get("evidence"), "admission evidence")
    _expect(
        evidence,
        {
            "architecture_contract_seal_sha256": architecture["seal_sha256"],
            "dependency_graph_seal_sha256": FROZEN_ARTIFACT_SEALS[
                "GLM52_SHARD_DEPENDENCY_GRAPH.json"
            ],
            "logical_weight_ledger_seal_sha256": ledger["seal_sha256"],
            "official_manifest_seal_sha256": manifest["seal_sha256"],
            "source_format_ledger_seal_sha256": formats["seal_sha256"],
            "streaming_schedule_seal_sha256": FROZEN_ARTIFACT_SEALS[
                "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json"
            ],
        },
        "admission dependent seals",
    )
    rows, _ = _manifest_rows(manifest)
    license_row = next((row for row in rows if row.get("path") == "LICENSE"), None)
    license_row = _require_object(license_row, "manifest LICENSE row")
    _expect(license_row.get("logical_bytes"), 1065, "LICENSE bytes")
    license_local = _require_object(license_row.get("local"), "manifest LICENSE local")
    _expect(license_local.get("sha256"), OFFICIAL_LICENSE_SHA256, "manifest LICENSE hash")
    _expect(license_local.get("state"), "PRESENT_VERIFIED", "manifest LICENSE state")
    return _SemanticResult(
        artifact_paths=(
            admission_path, manifest_path, architecture_path, ledger_path, format_path
        ),
        document_paths=(),
        facts={
            "repo": OFFICIAL_REPO,
            "immutable_revision": OFFICIAL_REVISION,
            "license_spdx": OFFICIAL_LICENSE,
            "license_sha256": OFFICIAL_LICENSE_SHA256,
            "immutable_tree_url": tree_url,
            "body_shards_verified": 0,
        },
        scope={
            "proves": "official repository, immutable revision, and license identity",
            "does_not_prove": ["weight-body verification", "BF16 reference forward"],
        },
    )


def _manifest_rows(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = _require_list(manifest.get("files"), "official manifest files")
    if any(not isinstance(row, dict) for row in files):
        _fail("official manifest contains a non-object row")
    rows = [dict(row) for row in files]
    paths = [row.get("path") for row in rows]
    if any(not isinstance(path, str) or not path for path in paths) \
            or len(set(paths)) != len(paths):
        _fail("official manifest paths are invalid or duplicated")
    weights = [row for row in rows if row.get("is_weight") is True]
    return rows, weights


def _validate_manifest(reader: _Reader) -> _SemanticResult:
    manifest_path = "GLM52_OFFICIAL_MANIFEST.json"
    format_path = "GLM52_SOURCE_FORMAT_LEDGER.json"
    admission_path = "GLM52_SOURCE_ADMISSION.json"
    manifest = reader.sealed(
        manifest_path,
        schema="hawking.glm52.official_manifest.v1",
        status="PASS_CONTROL_PLANE_AND_HEADERS_BODY_PENDING",
    )
    formats = reader.sealed(
        format_path,
        schema="hawking.glm52.source_format_ledger.v1",
        status="PASS_HEADER_DERIVED_BODY_PENDING",
    )
    admission = reader.sealed(
        admission_path,
        schema="hawking.glm52.source_admission.v1",
        status="ADMITTED_CONTROL_PLANE_HEADERS_AND_PLAN_BODY_PENDING",
    )
    for artifact, label in ((manifest, "manifest"), (formats, "format ledger")):
        _expect(artifact.get("repo"), OFFICIAL_REPO, f"{label} repo")
        _expect(artifact.get("revision"), OFFICIAL_REVISION, f"{label} revision")
    for key, expected in (
        ("file_count", OFFICIAL_FILE_COUNT),
        ("weight_shards", OFFICIAL_WEIGHT_SHARDS),
        ("source_logical_bytes", OFFICIAL_SOURCE_LOGICAL_BYTES),
        ("weight_container_logical_bytes", OFFICIAL_WEIGHT_CONTAINER_BYTES),
        ("nonweight_logical_bytes", OFFICIAL_NONWEIGHT_BYTES),
    ):
        _expect(manifest.get(key), expected, f"manifest {key}")
    rows, weights = _manifest_rows(manifest)
    _expect(len(rows), OFFICIAL_FILE_COUNT, "manifest row count")
    _expect(len(weights), OFFICIAL_WEIGHT_SHARDS, "manifest weight-row count")
    expected_names = {
        f"model-{index:05d}-of-00282.safetensors"
        for index in range(1, OFFICIAL_WEIGHT_SHARDS + 1)
    }
    _expect({row.get("path") for row in weights}, expected_names, "weight-shard names")
    _expect(
        sum(int(row.get("logical_bytes", -1)) for row in rows),
        OFFICIAL_SOURCE_LOGICAL_BYTES,
        "manifest row byte sum",
    )
    _expect(
        sum(int(row.get("logical_bytes", -1)) for row in weights),
        OFFICIAL_WEIGHT_CONTAINER_BYTES,
        "manifest weight byte sum",
    )
    for row in weights:
        path = str(row.get("path"))
        _expect(row.get("download_state"), "NOT_FETCHED", f"{path} download state")
        _expect(
            row.get("verification_state"),
            "HEADER_VERIFIED_BODY_NOT_FETCHED",
            f"{path} verification boundary",
        )
        local = _require_object(row.get("local"), f"{path} local state")
        _expect(
            local.get("state"),
            "HEADER_ONLY_REMOTE_RANGE_NO_LOCAL_WEIGHT_BODY",
            f"{path} local state",
        )
        _expect(local.get("logical_bytes"), 0, f"{path} local bytes")
        _expect(local.get("sha256"), None, f"{path} local SHA")
        if not _is_sha256(row.get("lfs_sha256")) or not _is_sha256(row.get("xet_hash")):
            _fail(f"{path} lacks frozen LFS/Xet content identity")
    boundary = _require_object(formats.get("verification_boundary"), "format boundary")
    _expect(boundary.get("headers_fetched_and_verified"), True, "header verification")
    _expect(boundary.get("tensor_payload_bodies_fetched"), False, "body fetched claim")
    _expect(boundary.get("tensor_payload_sha256_verified"), False, "body hash claim")
    _expect(formats.get("weight_shards"), OFFICIAL_WEIGHT_SHARDS, "format shards")
    _expect(formats.get("tensor_count"), OFFICIAL_TENSOR_COUNT, "format tensors")
    _expect(formats.get("tensor_payload_bytes"), OFFICIAL_TENSOR_PAYLOAD_BYTES, "format payload")
    dtype = _require_object(formats.get("dtype_summary"), "dtype summary")
    _expect(
        dtype,
        {
            "BF16": {
                "logical_weights": OFFICIAL_BF16_LOGICAL_WEIGHTS,
                "source_payload_bytes": OFFICIAL_BF16_PAYLOAD_BYTES,
                "tensor_count": OFFICIAL_BF16_TENSOR_COUNT,
            },
            "F32": {
                "logical_weights": OFFICIAL_F32_LOGICAL_WEIGHTS,
                "source_payload_bytes": OFFICIAL_F32_PAYLOAD_BYTES,
                "tensor_count": OFFICIAL_F32_TENSOR_COUNT,
            },
        },
        "source dtype summary",
    )
    gates = _require_object(admission.get("admission_gates"), "admission gates")
    _expect(gates.get("official_file_manifest_complete"), True, "manifest gate")
    _expect(gates.get("body_stream"), False, "body stream gate")
    return _SemanticResult(
        artifact_paths=(manifest_path, format_path, admission_path),
        document_paths=(),
        facts={
            "repo": OFFICIAL_REPO,
            "revision": OFFICIAL_REVISION,
            "official_files": OFFICIAL_FILE_COUNT,
            "weight_shards": OFFICIAL_WEIGHT_SHARDS,
            "source_logical_bytes": OFFICIAL_SOURCE_LOGICAL_BYTES,
            "weight_container_bytes": OFFICIAL_WEIGHT_CONTAINER_BYTES,
            "tensor_payload_bytes": OFFICIAL_TENSOR_PAYLOAD_BYTES,
            "tensor_count": OFFICIAL_TENSOR_COUNT,
            "bf16_tensor_count": OFFICIAL_BF16_TENSOR_COUNT,
            "headers_verified": True,
            "body_shards_fetched": 0,
            "body_shards_sha256_verified": 0,
            "body_verification_complete": False,
        },
        scope={
            "proves": "complete immutable remote file manifest and verified tensor headers",
            "manifest_completeness_is_not_body_verification": True,
            "body_verification_belongs_to_stop":
                "every_official_bf16_shard_fetched_verified",
            "does_not_prove": ["any complete shard body", "BF16 teacher execution"],
        },
    )


def _validate_weight_ledger(reader: _Reader) -> _SemanticResult:
    ledger_path = "GLM52_LOGICAL_WEIGHT_LEDGER.json"
    format_path = "GLM52_SOURCE_FORMAT_LEDGER.json"
    architecture_path = "GLM52_ARCHITECTURE_CONTRACT.json"
    ledger = reader.sealed(
        ledger_path,
        schema="hawking.glm52.logical_weight_ledger.v1",
        status="PASS_HEADER_DERIVED",
    )
    formats = reader.sealed(
        format_path,
        schema="hawking.glm52.source_format_ledger.v1",
        status="PASS_HEADER_DERIVED_BODY_PENDING",
    )
    architecture = reader.sealed(
        architecture_path,
        schema="hawking.glm52.architecture_contract.v1",
        status="PASS_CONFIG_INDEX_AND_HEADERS",
    )
    for artifact, label in (
        (ledger, "weight ledger"), (formats, "format ledger"),
        (architecture, "architecture"),
    ):
        _expect(artifact.get("repo"), OFFICIAL_REPO, f"{label} repo")
        _expect(artifact.get("revision"), OFFICIAL_REVISION, f"{label} revision")
    for key, expected in (
        ("tensor_count", OFFICIAL_TENSOR_COUNT),
        ("logical_weight_denominator", OFFICIAL_LOGICAL_WEIGHTS),
        ("source_payload_bytes", OFFICIAL_TENSOR_PAYLOAD_BYTES),
    ):
        _expect(ledger.get(key), expected, f"weight ledger {key}")
    _expect(
        ledger.get("source_dtype_summary"),
        formats.get("dtype_summary"),
        "ledger dtype cross-binding",
    )
    architecture_weights = _require_object(
        architecture.get("weights"), "architecture weights"
    )
    _expect(
        architecture_weights.get("logical_elements"),
        OFFICIAL_LOGICAL_WEIGHTS,
        "architecture logical weights",
    )
    _expect(
        architecture_weights.get("tensor_count"),
        OFFICIAL_TENSOR_COUNT,
        "architecture tensor count",
    )
    _expect(
        architecture_weights.get("dtype_summary"),
        formats.get("dtype_summary"),
        "architecture dtype cross-binding",
    )
    views = _require_object(ledger.get("major_accounting_views"), "accounting views")
    all_weights = _require_object(
        views.get("all_declared_model_weights"), "all-declared view"
    )
    _expect(
        all_weights,
        {
            "logical_weights": OFFICIAL_LOGICAL_WEIGHTS,
            "source_payload_bytes": OFFICIAL_TENSOR_PAYLOAD_BYTES,
            "tensor_count": OFFICIAL_TENSOR_COUNT,
        },
        "all-declared accounting view",
    )
    _expect(
        _require_object(ledger.get("mtp_policy"), "MTP policy").get(
            "included_in_complete_denominator"
        ),
        True,
        "MTP denominator inclusion",
    )
    alias = _require_object(ledger.get("alias_policy"), "alias policy")
    _expect(alias.get("stored_aliases"), 0, "stored aliases")
    _expect(alias.get("every_stored_tensor_counted_once"), True, "tensor count-once policy")
    budgets = _require_object(ledger.get("rate_budgets"), "rate budgets")
    expected_budgets = {
        "hard_1_bpw": OFFICIAL_LOGICAL_WEIGHTS // 8,
        "planned_0_98_bpw": OFFICIAL_LOGICAL_WEIGHTS * 49 // (50 * 8),
        "rate_0_75_bpw": OFFICIAL_LOGICAL_WEIGHTS * 3 // (4 * 8),
        "rate_0_50_bpw": OFFICIAL_LOGICAL_WEIGHTS // (2 * 8),
        "rate_0_33_represented_as_one_third_bpw": OFFICIAL_LOGICAL_WEIGHTS // (3 * 8),
        "rate_0_25_bpw": OFFICIAL_LOGICAL_WEIGHTS // (4 * 8),
    }
    for name, expected_bytes in expected_budgets.items():
        budget = _require_object(budgets.get(name), f"rate budget {name}")
        _expect(
            budget.get("maximum_complete_physical_bytes"),
            expected_bytes,
            f"rate budget {name} bytes",
        )
    return _SemanticResult(
        artifact_paths=(ledger_path, format_path, architecture_path),
        document_paths=(),
        facts={
            "repo": OFFICIAL_REPO,
            "revision": OFFICIAL_REVISION,
            "logical_weight_denominator": OFFICIAL_LOGICAL_WEIGHTS,
            "tensor_count": OFFICIAL_TENSOR_COUNT,
            "bf16_logical_weights": OFFICIAL_BF16_LOGICAL_WEIGHTS,
            "f32_router_bias_logical_weights": OFFICIAL_F32_LOGICAL_WEIGHTS,
            "tensor_payload_bytes": OFFICIAL_TENSOR_PAYLOAD_BYTES,
            "mtp_included": True,
            "stored_aliases": 0,
        },
        scope={
            "proves": "exact header-derived complete logical-weight denominator",
            "does_not_prove": ["weight-body hashes", "physical compact artifact size"],
        },
    )


def _validate_pre_audit(reader: _Reader) -> _SemanticResult:
    path = "GRAVITY_COMPLETENESS_AUDIT_GLM52_PRE.json"
    audit = reader.sealed(
        path,
        schema="hawking.gravity_completeness_audit.glm52_pre.v1",
        status="PASS_FROZEN_PRE_CAMPAIGN_BASELINE",
    )
    _expect(tuple(audit.get("axes", [])), _PRE_AUDIT_AXES, "pre-audit axes")
    snapshot = _require_object(audit.get("snapshot"), "pre-audit snapshot")
    _expect(snapshot.get("branch"), "campaign/glm52-bf16-xet-gravity", "pre-audit branch")
    _expect(
        snapshot.get("repository_head"),
        "8d634af9702aa965728f057661b5d1fad1883f45",
        "pre-audit repository head",
    )
    _expect(
        snapshot.get("at_utc"),
        "2026-07-21T20:29:51.121903Z",
        "pre-audit timestamp",
    )
    _expect(snapshot.get("later_glm_artifacts_excluded_from_scores"), True, "pre-audit freeze")
    if "before GLM source admission" not in str(snapshot.get("boundary")):
        _fail("pre-audit boundary is not pre-campaign")
    scores = _require_object(audit.get("scores"), "pre-audit scores")
    _expect(
        set(scores),
        {"GLM52_PRE", "GPT_OSS_120B", "KIMI_K26", "QWEN3_235B"},
        "pre-audit score inventory",
    )
    for model, row_value in scores.items():
        row = _require_object(row_value, f"pre-audit score {model}")
        axes = _require_object(row.get("axes"), f"pre-audit score axes {model}")
        _expect(set(axes), set(_PRE_AUDIT_AXES), f"pre-audit score axes {model}")
        total = sum(axes.values())
        _expect(row.get("total"), total, f"pre-audit total {model}")
        _expect(row.get("maximum"), len(_PRE_AUDIT_AXES) * 5, f"pre-audit maximum {model}")
        if not math.isclose(float(row.get("mean")), total / len(_PRE_AUDIT_AXES)):
            _fail(f"pre-audit mean mismatch for {model}")
    _expect(scores["GLM52_PRE"].get("total"), 16, "GLM pre-audit total")
    honest = _require_object(audit.get("honest_status"), "pre-audit honest status")
    not_proven = _require_list(honest.get("not_proven_at_entry"), "pre-audit unproven list")
    if not any("any GLM-5.2 scientific result" in str(item) for item in not_proven):
        _fail("pre-audit does not preserve the no-GLM-result boundary")
    evidence = _require_object(audit.get("evidence"), "pre-audit evidence")
    document_paths: list[str] = []
    evidence_rows = 0
    for model, rows_value in evidence.items():
        rows = _require_list(rows_value, f"pre-audit evidence {model}")
        for index, row_value in enumerate(rows):
            row = _require_object(row_value, f"pre-audit evidence {model}[{index}]")
            evidence_path = row.get("path")
            if not isinstance(evidence_path, str):
                _fail("pre-audit evidence path is invalid")
            raw = reader.raw(evidence_path)
            _expect(len(raw), row.get("bytes"), f"pre-audit evidence bytes {evidence_path}")
            _expect(
                hashlib.sha256(raw).hexdigest(),
                row.get("sha256"),
                f"pre-audit evidence hash {evidence_path}",
            )
            if not _require_list(row.get("fields"), f"pre-audit fields {evidence_path}"):
                _fail(f"pre-audit evidence has no field scope: {evidence_path}")
            document_paths.append(evidence_path)
            evidence_rows += 1
    _expect(evidence_rows, 18, "pre-audit evidence row count")
    return _SemanticResult(
        artifact_paths=(path,),
        document_paths=tuple(document_paths),
        facts={
            "audit_boundary": "FROZEN_PRE_CAMPAIGN",
            "axis_count": len(_PRE_AUDIT_AXES),
            "glm52_pre_total": 16,
            "glm52_pre_maximum": 105,
            "evidence_rows": evidence_rows,
            "later_glm_artifacts_excluded": True,
            "glm_scientific_result_at_entry": False,
        },
        scope={
            "proves": "frozen pre-campaign Gravity maturity baseline",
            "does_not_prove": ["post-campaign maturity", "any GLM scientific result"],
        },
    )


def _validate_external_matrix(reader: _Reader) -> _SemanticResult:
    path = "GRAVITY_EXTERNAL_BASELINE_MATRIX.json"
    matrix = reader.sealed(
        path,
        schema="hawking.gravity_external_baseline_matrix.v1",
        status="PASS_PRIMARY_SOURCE_COMPARISON",
    )
    methods = _require_list(matrix.get("methods"), "external methods")
    _expect(len(methods), 13, "external method count")
    _expect(matrix.get("research_cutoff"), "2026-07-21", "external research cutoff")
    names: list[str] = []
    paper_sources = 0
    code_sources = 0
    for index, method_value in enumerate(methods):
        method = _require_object(method_value, f"external method {index}")
        name = method.get("method")
        if not isinstance(name, str) or not name:
            _fail(f"external method {index} has no name")
        names.append(name)
        for key in ("structured_comparison", "rate", "glm_comparability"):
            if key not in method:
                _fail(f"external method {name} omits {key}")
        sources = _require_list(method.get("sources"), f"external sources {name}")
        if not sources:
            _fail(f"external method {name} has no primary source")
        for source_value in sources:
            source = _require_object(source_value, f"external source {name}")
            kind = source.get("kind")
            if kind == "paper":
                identity = _require_object(
                    source.get("content_identity"), f"paper identity {name}"
                )
                if not _is_sha256(identity.get("sha256")):
                    _fail(f"external paper source lacks content hash: {name}")
                paper_sources += 1
            elif kind == "code":
                commit = source.get("commit")
                if not isinstance(commit, str) or _SHA1_RE.fullmatch(commit) is None:
                    _fail(f"external code source is not commit pinned: {name}")
                code_sources += 1
            else:
                _fail(f"external source kind is unsupported: {kind!r}")
    if len(set(names)) != len(names):
        _fail("external matrix contains duplicate methods")
    _expect(
        names,
        [
            "BitNet b1.58 / b1.58 2B4T", "ParetoQ", "QuIP", "QuIP#",
            "AQLM", "VPTQ", "GPTVQ", "BiLLM", "STBLLM", "QMoE",
            "BTC-LLM", "NanoQuant", "LittleBit",
        ],
        "external method inventory",
    )
    qmoe = next(method for method in methods if method.get("method") == "QMoE")
    _expect(
        _require_object(qmoe.get("rate"), "QMoE rate").get(
            "canonical_artifact_bpw"
        ),
        0.807,
        "QMoE canonical artifact BPW",
    )
    if "closest prior" not in str(qmoe.get("glm_comparability")):
        _fail("QMoE is not retained as the closest giant-MoE prior")
    taxonomy = _require_object(matrix.get("rate_taxonomy"), "rate taxonomy")
    _expect(
        set(taxonomy),
        {
            "canonical_artifact_bpw", "decoded_tensor_payload_bpw",
            "nominal_or_method_bpw", "ranking_rule",
        },
        "rate taxonomy levels",
    )
    binding = _require_object(matrix.get("instrument_binding"), "matrix instrument")
    source_paths = (
        "tools/condense/glm52_external_baselines.py",
        "tools/condense/glm52_common.py",
    )
    _expect(
        hashlib.sha256(reader.raw(source_paths[0])).hexdigest(),
        binding.get("generator_sha256"),
        "external matrix generator hash",
    )
    _expect(
        hashlib.sha256(reader.raw(source_paths[1])).hexdigest(),
        binding.get("common_sha256"),
        "external matrix common hash",
    )
    _expect(binding.get("timestamp_free_deterministic_rebuild"), True, "matrix rebuild")
    claim_policy = _require_object(matrix.get("claim_policy"), "matrix claim policy")
    unsafe = _require_list(claim_policy.get("unsafe"), "matrix unsafe claims")
    if not any("First sub-1-bit PTQ" in str(item) for item in unsafe):
        _fail("external matrix omits its principal unsafe novelty claim")
    return _SemanticResult(
        artifact_paths=(path,),
        document_paths=source_paths,
        facts={
            "method_count": len(methods),
            "paper_source_count": paper_sources,
            "commit_pinned_code_source_count": code_sources,
            "primary_source_only": True,
            "rate_taxonomy_levels_distinguished": True,
            "qmoe_canonical_artifact_bpw": 0.807,
            "matched_cross_paper_leaderboard_claimed": False,
        },
        scope={
            "proves": "primary-source external method and accounting comparison",
            "does_not_prove": [
                "GLM result superiority",
                "matched model/hardware comparison",
                "campaign capability",
            ],
        },
    )


def _validate_adapter(reader: _Reader) -> _SemanticResult:
    path = "GLM52_ADAPTER_TWIN.json"
    parity_path = "GLM52_REFERENCE_PARITY.json"
    manifest_path = "GLM52_OFFICIAL_MANIFEST.json"
    twin = reader.sealed(
        path,
        schema="hawking.glm52.adapter_twin.v1",
        status="PASS_SYNTHETIC_TWIN_AND_OFFICIAL_HEADER_TOKENIZER_SCHEMA",
    )
    manifest = reader.sealed(
        manifest_path,
        schema="hawking.glm52.official_manifest.v1",
        status="PASS_CONTROL_PLANE_AND_HEADERS_BODY_PENDING",
    )
    parity = reader.sealed(
        parity_path,
        schema="hawking.glm52.reference_parity.v1",
        status="PASS_SYNTHETIC_MAIN_AND_MTP_SELF_CONSISTENCY_SOURCE_PARENT_PENDING",
    )
    binding = _require_object(twin.get("binding"), "adapter binding")
    _expect(binding.get("official_repo"), OFFICIAL_REPO, "adapter repo")
    _expect(binding.get("official_revision"), OFFICIAL_REVISION, "adapter revision")
    _expect(
        binding.get("source_class"),
        ["OFFICIAL_BF16_TEACHER", "VULTURE_XET_STREAMING", "TEXT_GENERATION"],
        "adapter source class",
    )
    _expect(twin.get("source_parent_parity_claimed"), False, "adapter parent parity")
    pending_reason = twin.get("source_parent_parity_pending_reason")
    if not isinstance(pending_reason, str) or "No BF16 payload shard" not in pending_reason:
        _fail("adapter twin does not disclose absent BF16 body parity")
    checks = _require_object(twin.get("checks"), "adapter checks")
    for key in (
        "expert_gate_then_up", "hardlinked_core_shards",
        "main_only_is_exact_core_filter", "mtp_has_full_indexer",
        "shared_layers_have_no_indexer_tensors",
    ):
        _expect(checks.get(key), True, f"adapter check {key}")
    sweep = _require_object(checks.get("official_schema_sweep"), "adapter schema sweep")
    for key, expected in (
        ("status", "PASS"), ("actual_tensor_count", OFFICIAL_TENSOR_COUNT),
        ("expected_tensor_count", OFFICIAL_TENSOR_COUNT),
        ("actual_logical_elements", OFFICIAL_LOGICAL_WEIGHTS),
        ("expected_logical_elements", OFFICIAL_LOGICAL_WEIGHTS),
    ):
        _expect(sweep.get(key), expected, f"adapter sweep {key}")
    for key in ("missing", "unknown", "shape_mismatches", "dtype_mismatches"):
        _expect(sweep.get(key), [], f"adapter sweep {key}")
    streaming = _require_object(
        checks.get("streaming_window_admission"), "adapter streaming admission"
    )
    _expect(streaming.get("status"), "PASS", "adapter streaming admission")
    fixture = _require_object(twin.get("synthetic_fixture"), "adapter fixture")
    _expect(fixture.get("profile"), "synthetic", "adapter fixture profile")
    _expect(fixture.get("deterministic"), True, "adapter determinism")
    _expect(fixture.get("network_access"), False, "adapter network access")
    long_context = _require_object(
        fixture.get("long_context_indexer_contract"), "adapter long-context scope"
    )
    _expect(long_context.get("capability_claimed"), False, "adapter capability claim")
    instrument = _require_object(twin.get("instrument_binding"), "adapter instrument")
    source_hashes = _require_object(
        instrument.get("local_source_sha256"), "adapter source hashes"
    )
    source_paths = tuple(sorted(source_hashes))
    for source_path in source_paths:
        _expect(
            hashlib.sha256(reader.raw(source_path)).hexdigest(),
            source_hashes[source_path],
            f"adapter instrument hash {source_path}",
        )
    parity_instrument = _require_object(
        parity.get("instrument_binding"), "reference parity instrument"
    )
    _expect(
        parity_instrument.get("local_source_sha256"),
        source_hashes,
        "adapter/reference instrument identity",
    )
    _expect(
        parity.get("synthetic_fixture"),
        twin.get("synthetic_fixture"),
        "adapter/reference synthetic fixture",
    )
    claim_boundary = _require_object(
        parity.get("claim_boundary"), "reference claim boundary"
    )
    _expect(claim_boundary.get("capability"), "NOT_CLAIMED", "reference capability")
    _expect(
        claim_boundary.get("official_bf16_parent_forward"),
        "PENDING_FIRST_ADMITTED_SOURCE_WINDOW",
        "reference BF16 boundary",
    )
    main_metrics = _require_object(
        parity.get("official_transformers_main_vs_numpy_reference"),
        "reference main metrics",
    )
    _expect(main_metrics.get("status"), "PASS", "reference main status")
    thresholds = _require_object(main_metrics.get("thresholds"), "reference thresholds")
    if float(main_metrics.get("maximum_absolute_error")) \
            > float(thresholds.get("maximum_absolute_logit_error")) \
            or float(main_metrics.get("cosine")) \
            < float(thresholds.get("minimum_logit_cosine")) \
            or float(main_metrics.get("top1_agreement")) \
            < float(thresholds.get("minimum_top1_agreement")) \
            or float(main_metrics.get("relative_frobenius_error")) \
            > float(thresholds.get("relative_frobenius_logit_error")):
        _fail("reference main metrics exceed their frozen thresholds")
    _expect(
        parity.get("reference_deterministic_exact_replay"),
        True,
        "reference deterministic replay",
    )
    metal = _require_object(parity.get("cpu_vs_metal"), "reference Metal parity")
    _expect(metal.get("available"), True, "reference Metal availability")
    _expect(
        _require_object(metal.get("metrics"), "reference Metal metrics").get("status"),
        "PASS",
        "reference Metal status",
    )
    mtp = _require_object(parity.get("mtp"), "reference MTP")
    _expect(mtp.get("external_pinned_runtime_executed"), False, "external MTP runtime")
    _expect(mtp.get("transformers_parity_claimed"), False, "MTP Transformers parity")
    long_probe = _require_object(
        parity.get("long_context_indexer_shape_probe"), "reference long-context probe"
    )
    _expect(long_probe.get("full_attention_or_model_executed"), False, "1M execution")
    _expect(long_probe.get("one_million_context_capability_claimed"), False, "1M capability")
    files, _ = _manifest_rows(manifest)
    manifest_by_path = {row["path"]: row for row in files}
    tokenizer = _require_object(
        checks.get("official_tokenizer_chat_assembly"), "adapter tokenizer"
    )
    assets = _require_object(tokenizer.get("asset_sha256"), "adapter tokenizer assets")
    for asset, expected_hash in assets.items():
        row = _require_object(manifest_by_path.get(asset), f"manifest asset {asset}")
        local = _require_object(row.get("local"), f"manifest local asset {asset}")
        _expect(local.get("sha256"), expected_hash, f"tokenizer asset hash {asset}")
        _expect(local.get("state"), "PRESENT_VERIFIED", f"tokenizer asset state {asset}")
    return _SemanticResult(
        artifact_paths=(path, parity_path, manifest_path),
        document_paths=source_paths,
        facts={
            "repo": OFFICIAL_REPO,
            "revision": OFFICIAL_REVISION,
            "official_schema_tensor_count": OFFICIAL_TENSOR_COUNT,
            "official_schema_logical_weights": OFFICIAL_LOGICAL_WEIGHTS,
            "synthetic_twin_green": True,
            "official_header_schema_green": True,
            "official_tokenizer_chat_assembly_green": True,
            "body_backed_parent_parity": False,
            "bf16_reference_forward_validated": False,
            "capability_claimed": False,
        },
        scope={
            "proves": "synthetic adapter twin plus official header/tokenizer schema",
            "body_backed": False,
            "does_not_prove": [
                "real BF16 tensor-value parity",
                "BF16 reference forward",
                "model capability",
            ],
        },
    )


def _validate_corpus(reader: _Reader) -> _SemanticResult:
    path = "GLM52_CORPUS_INTEGRITY.json"
    manifest_path = "GLM52_OFFICIAL_MANIFEST.json"
    corpus = reader.sealed(
        path, schema="hawking.glm52.corpus_integrity.v2", status="PASS"
    )
    manifest = reader.sealed(
        manifest_path,
        schema="hawking.glm52.official_manifest.v1",
        status="PASS_CONTROL_PLANE_AND_HEADERS_BODY_PENDING",
    )
    scope = _require_object(corpus.get("scope"), "corpus scope")
    expected_scope = {
        "model_payload_downloaded": False,
        "network_access_used": False,
        "capability_claim_permitted": False,
        "real_model_scores_required_for_capability_claim": True,
    }
    for key, expected in expected_scope.items():
        _expect(scope.get(key), expected, f"corpus scope {key}")
    gates = _require_object(corpus.get("integrity_gates"), "corpus gates")
    _expect(len(gates), 12, "corpus gate count")
    if set(gates.values()) != {"PASS"}:
        _fail("not every corpus integrity gate is PASS")
    validation = _require_object(corpus.get("validation"), "corpus validation")
    for key, expected in (
        ("status", "PASS"), ("record_count", 171), ("core_record_count", 135),
        ("long_context_record_count", 36), ("atomic_segment_count", 33_804),
        ("unique_prompt_hashes", 171), ("unique_context_window_hashes", 171),
        ("unique_source_document_ids", 171),
        ("unique_source_document_hashes", 171),
        ("unique_segment_hashes", 33_804),
    ):
        _expect(validation.get(key), expected, f"corpus validation {key}")
    partitions = _require_list(corpus.get("partitions"), "corpus partitions")
    _expect(len(partitions), 9, "corpus partition count")
    domains = _require_list(corpus.get("domains"), "corpus domains")
    _expect(len(domains), 15, "corpus domain count")
    ladder = _require_list(corpus.get("context_ladder"), "corpus context ladder")
    ladder_by_rung = {
        _require_object(row, "corpus ladder row").get("rung"): row for row in ladder
    }
    _expect(set(ladder_by_rung), {"2K", "8K", "32K", "128K", "256K", "1M"}, "corpus ladder rungs")
    for rung in ("2K", "8K", "32K", "128K"):
        _expect(ladder_by_rung[rung].get("admission"), "ADMITTED", f"corpus rung {rung}")
    _expect(
        ladder_by_rung["256K"].get("admission"),
        "NOT_ADMITTED_RESOURCE_VALIDATION_PENDING",
        "corpus 256K boundary",
    )
    _expect(
        ladder_by_rung["1M"].get("admission"),
        "NOT_ADMITTED_EXACT_RUNTIME_PENDING",
        "corpus 1M boundary",
    )
    tokenizer = _require_object(corpus.get("official_tokenizer"), "corpus tokenizer")
    for key, expected in (
        ("repository", OFFICIAL_REPO), ("revision", OFFICIAL_REVISION),
        ("file", "tokenizer.json"),
        ("sha256", "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"),
        ("bytes", 20_217_442), ("vocabulary_size", 154_856),
        ("local_files_only", True), ("host_path_sealed", False),
    ):
        _expect(tokenizer.get(key), expected, f"corpus tokenizer {key}")
    rows, _ = _manifest_rows(manifest)
    tokenizer_row = next((row for row in rows if row.get("path") == "tokenizer.json"), None)
    tokenizer_row = _require_object(tokenizer_row, "manifest tokenizer row")
    tokenizer_local = _require_object(tokenizer_row.get("local"), "manifest tokenizer local")
    _expect(tokenizer_local.get("sha256"), tokenizer["sha256"], "corpus tokenizer manifest hash")
    builder = _require_object(corpus.get("deterministic_builder"), "corpus builder")
    builder_path = builder.get("builder_path")
    if not isinstance(builder_path, str):
        _fail("corpus builder path is invalid")
    source_paths = [builder_path]
    _expect(
        hashlib.sha256(reader.raw(builder_path)).hexdigest(),
        builder.get("builder_sha256"),
        "corpus builder hash",
    )
    instruments = _require_object(builder.get("instrument_sha256"), "corpus instruments")
    for instrument_path, expected_hash in instruments.items():
        if not isinstance(instrument_path, str) or not instrument_path.startswith("tools/"):
            continue
        _expect(
            hashlib.sha256(reader.raw(instrument_path)).hexdigest(),
            expected_hash,
            f"corpus instrument hash {instrument_path}",
        )
        source_paths.append(instrument_path)
    quality = _require_object(corpus.get("quality_metric_contract"), "corpus quality contract")
    _expect(
        quality.get("corpus_manifest_is_not_a_model_quality_result"),
        True,
        "corpus quality boundary",
    )
    _expect(
        quality.get("capability_claim_requires_real_scored_execution"),
        True,
        "corpus capability boundary",
    )
    return _SemanticResult(
        artifact_paths=(path, manifest_path),
        document_paths=tuple(dict.fromkeys(source_paths)),
        facts={
            "repo": OFFICIAL_REPO,
            "revision": OFFICIAL_REVISION,
            "record_count": 171,
            "partition_count": 9,
            "domain_count": 15,
            "integrity_gate_count": 12,
            "admitted_context_rungs": ["2K", "8K", "32K", "128K"],
            "withheld_context_rungs": ["256K", "1M"],
            "network_access_used": False,
            "model_payload_downloaded": False,
            "capability_claimed": False,
        },
        scope={
            "proves": "offline deterministic corpus integrity and split hygiene",
            "host_paths_sealed": False,
            "does_not_prove": [
                "model quality",
                "256K execution safety",
                "1M execution safety",
                "capability",
            ],
        },
    )


_VALIDATORS: dict[str, Callable[[_Reader], _SemanticResult]] = {
    "kimi_final_evidence_verified": _validate_kimi_final,
    "kimi_raw_source_safely_released": _validate_kimi_release,
    "official_glm52_immutable_revision_sealed": _validate_revision,
    "bf16_source_manifest_complete": _validate_manifest,
    "exact_logical_weight_ledger_sealed": _validate_weight_ledger,
    "gravity_pre_audit_complete": _validate_pre_audit,
    "external_baseline_matrix_complete": _validate_external_matrix,
    "adapter_twin_green": _validate_adapter,
    "corpus_integrity_green": _validate_corpus,
}


def _derive(reader: _Reader, stop_condition: str) -> dict[str, Any]:
    validator = _VALIDATORS.get(stop_condition)
    if validator is None:
        _fail(f"stop condition is not offline-evidence-ready: {stop_condition!r}")
    result = validator(reader)
    return {
        "schema": PROOF_SCHEMA,
        "status": "PASS" if result.proven else "BLOCKED",
        "proven": result.proven,
        "blockers": list(result.blockers),
        "stop_condition": stop_condition,
        "campaign_id": CAMPAIGN_ID,
        "source_repo": OFFICIAL_REPO,
        "source_revision": OFFICIAL_REVISION,
        "validator_source": validator_source_binding(),
        "artifact_bindings": {
            path: reader.artifact_binding(path)
            for path in sorted(result.artifact_paths)
        },
        "document_bindings": {
            path: reader.document_binding(path)
            for path in sorted(set(result.document_paths))
        },
        "facts": result.facts,
        "scope": result.scope,
    }


def derive_stop_proof(
    root: str | os.PathLike[str], stop_condition: str
) -> dict[str, Any]:
    """Derive one deterministic, non-receipt semantic proof from trusted files."""

    return _derive(_Reader(root), stop_condition)


def derive_all_ready_stop_proofs(
    root: str | os.PathLike[str],
) -> dict[str, dict[str, Any]]:
    """Derive all and only the nine currently offline-evidence-ready proofs."""

    reader = _Reader(root)
    return {condition: _derive(reader, condition) for condition in READY_STOP_CONDITIONS}


def validate_stop_proof(
    root: str | os.PathLike[str],
    stop_condition: str,
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-derive from current trusted bytes and require byte-canonical equality."""

    if not isinstance(proof, Mapping):
        _fail("terminal semantic proof must be an object")
    expected_fields = {
        "schema", "status", "proven", "blockers", "stop_condition",
        "campaign_id", "source_repo", "source_revision", "validator_source",
        "artifact_bindings", "document_bindings", "facts", "scope",
    }
    if set(proof) != expected_fields:
        _fail("terminal semantic proof fields are incomplete or unexpected")
    _expect(proof.get("schema"), PROOF_SCHEMA, "terminal proof schema")
    status = proof.get("status")
    proven = proof.get("proven")
    if type(proven) is not bool or status not in {"PASS", "BLOCKED"} \
            or (status == "PASS") is not proven:
        _fail("terminal proof status/proven relation is invalid")
    blockers = proof.get("blockers")
    if not isinstance(blockers, list) or (proven and blockers) \
            or (not proven and not blockers):
        _fail("terminal proof blocker inventory is invalid")
    _expect(proof.get("stop_condition"), stop_condition, "terminal proof condition")
    _expect(proof.get("campaign_id"), CAMPAIGN_ID, "terminal proof campaign")
    _expect(proof.get("source_repo"), OFFICIAL_REPO, "terminal proof repository")
    _expect(proof.get("source_revision"), OFFICIAL_REVISION, "terminal proof revision")
    _expect(
        proof.get("validator_source"),
        validator_source_binding(),
        "terminal proof validator-source bytes",
    )
    expected = derive_stop_proof(root, stop_condition)
    try:
        matches = canonical(dict(proof)) == canonical(expected)
    except (TypeError, ValueError) as exc:
        raise TerminalProofError(f"terminal proof is not canonical JSON: {exc}") from exc
    if not matches:
        _fail("terminal semantic proof does not match current grounded derivation")
    return expected


def validate_all_ready_stop_proofs(
    root: str | os.PathLike[str], proofs: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    if not isinstance(proofs, Mapping) or set(proofs) != set(READY_STOP_CONDITIONS):
        _fail("ready-stop proof inventory is incomplete or unexpected")
    return {
        condition: validate_stop_proof(root, condition, proofs[condition])
        for condition in READY_STOP_CONDITIONS
    }
