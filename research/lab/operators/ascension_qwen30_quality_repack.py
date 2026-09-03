"""Build a separate, source-bound Qwen30 quality-repack candidate.

This lane deliberately leaves the admitted 1.130366 BPW Qwen30 control
untouched.  It re-encodes a *new* complete artifact from the immutable source
body.  Every unchanged tensor keeps the direct binary sign/scale layout; only
the two source-bound gate/up organs named in the sealed proposal receive a
deterministic sparse FP16 residual.  The residual is an experimental quality
branch, not a runtime, admission, HCLI, TPS, TG, or tournament result.

The candidate uses the pre-existing full-shard revalidation receipt as an
immutable source control.  It verifies the receipt, its audit binding, and the
current file identities before it writes any bytes; it never overwrites that
control receipt.  A separate immutable snapshot and selection receipt bind
the branch inputs, residual policy, and source-to-packed discriminators.
"""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import struct
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators import ascension_qwen30_complete_gravity as complete
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
QWEN30_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30"
BASELINE_ROOT = QWEN30_ROOT / "complete-gravity"
SOURCE_AUDIT = QWEN30_ROOT / "QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json"
QUALITY_ROOT = QWEN30_ROOT / "quality-candidates" / "gate-up-residual-v1"
SCIENTIFIC_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen-family/scientific-optimizer"
)
GATE_UP_PROPOSAL = (
    SCIENTIFIC_ROOT
    / "repack-proposals/QWEN30_GATE_UP_REPRESENTATION_REPACK_PROPOSAL_5318d99a625f683340d8fd65.json"
)
GATE_UP_QUALITY_RECEIPT = (
    SCIENTIFIC_ROOT
    / "capability-quality/QWEN30_DIRECT_PACKED_GATE_UP_QUALITY_22b24b962eeffb943130c62b.json"
)
BASELINE_MANIFEST_NAME = "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
BASELINE_ADMISSION_NAME = "QWEN30_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"
BASELINE_REVALIDATION_NAME = "QWEN30_CURRENT_SOURCE_SHARD_REVALIDATION.json"

SCHEMA = "hawking.ascension.qwen30_quality_repack_candidate.v1"
SELECTION_SCHEMA = "hawking.ascension.qwen30_quality_repack_selection.v1"
SOURCE_SNAPSHOT_SCHEMA = "hawking.ascension.qwen30_quality_repack_source_snapshot.v1"
STATUS_SCHEMA = "hawking.ascension.qwen30_quality_repack_status.v1"
BRANCH_ID = "qwen30-gate-up-sparse-fp16-residual-v1"
ARTIFACT_PREFIX = "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1"
MODEL_ID = "Qwen3-Coder-30B-A3B-Instruct-quality-gate-up-residual-v1"

RESIDUAL_MAGIC = b"HQ30GR2\0"
RESIDUAL_VERSION = 1
RESIDUAL_HEADER = struct.Struct("<8sIIIIII")
RESIDUAL_FRACTIONS: tuple[float, ...] = (0.0025, 0.005, 0.01)
MIN_PAIR_RELATIVE_L2_IMPROVEMENT = 0.005
CONTROL_COUNT = 16
DEFAULT_PARALLEL_WORKERS = 4
DEFAULT_PARALLEL_MEMORY_BUDGET_MIB = 3072
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_STREAMING_DIRECT_THRESHOLD_BYTES = 256 * _MIB
_STREAMING_DIRECT_WORKING_SET_BYTES = 192 * _MIB
_STREAMING_GROUPS_PER_CHUNK = 8192
_QUALITY_SELECTED_ORGANS = frozenset(
    {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    }
)


class QualityRepackError(complete.CompleteGravityError):
    """The quality branch cannot safely proceed from its immutable controls."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _raw_sha256(path: Path) -> str:
    return complete._sha256_file(path)


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    raw = complete._read_json(path)
    if raw is None:
        raise QualityRepackError(f"missing {label}: {path}")
    try:
        return verify(raw, label=str(path))
    except Exception as exc:
        raise QualityRepackError(f"untrustworthy {label}: {exc}") from exc


def _immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Write an append-only branch record, refusing a resealed substitution."""

    expected = seal(dict(payload))
    existing = complete._read_json(path)
    if existing is None:
        complete._atomic_json(path, expected)
        return expected
    try:
        verified = verify(existing, label=str(path))
    except Exception as exc:
        raise QualityRepackError(f"{label} already exists but is not sealed: {exc}") from exc
    if _canonical_sha256(verified) != _canonical_sha256(expected):
        raise QualityRepackError(f"{label} already exists with a different immutable binding: {path}")
    return verified


def _file_binding(path: Path, *, label: str) -> dict[str, Any]:
    document = _sealed(path, label=label)
    return {
        "path": str(path.resolve()),
        "document_sha256": _raw_sha256(path),
        "seal_sha256": document["seal_sha256"],
        "file_identity": complete._file_identity(path, label=label),
    }


def _metric(source: np.ndarray, reconstructed: np.ndarray) -> dict[str, float | bool]:
    left = np.ascontiguousarray(source, dtype=np.float32).reshape(-1)
    right = np.ascontiguousarray(reconstructed, dtype=np.float32).reshape(-1)
    if left.shape != right.shape:
        raise QualityRepackError("quality comparison geometry mismatch")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise QualityRepackError("quality comparison contains a non-finite value")
    left_norm = max(float(np.linalg.norm(left)), 1e-12)
    right_norm = max(float(np.linalg.norm(right)), 1e-12)
    return {
        "relative_l2": float(np.linalg.norm(left - right) / left_norm),
        "cosine": float(np.dot(left, right) / (left_norm * right_norm)),
        "rmse": float(np.sqrt(np.mean(np.square(left - right)))),
        "max_abs": float(np.max(np.abs(left - right))) if left.size else 0.0,
        "finite": True,
    }


def _source_value_sha256(values: np.ndarray) -> str:
    """Hash the canonical FP32 tensor view used by the sealed diagnostic.

    The source body is BF16, but the pre-existing source-bound gate/up receipt
    identifies its decoded values as deterministic little-endian FP32.  Keep
    the raw BF16 hash separately in the branch journal rather than comparing
    unlike representations.
    """

    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def _activation_controls(input_width: int, *, label: str) -> np.ndarray:
    if input_width <= 0:
        raise QualityRepackError("projection control input width must be positive")
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")
    generator = np.random.default_rng(seed)
    return generator.standard_normal((CONTROL_COUNT, input_width), dtype=np.float32)


def _silu(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -60.0, 60.0)
    return clipped / (1.0 + np.exp(-clipped))


def _projection_discriminator(
    source: np.ndarray, reconstructed: np.ndarray, *, label: str
) -> dict[str, Any]:
    if source.ndim != 2 or reconstructed.shape != source.shape:
        raise QualityRepackError("projection discriminator requires matching rank-2 weights")
    controls = _activation_controls(int(source.shape[1]), label=label)
    source_output = controls @ source.T
    reconstructed_output = controls @ reconstructed.T
    return {
        "control_count": CONTROL_COUNT,
        "activation_sha256": hashlib.sha256(controls.astype("<f4").tobytes()).hexdigest(),
        "source_output_sha256": hashlib.sha256(source_output.astype("<f4").tobytes()).hexdigest(),
        "reconstructed_output_sha256": hashlib.sha256(
            reconstructed_output.astype("<f4").tobytes()
        ).hexdigest(),
        "metrics": _metric(source_output, reconstructed_output),
    }


def _pair_swiglu_discriminator(
    gate_source: np.ndarray,
    up_source: np.ndarray,
    gate_reconstructed: np.ndarray,
    up_reconstructed: np.ndarray,
) -> dict[str, Any]:
    if (
        gate_source.ndim != 2
        or up_source.ndim != 2
        or gate_source.shape != up_source.shape
        or gate_reconstructed.shape != gate_source.shape
        or up_reconstructed.shape != up_source.shape
    ):
        raise QualityRepackError("gate/up discriminator requires matching rank-2 matrices")
    controls = _activation_controls(int(gate_source.shape[1]), label=f"{BRANCH_ID}:gate-up")
    source_output = _silu(controls @ gate_source.T) * (controls @ up_source.T)
    reconstructed_output = _silu(controls @ gate_reconstructed.T) * (controls @ up_reconstructed.T)
    return {
        "control_count": CONTROL_COUNT,
        "activation_sha256": hashlib.sha256(controls.astype("<f4").tobytes()).hexdigest(),
        "source_swiglu_sha256": hashlib.sha256(source_output.astype("<f4").tobytes()).hexdigest(),
        "reconstructed_swiglu_sha256": hashlib.sha256(
            reconstructed_output.astype("<f4").tobytes()
        ).hexdigest(),
        "metrics": _metric(source_output, reconstructed_output),
    }


def _select_residual_indices(error: np.ndarray, *, fraction: float) -> np.ndarray:
    """Choose top absolute-error coordinates with a deterministic tie rule."""

    if not (0.0 < fraction <= 1.0):
        raise QualityRepackError("sparse residual fraction must be in (0, 1]")
    flat = np.ascontiguousarray(error, dtype=np.float32).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise QualityRepackError("residual selection needs finite non-empty errors")
    count = max(1, min(flat.size, int(math.ceil(flat.size * fraction))))
    # ``partition`` finds the boundary in O(n).  Ties at that boundary are
    # resolved by ascending flat index, which makes the payload independent of
    # implementation-specific quickselect ordering.
    boundary = float(np.partition(flat, flat.size - count)[flat.size - count])
    strict = np.flatnonzero(flat > boundary)
    equal = np.flatnonzero(flat == boundary)
    remaining = count - int(strict.size)
    if remaining < 0 or remaining > int(equal.size):
        raise QualityRepackError("deterministic residual tie selection failed")
    indices = np.sort(np.concatenate((strict, equal[:remaining])).astype(np.uint32, copy=False))
    if int(indices.size) != count or np.unique(indices).size != indices.size:
        raise QualityRepackError("residual index selection is not unique")
    return indices


def _unpack_binary(payload: bytes) -> tuple[tuple[int, ...], np.ndarray]:
    """Decode the existing direct binary payload for a deterministic oracle."""

    if len(payload) < 32:
        raise QualityRepackError("direct binary payload is shorter than its header")
    magic, version, group_size, rank, _reserved, elements, _reserved2 = struct.unpack(
        "<8sIIHHQI", payload[:32]
    )
    if magic != complete.MAGIC or version != complete.VERSION or group_size != complete.GROUP_SIZE:
        raise QualityRepackError("unexpected direct binary base layout")
    if rank <= 0:
        raise QualityRepackError("direct binary payload has zero rank")
    dimensions_end = 32 + 4 * rank
    if len(payload) < dimensions_end:
        raise QualityRepackError("direct binary payload has truncated dimensions")
    shape = struct.unpack("<" + "I" * rank, payload[32:dimensions_end])
    if complete._tensor_count(shape) != elements:
        raise QualityRepackError("direct binary payload shape/elements mismatch")
    groups = (elements + group_size - 1) // group_size
    scales_end = dimensions_end + 2 * groups
    signs_end = scales_end + (groups * group_size) // 8
    if signs_end != len(payload):
        raise QualityRepackError("direct binary payload byte geometry mismatch")
    scales = np.frombuffer(payload[dimensions_end:scales_end], dtype="<f2").astype(np.float32)
    bits = np.unpackbits(
        np.frombuffer(payload[scales_end:signs_end], dtype=np.uint8), bitorder="little"
    )[: groups * group_size]
    signs = np.where(bits != 0, 1.0, -1.0).astype(np.float32)
    reconstructed = (signs * np.repeat(scales, group_size))[:elements]
    return tuple(int(value) for value in shape), reconstructed.reshape(shape)


def _pack_sparse_residual(
    values: np.ndarray, shape: Sequence[int], *, fraction: float
) -> tuple[bytes, dict[str, Any], np.ndarray]:
    """Encode binary sign/scale plus a deterministic sparse FP16 residual."""

    base_payload, base_quality, base_reconstructed = complete._pack_binary(values, shape)
    flat_source = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    flat_base = np.ascontiguousarray(base_reconstructed, dtype=np.float32).reshape(-1)
    indices = _select_residual_indices(np.abs(flat_source - flat_base), fraction=fraction)
    residual_values = (flat_source[indices] - flat_base[indices]).astype("<f2")
    dimensions = tuple(int(value) for value in shape)
    header = RESIDUAL_HEADER.pack(
        RESIDUAL_MAGIC,
        RESIDUAL_VERSION,
        complete.VERSION,
        complete.GROUP_SIZE,
        len(dimensions),
        len(base_payload),
        int(indices.size),
    )
    dimensions_payload = struct.pack("<" + "I" * len(dimensions), *dimensions)
    index_payload = indices.astype("<u4", copy=False).tobytes()
    residual_payload = residual_values.tobytes()
    payload = header + dimensions_payload + base_payload + index_payload + residual_payload
    reconstructed_flat = flat_base.copy()
    reconstructed_flat[indices] += residual_values.astype(np.float32)
    reconstructed = reconstructed_flat.reshape(dimensions)
    metadata = {
        "family": "binary_sign_scale_sparse_fp16_residual",
        "version": RESIDUAL_VERSION,
        "magic": RESIDUAL_MAGIC.decode("ascii"),
        "base_layout": {
            "magic": complete.MAGIC.decode("ascii"),
            "version": complete.VERSION,
            "group_size": complete.GROUP_SIZE,
            "sign_bit_order": "little",
            "scale_dtype": "float16",
        },
        "residual": {
            "selection": "largest_abs_binary_reconstruction_error_with_ascending_index_ties",
            "requested_fraction": fraction,
            "selected_count": int(indices.size),
            "index_dtype": "uint32_little_endian",
            "value_dtype": "float16_little_endian",
            "indices_sha256": hashlib.sha256(index_payload).hexdigest(),
            "values_sha256": hashlib.sha256(residual_payload).hexdigest(),
            "index_bytes": len(index_payload),
            "value_bytes": len(residual_payload),
        },
        "base_payload_bytes": len(base_payload),
        "header_and_shape_bytes": len(header) + len(dimensions_payload),
        "physical_payload_bytes": len(payload),
        "base_reconstruction_quality": base_quality,
    }
    return payload, metadata, reconstructed


def _unpack_sparse_residual(payload: bytes) -> tuple[dict[str, Any], np.ndarray]:
    if len(payload) < RESIDUAL_HEADER.size:
        raise QualityRepackError("sparse residual payload is shorter than its header")
    (
        magic,
        version,
        base_version,
        group_size,
        rank,
        base_bytes,
        residual_count,
    ) = RESIDUAL_HEADER.unpack(payload[: RESIDUAL_HEADER.size])
    if (
        magic != RESIDUAL_MAGIC
        or version != RESIDUAL_VERSION
        or base_version != complete.VERSION
        or group_size != complete.GROUP_SIZE
        or rank <= 0
    ):
        raise QualityRepackError("unexpected sparse residual layout")
    dimensions_end = RESIDUAL_HEADER.size + 4 * rank
    if len(payload) < dimensions_end + base_bytes:
        raise QualityRepackError("sparse residual payload has truncated base bytes")
    shape = struct.unpack("<" + "I" * rank, payload[RESIDUAL_HEADER.size:dimensions_end])
    total = complete._tensor_count(shape)
    base_end = dimensions_end + base_bytes
    indices_end = base_end + residual_count * 4
    values_end = indices_end + residual_count * 2
    if values_end != len(payload):
        raise QualityRepackError("sparse residual payload byte geometry mismatch")
    decoded_shape, reconstructed = _unpack_binary(payload[dimensions_end:base_end])
    if tuple(shape) != decoded_shape:
        raise QualityRepackError("sparse residual base shape mismatch")
    indices = np.frombuffer(payload[base_end:indices_end], dtype="<u4")
    values = np.frombuffer(payload[indices_end:values_end], dtype="<f2").astype(np.float32)
    if (
        int(indices.size) != residual_count
        or np.any(indices >= total)
        or (indices.size > 1 and np.any(indices[1:] <= indices[:-1]))
    ):
        raise QualityRepackError("sparse residual indices are invalid")
    flat = np.ascontiguousarray(reconstructed, dtype=np.float32).reshape(-1).copy()
    flat[indices] += values
    return {
        "shape": [int(value) for value in shape],
        "selected_count": int(residual_count),
        "indices_sha256": hashlib.sha256(payload[base_end:indices_end]).hexdigest(),
        "values_sha256": hashlib.sha256(payload[indices_end:values_end]).hexdigest(),
    }, flat.reshape(shape)


class QualityRepackGravity(complete.CompleteBinaryGravity):
    """A complete-artifact branch that cannot alter the admitted control."""

    def __init__(
        self,
        *,
        model_dir: Path,
        source_audit: Path,
        root: Path,
        proposal_path: Path,
        quality_receipt_path: Path,
        baseline_root: Path,
        repository: str = complete.DEFAULT_REPOSITORY,
        artifact_prefix: str = ARTIFACT_PREFIX,
        model_id: str = MODEL_ID,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            source_audit=source_audit,
            root=root,
            repository=repository,
            model_id=model_id,
            artifact_prefix=artifact_prefix,
            schema=SCHEMA,
        )
        self.proposal_path = proposal_path.expanduser().resolve()
        self.quality_receipt_path = quality_receipt_path.expanduser().resolve()
        self.baseline_root = baseline_root.expanduser().resolve()
        self.baseline_manifest_path = self.baseline_root / BASELINE_MANIFEST_NAME
        self.baseline_admission_path = self.baseline_root / BASELINE_ADMISSION_NAME
        self.baseline_revalidation_path = self.baseline_root / BASELINE_REVALIDATION_NAME
        # The candidate must reuse, never rewrite, the sealed full-source
        # receipt that protects the admitted baseline.
        self.source_revalidation_path = self.baseline_revalidation_path
        self.snapshot_path = self.root / f"{self.artifact_prefix}_SOURCE_BINDING_SNAPSHOT.json"
        self.selection_path = self.root / f"{self.artifact_prefix}_SELECTION_RECEIPT.json"
        self._policy: dict[str, Any] | None = None
        self._baseline_rows: dict[str, dict[str, Any]] = {}
        self._baseline_control: dict[str, Any] = {}
        self._current_revalidation: dict[str, Any] | None = None
        # Only the quality candidate uses this lease.  It serializes journal
        # append/index reconciliation even when a human accidentally starts a
        # second CLI process while launchd is already packing.
        self._coordinator_lock_path = self.root / f"{self.artifact_prefix}_PARALLEL_COORDINATOR.lock"

    @contextlib.contextmanager
    def _coordinator_lease(self) -> Any:
        """Hold the one candidate-local coordinator lease for a bounded run.

        Tensor workers never append the journal or mutate the compact index;
        they only atomically replace their unique deterministic payload path.
        The coordinator collects completed futures in fixed planned-order,
        fsyncs each JSONL row, then advances the sealed restart index.
        """

        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._coordinator_lock_path, os.O_CREAT | os.O_RDWR, 0o640)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise QualityRepackError(
                    "another Qwen30 quality-repack coordinator already holds the candidate journal lease"
                ) from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @staticmethod
    def _memory_snapshot() -> dict[str, int | float | None]:
        """Read conservative macOS headroom/swap counters without mutating state."""

        page_size = 16_384
        total_bytes: int | None = None
        reclaimable_bytes: int | None = None
        swapouts_pages: int | None = None
        swap_used_bytes: int | None = None
        try:
            page_size = int(
                subprocess.run(
                    ["/usr/sbin/sysctl", "-n", "hw.pagesize"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                ).stdout.strip()
            )
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        try:
            total_bytes = int(
                subprocess.run(
                    ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                ).stdout.strip()
            )
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        try:
            text = subprocess.run(
                ["/usr/bin/vm_stat"], check=False, capture_output=True, text=True, timeout=2.0
            ).stdout
            pages: dict[str, int] = {}
            for line in text.splitlines():
                match = re.match(r"^Pages ([A-Za-z ]+):\s+(\d+)\.?$", line)
                if match:
                    pages[match.group(1).strip().lower()] = int(match.group(2))
                    continue
                swapouts = re.match(r"^Swapouts:\s+(\d+)\.?$", line)
                if swapouts:
                    pages["swapouts"] = int(swapouts.group(1))
            reclaimable_pages = (
                pages.get("free", 0) + pages.get("inactive", 0) + pages.get("speculative", 0)
            )
            reclaimable_bytes = reclaimable_pages * page_size
            swapouts_pages = pages.get("swapouts")
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            swap = subprocess.run(
                ["/usr/sbin/sysctl", "vm.swapusage"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            ).stdout
            used = re.search(r"used\s+=\s+([0-9.]+)([MG])", swap)
            if used:
                multiplier = _MIB if used.group(2) == "M" else _GIB
                swap_used_bytes = int(float(used.group(1)) * multiplier)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return {
            "page_size": page_size,
            "total_bytes": total_bytes,
            "reclaimable_bytes": reclaimable_bytes,
            "swapouts_pages": swapouts_pages,
            "swap_used_bytes": swap_used_bytes,
        }

    @staticmethod
    def _rss_bytes() -> int | None:
        try:
            result = subprocess.run(
                ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            value = int(result.stdout.strip())
            return value * 1024
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _transient_bytes_for_info(info: Mapping[str, Any]) -> int:
        """Conservatively bound one worker's raw/FP32/padded/metric arrays."""

        offsets = info.get("data_offsets")
        shape = info.get("shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not isinstance(shape, list):
            raise QualityRepackError("parallel scheduler encountered malformed source tensor metadata")
        try:
            begin, end = (int(value) for value in offsets)
            elements = complete._tensor_count([int(value) for value in shape])
        except (TypeError, ValueError, complete.CompleteGravityError) as exc:
            raise QualityRepackError("parallel scheduler cannot estimate malformed tensor geometry") from exc
        if begin < 0 or end < begin or elements <= 0:
            raise QualityRepackError("parallel scheduler encountered invalid source tensor byte range")
        # ``_pack_binary`` keeps raw, FP32, padding, reconstructed, and error
        # vectors live around the quality metrics.  14x raw plus a fixed 16 MiB
        # covers BF16/F32 variants and NumPy allocator overhead without using
        # the whole machine merely because inactive file cache is reclaimable.
        return max(16 * _MIB, (end - begin) * 14 + elements * 4 + 8 * _MIB)

    def _can_stream_direct_tensor(self, *, tensor_name: str, info: Mapping[str, Any]) -> bool:
        """Whether a large unchanged BF16 tensor has the exact streaming path.

        The two selected gate/up organs must keep their sealed residual writer.
        All other Qwen30 tensors are direct sign/scale controls and can be
        materialized group-by-group without a decoded full-tensor shadow.
        """

        if tensor_name in _QUALITY_SELECTED_ORGANS:
            return False
        dtype = str(info.get("dtype", "")).upper()
        if dtype not in {"BF16", "BFLOAT16"}:
            return False
        try:
            return self._transient_bytes_for_info(info) > _STREAMING_DIRECT_THRESHOLD_BYTES
        except QualityRepackError:
            return False

    def _stream_direct_binary_tensor(
        self,
        *,
        tensor_name: str,
        shard: str,
        source_hash: str,
        info: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Encode one very large unselected BF16 tensor with bounded memory.

        This is byte-layout equivalent to ``_pack_binary``: group scales use
        the same FP64 mean-absolute calculation, signs retain the same padded
        little-endian tail bits, and the final header/scales/signs payload is
        still atomically replaced at the deterministic tensor path.  Only the
        transient representation changes, so it is not a new model family or
        a capability claim.
        """

        if not self._can_stream_direct_tensor(tensor_name=tensor_name, info=info):
            raise QualityRepackError("streaming direct encoder was asked for an ineligible tensor")
        source_path = self.model_dir / shard
        current = self._current_revalidation
        if current is None:
            raise QualityRepackError("source revalidation has not been admitted")
        rows = current.get("shards")
        expected_identity = rows.get(shard, {}).get("file_identity") if isinstance(rows, Mapping) else None
        before = complete._file_identity(source_path, label=f"source shard {shard}")
        if before != expected_identity:
            raise QualityRepackError(f"source shard identity diverged before streaming {tensor_name}")
        header = self._header(source_path)
        observed = header.get(tensor_name)
        if not isinstance(observed, Mapping) or dict(observed) != dict(info):
            raise QualityRepackError(f"source header changed before streaming {tensor_name}")
        dtype = str(info.get("dtype", "")).upper()
        shape = [int(value) for value in info.get("shape", [])]
        offsets = info.get("data_offsets")
        if dtype not in {"BF16", "BFLOAT16"} or not shape or not isinstance(offsets, list) or len(offsets) != 2:
            raise QualityRepackError(f"streaming direct tensor metadata is invalid: {tensor_name}")
        begin, end = (int(value) for value in offsets)
        elements = complete._tensor_count(shape)
        if begin < 0 or end < begin or end - begin != elements * 2:
            raise QualityRepackError(f"streaming direct tensor byte geometry is invalid: {tensor_name}")
        groups = (elements + complete.GROUP_SIZE - 1) // complete.GROUP_SIZE
        destination = self.tensor_dir / complete._artifact_name(tensor_name)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        raw_digest = hashlib.sha256()
        value_digest = hashlib.sha256()
        source_sq = 0.0
        reconstructed_sq = 0.0
        dot = 0.0
        error_sq = 0.0
        total_values = 0
        scale_descriptor, scale_path = tempfile.mkstemp(prefix=f".{destination.name}.scales.", dir=self.tensor_dir)
        sign_descriptor, sign_path = tempfile.mkstemp(prefix=f".{destination.name}.signs.", dir=self.tensor_dir)
        artifact_descriptor: int | None = None
        artifact_temporary: str | None = None
        try:
            with os.fdopen(scale_descriptor, "wb") as scales, os.fdopen(sign_descriptor, "wb") as signs:
                with source_path.open("rb") as source:
                    header_bytes = struct.unpack("<Q", source.read(8))[0]
                    source.seek(8 + header_bytes + begin)
                    remaining = elements
                    while remaining:
                        logical = min(remaining, _STREAMING_GROUPS_PER_CHUNK * complete.GROUP_SIZE)
                        raw = source.read(logical * 2)
                        if len(raw) != logical * 2:
                            raise QualityRepackError(f"streaming source tensor is truncated: {tensor_name}")
                        raw_digest.update(raw)
                        values = (
                            np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                        ).view(np.float32)
                        if not np.isfinite(values).all():
                            raise QualityRepackError(f"streaming source tensor has non-finite values: {tensor_name}")
                        value_digest.update(np.ascontiguousarray(values, dtype="<f4").tobytes())
                        local_groups = (logical + complete.GROUP_SIZE - 1) // complete.GROUP_SIZE
                        padded = np.pad(
                            values,
                            (0, local_groups * complete.GROUP_SIZE - logical),
                            constant_values=0.0,
                        ).reshape(local_groups, complete.GROUP_SIZE)
                        group_scales = np.mean(np.abs(padded), axis=1, dtype=np.float64).astype("<f2")
                        group_signs = np.packbits(
                            (padded >= 0.0).reshape(-1).astype(np.uint8), bitorder="little"
                        )
                        reconstructed = (
                            np.where(padded >= 0.0, 1.0, -1.0)
                            * group_scales.astype(np.float32)[:, None]
                        ).reshape(-1)[:logical]
                        values64 = values.astype(np.float64, copy=False)
                        reconstructed64 = reconstructed.astype(np.float64, copy=False)
                        difference64 = values64 - reconstructed64
                        source_sq += float(np.dot(values64, values64))
                        reconstructed_sq += float(np.dot(reconstructed64, reconstructed64))
                        dot += float(np.dot(values64, reconstructed64))
                        error_sq += float(np.dot(difference64, difference64))
                        total_values += logical
                        scales.write(group_scales.tobytes())
                        signs.write(group_signs.tobytes())
                        remaining -= logical
                scales.flush()
                signs.flush()
                os.fsync(scales.fileno())
                os.fsync(signs.fileno())
            if total_values != elements:
                raise QualityRepackError(f"streaming source tensor element count changed: {tensor_name}")
            after = complete._file_identity(source_path, label=f"source shard {shard}")
            if before != after:
                raise QualityRepackError(f"source shard changed while streaming {tensor_name}")
            expected_payload_bytes = complete._payload_bytes(shape)
            artifact_descriptor, artifact_temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=self.tensor_dir
            )
            artifact_digest = hashlib.sha256()
            direct_header = struct.pack(
                "<8sIIHHQI",
                complete.MAGIC,
                complete.VERSION,
                complete.GROUP_SIZE,
                len(shape),
                0,
                elements,
                0,
            ) + struct.pack("<" + "I" * len(shape), *shape)
            with os.fdopen(artifact_descriptor, "wb") as artifact, open(scale_path, "rb") as scales, open(sign_path, "rb") as signs:
                artifact_descriptor = None
                for source_part in (memoryview(direct_header),):
                    artifact.write(source_part)
                    artifact_digest.update(source_part)
                for staged in (scales, signs):
                    while block := staged.read(8 * _MIB):
                        artifact.write(block)
                        artifact_digest.update(block)
                artifact.flush()
                os.fsync(artifact.fileno())
            if os.path.getsize(artifact_temporary) != expected_payload_bytes:
                raise QualityRepackError(f"streaming direct payload byte count changed: {tensor_name}")
            os.chmod(artifact_temporary, 0o640)
            os.replace(artifact_temporary, destination)
            os.chmod(destination, 0o640)
            artifact_temporary = None
            source_norm = max(math.sqrt(source_sq), 1e-12)
            reconstructed_norm = max(math.sqrt(reconstructed_sq), 1e-12)
            quality = {
                "relative_l2": math.sqrt(error_sq) / source_norm,
                "cosine": dot / (source_norm * reconstructed_norm),
                "rmse": math.sqrt(error_sq / elements),
                "finite": True,
            }
            return {
                "tensor_name": tensor_name,
                "source_shard": shard,
                "source_shard_sha256": source_hash,
                "source_dtype": dtype,
                "shape": shape,
                "elements": elements,
                "artifact_path": str(destination),
                "artifact_bytes": expected_payload_bytes,
                "artifact_sha256": artifact_digest.hexdigest(),
                "layout": {
                    "family": "binary_sign_scale_control_reencoded_from_source",
                    "magic": complete.MAGIC.decode("ascii"),
                    "version": complete.VERSION,
                    "group_size": complete.GROUP_SIZE,
                    "sign_bit_order": "little",
                    "scale_dtype": "float16",
                },
                "component_quality": quality,
                "candidate_mutation": {
                    "changed_from_admitted_control": False,
                    "reason": "not one of the two sealed proposal organs",
                    "source_to_packed_discriminator": {
                        "state": "not_required_for_unchanged_control_layout",
                        "source_value_sha256": value_digest.hexdigest(),
                        "source_raw_bf16_sha256": raw_digest.hexdigest(),
                        "streaming_direct_encoder": "group_exact_bounded_memory_v1",
                    },
                    "baseline_rollback": self._baseline_rollback(tensor_name),
                },
            }
        finally:
            if artifact_descriptor is not None:
                os.close(artifact_descriptor)
            if artifact_temporary is not None and os.path.exists(artifact_temporary):
                os.unlink(artifact_temporary)
            for staged in (scale_path, sign_path):
                if os.path.exists(staged):
                    os.unlink(staged)

    def _parallel_memory_budget(self, configured_bytes: int) -> tuple[int, dict[str, int | float | None]]:
        snapshot = self._memory_snapshot()
        reclaimable = snapshot.get("reclaimable_bytes")
        total = snapshot.get("total_bytes")
        # Preserve at least 2 GiB (or 3% on a smaller host) for the existing
        # live campaign, launchd, Metal, file cache, and admission watcher.
        reserve = max(2 * _GIB, int(total * 0.03) if isinstance(total, int) else 0)
        headroom = configured_bytes
        if isinstance(reclaimable, int):
            headroom = max(0, min(configured_bytes, reclaimable - reserve))
        snapshot["reserve_bytes"] = reserve
        snapshot["configured_budget_bytes"] = configured_bytes
        snapshot["effective_budget_bytes"] = headroom
        return headroom, snapshot

    def _select_shard_disjoint_parallel_work(
        self,
        *,
        planned_order: Sequence[tuple[str, str]],
        progress: Mapping[str, Mapping[str, Any]],
        shard_evidence: Mapping[str, Mapping[str, Any]],
        worker_limit: int,
        memory_budget_bytes: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Choose a deterministic, fair one-tensor-per-shard work wave.

        The fixed source plan remains the authority.  We derive the first
        unresolved tensor per shard, select the least-advanced shards (ties by
        fixed planned ordinal), and never expose two workers to the same source
        shard.  That makes source identity before/after checks independent and
        lets a sole coordinator append resolved rows in ordinal order.
        """

        if worker_limit <= 0 or memory_budget_bytes <= 0:
            return [], {"reason": "nonpositive_worker_or_memory_budget"}
        completed_per_shard: dict[str, int] = {}
        first_unresolved: dict[str, tuple[int, str]] = {}
        for ordinal, (shard, tensor_name) in enumerate(planned_order):
            source_hash = str(shard_evidence[shard]["sha256"])
            if self._progress_row_binds_source(
                progress.get(tensor_name), shard=shard, source_hash=source_hash
            ):
                completed_per_shard[shard] = completed_per_shard.get(shard, 0) + 1
            elif shard not in first_unresolved:
                first_unresolved[shard] = (ordinal, tensor_name)
        ranked = sorted(
            (
                (completed_per_shard.get(shard, 0), ordinal, shard, tensor_name)
                for shard, (ordinal, tensor_name) in first_unresolved.items()
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        headers: dict[str, dict[str, Any]] = {}
        selected: list[dict[str, Any]] = []
        estimated_bytes = 0
        deferred_for_memory = 0
        for completed_count, ordinal, shard, tensor_name in ranked:
            if len(selected) >= worker_limit:
                break
            header = headers.get(shard)
            if header is None:
                header = self._header(self.model_dir / shard)
                headers[shard] = header
            info = header.get(tensor_name)
            if not isinstance(info, Mapping):
                raise QualityRepackError(f"source header lacks indexed tensor {tensor_name}")
            native_estimate = self._transient_bytes_for_info(info)
            streaming_direct = self._can_stream_direct_tensor(
                tensor_name=tensor_name, info=info
            )
            estimate = (
                min(native_estimate, _STREAMING_DIRECT_WORKING_SET_BYTES)
                if streaming_direct
                else native_estimate
            )
            if estimate > memory_budget_bytes or estimated_bytes + estimate > memory_budget_bytes:
                deferred_for_memory += 1
                continue
            selected.append(
                {
                    "planned_ordinal": ordinal,
                    "shard": shard,
                    "tensor_name": tensor_name,
                    "source_hash": str(shard_evidence[shard]["sha256"]),
                    "info": dict(info),
                    "estimated_transient_bytes": estimate,
                    "native_full_tensor_estimate_bytes": native_estimate,
                    "streaming_direct_encoder": streaming_direct,
                    "completed_in_shard_before_wave": completed_count,
                }
            )
            estimated_bytes += estimate
        selected.sort(key=lambda item: int(item["planned_ordinal"]))
        return selected, {
            "candidate_shards": len(first_unresolved),
            "selected_shards": [str(item["shard"]) for item in selected],
            "selected_planned_ordinals": [int(item["planned_ordinal"]) for item in selected],
            "estimated_transient_bytes": estimated_bytes,
            "memory_deferred_shards": deferred_for_memory,
            "fixed_plan_sha256": complete._canonical_sha256(list(planned_order)),
        }

    def _status_payload(self, phase: str, **fields: Any) -> dict[str, Any]:
        payload = super()._status_payload(phase, **fields)
        payload["schema"] = STATUS_SCHEMA
        payload["candidate_branch"] = BRANCH_ID
        payload["representation"] = "mixed_direct_binary_plus_selected_sparse_fp16_residual"
        payload["claim_boundary"] = {
            "admitted_qwen30_control_is_preserved_and_never_mutated": True,
            "this_is_a_new_complete_artifact_candidate_not_a_baseline_replacement": True,
            "native_admission_runtime_generation_hcli_tps_tg_and_tournament_are_not_claimed": True,
            "only_sealed_proposal_organs_may_use_the_residual_layout": True,
        }
        return payload

    def _admit_source(self) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
        audit, weight_map, shard_evidence = super()._admit_source()
        self._load_control_documents(audit=audit, weight_map=weight_map, shard_evidence=shard_evidence)
        return audit, weight_map, shard_evidence

    def _load_control_documents(
        self,
        *,
        audit: Mapping[str, Any],
        weight_map: Mapping[str, str],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> None:
        proposal = _sealed(self.proposal_path, label="gate/up quality repack proposal")
        quality = _sealed(self.quality_receipt_path, label="gate/up quality diagnostic")
        manifest = _sealed(self.baseline_manifest_path, label="admitted Qwen30 baseline manifest")
        admission = _sealed(self.baseline_admission_path, label="admitted Qwen30 baseline admission")
        revalidation = _sealed(self.baseline_revalidation_path, label="immutable Qwen30 source revalidation")
        if (
            proposal.get("schema") != "hawking.ascension.qwen30_gate_up_representation_repack_proposal.v1"
            or proposal.get("status")
            != "PROPOSED_NOT_APPLIED_COMPLETE_ACCOUNTING_AND_CAPABILITY_RETEST_REQUIRED"
            or proposal.get("quality_receipt_path") != str(self.quality_receipt_path)
            or proposal.get("quality_receipt_seal_sha256") != quality.get("seal_sha256")
        ):
            raise QualityRepackError("gate/up proposal no longer binds the exact sealed diagnostic")
        control = proposal.get("baseline_control")
        if not isinstance(control, Mapping):
            raise QualityRepackError("gate/up proposal has no baseline rollback control")
        if (
            control.get("manifest_path") != str(self.baseline_manifest_path)
            or control.get("manifest_seal_sha256") != manifest.get("seal_sha256")
            or control.get("admission_path") != str(self.baseline_admission_path)
            or control.get("admission_seal_sha256") != admission.get("seal_sha256")
            or control.get("preserve_as_rollback_control") is not True
            or control.get("replacement_forbidden_until_all_acceptance_gates_pass") is not True
        ):
            raise QualityRepackError("gate/up proposal baseline rollback binding changed")
        if manifest.get("status") != complete.COMPLETE_MANIFEST_STATUS:
            raise QualityRepackError("baseline manifest is not a complete candidate control")
        ledger = manifest.get("complete_physical_bpw_ledger")
        if not isinstance(ledger, Mapping) or float(ledger.get("complete_physical_bpw", math.inf)) != float(
            control.get("complete_physical_bpw", math.nan)
        ):
            raise QualityRepackError("gate/up proposal baseline BPW does not match its manifest")
        if admission.get("status") not in {
            "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED",
            "PASS_COMPLETE_BINARY_GRAVITY_ARTIFACT_ADMISSION",
        }:
            raise QualityRepackError("baseline admission is not an artifact-only admission control")
        source_binding = quality.get("source_binding")
        if (
            quality.get("schema") != "hawking.ascension.qwen30_direct_packed_gate_up_quality_diagnostic.v1"
            or quality.get("status")
            != "PASS_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_QUALITY_DIAGNOSTIC_NOT_MODEL_QUALITY"
            or not isinstance(source_binding, Mapping)
            or source_binding.get("revalidation_receipt_path") != str(self.baseline_revalidation_path)
            or source_binding.get("revalidation_receipt_seal_sha256") != revalidation.get("seal_sha256")
        ):
            raise QualityRepackError("gate/up diagnostic no longer binds the immutable source revalidation")
        if revalidation.get("status") != "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED":
            raise QualityRepackError("baseline source receipt is not a complete full-shard revalidation")
        if revalidation.get("source_audit_seal_sha256") != audit.get("seal_sha256"):
            raise QualityRepackError("baseline source revalidation is not bound to the current sealed audit")
        proposed_branch = proposal.get("proposed_candidate_branch")
        organs = proposed_branch.get("initial_organs") if isinstance(proposed_branch, Mapping) else None
        evidence_organs = source_binding.get("tensors") if isinstance(source_binding.get("tensors"), list) else None
        if not isinstance(organs, list) or not isinstance(evidence_organs, list) or len(organs) != 2:
            raise QualityRepackError("gate/up proposal must name exactly its two evidence-backed initial organs")
        selected = tuple(str(item) for item in organs)
        evidence_by_name = {
            str(item.get("tensor_name")): dict(item)
            for item in evidence_organs
            if isinstance(item, Mapping) and isinstance(item.get("tensor_name"), str)
        }
        if set(selected) != set(evidence_by_name) or len(set(selected)) != 2:
            raise QualityRepackError("proposal organs do not exactly match source-bound diagnostic organs")
        for tensor_name in selected:
            shard = weight_map.get(tensor_name)
            evidence = evidence_by_name[tensor_name]
            if (
                shard is None
                or evidence.get("source_shard") != shard
                or evidence.get("source_shard_sha256") != shard_evidence[shard].get("sha256")
                or not isinstance(evidence.get("source_value_sha256"), str)
            ):
                raise QualityRepackError(f"proposal organ lacks a current source binding: {tensor_name}")
        rows = manifest.get("tensors")
        if not isinstance(rows, list) or len(rows) != len(weight_map):
            raise QualityRepackError("baseline manifest does not have a full tensor catalog")
        baseline_rows = {
            str(row.get("tensor_name")): dict(row)
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("tensor_name"), str)
        }
        if set(baseline_rows) != set(weight_map):
            raise QualityRepackError("baseline manifest catalog does not exactly match source catalog")
        self._baseline_rows = baseline_rows
        self._baseline_control = {
            "manifest": _file_binding(self.baseline_manifest_path, label="baseline manifest"),
            "admission": _file_binding(self.baseline_admission_path, label="baseline admission"),
            "source_revalidation": _file_binding(
                self.baseline_revalidation_path, label="baseline source revalidation"
            ),
            "complete_physical_bpw": float(control["complete_physical_bpw"]),
            "preserve_as_rollback_control": True,
            "replacement_forbidden_until_all_acceptance_gates_pass": True,
        }
        self._proposal_context = {
            "proposal": _file_binding(self.proposal_path, label="gate/up quality repack proposal"),
            "quality_receipt": _file_binding(
                self.quality_receipt_path, label="gate/up quality diagnostic"
            ),
            "selected_organs": list(selected),
            "evidence_by_name": evidence_by_name,
            "source_content_identity_sha256": source_binding.get("source_content_identity_sha256"),
        }

    def _revalidate_current_source(
        self,
        *,
        audit: Mapping[str, Any],
        weight_map: Mapping[str, str],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        """Reuse only the immutable admitted-control receipt; never rewrite it."""

        binding = self._source_revalidation_binding(
            audit=audit, weight_map=weight_map, shard_evidence=shard_evidence
        )
        existing = _sealed(self.baseline_revalidation_path, label="immutable Qwen30 source revalidation")
        if not self._receipt_matches_current_source(
            existing,
            binding=binding,
            shard_evidence=shard_evidence,
            weight_map=weight_map,
        ):
            raise QualityRepackError(
                "immutable source revalidation no longer matches the exact local Qwen30 source; "
                "refusing to create or overwrite a quality candidate"
            )
        self._current_revalidation = existing
        self._ensure_selection(
            audit=audit,
            weight_map=weight_map,
            shard_evidence=shard_evidence,
            revalidation=existing,
        )
        return existing, False

    def _read_source_tensor(
        self, *, tensor_name: str, shard: str, expected_source_hash: str
    ) -> tuple[bytes, np.ndarray, list[int], str]:
        source_path = self.model_dir / shard
        current = self._current_revalidation
        if current is None:
            raise QualityRepackError("source revalidation has not been admitted")
        rows = current.get("shards")
        expected_identity = rows.get(shard, {}).get("file_identity") if isinstance(rows, Mapping) else None
        before = complete._file_identity(source_path, label=f"source shard {shard}")
        if before != expected_identity:
            raise QualityRepackError(f"source shard identity diverged from immutable revalidation: {shard}")
        header = self._header(source_path)
        info = header.get(tensor_name)
        if not isinstance(info, Mapping):
            raise QualityRepackError(f"source header lacks selected tensor {tensor_name}")
        dtype = str(info.get("dtype"))
        shape = [int(item) for item in info.get("shape", [])]
        offsets = info.get("data_offsets")
        if not shape or not isinstance(offsets, list) or len(offsets) != 2:
            raise QualityRepackError(f"invalid source tensor metadata: {tensor_name}")
        begin, end = (int(item) for item in offsets)
        if begin < 0 or end < begin:
            raise QualityRepackError(f"invalid source byte range: {tensor_name}")
        with source_path.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(8 + header_bytes + begin)
            raw = handle.read(end - begin)
        after = complete._file_identity(source_path, label=f"source shard {shard}")
        if before != after:
            raise QualityRepackError(f"source shard changed while reading {tensor_name}")
        values = complete._values_from_raw(raw, dtype, shape)
        return raw, values, shape, dtype

    def _selection_binding(
        self,
        *,
        audit: Mapping[str, Any],
        revalidation: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self._proposal_context or not self._baseline_control:
            raise QualityRepackError("proposal controls have not been loaded")
        return {
            "branch_id": BRANCH_ID,
            "source_audit": {
                "path": str(self.source_audit),
                "document_sha256": _raw_sha256(self.source_audit),
                "seal_sha256": audit["seal_sha256"],
            },
            "immutable_source_revalidation": {
                "path": str(self.baseline_revalidation_path),
                "document_sha256": _raw_sha256(self.baseline_revalidation_path),
                "seal_sha256": revalidation["seal_sha256"],
                "source_revision": revalidation.get("source_revision"),
                "sealed_shard_hashes_sha256": revalidation.get("sealed_shard_hashes_sha256"),
                "weight_map_sha256": revalidation.get("weight_map_sha256"),
            },
            "proposal": self._proposal_context["proposal"],
            "quality_receipt": self._proposal_context["quality_receipt"],
            "baseline_control": self._baseline_control,
            "selected_organs": self._proposal_context["selected_organs"],
            "source_content_identity_sha256": self._proposal_context[
                "source_content_identity_sha256"
            ],
            "residual_fraction_frontier": list(RESIDUAL_FRACTIONS),
            "minimum_pair_relative_l2_improvement": MIN_PAIR_RELATIVE_L2_IMPROVEMENT,
        }

    def _load_existing_selection(self, *, binding: Mapping[str, Any]) -> dict[str, Any] | None:
        existing = complete._read_json(self.selection_path)
        if existing is None:
            return None
        try:
            verified = verify(existing, label=str(self.selection_path))
        except Exception as exc:
            raise QualityRepackError(f"quality selection receipt is not sealed: {exc}") from exc
        if (
            verified.get("schema") != SELECTION_SCHEMA
            or verified.get("status")
            != "EARNED_SOURCE_BOUND_QUALITY_REPACK_SELECTION_UNQUALIFIED"
            or verified.get("binding") != dict(binding)
        ):
            raise QualityRepackError("existing quality selection does not match immutable branch bindings")
        selected = verified.get("selected_representation")
        if not isinstance(selected, Mapping) or not isinstance(selected.get("organs"), list):
            raise QualityRepackError("existing quality selection lacks selected organ payload bindings")
        return verified

    def _ensure_selection(
        self,
        *,
        audit: Mapping[str, Any],
        weight_map: Mapping[str, str],
        shard_evidence: Mapping[str, Mapping[str, Any]],
        revalidation: Mapping[str, Any],
    ) -> None:
        binding = self._selection_binding(audit=audit, revalidation=revalidation)
        snapshot = _immutable_json(
            self.snapshot_path,
            {
                "schema": SOURCE_SNAPSHOT_SCHEMA,
                "status": "EARNED_IMMUTABLE_SOURCE_AND_ROLLBACK_BINDING",
                "binding": binding,
                "claim_boundary": {
                    "baseline_control_is_preserved_and_not_mutated": True,
                    "snapshot_is_source_and_rollback_evidence_not_an_artifact_admission": True,
                    "current_source_is_accepted_only_while_the_original_full_shard_receipt_matches": True,
                },
            },
            label="quality source binding snapshot",
        )
        existing = self._load_existing_selection(binding=binding)
        if existing is not None:
            self._policy = existing
            return
        selected_names = list(self._proposal_context["selected_organs"])
        source_values: dict[str, np.ndarray] = {}
        source_raw: dict[str, bytes] = {}
        source_metadata: dict[str, dict[str, Any]] = {}
        for name in selected_names:
            shard = weight_map[name]
            raw, values, shape, dtype = self._read_source_tensor(
                tensor_name=name,
                shard=shard,
                expected_source_hash=str(shard_evidence[shard]["sha256"]),
            )
            evidence = self._proposal_context["evidence_by_name"][name]
            raw_sha256 = hashlib.sha256(raw).hexdigest()
            value_sha256 = _source_value_sha256(values)
            if (
                value_sha256 != evidence.get("source_value_sha256")
                or list(shape) != list(evidence.get("tensor_shape", []))
            ):
                raise QualityRepackError(f"selected organ source bytes differ from its sealed diagnostic: {name}")
            source_values[name] = values
            source_raw[name] = raw
            source_metadata[name] = {
                "source_shard": shard,
                "source_shard_sha256": shard_evidence[shard]["sha256"],
                "source_value_sha256": value_sha256,
                "source_raw_bf16_sha256": raw_sha256,
                "source_dtype": dtype,
                "shape": shape,
            }
        gate_name, up_name = selected_names
        sweep: list[dict[str, Any]] = []
        reconstructed_by_fraction: dict[float, dict[str, np.ndarray]] = {}
        payload_by_fraction: dict[float, dict[str, tuple[bytes, dict[str, Any]]]] = {}
        for fraction in (0.0, *RESIDUAL_FRACTIONS):
            reconstructed: dict[str, np.ndarray] = {}
            packed: dict[str, tuple[bytes, dict[str, Any]]] = {}
            organs: list[dict[str, Any]] = []
            for name in selected_names:
                values = source_values[name]
                if fraction == 0.0:
                    payload, quality, rebuilt = complete._pack_binary(values, values.shape)
                    rebuilt = np.ascontiguousarray(rebuilt, dtype=np.float32).reshape(values.shape)
                    metadata = {
                        "family": "binary_sign_scale_control",
                        "physical_payload_bytes": len(payload),
                        "base_reconstruction_quality": quality,
                    }
                else:
                    payload, metadata, rebuilt = _pack_sparse_residual(
                        values, values.shape, fraction=fraction
                    )
                reconstructed[name] = rebuilt
                packed[name] = (payload, metadata)
                organs.append(
                    {
                        "tensor_name": name,
                        "physical_payload_bytes": len(payload),
                        "component_physical_bpw": len(payload) * 8.0 / int(values.size),
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "source_to_packed_weight_metrics": _metric(values, rebuilt),
                        "source_to_packed_projection": _projection_discriminator(
                            values, rebuilt, label=f"{BRANCH_ID}:{name}"
                        ),
                        "residual": metadata.get("residual"),
                    }
                )
            pair = _pair_swiglu_discriminator(
                source_values[gate_name],
                source_values[up_name],
                reconstructed[gate_name],
                reconstructed[up_name],
            )
            sweep.append(
                {
                    "residual_fraction": fraction,
                    "organs": organs,
                    "pair_source_to_packed_swiglu": pair,
                    "all_component_bpw_at_or_below_1_5": all(
                        float(row["component_physical_bpw"]) <= 1.5 for row in organs
                    ),
                }
            )
            reconstructed_by_fraction[fraction] = reconstructed
            payload_by_fraction[fraction] = packed
        baseline_pair = float(sweep[0]["pair_source_to_packed_swiglu"]["metrics"]["relative_l2"])
        chosen: dict[str, Any] | None = None
        for candidate in sweep[1:]:
            relative_l2 = float(candidate["pair_source_to_packed_swiglu"]["metrics"]["relative_l2"])
            improvement = (baseline_pair - relative_l2) / max(baseline_pair, 1e-12)
            candidate["pair_relative_l2_improvement_fraction"] = improvement
            if (
                chosen is None
                and candidate["all_component_bpw_at_or_below_1_5"]
                and improvement >= MIN_PAIR_RELATIVE_L2_IMPROVEMENT
            ):
                chosen = candidate
        if chosen is None:
            raise QualityRepackError(
                "none of the bounded residual fractions improved the source-bound gate/up control "
                "enough to justify a full candidate repack"
            )
        chosen_fraction = float(chosen["residual_fraction"])
        organs: list[dict[str, Any]] = []
        for name in selected_names:
            payload, metadata = payload_by_fraction[chosen_fraction][name]
            rebuilt = reconstructed_by_fraction[chosen_fraction][name]
            organs.append(
                {
                    "tensor_name": name,
                    **source_metadata[name],
                    "representation": metadata,
                    "physical_payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "physical_payload_bytes": len(payload),
                    "source_to_packed_weight_metrics": _metric(source_values[name], rebuilt),
                    "source_to_packed_projection": _projection_discriminator(
                        source_values[name], rebuilt, label=f"{BRANCH_ID}:{name}"
                    ),
                }
            )
        selection = _immutable_json(
            self.selection_path,
            {
                "schema": SELECTION_SCHEMA,
                "status": "EARNED_SOURCE_BOUND_QUALITY_REPACK_SELECTION_UNQUALIFIED",
                "binding": binding,
                "source_binding_snapshot": _file_binding(
                    self.snapshot_path, label="quality source binding snapshot"
                ),
                "selection_method": {
                    "kind": "smallest_residual_fraction_with_measured_gate_up_swiglu_relative_l2_improvement",
                    "fractions_evaluated": [0.0, *RESIDUAL_FRACTIONS],
                    "minimum_pair_relative_l2_improvement": MIN_PAIR_RELATIVE_L2_IMPROVEMENT,
                    "control_count": CONTROL_COUNT,
                    "tie_break": "ascending_flat_index_at_equal_absolute_binary_reconstruction_error",
                },
                "quality_sweep": sweep,
                "selected_representation": {
                    "family": "binary_sign_scale_sparse_fp16_residual",
                    "residual_fraction": chosen_fraction,
                    "baseline_pair_relative_l2": baseline_pair,
                    "chosen_pair_relative_l2": chosen["pair_source_to_packed_swiglu"]["metrics"][
                        "relative_l2"
                    ],
                    "pair_relative_l2_improvement_fraction": chosen[
                        "pair_relative_l2_improvement_fraction"
                    ],
                    "organs": organs,
                },
                "rollback": {
                    "baseline_control": self._baseline_control,
                    "automatic_baseline_replacement_forbidden": True,
                    "rollback_action": "retain and use the separately admitted 1.130366 BPW Qwen30 control",
                },
                "claim_boundary": {
                    "selection_is_a_source_bound_component_quality_measurement_only": True,
                    "full_artifact_accounting_native_admission_runtime_hcli_capability_tps_tg_and_tournament_are_unearned": True,
                    "the_candidate_may_not_replace_the_admitted_control_automatically": True,
                },
            },
            label="quality repack selection receipt",
        )
        self._policy = selection

    def _policy_organ(self, tensor_name: str) -> dict[str, Any] | None:
        if self._policy is None:
            raise QualityRepackError("quality selection has not been loaded")
        selected = self._policy.get("selected_representation")
        organs = selected.get("organs") if isinstance(selected, Mapping) else None
        if not isinstance(organs, list):
            raise QualityRepackError("quality selection has no organ list")
        for organ in organs:
            if isinstance(organ, Mapping) and organ.get("tensor_name") == tensor_name:
                return dict(organ)
        return None

    def _baseline_rollback(self, tensor_name: str) -> dict[str, Any]:
        baseline = self._baseline_rows.get(tensor_name)
        if not isinstance(baseline, Mapping):
            raise QualityRepackError(f"baseline rollback catalog is missing {tensor_name}")
        return {
            "baseline_manifest_path": str(self.baseline_manifest_path),
            "baseline_manifest_seal_sha256": self._baseline_control["manifest"]["seal_sha256"],
            "baseline_artifact_path": baseline.get("artifact_path"),
            "baseline_artifact_sha256": baseline.get("artifact_sha256"),
            "baseline_artifact_bytes": baseline.get("artifact_bytes"),
            "rollback_action": "use the separately admitted baseline tensor; this candidate never overwrites it",
        }

    def _write_tensor(
        self, *, tensor_name: str, shard: str, source_hash: str, info: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self._can_stream_direct_tensor(tensor_name=tensor_name, info=info):
            return self._stream_direct_binary_tensor(
                tensor_name=tensor_name,
                shard=shard,
                source_hash=source_hash,
                info=info,
            )
        # Read the current header under the immutable shard identity below;
        # the coordinator-supplied metadata is only a bounded scheduling hint.
        del info
        raw, values, shape, dtype = self._read_source_tensor(
            tensor_name=tensor_name, shard=shard, expected_source_hash=source_hash
        )
        organ = self._policy_organ(tensor_name)
        if organ is None:
            payload, quality, _ = complete._pack_binary(values, shape)
            layout: dict[str, Any] = {
                "family": "binary_sign_scale_control_reencoded_from_source",
                "magic": complete.MAGIC.decode("ascii"),
                "version": complete.VERSION,
                "group_size": complete.GROUP_SIZE,
                "sign_bit_order": "little",
                "scale_dtype": "float16",
            }
            mutation: dict[str, Any] = {
                "changed_from_admitted_control": False,
                "reason": "not one of the two sealed proposal organs",
                "source_to_packed_discriminator": {
                    "state": "not_required_for_unchanged_control_layout",
                    "source_value_sha256": _source_value_sha256(values),
                    "source_raw_bf16_sha256": hashlib.sha256(raw).hexdigest(),
                },
            }
        else:
            selected = self._policy.get("selected_representation") if self._policy else None
            fraction = selected.get("residual_fraction") if isinstance(selected, Mapping) else None
            if not isinstance(fraction, (float, int)):
                raise QualityRepackError("quality selection has no residual fraction")
            if _source_value_sha256(values) != organ.get("source_value_sha256"):
                raise QualityRepackError(f"source payload changed after quality selection: {tensor_name}")
            payload, residual_layout, rebuilt = _pack_sparse_residual(values, shape, fraction=float(fraction))
            if hashlib.sha256(payload).hexdigest() != organ.get("physical_payload_sha256"):
                raise QualityRepackError(f"deterministic selected payload changed: {tensor_name}")
            if len(payload) != int(organ.get("physical_payload_bytes", -1)):
                raise QualityRepackError(f"selected payload byte count changed: {tensor_name}")
            expected_representation = organ.get("representation")
            if not isinstance(expected_representation, Mapping) or residual_layout != expected_representation:
                raise QualityRepackError(f"selected residual metadata changed: {tensor_name}")
            quality = _metric(values, rebuilt)
            layout = residual_layout
            mutation = {
                "changed_from_admitted_control": True,
                "reason": "explicitly selected by the sealed gate/up quality-repack proposal",
                "selection_receipt_path": str(self.selection_path),
                "selection_receipt_seal_sha256": self._policy["seal_sha256"],
                "source_to_packed_discriminator": {
                    "source_value_sha256": organ["source_value_sha256"],
                    "weight_metrics": organ["source_to_packed_weight_metrics"],
                    "projection": organ["source_to_packed_projection"],
                    "payload_sha256": organ["physical_payload_sha256"],
                },
            }
        destination = self.tensor_dir / complete._artifact_name(tensor_name)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=self.tensor_dir)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o640)
            os.replace(temporary, destination)
            os.chmod(destination, 0o640)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {
            "tensor_name": tensor_name,
            "source_shard": shard,
            "source_shard_sha256": source_hash,
            "source_dtype": dtype,
            "shape": shape,
            "elements": complete._tensor_count(shape),
            "artifact_path": str(destination),
            "artifact_bytes": len(payload),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "layout": layout,
            "component_quality": quality,
            "candidate_mutation": {
                **mutation,
                "baseline_rollback": self._baseline_rollback(tensor_name),
            },
        }

    def _progress_row_is_usable(
        self,
        *,
        tensor_name: str,
        shard: str,
        source_hash: str,
        row: Mapping[str, Any],
    ) -> bool:
        if not super()._progress_row_is_usable(
            tensor_name=tensor_name, shard=shard, source_hash=source_hash, row=row
        ):
            return False
        organ = self._policy_organ(tensor_name)
        mutation = row.get("candidate_mutation")
        if not isinstance(mutation, Mapping) or not isinstance(mutation.get("baseline_rollback"), Mapping):
            return False
        if organ is None:
            return mutation.get("changed_from_admitted_control") is False
        if mutation.get("changed_from_admitted_control") is not True:
            return False
        try:
            payload = Path(str(row["artifact_path"])).read_bytes()
            unpacked, _ = _unpack_sparse_residual(payload)
        except (OSError, QualityRepackError):
            return False
        representation = organ.get("representation")
        residual = representation.get("residual") if isinstance(representation, Mapping) else None
        return (
            hashlib.sha256(payload).hexdigest() == organ.get("physical_payload_sha256")
            and unpacked.get("selected_count") == residual.get("selected_count")
            and unpacked.get("indices_sha256") == residual.get("indices_sha256")
            and unpacked.get("values_sha256") == residual.get("values_sha256")
        ) if isinstance(residual, Mapping) else False

    def _manifest_representation(self) -> dict[str, Any]:
        selected = self._policy.get("selected_representation") if self._policy else {}
        return {
            "family": "mixed_direct_binary_sign_scale_plus_selected_sparse_fp16_residual",
            "unchanged_tensor_layout": "HQ30G1B1 binary sign plus FP16 group scale",
            "selected_organ_layout": "HQ30GR2 sparse FP16 residual over HQ30G1B1 base",
            "selected_organs": self._proposal_context.get("selected_organs", []),
            "selected_residual_fraction": selected.get("residual_fraction")
            if isinstance(selected, Mapping)
            else None,
            "native_reader_requirement": "a new exact sparse-residual reader and full native admission are required before any runtime use",
            "physical_direct_layout": True,
        }

    def _manifest_champion_classes(self, *, complete_bpw: float) -> dict[str, Any]:
        return {
            "current_bpw_candidate": {
                "candidate": BRANCH_ID,
                "complete_physical_bpw": complete_bpw,
                "status": "CANDIDATE_ONLY_NOT_A_BASELINE_REPLACEMENT",
            },
            "admitted_baseline_rollback_control": {
                "complete_physical_bpw": self._baseline_control.get("complete_physical_bpw"),
                "manifest_seal_sha256": self._baseline_control.get("manifest", {}).get("seal_sha256"),
                "status": "PRESERVED_SEPARATE_CONTROL",
            },
            "runtime_capability_hcli_tps_tg": {
                "status": "BLOCKED_BY_REQUIRED_NEW_NATIVE_READER_ADMISSION_AND_FULL_RETEST",
                "timing_or_capability_transfer_from_baseline_forbidden": True,
            },
        }

    @staticmethod
    def _manifest_quality_summary(ordered: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        changed = [
            row
            for row in ordered
            if isinstance(row.get("candidate_mutation"), Mapping)
            and row["candidate_mutation"].get("changed_from_admitted_control") is True
        ]
        return {
            "mean_component_cosine": float(np.mean([row["component_quality"]["cosine"] for row in ordered])),
            "mean_component_relative_l2": float(
                np.mean([row["component_quality"]["relative_l2"] for row in ordered])
            ),
            "changed_organs": len(changed),
            "verdict": "SOURCE_BOUND_QUALITY_REPACK_CANDIDATE_UNQUALIFIED_REQUIRES_INDEPENDENT_ADMISSION_AND_ALL_RUNTIME_CAPABILITY_GATES",
        }

    @staticmethod
    def _manifest_claim_boundary() -> dict[str, Any]:
        return {
            "complete_physical_tensor_coverage_is_true_only_after_terminal_catalog_admission": True,
            "complete_bpw_is_real_accounted_bytes_not_a_capability_result": True,
            "admitted_baseline_is_preserved_and_automatic_replacement_is_forbidden": True,
            "native_admission_runtime_generation_hcli_agent_os_tps_tg_and_tournament_are_unearned": True,
            "raw_source_remains_authority_teacher_only": True,
        }

    def _manifest_extra_fields(
        self,
        *,
        ordered: Sequence[Mapping[str, Any]],
        artifact_bytes: int,
        elements: int,
        complete_bpw: float,
    ) -> dict[str, Any]:
        del elements
        changed = [
            row["tensor_name"]
            for row in ordered
            if isinstance(row.get("candidate_mutation"), Mapping)
            and row["candidate_mutation"].get("changed_from_admitted_control") is True
        ]
        return {
            "quality_repack_branch": {
                "branch_id": BRANCH_ID,
                "source_binding_snapshot": _file_binding(
                    self.snapshot_path, label="quality source binding snapshot"
                ),
                "selection_receipt": _file_binding(
                    self.selection_path, label="quality repack selection receipt"
                ),
                "changed_organs": changed,
                "unchanged_reencoded_tensor_count": len(ordered) - len(changed),
                "tensor_payload_bytes_before_manifest": artifact_bytes,
                "candidate_complete_bpw_before_manifest_fixed_point": complete_bpw,
                "baseline_rollback_control": self._baseline_control,
                "admission_state": "NOT_REQUESTED_REQUIRES_EXACT_SPARSE_RESIDUAL_NATIVE_READER",
                "mandatory_follow_on_gates": [
                    "full_native_artifact_admission",
                    "all_layer_no_fallback_exact_token_and_autoregressive_generation",
                    "fresh_hcli_context_kv_agent_os_restart_and_storage_rollback",
                    "fresh_complete_token_profile_and_clean_base_true_tps_tg",
                ],
            }
        }

    def validate(self) -> dict[str, Any]:
        audit, weight_map, shard_evidence = self._admit_source()
        revalidation, _ = self._revalidate_current_source(
            audit=audit, weight_map=weight_map, shard_evidence=shard_evidence
        )
        if self._policy is None:
            raise QualityRepackError("quality selection was not established")
        selected = self._policy["selected_representation"]
        self._publish(
            "SOURCE_BOUND_QUALITY_REPACK_READY",
            source_binding_snapshot_path=str(self.snapshot_path),
            source_binding_snapshot_seal_sha256=_sealed(
                self.snapshot_path, label="quality source binding snapshot"
            )["seal_sha256"],
            selection_receipt_path=str(self.selection_path),
            selection_receipt_seal_sha256=self._policy["seal_sha256"],
            immutable_source_revalidation_seal_sha256=revalidation["seal_sha256"],
            selected_organs=[row["tensor_name"] for row in selected["organs"]],
            residual_fraction=selected["residual_fraction"],
            claim_boundary={
                "validation_does_not_write_or_admit_a_complete_candidate": True,
                "baseline_control_remains_unchanged": True,
            },
        )
        return self._policy

    def run(
        self,
        *,
        max_tensors: int,
        workers: int = DEFAULT_PARALLEL_WORKERS,
        memory_budget_bytes: int = DEFAULT_PARALLEL_MEMORY_BUDGET_MIB * _MIB,
    ) -> int:
        """Pack a bounded quality-candidate wave with one journal coordinator.

        This override intentionally leaves the baseline compiler untouched.
        ``workers`` are short-lived thread workers for atomic tensor payload
        materialization only.  The coordinator alone owns all status, JSONL,
        progress-index, ledger, manifest, and terminal writes; when the final
        source-bound cursor is reached it delegates back to the proven serial
        finalization path in :class:`CompleteBinaryGravity`.
        """

        if max_tensors <= 0:
            raise QualityRepackError("max_tensors must be positive")
        if workers <= 0:
            raise QualityRepackError("parallel workers must be positive")
        if memory_budget_bytes <= 0:
            raise QualityRepackError("parallel memory budget must be positive")
        with self._coordinator_lease():
            if workers == 1:
                return super().run(max_tensors=max_tensors)
            audit, weight_map, shard_evidence = self._admit_source()
            revalidation, revalidated_now = self._revalidate_current_source(
                audit=audit,
                weight_map=weight_map,
                shard_evidence=shard_evidence,
            )
            progress_binding = self._progress_binding(
                audit_seal=str(audit["seal_sha256"]),
                revalidation_seal=str(revalidation["seal_sha256"]),
            )
            planned_order = self._planned_tensor_order(weight_map)
            if len(planned_order) != len(weight_map):
                raise QualityRepackError("source index does not produce one deterministic entry per tensor")
            progress, progress_journal, scheduler = self._load_progress_index(
                binding=progress_binding,
                planned_order=planned_order,
                shard_evidence=shard_evidence,
            )
            self.tensor_dir.mkdir(parents=True, exist_ok=True)
            planned = len(planned_order)
            completed = int(scheduler["source_bound_completed_tensors"])
            if completed == planned and int(scheduler["next_cursor"]) == planned:
                # Keep final manifest/terminal construction exactly in the
                # existing all-row implementation.  It performs no new tensor
                # writes from this completed state and remains under the lease.
                return super().run(max_tensors=1)

            self._publish(
                "PACKING_COMPLETE_BINARY_GRAVITY_PARALLEL",
                source_revalidation={
                    "receipt_path": str(self.source_revalidation_path),
                    "receipt_seal_sha256": revalidation["seal_sha256"],
                    "full_shards_revalidated_this_cycle": revalidated_now,
                },
                parallel_scheduler={
                    "mode": "shard_disjoint_atomic_workers_single_coordinator_v1",
                    "workers_requested": workers,
                    "configured_memory_budget_bytes": memory_budget_bytes,
                    "fixed_planned_order_sha256": complete._canonical_sha256(list(planned_order)),
                },
                progress={
                    "planned_tensors": planned,
                    "completed_tensors": completed,
                    "batch_limit": max_tensors,
                    "progress_index_path": str(self.progress_index_path),
                    "next_cursor": scheduler["next_cursor"],
                    "next_source_shard": scheduler["next_source_shard"],
                    "next_tensor_name": scheduler["next_tensor_name"],
                },
            )
            performed = 0
            run_started = time.monotonic()
            first_memory = self._memory_snapshot()
            while performed < max_tensors and int(scheduler["next_cursor"]) < planned:
                effective_budget, before_memory = self._parallel_memory_budget(memory_budget_bytes)
                wave_limit = min(workers, max_tensors - performed)
                work, scheduling = self._select_shard_disjoint_parallel_work(
                    planned_order=planned_order,
                    progress=progress,
                    shard_evidence=shard_evidence,
                    worker_limit=wave_limit,
                    memory_budget_bytes=effective_budget,
                )
                if not work:
                    self._write_progress_index(
                        completed=progress,
                        journal=progress_journal,
                        binding=progress_binding,
                        scheduler=scheduler,
                    )
                    self._publish(
                        "MEMORY_BACKPRESSURE_WAITING_QUALITY_REPACK",
                        source_revalidation={
                            "receipt_path": str(self.source_revalidation_path),
                            "receipt_seal_sha256": revalidation["seal_sha256"],
                        },
                        parallel_scheduler={
                            "mode": "shard_disjoint_atomic_workers_single_coordinator_v1",
                            "workers_requested": workers,
                            "effective_memory_budget_bytes": effective_budget,
                            "memory_before": before_memory,
                            **scheduling,
                        },
                        progress={
                            "planned_tensors": planned,
                            "completed_tensors": int(scheduler["source_bound_completed_tensors"]),
                            "resume_required": True,
                            "next_cursor": scheduler["next_cursor"],
                            "next_source_shard": scheduler["next_source_shard"],
                            "next_tensor_name": scheduler["next_tensor_name"],
                        },
                    )
                    return 0
                wave_started = time.monotonic()
                # Workers receive disjoint source shards and distinct,
                # deterministic output files.  They do not receive the journal
                # handle, index object, or any mutable coordinator state.
                with ThreadPoolExecutor(max_workers=len(work), thread_name_prefix="q30-quality-pack") as pool:
                    futures = {
                        int(item["planned_ordinal"]): pool.submit(
                            self._write_tensor,
                            tensor_name=str(item["tensor_name"]),
                            shard=str(item["shard"]),
                            source_hash=str(item["source_hash"]),
                            info=item["info"],
                        )
                        for item in work
                    }
                    resolved = [(ordinal, future.result()) for ordinal, future in futures.items()]
                # Completion timing is intentionally irrelevant: rows are
                # committed in the fixed planned order so the journal remains
                # deterministic and every append/index update is single-writer.
                for ordinal, row in sorted(resolved, key=lambda item: item[0]):
                    if str(row.get("tensor_name")) in progress:
                        raise QualityRepackError(
                            f"parallel coordinator observed duplicate resolved tensor row at ordinal {ordinal}"
                        )
                    appended = self._append_progress(row)
                    progress_journal = self._advance_progress_index(
                        completed=progress,
                        journal=progress_journal,
                        row=row,
                        appended=appended,
                    )
                    scheduler = self._advance_scheduler(
                        scheduler=scheduler,
                        completed=progress,
                        planned_order=planned_order,
                        shard_evidence=shard_evidence,
                    )
                    performed += 1
                after_memory = self._memory_snapshot()
                elapsed = max(time.monotonic() - wave_started, 1e-9)
                completed = int(scheduler["source_bound_completed_tensors"])
                payload_bytes = sum(int(item["artifact_bytes"]) for item in progress.values())
                self._publish(
                    "PACKING_COMPLETE_BINARY_GRAVITY_PARALLEL",
                    current_tensor=str(sorted(resolved, key=lambda item: item[0])[-1][1]["tensor_name"]),
                    source_revalidation={
                        "receipt_path": str(self.source_revalidation_path),
                        "receipt_seal_sha256": revalidation["seal_sha256"],
                        "full_shards_revalidated_this_cycle": revalidated_now,
                    },
                    parallel_scheduler={
                        "mode": "shard_disjoint_atomic_workers_single_coordinator_v1",
                        "workers_requested": workers,
                        "workers_dispatched": len(work),
                        "effective_memory_budget_bytes": effective_budget,
                        "memory_before": before_memory,
                        "memory_after": after_memory,
                        "swapouts_pages_delta": (
                            int(after_memory["swapouts_pages"]) - int(before_memory["swapouts_pages"])
                            if isinstance(after_memory.get("swapouts_pages"), int)
                            and isinstance(before_memory.get("swapouts_pages"), int)
                            else None
                        ),
                        "rss_bytes": self._rss_bytes(),
                        "wave_elapsed_seconds": elapsed,
                        "wave_tensors_per_second": len(work) / elapsed,
                        **scheduling,
                    },
                    progress={
                        "planned_tensors": planned,
                        "completed_tensors": completed,
                        "new_tensors_this_cycle": performed,
                        "artifact_bytes": payload_bytes,
                        "batch_limit": max_tensors,
                        "next_cursor": scheduler["next_cursor"],
                        "next_source_shard": scheduler["next_source_shard"],
                        "next_tensor_name": scheduler["next_tensor_name"],
                    },
                )
            self._write_progress_index(
                completed=progress,
                journal=progress_journal,
                binding=progress_binding,
                scheduler=scheduler,
            )
            run_elapsed = max(time.monotonic() - run_started, 1e-9)
            if int(scheduler["next_cursor"]) == planned:
                # Finalization observes fully fsync'd progress after this
                # coordinator wave; it cannot see a worker's temporary file.
                return super().run(max_tensors=1)
            self._publish(
                "PACKING_COMPLETE_BINARY_GRAVITY_PARALLEL",
                source_revalidation={
                    "receipt_path": str(self.source_revalidation_path),
                    "receipt_seal_sha256": revalidation["seal_sha256"],
                },
                parallel_scheduler={
                    "mode": "shard_disjoint_atomic_workers_single_coordinator_v1",
                    "workers_requested": workers,
                    "cycle_tensors": performed,
                    "cycle_elapsed_seconds": run_elapsed,
                    "cycle_tensors_per_second": performed / run_elapsed,
                    "memory_at_cycle_start": first_memory,
                    "memory_at_cycle_end": self._memory_snapshot(),
                    "rss_bytes": self._rss_bytes(),
                },
                progress={
                    "planned_tensors": planned,
                    "completed_tensors": int(scheduler["source_bound_completed_tensors"]),
                    "new_tensors_this_cycle": performed,
                    "resume_required": True,
                    "progress_index_path": str(self.progress_index_path),
                    "next_cursor": scheduler["next_cursor"],
                    "next_source_shard": scheduler["next_source_shard"],
                    "next_tensor_name": scheduler["next_tensor_name"],
                },
            )
            return 0

    def watch(
        self,
        *,
        max_tensors: int,
        idle_seconds: float,
        workers: int = DEFAULT_PARALLEL_WORKERS,
        memory_budget_bytes: int = DEFAULT_PARALLEL_MEMORY_BUDGET_MIB * _MIB,
    ) -> int:
        if max_tensors <= 0:
            raise QualityRepackError("max_tensors must be positive")
        if idle_seconds < 0.0:
            raise QualityRepackError("idle_seconds must be non-negative")
        stop = False

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop
            stop = True

        previous_int = signal.signal(signal.SIGINT, request_stop)
        previous_term = signal.signal(signal.SIGTERM, request_stop)
        try:
            while not stop:
                self.run(
                    max_tensors=max_tensors,
                    workers=workers,
                    memory_budget_bytes=memory_budget_bytes,
                )
                terminal = complete._read_json(self.terminal_receipt_path)
                if terminal is not None:
                    try:
                        verified = verify(terminal, label=str(self.terminal_receipt_path))
                    except Exception:
                        verified = {}
                    if verified.get("status") == complete.COMPLETE_CANDIDATE_PHASE:
                        return 0
                if idle_seconds:
                    time.sleep(idle_seconds)
            self._publish("QUALITY_REPACK_STOPPED_BY_SIGNAL")
            return 0
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--root", type=Path, default=QUALITY_ROOT)
    parser.add_argument("--proposal", type=Path, default=GATE_UP_PROPOSAL)
    parser.add_argument("--quality-receipt", type=Path, default=GATE_UP_QUALITY_RECEIPT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--max-tensors", type=int, default=32)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help="bounded shard-disjoint tensor workers; the coordinator remains single-writer",
    )
    parser.add_argument(
        "--memory-budget-mib",
        type=int,
        default=DEFAULT_PARALLEL_MEMORY_BUDGET_MIB,
        help="maximum estimated transient worker memory before backpressure",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worker = QualityRepackGravity(
        model_dir=args.model_dir,
        source_audit=args.source_audit,
        root=args.root,
        proposal_path=args.proposal,
        quality_receipt_path=args.quality_receipt,
        baseline_root=args.baseline_root,
    )
    if args.validate_only:
        worker.validate()
        return 0
    if args.workers <= 0 or args.memory_budget_mib <= 0:
        raise SystemExit("--workers and --memory-budget-mib must be positive")
    if args.watch:
        return worker.watch(
            max_tensors=args.max_tensors,
            idle_seconds=args.idle_seconds,
            workers=args.workers,
            memory_budget_bytes=args.memory_budget_mib * _MIB,
        )
    return worker.run(
        max_tensors=args.max_tensors,
        workers=args.workers,
        memory_budget_bytes=args.memory_budget_mib * _MIB,
    )


if __name__ == "__main__":
    raise SystemExit(main())
