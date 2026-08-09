"""Durable deterministic physical Gravity workers for the two Qwen families.

This module deliberately owns *candidate research only*.  It turns the local,
verified Qwen30 and Qwen80 BF16 bodies into a long-running population of real
packed tensor/organ artifacts.  The raw bodies remain teachers/source
authorities; no result here is a manager, capability, HCLI, TPS, or tournament
receipt.

Unlike the old one-shot scouts, each worker has a stable source-content ID,
deterministic genome stream, sealed candidate records, artifact hashes,
Pareto/diversity frontiers, a negative-science lookup before proposal, and a
cross-family knowledge ledger.  It is intentionally safe to run under
``launchd`` for weeks: every successful mutation is atomically committed before
the sequence cursor advances, and disk headroom stops new artifacts rather than
silently evicting evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import sqlite3
import stat
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical"
LIFECYCLE_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/lifecycle"
FAMILY_ROOT = PHYSICAL_ROOT / "qwen-family" / "dual-gravity"
DASHBOARD_PATH = LIFECYCLE_ROOT / "ASCENSION_DUAL_MANAGER_PHYSICAL_STATUS.json"

SCHEMA = "hawking.ascension.dual_gravity_worker.v1"
IDENTITY_SCHEMA = "hawking.ascension.qwen_source_content_identity.v1"
SOURCE_REVALIDATION_SCHEMA = "hawking.ascension.complete_binary_source_revalidation.v1"
SOURCE_REVALIDATION_STATUS = "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED"
CANDIDATE_SCHEMA = "hawking.ascension.gravity_physical_candidate.v1"
FRONTIER_SCHEMA = "hawking.ascension.gravity_pareto_frontier.v1"
CHAMPION_SCHEMA = "hawking.ascension.gravity_champions.v1"
INDEX_SCHEMA = "hawking.ascension.gravity_candidate_index.v1"
MECHANISM_SCHEMA = "hawking.ascension.knowledge_representation_genome.v1"
NEGATIVE_SCHEMA = "hawking.ascension.negative_science.v1"
TRANSFER_SCHEMA = "hawking.ascension.transfer_matrix.v1"

CONTROL_FILENAMES = (
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
)
REQUIRED_CONTROL_FILENAMES = (
    "model.safetensors.index.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

MIN_FREE_BYTES = 160 * 1024**3
MAX_ARTIFACT_BYTES = 64 * 1024**2
MAX_REGION_ELEMENTS = 2_097_152
FRONTIER_LIMIT = 24
GROUP_BINARY = 128
GROUP_UNIFORM = 64
MAGIC_BINARY = b"HGRAVB01"
MAGIC_UNIFORM = b"HGRAVU01"
MAGIC_TERNARY = b"HGRAVT01"
MAGIC_RESIDUAL = b"HGRAVR01"
MAGIC_LOWRANK = b"HGRAVL01"
MAGIC_HADAMARD = b"HGRAVH01"
MAGIC_ADDITIVE = b"HGRAVA01"
MAGIC_ACTIVATION = b"HGRAVC01"
GPU_LEASE_STATUS_PATH = FAMILY_ROOT / "GPU_LEASE_STATUS.json"
GPU_LEASE_LOCK_PATH = FAMILY_ROOT / ".gpu-lease.lock"

# The first worker only knew these seven families. They are an immutable
# schedule contract: sequence positions below the v2 boundary must keep their
# original target/family/config mapping forever, including after future code
# revisions. New families are introduced only at the next complete legacy
# generation, so no existing stream position is silently reinterpreted.
LEGACY_REPRESENTATION_SCHEDULE_VERSION = "v1_legacy_seven_family"
EXPANDED_REPRESENTATION_SCHEDULE_VERSION = "v2_expanded_ten_family"
# This is a minimum only. On first upgraded launch, an existing worker seals
# the *next* full legacy generation boundary in WORKER_STATE so an already
# advancing v1 stream can never be remapped halfway through a generation.
REPRESENTATION_EXPANSION_START_GENERATION = 1
REPRESENTATION_SCHEDULE_SCHEMA = "hawking.ascension.representation_schedule_migration.v1"
LEGACY_REPRESENTATIONS = (
    "binary_sign_scale128",
    "uniform_q2_group64",
    "uniform_q3_group64",
    "ternary_threshold_group128",
    "binary_outlier_residual",
    "teacher_low_rank_q3",
    "uniform_q4_group64",
)
EXPANDED_REPRESENTATIONS = (
    *LEGACY_REPRESENTATIONS,
    "hadamard_lattice_q3_group128",
    "additive_residual_codebook_q2x2",
    "activation_corrected_rowwise_q3",
)


class DualGravityError(RuntimeError):
    """A physical worker cannot safely make the requested mutation."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    repository: str
    revision: str
    source_dir: Path
    source_audit: Path
    root: Path
    legacy_status: Path
    architecture: str
    model_family: str
    expert_count: int
    top_k: int
    query_heads: int
    kv_heads: int
    head_dim: int


SPECS: dict[str, ModelSpec] = {
    "qwen30": ModelSpec(
        key="qwen30",
        model_id="Qwen3-Coder-30B-A3B-Instruct",
        repository="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        revision="b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        source_dir=REPO_ROOT / "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct",
        source_audit=PHYSICAL_ROOT / "qwen30/QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json",
        root=PHYSICAL_ROOT / "qwen30",
        legacy_status=PHYSICAL_ROOT / "qwen30/QWEN30_REAL_CAMPAIGN_STATUS.json",
        architecture="Qwen3MoeForCausalLM",
        model_family="qwen3_moe",
        expert_count=128,
        top_k=8,
        query_heads=32,
        kv_heads=4,
        head_dim=128,
    ),
    "qwen80": ModelSpec(
        key="qwen80",
        model_id="Qwen3-Coder-Next-80B",
        repository="Qwen/Qwen3-Coder-Next",
        revision="a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
        source_dir=REPO_ROOT / "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next",
        source_audit=PHYSICAL_ROOT / "qwen80-acquisition/QWEN80_SOURCE_BODY_AUDIT_CANDIDATE.json",
        root=PHYSICAL_ROOT / "qwen80",
        legacy_status=PHYSICAL_ROOT / "qwen80/QWEN80_PHYSICAL_CAMPAIGN_STATUS.json",
        architecture="Qwen3NextForCausalLM",
        model_family="qwen3_next_hybrid",
        expert_count=512,
        top_k=10,
        query_heads=16,
        kv_heads=2,
        head_dim=256,
    ),
}


@dataclass(frozen=True)
class Target:
    name: str
    organ: str
    sensitive: bool
    notes: str


@dataclass(frozen=True)
class Proposal:
    sequence: int
    generation: int
    target: Target
    representation: str
    config: Mapping[str, Any]
    candidate_id: str
    schedule_version: str = LEGACY_REPRESENTATION_SCHEDULE_VERSION
    schedule_phase: str = "legacy_seven_family"
    schedule_boundary_sequence: int = 0
    schedule_start_generation: int = REPRESENTATION_EXPANSION_START_GENERATION


@dataclass
class CodecResult:
    payload: bytes
    reconstruction: np.ndarray
    metadata: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: bytes | bytearray | Mapping[str, Any] | Sequence[Any] | str) -> str:
    if isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raw = _canonical_bytes(value)
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _regular_file_identity(path: Path, *, label: str) -> dict[str, int]:
    """Return a cheap, no-hash identity only for a direct regular file.

    The complete-artifact compiler has already done the expensive full shard
    SHA-256 pass.  Component mutations reuse that sealed proof only while this
    identity is unchanged.  ``lstat`` is intentional: a symlink must never
    substitute for an audited model shard.
    """

    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise DualGravityError(f"cannot stat source {label}: {exc}") from exc
    if stat.S_ISLNK(observed.st_mode):
        raise DualGravityError(f"source {label} must be a regular file, not a symlink: {path}")
    if not stat.S_ISREG(observed.st_mode):
        raise DualGravityError(f"source {label} must be a regular file: {path}")
    return {
        "bytes": int(observed.st_size),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_jsonl_once(path: Path, row: Mapping[str, Any], *, record_id: str) -> None:
    """Append one immutable knowledge row while preventing duplicate retries."""

    lock_path = path.with_name(f".{path.name}.lock")
    with _locked(lock_path):
        existing_ids: set[str] = set()
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        prior = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(prior, Mapping) and isinstance(prior.get("record_id"), str):
                        existing_ids.add(str(prior["record_id"]))
        if record_id in existing_ids:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o640)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def _free_bytes(path: Path) -> int:
    return int(os.statvfs(path).f_bavail * os.statvfs(path).f_frsize)


def _command_snapshot(args: Sequence[str], *, timeout: float = 2.0) -> str | None:
    try:
        completed = subprocess.run(
            list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = completed.stdout.strip()
    return text[:2000] if text else None


def _system_resources() -> dict[str, Any]:
    try:
        physical_memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        physical_memory = None
    return {
        "load_average": [round(float(value), 3) for value in os.getloadavg()],
        "physical_memory_bytes": physical_memory,
        "swap": _command_snapshot(("/usr/sbin/sysctl", "-n", "vm.swapusage")),
        "memory_pressure": _command_snapshot(("/usr/bin/memory_pressure", "-Q")),
        "thermal": _command_snapshot(("/usr/bin/pmset", "-g", "therm")),
    }


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")[:120]


def _quality(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    source = np.ascontiguousarray(reference, dtype=np.float32).reshape(-1)
    restored = np.ascontiguousarray(reconstruction, dtype=np.float32).reshape(-1)
    if source.shape != restored.shape:
        raise DualGravityError("codec reconstruction shape does not match source region")
    delta = restored - source
    source_norm = max(float(np.linalg.norm(source)), 1e-12)
    restored_norm = max(float(np.linalg.norm(restored)), 1e-12)
    return {
        "relative_l2": float(np.linalg.norm(delta) / source_norm),
        "cosine": float(np.dot(source, restored) / (source_norm * restored_norm)),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size != flat.size:
        raise DualGravityError("source region contains non-finite values")
    return {
        "elements": int(finite.size),
        "mean": float(np.mean(finite, dtype=np.float64)),
        "std": float(np.std(finite, dtype=np.float64)),
        "absmax": float(np.max(np.abs(finite))) if finite.size else 0.0,
        "zero_fraction": float(np.count_nonzero(finite == 0.0) / max(finite.size, 1)),
    }


def _functional_probe(reference: np.ndarray, reconstruction: np.ndarray, *, seed: int) -> dict[str, Any]:
    """A deterministic tensor-teacher functional check, not model generation."""

    source = np.ascontiguousarray(reference, dtype=np.float32)
    restored = np.ascontiguousarray(reconstruction, dtype=np.float32)
    if source.ndim < 2:
        return {
            "status": "RAN_SCALAR_COMPONENT_CHECK",
            "relative_l2": _quality(source, restored)["relative_l2"],
            "cosine": _quality(source, restored)["cosine"],
            "claim_boundary": "tensor component only; no prompt, token loop, or model capability claim",
        }
    matrix = source.reshape(source.shape[0], -1)
    rebuilt = restored.reshape(restored.shape[0], -1)
    generator = np.random.default_rng(seed)
    activation = generator.standard_normal(matrix.shape[1], dtype=np.float32)
    expected = matrix @ activation
    observed = rebuilt @ activation
    result = _quality(expected, observed)
    return {
        "status": "RAN_TENSOR_TEACHER_MATVEC",
        "input_width": int(matrix.shape[1]),
        "output_width": int(matrix.shape[0]),
        "activation_statistics": _distribution(activation),
        **result,
        "claim_boundary": "teacher tensor matvec only; not full-model parity, mini-generation, HCLI, or capability",
    }


def _routing_probe(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    """Exact source-router control sample; it is not prompt-derived routing."""

    if reference.ndim != 2 or reference.shape[0] < top_k:
        return {"status": "NOT_APPLICABLE"}
    generator = np.random.default_rng(seed)
    activation = generator.standard_normal(reference.shape[1], dtype=np.float32)
    raw = np.ascontiguousarray(reference, dtype=np.float32) @ activation
    packed = np.ascontiguousarray(reconstruction, dtype=np.float32) @ activation
    raw_ids = np.argpartition(raw, -top_k)[-top_k:]
    packed_ids = np.argpartition(packed, -top_k)[-top_k:]
    raw_ids = raw_ids[np.argsort(raw[raw_ids])[::-1]]
    packed_ids = packed_ids[np.argsort(packed[packed_ids])[::-1]]
    raw_margin = float(raw[raw_ids[0]] - raw[raw_ids[1]]) if top_k > 1 else None
    return {
        "status": "RAN_SOURCE_ROUTER_CONTROL_SAMPLE",
        "top_k": top_k,
        "raw_route_ids": [int(item) for item in raw_ids],
        "packed_route_ids": [int(item) for item in packed_ids],
        "route_set_overlap": len(set(raw_ids.tolist()) & set(packed_ids.tolist())) / top_k,
        "raw_top1_margin": raw_margin,
        "activation_statistics": _distribution(activation),
        "claim_boundary": "deterministic source-router component control only; not prompt-dependent routing or capability",
    }


def _mps_component_probe(
    values: np.ndarray,
    *,
    seed: int,
    gpu_lease: Callable[[], contextlib.AbstractContextManager[Any]] | None = None,
) -> dict[str, Any]:
    """Measure a library MPS component smoke without calling it a custom kernel."""

    if values.ndim != 2 or values.size > MAX_REGION_ELEMENTS:
        return {"status": "NOT_SCHEDULED", "reason": "geometry_not_bounded_2d_component"}
    try:
        import torch
    except ImportError:
        return {"status": "UNAVAILABLE", "reason": "torch_not_installed"}
    if not torch.backends.mps.is_available():
        return {"status": "UNAVAILABLE", "reason": "mps_not_available"}
    try:
        lease = gpu_lease() if gpu_lease else contextlib.nullcontext()
        with lease:
            torch.manual_seed(seed)
            matrix = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float16)).to("mps")
            vector = torch.randn((1, matrix.shape[1]), dtype=torch.float16, device="mps")
            for _ in range(3):
                _ = vector @ matrix.T
            torch.mps.synchronize()
            started = time.perf_counter()
            iterations = 16
            for _ in range(iterations):
                _ = vector @ matrix.T
            torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            return {
                "status": "RAN_MPS_LIBRARY_COMPONENT_SMOKE",
                "device": "mps",
                "iterations": iterations,
                "elapsed_seconds": elapsed,
                "matvecs_per_second": iterations / max(elapsed, 1e-12),
                "shape": [int(item) for item in values.shape],
                "claim_boundary": "library matvec component only; not custom packed decode and not model tokens-per-second",
            }
    except Exception as exc:  # Environmental runtime failures are candidate evidence, not fatal corruption.
        return {"status": "FAILED", "reason": type(exc).__name__}


def _pack_unsigned(codes: np.ndarray, bits: int) -> bytes:
    if bits < 1 or bits > 8:
        raise DualGravityError("bit packing requires 1..8 bits")
    flat = np.ascontiguousarray(codes, dtype=np.uint8).reshape(-1)
    bit_matrix = ((flat[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little").tobytes()


def _unpack_unsigned(payload: bytes, count: int, bits: int) -> np.ndarray:
    bit_count = count * bits
    raw = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:bit_count]
    weights = (1 << np.arange(bits, dtype=np.uint8)).astype(np.uint16)
    return (raw.reshape(count, bits).astype(np.uint16) * weights).sum(axis=1).astype(np.uint8)


def _container(magic: bytes, header: Mapping[str, Any], body: bytes) -> bytes:
    encoded = _canonical_bytes(dict(header))
    return magic + struct.pack("<I", len(encoded)) + encoded + body


def _parse_container(payload: bytes, *, expected_magic: bytes) -> tuple[dict[str, Any], bytes]:
    """Decode the durable artifact envelope before a codec-specific decode.

    Every new representation uses this parser in its reconstruction path.  It
    keeps a candidate from accidentally reporting a reconstruction calculated
    from encoder-local arrays rather than from the physical bytes it wrote.
    """

    if len(expected_magic) != 8:
        raise DualGravityError("artifact magic must be exactly eight bytes")
    if len(payload) < 12 or payload[:8] != expected_magic:
        raise DualGravityError("artifact magic does not match codec")
    header_size = struct.unpack("<I", payload[8:12])[0]
    body_offset = 12 + header_size
    if body_offset > len(payload):
        raise DualGravityError("artifact header length exceeds physical bytes")
    try:
        header = json.loads(payload[12:body_offset].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DualGravityError("artifact header is not valid canonical JSON") from exc
    if not isinstance(header, Mapping):
        raise DualGravityError("artifact header is not an object")
    return dict(header), payload[body_offset:]


def _packed_byte_count(*, count: int, bits: int) -> int:
    if count < 0 or bits < 1 or bits > 8:
        raise DualGravityError("invalid packed-code geometry")
    return math.ceil(count * bits / 8)


def _decode_uniform_body(header: Mapping[str, Any], body: bytes) -> np.ndarray:
    """Decode the uniform body used directly and as a base layer elsewhere."""

    try:
        shape = tuple(int(item) for item in header["shape"])
        elements = int(header["elements"])
        bits = int(header["bits"])
        group_size = int(header["group_size"])
        groups = int(header["groups"])
        scale_bytes = int(header["scale_bytes"])
        code_bytes = int(header["code_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DualGravityError("uniform artifact header lacks required geometry") from exc
    if not shape or any(item <= 0 for item in shape) or math.prod(shape) != elements:
        raise DualGravityError("uniform artifact shape does not match element count")
    if bits < 2 or bits > 8 or group_size <= 0 or groups != math.ceil(elements / group_size):
        raise DualGravityError("uniform artifact group geometry is invalid")
    if scale_bytes != groups * np.dtype("<f2").itemsize or code_bytes != _packed_byte_count(count=groups * group_size, bits=bits):
        raise DualGravityError("uniform artifact byte ledger is invalid")
    if len(body) != scale_bytes + code_bytes:
        raise DualGravityError("uniform artifact physical body bytes do not match its ledger")
    scales = np.frombuffer(body[:scale_bytes], dtype="<f2", count=groups).astype(np.float32)
    unsigned = _unpack_unsigned(body[scale_bytes:], groups * group_size, bits)
    bound = (1 << (bits - 1)) - 1
    signed = unsigned.astype(np.int16) - bound
    rebuilt = signed.reshape(groups, group_size).astype(np.float32) * scales[:, None]
    return np.ascontiguousarray(rebuilt.reshape(-1)[:elements].reshape(shape), dtype=np.float32)


def _decode_uniform_codec(payload: bytes) -> np.ndarray:
    header, body = _parse_container(payload, expected_magic=MAGIC_UNIFORM)
    return _decode_uniform_body(header, body)


def _binary_parts(values: np.ndarray, *, group_size: int) -> tuple[np.ndarray, np.ndarray, bytes, np.ndarray]:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / group_size)
    padded = np.pad(flat, (0, groups * group_size - flat.size), constant_values=0.0).reshape(groups, group_size)
    if not np.isfinite(padded).all():
        raise DualGravityError("non-finite source values are not a valid lossy candidate input")
    scales = np.mean(np.abs(padded), axis=1, dtype=np.float64).astype("<f2")
    signs = np.packbits((padded >= 0.0).reshape(-1).astype(np.uint8), bitorder="little").tobytes()
    rebuilt = (np.where(padded >= 0.0, 1.0, -1.0) * scales.astype(np.float32)[:, None]).reshape(-1)[: flat.size]
    return padded, scales, signs, rebuilt.reshape(values.shape)


def _binary_codec(values: np.ndarray, *, group_size: int = GROUP_BINARY) -> CodecResult:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    _, scales, signs, reconstruction = _binary_parts(values, group_size=group_size)
    header = {
        "schema": "hawking.gravity.binary_sign_scale.v1",
        "representation": "binary_sign_scale",
        "shape": [int(item) for item in values.shape],
        "elements": int(flat.size),
        "group_size": group_size,
        "groups": int(scales.size),
        "scale_dtype": "float16",
        "bit_order": "little",
        "scale_bytes": int(scales.nbytes),
        "sign_bytes": len(signs),
    }
    payload = _container(MAGIC_BINARY, header, scales.tobytes() + signs)
    return CodecResult(payload=payload, reconstruction=reconstruction, metadata=header)


def _uniform_codec(values: np.ndarray, *, bits: int, group_size: int = GROUP_UNIFORM) -> CodecResult:
    if bits < 2 or bits > 8:
        raise DualGravityError("uniform codec supports 2..8 bits")
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / group_size)
    padded = np.pad(flat, (0, groups * group_size - flat.size), constant_values=0.0).reshape(groups, group_size)
    if not np.isfinite(padded).all():
        raise DualGravityError("non-finite source values are not a valid lossy candidate input")
    bound = (1 << (bits - 1)) - 1
    scales = (np.max(np.abs(padded), axis=1) / max(bound, 1)).astype("<f2")
    denominator = np.where(scales.astype(np.float32) > 0.0, scales.astype(np.float32), 1.0)
    signed = np.rint(padded / denominator[:, None]).clip(-bound, bound).astype(np.int16)
    unsigned = (signed.reshape(-1) + bound).astype(np.uint8)
    code_bytes = _pack_unsigned(unsigned, bits)
    header = {
        "schema": "hawking.gravity.uniform_group.v1",
        "representation": f"uniform_q{bits}_group_scale",
        "shape": [int(item) for item in values.shape],
        "elements": int(flat.size),
        "bits": bits,
        "group_size": group_size,
        "groups": groups,
        "scale_dtype": "float16",
        "code_bytes": len(code_bytes),
        "scale_bytes": int(scales.nbytes),
        "retained_padding_elements": int(groups * group_size - flat.size),
    }
    payload = _container(MAGIC_UNIFORM, header, scales.tobytes() + code_bytes)
    return CodecResult(payload=payload, reconstruction=_decode_uniform_codec(payload), metadata=header)


def _ternary_codec(values: np.ndarray, *, threshold_multiplier: float, group_size: int = GROUP_BINARY) -> CodecResult:
    if threshold_multiplier <= 0:
        raise DualGravityError("ternary threshold multiplier must be positive")
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / group_size)
    padded = np.pad(flat, (0, groups * group_size - flat.size), constant_values=0.0).reshape(groups, group_size)
    base = np.mean(np.abs(padded), axis=1, dtype=np.float64).astype(np.float32)
    thresholds = (base * threshold_multiplier).astype("<f2")
    active = np.abs(padded) >= thresholds.astype(np.float32)[:, None]
    selected_abs = np.where(active, np.abs(padded), 0.0)
    selected_count = np.maximum(active.sum(axis=1), 1)
    scales = (selected_abs.sum(axis=1) / selected_count).astype("<f2")
    codes = np.where(~active, 0, np.where(padded >= 0.0, 1, 2)).astype(np.uint8).reshape(-1)
    code_bytes = _pack_unsigned(codes, 2)
    decoded = _unpack_unsigned(code_bytes, codes.size, 2).reshape(groups, group_size)
    rebuilt = np.where(decoded == 1, 1.0, np.where(decoded == 2, -1.0, 0.0))
    rebuilt = (rebuilt * scales.astype(np.float32)[:, None]).reshape(-1)[: flat.size]
    header = {
        "schema": "hawking.gravity.ternary_threshold.v1",
        "representation": "ternary_threshold_group_scale",
        "shape": [int(item) for item in values.shape],
        "elements": int(flat.size),
        "group_size": group_size,
        "groups": groups,
        "threshold_multiplier": threshold_multiplier,
        "scale_dtype": "float16",
        "threshold_dtype": "float16",
        "code_bytes": len(code_bytes),
        "scale_bytes": int(scales.nbytes),
        "threshold_bytes": int(thresholds.nbytes),
    }
    payload = _container(MAGIC_TERNARY, header, thresholds.tobytes() + scales.tobytes() + code_bytes)
    return CodecResult(payload=payload, reconstruction=rebuilt.reshape(values.shape), metadata=header)


def _residual_codec(values: np.ndarray, *, outlier_ratio: float, group_size: int = GROUP_BINARY) -> CodecResult:
    if not 0.0 < outlier_ratio <= 0.1:
        raise DualGravityError("outlier residual ratio must be in (0, 0.1]")
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    _, scales, signs, base = _binary_parts(values, group_size=group_size)
    reconstructed = np.ascontiguousarray(base, dtype=np.float32).reshape(-1)
    residual = flat - reconstructed
    count = max(1, int(math.ceil(flat.size * outlier_ratio)))
    indices = np.argpartition(np.abs(residual), -count)[-count:].astype("<u4")
    indices.sort()
    values16 = residual[indices].astype("<f2")
    reconstructed[indices] += values16.astype(np.float32)
    header = {
        "schema": "hawking.gravity.binary_outlier_residual.v1",
        "representation": "binary_sign_scale_plus_sparse_fp16_residual",
        "shape": [int(item) for item in values.shape],
        "elements": int(flat.size),
        "group_size": group_size,
        "groups": int(scales.size),
        "outlier_ratio_requested": outlier_ratio,
        "outlier_count": int(count),
        "scale_dtype": "float16",
        "residual_dtype": "float16",
        "index_dtype": "uint32",
        "scale_bytes": int(scales.nbytes),
        "sign_bytes": len(signs),
        "index_bytes": int(indices.nbytes),
        "residual_bytes": int(values16.nbytes),
    }
    payload = _container(MAGIC_RESIDUAL, header, scales.tobytes() + signs + indices.tobytes() + values16.tobytes())
    return CodecResult(payload=payload, reconstruction=reconstructed.reshape(values.shape), metadata=header)


def _orthonormal_hadamard(groups: np.ndarray) -> np.ndarray:
    """Apply a deterministic normalized Walsh-Hadamard transform per group."""

    work = np.array(groups, dtype=np.float32, copy=True, order="C")
    if work.ndim != 2 or work.shape[1] <= 0 or work.shape[1] & (work.shape[1] - 1):
        raise DualGravityError("Hadamard lattice codec requires power-of-two 2D groups")
    width = int(work.shape[1])
    stride = 1
    while stride < width:
        view = work.reshape(work.shape[0], width // (2 * stride), 2, stride)
        left = view[:, :, 0, :].copy()
        right = view[:, :, 1, :].copy()
        view[:, :, 0, :] = left + right
        view[:, :, 1, :] = left - right
        stride *= 2
    return work / math.sqrt(width)


def _decode_hadamard_lattice_codec(payload: bytes) -> np.ndarray:
    header, body = _parse_container(payload, expected_magic=MAGIC_HADAMARD)
    try:
        shape = tuple(int(item) for item in header["shape"])
        elements = int(header["elements"])
        bits = int(header["bits"])
        group_size = int(header["group_size"])
        groups = int(header["groups"])
        scale_bytes = int(header["scale_bytes"])
        code_bytes = int(header["code_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DualGravityError("Hadamard lattice header lacks required geometry") from exc
    if not shape or any(item <= 0 for item in shape) or math.prod(shape) != elements:
        raise DualGravityError("Hadamard lattice shape does not match element count")
    if bits < 2 or bits > 8 or group_size <= 0 or group_size & (group_size - 1):
        raise DualGravityError("Hadamard lattice quantization geometry is invalid")
    if groups != math.ceil(elements / group_size):
        raise DualGravityError("Hadamard lattice group count is invalid")
    if scale_bytes != groups * np.dtype("<f2").itemsize or code_bytes != _packed_byte_count(count=groups * group_size, bits=bits):
        raise DualGravityError("Hadamard lattice byte ledger is invalid")
    if len(body) != scale_bytes + code_bytes:
        raise DualGravityError("Hadamard lattice physical body bytes do not match its ledger")
    scales = np.frombuffer(body[:scale_bytes], dtype="<f2", count=groups).astype(np.float32)
    codes = _unpack_unsigned(body[scale_bytes:], groups * group_size, bits)
    bound = (1 << (bits - 1)) - 1
    coefficients = (codes.astype(np.int16) - bound).reshape(groups, group_size).astype(np.float32) * scales[:, None]
    restored = _orthonormal_hadamard(coefficients).reshape(-1)[:elements]
    return np.ascontiguousarray(restored.reshape(shape), dtype=np.float32)


def _hadamard_lattice_codec(values: np.ndarray, *, bits: int, group_size: int = GROUP_BINARY) -> CodecResult:
    """Pack an orthonormal transform-domain integer lattice, then decode it."""

    if bits < 2 or bits > 8:
        raise DualGravityError("Hadamard lattice codec supports 2..8 bits")
    if group_size <= 0 or group_size & (group_size - 1):
        raise DualGravityError("Hadamard lattice group size must be a power of two")
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / group_size)
    padded = np.pad(flat, (0, groups * group_size - flat.size), constant_values=0.0).reshape(groups, group_size)
    if not np.isfinite(padded).all():
        raise DualGravityError("non-finite source values are not valid Hadamard lattice input")
    transformed = _orthonormal_hadamard(padded)
    bound = (1 << (bits - 1)) - 1
    scales = (np.max(np.abs(transformed), axis=1) / max(bound, 1)).astype("<f2")
    denominator = np.where(scales.astype(np.float32) > 0.0, scales.astype(np.float32), 1.0)
    signed = np.rint(transformed / denominator[:, None]).clip(-bound, bound).astype(np.int16)
    code_bytes = _pack_unsigned((signed.reshape(-1) + bound).astype(np.uint8), bits)
    header = {
        "schema": "hawking.gravity.hadamard_lattice_group.v1",
        "representation": "orthonormal_hadamard_integer_lattice_group_scale",
        "shape": [int(item) for item in values.shape],
        "elements": int(flat.size),
        "bits": bits,
        "group_size": group_size,
        "groups": groups,
        "transform": "normalized_walsh_hadamard_self_inverse",
        "lattice": "symmetric_integer_uniform",
        "scale_dtype": "float16",
        "code_bytes": len(code_bytes),
        "scale_bytes": int(scales.nbytes),
        "retained_padding_elements": int(groups * group_size - flat.size),
    }
    payload = _container(MAGIC_HADAMARD, header, scales.tobytes() + code_bytes)
    return CodecResult(payload=payload, reconstruction=_decode_hadamard_lattice_codec(payload), metadata=header)


_ADDITIVE_Q2_LEVELS = np.asarray((-1.5, -0.5, 0.5, 1.5), dtype=np.float32)


def _additive_q2_codes(normalized: np.ndarray) -> np.ndarray:
    """Nearest deterministic codebook index for the fixed four-level lattice."""

    return np.rint(np.ascontiguousarray(normalized, dtype=np.float32) + 1.5).clip(0, 3).astype(np.uint8)


def _decode_additive_residual_codec(payload: bytes) -> np.ndarray:
    header, body = _parse_container(payload, expected_magic=MAGIC_ADDITIVE)
    try:
        shape = tuple(int(item) for item in header["shape"])
        elements = int(header["elements"])
        group_size = int(header["group_size"])
        groups = int(header["groups"])
        base_scale_bytes = int(header["base_scale_bytes"])
        residual_scale_bytes = int(header["residual_scale_bytes"])
        base_code_bytes = int(header["base_code_bytes"])
        residual_code_bytes = int(header["residual_code_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DualGravityError("additive residual header lacks required geometry") from exc
    if not shape or any(item <= 0 for item in shape) or math.prod(shape) != elements:
        raise DualGravityError("additive residual shape does not match element count")
    if group_size <= 0 or groups != math.ceil(elements / group_size):
        raise DualGravityError("additive residual group geometry is invalid")
    expected_scale_bytes = groups * np.dtype("<f2").itemsize
    expected_code_bytes = _packed_byte_count(count=groups * group_size, bits=2)
    if (base_scale_bytes, residual_scale_bytes) != (expected_scale_bytes, expected_scale_bytes):
        raise DualGravityError("additive residual scale-byte ledger is invalid")
    if (base_code_bytes, residual_code_bytes) != (expected_code_bytes, expected_code_bytes):
        raise DualGravityError("additive residual code-byte ledger is invalid")
    expected_body = base_scale_bytes + residual_scale_bytes + base_code_bytes + residual_code_bytes
    if len(body) != expected_body:
        raise DualGravityError("additive residual physical body bytes do not match its ledger")
    cursor = 0
    base_scales = np.frombuffer(body[cursor : cursor + base_scale_bytes], dtype="<f2", count=groups).astype(np.float32)
    cursor += base_scale_bytes
    residual_scales = np.frombuffer(body[cursor : cursor + residual_scale_bytes], dtype="<f2", count=groups).astype(np.float32)
    cursor += residual_scale_bytes
    base_codes = _unpack_unsigned(body[cursor : cursor + base_code_bytes], groups * group_size, 2).reshape(groups, group_size)
    cursor += base_code_bytes
    residual_codes = _unpack_unsigned(body[cursor:], groups * group_size, 2).reshape(groups, group_size)
    base = _ADDITIVE_Q2_LEVELS[base_codes] * base_scales[:, None]
    correction = _ADDITIVE_Q2_LEVELS[residual_codes] * residual_scales[:, None]
    restored = (base + correction).reshape(-1)[:elements]
    return np.ascontiguousarray(restored.reshape(shape), dtype=np.float32)


def _additive_residual_codec(values: np.ndarray, *, group_size: int = GROUP_UNIFORM) -> CodecResult:
    """Encode two real additive q2 codebooks, not a label over uniform q4."""

    if group_size <= 0:
        raise DualGravityError("additive residual group size must be positive")
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / group_size)
    padded = np.pad(flat, (0, groups * group_size - flat.size), constant_values=0.0).reshape(groups, group_size)
    if not np.isfinite(padded).all():
        raise DualGravityError("non-finite source values are not valid additive residual input")
    base_scales = (np.max(np.abs(padded), axis=1) / 1.5).astype("<f2")
    base_denominator = np.where(base_scales.astype(np.float32) > 0.0, base_scales.astype(np.float32), 1.0)
    base_codes = _additive_q2_codes(padded / base_denominator[:, None])
    base = _ADDITIVE_Q2_LEVELS[base_codes] * base_scales.astype(np.float32)[:, None]
    residual = padded - base
    residual_scales = (np.max(np.abs(residual), axis=1) / 1.5).astype("<f2")
    residual_denominator = np.where(residual_scales.astype(np.float32) > 0.0, residual_scales.astype(np.float32), 1.0)
    residual_codes = _additive_q2_codes(residual / residual_denominator[:, None])
    base_code_bytes = _pack_unsigned(base_codes.reshape(-1), 2)
    residual_code_bytes = _pack_unsigned(residual_codes.reshape(-1), 2)
    header = {
        "schema": "hawking.gravity.additive_residual_codebook.v1",
        "representation": "two_stage_additive_q2_residual_codebooks_group_scale",
        "shape": [int(item) for item in values.shape],
        "elements": int(flat.size),
        "group_size": group_size,
        "groups": groups,
        "stage_count": 2,
        "codebook": [-1.5, -0.5, 0.5, 1.5],
        "codebook_bits_each": 2,
        "scale_dtype": "float16",
        "base_scale_bytes": int(base_scales.nbytes),
        "residual_scale_bytes": int(residual_scales.nbytes),
        "base_code_bytes": len(base_code_bytes),
        "residual_code_bytes": len(residual_code_bytes),
        "retained_padding_elements": int(groups * group_size - flat.size),
    }
    payload = _container(MAGIC_ADDITIVE, header, base_scales.tobytes() + residual_scales.tobytes() + base_code_bytes + residual_code_bytes)
    return CodecResult(payload=payload, reconstruction=_decode_additive_residual_codec(payload), metadata=header)


def _deterministic_calibration_direction(width: int, seed: int) -> np.ndarray:
    """A reproducible unit direction stored by rule rather than raw samples."""

    if width <= 0:
        raise DualGravityError("activation correction needs a positive input width")
    index = np.arange(1, width + 1, dtype=np.float64)
    first_frequency = 0.001953125 * (1 + seed % 251)
    second_frequency = 0.001220703125 * (1 + (seed // 251) % 509)
    direction = np.sin(index * first_frequency) + 0.5 * np.cos(index * second_frequency)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise DualGravityError("deterministic activation direction is degenerate")
    return np.ascontiguousarray(direction / norm, dtype=np.float32)


def _decode_activation_corrected_codec(payload: bytes) -> np.ndarray:
    header, body = _parse_container(payload, expected_magic=MAGIC_ACTIVATION)
    try:
        shape = tuple(int(item) for item in header["shape"])
        matrix_shape = tuple(int(item) for item in header["matrix_shape"])
        elements = int(header["elements"])
        base_header = header["base_uniform_header"]
        calibration = header["activation_correction"]
        correction_bytes = int(calibration["correction_bytes"])
        seed = int(calibration["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DualGravityError("activation-corrected header lacks required geometry") from exc
    if not isinstance(base_header, Mapping) or not isinstance(calibration, Mapping):
        raise DualGravityError("activation-corrected nested metadata is invalid")
    if len(shape) < 2 or any(item <= 0 for item in shape) or math.prod(shape) != elements:
        raise DualGravityError("activation-corrected shape does not match element count")
    if len(matrix_shape) != 2 or matrix_shape[0] <= 0 or matrix_shape[1] <= 0 or math.prod(matrix_shape) != elements:
        raise DualGravityError("activation-corrected matrix geometry is invalid")
    base_scale_bytes = int(base_header.get("scale_bytes", -1))
    base_code_bytes = int(base_header.get("code_bytes", -1))
    base_body_bytes = base_scale_bytes + base_code_bytes
    if correction_bytes != matrix_shape[0] * np.dtype("<f2").itemsize:
        raise DualGravityError("activation-corrected row correction ledger is invalid")
    if len(body) != base_body_bytes + correction_bytes:
        raise DualGravityError("activation-corrected physical body bytes do not match its ledger")
    base = _decode_uniform_body(base_header, body[:base_body_bytes]).reshape(matrix_shape)
    corrections = np.frombuffer(body[base_body_bytes:], dtype="<f2", count=matrix_shape[0]).astype(np.float32)
    direction = _deterministic_calibration_direction(matrix_shape[1], seed)
    restored = base + corrections[:, None] * direction[None, :]
    return np.ascontiguousarray(restored.reshape(shape), dtype=np.float32)


def _activation_corrected_codec(
    values: np.ndarray,
    *,
    bits: int,
    group_size: int = GROUP_UNIFORM,
    calibration_seed: int,
) -> CodecResult:
    """Store a q3 base plus a real source-output correction per output row.

    The correction is fitted only to a deterministic algebraic component
    direction. It is explicitly not prompt activation calibration, generation,
    or a capability measurement.
    """

    if values.ndim < 2:
        raise DualGravityError("activation-corrected candidate requires a matrix-like tensor")
    original_shape = tuple(int(item) for item in values.shape)
    matrix = np.ascontiguousarray(values, dtype=np.float32).reshape(values.shape[0], -1)
    base_result = _uniform_codec(matrix, bits=bits, group_size=group_size)
    base_header, base_body = _parse_container(base_result.payload, expected_magic=MAGIC_UNIFORM)
    base = _decode_uniform_body(base_header, base_body).reshape(matrix.shape)
    direction = _deterministic_calibration_direction(matrix.shape[1], calibration_seed)
    reference_output = matrix @ direction
    base_output = base @ direction
    corrections = (reference_output - base_output).astype("<f2")
    corrected_output = base_output + corrections.astype(np.float32)
    baseline_output = _quality(reference_output, base_output)
    corrected_output_quality = _quality(reference_output, corrected_output)
    header = {
        "schema": "hawking.gravity.activation_corrected_rowwise.v1",
        "representation": "uniform_group_base_plus_deterministic_rowwise_output_correction",
        "shape": list(original_shape),
        "matrix_shape": [int(item) for item in matrix.shape],
        "elements": int(values.size),
        "base_uniform_header": base_header,
        "activation_correction": {
            "kind": "deterministic_unit_direction_row_output_correction",
            "generator": "sin_cos_v1",
            "seed": int(calibration_seed),
            "input_width": int(matrix.shape[1]),
            "output_rows": int(matrix.shape[0]),
            "correction_dtype": "float16",
            "correction_bytes": int(corrections.nbytes),
            "baseline_direction_output": baseline_output,
            "corrected_direction_output": corrected_output_quality,
            "claim_boundary": "deterministic tensor component direction only; not prompt activation calibration or model routing",
        },
    }
    payload = _container(MAGIC_ACTIVATION, header, base_body + corrections.tobytes())
    return CodecResult(payload=payload, reconstruction=_decode_activation_corrected_codec(payload), metadata=header)


def _factor_codec(values: np.ndarray, *, bits: int, group_size: int = GROUP_UNIFORM) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    """Body-only quantization used for both low-rank factors."""

    result = _uniform_codec(values, bits=bits, group_size=group_size)
    # The enclosing low-rank payload has its own header, so retain only the
    # uniform codec body after its magic/header prefix.  Recreate it from known
    # metadata to avoid storing two redundant JSON headers.
    header_len = struct.unpack("<I", result.payload[8:12])[0]
    body = result.payload[12 + header_len :]
    return body, result.reconstruction, dict(result.metadata)


def _low_rank_factors(values: np.ndarray, *, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.ascontiguousarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    rows, columns = matrix.shape
    allowed = min(rows, columns)
    actual_rank = min(max(1, rank), allowed)
    oversample = min(12, max(0, allowed - actual_rank))
    generator = np.random.default_rng(seed)
    probe = generator.standard_normal((columns, actual_rank + oversample), dtype=np.float32)
    basis, _ = np.linalg.qr(matrix @ probe, mode="reduced")
    small = basis.T @ matrix
    left, singular, right = np.linalg.svd(small, full_matrices=False)
    return (
        np.ascontiguousarray(basis @ (left[:, :actual_rank] * singular[:actual_rank]), dtype=np.float32),
        np.ascontiguousarray(right[:actual_rank, :], dtype=np.float32),
    )


def _teacher_refine(
    teacher: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    steps: int,
    seed: int,
    gpu_lease: Callable[[], contextlib.AbstractContextManager[Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run bounded component-level teacher distillation on the true BF16 region."""

    try:
        import torch
    except ImportError:
        return left, right, {"status": "UNAVAILABLE", "reason": "torch_not_installed"}
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    try:
        lease = gpu_lease() if device.type == "mps" and gpu_lease else contextlib.nullcontext()
        with lease:
            torch.manual_seed(seed)
            target = torch.from_numpy(np.ascontiguousarray(teacher, dtype=np.float32)).to(device)
            first = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(left)).to(device))
            second = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(right)).to(device))
            optimizer = torch.optim.AdamW((first, second), lr=6e-4, weight_decay=1e-6)
            with torch.no_grad():
                initial = float(torch.mean(torch.square(first @ second - target)).item())
            losses: list[float] = []
            started = time.perf_counter()
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean(torch.square(first @ second - target))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))
            if device.type == "mps":
                torch.mps.synchronize()
            return (
                first.detach().to("cpu").float().numpy(),
                second.detach().to("cpu").float().numpy(),
                {
                    "status": "RAN_COMPONENT_TEACHER_DISTILLATION",
                    "device": device.type,
                    "steps": steps,
                    "initial_mse": initial,
                    "final_mse": losses[-1] if losses else initial,
                    "elapsed_seconds": time.perf_counter() - started,
                    "claim_boundary": "component teacher training only; not full-model QAT, capability, or manager training",
                },
            )
    except Exception as exc:
        return left, right, {"status": "FAILED", "device": device.type, "reason": type(exc).__name__}


def _low_rank_codec(
    values: np.ndarray,
    *,
    rank: int,
    bits: int,
    seed: int,
    train_steps: int,
    gpu_lease: Callable[[], contextlib.AbstractContextManager[Any]] | None = None,
) -> tuple[CodecResult, dict[str, Any]]:
    if values.ndim < 2:
        raise DualGravityError("low-rank candidate requires a matrix-like tensor")
    original_shape = tuple(int(item) for item in values.shape)
    matrix = np.ascontiguousarray(values, dtype=np.float32).reshape(values.shape[0], -1)
    left, right = _low_rank_factors(matrix, rank=rank, seed=seed)
    training: dict[str, Any] = {"status": "NOT_REQUESTED"}
    if train_steps > 0:
        left, right, training = _teacher_refine(
            matrix,
            left,
            right,
            steps=train_steps,
            seed=seed,
            gpu_lease=gpu_lease,
        )
    left_body, left_rebuilt, left_meta = _factor_codec(left, bits=bits)
    right_body, right_rebuilt, right_meta = _factor_codec(right, bits=bits)
    reconstruction = (left_rebuilt @ right_rebuilt).reshape(original_shape)
    header = {
        "schema": "hawking.gravity.low_rank_quantized_factors.v1",
        "representation": "teacher_refined_low_rank_plus_uniform_q_factors" if train_steps else "randomized_low_rank_plus_uniform_q_factors",
        "shape": list(original_shape),
        "matrix_shape": [int(item) for item in matrix.shape],
        "elements": int(values.size),
        "rank": int(left.shape[1]),
        "factor_bits": bits,
        "factor_group_size": GROUP_UNIFORM,
        "left": left_meta,
        "right": right_meta,
        "left_body_bytes": len(left_body),
        "right_body_bytes": len(right_body),
    }
    payload = _container(MAGIC_LOWRANK, header, left_body + right_body)
    return CodecResult(payload=payload, reconstruction=reconstruction, metadata=header), training


def _representation_config(representation: str, generation: int) -> dict[str, Any]:
    phase = generation % 3
    if representation == "binary_sign_scale128":
        return {"group_size": (128, 64, 256)[phase]}
    if representation in {"uniform_q2_group64", "uniform_q3_group64", "uniform_q4_group64"}:
        return {"bits": int(representation[9]), "group_size": (64, 128, 32)[phase]}
    if representation == "ternary_threshold_group128":
        return {"threshold_multiplier": (0.55, 0.8, 1.05)[phase], "group_size": (128, 64, 256)[phase]}
    if representation == "binary_outlier_residual":
        return {"outlier_ratio": (0.0025, 0.005, 0.01)[phase], "group_size": (128, 64, 256)[phase]}
    if representation == "teacher_low_rank_q3":
        return {"rank": (24, 32, 48)[phase], "bits": 3, "train_steps": (4, 6, 8)[phase]}
    if representation == "hadamard_lattice_q3_group128":
        return {"bits": 3, "group_size": (128, 64, 256)[phase], "transform": "normalized_walsh_hadamard_self_inverse"}
    if representation == "additive_residual_codebook_q2x2":
        return {"group_size": (64, 128, 32)[phase], "stages": 2, "bits_per_stage": 2}
    if representation == "activation_corrected_rowwise_q3":
        return {
            "bits": 3,
            "group_size": (64, 128, 32)[phase],
            "calibration_seed": (101, 211, 307)[phase],
            "correction": "deterministic_unit_direction_row_output_correction",
        }
    raise DualGravityError(f"unknown representation {representation}")


def _encode(
    values: np.ndarray,
    proposal: Proposal,
    *,
    gpu_lease: Callable[[], contextlib.AbstractContextManager[Any]] | None = None,
) -> tuple[CodecResult, dict[str, Any]]:
    config = dict(proposal.config)
    if proposal.representation == "binary_sign_scale128":
        return _binary_codec(values, group_size=int(config["group_size"])), {"status": "NOT_APPLICABLE"}
    if proposal.representation.startswith("uniform_q"):
        return _uniform_codec(values, bits=int(config["bits"]), group_size=int(config["group_size"])), {"status": "NOT_APPLICABLE"}
    if proposal.representation == "ternary_threshold_group128":
        return _ternary_codec(values, threshold_multiplier=float(config["threshold_multiplier"]), group_size=int(config["group_size"])), {"status": "NOT_APPLICABLE"}
    if proposal.representation == "binary_outlier_residual":
        return _residual_codec(values, outlier_ratio=float(config["outlier_ratio"]), group_size=int(config["group_size"])), {"status": "NOT_APPLICABLE"}
    if proposal.representation == "teacher_low_rank_q3":
        return _low_rank_codec(
            values,
            rank=int(config["rank"]),
            bits=int(config["bits"]),
            seed=int(_sha256(proposal.candidate_id)[:16], 16),
            train_steps=int(config["train_steps"]),
            gpu_lease=gpu_lease,
        )
    if proposal.representation == "hadamard_lattice_q3_group128":
        return _hadamard_lattice_codec(values, bits=int(config["bits"]), group_size=int(config["group_size"])), {"status": "NOT_APPLICABLE"}
    if proposal.representation == "additive_residual_codebook_q2x2":
        return _additive_residual_codec(values, group_size=int(config["group_size"])), {"status": "NOT_APPLICABLE"}
    if proposal.representation == "activation_corrected_rowwise_q3":
        return _activation_corrected_codec(
            values,
            bits=int(config["bits"]),
            group_size=int(config["group_size"]),
            calibration_seed=int(config["calibration_seed"]),
        ), {"status": "NOT_APPLICABLE"}
    raise DualGravityError(f"unsupported representation {proposal.representation}")


class DualGravityWorker:
    """One deterministic, restart-safe worker for one Qwen family."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.evolution_root = spec.root / "evolution"
        self.identity_dir = self.evolution_root / "source-identities"
        self.identity_path = self.evolution_root / "SOURCE_CONTENT_IDENTITY.json"
        self.state_path = self.evolution_root / "WORKER_STATE.json"
        self.candidate_dir = self.evolution_root / "candidates"
        self.artifact_dir = self.evolution_root / "artifacts"
        self.failure_dir = self.evolution_root / "failures"
        self.index_path = self.evolution_root / "CANDIDATE_INDEX.json"
        self.frontier_path = self.evolution_root / "PARETO_FRONTIER.json"
        self.champions_path = self.evolution_root / "CHAMPIONS.json"
        self.status_path = self.evolution_root / f"{spec.key.upper()}_DUAL_GRAVITY_STATUS.json"
        self.local_negative_path = self.evolution_root / "NEGATIVE_SCIENCE.jsonl"
        self.shared_mechanisms_path = FAMILY_ROOT / "ASCENSION_REPRESENTATION_GENOME.jsonl"
        self.shared_kernel_path = FAMILY_ROOT / "ASCENSION_KERNEL_GENOME.jsonl"
        self.shared_scheduler_path = FAMILY_ROOT / "ASCENSION_SCHEDULER_GENOME.jsonl"
        self.shared_negative_path = FAMILY_ROOT / "ASCENSION_NEGATIVE_SCIENCE.jsonl"
        self.transfer_path = FAMILY_ROOT / "ASCENSION_TRANSFER_MATRIX.json"
        self.mechanism_index_path = FAMILY_ROOT / "ASCENSION_MECHANISM_INDEX.sqlite"
        self._stopping = False

    @contextlib.contextmanager
    def _gpu_lease(self, *, stage: str) -> Iterator[None]:
        """Serialize actual MPS work while allowing CPU packing to overlap."""

        with _locked(GPU_LEASE_LOCK_PATH):
            acquired = _utc_now()
            _atomic_json(
                GPU_LEASE_STATUS_PATH,
                {
                    "schema": "hawking.ascension.gpu_lease.v1",
                    "status": "ACTIVE_EXCLUSIVE_GPU_LEASE",
                    "worker": self.spec.key,
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "stage": stage,
                    "acquired_at": acquired,
                    "claim_boundary": "this is a development component lease; qualifying complete-model benchmarks require their own exclusive receipt",
                },
            )
            try:
                yield
            finally:
                _atomic_json(
                    GPU_LEASE_STATUS_PATH,
                    {
                        "schema": "hawking.ascension.gpu_lease.v1",
                        "status": "RELEASED",
                        "worker": self.spec.key,
                        "pid": os.getpid(),
                        "ppid": os.getppid(),
                        "stage": stage,
                        "acquired_at": acquired,
                        "released_at": _utc_now(),
                    },
                )

    def _audit_weights(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        raw = _read_json(self.spec.source_audit)
        if raw is None:
            raise DualGravityError(f"source audit is missing: {self.spec.source_audit}")
        checked = verify(raw, label=str(self.spec.source_audit))
        source = checked.get("source") if isinstance(checked.get("source"), Mapping) else {}
        if source.get("repository") != self.spec.repository:
            raise DualGravityError("source audit repository does not match worker family")
        if source.get("revision") not in (None, self.spec.revision):
            raise DualGravityError("source audit revision does not match worker family")
        rows: dict[str, dict[str, Any]] = {}
        shards = source.get("shards") if isinstance(source.get("shards"), Mapping) else {}
        for name, row in shards.items():
            if isinstance(row, Mapping) and isinstance(row.get("sha256"), str):
                rows[str(name)] = {"bytes": int(row.get("bytes", -1)), "sha256": str(row["sha256"])}
        files = checked.get("files") if isinstance(checked.get("files"), list) else []
        for row in files:
            if not isinstance(row, Mapping):
                continue
            name = row.get("path")
            digest = row.get("sha256")
            if isinstance(name, str) and isinstance(digest, str) and name.endswith(".safetensors"):
                rows[name] = {"bytes": int(row.get("bytes", -1)), "sha256": digest}
        if not rows:
            raise DualGravityError("source audit has no verified safetensor shard hashes")
        return checked, rows

    def _source_identity(self, weight_map: Mapping[str, str]) -> dict[str, Any]:
        audit, audit_weights = self._audit_weights()
        required_shards = sorted(set(weight_map.values()))
        expected: list[dict[str, Any]] = []
        for shard in required_shards:
            if shard not in audit_weights:
                raise DualGravityError(f"source audit does not cover indexed shard {shard}")
            path = self.spec.source_dir / shard
            if not path.is_file():
                raise DualGravityError(f"source shard is absent: {path}")
            if path.stat().st_size != int(audit_weights[shard]["bytes"]):
                raise DualGravityError(f"source shard size differs from sealed audit: {shard}")
            expected.append({"path": shard, **audit_weights[shard]})
        controls: list[dict[str, Any]] = []
        for name in CONTROL_FILENAMES:
            path = self.spec.source_dir / name
            if not path.is_file():
                if name in REQUIRED_CONTROL_FILENAMES:
                    raise DualGravityError(f"required source control file is absent: {path}")
                continue
            controls.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
        content = {
            "repository": self.spec.repository,
            "revision": self.spec.revision,
            "architecture": self.spec.architecture,
            "verified_weight_shards": expected,
            "control_files": controls,
        }
        content_identity = _sha256(content)
        if self.identity_path.is_file():
            existing = _read_json(self.identity_path)
            if existing is None:
                raise DualGravityError("existing source identity is unreadable")
            verified_existing = verify(existing, label=str(self.identity_path))
            if verified_existing.get("content_identity_sha256") != content_identity:
                raise DualGravityError("source content identity changed; refusing to mix cohorts")
            return verified_existing
        record = seal(
            {
                "schema": IDENTITY_SCHEMA,
                "status": "IMMUTABLE_SOURCE_CONTENT_IDENTITY_BOUND",
                "recorded_at": _utc_now(),
                "model": {
                    "id": self.spec.model_id,
                    "repository": self.spec.repository,
                    "revision": self.spec.revision,
                    "architecture": self.spec.architecture,
                    "source_dir": str(self.spec.source_dir),
                },
                "weight_body_audit_path": str(self.spec.source_audit),
                "weight_body_audit_seal_sha256": audit.get("seal_sha256"),
                "source_content": content,
                "content_identity_sha256": content_identity,
                "claim_boundary": {
                    "content_identity_is_stable_across_status_heartbeats": True,
                    "source_body_is_teacher_authority_not_a_runtime_participant": True,
                    "shard_full_revalidation_is_owned_by_the_complete_artifact_compiler": True,
                },
            }
        )
        self.identity_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.identity_dir / f"{content_identity}.json", record)
        _atomic_json(self.identity_path, record)
        return record

    def _source_shard_path(self, shard: str) -> Path:
        """Resolve one index-referenced shard without admitting path escape."""

        if not isinstance(shard, str) or not shard:
            raise DualGravityError("source shard name must be a non-empty string")
        relative = Path(shard)
        if relative.is_absolute() or ".." in relative.parts:
            raise DualGravityError(f"source shard path escapes the source body: {shard!r}")
        source_root = self.spec.source_dir.resolve()
        candidate = self.spec.source_dir / relative
        try:
            # Resolve only the parent here.  Resolving the candidate itself
            # would follow a final symlink before _regular_file_identity can
            # reject it explicitly.
            candidate.parent.resolve().relative_to(source_root)
        except ValueError as exc:
            raise DualGravityError(f"source shard path escapes the source body: {shard!r}") from exc
        return candidate

    def _immutable_source_shards(
        self, identity: Mapping[str, Any], weight_map: Mapping[str, str]
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Validate the immutable source-content cohort and its exact shard set."""

        try:
            checked_identity = verify(identity, label=str(self.identity_path))
        except Exception as exc:
            raise DualGravityError(f"immutable source content identity is not trustworthy: {exc}") from exc
        if checked_identity.get("schema") != IDENTITY_SCHEMA:
            raise DualGravityError("immutable source content identity has an unexpected schema")
        if checked_identity.get("status") != "IMMUTABLE_SOURCE_CONTENT_IDENTITY_BOUND":
            raise DualGravityError("immutable source content identity is not bound")
        source_content = (
            checked_identity.get("source_content")
            if isinstance(checked_identity.get("source_content"), Mapping)
            else None
        )
        if source_content is None:
            raise DualGravityError("immutable source content identity is missing source_content")
        if checked_identity.get("content_identity_sha256") != _sha256(source_content):
            raise DualGravityError("immutable source content identity digest does not match source_content")
        if source_content.get("repository") != self.spec.repository:
            raise DualGravityError("immutable source content repository does not match worker family")
        if source_content.get("revision") != self.spec.revision:
            raise DualGravityError("immutable source content revision does not match worker family")
        if source_content.get("architecture") != self.spec.architecture:
            raise DualGravityError("immutable source content architecture does not match worker family")

        normalized_weight_map: dict[str, str] = {}
        for tensor_name, shard in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name or not isinstance(shard, str) or not shard:
                raise DualGravityError("source safetensors index contains an invalid tensor-to-shard entry")
            normalized_weight_map[tensor_name] = shard
        indexed_shards = set(normalized_weight_map.values())
        if not indexed_shards:
            raise DualGravityError("source safetensors index has no referenced shards")

        rows = source_content.get("verified_weight_shards")
        if not isinstance(rows, list) or not rows:
            raise DualGravityError("immutable source content identity has no verified weight shards")
        expected: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise DualGravityError("immutable source content identity has a malformed verified shard")
            path = row.get("path")
            digest = row.get("sha256")
            size = row.get("bytes")
            if not isinstance(path, str) or not path:
                raise DualGravityError("immutable source content identity has an unnamed verified shard")
            if path in expected:
                raise DualGravityError(f"immutable source content identity duplicates shard {path}")
            if not isinstance(digest, str) or len(digest) != 64:
                raise DualGravityError(f"immutable source content identity has an invalid SHA-256 for {path}")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise DualGravityError(f"immutable source content identity has a non-hex SHA-256 for {path}") from exc
            if type(size) is not int or size < 0:
                raise DualGravityError(f"immutable source content identity has invalid bytes for {path}")
            self._source_shard_path(path)
            expected[path] = {"sha256": digest, "bytes": size}
        if set(expected) != indexed_shards:
            missing = sorted(indexed_shards - set(expected))
            unexpected = sorted(set(expected) - indexed_shards)
            raise DualGravityError(
                "immutable source content shard set does not exactly match the safetensors index: "
                f"missing={missing} unexpected={unexpected}"
            )
        return checked_identity, expected

    def _current_source_revalidation(
        self,
        identity: Mapping[str, Any],
        weight_map: Mapping[str, str],
        *,
        target_shard: str,
    ) -> dict[str, Any]:
        """Admit a source tensor from the compiler's sealed current-shard receipt.

        This is deliberately a cheap per-candidate check.  It does not rehash
        a 30B/80B source body: the complete-artifact compiler owns that costly
        pass, and this worker requires its sealed receipt plus unchanged regular
        file identities before reading a tensor.
        """

        checked_identity, expected_shards = self._immutable_source_shards(identity, weight_map)
        if target_shard not in expected_shards:
            raise DualGravityError(f"source target shard is absent from immutable source content: {target_shard}")
        prefix = "QWEN30" if self.spec.key == "qwen30" else "QWEN80"
        receipt_path = self.spec.root / "complete-gravity" / f"{prefix}_CURRENT_SOURCE_SHARD_REVALIDATION.json"
        raw_receipt = _read_json(receipt_path)
        if raw_receipt is None:
            raise DualGravityError(f"source revalidation receipt is missing or unreadable: {receipt_path}")
        try:
            receipt = verify(raw_receipt, label=str(receipt_path))
        except Exception as exc:
            raise DualGravityError(f"source revalidation receipt is not trustworthy: {exc}") from exc
        if receipt.get("schema") != SOURCE_REVALIDATION_SCHEMA:
            raise DualGravityError("source revalidation receipt has an unexpected schema")
        if receipt.get("status") != SOURCE_REVALIDATION_STATUS:
            raise DualGravityError("source revalidation receipt is not an earned current-source result")

        source_content = checked_identity["source_content"]
        if receipt.get("source_repository") != source_content["repository"]:
            raise DualGravityError("source revalidation receipt repository does not match immutable source content")
        if receipt.get("source_revision") != source_content["revision"]:
            raise DualGravityError("source revalidation receipt revision does not match immutable source content")
        if receipt.get("source_model_dir") != str(self.spec.source_dir.resolve()):
            raise DualGravityError("source revalidation receipt model directory does not match current source body")

        control_rows = source_content.get("control_files")
        index_identity = next(
            (
                row
                for row in control_rows
                if isinstance(row, Mapping) and row.get("path") == "model.safetensors.index.json"
            ),
            None,
        ) if isinstance(control_rows, list) else None
        if not isinstance(index_identity, Mapping) or not isinstance(index_identity.get("sha256"), str):
            raise DualGravityError("immutable source content identity is missing the safetensors-index digest")
        if receipt.get("index_sha256") != index_identity["sha256"]:
            raise DualGravityError("source revalidation receipt index digest does not match immutable source content")

        normalized_weight_map = {str(tensor): str(shard) for tensor, shard in weight_map.items()}
        expected_hashes = {shard: row["sha256"] for shard, row in sorted(expected_shards.items())}
        if receipt.get("weight_map_sha256") != _sha256(dict(sorted(normalized_weight_map.items()))):
            raise DualGravityError("source revalidation receipt weight-map digest does not match the current index")
        if receipt.get("sealed_shard_hashes_sha256") != _sha256(expected_hashes):
            raise DualGravityError("source revalidation receipt shard-hash digest does not match immutable source content")
        if type(receipt.get("sealed_shard_count")) is not int or receipt["sealed_shard_count"] != len(expected_shards):
            raise DualGravityError("source revalidation receipt shard count does not match immutable source content")

        receipt_shards = receipt.get("shards")
        if not isinstance(receipt_shards, Mapping) or set(receipt_shards) != set(expected_shards):
            raise DualGravityError("source revalidation receipt shard set does not exactly match immutable source content")
        total_bytes = 0
        for shard in sorted(expected_shards):
            expected = expected_shards[shard]
            row = receipt_shards.get(shard)
            if not isinstance(row, Mapping):
                raise DualGravityError(f"source revalidation receipt shard row is malformed: {shard}")
            if row.get("expected_sha256") != expected["sha256"] or row.get("observed_sha256") != expected["sha256"]:
                raise DualGravityError(f"source revalidation receipt hash does not match immutable source content: {shard}")
            if type(row.get("expected_bytes")) is not int or row["expected_bytes"] != expected["bytes"]:
                raise DualGravityError(f"source revalidation receipt bytes do not match immutable source content: {shard}")
            receipt_identity = row.get("file_identity")
            if not isinstance(receipt_identity, Mapping):
                raise DualGravityError(f"source revalidation receipt has no file identity for {shard}")
            current_identity = _regular_file_identity(
                self._source_shard_path(shard), label=f"shard {shard}"
            )
            if dict(receipt_identity) != current_identity:
                raise DualGravityError(f"source shard file identity differs from the sealed revalidation receipt: {shard}")
            if current_identity["bytes"] != expected["bytes"]:
                raise DualGravityError(f"source shard observed bytes differ from immutable source content: {shard}")
            total_bytes += expected["bytes"]
        if type(receipt.get("observed_total_bytes")) is not int or receipt["observed_total_bytes"] != total_bytes:
            raise DualGravityError("source revalidation receipt total bytes do not match immutable source content")

        target_row = receipt_shards[target_shard]
        target_identity = target_row.get("file_identity")
        assert isinstance(target_identity, Mapping)  # validated above; keeps the proof precisely typed.
        return {
            "status": "VERIFIED_SEALED_CURRENT_SOURCE_REVALIDATION_BEFORE_TENSOR_READ",
            "receipt_path": str(receipt_path),
            "receipt_seal_sha256": receipt["seal_sha256"],
            "source_content_identity_sha256": checked_identity["content_identity_sha256"],
            "indexed_shard_count": len(expected_shards),
            "target_shard": target_shard,
            "target_shard_sha256": expected_shards[target_shard]["sha256"],
            "target_shard_bytes": expected_shards[target_shard]["bytes"],
            "target_file_identity": dict(target_identity),
            "claim_boundary": "sealed full-shard validation is reused via unchanged regular-file identities; no per-candidate full-shard hashing is inferred",
        }

    def _assert_revalidated_target_unchanged(self, proof: Mapping[str, Any]) -> None:
        """Catch a source mutation that races the tensor read without rehashing."""

        shard = proof.get("target_shard")
        expected_identity = proof.get("target_file_identity")
        if not isinstance(shard, str) or not isinstance(expected_identity, Mapping):
            raise DualGravityError("source revalidation proof lacks target shard identity")
        current_identity = _regular_file_identity(self._source_shard_path(shard), label=f"shard {shard}")
        if current_identity != dict(expected_identity):
            raise DualGravityError(f"source shard changed while its candidate tensor was being read: {shard}")

    def _default_state(self, identity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "model": self.spec.key,
            "model_id": self.spec.model_id,
            "source_content_identity_sha256": identity["content_identity_sha256"],
            "source_identity_path": str(self.identity_path),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "heartbeat": 0,
            "next_proposal_index": 0,
            "completed_candidate_count": 0,
            "failed_candidate_count": 0,
            "material_progress_count": 0,
            "last_material_progress_at": None,
            "integrity_audit_cursor": 0,
        }

    def _state(self, identity: Mapping[str, Any]) -> dict[str, Any]:
        existing = _read_json(self.state_path)
        if existing is None:
            return self._default_state(identity)
        if existing.get("schema") != SCHEMA or existing.get("model") != self.spec.key:
            raise DualGravityError("worker state belongs to a different schema or model")
        if existing.get("source_content_identity_sha256") != identity.get("content_identity_sha256"):
            raise DualGravityError("worker state binds a different immutable source cohort")
        return dict(existing)

    def _targets(self, weight_map: Mapping[str, str]) -> tuple[Target, ...]:
        targets: list[Target] = []

        def add(name: str, organ: str, sensitive: bool, notes: str) -> None:
            if name in weight_map and all(existing.name != name for existing in targets):
                targets.append(Target(name=name, organ=organ, sensitive=sensitive, notes=notes))

        if self.spec.key == "qwen30":
            for layer in (0, 12, 24, 36, 47):
                add(f"model.layers.{layer}.mlp.gate.weight", "moe_router", True, "128-expert top-8 router")
            for layer in (0, 24, 47):
                add(f"model.layers.{layer}.mlp.experts.0.gate_proj.weight", "routed_expert_gate", True, "routed expert sensitive organ")
                add(f"model.layers.{layer}.mlp.experts.63.up_proj.weight", "routed_expert_up", True, "routed expert activation projection")
                add(f"model.layers.{layer}.self_attn.q_proj.weight", "gqa_attention_q", True, "GQA attention projection region")
        else:
            for layer in (0, 12, 24, 36, 47):
                add(f"model.layers.{layer}.mlp.gate.weight", "moe_router", True, "512-expert top-10 router")
            for layer in (0, 24, 47):
                add(f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight", "shared_expert", True, "shared expert sensitive organ")
                add(f"model.layers.{layer}.mlp.experts.0.gate_proj.weight", "routed_expert_gate", True, "512-expert routed expert")
                add(f"model.layers.{layer}.linear_attn.in_proj_ba.weight", "gated_deltanet_projection", True, "DeltaNet projection")
                add(f"model.layers.{layer}.linear_attn.conv1d.weight", "gated_deltanet_convolution", True, "DeltaNet convolution")
            for layer in (3, 23, 47):
                add(f"model.layers.{layer}.self_attn.q_proj.weight", "hybrid_attention_q", True, "hybrid attention projection region")
        if not targets:
            raise DualGravityError("no expected physical target tensors were found in local source index")
        return tuple(targets)

    @staticmethod
    def _representations() -> tuple[str, ...]:
        """The frozen v1 family order retained for legacy sequence replay."""

        return LEGACY_REPRESENTATIONS

    @staticmethod
    def _expanded_representations() -> tuple[str, ...]:
        return EXPANDED_REPRESENTATIONS

    @staticmethod
    def _expansion_boundary_sequence(targets: Sequence[Target]) -> int:
        """The earliest possible v2 boundary for a fresh worker state."""

        if not targets:
            raise DualGravityError("cannot build a representation schedule without targets")
        return REPRESENTATION_EXPANSION_START_GENERATION * len(targets) * len(LEGACY_REPRESENTATIONS)

    def _schedule_plan(self, state: Mapping[str, Any], targets: Sequence[Target]) -> dict[str, Any]:
        """Resolve and durably pin the schedule migration before a proposal.

        The live v1 workers may have advanced while the upgraded code was being
        prepared.  Rather than reinterpret those already-issued positions, the
        first v2 process commits the next *complete* legacy generation boundary
        into its normal atomic checkpoint state.  Retry-before-checkpoint
        computes the same boundary from the same cursor; retry-after-checkpoint
        reads the pinned value.
        """

        legacy = self._representations()
        legacy_block = len(targets) * len(legacy)
        if legacy_block <= 0:
            raise DualGravityError("representation schedule has no legacy positions")
        existing = state.get("representation_schedule")
        if isinstance(existing, Mapping):
            try:
                boundary = int(existing["v2_start_sequence"])
                start_generation = int(existing["v2_start_generation"])
                target_count = int(existing["target_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DualGravityError("stored representation schedule is malformed") from exc
            if existing.get("schema") != REPRESENTATION_SCHEDULE_SCHEMA:
                raise DualGravityError("stored representation schedule schema is incompatible")
            if target_count != len(targets) or list(existing.get("legacy_representation_order", [])) != list(legacy):
                raise DualGravityError("stored representation schedule does not match immutable v1 mapping")
            if boundary != start_generation * legacy_block or start_generation < REPRESENTATION_EXPANSION_START_GENERATION:
                raise DualGravityError("stored representation schedule boundary is invalid")
            return dict(existing)
        next_sequence = max(0, int(state.get("next_proposal_index", 0)))
        # Strictly future: an old worker at the exact start of a legacy
        # generation must finish that whole generation before v2 begins.
        start_generation = max(
            REPRESENTATION_EXPANSION_START_GENERATION,
            next_sequence // legacy_block + 1,
        )
        boundary = start_generation * legacy_block
        plan = {
            "schema": REPRESENTATION_SCHEDULE_SCHEMA,
            "status": "PINNED_FUTURE_EXPANSION_BOUNDARY",
            "legacy_representation_schedule_version": LEGACY_REPRESENTATION_SCHEDULE_VERSION,
            "expanded_representation_schedule_version": EXPANDED_REPRESENTATION_SCHEDULE_VERSION,
            "legacy_representation_order": list(legacy),
            "expanded_representation_order": list(self._expanded_representations()),
            "target_count": len(targets),
            "legacy_block_size": legacy_block,
            "v2_start_generation": start_generation,
            "v2_start_sequence": boundary,
            "pinned_from_next_proposal_index": next_sequence,
            "claim_boundary": "schedule only; all representations remain source-region component artifacts until independent full-model gates are earned",
        }
        if isinstance(state, dict):
            state["representation_schedule"] = plan
        return plan

    def _proposal(self, state: Mapping[str, Any], targets: Sequence[Target], identity: Mapping[str, Any]) -> Proposal:
        sequence = int(state.get("next_proposal_index", 0))
        legacy = self._representations()
        schedule_plan = self._schedule_plan(state, targets)
        boundary_sequence = int(schedule_plan["v2_start_sequence"])
        start_generation = int(schedule_plan["v2_start_generation"])
        if sequence < boundary_sequence:
            representations = legacy
            block = len(targets) * len(representations)
            generation = sequence // block
            within = sequence % block
            schedule_version = LEGACY_REPRESENTATION_SCHEDULE_VERSION
            schedule_phase = "legacy_seven_family_frozen"
        else:
            representations = self._expanded_representations()
            block = len(targets) * len(representations)
            relative_sequence = sequence - boundary_sequence
            generation = start_generation + relative_sequence // block
            within = relative_sequence % block
            schedule_version = EXPANDED_REPRESENTATION_SCHEDULE_VERSION
            schedule_phase = "expanded_ten_family_after_complete_legacy_generation"
        target = targets[within // len(representations)]
        representation = representations[within % len(representations)]
        config = _representation_config(representation, generation)
        schedule = {
            "version": schedule_version,
            "phase": schedule_phase,
            "legacy_representation_order": list(legacy),
            "active_representation_order": list(representations),
            "expansion_start_generation": start_generation,
            "expansion_boundary_sequence": boundary_sequence,
            "sequence_mapping": "sequence<boundary uses v1 blocks; sequence>=boundary uses v2 offset blocks",
        }
        genome = {
            "source_content_identity_sha256": identity["content_identity_sha256"],
            "model": self.spec.key,
            "sequence": sequence,
            "generation": generation,
            "tensor": target.name,
            "organ": target.organ,
            "representation": representation,
            "config": config,
            "schedule": schedule,
        }
        # Preserve the historical v1 candidate-ID derivation for all legacy
        # positions. v2 IDs deliberately key only on immutable schedule/source
        # facts; the mutable implementation digest is recorded in the receipt
        # but cannot perturb a future candidate's lineage.
        if schedule_version == LEGACY_REPRESENTATION_SCHEDULE_VERSION:
            candidate_id_genome = {
                "source_content_identity_sha256": identity["content_identity_sha256"],
                "model": self.spec.key,
                "sequence": sequence,
                "generation": generation,
                "tensor": target.name,
                "organ": target.organ,
                "representation": representation,
                "config": config,
                "worker_code_sha256": _sha256_file(Path(__file__)),
            }
        else:
            candidate_id_genome = genome
        return Proposal(
            sequence=sequence,
            generation=generation,
            target=target,
            representation=representation,
            config=config,
            candidate_id=f"{self.spec.key}-{_sha256(candidate_id_genome)[:32]}",
            schedule_version=schedule_version,
            schedule_phase=schedule_phase,
            schedule_boundary_sequence=boundary_sequence,
            schedule_start_generation=start_generation,
        )

    def _load_index(self, identity: Mapping[str, Any]) -> list[dict[str, Any]]:
        existing = _read_json(self.index_path)
        if existing is None:
            return self._rebuild_index(identity)
        try:
            checked = verify(existing, label=str(self.index_path))
        except Exception:
            return self._rebuild_index(identity)
        if checked.get("source_content_identity_sha256") != identity.get("content_identity_sha256"):
            raise DualGravityError("candidate index binds a different source cohort")
        rows = checked.get("candidates")
        if not isinstance(rows, list):
            return self._rebuild_index(identity)
        valid: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            record_path = Path(str(row.get("record_path", "")))
            artifact_path = Path(str(row.get("artifact_path", "")))
            if not record_path.is_file() or not artifact_path.is_file():
                continue
            record = _read_json(record_path)
            try:
                if record is None:
                    continue
                checked_record = verify(record, label=str(record_path))
            except Exception:
                continue
            if checked_record.get("candidate_id") != row.get("candidate_id"):
                continue
            if int(row.get("artifact_bytes", -1)) != artifact_path.stat().st_size:
                continue
            valid.append(dict(row))
        if len(valid) != len(rows):
            self._write_index(identity, valid)
        return valid

    def _rebuild_index(self, identity: Mapping[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(self.candidate_dir.glob("*.json")):
            record = _read_json(path)
            try:
                if record is None:
                    continue
                checked = verify(record, label=str(path))
            except Exception:
                continue
            if checked.get("source_content_identity_sha256") != identity.get("content_identity_sha256"):
                continue
            artifact = Path(str(checked.get("artifact", {}).get("path", "")))
            if not artifact.is_file():
                continue
            expected_sha = checked.get("artifact", {}).get("sha256")
            if not isinstance(expected_sha, str) or _sha256_file(artifact) != expected_sha:
                continue
            records.append(self._index_row(checked, path))
        self._write_index(identity, records)
        return records

    def _write_index(self, identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
        document = seal(
            {
                "schema": INDEX_SCHEMA,
                "status": "DURABLE_CANDIDATE_INDEX",
                "recorded_at": _utc_now(),
                "model": self.spec.key,
                "source_content_identity_sha256": identity["content_identity_sha256"],
                "candidate_count": len(rows),
                "candidates": [dict(row) for row in rows],
                "claim_boundary": "index is research evidence only; no member is a full-model runtime or tournament candidate",
            }
        )
        _atomic_json(self.index_path, document)

    @staticmethod
    def _index_row(record: Mapping[str, Any], record_path: Path) -> dict[str, Any]:
        artifact = record.get("artifact") if isinstance(record.get("artifact"), Mapping) else {}
        measurement = record.get("measurement") if isinstance(record.get("measurement"), Mapping) else {}
        functional = measurement.get("functional_probe") if isinstance(measurement.get("functional_probe"), Mapping) else {}
        component = measurement.get("mps_component_probe") if isinstance(measurement.get("mps_component_probe"), Mapping) else {}
        source = record.get("source") if isinstance(record.get("source"), Mapping) else {}
        representation = record.get("representation") if isinstance(record.get("representation"), Mapping) else {}
        return {
            "candidate_id": record.get("candidate_id"),
            "record_path": str(record_path),
            "record_seal_sha256": record.get("seal_sha256"),
            "artifact_path": artifact.get("path"),
            "artifact_sha256": artifact.get("sha256"),
            "artifact_bytes": artifact.get("bytes"),
            "physical_bpw": artifact.get("physical_bpw"),
            "relative_l2": measurement.get("weight_reconstruction", {}).get("relative_l2") if isinstance(measurement.get("weight_reconstruction"), Mapping) else None,
            "cosine": measurement.get("weight_reconstruction", {}).get("cosine") if isinstance(measurement.get("weight_reconstruction"), Mapping) else None,
            "functional_relative_l2": functional.get("relative_l2"),
            "functional_cosine": functional.get("cosine"),
            "mps_matvecs_per_second": component.get("matvecs_per_second"),
            "representation": representation.get("family"),
            "tensor_name": source.get("tensor_name"),
            "organ": source.get("organ"),
            "generation": record.get("genome", {}).get("generation") if isinstance(record.get("genome"), Mapping) else None,
            "schedule_version": (
                record.get("genome", {}).get("schedule", {}).get("version")
                if isinstance(record.get("genome"), Mapping) and isinstance(record.get("genome", {}).get("schedule"), Mapping)
                else LEGACY_REPRESENTATION_SCHEDULE_VERSION
            ),
            "sensitive_gate_pass": record.get("sensitive_organ_gate", {}).get("passes") if isinstance(record.get("sensitive_organ_gate"), Mapping) else False,
            "created_at": record.get("recorded_at"),
        }

    def _active_negative_keys(self) -> set[str]:
        keys: set[str] = set()
        for row in _jsonl(self.shared_negative_path):
            if row.get("status") == "BURIED" and isinstance(row.get("mechanism_key"), str):
                keys.add(str(row["mechanism_key"]))
        return keys

    def _knowledge_snapshot(self, proposal: Proposal) -> dict[str, Any]:
        matching_transfer = [
            row
            for row in _jsonl(self.shared_mechanisms_path)
            if row.get("representation") == proposal.representation and row.get("model_family") != self.spec.model_family
        ][-8:]
        # A burial is intentionally configuration-specific.  A later genome
        # with a different rank, group geometry, threshold, or residual budget
        # is a materially different premise rather than a silent rerun.
        mechanism_key = (
            f"{proposal.representation}:{proposal.target.organ}:{self.spec.model_family}:"
            f"{_sha256(dict(proposal.config))[:16]}"
        )
        active = mechanism_key in self._active_negative_keys()
        return {
            "negative_science_checked": True,
            "mechanism_key": mechanism_key,
            "may_proceed": not active,
            "active_matching_burial": active,
            "cross_family_prior_count": len(matching_transfer),
            "cross_family_priors": matching_transfer,
            "claim_boundary": "transfer priors tune scheduling only; exact model evidence is still required",
        }

    def _crop_region(self, values: np.ndarray, *, proposal: Proposal) -> tuple[np.ndarray, dict[str, Any]]:
        source = np.ascontiguousarray(values, dtype=np.float32)
        if source.size <= MAX_REGION_ELEMENTS:
            return source, {"kind": "complete_tensor", "shape": [int(item) for item in source.shape], "elements": int(source.size)}
        if source.ndim < 2:
            region = source.reshape(-1)[:MAX_REGION_ELEMENTS]
            return region, {"kind": "flat_prefix", "shape": [int(item) for item in region.shape], "elements": int(region.size)}
        row_width = int(math.prod(source.shape[1:]))
        rows = max(1, min(source.shape[0], MAX_REGION_ELEMENTS // max(row_width, 1)))
        span = int(source.shape[0]) - rows + 1
        seed = int(_sha256(proposal.candidate_id)[:16], 16)
        start = seed % max(span, 1)
        region = np.ascontiguousarray(source[start : start + rows])
        return region, {
            "kind": "row_window",
            "axis": 0,
            "start": int(start),
            "stop": int(start + rows),
            "source_shape": [int(item) for item in source.shape],
            "shape": [int(item) for item in region.shape],
            "elements": int(region.size),
        }

    @staticmethod
    def _sensitive_gate(target: Target, functional: Mapping[str, Any]) -> dict[str, Any]:
        threshold = {
            "moe_router": 0.18,
            "gqa_attention_q": 0.28,
            "hybrid_attention_q": 0.28,
            "gated_deltanet_projection": 0.24,
            "gated_deltanet_convolution": 0.20,
            "shared_expert": 0.26,
            "routed_expert_gate": 0.28,
            "routed_expert_up": 0.30,
        }.get(target.organ, 0.30)
        observed = functional.get("relative_l2")
        passes = isinstance(observed, (int, float)) and math.isfinite(float(observed)) and float(observed) <= threshold
        return {
            "status": "RAN_COMPONENT_SENSITIVE_ORGAN_GATE",
            "organ": target.organ,
            "sensitive": target.sensitive,
            "functional_relative_l2": observed,
            "threshold": threshold,
            "passes": passes,
            "claim_boundary": "component organ gate only; it cannot establish full-model capability",
        }

    def _state_kv_snapshot(self) -> dict[str, Any]:
        """Expose exact static state geometry without inventing dynamic runtime data."""

        lane_root = self.spec.root / "state-kv"
        prefix = "QWEN30" if self.spec.key == "qwen30" else "QWEN80"
        status_path = lane_root / f"{prefix}_STATE_KV_STATUS.json"
        measured = _read_json(status_path)
        if measured is not None:
            # The state/KV worker publishes its results below a sealed
            # ``outcome`` object.  Verify that receipt before using it in an
            # evolutionary genome: an unsealed JSON status must never become
            # a source of scheduling or state-memory evidence.
            try:
                checked = verify(measured, label=str(status_path))
            except Exception as exc:
                return {
                    "status": "INVALID_SEALED_STATE_KV_RECEIPT",
                    "status_path": str(status_path),
                    "error": str(exc),
                    "claim_boundary": "invalid state/KV evidence is excluded from candidate genomes",
                }
            outcome = checked.get("outcome") if isinstance(checked.get("outcome"), Mapping) else {}
            geometry = outcome.get("geometry") if isinstance(outcome.get("geometry"), Mapping) else None
            receipt_keys = (
                ("attention_kv", "receipt_path", "receipt_seal_sha256"),
                ("deltanet_recurrent_state", "deltanet_receipt_path", "deltanet_receipt_seal_sha256"),
            )
            source_receipts: list[dict[str, Any]] = []
            for component, path_key, seal_key in receipt_keys:
                raw_path = outcome.get(path_key)
                if not isinstance(raw_path, str):
                    continue
                receipt_path = Path(raw_path)
                receipt = _read_json(receipt_path)
                try:
                    verified_receipt = verify(receipt or {}, label=str(receipt_path))
                    observed_seal = verified_receipt.get("seal_sha256")
                    expected_seal = outcome.get(seal_key)
                    if expected_seal != observed_seal:
                        receipt_status = "INVALID"
                        error = "component receipt seal does not match the sealed state/KV status reference"
                    else:
                        receipt_status = "VERIFIED"
                except Exception as exc:
                    receipt_status = "INVALID"
                    observed_seal = None
                    error = str(exc)
                row: dict[str, Any] = {
                    "component": component,
                    "path": str(receipt_path),
                    "expected_seal_sha256": outcome.get(seal_key),
                    "observed_seal_sha256": observed_seal,
                    "status": receipt_status,
                }
                if receipt_status == "INVALID":
                    row["error"] = error
                source_receipts.append(row)
            traffic_receipts: list[dict[str, Any]] = []
            state_traffic = outcome.get("state_traffic")
            if isinstance(state_traffic, Mapping):
                if isinstance(state_traffic.get("path"), str):
                    traffic_bindings = [("state_traffic", state_traffic)]
                else:
                    traffic_bindings = [
                        (str(component), binding)
                        for component, binding in state_traffic.items()
                        if isinstance(binding, Mapping)
                    ]
                for component, binding in traffic_bindings:
                    raw_path = binding.get("path")
                    expected_seal = binding.get("seal_sha256")
                    row: dict[str, Any] = {
                        "component": component,
                        "path": raw_path,
                        "expected_seal_sha256": expected_seal,
                    }
                    if not isinstance(raw_path, str) or not isinstance(expected_seal, str):
                        row.update({"status": "INVALID", "error": "traffic binding lacks path or sealed receipt hash"})
                    else:
                        receipt_path = Path(raw_path)
                        try:
                            verified_receipt = verify(_read_json(receipt_path) or {}, label=str(receipt_path))
                            observed_seal = verified_receipt.get("seal_sha256")
                            row["observed_seal_sha256"] = observed_seal
                            if observed_seal == expected_seal:
                                row["status"] = "VERIFIED"
                            else:
                                row.update({
                                    "status": "INVALID",
                                    "error": "traffic receipt seal does not match the sealed state/KV status reference",
                                })
                        except Exception as exc:
                            row.update({"status": "INVALID", "error": str(exc)})
                    traffic_receipts.append(row)
            restart_recovery: dict[str, Any] | None = None
            recovery_binding = checked.get("sealed_restart_recovery")
            if isinstance(recovery_binding, Mapping):
                raw_path = recovery_binding.get("path")
                expected_seal = recovery_binding.get("seal_sha256")
                restart_recovery = {
                    "path": raw_path,
                    "expected_seal_sha256": expected_seal,
                    "manifest_count": recovery_binding.get("manifest_count"),
                    "recovery_pid": recovery_binding.get("recovery_pid"),
                }
                if not isinstance(raw_path, str) or not isinstance(expected_seal, str):
                    restart_recovery.update({"status": "INVALID", "error": "recovery binding lacks path or sealed receipt hash"})
                else:
                    recovery_path = Path(raw_path)
                    try:
                        recovery = verify(_read_json(recovery_path) or {}, label=str(recovery_path))
                        observed_seal = recovery.get("seal_sha256")
                        restart_recovery["observed_seal_sha256"] = observed_seal
                        if observed_seal == expected_seal and recovery.get("status") == "SEALED_RESTART_EQUIVALENT_DURABLE_COMPONENT_STATE_RECOVERY_VERIFIED":
                            restart_recovery["status"] = "VERIFIED_SEPARATE_PROCESS_REHYDRATION"
                        else:
                            restart_recovery.update({"status": "INVALID", "error": "recovery receipt does not prove the expected sealed restart state"})
                    except Exception as exc:
                        restart_recovery.update({"status": "INVALID", "error": str(exc)})
            evidence_invalid = (
                any(row["status"] != "VERIFIED" for row in source_receipts)
                or any(row["status"] != "VERIFIED" for row in traffic_receipts)
                or (restart_recovery is not None and restart_recovery.get("status") != "VERIFIED_SEPARATE_PROCESS_REHYDRATION")
            )
            if geometry is None or evidence_invalid:
                return {
                    "status": "STATE_KV_RECEIPT_INCOMPLETE_OR_INVALID",
                    "status_path": str(status_path),
                    "seal_sha256": checked.get("seal_sha256"),
                    "source_component_receipts": source_receipts,
                    "traffic_receipts": traffic_receipts,
                    "sealed_restart_recovery": restart_recovery,
                    "claim_boundary": "incomplete state/KV evidence is excluded from candidate genomes",
                }
            return {
                "status": checked.get("phase"),
                "status_path": str(status_path),
                "seal_sha256": checked.get("seal_sha256"),
                "heartbeat": checked.get("heartbeat"),
                "measured_source_bound_component_lane": True,
                "component_artifact_count": outcome.get("artifact_count"),
                "geometry": geometry,
                "source_component_receipts": source_receipts,
                "traffic_receipts": traffic_receipts,
                "sealed_restart_recovery": restart_recovery,
                "claim_boundary": "state/KV lane is source-bound component evidence only; it remains separate from model weight BPW and complete-token qualification",
            }
        if self.spec.key == "qwen30":
            attention_layers = 48
            kv_bytes_per_token_fp16 = attention_layers * self.spec.kv_heads * self.spec.head_dim * 2 * 2
            geometry = {
                "attention_layers": attention_layers,
                "kv_heads": self.spec.kv_heads,
                "head_dim": self.spec.head_dim,
                "key_value_streams": 2,
                "reference_fp16_kv_bytes_per_token": kv_bytes_per_token_fp16,
                "reference_fp16_kv_bytes_per_4096_token_session": kv_bytes_per_token_fp16 * 4096,
            }
        else:
            # The source index's hybrid schedule has 12 full-attention layers
            # and 36 DeltaNet layers; the latter's exact component geometry is
            # 32 × 128 × 128 recurrent elements per layer.
            attention_layers = 12
            kv_bytes_per_token_fp16 = attention_layers * self.spec.kv_heads * self.spec.head_dim * 2 * 2
            deltanet_elements_per_layer = 32 * 128 * 128
            geometry = {
                "attention_layers": attention_layers,
                "kv_heads": self.spec.kv_heads,
                "head_dim": self.spec.head_dim,
                "key_value_streams": 2,
                "reference_fp16_attention_kv_bytes_per_token": kv_bytes_per_token_fp16,
                "reference_fp16_attention_kv_bytes_per_4096_token_session": kv_bytes_per_token_fp16 * 4096,
                "deltanet_layers": 36,
                "deltanet_recurrent_elements_per_layer": deltanet_elements_per_layer,
                "reference_fp16_deltanet_recurrent_bytes_per_session": 36 * deltanet_elements_per_layer * 2,
            }
        return {
            "status": "EXACT_STATIC_GEOMETRY_READY_DYNAMIC_SOURCE_BOUND_STATE_LANE_PENDING",
            "status_path": str(status_path),
            "geometry": geometry,
            "candidate_formats": [
                "native_reference_fp16",
                "grouped_q8_state",
                "grouped_q4_state",
                "protected_residual_state",
            ],
            "claim_boundary": "geometry accounting only; no real prompt-derived KV/state, long-context, session, or restart pass yet",
        }

    def _kernel_snapshot(self) -> dict[str, Any]:
        kernel_root = PHYSICAL_ROOT / "kernel"
        names = [
            "QWEN_DUAL_ROUTE_METAL_COMPONENT_PROBE.json",
            "QWEN30_GQA_METAL_COMPONENT_PROBE.json",
            "QWEN_NEXT_GATED_DELTANET_METAL_COMPONENT_PROBE.json",
            "QWEN_BINARY_SIGN_SCALE_MATVEC_METAL_COMPONENT_PROBE.json",
            "QWEN_UNIFORM_Q4_GROUP64_MATVEC_METAL_COMPONENT_PROBE.json",
        ]
        rows = []
        for name in names:
            path = kernel_root / name
            if path.is_file():
                document = _read_json(path) or {}
                # Bind the exact evidence bytes without embedding a growing
                # copy of every full component receipt into every candidate.
                rows.append(
                    {
                        "path": str(path),
                        "sha256": _sha256_file(path),
                        "schema": document.get("schema"),
                        "status": document.get("status"),
                        "device": document.get("device"),
                    }
                )
        return {
            "available_component_evidence": rows,
            "required_before_operational": {
                "custom_kernel_base_true_tps_minimum": 100.0,
                "full_token_tg3_base_true_tps_minimum": 333.0,
                "current_full_token_result": None,
                "status": "BLOCKED_UNTIL_EXACT_COMPLETE_NATIVE_RUNTIME_EXISTS",
            },
            "claim_boundary": "component probes and library MPS measurements are never model TPS",
        }

    def _complete_pack_snapshot(self) -> dict[str, Any]:
        root = self.spec.root / "complete-gravity"
        prefix = "QWEN30" if self.spec.key == "qwen30" else "QWEN80"
        status = _read_json(root / f"{prefix}_COMPLETE_GRAVITY_STATUS.json") or {}
        manifest = _read_json(root / f"{prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json")
        manifest_summary: dict[str, Any] = {
            "manifest_path": str(root / f"{prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"),
            "manifest_present": manifest is not None,
        }
        if manifest is not None:
            try:
                checked = verify(manifest, label=str(root / f"{prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"))
                ledger = (
                    checked.get("complete_physical_bpw_ledger")
                    if isinstance(checked.get("complete_physical_bpw_ledger"), Mapping)
                    else {}
                )
                manifest_summary.update(
                    {
                        "manifest_status": checked.get("status"),
                        "manifest_seal_sha256": checked.get("seal_sha256"),
                        "complete_physical_bpw": ledger.get("complete_physical_bpw"),
                        "storage_threshold_pass": ledger.get("passes_storage_threshold"),
                    }
                )
            except Exception as exc:
                manifest_summary.update(
                    {
                        "manifest_status": "INVALID_SEALED_MANIFEST",
                        "manifest_error": str(exc),
                    }
                )
        return {
            "status_path": str(root / f"{prefix}_COMPLETE_GRAVITY_STATUS.json"),
            "phase": status.get("phase"),
            "pid": status.get("pid"),
            "recorded_at": status.get("recorded_at"),
            "progress": status.get("progress"),
            **manifest_summary,
            "claim_boundary": "fixed full-pack baseline is separate from this component population until a complete all-tensor genome is compiled",
        }

    def _runtime_snapshot(self) -> dict[str, Any]:
        """Surface native-runtime gates without turning blocked receipts into claims."""

        prefix = "QWEN30" if self.spec.key == "qwen30" else "QWEN80"
        runtime_path = self.spec.root / "complete-runtime" / f"{prefix}_COMPLETE_RUNTIME_STATUS.json"
        tg_path = self.spec.root / "tg3" / f"{prefix}_TG3_ASCENT_STATUS.json"
        runtime = _read_json(runtime_path) or {}
        tg = _read_json(tg_path) or {}
        runtime_phase = runtime.get("phase")
        tg_phase = tg.get("phase")
        base_true_tps = runtime.get("base_true_tps")
        if not isinstance(base_true_tps, (int, float)):
            base_true_tps = None
        tg_rung = runtime.get("tg_rung") if isinstance(runtime.get("tg_rung"), str) else None
        if tg_rung is None and isinstance(tg.get("tg_rung"), str):
            tg_rung = tg.get("tg_rung")
        native_ready = runtime_phase in {
            "EARNED_COMPLETE_NATIVE_RUNTIME",
            "EARNED_COMPLETE_NATIVE_RUNTIME_100_TPS_OPERATIONAL",
            "EARNED_TG3_COMPLETE_NATIVE_RUNTIME",
        }
        return {
            "complete_runtime_status_path": str(runtime_path),
            "tg3_status_path": str(tg_path),
            "complete_runtime_phase": runtime_phase,
            "tg3_phase": tg_phase,
            "pid": runtime.get("pid"),
            "current_base_true_tps": base_true_tps,
            "tg_rung": tg_rung,
            "hcli_status": runtime.get("hcli_status") or ("ELIGIBLE_FOR_REAL_TEST" if native_ready else "BLOCKED_NO_COMPLETE_NATIVE_RUNTIME"),
            "current_kernel_bottleneck": (
                "UNBLOCKED_BY_EARNED_COMPLETE_NATIVE_RUNTIME"
                if native_ready
                else "EXACT_COMPLETE_NATIVE_DECODER_AND_FULL_TOKEN_GRAPH_NOT_YET_IMPLEMENTED"
            ),
            "claim_boundary": "missing/blocked runtime receipts mean no TPS, HCLI, or TG qualification is inferred",
        }

    @staticmethod
    def _finite(value: Any, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        return number if math.isfinite(number) else fallback

    def _frontier_and_champions(
        self, identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidates = [
            dict(row)
            for row in rows
            if isinstance(row.get("physical_bpw"), (int, float))
            and isinstance(row.get("functional_relative_l2"), (int, float))
            and isinstance(row.get("relative_l2"), (int, float))
        ]

        def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
            lv = (self._finite(left.get("physical_bpw"), math.inf), self._finite(left.get("functional_relative_l2"), math.inf), self._finite(left.get("relative_l2"), math.inf))
            rv = (self._finite(right.get("physical_bpw"), math.inf), self._finite(right.get("functional_relative_l2"), math.inf), self._finite(right.get("relative_l2"), math.inf))
            return all(a <= b for a, b in zip(lv, rv)) and any(a < b for a, b in zip(lv, rv))

        pure = [row for row in candidates if not any(other is not row and dominates(other, row) for other in candidates)]
        pure.sort(key=lambda row: (self._finite(row.get("functional_relative_l2"), math.inf), self._finite(row.get("physical_bpw"), math.inf), str(row.get("candidate_id"))))
        retained = list(pure[:FRONTIER_LIMIT])
        represented = {str(row.get("representation")) for row in retained}
        for representation in sorted({str(row.get("representation")) for row in candidates} - represented):
            family = [row for row in candidates if row.get("representation") == representation]
            family.sort(key=lambda row: (self._finite(row.get("functional_relative_l2"), math.inf), self._finite(row.get("physical_bpw"), math.inf), str(row.get("candidate_id"))))
            if family and len(retained) < FRONTIER_LIMIT:
                retained.append(family[0])
        frontier = seal(
            {
                "schema": FRONTIER_SCHEMA,
                "status": "COMPONENT_PHYSICAL_PARETO_FRONTIER_NOT_FULL_MODEL_SELECTION",
                "recorded_at": _utc_now(),
                "model": self.spec.key,
                "source_content_identity_sha256": identity["content_identity_sha256"],
                "objective_order": ["min_physical_bpw", "min_functional_component_relative_l2", "min_weight_relative_l2"],
                "candidate_count": len(candidates),
                "pure_pareto_count": len(pure),
                "diversity_retained_count": len(retained),
                "representation_classes_present": sorted({str(row.get("representation")) for row in candidates}),
                "members": retained,
                "claim_boundary": {
                    "frontier_is_component_physical_research_only": True,
                    "no_frontier_member_is_capability_or_runtime_qualified": True,
                    "selection_cannot_replace_complete_all_tensor_compile": True,
                },
            }
        )
        best_bpw = min(candidates, key=lambda row: (self._finite(row.get("physical_bpw"), math.inf), str(row.get("candidate_id"))), default=None)
        best_fidelity = min(candidates, key=lambda row: (self._finite(row.get("functional_relative_l2"), math.inf), str(row.get("candidate_id"))), default=None)
        timed = [row for row in candidates if isinstance(row.get("mps_matvecs_per_second"), (int, float))]
        fastest = max(timed, key=lambda row: (self._finite(row.get("mps_matvecs_per_second"), -math.inf), str(row.get("candidate_id"))), default=None)
        champions = seal(
            {
                "schema": CHAMPION_SCHEMA,
                "status": "COMPONENT_CHAMPIONS_NOT_TOURNAMENT_SELECTION",
                "recorded_at": _utc_now(),
                "model": self.spec.key,
                "source_content_identity_sha256": identity["content_identity_sha256"],
                "current_lowest_bpw_champion": best_bpw,
                "current_most_faithful_component_champion": best_fidelity,
                "current_fastest_component_champion": fastest,
                "current_capable_champion": {
                    "candidate": None,
                    "status": "UNSET_BLOCKED_NO_NATIVE_COMPLETE_TOKEN_RUNTIME_OR_CAPABILITY_EVIDENCE",
                },
                "current_full_runtime_champion": {
                    "candidate": None,
                    "status": "UNSET_BLOCKED_100_TPS_CUSTOM_KERNEL_AND_TG3_REQUIRE_FULL_TOKEN_RUNTIME",
                },
                "tournament_selection": {
                    "status": "FORBIDDEN",
                    "reason": "component candidates cannot self-select into the protected tournament",
                },
                "claim_boundary": "component MPS matvec throughput is not a model TPS value",
            }
        )
        return frontier, champions

    def _index_knowledge_record(self, row: Mapping[str, Any], *, kind: str) -> None:
        """Maintain the actual shared mechanism index alongside append-only JSONL."""

        record_id = row.get("record_id")
        if not isinstance(record_id, str):
            raise DualGravityError("knowledge record lacks record_id")
        lock_path = FAMILY_ROOT / ".mechanism-index.lock"
        with _locked(lock_path):
            FAMILY_ROOT.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.mechanism_index_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mechanisms (
                      record_id TEXT PRIMARY KEY,
                      kind TEXT NOT NULL,
                      model_family TEXT,
                      representation TEXT,
                      organ TEXT,
                      candidate_id TEXT,
                      recorded_at TEXT,
                      sealed_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS mechanisms_lookup ON mechanisms (kind, model_family, representation, organ)"
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO mechanisms
                    (record_id, kind, model_family, representation, organ, candidate_id, recorded_at, sealed_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        kind,
                        row.get("model_family"),
                        row.get("representation"),
                        row.get("organ") or row.get("tensor_or_organ"),
                        row.get("candidate_id"),
                        row.get("recorded_at"),
                        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def _scheduler_record(self, record: Mapping[str, Any], proposal: Proposal) -> dict[str, Any]:
        training = record.get("training") if isinstance(record.get("training"), Mapping) else {}
        measurement = record.get("measurement") if isinstance(record.get("measurement"), Mapping) else {}
        mps = measurement.get("mps_component_probe") if isinstance(measurement.get("mps_component_probe"), Mapping) else {}
        return seal(
            {
                "schema": "hawking.ascension.knowledge_scheduler_genome.v1",
                "record_id": f"scheduler:{record['candidate_id']}",
                "recorded_at": _utc_now(),
                "task_class": "qwen_component_gravity_mutation",
                "resource_class": {
                    "cpu": "source_tensor_read_pack_measure",
                    "gpu": training.get("device") == "mps" or mps.get("device") == "mps",
                    "gpu_lease": str(GPU_LEASE_STATUS_PATH),
                    "disk": "atomic_candidate_artifact_and_receipt",
                },
                "dependency": "verified source content identity + negative science preflight",
                "critical_path_status": "feeds complete artifact and native runtime, which remain blocking",
                "preemptibility": "checkpoint after every sealed candidate",
                "checkpoint_cost": {"candidate_artifact_bytes": record.get("artifact", {}).get("bytes") if isinstance(record.get("artifact"), Mapping) else None},
                "measured_progress_per_watt": "NOT_MEASURED",
                "contention_outcome": "GPU component work serialized by durable lease; CPU packing may overlap other worker",
                "validity_scope": f"{self.spec.model_family}:{proposal.target.organ}:{proposal.representation}",
                "candidate_id": record.get("candidate_id"),
                "schedule_version": proposal.schedule_version,
                "schedule_phase": proposal.schedule_phase,
                "model_family": self.spec.model_family,
                "representation": proposal.representation,
                "organ": proposal.target.organ,
                "claim_boundary": "development scheduling evidence only; not a qualifying benchmark resource receipt",
            }
        )

    def _kernel_record(self, record: Mapping[str, Any], proposal: Proposal) -> dict[str, Any]:
        source = record.get("source") if isinstance(record.get("source"), Mapping) else {}
        measurement = record.get("measurement") if isinstance(record.get("measurement"), Mapping) else {}
        mps = measurement.get("mps_component_probe") if isinstance(measurement.get("mps_component_probe"), Mapping) else {}
        return seal(
            {
                "schema": "hawking.ascension.knowledge_kernel_genome.v1",
                "record_id": f"kernel:{record['candidate_id']}",
                "recorded_at": _utc_now(),
                "operator": "packed_representation_component_matvec",
                "model_family": self.spec.model_family,
                "tensor_geometry": source.get("region"),
                "representation": proposal.representation,
                "kernel_grammar": "binary/uniform/ternary/sparse-residual/low-rank/Hadamard-lattice/additive-codebook/activation-corrected direct component candidate",
                "tile": "not yet autotuned; source-bound component geometry only",
                "threadgroup": "see byte-bound Metal component receipts when available",
                "memory_layout": record.get("representation", {}).get("codec_metadata") if isinstance(record.get("representation"), Mapping) else None,
                "command_graph": "single component dispatch / library smoke; no complete token graph",
                "measured_latency": mps.get("elapsed_seconds"),
                "bandwidth": "NOT_MEASURED",
                "occupancy": "NOT_MEASURED",
                "energy": "NOT_MEASURED",
                "parity": measurement.get("weight_reconstruction"),
                "capability": "NOT_MEASURED_NO_COMPLETE_MODEL",
                "hardware": mps.get("device") or "Apple M3 Ultra component receipts",
                "source_revision": self.spec.revision,
                "validity_scope": f"component_only:{proposal.target.organ}",
                "candidate_id": record.get("candidate_id"),
                "schedule_version": proposal.schedule_version,
                "schedule_phase": proposal.schedule_phase,
                "organ": proposal.target.organ,
                "component_evidence": record.get("kernel_snapshot"),
                "claim_boundary": "does not report model TPS, TG10, TG3, or a complete executable model",
            }
        )

    def _write_knowledge(
        self, record: Mapping[str, Any], proposal: Proposal, *, passed_sensitive_gate: bool
    ) -> None:
        source = record.get("source") if isinstance(record.get("source"), Mapping) else {}
        artifact = record.get("artifact") if isinstance(record.get("artifact"), Mapping) else {}
        measurement = record.get("measurement") if isinstance(record.get("measurement"), Mapping) else {}
        quality = measurement.get("weight_reconstruction") if isinstance(measurement.get("weight_reconstruction"), Mapping) else {}
        functional = measurement.get("functional_probe") if isinstance(measurement.get("functional_probe"), Mapping) else {}
        routing = measurement.get("routing_probe") if isinstance(measurement.get("routing_probe"), Mapping) else {}
        mechanism_key = (
            f"{proposal.representation}:{proposal.target.organ}:{self.spec.model_family}:"
            f"{_sha256(dict(proposal.config))[:16]}"
        )
        representation_row = seal(
            {
                "schema": MECHANISM_SCHEMA,
                "record_id": f"representation:{record['candidate_id']}",
                "recorded_at": _utc_now(),
                "tensor_or_organ": source.get("tensor_name"),
                "model_family": self.spec.model_family,
                "geometry": source.get("region"),
                "weight_statistics": source.get("weight_statistics"),
                "activation_statistics": functional.get("activation_statistics"),
                "routing_behavior": routing,
                "representation": proposal.representation,
                "schedule": {
                    "version": proposal.schedule_version,
                    "phase": proposal.schedule_phase,
                    "expansion_start_generation": proposal.schedule_start_generation,
                    "expansion_boundary_sequence": proposal.schedule_boundary_sequence,
                },
                "precision": artifact.get("codec"),
                "codebook_or_basis": proposal.config,
                "residual": proposal.representation in {"binary_outlier_residual", "additive_residual_codebook_q2x2"},
                "doctor_qat": record.get("training"),
                "complete_bpw_delta": {"component_physical_bpw": artifact.get("physical_bpw"), "full_model_bpw": None},
                "capability_delta": "NOT_MEASURED_NO_COMPLETE_NATIVE_MODEL",
                "runtime_delta": "NOT_MEASURED_NO_COMPLETE_NATIVE_MODEL",
                "kernel_requirements": record.get("kernel_snapshot"),
                "failure_modes": [] if passed_sensitive_gate else ["component_sensitive_organ_gate_failed"],
                "reopen_conditions": "new representation/configuration or full-model parity evidence",
                "mechanism_key": mechanism_key,
                "candidate_id": record.get("candidate_id"),
                "evidence_binding": {"candidate_seal_sha256": record.get("seal_sha256"), "artifact_sha256": artifact.get("sha256")},
                "measurement": {"weight": quality, "functional": functional},
                "claim_boundary": "knowledge transfer prior only; not cross-model qualification",
            }
        )
        _append_jsonl_once(self.shared_mechanisms_path, representation_row, record_id=str(representation_row["record_id"]))
        self._index_knowledge_record(representation_row, kind="representation")
        kernel_row = self._kernel_record(record, proposal)
        _append_jsonl_once(self.shared_kernel_path, kernel_row, record_id=str(kernel_row["record_id"]))
        self._index_knowledge_record(kernel_row, kind="kernel")
        scheduler_row = self._scheduler_record(record, proposal)
        _append_jsonl_once(self.shared_scheduler_path, scheduler_row, record_id=str(scheduler_row["record_id"]))
        self._index_knowledge_record(scheduler_row, kind="scheduler")
        if not passed_sensitive_gate:
            negative = seal(
                {
                    "schema": NEGATIVE_SCHEMA,
                    "record_id": f"negative:{record['candidate_id']}",
                    "recorded_at": _utc_now(),
                    "status": "BURIED",
                    "mechanism": proposal.representation,
                    "mechanism_key": mechanism_key,
                    "model_geometry": f"{self.spec.model_family}:{proposal.target.organ}:{source.get('region', {}).get('shape')}",
                    "measured_outcome": {"weight": quality, "functional": functional, "physical_bpw": artifact.get("physical_bpw")},
                    "failure_reason": "component_sensitive_organ_gate_failed",
                    "reopen_condition": "different genome/configuration, materially different calibration, or direct full-model parity evidence",
                    "evidence_binding": {"candidate_path": str(self.candidate_dir / f"{record['candidate_id']}.json"), "candidate_seal_sha256": record.get("seal_sha256"), "artifact_sha256": artifact.get("sha256")},
                    "claim_boundary": "specific model/organ/configuration negative result; not a universal representation ban",
                }
            )
            _append_jsonl_once(self.shared_negative_path, negative, record_id=str(negative["record_id"]))
            _append_jsonl_once(self.local_negative_path, negative, record_id=str(negative["record_id"]))
            self._index_knowledge_record(negative, kind="negative_science")
        self._update_transfer_matrix(proposal, record, passed_sensitive_gate)

    def _update_transfer_matrix(self, proposal: Proposal, record: Mapping[str, Any], passed: bool) -> None:
        lock_path = FAMILY_ROOT / ".transfer-matrix.lock"
        with _locked(lock_path):
            existing = _read_json(self.transfer_path) or {}
            rows = existing.get("entries") if isinstance(existing.get("entries"), list) else []
            entry_id = f"{self.spec.key}:{proposal.representation}:{proposal.target.organ}"
            retained = [row for row in rows if not (isinstance(row, Mapping) and row.get("entry_id") == entry_id)]
            retained.append(
                {
                    "entry_id": entry_id,
                    "source_family": self.spec.model_family,
                    "target_family": "qwen_family_peer",
                    "mechanism": proposal.representation,
                    "transfer_status": "GENERATED_VARIANT" if passed else "RESEARCH_CANDIDATE",
                    "compatibility_conditions": [proposal.target.organ, "same representation/configuration class", "direct target-family validation required"],
                    "required_validation": ["exact target tensor parity", "complete-model runtime", "100 TPS custom kernel", "TG3"],
                    "negative_science_link": None if passed else f"negative:{record['candidate_id']}",
                    "candidate_id": record.get("candidate_id"),
                    "updated_at": _utc_now(),
                }
            )
            document = seal(
                {
                    "schema": TRANSFER_SCHEMA,
                    "status": "LIVE_COMPONENT_TRANSFER_INDEX",
                    "recorded_at": _utc_now(),
                    "entries": retained,
                    "claim_boundary": "transfer status does not waive direct target model gates",
                }
            )
            _atomic_json(self.transfer_path, document)

    def _commit_candidate(
        self,
        *,
        identity: Mapping[str, Any],
        proposal: Proposal,
        values: np.ndarray,
        source_region: Mapping[str, Any],
        source_shard: str,
        source_shard_sha256: str,
        source_revalidation: Mapping[str, Any],
        knowledge: Mapping[str, Any],
        parent_ids: Sequence[str],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        codec, training = _encode(
            values,
            proposal,
            gpu_lease=lambda: self._gpu_lease(stage="component_teacher_distillation"),
        )
        if len(codec.payload) > MAX_ARTIFACT_BYTES:
            raise DualGravityError(f"candidate payload exceeds protected bound: {len(codec.payload)} bytes")
        artifact_path = self.artifact_dir / f"{proposal.candidate_id}.gravity"
        _atomic_bytes(artifact_path, codec.payload)
        artifact_sha = _sha256_file(artifact_path)
        state_kv = self._state_kv_snapshot()
        reconstruction = _quality(values, codec.reconstruction)
        functional = _functional_probe(values, codec.reconstruction, seed=int(_sha256(proposal.candidate_id + ":functional")[:16], 16))
        routing = (
            _routing_probe(
                values,
                codec.reconstruction,
                top_k=self.spec.top_k,
                seed=int(_sha256(proposal.candidate_id + ":router")[:16], 16),
            )
            if proposal.target.organ == "moe_router"
            else {"status": "NOT_APPLICABLE"}
        )
        mps = _mps_component_probe(
            codec.reconstruction,
            seed=int(_sha256(proposal.candidate_id + ":mps")[:16], 16),
            gpu_lease=lambda: self._gpu_lease(stage="mps_component_smoke"),
        )
        sensitive = self._sensitive_gate(proposal.target, functional)
        record_path = self.candidate_dir / f"{proposal.candidate_id}.json"
        record = seal(
            {
                "schema": CANDIDATE_SCHEMA,
                "status": "PHYSICAL_COMPONENT_CANDIDATE_MEASURED_NOT_FULL_MODEL",
                "recorded_at": _utc_now(),
                "candidate_id": proposal.candidate_id,
                "source_content_identity_sha256": identity["content_identity_sha256"],
                "genome": {
                    "sequence": proposal.sequence,
                    "generation": proposal.generation,
                    "schedule": {
                        "version": proposal.schedule_version,
                        "phase": proposal.schedule_phase,
                        "expansion_start_generation": proposal.schedule_start_generation,
                        "expansion_boundary_sequence": proposal.schedule_boundary_sequence,
                        "claim_boundary": "v1 positions are frozen; v2 adds component representations only",
                    },
                    "parent_ids": list(parent_ids),
                    "representation": proposal.representation,
                    "configuration": dict(proposal.config),
                    "state_kv_format": state_kv,
                    "kernel_grammar": "source-bound packed component; direct Metal binding required before full promotion",
                    "residency_prefetch_policy": "CPU artifact packing with exclusive GPU component lease; complete-model residency unimplemented",
                    "deterministic_genome_sha256": _sha256(
                        {
                            "candidate_id": proposal.candidate_id,
                            "config": proposal.config,
                            "schedule_version": proposal.schedule_version,
                            "schedule_boundary_sequence": proposal.schedule_boundary_sequence,
                        }
                    ),
                    "worker_code_sha256": _sha256_file(Path(__file__)),
                },
                "source": {
                    "repository": self.spec.repository,
                    "revision": self.spec.revision,
                    "tensor_name": proposal.target.name,
                    "organ": proposal.target.organ,
                    "sensitive_organ": proposal.target.sensitive,
                    "source_shard": source_shard,
                    "source_shard_sha256_from_identity": source_shard_sha256,
                    "current_source_revalidation": dict(source_revalidation),
                    "region": dict(source_region),
                    "weight_statistics": _distribution(values),
                    "region_value_sha256": _sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()),
                },
                "representation": {
                    "family": proposal.representation,
                    "codec_metadata": codec.metadata,
                    "claim_boundary": "artifact represents only the declared source tensor region",
                },
                "artifact": {
                    "path": str(artifact_path),
                    "sha256": artifact_sha,
                    "bytes": artifact_path.stat().st_size,
                    "physical_bpw": artifact_path.stat().st_size * 8.0 / max(values.size, 1),
                    "codec": codec.metadata.get("schema"),
                },
                "measurement": {
                    "weight_reconstruction": reconstruction,
                    "functional_probe": functional,
                    "routing_probe": routing,
                    "mps_component_probe": mps,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                "training": training,
                "sensitive_organ_gate": sensitive,
                "knowledge_preflight": dict(knowledge),
                "kernel_snapshot": self._kernel_snapshot(),
                "state_machine": {
                    "immutable_source_content_identity": "VERIFIED",
                    "current_source_revalidation": {
                        "status": source_revalidation.get("status"),
                        "receipt_seal_sha256": source_revalidation.get("receipt_seal_sha256"),
                        "target_shard": source_revalidation.get("target_shard"),
                    },
                    "deterministic_genome_proposal": "COMMITTED",
                    "physical_byte_account": "COMMITTED",
                    "representative_tensor_parity": "RAN",
                    "sensitive_organ_gate": "PASSED" if sensitive["passes"] else "FAILED_COMPONENT_ONLY",
                    "mini_generation": "BLOCKED_NATIVE_COMPLETE_DECODER_UNIMPLEMENTED",
                    "kv_state": state_kv,
                    "complete_pack": self._complete_pack_snapshot(),
                    "complete_generation": "BLOCKED_NATIVE_COMPLETE_DECODER_UNIMPLEMENTED",
                    "native_metal": "COMPONENT_EVIDENCE_ONLY",
                    "complete_token_profile": "BLOCKED_NATIVE_COMPLETE_DECODER_UNIMPLEMENTED",
                    "capability": "BLOCKED_NO_COMPLETE_MODEL",
                    "runtime_tg": "BLOCKED_100_TPS_AND_TG3_REQUIRE_COMPLETE_TOKEN_RUNTIME",
                },
                "claim_boundary": {
                    "raw_bf16_body_is_teacher_only": True,
                    "candidate_is_not_complete_model_artifact": True,
                    "candidate_is_not_native_runtime": True,
                    "candidate_is_not_100_tps_or_tg3": True,
                    "candidate_is_not_hcli_or_capability_qualified": True,
                    "candidate_cannot_select_tournament_winner": True,
                },
            }
        )
        _atomic_json(record_path, record)
        return record

    def _commit_failure(
        self,
        *,
        identity: Mapping[str, Any],
        proposal: Proposal,
        knowledge: Mapping[str, Any],
        error: Exception,
        hard_block: bool,
    ) -> None:
        """Seal a failed experiment rather than letting an error vanish into logs."""

        record = seal(
            {
                "schema": "hawking.ascension.gravity_candidate_failure.v1",
                "status": "HARD_SOURCE_BLOCK" if hard_block else "FAILED_EXPERIMENT_ADVANCED",
                "recorded_at": _utc_now(),
                "candidate_id": proposal.candidate_id,
                "source_content_identity_sha256": identity["content_identity_sha256"],
                "genome": {
                    "sequence": proposal.sequence,
                    "generation": proposal.generation,
                    "schedule": {
                        "version": proposal.schedule_version,
                        "phase": proposal.schedule_phase,
                        "expansion_start_generation": proposal.schedule_start_generation,
                        "expansion_boundary_sequence": proposal.schedule_boundary_sequence,
                    },
                    "representation": proposal.representation,
                    "configuration": dict(proposal.config),
                    "tensor": proposal.target.name,
                    "organ": proposal.target.organ,
                },
                "knowledge_preflight": dict(knowledge),
                "failure": {"type": type(error).__name__, "message": str(error), "hard_block": hard_block},
                "reopen_condition": "repair source identity for hard provenance block; otherwise materially different genome or fixed codec/kernel premise",
                "claim_boundary": "failed component experiment only; no candidate capability/runtime conclusion",
            }
        )
        _atomic_json(self.failure_dir / f"{proposal.candidate_id}.json", record)
        mechanism_key = (
            f"{proposal.representation}:{proposal.target.organ}:{self.spec.model_family}:"
            f"{_sha256(dict(proposal.config))[:16]}"
        )
        negative = seal(
            {
                "schema": NEGATIVE_SCHEMA,
                "record_id": f"negative-failure:{proposal.candidate_id}",
                "recorded_at": _utc_now(),
                "status": "BURIED",
                "mechanism": proposal.representation,
                "mechanism_key": mechanism_key,
                "model_geometry": f"{self.spec.model_family}:{proposal.target.organ}",
                "measured_outcome": {"error": type(error).__name__},
                "failure_reason": str(error),
                "reopen_condition": record["reopen_condition"],
                "evidence_binding": {"failure_record_path": str(self.failure_dir / f"{proposal.candidate_id}.json"), "failure_record_seal_sha256": record["seal_sha256"]},
                "claim_boundary": "specific failed premise; no universal representation conclusion",
            }
        )
        _append_jsonl_once(self.shared_negative_path, negative, record_id=str(negative["record_id"]))
        _append_jsonl_once(self.local_negative_path, negative, record_id=str(negative["record_id"]))
        self._index_knowledge_record(negative, kind="negative_science")

    def _integrity_step(self, state: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"status": "NO_ARTIFACTS_YET"}
        cursor = int(state.get("integrity_audit_cursor", 0)) % len(rows)
        row = rows[cursor]
        artifact = Path(str(row.get("artifact_path", "")))
        actual = _sha256_file(artifact) if artifact.is_file() else None
        state["integrity_audit_cursor"] = cursor + 1
        return {
            "status": "VERIFIED_ONE_DURABLE_ARTIFACT" if actual == row.get("artifact_sha256") else "ARTIFACT_INTEGRITY_FAILURE",
            "candidate_id": row.get("candidate_id"),
            "expected_sha256": row.get("artifact_sha256"),
            "actual_sha256": actual,
        }

    def _publish(
        self,
        state: Mapping[str, Any],
        identity: Mapping[str, Any],
        *,
        phase: str,
        current: Mapping[str, Any] | None,
        next_proposal: Proposal | None,
        rows: Sequence[Mapping[str, Any]],
        frontier: Mapping[str, Any] | None = None,
        champions: Mapping[str, Any] | None = None,
        integrity: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> None:
        complete = self._complete_pack_snapshot()
        state_kv = self._state_kv_snapshot()
        runtime = self._runtime_snapshot()
        status = {
            "schema": SCHEMA,
            "status": "REAL_DETERMINISTIC_EVOLUTION_ADVANCING" if phase == "EVOLVING_PHYSICAL_CANDIDATE" else "BLOCKED_OR_RECOVERING",
            "recorded_at": _utc_now(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "model": {
                "key": self.spec.key,
                "id": self.spec.model_id,
                "architecture": self.spec.architecture,
                "expert_topology": {"experts": self.spec.expert_count, "top_k": self.spec.top_k},
                "attention_topology": {"query_heads": self.spec.query_heads, "kv_heads": self.spec.kv_heads, "head_dim": self.spec.head_dim},
            },
            "phase": phase,
            "heartbeat": state.get("heartbeat"),
            "last_material_progress_at": state.get("last_material_progress_at"),
            "source_content_identity": {"path": str(self.identity_path), "sha256": identity.get("content_identity_sha256"), "seal_sha256": identity.get("seal_sha256")},
            "population": {
                "candidate_count": len(rows),
                "completed_candidate_count": state.get("completed_candidate_count"),
                "failed_candidate_count": state.get("failed_candidate_count"),
                "artifact_bytes": sum(int(row.get("artifact_bytes", 0)) for row in rows),
                "no_automatic_artifact_eviction": True,
                "minimum_free_bytes_guard": MIN_FREE_BYTES,
                "current_free_bytes": _free_bytes(self.spec.root),
            },
            "current_experiment": dict(current) if current else None,
            "next_experiment": (
                {
                    "candidate_id": next_proposal.candidate_id,
                    "sequence": next_proposal.sequence,
                    "generation": next_proposal.generation,
                    "schedule_version": next_proposal.schedule_version,
                    "schedule_phase": next_proposal.schedule_phase,
                    "schedule_start_generation": next_proposal.schedule_start_generation,
                    "schedule_boundary_sequence": next_proposal.schedule_boundary_sequence,
                    "tensor": next_proposal.target.name,
                    "organ": next_proposal.target.organ,
                    "representation": next_proposal.representation,
                    "configuration": dict(next_proposal.config),
                }
                if next_proposal
                else None
            ),
            "pareto_frontier": {"path": str(self.frontier_path), "seal_sha256": frontier.get("seal_sha256") if frontier else None},
            "champions": {
                "path": str(self.champions_path),
                "seal_sha256": champions.get("seal_sha256") if champions else None,
                "current_lowest_bpw_component": champions.get("current_lowest_bpw_champion") if champions else None,
                "current_capable": champions.get("current_capable_champion") if champions else None,
                "current_fastest_component": champions.get("current_fastest_component_champion") if champions else None,
            },
            "knowledge_plane": {
                "representation_ledger": str(self.shared_mechanisms_path),
                "negative_science_ledger": str(self.shared_negative_path),
                "transfer_matrix": str(self.transfer_path),
                "negative_science_checked_before_proposal": True,
            },
            "complete_pack": complete,
            "state_kv": state_kv,
            "runtime": runtime,
            "resource_request": {
                "cpu": "source-bound artifact pack / component evaluation",
                "gpu": "exclusive short MPS lease only when a candidate runs training or smoke",
                "disk": "append-only candidate evidence with protected free-space guard",
                "network": "none; both source bodies are locally verified",
            },
            "kernel_and_runtime": self._kernel_snapshot(),
            "artifact_integrity": dict(integrity) if integrity else None,
            "error": dict(error) if error else None,
            "claim_boundary": {
                "raw_source_is_teacher_only": True,
                "no_full_model_capability_or_runtime_claim": True,
                "100_tps_custom_kernel_gate_remains_unearned": True,
                "tg3_333_tps_gate_remains_unearned": True,
                "tournament_remains_protected_and_fail_closed": True,
            },
        }
        _atomic_json(self.status_path, status)
        # The legacy path is the public physical-lane compatibility view.  It
        # intentionally replaces a misleading refresh-only status with the
        # same truthful live state instead of retaining two competing workers.
        _atomic_json(self.spec.legacy_status, status)
        self._update_dashboard(status)

    def _update_dashboard(self, worker_status: Mapping[str, Any]) -> None:
        lock_path = LIFECYCLE_ROOT / ".ascension-dual-physical-status.lock"
        with _locked(lock_path):
            existing = _read_json(DASHBOARD_PATH) or {}
            workers = existing.get("workers") if isinstance(existing.get("workers"), Mapping) else {}
            workers = dict(workers)
            complete = worker_status.get("complete_pack") if isinstance(worker_status.get("complete_pack"), Mapping) else {}
            champions = worker_status.get("champions") if isinstance(worker_status.get("champions"), Mapping) else {}
            runtime = worker_status.get("runtime") if isinstance(worker_status.get("runtime"), Mapping) else {}
            current = worker_status.get("current_experiment") if isinstance(worker_status.get("current_experiment"), Mapping) else {}
            population = worker_status.get("population") if isinstance(worker_status.get("population"), Mapping) else {}
            model = worker_status.get("model") if isinstance(worker_status.get("model"), Mapping) else {}
            workers[self.spec.key] = {
                "status_path": str(self.status_path),
                "status": worker_status.get("status"),
                "phase": worker_status.get("phase"),
                "pid": worker_status.get("pid"),
                "ppid": worker_status.get("ppid"),
                "heartbeat": worker_status.get("heartbeat"),
                "source_revision": self.spec.revision,
                "source_model": model.get("id"),
                "candidate_number": population.get("completed_candidate_count"),
                "last_material_progress_at": worker_status.get("last_material_progress_at"),
                "population": population,
                "current_experiment": current or None,
                "current_representation": current.get("representation"),
                "last_completed_experiment": current or None,
                "next_experiment": worker_status.get("next_experiment"),
                "champions": champions,
                "best_complete_bpw": {
                    "value": complete.get("complete_physical_bpw"),
                    "status": complete.get("manifest_status") or complete.get("phase"),
                    "claim_boundary": "null until a verified all-tensor manifest exists",
                },
                "best_capable_bpw": {
                    "value": None,
                    "status": "UNSET_BLOCKED_NO_COMPLETE_CAPABILITY_EVIDENCE",
                },
                "current_base_true_tps": runtime.get("current_base_true_tps"),
                "tg_rung": runtime.get("tg_rung") or "BLOCKED_NO_COMPLETE_NATIVE_TOKEN_RUNTIME",
                "current_kernel_bottleneck": runtime.get("current_kernel_bottleneck"),
                "complete_pack": complete,
                "state_kv": worker_status.get("state_kv"),
                "runtime": runtime,
                "kernel_and_runtime": worker_status.get("kernel_and_runtime"),
            }
            # The legacy controller is retained as an archival compatibility
            # view.  The live physical campaign instead has its own non-V3
            # gatekeeper, which reads these source-bound lanes and cannot
            # launch/select a tournament winner.  Prefer that live workflow
            # in the dashboard so callers do not mistake a stale V3 seed
            # archive for the physical campaign's actual state.
            legacy_tournament_path = LIFECYCLE_ROOT / "ASCENSION_MANAGER_TOURNAMENT_WORKFLOW.json"
            legacy_tournament = _read_json(legacy_tournament_path) or {}
            physical_lifecycle_root = PHYSICAL_ROOT / "lifecycle"
            physical_tournament_path = physical_lifecycle_root / "ASCENSION_PHYSICAL_TOURNAMENT_WORKFLOW.json"
            physical_gate_path = physical_lifecycle_root / "ASCENSION_PHYSICAL_TOURNAMENT_GATE_STATUS.json"
            physical_tournament = _read_json(physical_tournament_path) or {}
            physical_gate = _read_json(physical_gate_path) or {}
            qwen80_acquisition = _read_json(PHYSICAL_ROOT / "qwen80-acquisition/QWEN80_ACQUISITION_STATUS.json") or {}
            gpu_lease = _read_json(GPU_LEASE_STATUS_PATH) or {"status": "NO_LEASE_RECORD_YET"}
            document = {
                "schema": "hawking.ascension.dual_manager_physical_status.v1",
                "status": "DUAL_PHYSICAL_EVOLUTION_ACTIVE" if any(row.get("status") == "REAL_DETERMINISTIC_EVOLUTION_ADVANCING" for row in workers.values() if isinstance(row, Mapping)) else "DUAL_PHYSICAL_EVOLUTION_AWAITING_WORKER",
                "recorded_at": _utc_now(),
                "protected_authority": "non-V3 physical gatekeeper and protected final review; workers may not promote, launch, or select",
                "workers": workers,
                "knowledge_plane": {
                    "representation_ledger": str(self.shared_mechanisms_path),
                    "kernel_ledger": str(self.shared_kernel_path),
                    "scheduler_ledger": str(self.shared_scheduler_path),
                    "negative_science_ledger": str(self.shared_negative_path),
                    "transfer_matrix": str(self.transfer_path),
                    "mechanism_index": str(self.mechanism_index_path),
                    "representation_record_count": len(_jsonl(self.shared_mechanisms_path)),
                    "kernel_record_count": len(_jsonl(self.shared_kernel_path)),
                    "scheduler_record_count": len(_jsonl(self.shared_scheduler_path)),
                    "negative_record_count": len(_jsonl(self.shared_negative_path)),
                },
                "global_resources": {
                    "gpu_owner": gpu_lease,
                    "cpu_jobs": [
                        {"worker": key, "lane": "gravity_evolution", "pid": row.get("pid"), "phase": row.get("phase")}
                        for key, row in workers.items()
                        if isinstance(row, Mapping)
                    ] + [
                        {
                            "worker": key,
                            "lane": "complete_gravity_pack",
                            "pid": row.get("complete_pack", {}).get("pid") if isinstance(row.get("complete_pack"), Mapping) else None,
                            "phase": row.get("complete_pack", {}).get("phase") if isinstance(row.get("complete_pack"), Mapping) else None,
                        }
                        for key, row in workers.items()
                        if isinstance(row, Mapping)
                    ] + [
                        {
                            "worker": key,
                            "lane": "native_runtime_watchdog",
                            "pid": row.get("runtime", {}).get("pid") if isinstance(row.get("runtime"), Mapping) else None,
                            "phase": row.get("runtime", {}).get("complete_runtime_phase") if isinstance(row.get("runtime"), Mapping) else None,
                        }
                        for key, row in workers.items()
                        if isinstance(row, Mapping)
                    ],
                    "network_job": {
                        "qwen80_acquisition_phase": qwen80_acquisition.get("phase"),
                        "pid": qwen80_acquisition.get("pid"),
                        "claim": "no active transfer is assumed unless acquisition status says otherwise",
                    },
                    "free_disk_bytes": _free_bytes(PHYSICAL_ROOT),
                    "minimum_reserved_free_bytes": MIN_FREE_BYTES,
                    "host": _system_resources(),
                },
                "tournament": {
                    "workflow_path": str(physical_tournament_path),
                    "gate_status_path": str(physical_gate_path),
                    "runtime_phase": physical_tournament.get("runtime_phase"),
                    "status": physical_tournament.get("status"),
                    "gate_status": physical_gate.get("status"),
                    "ready_for_final_review": physical_gate.get("ready_for_final_review"),
                    "admission": "BLOCKED until exact native full-token runtime earns 100 TPS custom-kernel gate, TG3 333 TPS, and protected review",
                    "legacy_controller": {
                        "path": str(legacy_tournament_path),
                        "runtime_phase": legacy_tournament.get("runtime_phase"),
                        "status": legacy_tournament.get("status"),
                        "scope": "archival compatibility only; not the live physical campaign authority",
                    },
                },
                "claim_boundary": "live workers are component/representation research, not tournament entrants",
            }
            _atomic_json(DASHBOARD_PATH, document)

    def run_cycle(self) -> None:
        weight_map = load_weight_map(self.spec.source_dir)
        identity = self._source_identity(weight_map)
        state = self._state(identity)
        rows = self._load_index(identity)
        targets = self._targets(weight_map)
        proposal = self._proposal(state, targets, identity)
        known_ids = {str(row.get("candidate_id")) for row in rows}
        current = {
            "candidate_id": proposal.candidate_id,
            "sequence": proposal.sequence,
            "generation": proposal.generation,
            "schedule_version": proposal.schedule_version,
            "schedule_phase": proposal.schedule_phase,
            "schedule_start_generation": proposal.schedule_start_generation,
            "schedule_boundary_sequence": proposal.schedule_boundary_sequence,
            "tensor": proposal.target.name,
            "organ": proposal.target.organ,
            "representation": proposal.representation,
            "configuration": dict(proposal.config),
        }
        frontier, champions = self._frontier_and_champions(identity, rows)
        _atomic_json(self.frontier_path, frontier)
        _atomic_json(self.champions_path, champions)
        if proposal.candidate_id in known_ids:
            state["next_proposal_index"] = proposal.sequence + 1
            state["heartbeat"] = int(state.get("heartbeat", 0)) + 1
            state["updated_at"] = _utc_now()
            state["completed_candidate_count"] = len(rows)
            _atomic_json(self.state_path, state)
            next_proposal = self._proposal(state, targets, identity)
            self._publish(state, identity, phase="RECOVERED_ALREADY_COMMITTED_MUTATION", current=current, next_proposal=next_proposal, rows=rows, frontier=frontier, champions=champions)
            return
        if _free_bytes(self.spec.root) < MIN_FREE_BYTES:
            state["heartbeat"] = int(state.get("heartbeat", 0)) + 1
            state["updated_at"] = _utc_now()
            _atomic_json(self.state_path, state)
            self._publish(state, identity, phase="SPACE_GUARD_BLOCKED_NO_EVIDENCE_EVICTED", current=current, next_proposal=proposal, rows=rows, frontier=frontier, champions=champions, error={"reason": "free_disk_below_protected_reserve"})
            return
        knowledge = self._knowledge_snapshot(proposal)
        if not knowledge["may_proceed"]:
            # This is a durable negative-science decision, not a false active
            # heartbeat.  Advance to the next materially different genome.
            state["next_proposal_index"] = proposal.sequence + 1
            state["heartbeat"] = int(state.get("heartbeat", 0)) + 1
            state["updated_at"] = _utc_now()
            _atomic_json(self.state_path, state)
            next_proposal = self._proposal(state, targets, identity)
            self._publish(state, identity, phase="NEGATIVE_SCIENCE_SKIP_ADVANCED_TO_NEXT_GENOME", current=current, next_proposal=next_proposal, rows=rows, frontier=frontier, champions=champions, error={"reason": "active_matching_negative_science", "knowledge_preflight": knowledge})
            return
        self._publish(state, identity, phase="EVOLVING_PHYSICAL_CANDIDATE", current=current, next_proposal=None, rows=rows, frontier=frontier, champions=champions)
        try:
            source_shard = weight_map.get(proposal.target.name)
            if not isinstance(source_shard, str) or not source_shard:
                raise DualGravityError(f"source safetensors index has no shard for target tensor {proposal.target.name}")
            source_revalidation = self._current_source_revalidation(
                identity,
                weight_map,
                target_shard=source_shard,
            )
            values = load_tensor(self.spec.source_dir, weight_map, proposal.target.name)
            self._assert_revalidated_target_unchanged(source_revalidation)
            region, region_meta = self._crop_region(values, proposal=proposal)
            parent_ids = [
                str(row["candidate_id"])
                for row in rows
                if row.get("tensor_name") == proposal.target.name
                and row.get("representation") == proposal.representation
                and isinstance(row.get("generation"), int)
                and int(row["generation"]) < proposal.generation
            ]
            parent_ids = parent_ids[-2:]
            record = self._commit_candidate(
                identity=identity,
                proposal=proposal,
                values=region,
                source_region=region_meta,
                source_shard=source_shard,
                source_shard_sha256=str(source_revalidation["target_shard_sha256"]),
                source_revalidation=source_revalidation,
                knowledge=knowledge,
                parent_ids=parent_ids,
            )
            record_path = self.candidate_dir / f"{proposal.candidate_id}.json"
            rows = [*rows, self._index_row(record, record_path)]
            self._write_index(identity, rows)
            passed = bool(record.get("sensitive_organ_gate", {}).get("passes"))
            self._write_knowledge(record, proposal, passed_sensitive_gate=passed)
            state["next_proposal_index"] = proposal.sequence + 1
            state["heartbeat"] = int(state.get("heartbeat", 0)) + 1
            state["completed_candidate_count"] = len(rows)
            state["material_progress_count"] = int(state.get("material_progress_count", 0)) + 1
            state["last_material_progress_at"] = _utc_now()
            state["updated_at"] = _utc_now()
            integrity = self._integrity_step(state, rows)
            _atomic_json(self.state_path, state)
            frontier, champions = self._frontier_and_champions(identity, rows)
            _atomic_json(self.frontier_path, frontier)
            _atomic_json(self.champions_path, champions)
            next_proposal = self._proposal(state, targets, identity)
            self._publish(state, identity, phase="EVOLVING_PHYSICAL_CANDIDATE", current={**current, "result": record.get("status"), "sensitive_gate_pass": passed}, next_proposal=next_proposal, rows=rows, frontier=frontier, champions=champions, integrity=integrity)
        except Exception as exc:
            # Preserve a deterministic failure event, then move on only for an
            # experiment-specific error.  Source/identity corruption is held
            # as a hard block so we never paper over provenance failure.
            hard = isinstance(exc, DualGravityError) and ("source" in str(exc).lower() or "identity" in str(exc).lower())
            self._commit_failure(
                identity=identity,
                proposal=proposal,
                knowledge=knowledge,
                error=exc,
                hard_block=hard,
            )
            state["heartbeat"] = int(state.get("heartbeat", 0)) + 1
            state["updated_at"] = _utc_now()
            state["failed_candidate_count"] = int(state.get("failed_candidate_count", 0)) + 1
            if not hard:
                state["next_proposal_index"] = proposal.sequence + 1
            _atomic_json(self.state_path, state)
            next_proposal = None if hard else self._proposal(state, targets, identity)
            self._publish(state, identity, phase="HARD_SOURCE_BLOCK" if hard else "CANDIDATE_FAILURE_ADVANCED_TO_NEXT_GENOME", current=current, next_proposal=next_proposal, rows=rows, frontier=frontier, champions=champions, error={"type": type(exc).__name__, "message": str(exc), "hard_block": hard})

    def watch(self, *, idle_seconds: float) -> int:
        if idle_seconds <= 0:
            raise DualGravityError("idle seconds must be positive")

        def stop(_signal: int, _frame: Any) -> None:
            self._stopping = True

        old_term = signal.signal(signal.SIGTERM, stop)
        old_int = signal.signal(signal.SIGINT, stop)
        try:
            while not self._stopping:
                self.run_cycle()
                if not self._stopping:
                    time.sleep(idle_seconds)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(SPECS))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("once", help="commit one deterministic physical candidate")
    watch = commands.add_parser("watch", help="run the detached durable worker")
    watch.add_argument("--idle-seconds", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worker = DualGravityWorker(SPECS[args.model])
    if args.command == "once":
        worker.run_cycle()
        return 0
    return worker.watch(idle_seconds=float(args.idle_seconds))


__all__ = ["DualGravityError", "DualGravityWorker", "ModelSpec", "SPECS", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
