"""Pack Qwen30 as uniform Qn group-G for Lane-N bisection arms.

Supports:
  * bits=3, group=128 → nominal 3.125 bpw (3 + 16/128)
  * bits=2, group=128 → nominal 2.125 bpw (2 + 16/128)

Physical layout (flat elements):
  header HQ30UQn\\0 (n in {'2','3'}), version 1, group_size G
  FP16 scales (max_abs/bound per group)
  bit-packed little-endian codes (n bits per weight), groups padded

Metal must decode the same bit packing. Diagnostic candidate only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators import ascension_qwen30_complete_gravity as complete
from lab.receipts import seal


MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
MODEL_DIR = MAIN_HAWKING / (
    "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
)
QWEN30_ROOT = MAIN_HAWKING / (
    "workspace/campaign/records/ascension-sandbox/physical/qwen30"
)
SOURCE_AUDIT = QWEN30_ROOT / "QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json"
BASELINE_REVALIDATION = (
    QWEN30_ROOT / "complete-gravity" / "QWEN30_CURRENT_SOURCE_SHARD_REVALIDATION.json"
)
VERSION = 1
TERMINAL_SCHEMA = "hawking.ascension.complete_binary_terminal_status.v1"
EXPECTED_TENSOR_COUNT = 18_867


class UniformQnError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _config(bits: int, group_size: int) -> dict[str, Any]:
    if bits not in (2, 3):
        raise UniformQnError("only bits=2 or 3 supported for Lane-N bisection")
    if group_size != 128:
        raise UniformQnError("Lane-N bisection uses group_size=128")
    magic = f"HQ30UQ{bits}\0".encode("ascii")
    if len(magic) != 8:
        raise UniformQnError("magic length")
    return {
        "bits": bits,
        "group_size": group_size,
        "bound": (1 << (bits - 1)) - 1,  # 1 for q2, 3 for q3
        "magic": magic,
        "ext": f".hq30uq{bits}",
        "schema": f"hawking.ascension.qwen30_uniform_q{bits}_group{group_size}_candidate.v1",
        "status": f"CANDIDATE_UNIFORM_Q{bits}_GROUP{group_size}_DIAGNOSTIC_UNQUALIFIED",
        "phase": f"EARNED_COMPLETE_PHYSICAL_UNIFORM_Q{bits}_CANDIDATE_UNQUALIFIED",
        "branch_id": f"qwen30-uniform-q{bits}-group{group_size}-v1",
        "artifact_prefix": f"QWEN30_UNIFORM_Q{bits}_GROUP{group_size}_V1",
        "model_id": f"Qwen3-Coder-30B-A3B-Instruct-uniform-q{bits}-group{group_size}-v1",
        "family": f"uniform_q{bits}_group{group_size}_fp16_scale",
        "nominal_bpw": bits + 16.0 / group_size,
        "root": QWEN30_ROOT
        / "quality-candidates"
        / f"uniform-q{bits}-group{group_size}-v1",
    }


def _artifact_name(tensor_name: str, ext: str) -> str:
    return hashlib.sha256(tensor_name.encode("utf-8")).hexdigest() + ext


def _payload_bytes(shape: Sequence[int], *, bits: int, group_size: int) -> int:
    elements = math.prod(int(d) for d in shape)
    groups = (elements + group_size - 1) // group_size
    padded = groups * group_size
    code_bits = padded * bits
    code_bytes = (code_bits + 7) // 8
    return 32 + 4 * len(shape) + 2 * groups + code_bytes


def _pack_bits_le(codes_u8: np.ndarray, bits: int) -> bytes:
    """Little-endian bit packing: bit 0 of value 0 is LSB of byte 0."""
    flat = np.ascontiguousarray(codes_u8, dtype=np.uint8).reshape(-1)
    bit_matrix = ((flat[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little").tobytes()


def _pack_uniform_qn(
    values: np.ndarray,
    shape: Sequence[int],
    *,
    bits: int,
    group_size: int,
    magic: bytes,
) -> tuple[bytes, dict[str, Any]]:
    bound = (1 << (bits - 1)) - 1
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    total = int(flat.size)
    groups = (total + group_size - 1) // group_size
    padded = np.pad(flat, (0, groups * group_size - total), constant_values=0.0).reshape(
        groups, group_size
    )
    if not np.isfinite(padded).all():
        raise UniformQnError("non-finite source")
    max_abs = np.max(np.abs(padded), axis=1).astype(np.float32)
    scales_f16 = (max_abs / float(max(bound, 1))).astype("<f2")
    scales = scales_f16.astype(np.float32)
    denom = np.where(scales > 0.0, scales, 1.0)
    q = np.rint(padded / denom[:, None]).clip(-bound, bound).astype(np.int16)
    # Offset so codes are unsigned 0..2*bound
    codes_u8 = (q + bound).astype(np.uint8)
    code_bytes = _pack_bits_le(codes_u8.reshape(-1), bits)
    dimensions = tuple(int(item) for item in shape)
    header = struct.pack(
        "<8sIIHHQI",
        magic,
        VERSION,
        group_size,
        len(dimensions),
        0,
        total,
        0,
    )
    header += struct.pack("<" + "I" * len(dimensions), *dimensions)
    payload = header + scales_f16.tobytes() + code_bytes
    expected = _payload_bytes(dimensions, bits=bits, group_size=group_size)
    if len(payload) != expected:
        raise UniformQnError(f"payload {len(payload)} != expected {expected}")
    reconstructed = (q.astype(np.float32) * scales[:, None]).reshape(-1)[:total]
    original_norm = max(float(np.linalg.norm(flat)), 1e-12)
    recon_norm = max(float(np.linalg.norm(reconstructed)), 1e-12)
    metrics = {
        "relative_l2": float(np.linalg.norm(flat - reconstructed) / original_norm),
        "cosine": float(np.dot(flat, reconstructed) / (original_norm * recon_norm)),
        "rmse": float(np.sqrt(np.mean(np.square(flat - reconstructed)))),
        "finite": True,
    }
    return payload, metrics


def _pack_one_job(job: dict[str, Any]) -> dict[str, Any]:
    bits = int(job["bits"])
    group_size = int(job["group_size"])
    magic = bytes(job["magic"])
    ext = job["ext"]
    model_dir = Path(job["model_dir"])
    tensor_dir = Path(job["tensor_dir"])
    tensor_name = job["tensor_name"]
    shard = job["shard"]
    source_hash = job["source_hash"]
    dtype = job["dtype"]
    shape = [int(x) for x in job["shape"]]
    begin, end = int(job["begin"]), int(job["end"])
    destination = tensor_dir / _artifact_name(tensor_name, ext)
    expected_size = _payload_bytes(shape, bits=bits, group_size=group_size)
    if destination.is_file() and destination.stat().st_size == expected_size:
        payload = destination.read_bytes()
        if payload[:8] == magic and len(payload) == expected_size:
            return {
                "tensor_name": tensor_name,
                "source_shard": shard,
                "source_shard_sha256": source_hash,
                "source_dtype": dtype,
                "shape": shape,
                "elements": math.prod(shape),
                "artifact_path": str(destination),
                "artifact_bytes": len(payload),
                "artifact_sha256": _sha256_bytes(payload),
                "layout": {
                    "magic": magic.decode("ascii"),
                    "version": VERSION,
                    "group_size": group_size,
                    "bits": bits,
                    "code_packing": "little_endian_bit_stream",
                    "scale_dtype": "float16",
                    "scale_rule": f"max_abs_div_{ (1<<(bits-1))-1 }_fp16_authority",
                },
                "component_quality": {"cosine": None, "resumed": True, "finite": True},
                "resumed": True,
            }
    source_path = model_dir / shard
    with source_path.open("rb") as handle:
        header_bytes = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_bytes + begin)
        raw = handle.read(end - begin)
    values = complete._values_from_raw(raw, dtype, shape)
    payload, quality = _pack_uniform_qn(
        values, shape, bits=bits, group_size=group_size, magic=magic
    )
    del values
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=tensor_dir)
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
        "elements": math.prod(shape),
        "artifact_path": str(destination),
        "artifact_bytes": len(payload),
        "artifact_sha256": _sha256_bytes(payload),
        "layout": {
            "magic": magic.decode("ascii"),
            "version": VERSION,
            "group_size": group_size,
            "bits": bits,
            "code_packing": "little_endian_bit_stream",
            "scale_dtype": "float16",
            "scale_rule": f"max_abs_div_{(1<<(bits-1))-1}_fp16_authority",
        },
        "component_quality": quality,
        "resumed": False,
    }


def _load_source_index(model_dir: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = {str(k): str(v) for k, v in index["weight_map"].items()}
    meta: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(weight_map.values())):
        path = model_dir / shard
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in weight_map:
                meta[name] = {
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                    "data_offsets": info["data_offsets"],
                    "shard": shard,
                }
    if len(meta) != len(weight_map):
        raise UniformQnError("source index/meta mismatch")
    return weight_map, meta


def build(*, bits: int, group_size: int, workers: int) -> Path:
    cfg = _config(bits, group_size)
    root: Path = cfg["root"]
    tensor_dir = root / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    progress_path = root / f"{cfg['artifact_prefix']}_PROGRESS.jsonl"
    status_path = root / f"{cfg['artifact_prefix']}_STATUS.json"
    manifest_path = root / f"{cfg['artifact_prefix']}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    terminal_path = root / f"{cfg['artifact_prefix']}_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json"

    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    revalidation = json.loads(BASELINE_REVALIDATION.read_text(encoding="utf-8"))
    source_revision = revalidation["source_revision"]
    weight_map, meta = _load_source_index(MODEL_DIR)
    if len(weight_map) != EXPECTED_TENSOR_COUNT:
        raise UniformQnError(f"tensor count {len(weight_map)}")
    body = audit.get("source") if isinstance(audit.get("source"), Mapping) else audit
    shards = body.get("shards") if isinstance(body, Mapping) else None
    if not isinstance(shards, Mapping):
        raise UniformQnError("audit shards missing")
    sealed_hashes: dict[str, str] = {}
    for shard in sorted(set(weight_map.values())):
        row = shards.get(shard)
        sealed_hashes[shard] = row["sha256"] if isinstance(row, Mapping) else str(row)

    done: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done[str(row["tensor_name"])] = row

    ordered_names = sorted(weight_map.keys())
    jobs = []
    for name in ordered_names:
        if name in done and Path(done[name]["artifact_path"]).is_file():
            continue
        info = meta[name]
        begin, end = info["data_offsets"]
        jobs.append(
            {
                "model_dir": str(MODEL_DIR),
                "tensor_dir": str(tensor_dir),
                "tensor_name": name,
                "shard": info["shard"],
                "source_hash": sealed_hashes[info["shard"]],
                "dtype": info["dtype"],
                "shape": info["shape"],
                "begin": begin,
                "end": end,
                "bits": bits,
                "group_size": group_size,
                "magic": list(cfg["magic"]),
                "ext": cfg["ext"],
            }
        )

    t0 = time.time()
    with progress_path.open("a", encoding="utf-8") as progress_handle:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_pack_one_job, job): job["tensor_name"] for job in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                done[row["tensor_name"]] = row
                progress_handle.write(json.dumps(row, sort_keys=True) + "\n")
                progress_handle.flush()
                if i % 500 == 0:
                    _atomic_json(
                        status_path,
                        {
                            "schema": cfg["schema"],
                            "phase": f"PACKING_UNIFORM_Q{bits}",
                            "done": len(done),
                            "elapsed_s": time.time() - t0,
                        },
                    )

    ordered = [done[n] for n in ordered_names]
    quality_rows = [
        r for r in ordered if isinstance(r.get("component_quality", {}).get("cosine"), float)
    ]
    mean_cos = float(np.mean([r["component_quality"]["cosine"] for r in quality_rows])) if quality_rows else float("nan")
    mean_l2 = (
        float(np.mean([r["component_quality"].get("relative_l2", 0.0) for r in quality_rows]))
        if quality_rows
        else float("nan")
    )
    payload_bytes = sum(int(r["artifact_bytes"]) for r in ordered)
    elements = sum(int(r["elements"]) for r in ordered)
    draft_tensors = [
        {
            "tensor_name": r["tensor_name"],
            "source_shard": r["source_shard"],
            "source_shard_sha256": r["source_shard_sha256"],
            "source_dtype": r["source_dtype"],
            "shape": r["shape"],
            "elements": r["elements"],
            "artifact_path": r["artifact_path"],
            "artifact_bytes": r["artifact_bytes"],
            "artifact_sha256": r["artifact_sha256"],
            "layout": r["layout"],
            "component_quality": r["component_quality"],
        }
        for r in ordered
    ]
    reval_path = str(BASELINE_REVALIDATION)
    reval_seal = revalidation["seal_sha256"]

    def make_manifest(manifest_bytes_estimate: int) -> dict[str, Any]:
        all_bytes = payload_bytes + manifest_bytes_estimate
        complete_bpw = (all_bytes * 8.0) / float(elements)
        return {
            "schema": cfg["schema"],
            "status": cfg["status"],
            "recorded_at": _utc_now(),
            "branch_id": cfg["branch_id"],
            "model_id": cfg["model_id"],
            "artifact_prefix": cfg["artifact_prefix"],
            "representation": {
                "family": cfg["family"],
                "group_size": group_size,
                "bits_per_weight": bits,
                "nominal_bpw": cfg["nominal_bpw"],
                "physical_direct_layout": True,
                "training": "none",
                "diagnostic_label": f"Lane-N bisection arm uniform q{bits} group-{group_size}",
            },
            "source": {
                "model_dir": str(MODEL_DIR),
                "repository": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "tensor_count": len(ordered),
            },
            "source_body_audit_seal_sha256": audit["seal_sha256"],
            "source_revalidation_receipt_path": reval_path,
            "source_revalidation_receipt_seal_sha256": reval_seal,
            "complete_physical_bpw_ledger": {
                "all_required_weight_artifact_bytes": all_bytes,
                "complete_physical_bpw": complete_bpw,
                "explicitly_excluded_separate_state": [
                    "KV_cache_bytes",
                    "Qwen80_recurrent_state_bytes",
                    "Context_OS_cache_bytes",
                    "Agent_OS_memory_bytes",
                ],
                "manifest_bytes_billed": manifest_bytes_estimate,
                "passes_storage_threshold": complete_bpw <= 1.5,
                "source_weight_elements": elements,
                "tensor_payload_bytes": payload_bytes,
                "threshold_bpw": 1.5,
                "nominal_codec_bpw": cfg["nominal_bpw"],
            },
            "quality_summary": {
                "mean_component_cosine": mean_cos,
                "mean_component_relative_l2": mean_l2,
                "quality_rows_with_cosine": len(quality_rows),
                "verdict": f"DIAGNOSTIC_UNIFORM_Q{bits}_GROUP{group_size}_COHERENCE_PROBE",
            },
            "claim_boundary": {
                "complete_physical_tensor_coverage_is_true": True,
                "diagnostic_uniform_qn_not_production_promotion": True,
                "weights_remain_packed_at_token_time": True,
                "no_fp16_reconstruction_of_matrix_bodies": True,
                "not_tg_or_tournament_qualification": True,
            },
            "champion_classes": {
                "current_bpw_champion": {
                    "candidate": cfg["model_id"],
                    "complete_physical_bpw": complete_bpw,
                    "status": "CANDIDATE_ONLY",
                }
            },
            "tensors": draft_tensors,
        }

    est = 22_000_000
    for _ in range(4):
        draft = make_manifest(est)
        sealed = seal(draft)
        size = len(json.dumps(sealed, indent=2, sort_keys=True, ensure_ascii=False).encode()) + 1
        if size == est:
            break
        est = size
    else:
        sealed = seal(make_manifest(est))
    _atomic_json(manifest_path, sealed)

    terminal = seal(
        {
            "schema": TERMINAL_SCHEMA,
            "status": cfg["phase"],
            "recorded_at": _utc_now(),
            "binding": {
                "model_id": cfg["model_id"],
                "artifact_prefix": cfg["artifact_prefix"],
                "manifest_schema": cfg["schema"],
                "branch_id": cfg["branch_id"],
                "source_body_audit_seal_sha256": audit["seal_sha256"],
                "source_revalidation_receipt_path": reval_path,
                "source_revalidation_receipt_seal_sha256": reval_seal,
                "source_revision": source_revision,
            },
            "candidate": {
                "manifest_path": str(manifest_path),
                "manifest_seal_sha256": sealed["seal_sha256"],
                "all_required_weight_artifact_bytes": sealed["complete_physical_bpw_ledger"][
                    "all_required_weight_artifact_bytes"
                ],
                "complete_physical_bpw": sealed["complete_physical_bpw_ledger"][
                    "complete_physical_bpw"
                ],
                "tensor_count": len(ordered),
                "mean_component_cosine": mean_cos,
            },
        }
    )
    _atomic_json(terminal_path, terminal)
    _atomic_json(
        status_path,
        {
            "schema": cfg["schema"],
            "phase": cfg["phase"],
            "manifest_seal_sha256": sealed["seal_sha256"],
            "complete_physical_bpw": sealed["complete_physical_bpw_ledger"]["complete_physical_bpw"],
            "mean_component_cosine": mean_cos,
            "elapsed_s": time.time() - t0,
        },
    )
    return manifest_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, required=True, choices=[2, 3])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    path = build(bits=args.bits, group_size=args.group_size, workers=args.workers)
    print(json.dumps({"manifest_path": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
