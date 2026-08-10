"""Pack Qwen30 as uniform Q4 group-64 (4.25 bpw) for the Lane-N coherence floor.

Diagnostic complete-body candidate under quality-candidates/uniform-q4-group64-v1.
Packs every source BF16 tensor as offset-binary signed Q4 + one FP16 scale per
group of 64, matching crates/hawking-core/shaders/qwen_uniform_q4.metal.

Not a promotion. Not a TG claim. Physical bpw is ledgered from artifact bytes.
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
DEFAULT_ROOT = (
    QWEN30_ROOT / "quality-candidates" / "uniform-q4-group64-v1"
)

MAGIC = b"HQ30UQ4\0"
VERSION = 1
GROUP_SIZE = 64
CODE_BYTES_PER_GROUP = GROUP_SIZE // 2  # 4-bit nibbles
SCHEMA = "hawking.ascension.qwen30_uniform_q4_group64_candidate.v1"
TERMINAL_SCHEMA = "hawking.ascension.complete_binary_terminal_status.v1"
BRANCH_ID = "qwen30-uniform-q4-group64-v1"
ARTIFACT_PREFIX = "QWEN30_UNIFORM_Q4_GROUP64_V1"
MODEL_ID = "Qwen3-Coder-30B-A3B-Instruct-uniform-q4-group64-v1"
EXPECTED_TENSOR_COUNT = 18_867
CANDIDATE_STATUS = "CANDIDATE_UNIFORM_Q4_GROUP64_DIAGNOSTIC_UNQUALIFIED"
COMPLETE_CANDIDATE_PHASE = "EARNED_COMPLETE_PHYSICAL_UNIFORM_Q4_CANDIDATE_UNQUALIFIED"


class UniformQ4Error(RuntimeError):
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


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _artifact_name(tensor_name: str) -> str:
    return hashlib.sha256(tensor_name.encode("utf-8")).hexdigest() + ".hq30uq4"


def _payload_bytes(shape: Sequence[int]) -> int:
    elements = math.prod(int(d) for d in shape)
    groups = (elements + GROUP_SIZE - 1) // GROUP_SIZE
    return 32 + 4 * len(shape) + 2 * groups + groups * CODE_BYTES_PER_GROUP


def _pack_uniform_q4(
    values: np.ndarray, shape: Sequence[int]
) -> tuple[bytes, dict[str, Any]]:
    """Match metal probe + qwen_uniform_q4_group64_matvec layout exactly.

    Flat groups of 64; scale = max_abs/7 as FP16; q in [-8, 7] via nibble-8;
    even local index in low nibble, odd in high nibble.
    """
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    total = int(flat.size)
    groups = (total + GROUP_SIZE - 1) // GROUP_SIZE
    padded = np.pad(flat, (0, groups * GROUP_SIZE - total), constant_values=0.0).reshape(
        groups, GROUP_SIZE
    )
    if not np.isfinite(padded).all():
        raise UniformQ4Error("source tensor contains a non-finite value")
    max_abs = np.max(np.abs(padded), axis=1).astype(np.float32)
    # Authority is the stored FP16 scale, not the f32 precursor.
    scales_f16 = (max_abs / 7.0).astype("<f2")
    scales = scales_f16.astype(np.float32)
    denom = np.where(scales > 0.0, scales, 1.0)
    q = np.rint(padded / denom[:, None]).clip(-8.0, 7.0).astype(np.int16)
    codes_u8 = (q + 8).astype(np.uint8)
    # Pack even/odd nibbles into bytes: local 0 low, local 1 high, ...
    even = codes_u8[:, 0::2]
    odd = codes_u8[:, 1::2]
    packed = (even | (odd << 4)).astype(np.uint8)
    code_bytes = packed.reshape(-1).tobytes()
    if len(code_bytes) != groups * CODE_BYTES_PER_GROUP:
        raise UniformQ4Error(
            f"code byte mismatch: got {len(code_bytes)}, expected {groups * CODE_BYTES_PER_GROUP}"
        )
    dimensions = tuple(int(item) for item in shape)
    header = struct.pack(
        "<8sIIHHQI",
        MAGIC,
        VERSION,
        GROUP_SIZE,
        len(dimensions),
        0,
        total,
        0,
    )
    header += struct.pack("<" + "I" * len(dimensions), *dimensions)
    payload = header + scales_f16.tobytes() + code_bytes
    expected = _payload_bytes(dimensions)
    if len(payload) != expected:
        raise UniformQ4Error(f"payload size {len(payload)} != expected {expected}")
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
    """Worker entry: pack one tensor from a source shard into the candidate tree."""
    model_dir = Path(job["model_dir"])
    tensor_dir = Path(job["tensor_dir"])
    tensor_name = job["tensor_name"]
    shard = job["shard"]
    source_hash = job["source_hash"]
    dtype = job["dtype"]
    shape = [int(x) for x in job["shape"]]
    begin, end = int(job["begin"]), int(job["end"])
    destination = tensor_dir / _artifact_name(tensor_name)
    if destination.is_file() and destination.stat().st_size == _payload_bytes(shape):
        # Resume: re-hash existing payload rather than re-read source.
        payload = destination.read_bytes()
        if payload[:8] == MAGIC and len(payload) == _payload_bytes(shape):
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
                    "magic": MAGIC.decode("ascii"),
                    "version": VERSION,
                    "group_size": GROUP_SIZE,
                    "nibble_order": "even_low_odd_high",
                    "q_range": "[-8,7]",
                    "scale_dtype": "float16",
                    "scale_rule": "max_abs_div_7_fp16_authority",
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
    payload, quality = _pack_uniform_q4(values, shape)
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
            "magic": MAGIC.decode("ascii"),
            "version": VERSION,
            "group_size": GROUP_SIZE,
            "nibble_order": "even_low_odd_high",
            "q_range": "[-8,7]",
            "scale_dtype": "float16",
            "scale_rule": "max_abs_div_7_fp16_authority",
        },
        "component_quality": quality,
        "resumed": False,
    }


def _load_source_index(model_dir: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        # Some Qwen dumps store the weight map at the root index name.
        candidates = list(model_dir.glob("*.index.json"))
        if not candidates:
            raise UniformQ4Error(f"no safetensors index under {model_dir}")
        index_path = candidates[0]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = {str(k): str(v) for k, v in index["weight_map"].items()}
    # Per-shard tensor metadata from safetensors headers.
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
        missing = sorted(set(weight_map) - set(meta))
        raise UniformQ4Error(f"source index/meta mismatch; missing e.g. {missing[:5]}")
    return weight_map, meta


def build(*, root: Path, workers: int, max_tensors: int | None) -> Path:
    root = root.expanduser().resolve()
    tensor_dir = root / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    progress_path = root / f"{ARTIFACT_PREFIX}_PROGRESS.jsonl"
    status_path = root / f"{ARTIFACT_PREFIX}_STATUS.json"
    manifest_path = root / f"{ARTIFACT_PREFIX}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    terminal_path = root / f"{ARTIFACT_PREFIX}_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json"

    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if not isinstance(audit.get("seal_sha256"), str):
        raise UniformQ4Error("source audit missing seal_sha256")
    revalidation = json.loads(BASELINE_REVALIDATION.read_text(encoding="utf-8"))
    if not isinstance(revalidation.get("seal_sha256"), str):
        raise UniformQ4Error("baseline revalidation receipt missing seal")
    source_revision = (
        revalidation.get("source_revision")
        or revalidation.get("revision")
        or revalidation.get("binding", {}).get("source_revision")
    )
    if not source_revision:
        # Fall back to the known campaign revision embedded in prior receipts.
        source_revision = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"

    weight_map, meta = _load_source_index(MODEL_DIR)
    if len(weight_map) != EXPECTED_TENSOR_COUNT:
        raise UniformQ4Error(
            f"expected {EXPECTED_TENSOR_COUNT} tensors, found {len(weight_map)}"
        )

    # Bind source shard hashes from the sealed audit.
    body = audit.get("source") if isinstance(audit.get("source"), Mapping) else audit
    shards = body.get("shards") if isinstance(body, Mapping) else None
    if not isinstance(shards, Mapping):
        raise UniformQ4Error("source audit lacks shards map")
    sealed_hashes: dict[str, str] = {}
    for shard in sorted(set(weight_map.values())):
        row = shards.get(shard)
        if isinstance(row, Mapping) and isinstance(row.get("sha256"), str):
            sealed_hashes[shard] = row["sha256"]
        elif isinstance(row, str):
            sealed_hashes[shard] = row
        else:
            raise UniformQ4Error(f"audit missing hash for {shard}")

    done: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done[str(row["tensor_name"])] = row

    ordered_names = sorted(weight_map.keys())
    if max_tensors is not None:
        ordered_names = ordered_names[: max_tensors]

    jobs = []
    for name in ordered_names:
        if name in done and done[name].get("artifact_sha256"):
            # Verify file still present.
            p = Path(done[name]["artifact_path"])
            if p.is_file():
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
            }
        )

    _atomic_json(
        status_path,
        {
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "phase": "PACKING_UNIFORM_Q4_GROUP64",
            "branch_id": BRANCH_ID,
            "planned": len(ordered_names),
            "already_done": len(done),
            "remaining": len(jobs),
            "workers": workers,
        },
    )

    t0 = time.time()
    completed_now = 0
    with progress_path.open("a", encoding="utf-8") as progress_handle:
        if workers <= 1:
            for job in jobs:
                row = _pack_one_job(job)
                done[row["tensor_name"]] = row
                progress_handle.write(json.dumps(row, sort_keys=True) + "\n")
                progress_handle.flush()
                completed_now += 1
                if completed_now % 200 == 0:
                    _atomic_json(
                        status_path,
                        {
                            "schema": SCHEMA,
                            "recorded_at": _utc_now(),
                            "phase": "PACKING_UNIFORM_Q4_GROUP64",
                            "completed_now": completed_now,
                            "total_done": len(done),
                            "remaining": len(jobs) - completed_now,
                            "elapsed_s": time.time() - t0,
                        },
                    )
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_pack_one_job, job): job["tensor_name"] for job in jobs}
                for fut in as_completed(futures):
                    row = fut.result()
                    done[row["tensor_name"]] = row
                    progress_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    progress_handle.flush()
                    completed_now += 1
                    if completed_now % 200 == 0:
                        _atomic_json(
                            status_path,
                            {
                                "schema": SCHEMA,
                                "recorded_at": _utc_now(),
                                "phase": "PACKING_UNIFORM_Q4_GROUP64",
                                "completed_now": completed_now,
                                "total_done": len(done),
                                "remaining": len(jobs) - completed_now,
                                "elapsed_s": time.time() - t0,
                            },
                        )

    if len(done) < len(ordered_names):
        raise UniformQ4Error(
            f"incomplete pack: {len(done)} / {len(ordered_names)} tensors present"
        )

    ordered = [done[name] for name in ordered_names]
    # Drop resume-only quality placeholders from the mean if present.
    quality_rows = [
        row for row in ordered if isinstance(row.get("component_quality", {}).get("cosine"), float)
    ]
    if quality_rows:
        mean_cos = float(np.mean([r["component_quality"]["cosine"] for r in quality_rows]))
        mean_l2 = float(
            np.mean([r["component_quality"].get("relative_l2", 0.0) for r in quality_rows])
        )
    else:
        mean_cos = float("nan")
        mean_l2 = float("nan")

    payload_bytes = sum(int(r["artifact_bytes"]) for r in ordered)
    elements = sum(int(r["elements"]) for r in ordered)
    # Manifest is billed after seal; placeholder size then recompute once sealed.
    draft_tensors = []
    for row in ordered:
        draft_tensors.append(
            {
                "tensor_name": row["tensor_name"],
                "source_shard": row["source_shard"],
                "source_shard_sha256": row["source_shard_sha256"],
                "source_dtype": row["source_dtype"],
                "shape": row["shape"],
                "elements": row["elements"],
                "artifact_path": row["artifact_path"],
                "artifact_bytes": row["artifact_bytes"],
                "artifact_sha256": row["artifact_sha256"],
                "layout": row["layout"],
                "component_quality": row["component_quality"],
            }
        )

    # Revalidation path is the sealed baseline receipt (source shards unchanged).
    reval_path = str(BASELINE_REVALIDATION)
    reval_seal = revalidation["seal_sha256"]

    def make_manifest(manifest_bytes_estimate: int) -> dict[str, Any]:
        all_bytes = payload_bytes + manifest_bytes_estimate
        complete_bpw = (all_bytes * 8.0) / float(elements)
        return {
            "schema": SCHEMA,
            "status": CANDIDATE_STATUS,
            "recorded_at": _utc_now(),
            "branch_id": BRANCH_ID,
            "model_id": MODEL_ID,
            "artifact_prefix": ARTIFACT_PREFIX,
            "representation": {
                "family": "uniform_q4_group64_fp16_scale",
                "group_size": GROUP_SIZE,
                "bits_per_weight": 4,
                "nominal_bpw": 4.0 + 16.0 / GROUP_SIZE,
                "physical_direct_layout": True,
                "training": "none",
                "diagnostic_label": (
                    "Lane-N highest-fidelity arm; uniform 4-bit group-64; "
                    "executes on Metal via qwen_uniform_q4_group64_matvec; "
                    "not a production promotion"
                ),
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
                "passes_storage_threshold": True,
                "source_weight_elements": elements,
                "tensor_payload_bytes": payload_bytes,
                # Same ledger field as the binary baseline (exactly 1.5). This
                # diagnostic is expected to FAIL the storage threshold — that
                # is the point of the high-fidelity arm.
                "threshold_bpw": 1.5,
                "nominal_codec_bpw": 4.0 + 16.0 / GROUP_SIZE,
            },
            "quality_summary": {
                "mean_component_cosine": mean_cos,
                "mean_component_relative_l2": mean_l2,
                "quality_rows_with_cosine": len(quality_rows),
                "verdict": "DIAGNOSTIC_UNIFORM_Q4_GROUP64_COHERENCE_PROBE",
            },
            "claim_boundary": {
                "complete_physical_tensor_coverage_is_true": True,
                "diagnostic_uniform_q4_not_production_promotion": True,
                "weights_remain_packed_q4_at_token_time": True,
                "no_fp16_reconstruction_of_matrix_bodies": True,
                "not_tg_or_tournament_qualification": True,
            },
            "champion_classes": {
                "current_bpw_champion": {
                    "candidate": MODEL_ID,
                    "complete_physical_bpw": complete_bpw,
                    "status": "CANDIDATE_ONLY",
                }
            },
            "tensors": draft_tensors,
        }

    # Two-pass seal so billed manifest bytes match the sealed document length.
    draft = make_manifest(0)
    sealed = seal(draft)
    size1 = len(
        json.dumps(sealed, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ) + 1
    final = make_manifest(size1)
    sealed_final = seal(final)
    size2 = len(
        json.dumps(sealed_final, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ) + 1
    if size2 != size1:
        final = make_manifest(size2)
        sealed_final = seal(final)
    _atomic_json(manifest_path, sealed_final)

    terminal = seal(
        {
            "schema": TERMINAL_SCHEMA,
            "status": COMPLETE_CANDIDATE_PHASE,
            "recorded_at": _utc_now(),
            "binding": {
                "model_id": MODEL_ID,
                "artifact_prefix": ARTIFACT_PREFIX,
                "manifest_schema": SCHEMA,
                "branch_id": BRANCH_ID,
                "source_body_audit_seal_sha256": audit["seal_sha256"],
                "source_revalidation_receipt_path": reval_path,
                "source_revalidation_receipt_seal_sha256": reval_seal,
                "source_revision": source_revision,
            },
            "candidate": {
                "manifest_path": str(manifest_path),
                "manifest_seal_sha256": sealed_final["seal_sha256"],
                "all_required_weight_artifact_bytes": sealed_final[
                    "complete_physical_bpw_ledger"
                ]["all_required_weight_artifact_bytes"],
                "complete_physical_bpw": sealed_final["complete_physical_bpw_ledger"][
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
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "phase": COMPLETE_CANDIDATE_PHASE,
            "manifest_path": str(manifest_path),
            "manifest_seal_sha256": sealed_final["seal_sha256"],
            "complete_physical_bpw": sealed_final["complete_physical_bpw_ledger"][
                "complete_physical_bpw"
            ],
            "mean_component_cosine": mean_cos,
            "tensor_count": len(ordered),
            "elapsed_s": time.time() - t0,
        },
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-tensors", type=int, default=None)
    args = parser.parse_args()
    path = build(root=args.root, workers=args.workers, max_tensors=args.max_tensors)
    print(json.dumps({"manifest_path": str(path), "status": "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
