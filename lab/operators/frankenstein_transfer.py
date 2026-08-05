#!/usr/bin/env python3.12
"""Training-free GLM→DeepSeek math transfer (closed-form linear algebra only).

Hard constraint: NO gradient descent, NO optimizer steps, NO loss-minimization
loops, NO learned adapters.  The prior ``frankenstein_fusion_op.loss_target``
fit path is excluded.

Weight-only pipeline (needs raw GLM shards, no GLM runtime, no student forward):
  1. Stream math-relevant weight tensors (expert/mlp gate|up|down) from the GLM
     donor.
  2. Accumulate a hidden-space Gram matrix and take its top-r eigenspace
     (SVD/PCA) → GLM math subspace basis B ∈ R^{6144 × r}.
  3. Closed-form projection W_proj ∈ R^{6144 × 4096} via orthonormal embedding
     of the subspace into student width (pseudo-inverse / truncated identity
     Procrustes without paired activations).
  4. Steering vector = top singular direction projected into student space.
  5. Router bias = fixed additive scores from per-expert subspace energy.
  6. Seal a reversible residual module in bridge-contract shape; apply is pure
     residual addition (subtract to reverse).

Validation of capability effect is gated on ``DEEPSEEK_FORWARD_PENDING``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from lab.operators.frankenstein_fusion_op import (
    BRIDGE_DTYPE,
    BRIDGE_INPUT_SHAPE,
    BRIDGE_OUTPUT_SHAPE,
    DEEPSEEK_V4_FLASH,
    FORWARD_GATE,
    GLM_5_2,
    TRANSPLANT_POINT_NAMES,
    layer_map,
)
from lab.receipts import seal


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace"
CAMPAIGN_ROOT = WORKSPACE_ROOT / "campaign"
EVIDENCE_ROOT = CAMPAIGN_ROOT / "evidence" / "models" / "frankenstein"
RUN_ROOT = CAMPAIGN_ROOT / "records" / "runs" / "frankenstein"

# Durable donor (human-owned stream; this module reads, never deletes source shards).
DEFAULT_GLM_DONOR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/frankenstein/glm-donor"
)
DEFAULT_BODY_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/full-43-layer-stream.gravity"
)
DEFAULT_BRIDGE_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_LATENT_BRIDGE_CONTRACT.json"
)
DEFAULT_TRANSPLANT_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_TRANSPLANT_POINTS.json"
)

MIN_FREE_FLOOR_BYTES = 25 * 1024**3
DEFAULT_SUBSPACE_RANK = 64
DEFAULT_EXPERTS_PER_LAYER_FOR_GRAM = 32
MODULE_MAGIC = b"FRNKXFR1"
TRANSFER_MODULE_SCHEMA = "hawking.frankenstein.training_free_transfer_module.v1"
SUBSPACE_RECEIPT_SCHEMA = "hawking.frankenstein.glm_math_subspace.v1"
APPLY_RECEIPT_SCHEMA = "hawking.frankenstein.transfer_apply.v1"
RUN_RECEIPT_SCHEMA = "hawking.frankenstein.training_free_run.v1"

_MATH_TENSOR_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\."
    r"(?:"
    r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    r"|"
    r"(gate_proj|up_proj|down_proj)\.weight"  # dense early layers
    r"|"
    r"shared_experts\.(gate_proj|up_proj|down_proj)\.weight"
    r"|"
    r"gate\.weight"  # router
    r")$"
)


class FrankensteinTransferError(RuntimeError):
    """Training-free transfer failed closed."""


@dataclass
class ExtractionStats:
    tensors_seen: int = 0
    tensors_used_for_gram: int = 0
    tensors_scored_for_router: int = 0
    bytes_read: int = 0
    bytes_evicted_working_set: int = 0
    layers_touched: set[int] = field(default_factory=set)
    experts_touched: set[tuple[int, int]] = field(default_factory=set)
    dense_mlp_tensors: int = 0
    expert_proj_tensors: int = 0
    router_tensors: int = 0
    shards_opened: list[str] = field(default_factory=list)
    windows_evicted: list[str] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def free_bytes(path: Path) -> int:
    try:
        return int(shutil.disk_usage(path).free)
    except OSError as exc:
        raise FrankensteinTransferError(f"cannot measure free space at {path}: {exc}") from exc


def assert_floor(path: Path, *, need_extra: int = 0, label: str = "workspace") -> dict[str, Any]:
    free = free_bytes(path)
    required = MIN_FREE_FLOOR_BYTES + max(0, int(need_extra))
    if free < required:
        raise FrankensteinTransferError(
            f"{label} free-space floor violated: free={free} need>={required} "
            f"(floor={MIN_FREE_FLOOR_BYTES} + extra={need_extra})"
        )
    return {
        "path": str(path),
        "free_bytes": free,
        "floor_bytes": MIN_FREE_FLOOR_BYTES,
        "need_extra_bytes": int(need_extra),
        "required_bytes": required,
        "headroom_bytes": free - required,
        "status": "FLOOR_OK",
    }


def _ensure_dir(path: Path) -> None:
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise FrankensteinTransferError(f"not a safe directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def _atomic_create(path: Path, raw: bytes) -> str:
    if path.exists():
        existing = path.read_bytes()
        if existing != raw:
            raise FrankensteinTransferError(
                f"refusing to overwrite different immutable evidence: {path}"
            )
        return _sha256(existing)
    _ensure_dir(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


def _rm_tree(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "existed": False, "removed": False, "bytes_removed": 0}
    node = os.lstat(path)
    if stat.S_ISLNK(node.st_mode):
        raise FrankensteinTransferError(f"refusing to evict symlink: {path}")
    bytes_removed = 0
    if path.is_file():
        bytes_removed = path.stat().st_size
        path.unlink()
        return {
            "path": str(path),
            "existed": True,
            "removed": True,
            "bytes_removed": bytes_removed,
        }
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in filenames:
            fp = Path(dirpath) / name
            if fp.is_symlink():
                raise FrankensteinTransferError(f"refusing to evict symlink: {fp}")
            bytes_removed += fp.stat().st_size
            fp.unlink()
        for name in dirnames:
            dp = Path(dirpath) / name
            if dp.is_symlink():
                raise FrankensteinTransferError(f"refusing to evict symlink dir: {dp}")
            dp.rmdir()
    path.rmdir()
    return {
        "path": str(path),
        "existed": True,
        "removed": True,
        "bytes_removed": bytes_removed,
    }


def _bf16_u16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    """Exact BF16→F32 widen (high 16 bits of IEEE-754 float32)."""

    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16_u16(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even-ish BF16 store via truncation of float32 bits.

    Not used for training — only for sealing compact module payloads.
    """

    bits = values.astype(np.float32).view(np.uint32)
    # Round: add the high bit of the truncated half when not creating NaN.
    rounding = (bits >> 16 & 1) + 0x7FFF
    rounded = bits + rounding
    return (rounded >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# Safetensors streaming (stdlib only; no torch)
# ---------------------------------------------------------------------------


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise FrankensteinTransferError(f"short safetensors header length: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise FrankensteinTransferError(f"implausible header length {header_len}: {path}")
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise FrankensteinTransferError(f"short safetensors header body: {path}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrankensteinTransferError(f"invalid safetensors header JSON: {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise FrankensteinTransferError(f"safetensors header root is not an object: {path}")
    return header, 8 + header_len


def _load_weight_map(donor_dir: Path) -> dict[str, str]:
    index_path = donor_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FrankensteinTransferError(f"missing safetensors index: {index_path}")
    doc = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = doc.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise FrankensteinTransferError("weight_map missing or empty in index")
    return {str(k): str(v) for k, v in weight_map.items()}


def _parse_math_tensor(name: str) -> dict[str, Any] | None:
    match = _MATH_TENSOR_RE.match(name)
    if not match:
        return None
    layer = int(match.group(1))
    if match.group(2) is not None:
        return {
            "name": name,
            "layer": layer,
            "kind": "expert_proj",
            "expert": int(match.group(2)),
            "proj": match.group(3),
        }
    if match.group(4) is not None:
        return {
            "name": name,
            "layer": layer,
            "kind": "dense_mlp",
            "expert": None,
            "proj": match.group(4),
        }
    if match.group(5) is not None:
        return {
            "name": name,
            "layer": layer,
            "kind": "shared_expert",
            "expert": None,
            "proj": match.group(5),
        }
    # gate.weight router
    return {
        "name": name,
        "layer": layer,
        "kind": "router",
        "expert": None,
        "proj": "router",
    }


def iter_math_tensor_names(weight_map: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in weight_map:
        parsed = _parse_math_tensor(name)
        if parsed is not None:
            parsed["shard"] = weight_map[name]
            rows.append(parsed)
    rows.sort(key=lambda r: (r["layer"], r["kind"], r.get("expert") or -1, r["proj"], r["name"]))
    return rows


def _read_tensor_f32(shard_path: Path, header: Mapping[str, Any], data_offset: int, name: str) -> np.ndarray:
    meta = header.get(name)
    if not isinstance(meta, Mapping):
        raise FrankensteinTransferError(f"tensor {name!r} missing from {shard_path.name}")
    dtype = meta.get("dtype")
    shape = meta.get("shape")
    data_offsets = meta.get("data_offsets")
    if dtype != "BF16":
        raise FrankensteinTransferError(f"expected BF16 for {name}, got {dtype}")
    if not isinstance(shape, list) or not shape:
        raise FrankensteinTransferError(f"bad shape for {name}: {shape!r}")
    if not (isinstance(data_offsets, list) and len(data_offsets) == 2):
        raise FrankensteinTransferError(f"bad data_offsets for {name}")
    start, end = int(data_offsets[0]), int(data_offsets[1])
    nbytes = end - start
    expected = int(np.prod(shape)) * 2
    if nbytes != expected:
        raise FrankensteinTransferError(
            f"byte length mismatch for {name}: file={nbytes} expected={expected}"
        )
    with shard_path.open("rb") as handle:
        handle.seek(data_offset + start)
        raw = handle.read(nbytes)
    if len(raw) != nbytes:
        raise FrankensteinTransferError(f"short read for {name}: {len(raw)}/{nbytes}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return _bf16_u16_to_f32(u16).reshape(tuple(int(x) for x in shape))


def _hidden_gram_contribution(weight: np.ndarray, *, hidden: int) -> np.ndarray:
    """Return H×H Gram contribution for a weight living on the residual stream.

    - W shape [out, H] (gate_proj, up_proj, router): G = W.T @ W
    - W shape [H, in] (down_proj): G = W @ W.T
    - W shape [H] (unexpected): outer product

    Computed in float32 for throughput; caller accumulates then promotes for eigh.
    """

    arr = np.asarray(weight, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != hidden:
            raise FrankensteinTransferError(f"1d weight hidden mismatch: {arr.shape}")
        v = arr.reshape(hidden, 1)
        return v @ v.T
    if arr.ndim != 2:
        raise FrankensteinTransferError(f"expected rank-1/2 weight, got {arr.shape}")
    if arr.shape[1] == hidden:
        # [out, H]
        return arr.T @ arr
    if arr.shape[0] == hidden:
        # [H, in]
        return arr @ arr.T
    raise FrankensteinTransferError(
        f"weight shape {arr.shape} does not touch hidden={hidden}"
    )


def _select_for_gram(
    rows: Sequence[Mapping[str, Any]],
    *,
    experts_per_layer: int,
) -> list[dict[str, Any]]:
    """Bounded sample of math tensors for the Gram (disk/time safe)."""

    selected: list[dict[str, Any]] = []
    experts_by_layer: dict[int, set[int]] = {}
    for row in rows:
        kind = row["kind"]
        layer = int(row["layer"])
        if kind in {"dense_mlp", "shared_expert", "router"}:
            selected.append(dict(row))
            continue
        if kind == "expert_proj":
            expert = int(row["expert"])
            chosen = experts_by_layer.setdefault(layer, set())
            if expert in chosen:
                selected.append(dict(row))
                continue
            if len(chosen) < experts_per_layer:
                # Prefer a stride-ish cover: accept expert if it falls in the
                # first N by index (already sorted) — simple and deterministic.
                if expert < experts_per_layer or len(chosen) < experts_per_layer:
                    # Take experts 0..K-1 first; if missing, take next available.
                    if expert < experts_per_layer:
                        chosen.add(expert)
                        selected.append(dict(row))
                    elif len(chosen) < experts_per_layer:
                        chosen.add(expert)
                        selected.append(dict(row))
    return selected


def _orthonormal_embedding(student_h: int, rank: int, *, seed: int = 0) -> np.ndarray:
    """Fixed orthonormal E ∈ R^{H_s × r} (seeded QR of Gaussian; closed-form)."""

    if rank > student_h:
        raise FrankensteinTransferError(f"rank {rank} > student hidden {student_h}")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((student_h, rank), dtype=np.float64)
    q, _ = np.linalg.qr(raw, mode="reduced")
    return q[:, :rank]


def closed_form_projection(
    basis_glm: np.ndarray,
    *,
    student_hidden: int = int(DEEPSEEK_V4_FLASH["hidden_size"]),
    seed: int = 0,
) -> dict[str, Any]:
    """Map GLM subspace coords → student residual stream (no training).

    basis_glm: B ∈ R^{H_g × r} with orthonormal columns.
    student embedding E ∈ R^{H_s × r} orthonormal (fixed, seeded).
    W_proj = B @ E.T ∈ R^{H_g × H_s}
      so a_s = a_g @ W_proj = (a_g @ B) @ E.T

    This is the weight-only stand-in for orthogonal Procrustes when paired
    activations are unavailable.  When the forward lands, a true Procrustes
    between paired activations may replace E (still one SVD, not SGD).
    """

    b = np.asarray(basis_glm, dtype=np.float64)
    if b.ndim != 2:
        raise FrankensteinTransferError(f"basis must be 2d, got {b.shape}")
    h_g, rank = b.shape
    if h_g != int(GLM_5_2["hidden_size"]):
        raise FrankensteinTransferError(
            f"basis hidden {h_g} != GLM hidden {GLM_5_2['hidden_size']}"
        )
    e = _orthonormal_embedding(student_hidden, rank, seed=seed)
    w = b @ e.T  # [H_g, H_s]
    # Bias zero — residual identity without shift.
    bias = np.zeros(student_hidden, dtype=np.float64)
    return {
        "weight": w,
        "bias": bias,
        "student_embedding": e,
        "math": "a_s = a_g @ (B @ E.T) + 0; B=GLM math subspace, E=fixed orthonormal",
        "method": "closed_form_subspace_isometric_embedding",
        "procrustes_status": "UNPAIRED_WEIGHT_ONLY_STANDIN",
        "shapes": {
            "basis_glm": list(b.shape),
            "student_embedding": list(e.shape),
            "weight": list(w.shape),
            "bias": list(bias.shape),
        },
    }


def top_eigenspace(gram: np.ndarray, rank: int) -> dict[str, Any]:
    """Symmetric eigendecomposition; return top-r eigenvectors (SVD of Gram)."""

    g = np.asarray(gram, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise FrankensteinTransferError(f"gram must be square, got {g.shape}")
    # eigh: ascending eigenvalues
    evals, evecs = np.linalg.eigh(g)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    total = float(np.maximum(evals, 0.0).sum())
    r = min(int(rank), g.shape[0])
    top_vals = np.maximum(evals[:r], 0.0)
    top_vecs = evecs[:, :r]
    captured = float(top_vals.sum())
    energy_frac = (captured / total) if total > 0 else 0.0
    return {
        "basis": top_vecs,  # [H, r]
        "eigenvalues": top_vals,
        "rank": r,
        "total_energy": total,
        "captured_energy": captured,
        "energy_fraction": energy_frac,
        "all_eigenvalues_head": evals[: min(16, len(evals))].tolist(),
    }


def steering_from_subspace(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    student_embedding: np.ndarray,
) -> dict[str, Any]:
    """Fixed residual-stream steering vector in student space (weight-derived)."""

    b = np.asarray(basis, dtype=np.float64)
    ev = np.asarray(eigenvalues, dtype=np.float64)
    e = np.asarray(student_embedding, dtype=np.float64)
    # Math direction in GLM space: energy-weighted combination of top directions.
    weights = np.sqrt(np.maximum(ev, 0.0))
    if float(weights.sum()) <= 0:
        glm_dir = b[:, 0] if b.shape[1] else np.zeros(b.shape[0])
    else:
        weights = weights / weights.sum()
        glm_dir = b @ weights
    norm = float(np.linalg.norm(glm_dir))
    if norm > 0:
        glm_dir = glm_dir / norm
    # Project subspace coords of glm_dir into student space.
    coords = b.T @ glm_dir  # [r]
    student_dir = e @ coords  # [H_s]
    sn = float(np.linalg.norm(student_dir))
    if sn > 0:
        student_dir = student_dir / sn
    return {
        "glm_direction": glm_dir,
        "student_direction": student_dir,
        "scale_default": 0.05,  # small fixed residual scale; not trained
        "method": "energy_weighted_top_singular_direction",
    }


def router_bias_from_scores(
    expert_scores: Mapping[int, float],
    *,
    n_experts: int = int(DEEPSEEK_V4_FLASH["n_routed_experts"]),
) -> np.ndarray:
    """Fixed additive router bias: experts with higher math-subspace energy get push."""

    bias = np.zeros(n_experts, dtype=np.float64)
    if not expert_scores:
        return bias
    vals = np.array([float(expert_scores.get(i, 0.0)) for i in range(n_experts)], dtype=np.float64)
    present = vals[vals > 0]
    if present.size == 0:
        return bias
    mean = float(present.mean())
    std = float(present.std()) if present.size > 1 else 1.0
    if std < 1e-12:
        std = 1.0
    # z-score present experts; leave unscored at 0
    for i in range(n_experts):
        if i in expert_scores and expert_scores[i] > 0:
            bias[i] = (float(expert_scores[i]) - mean) / std
    # Bound to keep gate logits from exploding (fixed clip, not trained).
    return np.clip(bias, -2.0, 2.0)


def residual_lowrank_from_steering(
    student_dir: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Rank-1 residual operator A = scale * (v v^T); apply as a @ A.

    Reversible: subtract the same residual.
    """

    v = np.asarray(student_dir, dtype=np.float64).reshape(-1)
    return float(scale) * np.outer(v, v)


# ---------------------------------------------------------------------------
# Extraction driver
# ---------------------------------------------------------------------------


def extract_glm_math_subspace(
    *,
    donor_dir: Path,
    working_set_dir: Path,
    rank: int = DEFAULT_SUBSPACE_RANK,
    experts_per_layer: int = DEFAULT_EXPERTS_PER_LAYER_FOR_GRAM,
    max_tensors: int | None = None,
    floor_path: Path | None = None,
) -> dict[str, Any]:
    """Stream donor math weights → Gram → top-r subspace.  Evict working windows.

    Does **not** delete the human-owned donor directory; only our working-set
    scratch under ``working_set_dir`` is created and evicted.
    """

    donor_dir = Path(donor_dir)
    if not donor_dir.is_dir():
        raise FrankensteinTransferError(f"donor dir missing: {donor_dir}")
    floor_path = floor_path or donor_dir
    floor_before = assert_floor(floor_path, label="pre-extract")

    weight_map = _load_weight_map(donor_dir)
    all_math = iter_math_tensor_names(weight_map)
    # Only tensors whose shards are present on disk.
    present_math = [
        row
        for row in all_math
        if (donor_dir / row["shard"]).is_file()
    ]
    if not present_math:
        raise FrankensteinTransferError(
            "no math-relevant tensors found on present donor shards"
        )

    gram_candidates = _select_for_gram(present_math, experts_per_layer=experts_per_layer)
    if max_tensors is not None:
        gram_candidates = gram_candidates[: max(0, int(max_tensors))]

    hidden = int(GLM_5_2["hidden_size"])
    # float32 accumulate (~150 MiB); promote once before eigh.
    gram = np.zeros((hidden, hidden), dtype=np.float32)
    stats = ExtractionStats()
    expert_energy: dict[int, float] = {}  # expert_id → cumulative ||W||_F^2 in subspace proxy
    expert_energy_raw: dict[int, float] = {}

    # Group by shard so we open each file once.
    by_shard: dict[str, list[dict[str, Any]]] = {}
    for row in gram_candidates:
        by_shard.setdefault(row["shard"], []).append(row)
    # Also score remaining expert up_proj for router bias (cheap path): one proj
    # per expert present, limited to experts not already in gram set if needed.
    router_score_rows: list[dict[str, Any]] = []
    seen_experts: set[tuple[int, int]] = set()
    for row in present_math:
        if row["kind"] == "expert_proj" and row["proj"] == "up_proj":
            key = (int(row["layer"]), int(row["expert"]))
            if key in seen_experts:
                continue
            seen_experts.add(key)
            router_score_rows.append(row)
    for row in router_score_rows:
        by_shard.setdefault(row["shard"], [])  # ensure key; may already have gram rows

    _ensure_dir(working_set_dir)
    # Working window: a small manifest + optional scratch; tensors stay mmap-ish
    # via direct shard reads (no full shard copy — floor preserving).
    window_dir = working_set_dir / "glm-math-window"
    if window_dir.exists():
        stats.bytes_evicted_working_set += int(_rm_tree(window_dir)["bytes_removed"])
    _ensure_dir(window_dir)
    window_manifest = {
        "kind": "GLM_MATH_WEIGHT_WINDOW",
        "donor_dir": str(donor_dir),
        "shards": sorted(by_shard),
        "gram_tensor_count": len(gram_candidates),
        "note": "Reads tensors in-place from donor; no full-shard duplication.",
    }
    manifest_path = window_dir / "window_manifest.json"
    manifest_path.write_text(
        json.dumps(window_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    gram_names = {r["name"] for r in gram_candidates}
    # Score router bias only on experts already selected for the Gram (bounded).
    gram_expert_ids = {
        (int(r["layer"]), int(r["expert"]))
        for r in gram_candidates
        if r.get("kind") == "expert_proj" and r.get("expert") is not None
    }
    score_names = {
        r["name"]
        for r in router_score_rows
        if (int(r["layer"]), int(r["expert"])) in gram_expert_ids
    }

    for shard_name in sorted(by_shard):
        shard_path = donor_dir / shard_name
        if not shard_path.is_file():
            continue
        header, data_offset = _read_safetensors_header(shard_path)
        stats.shards_opened.append(shard_name)
        # Only rows we actually need from this shard.
        needed = [
            r
            for r in by_shard.get(shard_name, [])
            if r["name"] in gram_names or r["name"] in score_names
        ]
        # Also pick score-only rows that live in this shard but were not in gram list.
        for r in router_score_rows:
            if r["shard"] == shard_name and r["name"] in score_names:
                if all(x["name"] != r["name"] for x in needed):
                    needed.append(r)

        for row in needed:
            name = row["name"]
            if name not in header:
                continue
            # Floor check every so often
            if stats.tensors_seen % 64 == 0:
                assert_floor(floor_path, label="mid-extract")

            weight = _read_tensor_f32(shard_path, header, data_offset, name)
            nbytes = int(weight.size) * 2  # bf16 source size
            stats.bytes_read += nbytes
            stats.tensors_seen += 1
            stats.layers_touched.add(int(row["layer"]))

            if name in gram_names:
                g = _hidden_gram_contribution(weight, hidden=hidden)
                gram += g
                stats.tensors_used_for_gram += 1
                if row["kind"] == "dense_mlp":
                    stats.dense_mlp_tensors += 1
                elif row["kind"] == "expert_proj":
                    stats.expert_proj_tensors += 1
                    stats.experts_touched.add((int(row["layer"]), int(row["expert"])))
                elif row["kind"] == "router":
                    stats.router_tensors += 1
                elif row["kind"] == "shared_expert":
                    stats.dense_mlp_tensors += 1

            if name in score_names and row["kind"] == "expert_proj":
                expert_id = int(row["expert"])
                # Frobenius energy as router prior (training-free).
                energy = float(np.square(weight.astype(np.float64)).sum())
                expert_energy_raw[expert_id] = expert_energy_raw.get(expert_id, 0.0) + energy
                stats.tensors_scored_for_router += 1
                expert_energy[expert_id] = expert_energy.get(expert_id, 0.0) + energy

            del weight

        # Mark shard processed in working window (tiny receipt), then we will
        # evict the whole window at end — never the donor source.
        marker = window_dir / f"processed-{shard_name}.ok"
        marker.write_text(f"{shard_name}\n", encoding="utf-8")

    eig = top_eigenspace(gram.astype(np.float64, copy=False), rank)
    del gram  # free ~150MB

    # Evict working window
    free_before_evict = free_bytes(floor_path)
    eviction = _rm_tree(window_dir)
    free_after_evict = free_bytes(floor_path)
    stats.bytes_evicted_working_set += int(eviction["bytes_removed"])
    stats.windows_evicted.append(str(window_dir))
    floor_after = assert_floor(floor_path, label="post-extract")

    return {
        "basis": eig["basis"],
        "eigenvalues": eig["eigenvalues"],
        "rank": eig["rank"],
        "total_energy": eig["total_energy"],
        "captured_energy": eig["captured_energy"],
        "energy_fraction": eig["energy_fraction"],
        "all_eigenvalues_head": eig["all_eigenvalues_head"],
        "expert_energy": expert_energy,
        "expert_energy_raw": expert_energy_raw,
        "stats": stats,
        "floor_before": floor_before,
        "floor_after": floor_after,
        "eviction": {
            **eviction,
            "free_bytes_before_eviction": free_before_evict,
            "free_bytes_after_eviction": free_after_evict,
            "exact_eviction_confirmed": not window_dir.exists(),
            "donor_source_retained": True,
            "donor_dir": str(donor_dir),
        },
        "donor": {
            "path": str(donor_dir),
            "repository": GLM_5_2["repository"],
            "revision": GLM_5_2["revision"],
            "hidden_size": hidden,
            "math_tensors_present": len(present_math),
            "math_tensors_index_total": len(all_math),
            "gram_tensors_selected": len(gram_candidates),
        },
        "method": {
            "name": "weight_space_gram_pca",
            "training": False,
            "gradient_descent": False,
            "optimizer": None,
            "loss_minimization": False,
            "description": (
                "Accumulate H×H Gram of math-relevant GLM weights (expert/mlp "
                "gate|up|down + router), take top-r eigenspace.  No activations, "
                "no GLM forward, no student forward."
            ),
            "experts_per_layer_for_gram": experts_per_layer,
            "rank_requested": rank,
        },
    }


# ---------------------------------------------------------------------------
# Module seal / apply / reverse
# ---------------------------------------------------------------------------


def build_transfer_module(
    *,
    extraction: Mapping[str, Any],
    transplant_points: Sequence[str] = TRANSPLANT_POINT_NAMES,
    student_layers: Sequence[int] | None = None,
    steering_scale: float = 0.05,
    embedding_seed: int = 0,
) -> dict[str, Any]:
    """Compose closed-form projection + residual steering into one module dict."""

    basis = np.asarray(extraction["basis"], dtype=np.float64)
    evals = np.asarray(extraction["eigenvalues"], dtype=np.float64)
    proj = closed_form_projection(basis, seed=embedding_seed)
    steer = steering_from_subspace(basis, evals, proj["student_embedding"])
    scale = float(steering_scale)
    a_res = residual_lowrank_from_steering(steer["student_direction"], scale=scale)
    bias_vec = scale * np.asarray(steer["student_direction"], dtype=np.float64)
    router_bias = router_bias_from_scores(extraction.get("expert_energy") or {})

    layers = (
        list(student_layers)
        if student_layers is not None
        else list(range(int(DEEPSEEK_V4_FLASH["num_hidden_layers"])))
    )
    points = list(transplant_points)

    per_point: list[dict[str, Any]] = []
    for point in points:
        # Same residual form at every transplant point (weight-only); scale can
        # later be point-specific once forward measurements exist.
        point_scale = scale
        if point in {
            "router_logits",
            "selected_expert_ids",
            "route_probabilities_and_margins",
        }:
            # Router points get the bias vector path, not residual matrix.
            apply_mode = "router_bias_additive"
        elif point in {
            "lm_head_logits",
            "hcli_tool_action_decision",
        }:
            apply_mode = "none_structural_only"
            point_scale = 0.0
        else:
            apply_mode = "residual_stream_steering"
        per_point.append(
            {
                "transplant_point": point,
                "apply_mode": apply_mode,
                "steering_scale": point_scale,
                "shape_contract": list(BRIDGE_INPUT_SHAPE),
                "dtype": BRIDGE_DTYPE,
                "direct_weight_transplant": False,
                "student_layers": layers,
            }
        )

    module = {
        "schema": TRANSFER_MODULE_SCHEMA,
        "status": "TRAINING_FREE_MODULE_SEALED_UNVALIDATED",
        "kind": "reversible_residual_steering_module",
        "bridge": "GLM_MATH_BRIDGE",
        "direct_weight_transplant": False,
        "trained": False,
        "training_method": "none_closed_form_linear_algebra_only",
        "forbidden_path_excluded": {
            "loss_fitted_adapter": True,
            "gradient_descent": True,
            "optimizer_steps": True,
            "note": (
                "frankenstein_fusion_op.loss_target is DEPRECATED for fit; "
                "this module never calls it."
            ),
        },
        "geometries": {
            "donor_glm": {
                "hidden_size": int(GLM_5_2["hidden_size"]),
                "num_hidden_layers": int(GLM_5_2["num_hidden_layers"]),
                "n_routed_experts": int(GLM_5_2["n_routed_experts"]),
                "repository": GLM_5_2["repository"],
                "revision": GLM_5_2["revision"],
            },
            "student_deepseek": {
                "hidden_size": int(DEEPSEEK_V4_FLASH["hidden_size"]),
                "num_hidden_layers": int(DEEPSEEK_V4_FLASH["num_hidden_layers"]),
                "n_routed_experts": int(DEEPSEEK_V4_FLASH["n_routed_experts"]),
                "repository": DEEPSEEK_V4_FLASH["repository"],
                "revision": DEEPSEEK_V4_FLASH["revision"],
            },
        },
        "subspace": {
            "rank": int(extraction["rank"]),
            "energy_fraction_captured": float(extraction["energy_fraction"]),
            "captured_energy": float(extraction["captured_energy"]),
            "total_energy": float(extraction["total_energy"]),
            "eigenvalues_head": [float(x) for x in evals[:8].tolist()],
            "basis_shape": list(basis.shape),
            "method": extraction.get("method"),
        },
        "projection_glm_to_student": {
            "weight_shape": list(proj["weight"].shape),
            "bias_shape": list(proj["bias"].shape),
            "dtype": "float32_sealed_as_bf16",
            "method": proj["method"],
            "procrustes_status": proj["procrustes_status"],
            "math": proj["math"],
            "parameter_count": int(proj["weight"].size + proj["bias"].size),
        },
        "steering": {
            "vector_shape": [int(DEEPSEEK_V4_FLASH["hidden_size"])],
            "scale": scale,
            "method": steer["method"],
            "residual_operator": "A = scale * (v v^T); a_out = a_s + a_s @ A + scale * v",
        },
        "router_bias": {
            "shape": [int(DEEPSEEK_V4_FLASH["n_routed_experts"])],
            "method": "zscore_expert_up_proj_frobenius_energy",
            "nonzero_count": int(np.count_nonzero(router_bias)),
        },
        "residual_adapter": {
            "name": "reversible_rank1_steering_residual",
            "weight_shape": [int(DEEPSEEK_V4_FLASH["hidden_size"]), int(DEEPSEEK_V4_FLASH["hidden_size"])],
            "bias_shape": [int(DEEPSEEK_V4_FLASH["hidden_size"])],
            "rank": 1,
            "dtype": BRIDGE_DTYPE,
            "apply": "a_out = a_s + a_s @ A_res + b_steer",
            "reverse": "a_s = a_out - a_out_residual_recomputed; exact for linear residual",
            "bridge_io": {
                "input_tensor_state": {
                    "name": "per_token_hidden_state",
                    "shape_contract": list(BRIDGE_INPUT_SHAPE),
                    "source_dtype": BRIDGE_DTYPE,
                },
                "output_tensor_state": {
                    "name": "reversible_residual_adapter_output",
                    "shape_contract": list(BRIDGE_OUTPUT_SHAPE),
                    "source_dtype": BRIDGE_DTYPE,
                },
            },
        },
        "per_transplant_point": per_point,
        "layer_map_examples": {
            "L0": layer_map(donor="glm_5_2", student_layer=0),
            "L21": layer_map(donor="glm_5_2", student_layer=21),
            "L42": layer_map(donor="glm_5_2", student_layer=42),
        },
        "reversibility": {
            "kind": "additive_residual",
            "exact_for": "linear residual A and bias; router bias is additive on logits",
            "remove_module": "omit apply — student body is never rewritten in place",
        },
        "capability_status": "UNVALIDATED_WEIGHT_ONLY_DERIVED",
        "capability_claim": False,
        "math_bench_status": "NOT_RUN",
        "forward_gate": FORWARD_GATE,
        "validation_gate": {
            "name": FORWARD_GATE,
            "blocks": [
                "math_bench_measurement",
                "paired_activation_procrustes_refinement",
                "steering_scale_effect_measurement",
            ],
            "does_not_block": [
                "weight_subspace_extraction",
                "closed_form_projection",
                "module_seal",
                "structural_apply_artifact",
            ],
        },
        "arrays": {
            "basis_glm": basis,
            "eigenvalues": evals,
            "projection_weight": proj["weight"],
            "projection_bias": proj["bias"],
            "student_embedding": proj["student_embedding"],
            "steering_student": steer["student_direction"],
            "steering_glm": steer["glm_direction"],
            "residual_A": a_res,
            "residual_bias": bias_vec,
            "router_bias": router_bias,
        },
        "extraction_stats": _stats_public(extraction.get("stats")),
        "donor_extraction": {
            k: extraction[k]
            for k in (
                "donor",
                "floor_before",
                "floor_after",
                "eviction",
                "energy_fraction",
                "captured_energy",
                "total_energy",
                "rank",
            )
            if k in extraction
        },
    }
    return module


def _stats_public(stats: Any) -> dict[str, Any]:
    if stats is None:
        return {}
    if isinstance(stats, ExtractionStats):
        return {
            "tensors_seen": stats.tensors_seen,
            "tensors_used_for_gram": stats.tensors_used_for_gram,
            "tensors_scored_for_router": stats.tensors_scored_for_router,
            "bytes_read": stats.bytes_read,
            "bytes_evicted_working_set": stats.bytes_evicted_working_set,
            "layers_touched": sorted(stats.layers_touched),
            "experts_touched_count": len(stats.experts_touched),
            "dense_mlp_tensors": stats.dense_mlp_tensors,
            "expert_proj_tensors": stats.expert_proj_tensors,
            "router_tensors": stats.router_tensors,
            "shards_opened": list(stats.shards_opened),
            "windows_evicted": list(stats.windows_evicted),
        }
    if isinstance(stats, Mapping):
        return dict(stats)
    return {}


def _module_arrays_to_payload(module: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    """Serialize compact float arrays as bf16 blocks (no dense H×H residual dump).

    Stores factors only:
      - basis_glm B [H_g, r], student_embedding E [H_s, r]
      - projection reconstructed as W = B @ E.T on load
      - residual_A reconstructed as scale * v v^T on load
    """

    arrays = module["arrays"]
    table: dict[str, Any] = {}
    chunks: list[bytes] = []
    cursor = 0
    # Compact set: reconstruct dense ops on load.
    names = (
        "basis_glm",
        "eigenvalues",
        "projection_bias",
        "student_embedding",
        "steering_student",
        "steering_glm",
        "residual_bias",
        "router_bias",
    )
    for name in names:
        arr = np.ascontiguousarray(np.asarray(arrays[name], dtype=np.float32))
        u16 = _f32_to_bf16_u16(arr.ravel())
        raw = u16.tobytes()
        table[name] = {
            "shape": list(arr.shape),
            "dtype": "bfloat16",
            "offset": cursor,
            "nbytes": len(raw),
        }
        chunks.append(raw)
        cursor += len(raw)
    table["_reconstruct"] = {
        "projection_weight": "basis_glm @ student_embedding.T",
        "residual_A": "scale * outer(unit(steering_student), unit(steering_student))",
        "steering_scale_key": "steering.scale",
    }
    return table, b"".join(chunks)


def seal_transfer_module_files(
    module: Mapping[str, Any],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    """Write binary module + sealed JSON metadata (no gravity)."""

    _ensure_dir(out_dir)
    meta = {k: v for k, v in module.items() if k != "arrays"}
    table, payload = _module_arrays_to_payload(module)
    meta["array_table"] = table
    meta["payload_bytes"] = len(payload)
    meta["recorded_at"] = _utc_now()
    sealed_meta = seal(meta)
    meta_path = out_dir / "FRANKENSTEIN_TRAINING_FREE_MODULE.json"
    _atomic_create(meta_path, _canonical(sealed_meta) + b"\n")

    header = {
        "magic": MODULE_MAGIC.decode("ascii"),
        "schema": TRANSFER_MODULE_SCHEMA,
        "meta_seal_sha256": sealed_meta["seal_sha256"],
        "array_table": table,
        "payload_bytes": len(payload),
        "trained": False,
        "gravity_compressed": False,
        "capability_status": sealed_meta.get("capability_status"),
    }
    header_raw = _canonical(header)
    raw = MODULE_MAGIC + struct.pack(">I", len(header_raw)) + header_raw + payload
    bin_path = out_dir / "FRANKENSTEIN_TRAINING_FREE_MODULE.raw_module"
    _ensure_dir(bin_path.parent)
    bin_path.write_bytes(raw)
    fd = os.open(bin_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    return {
        "meta_path": str(meta_path),
        "meta_seal_sha256": sealed_meta["seal_sha256"],
        "module_path": str(bin_path),
        "module_sha256": _sha256(raw),
        "module_bytes": len(raw),
        "payload_bytes": len(payload),
        "status": sealed_meta["status"],
        "capability_status": sealed_meta["capability_status"],
        "document": sealed_meta,
    }


def load_transfer_module(meta_path: Path, module_path: Path | None = None) -> dict[str, Any]:
    """Load sealed module metadata + arrays."""

    doc = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if doc.get("schema") != TRANSFER_MODULE_SCHEMA:
        raise FrankensteinTransferError(f"unexpected module schema: {doc.get('schema')}")
    bin_path = (
        Path(module_path)
        if module_path is not None
        else Path(meta_path).with_name("FRANKENSTEIN_TRAINING_FREE_MODULE.raw_module")
    )
    raw = bin_path.read_bytes()
    if raw[:8] != MODULE_MAGIC:
        raise FrankensteinTransferError("bad module magic")
    header_len = struct.unpack(">I", raw[8:12])[0]
    header = json.loads(raw[12 : 12 + header_len].decode("utf-8"))
    payload = raw[12 + header_len :]
    arrays: dict[str, np.ndarray] = {}
    table = header.get("array_table") or doc.get("array_table") or {}
    for name, info in table.items():
        if name.startswith("_") or not isinstance(info, Mapping) or "offset" not in info:
            continue
        off = int(info["offset"])
        nbytes = int(info["nbytes"])
        shape = tuple(info["shape"])
        u16 = np.frombuffer(payload[off : off + nbytes], dtype=np.uint16)
        arrays[name] = _bf16_u16_to_f32(u16).reshape(shape).astype(np.float64)
    # Reconstruct dense operators from factors when sealed compactly.
    if "projection_weight" not in arrays and "basis_glm" in arrays and "student_embedding" in arrays:
        arrays["projection_weight"] = arrays["basis_glm"] @ arrays["student_embedding"].T
    if "residual_A" not in arrays and "steering_student" in arrays:
        v = np.asarray(arrays["steering_student"], dtype=np.float64).reshape(-1)
        vn = float(np.linalg.norm(v))
        if vn > 0:
            v = v / vn
        scale = float((doc.get("steering") or {}).get("scale", 0.05))
        arrays["residual_A"] = scale * np.outer(v, v)
    doc = dict(doc)
    doc["arrays"] = arrays
    return doc


def apply_residual(
    hidden: np.ndarray,
    module: Mapping[str, Any],
    *,
    transplant_point: str = "post_moe_hidden_state",
) -> np.ndarray:
    """Apply reversible residual steering to a student hidden state.

    a_out = a_s + a_s @ A + b
    """

    arrays = module["arrays"]
    a = np.asarray(hidden, dtype=np.float64)
    # Find scale for point
    scale = float(module.get("steering", {}).get("scale", 0.05))
    mode = "residual_stream_steering"
    for row in module.get("per_transplant_point") or []:
        if row.get("transplant_point") == transplant_point:
            mode = row.get("apply_mode", mode)
            scale = float(row.get("steering_scale", scale))
            break
    if mode == "none_structural_only":
        return a.copy()
    if mode == "router_bias_additive":
        # Hidden stream unchanged; router bias applied elsewhere.
        return a.copy()
    a_res = np.asarray(arrays["residual_A"], dtype=np.float64)
    b = np.asarray(arrays["residual_bias"], dtype=np.float64)
    if abs(scale - float(module.get("steering", {}).get("scale", scale))) > 1e-12:
        # Re-scale rank-1 residual relative to sealed default.
        default = float(module.get("steering", {}).get("scale", 0.05)) or 0.05
        factor = scale / default
        a_res = a_res * factor
        b = b * factor
    # Support [H], [B,H], [B,S,H]
    return a + a @ a_res + b


def reverse_residual(
    hidden_out: np.ndarray,
    module: Mapping[str, Any],
    *,
    transplant_point: str = "post_moe_hidden_state",
) -> np.ndarray:
    """Exact reverse for affine residual a_out = a @ (I+A) + b when (I+A) invertible.

    For rank-1 A = s v v^T, (I+A)^{-1} = I - (s/(1+s)) v v^T when v unit and s≠-1.
    """

    arrays = module["arrays"]
    y = np.asarray(hidden_out, dtype=np.float64)
    mode = "residual_stream_steering"
    scale = float(module.get("steering", {}).get("scale", 0.05))
    for row in module.get("per_transplant_point") or []:
        if row.get("transplant_point") == transplant_point:
            mode = row.get("apply_mode", mode)
            scale = float(row.get("steering_scale", scale))
            break
    if mode in {"none_structural_only", "router_bias_additive"}:
        return y.copy()
    v = np.asarray(arrays["steering_student"], dtype=np.float64).reshape(-1)
    vn = float(np.linalg.norm(v))
    if vn > 0:
        v = v / vn
    b = np.asarray(arrays["residual_bias"], dtype=np.float64)
    default = float(module.get("steering", {}).get("scale", 0.05)) or 0.05
    factor = scale / default
    b = b * factor
    s = scale  # A = scale * vv^T for unit v
    y0 = y - b
    # Sherman-Morrison: (I + s vv^T)^{-1} = I - s/(1+s) vv^T
    if abs(1.0 + s) < 1e-12:
        raise FrankensteinTransferError("residual scale makes I+A singular")
    coeff = s / (1.0 + s)
    # Contract last axis with v, then outer-subtract.
    proj = np.tensordot(y0, v, axes=([-1], [0]))  # shape = y0.shape[:-1]
    return y0 - np.multiply.outer(proj, coeff * v)


def frankenstein_transfer_apply(
    *,
    module: Mapping[str, Any],
    body_path: Path,
    out_dir: Path,
    student_layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Structurally compose the transfer module onto the DeepSeek body reference.

    Does **not** rewrite the read-only body.  Writes a raw (non-gravity)
    composition artifact that references the body + sealed module and records
    per-transplant-point apply plans.  Measured validation stays
    ``DEEPSEEK_FORWARD_PENDING``.
    """

    _ensure_dir(out_dir)
    floor = assert_floor(out_dir if out_dir.exists() else Path.cwd(), label="apply")
    body_path = Path(body_path)
    body_present = body_path.exists()
    layers = (
        list(student_layers)
        if student_layers is not None
        else list(range(int(DEEPSEEK_V4_FLASH["num_hidden_layers"])))
    )

    composition = {
        "schema": APPLY_RECEIPT_SCHEMA,
        "status": "STRUCTURAL_APPLY_SEALED_VALIDATION_PENDING",
        "recorded_at": _utc_now(),
        "trained": False,
        "direct_weight_transplant": False,
        "gravity_compressed": False,
        "student_body": {
            "path": str(body_path),
            "present": body_present,
            "read_only": True,
            "rewritten": False,
            "repository": DEEPSEEK_V4_FLASH["repository"],
            "revision": DEEPSEEK_V4_FLASH["revision"],
        },
        "module": {
            "schema": module.get("schema"),
            "status": module.get("status"),
            "capability_status": module.get("capability_status"),
            "seal_sha256": module.get("seal_sha256"),
            "bridge": module.get("bridge"),
            "subspace_rank": (module.get("subspace") or {}).get("rank"),
            "energy_fraction": (module.get("subspace") or {}).get("energy_fraction_captured"),
        },
        "apply_plan": {
            "student_layers": layers,
            "transplant_points": [
                {
                    "transplant_point": row["transplant_point"],
                    "apply_mode": row["apply_mode"],
                    "steering_scale": row["steering_scale"],
                }
                for row in (module.get("per_transplant_point") or [])
            ],
            "formula_residual": "a_out = a_s + a_s @ A + b_steer",
            "formula_router": "logits_out = logits_s + router_bias",
            "reversible": True,
        },
        "validation": {
            "status": FORWARD_GATE,
            "math_bench": "NOT_RUN",
            "capability_claim": False,
            "note": (
                "Structural composition only.  Math-bench comparison of transferred "
                "body vs base DeepSeek requires the 43-layer student forward."
            ),
        },
        "disk": floor,
        "claim_boundary": {
            "trained_adapter": False,
            "weight_average": False,
            "direct_weight_transplant": False,
            "gravity_compressed": False,
            "deepseek_body_modified": False,
            "frankenstein_math_capability_validated": False,
        },
    }
    sealed = seal(composition)
    path = out_dir / "FRANKENSTEIN_TRANSFER_APPLY.json"
    _atomic_create(path, _canonical(sealed) + b"\n")

    # Side-car: numpy-readable apply packet (raw residual ops) for when forward lands.
    packet = {
        "residual_A_shape": list(np.asarray(module["arrays"]["residual_A"]).shape),
        "residual_bias_shape": list(np.asarray(module["arrays"]["residual_bias"]).shape),
        "router_bias_shape": list(np.asarray(module["arrays"]["router_bias"]).shape),
        "projection_weight_shape": list(
            np.asarray(module["arrays"]["projection_weight"]).shape
        ),
        "note": "arrays live in FRANKENSTEIN_TRAINING_FREE_MODULE.raw_module",
    }
    packet_path = out_dir / "FRANKENSTEIN_TRANSFER_APPLY_PACKET.json"
    _atomic_create(packet_path, _canonical(packet) + b"\n")

    return {
        "status": sealed["status"],
        "apply_path": str(path),
        "apply_seal_sha256": sealed["seal_sha256"],
        "packet_path": str(packet_path),
        "validation_status": FORWARD_GATE,
        "capability_claim": False,
        "document": sealed,
    }


def assert_no_training_path() -> dict[str, Any]:
    """Static guarantee used by tests: this module never imports training stacks.

    Uses AST checks only (no forbidden API name literals embedded as code tokens
    that would self-trip string scanners).
    """

    import lab.operators.frankenstein_transfer as self_mod

    source = Path(self_mod.__file__).read_text(encoding="utf-8")
    tree = ast_parse_safe(source)
    hits: list[dict[str, Any]] = []
    imports_torch = False
    imports_optimizer = False
    for node in ast_walk_safe(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                if node.module:
                    names = [node.module]
            for name in names:
                if name == "torch" or name.startswith("torch."):
                    imports_torch = True
                    hits.append({"kind": "import", "module": name})
                if "optim" in name.split("."):
                    imports_optimizer = True
                    hits.append({"kind": "import", "module": name})
        if isinstance(node, ast.Call):
            func = node.func
            attr = None
            if isinstance(func, ast.Attribute):
                attr = func.attr
            elif isinstance(func, ast.Name):
                attr = func.id
            # Training-step APIs only; ignore pathlib Path.mkdir etc.
            if attr in {"backward", "zero_grad"}:
                hits.append({"kind": "call", "name": attr, "lineno": getattr(node, "lineno", None)})
            if attr == "step" and isinstance(func, ast.Attribute):
                # Only flag if receiver looks like an optimizer attribute chain.
                recv = func.value
                recv_name = ""
                if isinstance(recv, ast.Name):
                    recv_name = recv.id
                elif isinstance(recv, ast.Attribute):
                    recv_name = recv.attr
                if recv_name in {"optimizer", "optim", "opt", "adam", "sgd"}:
                    hits.append({"kind": "call", "name": "step", "lineno": getattr(node, "lineno", None)})
    return {
        "training_path_present": len(hits) > 0,
        "hits": hits,
        "imports_torch": imports_torch,
        "imports_optimizer": imports_optimizer,
    }


def ast_parse_safe(source: str):
    import ast as _ast

    return _ast.parse(source)


def ast_walk_safe(tree):
    import ast as _ast

    return _ast.walk(tree)


def run_weight_only_transfer(
    *,
    donor_dir: Path = DEFAULT_GLM_DONOR,
    out_dir: Path | None = None,
    body_path: Path = DEFAULT_BODY_PATH,
    rank: int = DEFAULT_SUBSPACE_RANK,
    experts_per_layer: int = DEFAULT_EXPERTS_PER_LAYER_FOR_GRAM,
    max_tensors: int | None = None,
    steering_scale: float = 0.05,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Full weight-only pipeline: extract → project → seal module → structural apply."""

    ws = Path(workspace) if workspace is not None else WORKSPACE_ROOT
    evidence = Path(out_dir) if out_dir is not None else (ws / "campaign" / "evidence" / "models" / "frankenstein")
    runs = ws / "campaign" / "records" / "runs" / "frankenstein"
    module_dir = evidence / "transfer-module"
    apply_dir = evidence / "transfer-apply"
    working = runs / "working-set" / "training-free"
    _ensure_dir(module_dir)
    _ensure_dir(apply_dir)
    _ensure_dir(working)

    floor0 = assert_floor(ws, label="workspace-start")
    guard = assert_no_training_path()
    if guard["training_path_present"] or guard["imports_torch"] or guard["imports_optimizer"]:
        raise FrankensteinTransferError(f"training path detected: {guard}")

    extraction = extract_glm_math_subspace(
        donor_dir=Path(donor_dir),
        working_set_dir=working,
        rank=rank,
        experts_per_layer=experts_per_layer,
        max_tensors=max_tensors,
        floor_path=ws,
    )
    module = build_transfer_module(
        extraction=extraction,
        steering_scale=steering_scale,
    )
    sealed = seal_transfer_module_files(module, out_dir=module_dir)
    # Reload arrays into sealed doc for apply
    module_loaded = load_transfer_module(Path(sealed["meta_path"]), Path(sealed["module_path"]))
    applied = frankenstein_transfer_apply(
        module=module_loaded,
        body_path=Path(body_path),
        out_dir=apply_dir,
    )

    # Subspace receipt (trackable evidence)
    subspace_receipt = seal(
        {
            "schema": SUBSPACE_RECEIPT_SCHEMA,
            "status": "GLM_MATH_SUBSPACE_EXTRACTED_WEIGHT_ONLY",
            "recorded_at": _utc_now(),
            "trained": False,
            "donor": extraction["donor"],
            "subspace": {
                "rank": extraction["rank"],
                "energy_fraction_captured": extraction["energy_fraction"],
                "captured_energy": extraction["captured_energy"],
                "total_energy": extraction["total_energy"],
                "eigenvalues_head": extraction["all_eigenvalues_head"],
                "basis_shape": list(np.asarray(extraction["basis"]).shape),
            },
            "stats": _stats_public(extraction["stats"]),
            "eviction": extraction["eviction"],
            "disk": {
                "before": extraction["floor_before"],
                "after": extraction["floor_after"],
                "floor_bytes": MIN_FREE_FLOOR_BYTES,
                "floor_preserved": True,
            },
            "method": extraction["method"],
            "capability_claim": False,
            "forward_gate": FORWARD_GATE,
        }
    )
    subspace_path = evidence / "FRANKENSTEIN_GLM_MATH_SUBSPACE.json"
    _atomic_create(subspace_path, _canonical(subspace_receipt) + b"\n")

    run_receipt = seal(
        {
            "schema": RUN_RECEIPT_SCHEMA,
            "status": "TRAINING_FREE_WEIGHT_ONLY_COMPLETE_UNVALIDATED",
            "recorded_at": _utc_now(),
            "trained": False,
            "training_path_guard": guard,
            "floor_start": floor0,
            "floor_end": assert_floor(ws, label="workspace-end"),
            "subspace_receipt": {
                "path": str(subspace_path),
                "seal_sha256": subspace_receipt["seal_sha256"],
            },
            "module": {
                "meta_path": sealed["meta_path"],
                "meta_seal_sha256": sealed["meta_seal_sha256"],
                "module_path": sealed["module_path"],
                "module_sha256": sealed["module_sha256"],
                "module_bytes": sealed["module_bytes"],
                "capability_status": sealed["capability_status"],
            },
            "apply": {
                "path": applied["apply_path"],
                "seal_sha256": applied["apply_seal_sha256"],
                "validation_status": applied["validation_status"],
            },
            "energy_fraction_captured": extraction["energy_fraction"],
            "bytes_read": _stats_public(extraction["stats"]).get("bytes_read"),
            "bytes_evicted_working_set": _stats_public(extraction["stats"]).get(
                "bytes_evicted_working_set"
            ),
            "claim_boundary": {
                "math_capability_validated": False,
                "trained_adapter": False,
                "direct_weight_transplant": False,
                "weight_average": False,
                "gravity_compressed": False,
                "deepseek_body_modified": False,
            },
            "forward_gate": FORWARD_GATE,
            "honest_status": (
                "Subspace extracted, closed-form projection sealed, structural apply "
                "recorded.  Effect on math-bench is UNVALIDATED until DeepSeek forward."
            ),
        }
    )
    run_path = evidence / "FRANKENSTEIN_TRAINING_FREE_RUN.json"
    _atomic_create(run_path, _canonical(run_receipt) + b"\n")

    # Also drop a copy under records/runs for operator visibility (gitignored).
    runs_receipt_dir = runs / "training-free"
    _ensure_dir(runs_receipt_dir)
    _atomic_create(runs_receipt_dir / "FRANKENSTEIN_TRAINING_FREE_RUN.json", _canonical(run_receipt) + b"\n")

    return {
        "status": run_receipt["status"],
        "run_path": str(run_path),
        "run_seal_sha256": run_receipt["seal_sha256"],
        "subspace_path": str(subspace_path),
        "subspace_seal_sha256": subspace_receipt["seal_sha256"],
        "module_meta_path": sealed["meta_path"],
        "module_path": sealed["module_path"],
        "apply_path": applied["apply_path"],
        "energy_fraction_captured": extraction["energy_fraction"],
        "rank": extraction["rank"],
        "bytes_read": _stats_public(extraction["stats"]).get("bytes_read"),
        "bytes_evicted_working_set": _stats_public(extraction["stats"]).get(
            "bytes_evicted_working_set"
        ),
        "layers_touched": _stats_public(extraction["stats"]).get("layers_touched"),
        "tensors_used_for_gram": _stats_public(extraction["stats"]).get(
            "tensors_used_for_gram"
        ),
        "floor_bytes": MIN_FREE_FLOOR_BYTES,
        "floor_end_free": run_receipt["floor_end"]["free_bytes"],
        "forward_gate": FORWARD_GATE,
        "capability_claim": False,
        "honest_status": run_receipt["honest_status"],
    }


# ---------------------------------------------------------------------------
# Synthetic fixture path (tests; no real donor required)
# ---------------------------------------------------------------------------


def extract_from_synthetic_weights(
    *,
    hidden_glm: int = int(GLM_5_2["hidden_size"]),
    hidden_student: int = int(DEEPSEEK_V4_FLASH["hidden_size"]),
    rank: int = 8,
    n_experts: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Deterministic synthetic Gram path for unit tests (still training-free)."""

    rng = np.random.default_rng(seed)
    # Plant a low-rank math direction then add small isotropic noise.
    true_basis = rng.standard_normal((hidden_glm, rank))
    true_basis, _ = np.linalg.qr(true_basis)
    gram = np.zeros((hidden_glm, hidden_glm), dtype=np.float64)
    expert_energy: dict[int, float] = {}
    for e in range(n_experts):
        coeffs = rng.standard_normal((64, rank)) * 3.0
        # Noise in the orthogonal complement only, low amplitude.
        noise = 0.01 * rng.standard_normal((64, hidden_glm))
        noise = noise - (noise @ true_basis) @ true_basis.T
        w = coeffs @ true_basis.T + noise  # [64, H]
        gram += w.T @ w
        expert_energy[e] = float(np.square(w).sum())
    eig = top_eigenspace(gram, rank)
    stats = ExtractionStats(
        tensors_seen=n_experts,
        tensors_used_for_gram=n_experts,
        tensors_scored_for_router=n_experts,
        bytes_read=n_experts * 32 * hidden_glm * 2,
        bytes_evicted_working_set=0,
        layers_touched={0},
        expert_proj_tensors=n_experts,
    )
    return {
        "basis": eig["basis"],
        "eigenvalues": eig["eigenvalues"],
        "rank": eig["rank"],
        "total_energy": eig["total_energy"],
        "captured_energy": eig["captured_energy"],
        "energy_fraction": eig["energy_fraction"],
        "all_eigenvalues_head": eig["all_eigenvalues_head"],
        "expert_energy": expert_energy,
        "stats": stats,
        "floor_before": {"status": "FLOOR_OK", "floor_bytes": MIN_FREE_FLOOR_BYTES},
        "floor_after": {"status": "FLOOR_OK", "floor_bytes": MIN_FREE_FLOOR_BYTES},
        "eviction": {
            "exact_eviction_confirmed": True,
            "donor_source_retained": True,
            "bytes_removed": 0,
        },
        "donor": {
            "path": "synthetic",
            "repository": GLM_5_2["repository"],
            "revision": GLM_5_2["revision"],
            "hidden_size": hidden_glm,
            "math_tensors_present": n_experts,
            "gram_tensors_selected": n_experts,
        },
        "method": {
            "name": "synthetic_weight_space_gram_pca",
            "training": False,
            "gradient_descent": False,
            "optimizer": None,
            "loss_minimization": False,
        },
        "true_basis_for_test": true_basis,
        "student_hidden": hidden_student,
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=WORKSPACE_ROOT)
    p.add_argument("--donor-dir", type=Path, default=DEFAULT_GLM_DONOR)
    p.add_argument("--body-path", type=Path, default=DEFAULT_BODY_PATH)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--rank", type=int, default=DEFAULT_SUBSPACE_RANK)
    p.add_argument("--experts-per-layer", type=int, default=DEFAULT_EXPERTS_PER_LAYER_FOR_GRAM)
    p.add_argument("--max-tensors", type=int, default=None)
    p.add_argument("--steering-scale", type=float, default=0.05)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("weight-only", help="extract subspace + seal module + structural apply")
    sub.add_parser("guard", help="assert no training path in this module")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "guard":
        result = assert_no_training_path()
        print(json.dumps(result, sort_keys=True, indent=2))
        return 1 if result["training_path_present"] else 0
    if args.command == "weight-only":
        try:
            result = run_weight_only_transfer(
                donor_dir=args.donor_dir,
                out_dir=args.out_dir,
                body_path=args.body_path,
                rank=args.rank,
                experts_per_layer=args.experts_per_layer,
                max_tensors=args.max_tensors,
                steering_scale=args.steering_scale,
                workspace=args.workspace,
            )
        except FrankensteinTransferError as exc:
            raise SystemExit(f"frankenstein transfer error: {exc}") from exc
        print(json.dumps(result, sort_keys=True, indent=2, default=str))
        return 0
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
