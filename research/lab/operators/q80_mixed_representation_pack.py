#!/usr/bin/env python3
"""Stream-pack Qwen3-Coder-Next as the mixed ≤1.5 representation.

Closes the pack lane gap (`artifact_packed`). Does not write Metal kernels
and does not generate tokens. On-disk complete BPW is measured from the
bytes that were written.

Recipe (receipts/QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json):

  routed gate_proj  HGRAVB01 binary_group 128
  routed up_proj    HGRAVR02 binary + rice_q1_rms @ 2%
  routed down_proj  HGRAVS01 r160_b3, fit on post-SwiGLU X
  non-expert        HGRAVU01 uniform-q8 group-64  (sensitive 3% untouched)

Physical layout follows token-graph consumption order. The catalog is a
compact mmap table, not a 74k-row JSON array.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.ascension_dual_gravity_worker import (  # noqa: E402
    ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
    GROUP_BINARY,
    GROUP_UNIFORM,
    MAGIC_ACT_SVD,
    MAGIC_BINARY,
    _activation_weighted_svd_low_rank_codec,
    _binary_codec,
    _container,
    _decode_activation_weighted_svd_low_rank_codec,
    _decode_uniform_codec,
    _factor_codec,
    _parse_container,
    _uniform_codec,
)
from lab.operators.hgravs01_adapter import HGRAVS01_SCHEMA  # noqa: E402
from lab.operators.q80_capture_index import (  # noqa: E402
    inspect_index,
    _decode_path,
    _load_npy,
)
from lab.operators.qwen30b_gravity_pack import (  # noqa: E402
    _HEADER_CACHE,
    load_tensor,
    load_weight_map,
    read_safetensors_header,
)
from lab.operators.residual_compact_codec import (  # noqa: E402
    _rebuild_binary,
    decode_residual_compact,
    encode_residual_compact,
)
from lab.receipts import seal  # noqa: E402

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
MODEL_DIR = MAIN_HAWKING / (
    "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
)
CAPTURE_DIR = MAIN_HAWKING / (
    "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    "/quality-diagnostics/source-bf16-capture-n192-scale64"
)
DEFAULT_ROOT = MAIN_HAWKING / (
    "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    "/quality-candidates/mixed-1p5-v1"
)
REVALIDATION = MAIN_HAWKING / (
    "workspace/campaign/records/ascension-sandbox/physical/qwen80"
    "/complete-gravity/QWEN80_CURRENT_SOURCE_SHARD_REVALIDATION.json"
)

SCHEMA = "hawking.ascension.qwen80_mixed_representation_candidate.v1"
TERMINAL_SCHEMA = "hawking.ascension.complete_binary_terminal_status.v1"
BRANCH_ID = "qwen80-mixed-1p5-v1"
MODEL_ID = "Qwen3-Coder-Next-mixed-1p5-v1"
ARTIFACT_PREFIX = "QWEN80_MIXED_1P5_V1"
CANDIDATE_STATUS = "CANDIDATE_MIXED_REPRESENTATION_PACKED_NOT_GENERATED"
EXPECTED_TENSOR_COUNT = 74_391
N_LAYERS = 48
N_EXPERTS = 512
HIDDEN = 2048
MOE_INTERMEDIATE = 512
F_EXPERT = 0.9703169371044981
F_NONEXPERT = 0.029683062895501933
RANK = 160
FACTOR_BITS = 3
OUTLIER_RATIO = 0.02
DISK_FLOOR = 15 * 1024**3
SOURCE_ELEMENTS = 79_674_391_296

CATALOG_MAGIC = b"HQ80M15\0"
CATALOG_VERSION = 1
RECORD_SIZE = 128
CODEC_BINARY = 0
CODEC_RESIDUAL = 1
CODEC_HGRAVS01 = 2
CODEC_UNIFORM8 = 3
ORGAN_GATE = 0
ORGAN_UP = 1
ORGAN_DOWN = 2
ORGAN_NONEXPERT = 3
FLAG_SENSITIVE = 1 << 0
FLAG_GRAM_RANKDEF = 1 << 1
FLAG_WEIGHT_SPACE = 1 << 2
FLAG_ACTIVATION_WEIGHTED = 1 << 3
FIT_AW_R160 = 0
FIT_AW_RANKDEF = 1
FIT_WEIGHT_SPACE = 2

CAPTURE_SHA256 = "17a1e9b60a53cc491601a549880c2d215ff16395ee36abaa05fb95eb7fe2aabe"
CAPTURE_SCHEMA = (
    "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_result.v1"
)
CAPTURE_STATUS = (
    "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_SOURCE_BF16_LAYER_MAJOR"
    "_ALL_LAYER_ROUTE_AND_HIDDEN_CAPTURE"
)

_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class PackError(RuntimeError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(path)
    return int(st.f_bavail) * int(st.f_frsize)


def require_disk(path: Path, label: str) -> None:
    avail = free_bytes(path)
    if avail < DISK_FLOOR:
        raise PackError(
            f"{label}: free {avail} bytes below 15 GiB floor {DISK_FLOOR}"
        )


def capture_identity(capture_dir: Path) -> dict[str, Any]:
    return {
        "path": str(capture_dir),
        "capture_result_path": str(capture_dir / "capture-result.json"),
        "sha256": CAPTURE_SHA256,
        "schema": CAPTURE_SCHEMA,
        "status": CAPTURE_STATUS,
        "fit_kind": "real_routed_activation_capture",
        "not_synthetic_unit_direction": True,
    }


def classify(name: str) -> tuple[int, int]:
    m = _EXPERT_RE.match(name)
    if m:
        which = m.group(3)
        if which == "gate":
            return ORGAN_GATE, CODEC_BINARY
        if which == "up":
            return ORGAN_UP, CODEC_RESIDUAL
        return ORGAN_DOWN, CODEC_HGRAVS01
    return ORGAN_NONEXPERT, CODEC_UNIFORM8


def is_gqa_layer(layer: int) -> bool:
    return int(layer) % 4 == 3


def layer_of(name: str) -> int | None:
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def execution_order(names: Iterable[str]) -> list[str]:
    """Token-graph consumption order. Every input name appears exactly once."""

    remaining = set(names)
    out: list[str] = []

    def take(name: str) -> None:
        if name in remaining:
            remaining.remove(name)
            out.append(name)

    take("model.embed_tokens.weight")
    for layer in range(N_LAYERS):
        p = f"model.layers.{layer}."
        take(p + "input_layernorm.weight")
        if is_gqa_layer(layer):
            for suffix in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "self_attn.o_proj.weight",
            ):
                take(p + suffix)
        else:
            for suffix in (
                "linear_attn.in_proj_qkvz.weight",
                "linear_attn.in_proj_ba.weight",
                "linear_attn.conv1d.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ):
                take(p + suffix)
        leftovers = sorted(
            n
            for n in remaining
            if n.startswith(p)
            and ".mlp." not in n
            and not n.endswith("post_attention_layernorm.weight")
        )
        for n in leftovers:
            take(n)
        take(p + "post_attention_layernorm.weight")
        take(p + "mlp.gate.weight")
        take(p + "mlp.shared_expert.gate_proj.weight")
        take(p + "mlp.shared_expert.up_proj.weight")
        take(p + "mlp.shared_expert.down_proj.weight")
        take(p + "mlp.shared_expert_gate.weight")
        mlp_left = sorted(
            n
            for n in remaining
            if n.startswith(p + "mlp.") and ".experts." not in n
        )
        for n in mlp_left:
            take(n)
        for expert in range(N_EXPERTS):
            e = p + f"mlp.experts.{expert}."
            take(e + "gate_proj.weight")
            take(e + "up_proj.weight")
            take(e + "down_proj.weight")
    take("model.norm.weight")
    take("lm_head.weight")
    if remaining:
        raise PackError(
            "execution_order dropped tensors: " + ", ".join(sorted(remaining)[:12])
        )
    return out


def decode_binary(payload: bytes) -> np.ndarray:
    header, body = _parse_container(payload, expected_magic=MAGIC_BINARY)
    scale_bytes = int(header["scale_bytes"])
    groups = int(header["groups"])
    scales = np.frombuffer(body[:scale_bytes], dtype="<f2", count=groups)
    rebuilt = _rebuild_binary(
        scales,
        body[scale_bytes : scale_bytes + int(header["sign_bytes"])],
        elements=int(header["elements"]),
        group_size=int(header["group_size"]),
    )
    return np.ascontiguousarray(rebuilt.reshape(header["shape"]), dtype=np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def post_swiglu(x_hidden: np.ndarray, gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """silu(X @ W_gate.T) * (X @ W_up.T). X is [N, 2048], W is [512, 2048]."""

    x = np.ascontiguousarray(x_hidden, dtype=np.float32)
    g = np.ascontiguousarray(gate, dtype=np.float32)
    u = np.ascontiguousarray(up, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != g.shape[1] or g.shape != u.shape:
        raise PackError(
            f"swiglu geometry X{x.shape} gate{g.shape} up{u.shape}"
        )
    return silu(x @ g.T) * (x @ u.T)


def cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(left @ right)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    if den <= 1e-12:
        return 1.0 if num == 0.0 else 0.0
    return num / den


def encode_gate(values: np.ndarray) -> tuple[bytes, np.ndarray]:
    result = _binary_codec(values, group_size=GROUP_BINARY)
    decoded = decode_binary(result.payload)
    if decoded.shape != values.shape:
        raise PackError("binary decode shape mismatch")
    return result.payload, decoded


def encode_up(values: np.ndarray) -> tuple[bytes, np.ndarray]:
    result = encode_residual_compact(
        values,
        outlier_ratio=OUTLIER_RATIO,
        group_size=GROUP_BINARY,
        index_mode="rice",
        value_bits=1,
        value_scale="rms",
    )
    decoded = decode_residual_compact(result.payload)
    return result.payload, decoded


def encode_down_activation(
    values: np.ndarray,
    x_swiglu: np.ndarray,
    identity: Mapping[str, Any],
) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    codec = _activation_weighted_svd_low_rank_codec(
        values,
        rank=RANK,
        bits=FACTOR_BITS,
        X_fit=x_swiglu,
        capture_identity=identity,
        X_hold=None,
    )
    return codec.payload, np.asarray(codec.reconstruction, dtype=np.float32), dict(
        codec.metadata
    )


def encode_down_weight_space(
    values: np.ndarray,
    identity: Mapping[str, Any],
    *,
    reason: str,
) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    matrix = np.ascontiguousarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise PackError("down_proj weight-space SVD requires a matrix")
    actual = min(RANK, matrix.shape[0], matrix.shape[1])
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    left = (u[:, :actual] * s[:actual]).astype(np.float32)
    right = vt[:actual, :].astype(np.float32)
    left_body, _, left_meta = _factor_codec(left, bits=FACTOR_BITS)
    right_body, _, right_meta = _factor_codec(right, bits=FACTOR_BITS)
    header = {
        "schema": HGRAVS01_SCHEMA,
        "representation": ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "elements": int(matrix.size),
        "rank": int(actual),
        "factor_bits": int(FACTOR_BITS),
        "factor_group_size": GROUP_UNIFORM,
        "left": left_meta,
        "right": right_meta,
        "left_body_bytes": len(left_body),
        "right_body_bytes": len(right_body),
        "fit": {
            "fit": "weight_space_truncated_svd",
            "reason": reason,
            "rank": int(actual),
            "n_fit_tokens": 0,
            "requested_rank": RANK,
            "rank_clamped_to_n_fit": False,
        },
        "activation_capture": {
            "path": identity.get("path"),
            "capture_result_path": identity.get("capture_result_path"),
            "sha256": identity.get("sha256"),
            "schema": identity.get("schema"),
            "status": identity.get("status"),
            "fit_kind": "real_routed_activation_capture",
            "not_synthetic_unit_direction": True,
            "weight_space_svd_reported": True,
        },
    }
    payload = _container(MAGIC_ACT_SVD, header, left_body + right_body)
    decoded = _decode_activation_weighted_svd_low_rank_codec(payload)
    return payload, decoded, header


def encode_uniform8(values: np.ndarray) -> tuple[bytes, np.ndarray]:
    result = _uniform_codec(values, bits=8, group_size=GROUP_UNIFORM)
    decoded = _decode_uniform_codec(result.payload)
    return result.payload, decoded


class CaptureHiddens:
    """Layer-streamed router-input X from capture-index.v1. Never parses the 1.38 GB JSON."""

    def __init__(self, run_dir: Path):
        status, root, header = inspect_index(run_dir)
        if status != "ok" or root is None or header is None:
            raise PackError(f"capture index not usable under {run_dir}: {status}")
        src = header.get("source") or {}
        if src.get("sha256") != CAPTURE_SHA256:
            raise PackError(
                f"capture sha256 {src.get('sha256')} != bound {CAPTURE_SHA256}"
            )
        self.run_dir = Path(run_dir)
        self.root = root
        self.header = header
        self.layer = _load_npy(root, "layer.npy")
        self.hidden_retained = _load_npy(root, "hidden_retained.npy")
        self.elements = _load_npy(root, "elements.npy")
        self.hidden_offset = _load_npy(root, "hidden_offset.npy")
        self.path_id = _load_npy(root, "path_id.npy")
        self.path_offsets = _load_npy(root, "path_offsets.npy")
        self.path_blob = _load_npy(root, "path_blob.npy")
        self.key_layer = _load_npy(root, "key_layer.npy")
        self.key_expert = _load_npy(root, "key_expert.npy")
        self.key_offsets = _load_npy(root, "key_offsets.npy")
        self.key_row_ids = _load_npy(root, "key_row_ids.npy")
        self._key_index: dict[tuple[int, int], int] = {}
        for i in range(int(self.key_layer.size)):
            self._key_index[(int(self.key_layer[i]), int(self.key_expert[i]))] = i
        self.counts = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.int32)
        for (layer, expert), idx in self._key_index.items():
            n = int(self.key_offsets[idx + 1]) - int(self.key_offsets[idx])
            self.counts[layer, expert] = n

    def n_fit(self, layer: int, expert: int) -> int:
        return int(self.counts[layer, expert])

    def load_layer(self, layer: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Return (rid_to_x, expert_rids). Stack per expert only at encode time."""

        experts_rids: dict[int, np.ndarray] = {}
        unique: set[int] = set()
        for expert in range(N_EXPERTS):
            idx = self._key_index.get((layer, expert))
            if idx is None:
                continue
            lo = int(self.key_offsets[idx])
            hi = int(self.key_offsets[idx + 1])
            rids = np.asarray(self.key_row_ids[lo:hi], dtype=np.int32)
            if rids.size == 0:
                continue
            experts_rids[expert] = rids
            unique.update(int(r) for r in rids.tolist())
        rid_to_x: dict[int, np.ndarray] = {}

        def _load_rid(rid: int) -> tuple[int, np.ndarray]:
            if int(self.hidden_retained[rid]) == 0:
                raise PackError(f"key_row_ids cited a non-retained row {rid}")
            if int(self.layer[rid]) != layer:
                raise PackError(f"row {rid} layer {int(self.layer[rid])} != {layer}")
            rel = _decode_path(self.path_blob, self.path_offsets, int(self.path_id[rid]))
            fpath = self.run_dir / rel
            off = int(self.hidden_offset[rid])
            n_elem = int(self.elements[rid])
            if off == 0:
                x = np.fromfile(fpath, dtype="<f4")
            else:
                x = np.memmap(
                    fpath, dtype="<f4", mode="r", offset=off, shape=(n_elem,)
                ).copy()
            if x.size != n_elem or n_elem != HIDDEN:
                raise PackError(f"hidden {fpath} has {x.size} != {n_elem}/{HIDDEN}")
            return int(rid), np.ascontiguousarray(x, dtype=np.float32)

        rids = list(unique)
        if len(rids) <= 32:
            loaded = [_load_rid(r) for r in rids]
        else:
            with ThreadPoolExecutor(max_workers=16) as pool:
                loaded = list(pool.map(_load_rid, rids))
        for rid, x in loaded:
            rid_to_x[rid] = x
        return rid_to_x, experts_rids

    def expert_hidden(
        self,
        rid_to_x: Mapping[int, np.ndarray],
        expert_rids: Mapping[int, np.ndarray],
        expert: int,
    ) -> np.ndarray | None:
        rids = expert_rids.get(expert)
        if rids is None or rids.size == 0:
            return None
        return np.stack([rid_to_x[int(r)] for r in rids.tolist()], axis=0)


def warm_shard_headers(model_dir: Path, weight_map: Mapping[str, str]) -> None:
    for shard_name in sorted(set(weight_map.values())):
        shard = model_dir / shard_name
        if shard not in _HEADER_CACHE:
            _HEADER_CACHE[shard] = read_safetensors_header(shard)


def segment_id_for(name: str) -> int:
    if name == "model.embed_tokens.weight":
        return 0
    if name in {"model.norm.weight", "lm_head.weight"}:
        return N_LAYERS + 1
    layer = layer_of(name)
    if layer is None:
        raise PackError(f"no segment for {name}")
    return layer + 1


def segment_filename(seg_id: int) -> str:
    if seg_id == 0:
        return "00_embed.hq80seg"
    if seg_id == N_LAYERS + 1:
        return "99_terminal.hq80seg"
    return f"L{seg_id - 1:02d}.hq80seg"


def pack_record(
    *,
    name: str,
    payload: bytes,
    shape: list[int],
    elements: int,
    segment_id: int,
    offset: int,
    organ: int,
    codec: int,
    flags: int,
    n_fit_rows: int,
    achieved_rank: int,
    codec_bpw: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "payload": payload,
        "shape": shape,
        "elements": int(elements),
        "segment_id": int(segment_id),
        "offset": int(offset),
        "organ": int(organ),
        "codec": int(codec),
        "flags": int(flags),
        "n_fit_rows": int(n_fit_rows),
        "achieved_rank": int(achieved_rank),
        "codec_bpw": float(codec_bpw),
        "sha256": sha256_hex(payload),
        "nbytes": len(payload),
    }


def encode_named(
    name: str,
    values: np.ndarray,
    *,
    identity: Mapping[str, Any],
    x_swiglu: np.ndarray | None,
    n_fit_rows: int,
) -> dict[str, Any]:
    organ, codec = classify(name)
    flags = 0
    achieved_rank = 0
    if codec == CODEC_BINARY:
        payload, decoded = encode_gate(values)
    elif codec == CODEC_RESIDUAL:
        payload, decoded = encode_up(values)
    elif codec == CODEC_HGRAVS01:
        if n_fit_rows >= 1:
            if x_swiglu is None:
                raise PackError(f"{name}: activation-weighted down_proj missing X")
            payload, decoded, meta = encode_down_activation(values, x_swiglu, identity)
            achieved_rank = int(meta.get("rank") or RANK)
            flags |= FLAG_ACTIVATION_WEIGHTED
            if n_fit_rows < 512:
                flags |= FLAG_GRAM_RANKDEF
        else:
            payload, decoded, meta = encode_down_weight_space(
                values,
                identity,
                reason="never_routed_in_bound_25258_token_capture",
            )
            achieved_rank = int(meta.get("rank") or RANK)
            flags |= FLAG_WEIGHT_SPACE
    else:
        payload, decoded = encode_uniform8(values)
        flags |= FLAG_SENSITIVE
    bpw = 8.0 * len(payload) / max(int(values.size), 1)
    return {
        "payload": payload,
        "decoded": decoded,
        "organ": organ,
        "codec": codec,
        "flags": flags,
        "n_fit_rows": int(n_fit_rows),
        "achieved_rank": int(achieved_rank),
        "codec_bpw": float(bpw),
        "cosine": cosine_flat(values, decoded),
    }


def write_catalog(
    path: Path,
    records: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> bytes:
    names = [r["name"] for r in records]
    name_blob = bytearray()
    offs: list[int] = []
    for name in names:
        raw = name.encode("utf-8")
        offs.append(len(name_blob))
        name_blob.extend(raw)
    table = bytearray()
    for rec, off in zip(records, offs):
        raw_name = rec["name"].encode("utf-8")
        dims = [0, 0, 0, 0]
        shape = rec["shape"]
        if len(shape) > 4:
            raise PackError(f"{rec['name']} rank {len(shape)} exceeds catalog")
        for i, d in enumerate(shape):
            dims[i] = int(d)
        digest = bytes.fromhex(rec["sha256"])
        if len(digest) != 32:
            raise PackError("catalog sha256 is not 32 bytes")
        rec_bytes = struct.pack(
            "<IHBBBB",
            off,
            len(raw_name),
            rec["codec"],
            rec["organ"],
            len(shape),
            0,
        )
        rec_bytes += b"\x00\x00"
        rec_bytes += struct.pack(
            "<IIIIQHHQQ32sIIf",
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            rec["elements"],
            rec["segment_id"],
            rec["achieved_rank"],
            rec["offset"],
            rec["nbytes"],
            digest,
            rec["flags"],
            rec["n_fit_rows"],
            float(rec["codec_bpw"]),
        )
        if len(rec_bytes) > RECORD_SIZE:
            raise PackError(f"record packed {len(rec_bytes)} > {RECORD_SIZE}")
        rec_bytes = rec_bytes + b"\x00" * (RECORD_SIZE - len(rec_bytes))
        table.extend(rec_bytes)
    seg_blob = bytearray()
    for seg in segments:
        name = str(seg["filename"]).encode("utf-8")
        digest = bytes.fromhex(seg["sha256"])
        seg_blob.extend(
            struct.pack(
                "<HHQ32s",
                int(seg["id"]),
                len(name),
                int(seg["bytes"]),
                digest,
            )
        )
        seg_blob.extend(name)
    blob = (
        CATALOG_MAGIC
        + struct.pack(
            "<IIIIII",
            CATALOG_VERSION,
            len(records),
            len(segments),
            0,
            len(name_blob),
            0,
        )
        + bytes(seg_blob)
        + bytes(table)
        + bytes(name_blob)
    )
    path.write_bytes(blob)
    return blob


def read_catalog(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw[:8] != CATALOG_MAGIC:
        raise PackError(f"catalog magic {raw[:8]!r} is not HQ80M15")
    version, n_tensors, n_segments, flags, name_blob_bytes, _reserved = struct.unpack_from(
        "<IIIIII", raw, 8
    )
    if version != CATALOG_VERSION:
        raise PackError(f"catalog version {version}")
    cursor = 32
    segments = []
    for _ in range(n_segments):
        seg_id, name_len, nbytes = struct.unpack_from("<HHQ", raw, cursor)
        digest = raw[cursor + 12 : cursor + 44]
        cursor += 44
        filename = raw[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        segments.append(
            {
                "id": seg_id,
                "filename": filename,
                "bytes": nbytes,
                "sha256": digest.hex(),
            }
        )
    table = raw[cursor : cursor + n_tensors * RECORD_SIZE]
    cursor += n_tensors * RECORD_SIZE
    name_blob = raw[cursor : cursor + name_blob_bytes]
    records = []
    for i in range(n_tensors):
        rec = table[i * RECORD_SIZE : (i + 1) * RECORD_SIZE]
        name_off, name_len, codec, organ, ndim, _pad = struct.unpack_from(
            "<IHBBBB", rec, 0
        )
        (
            d0,
            d1,
            d2,
            d3,
            elements,
            segment_id,
            achieved_rank,
            offset,
            nbytes,
            digest,
            rec_flags,
            n_fit_rows,
            codec_bpw,
        ) = struct.unpack_from("<IIIIQHHQQ32sIIf", rec, 12)
        name = name_blob[name_off : name_off + name_len].decode("utf-8")
        dims = [d0, d1, d2, d3][:ndim]
        records.append(
            {
                "name": name,
                "codec": codec,
                "organ": organ,
                "shape": dims,
                "elements": elements,
                "segment_id": segment_id,
                "achieved_rank": achieved_rank,
                "offset": offset,
                "nbytes": nbytes,
                "sha256": digest.hex(),
                "flags": rec_flags,
                "n_fit_rows": n_fit_rows,
                "codec_bpw": codec_bpw,
            }
        )
    return {
        "version": version,
        "flags": flags,
        "segments": segments,
        "records": records,
    }


def read_revalidation(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PackError("revalidation is not an object")
    return raw


def organ_byte_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {
        "routed_gate_proj": {"bytes": 0, "elements": 0, "tensors": 0},
        "routed_up_proj": {"bytes": 0, "elements": 0, "tensors": 0},
        "routed_down_proj": {"bytes": 0, "elements": 0, "tensors": 0},
        "nonexpert_8bit": {"bytes": 0, "elements": 0, "tensors": 0},
    }
    key = {
        ORGAN_GATE: "routed_gate_proj",
        ORGAN_UP: "routed_up_proj",
        ORGAN_DOWN: "routed_down_proj",
        ORGAN_NONEXPERT: "nonexpert_8bit",
    }
    for rec in records:
        slot = buckets[key[rec["organ"]]]
        slot["bytes"] += rec["nbytes"]
        slot["elements"] += rec["elements"]
        slot["tensors"] += 1
    for slot in buckets.values():
        elems = max(int(slot["elements"]), 1)
        slot["physical_bpw"] = 8.0 * slot["bytes"] / elems
    return buckets


class Packer:
    def __init__(
        self,
        *,
        root: Path,
        model_dir: Path,
        capture_dir: Path,
        revalidation_path: Path,
        workers: int,
        max_layers: int | None = None,
        max_experts: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.model_dir = Path(model_dir)
        self.capture_dir = Path(capture_dir)
        self.revalidation_path = Path(revalidation_path)
        self.workers = max(1, int(workers))
        self.max_layers = max_layers
        self.max_experts = max_experts
        self.identity = capture_identity(self.capture_dir)
        self.weight_map = load_weight_map(self.model_dir)
        if (
            self.max_layers is None
            and self.max_experts is None
            and len(self.weight_map) != EXPECTED_TENSOR_COUNT
        ):
            raise PackError(
                f"source index has {len(self.weight_map)} tensors, "
                f"expected {EXPECTED_TENSOR_COUNT}"
            )
        self.names = execution_order(self.weight_map)
        if self.max_layers is not None:
            keep = set()
            for name in self.names:
                layer = layer_of(name)
                if layer is None or layer < self.max_layers:
                    if self.max_experts is not None:
                        m = _EXPERT_RE.match(name)
                        if m and int(m.group(2)) >= self.max_experts:
                            continue
                    keep.add(name)
            self.names = [n for n in self.names if n in keep]
        self.captures = CaptureHiddens(self.capture_dir)
        self.seg_dir = self.root / "segments"
        self.records: list[dict[str, Any]] = []
        self.segments: list[dict[str, Any]] = []
        self.fit_rows = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.uint16)
        self.fit_kind = np.full((N_LAYERS, N_EXPERTS), 255, dtype=np.uint8)
        self.cosines: list[float] = []
        self.lock = threading.Lock()

    def census(self) -> dict[str, Any]:
        counts = self.captures.counts.reshape(-1)
        return {
            "n_pairs": int(counts.size),
            "n_zero": int((counts == 0).sum()),
            "n_lt_160": int((counts < 160).sum()),
            "n_ge_160": int((counts >= 160).sum()),
            "n_lt_512": int((counts < 512).sum()),
            "n_ge_512": int((counts >= 512).sum()),
            "min": int(counts.min()) if counts.size else 0,
            "p50": int(np.median(counts)) if counts.size else 0,
            "max": int(counts.max()) if counts.size else 0,
            "mean": float(counts.mean()) if counts.size else 0.0,
        }

    def _load(self, name: str) -> np.ndarray:
        return np.ascontiguousarray(
            load_tensor(self.model_dir, self.weight_map, name), dtype=np.float32
        )

    def _pack_one(
        self,
        name: str,
        values: np.ndarray,
        x_swiglu: np.ndarray | None,
        n_fit_rows: int,
    ) -> dict[str, Any]:
        encoded = encode_named(
            name,
            values,
            identity=self.identity,
            x_swiglu=x_swiglu,
            n_fit_rows=n_fit_rows,
        )
        decoded = encoded.pop("decoded")
        if decoded.shape != values.shape:
            raise PackError(f"{name}: decode shape {decoded.shape}")
        encoded["shape"] = [int(x) for x in decoded.shape]
        encoded["elements"] = int(decoded.size)
        del decoded
        return encoded

    def pack_nonexpert_group(self, names: list[str], seg_id: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        for name in names:
            values = self._load(name)
            encoded = self._pack_one(name, values, None, 0)
            rec = pack_record(
                name=name,
                payload=encoded["payload"],
                shape=encoded["shape"],
                elements=encoded["elements"],
                segment_id=seg_id,
                offset=offset,
                organ=encoded["organ"],
                codec=encoded["codec"],
                flags=encoded["flags"],
                n_fit_rows=0,
                achieved_rank=0,
                codec_bpw=encoded["codec_bpw"],
            )
            rec["_cosine"] = encoded["cosine"]
            del encoded
            rows.append(rec)
            offset += rec["nbytes"]
        return rows

    def pack_layer_experts(
        self,
        layer: int,
        rid_to_x: Mapping[int, np.ndarray],
        expert_rids: Mapping[int, np.ndarray],
        seg_id: int,
        start_offset: int,
    ) -> list[dict[str, Any]]:
        experts = list(range(N_EXPERTS))
        if self.max_experts is not None:
            experts = [e for e in experts if e < self.max_experts]

        def job(expert: int) -> tuple[int, dict[str, dict[str, Any]], int, int]:
            g_name = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
            u_name = f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"
            d_name = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
            gate = self._load(g_name)
            up = self._load(u_name)
            down = self._load(d_name)
            x_h = self.captures.expert_hidden(rid_to_x, expert_rids, expert)
            n_fit = 0 if x_h is None else int(x_h.shape[0])
            x_s = None
            if n_fit >= 1:
                x_s = post_swiglu(x_h, gate, up)
                if x_s.shape[1] != MOE_INTERMEDIATE:
                    raise PackError(f"swiglu width {x_s.shape} at L{layer}.E{expert}")
            g = self._pack_one(g_name, gate, None, 0)
            u = self._pack_one(u_name, up, None, 0)
            d = self._pack_one(d_name, down, x_s, n_fit)
            return expert, {g_name: g, u_name: u, d_name: d}, n_fit, (
                FIT_WEIGHT_SPACE
                if n_fit == 0
                else (FIT_AW_R160 if n_fit >= 160 else FIT_AW_RANKDEF)
            )

        if self.workers <= 1:
            results = [job(e) for e in experts]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futs = {pool.submit(job, e): e for e in experts}
                for fut in as_completed(futs):
                    results.append(fut.result())
        results.sort(key=lambda item: item[0])
        rows: list[dict[str, Any]] = []
        offset = start_offset
        for expert, encoded_by_name, n_fit, kind in results:
            self.fit_rows[layer, expert] = min(n_fit, 65535)
            self.fit_kind[layer, expert] = kind
            for name in (
                f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
            ):
                enc = encoded_by_name[name]
                rec = pack_record(
                    name=name,
                    payload=enc["payload"],
                    shape=enc["shape"],
                    elements=enc["elements"],
                    segment_id=seg_id,
                    offset=offset,
                    organ=enc["organ"],
                    codec=enc["codec"],
                    flags=enc["flags"],
                    n_fit_rows=enc["n_fit_rows"],
                    achieved_rank=enc["achieved_rank"],
                    codec_bpw=enc["codec_bpw"],
                )
                rec["_cosine"] = enc["cosine"]
                rows.append(rec)
                offset += rec["nbytes"]
        return rows

    def _segment_sidecar(self, seg_id: int) -> Path:
        return self.seg_dir / f"{segment_filename(seg_id)}.records.json"

    def _restore_fit(self, rec: Mapping[str, Any]) -> None:
        m = _EXPERT_RE.match(str(rec.get("name") or ""))
        if not m or m.group(3) != "down":
            return
        layer = int(m.group(1))
        expert = int(m.group(2))
        flags = int(rec.get("flags") or 0)
        self.fit_rows[layer, expert] = min(int(rec.get("n_fit_rows") or 0), 65535)
        if flags & FLAG_WEIGHT_SPACE:
            self.fit_kind[layer, expert] = FIT_WEIGHT_SPACE
        elif flags & FLAG_ACTIVATION_WEIGHTED:
            self.fit_kind[layer, expert] = (
                FIT_AW_RANKDEF if flags & FLAG_GRAM_RANKDEF else FIT_AW_R160
            )

    def try_resume_segment(self, seg_id: int, expected_names: list[str]) -> dict[str, Any] | None:
        filename = segment_filename(seg_id)
        path = self.seg_dir / filename
        side = self._segment_sidecar(seg_id)
        if not path.is_file() or not side.is_file():
            return None
        meta = json.loads(side.read_text(encoding="utf-8"))
        if meta.get("sha256") != sha256_file(path):
            return None
        rows = meta.get("records") or []
        names = [r["name"] for r in rows]
        if names != expected_names:
            return None
        return meta

    def write_segment(self, seg_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
        filename = segment_filename(seg_id)
        path = self.seg_dir / filename
        hasher = hashlib.sha256()
        with path.open("wb") as handle:
            for rec in rows:
                payload = rec.pop("payload")
                if rec["offset"] != handle.tell():
                    raise PackError(
                        f"{rec['name']}: offset {rec['offset']} != {handle.tell()}"
                    )
                handle.write(payload)
                hasher.update(payload)
        digest = hasher.hexdigest()
        nbytes = path.stat().st_size
        expected = sum(r["nbytes"] for r in rows)
        if nbytes != expected:
            raise PackError(f"{filename}: size {nbytes} != {expected}")
        meta = {
            "id": seg_id,
            "filename": filename,
            "path": str(path),
            "bytes": nbytes,
            "sha256": digest,
            "tensor_count": len(rows),
            "records": [{k: v for k, v in r.items() if k != "payload"} for r in rows],
        }
        slim = []
        for rec in rows:
            item = {k: v for k, v in rec.items() if k != "payload"}
            slim.append(item)
        self._segment_sidecar(seg_id).write_text(
            json.dumps(
                {
                    "id": seg_id,
                    "filename": filename,
                    "bytes": nbytes,
                    "sha256": digest,
                    "records": slim,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "id": seg_id,
            "filename": filename,
            "path": str(path),
            "bytes": nbytes,
            "sha256": digest,
            "tensor_count": len(rows),
        }

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        require_disk(self.root, "start")
        self.root.mkdir(parents=True, exist_ok=True)
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "FORMAT.md").write_text(
            (REPO_ROOT / "docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        warm_shard_headers(self.model_dir, self.weight_map)
        census = self.census()
        print(f"[q80-pack] n_fit census {json.dumps(census)}", flush=True)
        print(
            f"[q80-pack] {len(self.names)} tensors, workers={self.workers}",
            flush=True,
        )

        by_seg: dict[int, list[str]] = defaultdict(list)
        for name in self.names:
            by_seg[segment_id_for(name)].append(name)

        for seg_id in sorted(by_seg):
            require_disk(self.root, f"segment {seg_id}")
            names = by_seg[seg_id]
            t0 = time.perf_counter()
            resumed = self.try_resume_segment(seg_id, names)
            if resumed is not None:
                self.segments.append(
                    {
                        "id": seg_id,
                        "filename": resumed.get("filename") or segment_filename(seg_id),
                        "path": str(self.seg_dir / segment_filename(seg_id)),
                        "bytes": int(resumed.get("bytes") or 0)
                        or int((self.seg_dir / segment_filename(seg_id)).stat().st_size),
                        "sha256": resumed["sha256"],
                        "tensor_count": len(resumed["records"]),
                    }
                )
                for rec in resumed["records"]:
                    self.cosines.append(float(rec.get("cosine") or rec.get("_cosine") or 0.0))
                    self._restore_fit(rec)
                    self.records.append({k: v for k, v in rec.items() if k not in {"_cosine", "cosine", "payload"}})
                print(
                    f"[q80-pack] resume {segment_filename(seg_id)} "
                    f"tensors={len(resumed['records'])} "
                    f"wall_s={time.perf_counter() - t0:.1f}",
                    flush=True,
                )
                continue
            if seg_id == 0 or seg_id == N_LAYERS + 1:
                rows = self.pack_nonexpert_group(names, seg_id)
            else:
                layer = seg_id - 1
                prefix = [n for n in names if ".mlp.experts." not in n]
                rows = self.pack_nonexpert_group(prefix, seg_id)
                start = sum(r["nbytes"] for r in rows)
                print(
                    f"[q80-pack] L{layer:02d} loading hidden X "
                    f"(prefix {len(prefix)} tensors)",
                    flush=True,
                )
                rid_to_x, expert_rids = self.captures.load_layer(layer)
                rows.extend(
                    self.pack_layer_experts(
                        layer, rid_to_x, expert_rids, seg_id, start
                    )
                )
                del rid_to_x, expert_rids
                gc.collect()
            seg = self.write_segment(seg_id, rows)
            self.segments.append(seg)
            for rec in rows:
                self.cosines.append(float(rec.pop("_cosine")))
                self.records.append(rec)
            print(
                f"[q80-pack] segment {seg['filename']} "
                f"tensors={seg['tensor_count']} bytes={seg['bytes']} "
                f"wall_s={time.perf_counter() - t0:.1f}",
                flush=True,
            )

        if self.max_layers is None and self.max_experts is None:
            if len(self.records) != EXPECTED_TENSOR_COUNT:
                raise PackError(
                    f"packed {len(self.records)} tensors, expected {EXPECTED_TENSOR_COUNT}"
                )

        catalog_path = self.root / "catalog.hq80m15"
        catalog_bytes = write_catalog(catalog_path, self.records, self.segments)
        fit_rows_path = self.root / "fit_rows.u16le"
        fit_kind_path = self.root / "fit_kind.u8"
        fit_rows_path.write_bytes(self.fit_rows.astype("<u2").tobytes())
        fit_kind_path.write_bytes(self.fit_kind.tobytes())

        payload_bytes = sum(r["nbytes"] for r in self.records)
        catalog_nbytes = catalog_path.stat().st_size
        fit_bytes = fit_rows_path.stat().st_size + fit_kind_path.stat().st_size
        format_bytes = (self.root / "FORMAT.md").stat().st_size
        side_bytes = catalog_nbytes + fit_bytes + format_bytes
        elements = sum(r["elements"] for r in self.records)
        organ = organ_byte_breakdown(self.records)
        expert_elems = (
            organ["routed_gate_proj"]["elements"]
            + organ["routed_up_proj"]["elements"]
            + organ["routed_down_proj"]["elements"]
        )
        expert_bytes = (
            organ["routed_gate_proj"]["bytes"]
            + organ["routed_up_proj"]["bytes"]
            + organ["routed_down_proj"]["bytes"]
        )
        nonexpert_elems = organ["nonexpert_8bit"]["elements"]
        nonexpert_bytes = organ["nonexpert_8bit"]["bytes"]
        expert_bpw = 8.0 * expert_bytes / max(expert_elems, 1)
        nonexpert_bpw = 8.0 * nonexpert_bytes / max(nonexpert_elems, 1)
        design_complete = F_EXPERT * expert_bpw + F_NONEXPERT * nonexpert_bpw

        reval = read_revalidation(self.revalidation_path)
        reval_seal = str(reval.get("seal_sha256") or "")
        source_audit = str(reval.get("source_audit_seal_sha256") or "")
        source_revision = str(reval.get("source_revision") or "")

        written = self.fit_kind != 255
        n_zero = int(((self.fit_kind == FIT_WEIGHT_SPACE) & written).sum())
        n_rankdef = int(((self.fit_kind == FIT_AW_RANKDEF) & written).sum())
        n_aw = int(((self.fit_kind == FIT_AW_R160) & written).sum())

        manifest_path = self.root / (
            f"{ARTIFACT_PREFIX}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
        )
        terminal_path = self.root / (
            f"{ARTIFACT_PREFIX}_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json"
        )

        def make_manifest(billed: int) -> dict[str, Any]:
            total = payload_bytes + side_bytes + billed
            physical = (total * 8.0) / max(elements, 1)
            return {
                "schema": SCHEMA,
                "status": CANDIDATE_STATUS,
                "branch_id": BRANCH_ID,
                "model_id": MODEL_ID,
                "artifact_prefix": ARTIFACT_PREFIX,
                "source_body_audit_seal_sha256": source_audit,
                "source_revalidation_receipt_path": str(self.revalidation_path),
                "source_revalidation_receipt_seal_sha256": reval_seal,
                "source": {
                    "repository": "Qwen/Qwen3-Coder-Next",
                    "model_dir": str(self.model_dir),
                    "tensor_count": len(self.records),
                    "revision": source_revision,
                },
                "representation": {
                    "family": "mixed_per_component",
                    "gate_proj": "binary_group_128",
                    "up_proj": "binary_plus_rice_q1_rms_residual_2pct",
                    "down_proj": "hgravs01_r160_b3_activation_weighted_post_swiglu",
                    "nonexpert": "uniform_q8_group64",
                    "sensitive_3pct_untouched": True,
                    "identity": (
                        "complete_bpw = 0.97032*expert_bpw + 0.02968*nonexpert_bpw"
                    ),
                },
                "activation_capture": self.identity,
                "down_proj_fit": {
                    "requested_rank": RANK,
                    "rank_clamped_to_n_fit": False,
                    "n_activation_weighted_r160": n_aw,
                    "n_activation_weighted_gram_rank_deficient": n_rankdef,
                    "n_weight_space_never_routed": n_zero,
                    "n_fit_census": census,
                    "note": (
                        "Rank is never clamped to n_fit_rows. Never-routed "
                        "experts (no X) use weight-space truncated SVD at "
                        "r160, same HGRAVS01 body; this is reported, not silent."
                    ),
                },
                "complete_physical_bpw_ledger": {
                    "source_weight_elements": elements,
                    "tensor_payload_bytes": payload_bytes,
                    "catalog_bytes": catalog_nbytes,
                    "fit_table_bytes": fit_bytes,
                    "format_bytes": format_bytes,
                    "manifest_bytes_billed": billed,
                    "side_table_bytes": side_bytes + billed,
                    "all_required_weight_artifact_bytes": total,
                    "complete_physical_bpw": physical,
                    "payload_only_physical_bpw": 8.0 * payload_bytes / max(elements, 1),
                    "expert_physical_bpw": expert_bpw,
                    "nonexpert_physical_bpw": nonexpert_bpw,
                    "design_identity_complete_bpw": design_complete,
                    "threshold_bpw": 1.5,
                    "passes_storage_threshold": physical <= 1.5,
                    "organ_breakdown": organ,
                },
                "quality_summary": {
                    "mean_component_cosine": (
                        float(sum(self.cosines) / len(self.cosines))
                        if self.cosines
                        else 0.0
                    ),
                    "quality_rows_with_cosine": len(self.cosines),
                    "verdict": "PACKED_NOT_A_COHERENCE_CLAIM",
                },
                "catalog": {
                    "path": str(catalog_path),
                    "sha256": sha256_hex(catalog_bytes),
                    "bytes": catalog_nbytes,
                    "format": "hawking.q80.mixed_catalog.hq80m15.v1",
                },
                "segments": [
                    {
                        "id": s["id"],
                        "path": s["path"],
                        "filename": s["filename"],
                        "bytes": s["bytes"],
                        "sha256": s["sha256"],
                        "tensor_count": s["tensor_count"],
                    }
                    for s in self.segments
                ],
                "claim_boundary": {
                    "artifact_packed": True,
                    "decode_kernel_exists": False,
                    "coherence_generation_tested": False,
                    "packing_is_not_a_le_1_5_coherence_claim": True,
                    "generation_is_the_gate": True,
                    "sensitive_3pct_untouched": True,
                    "did_not_mutate_source": True,
                    "did_not_write_metal_kernels": True,
                    "did_not_create_giant_json_index": True,
                },
            }

        billed = 0
        sealed = None
        for _ in range(6):
            sealed = seal(make_manifest(billed))
            actual = len(
                (json.dumps(sealed, indent=2, sort_keys=False) + "\n").encode("utf-8")
            )
            # write_pretty matches Python json indent=2
            rendered = (json.dumps(sealed, indent=2) + "\n").encode("utf-8")
            actual = len(rendered)
            if actual == billed:
                break
            billed = actual
        else:
            raise PackError("manifest byte-billing did not converge")
        assert sealed is not None
        manifest_path.write_bytes((json.dumps(sealed, indent=2) + "\n").encode("utf-8"))
        if manifest_path.stat().st_size != billed:
            # last-chance reseal against the bytes that actually landed
            billed = manifest_path.stat().st_size
            sealed = seal(make_manifest(billed))
            manifest_path.write_bytes(
                (json.dumps(sealed, indent=2) + "\n").encode("utf-8")
            )

        terminal = seal(
            {
                "schema": TERMINAL_SCHEMA,
                "status": "EARNED_COMPLETE_PHYSICAL_MIXED_REPRESENTATION_PACKED_NOT_GENERATED",
                "binding": {
                    "model_id": MODEL_ID,
                    "artifact_prefix": ARTIFACT_PREFIX,
                    "manifest_schema": SCHEMA,
                    "branch_id": BRANCH_ID,
                    "source_body_audit_seal_sha256": source_audit,
                    "source_revalidation_receipt_path": str(self.revalidation_path),
                    "source_revalidation_receipt_seal_sha256": reval_seal,
                    "source_revision": source_revision,
                    "activation_capture_sha256": CAPTURE_SHA256,
                },
                "candidate": {
                    "manifest_path": str(manifest_path),
                    "manifest_seal_sha256": sealed["seal_sha256"],
                    "catalog_path": str(catalog_path),
                    "catalog_sha256": sha256_hex(catalog_bytes),
                    "all_required_weight_artifact_bytes": sealed[
                        "complete_physical_bpw_ledger"
                    ]["all_required_weight_artifact_bytes"],
                    "complete_physical_bpw": sealed["complete_physical_bpw_ledger"][
                        "complete_physical_bpw"
                    ],
                    "tensor_count": len(self.records),
                },
                "claim_boundary": sealed["claim_boundary"],
            }
        )
        terminal_path.write_text(json.dumps(terminal, indent=2) + "\n", encoding="utf-8")

        wall = time.perf_counter() - started
        ledger = sealed["complete_physical_bpw_ledger"]
        report = {
            "status": "ok",
            "wall_s": wall,
            "root": str(self.root),
            "manifest_path": str(manifest_path),
            "terminal_path": str(terminal_path),
            "manifest_seal_sha256": sealed["seal_sha256"],
            "terminal_seal_sha256": terminal["seal_sha256"],
            "tensor_count": len(self.records),
            "complete_physical_bpw": ledger["complete_physical_bpw"],
            "design_identity_complete_bpw": ledger["design_identity_complete_bpw"],
            "expert_physical_bpw": ledger["expert_physical_bpw"],
            "nonexpert_physical_bpw": ledger["nonexpert_physical_bpw"],
            "all_required_weight_artifact_bytes": ledger[
                "all_required_weight_artifact_bytes"
            ],
            "tensor_payload_bytes": payload_bytes,
            "organ_breakdown": organ,
            "down_proj_fit": sealed["down_proj_fit"],
            "mean_component_cosine": sealed["quality_summary"]["mean_component_cosine"],
            "n_fit_census": census,
            "claim_boundary": sealed["claim_boundary"],
        }
        (self.root / "PACK_REPORT.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--capture", type=Path, default=CAPTURE_DIR)
    p.add_argument("--revalidation", type=Path, default=REVALIDATION)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--max-experts", type=int, default=None)
    p.add_argument(
        "--census-only",
        action="store_true",
        help="print n_fit census and exit (no pack)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model_dir.is_dir():
        raise SystemExit(f"model dir missing: {args.model_dir}")
    if args.census_only:
        cap = CaptureHiddens(args.capture)
        print(json.dumps({"n_fit_census": {
            "n_pairs": int(cap.counts.size),
            "n_zero": int((cap.counts == 0).sum()),
            "n_lt_160": int((cap.counts < 160).sum()),
            "n_ge_160": int((cap.counts >= 160).sum()),
            "n_lt_512": int((cap.counts < 512).sum()),
            "n_ge_512": int((cap.counts >= 512).sum()),
            "min": int(cap.counts.min()),
            "p50": int(np.median(cap.counts)),
            "max": int(cap.counts.max()),
            "mean": float(cap.counts.mean()),
        }}, indent=2))
        return 0
    packer = Packer(
        root=args.root,
        model_dir=args.model_dir,
        capture_dir=args.capture,
        revalidation_path=args.revalidation,
        workers=args.workers,
        max_layers=args.max_layers,
        max_experts=args.max_experts,
    )
    packer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
