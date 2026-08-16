#!/usr/bin/env python3
"""Pack Qwen3.8 under 1.5 BPW and materialize a native Q4 generate catalog.

Recipe (dense SwiGLU, real mixed-2p0 MLP, no MoE machinery):

  mlp.gate_proj     HGRAVB01 binary_g128          (copied from mixed-2p0-v1)
  mlp.up_proj       HGRAVR02 rice_q1_rms @ 2%     (copied from mixed-2p0-v1)
  mlp.down_proj     HGRAVS01 r160_b3 on real X    (copied from mixed-2p0-v1)
  attention GEMVs   rice_q1_rms @ 2% from BF16    (this lane)
  embed + lm_head   HQ30UQ4 group-64              (oracle Q4, generate-id later)
  norms / A_log / dt_bias / conv1d   f32          (same as the sealed Q4 catalog)

The generate vehicle is a hard-linked copy of uniform-q4-v1 with overwritten
Q4 files of the *reconstructed* mixed/rice weights. hybrid_greedy only speaks
HQ30UQ4 + f32v2; TPS is projected from packed bytes, not from this vehicle.

Does not mutate the BF16 source or the sealed uniform-q4-v1 files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

MAIN = Path("/Users/scammermike/Downloads/hawking")
sys.path.insert(0, str(MAIN))

from lab.operators.ascension_dual_gravity_worker import (  # noqa: E402
    GROUP_BINARY,
    MAGIC_ACT_SVD,
    MAGIC_BINARY,
    _decode_activation_weighted_svd_low_rank_codec,
    _parse_container,
)
from lab.operators.qwen30b_gravity_pack import (  # noqa: E402
    load_tensor,
    load_weight_map,
)
from lab.operators.residual_compact_codec import (  # noqa: E402
    _rebuild_binary,
    decode_residual_compact,
    encode_residual_compact,
)

MODEL_DIR = MAIN / "workspace/campaign/records/runs/qwen38-27b/bf16"
Q4_ORACLE = MAIN / "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1"
MIXED_2P0 = MAIN / "workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1"
DEFAULT_ROOT = MAIN / "workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1"

N_LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
KEY_HEADS = 16
VALUE_HEADS = 48
KEY_DIM = 128
VALUE_DIM = 128
VALUES_PER_KEY = 3
QKV_ROWS = 10240
Z_ROWS = 6144
A_ROWS = 48
B_ROWS = 48
QKVZ_ROWS = 16384
BA_ROWS = 96
Q_ROWS = 12288
KV_ROWS = 1024
O_ROWS = 5120
O_COLS = 6144

CATALOG_MAGIC = b"HQ38M20\0"
CATALOG_VERSION = 1
RECORD_SIZE = 128
CODEC_BINARY = 0
CODEC_RESIDUAL = 1
CODEC_HGRAVS01 = 2
CODEC_UNIFORM4 = 3
CODEC_F32 = 4
ORGAN_GATE = 0
ORGAN_UP = 1
ORGAN_DOWN = 2
ORGAN_ATTN = 3
ORGAN_EMB = 4
ORGAN_HEAD = 5
ORGAN_SMALL = 6

Q4_MAGIC = b"HQ30UQ4\0"
Q4_VERSION = 1
Q4_GROUP = 64


class PackError(RuntimeError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_filename(name: str, ext: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest() + f".{ext}"


def is_gqa_layer(layer: int) -> bool:
    return int(layer) % 4 == 3


def log(msg: str) -> None:
    print(f"[sub15] {msg}", flush=True)


def cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(left @ right)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    if den <= 1e-12:
        return 1.0 if num == 0.0 else 0.0
    return num / den


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


def decode_mixed_payload(codec: int, payload: bytes) -> np.ndarray:
    if codec == CODEC_BINARY:
        return decode_binary(payload)
    if codec == CODEC_RESIDUAL:
        return decode_residual_compact(payload)
    if codec == CODEC_HGRAVS01:
        return np.ascontiguousarray(
            _decode_activation_weighted_svd_low_rank_codec(payload), dtype=np.float32
        )
    if codec == CODEC_UNIFORM4:
        from lab.operators.ascension_dual_gravity_worker import _decode_uniform_codec

        return np.ascontiguousarray(_decode_uniform_codec(payload), dtype=np.float32)
    raise PackError(f"unknown mixed codec {codec}")


def encode_rice(values: np.ndarray) -> tuple[bytes, np.ndarray]:
    result = encode_residual_compact(
        values,
        outlier_ratio=0.02,
        group_size=GROUP_BINARY,
        index_mode="rice",
        value_bits=1,
        value_scale="rms",
    )
    decoded = decode_residual_compact(result.payload)
    return result.payload, np.ascontiguousarray(decoded, dtype=np.float32)


def pack_hq30uq4(values: np.ndarray, shape: list[int]) -> bytes:
    """Match hawking-core pack_uniform_q4_group64 (HQ30UQ4 v1)."""
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if int(flat.size) != int(np.prod(shape)):
        raise PackError(f"q4 value count {flat.size} != shape product {np.prod(shape)}")
    if not np.isfinite(flat).all():
        raise PackError("q4 source is non-finite")
    n = int(flat.size)
    groups = (n + Q4_GROUP - 1) // Q4_GROUP
    padded = np.zeros(groups * Q4_GROUP, dtype=np.float32)
    padded[:n] = flat
    grouped = padded.reshape(groups, Q4_GROUP)
    max_abs = np.max(np.abs(grouped), axis=1)
    scale = (max_abs / 7.0).astype(np.float16)
    scale_f = scale.astype(np.float32)
    inv = np.where(scale_f == 0.0, 0.0, 1.0 / scale_f)
    quant = np.rint(grouped * inv[:, None]).clip(-8.0, 7.0).astype(np.int16)
    code = (quant + 8).astype(np.uint8)
    even = code[:, 0::2]
    odd = code[:, 1::2]
    packed = even | (np.left_shift(odd, 4))
    header = bytearray()
    header += Q4_MAGIC
    header += struct.pack("<I", Q4_VERSION)
    header += struct.pack("<I", Q4_GROUP)
    header += struct.pack("<H", len(shape))
    header += struct.pack("<H", 0)
    header += struct.pack("<Q", n)
    header += struct.pack("<I", 0)
    for dim in shape:
        header += struct.pack("<I", int(dim))
    return bytes(header) + scale.tobytes() + packed.tobytes()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def fuse_in_proj_qkvz(qkv: np.ndarray, z: np.ndarray) -> np.ndarray:
    qkv = np.ascontiguousarray(qkv, dtype=np.float32).reshape(QKV_ROWS, HIDDEN)
    z = np.ascontiguousarray(z, dtype=np.float32).reshape(Z_ROWS, HIDDEN)
    value_rows = VALUES_PER_KEY * VALUE_DIM  # 384
    qkvz_per_key = KEY_DIM * 2 + value_rows * 2  # 1024
    fused = np.empty((QKVZ_ROWS, HIDDEN), dtype=np.float32)
    for kh in range(KEY_HEADS):
        dst = kh * qkvz_per_key
        q_src = kh * KEY_DIM
        k_src = KEY_HEADS * KEY_DIM + kh * KEY_DIM
        v_src = KEY_HEADS * KEY_DIM * 2 + kh * value_rows
        z_src = kh * value_rows
        fused[dst : dst + KEY_DIM] = qkv[q_src : q_src + KEY_DIM]
        fused[dst + KEY_DIM : dst + 2 * KEY_DIM] = qkv[k_src : k_src + KEY_DIM]
        fused[dst + 2 * KEY_DIM : dst + 2 * KEY_DIM + value_rows] = qkv[
            v_src : v_src + value_rows
        ]
        fused[dst + 2 * KEY_DIM + value_rows : dst + qkvz_per_key] = z[
            z_src : z_src + value_rows
        ]
    return fused


def fuse_in_proj_ba(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(B_ROWS, HIDDEN)
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(A_ROWS, HIDDEN)
    fused = np.empty((BA_ROWS, HIDDEN), dtype=np.float32)
    for kh in range(KEY_HEADS):
        src = kh * VALUES_PER_KEY
        dst = kh * (VALUES_PER_KEY * 2)
        fused[dst : dst + VALUES_PER_KEY] = b[src : src + VALUES_PER_KEY]
        fused[dst + VALUES_PER_KEY : dst + 2 * VALUES_PER_KEY] = a[
            src : src + VALUES_PER_KEY
        ]
    return fused


def read_mixed_catalog(root: Path) -> dict[str, Any]:
    raw = (root / "catalog.hq38m20").read_bytes()
    if raw[:8] != CATALOG_MAGIC:
        raise PackError(f"mixed catalog magic {raw[:8]!r}")
    version, n_tensors, n_segments, _flags, name_blob_bytes, _ = struct.unpack_from(
        "<IIIIII", raw, 8
    )
    if version != CATALOG_VERSION:
        raise PackError(f"mixed catalog version {version}")
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
        records.append(
            {
                "name": name,
                "codec": codec,
                "organ": organ,
                "shape": [d0, d1, d2, d3][:ndim],
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
    return {"segments": segments, "records": records}


class MixedReader:
    def __init__(self, root: Path):
        cat = read_mixed_catalog(root)
        self.root = root
        self.rows = {r["name"]: r for r in cat["records"]}
        self.segments = {s["id"]: s for s in cat["segments"]}

    def payload(self, name: str) -> bytes:
        rec = self.rows[name]
        seg = self.segments[rec["segment_id"]]
        path = self.root / "segments" / seg["filename"]
        with path.open("rb") as handle:
            handle.seek(int(rec["offset"]))
            blob = handle.read(int(rec["nbytes"]))
        if sha256_hex(blob) != rec["sha256"]:
            raise PackError(f"{name}: mixed payload sha256 mismatch")
        return blob

    def load(self, name: str) -> np.ndarray:
        rec = self.rows[name]
        return decode_mixed_payload(int(rec["codec"]), self.payload(name))


def init_generate_root(root: Path) -> dict[str, Any]:
    tensors = root / "tensors"
    tensors.mkdir(parents=True, exist_ok=True)
    src = Q4_ORACLE / "tensors"
    linked = 0
    existed = 0
    for item in src.iterdir():
        dest = tensors / item.name
        if dest.exists() or dest.is_symlink():
            existed += 1
            continue
        os.link(item, dest)
        linked += 1
    manifest_src = json.loads((Q4_ORACLE / "manifest.json").read_text())
    dest_manifest = root / "manifest.json"
    if not dest_manifest.exists():
        dest_manifest.write_text(json.dumps(manifest_src, indent=2) + "\n")
    log(f"generate root {root}: hardlinked {linked}, already {existed}")
    return manifest_src


def overwrite_q4(root: Path, name: str, values: np.ndarray, shape: list[int]) -> int:
    payload = pack_hq30uq4(values, shape)
    path = root / "tensors" / artifact_filename(name, "hq30uq4")
    if path.exists() or path.is_symlink():
        path.unlink()
    write_atomic(path, payload)
    return len(payload)


def mlp_names(layer: int) -> list[tuple[str, list[int]]]:
    p = f"language_model.model.layers.{layer}.mlp."
    return [
        (p + "gate_proj.weight", [INTERMEDIATE, HIDDEN]),
        (p + "up_proj.weight", [INTERMEDIATE, HIDDEN]),
        (p + "down_proj.weight", [HIDDEN, INTERMEDIATE]),
    ]


def attn_source_names(layer: int) -> list[tuple[str, list[int], str]]:
    """(source_name, shape, role) for rice-packed attention GEMVs."""
    p = f"language_model.model.layers.{layer}."
    if is_gqa_layer(layer):
        return [
            (p + "self_attn.q_proj.weight", [Q_ROWS, HIDDEN], "q_proj"),
            (p + "self_attn.k_proj.weight", [KV_ROWS, HIDDEN], "k_proj"),
            (p + "self_attn.v_proj.weight", [KV_ROWS, HIDDEN], "v_proj"),
            (p + "self_attn.o_proj.weight", [O_ROWS, O_COLS], "o_proj"),
        ]
    return [
        (p + "linear_attn.in_proj_qkv.weight", [QKV_ROWS, HIDDEN], "in_proj_qkv"),
        (p + "linear_attn.in_proj_z.weight", [Z_ROWS, HIDDEN], "in_proj_z"),
        (p + "linear_attn.in_proj_a.weight", [A_ROWS, HIDDEN], "in_proj_a"),
        (p + "linear_attn.in_proj_b.weight", [B_ROWS, HIDDEN], "in_proj_b"),
        (p + "linear_attn.out_proj.weight", [O_ROWS, O_COLS], "out_proj"),
    ]


def materialize_mlp(root: Path, mixed: MixedReader, layers: range) -> list[dict[str, Any]]:
    out = []
    for layer in layers:
        for name, shape in mlp_names(layer):
            t0 = time.perf_counter()
            rec = mixed.rows[name]
            decoded = mixed.load(name)
            if list(decoded.shape) != shape:
                raise PackError(f"{name} decoded shape {decoded.shape} != {shape}")
            nbytes = overwrite_q4(root, name, decoded, shape)
            row = {
                "name": name,
                "organ": int(rec["organ"]),
                "codec": int(rec["codec"]),
                "codec_name": {0: "binary_g128", 1: "rice_q1_rms_2pct", 2: "hgravs01_r160_b3"}[
                    int(rec["codec"])
                ],
                "shape": shape,
                "elements": int(rec["elements"]),
                "packed_bytes": int(rec["nbytes"]),
                "packed_bpw": float(rec["codec_bpw"]),
                "q4_vehicle_bytes": nbytes,
                "wall_s": time.perf_counter() - t0,
            }
            out.append(row)
            log(
                f"mlp L{layer:02d} {name.rsplit('.', 2)[-2]} "
                f"packed={row['packed_bpw']:.3f} bpw q4={nbytes} "
                f"{row['wall_s']:.1f}s"
            )
            del decoded
    return out


def pack_and_materialize_attention(
    root: Path,
    packed_dir: Path,
    model_dir: Path,
    weight_map: dict[str, str],
    layers: range,
) -> list[dict[str, Any]]:
    packed_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    # Keep decoded pieces per layer so we can fuse after both halves land.
    pending: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for layer in layers:
        for name, shape, role in attn_source_names(layer):
            payload_path = packed_dir / f"{sha256_hex(name.encode())}.rice"
            meta_path = packed_dir / f"{sha256_hex(name.encode())}.json"
            t0 = time.perf_counter()
            if payload_path.exists() and meta_path.exists():
                payload = payload_path.read_bytes()
                decoded = decode_residual_compact(payload)
                meta = json.loads(meta_path.read_text())
                log(f"attn reuse {name} {meta.get('packed_bpw'):.3f} bpw")
            else:
                values = np.ascontiguousarray(
                    load_tensor(model_dir, weight_map, name), dtype=np.float32
                )
                if list(values.shape) != shape:
                    raise PackError(f"{name} source {values.shape} != {shape}")
                payload, decoded = encode_rice(values)
                meta = {
                    "name": name,
                    "role": role,
                    "shape": shape,
                    "elements": int(values.size),
                    "packed_bytes": len(payload),
                    "packed_bpw": 8.0 * len(payload) / max(int(values.size), 1),
                    "cosine_vs_bf16": cosine_flat(values, decoded),
                }
                write_atomic(payload_path, payload)
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
                del values
                log(
                    f"attn L{layer:02d} {role} bpw={meta['packed_bpw']:.3f} "
                    f"cos={meta['cosine_vs_bf16']:.4f} {time.perf_counter()-t0:.1f}s"
                )
            decoded = np.ascontiguousarray(decoded, dtype=np.float32).reshape(shape)
            pending[layer][role] = decoded
            out.append(meta)

        # Fuse + write Q4 vehicle tensors for this layer, then free.
        if is_gqa_layer(layer):
            p = f"language_model.model.layers.{layer}.self_attn."
            for role, q4_name, shape in (
                ("q_proj", p + "q_proj.weight", [Q_ROWS, HIDDEN]),
                ("k_proj", p + "k_proj.weight", [KV_ROWS, HIDDEN]),
                ("v_proj", p + "v_proj.weight", [KV_ROWS, HIDDEN]),
                ("o_proj", p + "o_proj.weight", [O_ROWS, O_COLS]),
            ):
                overwrite_q4(root, q4_name, pending[layer][role], shape)
        else:
            p = f"language_model.model.layers.{layer}.linear_attn."
            qkvz = fuse_in_proj_qkvz(
                pending[layer]["in_proj_qkv"], pending[layer]["in_proj_z"]
            )
            ba = fuse_in_proj_ba(
                pending[layer]["in_proj_b"], pending[layer]["in_proj_a"]
            )
            overwrite_q4(root, p + "in_proj_qkvz.weight", qkvz, [QKVZ_ROWS, HIDDEN])
            overwrite_q4(root, p + "in_proj_ba.weight", ba, [BA_ROWS, HIDDEN])
            overwrite_q4(
                root,
                p + "out_proj.weight",
                pending[layer]["out_proj"],
                [O_ROWS, O_COLS],
            )
            del qkvz, ba
        pending.pop(layer, None)
    return out


def classify_q4_row(name: str) -> str:
    if name.endswith("embed_tokens.weight"):
        return "embed"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if ".mlp." in name:
        return "mlp"
    if name.endswith(".f32v2") or name.endswith(
        (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "model.norm.weight",
            "q_norm.weight",
            "k_norm.weight",
            "conv1d.weight",
            "A_log",
            "dt_bias",
            "linear_attn.norm.weight",
        )
    ):
        return "small_f32"
    return "attention"


def ledger(
    *,
    root: Path,
    manifest: dict[str, Any],
    mlp_rows: list[dict[str, Any]],
    attn_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_class: dict[str, dict[str, float]] = {
        k: {"bytes": 0, "elements": 0, "tensors": 0}
        for k in (
            "mlp_gate_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "attention_gemv_rice",
            "embed_q4",
            "lm_head_q4",
            "small_f32",
        )
    }

    def add(slot: str, nbytes: int, elements: int, tensors: int = 1) -> None:
        by_class[slot]["bytes"] += nbytes
        by_class[slot]["elements"] += elements
        by_class[slot]["tensors"] += tensors

    for rec in mlp_rows:
        organ = int(rec["organ"])
        slot = {ORGAN_GATE: "mlp_gate_proj", ORGAN_UP: "mlp_up_proj", ORGAN_DOWN: "mlp_down_proj"}[
            organ
        ]
        add(slot, int(rec["packed_bytes"]), int(rec["elements"]))
    for rec in attn_rows:
        add("attention_gemv_rice", int(rec["packed_bytes"]), int(rec["elements"]))

    name_to_row = {t["name"]: t for t in manifest["tensors"]}
    embed = name_to_row["language_model.model.embed_tokens.weight"]
    head = name_to_row["language_model.lm_head.weight"]
    add("embed_q4", int(embed["bytes"]), int(embed["elements"]))
    add("lm_head_q4", int(head["bytes"]), int(head["elements"]))
    for t in manifest["tensors"]:
        if t["kind"] == "f32":
            add("small_f32", int(t["bytes"]), int(t["elements"]))

    total_bytes = sum(s["bytes"] for s in by_class.values())
    total_elems = sum(s["elements"] for s in by_class.values())
    for slot in by_class.values():
        elems = max(int(slot["elements"]), 1)
        slot["physical_bpw"] = 8.0 * slot["bytes"] / elems
    complete = 8.0 * total_bytes / max(total_elems, 1)

    # Projection: scale GPU-bound part of the 38.217 ms wall, hold 1.415 ms fixed.
    measured_wall_ms = 38.217
    measured_bpw = 4.252735126866492
    fixed_ms = 1.415
    gpu_ms = measured_wall_ms - fixed_ms
    proj_ms = gpu_ms * (complete / measured_bpw) + fixed_ms
    proj_tps = 1000.0 / proj_ms if proj_ms > 0 else 0.0

    return {
        "complete_physical_bpw": complete,
        "all_required_weight_artifact_bytes": int(total_bytes),
        "source_weight_elements": int(total_elems),
        "per_tensor_class": by_class,
        "projection": {
            "measured_complete_wall_ms": measured_wall_ms,
            "measured_bpw": measured_bpw,
            "fixed_overhead_ms": fixed_ms,
            "formula": "ms = 1.415 + (38.217-1.415) * (bpw / 4.252735126866492)",
            "projected_ms_per_token": proj_ms,
            "projected_tps": proj_tps,
        },
        "generate_vehicle": {
            "artifact_root": str(root),
            "schema": manifest.get("schema"),
            "note": "HQ30UQ4 of reconstructed mixed/rice weights; packed BPW is the ledger above",
        },
    }


def write_reports(root: Path, body: dict[str, Any]) -> None:
    path = root / "PACK_REPORT.json"
    path.write_text(json.dumps(body, indent=2) + "\n")
    (root / "FORMAT.md").write_text(
        "Qwen3.8 dense mixed-sub15 pack. "
        "MLP copied from mixed-2p0-v1 (binary gate, rice up, HGRAVS01 down on real X). "
        "Attention GEMVs rice_q1_rms_2pct from BF16. "
        "embed/lm_head HQ30UQ4. Small tensors f32.\n"
    )
    log(f"wrote {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--mixed", type=Path, default=MIXED_2P0)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument(
        "--phase",
        choices=("init", "mlp", "attn", "report", "all"),
        default="all",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    layers = range(args.max_layers if args.max_layers is not None else N_LAYERS)
    packed_attn = root / "packed" / "attn"
    mlp_json = root / "packed" / "mlp_rows.json"
    attn_json = root / "packed" / "attn_rows.json"
    (root / "packed").mkdir(exist_ok=True)

    if args.phase in ("init", "mlp", "attn", "all", "report"):
        manifest = init_generate_root(root)

    mlp_rows: list[dict[str, Any]] = []
    if mlp_json.exists():
        mlp_rows = json.loads(mlp_json.read_text())
    if args.phase in ("mlp", "all"):
        mixed = MixedReader(args.mixed)
        mlp_rows = materialize_mlp(root, mixed, layers)
        mlp_json.write_text(json.dumps(mlp_rows, indent=2) + "\n")

    attn_rows: list[dict[str, Any]] = []
    if attn_json.exists():
        attn_rows = json.loads(attn_json.read_text())
    if args.phase in ("attn", "all"):
        wm = load_weight_map(args.model_dir)
        attn_rows = pack_and_materialize_attention(
            root, packed_attn, args.model_dir, wm, layers
        )
        attn_json.write_text(json.dumps(attn_rows, indent=2) + "\n")

    if args.phase in ("report", "all") and mlp_rows and attn_rows:
        body = {
            "schema": "hawking.ascent.qwen38_mixed_sub15.v1",
            "status": "PACKED",
            "date": "2026-08-16",
            "lane": "qwen38-sub15-pack",
            "root": str(root),
            "activation": {
                "mlp_down_fit": "reused mixed-2p0-v1 HGRAVS01 (real BF16 post-SwiGLU X)",
                "not_synthetic": True,
                "capture": str(
                    MAIN
                    / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
                ),
            },
            "recipe": {
                "mlp.gate_proj": "HGRAVB01 binary_g128 (from mixed-2p0-v1)",
                "mlp.up_proj": "HGRAVR02 rice_q1_rms_2pct (from mixed-2p0-v1)",
                "mlp.down_proj": "HGRAVS01 r160_b3 real-X (from mixed-2p0-v1)",
                "attention_gemv": "rice_q1_rms_2pct from BF16",
                "embed": "HQ30UQ4 group-64 (oracle)",
                "lm_head": "HQ30UQ4 group-64 (oracle)",
                "small": "f32 (oracle)",
            },
            "did_not_duplicate_rescreen": True,
            "sibling_mixed_2p0_bpw": 2.0855934079220506,
            **ledger(
                root=root, manifest=manifest, mlp_rows=mlp_rows, attn_rows=attn_rows
            ),
        }
        write_reports(root, body)
        print(json.dumps(body, indent=2), flush=True)
        if body["complete_physical_bpw"] >= 1.5:
            log(f"WARNING bpw {body['complete_physical_bpw']} is not < 1.5")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
