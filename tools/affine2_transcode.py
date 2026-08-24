#!/usr/bin/env python3
"""Convert PocketAiHub Qwen3.8 MLX affine-2bit safetensors into HQ38M20 + HGRAVF01.

Does not import or call mlx / mlx_lm. Copies packed 2-bit codes as-is and
converts bf16 scale/bias to IEEE fp16. Small unquantized tensors become f32v2
(HF-δ on residual / q_norm / k_norm). Vision tensors are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAIN = Path("/Users/scammermike/Downloads/hawking")
DEFAULT_SRC = MAIN / "workspace/campaign/records/runs/qwen38-27b/abliterated-mlx-2bit/2bit"
DEFAULT_DST = REPO / "workspace/campaign/records/runs/qwen38-27b/affine2-native"

CATALOG_MAGIC = b"HQ38M20\0"
CATALOG_VERSION = 1
RECORD_SIZE = 128
CODEC_AFFINE = 5
CODEC_F32 = 4
ORGAN_GATE = 0
ORGAN_UP = 1
ORGAN_DOWN = 2
ORGAN_ATTN = 3
ORGAN_EMB = 4
ORGAN_HEAD = 5
ORGAN_SMALL = 6
MAGIC_AFFINE = b"HGRAVF01"
SCHEMA_AFFINE = "hawking.gravity.affine_scale_bias.v1"
AFFINE_REPR = "affine_q2_group32_fp16_scale_bias"


class TranscodeError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[affine2] {msg}", flush=True)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def artifact_filename(name: str, ext: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest() + f".{ext}"


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def organ_for(name: str) -> int:
    if name.endswith("mlp.gate_proj.weight"):
        return ORGAN_GATE
    if name.endswith("mlp.up_proj.weight"):
        return ORGAN_UP
    if name.endswith("mlp.down_proj.weight"):
        return ORGAN_DOWN
    if name.endswith("embed_tokens.weight"):
        return ORGAN_EMB
    if name.endswith("lm_head.weight"):
        return ORGAN_HEAD
    if "self_attn." in name or "linear_attn." in name:
        return ORGAN_ATTN
    return ORGAN_SMALL


def is_hf_delta_norm(name: str) -> bool:
    return (
        name.endswith("input_layernorm.weight")
        or name.endswith("post_attention_layernorm.weight")
        or name.endswith("model.norm.weight")
        or name.endswith("q_norm.weight")
        or name.endswith("k_norm.weight")
    )


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    bits = np.frombuffer(raw, dtype="<u2")
    return (bits.astype(np.uint32) << 16).view(np.float32).copy()


def bf16_bytes_to_f16_bytes(raw: bytes) -> bytes:
    return bf16_bytes_to_f32(raw).astype(np.float16).tobytes()


def wrap_hgrafv01(
    rows: int,
    cols: int,
    scales_f16: bytes,
    biases_f16: bytes,
    codes: bytes,
) -> bytes:
    groups = rows * (cols // 32)
    if len(scales_f16) != groups * 2 or len(biases_f16) != groups * 2:
        raise TranscodeError(
            f"scale/bias bytes {len(scales_f16)}/{len(biases_f16)} != {groups * 2}"
        )
    if len(codes) != groups * 8:
        raise TranscodeError(f"code bytes {len(codes)} != {groups * 8}")
    header = {
        "schema": SCHEMA_AFFINE,
        "representation": AFFINE_REPR,
        "shape": [rows, cols],
        "elements": rows * cols,
        "bits": 2,
        "group_size": 32,
        "groups": groups,
        "scale_bytes": len(scales_f16),
        "bias_bytes": len(biases_f16),
        "code_bytes": len(codes),
        "source": "mlx_quantized_linear",
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return (
        MAGIC_AFFINE
        + struct.pack("<I", len(header_bytes))
        + header_bytes
        + scales_f16
        + biases_f16
        + codes
    )


def wrap_f32v2(values: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    return struct.pack("<Q", int(flat.size)) + flat.tobytes()


def write_catalog(
    path: Path,
    records: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> None:
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
            raise TranscodeError(f"{rec['name']} rank {len(shape)} exceeds catalog")
        for i, d in enumerate(shape):
            dims[i] = int(d)
        digest = bytes.fromhex(rec["sha256"])
        rec_bytes = struct.pack(
            "<IHBBBB",
            off,
            len(raw_name),
            int(rec["codec"]),
            int(rec["organ"]),
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
            int(rec["elements"]),
            int(rec["segment_id"]),
            int(rec.get("achieved_rank") or 0),
            int(rec["offset"]),
            int(rec["nbytes"]),
            digest,
            int(rec.get("flags") or 0),
            int(rec.get("n_fit_rows") or 0),
            float(rec["codec_bpw"]),
        )
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
    write_atomic(path, blob)


class ShardReader:
    def __init__(self, path: Path):
        self.path = path
        raw = path.read_bytes() if path.stat().st_size < 64 else None
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
            self.data_off = 8 + header_len
            self.meta = {k: v for k, v in header.items() if k != "__metadata__"}
        del raw

    def read(self, name: str) -> tuple[list[int], str, bytes]:
        info = self.meta[name]
        start, end = info["data_offsets"]
        with self.path.open("rb") as handle:
            handle.seek(self.data_off + int(start))
            blob = handle.read(int(end) - int(start))
        return list(info["shape"]), str(info["dtype"]), blob


def load_index(src: Path) -> dict[str, str]:
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    return dict(idx["weight_map"])


def transcode(src: Path, dst: Path) -> dict[str, Any]:
    if not (src / "model.safetensors.index.json").is_file():
        raise TranscodeError(f"missing safetensors index under {src}")
    weight_map = load_index(src)
    lang = [k for k in weight_map if k.startswith("language_model.")]
    bases = {k[: -len(".scales")] for k in lang if k.endswith(".scales")}
    gemv = sorted(k for k in lang if k.endswith(".weight") and k[: -len(".weight")] in bases)
    small = sorted(
        k
        for k in lang
        if not k.endswith(".scales")
        and not k.endswith(".biases")
        and not (k.endswith(".weight") and k[: -len(".weight")] in bases)
    )
    if len(gemv) != 498 or len(small) != 353:
        raise TranscodeError(
            f"expected 498 GEMV + 353 small language tensors, got {len(gemv)}+{len(small)}"
        )

    shards: dict[str, ShardReader] = {}

    def reader_for(name: str) -> ShardReader:
        fname = weight_map[name]
        if fname not in shards:
            shards[fname] = ShardReader(src / fname)
        return shards[fname]

    dst.mkdir(parents=True, exist_ok=True)
    seg_dir = dst / "segments"
    seg_dir.mkdir(exist_ok=True)

    records: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    reused = 0
    wrote = 0

    def add_payload(name: str, codec: int, organ: int, shape: list[int], payload: bytes) -> None:
        nonlocal wrote, reused
        ext = "hgrafv01" if codec == CODEC_AFFINE else "f32v2"
        filename = artifact_filename(name, ext)
        path = seg_dir / filename
        if path.is_file() and path.stat().st_size == len(payload):
            reused += 1
        else:
            write_atomic(path, payload)
            wrote += 1
        sid = len(segments)
        digest = sha256_hex(payload)
        segments.append(
            {
                "id": sid,
                "filename": filename,
                "bytes": len(payload),
                "sha256": digest,
            }
        )
        elements = 1
        for dim in shape:
            elements *= int(dim)
        records.append(
            {
                "name": name,
                "codec": codec,
                "organ": organ,
                "shape": list(shape),
                "elements": elements,
                "segment_id": sid,
                "achieved_rank": 0,
                "offset": 0,
                "nbytes": len(payload),
                "sha256": digest,
                "flags": 0,
                "n_fit_rows": 0,
                "codec_bpw": 8.0 * len(payload) / max(elements, 1),
            }
        )

    for i, name in enumerate(gemv):
        base = name[: -len(".weight")]
        weight_r = reader_for(name)
        w_shape, w_dtype, w_bytes = weight_r.read(name)
        s_shape, s_dtype, s_bytes = reader_for(base + ".scales").read(base + ".scales")
        b_shape, b_dtype, b_bytes = reader_for(base + ".biases").read(base + ".biases")
        if w_dtype != "U32":
            raise TranscodeError(f"{name} dtype {w_dtype} is not U32")
        if s_dtype != "BF16" or b_dtype != "BF16":
            raise TranscodeError(f"{name} scale/bias dtypes {s_dtype}/{b_dtype}")
        if len(w_shape) != 2 or len(s_shape) != 2:
            raise TranscodeError(f"{name} rank is not 2")
        rows, packed_cols = int(w_shape[0]), int(w_shape[1])
        cols = packed_cols * 16
        if cols % 32 != 0:
            raise TranscodeError(f"{name} recovered cols={cols} is not a multiple of 32")
        if list(s_shape) != [rows, cols // 32] or list(b_shape) != [rows, cols // 32]:
            raise TranscodeError(
                f"{name} scale/bias shape {s_shape}/{b_shape} != {[rows, cols // 32]}"
            )
        if len(w_bytes) != rows * packed_cols * 4:
            raise TranscodeError(f"{name} packed byte count drifted")
        payload = wrap_hgrafv01(
            rows,
            cols,
            bf16_bytes_to_f16_bytes(s_bytes),
            bf16_bytes_to_f16_bytes(b_bytes),
            w_bytes,
        )
        add_payload(name, CODEC_AFFINE, organ_for(name), [rows, cols], payload)
        if i % 25 == 0 or i + 1 == len(gemv):
            log(f"gemv {i + 1}/{len(gemv)} {name} {rows}x{cols} {len(payload)}B")

    for i, name in enumerate(small):
        shape, dtype, raw = reader_for(name).read(name)
        if dtype != "BF16":
            raise TranscodeError(f"{name} dtype {dtype} is not BF16")
        values = bf16_bytes_to_f32(raw)
        if is_hf_delta_norm(name):
            values = values - 1.0
        payload = wrap_f32v2(values)
        add_payload(name, CODEC_F32, ORGAN_SMALL, [int(d) for d in shape], payload)
        if i % 50 == 0 or i + 1 == len(small):
            log(f"f32  {i + 1}/{len(small)} {name} {shape}")

    if len(records) != 851:
        raise TranscodeError(f"catalog would have {len(records)} tensors, expected 851")

    catalog_path = dst / "catalog.hq38m20"
    write_catalog(catalog_path, records, segments)
    report = {
        "schema": "hawking.ascent.qwen38_affine2_native_catalog.v1",
        "status": "EMITTED",
        "source": str(src),
        "root": str(dst),
        "catalog": str(catalog_path),
        "tensors": len(records),
        "segments": len(segments),
        "wrote": wrote,
        "reused": reused,
        "affine": sum(1 for r in records if r["codec"] == CODEC_AFFINE),
        "f32": sum(1 for r in records if r["codec"] == CODEC_F32),
        "payload_bytes": int(sum(r["nbytes"] for r in records)),
        "wall_s": time.perf_counter() - t0,
        "note": "HGRAVF01 copies MLX 2-bit codes; bf16 scale/bias converted to fp16. No mlx.",
    }
    (dst / "TRANSCODE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    (dst / "FORMAT.md").write_text(
        "Qwen3.8 native affine-2bit catalog. GEMVs are HGRAVF01 "
        "(w = q * scale + bias, group-32, 2-bit). Small tensors are f32v2.\n"
    )
    tok_src = src / "tokenizer.json"
    tok_dst = dst / "tokenizer.json"
    if tok_src.is_file() and not tok_dst.exists():
        try:
            os.link(tok_src, tok_dst)
        except OSError:
            tok_dst.write_bytes(tok_src.read_bytes())
    log(
        f"emitted {catalog_path} tensors={len(records)} "
        f"affine={report['affine']} f32={report['f32']} "
        f"bytes={report['payload_bytes']} wall={report['wall_s']:.1f}s"
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = transcode(args.src, args.dst)
    except TranscodeError as err:
        print(f"affine2_transcode: {err}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
