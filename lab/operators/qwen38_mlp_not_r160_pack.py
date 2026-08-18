#!/usr/bin/env python3
"""Pack a HQ38M20 Qwen3.8 artifact whose MLP is not r160_b3 down_proj.

Copies mixed-2p0-v1 (HGRAVB01 gate / HGRAVR02 up / HGRAVU01 attention) and
re-encodes selected MLP organs as HGRAVU01 from the BF16 source. Default is
down_proj only at 4 bits: that is the organ whose mixed-2p0 weight cosine is
0.17-0.21. Packed bytes stay packed. Never materializes a dense W for the
runtime — the catalog points at HGRAVU01 payloads the existing uniform tile
already consumes.

Does not mutate mixed-2p0-v1 or the BF16 source. Non-replaced segments are
hard-linked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.ascension_dual_gravity_worker import (  # noqa: E402
    GROUP_UNIFORM,
    MAGIC_UNIFORM,
    _container,
    _pack_unsigned,
)
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map  # noqa: E402

MAIN = Path("/Users/scammermike/Downloads/hawking")
MODEL_DIR = MAIN / "workspace/campaign/records/runs/qwen38-27b/bf16"
MIXED_2P0 = MAIN / "workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1"
RUNS = MAIN / "workspace/campaign/records/runs/qwen38-27b"

SOURCE_ELEMENTS = 26_895_998_464
CATALOG_MAGIC = b"HQ38M20\0"
CATALOG_VERSION = 1
RECORD_SIZE = 128
CODEC_UNIFORM = 3
ORGAN_GATE = 0
ORGAN_UP = 1
ORGAN_DOWN = 2
ORGAN_NONMLP = 3
ORGAN_ATTN = 4       # virtual: attention GEMVs only, never embed / lm_head / norms


class PackError(RuntimeError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def log(msg: str) -> None:
    print(f"[mlp-not-r160] {msg}", flush=True)


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def encode_uniform_payload(values: np.ndarray, bits: int) -> bytes:
    if bits < 2 or bits > 8:
        raise PackError(f"uniform bits {bits} not in 2..8")
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(flat).all():
        raise PackError("non-finite source")
    groups = math.ceil(flat.size / GROUP_UNIFORM)
    padded = np.zeros(groups * GROUP_UNIFORM, dtype=np.float32)
    padded[: flat.size] = flat
    grouped = padded.reshape(groups, GROUP_UNIFORM)
    bound = (1 << (bits - 1)) - 1
    scales = (np.max(np.abs(grouped), axis=1) / max(bound, 1)).astype("<f2")
    denominator = np.where(scales.astype(np.float32) > 0.0, scales.astype(np.float32), 1.0)
    signed = np.rint(grouped / denominator[:, None]).clip(-bound, bound).astype(np.int16)
    unsigned = (signed.reshape(-1) + bound).astype(np.uint8)
    code_bytes = _pack_unsigned(unsigned, bits)
    header = {
        "schema": "hawking.gravity.uniform_group.v1",
        "representation": f"uniform_q{bits}_group_scale",
        "shape": [int(item) for item in values.shape],
        "elements": int(flat.size),
        "bits": bits,
        "group_size": GROUP_UNIFORM,
        "groups": groups,
        "scale_dtype": "float16",
        "code_bytes": len(code_bytes),
        "scale_bytes": int(scales.nbytes),
        "retained_padding_elements": int(groups * GROUP_UNIFORM - flat.size),
    }
    return _container(MAGIC_UNIFORM, header, scales.tobytes() + code_bytes)


def strided_weight_cosine(source: np.ndarray, bits: int, stride: int = 17) -> float:
    """Cheap screen: reconstruct every `stride`-th group, score vs source."""
    flat = np.ascontiguousarray(source, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / GROUP_UNIFORM)
    padded = np.zeros(groups * GROUP_UNIFORM, dtype=np.float32)
    padded[: flat.size] = flat
    grouped = padded.reshape(groups, GROUP_UNIFORM)
    idx = np.arange(0, groups, stride)
    bound = (1 << (bits - 1)) - 1
    scales = np.max(np.abs(grouped[idx]), axis=1) / max(bound, 1)
    denom = np.where(scales > 0.0, scales, 1.0)
    signed = np.rint(grouped[idx] / denom[:, None]).clip(-bound, bound)
    recon = signed * denom[:, None]
    a = grouped[idx].reshape(-1).astype(np.float64)
    b = recon.reshape(-1).astype(np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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


def write_catalog(
    path: Path,
    records: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> bytes:
    name_blob = bytearray()
    offs: list[int] = []
    for rec in records:
        raw = rec["name"].encode("utf-8")
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
        rec_bytes = rec_bytes + b"\x00" * (RECORD_SIZE - len(rec_bytes))
        table.extend(rec_bytes)
    seg_blob = bytearray()
    for seg in segments:
        name = str(seg["filename"]).encode("utf-8")
        digest = bytes.fromhex(seg["sha256"])
        seg_blob.extend(
            struct.pack("<HHQ32s", int(seg["id"]), len(name), int(seg["bytes"]), digest)
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
    return blob


ATTN_SUFFIXES = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
    "linear_attn.out_proj.weight",
)


def organ_of(name: str, fallback: int) -> int:
    if name.endswith("mlp.gate_proj.weight"):
        return ORGAN_GATE
    if name.endswith("mlp.up_proj.weight"):
        return ORGAN_UP
    if name.endswith("mlp.down_proj.weight"):
        return ORGAN_DOWN
    if name.endswith(ATTN_SUFFIXES):
        return ORGAN_ATTN
    return fallback


def organ_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {
        "mlp_gate_proj": {"bytes": 0, "elements": 0, "tensors": 0},
        "mlp_up_proj": {"bytes": 0, "elements": 0, "tensors": 0},
        "mlp_down_proj": {"bytes": 0, "elements": 0, "tensors": 0},
        "attention_embed_norm": {"bytes": 0, "elements": 0, "tensors": 0},
    }
    key = {
        ORGAN_GATE: "mlp_gate_proj",
        ORGAN_UP: "mlp_up_proj",
        ORGAN_DOWN: "mlp_down_proj",
        ORGAN_NONMLP: "attention_embed_norm",
        ORGAN_ATTN: "attention_embed_norm",
    }
    for rec in records:
        slot = buckets[key[int(rec["organ"])]]
        slot["bytes"] += int(rec["nbytes"])
        slot["elements"] += int(rec["elements"])
        slot["tensors"] += 1
    for slot in buckets.values():
        elems = max(int(slot["elements"]), 1)
        slot["physical_bpw"] = 8.0 * slot["bytes"] / elems
    return buckets


def hardlink_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--down-bits", type=int, default=4)
    p.add_argument("--gate-bits", type=int, default=None)
    p.add_argument("--up-bits", type=int, default=None)
    p.add_argument("--attn-bits", type=int, default=None,
                   help="re-encode the attention GEMVs only; embed, lm_head and norms are untouched")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--mixed", type=Path, default=MIXED_2P0)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def bits_ok(bits: int | None, flag: str) -> None:
    if bits is None:
        return
    if bits < 2 or bits > 8:
        raise PackError(f"{flag} {bits} not in 2..8")


def main() -> int:
    args = parse_args()
    bits_ok(args.down_bits, "--down-bits")
    bits_ok(args.gate_bits, "--gate-bits")
    bits_ok(args.up_bits, "--up-bits")
    bits_ok(args.attn_bits, "--attn-bits")
    replace_bits = {
        ORGAN_DOWN: int(args.down_bits),
    }
    if args.gate_bits is not None:
        replace_bits[ORGAN_GATE] = int(args.gate_bits)
    if args.up_bits is not None:
        replace_bits[ORGAN_UP] = int(args.up_bits)
    if args.attn_bits is not None:
        replace_bits[ORGAN_ATTN] = int(args.attn_bits)
    tag = args.tag
    if tag is None:
        parts = [f"down-q{args.down_bits}"]
        if args.gate_bits is not None:
            parts.append(f"gate-q{args.gate_bits}")
        if args.up_bits is not None:
            parts.append(f"up-q{args.up_bits}")
        if args.attn_bits is not None:
            parts.append(f"attn-q{args.attn_bits}")
        tag = "mixed-" + "-".join(parts) + "-v1"
    root: Path = args.root or (RUNS / tag)
    root.mkdir(parents=True, exist_ok=True)
    (root / "segments").mkdir(exist_ok=True)
    log(f"root {root}")
    log(f"replace organs { {k: replace_bits[k] for k in replace_bits} }")

    src_cat = read_mixed_catalog(args.mixed)
    weight_map = load_weight_map(args.model_dir)
    src_segments = {int(s["id"]): s for s in src_cat["segments"]}
    max_seg = max(src_segments)

    out_records: list[dict[str, Any]] = []
    out_segments: list[dict[str, Any]] = []
    used_src_segs: set[int] = set()
    next_seg = max_seg + 1
    encoded = 0
    copied = 0
    cosines: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for rec in src_cat["records"]:
        organ = organ_of(rec["name"], int(rec["organ"]))
        if organ not in replace_bits:
            out_records.append(dict(rec))
            used_src_segs.add(int(rec["segment_id"]))
            copied += 1
            continue
        bits = replace_bits[organ]
        name = rec["name"]
        t0 = time.perf_counter()
        values = np.ascontiguousarray(
            load_tensor(args.model_dir, weight_map, name), dtype=np.float32
        )
        if list(values.shape) != list(rec["shape"]):
            raise PackError(f"{name} source {values.shape} != catalog {rec['shape']}")
        cosine = strided_weight_cosine(values, bits)
        payload = encode_uniform_payload(values, bits)
        del values
        digest = sha256_hex(payload)
        filename = f"replace_{name.replace('.', '_')}.hq38seg"
        dest = root / "segments" / filename
        write_atomic(dest, payload)
        seg_id = next_seg
        next_seg += 1
        out_segments.append(
            {
                "id": seg_id,
                "filename": filename,
                "bytes": len(payload),
                "sha256": sha256_hex(dest.read_bytes()),
            }
        )
        new_rec = dict(rec)
        new_rec.update(
            {
                "codec": CODEC_UNIFORM,
                "organ": ORGAN_NONMLP if organ == ORGAN_ATTN else organ,
                "segment_id": seg_id,
                "offset": 0,
                "nbytes": len(payload),
                "sha256": digest,
                "achieved_rank": 0,
                "n_fit_rows": 0,
                "flags": 0,
                "codec_bpw": 8.0 * len(payload) / max(int(rec["elements"]), 1),
            }
        )
        out_records.append(new_rec)
        encoded += 1
        cosines.append(
            {
                "name": name,
                "bits": bits,
                "strided_weight_cosine": cosine,
                "nbytes": len(payload),
                "codec_bpw": new_rec["codec_bpw"],
                "encode_s": time.perf_counter() - t0,
            }
        )
        log(
            f"encode {name} q{bits} {len(payload)} bytes cos={cosine:.5f} "
            f"{time.perf_counter() - t0:.2f}s"
        )

    for seg_id in sorted(used_src_segs):
        meta = src_segments[seg_id]
        src = args.mixed / "segments" / meta["filename"]
        dest = root / "segments" / meta["filename"]
        hardlink_or_copy(src, dest)
        out_segments.append(
            {
                "id": int(meta["id"]),
                "filename": meta["filename"],
                "bytes": int(meta["bytes"]),
                "sha256": meta["sha256"],
            }
        )

    out_segments.sort(key=lambda s: int(s["id"]))
    write_catalog(root / "catalog.hq38m20", out_records, out_segments)
    breakdown = organ_breakdown(out_records)
    payload_bytes = sum(int(r["nbytes"]) for r in out_records)
    catalog_bytes = (root / "catalog.hq38m20").stat().st_size
    complete_bpw = 8.0 * (payload_bytes + catalog_bytes) / SOURCE_ELEMENTS
    mlp_elems = (
        breakdown["mlp_gate_proj"]["elements"]
        + breakdown["mlp_up_proj"]["elements"]
        + breakdown["mlp_down_proj"]["elements"]
    )
    mlp_bytes = (
        breakdown["mlp_gate_proj"]["bytes"]
        + breakdown["mlp_up_proj"]["bytes"]
        + breakdown["mlp_down_proj"]["bytes"]
    )
    report = {
        "schema": "hawking.ascent.qwen38_mlp_not_r160.v1",
        "status": "PACKED",
        "date": time.strftime("%Y-%m-%d"),
        "lane": "auto-q80-qwen3-complete-still-bpw",
        "root": str(root),
        "source_mixed": str(args.mixed),
        "source_bf16": str(args.model_dir),
        "recipe": {
            "attention_embed_norm": (
                f"attention GEMVs HGRAVU01 uniform_q{args.attn_bits}_group64 from BF16; "
                "embed / lm_head / norms copied mixed-2p0-v1 HGRAVU01"
                if args.attn_bits is not None
                else "copied mixed-2p0-v1 HGRAVU01"
            ),
            "mlp.gate_proj": (
                f"HGRAVU01 uniform_q{args.gate_bits}_group64"
                if args.gate_bits is not None
                else "copied mixed-2p0-v1 HGRAVB01"
            ),
            "mlp.up_proj": (
                f"HGRAVU01 uniform_q{args.up_bits}_group64"
                if args.up_bits is not None
                else "copied mixed-2p0-v1 HGRAVR02"
            ),
            "mlp.down_proj": f"HGRAVU01 uniform_q{args.down_bits}_group64 (NOT HGRAVS01 r160_b3)",
            "native_catalog": "HQ38M20",
            "reconstruct_to_q4": False,
        },
        "replaced_organs": sorted(replace_bits.keys()),
        "encoded_tensors": encoded,
        "copied_tensors": copied,
        "tensor_count": len(out_records),
        "source_weight_elements": SOURCE_ELEMENTS,
        "tensor_payload_bytes": payload_bytes,
        "catalog_bytes": catalog_bytes,
        "all_required_weight_artifact_bytes": payload_bytes + catalog_bytes,
        "complete_physical_bpw": complete_bpw,
        "mlp_physical_bpw": 8.0 * mlp_bytes / max(mlp_elems, 1),
        "nonmlp_physical_bpw": breakdown["attention_embed_norm"]["physical_bpw"],
        "organ_breakdown": breakdown,
        "replaced_strided_weight_cosine": {
            "n": len(cosines),
            "min": min((c["strided_weight_cosine"] for c in cosines), default=None),
            "median": (
                sorted(c["strided_weight_cosine"] for c in cosines)[len(cosines) // 2]
                if cosines
                else None
            ),
            "max": max((c["strided_weight_cosine"] for c in cosines), default=None),
        },
        "projection": {
            "measured_complete_wall_ms": 38.217,
            "measured_bpw": 4.252735126866492,
            "fixed_overhead_ms": 1.229,
            "formula": "ms = 1.229 + (38.217-1.229) * (bpw / 4.252735126866492)",
            "projected_ms_per_token": 1.229
            + (38.217 - 1.229) * (complete_bpw / 4.252735126866492),
        },
        "wall_s": time.perf_counter() - t_all,
        "replaced_layer_rows": cosines,
        "claim_boundary": {
            "did_not_mutate_source": True,
            "did_not_use_synthetic_activations": True,
            "did_not_reconstruct_to_q4": True,
            "generation_is_the_gate": True,
            "r160_b3_down_removed": True,
        },
    }
    report["projection"]["projected_tps"] = (
        1000.0 / report["projection"]["projected_ms_per_token"]
        if report["projection"]["projected_ms_per_token"] > 0
        else 0.0
    )
    write_atomic(
        root / "PACK_REPORT.json",
        (json.dumps(report, indent=2) + "\n").encode(),
    )
    write_atomic(
        root / "FORMAT.md",
        (
            "HQ38M20 native mixed catalog. MLP down_proj is HGRAVU01, not "
            "HGRAVS01 r160_b3. Packed bytes stay packed.\n"
        ).encode(),
    )
    log(
        f"done bpw={complete_bpw:.4f} mlp_bpw={report['mlp_physical_bpw']:.4f} "
        f"down_bpw={breakdown['mlp_down_proj']['physical_bpw']:.4f} "
        f"proj_ms={report['projection']['projected_ms_per_token']:.3f} "
        f"wall={report['wall_s']:.1f}s"
    )
    print(json.dumps({"root": str(root), "complete_physical_bpw": complete_bpw}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackError as exc:
        print(f"[mlp-not-r160] ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
