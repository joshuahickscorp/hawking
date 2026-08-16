#!/usr/bin/env python3
"""DSV4F per-token ACTIVE byte census + native-code entropy + requant probe.

Does not touch the GPU. Reads the sealed full-stream manifest and a few
content-addressed chunks. Writes a JSON document the roof-rung instrument
can consume.

    python3 tools/ascent/dsv4f_active_byte_census.py
"""

from __future__ import annotations

import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/"
    "full-43-layer-stream.gravity"
)
CAPTURE = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/"
    "activation-x-capture-3k"
)
DIAGNOSIS = (
    REPO / "receipts" / "ascent-2026-08-16" / "DSV4F_GPU_BODY_DIAGNOSIS.json"
)
OUT = REPO / "receipts" / "ascent-2026-08-16" / "DSV4F_BYTE_REDUCTION.json"

LAYERS = 43
HIDDEN = 4096
VOCAB = 129_280
ROUTED = 256
TOP_K = 6
SHARED = 1
INTER = 2048
Q_LORA = 1024
O_LORA = 1024
WKV_ROWS = 512
WQ_B_ROWS = 32_768
WO_A_ROWS = 8192
WO_B_ROWS = 4096
WO_B_COLS = 8192
HEADS = 64
HEAD_DIM = 512
FP8_BLOCK = 128
FP4_BLOCK = 32
E4M3_MAX = 448.0
E2M1_MAX = 6.0
E2M1 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def decode_e4m3fn(bits: int) -> float:
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        raise ValueError("E4M3FN NaN")
    if exponent == 0:
        mag = mantissa * 0.001953125
    else:
        mag = struct.unpack(
            "<f", struct.pack("<I", ((exponent + 120) << 23) | (mantissa << 20))
        )[0]
    return mag if (bits & 0x80) == 0 else -mag


def decode_e8m0fnu(bits: int) -> float:
    if bits == 0xFF:
        raise ValueError("E8M0FNU NaN")
    if bits == 0:
        return struct.unpack("<f", struct.pack("<I", 0x00400000))[0]
    return struct.unpack("<f", struct.pack("<I", bits << 23))[0]


def decode_e2m1fn(nibble: int) -> float:
    return E2M1[nibble & 0x0F]


def shannon_bits(counts: Counter[int], n: int) -> float:
    if n <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent


def fp8_scale_bytes(rows: int, cols: int) -> int:
    return (rows // FP8_BLOCK) * (cols // FP8_BLOCK)


def fp4_scale_bytes(rows: int, cols: int) -> int:
    return rows * (cols // FP4_BLOCK)


def geometry_census() -> dict[str, Any]:
    mla_w = (
        Q_LORA * HIDDEN
        + WQ_B_ROWS * Q_LORA
        + WKV_ROWS * HIDDEN
        + WO_A_ROWS * HIDDEN
        + WO_B_ROWS * WO_B_COLS
    )
    mla_s = (
        fp8_scale_bytes(Q_LORA, HIDDEN)
        + fp8_scale_bytes(WQ_B_ROWS, Q_LORA)
        + fp8_scale_bytes(WKV_ROWS, HIDDEN)
        + fp8_scale_bytes(WO_A_ROWS, HIDDEN)
        + fp8_scale_bytes(WO_B_ROWS, WO_B_COLS)
    )
    mla_bytes = LAYERS * (mla_w + mla_s)

    expert_w = 3 * INTER * HIDDEN
    expert_fp4 = expert_w // 2
    expert_s = 3 * fp4_scale_bytes(INTER, HIDDEN)
    expert_bytes = LAYERS * TOP_K * (expert_fp4 + expert_s)
    expert_weights = LAYERS * TOP_K * expert_w

    shared_w = 3 * INTER * HIDDEN
    shared_s = 3 * fp8_scale_bytes(INTER, HIDDEN)
    shared_bytes = LAYERS * SHARED * (shared_w + shared_s)

    router_w = LAYERS * ROUTED * HIDDEN
    router_bytes = router_w * 2

    lm_w = VOCAB * HIDDEN
    lm_bytes = lm_w * 2

    classes = {
        "mla_fp8_plus_scale": {
            "weights": LAYERS * mla_w,
            "bytes": mla_bytes,
            "dtype": "F8_E4M3 + UE8M0 128x128",
            "bpw": 8.0 * mla_bytes / (LAYERS * mla_w),
            "how": (
                f"{LAYERS} layers x (wq_a {Q_LORA}x{HIDDEN} + wq_b {WQ_B_ROWS}x{Q_LORA} "
                f"+ wkv {WKV_ROWS}x{HIDDEN} + wo_a {WO_A_ROWS}x{HIDDEN} "
                f"+ wo_b {WO_B_ROWS}x{WO_B_COLS}) FP8 + 128x128 UE8M0"
            ),
        },
        "routed_fp4_plus_scale": {
            "weights": expert_weights,
            "bytes": expert_bytes,
            "dtype": "I8 packed E2M1FN_x2 + UE8M0 per 32-K",
            "bpw": 8.0 * expert_bytes / expert_weights,
            "how": (
                f"{LAYERS} layers x top-{TOP_K} of {ROUTED} x 3 proj x {INTER}x{HIDDEN} "
                "native 16-level FP4"
            ),
        },
        "shared_fp8_plus_scale": {
            "weights": LAYERS * shared_w,
            "bytes": shared_bytes,
            "dtype": "F8_E4M3 + UE8M0 128x128",
            "bpw": 8.0 * shared_bytes / (LAYERS * shared_w),
            "how": f"{LAYERS} layers x 1 shared x 3 proj x {INTER}x{HIDDEN} FP8",
        },
        "router_bf16": {
            "weights": router_w,
            "bytes": router_bytes,
            "dtype": "BF16",
            "bpw": 16.0,
            "how": f"{LAYERS} layers x {ROUTED}x{HIDDEN} BF16 gate.weight",
        },
        "lm_head_bf16": {
            "weights": lm_w,
            "bytes": lm_bytes,
            "dtype": "BF16",
            "bpw": 16.0,
            "how": f"full GEMV {VOCAB}x{HIDDEN} BF16",
        },
    }
    total_b = sum(c["bytes"] for c in classes.values())
    total_w = sum(c["weights"] for c in classes.values())
    return {
        "classes": classes,
        "total_unique_weight_scale_bytes": total_b,
        "total_gb": total_b / 1e9,
        "active_weights": total_w,
        "active_decode_bpw": 8.0 * total_b / total_w,
    }


def load_manifest() -> dict[str, Any]:
    with (ARTIFACT / "manifest.json").open() as fh:
        return json.load(fh)


def classify_name(name: str) -> tuple[str, int | None]:
    if name == "embed.weight":
        return "embed_full_table", None
    if name in ("head.weight", "lm_head.weight"):
        return "lm_head", None
    if name.startswith("mtp."):
        return "mtp_excluded", None
    if not name.startswith("layers."):
        return "other_global", None
    rest = name[len("layers.") :]
    layer_s, _, tail = rest.partition(".")
    try:
        layer = int(layer_s)
    except ValueError:
        return "other_global", None
    if layer >= LAYERS:
        return "mtp_excluded", layer
    if "indexer" in tail or "compressor" in tail:
        return "indexer_compressor", layer
    if tail.startswith("attn.wq_") or tail.startswith("attn.wo_") or tail.startswith(
        "attn.wkv"
    ):
        return "mla_fp8_pair", layer
    if tail.startswith("attn."):
        return "mla_aux_norm_sink", layer
    if tail.startswith("ffn.experts.") or tail.startswith("mlp.experts."):
        return "routed_all_256", layer
    if tail.startswith("ffn.shared_experts.") or tail.startswith("ffn.shared_expert."):
        return "shared_fp8_pair", layer
    if "tid2eid" in tail:
        return "hash_tid2eid", layer
    if tail.startswith("ffn.gate.") or tail.startswith("mlp.gate."):
        return "router", layer
    if tail.startswith("hc_"):
        return "mhc_f32", layer
    if "indexer" in tail or "compressor" in tail:
        return "indexer_compressor", layer
    if tail.endswith("_norm.weight") or tail in (
        "attn_norm.weight",
        "ffn_norm.weight",
    ):
        return "norms", layer
    return "other_layer", layer


def manifest_census(tensors: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = {}
    expert_sizes: Counter[int] = Counter()
    for name, meta in tensors.items():
        klass, _layer = classify_name(name)
        rec = buckets.setdefault(klass, {"tensors": 0, "bytes": 0})
        rec["tensors"] += 1
        rec["bytes"] += int(meta["bytes"])
        if klass == "routed_all_256" and name.endswith(".weight"):
            # group by expert: layers.L.mlp.experts.E.wN.weight
            expert_sizes[int(meta["bytes"])] += 1

    routed_storage = buckets.get("routed_all_256", {}).get("bytes", 0)
    # 43 layers x 256 experts. ACTIVE = top-6 / 256 of that storage.
    routed_active = routed_storage * TOP_K // ROUTED if routed_storage else 0

    mla = buckets.get("mla_fp8_pair", {}).get("bytes", 0)
    shared = buckets.get("shared_fp8_pair", {}).get("bytes", 0)
    router = buckets.get("router", {}).get("bytes", 0)
    # router bucket includes gate.bias (F32, tiny) + gate.weight (BF16)
    lm = buckets.get("lm_head", {}).get("bytes", 0)

    active_five = {
        "mla_fp8_plus_scale": mla,
        "routed_fp4_plus_scale": routed_active,
        "shared_fp8_plus_scale": shared,
        "router_incl_bias": router,
        "lm_head_bf16": lm,
    }
    return {
        "storage_buckets": buckets,
        "routed_expert_weight_byte_sizes": dict(expert_sizes),
        "routed_experts_uniform": len(expert_sizes) <= 1,
        "active_five_from_manifest": active_five,
        "active_five_sum": sum(active_five.values()),
        "not_in_10280_gpu_unique": {
            "indexer_compressor_base": buckets.get("indexer_compressor", {}),
            "mhc_f32": buckets.get("mhc_f32", {}),
            "norms": buckets.get("norms", {}),
            "hash_tid2eid": buckets.get("hash_tid2eid", {}),
            "mla_aux_norm_sink": buckets.get("mla_aux_norm_sink", {}),
            "embed_full_table_not_served": buckets.get("embed_full_table", {}),
            "mtp_excluded": buckets.get("mtp_excluded", {}),
        },
        "note": (
            "routed_all_256 is STORAGE of every expert. ACTIVE billed as "
            f"top-{TOP_K}/{ROUTED} of that mass because every expert has the "
            "same stored bytes (verified by routed_experts_uniform)."
        ),
    }


def read_tensor(tensors: dict[str, Any], name: str, max_bytes: int | None = None) -> bytes:
    meta = tensors[name]
    buf = bytearray()
    for seg in meta["segments"]:
        if max_bytes is not None and len(buf) >= max_bytes:
            break
        path = ARTIFACT / seg["chunk_relpath"]
        data = path.read_bytes()
        if len(data) != int(seg["bytes"]):
            raise RuntimeError(
                f"{name} segment {seg['chunk_relpath']} size {len(data)} != {seg['bytes']}"
            )
        buf.extend(data)
    if max_bytes is not None:
        return bytes(buf[:max_bytes])
    if len(buf) != int(meta["bytes"]):
        raise RuntimeError(f"{name} assembled {len(buf)} != {meta['bytes']}")
    return bytes(buf)


def entropy_of_bytes(raw: bytes, kind: str) -> dict[str, Any]:
    if kind == "fp4_nibbles":
        counts: Counter[int] = Counter()
        for b in raw:
            counts[b & 0x0F] += 1
            counts[b >> 4] += 1
        n = 2 * len(raw)
        return {
            "kind": kind,
            "symbols": 16,
            "n": n,
            "unique": len(counts),
            "entropy_bits": shannon_bits(counts, n),
            "zero_frac": counts[0] / n if n else 0.0,
            "histogram": {str(k): counts[k] for k in range(16)},
        }
    if kind == "fp8_bytes":
        counts = Counter(raw)
        n = len(raw)
        return {
            "kind": kind,
            "symbols": 256,
            "n": n,
            "unique": len(counts),
            "entropy_bits": shannon_bits(counts, n),
            "zero_frac": counts[0] / n if n else 0.0,
            "nan_frac": (counts[0x7F] + counts[0xFF]) / n if n else 0.0,
        }
    if kind == "bf16_bytes":
        # Treat as 16-bit codes (byte-pair little-endian).
        counts = Counter()
        n = len(raw) // 2
        for i in range(n):
            counts[raw[2 * i] | (raw[2 * i + 1] << 8)] += 1
        return {
            "kind": kind,
            "symbols": 65536,
            "n": n,
            "unique": len(counts),
            "entropy_bits": shannon_bits(counts, n),
        }
    raise ValueError(kind)


def sample_entropy(tensors: dict[str, Any]) -> dict[str, Any]:
    samples = [
        ("layers.0.attn.wq_a.weight", "fp8_bytes", None),
        ("layers.0.attn.wq_b.weight", "fp8_bytes", 8_388_608),
        ("layers.0.attn.wo_a.weight", "fp8_bytes", 8_388_608),
        ("layers.21.attn.wq_a.weight", "fp8_bytes", None),
        ("layers.42.attn.wq_a.weight", "fp8_bytes", None),
        ("layers.0.ffn.shared_experts.w1.weight", "fp8_bytes", None),
        ("layers.21.ffn.shared_experts.w1.weight", "fp8_bytes", None),
        ("layers.0.ffn.experts.0.w1.weight", "fp4_nibbles", None),
        ("layers.0.ffn.experts.0.w2.weight", "fp4_nibbles", None),
        ("layers.0.ffn.experts.0.w3.weight", "fp4_nibbles", None),
        ("layers.21.ffn.experts.17.w1.weight", "fp4_nibbles", None),
        ("layers.42.ffn.experts.200.w1.weight", "fp4_nibbles", None),
        ("layers.3.ffn.gate.weight", "bf16_bytes", None),
        ("head.weight", "bf16_bytes", 8_388_608),
    ]
    # shared expert name may be shared_expert singular — probe.
    names = set(tensors)
    out = []
    for name, kind, cap in samples:
        if name not in names:
            # try singular shared
            alt = name.replace("shared_experts", "shared_expert")
            if alt in names:
                name = alt
            else:
                out.append({"name": name, "error": "missing"})
                continue
        raw = read_tensor(tensors, name, cap)
        rec = entropy_of_bytes(raw, kind)
        rec["name"] = name
        rec["sampled_bytes"] = len(raw)
        rec["tensor_bytes"] = int(tensors[name]["bytes"])
        rec["dtype"] = tensors[name]["dtype"]
        rec["shape"] = tensors[name]["shape"]
        out.append(rec)
    return {"samples": out}


def nearest_e2m1(value: float) -> int:
    best = 0
    best_d = abs(E2M1[0] - value)
    for i, cand in enumerate(E2M1):
        d = abs(cand - value)
        if d < best_d or (d == best_d and (i & 1) == 0 and (best & 1) != 0):
            best = i
            best_d = d
    return best


def rounded_ue8m0_for_max(amax: float, codec_max: float) -> int:
    floor = 1.0 / (1 << 30)
    clamped = amax if amax > floor else floor
    scaled = clamped * (1.0 / codec_max)
    raw = struct.unpack("<I", struct.pack("<f", scaled))[0]
    exp_field = (raw >> 23) & 0xFF
    mantissa = raw & 0x007FFFFF
    if exp_field == 0 or exp_field == 0xFF:
        raise ValueError("scale out of range")
    exponent = exp_field - 127 + (1 if mantissa else 0)
    e8 = exponent + 127
    if not 0 <= e8 <= 254:
        raise ValueError(f"e8m0 {e8}")
    return e8


def fp8_block_scale(scales: bytes, row: int, col: int, scale_cols: int) -> float:
    """UE8M0 at the 128x128 block containing (row, col)."""
    return decode_e8m0fnu(scales[(row // FP8_BLOCK) * scale_cols + (col // FP8_BLOCK)])


def requant_fp8_row_to_fp4(
    weight_row: bytes, scales: bytes, row: int, logical_k: int
) -> tuple[bytearray, bytearray]:
    """Decode one FP8 row (128x128 block scales) and requantize to native FP4+UE8M0/32."""
    packed = bytearray(logical_k // 2)
    out_scales = bytearray(logical_k // FP4_BLOCK)
    scale_cols = logical_k // FP8_BLOCK
    f32 = [0.0] * logical_k
    for col in range(logical_k):
        f32[col] = decode_e4m3fn(weight_row[col]) * fp8_block_scale(
            scales, row, col, scale_cols
        )
    for block in range(logical_k // FP4_BLOCK):
        sl = f32[block * FP4_BLOCK : (block + 1) * FP4_BLOCK]
        amax = max(abs(v) for v in sl)
        sb = rounded_ue8m0_for_max(amax if amax > 0 else 1e-30, E2M1_MAX)
        scale = decode_e8m0fnu(sb)
        out_scales[block] = sb
        for i, v in enumerate(sl):
            nib = nearest_e2m1(v / scale)
            idx = block * FP4_BLOCK + i
            if idx & 1 == 0:
                packed[idx // 2] = (packed[idx // 2] & 0xF0) | nib
            else:
                packed[idx // 2] = (packed[idx // 2] & 0x0F) | (nib << 4)
    return packed, out_scales


def act_quant_f32(x: list[float]) -> tuple[bytearray, bytearray]:
    n = len(x)
    act = bytearray(n)
    scales = bytearray(n // FP8_BLOCK)
    # reuse E4M3 nearest via exhaustive table
    table = []
    for raw in range(256):
        try:
            table.append((raw, decode_e4m3fn(raw)))
        except ValueError:
            continue
    for b in range(n // FP8_BLOCK):
        sl = x[b * FP8_BLOCK : (b + 1) * FP8_BLOCK]
        amax = max(abs(v) for v in sl)
        sb = rounded_ue8m0_for_max(amax if amax > 0 else 1e-30, E4M3_MAX)
        scale = decode_e8m0fnu(sb)
        scales[b] = sb
        for i, v in enumerate(sl):
            target = max(-E4M3_MAX, min(E4M3_MAX, v / scale))
            best = 0
            best_d = 1e30
            for raw, cand in table:
                d = abs(cand - target)
                if d < best_d or (d == best_d and (raw & 1) == 0 and (best & 1) != 0):
                    best = raw
                    best_d = d
            act[b * FP8_BLOCK + i] = best
    return act, scales


def matvec_fp8(
    act: bytes, act_scales: bytes, weight: bytes, w_scales: bytes, rows: int, k: int
) -> list[float]:
    out = [0.0] * rows
    sc = k // FP8_BLOCK
    for r in range(rows):
        acc = 0.0
        wr = r * k
        for b in range(sc):
            block = 0.0
            base = b * FP8_BLOCK
            for c in range(FP8_BLOCK):
                block += decode_e4m3fn(act[base + c]) * decode_e4m3fn(
                    weight[wr + base + c]
                )
            acc += (
                block
                * decode_e8m0fnu(act_scales[b])
                * fp8_block_scale(w_scales, r, base, sc)
            )
        out[r] = acc
    return out


def matvec_fp4(
    act: bytes, act_scales: bytes, packed: bytes, w_scales: bytes, rows: int, k: int
) -> list[float]:
    out = [0.0] * rows
    packed_k = k // 2
    sc = k // FP4_BLOCK
    for r in range(rows):
        acc = 0.0
        pr = r * packed_k
        sr = r * sc
        for b in range(sc):
            block = 0.0
            start = b * FP4_BLOCK
            for c in range(start, start + FP4_BLOCK):
                p = packed[pr + c // 2]
                nib = p & 0x0F if (c & 1) == 0 else p >> 4
                block += decode_e4m3fn(act[c]) * decode_e2m1fn(nib)
            acc += (
                block
                * decode_e8m0fnu(act_scales[start // FP8_BLOCK])
                * decode_e8m0fnu(w_scales[sr + b])
            )
        out[r] = acc
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def requant_probe(tensors: dict[str, Any]) -> dict[str, Any]:
    """FP8→native-FP4 requant of layers.0.attn.wq_a, then GEMV vs captured X.

    Uses the first retained L0 hidden (4096 f32). This is a contract preview,
    not a production numeric seal.
    """
    w_name = "layers.0.attn.wq_a.weight"
    s_name = "layers.0.attn.wq_a.scale"
    if w_name not in tensors or s_name not in tensors:
        return {"error": "wq_a pair missing"}
    weight = read_tensor(tensors, w_name)
    scales = read_tensor(tensors, s_name)
    rows, k = 1024, 4096
    expected_scales = (rows // FP8_BLOCK) * (k // FP8_BLOCK)
    if len(weight) != rows * k or len(scales) != expected_scales:
        return {
            "error": "wq_a geometry mismatch",
            "w": len(weight),
            "s": len(scales),
            "expected_scales": expected_scales,
        }

    x_path = CAPTURE / "hidden" / "L00" / "vocab_bos_v1" / "000000.f32le"
    if not x_path.is_file():
        return {"error": "capture X missing", "path": str(x_path)}
    x = list(struct.unpack(f"<{HIDDEN}f", x_path.read_bytes()))
    act, act_s = act_quant_f32(x)

    packed = bytearray(rows * (k // 2))
    fp4_scales = bytearray(rows * (k // FP4_BLOCK))
    for r in range(rows):
        p, s = requant_fp8_row_to_fp4(weight[r * k : (r + 1) * k], scales, r, k)
        packed[r * (k // 2) : (r + 1) * (k // 2)] = p
        fp4_scales[r * (k // FP4_BLOCK) : (r + 1) * (k // FP4_BLOCK)] = s

    y8 = matvec_fp8(act, act_s, weight, scales, rows, k)
    y4 = matvec_fp4(act, act_s, bytes(packed), bytes(fp4_scales), rows, k)
    diffs = [abs(a - b) for a, b in zip(y8, y4)]
    recon = []
    scale_cols = k // FP8_BLOCK
    for col in range(k):
        orig = decode_e4m3fn(weight[col]) * fp8_block_scale(scales, 0, col, scale_cols)
        p = packed[col // 2]
        nib = p & 0x0F if (col & 1) == 0 else p >> 4
        rec = decode_e2m1fn(nib) * decode_e8m0fnu(fp4_scales[col // FP4_BLOCK])
        recon.append(abs(orig - rec))
    return {
        "organ": w_name,
        "rows": rows,
        "k": k,
        "x": str(x_path.relative_to(CAPTURE.parent.parent.parent.parent))
        if False
        else str(x_path),
        "matvec_cosine_fp8_vs_fp4": cosine(y8, y4),
        "matvec_max_abs": max(diffs),
        "matvec_mean_abs": sum(diffs) / len(diffs),
        "matvec_rel_l2": math.sqrt(sum(d * d for d in diffs))
        / (math.sqrt(sum(v * v for v in y8)) or 1.0),
        "weight_row0_max_abs": max(recon),
        "weight_row0_mean_abs": sum(recon) / len(recon),
        "note": (
            "CPU source-algorithm preview. Breaks hc_sha by construction. "
            "ALU cost of the extra decode is affordable (DSV4F ALU 0.77% idle)."
        ),
    }


def project_candidates(geo: dict[str, Any]) -> list[dict[str, Any]]:
    c = geo["classes"]
    mla = c["mla_fp8_plus_scale"]["bytes"]
    routed = c["routed_fp4_plus_scale"]["bytes"]
    shared = c["shared_fp8_plus_scale"]["bytes"]
    router = c["router_bf16"]["bytes"]
    lm = c["lm_head_bf16"]["bytes"]
    w_mla = c["mla_fp8_plus_scale"]["weights"]
    w_routed = c["routed_fp4_plus_scale"]["weights"]
    w_shared = c["shared_fp8_plus_scale"]["weights"]
    w_router = c["router_bf16"]["weights"]
    w_lm = c["lm_head_bf16"]["weights"]

    def pack(weights: int, bpw: float) -> int:
        return int(math.ceil(weights * bpw / 8.0))

    def row(
        name: str,
        parts: dict[str, int],
        *,
        contract: str,
        hc_sha: str,
        mechanism: str,
        alu: str,
    ) -> dict[str, Any]:
        total = sum(parts.values())
        return {
            "name": name,
            "mechanism": mechanism,
            "bytes": total,
            "gb": total / 1e9,
            "parts": parts,
            "numeric_contract": contract,
            "hc_sha": hc_sha,
            "alu_note": alu,
        }

    keep = "c94da765c4bbf795b598d96209cd80821e5a81ab97a8712586f54b8c8b612597"
    native = {
        "mla": mla,
        "routed": routed,
        "shared": shared,
        "router": router,
        "lm_head": lm,
    }
    rows = [
        row(
            "C0_current_native",
            native,
            contract=(
                "bit-identity of the BOS graph HC BF16: "
                f"{keep}. e2e asserts this. 0 fallbacks. greedy token=5."
            ),
            hc_sha=keep,
            mechanism="no change; native FP8 MLA / FP4 routed / FP8 shared / BF16 head+router",
            alu="ALU idle 0.77%; occupancy is the remaining GPU lever at these bytes",
        ),
        row(
            "C1_lossless_entropy_placeholder",
            native,  # filled after entropy
            contract=(
                "bit-identity PRESERVED: decoder reconstructs the exact native "
                "E2M1/E4M3/E8M0 codes before the existing matvec. hc_sha stays "
                f"{keep}."
            ),
            hc_sha=keep,
            mechanism=(
                "ANS/Huffman of already-quantized native codes. Does not drop levels. "
                "Reconstruction work is extra ALU; DSV4F can afford it (opposite of Q80)."
            ),
            alu="MORE reconstruction is affordable here because ALU is idle; bytes are the scarce resource",
        ),
        row(
            "C2_mla_shared_fp8_to_fp4_lm_fp8",
            {
                "mla": pack(w_mla, 4.25),
                "routed": routed,
                "shared": pack(w_shared, 4.25),
                "router": pack(w_router, 8.0),
                "lm_head": pack(w_lm, 8.0005),
            },
            contract=(
                "NEW hc_sha (bit-identity broken). Proposed quality bar: "
                "BOS greedy token-id unchanged; |logit-16.78185|<=0.05 (existing "
                "oracle slack); final-HC cosine vs sealed BF16 HC >= 0.995; "
                "max-abs on wq_a probe reported in this receipt. "
                "The e2e assert MUST be rewritten to the new sealed hash plus "
                "these numeric bounds — do not silently drop the assert."
            ),
            hc_sha="MUST_RESEAL — any requant changes c94da765 by construction",
            mechanism=(
                "Requantize still-wide classes onto the SAME native family DSV4F "
                "already executes: MLA+shared FP8 E4M3 → FP4 E2M1+UE8M0/32; "
                "lm_head BF16→FP8 E4M3+UE8M0/128; router BF16→FP8. Routed experts "
                "stay native 16-level FP4. Q80 mixed rates are not used."
            ),
            alu="decode path already exists (FP4/FP8 kernels). Extra ALU vs current FP8 MLA is cheap; ALU is 0.77%.",
        ),
        row(
            "C3_all_fp4_including_head",
            {
                "mla": pack(w_mla, 4.25),
                "routed": routed,
                "shared": pack(w_shared, 4.25),
                "router": pack(w_router, 4.25),
                "lm_head": pack(w_lm, 4.25),
            },
            contract=(
                "NEW hc_sha. Same greedy-token + cosine>=0.99 + logit slack. "
                "LM head is currently 341 GB/s / 82.8% of ceiling — compressing "
                "it is the one organ where bytes map 1:1 onto GPU microseconds."
            ),
            hc_sha="MUST_RESEAL",
            mechanism="C2 plus lm_head and router also to native FP4.",
            alu="lm_head is bandwidth-saturated already; FP4 decode tax is the Q80-opposite trade and is acceptable",
        ),
        row(
            "C4_c2_plus_topk4",
            {
                "mla": pack(w_mla, 4.25),
                "routed": routed * 4 // 6,
                "shared": pack(w_shared, 4.25),
                "router": pack(w_router, 8.0),
                "lm_head": pack(w_lm, 8.0005),
            },
            contract=(
                "NEW hc_sha AND a routing contract: top-4 of 256 instead of top-6. "
                "This is fewer ACTIVE weights, not a lower BPW. Quality must be "
                "re-proven on the fullseq capture set, not just BOS greedy."
            ),
            hc_sha="MUST_RESEAL — routing change moves HC even with identical tensors",
            mechanism="C2 representation plus num_experts_per_tok 6→4 (source config is 6).",
            alu="fewer expert matvecs; occupancy-neutral; bytes drop linearly",
        ),
        row(
            "C5_c3_plus_routed_3bit",
            {
                "mla": pack(w_mla, 4.25),
                "routed": pack(w_routed, 3.25),  # 8-level + 0.25 scale
                "shared": pack(w_shared, 4.25),
                "router": pack(w_router, 4.25),
                "lm_head": pack(w_lm, 4.25),
            },
            contract=(
                "NEW hc_sha. Second quantization of already 16-level expert codes "
                "down to 8 levels. Requires a measured expert-output cosine "
                "(propose >=0.99 vs native FP4) AND fullseq greedy-token match "
                "rate. Q80's 0.14-1.13 expert BPW does NOT apply — those started "
                "from BF16, not E2M1."
            ),
            hc_sha="MUST_RESEAL",
            mechanism=(
                "Drop routed experts from 16-level E2M1 to 8-level (3 BPW + 0.25 scale). "
                "This is a NEW codec on already-quantized data."
            ),
            alu="cheaper decode than E2M1 (smaller LUT). Bytes, not ALU, are why you would do this.",
        ),
        row(
            "C6_attention_2bpw_experts_2bpw",
            {
                "mla": pack(w_mla, 2.0),
                "routed": pack(w_routed, 2.0),
                "shared": pack(w_shared, 2.0),
                "router": pack(w_router, 2.0),
                "lm_head": pack(w_lm, 2.0),
            },
            contract=(
                "NEW hc_sha. This is a research codec (structured residual / "
                "low-rank+codes on already-quantized native values), not a "
                "native-family requant. Propose: final-HC cosine >=0.98 and "
                "fullseq token match as the admission bar. Not bit-identical."
            ),
            hc_sha="MUST_RESEAL",
            mechanism=(
                "Sub-native-FP4 on every class. Only path into the 3-4 GB band "
                "without cutting top-k. Reconstruction will be heavier than "
                "E2M1; DSV4F's idle ALU is the reason this is even discussable."
            ),
            alu="Q80 could not afford this (reconstruction was the wall). DSV4F can, IF the codec actually cuts bytes.",
        ),
        row(
            "C7_half_gb_fantasy",
            {
                "mla": pack(w_mla, 0.31),
                "routed": pack(w_routed, 0.31),
                "shared": pack(w_shared, 0.31),
                "router": pack(w_router, 0.31),
                "lm_head": pack(w_lm, 0.31),
            },
            contract="not a serious contract — arithmetic only",
            hc_sha="N/A",
            mechanism=(
                "The 0.5 GB figure that would make 20 ms possible at today's "
                "6.3% of roof. Requires ~0.31 active BPW on 12.75B weights, "
                "or deleting ~95% of the forward. Not reachable on this model."
            ),
            alu="irrelevant; the weights would have to not exist",
        ),
    ]
    return rows


def main() -> int:
    geo = geometry_census()
    print(
        f"geometry total {geo['total_unique_weight_scale_bytes']} "
        f"({geo['total_gb']:.6f} GB) bpw={geo['active_decode_bpw']:.6f}",
        file=sys.stderr,
    )

    print("loading manifest…", file=sys.stderr)
    manifest = load_manifest()
    man = manifest_census(manifest["tensors"])
    print(
        f"manifest active-five sum {man['active_five_sum']} "
        f"uniform_experts={man['routed_experts_uniform']}",
        file=sys.stderr,
    )

    print("entropy samples…", file=sys.stderr)
    ent = sample_entropy(manifest["tensors"])
    for s in ent["samples"]:
        if "error" in s:
            print(f"  {s['name']}: {s['error']}", file=sys.stderr)
        else:
            print(
                f"  {s['name']}: H={s['entropy_bits']:.4f} bits "
                f"unique={s['unique']}/{s.get('symbols')}",
                file=sys.stderr,
            )

    # Fill C1 from measured entropy (mean per class).
    fp8_hs = [
        s["entropy_bits"]
        for s in ent["samples"]
        if s.get("kind") == "fp8_bytes" and "attn" in s.get("name", "")
    ]
    fp4_hs = [
        s["entropy_bits"]
        for s in ent["samples"]
        if s.get("kind") == "fp4_nibbles"
    ]
    shared_hs = [
        s["entropy_bits"]
        for s in ent["samples"]
        if s.get("kind") == "fp8_bytes" and "shared" in s.get("name", "")
    ]
    bf16_head = next(
        (s["entropy_bits"] for s in ent["samples"] if s.get("name") == "head.weight"),
        None,
    )
    bf16_router = next(
        (
            s["entropy_bits"]
            for s in ent["samples"]
            if s.get("kind") == "bf16_bytes" and "gate" in s.get("name", "")
        ),
        None,
    )
    mean_fp8 = sum(fp8_hs) / len(fp8_hs) if fp8_hs else 8.0
    mean_fp4 = sum(fp4_hs) / len(fp4_hs) if fp4_hs else 4.0
    mean_shared = sum(shared_hs) / len(shared_hs) if shared_hs else mean_fp8
    # lossless BPW = code entropy + existing scale BPW
    mla_scale_bpw = geo["classes"]["mla_fp8_plus_scale"]["bpw"] - 8.0
    routed_scale_bpw = geo["classes"]["routed_fp4_plus_scale"]["bpw"] - 4.0
    shared_scale_bpw = geo["classes"]["shared_fp8_plus_scale"]["bpw"] - 8.0
    c1_mla_bpw = mean_fp8 + mla_scale_bpw
    c1_routed_bpw = mean_fp4 + routed_scale_bpw
    c1_shared_bpw = mean_shared + shared_scale_bpw
    c1_lm_bpw = bf16_head if bf16_head is not None else 16.0
    c1_router_bpw = bf16_router if bf16_router is not None else 16.0

    print("requant probe (wq_a FP8→FP4 vs captured X)…", file=sys.stderr)
    probe = requant_probe(manifest["tensors"])
    print(json.dumps({k: probe[k] for k in probe if k != "note"}, indent=2), file=sys.stderr)

    cands = project_candidates(geo)
    # overwrite C1 parts with measured lossless floors (plus 0.5% ANS overhead)
    overhead = 1.005
    c = geo["classes"]

    def pack(weights: int, bpw: float) -> int:
        return int(math.ceil(weights * bpw * overhead / 8.0))

    for cand in cands:
        if cand["name"] == "C1_lossless_entropy_placeholder":
            cand["name"] = "C1_lossless_entropy_of_native_codes"
            cand["parts"] = {
                "mla": pack(c["mla_fp8_plus_scale"]["weights"], c1_mla_bpw),
                "routed": pack(c["routed_fp4_plus_scale"]["weights"], c1_routed_bpw),
                "shared": pack(c["shared_fp8_plus_scale"]["weights"], c1_shared_bpw),
                "router": pack(c["router_bf16"]["weights"], c1_router_bpw),
                "lm_head": pack(c["lm_head_bf16"]["weights"], c1_lm_bpw),
            }
            cand["bytes"] = sum(cand["parts"].values())
            cand["gb"] = cand["bytes"] / 1e9
            cand["measured_lossless_bpw"] = {
                "mla_fp8_codes": mean_fp8,
                "mla_plus_scale": c1_mla_bpw,
                "routed_fp4_codes": mean_fp4,
                "routed_plus_scale": c1_routed_bpw,
                "shared_fp8_codes": mean_shared,
                "shared_plus_scale": c1_shared_bpw,
                "lm_head_bf16_codes": c1_lm_bpw,
                "router_bf16_codes": c1_router_bpw,
                "ans_overhead": overhead,
            }

    diagnosis = json.loads(DIAGNOSIS.read_text()) if DIAGNOSIS.is_file() else {}
    doc = {
        "schema": "hawking.ascension.dsv4f_byte_reduction.v1",
        "date": "2026-08-16",
        "artifact": {
            "path": str(ARTIFACT),
            "seal_sha256": manifest.get("seal_sha256"),
            "schema": manifest.get("schema"),
            "status": manifest.get("status"),
            "source": "deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1",
        },
        "hc_sha_gate": {
            "value": "c94da765c4bbf795b598d96209cd80821e5a81ab97a8712586f54b8c8b612597",
            "asserted_by": "crates/hawking-core/tests/dsv4f_native_token_graph_e2e.rs SEALED_GRAPH_HC_BF16_SHA256",
            "what_it_hashes": "final HC state as BF16 bits after the 43-layer BOS graph",
            "cpu_oracle_is_not_this_gate": True,
            "any_representation_change_breaks_it": True,
        },
        "alu_is_idle": {
            "utilization": 0.0077,
            "source": "DSV4F_GPU_BODY_DIAGNOSIS.json discriminator.alu_idle_proof",
            "consequence": (
                "Unlike Q80, a codec that spends more reconstruction work is "
                "affordable on DSV4F IF it cuts bytes. Rank candidates by bytes "
                "first, reconstruction cost second."
            ),
        },
        "geometry_census": geo,
        "manifest_census": man,
        "entropy": ent,
        "requant_probe": probe,
        "candidates": cands,
        "diagnosis_byte_budget": diagnosis.get("byte_budget_unique_stored"),
        "q80_rates_do_not_transfer": {
            "q80_mixed_complete_physical_bpw": 1.4444457,
            "q80_expert_started_from": "BF16-derived mixed (binary/rice/low-rank)",
            "dsv4f_experts_are": "source-native 16-level E2M1FN + UE8M0/32 (4.25 BPW)",
            "dsv4f_mla_is": "source-native E4M3FN + UE8M0 128x128 (8.0005 BPW)",
            "why": (
                "Q80's 0.14-1.13 expert organ rates measure redundancy in BF16. "
                "DSV4F experts are already 16 reconstruction levels. Applying "
                "those rates here is a unit error."
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
