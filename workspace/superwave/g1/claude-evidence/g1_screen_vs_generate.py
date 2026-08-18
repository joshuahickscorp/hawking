#!/usr/bin/env python3
"""Screen-vs-generate discriminator. CPU/numpy. No GPU, no generate, no pack.

Writes /tmp/g1_screen_vs_generate.json. Reads artifacts and BF16 parent only.
"""
from __future__ import annotations

import json
import math
import os
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ART = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b")
SRC = ART / "bf16"
CAP = ART / "activation-capture-v1"
G0 = ART / "uniform-q4-v1"
OUT = Path("/tmp/g1_screen_vs_generate.json")

HIDDEN = 5120
VOCAB = 248320
INTER = 17408
STOP_LO, STOP_HI = 248044, 248076  # inclusive
EOS = 248046  # <|im_end|>
THINK = 248068
IM_START = 248045
EOT = 248044
PARIS = 11751
ASSISTANT = 74455

SPECIAL = {
    248044: "<|endoftext|>",
    248045: "<|im_start|>",
    248046: "<|im_end|>",
    248068: "<think>",
    248069: "</think>",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return num / den if den > 1e-12 else 0.0


def mean_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.mean(num[ok] / den[ok]))


def min_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.min(num[ok] / den[ok]))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a))
    return num / den if den > 1e-12 else num


# ---------------------------------------------------------------------------
# BF16
# ---------------------------------------------------------------------------

_WMAP = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
_HEADER_CACHE: dict[Path, dict] = {}


def read_header(shard: Path) -> dict:
    if shard not in _HEADER_CACHE:
        with shard.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _HEADER_CACHE[shard] = json.loads(fh.read(n))
    return _HEADER_CACHE[shard]


def tensor_loc(name: str) -> tuple[Path, int, int, tuple[int, ...], str, int]:
    shard = SRC / _WMAP[name]
    header = read_header(shard)
    info = header[name]
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
    data0 = 8 + n
    shape = tuple(int(x) for x in info["shape"])
    return shard, data0 + lo, data0 + hi, shape, info.get("dtype", "BF16"), n


def load_tensor(name: str) -> np.ndarray:
    shard, lo, hi, shape, dtype, _ = tensor_loc(name)
    with shard.open("rb") as fh:
        fh.seek(lo)
        raw = fh.read(hi - lo)
    if dtype in ("BF16", "BFLOAT16"):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return np.ascontiguousarray(u32.view(np.float32).reshape(shape))
    if dtype in ("F32", "FLOAT32"):
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    raise RuntimeError(f"unsupported dtype {dtype} for {name}")


def load_bf16_rows(name: str, rows: np.ndarray) -> np.ndarray:
    shard, lo, hi, shape, dtype, _ = tensor_loc(name)
    assert dtype in ("BF16", "BFLOAT16")
    cols = shape[1]
    row_bytes = cols * 2
    out = np.empty((len(rows), cols), dtype=np.float32)
    with shard.open("rb") as fh:
        for i, r in enumerate(rows):
            fh.seek(lo + int(r) * row_bytes)
            raw = fh.read(row_bytes)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            out[i] = u32.view(np.float32)
    return out


def load_hidden(layer: int) -> np.ndarray:
    path = CAP / "hidden" / f"L{layer:02d}.f32"
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != 256 * HIDDEN:
        raise RuntimeError(f"hidden L{layer} size {raw.size}")
    return np.ascontiguousarray(raw.reshape(256, HIDDEN))


# ---------------------------------------------------------------------------
# codecs (requantize from BF16)
# ---------------------------------------------------------------------------

def group_pad(W: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    groups = (n + group_size - 1) // group_size
    padded = np.zeros((groups, group_size), dtype=np.float32)
    padded.reshape(-1)[:n] = flat
    return padded, n


def hgravu_absmax_recon(W: np.ndarray, bits: int, group_size: int = 64) -> np.ndarray:
    """HGRAVU01: scale=f16(max/bound), q in [-bound, bound], bound=2^(bits-1)-1."""
    bound = (1 << (bits - 1)) - 1
    padded, n = group_pad(W, group_size)
    absmax = np.max(np.abs(padded), axis=1)
    scale = (absmax / max(bound, 1)).astype(np.float32)
    # snap to f16 like the packer
    scale = scale.astype(np.float16).astype(np.float32)
    den = np.where(scale > 0.0, scale, 1.0)
    codes = np.rint(padded / den[:, None]).clip(-bound, bound)
    recon = (codes * scale[:, None]).reshape(-1)[:n]
    return recon.reshape(W.shape).astype(np.float32)


def hq30uq4_recon(W: np.ndarray, group_size: int = 64) -> np.ndarray:
    """G0 HQ30UQ4: scale=f16(max/7), q in [-8, 7], nibble-8."""
    padded, n = group_pad(W, group_size)
    absmax = np.max(np.abs(padded), axis=1)
    scale = (absmax / 7.0).astype(np.float32)
    scale = scale.astype(np.float16).astype(np.float32)
    den = np.where(scale > 0.0, scale, 1.0)
    codes = np.rint(padded / den[:, None]).clip(-8, 7)
    recon = (codes * scale[:, None]).reshape(-1)[:n]
    return recon.reshape(W.shape).astype(np.float32)


def binary_g128_recon(W: np.ndarray, group_size: int = 128) -> np.ndarray:
    """HGRAVB01: per-group mean-abs scale, sign * scale. Matches L0 gate 0.79836."""
    padded, n = group_pad(W, group_size)
    scale = np.mean(np.abs(padded), axis=1).astype(np.float32)
    scale = scale.astype(np.float16).astype(np.float32)
    signs = np.where(padded >= 0.0, 1.0, -1.0)
    recon = (signs * scale[:, None]).reshape(-1)[:n]
    return recon.reshape(W.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# packed decoders
# ---------------------------------------------------------------------------

def parse_catalog(root: Path) -> list[dict]:
    raw = (root / "catalog.hq38m20").read_bytes()
    assert raw[:8] == b"HQ38M20\0", raw[:8]
    _ver, n_tensors, n_segments = struct.unpack_from("<III", raw, 8)
    name_blob_bytes = struct.unpack_from("<I", raw, 24)[0]
    cur = 32
    by_id = {}
    for _ in range(n_segments):
        sid, nlen = struct.unpack_from("<HH", raw, cur)
        filename = raw[cur + 44 : cur + 44 + nlen].decode()
        by_id[sid] = filename
        cur += 44 + nlen
    table = raw[cur : cur + n_tensors * 128]
    cur += n_tensors * 128
    blob = raw[cur : cur + name_blob_bytes]
    rows = []
    for i in range(n_tensors):
        rec = table[i * 128 : (i + 1) * 128]
        name_off, name_len = struct.unpack_from("<IH", rec, 0)
        codec = rec[6]
        ndim = rec[8]
        shape = [struct.unpack_from("<I", rec, 12 + d * 4)[0] for d in range(ndim)]
        seg = struct.unpack_from("<H", rec, 36)[0]
        off, nbytes = struct.unpack_from("<QQ", rec, 40)
        name = blob[name_off : name_off + name_len].decode()
        segname = by_id[seg]
        path = Path(segname)
        if not path.is_absolute():
            cand = root / segname
            if not cand.exists():
                cand = root / "segments" / Path(segname).name
            path = cand
        rows.append(
            dict(name=name, codec=codec, shape=shape, path=str(path), off=off, nbytes=nbytes)
        )
    return rows


def catalog_row(rows: list[dict], suffix: str) -> dict:
    for r in rows:
        if r["name"].endswith(suffix) or r["name"] == suffix:
            return r
    raise KeyError(suffix)


def split_hgravu(payload: bytes) -> tuple[dict, bytes]:
    assert payload[:8] == b"HGRAVU01", payload[:8]
    hlen = struct.unpack_from("<I", payload, 8)[0]
    hdr = json.loads(payload[12 : 12 + hlen])
    body = payload[12 + hlen :]
    return hdr, body


def hgravu_extract_codes_vec(codes_u: np.ndarray, element0: int, n: int, bits: int) -> np.ndarray:
    """LSB-first unsigned extract, matching extract_unsigned / gk_uniform_extract."""
    bit0 = (element0 + np.arange(n, dtype=np.int64)) * bits
    byte0 = bit0 >> 3
    shift = (bit0 & 7).astype(np.uint8)
    packed = codes_u[byte0].astype(np.uint16)
    span = (shift.astype(np.int32) + bits) > 8
    if np.any(span):
        packed = packed.copy()
        packed[span] |= codes_u[byte0[span] + 1].astype(np.uint16) << 8
    return ((packed >> shift) & ((1 << bits) - 1)).astype(np.int16)


def decode_hgravu_rows(path: Path, off: int, nbytes: int, rows: np.ndarray) -> np.ndarray:
    with path.open("rb") as fh:
        fh.seek(off)
        payload = fh.read(nbytes)
    hdr, body = split_hgravu(payload)
    bits = int(hdr["bits"])
    cols = int(hdr["shape"][1])
    group = int(hdr["group_size"])
    bound = (1 << (bits - 1)) - 1
    scale_bytes = int(hdr["scale_bytes"])
    scales = np.frombuffer(body[:scale_bytes], dtype="<f2").astype(np.float32)
    codes_u = np.frombuffer(body[scale_bytes : scale_bytes + int(hdr["code_bytes"])], dtype=np.uint8)
    out = np.empty((len(rows), cols), dtype=np.float32)
    for i, r in enumerate(rows):
        r = int(r)
        element0 = r * cols
        q = hgravu_extract_codes_vec(codes_u, element0, cols, bits).astype(np.int32) - bound
        g_idx = (element0 + np.arange(cols)) // group
        out[i] = q.astype(np.float32) * scales[g_idx]
    return out


def decode_hq30uq4_rows(path: Path, rows: np.ndarray) -> np.ndarray:
    with path.open("rb") as fh:
        header = fh.read(40)
        assert header[:8] == b"HQ30UQ4\0", header[:8]
        group_size = struct.unpack_from("<I", header, 12)[0]
        rank = struct.unpack_from("<H", header, 16)[0]
        nrows = struct.unpack_from("<I", header, 32)[0]
        ncols = struct.unpack_from("<I", header, 36)[0]
        assert group_size == 64 and rank == 2
        n_groups = (nrows * ncols) // 64
        gpr = ncols // 64
        scale_off = 40
        code_off = 40 + n_groups * 2
        out = np.empty((len(rows), ncols), dtype=np.float32)
        for i, r in enumerate(rows):
            r = int(r)
            fh.seek(scale_off + r * gpr * 2)
            scales = np.frombuffer(fh.read(gpr * 2), dtype="<f2").astype(np.float32)
            fh.seek(code_off + r * gpr * 32)
            codes = np.frombuffer(fh.read(gpr * 32), dtype=np.uint8)
            # even nibble low, odd high; q = nibble - 8
            lo = (codes & 0x0F).astype(np.int16) - 8
            hi = (codes >> 4).astype(np.int16) - 8
            q = np.empty(ncols, dtype=np.int16)
            q[0::2] = lo
            q[1::2] = hi
            out[i] = q.astype(np.float32) * np.repeat(scales, 64)
    return out


def hq30uq4_logit_chunk(path: Path, hidden: np.ndarray, row0: int, nrow: int) -> np.ndarray:
    """logits[row0:row0+nrow] = W[row0:row0+nrow] @ hidden, W is HQ30UQ4."""
    with path.open("rb") as fh:
        header = fh.read(40)
        group_size = struct.unpack_from("<I", header, 12)[0]
        nrows = struct.unpack_from("<I", header, 32)[0]
        ncols = struct.unpack_from("<I", header, 36)[0]
        n_groups = (nrows * ncols) // 64
        gpr = ncols // 64
        scale_off = 40
        code_off = 40 + n_groups * 2
        fh.seek(scale_off + row0 * gpr * 2)
        scales = np.frombuffer(fh.read(nrow * gpr * 2), dtype="<f2").astype(np.float32).reshape(nrow, gpr)
        fh.seek(code_off + row0 * gpr * 32)
        codes = np.frombuffer(fh.read(nrow * gpr * 32), dtype=np.uint8).reshape(nrow, gpr * 32)
    lo = (codes & 0x0F).astype(np.int16) - 8
    hi = (codes >> 4).astype(np.int16) - 8
    q = np.empty((nrow, ncols), dtype=np.int16)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    W = q.astype(np.float32) * np.repeat(scales, 64, axis=1)
    return W @ hidden.astype(np.float32)


def hgravu_logit_chunk(path: Path, off: int, nbytes: int, hidden: np.ndarray, row0: int, nrow: int, hdr=None, body=None) -> np.ndarray:
    if hdr is None:
        with path.open("rb") as fh:
            fh.seek(off)
            payload = fh.read(nbytes)
        hdr, body = split_hgravu(payload)
    bits = int(hdr["bits"])
    cols = int(hdr["shape"][1])
    group = int(hdr["group_size"])
    bound = (1 << (bits - 1)) - 1
    scale_bytes = int(hdr["scale_bytes"])
    scales = np.frombuffer(body[:scale_bytes], dtype="<f2").astype(np.float32)
    codes = body[scale_bytes : scale_bytes + int(hdr["code_bytes"])]
    W = np.empty((nrow, cols), dtype=np.float32)
    for i in range(nrow):
        r = row0 + i
        element0 = r * cols
        q = hgravu_extract_codes(codes, element0, cols, bits).astype(np.int32) - bound
        g0 = element0 // group
        g_idx = g0 + (np.arange(cols) // group)
        W[i] = q.astype(np.float32) * scales[g_idx]
    return W @ hidden.astype(np.float32), hdr, body


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x64 = x.astype(np.float64)
    var = np.mean(x64 * x64, axis=-1, keepdims=True)
    y = x64 / np.sqrt(var + eps)
    return (y * w.astype(np.float64)).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def topk(logits: np.ndarray, k: int = 8) -> list[dict]:
    idx = np.argpartition(-logits, k)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return [
        dict(id=int(i), logit=float(logits[i]), name=SPECIAL.get(int(i)))
        for i in idx
    ]


def logit_stats(logits: np.ndarray) -> dict:
    order = np.argsort(-logits)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    mx = float(np.max(logits))
    # softmax mass on stop/control set
    z = logits - mx
    e = np.exp(z.astype(np.float64))
    p = e / e.sum()
    stop_ids = list(range(STOP_LO, STOP_HI + 1))
    return dict(
        argmax=int(order[0]),
        argmax_name=SPECIAL.get(int(order[0])),
        argmax_logit=float(logits[order[0]]),
        eos_rank=int(ranks[EOS]) + 1,
        eos_logit=float(logits[EOS]),
        think_rank=int(ranks[THINK]) + 1,
        think_logit=float(logits[THINK]),
        eot_rank=int(ranks[EOT]) + 1,
        eot_logit=float(logits[EOT]),
        paris_rank=int(ranks[PARIS]) + 1,
        paris_logit=float(logits[PARIS]),
        stop_set_prob=float(p[stop_ids].sum()),
        eos_prob=float(p[EOS]),
        think_prob=float(p[THINK]),
        top8=topk(logits, 8),
        margin_top1_minus_eos=float(logits[order[0]] - logits[EOS]),
        margin_think_minus_eos=float(logits[THINK] - logits[EOS]),
    )


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_identity() -> dict:
    man = json.loads((G0 / "manifest.json").read_text())
    g0_lm = next(t for t in man["tensors"] if t["name"].endswith("lm_head.weight"))
    g0_emb = next(t for t in man["tensors"] if t["name"].endswith("embed_tokens.weight"))
    g0_lm_p = G0 / "tensors" / g0_lm["artifact"]
    g0_emb_p = G0 / "tensors" / g0_emb["artifact"]
    s15_lm = ART / "mixed-sub15-v1/tensors" / g0_lm["artifact"]
    s15_emb = ART / "mixed-sub15-v1/tensors" / g0_emb["artifact"]
    mixed = {}
    for name in ("mixed-2p0-v1", "mixed-q4down-v1", "mixed-q3mlp-v1"):
        rows = parse_catalog(ART / name)
        lm = catalog_row(rows, "lm_head.weight")
        emb = catalog_row(rows, "embed_tokens.weight")
        mixed[name] = dict(
            lm_head_ino=os.stat(lm["path"]).st_ino,
            embed_ino=os.stat(emb["path"]).st_ino,
            lm_head_nlink=os.stat(lm["path"]).st_nlink,
            lm_magic=open(lm["path"], "rb").read(8).decode("latin1"),
            embed_magic=open(emb["path"], "rb").read(8).decode("latin1"),
            lm_path=lm["path"],
            embed_path=emb["path"],
            lm_off=lm["off"],
            lm_nbytes=lm["nbytes"],
            embed_off=emb["off"],
            embed_nbytes=emb["nbytes"],
        )
    return dict(
        g0_lm_ino=g0_lm_p.stat().st_ino,
        g0_emb_ino=g0_emb_p.stat().st_ino,
        g0_lm_nlink=g0_lm_p.stat().st_nlink,
        g0_emb_nlink=g0_emb_p.stat().st_nlink,
        g0_lm_path=str(g0_lm_p),
        g0_emb_path=str(g0_emb_p),
        sub15_lm_ino=s15_lm.stat().st_ino,
        sub15_emb_ino=s15_emb.stat().st_ino,
        sub15_shares_g0_lm=s15_lm.stat().st_ino == g0_lm_p.stat().st_ino,
        sub15_shares_g0_emb=s15_emb.stat().st_ino == g0_emb_p.stat().st_ino,
        mixed=mixed,
        mixed_share_one_lm=len({mixed[k]["lm_head_ino"] for k in mixed}) == 1,
        mixed_share_one_emb=len({mixed[k]["embed_ino"] for k in mixed}) == 1,
        g0_min_q4_cosine_field=man.get("min_q4_cosine"),
        g0_complete_bpw=man.get("complete_physical_bpw"),
    )


def stage_floor_products() -> dict:
    d = json.loads(Path("/tmp/g1_mlp_floor.json").read_text())
    by = defaultdict(list)
    for org in d["organs"]:
        for c in org["candidates"]:
            by[(org["role"], c["codec"])].append(
                dict(layer=org["layer"], hold=c.get("hold_output_cosine"), w=c.get("weight_cosine"))
            )

    def agg(rows, key):
        xs = [r[key] for r in rows if r[key] is not None]
        prod = 1.0
        for x in xs:
            prod *= x
        return dict(
            n=len(xs),
            min=min(xs),
            median=sorted(xs)[len(xs) // 2],
            max=max(xs),
            prod=prod,
            geomean=prod ** (1 / len(xs)),
            n_ge_095=sum(x >= 0.95 for x in xs),
            n_ge_099=sum(x >= 0.99 for x in xs),
        )

    out = {}
    want = [
        "uniform_q3_g64",
        "uniform_q4_g64",
        "binary_g128",
        "residual_rice_q1_rms_2pct",
        "hgravs01_r160_b3_act_thin",
        "hgravs01_r160_b3_packed_2p0",
    ]
    for role in ("gate_proj", "up_proj", "down_proj"):
        out[role] = {c: dict(hold=agg(by[(role, c)], "hold"), weight=agg(by[(role, c)], "w")) for c in want if (role, c) in by}

    # 192-tensor products for q3 / q4
    def prod_all(codec):
        holds = []
        for role in ("gate_proj", "up_proj", "down_proj"):
            holds.extend(r["hold"] for r in by[(role, codec)])
        p = 1.0
        for h in holds:
            p *= h
        return dict(n=len(holds), min=min(holds), prod=p, geomean=p ** (1 / len(holds)))

    out["mlp_192_q3"] = prod_all("uniform_q3_g64")
    out["mlp_192_q4"] = prod_all("uniform_q4_g64")
    # recipe-shaped products
    # q3mlp: all three uniform q3
    p = 1.0
    for role in ("gate_proj", "up_proj", "down_proj"):
        p *= out[role]["uniform_q3_g64"]["hold"]["prod"]
    out["recipe_q3mlp_mlp_hold_prod"] = p
    p = 1.0
    for role in ("gate_proj", "up_proj", "down_proj"):
        p *= out[role]["uniform_q4_g64"]["hold"]["prod"]
    out["recipe_g0_mlp_hold_prod"] = p
    # q4down: binary gate, residual up, q4 down
    out["recipe_q4down_mlp_hold_prod"] = (
        out["gate_proj"]["binary_g128"]["hold"]["prod"]
        * out["up_proj"]["residual_rice_q1_rms_2pct"]["hold"]["prod"]
        * out["down_proj"]["uniform_q4_g64"]["hold"]["prod"]
    )
    out["recipe_2p0_mlp_hold_prod_honest_hgravs"] = (
        out["gate_proj"]["binary_g128"]["hold"]["prod"]
        * out["up_proj"]["residual_rice_q1_rms_2pct"]["hold"]["prod"]
        * out["down_proj"]["hgravs01_r160_b3_act_thin"]["hold"]["prod"]
    )
    out["recipe_2p0_mlp_hold_prod_packed_insample"] = (
        out["gate_proj"]["binary_g128"]["hold"]["prod"]
        * out["up_proj"]["residual_rice_q1_rms_2pct"]["hold"]["prod"]
        * out["down_proj"]["hgravs01_r160_b3_packed_2p0"]["hold"]["prod"]
    )
    out["activation"] = d["activation"]
    out["ns014_rpd"] = 92 / 2048
    out["qwen38_gate_up_rpd"] = 256 / 5120
    out["qwen38_down_rpd"] = 256 / 17408
    return out


def stage_stop_rows(ident: dict) -> dict:
    stop_rows = np.arange(STOP_LO, STOP_HI + 1)
    control_rows = np.array(
        [0, 1, 2, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536, 11751, 200000, 209714, 209715, 209716],
        dtype=np.int64,
    )
    all_rows = np.unique(np.concatenate([stop_rows, control_rows]))
    log(f"load BF16 lm_head rows n={len(all_rows)}")
    bf = load_bf16_rows("language_model.lm_head.weight", all_rows)
    g0_p = Path(ident["g0_lm_path"])
    log("decode G0 HQ30UQ4 lm_head rows")
    g0 = decode_hq30uq4_rows(g0_p, all_rows)
    mixed = ident["mixed"]["mixed-q3mlp-v1"]
    log("decode HGRAVU01 lm_head rows")
    hv = decode_hgravu_rows(Path(mixed["lm_path"]), mixed["lm_off"], mixed["lm_nbytes"], all_rows)

    def pack(rows_idx, label):
        idx = [int(np.where(all_rows == r)[0][0]) for r in rows_idx]
        b = bf[idx]
        g = g0[idx]
        h = hv[idx]
        per = []
        for i, r in enumerate(rows_idx):
            per.append(
                dict(
                    row=int(r),
                    name=SPECIAL.get(int(r)),
                    g0_vs_bf16=cosine(b[i], g[i]),
                    hgravu_vs_bf16=cosine(b[i], h[i]),
                    g0_vs_hgravu=cosine(g[i], h[i]),
                    g0_rel_l2=rel_l2(b[i], g[i]),
                    hgravu_rel_l2=rel_l2(b[i], h[i]),
                    bf16_norm=float(np.linalg.norm(b[i])),
                    overflow_row=int(r) >= 209715,
                )
            )
        return dict(
            n=len(rows_idx),
            label=label,
            g0_vs_bf16_mean=mean_row_cosine(b, g),
            g0_vs_bf16_min=min_row_cosine(b, g),
            hgravu_vs_bf16_mean=mean_row_cosine(b, h),
            hgravu_vs_bf16_min=min_row_cosine(b, h),
            g0_vs_hgravu_mean=mean_row_cosine(g, h),
            rows=per,
        )

    # verify requant vs packed on a couple of rows
    sample = np.array([0, THINK, EOS, 209715], dtype=np.int64)
    bf_s = load_bf16_rows("language_model.lm_head.weight", sample)
    hq_req = hq30uq4_recon(bf_s)
    hv_req = hgravu_absmax_recon(bf_s, 4)
    g0_s = decode_hq30uq4_rows(g0_p, sample)
    hv_s = decode_hgravu_rows(Path(mixed["lm_path"]), mixed["lm_off"], mixed["lm_nbytes"], sample)
    verify = []
    for i, r in enumerate(sample):
        verify.append(
            dict(
                row=int(r),
                packed_g0_vs_requant=cosine(g0_s[i], hq_req[i]),
                packed_hgravu_vs_requant=cosine(hv_s[i], hv_req[i]),
                packed_g0_vs_bf16=cosine(g0_s[i], bf_s[i]),
                packed_hgravu_vs_bf16=cosine(hv_s[i], bf_s[i]),
            )
        )
    return dict(
        stop=pack(stop_rows, "stop_control_248044_248076"),
        control=pack(control_rows, "control_and_overflow_boundary"),
        packed_vs_requant=verify,
        overflow_element=1073741824,
        overflow_row_bits4_k5120=209715,
    )


def capture_sites() -> list[dict]:
    meta = json.loads((CAP / "capture-result.json").read_text())
    sites = []
    base = 0
    for pi, p in enumerate(meta["prompts"]):
        ids = p["ids"]
        n = p["n_tokens"]
        # first-token site: index of 198 after assistant, whose next id is think
        for i in range(n - 1):
            if ids[i] == 198 and i > 0 and ids[i - 1] == ASSISTANT and ids[i + 1] == THINK:
                sites.append(
                    dict(
                        kind="generate_first_token",
                        prompt=pi,
                        text=p["prompt"][:60],
                        t=base + i,
                        this_id=ids[i],
                        next_id=ids[i + 1],
                        next_name=SPECIAL.get(ids[i + 1]),
                    )
                )
            if ids[i] == THINK and ids[i + 1] == 198:
                sites.append(
                    dict(
                        kind="after_think",
                        prompt=pi,
                        text=p["prompt"][:60],
                        t=base + i,
                        this_id=ids[i],
                        next_id=ids[i + 1],
                        next_name="newline",
                    )
                )
        base += n
    return sites


def full_logits_for_hidden(hidden: np.ndarray, kind: str, ident: dict) -> np.ndarray:
    logits = np.empty(VOCAB, dtype=np.float32)
    chunk = 1024
    if kind == "bf16":
        shard, lo, hi, shape, dtype, _ = tensor_loc("language_model.lm_head.weight")
        row_bytes = HIDDEN * 2
        with shard.open("rb") as fh:
            for r0 in range(0, VOCAB, chunk):
                n = min(chunk, VOCAB - r0)
                fh.seek(lo + r0 * row_bytes)
                raw = fh.read(n * row_bytes)
                u16 = np.frombuffer(raw, dtype=np.uint16).reshape(n, HIDDEN)
                W = (u16.astype(np.uint32) << 16).view(np.float32)
                logits[r0 : r0 + n] = W @ hidden
        return logits
    if kind == "g0":
        path = Path(ident["g0_lm_path"])
        for r0 in range(0, VOCAB, chunk):
            n = min(chunk, VOCAB - r0)
            logits[r0 : r0 + n] = hq30uq4_logit_chunk(path, hidden, r0, n)
        return logits
    if kind == "hgravu":
        mixed = ident["mixed"]["mixed-q3mlp-v1"]
        path = Path(mixed["lm_path"])
        # load once
        with path.open("rb") as fh:
            fh.seek(mixed["lm_off"])
            payload = fh.read(mixed["lm_nbytes"])
        hdr, body = split_hgravu(payload)
        hdr_ref = [hdr]
        body_ref = [body]
        # python extract of all rows is slow; do 2048-row chunks
        # pre-extract is the bottleneck. Use a faster vectorized extract per chunk.
        bits = int(hdr["bits"])
        cols = int(hdr["shape"][1])
        group = int(hdr["group_size"])
        bound = (1 << (bits - 1)) - 1
        scale_bytes = int(hdr["scale_bytes"])
        scales = np.frombuffer(body[:scale_bytes], dtype="<f2").astype(np.float32)
        codes = np.frombuffer(body[scale_bytes : scale_bytes + int(hdr["code_bytes"])], dtype=np.uint8)
        # vectorized extract for a chunk
        for r0 in range(0, VOCAB, chunk):
            n = min(chunk, VOCAB - r0)
            element0 = r0 * cols
            n_el = n * cols
            bit0 = (element0 + np.arange(n_el, dtype=np.int64)) * bits
            byte0 = bit0 >> 3
            shift = (bit0 & 7).astype(np.uint8)
            packed = codes[byte0].astype(np.uint16)
            # may span
            span = (shift.astype(np.int32) + bits) > 8
            if np.any(span):
                packed = packed.copy()
                packed[span] |= codes[byte0[span] + 1].astype(np.uint16) << 8
            q = ((packed >> shift) & ((1 << bits) - 1)).astype(np.int16) - bound
            W = q.astype(np.float32).reshape(n, cols)
            g0 = (element0 + np.arange(n_el)) // group
            W *= scales[g0].reshape(n, cols)
            logits[r0 : r0 + n] = W @ hidden
        return logits
    raise ValueError(kind)


def stage_logits(ident: dict) -> dict:
    sites = capture_sites()
    log(f"capture sites: {len(sites)}")
    # L63 post-norm is the last captured site. Not confirmed final-norm.
    X = load_hidden(63)
    try:
        w_norm = load_tensor("language_model.model.norm.weight")
        norm_note = "applied language_model.model.norm.weight on L63 post-norm (UNCONFIRMED as true final-norm site)"
    except Exception as e:
        w_norm = np.ones(HIDDEN, dtype=np.float32)
        norm_note = f"norm load failed ({e}); identity"
    out_sites = []
    kinds = ("bf16", "g0", "hgravu")
    # restrict to first two prompts x two kinds of site = 4 sites to keep wall honest
    picked = []
    for kind in ("generate_first_token", "after_think"):
        picked.extend([s for s in sites if s["kind"] == kind][:2])
    cache_h = {}
    for s in picked:
        t = s["t"]
        h_raw = X[t]
        h = rmsnorm(h_raw, w_norm)
        rec = dict(site=s, hidden_l2=float(np.linalg.norm(h)), raw_l2=float(np.linalg.norm(h_raw)))
        for kind in kinds:
            key = (kind, t)
            log(f"logits {kind} t={t} kind={s['kind']}")
            t0 = time.time()
            logits = full_logits_for_hidden(h, kind, ident)
            stats = logit_stats(logits)
            stats["wall_s"] = time.time() - t0
            rec[kind] = stats
        # pairwise logit cosine
        # recompute cheap? we didn't keep vectors. skip or store top only.
        out_sites.append(rec)
    return dict(norm_note=norm_note, n_sites=len(out_sites), sites=out_sites, all_sites=sites)


def stage_residual_and_walk() -> dict:
    """Teacher-forced residual-proxy and student-forced MLP-error walk.

    X[l] = captured post-norm hidden (MLP input). Residual-proxy uses X as the
    skip base (same convention as g1-mse-scale-rule residual-proxy).
    """
    hold = np.arange(192, 256)  # last 64, matches floor hold split
    layers_full = list(range(64))
    # 4 think-end / first-token sites for the walk
    sites = capture_sites()
    walk_idx = [s["t"] for s in sites if s["kind"] in ("generate_first_token", "after_think")][:4]
    if not walk_idx:
        walk_idx = [54, 55, 114, 115]

    per_layer = []
    # student error state: (n_tok, 5120) per recipe
    recipes = {
        "g0_q4_hq30": dict(gate="hq4", up="hq4", down="hq4"),
        "q3mlp": dict(gate="u3", up="u3", down="u3"),
        "q4down": dict(gate="bin", up="bin", down="hq4"),  # up residual ≈ worse than binary? residual is better; use bin as lower bound? use bin for gate, u4 for down, bin for up as conservative
        "q4down_gatebin_downq4_upbin": dict(gate="bin", up="bin", down="u4"),
    }
    # actually q4down up is residual (better than binary). Use binary as a *pessimistic* stand-in only if we must.
    # For ranking we also want an honest-enough q4down: gate bin, up bin (underestimate), down u4.
    walk_err = {k: np.zeros((len(walk_idx), HIDDEN), dtype=np.float32) for k in recipes}
    walk_trace = {k: [] for k in recipes}

    def recon(W, tag):
        if tag == "hq4":
            return hq30uq4_recon(W)
        if tag == "u4":
            return hgravu_absmax_recon(W, 4)
        if tag == "u3":
            return hgravu_absmax_recon(W, 3)
        if tag == "bin":
            return binary_g128_recon(W)
        raise ValueError(tag)

    for layer in layers_full:
        t1 = time.time()
        X = load_hidden(layer)
        Xh = X[hold]
        Xw = X[walk_idx]
        Wg = load_tensor(f"language_model.model.layers.{layer}.mlp.gate_proj.weight")
        Wu = load_tensor(f"language_model.model.layers.{layer}.mlp.up_proj.weight")
        Wd = load_tensor(f"language_model.model.layers.{layer}.mlp.down_proj.weight")

        # BF16 MLP on hold
        Yg = Xh @ Wg.T
        Yu = Xh @ Wu.T
        act = silu(Yg) * Yu
        Yd = act @ Wd.T
        res = Xh + Yd

        row = dict(layer=layer, wall_s=None)
        row["yd_over_x"] = float(np.linalg.norm(Yd) / max(np.linalg.norm(Xh), 1e-12))
        row["mean_token_yd_over_x"] = float(
            np.mean(np.linalg.norm(Yd, axis=1) / np.maximum(np.linalg.norm(Xh, axis=1), 1e-12))
        )

        # teacher-forced hold for G0 Q4 and q3
        for tag, bitspec in (("g0_q4_hq30", "hq4"), ("q3mlp", "u3"), ("q4_hgravu", "u4")):
            Wg_h = recon(Wg, bitspec if bitspec != "u4" else "u4") if bitspec != "hq4" else recon(Wg, "hq4")
            if bitspec == "u3":
                Wg_h, Wu_h, Wd_h = recon(Wg, "u3"), recon(Wu, "u3"), recon(Wd, "u3")
            elif bitspec == "hq4":
                Wg_h, Wu_h, Wd_h = recon(Wg, "hq4"), recon(Wu, "hq4"), recon(Wd, "hq4")
            else:
                Wg_h, Wu_h, Wd_h = recon(Wg, "u4"), recon(Wu, "u4"), recon(Wd, "u4")
            Yg_h = Xh @ Wg_h.T
            Yu_h = Xh @ Wu_h.T
            act_h = silu(Yg_h) * Yu_h
            Yd_h = act_h @ Wd_h.T
            res_h = Xh + Yd_h
            row[tag] = dict(
                gate_out_cos=mean_row_cosine(Yg, Yg_h),
                up_out_cos=mean_row_cosine(Yu, Yu_h),
                down_out_cos=mean_row_cosine(Yd, Yd_h),
                residual_proxy_cos=mean_row_cosine(res, res_h),
                residual_proxy_min=min_row_cosine(res, res_h),
                down_rel_l2=rel_l2(Yd, Yd_h),
                residual_rel_l2=rel_l2(res, res_h),
            )

        # binary gate / residual-as-binary / q4 down on hold (q4down-ish)
        Wg_b, Wu_b, Wd_4 = recon(Wg, "bin"), recon(Wu, "bin"), recon(Wd, "u4")
        act_b = silu(Xh @ Wg_b.T) * (Xh @ Wu_b.T)
        Yd_b = act_b @ Wd_4.T
        row["q4down_bin_up"] = dict(
            residual_proxy_cos=mean_row_cosine(res, Xh + Yd_b),
            down_out_cos=mean_row_cosine(Yd, Yd_b),
        )

        # student-forced walk on 4 tokens
        Yg_w = Xw @ Wg.T
        Yu_w = Xw @ Wu.T
        act_w = silu(Yg_w) * Yu_w
        Yd_w = act_w @ Wd.T
        for rname, spec in recipes.items():
            Xs = Xw + walk_err[rname]
            Wg_h, Wu_h, Wd_h = recon(Wg, spec["gate"]), recon(Wu, spec["up"]), recon(Wd, spec["down"])
            act_s = silu(Xs @ Wg_h.T) * (Xs @ Wu_h.T)
            Yd_s = act_s @ Wd_h.T
            # Δ[l+1] = Δ[l] + (MLP_hat(X+Δ) - MLP(X))
            walk_err[rname] = walk_err[rname] + (Yd_s - Yd_w)
            walk_trace[rname].append(
                dict(
                    layer=layer,
                    hidden_cos=mean_row_cosine(Xw, Xw + walk_err[rname]),
                    err_l2=float(np.linalg.norm(walk_err[rname]) / max(math.sqrt(len(walk_idx)), 1e-12)),
                    mean_token_cos=mean_row_cosine(Xw, Xw + walk_err[rname]),
                )
            )

        row["wall_s"] = time.time() - t1
        per_layer.append(row)
        if layer % 8 == 0 or layer in (54, 58, 62, 63):
            log(
                f"L{layer:02d} yd/x={row['mean_token_yd_over_x']:.4f} "
                f"q4_res={row['g0_q4_hq30']['residual_proxy_cos']:.6f} "
                f"q3_res={row['q3mlp']['residual_proxy_cos']:.6f} "
                f"walk_q3={walk_trace['q3mlp'][-1]['hidden_cos']:.6f} "
                f"{row['wall_s']:.1f}s"
            )
        # free
        del Wg, Wu, Wd

    def prod_key(tag, field):
        p = 1.0
        xs = []
        for row in per_layer:
            x = row[tag][field]
            xs.append(x)
            p *= x
        return dict(n=len(xs), min=min(xs), median=sorted(xs)[len(xs) // 2], prod=p, geomean=p ** (1 / len(xs)))

    summary = dict(
        g0_residual_proxy=prod_key("g0_q4_hq30", "residual_proxy_cos"),
        q3_residual_proxy=prod_key("q3mlp", "residual_proxy_cos"),
        u4_residual_proxy=prod_key("q4_hgravu", "residual_proxy_cos"),
        g0_down_out=prod_key("g0_q4_hq30", "down_out_cos"),
        q3_down_out=prod_key("q3mlp", "down_out_cos"),
        q4down_residual_proxy=prod_key("q4down_bin_up", "residual_proxy_cos"),
        walk_final={k: walk_trace[k][-1] for k in recipes},
        walk_idx=walk_idx,
        mean_yd_over_x=float(np.mean([r["mean_token_yd_over_x"] for r in per_layer])),
        min_yd_over_x=float(np.min([r["mean_token_yd_over_x"] for r in per_layer])),
        max_yd_over_x=float(np.max([r["mean_token_yd_over_x"] for r in per_layer])),
    )
    return dict(summary=summary, per_layer=per_layer, walk_trace=walk_trace)


def stage_packed_organs(ident: dict) -> dict:
    """Real packed L0/L62 gate+down vs BF16 on hold tokens. Artifact-faithful."""
    hold = np.arange(192, 256)
    X0 = load_hidden(0)[hold]
    X62 = load_hidden(62)[hold]
    out = {}

    def score_matvec(X, W_hat, W):
        y = X @ W.T
        yh = X @ W_hat.T
        return dict(
            out_cos=mean_row_cosine(y, yh),
            out_min=min_row_cosine(y, yh),
            weight_cos=cosine(W, W_hat),
            weight_rel_l2=rel_l2(W, W_hat),
        )

    for layer, X in ((0, X0), (62, X62)):
        for role, shape_ok in (("gate_proj", (INTER, HIDDEN)), ("down_proj", (HIDDEN, INTER))):
            name = f"language_model.model.layers.{layer}.mlp.{role}.weight"
            W = load_tensor(name)
            rec = {"bf16_shape": list(W.shape)}
            # G0 HQ30UQ4
            g0_man = json.loads((G0 / "manifest.json").read_text())
            g0_t = next(t for t in g0_man["tensors"] if t["name"] == name)
            g0_path = G0 / "tensors" / g0_t["artifact"]
            # decode all rows of this tensor via requant of packed? decode_hq30uq4_rows all rows
            rows = np.arange(W.shape[0])
            log(f"packed decode G0 L{layer} {role} rows={W.shape[0]}")
            Wg0 = decode_hq30uq4_rows(g0_path, rows)
            rec["g0"] = score_matvec(X, Wg0, W)

            # q3mlp HGRAVU01
            q3_rows = parse_catalog(ART / "mixed-q3mlp-v1")
            r = catalog_row(q3_rows, name)
            log(f"packed decode q3mlp L{layer} {role}")
            Wq3 = decode_hgravu_rows(Path(r["path"]), r["off"], r["nbytes"], rows)
            rec["q3mlp"] = score_matvec(X, Wq3, W)

            # q4down: down is U01 q4, gate is binary — skip full binary decode here; down only
            if role == "down_proj":
                q4d_rows = parse_catalog(ART / "mixed-q4down-v1")
                r4 = catalog_row(q4d_rows, name)
                log(f"packed decode q4down L{layer} {role} magic peek")
                with open(r4["path"], "rb") as fh:
                    fh.seek(r4["off"])
                    mag = fh.read(8)
                rec["q4down_magic"] = mag.decode("latin1")
                if mag == b"HGRAVU01":
                    Wq4d = decode_hgravu_rows(Path(r4["path"]), r4["off"], r4["nbytes"], rows)
                    rec["q4down"] = score_matvec(X, Wq4d, W)

            # sub15: HQ30UQ4 of reconstructed weights
            s15_man = json.loads((ART / "mixed-sub15-v1/manifest.json").read_text())
            s15_t = next(t for t in s15_man["tensors"] if t["name"] == name)
            s15_path = ART / "mixed-sub15-v1/tensors" / s15_t["artifact"]
            log(f"packed decode sub15 L{layer} {role} ino={s15_path.stat().st_ino} g0ino={g0_path.stat().st_ino}")
            rec["sub15_same_file_as_g0"] = s15_path.stat().st_ino == g0_path.stat().st_ino
            if not rec["sub15_same_file_as_g0"]:
                Ws15 = decode_hq30uq4_rows(s15_path, rows)
                rec["sub15"] = score_matvec(X, Ws15, W)
            else:
                rec["sub15"] = dict(note="HARD-LINKED to G0 — identical packed bytes")
            out[f"L{layer}_{role}"] = rec
            del W, Wg0, Wq3
    return out


def main() -> None:
    t0 = time.time()
    result = dict(
        schema="hawking.g1.screen_vs_generate.v1",
        date=time.strftime("%Y-%m-%d"),
        label="MEASURED",
        claim_boundary=dict(
            no_gpu=True,
            no_generate=True,
            no_pack=True,
            no_resident_touch=True,
            capture_is_post_norm_hidden=True,
            lm_head_site_not_confirmed_final_norm=True,
            mixer_x_not_captured=True,
        ),
    )
    log("identity")
    result["identity"] = stage_identity()
    log("floor products")
    result["floor_products"] = stage_floor_products()
    log("stop rows")
    result["stop_rows"] = stage_stop_rows(result["identity"])
    # persist partial
    OUT.write_text(json.dumps(result, indent=2))
    log("logits")
    result["logits"] = stage_logits(result["identity"])
    OUT.write_text(json.dumps(result, indent=2))
    log("packed organs (L0/L62)")
    result["packed_organs"] = stage_packed_organs(result["identity"])
    OUT.write_text(json.dumps(result, indent=2))
    log("residual + student walk (64 layers)")
    result["residual_walk"] = stage_residual_and_walk()
    result["wall_s"] = time.time() - t0
    OUT.write_text(json.dumps(result, indent=2))
    log(f"done wall_s={result['wall_s']:.1f} -> {OUT}")


if __name__ == "__main__":
    main()
