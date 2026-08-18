#!/usr/bin/env python3
"""G1 lossless-entropy: BF16 field entropy + tile-local codecs on real tensors.

Read-only. Writes JSON to --out. Peak working set is one chunk (~32 MiB) plus
class histograms. Does not touch the live G0 tree, the resident, or the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b")
BF16 = ROOT / "bf16"
Q4 = ROOT / "uniform-q4-v1"
G128 = ROOT / "q4-mse-g128-hq30uq4-v1"
Q3 = ROOT / "mixed-q3mlp-v1"
N_LANG = 26_895_998_464
CHUNK = 16_777_216  # 2^24 elems, divisible by 512
TILES = (8, 64, 128, 512)  # thread packet, G0 group, g128 group, geo_tpr64 pass
PIN = {
    "config.json": "4de5e964cc7262209925609db8e62b3238be9625bac8d7d9840d4e71450651fe",
    "model.safetensors.index.json": "1db862301da01efa0a977a8f6944195d79bcab9683863c7e5f2e9aa33f8d1ce3",
}


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def shannon(counts) -> float:
    c = np.asarray(counts, dtype=np.float64)
    tot = float(c.sum())
    if tot <= 0:
        return 0.0
    p = c[c > 0] / tot
    return float(-(p * np.log2(p)).sum())


def classify(name: str) -> str:
    if name.endswith("embed_tokens.weight"):
        return "embed"
    if name.endswith("lm_head.weight") or name.endswith("language_model.lm_head.weight"):
        return "lm_head"
    for key, cls in (
        (".mlp.gate_proj.weight", "mlp.gate_proj"),
        (".mlp.up_proj.weight", "mlp.up_proj"),
        (".mlp.down_proj.weight", "mlp.down_proj"),
        ("in_proj_qkvz", "linear_attn.in_proj_qkvz"),
        ("in_proj_qkv.weight", "linear_attn.in_proj_qkv"),
        ("in_proj_z.weight", "linear_attn.in_proj_z"),
        ("in_proj_ba", "linear_attn.in_proj_ba"),
        ("in_proj_a.weight", "linear_attn.in_proj_a"),
        ("in_proj_b.weight", "linear_attn.in_proj_b"),
        ("linear_attn.out_proj", "linear_attn.out_proj"),
        ("self_attn.q_proj", "self_attn.q_proj"),
        ("self_attn.k_proj", "self_attn.k_proj"),
        ("self_attn.v_proj", "self_attn.v_proj"),
        ("self_attn.o_proj", "self_attn.o_proj"),
    ):
        if key in name:
            return cls
    if "conv1d" in name:
        return "small.conv1d"
    if "A_log" in name:
        return "small.A_log"
    if "dt_bias" in name:
        return "small.dt_bias"
    if "q_norm" in name:
        return "small.q_norm"
    if "k_norm" in name:
        return "small.k_norm"
    if "input_layernorm" in name:
        return "small.input_layernorm"
    if "post_attention_layernorm" in name:
        return "small.post_attention_layernorm"
    if name.endswith("model.norm.weight"):
        return "small.final_norm"
    if "linear_attn.norm" in name:
        return "small.linear_attn.norm"
    return "small.other"


class FieldAcc:
    """Order-0 histograms of BF16 fields + tile-local exponent geometry."""

    def __init__(self):
        self.n = 0
        self.sign = np.zeros(2, dtype=np.uint64)
        self.exp = np.zeros(256, dtype=np.uint64)
        self.mant = np.zeros(128, dtype=np.uint64)
        self.joint16 = np.zeros(65536, dtype=np.uint64)
        self.n_zero = 0
        self.n_sub = 0
        self.n_inf = 0
        self.n_nan = 0
        self.tensors = 0
        self.tiles = {
            T: {
                "n_tiles": 0,
                "n_uniform_exp": 0,
                "n_unique_sum": 0,
                "span_sum": 0,
                "n_at_emax": 0,
                "n_weights": 0,
                "delta_hist": np.zeros(256, dtype=np.uint64),
            }
            for T in TILES
        }

    def add_u16(self, u16: np.ndarray, do_tiles: bool = True) -> None:
        u16 = np.ascontiguousarray(u16, dtype=np.uint16).reshape(-1)
        n = int(u16.size)
        if n == 0:
            return
        self.n += n
        sign = (u16 >> np.uint16(15)).astype(np.uint8)
        exp = ((u16 >> np.uint16(7)) & np.uint16(0xFF)).astype(np.uint16)
        mant = (u16 & np.uint16(0x7F)).astype(np.uint16)
        self.sign += np.bincount(sign, minlength=2).astype(np.uint64)
        self.exp += np.bincount(exp, minlength=256).astype(np.uint64)
        self.mant += np.bincount(mant, minlength=128).astype(np.uint64)
        self.joint16 += np.bincount(u16.astype(np.int64), minlength=65536).astype(np.uint64)
        z = (exp == 0) & (mant == 0)
        self.n_zero += int(z.sum())
        self.n_sub += int(((exp == 0) & (mant != 0)).sum())
        self.n_inf += int(((exp == 255) & (mant == 0)).sum())
        self.n_nan += int(((exp == 255) & (mant != 0)).sum())
        if do_tiles:
            for T in TILES:
                full = (n // T) * T
                if full == 0:
                    continue
                e = exp[:full].reshape(-1, T)
                emax = e.max(axis=1)
                emin = e.min(axis=1)
                uniform = emax == emin
                # unique count via sort-eq
                es = np.sort(e, axis=1)
                nunique = 1 + (es[:, 1:] != es[:, :-1]).sum(axis=1)
                at = (e == emax[:, None]).sum()
                delta = (emax[:, None] - e).astype(np.int16)
                dh = np.bincount(delta.reshape(-1).astype(np.int32), minlength=256)
                t = self.tiles[T]
                t["n_tiles"] += int(e.shape[0])
                t["n_uniform_exp"] += int(uniform.sum())
                t["n_unique_sum"] += int(nunique.sum())
                t["span_sum"] += int((emax.astype(np.int32) - emin.astype(np.int32)).sum())
                t["n_at_emax"] += int(at)
                t["n_weights"] += int(full)
                t["delta_hist"][: dh.size] += dh.astype(np.uint64)

    def snapshot(self) -> dict:
        h_s = shannon(self.sign)
        h_e = shannon(self.exp)
        h_m = shannon(self.mant)
        h16 = shannon(self.joint16)
        # H(exp,mant) from folding sign off the 16-bit hist
        em = np.zeros(32768, dtype=np.uint64)
        em += self.joint16[:32768]
        em += self.joint16[32768:]
        h_em = shannon(em)
        h_m_given_e = max(0.0, h_em - h_e)
        n_unique16 = int((self.joint16 > 0).sum())
        n_unique_exp = int((self.exp > 0).sum())
        tiles = {}
        for T, t in self.tiles.items():
            nt = max(t["n_tiles"], 1)
            nw = max(t["n_weights"], 1)
            p_uni = t["n_uniform_exp"] / nt
            p_emax = t["n_at_emax"] / nw
            p_esc = 1.0 - p_emax
            # lossless shared-exp on uniform tiles, raw 16 elsewhere
            bfp_fallback = p_uni * (8.0 + 8.0 / T) + (1.0 - p_uni) * 16.0
            # 1-bit flag + 8-bit inlier (sign+mant) + 16-bit outlier + 8-bit emax/T
            flag_esc = 8.0 / T + 1.0 + 8.0 * p_emax + 16.0 * p_esc
            # field: emax/T + H(delta) + H_sign + H_mant  (shared model, lossless if delta kept)
            h_delta = shannon(t["delta_hist"])
            field_tile = 8.0 / T + h_delta + h_s + h_m
            # shared-model ANS of 16-bit patterns, 4-byte flush per tile
            ans_shared = h16 + 32.0 / T
            # width table on delta (ceil(log2(max_delta+1))) — only pays if unused
            dpos = np.flatnonzero(t["delta_hist"])
            max_d = int(dpos[-1]) if dpos.size else 0
            # bits to store delta uniformly
            dw = int(np.ceil(np.log2(max_d + 1))) if max_d > 0 else 0
            var_delta = 8.0 / T + dw + 1.0 + 7.0  # emax + fixed-width delta + sign + mant
            tiles[str(T)] = {
                "n_tiles": int(t["n_tiles"]),
                "n_weights": int(t["n_weights"]),
                "frac_uniform_exp": p_uni,
                "mean_nunique_exp": t["n_unique_sum"] / nt,
                "mean_exp_span": t["span_sum"] / nt,
                "p_at_emax": p_emax,
                "p_esc_from_emax": p_esc,
                "H_delta_exp": h_delta,
                "max_delta": max_d,
                "scheme_bfp_uniform_or_raw16": bfp_fallback,
                "scheme_flag_escape": flag_esc,
                "scheme_field_emax_delta": field_tile,
                "scheme_varwidth_delta": var_delta,
                "scheme_ans_shared_plus_flush": ans_shared,
                "flush_bpw": 32.0 / T,
            }
        used_exp = [int(i) for i in np.flatnonzero(self.exp)]
        return {
            "elements": int(self.n),
            "tensors": int(self.tensors),
            "H_sign": h_s,
            "H_exp": h_e,
            "H_mant": h_m,
            "H_exp_mant": h_em,
            "H_mant_given_exp": h_m_given_e,
            "H_joint16": h16,
            "H_fields_indep": h_s + h_e + h_m,
            "nominal_bits": 16.0,
            "gap_joint16": 16.0 - h16,
            "gap_fields_indep": 16.0 - (h_s + h_e + h_m),
            "n_unique_patterns": n_unique16,
            "n_unique_exp": n_unique_exp,
            "used_exp_min": used_exp[0] if used_exp else None,
            "used_exp_max": used_exp[-1] if used_exp else None,
            "p_zero": self.n_zero / max(self.n, 1),
            "p_subnormal": self.n_sub / max(self.n, 1),
            "p_inf": self.n_inf / max(self.n, 1),
            "p_nan": self.n_nan / max(self.n, 1),
            "p_sign1": float(self.sign[1] / max(self.n, 1)),
            "exp_hist_nonzero": {str(int(i)): int(self.exp[i]) for i in np.flatnonzero(self.exp)},
            "tiles": tiles,
        }


class QAcc:
    """Index-stream histograms at a known bit width + tile width / escape / ANS."""

    def __init__(self, bits: int):
        self.bits = bits
        self.levels = 1 << bits
        self.hist = np.zeros(self.levels, dtype=np.uint64)
        self.n = 0
        self.tensors = 0
        self.tiles = {
            T: {
                "n_tiles": 0,
                "width_sum": 0.0,  # offset-span width
                "signed_width_sum": 0.0,
                "n_fullwidth": 0,
                "width_hist": np.zeros(bits + 1, dtype=np.uint64),
            }
            for T in TILES
        }

    def add_codes(self, codes: np.ndarray) -> None:
        """codes are unsigned in 0 .. 2^bits-1 (nibble or HGRAVU01 unsigned)."""
        c = np.ascontiguousarray(codes, dtype=np.int32).reshape(-1)
        n = int(c.size)
        if n == 0:
            return
        self.n += n
        self.hist += np.bincount(c, minlength=self.levels).astype(np.uint64)
        qmin = -(1 << (self.bits - 1))
        q = c + qmin  # interpret as offset-binary; width stats on this view
        for T in TILES:
            full = (n // T) * T
            if full == 0:
                continue
            g = q[:full].reshape(-1, T)
            mx = g.max(axis=1)
            mn = g.min(axis=1)
            span = mx.astype(np.int32) - mn.astype(np.int32) + 1
            w = np.ceil(np.log2(np.maximum(span, 1))).astype(np.int16)
            w = np.clip(w, 0, self.bits)
            mag = np.maximum(np.abs(mn.astype(np.int32)), np.abs(mx.astype(np.int32)))
            sw = np.where(mag == 0, 0, np.ceil(np.log2(mag.astype(np.float64) + 1.0)).astype(np.int16) + 1)
            sw = np.clip(sw, 0, self.bits)
            t = self.tiles[T]
            t["n_tiles"] += int(g.shape[0])
            t["width_sum"] += float(w.sum())
            t["signed_width_sum"] += float(sw.sum())
            t["n_fullwidth"] += int((w == self.bits).sum())
            t["width_hist"] += np.bincount(w, minlength=self.bits + 1).astype(np.uint64)

    def snapshot(self) -> dict:
        h = shannon(self.hist)
        tiles = {}
        for T, t in self.tiles.items():
            nt = max(t["n_tiles"], 1)
            mean_w = t["width_sum"] / nt
            mean_sw = t["signed_width_sum"] / nt
            table_bits = int(np.ceil(np.log2(self.bits + 1))) if self.bits > 0 else 0
            table_bpw = table_bits / T
            # top-(2^k-1) escape from global hist
            n = float(self.hist.sum()) or 1.0
            order = np.argsort(-self.hist)
            esc = {}
            for k in range(1, self.bits):
                n_in = (1 << k) - 1
                p_in = float(sum(int(self.hist[i]) for i in order[:n_in])) / n
                p_esc = 1.0 - p_in
                leftover = self.bits - k
                esc[f"k{k}"] = {
                    "p_in": p_in,
                    "p_esc": p_esc,
                    "seq_k_plus_leftover": k + p_esc * leftover,
                    "group_scanned_leftover": k + p_esc * leftover + table_bits / T,
                    "ra_local_index": k + p_esc * (int(np.ceil(np.log2(T))) + leftover),
                }
            tiles[str(T)] = {
                "n_tiles": int(t["n_tiles"]),
                "mean_offset_width": mean_w,
                "mean_signed_width": mean_sw,
                "frac_fullwidth": t["n_fullwidth"] / nt,
                "width_hist": [int(x) for x in t["width_hist"].tolist()],
                "width_table_bpw": table_bpw,
                "varwidth_offset_bpw": mean_w + table_bpw,
                "ans_shared_plus_flush": h + 32.0 / T,
                "flush_bpw": 32.0 / T,
                "escape": esc,
            }
        return {
            "bits": self.bits,
            "elements": int(self.n),
            "tensors": int(self.tensors),
            "shannon_H": h,
            "nominal": float(self.bits),
            "gap": float(self.bits) - h,
            "hist": [int(x) for x in self.hist.tolist()],
            "tiles": tiles,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_st_index() -> tuple[dict, dict]:
    idx = json.loads((BF16 / "model.safetensors.index.json").read_text())
    return idx["weight_map"], {}


_HDR: dict[str, dict] = {}


def tensor_u16_iter(weight_map: dict, name: str):
    shard = BF16 / weight_map[name]
    key = str(shard)
    if key not in _HDR:
        with shard.open("rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            raw = f.read(hlen)
        info = json.loads(raw)
        _HDR[key] = {"tensors": info, "base": 8 + hlen}
    t = _HDR[key]["tensors"][name]
    begin, end = t["data_offsets"]
    off = _HDR[key]["base"] + begin
    nbytes = end - begin
    mm = np.memmap(shard, dtype=np.uint8, mode="r")
    raw = mm[off : off + nbytes]
    u16 = np.frombuffer(raw, dtype="<u2")
    n = int(u16.size)
    for i in range(0, n, CHUNK):
        yield np.asarray(u16[i : i + CHUNK])
    del mm


def parse_hq30uq4(path: Path) -> tuple[int, int, int, int]:
    """Return (group, elements, code_off, code_bytes_per_group)."""
    with path.open("rb") as f:
        hdr = f.read(40)
    if hdr[:8] != b"HQ30UQ4\0":
        raise RuntimeError(f"bad magic {path}")
    version, group, rank = struct.unpack_from("<IIH", hdr, 8)
    elements = struct.unpack_from("<Q", hdr, 20)[0]
    if version != 1:
        raise RuntimeError(f"bad version {path}")
    dim_bytes = rank * 4
    groups = (elements + group - 1) // group
    scale_off = 32 + dim_bytes
    code_off = scale_off + groups * 2
    cpg = group // 2
    expect = code_off + groups * cpg
    size = path.stat().st_size
    if expect != size:
        raise RuntimeError(f"size mismatch {path} {size} != {expect} g={group} E={elements}")
    return group, elements, code_off, cpg


def iter_hq30uq4_codes(path: Path):
    group, elements, code_off, cpg = parse_hq30uq4(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    codes = mm[code_off:]
    remaining = elements
    groups = (elements + group - 1) // group
    # walk groups in chunks
    step = max(1, CHUNK // group)
    for g0 in range(0, groups, step):
        g1 = min(groups, g0 + step)
        blob = np.asarray(codes[g0 * cpg : g1 * cpg])
        lo = blob & 0x0F
        hi = blob >> 4
        paired = np.empty(blob.size * 2, dtype=np.uint8)
        paired[0::2] = lo
        paired[1::2] = hi
        take = min(remaining, paired.size)
        yield paired[:take]
        remaining -= take
    del mm


def parse_hgravu01(path: Path) -> dict:
    with path.open("rb") as f:
        mag = f.read(8)
        if mag != b"HGRAVU01":
            raise RuntimeError(f"bad HGRAVU01 magic {path}")
        (jlen,) = struct.unpack("<I", f.read(4))
        meta = json.loads(f.read(jlen))
        payload_off = 12 + jlen
    return {**meta, "payload_off": payload_off, "path": str(path)}


def iter_hgravu01_codes(path: Path):
    meta = parse_hgravu01(path)
    bits = int(meta["bits"])
    elements = int(meta["elements"])
    group = int(meta.get("group") or meta.get("group_size") or 64)
    groups = (elements + group - 1) // group
    bits_per_g = group * bits
    bytes_per_g = (bits_per_g + 7) // 8
    scale_bytes = groups * 2
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    codes = mm[meta["payload_off"] + scale_bytes :]
    remaining = elements
    step = max(1, CHUNK // group)
    for g0 in range(0, groups, step):
        g1 = min(groups, g0 + step)
        blob = np.asarray(codes[g0 * bytes_per_g : g1 * bytes_per_g])
        ng = g1 - g0
        bits_arr = np.unpackbits(blob, bitorder="little")
        need = ng * group * bits
        bits_arr = bits_arr[:need].reshape(ng * group, bits)
        weights = (2 ** np.arange(bits, dtype=np.uint16))
        unsigned = bits_arr.dot(weights).astype(np.uint16)
        take = min(remaining, unsigned.size)
        yield unsigned[:take]
        remaining -= take
    del mm


def rans_size_order0(symbols: np.ndarray, alphabet: int) -> dict:
    s = np.ascontiguousarray(symbols, dtype=np.int32).reshape(-1)
    if s.size == 0:
        return {"n": 0, "stream_bytes": 0, "stream_bpw": 0.0}
    counts = np.bincount(s, minlength=alphabet).astype(np.int64)
    counts = np.maximum(counts, 1)
    SCALE = 1 << 12
    freqs = counts * SCALE // counts.sum()
    freqs = np.maximum(freqs, 1)
    extra = int(freqs.sum() - SCALE)
    if extra > 0:
        while extra > 0:
            i = int(np.argmax(freqs))
            take = min(extra, int(freqs[i] - 1))
            if take <= 0:
                break
            freqs[i] -= take
            extra -= take
    elif extra < 0:
        freqs[int(np.argmax(freqs))] += -extra
    cum = np.zeros(alphabet + 1, dtype=np.uint32)
    acc = 0
    for i, f in enumerate(freqs):
        acc += int(f)
        cum[i + 1] = acc
    L = 1 << 23
    x = L
    out = bytearray()
    for sym in s[::-1]:
        freq = int(freqs[int(sym)])
        start = int(cum[int(sym)])
        x_max = ((L >> 12) << 8) * freq
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        q, r = divmod(x, freq)
        x = (q << 12) + r + start
    for _ in range(4):
        out.append(x & 0xFF)
        x >>= 8
    n = int(s.size)
    return {"n": n, "stream_bytes": len(out), "stream_bpw": len(out) * 8 / n, "table_bytes": alphabet * 2}


def rans_per_tile(symbols: np.ndarray, alphabet: int, tile: int, max_tiles: int = 256) -> dict:
    s = np.ascontiguousarray(symbols, dtype=np.int32).reshape(-1)
    n = int(s.size)
    full = (n // tile) * tile
    if full == 0:
        return {"tile": tile, "sampled_tiles": 0}
    g = s[:full].reshape(-1, tile)
    take = min(g.shape[0], max_tiles)
    idx = np.linspace(0, g.shape[0] - 1, take, dtype=np.int64)
    bits = 0.0
    for gi in idx:
        r = rans_size_order0(g[gi], alphabet)
        bits += r["stream_bytes"] * 8
    return {
        "tile": tile,
        "sampled_tiles": take,
        "measured_stream_bpw_incl_flush": (bits / take) / tile,
        "flush_bpw": 32.0 / tile,
        "shared_table_bpw": (alphabet * 2 * 8) / max(n, 1),
    }


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def quantize_g0(x: np.ndarray, bits: int, group: int) -> np.ndarray:
    n = int(x.size)
    pad = (group - (n % group)) % group
    if pad:
        x = np.pad(x, (0, pad), constant_values=0.0)
    g = x.reshape(-1, group)
    bound = (1 << (bits - 1)) - 1
    if bound <= 0:
        # 1-bit two-level: q in {0,1}, scale = amax, nearest
        amax = np.max(np.abs(g), axis=1)
        den = np.where(amax > 0.0, amax, 1.0)
        q = np.rint(np.abs(g) / den[:, None])
        q = np.clip(q, 0, 1).astype(np.int16)
        return q.reshape(-1)[:n]
    amax = np.max(np.abs(g), axis=1)
    scale = np.float16(amax / bound).astype(np.float32)
    den = np.where(scale > 0.0, scale, 1.0)
    qmin = -(1 << (bits - 1))
    q = np.rint(g / den[:, None])
    q = np.clip(q, qmin, bound).astype(np.int16)
    return q.reshape(-1)[:n]


def quantize_sign(x: np.ndarray) -> np.ndarray:
    return (x >= 0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/g1_lossless_entropy.json")
    ap.add_argument(
        "--phase",
        default="all",
        choices=["all", "bf16", "q4", "g128", "q3", "requant", "rans"],
    )
    args = ap.parse_args()
    t0 = time.time()
    out: dict = {
        "schema": "hawking.g1.lossless_entropy.measure.v1",
        "N": N_LANG,
        "paths": {"bf16": str(BF16), "q4": str(Q4), "g128": str(G128), "q3": str(Q3)},
        "tiles": list(TILES),
        "pin_check": {},
        "started_unix": time.time(),
        "peak_rss_mb_start": rss_mb(),
    }

    # --- pin ---
    pin = {}
    for rel, expect in PIN.items():
        got = sha256_file(BF16 / rel)
        pin[rel] = {"expected": expect, "got": got, "match": got == expect}
    out["pin_check"] = pin

    weight_map, _ = load_st_index()
    by_class: dict[str, list[str]] = defaultdict(list)
    for name in weight_map:
        if name.startswith("vision_tower."):
            continue
        by_class[classify(name)].append(name)
    out["bf16_class_counts"] = {c: len(v) for c, v in sorted(by_class.items())}

    phases = ["bf16", "q4", "g128", "q3", "requant", "rans"] if args.phase == "all" else [args.phase]

    # --- BF16 fields ---
    if "bf16" in phases:
        accs: dict[str, FieldAcc] = defaultdict(FieldAcc)
        grand = FieldAcc()
        n_done = 0
        for cls, names in sorted(by_class.items()):
            for name in names:
                do_tiles = not cls.startswith("small.")
                for chunk in tensor_u16_iter(weight_map, name):
                    accs[cls].add_u16(chunk, do_tiles=do_tiles)
                    grand.add_u16(chunk, do_tiles=do_tiles)
                accs[cls].tensors += 1
                grand.tensors += 1
                n_done += 1
                if n_done % 50 == 0:
                    print(f"bf16 {n_done} tensors rss={rss_mb():.1f}MB", flush=True)
        out["bf16_fields"] = {c: a.snapshot() for c, a in sorted(accs.items())}
        out["bf16_fields_all_language"] = grand.snapshot()
        print(f"bf16 done H16={out['bf16_fields_all_language']['H_joint16']:.6f} rss={rss_mb():.1f}", flush=True)

    # --- G0 Q4 spot-check (one tensor per class + all of L0 gate) ---
    if "q4" in phases:
        man = json.loads((Q4 / "manifest.json").read_text())
        out["q4_manifest_complete_physical_bpw"] = man.get("complete_physical_bpw")
        accs_q: dict[str, QAcc] = {}
        # full production would re-derive the prior 268s census; we do
        # one tensor per class (first) PLUS every mlp.gate_proj (64) as a
        # class-complete CROSS-CHECK against the landed 3.4846.
        seen_class = set()
        gate_acc = QAcc(4)
        for row in man["tensors"]:
            if row.get("kind") != "q4":
                continue
            cls = classify(row["name"])
            path = Q4 / "tensors" / row["artifact"]
            take = (cls not in seen_class) or (cls == "mlp.gate_proj")
            if not take:
                continue
            if cls not in accs_q:
                accs_q[cls] = QAcc(4)
            for chunk in iter_hq30uq4_codes(path):
                accs_q[cls].add_codes(chunk)
                if cls == "mlp.gate_proj":
                    gate_acc.add_codes(chunk)
            accs_q[cls].tensors += 1
            if cls == "mlp.gate_proj":
                gate_acc.tensors += 1
            seen_class.add(cls)
            print(f"q4 {cls} {row['name'][-40:]} n={accs_q[cls].n} rss={rss_mb():.1f}", flush=True)
        out["q4_spot"] = {c: a.snapshot() for c, a in sorted(accs_q.items())}
        out["q4_gate_all_layers"] = gate_acc.snapshot()
        print(f"q4 gate-all H={out['q4_gate_all_layers']['shannon_H']:.10f}", flush=True)

    # --- g128 MSE HQ30UQ4 full GEMV census ---
    if "g128" in phases:
        man = json.loads((G128 / "manifest.json").read_text())
        out["g128_manifest"] = {
            "complete_physical_bpw": man.get("complete_physical_bpw"),
            "nominal_codec_bpw": man.get("nominal_codec_bpw"),
            "q4_group_size": man.get("q4_group_size"),
            "embed_group_size": man.get("embed_group_size"),
            "q4_tensors": man.get("q4_tensors"),
            "source_weight_elements": man.get("source_weight_elements"),
            "tensor_payload_bytes": man.get("tensor_payload_bytes"),
        }
        accs_g: dict[str, QAcc] = defaultdict(lambda: QAcc(4))
        grand_g = QAcc(4)
        n_t = 0
        for row in man["tensors"]:
            if row.get("kind") != "q4":
                continue
            cls = classify(row["name"])
            path = G128 / "tensors" / row["artifact"]
            for chunk in iter_hq30uq4_codes(path):
                accs_g[cls].add_codes(chunk)
                grand_g.add_codes(chunk)
            accs_g[cls].tensors += 1
            grand_g.tensors += 1
            n_t += 1
            if n_t % 40 == 0:
                print(f"g128 {n_t} Hsofar={shannon(grand_g.hist):.6f} rss={rss_mb():.1f}", flush=True)
        out["g128_index"] = {c: a.snapshot() for c, a in sorted(accs_g.items())}
        out["g128_index_all"] = grand_g.snapshot()
        print(f"g128 all H={out['g128_index_all']['shannon_H']:.10f} n={grand_g.n}", flush=True)

        # scale-stream H (fp16 bit patterns), one tensor per class
        scale_h = {}
        seen = set()
        for row in man["tensors"]:
            if row.get("kind") != "q4":
                continue
            cls = classify(row["name"])
            if cls in seen:
                continue
            path = G128 / "tensors" / row["artifact"]
            group, elements, code_off, cpg = parse_hq30uq4(path)
            groups = (elements + group - 1) // group
            rank = struct.unpack_from("<H", path.read_bytes()[:40], 16)[0]
            scale_off = 32 + rank * 4
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            raw = np.frombuffer(np.asarray(mm[scale_off:code_off]), dtype="<u2")
            hist = np.bincount(raw.astype(np.int64), minlength=65536)
            scale_h[cls] = {
                "n": int(raw.size),
                "group": group,
                "H16": shannon(hist),
                "unique": int((hist > 0).sum()),
                "recoverable_vs_16": 16.0 - shannon(hist),
                "bpw_if_coded": shannon(hist) / group,
            }
            seen.add(cls)
            del mm
        out["g128_scale_stream"] = scale_h

    # --- Q3 HGRAVU01 packed MLP (all 192 replace segments) ---
    if "q3" in phases:
        segdir = Q3 / "segments"
        accs3: dict[str, QAcc] = defaultdict(lambda: QAcc(3))
        grand3 = QAcc(3)
        n_t = 0
        for fn in sorted(os.listdir(segdir)):
            if not fn.startswith("replace_") or not fn.endswith(".hq38seg"):
                continue
            path = segdir / fn
            # class from filename
            cls = "other"
            for key, c in (
                ("gate_proj", "mlp.gate_proj"),
                ("up_proj", "mlp.up_proj"),
                ("down_proj", "mlp.down_proj"),
            ):
                if key in fn:
                    cls = c
                    break
            try:
                for chunk in iter_hgravu01_codes(path):
                    accs3[cls].add_codes(chunk)
                    grand3.add_codes(chunk)
            except Exception as e:
                print(f"q3 FAIL {fn}: {e}", flush=True)
                continue
            accs3[cls].tensors += 1
            grand3.tensors += 1
            n_t += 1
            if n_t % 32 == 0:
                print(f"q3 {n_t} H={shannon(grand3.hist):.6f} rss={rss_mb():.1f}", flush=True)
        out["q3_index"] = {c: a.snapshot() for c, a in sorted(accs3.items())}
        out["q3_index_all"] = grand3.snapshot()
        print(f"q3 all H={out['q3_index_all']['shannon_H']:.10f} n={grand3.n} t={grand3.tensors}", flush=True)

    # --- requant 1-bit / 2-bit Shannon on sample layers (G0 convention) ---
    if "requant" in phases:
        sample_layers = {0, 31, 63}
        sample_cls = [
            "mlp.gate_proj",
            "mlp.down_proj",
            "self_attn.q_proj",
            "linear_attn.out_proj",
        ]
        rows = []
        for cls in sample_cls:
            names = by_class.get(cls, [])
            if cls in ("embed", "lm_head"):
                names = names[:1]
            picked = []
            for n in names:
                if ".layers." in n:
                    lyr = int(n.split(".layers.", 1)[1].split(".", 1)[0])
                    if lyr in sample_layers:
                        picked.append(n)
                else:
                    picked.append(n)
            for name in picked:
                # load in chunks, accumulate hists for bits=1,2 and sign
                h1 = np.zeros(2, dtype=np.uint64)
                h2 = np.zeros(4, dtype=np.uint64)
                hs = np.zeros(2, dtype=np.uint64)
                n_el = 0
                for chunk in tensor_u16_iter(weight_map, name):
                    x = bf16_to_f32(chunk)
                    n_el += int(x.size)
                    q1 = quantize_g0(x, 1, 128)
                    q2 = quantize_g0(x, 2, 128)
                    # q2 is signed in {-2,-1,0,1}; shift to 0..3
                    h1 += np.bincount(q1.astype(np.int32), minlength=2).astype(np.uint64)
                    h2 += np.bincount((q2 + 2).astype(np.int32), minlength=4).astype(np.uint64)
                    hs += np.bincount(quantize_sign(x), minlength=2).astype(np.uint64)
                rows.append(
                    {
                        "name": name,
                        "class": cls,
                        "elements": n_el,
                        "H1_absmax_01": shannon(h1),
                        "H2_g0": shannon(h2),
                        "H_sign": shannon(hs),
                        "p1": [int(x) for x in h1.tolist()],
                        "complete_H1_g128": shannon(h1) + 16.0 / 128.0,
                        "complete_H2_g128": shannon(h2) + 16.0 / 128.0,
                        "complete_Hsign_g128": shannon(hs) + 16.0 / 128.0,
                    }
                )
                print(f"requant {name[-50:]} H1={rows[-1]['H1_absmax_01']:.4f} H2={rows[-1]['H2_g0']:.4f} Hs={rows[-1]['H_sign']:.4f}", flush=True)
        out["requant_samples"] = rows

    # --- rANS size probe on one gate L0 Q4 / BF16-exp / g128 ---
    if "rans" in phases:
        probes = {}
        # G0 Q4 L0 gate
        man = json.loads((Q4 / "manifest.json").read_text())
        gate0 = next(r for r in man["tensors"] if r["name"].endswith("layers.0.mlp.gate_proj.weight") and r.get("kind") == "q4")
        path = Q4 / "tensors" / gate0["artifact"]
        # first 2,097,152 codes
        buf = []
        need = 2_097_152
        for chunk in iter_hq30uq4_codes(path):
            buf.append(chunk)
            if sum(c.size for c in buf) >= need:
                break
        s = np.concatenate(buf)[:need].astype(np.int32)
        probes["q4_gate_L0_2M"] = {
            "H": shannon(np.bincount(s, minlength=16)),
            "global_rans": rans_size_order0(s, 16),
            "per_tile": {str(T): rans_per_tile(s, 16, T) for T in TILES},
        }
        # g128 same tensor
        man2 = json.loads((G128 / "manifest.json").read_text())
        gate0b = next(r for r in man2["tensors"] if r["name"].endswith("layers.0.mlp.gate_proj.weight") and r.get("kind") == "q4")
        pathb = G128 / "tensors" / gate0b["artifact"]
        buf = []
        for chunk in iter_hq30uq4_codes(pathb):
            buf.append(chunk)
            if sum(c.size for c in buf) >= need:
                break
        s2 = np.concatenate(buf)[:need].astype(np.int32)
        probes["g128_gate_L0_2M"] = {
            "H": shannon(np.bincount(s2, minlength=16)),
            "global_rans": rans_size_order0(s2, 16),
            "per_tile": {str(T): rans_per_tile(s2, 16, T) for T in TILES},
        }
        # BF16 exponent of same source tensor, 2M
        name = "language_model.model.layers.0.mlp.gate_proj.weight"
        exp_buf = []
        u16_buf = []
        got = 0
        for chunk in tensor_u16_iter(weight_map, name):
            take = min(chunk.size, need - got)
            u16_buf.append(chunk[:take])
            exp_buf.append(((chunk[:take] >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int32))
            got += take
            if got >= need:
                break
        e = np.concatenate(exp_buf)
        u = np.concatenate(u16_buf)
        probes["bf16_gate_L0_2M_exp"] = {
            "H": shannon(np.bincount(e, minlength=256)),
            "global_rans": rans_size_order0(e, 256),
            "per_tile": {str(T): rans_per_tile(e, 256, T) for T in TILES},
        }
        # 16-bit patterns: alphabet 65536 is too wide for the tiny rANS; report H only
        probes["bf16_gate_L0_2M_u16"] = {
            "H": shannon(np.bincount(u.astype(np.int64), minlength=65536)),
            "note": "alphabet 65536; rANS size not measured; use H+32/T",
        }
        out["rans_probes"] = probes
        print("rans probes done", json.dumps({k: v.get("H") for k, v in probes.items()}), flush=True)

    out["wall_s"] = time.time() - t0
    out["peak_rss_mb"] = rss_mb()
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"WROTE {args.out} wall={out['wall_s']:.2f}s rss={out['peak_rss_mb']:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
