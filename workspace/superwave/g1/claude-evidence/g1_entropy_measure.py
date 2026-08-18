#!/usr/bin/env python3
"""G1 entropy-coding lane: measure Qwen3.8 quantized-index entropy and RA codecs.

Read-only against the Qwen3.8 BF16 source and the sealed uniform-q4-v1 artifact.
Writes JSON to --out. Does not write any model artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b")
BF16 = ROOT / "bf16"
Q4 = ROOT / "uniform-q4-v1"
BIT_WIDTHS = (2, 3, 4, 5, 6, 8)
GROUP = 64
HEADER = 32


def classify(name: str) -> str:
    if name.endswith("embed_tokens.weight"):
        return "embed"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if ".mlp.gate_proj.weight" in name:
        return "mlp.gate_proj"
    if ".mlp.up_proj.weight" in name:
        return "mlp.up_proj"
    if ".mlp.down_proj.weight" in name:
        return "mlp.down_proj"
    if "in_proj_qkvz" in name:
        return "linear_attn.in_proj_qkvz"
    if "in_proj_qkv.weight" in name:
        return "linear_attn.in_proj_qkv"
    if "in_proj_z.weight" in name:
        return "linear_attn.in_proj_z"
    if "in_proj_ba" in name:
        return "linear_attn.in_proj_ba"
    if "in_proj_a.weight" in name:
        return "linear_attn.in_proj_a"
    if "in_proj_b.weight" in name:
        return "linear_attn.in_proj_b"
    if "linear_attn.out_proj" in name:
        return "linear_attn.out_proj"
    if "self_attn.q_proj" in name:
        return "self_attn.q_proj"
    if "self_attn.k_proj" in name:
        return "self_attn.k_proj"
    if "self_attn.v_proj" in name:
        return "self_attn.v_proj"
    if "self_attn.o_proj" in name:
        return "self_attn.o_proj"
    if "conv1d" in name:
        return "small.conv1d"
    if "A_log" in name:
        return "small.A_log"
    if "dt_bias" in name:
        return "small.dt_bias"
    if "linear_attn.norm" in name:
        return "small.linear_attn.norm"
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
    return "other"


GEMV_CLASSES = {
    "embed",
    "lm_head",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "linear_attn.in_proj_qkvz",
    "linear_attn.in_proj_ba",
    "linear_attn.out_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
}


def shannon(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64)
    p = p[p > 0] / total
    return float(-(p * np.log2(p)).sum())


def entropy_from_list(values: np.ndarray, minlength: int) -> float:
    if values.size == 0:
        return 0.0
    lo = int(values.min())
    shifted = values.astype(np.int32) - lo
    counts = np.bincount(shifted, minlength=minlength)
    return shannon(counts)


class Acc:
    def __init__(self, bits: int):
        self.bits = bits
        self.levels = 1 << bits
        self.qmin = -(1 << (bits - 1))
        self.hist = np.zeros(self.levels, dtype=np.uint64)
        self.width_hist = np.zeros(bits + 1, dtype=np.uint64)  # offset-span bits
        self.signed_width_hist = np.zeros(bits + 1, dtype=np.uint64)
        self.elements = 0
        self.groups = 0
        self.group_entropy_sum = 0.0
        self.group_entropy_n = 0
        self.sat_lo = 0
        self.sat_hi = 0
        self.tensors = 0

    def _add_groups(self, g: np.ndarray, sample_group_entropy: bool) -> None:
        if g.size == 0:
            return
        qmax = g.max(axis=1)
        qmin = g.min(axis=1)
        span = qmax.astype(np.int32) - qmin.astype(np.int32) + 1
        w = np.ceil(np.log2(np.maximum(span, 1))).astype(np.int16)
        w = np.clip(w, 0, self.bits)
        self.width_hist += np.bincount(w, minlength=self.bits + 1).astype(np.uint64)
        m = np.maximum(np.abs(qmin.astype(np.int32)), np.abs(qmax.astype(np.int32)))
        sw = np.where(
            m == 0,
            0,
            np.ceil(np.log2(m.astype(np.float64) + 1.0)).astype(np.int16) + 1,
        )
        sw = np.clip(sw, 0, self.bits)
        self.signed_width_hist += np.bincount(sw, minlength=self.bits + 1).astype(np.uint64)
        ng = int(g.shape[0])
        self.groups += ng
        if sample_group_entropy and self.group_entropy_n < 8192:
            take = min(ng, 8192 - self.group_entropy_n)
            if take < ng:
                idx = np.linspace(0, ng - 1, take, dtype=np.int64)
                gs = g[idx]
            else:
                gs = g[:take]
            for row in gs:
                self.group_entropy_sum += entropy_from_list(row, self.levels)
                self.group_entropy_n += 1

    def add_q(self, q: np.ndarray, sample_group_entropy: bool) -> None:
        q = np.ascontiguousarray(q, dtype=np.int16).reshape(-1)
        n = int(q.size)
        if n == 0:
            return
        offset = -self.qmin
        codes = q.astype(np.int32) + offset
        self.hist += np.bincount(codes, minlength=self.levels).astype(np.uint64)
        self.sat_lo += int((q == self.qmin).sum())
        self.sat_hi += int((q == (self.qmin + self.levels - 1)).sum())
        self.elements += n
        full = (n // GROUP) * GROUP
        if full:
            self._add_groups(q[:full].reshape(-1, GROUP), sample_group_entropy)
        if full < n:
            tail = q[full:]
            self._add_groups(tail.reshape(1, -1), sample_group_entropy)

    def snapshot(self) -> dict:
        h = shannon(self.hist)
        nom = float(self.bits)
        gap = nom - h
        wh = self.width_hist.astype(np.float64)
        wsum = float(wh.sum()) or 1.0
        mean_w = float(np.dot(np.arange(len(wh)), wh) / wsum)
        swh = self.signed_width_hist.astype(np.float64)
        swsum = float(swh.sum()) or 1.0
        mean_sw = float(np.dot(np.arange(len(swh)), swh) / swsum)
        table_bpw = math.ceil(math.log2(self.bits + 1)) / GROUP
        var_bpw = mean_w + table_bpw
        signed_var_bpw = mean_sw + table_bpw
        hg = (self.group_entropy_sum / self.group_entropy_n) if self.group_entropy_n else None
        # rice of zigzag(q)
        rice = rice_bits_from_hist(self.hist, self.qmin)
        # escape / outlier schemes
        escape = escape_schemes(self.hist, self.bits, self.qmin)
        return {
            "bits": self.bits,
            "elements": int(self.elements),
            "groups": int(self.groups),
            "tensors": int(self.tensors),
            "shannon_H_bits": h,
            "nominal_bits": nom,
            "gap_bits": gap,
            "hist": [int(x) for x in self.hist.tolist()],
            "p_sat_lo": self.sat_lo / max(self.elements, 1),
            "p_sat_hi": self.sat_hi / max(self.elements, 1),
            "mean_group_offset_width": mean_w,
            "mean_group_signed_width": mean_sw,
            "width_hist": [int(x) for x in self.width_hist.tolist()],
            "signed_width_hist": [int(x) for x in self.signed_width_hist.tolist()],
            "width_table_bpw": table_bpw,
            "per_group_varwidth_offset_bpw": var_bpw,
            "per_group_varwidth_signed_bpw": signed_var_bpw,
            "mean_group_shannon": hg,
            "group_entropy_samples": int(self.group_entropy_n),
            "rice_of_zigzag_q": rice,
            "escape": escape,
        }


def rice_expected_bits(values_nonneg: np.ndarray, k: int) -> float:
    if values_nonneg.size == 0:
        return 0.0
    q = values_nonneg.astype(np.uint64) >> np.uint64(k)
    return float(q.mean() + 1.0 + k)


def rice_bits_from_hist(hist: np.ndarray, qmin: int) -> dict:
    # reconstruct multiset as histogram over zigzag(q)
    # zigzag(q) = (q<<1) ^ (q>>31)
    zs = []
    cs = []
    for i, c in enumerate(hist.tolist()):
        if c == 0:
            continue
        q = i + qmin
        z = (q << 1) ^ (q >> 31)
        zs.append(z)
        cs.append(c)
    if not zs:
        return {"best_k": 0, "bpw": 0.0}
    zs = np.array(zs, dtype=np.int64)
    cs = np.array(cs, dtype=np.float64)
    n = float(cs.sum())
    best = None
    for k in range(0, 12):
        q = (zs.astype(np.uint64) >> np.uint64(k)).astype(np.float64)
        bits = float((q * cs).sum() / n + 1.0 + k)
        if best is None or bits < best[1]:
            best = (k, bits)
    assert best is not None
    return {"best_k": int(best[0]), "bpw": float(best[1]), "note": "order0 rice of zigzag(q); sequential"}


def escape_schemes(hist: np.ndarray, bits: int, qmin: int) -> dict:
    n = float(hist.sum()) or 1.0
    out = {}
    # inlier half-range R: values in [-(2^{k-1}-1), 2^{k-1}-1] fit in k-bit signed
    # excluding the extra negative code. Escape everything outside.
    for k in range(1, bits):
        in_lo = -(1 << (k - 1)) + (0 if k == 1 else 0)
        # k-bit signed typical range [-2^{k-1}, 2^{k-1}-1] but reserve one ESC
        # Use reserved max-unsigned as ESC: inliers are 2^k - 1 codes centered on 0
        n_in = (1 << k) - 1
        # map q to 0..2^bits-1 already in hist index. Choose inlier set as the
        # n_in most frequent symbols (optimal static escape alphabet).
        order = np.argsort(-hist)
        inlier = set(int(i) for i in order[:n_in].tolist())
        p_in = float(sum(int(hist[i]) for i in inlier)) / n
        p_esc = 1.0 - p_in
        # sequential packed: k bits always, plus (bits) extra on escape
        seq_bpw = k + p_esc * bits
        # RA: k-bit main + per-escape (group-local 6-bit pos + bits value)
        # plus 1-bit? no: ESC is a reserved main code so no extra flag
        # outlier table per group: count is implicit by scanning 64 mains
        # cost of table: p_esc * (6 + bits) if group-local index, stored densely
        ra_bpw = k + p_esc * (6 + bits)
        # rice positions: if independent, gap Geo(p_esc)
        # E[rice bits per outlier] via best k on geometric is ~ 1/p * H_2(p) wait
        # position bits per weight = rice(gap)*p_esc. We'll fill empirically later.
        out[f"k{k}"] = {
            "inlier_width": k,
            "p_in": p_in,
            "p_esc": p_esc,
            "seq_packed_bpw": seq_bpw,
            "ra_grouplocal_table_bpw": ra_bpw,
        }
    return out


def parse_q4_layout(path: Path, elements: int) -> tuple[int, int, int]:
    """Return (code_offset, groups, file_size) after validating the header."""
    size = path.stat().st_size
    with path.open("rb") as f:
        hdr = f.read(40)
    if hdr[:8] != b"HQ30UQ4\0":
        raise RuntimeError(f"bad magic {path}")
    version, group, rank = struct.unpack_from("<IIH", hdr, 8)
    el = struct.unpack_from("<Q", hdr, 20)[0]
    if version != 1 or group != 64 or el != elements:
        raise RuntimeError(f"header mismatch {path} v={version} g={group} el={el} want {elements}")
    dim_bytes = rank * 4
    groups = (elements + 63) // 64
    scale_off = HEADER + dim_bytes
    code_off = scale_off + groups * 2
    expect = code_off + groups * 32
    if expect != size:
        raise RuntimeError(f"size mismatch {path} {size} != {expect}")
    return code_off, groups, size


def hist_q4_file(path: Path, elements: int, acc: Acc, sample_ge: bool) -> None:
    code_off, groups, _ = parse_q4_layout(path, elements)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    codes = mm[code_off:]
    # process 2M groups at a time (64MB)
    step = 2_000_000
    remaining = elements
    for g0 in range(0, groups, step):
        g1 = min(groups, g0 + step)
        blob = np.asarray(codes[g0 * 32 : g1 * 32])
        # even nibble low, odd high; q = nibble - 8
        lo = (blob & 0x0F).astype(np.int16) - 8
        hi = (blob >> 4).astype(np.int16) - 8
        # interleave back to element order
        paired = np.empty(blob.size * 2, dtype=np.int16)
        paired[0::2] = lo
        paired[1::2] = hi
        take = min(remaining, paired.size)
        acc.add_q(paired[:take], sample_group_entropy=sample_ge)
        remaining -= take
    del mm


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def quantize_g0(x: np.ndarray, bits: int) -> np.ndarray:
    """G0 HQ30UQ4 convention generalized: scale=amax/bound, q in [-2^{n-1}, bound]."""
    n = int(x.size)
    pad = (GROUP - (n % GROUP)) % GROUP
    if pad:
        x = np.pad(x, (0, pad), constant_values=0.0)
    g = x.reshape(-1, GROUP)
    bound = (1 << (bits - 1)) - 1
    amax = np.max(np.abs(g), axis=1)
    scale = np.float16(amax / max(bound, 1)).astype(np.float32)
    den = np.where(scale > 0.0, scale, 1.0)
    qmin = -(1 << (bits - 1))
    q = np.rint(g / den[:, None])
    q = np.clip(q, qmin, bound).astype(np.int16)
    return q.reshape(-1)[:n]


def load_safetensors_index() -> dict:
    idx = json.loads((BF16 / "model.safetensors.index.json").read_text())
    return idx["weight_map"]


def read_st_header(path: Path) -> dict:
    with path.open("rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        raw = f.read(hlen)
    info = json.loads(raw)
    return {"header_nbytes": hlen, "tensors": info, "data_base": 8 + hlen}


_HEADER_CACHE: dict[str, dict] = {}


def tensor_loc(weight_map: dict, name: str) -> tuple[Path, int, int, list]:
    shard = BF16 / weight_map[name]
    key = str(shard)
    if key not in _HEADER_CACHE:
        _HEADER_CACHE[key] = read_st_header(shard)
    h = _HEADER_CACHE[key]
    t = h["tensors"][name]
    begin, end = t["data_offsets"]
    return shard, h["data_base"] + begin, end - begin, t["shape"]


def iter_bf16_f32(weight_map: dict, name: str, chunk_elems: int = 4_194_304):
    path, off, nbytes, shape = tensor_loc(weight_map, name)
    elements = nbytes // 2
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    raw = mm[off : off + nbytes]
    for i in range(0, elements, chunk_elems):
        n = min(chunk_elems, elements - i)
        yield bf16_to_f32(np.asarray(raw[i * 2 : (i + n) * 2]))
    del mm
    return shape


def process_bf16_tensor(weight_map, name, accs, sample_ge):
    n_el = 0
    for chunk in iter_bf16_f32(weight_map, name):
        n_el += int(chunk.size)
        for bits, acc in accs.items():
            q = quantize_g0(chunk, bits)
            acc.add_q(q, sample_group_entropy=sample_ge)
    return n_el


def layer_of(name: str) -> int | None:
    marker = ".layers."
    if marker not in name:
        return None
    rest = name.split(marker, 1)[1]
    return int(rest.split(".", 1)[0])


def rice_gap_empirical(q: np.ndarray, inlier_abs: int) -> dict:
    """Measure rice-of-gaps for |q| > inlier_abs on one tensor-worth of q."""
    q = np.ascontiguousarray(q, dtype=np.int16).reshape(-1)
    mask = np.abs(q) > inlier_abs
    idx = np.flatnonzero(mask)
    n = int(q.size)
    n_out = int(idx.size)
    if n_out == 0:
        return {
            "n": n,
            "n_out": 0,
            "p_out": 0.0,
            "rice_k": 0,
            "pos_bits_total": 0.0,
            "pos_bpw": 0.0,
            "first_index": None,
        }
    first = int(idx[0])
    if n_out == 1:
        diffs = np.array([], dtype=np.int64)
        rice_k = 0
        rice_bits = 0.0
    else:
        diffs = np.diff(idx).astype(np.int64)  # >= 1
        # rice of (diff-1) so 1 -> 0
        gaps = diffs - 1
        best = None
        for k in range(0, 16):
            bits = float((gaps >> k).sum() + gaps.size * (1 + k))
            if best is None or bits < best[1]:
                best = (k, bits)
        rice_k, rice_bits = best
    # 32-bit first index + rice diffs, matching residual_compact
    pos_bits = 32.0 + rice_bits
    # extra value: full signed q of outliers (bits each) — or leftover
    # leftover = bits needed beyond inlier
    return {
        "n": n,
        "n_out": n_out,
        "p_out": n_out / n,
        "rice_k": int(rice_k) if n_out > 1 else 0,
        "pos_bits_total": pos_bits,
        "pos_bpw": pos_bits / n,
        "first_index": first,
    }


def rans_size_order0(symbols: np.ndarray, alphabet: int) -> dict:
    """Tiny 32-bit rANS, static 12-bit freqs, one stream. MEASURED size."""
    # Use uint16 symbols already in 0..alphabet-1
    s = np.ascontiguousarray(symbols, dtype=np.int32).reshape(-1)
    if s.size == 0:
        return {"bytes": 0, "bpw": 0.0, "n": 0}
    counts = np.bincount(s, minlength=alphabet).astype(np.int64)
    # avoid zero freq
    counts = np.maximum(counts, 1)
    SCALE = 1 << 12
    freqs = counts * SCALE // counts.sum()
    freqs = np.maximum(freqs, 1)
    extra = int(freqs.sum() - SCALE)
    # fix sum
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
    SCALE_BITS = 12
    x = L
    out = bytearray()
    for sym in s[::-1]:
        freq = int(freqs[int(sym)])
        start = int(cum[int(sym)])
        x_max = ((L >> SCALE_BITS) << 8) * freq
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        q, r = divmod(x, freq)
        x = (q << SCALE_BITS) + r + start
    for _ in range(4):
        out.append(x & 0xFF)
        x >>= 8
    # model table: alphabet * 2 bytes freqs
    table = alphabet * 2
    payload = len(out)
    n = int(s.size)
    return {
        "n": n,
        "stream_bytes": payload,
        "table_bytes": table,
        "total_bytes": payload + table,
        "stream_bpw": payload * 8 / n,
        "total_bpw": (payload + table) * 8 / n,
        "flush_bytes": 4,
    }


def rans_per_group(q: np.ndarray, bits: int, group: int = GROUP) -> dict:
    qmin = -(1 << (bits - 1))
    s = (np.ascontiguousarray(q, dtype=np.int16).reshape(-1).astype(np.int32) - qmin)
    n = int(s.size)
    pad = (group - (n % group)) % group
    if pad:
        s = np.pad(s, (0, pad), constant_values=0)
    g = s.reshape(-1, group)
    # shared static model from whole tensor
    alphabet = 1 << bits
    counts = np.bincount(s[:n], minlength=alphabet).astype(np.int64)
    # measure mean stream bytes per group using actual rANS + 4B flush
    stream_bits = 0
    take = min(g.shape[0], 256)
    idx = np.linspace(0, g.shape[0] - 1, take, dtype=np.int64)
    for gi in idx:
        r = rans_size_order0(g[gi], alphabet)
        stream_bits += r["stream_bytes"] * 8
    mean_stream_bpw = (stream_bits / take) / group
    table_bpw = (alphabet * 2 * 8) / n  # one shared table
    # theoretical: H_group + 32/group
    return {
        "sampled_groups": take,
        "measured_stream_bpw_incl_flush": mean_stream_bpw,
        "shared_table_bpw": table_bpw,
        "measured_total_bpw": mean_stream_bpw + table_bpw,
        "flush_bpw_if_4B": 32.0 / group,
        "group": group,
    }


def pick_bf16_names(weight_map: dict) -> dict[str, list[str]]:
    by = defaultdict(list)
    for name in weight_map:
        if name.startswith("vision_tower."):
            continue
        by[classify(name)].append(name)
    for k in by:
        by[k].sort(key=lambda n: (layer_of(n) is None, layer_of(n) or -1, n))
    return by


# Layers spanning DeltaNet and GQA, early/mid/late.
SAMPLE_LAYERS = {0, 3, 7, 15, 16, 31, 32, 47, 48, 63}


def select_names(by_class: dict[str, list[str]], full_small: bool = True) -> list[tuple[str, str]]:
    chosen = []
    for cls, names in by_class.items():
        if cls.startswith("small.") or cls == "other":
            if full_small:
                for n in names:
                    chosen.append((cls, n))
            continue
        if cls in ("embed", "lm_head"):
            chosen.append((cls, names[0]))
            continue
        # fused classes only exist in Q4; BF16 has splits
        for n in names:
            lyr = layer_of(n)
            if lyr is None or lyr in SAMPLE_LAYERS:
                chosen.append((cls, n))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/g1_entropy_measure.json")
    ap.add_argument("--phase", default="all", choices=["q4", "bf16", "codec", "all"])
    ap.add_argument("--max-embed-elems", type=int, default=0, help="0 = full tensor")
    args = ap.parse_args()
    t0 = time.time()
    result = {
        "schema": "hawking.g1.entropy_coding.measure.v1",
        "quantizer": {
            "name": "g0_hq30uq4_generalized",
            "group": GROUP,
            "scale": "fp16(amax / ((1<<(bits-1))-1))",
            "q_range": "[-(1<<(bits-1)), (1<<(bits-1))-1]",
            "round": "rint ties-to-even via numpy.rint",
            "match_source": "crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs pack_uniform_q4_group64 uses max_abs/7 and clamp[-8,7]",
        },
        "paths": {"bf16": str(BF16), "q4": str(Q4)},
        "started_unix": t0,
    }

    man = json.loads((Q4 / "manifest.json").read_text())
    result["q4_manifest_complete_physical_bpw"] = man["complete_physical_bpw"]
    result["q4_manifest_nominal_codec_bpw"] = man["nominal_codec_bpw"]
    result["q4_manifest_source_weight_elements"] = man["source_weight_elements"]

    if args.phase in ("q4", "all"):
        print("PHASE q4: histogram every HQ30UQ4 GEMV", flush=True)
        q4_acc: dict[str, Acc] = {}
        q4_layer: dict[str, dict[int, Acc]] = defaultdict(dict)
        n_files = 0
        for row in man["tensors"]:
            if row["kind"] != "q4":
                continue
            cls = classify(row["name"])
            if cls not in GEMV_CLASSES:
                continue
            if cls not in q4_acc:
                q4_acc[cls] = Acc(4)
            acc = q4_acc[cls]
            lyr = layer_of(row["name"])
            sample_ge = acc.group_entropy_n < 8192
            path = Q4 / "tensors" / row["artifact"]
            hist_q4_file(path, int(row["elements"]), acc, sample_ge)
            acc.tensors += 1
            n_files += 1
            if lyr is not None and lyr in (0, 31, 63):
                if lyr not in q4_layer[cls]:
                    q4_layer[cls][lyr] = Acc(4)
                # re-read for layer probe (small number of tensors)
                hist_q4_file(path, int(row["elements"]), q4_layer[cls][lyr], False)
                q4_layer[cls][lyr].tensors += 1
            if n_files % 20 == 0:
                print(f"  q4 files {n_files} last={row['name']} elems={row['elements']}", flush=True)
        result["q4_production"] = {k: v.snapshot() for k, v in sorted(q4_acc.items())}
        result["q4_layer_probe"] = {
            cls: {str(lyr): acc.snapshot()["shannon_H_bits"] for lyr, acc in layers.items()}
            for cls, layers in q4_layer.items()
        }
        print("  q4 classes", {k: v.elements for k, v in q4_acc.items()}, flush=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print("  checkpoint after q4 ->", args.out, flush=True)

    if args.phase in ("bf16", "all"):
        print("PHASE bf16: requant at", BIT_WIDTHS, flush=True)
        weight_map = load_safetensors_index()
        by = pick_bf16_names(weight_map)
        result["bf16_class_counts"] = {k: len(v) for k, v in sorted(by.items())}
        chosen = select_names(by)
        # group chosen by class
        accs_by_cls = {cls: {b: Acc(b) for b in BIT_WIDTHS} for cls, _ in chosen}
        # don't duplicate empty
        accs_by_cls = defaultdict(lambda: {b: Acc(b) for b in BIT_WIDTHS})
        per_tensor = []
        for i, (cls, name) in enumerate(chosen):
            # skip small f32 classes for multi-bit GEMV study except we still can
            if cls.startswith("small."):
                continue
            t1 = time.time()
            accs = accs_by_cls[cls]
            # embed/lm_head optionally truncated
            if cls in ("embed", "lm_head") and args.max_embed_elems:
                n_seen = 0
                for chunk in iter_bf16_f32(weight_map, name):
                    if n_seen >= args.max_embed_elems:
                        break
                    take = chunk[: max(0, args.max_embed_elems - n_seen)]
                    n_seen += int(take.size)
                    for bits, acc in accs.items():
                        acc.add_q(quantize_g0(take, bits), sample_group_entropy=acc.group_entropy_n < 4096)
                n_el = n_seen
            else:
                n_el = process_bf16_tensor(weight_map, name, accs, sample_ge=True)
            for acc in accs.values():
                acc.tensors += 1
            dt = time.time() - t1
            rec = {"class": cls, "name": name, "elements": n_el, "wall_s": dt}
            # record H at 4 bits for this tensor
            rec["H4"] = accs[4].snapshot()["shannon_H_bits"] if accs[4].elements else None
            per_tensor.append(rec)
            print(f"  [{i+1}/{len(chosen)}] {cls} {name} n={n_el} {dt:.2f}s H4={rec['H4']}", flush=True)
        result["bf16_sweep"] = {
            cls: {str(b): acc.snapshot() for b, acc in accs.items()}
            for cls, accs in accs_by_cls.items()
            if any(a.elements for a in accs.values())
        }
        result["bf16_per_tensor"] = per_tensor
        Path(args.out).write_text(json.dumps(result, indent=2))
        print("  checkpoint after bf16 ->", args.out, flush=True)

        # codec sample: one representative tensor per major class at 4 bits
        print("PHASE codec samples on one tensor/class at 4/3/2 bits", flush=True)
        codec = {}
        reps = {}
        for cls, names in by.items():
            if cls.startswith("small.") or cls == "other":
                continue
            # pick layer 0 or the only tensor
            pick = None
            for n in names:
                if layer_of(n) == 0 or layer_of(n) is None:
                    pick = n
                    break
            if pick is None:
                pick = names[0]
            reps[cls] = pick
        for cls, name in reps.items():
            # 2M weights is enough for codec sizing; rANS is Python-slow.
            chunks = []
            n = 0
            for chunk in iter_bf16_f32(weight_map, name, chunk_elems=2_097_152):
                chunks.append(chunk)
                n += int(chunk.size)
                if n >= 2_097_152:
                    break
            x = np.concatenate(chunks)[:2_097_152]
            entry = {"name": name, "elements_used": int(x.size)}
            for bits in (2, 3, 4, 5, 6, 8):
                q = quantize_g0(x, bits)
                # rice gaps at several inlier radii
                rice_rows = {}
                for R in (0, 1, 3, 7, 15):
                    if R >= (1 << (bits - 1)):
                        continue
                    rg = rice_gap_empirical(q, R)
                    # value leftovers: store full q of outliers at `bits` bits
                    val_bpw = rg["p_out"] * bits
                    # main: if R==0, no main (all outliers). else signed width for [-R,R]
                    if R == 0:
                        main_bpw = 0.0
                    else:
                        main_bpw = math.ceil(math.log2(2 * R + 1))
                    rice_rows[f"R{R}"] = {
                        **rg,
                        "main_bpw": main_bpw,
                        "outlier_value_bpw": val_bpw,
                        "total_seq_rice_bpw": main_bpw + rg["pos_bpw"] + val_bpw,
                        "ra_grouplocal_bpw": main_bpw + rg["p_out"] * (6 + bits),
                    }
                rans = {}
                for gsz in (64, 256, 1024):
                    rans[f"G{gsz}"] = rans_per_group(q, bits, group=gsz)
                # global rANS on this sample
                qmin = -(1 << (bits - 1))
                sym = q.astype(np.int32) - qmin
                global_rans = rans_size_order0(sym, 1 << bits)
                entry[str(bits)] = {
                    "H": entropy_from_list(q, 1 << bits),
                    "rice_outlier": rice_rows,
                    "rans_per_group": rans,
                    "rans_global": global_rans,
                }
            codec[cls] = entry
            print(f"  codec {cls} n={x.size} H4={entry['4']['H']:.4f} ransG64={entry['4']['rans_per_group']['G64']['measured_total_bpw']:.4f}", flush=True)
        result["codec_samples"] = codec

    result["wall_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("WROTE", args.out, "wall_s", result["wall_s"], flush=True)


if __name__ == "__main__":
    main()
