#!/usr/bin/env python3
"""Measure inherited Q4-container constraints vs the BF16 parent.

Read-only. Headers + f32v2 smalls + G0 catalog. No GEMV payloads. No GPU.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

BF16 = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
)
G0 = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1"
)
N_LANG = 26_895_998_464
LANG_PREFIX = "language_model."
VIS_PREFIX = "vision_tower."


def bpw(bytes_, elems):
    if elems == 0:
        return float("nan")
    return (bytes_ * 8.0) / elems


def read_st_header(path: Path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        raw = f.read(hlen)
    meta = json.loads(raw)
    tensors = {}
    for name, info in meta.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        tensors[name] = {
            "dtype": info["dtype"],
            "shape": list(info["shape"]),
            "nbytes": int(end - begin),
            "elements": int(prod(info["shape"])),
        }
    return hlen, tensors


def prod(xs):
    p = 1
    for x in xs:
        p *= int(x)
    return p


def classify_lang(name: str) -> str:
    if name == "language_model.model.embed_tokens.weight":
        return "embed"
    if name == "language_model.lm_head.weight":
        return "lm_head"
    if name == "language_model.model.norm.weight":
        return "norm.final"
    if ".input_layernorm.weight" in name:
        return "norm.input"
    if ".post_attention_layernorm.weight" in name:
        return "norm.post_attn"
    if ".mlp.gate_proj.weight" in name:
        return "mlp.gate"
    if ".mlp.up_proj.weight" in name:
        return "mlp.up"
    if ".mlp.down_proj.weight" in name:
        return "mlp.down"
    if ".linear_attn.in_proj_qkv.weight" in name:
        return "dn.in_proj_qkv"
    if ".linear_attn.in_proj_z.weight" in name:
        return "dn.in_proj_z"
    if ".linear_attn.in_proj_a.weight" in name:
        return "dn.in_proj_a"
    if ".linear_attn.in_proj_b.weight" in name:
        return "dn.in_proj_b"
    if ".linear_attn.out_proj.weight" in name:
        return "dn.out_proj"
    if ".linear_attn.conv1d.weight" in name:
        return "dn.conv1d"
    if ".linear_attn.A_log" in name:
        return "dn.A_log"
    if ".linear_attn.dt_bias" in name:
        return "dn.dt_bias"
    if ".linear_attn.norm.weight" in name:
        return "dn.norm"
    if ".self_attn.q_proj.weight" in name:
        return "gqa.q"
    if ".self_attn.k_proj.weight" in name:
        return "gqa.k"
    if ".self_attn.v_proj.weight" in name:
        return "gqa.v"
    if ".self_attn.o_proj.weight" in name:
        return "gqa.o"
    if ".self_attn.q_norm.weight" in name:
        return "gqa.q_norm"
    if ".self_attn.k_norm.weight" in name:
        return "gqa.k_norm"
    return "UNKNOWN"


def classify_g0(name: str) -> str:
    c = classify_lang(name)
    if c != "UNKNOWN":
        return c
    if ".linear_attn.in_proj_qkvz.weight" in name:
        return "dn.in_proj_qkvz"
    if ".linear_attn.in_proj_ba.weight" in name:
        return "dn.in_proj_ba"
    return "UNKNOWN"


def q4_formula_bytes(elements: int, rank: int, group: int = 64) -> int:
    groups = (elements + group - 1) // group
    header = 32 + 4 * rank
    return header + groups * 2 + groups * (group // 2)


def f32v2_formula_bytes(elements: int) -> int:
    return 8 + 4 * elements


def q4_absmax_recon(values):
    """HQ30UQ4-faithful: flat groups of 64, s=f16(max_abs/7), q in [-8,7], rint ties-even."""
    import math

    n = len(values)
    recon = [0.0] * n
    gsz = 64
    groups = (n + gsz - 1) // gsz
    pad_elems = groups * gsz - n
    for g in range(groups):
        start = g * gsz
        end = min(start + gsz, n)
        max_abs = 0.0
        for i in range(start, end):
            a = abs(values[i])
            if a > max_abs:
                max_abs = a
        # f16(max_abs/7)
        s = f16_from_f32(max_abs / 7.0)
        for i in range(start, end):
            if s == 0.0:
                q = 0
            else:
                q = rint_ties_even(values[i] / s)
                if q < -8:
                    q = -8
                elif q > 7:
                    q = 7
            recon[i] = q * s
    return recon, pad_elems


def f16_from_f32(x: float) -> float:
    # IEEE f16 round-to-nearest-even via struct
    import struct as st

    packed = st.pack("<e", x)  # Python 3 'e' is IEEE f16
    return st.unpack("<e", packed)[0]


def rint_ties_even(x: float) -> int:
    import math

    if not math.isfinite(x):
        return 0
    truncated = math.trunc(x)
    frac = abs(x - truncated)
    if frac > 0.5:
        return truncated + (1 if x >= 0 else -1)
    if frac < 0.5:
        return truncated
    # exact .5: ties to even
    if truncated % 2 == 0:
        return truncated
    return truncated + (1 if x >= 0 else -1)


def cosine(a, b):
    dot = sa = sb = 0.0
    for x, y in zip(a, b):
        xf = float(x)
        yf = float(y)
        dot += xf * yf
        sa += xf * xf
        sb += yf * yf
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return dot / (sa ** 0.5 * sb ** 0.5)


def rel_l2(a, b):
    num = den = 0.0
    for x, y in zip(a, b):
        d = float(x) - float(y)
        num += d * d
        den += float(x) * float(x)
    if den == 0.0:
        return float("nan")
    return (num ** 0.5) / (den ** 0.5)


def read_f32v2(path: Path):
    data = path.read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    expect = 8 + 4 * n
    if len(data) != expect:
        raise RuntimeError(f"{path} size {len(data)} != {expect}")
    vals = list(struct.unpack_from("<" + "f" * n, data, 8))
    return vals


def main():
    out = {}

    index = json.loads((BF16 / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    total_size = index.get("total_size")
    prefixes = Counter(n.split(".")[0] for n in weight_map)
    deeper = Counter()
    for n in weight_map:
        if n.startswith("language_model."):
            deeper["language_model"] += 1
        elif n.startswith("vision_tower."):
            deeper["vision_tower"] += 1
        else:
            deeper["OTHER:" + n.split(".")[0]] += 1

    shards = sorted(set(weight_map.values()))
    header_census = {}
    dtype_c = Counter()
    lang = {}
    vis = {}
    other = {}
    missing = []
    for shard in shards:
        hlen, tensors = read_st_header(BF16 / shard)
        for name, loc in weight_map.items():
            if loc != shard:
                continue
            if name not in tensors:
                missing.append(name)
                continue
            info = tensors[name]
            dtype_c[info["dtype"]] += 1
            rec = {
                "shape": info["shape"],
                "elements": info["elements"],
                "dtype": info["dtype"],
                "nbytes": info["nbytes"],
                "shard": shard,
            }
            if name.startswith(LANG_PREFIX):
                lang[name] = rec
            elif name.startswith(VIS_PREFIX):
                vis[name] = rec
            else:
                other[name] = rec
        header_census[shard] = {"header_nbytes": hlen, "n_tensors_in_header": len(tensors)}

    # expected names not in headers
    for name in weight_map:
        if name not in lang and name not in vis and name not in other and name not in missing:
            missing.append(name)

    lang_elems = sum(t["elements"] for t in lang.values())
    vis_elems = sum(t["elements"] for t in vis.values())
    other_elems = sum(t["elements"] for t in other.values())

    lang_classes = defaultdict(lambda: {"n": 0, "elements": 0, "shapes": Counter()})
    unknown_lang = []
    for name, rec in lang.items():
        c = classify_lang(name)
        if c == "UNKNOWN":
            unknown_lang.append(name)
        lang_classes[c]["n"] += 1
        lang_classes[c]["elements"] += rec["elements"]
        lang_classes[c]["shapes"][tuple(rec["shape"])] += 1

    mtp = [n for n in weight_map if "mtp" in n.lower()]

    # mixer rule
    mixer = {}
    for layer in range(64):
        mixer[layer] = "gqa" if (layer + 1) % 4 == 0 else "dn"

    # required source set from geometry
    required = []
    required.append("language_model.model.embed_tokens.weight")
    for layer in range(64):
        p = f"language_model.model.layers.{layer}."
        required += [
            p + "input_layernorm.weight",
            p + "post_attention_layernorm.weight",
            p + "mlp.gate_proj.weight",
            p + "mlp.up_proj.weight",
            p + "mlp.down_proj.weight",
        ]
        if mixer[layer] == "dn":
            required += [
                p + "linear_attn.in_proj_qkv.weight",
                p + "linear_attn.in_proj_z.weight",
                p + "linear_attn.in_proj_a.weight",
                p + "linear_attn.in_proj_b.weight",
                p + "linear_attn.out_proj.weight",
                p + "linear_attn.conv1d.weight",
                p + "linear_attn.A_log",
                p + "linear_attn.dt_bias",
                p + "linear_attn.norm.weight",
            ]
        else:
            required += [
                p + "self_attn.q_proj.weight",
                p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight",
                p + "self_attn.o_proj.weight",
                p + "self_attn.q_norm.weight",
                p + "self_attn.k_norm.weight",
            ]
    required.append("language_model.model.norm.weight")
    required.append("language_model.lm_head.weight")

    required_set = set(required)
    lang_set = set(lang)
    missing_required = sorted(required_set - lang_set)
    extra_lang = sorted(lang_set - required_set)

    # fused catalog names
    fused_required = []
    for name in required:
        if name.endswith("linear_attn.in_proj_qkv.weight"):
            fused_required.append(name.replace("in_proj_qkv.weight", "in_proj_qkvz.weight"))
        elif name.endswith("linear_attn.in_proj_z.weight"):
            continue
        elif name.endswith("linear_attn.in_proj_b.weight"):
            fused_required.append(name.replace("in_proj_b.weight", "in_proj_ba.weight"))
        elif name.endswith("linear_attn.in_proj_a.weight"):
            continue
        else:
            fused_required.append(name)

    out["bf16"] = {
        "index_entries": len(weight_map),
        "total_size": total_size,
        "shards": len(shards),
        "missing_in_headers": missing,
        "dtype": dict(dtype_c),
        "prefix_first_token": dict(prefixes),
        "prefix_root": dict(deeper),
        "language_tensors": len(lang),
        "language_elements": lang_elems,
        "vision_tensors": len(vis),
        "vision_elements": vis_elems,
        "other_tensors": len(other),
        "other_names": sorted(other),
        "other_elements": other_elems,
        "mtp_names": mtp,
        "unknown_lang_names": unknown_lang,
        "required_source_count": len(required),
        "missing_required": missing_required,
        "extra_lang": extra_lang,
        "fused_catalog_count": len(fused_required),
        "lang_classes": {
            k: {
                "n": v["n"],
                "elements": v["elements"],
                "shapes": {"x".join(map(str, s)): c for s, c in v["shapes"].items()},
            }
            for k, v in sorted(lang_classes.items())
        },
        "N_lang_matches_contract": lang_elems == N_LANG,
    }

    man = json.loads((G0 / "manifest.json").read_text())
    rows = man["tensors"]
    q4 = [r for r in rows if r["kind"] == "q4"]
    f32 = [r for r in rows if r["kind"] == "f32"]

    q4_elems = sum(r["elements"] for r in q4)
    f32_elems = sum(r["elements"] for r in f32)
    q4_bytes = sum(r["bytes"] for r in q4)
    f32_bytes = sum(r["bytes"] for r in f32)
    payload = sum(r["bytes"] for r in rows)
    elems = sum(r["elements"] for r in rows)

    # lstat every listed artifact
    size_mismatch = []
    missing_art = []
    listed_lstat = 0
    for r in rows:
        p = G0 / "tensors" / r["artifact"]
        if not p.exists():
            missing_art.append(r["name"])
            continue
        sz = p.stat().st_size
        listed_lstat += sz
        if sz != r["bytes"]:
            size_mismatch.append((r["name"], sz, r["bytes"]))

    # unused sidecars
    all_files = list((G0 / "tensors").iterdir()) if (G0 / "tensors").is_dir() else []
    listed_names = {r["artifact"] for r in rows}
    unused = []
    unused_bytes = 0
    for f in all_files:
        if f.name not in listed_names and f.is_file():
            unused.append(f.name)
            unused_bytes += f.stat().st_size

    q4_not_div64 = [r["name"] for r in q4 if r["elements"] % 64 != 0]
    f32_not_div64 = [
        {"name": r["name"], "elements": r["elements"], "shape": r["shape"]}
        for r in f32
        if r["elements"] % 64 != 0
    ]
    f32_div64 = [r for r in f32 if r["elements"] % 64 == 0]
    f32_div64_elems = sum(r["elements"] for r in f32_div64)
    f32_not_div64_elems = f32_elems - f32_div64_elems

    q4_groups = 0
    q4_scale = 0
    q4_code = 0
    q4_hdr = 0
    q4_formula_mismatch = 0
    embed_scale = 0
    gemv_scale = 0
    gemv_code = 0
    gemv_groups = 0
    k_set = Counter()
    q4_ranks = Counter()
    for r in q4:
        e = r["elements"]
        rank = len(r["shape"])
        q4_ranks[rank] += 1
        groups = (e + 63) // 64
        q4_groups += groups
        q4_scale += groups * 2
        q4_code += groups * 32
        hdr = 32 + 4 * rank
        q4_hdr += hdr
        expect = q4_formula_bytes(e, rank)
        if expect != r["bytes"]:
            q4_formula_mismatch += 1
        if r["name"].endswith("embed_tokens.weight"):
            embed_scale += groups * 2
        else:
            gemv_scale += groups * 2
            gemv_code += groups * 32
            gemv_groups += groups
        if rank == 2:
            k_set[r["shape"][-1]] += 1

    f32_hdr = 0
    f32_formula_mismatch = 0
    f32_ranks = Counter()
    f32_by_class = defaultdict(lambda: {"n": 0, "elements": 0, "bytes": 0, "shapes": Counter()})
    for r in f32:
        rank = len(r["shape"])
        f32_ranks[rank] += 1
        f32_hdr += 8
        if f32v2_formula_bytes(r["elements"]) != r["bytes"]:
            f32_formula_mismatch += 1
        c = classify_g0(r["name"])
        f32_by_class[c]["n"] += 1
        f32_by_class[c]["elements"] += r["elements"]
        f32_by_class[c]["bytes"] += r["bytes"]
        f32_by_class[c]["shapes"][tuple(r["shape"])] += 1

    # smallest Q4 vs largest f32
    smallest_q4 = min(q4, key=lambda r: r["elements"])
    largest_f32 = max(f32, key=lambda r: r["elements"])

    # hypothetical Q4 of smalls
    f32_as_q4_bytes = 0
    f32_as_q4_codes = 0
    f32_as_q4_scales = 0
    f32_as_q4_hdr = 0
    f32_as_q4_pad_elems = 0
    for r in f32:
        e = r["elements"]
        rank = len(r["shape"])
        groups = (e + 63) // 64
        f32_as_q4_hdr += 32 + 4 * rank
        f32_as_q4_scales += groups * 2
        f32_as_q4_codes += groups * 32
        f32_as_q4_pad_elems += groups * 64 - e
        f32_as_q4_bytes += q4_formula_bytes(e, rank)

    f32_as_q4_div64_only_bytes = 0
    for r in f32_div64:
        f32_as_q4_div64_only_bytes += q4_formula_bytes(r["elements"], len(r["shape"]))
    # remainder stay f32v2
    f32_remain_bytes = sum(r["bytes"] for r in f32 if r["elements"] % 64 != 0)

    # codes-only for Q4 elems
    codes_only_q4 = q4_code  # no header no scale
    # 4-bit codes for ALL language elems (pad last groups of smalls)
    all_groups = (elems + 63) // 64  # wrong: pad is per-tensor
    all_code_bytes = 0
    all_scale_bytes = 0
    all_pad = 0
    for r in rows:
        e = r["elements"]
        g = (e + 63) // 64
        all_code_bytes += g * 32
        all_scale_bytes += g * 2
        all_pad += g * 64 - e

    man_bytes = (G0 / "manifest.json").stat().st_size

    out["g0"] = {
        "schema": man.get("schema"),
        "source_weight_elements": man.get("source_weight_elements"),
        "tensor_payload_bytes": man.get("tensor_payload_bytes"),
        "complete_physical_bpw_manifest": man.get("complete_physical_bpw"),
        "q4_tensors": len(q4),
        "f32_tensors": len(f32),
        "tensor_count": len(rows),
        "skipped_vision": man.get("skipped_vision_tensors"),
        "q4_group_size": man.get("q4_group_size"),
        "q4_elements": q4_elems,
        "f32_elements": f32_elems,
        "sum_elements": elems,
        "q4_bytes_catalog": q4_bytes,
        "f32_bytes_catalog": f32_bytes,
        "payload_catalog": payload,
        "listed_lstat": listed_lstat,
        "missing_art": missing_art,
        "size_mismatch_n": len(size_mismatch),
        "unused_sidecar_n": len(unused),
        "unused_sidecar_bytes": unused_bytes,
        "unused_ext": dict(Counter(Path(n).suffix for n in unused)),
        "manifest_bytes": man_bytes,
        "q4_not_div64": q4_not_div64,
        "f32_not_div64_n": len(f32_not_div64),
        "f32_not_div64_elems": f32_not_div64_elems,
        "f32_div64_n": len(f32_div64),
        "f32_div64_elems": f32_div64_elems,
        "q4_groups": q4_groups,
        "q4_scale_bytes": q4_scale,
        "q4_code_bytes": q4_code,
        "q4_header_bytes": q4_hdr,
        "q4_formula_mismatch": q4_formula_mismatch,
        "embed_scale_bytes": embed_scale,
        "gemv_scale_bytes": gemv_scale,
        "gemv_code_bytes": gemv_code,
        "gemv_groups": gemv_groups,
        "k_axis_counts": {str(k): v for k, v in k_set.items()},
        "q4_ranks": dict(q4_ranks),
        "f32_ranks": dict(f32_ranks),
        "f32_header_bytes": f32_hdr,
        "f32_formula_mismatch": f32_formula_mismatch,
        "smallest_q4": {
            "name": smallest_q4["name"],
            "elements": smallest_q4["elements"],
            "shape": smallest_q4["shape"],
            "bytes": smallest_q4["bytes"],
        },
        "largest_f32": {
            "name": largest_f32["name"],
            "elements": largest_f32["elements"],
            "shape": largest_f32["shape"],
            "bytes": largest_f32["bytes"],
        },
        "fused_name_match": sorted({r["name"] for r in rows}) == sorted(fused_required),
        "g0_minus_fused_required": sorted({r["name"] for r in rows} - set(fused_required)),
        "fused_required_minus_g0": sorted(set(fused_required) - {r["name"] for r in rows}),
        "f32_by_class": {
            k: {
                "n": v["n"],
                "elements": v["elements"],
                "bytes": v["bytes"],
                "shapes": {"x".join(map(str, s)): c for s, c in v["shapes"].items()},
            }
            for k, v in sorted(f32_by_class.items())
        },
    }

    # small tensor value stats + Q4 recon (all 353, 2.6M elems)
    class_stats = defaultdict(
        lambda: {
            "n": 0,
            "elements": 0,
            "min": float("inf"),
            "max": float("-inf"),
            "absmax": 0.0,
            "sum": 0.0,
            "sumsq": 0.0,
            "n_zero": 0,
            "n_nonfinite": 0,
            "cos_sum": 0.0,
            "cos_min": float("inf"),
            "rel_l2_max": 0.0,
            "rel_l2_sum": 0.0,
        }
    )
    worst_cos = {"name": None, "cos": 1.0, "cls": None}
    for r in f32:
        vals = read_f32v2(G0 / "tensors" / r["artifact"])
        c = classify_g0(r["name"])
        st = class_stats[c]
        st["n"] += 1
        st["elements"] += len(vals)
        mn = min(vals)
        mx = max(vals)
        am = max(abs(mn), abs(mx))
        st["min"] = min(st["min"], mn)
        st["max"] = max(st["max"], mx)
        st["absmax"] = max(st["absmax"], am)
        for v in vals:
            st["sum"] += v
            st["sumsq"] += v * v
            if v == 0.0:
                st["n_zero"] += 1
            if v != v or v in (float("inf"), float("-inf")):
                st["n_nonfinite"] += 1
        recon, _pad = q4_absmax_recon(vals)
        cos = cosine(vals, recon)
        rl = rel_l2(vals, recon)
        st["cos_sum"] += cos
        st["cos_min"] = min(st["cos_min"], cos)
        st["rel_l2_max"] = max(st["rel_l2_max"], rl)
        st["rel_l2_sum"] += rl
        if cos < worst_cos["cos"]:
            worst_cos = {"name": r["name"], "cos": cos, "cls": c, "rel_l2": rl}

    out["smalls_q4_probe"] = {
        "worst": worst_cos,
        "classes": {
            k: {
                **{
                    kk: vv
                    for kk, vv in v.items()
                    if kk not in ("cos_sum", "rel_l2_sum", "sum", "sumsq")
                },
                "mean": v["sum"] / v["elements"] if v["elements"] else None,
                "rms": (v["sumsq"] / v["elements"]) ** 0.5 if v["elements"] else None,
                "cos_mean": v["cos_sum"] / v["n"] if v["n"] else None,
                "rel_l2_mean": v["rel_l2_sum"] / v["n"] if v["n"] else None,
            }
            for k, v in sorted(class_stats.items())
        },
    }

    # prize ledger
    headers_total = q4_hdr + f32_hdr
    scale_all = q4_scale
    scale_gemv = gemv_scale
    f32_body = f32_bytes - f32_hdr
    f32_vs_codes = f32_body - (f32_elems // 2)  # 4-bit codes, no pad account
    # more honest: vs Q4-family payload (codes+scales+hdr) for same tensors
    f32_exemption_vs_q4_family = f32_bytes - f32_as_q4_bytes
    # vs 4-bit codes only (pad last group billed)
    f32_vs_q4_codes_padded = f32_bytes - f32_as_q4_codes
    # vs 4-bit codes of exact elements (no pad, no scale, no hdr)
    exact_4bit_smalls = (f32_elems * 4 + 7) // 8

    # function bits = 4-bit codes of all language elements, no pad
    function_4bit_bytes = (elems * 4) // 8  # elems % 8? 2645504+26893352960 even
    assert (elems * 4) % 8 == 0

    complete_bytes = payload
    complete_bpw = bpw(complete_bytes, N_LANG)
    codes_q4_bpw = bpw(q4_code, N_LANG)
    scale_all_bpw = bpw(scale_all, N_LANG)
    scale_gemv_bpw = bpw(scale_gemv, N_LANG)
    hdr_bpw = bpw(headers_total, N_LANG)
    f32_body_bpw = bpw(f32_body, N_LANG)
    f32_ex_vs_q4_bpw = bpw(f32_exemption_vs_q4_family, N_LANG)

    # floors
    # A: drop headers only (lossless container)
    floor_a_bytes = complete_bytes - headers_total
    # B: Q4-family smalls (same codec, including their own scale+hdr) + keep GEMV as-is
    #    replace f32_bytes with f32_as_q4_bytes
    floor_b_bytes = complete_bytes - f32_bytes + f32_as_q4_bytes
    # B2: Q4 only the E%64==0 smalls, rest stay f32v2
    floor_b2_bytes = (
        complete_bytes
        - sum(r["bytes"] for r in f32_div64)
        + f32_as_q4_div64_only_bytes
    )
    # C: drop scale plane (kills this kernel) keep codes + f32 smalls + headers
    floor_c_bytes = complete_bytes - scale_all
    # D: codes of 402 + 4-bit exact smalls, no scale no header no pad
    floor_d_bytes = q4_code + (f32_elems * 4) // 8
    # E: 4-bit codes of every language element, nothing else
    floor_e_bytes = function_4bit_bytes
    # F: codes+scales of 402 (keep codec function) + 4-bit exact smalls, no headers
    floor_f_bytes = q4_code + q4_scale + (f32_elems * 4) // 8

    out["prize"] = {
        "N": N_LANG,
        "complete_bytes": complete_bytes,
        "complete_bpw": complete_bpw,
        "q4_code_bytes": q4_code,
        "q4_scale_bytes": scale_all,
        "gemv_scale_bytes": scale_gemv,
        "embed_scale_bytes": embed_scale,
        "headers_total": headers_total,
        "q4_headers": q4_hdr,
        "f32_headers": f32_hdr,
        "group_pad_elems_q4": 0,
        "group_pad_elems_if_smalls_q4": all_pad,
        "f32_body_bytes": f32_body,
        "f32_as_q4_family_bytes": f32_as_q4_bytes,
        "f32_as_q4_codes": f32_as_q4_codes,
        "f32_as_q4_scales": f32_as_q4_scales,
        "f32_as_q4_hdr": f32_as_q4_hdr,
        "f32_as_q4_pad_elems": f32_as_q4_pad_elems,
        "f32_exemption_vs_q4_family_bytes": f32_exemption_vs_q4_family,
        "f32_vs_q4_codes_padded_bytes": f32_vs_q4_codes_padded,
        "exact_4bit_smalls_bytes": exact_4bit_smalls,
        "function_4bit_all_bytes": function_4bit_bytes,
        "unused_sidecar_bytes": unused_bytes,
        "manifest_bytes": man_bytes,
        "bpw": {
            "complete": complete_bpw,
            "q4_codes_on_N": codes_q4_bpw,
            "scale_all_on_N": scale_all_bpw,
            "scale_gemv_on_N": scale_gemv_bpw,
            "headers_on_N": hdr_bpw,
            "f32_body_on_N": f32_body_bpw,
            "f32_exemption_vs_q4_family_on_N": f32_ex_vs_q4_bpw,
            "unused_sidecar_on_N_not_in_complete": bpw(unused_bytes, N_LANG),
            "manifest_on_N_not_in_complete": bpw(man_bytes, N_LANG),
        },
        "floors": {
            "A_drop_headers_only": {
                "bytes": floor_a_bytes,
                "bpw": bpw(floor_a_bytes, N_LANG),
                "delta_bpw": complete_bpw - bpw(floor_a_bytes, N_LANG),
                "lossless": True,
            },
            "B_smalls_as_q4_family": {
                "bytes": floor_b_bytes,
                "bpw": bpw(floor_b_bytes, N_LANG),
                "delta_bpw": complete_bpw - bpw(floor_b_bytes, N_LANG),
                "lossless": False,
                "note": "same HQ30UQ4 including their scale+hdr; generate untested",
            },
            "B2_smalls_div64_as_q4_rest_f32": {
                "bytes": floor_b2_bytes,
                "bpw": bpw(floor_b2_bytes, N_LANG),
                "delta_bpw": complete_bpw - bpw(floor_b2_bytes, N_LANG),
                "lossless": False,
            },
            "C_drop_scale_plane_kills": {
                "bytes": floor_c_bytes,
                "bpw": bpw(floor_c_bytes, N_LANG),
                "delta_bpw": complete_bpw - bpw(floor_c_bytes, N_LANG),
                "lossless": False,
                "kills": True,
            },
            "D_q4_codes_plus_4bit_smalls_no_meta": {
                "bytes": floor_d_bytes,
                "bpw": bpw(floor_d_bytes, N_LANG),
                "delta_bpw": complete_bpw - bpw(floor_d_bytes, N_LANG),
                "note": "container stripped; codes remain; scales gone (kills this kernel)",
            },
            "E_4bit_codes_every_language_elem": {
                "bytes": floor_e_bytes,
                "bpw": bpw(floor_e_bytes, N_LANG),
                "delta_bpw": complete_bpw - bpw(floor_e_bytes, N_LANG),
            },
            "F_keep_q4_codec_meta_4bit_smalls_no_hdr": {
                "bytes": floor_f_bytes,
                "bpw": bpw(floor_f_bytes, N_LANG),
                "delta_bpw": complete_bpw - bpw(floor_f_bytes, N_LANG),
                "note": "same GEMV codec (codes+scales); smalls at 4-bit codes; no headers",
            },
        },
    }

    # 11-bit global scale codebook projection (cite uniqueness from byte-deletions; do not rescan 800MB)
    # we do not rescan; just arithmetic from measured group counts
    gemv_groups = gemv_groups
    codebook_11bit_index_bytes = (gemv_groups * 11 + 7) // 8
    # 1903 unique cited
    codebook_payload = 1903 * 2
    codebook_save_vs_f16 = gemv_scale - (codebook_11bit_index_bytes + codebook_payload)
    out["prize"]["scale_codebook_11bit_gemv_PROJECTED"] = {
        "unique_f16_CITED": 1903,
        "index_bytes": codebook_11bit_index_bytes,
        "codebook_bytes": codebook_payload,
        "save_bytes": codebook_save_vs_f16,
        "save_bpw": bpw(codebook_save_vs_f16, N_LANG),
        "note": "uniqueness CITED g1-byte-deletions.md; arithmetic this lane on measured gemv_groups",
    }

    # K divisibility
    ks = sorted(int(k) for k in k_set)
    out["geometry"] = {
        "unique_K": ks,
        "K_mod_64": {str(k): k % 64 for k in ks},
        "K_mod_128": {str(k): k % 128 for k in ks},
        "K_mod_512": {str(k): k % 512 for k in ks},
        "gcd_K": None,
    }
    g = ks[0]
    for k in ks[1:]:
        while k:
            g, k = k, g % k
    out["geometry"]["gcd_K"] = g

    path = Path("/tmp/g1_q4_geometry_constraints_measure.json")
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(path)
    print(json.dumps({
        "lang_tensors": out["bf16"]["language_tensors"],
        "lang_elems": out["bf16"]["language_elements"],
        "vis_tensors": out["bf16"]["vision_tensors"],
        "vis_elems": out["bf16"]["vision_elements"],
        "other": out["bf16"]["other_names"],
        "mtp": out["bf16"]["mtp_names"],
        "missing_required": out["bf16"]["missing_required"],
        "extra_lang": out["bf16"]["extra_lang"],
        "N_ok": out["bf16"]["N_lang_matches_contract"],
        "g0_q4": out["g0"]["q4_tensors"],
        "g0_f32": out["g0"]["f32_tensors"],
        "payload": out["g0"]["payload_catalog"],
        "lstat": out["g0"]["listed_lstat"],
        "complete_bpw": out["prize"]["complete_bpw"],
        "scale_all": out["prize"]["q4_scale_bytes"],
        "scale_gemv": out["prize"]["gemv_scale_bytes"],
        "headers": out["prize"]["headers_total"],
        "f32_exemption_vs_q4": out["prize"]["f32_exemption_vs_q4_family_bytes"],
        "floors": {k: v["bpw"] for k, v in out["prize"]["floors"].items()},
        "worst_small_q4": out["smalls_q4_probe"]["worst"],
        "fused_match": out["g0"]["fused_name_match"],
    }, indent=2))


if __name__ == "__main__":
    main()
