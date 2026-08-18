#!/usr/bin/env python3
"""Parse mixed-q3mlp-v1 HQ38M20 catalog and emit geometry tables. CPU only."""
from __future__ import annotations

import json
import os
import struct
from collections import Counter, defaultdict

ROOT = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1"
CAT = os.path.join(ROOT, "catalog.hq38m20")
GPU_CORES = 60  # qwen38_token_ns_ledger.rs geo_tpr64_occupancy


def read_u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def read_u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def read_u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def classify(name: str) -> str:
    if name.endswith("mlp.gate_proj.weight"):
        return "mlp.gate"
    if name.endswith("mlp.up_proj.weight"):
        return "mlp.up"
    if name.endswith("mlp.down_proj.weight"):
        return "mlp.down"
    if name.endswith("linear_attn.in_proj_qkv.weight"):
        return "dn.in_proj_qkv"
    if name.endswith("linear_attn.in_proj_z.weight"):
        return "dn.in_proj_z"
    if name.endswith("linear_attn.in_proj_a.weight"):
        return "dn.in_proj_a"
    if name.endswith("linear_attn.in_proj_b.weight"):
        return "dn.in_proj_b"
    if name.endswith("linear_attn.in_proj_qkvz.weight"):
        return "dn.in_proj_qkvz_FUSED"
    if name.endswith("linear_attn.in_proj_ba.weight"):
        return "dn.in_proj_ba_FUSED"
    if name.endswith("linear_attn.out_proj.weight"):
        return "dn.out"
    if name.endswith("self_attn.q_proj.weight"):
        return "gqa.q"
    if name.endswith("self_attn.k_proj.weight"):
        return "gqa.k"
    if name.endswith("self_attn.v_proj.weight"):
        return "gqa.v"
    if name.endswith("self_attn.o_proj.weight"):
        return "gqa.o"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if name.endswith("embed_tokens.weight"):
        return "embed"
    return "other"


def parse_catalog(path):
    raw = open(path, "rb").read()
    assert raw[:8] == b"HQ38M20\0", raw[:8]
    version = read_u32(raw, 8)
    n_tensors = read_u32(raw, 12)
    n_segments = read_u32(raw, 16)
    name_blob_bytes = read_u32(raw, 24)
    cursor = 32
    by_id = {}
    for _ in range(n_segments):
        sid = read_u16(raw, cursor)
        name_len = read_u16(raw, cursor + 2)
        cursor += 44
        filename = raw[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        if os.path.isabs(filename):
            by_id[sid] = filename
        else:
            by_id[sid] = os.path.join(ROOT, "segments", filename)
    table_bytes = n_tensors * 128
    table = raw[cursor : cursor + table_bytes]
    cursor += table_bytes
    name_blob = raw[cursor : cursor + name_blob_bytes]
    rows = []
    for i in range(n_tensors):
        rec = table[i * 128 : (i + 1) * 128]
        name_off = read_u32(rec, 0)
        name_len = read_u16(rec, 4)
        codec = rec[6]
        ndim = rec[8]
        shape = [read_u32(rec, 12 + d * 4) for d in range(ndim)]
        seg_id = read_u16(rec, 36)
        offset = read_u64(rec, 40)
        nbytes = read_u64(rec, 48)
        name = name_blob[name_off : name_off + name_len].decode("utf-8")
        rows.append(
            {
                "name": name,
                "codec": codec,
                "shape": shape,
                "elements": int(__import__("math").prod(shape)) if shape else 0,
                "segment_id": seg_id,
                "segment_path": by_id[seg_id],
                "offset": offset,
                "nbytes": nbytes,
                "class": classify(name),
            }
        )
    return {
        "version": version,
        "n_tensors": n_tensors,
        "n_segments": n_segments,
        "catalog_bytes": len(raw),
        "rows": rows,
    }


def peek_hgravu(row, max_json=800):
    with open(row["segment_path"], "rb") as f:
        f.seek(row["offset"])
        prefix = f.read(12 + max_json)
    if prefix[:8] != b"HGRAVU01":
        return {"magic": prefix[:8].decode("latin1", "replace"), "ok": False}
    hlen = read_u32(prefix, 8)
    if 12 + hlen > len(prefix):
        with open(row["segment_path"], "rb") as f:
            f.seek(row["offset"])
            prefix = f.read(12 + hlen)
    header = prefix[12 : 12 + hlen]
    try:
        obj = json.loads(header)
    except Exception as e:
        return {"ok": False, "magic": "HGRAVU01", "header_len": hlen, "err": str(e)}
    bits = int(obj.get("bits", -1))
    group = int(obj.get("group_size", -1))
    shape = obj.get("shape")
    elements = int(obj.get("elements", 0))
    groups = int(obj.get("groups", 0)) if "groups" in obj else (elements + group - 1) // group if group else 0
    scale_bytes = int(obj.get("scale_bytes", groups * 2))
    code_bytes = int(obj.get("code_bytes", (elements * bits + 7) // 8 if bits > 0 else 0))
    bound = (1 << (bits - 1)) - 1 if bits >= 1 else None
    body = scale_bytes + code_bytes
    return {
        "ok": True,
        "magic": "HGRAVU01",
        "header_len": hlen,
        "header": obj,
        "bits": bits,
        "group_size": group,
        "shape": shape,
        "elements": elements,
        "groups": groups,
        "scale_bytes": scale_bytes,
        "code_bytes": code_bytes,
        "bound": bound,
        "body_bytes": body,
        "envelope": 12 + hlen + body,
        "catalog_nbytes": row["nbytes"],
        "representation": obj.get("representation"),
        "schema": obj.get("schema"),
    }


def ceil_div(a, b):
    return (a + b - 1) // b


def geometry_row(M, K, tpr, rpt, tg, w):
    needed = tpr * rpt
    if needed > tg:
        kind = "ILLEGAL"
    elif needed < tg:
        kind = "SLACK"
    else:
        kind = "TIGHT"
    tgs = ceil_div(M, rpt)
    threads = tgs * tg
    sgs = tgs * (tg // 32)
    sg_per_row = tpr / 32.0
    if tpr <= 32:
        reduce = "simd"
    else:
        reduce = f"simd+tg({int(sg_per_row)})"
    last_rows = M % rpt
    idle_row_threads = 0 if last_rows == 0 else (rpt - last_rows) * tpr
    tile = tpr * w
    last_tile = K % tile
    idle_col_lanes = 0 if last_tile == 0 else tpr - ceil_div(last_tile, w)
    # slack idle threads in every TG
    slack_idle = max(tg - needed, 0)
    steps = ceil_div(K, tile) if kind != "ILLEGAL" else None
    elems = steps * w if steps is not None else None
    tg_per_core = tgs / GPU_CORES
    return {
        "tpr": tpr,
        "rpt": rpt,
        "tg": tg,
        "w": w,
        "kind": kind,
        "tgs": tgs,
        "threads": threads,
        "sgs": sgs,
        "tg_per_core": tg_per_core,
        "saturates_60": tgs >= GPU_CORES,
        "reduce": reduce,
        "steps": steps,
        "elems_per_thread": elems,
        "tile": tile,
        "idle_last_row_threads": idle_row_threads,
        "idle_last_col_lanes": idle_col_lanes,
        "slack_idle_per_tg": slack_idle,
        "sg_per_tg": tg // 32,
    }


def main():
    cat = parse_catalog(CAT)
    rows = cat["rows"]
    print("CATALOG")
    print(
        json.dumps(
            {
                "version": cat["version"],
                "n_tensors": cat["n_tensors"],
                "n_segments": cat["n_segments"],
                "catalog_bytes": cat["catalog_bytes"],
            },
            indent=2,
        )
    )
    class_counts = Counter(r["class"] for r in rows)
    codec_counts = Counter(r["codec"] for r in rows)
    print("CLASS_COUNTS", dict(class_counts))
    print("CODEC_COUNTS", dict(codec_counts))

    # shape census
    shape_by_class = defaultdict(Counter)
    nbytes_by_class = defaultdict(list)
    elems_by_class = defaultdict(list)
    for r in rows:
        shape_by_class[r["class"]][tuple(r["shape"])] += 1
        nbytes_by_class[r["class"]].append(r["nbytes"])
        elems_by_class[r["class"]].append(r["elements"])

    print("\nSHAPE_CENSUS")
    for cls in sorted(shape_by_class):
        shapes = dict(shape_by_class[cls])
        nb = nbytes_by_class[cls]
        el = elems_by_class[cls]
        print(
            json.dumps(
                {
                    "class": cls,
                    "n": class_counts[cls],
                    "shapes": {str(k): v for k, v in shapes.items()},
                    "nbytes_min": min(nb),
                    "nbytes_max": max(nb),
                    "nbytes_sum": sum(nb),
                    "elems_min": min(el),
                    "elems_max": max(el),
                    "elems_sum": sum(el),
                }
            )
        )

    gemv_classes = [
        "mlp.gate",
        "mlp.up",
        "mlp.down",
        "dn.in_proj_qkv",
        "dn.in_proj_z",
        "dn.in_proj_a",
        "dn.in_proj_b",
        "dn.out",
        "gqa.q",
        "gqa.k",
        "gqa.v",
        "gqa.o",
        "lm_head",
    ]
    print("\nFUSED_PRESENT", {k: class_counts[k] for k in class_counts if "FUSED" in k})

    # peek one of each class + all GEMV headers for bits/shape
    print("\nHEADER_PEEKS")
    seen = set()
    peeks = {}
    bits_census = Counter()
    gemv_bits_by_class = defaultdict(Counter)
    header_fail = 0
    for r in rows:
        cls = r["class"]
        if cls == "other":
            continue
        peek = peek_hgravu(r)
        if not peek.get("ok"):
            header_fail += 1
            if cls not in seen:
                print("PEEK_FAIL", cls, r["name"], peek)
            continue
        bits_census[peek["bits"]] += 1
        if cls in gemv_classes or cls == "embed":
            gemv_bits_by_class[cls][peek["bits"]] += 1
        if cls not in seen:
            seen.add(cls)
            slim = {k: peek[k] for k in peek if k != "header"}
            slim["name"] = r["name"]
            peeks[cls] = peek
            print("PEEK", json.dumps(slim, default=str))
            print("  JSON", json.dumps(peek["header"], sort_keys=True))

    print("\nBITS_CENSUS_NON_OTHER", dict(bits_census))
    print("GEMV_BITS", {k: dict(v) for k, v in gemv_bits_by_class.items()})
    print("HEADER_FAIL", header_fail)

    # other tensor shape histogram
    others = [r for r in rows if r["class"] == "other"]
    other_shapes = Counter(tuple(r["shape"]) for r in others)
    print("\nOTHER_n", len(others))
    print("OTHER_SHAPES", {str(k): v for k, v in other_shapes.most_common()})
    print("OTHER_ELEMS_MAX", max(r["elements"] for r in others) if others else None)
    print("OTHER_ELEMS_LE_65536", sum(1 for r in others if r["elements"] <= 65536))

    # distinct GEMV shapes
    print("\nDISTINCT_GEMV")
    distinct = {}
    for cls in gemv_classes:
        rs = [r for r in rows if r["class"] == cls]
        assert rs, cls
        shape = tuple(rs[0]["shape"])
        assert all(tuple(r["shape"]) == shape for r in rs), cls
        peek = peeks[cls]
        M, K = shape
        bits = peek["bits"]
        distinct[cls] = {
            "class": cls,
            "n": len(rs),
            "M": M,
            "K": K,
            "bits": bits,
            "group": peek["group_size"],
            "bound": peek["bound"],
            "nbytes_each": rs[0]["nbytes"],
            "code_bytes": peek["code_bytes"],
            "scale_bytes": peek["scale_bytes"],
            "wbytes_row": (peek["code_bytes"] + peek["scale_bytes"]) // M,
            "physical_bpw": 8 * rs[0]["nbytes"] / (M * K),
            "body_bpw": 8 * (peek["code_bytes"] + peek["scale_bytes"]) / (M * K),
        }
        print(json.dumps(distinct[cls]))

    tprs = [32, 64, 128]
    rpts = [1, 2, 4]
    tgsizes = [128, 256]
    # also incumbent simd8 rpt=8
    extra = [(32, 8, 256)]

    print("\nGEOM_GRID")
    for cls, d in distinct.items():
        M, K, bits = d["M"], d["K"], d["bits"]
        w_inc = 8 if bits == 3 else 1
        print(f"# {cls} {M}x{K} bits={bits}")
        combos = [(tpr, rpt, tg) for tpr in tprs for rpt in rpts for tg in tgsizes]
        combos += extra
        for tpr, rpt, tg in combos:
            for w, wtag in ((8, "W8"), (1, "W1")):
                if w == 1 and bits == 3:
                    continue  # bits3 incumbent is W8; W1 is the bits4 path
                if w == 1 and (tpr, rpt, tg) not in ((32, 8, 256), (32, 1, 32)):
                    # only emit W1 for incumbent-like; W8 is the geo_tpr64-class
                    if not (tpr == 32 and rpt == 8 and tg == 256):
                        continue
                g = geometry_row(M, K, tpr, rpt, tg, w)
                g["class"] = cls
                g["wtag"] = wtag
                g["n"] = d["n"]
                g["tgs_token"] = g["tgs"] * d["n"]
                print(json.dumps(g))

    print("\nTOKEN_TG_SUMS")
    # for each tight geometry + incumbent
    plans = [
        ("incumbent_simd8", 32, 8, 256, "incumbent"),
        ("tpr32_rpt1_tg32_ILLEGAL_SET", 32, 1, 32, "note"),
        ("tpr32_rpt4_tg128", 32, 4, 128, "tight"),
        ("tpr64_rpt2_tg128_G0", 64, 2, 128, "tight"),
        ("tpr128_rpt1_tg128", 128, 1, 128, "tight"),
        ("tpr64_rpt4_tg256", 64, 4, 256, "tight"),
        ("tpr128_rpt2_tg256", 128, 2, 256, "tight"),
        ("tpr32_rpt2_tg128_SLACK", 32, 2, 128, "slack"),
        ("tpr64_rpt1_tg128_SLACK", 64, 1, 128, "slack"),
        ("tpr32_rpt1_tg128_SLACK", 32, 1, 128, "slack"),
        ("tpr128_rpt1_tg256_SLACK", 128, 1, 256, "slack"),
        ("tpr64_rpt2_tg256_SLACK", 64, 2, 256, "slack"),
        ("tpr32_rpt4_tg256_SLACK", 32, 4, 256, "slack"),
        ("tpr128_rpt4_tg256_ILLEGAL", 128, 4, 256, "illegal"),
        ("tpr64_rpt4_tg128_ILLEGAL", 64, 4, 128, "illegal"),
        ("tpr128_rpt2_tg128_ILLEGAL", 128, 2, 128, "illegal"),
    ]
    for name, tpr, rpt, tg, tag in plans:
        total_tgs = 0
        total_threads = 0
        per = []
        legal = tpr * rpt <= tg
        for cls, d in distinct.items():
            g = geometry_row(d["M"], d["K"], tpr, rpt, tg, 8)
            total_tgs += g["tgs"] * d["n"]
            total_threads += g["threads"] * d["n"]
            per.append(
                {
                    "class": cls,
                    "tgs": g["tgs"],
                    "times": d["n"],
                    "tgs_token": g["tgs"] * d["n"],
                    "tg_per_core": round(g["tg_per_core"], 4),
                    "kind": g["kind"],
                    "reduce": g["reduce"],
                    "steps": g["steps"],
                    "idle_row": g["idle_last_row_threads"],
                    "idle_col": g["idle_last_col_lanes"],
                    "saturates": g["saturates_60"],
                }
            )
        print(
            json.dumps(
                {
                    "plan": name,
                    "tag": tag,
                    "tpr": tpr,
                    "rpt": rpt,
                    "tg": tg,
                    "legal": legal,
                    "token_tgs": total_tgs,
                    "token_threads": total_threads,
                    "token_tg_per_core_if_all_serial": total_tgs / GPU_CORES,
                    "per": per,
                }
            )
        )

    # AI at RPT
    print("\nAI")
    for cls, d in distinct.items():
        K = d["K"]
        Wb = d["wbytes_row"]
        for rpt in (1, 2, 4, 8):
            x_tg = 4 * K / rpt
            bytes_out = Wb + x_tg + 4
            ai = 2 * K / bytes_out
            print(
                json.dumps(
                    {
                        "class": cls,
                        "rpt": rpt,
                        "Wbytes": Wb,
                        "x_tg": x_tg,
                        "bytes_out": bytes_out,
                        "AI": ai,
                    }
                )
            )

    # G0 fused comparison TG
    print("\nG0_FUSED_TPR64")
    g0 = [
        ("in_proj_qkvz", 16384, 5120, 48),
        ("in_proj_ba", 96, 5120, 48),
        ("dn.out", 5120, 6144, 48),
        ("gqa.q", 12288, 5120, 16),
        ("gqa.k", 1024, 5120, 16),
        ("gqa.v", 1024, 5120, 16),
        ("gqa.o", 5120, 6144, 16),
        ("mlp.gate", 17408, 5120, 64),
        ("mlp.up", 17408, 5120, 64),
        ("mlp.down", 5120, 17408, 64),
        ("lm_head", 248320, 5120, 1),
    ]
    tot = 0
    for name, M, K, n in g0:
        tgs = ceil_div(M, 2)
        tot += tgs * n
        print(json.dumps({"class": name, "M": M, "tgs": tgs, "n": n, "tgs_token": tgs * n, "tg_per_core": tgs / 60}))
    print("G0_TOKEN_GEMV_TGS", tot)

    # dispatch accounting
    print("\nDISPATCH")
    print(
        json.dumps(
            {
                "g0": 1 + 64 * 15 + 3,
                "candidate_split": 1 + 48 * 19 + 16 * 15 + 3,
                "gemv_bits3": 192,
                "gemv_bits4": 48 * 5 + 16 * 4 + 1,
                "fuse_dispatches": 48 * 2,
                "delta": (1 + 48 * 19 + 16 * 15 + 3) - (1 + 64 * 15 + 3),
            }
        )
    )

    # K divisibility
    print("\nK_DIV")
    for K in (5120, 6144, 17408, 1024):
        rec = {"K": K}
        for tpr in (32, 64, 128):
            for w in (1, 8):
                rec[f"tpr{tpr}_w{w}"] = K % (tpr * w) == 0
                rec[f"steps_tpr{tpr}_w{w}"] = K // (tpr * w)
        print(json.dumps(rec))

    payload = sum(r["nbytes"] for r in rows)
    print("\nPAYLOAD_SUM", payload)
    print("BPW_G0_DEF", 8 * payload / 26895998464)


if __name__ == "__main__":
    main()
