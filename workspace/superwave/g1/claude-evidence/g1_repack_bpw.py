#!/usr/bin/env python3
"""Exact complete-BPW calculator for G1 packing recipes.

Formulas are the on-disk container identities from hawking-core, not
nominal bit-widths. Catalog tax uses the HQ38M20 layout.
"""
from __future__ import annotations

import json
from collections import defaultdict

N = 26_895_998_464
E_MLP = 17_112_760_320
E_ATTN = 7_237_795_840
E_TAB = 2_542_796_800
E_SMALL = 2_645_504
assert N == E_MLP + E_ATTN + E_TAB + E_SMALL

# G0 measured
G0_BYTES = 14_297_694_680
G0_BPW = 8 * G0_BYTES / N

# mixed siblings (MEASURED, cited)
MIXED_Q3MLP_BYTES = 12_149_632_429
MIXED_Q3MLP_BPW = 3.6138647373176767
MIXED_2P0_ART_BPW = 2.0855934079220506
MIXED_SUB15_BPW = 1.2910781930062503

SCHEMA_U = "hawking.gravity.uniform_group.v1"
SCHEMA_B = "hawking.gravity.binary_sign_scale.v1"


def hq30uq4_bytes(n: int, rank: int = 2) -> int:
    """HQ30UQ4 g64: 32 + 4*rank header + 2 B scale/group + 32 B codes/group."""
    assert n % 64 == 0, n
    groups = n // 64
    return 32 + 4 * rank + groups * 2 + groups * 32


def f32v2_bytes(n: int) -> int:
    return 8 + 4 * n


def packed_code_bytes(n: int, bits: int, group: int) -> int:
    groups = (n + group - 1) // group
    padded = groups * group
    return (padded * bits + 7) // 8


def hgravu_json(rows: int, cols: int, bits: int, group: int = 64) -> bytes:
    n = rows * cols
    groups = (n + group - 1) // group
    header = {
        "schema": SCHEMA_U,
        "representation": f"uniform_q{bits}_group_scale",
        "shape": [rows, cols],
        "elements": n,
        "bits": bits,
        "group_size": group,
        "groups": groups,
        "scale_bytes": groups * 2,
        "code_bytes": packed_code_bytes(n, bits, group),
    }
    return json.dumps(header, separators=(",", ":")).encode()


def hgravu_bytes(rows: int, cols: int, bits: int, group: int = 64) -> int:
    n = rows * cols
    groups = (n + group - 1) // group
    body = groups * 2 + packed_code_bytes(n, bits, group)
    return 8 + 4 + len(hgravu_json(rows, cols, bits, group)) + body


def hgravb_json(rows: int, cols: int, group: int = 128) -> bytes:
    n = rows * cols
    groups = rows * (cols // group)
    header = {
        "schema": SCHEMA_B,
        "representation": "binary_sign_scale",
        "shape": [rows, cols],
        "elements": n,
        "group_size": group,
        "groups": groups,
        "scale_bytes": groups * 2,
        "sign_bytes": n // 8,
    }
    return json.dumps(header, separators=(",", ":")).encode()


def hgravb_bytes(rows: int, cols: int, group: int = 128) -> int:
    assert cols % group == 0
    n = rows * cols
    groups = rows * (cols // group)
    body = groups * 2 + n // 8
    return 8 + 4 + len(hgravb_json(rows, cols, group)) + body


# --- shapes (fused G0 catalog) ---
SHAPES = {
    "mlp.gate_proj": (17408, 5120),
    "mlp.up_proj": (17408, 5120),
    "mlp.down_proj": (5120, 17408),
    "dn.in_proj_qkvz": (16384, 5120),
    "dn.in_proj_ba": (96, 5120),
    "dn.out_proj": (5120, 6144),
    "gqa.q_proj": (12288, 5120),
    "gqa.k_proj": (1024, 5120),
    "gqa.v_proj": (1024, 5120),
    "gqa.o_proj": (5120, 6144),
    "embed": (248320, 5120),
    "lm_head": (248320, 5120),
}

SMALL = {
    "dn.conv1d": (48, 10240 * 4),
    "input_layernorm": (64, 5120),
    "post_attention_layernorm": (64, 5120),
    "dn.norm": (48, 128),
    "final_norm": (1, 5120),
    "gqa.q_norm": (16, 256),
    "gqa.k_norm": (16, 256),
    "dn.A_log": (48, 48),
    "dn.dt_bias": (48, 48),
}


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def payload(kind: str, rows: int, cols: int) -> int:
    if kind == "UQ4":
        return hq30uq4_bytes(rows * cols, rank=2)
    if kind == "HU2":
        return hgravu_bytes(rows, cols, 2)
    if kind == "HU3":
        return hgravu_bytes(rows, cols, 3)
    if kind == "HU4":
        return hgravu_bytes(rows, cols, 4)
    if kind == "HB1":
        return hgravb_bytes(rows, cols, 128)
    if kind == "F32":
        return f32v2_bytes(rows * cols)
    raise ValueError(kind)


def name_for(layer: int | None, cls: str) -> str:
    if cls == "embed":
        return "language_model.model.embed_tokens.weight"
    if cls == "lm_head":
        return "language_model.lm_head.weight"
    if cls == "final_norm":
        return "language_model.model.norm.weight"
    suffix = {
        "mlp.gate_proj": "mlp.gate_proj.weight",
        "mlp.up_proj": "mlp.up_proj.weight",
        "mlp.down_proj": "mlp.down_proj.weight",
        "dn.in_proj_qkvz": "linear_attn.in_proj_qkvz.weight",
        "dn.in_proj_ba": "linear_attn.in_proj_ba.weight",
        "dn.out_proj": "linear_attn.out_proj.weight",
        "gqa.q_proj": "self_attn.q_proj.weight",
        "gqa.k_proj": "self_attn.k_proj.weight",
        "gqa.v_proj": "self_attn.v_proj.weight",
        "gqa.o_proj": "self_attn.o_proj.weight",
        "dn.conv1d": "linear_attn.conv1d.weight",
        "input_layernorm": "input_layernorm.weight",
        "post_attention_layernorm": "post_attention_layernorm.weight",
        "dn.norm": "linear_attn.norm.weight",
        "gqa.q_norm": "self_attn.q_norm.weight",
        "gqa.k_norm": "self_attn.k_norm.weight",
        "dn.A_log": "linear_attn.A_log",
        "dn.dt_bias": "linear_attn.dt_bias",
    }[cls]
    return f"language_model.model.layers.{layer}.{suffix}"


def catalog_bytes(names: list[str], n_segments: int = 66) -> int:
    # HQ38M20: 32-byte prefix + per-segment (44 + name_len) + 128*n + name blob
    prefix = 32
    # segment names seg_00.bin .. 
    seg = 0
    for i in range(n_segments):
        fname = f"seg_{i:02d}.bin"
        seg += 44 + len(fname)
    table = 128 * len(names)
    blob = sum(len(n.encode()) for n in names)
    return prefix + seg + table + blob


def island_row_bytes(in_dim: int) -> int:
    return f32v2_bytes(in_dim)


class Acc:
    def __init__(self):
        self.bytes = 0
        self.elems = 0
        self.names: list[str] = []
        self.by_class = defaultdict(lambda: {"bytes": 0, "elems": 0, "n": 0, "kinds": defaultdict(int)})
        self.rows = []

    def add(self, cls: str, layer: int | None, kind: str, rows: int, cols: int, extra: int = 0):
        n = rows * cols
        b = payload(kind, rows, cols) + extra
        self.bytes += b
        self.elems += n
        nm = name_for(layer, cls)
        self.names.append(nm)
        slot = self.by_class[cls]
        slot["bytes"] += b
        slot["elems"] += n
        slot["n"] += 1
        slot["kinds"][kind] += 1
        self.rows.append((layer, cls, kind, rows, cols, n, b, 8 * b / n))


def gqa_layers():
    return [i for i in range(64) if is_gqa(i)]


def dn_layers():
    return [i for i in range(64) if not is_gqa(i)]


def add_small(acc: Acc):
    for cls, (count, width) in SMALL.items():
        if cls == "final_norm":
            acc.add(cls, None, "F32", 1, width)
        elif cls.startswith("dn.") or cls.startswith("gqa."):
            layers = dn_layers() if cls.startswith("dn.") else gqa_layers()
            assert len(layers) == count
            for L in layers:
                acc.add(cls, L, "F32", 1, width)
        else:
            for L in range(64):
                acc.add(cls, L, "F32", 1, width)


def add_islands(acc: Acc):
    """Double-store output row 3994 as f32v2 sidecar (does not shrink GEMV)."""
    extra_names = []
    extra_bytes = 0
    # 64 down + 48 lin_o + 16 o
    for L in range(64):
        extra_bytes += island_row_bytes(17408)
        extra_names.append(name_for(L, "mlp.down_proj") + "::island_row_3994")
    for L in dn_layers():
        extra_bytes += island_row_bytes(6144)
        extra_names.append(name_for(L, "dn.out_proj") + "::island_row_3994")
    for L in gqa_layers():
        extra_bytes += island_row_bytes(6144)
        extra_names.append(name_for(L, "gqa.o_proj") + "::island_row_3994")
    acc.bytes += extra_bytes
    acc.names.extend(extra_names)
    acc.by_class["island_row_3994"]["bytes"] += extra_bytes
    acc.by_class["island_row_3994"]["elems"] += 0
    acc.by_class["island_row_3994"]["n"] += 128
    return extra_bytes


def summarize(name: str, acc: Acc, islands: bool) -> dict:
    if islands:
        isle = add_islands(acc)
    else:
        isle = 0
    cat = catalog_bytes(acc.names)
    tensor_bpw = 8 * acc.bytes / N
    art_bpw = 8 * (acc.bytes + cat) / N
    by = {}
    for cls, s in acc.by_class.items():
        by[cls] = {
            "n": s["n"],
            "elems": s["elems"],
            "bytes": s["bytes"],
            "bpw": (8 * s["bytes"] / s["elems"]) if s["elems"] else None,
            "kinds": dict(s["kinds"]),
        }
    # class rollups
    mlp_e = sum(acc.by_class[c]["elems"] for c in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    mlp_b = sum(acc.by_class[c]["bytes"] for c in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    attn_cls = [
        "dn.in_proj_qkvz",
        "dn.in_proj_ba",
        "dn.out_proj",
        "gqa.q_proj",
        "gqa.k_proj",
        "gqa.v_proj",
        "gqa.o_proj",
    ]
    attn_e = sum(acc.by_class[c]["elems"] for c in attn_cls)
    attn_b = sum(acc.by_class[c]["bytes"] for c in attn_cls)
    tab_e = acc.by_class["embed"]["elems"] + acc.by_class["lm_head"]["elems"]
    tab_b = acc.by_class["embed"]["bytes"] + acc.by_class["lm_head"]["bytes"]
    sm_e = sum(acc.by_class[c]["elems"] for c in SMALL)
    sm_b = sum(acc.by_class[c]["bytes"] for c in SMALL)
    return {
        "name": name,
        "tensor_payload_bytes": acc.bytes,
        "island_sidecar_bytes": isle,
        "catalog_bytes": cat,
        "artifact_bytes": acc.bytes + cat,
        "elems": acc.elems,
        "n_tensors_catalog": len(acc.names),
        "complete_physical_bpw": tensor_bpw,
        "artifact_complete_bpw": art_bpw,
        "mlp_bpw": 8 * mlp_b / mlp_e,
        "attn_bpw": 8 * attn_b / attn_e,
        "tab_bpw": 8 * tab_b / tab_e,
        "small_bpw": 8 * sm_b / sm_e,
        "mlp_bytes": mlp_b,
        "attn_bytes": attn_b,
        "tab_bytes": tab_b,
        "small_bytes": sm_b,
        "by_class": by,
        "vs_1_5": tensor_bpw - 1.5,
        "vs_g0": tensor_bpw - G0_BPW,
    }


# ---------- G0 check ----------
def build_g0():
    acc = Acc()
    for L in range(64):
        acc.add("mlp.gate_proj", L, "UQ4", *SHAPES["mlp.gate_proj"])
        acc.add("mlp.up_proj", L, "UQ4", *SHAPES["mlp.up_proj"])
        acc.add("mlp.down_proj", L, "UQ4", *SHAPES["mlp.down_proj"])
        if is_gqa(L):
            for c in ("gqa.q_proj", "gqa.k_proj", "gqa.v_proj", "gqa.o_proj"):
                acc.add(c, L, "UQ4", *SHAPES[c])
        else:
            for c in ("dn.in_proj_qkvz", "dn.in_proj_ba", "dn.out_proj"):
                acc.add(c, L, "UQ4", *SHAPES[c])
    acc.add("embed", None, "UQ4", *SHAPES["embed"])
    acc.add("lm_head", None, "UQ4", *SHAPES["lm_head"])
    add_small(acc)
    return acc


# ---------- Conservative: Q3 MLP + Q4 rest ----------
def build_C():
    acc = Acc()
    for L in range(64):
        acc.add("mlp.gate_proj", L, "HU3", *SHAPES["mlp.gate_proj"])
        acc.add("mlp.up_proj", L, "HU3", *SHAPES["mlp.up_proj"])
        acc.add("mlp.down_proj", L, "HU3", *SHAPES["mlp.down_proj"])
        if is_gqa(L):
            for c in ("gqa.q_proj", "gqa.k_proj", "gqa.v_proj", "gqa.o_proj"):
                acc.add(c, L, "UQ4", *SHAPES[c])
        else:
            for c in ("dn.in_proj_qkvz", "dn.in_proj_ba", "dn.out_proj"):
                acc.add(c, L, "UQ4", *SHAPES[c])
    acc.add("embed", None, "UQ4", *SHAPES["embed"])
    acc.add("lm_head", None, "UQ4", *SHAPES["lm_head"])
    add_small(acc)
    return acc


# ---------- Recommended: C + late down Q4 + island ----------
LATE_DOWN = set(range(47, 64))  # 17 layers, write-gain evidence L47/L63


def build_R():
    acc = Acc()
    for L in range(64):
        acc.add("mlp.gate_proj", L, "HU3", *SHAPES["mlp.gate_proj"])
        acc.add("mlp.up_proj", L, "HU3", *SHAPES["mlp.up_proj"])
        down_kind = "UQ4" if L in LATE_DOWN else "HU3"
        acc.add("mlp.down_proj", L, down_kind, *SHAPES["mlp.down_proj"])
        if is_gqa(L):
            for c in ("gqa.q_proj", "gqa.k_proj", "gqa.v_proj", "gqa.o_proj"):
                acc.add(c, L, "UQ4", *SHAPES[c])
        else:
            for c in ("dn.in_proj_qkvz", "dn.in_proj_ba", "dn.out_proj"):
                acc.add(c, L, "UQ4", *SHAPES[c])
    acc.add("embed", None, "UQ4", *SHAPES["embed"])
    acc.add("lm_head", None, "UQ4", *SHAPES["lm_head"])
    add_small(acc)
    return acc


# ---------- Aggressive candidates ----------
def mlp_kind_A(L: int, variant: str) -> str:
    if variant == "A1":
        if L <= 31:
            return "HB1"
        if L <= 51:
            return "HU2"
        return "HU3"
    if variant == "A2":
        # all MLP equal, more binary to pay for Q3 writes
        if L <= 51:
            return "HB1"
        return "HU3"
    if variant == "A3":
        return "HB1" if L <= 55 else "HU3"
    raise ValueError(variant)


def attn_kind_A(L: int, cls: str, variant: str) -> str:
    # Never leave attention at Q4. Never uniquely starve down (MLP handled elsewhere).
    write = cls in ("dn.out_proj", "gqa.o_proj")
    kv = cls in ("gqa.k_proj", "gqa.v_proj")
    ba = cls == "dn.in_proj_ba"
    if variant in ("A1", "A2", "A3"):
        if write:
            return "UQ4" if L >= 52 else "HU3"
        if kv:
            return "UQ4" if L >= 52 else "HU3"
        if ba:
            return "UQ4" if L >= 52 else "HU3"
        # in_proj / q
        if L <= 31:
            return "HB1"
        if L <= 51:
            return "HU2"
        return "HU3"
    raise ValueError(variant)


def build_A(variant: str, embed="HU2", lm="HU3"):
    acc = Acc()
    for L in range(64):
        mk = mlp_kind_A(L, variant)
        acc.add("mlp.gate_proj", L, mk, *SHAPES["mlp.gate_proj"])
        acc.add("mlp.up_proj", L, mk, *SHAPES["mlp.up_proj"])
        acc.add("mlp.down_proj", L, mk, *SHAPES["mlp.down_proj"])
        if is_gqa(L):
            for c in ("gqa.q_proj", "gqa.k_proj", "gqa.v_proj", "gqa.o_proj"):
                acc.add(c, L, attn_kind_A(L, c, variant), *SHAPES[c])
        else:
            for c in ("dn.in_proj_qkvz", "dn.in_proj_ba", "dn.out_proj"):
                acc.add(c, L, attn_kind_A(L, c, variant), *SHAPES[c])
    acc.add("embed", None, embed, *SHAPES["embed"])
    acc.add("lm_head", None, lm, *SHAPES["lm_head"])
    add_small(acc)
    return acc


def dump(s: dict):
    print(f"\n==== {s['name']} ====")
    print(f"tensor_payload_bytes     {s['tensor_payload_bytes']}")
    print(f"island_sidecar_bytes     {s['island_sidecar_bytes']}")
    print(f"catalog_bytes            {s['catalog_bytes']}")
    print(f"artifact_bytes           {s['artifact_bytes']}")
    print(f"n_catalog_rows           {s['n_tensors_catalog']}")
    print(f"elems                    {s['elems']}")
    print(f"complete_physical_bpw    {s['complete_physical_bpw']:.16f}")
    print(f"artifact_complete_bpw    {s['artifact_complete_bpw']:.16f}")
    print(f"mlp_bpw                  {s['mlp_bpw']:.16f}")
    print(f"attn_bpw                 {s['attn_bpw']:.16f}")
    print(f"tab_bpw                  {s['tab_bpw']:.16f}")
    print(f"small_bpw                {s['small_bpw']:.16f}")
    print(f"vs_1.5                   {s['vs_1_5']:+.16f}")
    print(f"vs_g0                    {s['vs_g0']:+.16f}")
    print("class breakdown:")
    for cls in sorted(s["by_class"]):
        c = s["by_class"][cls]
        bpw = f"{c['bpw']:.10f}" if c["bpw"] is not None else "n/a"
        print(f"  {cls:28s} n={c['n']:3d} elems={c['elems']:12d} bytes={c['bytes']:12d} bpw={bpw} kinds={c['kinds']}")


def header_samples():
    print("=== header samples ===")
    for bits in (2, 3, 4):
        for name, (r, c) in [
            ("gate", (17408, 5120)),
            ("down", (5120, 17408)),
            ("qkvz", (16384, 5120)),
            ("out", (5120, 6144)),
            ("ba", (96, 5120)),
            ("q", (12288, 5120)),
            ("k", (1024, 5120)),
            ("embed", (248320, 5120)),
        ]:
            jb = hgravu_json(r, c, bits)
            print(f"HU{bits} {name:6s} json={len(jb):3d} total={hgravu_bytes(r,c,bits)}")
    for name, (r, c) in [
        ("gate", (17408, 5120)),
        ("down", (5120, 17408)),
        ("qkvz", (16384, 5120)),
        ("out", (5120, 6144)),
        ("ba", (96, 5120)),
        ("q", (12288, 5120)),
        ("k", (1024, 5120)),
    ]:
        jb = hgravb_json(r, c)
        print(f"HB1 {name:6s} json={len(jb):3d} total={hgravb_bytes(r,c)} body_check_gate_ref=12533760")
    # measured binary gate
    print("HB1 gate vs MEASURED 12534021:", hgravb_bytes(17408, 5120), "delta", hgravb_bytes(17408, 5120) - 12534021)
    print("UQ4 gate", hq30uq4_bytes(17408 * 5120), "G0 class /64", 3_030_387_200 // 64)


def layer_table(builder, kind_fn_mlp, kind_fn_attn, path):
    lines = [
        "layer,mixer,mlp.gate,mlp.up,mlp.down,dn.qkvz,dn.ba,dn.out,gqa.q,gqa.k,gqa.v,gqa.o"
    ]
    for L in range(64):
        mix = "gqa" if is_gqa(L) else "delta_net"
        mg = kind_fn_mlp(L)
        if is_gqa(L):
            row = [str(L), mix, mg, mg, mg, "", "", "",
                   kind_fn_attn(L, "gqa.q_proj"),
                   kind_fn_attn(L, "gqa.k_proj"),
                   kind_fn_attn(L, "gqa.v_proj"),
                   kind_fn_attn(L, "gqa.o_proj")]
        else:
            row = [str(L), mix, mg, mg, mg,
                   kind_fn_attn(L, "dn.in_proj_qkvz"),
                   kind_fn_attn(L, "dn.in_proj_ba"),
                   kind_fn_attn(L, "dn.out_proj"),
                   "", "", "", ""]
        lines.append(",".join(row))
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    header_samples()

    g0 = summarize("G0_recompute", build_g0(), islands=False)
    dump(g0)
    print("G0 MEASURED bytes", G0_BYTES, "delta", g0["tensor_payload_bytes"] - G0_BYTES)
    print("G0 MEASURED bpw  ", f"{G0_BPW:.16f}")
    print("match", g0["tensor_payload_bytes"] == G0_BYTES)

    c = summarize("G1-C conservative", build_C(), islands=False)
    dump(c)
    print("mixed-q3mlp MEASURED bytes", MIXED_Q3MLP_BYTES, "delta", c["tensor_payload_bytes"] - MIXED_Q3MLP_BYTES)
    print("mixed-q3mlp MEASURED bpw  ", MIXED_Q3MLP_BPW)

    r = summarize("G1-R recommended", build_R(), islands=True)
    dump(r)

    for var in ("A1", "A2", "A3"):
        for emb, lm in (("HU2", "HU3"), ("HU2", "HU2"), ("HU3", "HU3")):
            a = summarize(f"G1-A {var} embed={emb} lm={lm}", build_A(var, emb, lm), islands=True)
            dump(a)

    # emit JSON
    out = {
        "g0": g0,
        "C": c,
        "R": r,
        "A1_HU2_HU3": summarize("A1", build_A("A1"), islands=True),
        "A2_HU2_HU3": summarize("A2", build_A("A2"), islands=True),
        "A3_HU2_HU3": summarize("A3", build_A("A3"), islands=True),
        "A2_HU2_HU2": summarize("A2e2l2", build_A("A2", "HU2", "HU2"), islands=True),
    }
    # strip by_class huge? keep it
    with open("/tmp/g1_repack_bpw.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote /tmp/g1_repack_bpw.json")

    layer_table(
        None,
        lambda L: "HU3",
        lambda L, c: "UQ4",
        "/tmp/g1_recipe_C_layers.csv",
    )
    layer_table(
        None,
        lambda L: "UQ4" if L in LATE_DOWN and False else "HU3",  # placeholder
        lambda L, c: "UQ4",
        "/tmp/g1_recipe_R_layers.csv",
    )
    # fix R table
    open("/tmp/g1_recipe_R_layers.csv", "w").write("")
    lines = ["layer,mixer,mlp.gate,mlp.up,mlp.down,dn.qkvz,dn.ba,dn.out,gqa.q,gqa.k,gqa.v,gqa.o"]
    for L in range(64):
        mix = "gqa" if is_gqa(L) else "delta_net"
        down = "UQ4" if L in LATE_DOWN else "HU3"
        if is_gqa(L):
            lines.append(f"{L},{mix},HU3,HU3,{down},,,,UQ4,UQ4,UQ4,UQ4")
        else:
            lines.append(f"{L},{mix},HU3,HU3,{down},UQ4,UQ4,UQ4,,,,")
    open("/tmp/g1_recipe_R_layers.csv", "w").write("\n".join(lines) + "\n")

    lines = ["layer,mixer,mlp.gate,mlp.up,mlp.down,dn.qkvz,dn.ba,dn.out,gqa.q,gqa.k,gqa.v,gqa.o"]
    for L in range(64):
        mix = "gqa" if is_gqa(L) else "delta_net"
        mk = mlp_kind_A(L, "A2")
        if is_gqa(L):
            lines.append(
                f"{L},{mix},{mk},{mk},{mk},,,,"
                f"{attn_kind_A(L,'gqa.q_proj','A2')},"
                f"{attn_kind_A(L,'gqa.k_proj','A2')},"
                f"{attn_kind_A(L,'gqa.v_proj','A2')},"
                f"{attn_kind_A(L,'gqa.o_proj','A2')}"
            )
        else:
            lines.append(
                f"{L},{mix},{mk},{mk},{mk},"
                f"{attn_kind_A(L,'dn.in_proj_qkvz','A2')},"
                f"{attn_kind_A(L,'dn.in_proj_ba','A2')},"
                f"{attn_kind_A(L,'dn.out_proj','A2')},,,,"
            )
    open("/tmp/g1_recipe_A_layers.csv", "w").write("\n".join(lines) + "\n")
    print("wrote layer CSVs")
