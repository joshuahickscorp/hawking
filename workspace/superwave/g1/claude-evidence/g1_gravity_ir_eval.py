#!/usr/bin/env python3
"""Gravity IR v0 cost evaluator.

Design authority: workspace/superwave/g1/g1-gravity-ir.md
This script is the cost model. It does not pack, generate, or touch the resident.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy

SOURCE_N = 26_895_998_464
G0_BYTES = 14_297_694_680
G0_BPW = 4.252735126866492
G0_S0 = 0.4078534106896186
MSE_BYTES = 13_897_447_220
MSE_BPW = 4.133684715546539
MSE_S0 = 0.47754866875899726
MSE_CATALOG = 224_763
G0_MANIFEST_SHA = "d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df"
LIVE_TOKEN_NS = 39_326_090
LIVE_TPS = 25.4284
LEDGER_TOKEN_NS = 35_227_917
ACTIVE_BUDGET_RECEIPT = 13_622_264_240
EMBED_ROW_BODY = 2720  # one HQ30UQ4 row, codes+scales, no header
HGRAVS_DOWN_BYTES = 93_847_197  # mixed-2p0 64x down MEASURED
ISLAND_WRITE_BYTES = 6_030_336  # 128 f32v2 rows, G1-R
SEQ_DEFAULT = 19

# --- containers ----------------------------------------------------------------


def prod(shape):
    p = 1
    for x in shape:
        p *= x
    return p


def hq30_payload(n, bits=4, group=64, rank=2):
    """HQ30UQ4-family payload. Codes packed bits*g/8 per group, f16 scale/group."""
    header = 32 + 4 * rank
    groups = (n + group - 1) // group
    code = groups * math.ceil(bits * group / 8)
    scale = groups * 2
    return {
        "header_bytes": header,
        "code_bytes": code,
        "scale_bytes": scale,
        "stored_bytes": header + code + scale,
    }


def f32v2_payload(n):
    return {
        "header_bytes": 8,
        "code_bytes": 4 * n,
        "scale_bytes": 0,
        "stored_bytes": 8 + 4 * n,
    }


def hgravu01_payload(n, bits, group, stored_bytes):
    """Invert JSON envelope from a MEASURED stored_bytes (12 + json + scale + code)."""
    groups = n // group
    scale = groups * 2
    code = math.ceil(bits * n / 8)
    envelope = stored_bytes - scale - code
    json_len = envelope - 12
    return {
        "header_bytes": envelope,
        "json_len": json_len,
        "code_bytes": code,
        "scale_bytes": scale,
        "stored_bytes": stored_bytes,
    }


# --- decode / kernel records ---------------------------------------------------

KERNEL = {
    "geo64": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "geo128": "qwen_uniform_q4_group128_matvec_geo_tpr64_tg128",
    "embed_q4": "qwen_uniform_q4_embedding_lookup",
    "embed_u01": "qwen38_hgravu_embedding_lookup",
    "simd": "q80_hgravs01_factor_matvec_simd",
    "simd3": "q80_hgravs01_factor_matvec_simd3",
    "f32": "f32_buffer",
    "row_dot": "qwen38_exact_row_dot_f32",
    "factor": "q80_hgravs01_factor_matvec_simd3",
}


def decode_quant(shape, bits, group, kernel, alphabet, count=1):
    rows, cols = shape[0], shape[1]
    n = rows * cols
    groups = n // group
    code_b = groups * math.ceil(bits * group / 8)
    scale_b = groups * 2
    if kernel == KERNEL["geo64"] or kernel == KERNEL["geo128"]:
        scale_loads_per_out = 0.125  # 1 half / 8 codes, W=8 TILE=512
    elif kernel in (KERNEL["simd"], KERNEL["simd3"]):
        scale_loads_per_out = 1.0  # 1-wide incumbent loop
    else:
        scale_loads_per_out = None
    tg = 128
    tpr = 64
    tgs = math.ceil(rows / 2)
    return {
        "code_bytes_touched": code_b * count,
        "scale_bytes_touched": scale_b * count,
        "unpack_ops": n * count,
        "macs": n * count,
        "scale_loads_per_out": scale_loads_per_out,
        "materialize_w": False,
        "launch": {
            "tg": tg,
            "threads_per_row": tpr,
            "threadgroups": tgs * count,
            "grid": tgs * tg,
        }
        if kernel.startswith("qwen_uniform_q4_group")
        else None,
        "alphabet": alphabet,
        "kernel": kernel,
    }


# --- node constructors ---------------------------------------------------------


def quant_node(
    id_,
    count,
    shape,
    container,
    bits,
    group,
    scale_rule,
    alphabet,
    kernel,
    role,
    stored_override=None,
    classification="ESSENTIAL",
    active="full",
):
    n = prod(shape)
    if container == "HQ30UQ4":
        pay = hq30_payload(n, bits, group)
        stored_one = pay["stored_bytes"]
    elif container == "f32v2":
        pay = f32v2_payload(n)
        stored_one = pay["stored_bytes"]
    elif container == "HGRAVU01":
        if stored_override is None:
            raise ValueError(f"{id_}: HGRAVU01 needs MEASURED stored_override")
        stored_one = stored_override // count
        pay = hgravu01_payload(n, bits, group, stored_one)
    else:
        raise ValueError(container)
    stored = stored_one * count if stored_override is None else stored_override
    if abs(stored - stored_one * count) > 0 and stored_override is not None:
        # allow class-total override (JSON length may vary 1-2 B)
        stored = stored_override
    if active == "full":
        active_b = stored
    elif active == "embed_row":
        # one gathered row: codes + scales, no header
        groups_row = shape[1] // group
        active_b = groups_row * math.ceil(bits * group / 8) + groups_row * 2
    elif active == "none":
        active_b = 0
    else:
        raise ValueError(active)
    node = {
        "id": id_,
        "kind": "DenseTensor" if container == "f32v2" else "QuantTensor",
        "count": count,
        "shape": list(shape),
        "source_elements": n * count,
        "container": container,
        "bits": bits if container != "f32v2" else 32,
        "group": group if container != "f32v2" else None,
        "scale_rule": scale_rule,
        "alphabet": alphabet,
        "header_bytes": pay["header_bytes"] * count
        if stored_override is None
        else pay["header_bytes"] * count,
        "code_bytes": pay["code_bytes"] * count,
        "scale_bytes": pay["scale_bytes"] * count,
        "stored_bytes": stored,
        "active_bytes_per_token": active_b,
        "kernel": kernel,
        "kernel_exists": kernel != KERNEL["row_dot"],
        "materialize_w": False,
        "class": classification,
        "role": role,
    }
    if container == "HGRAVU01":
        node["json_len_one"] = pay["json_len"]
    if node["kind"] == "QuantTensor":
        node["decode"] = decode_quant(shape, bits, group, kernel, alphabet, count)
    else:
        node["decode"] = {
            "code_bytes_touched": 0,
            "scale_bytes_touched": 0,
            "unpack_ops": 0,
            "macs": n * count,
            "scale_loads_per_out": 0.0,
            "materialize_w": False,
            "kernel": kernel,
        }
    return node


def island_node():
    # 64 down rows in_dim=17408 + 48 lin_o + 16 o in_dim=6144. f32v2.
    down_one = 8 + 4 * 17408
    out_one = 8 + 4 * 6144
    stored = 64 * down_one + 64 * out_one
    assert stored == ISLAND_WRITE_BYTES, stored
    return {
        "id": "island.row_3994",
        "kind": "ExactIsland",
        "count": 128,
        "selector": "compile_time_output_row",
        "indices": [3994],
        "index_bits": 0,
        "values_dtype": "f32",
        "when": {"skip_layers": [7], "reason": "L7 hidden col 3994 identically 0; post_attn gamma[3994]=0"},
        "apply_layers": "all_write: 64 down + 48 lin_o + 16 o",
        "source_elements": 0,  # does not change N
        "stored_bytes": stored,
        "active_bytes_per_token": stored,  # epilogue reads every sidecar every token
        "double_store": True,  # Qn body still holds the approximate row
        "kernel": KERNEL["row_dot"],
        "kernel_exists": False,
        "materialize_w": False,
        "class": "CONDITIONAL",
        "role": "correct",
        "decode": {
            "code_bytes_touched": 0,
            "scale_bytes_touched": 0,
            "unpack_ops": 0,
            "macs": 64 * 17408 + 64 * 6144,
            "scale_loads_per_out": 0.0,
            "materialize_w": False,
            "kernel": KERNEL["row_dot"],
            "note": "1-row f32 dot into out[3994] after the write GEMV. Do not branch inside TPR64.",
        },
        "breakdown": {
            "down_f32v2": 64 * down_one,
            "out_o_f32v2": 64 * out_one,
        },
    }


def generated_down_node():
    # 64 x down as HGRAVS01 r160_b3. MEASURED payload from mixed-2p0.
    n = 5120 * 17408
    return {
        "id": "mlp.down.generated",
        "kind": "GeneratedBlock",
        "count": 64,
        "shape": [5120, 17408],
        "source_elements": n * 64,
        "generator": "activation_weighted_svd",
        "rank": 160,
        "factor_elems": 5120 * 160 + 160 * 17408,  # 3,604,480
        "left": {"container": "HGRAVS01", "bits": 3, "side": "U·S"},
        "right": {"container": "HGRAVS01", "bits": 3, "side": "Vh"},
        "residual": None,
        "stored_bytes": HGRAVS_DOWN_BYTES,
        "active_bytes_per_token": HGRAVS_DOWN_BYTES,
        "kernel": KERNEL["factor"],
        "kernel_exists": True,
        "materialize_w": False,
        "class": "PREDICTABLE",
        "role": "mlp.down",
        "consume": "compose: y = L @ (R @ x)",
        "decode": {
            "code_bytes_touched": None,
            "scale_bytes_touched": None,
            "unpack_ops": None,
            "macs": 64 * (5120 * 160 + 160 * 17408),
            "scale_loads_per_out": 1.0,
            "materialize_w": False,
            "kernel": KERNEL["factor"],
            "stages": 2,
        },
        "quality": {
            "construction": "mixed-2p0 down HGRAVS01",
            "physical_bpw_class": 0.13161714918473189,
            "honest_hold_min": 0.730175,
            "verdict": "KILLS as constructed. REOPEN_IF activation-weighted SVD on a Q4-vehicle capture beats Q4 error at lower total BPW.",
        },
    }


def runtime_nodes(seq=SEQ_DEFAULT):
    rec = 48 * 48 * 128 * 128 * 4
    conv = 48 * 10240 * 3 * 4
    kv_one = 4 * 256 * 4
    kv_write = 16 * 2 * kv_one
    kv_read = 16 * 2 * seq * kv_one
    assert rec == 150_994_944
    assert conv == 5_898_240
    assert kv_write == 131_072
    return [
        {
            "id": "state.dn.rec",
            "kind": "RuntimeState",
            "family": "deltanet_rec",
            "count": 48,
            "resident_bytes": rec,
            "rw_bytes_per_token": rec * 2,
            "seq_term": False,
            "in_complete_bpw": False,
            "kernel": "qwen38_gated_delta_decode_vi",
            "class": "ESSENTIAL",
        },
        {
            "id": "state.dn.conv",
            "kind": "RuntimeState",
            "family": "deltanet_conv",
            "count": 48,
            "resident_bytes": conv,
            "rw_bytes_per_token": conv * 2,
            "seq_term": False,
            "in_complete_bpw": False,
            "kernel": "qwen38_qkvz_rearrange_conv_l2_f32",
            "class": "ESSENTIAL",
        },
        {
            "id": "state.gqa.kv",
            "kind": "RuntimeState",
            "family": "gqa_kv",
            "count": 16,
            "resident_bytes": 16 * 2 * 128 * kv_one,  # max_seq=128 default greedy
            "rw_bytes_per_token": kv_write + kv_read,
            "kv_write": kv_write,
            "kv_read_at_seq": kv_read,
            "seq": seq,
            "seq_term": True,
            "in_complete_bpw": False,
            "kernel": "qwen38_gqa_qk_norm_rope_cache_f32 + mha_decode_f32",
            "class": "ESSENTIAL",
        },
    ]


# --- class tables --------------------------------------------------------------

G0_Q4 = [
    ("mlp.gate", 64, (17408, 5120), "mlp"),
    ("mlp.up", 64, (17408, 5120), "mlp"),
    ("mlp.down", 64, (5120, 17408), "mlp"),
    ("dn.qkvz", 48, (16384, 5120), "attn"),
    ("dn.ba", 48, (96, 5120), "attn"),
    ("dn.out", 48, (5120, 6144), "attn"),
    ("gqa.q", 16, (12288, 5120), "attn"),
    ("gqa.k", 16, (1024, 5120), "attn"),
    ("gqa.v", 16, (1024, 5120), "attn"),
    ("gqa.o", 16, (5120, 6144), "attn"),
    ("lm_head", 1, (248320, 5120), "table"),
]

G0_SMALL = [
    ("dn.conv1d", 48, (10240, 4, 1)),
    ("norm.input", 64, (5120,)),
    ("norm.post_attn", 64, (5120,)),
    ("norm.final", 1, (5120,)),
    ("dn.norm", 48, (128,)),
    ("dn.A_log", 48, (48,)),
    ("dn.dt_bias", 48, (48,)),
    ("gqa.q_norm", 16, (256,)),
    ("gqa.k_norm", 16, (256,)),
]

# MEASURED class payloads from g1-pack-q4-mse-g128.md §2
MSE_CLASS_BYTES = {
    "mlp.gate": 2_941_273_600,
    "mlp.up": 2_941_273_600,
    "mlp.down": 2_941_273_600,
    "dn.qkvz": 2_076_193_920,
    "dn.ba": 12_177_984,
    "dn.out": 778_581_024,
    "gqa.q": 519_049_584,
    "gqa.k": 43_258_144,
    "gqa.v": 43_258_144,
    "gqa.o": 259_527_008,
    "lm_head": 655_565_086,
    "embed": 675_430_686,
}


def program_g0():
    nodes = []
    for id_, c, sh, role in G0_Q4:
        nodes.append(
            quant_node(
                id_,
                c,
                sh,
                "HQ30UQ4",
                4,
                64,
                "absmax",
                "nibble_minus_8",
                KERNEL["geo64"],
                role,
            )
        )
    nodes.append(
        quant_node(
            "embed",
            1,
            (248320, 5120),
            "HQ30UQ4",
            4,
            64,
            "absmax",
            "nibble_minus_8",
            KERNEL["embed_q4"],
            "table",
            active="embed_row",
        )
    )
    for id_, c, sh in G0_SMALL:
        nodes.append(
            quant_node(id_, c, sh, "f32v2", 32, None, "identity", None, KERNEL["f32"], "small")
        )
    ops = [
        {"op": "fuse", "when": "pack", "src": ["in_proj_qkv", "in_proj_z"], "dst": "dn.qkvz", "layout": "per_key_qkvz"},
        {"op": "fuse", "when": "pack", "src": ["in_proj_b", "in_proj_a"], "dst": "dn.ba", "layout": "per_key_b3_a3"},
        {"op": "lookup", "node": "embed", "kernel": KERNEL["embed_q4"]},
        {"op": "decode", "nodes": [n["id"] for n in nodes if n["kind"] == "QuantTensor"]},
        {"op": "matvec", "nodes": [id_ for id_, *_ in G0_Q4], "kernel": KERNEL["geo64"]},
        {"op": "fuse", "when": "runtime", "src": ["mlp.gate", "mlp.up"], "dst": "swiglu", "kernel": "gk_swiglu_f32", "status": "NOT_FUSED_on_G0", "note": "two GEMVs + standalone silu"},
    ]
    return {
        "id": "g0.uniform-q4-v1",
        "artifact": "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1",
        "manifest_sha256": G0_MANIFEST_SHA,
        "schema_artifact": "hawking.ascent.qwen38_language_uniform_q4.v1",
        "nodes": nodes,
        "ops": ops,
        "runtime": runtime_nodes(),
        "catalog_bytes": 0,
        "s0_192": G0_S0,
        "s0_tag": "CITED",
    }


def program_mse():
    nodes = []
    for id_, c, sh, role in G0_Q4:
        group = 128
        nodes.append(
            quant_node(
                id_,
                c,
                sh,
                "HGRAVU01",
                4,
                group,
                "mse",
                "nibble_minus_bound",
                KERNEL["simd"],
                role,
                stored_override=MSE_CLASS_BYTES[id_],
            )
        )
    nodes.append(
        quant_node(
            "embed",
            1,
            (248320, 5120),
            "HGRAVU01",
            4,
            64,
            "absmax",
            "nibble_minus_bound",
            KERNEL["embed_u01"],
            "table",
            stored_override=MSE_CLASS_BYTES["embed"],
            active="embed_row",
        )
    )
    for id_, c, sh in G0_SMALL:
        nodes.append(
            quant_node(id_, c, sh, "f32v2", 32, None, "identity", None, KERNEL["f32"], "small")
        )
    ops = [
        {"op": "fuse", "when": "pack", "src": ["in_proj_qkv", "in_proj_z"], "dst": "dn.qkvz"},
        {"op": "fuse", "when": "pack", "src": ["in_proj_b", "in_proj_a"], "dst": "dn.ba"},
        {"op": "lookup", "node": "embed", "kernel": KERNEL["embed_u01"]},
        {"op": "decode", "nodes": [n["id"] for n in nodes if n["kind"] == "QuantTensor"]},
        {"op": "matvec", "nodes": [id_ for id_, *_ in G0_Q4], "kernel": KERNEL["simd"], "note": "geo_tpr64 will not bind (HGRAVU01 group!=64)"},
    ]
    return {
        "id": "q4-mse-g128-v1",
        "artifact": "workspace/campaign/records/runs/qwen38-27b/q4-mse-g128-v1",
        "schema_artifact": "HQ38M20 + HGRAVU01",
        "nodes": nodes,
        "ops": ops,
        "runtime": runtime_nodes(),
        "catalog_bytes": MSE_CATALOG,
        "s0_192": MSE_S0,
        "s0_tag": "CITED packed VERIFY",
        "scale_plane": "THIN-CAPTURE (256 tok, rpd 0.0500/0.0147). Legal to pack. Not a production plane.",
    }


def program_island():
    p = program_g0()
    p = deepcopy(p)
    p["id"] = "fwd.exact-island-3994-on-g0"
    p["artifact"] = None
    p["nodes"].append(island_node())
    p["ops"].append(
        {
            "op": "correct",
            "after": "matvec:down|out|o",
            "node": "island.row_3994",
            "kernel": KERNEL["row_dot"],
            "kernel_exists": False,
            "note": "overwrite out[3994]; do not if(row==3994) inside TPR64",
        }
    )
    p["s0_192"] = None
    p["s0_tag"] = "UNMEASURED as a program. Island-only organ deltas CITED g1-channel-3994-island.md"
    return p


def program_generated():
    p = program_g0()
    p = deepcopy(p)
    p["id"] = "fwd.generated-down-r160-on-g0"
    p["artifact"] = None
    p["nodes"] = [n for n in p["nodes"] if n["id"] != "mlp.down"]
    p["nodes"].append(generated_down_node())
    p["ops"] = [op for op in p["ops"] if not (op.get("op") == "matvec" and "mlp.down" in op.get("nodes", []))]
    p["ops"].append(
        {
            "op": "generate",
            "node": "mlp.down.generated",
            "consume": "compose L@(R@x)",
            "kernel": KERNEL["factor"],
        }
    )
    p["s0_192"] = None
    p["s0_tag"] = "UNMEASURED as a program. Construction KILLS (mixed-2p0 / honest hold 0.730). Expressible."
    return p


# --- cost model ----------------------------------------------------------------


def eval_program(p, seq=SEQ_DEFAULT):
    stored = 0
    source_e = 0
    active_w = 0
    embed_stored = 0
    embed_active = 0
    small_stored = 0
    q_count = 0
    d_count = 0
    island_stored = 0
    gen_stored = 0
    kernels = {}
    missing_kernels = []
    decode_code = 0
    decode_scale = 0
    unpack = 0
    macs = 0
    materialize = False
    classes = {}
    for n in p["nodes"]:
        stored += n["stored_bytes"]
        source_e += n.get("source_elements", 0)
        if n["id"] == "embed":
            embed_stored = n["stored_bytes"]
            embed_active = n["active_bytes_per_token"]
        else:
            active_w += n["stored_bytes"]
        if n["kind"] == "DenseTensor":
            d_count += n["count"]
            small_stored += n["stored_bytes"]
        elif n["kind"] == "QuantTensor":
            q_count += n["count"]
        elif n["kind"] == "ExactIsland":
            island_stored += n["stored_bytes"]
        elif n["kind"] == "GeneratedBlock":
            gen_stored += n["stored_bytes"]
        k = n.get("kernel")
        kernels[k] = kernels.get(k, 0) + n["count"]
        if n.get("kernel_exists") is False:
            missing_kernels.append(n["id"])
        dec = n.get("decode") or {}
        decode_code += dec.get("code_bytes_touched") or 0
        decode_scale += dec.get("scale_bytes_touched") or 0
        unpack += dec.get("unpack_ops") or 0
        macs += dec.get("macs") or 0
        materialize = materialize or bool(dec.get("materialize_w"))
        classes[n["id"]] = {
            "kind": n["kind"],
            "count": n["count"],
            "container": n.get("container"),
            "bits": n.get("bits"),
            "group": n.get("group"),
            "scale_rule": n.get("scale_rule"),
            "alphabet": n.get("alphabet"),
            "stored_bytes": n["stored_bytes"],
            "source_elements": n.get("source_elements"),
            "physical_bpw": (8 * n["stored_bytes"] / n["source_elements"])
            if n.get("source_elements")
            else None,
            "kernel": k,
            "class": n.get("class"),
        }
    # RuntimeState is not in complete BPW
    state_resident = sum(s["resident_bytes"] for s in p["runtime"])
    state_rw = sum(s["rw_bytes_per_token"] for s in p["runtime"])
    complete_bpw = 8 * stored / SOURCE_N
    artifact_bpw = 8 * (stored + p.get("catalog_bytes", 0)) / SOURCE_N
    active_budget = stored - embed_stored  # receipt convention
    active_weights = active_budget + embed_active
    return {
        "program": p["id"],
        "stored_bytes": stored,
        "catalog_bytes": p.get("catalog_bytes", 0),
        "complete_physical_bpw": complete_bpw,
        "artifact_complete_bpw": artifact_bpw,
        "source_elements_accounted": source_e,
        "source_n": SOURCE_N,
        "source_elements_match_n": source_e == SOURCE_N
        or source_e == SOURCE_N,  # island adds 0
        "q_or_gen_plus_dense_count": q_count + d_count,
        "quant_tensors": q_count,
        "dense_tensors": d_count,
        "island_bytes": island_stored,
        "generated_bytes": gen_stored,
        "embed_stored": embed_stored,
        "embed_active_row": embed_active,
        "small_stored": small_stored,
        "active_budget_bytes": active_budget,
        "active_weight_bytes": active_weights,
        "state_resident_bytes": state_resident,
        "state_rw_bytes_per_token": state_rw,
        "active_total_bytes": active_weights + state_rw,
        "decode": {
            "code_bytes_touched": decode_code,
            "scale_bytes_touched": decode_scale,
            "unpack_ops": unpack,
            "macs": macs,
            "materialize_w": materialize,
        },
        "kernels": kernels,
        "missing_kernels": missing_kernels,
        "s0_192": p.get("s0_192"),
        "s0_tag": p.get("s0_tag"),
        "classes": classes,
        "seq": seq,
    }


def census(p):
    rows = []
    for n in p["nodes"]:
        rows.append(
            (
                n["id"],
                n["kind"],
                n["count"],
                n.get("container"),
                n.get("bits"),
                n.get("group"),
                n.get("scale_rule"),
                n.get("alphabet"),
                n.get("kernel"),
                n["stored_bytes"],
            )
        )
    return rows


def round_trip(p, expect_bytes, expect_bpw, expect_census, label):
    ev = eval_program(p)
    errors = []
    if ev["stored_bytes"] != expect_bytes:
        errors.append(f"bytes {ev['stored_bytes']} != {expect_bytes}")
    if abs(ev["complete_physical_bpw"] - expect_bpw) > 1e-15:
        errors.append(f"bpw {ev['complete_physical_bpw']!r} != {expect_bpw!r}")
    got = census(p)
    if got != expect_census:
        # compare loosely on first 8 fields
        if len(got) != len(expect_census):
            errors.append(f"census len {len(got)} != {len(expect_census)}")
        else:
            for a, b in zip(got, expect_census):
                if a != b:
                    errors.append(f"census row {a[0]} {a} != {b}")
                    break
    # invert: rebuild stored from census rows
    rebuilt = sum(r[-1] for r in got)
    if rebuilt != ev["stored_bytes"]:
        errors.append(f"invert stored {rebuilt} != {ev['stored_bytes']}")
    # source N
    se = ev["source_elements_accounted"]
    if se != SOURCE_N:
        errors.append(f"source_elements {se} != {SOURCE_N}")
    ok = not errors
    return {
        "label": label,
        "ok": ok,
        "errors": errors,
        "stored_bytes": ev["stored_bytes"],
        "complete_physical_bpw": ev["complete_physical_bpw"],
        "quant_tensors": ev["quant_tensors"],
        "dense_tensors": ev["dense_tensors"],
        "kernels": ev["kernels"],
    }


def g0_expect_census():
    p = program_g0()
    return census(p)


def mse_expect_census():
    p = program_mse()
    return census(p)


def token_cost_structure(ev, kernel_class):
    """Do not invent TOKEN_NS. Attach measured G0 partition only when the consume path is G0."""
    out = {
        "complete_physical_bpw": ev["complete_physical_bpw"],
        "active_weight_bytes": ev["active_weight_bytes"],
        "state_rw_bytes_per_token": ev["state_rw_bytes_per_token"],
        "kernel_class": kernel_class,
        "materialize_w": ev["decode"]["materialize_w"],
    }
    if kernel_class == "geo_tpr64_g64":
        out["token_ns"] = {
            "live_decode_phase_median": LIVE_TOKEN_NS,
            "live_tps": LIVE_TPS,
            "live_tag": "MEASURED DIRTY_ENGINEERING g1-baseline-remeasure",
            "ledger_encode_submit_wait": LEDGER_TOKEN_NS,
            "ledger_tag": "CITED G024 DIRTY_ENGINEERING",
            "addressing_frac_mlp": 0.871692,
            "decode_frac_mlp": 0.083585,
            "note": "partition is of isolated GEMV CBs, not a projection onto a new program",
        }
    else:
        out["token_ns"] = {
            "value": None,
            "tag": "UNMEASURED",
            "reason": f"kernel_class={kernel_class} is not the G0 geo_tpr64_g64 consume path. Filesize is not token cost.",
        }
    return out


def main():
    g0 = program_g0()
    mse = program_mse()
    isl = program_island()
    gen = program_generated()

    ev_g0 = eval_program(g0)
    ev_mse = eval_program(mse)
    ev_isl = eval_program(isl)
    ev_gen = eval_program(gen)

    rt_g0 = round_trip(g0, G0_BYTES, G0_BPW, g0_expect_census(), "G0")
    rt_mse = round_trip(mse, MSE_BYTES, MSE_BPW, mse_expect_census(), "q4-mse-g128-v1")

    # identity checks
    assert ev_g0["stored_bytes"] == G0_BYTES
    assert ev_g0["complete_physical_bpw"] == G0_BPW
    assert ev_g0["quant_tensors"] == 402
    assert ev_g0["dense_tensors"] == 353
    assert ev_g0["active_budget_bytes"] == ACTIVE_BUDGET_RECEIPT
    assert ev_mse["stored_bytes"] == MSE_BYTES
    assert ev_mse["complete_physical_bpw"] == MSE_BPW
    assert ev_isl["stored_bytes"] == G0_BYTES + ISLAND_WRITE_BYTES
    assert ev_gen["stored_bytes"] == G0_BYTES - 3_030_387_200 + HGRAVS_DOWN_BYTES
    assert ev_g0["decode"]["materialize_w"] is False
    assert ev_mse["decode"]["materialize_w"] is False
    assert KERNEL["row_dot"] in ev_isl["kernels"]
    assert ev_isl["missing_kernels"] == ["island.row_3994"]

    extra_mse_vs_hq30 = MSE_BYTES - (
        # HQ30 g128 on 401 + HQ30 g64 embed + f32
        sum(hq30_payload(prod(sh), 4, 128)["stored_bytes"] * c for id_, c, sh, _ in G0_Q4)
        + hq30_payload(248320 * 5120, 4, 64)["stored_bytes"]
        + sum(f32v2_payload(prod(sh))["stored_bytes"] * c for _, c, sh in G0_SMALL)
    )

    report = {
        "schema": "hawking.gravity.ir.v0.eval",
        "source_n": SOURCE_N,
        "round_trip": {"G0": rt_g0, "q4-mse-g128-v1": rt_mse},
        "eval": {
            "g0": {k: ev_g0[k] for k in ev_g0 if k != "classes"},
            "q4-mse-g128-v1": {k: ev_mse[k] for k in ev_mse if k != "classes"},
            "exact-island-3994": {k: ev_isl[k] for k in ev_isl if k != "classes"},
            "generated-down-r160": {k: ev_gen[k] for k in ev_gen if k != "classes"},
        },
        "class_bpw": {
            "g0": {k: v["physical_bpw"] for k, v in ev_g0["classes"].items()},
            "mse": {k: v["physical_bpw"] for k, v in ev_mse["classes"].items()},
            "island": {k: v["physical_bpw"] for k, v in ev_isl["classes"].items() if v["physical_bpw"] is not None},
            "generated": {k: v["physical_bpw"] for k, v in ev_gen["classes"].items() if v["physical_bpw"] is not None},
        },
        "mse_header_tax_vs_hq30_design": extra_mse_vs_hq30,
        "island_delta_bpw": 8 * ISLAND_WRITE_BYTES / SOURCE_N,
        "generated_delta_bpw_vs_g0": ev_gen["complete_physical_bpw"] - G0_BPW,
        "token_cost": {
            "g0": token_cost_structure(ev_g0, "geo_tpr64_g64"),
            "q4-mse-g128-v1": token_cost_structure(ev_mse, "factor_simd_hgravu01_g128"),
            "exact-island-3994": token_cost_structure(ev_isl, "geo_tpr64_g64 + MISSING row_dot"),
            "generated-down-r160": token_cost_structure(ev_gen, "geo_tpr64_g64 + factor_simd3_down"),
        },
        "mse_json_len_one": {
            id_: ev_mse["classes"][id_].get("physical_bpw")
            and mse["nodes"][[n["id"] for n in mse["nodes"]].index(id_)].get("json_len_one")
            for id_ in ["mlp.gate", "lm_head", "embed", "dn.ba"]
        },
        "invariants": {
            "materialize_w_any": False,
            "source_n_fixed": True,
            "runtime_state_excluded_from_bpw": True,
            "catalog_excluded_from_complete_bpw": True,
            "g0_active_budget_matches_receipt": ev_g0["active_budget_bytes"] == ACTIVE_BUDGET_RECEIPT,
        },
    }

    print(json.dumps(report, indent=2, sort_keys=False))

    # human ledger
    print("\n===== LEDGER =====")
    for name, ev in [
        ("G0", ev_g0),
        ("q4-mse-g128-v1", ev_mse),
        ("island-3994", ev_isl),
        ("generated-down", ev_gen),
    ]:
        print(
            f"{name:18} stored={ev['stored_bytes']:12d}  "
            f"bpw={ev['complete_physical_bpw']:.15f}  "
            f"active_w={ev['active_weight_bytes']:12d}  "
            f"state_rw={ev['state_rw_bytes_per_token']:10d}  "
            f"Q={ev['quant_tensors']} D={ev['dense_tensors']}  "
            f"missing={ev['missing_kernels']}"
        )
    print("G0 round-trip", rt_g0["ok"], rt_g0["errors"])
    print("MSE round-trip", rt_mse["ok"], rt_mse["errors"])
    print("mse header tax vs HQ30 design", extra_mse_vs_hq30)
    print("island +BPW", 8 * ISLAND_WRITE_BYTES / SOURCE_N)
    print("generated BPW", ev_gen["complete_physical_bpw"])
    print("generated stored", ev_gen["stored_bytes"])
    print("8*G0/N", 8 * G0_BYTES / SOURCE_N)
    print("8*MSE/N", 8 * MSE_BYTES / SOURCE_N)
    print("active_budget G0", ev_g0["active_budget_bytes"], "receipt", ACTIVE_BUDGET_RECEIPT)
    print("embed row G0", ev_g0["embed_active_row"])
    print("state rw", ev_g0["state_rw_bytes_per_token"])


if __name__ == "__main__":
    main()
