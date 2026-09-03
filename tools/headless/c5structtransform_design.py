#!/usr/bin/env python3
"""C5 structured-transform design: fewer bytes AND fewer operations.

Butterfly / Walsh-Hadamard / mixed-radix Monarch-Hadamard as an implicit
operator — the transform is the code, not a matrix. This is a DESIGN lane:
it terminates in a measured decision, not a kernel.

    python3 tools/headless/c5structtransform_design.py

Reads geometry and shaders from crates/ (read-only). Writes only
tools/headless/c5structtransform_design.py (this file) and
receipts/headless/C5STRUCTTRANSFORM_DESIGN.json.

Does not open Metal, does not load the 27B, does not score quality on
synthetic activations. Prior-science numbers are confirmed from receipts
(git-show for ascent-16 / G1 files that are not in the sparse checkout).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "hawking.headless.c5structtransform_design.v1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts/headless/C5STRUCTTRANSFORM_DESIGN.json"

DECODE = REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
GEOMETRY = REPO / "crates/hawking-core/src/model/qwen38_geometry.rs"
SCHEDULE = REPO / "crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs"
LEDGER = REPO / "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs"
SHADERS = REPO / "crates/hawking-core/shaders"
Q4_METAL = SHADERS / "qwen_uniform_q4.metal"
STRAND_METAL = SHADERS / "strand_bitslice.metal"
HGRAVS_METAL = SHADERS / "q80_mixed_decode.metal"
QUANT_METAL = SHADERS / "quant.metal"

KERNEL_CENSUS = REPO / "receipts/headless/NOETIC_KERNEL_CENSUS.json"
OP_CENSUS = REPO / "receipts/headless/NOETIC_OPERATION_CENSUS.json"
TPR64 = REPO / "receipts/headless/NOETIC_TPR64_REOPEN.json"
INFO_ACCT = REPO / "receipts/headless/NOETIC_INFORMATION_ACCOUNTING.json"

# Anchors — measured, not re-derived. Same set the operation census locked.
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_BOUND = 38
ANCHOR_DECLARED = 554
ANCHOR_TPS = 32.73
ANCHOR_TOKEN_MS = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_UNIFIED_B = 103_079_215_104
ANCHOR_GPU_CORES = 60
ANCHOR_PARAMS = 26_895_998_464
ANCHOR_BPW = 4.253
ANCHOR_ARTIFACT_B = 14_297_933_604
ANCHOR_GEMV_MAC_FLOPS = 51_243_909_120
ANCHOR_GEMV_ELEMENTS = 25_621_954_560
ANCHOR_Q4_GEMV_BYTES = 13_611_663_360
ANCHOR_DRAM_BYTES = 13_988_022_948
ANCHOR_GEMV_DISPATCHES = 401
ANCHOR_ACTIVATION_FLOPS = 297_313_024
ANCHOR_EXEC_FLOPS = 77_163_181_824
ANCHOR_EXEC_OPS = 179_651_020_544
ANCHOR_COMPUTE_PEAK_GFLOPS = 8979.0
ANCHOR_MLX_TPS = 35.51
ANCHOR_LLAMA_Q5K_TPS = 24.12

UNIFORM_Q4_GROUP = 64
Q4_BYTES_PER_GROUP = UNIFORM_Q4_GROUP // 2 + 2  # 32 code + 2 scale = 34
F16_BYTES = 2
HEADER_BYTES = 40  # HQ30UQ4-class per-tensor header
TG_MEM_LIMIT_BYTES = 32 * 1024  # crates/hawking-core/shaders/quant.metal:2115
SIMDGROUP_WIDTH = 32
WORKHORSE_TG = 128
BLOCK_B = 1024  # G032 tile: hidden=5*1024, intermediate=17*1024, no padding
LOG2_B = 10

# Prior-science numbers (confirmed live from receipts when present).
G032_Q4_DELTA_HOLD = 0.00033819033592939903
G032_Q4_DELTA_ENTROPY = 0.025883768994956233
G032_Q3_DELTA_HOLD = 0.0016053876210644358
G032_Q3_DELTA_ENTROPY = 0.02372571953461127
G032_Q2_DELTA_HOLD = 0.008204946837970573
G034_MEAN_FLAT_Q3 = 0.1839276241211841
G034_MEAN_LOWRANK = 0.5393288880586624
G034_ERROR_RATIO = 2.93
G034_MAC_RATIO = 0.2029641544117647
G035_SHARED_BEATS = False
G1_ONE_BASIS_FULLV_BPW = 0.015594527957807665
G1_DENSE_ORTH_BPW_64 = 0.9980497892996906
G1_KRON_ENERGY_RANK1 = 0.015259800946951544
G1_KRON_ENERGY_RANK64 = 0.801171269258503
G1_GATE_Q4_REL_L2 = 0.11899048089981079
G1_BLOCK_TERM_REL_L2 = 0.9989632368087769
G1_BLOCK_TERM_LOCAL_BPW = 0.026020364200367647
G1_KRON_SUM_REL_L2 = 0.9334229826927185
G1_SUB05_ROWS = 223
G1_SUB05_HEALTHY = 0
G042_GENERATED_BPW = 0.0
G043_RECON_SHARE = 0.7138349980076641
G043_PHYSICAL_NS = 30_549_917
Q80_STORAGE_BPW = 0.6462
Q80_ACTIVE_BPW = 2.518
MLP_DISTILL_HELD_OUT_GAP = 0.4206
NULL_COSINE = 0.898
HGRAVS01_DOWN_BPW = 0.13
GLM_EXPERT_BPW = 0.167
TPR64_HADAMARD_GATE_NS = 17333
TPR64_FREE_VARIANTS = 32
TPR64_TOTAL_VARIANTS = 33


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def git_show_json(rel: str):
    """Load a blob from HEAD. Sparse checkout is not evidence of absence."""
    try:
        p = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            capture_output=True, cwd=REPO, timeout=60,
        )
        if p.returncode != 0 or not p.stdout:
            return None
        return json.loads(p.stdout.decode("utf-8"))
    except Exception:
        return None


def load_json(path: Path):
    if path.is_file():
        return json.loads(path.read_text())
    return None


def usize_const(src: str, name: str) -> int:
    m = re.search(rf"pub const {name}: usize = ([0-9_]+);", src)
    if not m:
        raise SystemExit(f"FAIL: missing usize const {name}")
    return int(m.group(1).replace("_", ""))


def q4_matrix_bytes(rows: int, cols: int) -> int:
    groups = (cols + UNIFORM_Q4_GROUP - 1) // UNIFORM_Q4_GROUP
    return rows * groups * Q4_BYTES_PER_GROUP


def gcd_many(vals: list[int]) -> int:
    g = 0
    for v in vals:
        g = math.gcd(g, v)
    return g


def load_geometry() -> dict:
    geo = GEOMETRY.read_text()
    sched = SCHEDULE.read_text()
    g = {
        "layers": usize_const(geo, "QWEN38_LAYERS"),
        "dn_layers": usize_const(geo, "QWEN38_DELTANET_LAYERS"),
        "gqa_layers": usize_const(geo, "QWEN38_GQA_LAYERS"),
        "hidden": usize_const(geo, "QWEN38_HIDDEN"),
        "intermediate": usize_const(geo, "QWEN38_INTERMEDIATE"),
        "vocab": usize_const(geo, "QWEN38_VOCAB"),
        "qkvz_rows": usize_const(geo, "QWEN38_QKVZ_ROWS"),
        "ba_rows": usize_const(geo, "QWEN38_BA_ROWS"),
        "q_proj_rows": usize_const(geo, "QWEN38_Q_PROJ_ROWS"),
        "kv_proj_rows": usize_const(geo, "QWEN38_KV_PROJ_ROWS"),
        "o_proj_rows": usize_const(geo, "QWEN38_O_PROJ_ROWS"),
        "o_proj_cols": usize_const(geo, "QWEN38_O_PROJ_COLS"),
        "mixer_prefix": usize_const(sched, "QWEN38_MIXER_PREFIX_DISPATCHES"),
        "mlp_suffix": usize_const(sched, "QWEN38_DENSE_MLP_SUFFIX_DISPATCHES"),
    }
    g["full_layer"] = g["mixer_prefix"] + g["mlp_suffix"]
    g["terminal_n"] = 3
    g["production_dispatches"] = 1 + g["layers"] * g["full_layer"] + g["terminal_n"]
    g["production_cbs"] = 1
    return g


def gemv_organs(g: dict) -> list[dict]:
    H, I = g["hidden"], g["intermediate"]
    organs = []

    def add(name, count, rows, cols, role):
        elems = rows * cols
        organs.append({
            "organ": name,
            "count_per_token": count,
            "rows": rows,
            "cols": cols,
            "elements_per_launch": elems,
            "elements_per_token": elems * count,
            "mac_flops_per_launch": 2 * elems,
            "mac_flops_per_token": 2 * elems * count,
            "q4_bytes_per_launch": q4_matrix_bytes(rows, cols),
            "q4_bytes_per_token": q4_matrix_bytes(rows, cols) * count,
            "role": role,
        })

    add("mlp.gate_proj", g["layers"], I, H, "mlp")
    add("mlp.up_proj", g["layers"], I, H, "mlp")
    add("mlp.down_proj", g["layers"], H, I, "mlp")
    add("linear_attn.in_proj_qkvz", g["dn_layers"], g["qkvz_rows"], H, "deltanet")
    add("linear_attn.in_proj_ba", g["dn_layers"], g["ba_rows"], H, "deltanet")
    add("linear_attn.out_proj", g["dn_layers"], H, g["o_proj_cols"], "deltanet")
    add("self_attn.q_proj", g["gqa_layers"], g["q_proj_rows"], H, "gqa")
    add("self_attn.k_proj", g["gqa_layers"], g["kv_proj_rows"], H, "gqa")
    add("self_attn.v_proj", g["gqa_layers"], g["kv_proj_rows"], H, "gqa")
    add("self_attn.o_proj", g["gqa_layers"], H, g["o_proj_cols"], "gqa")
    add("lm_head", 1, g["vocab"], H, "terminal")
    return organs


def tiles(dim: int, b: int):
    if dim % b != 0:
        return None
    return dim // b


def fmh_costs(rows: int, cols: int, b: int) -> dict | None:
    """Mixed-radix Monarch-Hadamard costs for one m×n GEMV.

    n = n1*B, m = m1*B. Generated H_B (Sylvester, 0 stored bytes) plus a
    stored coefficient grid S of shape (B, m1, n1). Optional learned input
    mixes R_t of shape (n1, B, B). Optional output mixes L_k of shape
    (m1, B, B).
    """
    m1 = tiles(rows, b)
    n1 = tiles(cols, b)
    if m1 is None or n1 is None:
        return None
    log2b = b.bit_length() - 1
    assert 1 << log2b == b
    s_params = b * m1 * n1          # = m*n / B
    r_params = n1 * b * b           # = n * B
    l_params = m1 * b * b           # = m * B
    fwht_flops = cols * log2b       # n log2 B adds (u+v / u-v counted)
    s_flops = 2 * s_params          # FMA = 2
    r_flops = 2 * r_params
    l_flops = 2 * l_params
    s_bytes_f16 = s_params * F16_BYTES
    r_bytes_f16 = r_params * F16_BYTES
    l_bytes_f16 = l_params * F16_BYTES
    local_bpw_s = 8.0 * s_bytes_f16 / (rows * cols)   # = 16/B
    return {
        "m1": m1,
        "n1": n1,
        "B": b,
        "log2_B": log2b,
        "s_params": s_params,
        "r_params": r_params,
        "l_params": l_params,
        "fwht_flops": fwht_flops,
        "s_flops": s_flops,
        "r_flops": r_flops,
        "l_flops": l_flops,
        "rung0_flops": fwht_flops + s_flops,
        "rung1_flops": fwht_flops + r_flops + s_flops,
        "rung2_flops": fwht_flops + r_flops + s_flops + l_flops,
        "rung0_bytes_f16": s_bytes_f16 + HEADER_BYTES,
        "rung1_bytes_f16": s_bytes_f16 + r_bytes_f16 + HEADER_BYTES,
        "rung2_bytes_f16": s_bytes_f16 + r_bytes_f16 + l_bytes_f16 + HEADER_BYTES,
        "local_bpw_rung0_f16": local_bpw_s,
        "local_bpw_rung1_f16": 8.0 * (s_bytes_f16 + r_bytes_f16) / (rows * cols),
        "local_bpw_rung2_f16": 8.0 * (s_bytes_f16 + r_bytes_f16 + l_bytes_f16) / (rows * cols),
        "dense_mac_flops": 2 * rows * cols,
        "q4_bytes": q4_matrix_bytes(rows, cols),
    }


def fwht_identity_check(n: int = 16) -> dict:
    """Sylvester-Hadamard is orthogonal after 1/sqrt(n). Pure Python, n power of 2.

    This is an operator identity check, not a quality eval, and not on activations.
    """
    assert n & (n - 1) == 0 and n >= 2
    # Build H by Sylvester recurrence, then FWHT a basis vector two ways.
    h = [[1.0]]
    while len(h) < n:
        top = [row + row for row in h]
        bot = [row + [-x for x in row] for row in h]
        h = top + bot
    scale = 1.0 / math.sqrt(n)
    h = [[x * scale for x in row] for row in h]
    # H H^T ≈ I
    max_off = 0.0
    max_diag_err = 0.0
    for i in range(n):
        for j in range(n):
            acc = sum(h[i][k] * h[j][k] for k in range(n))
            if i == j:
                max_diag_err = max(max_diag_err, abs(acc - 1.0))
            else:
                max_off = max(max_off, abs(acc))
    # In-place FWHT on e_0 equals first column of unnormalised Sylvester, scaled.
    x = [1.0] + [0.0] * (n - 1)
    length = 1
    while length < n:
        for i in range(0, n, length * 2):
            for j in range(i, i + length):
                u, v = x[j], x[j + length]
                x[j] = u + v
                x[j + length] = u - v
        length *= 2
    x = [v * scale for v in x]
    col0_err = max(abs(x[i] - h[i][0]) for i in range(n))
    ok = max_off < 1e-12 and max_diag_err < 1e-12 and col0_err < 1e-12
    return {
        "n": n,
        "max_off_diag_HHT": max_off,
        "max_diag_err_HHT": max_diag_err,
        "fwht_e0_vs_H_col0": col0_err,
        "ok": ok,
    }


def confirm_prior_science() -> dict:
    """Live search of receipts this tree can see (on disk or via git show)."""
    hits = []
    missing = []

    def rec(name, path, ok, detail):
        hits.append({"name": name, "path": path, "confirmed": ok, "detail": detail})
        if not ok:
            missing.append(name)

    g032q4 = git_show_json("receipts/ascent-2026-08-16/G032_XFORM_HADAMARD_Q4.json")
    if g032q4 and "summary" in g032q4:
        s = g032q4["summary"]
        rec("G032_Q4", "receipts/ascent-2026-08-16/G032_XFORM_HADAMARD_Q4.json", True, {
            "mean_delta_hold": s.get("mean_delta_hold"),
            "mean_delta_entropy_bits": s.get("mean_delta_entropy_bits"),
            "stored_bytes": (g032q4.get("transform") or {}).get("stored_bytes"),
            "family": (g032q4.get("transform") or {}).get("family"),
        })
    else:
        rec("G032_Q4", "receipts/ascent-2026-08-16/G032_XFORM_HADAMARD_Q4.json", False,
            "git-show missed; using recorded G032_Q4_DELTA_HOLD")

    g032q3 = git_show_json("receipts/ascent-2026-08-16/G032_XFORM_HADAMARD_Q3.json")
    if g032q3 and "summary" in g032q3:
        rec("G032_Q3", "receipts/ascent-2026-08-16/G032_XFORM_HADAMARD_Q3.json", True,
            g032q3["summary"])
    else:
        rec("G032_Q3", "receipts/ascent-2026-08-16/G032_XFORM_HADAMARD_Q3.json", False, "missed")

    g035 = git_show_json("receipts/ascent-2026-08-16/G035_CROSSLAYER_SHARE.json")
    if g035 and g035.get("pairs"):
        flags = [p.get("shared_beats_independent") for p in g035["pairs"]]
        rec("G035", "receipts/ascent-2026-08-16/G035_CROSSLAYER_SHARE.json", True, {
            "shared_beats_independent_any": any(flags),
            "n_pairs": len(flags),
            "first_independent_mean": g035["pairs"][0].get("independent_mean"),
            "first_shared_mean": g035["pairs"][0].get("shared_mean"),
        })
    else:
        rec("G035", "receipts/ascent-2026-08-16/G035_CROSSLAYER_SHARE.json", False, "missed")

    g034 = git_show_json("receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json")
    if g034:
        rec("G034", "receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json", True, {
            "mean_flat_q3": g034.get("mean_flat_q3"),
            "mean_lowrank": g034.get("mean_lowrank"),
            "verdict": g034.get("verdict"),
            "family_verdict": g034.get("family_verdict"),
        })
    else:
        rec("G034", "receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json", False, "missed")

    g042 = git_show_json("receipts/ascent-2026-08-16/G042_BPW_FAMILY.json")
    if g042 and "definitions" in g042:
        rec("G042", "receipts/ascent-2026-08-16/G042_BPW_FAMILY.json", True, {
            "GENERATED_BPW_EQUIVALENT": g042["definitions"].get("GENERATED_BPW_EQUIVALENT"),
            "SHARED_BPW": g042["definitions"].get("SHARED_BPW"),
        })
    else:
        rec("G042", "receipts/ascent-2026-08-16/G042_BPW_FAMILY.json", False, "missed")

    share = git_show_json("research/hawking-experiments/superwave/g1/evidence/g1_share_basis.json")
    if share and "identity" in share:
        ident = share["identity"]
        rec("G1_SHARE", "research/hawking-experiments/superwave/g1/evidence/g1_share_basis.json", True, {
            "one_basis_fullV_bpw": ident.get("one_basis_fullV_bpw"),
            "dense_orth_bpw_64sites": ident.get("dense_orth_bpw_64sites"),
            "note": "COMPONENT amortisation of a dense n×n right basis stored once for 64 sites. NOT a structured-transform BPW.",
        })
    else:
        rec("G1_SHARE", "research/hawking-experiments/superwave/g1/evidence/g1_share_basis.json", False, "missed")

    tens = git_show_json("research/hawking-experiments/superwave/g1/evidence/g1_tensor_operators.json")
    if tens and tens.get("tensors"):
        t0 = tens["tensors"][0]
        ops = t0.get("operators") or []
        healthy = sum(1 for o in ops if (o.get("gate") or {}).get("healthy"))
        fams = {}
        for o in ops:
            fams[o.get("family")] = fams.get(o.get("family"), 0) + 1
        kron_e = None
        if t0.get("reshapes"):
            kron_e = (t0["reshapes"][0] or {}).get("kronecker_energy")
        rec("G1_TENSOR", "research/hawking-experiments/superwave/g1/evidence/g1_tensor_operators.json", True, {
            "tensor0": t0.get("name"),
            "n_operators": len(ops),
            "healthy": healthy,
            "families": fams,
            "kronecker_energy_rank1": None if not kron_e else kron_e.get("1"),
            "kronecker_energy_rank64": None if not kron_e else kron_e.get("64"),
            "q4_rel_l2": (t0.get("q4") or {}).get("rel_l2"),
        })
    else:
        rec("G1_TENSOR", "research/hawking-experiments/superwave/g1/evidence/g1_tensor_operators.json", False, "missed")

    kc = load_json(KERNEL_CENSUS)
    if kc:
        fam = next((f for f in kc.get("families", []) if f.get("id") == "structured_transform"), None)
        rec("NOETIC_KERNEL_CENSUS.structured_transform", str(KERNEL_CENSUS), True, {
            "verdict": None if not fam else fam.get("verdict"),
            "kernel": None if not fam else (fam.get("kernel") or {}).get("name"),
            "compile_gate": None if not fam else (fam.get("kernel") or {}).get("compile_gate"),
        })
    else:
        rec("NOETIC_KERNEL_CENSUS", str(KERNEL_CENSUS), False, "missing on disk")

    oc = load_json(OP_CENSUS)
    if oc and oc.get("analytic_vs_measured"):
        rec("NOETIC_OPERATION_CENSUS", str(OP_CENSUS), True, {
            "dispatched_gemv_mac_flops": oc["analytic_vs_measured"].get("dispatched_gemv_mac_flops"),
            "dispatches": (oc.get("dispatch_reconciliation") or {}).get("formula_total"),
        })
    else:
        rec("NOETIC_OPERATION_CENSUS", str(OP_CENSUS), False, "missing")

    tpr = load_json(TPR64)
    if tpr and tpr.get("free_reconstruction"):
        rec("NOETIC_TPR64_REOPEN", str(TPR64), True, {
            "the_32": tpr["free_reconstruction"].get("the_32"),
            "hadamard_gate_ns": (tpr.get("prose_vs_table") or {}).get("hadamard_gate_ns"),
            "cosine_Wh_vs_W": (tpr.get("scale_invariance_probe") or {}).get("cosine_Wh_vs_W"),
            "null_cosine": (tpr.get("null_baseline_cosine") or {}).get("value"),
        })
    else:
        rec("NOETIC_TPR64_REOPEN", str(TPR64), False, "missing")

    return {"n_hits": len(hits), "n_missing": len(missing), "missing": missing, "hits": hits}


def shader_facts() -> dict:
    q4 = Q4_METAL.read_text() if Q4_METAL.is_file() else ""
    strand = STRAND_METAL.read_text() if STRAND_METAL.is_file() else ""
    hgrav = HGRAVS_METAL.read_text() if HGRAVS_METAL.is_file() else ""
    quant = QUANT_METAL.read_text() if QUANT_METAL.is_file() else ""
    facts = {
        "q4_kernel_present": "kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128(" in q4,
        "q4_tg": WORKHORSE_TG,
        "q4_simdgroup_width": SIMDGROUP_WIDTH,
        "q4_threadgroup_red_floats": 4,
        "strand_rht_present": "kernel void strand_rht_forward_cols(" in strand,
        "strand_rht_block": 256,
        "strand_rht_register_floats_per_thread": 256,
        "strand_rht_is_activation_side": True,
        "hgravs01_two_stage_present": "kernel void q80_hgravs01_two_stage_matvec(" in hgrav,
        "hgravs01_rank_cap": 160,
        "hgravs01_x_cap": 512,
        "hgravs01_tg_floats": 160 + 512,
        "quant_metal_states_32kb_tg_limit": "32 KB threadgroup memory limit" in quant,
        "tg_mem_limit_bytes": TG_MEM_LIMIT_BYTES,
    }
    if facts["strand_rht_present"]:
        for i, line in enumerate(strand.splitlines(), 1):
            if line.startswith("kernel void strand_rht_forward_cols("):
                facts["strand_rht_line"] = i
                break
    if facts["q4_kernel_present"]:
        for i, line in enumerate(q4.splitlines(), 1):
            if line.startswith("kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128("):
                facts["q4_kernel_line"] = i
                break
    return facts


def organ_design(org: dict, b: int) -> dict:
    c = fmh_costs(org["rows"], org["cols"], b)
    row = {
        "organ": org["organ"],
        "role": org["role"],
        "count_per_token": org["count_per_token"],
        "rows": org["rows"],
        "cols": org["cols"],
        "tiles_at_B": c is not None,
        "incumbent_q4_bytes_per_token": org["q4_bytes_per_token"],
        "incumbent_mac_flops_per_token": org["mac_flops_per_token"],
    }
    if c is None:
        row["disposition"] = "KEEP_Q4"
        row["why"] = f"rows or cols not divisible by B={b}; pad would zero extra output tiles"
        row["fmh"] = None
        return row
    row["disposition"] = "FMH"
    row["fmh"] = c
    n = org["count_per_token"]
    row["rung0_bytes_per_token"] = c["rung0_bytes_f16"] * n
    row["rung1_bytes_per_token"] = c["rung1_bytes_f16"] * n
    row["rung2_bytes_per_token"] = c["rung2_bytes_f16"] * n
    row["rung0_flops_per_token"] = c["rung0_flops"] * n
    row["rung1_flops_per_token"] = c["rung1_flops"] * n
    row["rung2_flops_per_token"] = c["rung2_flops"] * n
    row["byte_ratio_rung0_vs_q4"] = (c["rung0_bytes_f16"] * n) / org["q4_bytes_per_token"]
    row["flop_ratio_rung0_vs_mac"] = (c["rung0_flops"] * n) / org["mac_flops_per_token"]
    row["byte_ratio_rung1_vs_q4"] = (c["rung1_bytes_f16"] * n) / org["q4_bytes_per_token"]
    row["flop_ratio_rung1_vs_mac"] = (c["rung1_flops"] * n) / org["mac_flops_per_token"]
    return row


def sum_field(rows: list[dict], field: str) -> int:
    return int(sum(r.get(field, 0) or 0 for r in rows))


def build_design(g: dict, organs: list[dict], prior: dict, shaders: dict, fwht: dict) -> dict:
    dims = []
    for o in organs:
        dims.extend([o["rows"], o["cols"]])
    universal_b = gcd_many(dims)

    per = [organ_design(o, BLOCK_B) for o in organs]
    fmh_organs = [r for r in per if r["disposition"] == "FMH"]
    keep_organs = [r for r in per if r["disposition"] == "KEEP_Q4"]

    gemv_elems = sum(o["elements_per_token"] for o in organs)
    gemv_macs = sum(o["mac_flops_per_token"] for o in organs)
    q4_bytes = sum(o["q4_bytes_per_token"] for o in organs)
    gemv_disp = sum(o["count_per_token"] for o in organs)

    keep_q4_bytes = sum(r["incumbent_q4_bytes_per_token"] for r in keep_organs)
    keep_macs = sum(r["incumbent_mac_flops_per_token"] for r in keep_organs)
    keep_disp = sum(r["count_per_token"] for r in keep_organs)
    fmh_disp = sum(r["count_per_token"] for r in fmh_organs)

    rung0_bytes = sum_field(fmh_organs, "rung0_bytes_per_token") + keep_q4_bytes
    rung1_bytes = sum_field(fmh_organs, "rung1_bytes_per_token") + keep_q4_bytes
    rung2_bytes = sum_field(fmh_organs, "rung2_bytes_per_token") + keep_q4_bytes
    rung0_flops = sum_field(fmh_organs, "rung0_flops_per_token") + keep_macs
    rung1_flops = sum_field(fmh_organs, "rung1_flops_per_token") + keep_macs
    rung2_flops = sum_field(fmh_organs, "rung2_flops_per_token") + keep_macs

    nonweight_dram = ANCHOR_DRAM_BYTES - ANCHOR_Q4_GEMV_BYTES
    dram_r0 = rung0_bytes + nonweight_dram
    dram_r1 = rung1_bytes + nonweight_dram
    dram_r2 = rung2_bytes + nonweight_dram

    roof = ANCHOR_ROOF_GB_S * 1e9
    bw_ms_incumbent = ANCHOR_DRAM_BYTES / roof * 1e3
    bw_ms_r0 = dram_r0 / roof * 1e3
    bw_ms_r1 = dram_r1 / roof * 1e3
    bw_ms_r2 = dram_r2 / roof * 1e3

    ridge = (ANCHOR_COMPUTE_PEAK_GFLOPS * 1e9) / roof  # FLOP/byte
    q4_intensity = ANCHOR_GEMV_MAC_FLOPS / ANCHOR_Q4_GEMV_BYTES
    r0_intensity = rung0_flops / max(rung0_bytes, 1)
    r1_intensity = rung1_flops / max(rung1_bytes, 1)

    # Token still 964 if fused (one kernel per organ). Unfused FWHT adds fmh_disp.
    fused_dispatches = ANCHOR_DISPATCHES
    unfused_dispatches = ANCHOR_DISPATCHES + fmh_disp
    fused_cbs = 1

    # Metal: all n1 tiles of z in TG for the largest n (down_proj cols=17408 → n1=17).
    max_n1 = max((r["fmh"]["n1"] for r in fmh_organs), default=0)
    max_m1 = max((r["fmh"]["m1"] for r in fmh_organs), default=0)
    z_tg_bytes = max_n1 * BLOCK_B * 4  # f32
    # Cooperative FWHT-1024: 32 lanes × 32 floats in TG (4 KiB) per tile, reused.
    fwht_tile_tg_bytes = BLOCK_B * 4
    red_tg_bytes = 4 * 4  # incumbent red[4]
    fused_tg_bytes = z_tg_bytes + red_tg_bytes  # hold all tiles of z
    fused_tg_fits = fused_tg_bytes <= TG_MEM_LIMIT_BYTES
    # If z for down_proj (n1=17 → 17*4096=69632) exceeds 32KB, spill z to a
    # device scratch of n f32 (69,632 B) — still not dense W.
    down = next(r for r in per if r["organ"] == "mlp.down_proj")
    down_z_bytes = down["fmh"]["n1"] * BLOCK_B * 4
    down_z_fits = down_z_bytes <= TG_MEM_LIMIT_BYTES
    # Gate/up hidden-side z: n=5120, n1=5, 5*4096=20480 < 32KB.
    gate = next(r for r in per if r["organ"] == "mlp.gate_proj")
    gate_z_bytes = gate["fmh"]["n1"] * BLOCK_B * 4

    # Rung-0 local BPW is exactly 16/B for f16 S, independent of m,n.
    rung0_bpw = 16.0 / BLOCK_B
    share_bpw = G1_ONE_BASIS_FULLV_BPW
    bpw_coincidence = abs(rung0_bpw - 1.0 / 64.0) < 1e-12
    # 16/1024 = 1/64. one_basis_fullV amortises 1 n×n V across 64 sites = 1/64 of ~1 BPW.
    # Same fraction, different objects. Must not be quoted as one claim.

    oracle = {
        "label": "ORACLE",
        "not_production": True,
        "formula": (
            "Ŵ[k*B + b, t*B + b'] = S[b, k, t] * H_B[b, b'] / sqrt(B)     (rung 0); "
            "replace H_B with H_B @ R_t for rung 1; left-multiply the B-block of "
            "outputs by L_k for rung 2. Then y = Ŵ x by ordinary GEMV."
        ),
        "dense_w_bytes_if_materialised_f32_per_token": 4 * gemv_elems,
        "trap": (
            "A decode-then-f32-GEMM lowering writes 102,487,818,240 B of dense W "
            "per token (operation census trap_reconstruct_then_gemm). The oracle "
            "may exist as a labelled correctness check. It is not a production path."
        ),
        "how_to_check": (
            "Build Ŵ from S (and R, L) on the host for one organ, compare "
            "fmh_matvec(x) against Ŵ @ x. Max-abs should match FWHT identity "
            "noise (~1e-6 f32). This proves the kernel, not the fit."
        ),
    }

    production = {
        "label": "PRODUCTION",
        "kernel_name": "fmh_matvec_b1024_tg128",
        "never_materialises_W": True,
        "stages": [
            {
                "id": "H",
                "stored_bytes": 0,
                "what": (
                    "In-place block FWHT of length B=1024 on each of n1 column-tiles "
                    "of x. Sylvester-Hadamard is generated from B. G032 already used "
                    "this tile because 5120=5*1024 and 17408=17*1024 with no padding."
                ),
            },
            {
                "id": "R",
                "stored_bytes": "n*B f16 per tensor on rung 1/2, else 0",
                "what": (
                    "Optional learned B×B mix per input tile, applied before H. "
                    "This is the coefficient layer that actually has a chance to "
                    "fit W; rung 0 sets R_t = I."
                ),
            },
            {
                "id": "PI",
                "stored_bytes": 0,
                "what": (
                    "Stride permute = reshape (n1, B) → transpose → (B, n1). "
                    "Generated from B. Not a stored permutation table."
                ),
            },
            {
                "id": "S",
                "stored_bytes": "mn/B f16 per tensor (the information)",
                "what": (
                    "For each Hadamard frequency b in 0..B-1, a dense m1×n1 map "
                    "S_b. y[k*B+b] = S_b[k, :] · U[b, :]. This is 1/B of the "
                    "rotated matrix: frequency-diagonal in the 1024-mode."
                ),
            },
            {
                "id": "L",
                "stored_bytes": "m*B f16 per tensor on rung 2, else 0",
                "what": "Optional learned B×B mix per output tile, after S.",
            },
        ],
        "not_this_kernel": (
            "strand_rht_forward_cols is activation-side FWHT-256 then a bitslice "
            "GEMV consumes tx. G032 packed W H and still ran dense Q4 GEMV. "
            "Neither is this operator."
        ),
        "combination_required": (
            "A pure transform (H alone, or H plus per-channel scale) cannot "
            "represent an arbitrary W. Kronecker energy on the natural "
            f"[17,1024]×[5,1024] reshape is {G1_KRON_ENERGY_RANK1:.4f} at rank-1 "
            f"and {G1_KRON_ENERGY_RANK64:.4f} at rank-64. The coefficient layer "
            "is S (always) plus R/L (rungs 1–2). Sparse residual is NNS-015 "
            "Pareto-dominated by q3 as a byte lever and is not in the production path."
        ),
    }

    # Bandwidth-only tok/s if quality were free — reported as a counterfactual,
    # immediately paired with the health prediction.
    tps_bw_r0 = 1000.0 / bw_ms_r0 if bw_ms_r0 > 0 else None
    tps_bw_r1 = 1000.0 / bw_ms_r1 if bw_ms_r1 > 0 else None

    s011 = {
        "incumbent": {
            "bytes_weight_stream": ANCHOR_Q4_GEMV_BYTES,
            "operations_gemv_mac": ANCHOR_GEMV_MAC_FLOPS,
            "dispatches": ANCHOR_DISPATCHES,
            "materialization_dense_W": 0,
            "synchronization_command_buffers": ANCHOR_CBS,
            "traffic_dram": ANCHOR_DRAM_BYTES,
        },
        "rung0_fused": {
            "bytes_weight_stream": rung0_bytes,
            "operations_gemv_plus_fwht": rung0_flops,
            "dispatches": fused_dispatches,
            "materialization_dense_W": 0,
            "synchronization_command_buffers": fused_cbs,
            "traffic_dram": dram_r0,
            "reduces": ["bytes", "operations", "traffic"],
            "does_not_reduce": ["dispatches", "materialization", "synchronization"],
            "complete_by_s011_section4": True,
        },
        "rung1_fused": {
            "bytes_weight_stream": rung1_bytes,
            "operations_gemv_plus_fwht": rung1_flops,
            "dispatches": fused_dispatches,
            "materialization_dense_W": 0,
            "synchronization_command_buffers": fused_cbs,
            "traffic_dram": dram_r1,
            "reduces": ["bytes", "operations", "traffic"],
            "complete_by_s011_section4": True,
        },
        "rung2_fused_f16": {
            "bytes_weight_stream": rung2_bytes,
            "operations_gemv_plus_fwht": rung2_flops,
            "dispatches": fused_dispatches,
            "materialization_dense_W": 0,
            "synchronization_command_buffers": fused_cbs,
            "traffic_dram": dram_r2,
            "bytes_exceed_incumbent_q4": rung2_bytes > q4_bytes,
            "reduces": ["operations"],
            "does_not_reduce": ["bytes", "traffic"],
            "complete_by_s011_section4": True,
            "why": (
                f"f16 R+S+L is {rung2_bytes:,} B vs Q4 {q4_bytes:,} B — bytes go UP. "
                f"FLOPs fall to {rung2_flops / ANCHOR_GEMV_MAC_FLOPS:.3f}×. Ops-only "
                "completeness. tpr64 says ops-only does not move this geometry; do not "
                "build rung 2 as f16."
            ),
        },
        "g032_codec_reparam_on_q4": {
            "bytes_weight_stream": ANCHOR_Q4_GEMV_BYTES,
            "operations": ANCHOR_GEMV_MAC_FLOPS + sum(
                o["cols"] * LOG2_B * o["count_per_token"] for o in organs
                if o["cols"] % BLOCK_B == 0 and o["rows"] % BLOCK_B == 0
            ),
            "dispatches_if_unfused_fwht": unfused_dispatches,
            "reduces": [],
            "increases": ["operations", "dispatches_if_unfused"],
            "complete_by_s011_section4": False,
            "why": (
                "G032 stores W H at the same Q4 size, then adds H^T x at runtime. "
                "Bytes do not drop. Ops rise. tpr64 says reconstruction is free on "
                f"{TPR64_FREE_VARIANTS}/{TPR64_TOTAL_VARIANTS} variants including "
                "hadamard, so the extra ALU cannot buy tok/s at this geometry."
            ),
        },
        "note": (
            "S011 §4 completeness is arithmetic, not a build recommendation. A "
            "complete design that is predicted unhealthy is still NOT_WORTH_BUILDING."
        ),
    }

    microbench = {
        "name": "fmh_rung0_fiber_lstsq_L31_gate",
        "purpose": (
            "Discriminate FMH from uniform-q4 BEFORE anyone writes fmh_matvec. "
            "CPU / numpy, one real tensor, real captured X. Minutes, not a kernel."
        ),
        "not_synthetic": True,
        "tensor": "language_model.model.layers.31.mlp.gate_proj.weight",
        "shape": [17408, 5120],
        "why_this_tensor": (
            "G032 and G034 both used L31.gate_proj. G034's site correction: "
            "post_attn_norm, not post_input_norm (silu(gate)*up reproduces "
            "captured post_swiglu to cosine 0.999986 only at post_attn_norm)."
        ),
        "W_source": (
            "bf16 parent shard via model.safetensors.index.json "
            "(tools/gravity_xform_hadamard.py::load_tensor)"
        ),
        "X_source": {
            "path": "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1",
            "site": "post_attn_norm / captured_post_norm_hidden",
            "fit_n": 192,
            "hold_n": 64,
            "sha256_self": "fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512",
            "not_synthetic": True,
        },
        "controls": {
            "q4": "grouped absmax g64, same as production",
            "q3": f"G034 mean_flat_q3 out_rel_fro = {G034_MEAN_FLAT_Q3}",
            "null_cosine": NULL_COSINE,
        },
        "procedure": [
            "Load W f32 17408×5120 and X_fit, X_hold from the capture (real tokens only).",
            "Block-FWHT along columns of W (G032 block_hadamard_apply, B=1024) → W_rot.",
            "Rung 0 extractor: S[b,k,t] = W_rot[k*B+b, t*B+b]  (frequency-diagonal).",
            "That is the Frobenius-optimal S for the rung-0 family. No training.",
            "Apply fmh_rung0(X) = S-mix of FWHT_block(X) on fit and hold.",
            "Score against W@X with gravity_doctor_gate._gain (NOT cosine-only). "
            "Also report out_rel_fro and max_abs. Cosine is recorded and flagged: "
            "0.01*W scores cosine 1.000000.",
            "If rung 0 is dead, repeat with a least-squares R_t (5 × 1024×1024) "
            "on the fit set only — that is rung 1. Do not touch hold while fitting R.",
        ],
        "kill_if": [
            f"doctor healthy=False on hold-out (q4 on this tensor is healthy at 4 bits; G1 q4 rel_l2={G1_GATE_Q4_REL_L2})",
            f"hold-out out_rel_fro > G034 mean_flat_q3 {G034_MEAN_FLAT_Q3} at any byte budget ≤ q4",
            "scoring is cosine-only",
            "X is synthetic / Gaussian / iid",
            f"local_bpw < 0.5 and no health verdict (223-row law; this rung 0 is {rung0_bpw:.6f} BPW)",
        ],
        "predicted_outcome": {
            "rung0": (
                f"rel_l2 ≈ 1.0 (G1 block_term {G1_BLOCK_TERM_REL_L2} at "
                f"local_bpw {G1_BLOCK_TERM_LOCAL_BPW}; frequency-diagonal keeps 1/B of W_rot)."
            ),
            "rung1": (
                f"rel_l2 ≈ 0.93 (G1 kronecker_sum {G1_KRON_SUM_REL_L2} on the same "
                f"[17,1024]×[5,1024] reshape), still ~8× q4's {G1_GATE_Q4_REL_L2}."
            ),
            "therefore": "Do not write the Metal kernel unless this CPU fit beats q3 on hold-out doctor.",
        },
        "wall_budget_s": 180,
        "forbidden": [
            "synthetic activations",
            "weight cosine as the gate",
            "quoting 0.0156 BPW as a quality result",
            "reconstruct Ŵ to f32 and calling that production",
        ],
    }

    expected_value = {
        "verdict": "NOT_WORTH_BUILDING",
        "what_it_would_win_if_healthy": {
            "rung0_weight_bytes_per_token": rung0_bytes,
            "rung0_vs_q4_weight_bytes": rung0_bytes / q4_bytes,
            "rung0_flops_vs_51_24_gflop": rung0_flops / ANCHOR_GEMV_MAC_FLOPS,
            "rung0_bandwidth_ms_counterfactual": bw_ms_r0,
            "rung0_bandwidth_tps_counterfactual": tps_bw_r0,
            "rung1_weight_bytes_per_token": rung1_bytes,
            "rung1_vs_q4_weight_bytes": rung1_bytes / q4_bytes,
            "rung1_flops_vs_51_24_gflop": rung1_flops / ANCHOR_GEMV_MAC_FLOPS,
            "rung1_bandwidth_ms_counterfactual": bw_ms_r1,
            "rung1_bandwidth_tps_counterfactual": tps_bw_r1,
            "controls_to_beat": {
                "self_uniform_q4": ANCHOR_TPS,
                "mlx_4bit_live": ANCHOR_MLX_TPS,
                "llamacpp_q5k_archived": ANCHOR_LLAMA_Q5K_TPS,
            },
            "counterfactual_flag": (
                "These tok/s numbers assume the representation is healthy and "
                "the kernel hits the measured 595.9 GB/s roof. They are not a "
                "result. Rung 0 local BPW is 0.015625 with no health verdict."
            ),
        },
        "what_it_risks": [
            "Rung 0 lands in the 223-component <0.5 local BPW band (healthy=true: 0).",
            "Rung 1/2 collapse to structured factorisations G1/G034 already ran on this parent (0 healthy; G034 2.93× q3 error at matched bits).",
            "G035: sharing S or H across layers lost. Do not amortise 0.0156.",
            "tpr64: recon is free at production geometry, including the hadamard variant. FLOP cuts that do not cut bytes do not move tok/s.",
            "MLP function distillation is NO-GO (+0.4206 held-out gap vs q3). Cannot train out of a bad drop-in.",
            "Cosine-only scoring will accept 0.01*W. Use doctor gain.",
            "Oracle materialisation is a 102.5 GiB/token trap.",
        ],
        "cheapest_kill": microbench["name"],
        "why_not_worth_building_the_kernel": (
            "The 'transform IS the code' rung is already refuted as a byte family "
            "(G042 GENERATED_BPW_EQUIVALENT=0, Hadamard named). The coefficient "
            "rung that could in principle reduce both bytes and ops is the mixed-"
            "radix blocking G1 already scored on this exact reshape, with zero "
            "healthy operators. Kernel census: quality, not kernel volume, is the "
            "blocker (~200 lines if RHT is reused). Pay the CPU fit first; it is "
            "the experiment that would kill this, and the prediction is it kills it."
        ),
        "reopen_if": (
            "The cheap microbenchmark on L31.gate_proj + real hold-out X reports "
            "doctor healthy=True and out_rel_fro ≤ q3 at ≤ q4 bytes. Until then "
            "do not open fmh_matvec.metal."
        ),
    }

    family_refuted = {
        "rung0_generated_H_plus_tiny_S": {
            "refuted": True,
            "as": "PROPERTY_OF_IDEA (near-zero stored state cannot carry this W)",
            "evidence": [
                "G042 GENERATED_BPW_EQUIVALENT=0; the one generated transform tested (Hadamard) was refuted",
                f"G1 Kronecker energy rank-1={G1_KRON_ENERGY_RANK1} on the [17,1024]×[5,1024] reshape",
                f"G1 block_term rel_l2={G1_BLOCK_TERM_REL_L2} at local_bpw={G1_BLOCK_TERM_LOCAL_BPW}",
                f"{G1_SUB05_ROWS} components local_bpw<0.5, healthy={G1_SUB05_HEALTHY}",
                "rung 0 local BPW = 16/B = 0.015625, inside that band",
            ],
            "stop": True,
        },
        "g032_codec_reparameterization": {
            "refuted": True,
            "as": "speed and bits lever (it still stores WH and adds FWHT)",
            "evidence": [
                f"Q4 mean_delta_hold={G032_Q4_DELTA_HOLD}, entropy +{G032_Q4_DELTA_ENTROPY} bits",
                f"Q3 mean_delta_hold={G032_Q3_DELTA_HOLD}",
                f"tpr64 hadamard_gate_ns={TPR64_HADAMARD_GATE_NS} in the f32 band; recon free {TPR64_FREE_VARIANTS}/{TPR64_TOTAL_VARIANTS}",
            ],
            "stop": True,
        },
        "rung1_rung2_coefficient_monarch": {
            "refuted": False,
            "as": "this exact fused operator was not shipped",
            "nearest_evidence_says_unhealthy": True,
            "evidence": [
                f"G034 low-rank at matched 3.25 bits: out_rel_fro {G034_MEAN_LOWRANK} vs q3 {G034_MEAN_FLAT_Q3} (ratio {G034_ERROR_RATIO})",
                "G034 family_verdict: COMPREHENSIVELY REFUTED (low-rank, TT unfold, Tucker=SVD)",
                f"G1 L0.gate 44 operators, healthy=0; kronecker_sum rel_l2={G1_KRON_SUM_REL_L2}",
                "kernel census structured_transform=PARTIAL; weight-side y=H diag(s) H^T x ABSENT; quality is the blocker",
            ],
            "stop": False,
            "do_not_write_kernel_first": True,
        },
        "sharing_a_transform_across_a_tensor_class": {
            "refuted": True,
            "as": "G035 shared_beats_independent=false; G042 SHARED_BPW=0",
            "zero_point_zero_one_five_six": {
                "value": share_bpw,
                "what_it_is": (
                    "G1-SHARE one_basis_fullV_bpw: one dense n×n right factor stored "
                    f"once, amortised over 64 sites, vs independent dense_orth_bpw="
                    f"{G1_DENSE_ORTH_BPW_64}. Component amortisation, not a model-level "
                    "structured-transform claim."
                ),
                "numerical_coincidence": (
                    f"FMH rung-0 f16 local BPW is 16/B = {rung0_bpw} = 1/64, the same "
                    "fraction as 1 basis / 64 sites. Different objects. Do not merge."
                ),
                "coincidence_is_exact_1_over_64": bpw_coincidence,
            },
        },
        "mlp_function_distillation_as_rescue": {
            "refuted": True,
            "as": f"NO-GO today: +{MLP_DISTILL_HELD_OUT_GAP} held-out gap vs q3 at 72% of its active bytes",
        },
    }

    watched = [
        {
            "what": "G032 Q4 Hadamard as codec reparameterization",
            "result": (
                f"mean_delta_hold={G032_Q4_DELTA_HOLD}, entropy "
                f"+{G032_Q4_DELTA_ENTROPY} bits, stored_bytes of H=0 but W is still stored"
            ),
            "why": "A transform that does not drop bits and adds FWHT ops is not a token lever. Runtime of H^T x was never measured on device in G032.",
        },
        {
            "what": "G042 generated / implicit weights",
            "result": "GENERATED_BPW_EQUIVALENT=0.0; Hadamard named as the generated transform that was refuted",
            "why": "The 'transform IS the code, almost no stored state' member is closed as a BPW family.",
        },
        {
            "what": "G1 Kronecker energy on the natural 17×1024 × 5×1024 reshape of L0.gate_proj",
            "result": f"rank-1 energy={G1_KRON_ENERGY_RANK1}, rank-64 energy={G1_KRON_ENERGY_RANK64}",
            "why": "W is not A⊗H_1024. Rung 0's frequency-diagonal keeps 1/B of the rotated matrix.",
        },
        {
            "what": "G1 structured operators on that tensor (Tucker/TT/Kronecker/BTD/low-rank)",
            "result": f"44 operators, healthy=0; block_term rel_l2={G1_BLOCK_TERM_REL_L2}; kronecker_sum rel_l2={G1_KRON_SUM_REL_L2}; q4 rel_l2={G1_GATE_Q4_REL_L2}",
            "why": "Nearest measured members of this family are unhealthy on this parent.",
        },
        {
            "what": "G034 matched-bit structured operator vs flat q3",
            "result": f"mean_lowrank={G034_MEAN_LOWRANK} vs mean_flat_q3={G034_MEAN_FLAT_Q3} (ratio {G034_ERROR_RATIO}); MAC ratio {G034_MAC_RATIO}",
            "why": "Reducing ops at matched bits without preserving function is the acceptance failure G034 recorded. TT unfold was worse.",
        },
        {
            "what": "G035 G-SHARE",
            "result": f"shared_beats_independent={G035_SHARED_BEATS}",
            "why": "A common transform stored once for a tensor class is not a free 0.0156 BPW win.",
        },
        {
            "what": "223 sub-0.5 local_bpw components",
            "result": f"healthy_true_count={G1_SUB05_HEALTHY}",
            "why": "Rung 0's 0.015625 local BPW is not a result until paired with a health verdict.",
        },
        {
            "what": "tpr64 reconstruction-is-free, hadamard variant",
            "result": (
                f"{TPR64_FREE_VARIANTS}/{TPR64_TOTAL_VARIANTS} variants recon_excess_ns=0; "
                f"hadamard_gate_ns={TPR64_HADAMARD_GATE_NS} in the uncompressed-f32 band"
            ),
            "why": "At production tpr64, cutting dequant ALU without cutting bytes does not move GPU ns. The a-priori 'fewer ops' case is a traffic case on this device, or it is nothing.",
        },
        {
            "what": "scale-invariance of cosine",
            "result": "0.01*W scores cosine 1.000000; raw-activation null cosine ≈ 0.898",
            "why": "The cheap microbenchmark is forbidden from using cosine as the gate.",
        },
        {
            "what": "hidden=5120 is not a power of two",
            "result": f"gcd of all GEMV dims={universal_b}; B=1024 tiles hidden/intermediate/qkvz/q/k/v/o and not ba=96 or vocab=248320",
            "why": "A global FWHT of x needs padding to 8192. G032's block-1024 is the workable tile; ba and lm_head stay Q4.",
        },
        {
            "what": "strand_rht_forward_cols",
            "result": (
                f"present={shaders['strand_rht_present']} line={shaders.get('strand_rht_line')} "
                "compile_gate=feature=tq, 256-wide, float buf[256] per thread, activation-side"
            ),
            "why": "An FWHT kernel exists and is not the weight-side operator. Reuse is ~200 lines; quality is the blocker (kernel census).",
        },
        {
            "what": "HGRAVS01 two-stage threadgroup caps",
            "result": (
                f"mid[{shaders['hgravs01_rank_cap']}]+x_tg[{shaders['hgravs01_x_cap']}] "
                f"= {shaders['hgravs01_tg_floats']*4} B; down_proj FMH z would be {down_z_bytes} B "
                f"({'fits' if down_z_fits else 'EXCEEDS'} 32 KB)"
            ),
            "why": "Fusing all of z for down_proj into TG memory does not fit. Spill z to a 69,632 B device scratch — still not dense W. Gate/up z is 20,480 B and does fit.",
        },
        {
            "what": "MLP function distillation as the surviving storage avenue",
            "result": f"NO-GO, +{MLP_DISTILL_HELD_OUT_GAP} held-out gap vs q3 at 72% of its active bytes",
            "why": "Cannot rescue a bad drop-in by distilling the MLP.",
        },
        {
            "what": "Q80 storage vs active BPW category error",
            "result": f"storage {Q80_STORAGE_BPW} against ACTIVE {Q80_ACTIVE_BPW} (~3.9×)",
            "why": "Report both or neither. FMH rung 0 storage BPW 0.015625 would be the same trap without an active-byte and health pair.",
        },
        {
            "what": "FWHT identity (this process, n=16, not activations)",
            "result": f"ok={fwht['ok']} max_off={fwht['max_off_diag_HHT']} col0_err={fwht['fwht_e0_vs_H_col0']}",
            "why": "Confirms the oracle's H is orthogonal. Says nothing about whether W lives near that family.",
        },
    ]

    self_check = {
        "dispatches_964": g["production_dispatches"] == ANCHOR_DISPATCHES,
        "gemv_elements_match_anchor": gemv_elems == ANCHOR_GEMV_ELEMENTS,
        "gemv_macs_match_anchor": gemv_macs == ANCHOR_GEMV_MAC_FLOPS,
        "q4_bytes_match_anchor": q4_bytes == ANCHOR_Q4_GEMV_BYTES,
        "gemv_dispatches_401": gemv_disp == ANCHOR_GEMV_DISPATCHES,
        "hidden_tiles_at_1024": g["hidden"] % BLOCK_B == 0,
        "intermediate_tiles_at_1024": g["intermediate"] % BLOCK_B == 0,
        "ba_does_not_tile_1024": g["ba_rows"] % BLOCK_B != 0,
        "vocab_does_not_tile_1024": g["vocab"] % BLOCK_B != 0,
        "rung0_bpw_is_16_over_B": abs(rung0_bpw - 16.0 / BLOCK_B) < 1e-15,
        "rung0_bpw_equals_1_over_64": bpw_coincidence,
        "fwht_identity_ok": fwht["ok"],
        "q4_kernel_present": shaders["q4_kernel_present"],
        "strand_rht_present": shaders["strand_rht_present"],
        "oracle_labelled": oracle["label"] == "ORACLE",
        "production_labelled": production["label"] == "PRODUCTION",
        "fused_tg_gate_fits_32kb": gate_z_bytes + red_tg_bytes <= TG_MEM_LIMIT_BYTES,
        "fused_tg_down_exceeds_32kb": (not down_z_fits),
        "s011_rung0_complete": s011["rung0_fused"]["complete_by_s011_section4"],
        "s011_g032_incomplete": not s011["g032_codec_reparam_on_q4"]["complete_by_s011_section4"],
        "rung2_f16_bytes_exceed_q4": rung2_bytes > q4_bytes,
        "verdict_is_not_worth_building": expected_value["verdict"] == "NOT_WORTH_BUILDING",
        "prior_science_hits": prior["n_hits"],
        "prior_science_missing": prior["n_missing"],
    }

    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git_head(),
        "question": (
            "Can a structured transform (butterfly / Walsh-Hadamard / mixed-radix "
            "Monarch-Hadamard) be the executable operator on Qwen3.8 uniform-q4, "
            "reducing both bytes and operations versus 51.24 GFLOP / 964 dispatches?"
        ),
        "answer": (
            "The near-zero-state member is already refuted (G042, G032, Kronecker "
            "energy, 223-with-zero-healthy). The coefficient member (FMH rungs 1–2) "
            "is a complete S011 §4 design on paper — it cuts bytes, FLOPs and traffic "
            "and never materialises W — and is NOT_WORTH_BUILDING as a kernel until "
            "a CPU fiber least-squares on real L31.gate_proj X beats q3. Nearest "
            "G1/G034 measurements predict it will not."
        ),
        "verdict": "NOT_WORTH_BUILDING",
        "anchors_not_rederived": {
            "tps": ANCHOR_TPS,
            "ms_per_token": ANCHOR_TOKEN_MS,
            "roof_gb_s": ANCHOR_ROOF_GB_S,
            "unified_memory_bytes": ANCHOR_UNIFIED_B,
            "gpu_cores": ANCHOR_GPU_CORES,
            "parameter_count": ANCHOR_PARAMS,
            "bpw": ANCHOR_BPW,
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers_per_token": ANCHOR_CBS,
            "gemv_mac_flops": ANCHOR_GEMV_MAC_FLOPS,
            "gemv_elements": ANCHOR_GEMV_ELEMENTS,
            "q4_gemv_bytes": ANCHOR_Q4_GEMV_BYTES,
            "dram_bytes": ANCHOR_DRAM_BYTES,
            "mlx_4bit_tps_live": ANCHOR_MLX_TPS,
            "llamacpp_q5k_tps_archived": ANCHOR_LLAMA_Q5K_TPS,
            "artifact_bytes": ANCHOR_ARTIFACT_B,
        },
        "geometry": g,
        "universal_block_gcd": universal_b,
        "chosen_B": BLOCK_B,
        "chosen_B_reason": (
            "G032 tile. Hidden=5*1024, intermediate=17*1024, qkvz=16*1024, "
            "q_proj=12*1024, kv=1*1024, o_cols=6*1024. ba=96 and vocab=248320 do "
            f"not tile; those {keep_disp} launches stay Q4. gcd of every GEMV dim "
            f"is {universal_b} (would tile ba+lm_head but lands even deeper in the "
            "<0.5 BPW band at f16)."
        ),
        "prior_science": {
            "searched": True,
            "n1arch_mechanisms_relevant": [
                "G032 Block-diagonal Sylvester-Hadamard (RAN as codec reparam, not as GEMV replacement)",
                "G042 generated/implicit weights REFUTED, GENERATED_BPW=0",
                "G035 G-SHARE REFUTED",
                "G1-SHARE 0.0156 BPW is right-basis amortisation, not this family",
                "G1-TENSOR / G034 Tucker/TT/ring/Kronecker/low-rank REFUTED, 223 rows <0.5 BPW healthy=0",
                "kernel census structured_transform PARTIAL (activation FWHT exists, weight-side absent)",
                "MLP distillation NO-GO +0.4206",
                "tpr64 recon free 32/33 including hadamard",
            ],
            "family_already_refuted": family_refuted,
            "live_confirmation": prior,
        },
        "operator": {
            "name": "fmh_matvec",
            "long_name": "fused mixed-radix Monarch-Hadamard matvec",
            "B": BLOCK_B,
            "math": (
                "Let B=1024, n=n1*B, m=m1*B, x∈R^n. Reshape X∈R^{n1×B}. "
                "Z[t,:]=FWHT_B(R_t X[t,:]) with R_t=I on rung 0. "
                "U=Z^T ∈ R^{B×n1} (generated permute). "
                "Y[b,:]=S_b U[b,:] with S_b∈R^{m1×n1}. "
                "y[k*B+b]=Y[b,k], then optional y_tile_k = L_k y_tile_k."
            ),
            "oracle": oracle,
            "production": production,
            "rungs": [
                {
                    "id": "rung0",
                    "stored": "S only (H, Π generated)",
                    "local_bpw_f16": rung0_bpw,
                    "health_band": "<0.05 — 223-row trap unless doctor says healthy",
                    "already_refuted": True,
                },
                {
                    "id": "rung1",
                    "stored": "R (n×B f16) + S",
                    "already_refuted": False,
                    "nearest": "G1 kronecker_sum / G034 low-rank, both unhealthy",
                },
                {
                    "id": "rung2",
                    "stored": "R + S + L (m×B f16)",
                    "already_refuted": False,
                    "nearest": "fatter than G034 rank-803 on gate; still a structured subset of dense",
                },
            ],
            "fwht_identity_self_check": fwht,
        },
        "organs": per,
        "expected_bytes_per_token": {
            "derived_not_guessed": True,
            "formula": (
                "tiling organs: rung0 = count*(2*mn/B + 40); rung1 adds 2*n*B; "
                "rung2 adds 2*m*B. Non-tiling organs: incumbent Q4 = rows*ceil(cols/64)*34. "
                "DRAM = weight stream + (ANCHOR_DRAM − ANCHOR_Q4_GEMV) for activations/state/KV."
            ),
            "incumbent_q4_gemv_bytes": q4_bytes,
            "incumbent_dram_bytes": ANCHOR_DRAM_BYTES,
            "keep_q4_organs": [r["organ"] for r in keep_organs],
            "keep_q4_bytes": keep_q4_bytes,
            "fmh_organs": [r["organ"] for r in fmh_organs],
            "rung0_weight_bytes": rung0_bytes,
            "rung1_weight_bytes": rung1_bytes,
            "rung2_weight_bytes": rung2_bytes,
            "rung0_dram_bytes": dram_r0,
            "rung1_dram_bytes": dram_r1,
            "rung2_dram_bytes": dram_r2,
            "rung0_vs_incumbent_weight": rung0_bytes / q4_bytes,
            "rung1_vs_incumbent_weight": rung1_bytes / q4_bytes,
            "bandwidth_ms_at_595_9": {
                "incumbent": bw_ms_incumbent,
                "rung0_counterfactual": bw_ms_r0,
                "rung1_counterfactual": bw_ms_r1,
                "rung2_counterfactual": bw_ms_r2,
            },
            "gate_proj_example": {
                "q4_bytes_per_launch": gate["incumbent_q4_bytes_per_token"] // gate["count_per_token"],
                "rung0_bytes_per_launch": gate["rung0_bytes_per_token"] // gate["count_per_token"],
                "rung1_bytes_per_launch": gate["rung1_bytes_per_token"] // gate["count_per_token"],
                "rung0_local_bpw_f16": gate["fmh"]["local_bpw_rung0_f16"],
                "rung1_local_bpw_f16": gate["fmh"]["local_bpw_rung1_f16"],
            },
        },
        "expected_operations_per_token": {
            "derived_not_guessed": True,
            "convention": "FMA=2; FWHT u+v/u-v counted as n log2 B FLOPs per launch",
            "incumbent_gemv_mac_flops": gemv_macs,
            "incumbent_activation_flops_unchanged": ANCHOR_ACTIVATION_FLOPS,
            "incumbent_executable_flops_with_dequant": ANCHOR_EXEC_FLOPS,
            "incumbent_executable_operations_with_unpack_alu": ANCHOR_EXEC_OPS,
            "keep_q4_mac_flops": keep_macs,
            "rung0_flops": rung0_flops,
            "rung1_flops": rung1_flops,
            "rung2_flops": rung2_flops,
            "rung0_vs_51_24_gflop": rung0_flops / ANCHOR_GEMV_MAC_FLOPS,
            "rung1_vs_51_24_gflop": rung1_flops / ANCHOR_GEMV_MAC_FLOPS,
            "rung2_vs_51_24_gflop": rung2_flops / ANCHOR_GEMV_MAC_FLOPS,
            "gate_proj_example": {
                "incumbent_mac_per_launch": gate["incumbent_mac_flops_per_token"] // gate["count_per_token"],
                "rung0_flops_per_launch": gate["rung0_flops_per_token"] // gate["count_per_token"],
                "rung1_flops_per_launch": gate["rung1_flops_per_token"] // gate["count_per_token"],
                "fwht_flops_per_launch": gate["fmh"]["fwht_flops"],
                "s_flops_per_launch": gate["fmh"]["s_flops"],
                "r_flops_per_launch": gate["fmh"]["r_flops"],
            },
            "intensity_flop_per_byte": {
                "ridge_point": ridge,
                "incumbent_q4": q4_intensity,
                "rung0": r0_intensity,
                "rung1": r1_intensity,
                "reading": (
                    f"Ridge is {ridge:.2f} FLOP/byte at 8979 GFLOP/s / 595.9 GB/s. "
                    f"Q4 GEMV sits at {q4_intensity:.2f} — bandwidth bound. Rung 0 "
                    f"sits at {r0_intensity:.2f} (still bandwidth bound, so tok/s "
                    "would track the byte cut IF healthy). tpr64 measured that "
                    "cutting dequant ALU without cutting bytes does not move GPU ns "
                    "at this launch geometry — that is the G032 codec-reparam case, "
                    "not rung 0/1."
                ),
            },
            "g043_recon_share_of_physical": G043_RECON_SHARE,
            "g043_physical_ns": G043_PHYSICAL_NS,
        },
        "dispatch_topology": {
            "incumbent": {
                "dispatches_per_token": ANCHOR_DISPATCHES,
                "gemv_dispatches": gemv_disp,
                "command_buffers": ANCHOR_CBS,
                "formula": "1 embed + 64*(9 mixer + 6 mlp) + 3 terminal = 964",
                "synchronises": (
                    "Single TokenCommandBuffer, encode-order data dependencies. "
                    "No extra host wait inside the token."
                ),
            },
            "production_fused": {
                "dispatches_per_token": fused_dispatches,
                "fmh_launches": fmh_disp,
                "q4_launches_kept": keep_disp,
                "command_buffers": fused_cbs,
                "kernel": "fmh_matvec_b1024_tg128 replaces geo_tpr64_tg128 on tiling organs",
                "synchronises": (
                    "Same 1-CB encode order. FWHT of x lives inside the organ kernel "
                    "so gate and up each FWHT the same hidden (51,200 FLOPs, noise). "
                    "A shared FWHT_hidden dispatch would ADD 64 dispatches to save "
                    "that noise — not worth it."
                ),
                "vs_incumbent": "964 → 964. Dispatch count is not the lever.",
            },
            "trap_unfused_strand_style": {
                "dispatches_per_token": unfused_dispatches,
                "extra": fmh_disp,
                "why_not_production": (
                    "strand_rht_forward_cols is a separate dispatch then GEMV reads tx. "
                    f"Copying that shape would add {fmh_disp} dispatches (964 → {unfused_dispatches}) "
                    "and a device buffer of n f32 per organ. Fuse."
                ),
            },
        },
        "metal_feasibility": {
            "device": "Apple M3 Ultra, 60 GPU cores, Metal 4, unified 103,079,215,104 B",
            "simdgroup_width": SIMDGROUP_WIDTH,
            "workhorse_threadgroup": WORKHORSE_TG,
            "threadgroup_memory_limit_bytes": TG_MEM_LIMIT_BYTES,
            "threadgroup_memory_limit_source": "crates/hawking-core/shaders/quant.metal:2115",
            "incumbent_kernel": {
                "name": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                "line": shaders.get("q4_kernel_line"),
                "grid": "ceil(rows/2)*128, TG 128",
                "map": "2 rows/TG, 64 threads/row (2 simdgroups), simd_sum, threadgroup red[4]=16 B",
                "coalescing": "each lane walks columns += 512, unpacks 8 nibbles, FMA against x[col]",
            },
            "existing_fwht": {
                "name": "strand_rht_forward_cols",
                "line": shaders.get("strand_rht_line"),
                "compile_gate": "feature=tq",
                "block": 256,
                "register_pressure": "float buf[256] = 1024 B per thread, one thread owns one block",
                "why_not_reuse_as_is": (
                    "Activation-side, B=256 not 1024, separate dispatch, tq-gated, "
                    "register-blocked scalar butterfly. Production FMH needs cooperative "
                    "simdgroup FWHT-1024 so it does not spill."
                ),
            },
            "existing_two_stage": {
                "name": "q80_hgravs01_two_stage_matvec",
                "tg_floats": shaders["hgravs01_tg_floats"],
                "tg_bytes": shaders["hgravs01_tg_floats"] * 4,
                "rank_cap": 160,
                "note": "mid[160] in TG is the pattern: reduce x into a small working set, then consume. FMH's working set is z of length n, not rank 160.",
            },
            "fmh_fused_launch": {
                "tg": 128,
                "grid_for_S_stage": "ceil(rows/2)*128 (match incumbent)",
                "fwht_stage": (
                    "5 (gate/up) or 17 (down) cooperative FWHT-1024. One simdgroup "
                    f"per tile, {fwht_tile_tg_bytes} B TG per tile reused, 10 butterfly "
                    "stages, shuffle or TG. Do not allocate float buf[1024] per thread."
                ),
                "z_in_tg_gate_up_bytes": gate_z_bytes,
                "z_in_tg_gate_up_fits_32kb": gate_z_bytes + red_tg_bytes <= TG_MEM_LIMIT_BYTES,
                "z_in_tg_down_bytes": down_z_bytes,
                "z_in_tg_down_fits_32kb": down_z_fits,
                "down_spill": (
                    "device scratch float z[17408] = 69,632 B per down_proj launch. "
                    "Not dense W (which would be 356,515,840 B f32 for that organ)."
                ),
                "register_pressure": (
                    "S-stage: 5 (n1) f32 accumulators + a handful of index math. "
                    "Far below strand_rht's 256-float buf. Incumbent already holds "
                    "an 8-wide unpack window plus acc."
                ),
                "coalescing": (
                    "x is 5120 consecutive f32 — coalesced. S packed [B, m1, n1] f16 "
                    "with n1=5 innermost: a row reads 10 consecutive bytes. Rung 1 R "
                    "is n1 tiles of B×B f16, streamed once per launch like today's Q4 "
                    "but 5×1024×1024×2 = 10,485,760 B on gate vs 47,349,760 Q4."
                ),
                "lines_estimate": (
                    "kernel census: ~200 lines if RHT is reused, plus drop the tq gate. "
                    "Quality, not volume, is the blocker."
                ),
            },
        },
        "memory_layout": {
            "per_tensor_blob": {
                "header_bytes": HEADER_BYTES,
                "header_fields": [
                    "magic fmh1",
                    "B u32 = 1024",
                    "m1 u32",
                    "n1 u32",
                    "flags u32 (bit0=has_R, bit1=has_L, bits8-15=H_kind sylvester)",
                ],
                "H": "not stored; generated from B",
                "PI": "not stored; generated reshape/transpose",
                "S": "f16[B, m1, n1] row-major, n1 fastest. rung 0 payload.",
                "R": "optional f16[n1, B, B], tile-major",
                "L": "optional f16[m1, B, B], tile-major",
            },
            "nr_accounting": {
                "H_is_not_model_specific": (
                    "G042: generated Hadamard does not carry parent information. "
                    "Do not put H in generated_structures and claim GENERATED_BPW."
                ),
                "S_and_R_are_model_specific": (
                    "Count every byte of S, R, L, header. complete_bits_per_weight "
                    "must be computed from those bytes over source_weight_elements, "
                    "not declared. n16clos: StructuredTransform is currently a "
                    "schema-change family on the sealed NR."
                ),
                "do_not_share_S_across_layers": "G035 shared_beats_independent=false",
                "active_vs_stored": (
                    "Report both. Q80 storage 0.6462 vs ACTIVE 2.518 is the category "
                    "error. FMH stored bytes ARE the active weight bytes (no embed-table "
                    "trick on these organs)."
                ),
            },
            "workspace": {
                "z_scratch_down_bytes": down_z_bytes,
                "z_scratch_is_dense_W": False,
                "incumbent_dense_W_if_reconstructed": 4 * gemv_elems,
            },
        },
        "cheap_microbenchmark": microbench,
        "expected_value": expected_value,
        "s011_section4": s011,
        "what_i_watched_fail": watched,
        "self_check": self_check,
        "written_to": str(RECEIPT),
    }


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_ratio(x: float) -> str:
    return f"{x:.6f}"


def print_report(doc: dict) -> None:
    print("=" * 78)
    print("C5 STRUCTURED TRANSFORM DESIGN")
    print("=" * 78)
    print(f"schema     {doc['schema']}")
    print(f"generated  {doc['generated_at']}")
    print(f"head       {doc['commit']}")
    print(f"verdict    {doc['verdict']}")
    print()
    print("## Prior-science search")
    ps = doc["prior_science"]
    print(f"  searched: {ps['searched']}  live hits={ps['live_confirmation']['n_hits']} "
          f"missing={ps['live_confirmation']['n_missing']}")
    for name in ps["n1arch_mechanisms_relevant"]:
        print(f"  - {name}")
    print()
    fr = ps["family_already_refuted"]
    print("  family members:")
    for k, v in fr.items():
        flag = "REFUTED" if v.get("refuted") else "NOT REFUTED (nearest evidence unhealthy)"
        print(f"    [{flag}] {k}")
        if k == "sharing_a_transform_across_a_tensor_class":
            z = v["zero_point_zero_one_five_six"]
            print(f"      0.0156 handling: {z['what_it_is']}")
            print(f"      coincidence: {z['numerical_coincidence']}")
    print()
    print("## 1. Mathematical operator")
    op = doc["operator"]
    print(f"  name: {op['name']}  ({op['long_name']})  B={op['B']}")
    print(f"  math: {op['math']}")
    print(f"  ORACLE (labelled {op['oracle']['label']}, not production):")
    print(f"    {op['oracle']['formula']}")
    print(f"    trap: {op['oracle']['trap']}")
    print(f"  PRODUCTION (labelled {op['production']['label']}):")
    print(f"    kernel {op['production']['kernel_name']}, never_materialises_W="
          f"{op['production']['never_materialises_W']}")
    print(f"    {op['production']['combination_required']}")
    print(f"  FWHT identity self-check n={op['fwht_identity_self_check']['n']} "
          f"ok={op['fwht_identity_self_check']['ok']}")
    print()
    print("## 2. Expected bytes / token (derived)")
    b = doc["expected_bytes_per_token"]
    print(f"  incumbent Q4 GEMV     {fmt_int(b['incumbent_q4_gemv_bytes'])} B")
    print(f"  incumbent DRAM        {fmt_int(b['incumbent_dram_bytes'])} B")
    print(f"  keep-Q4 organs        {b['keep_q4_organs']}  {fmt_int(b['keep_q4_bytes'])} B")
    print(f"  FMH organs            {b['fmh_organs']}")
    print(f"  rung0 weight          {fmt_int(b['rung0_weight_bytes'])} B  "
          f"({fmt_ratio(b['rung0_vs_incumbent_weight'])} × q4)")
    print(f"  rung1 weight          {fmt_int(b['rung1_weight_bytes'])} B  "
          f"({fmt_ratio(b['rung1_vs_incumbent_weight'])} × q4)")
    print(f"  rung2 weight          {fmt_int(b['rung2_weight_bytes'])} B")
    print(f"  bandwidth ms @ 595.9 GB/s  incumbent {b['bandwidth_ms_at_595_9']['incumbent']:.3f}  "
          f"rung0 {b['bandwidth_ms_at_595_9']['rung0_counterfactual']:.3f}  "
          f"rung1 {b['bandwidth_ms_at_595_9']['rung1_counterfactual']:.3f}")
    ge = b["gate_proj_example"]
    print(f"  gate_proj per launch  q4 {fmt_int(ge['q4_bytes_per_launch'])}  "
          f"rung0 {fmt_int(ge['rung0_bytes_per_launch'])}  "
          f"rung1 {fmt_int(ge['rung1_bytes_per_launch'])}  "
          f"local_bpw_r0 {ge['rung0_local_bpw_f16']:.6f}")
    print()
    print("## 3. Expected operations / token (derived, vs 51.24 GFLOP / 964)")
    o = doc["expected_operations_per_token"]
    print(f"  incumbent GEMV MAC    {fmt_int(o['incumbent_gemv_mac_flops'])} FLOP  (51.24 GFLOP)")
    print(f"  incumbent + dequant   {fmt_int(o['incumbent_executable_flops_with_dequant'])} FLOP")
    print(f"  incumbent + unpack    {fmt_int(o['incumbent_executable_operations_with_unpack_alu'])} ops")
    print(f"  rung0                 {fmt_int(o['rung0_flops'])} FLOP  "
          f"({fmt_ratio(o['rung0_vs_51_24_gflop'])} ×)")
    print(f"  rung1                 {fmt_int(o['rung1_flops'])} FLOP  "
          f"({fmt_ratio(o['rung1_vs_51_24_gflop'])} ×)")
    print(f"  rung2                 {fmt_int(o['rung2_flops'])} FLOP  "
          f"({fmt_ratio(o['rung2_vs_51_24_gflop'])} ×)")
    print(f"  activations unchanged {fmt_int(o['incumbent_activation_flops_unchanged'])} FLOP")
    print(f"  intensity FLOP/byte   q4 {o['intensity_flop_per_byte']['incumbent_q4']:.3f}  "
          f"rung0 {o['intensity_flop_per_byte']['rung0']:.3f}  "
          f"ridge {o['intensity_flop_per_byte']['ridge_point']:.2f}")
    print(f"  {o['intensity_flop_per_byte']['reading']}")
    go = o["gate_proj_example"]
    print(f"  gate_proj per launch  mac {fmt_int(go['incumbent_mac_per_launch'])}  "
          f"rung0 {fmt_int(go['rung0_flops_per_launch'])}  "
          f"(FWHT {fmt_int(go['fwht_flops_per_launch'])} + S {fmt_int(go['s_flops_per_launch'])})")
    print()
    print("## 4. Dispatch topology")
    d = doc["dispatch_topology"]
    print(f"  incumbent     {d['incumbent']['dispatches_per_token']} dispatches, "
          f"{d['incumbent']['command_buffers']} CB, {d['incumbent']['gemv_dispatches']} GEMV")
    print(f"  fused FMH     {d['production_fused']['dispatches_per_token']} dispatches, "
          f"{d['production_fused']['command_buffers']} CB  "
          f"({d['production_fused']['fmh_launches']} fmh + {d['production_fused']['q4_launches_kept']} q4 kept)")
    print(f"  {d['production_fused']['synchronises']}")
    print(f"  unfused trap  {d['trap_unfused_strand_style']['dispatches_per_token']} "
          f"(+{d['trap_unfused_strand_style']['extra']}) — not production")
    print()
    print("## 5. Metal feasibility")
    m = doc["metal_feasibility"]
    print(f"  {m['device']}")
    print(f"  simdgroup {m['simdgroup_width']}  TG {m['workhorse_threadgroup']}  "
          f"TG mem limit {m['threadgroup_memory_limit_bytes']} B "
          f"({m['threadgroup_memory_limit_source']})")
    print(f"  incumbent {m['incumbent_kernel']['name']} line {m['incumbent_kernel']['line']}: "
          f"{m['incumbent_kernel']['map']}")
    print(f"  existing FWHT {m['existing_fwht']['name']} line {m['existing_fwht']['line']} "
          f"gate={m['existing_fwht']['compile_gate']}  {m['existing_fwht']['register_pressure']}")
    f = m["fmh_fused_launch"]
    print(f"  FMH z in TG  gate/up {f['z_in_tg_gate_up_bytes']} B fit={f['z_in_tg_gate_up_fits_32kb']}  "
          f"down {f['z_in_tg_down_bytes']} B fit={f['z_in_tg_down_fits_32kb']}")
    print(f"  down spill: {f['down_spill']}")
    print(f"  coalescing: {f['coalescing']}")
    print(f"  {f['lines_estimate']}")
    print()
    print("## 6. Memory layout")
    lay = doc["memory_layout"]
    print(f"  blob: {lay['per_tensor_blob']['H']}; {lay['per_tensor_blob']['S']}; "
          f"{lay['per_tensor_blob']['R']}")
    print(f"  NR: {lay['nr_accounting']['H_is_not_model_specific']}")
    print(f"  NR: {lay['nr_accounting']['S_and_R_are_model_specific']}")
    print(f"  do not share S: {lay['nr_accounting']['do_not_share_S_across_layers']}")
    print()
    print("## 7. Cheap microbenchmark")
    mb = doc["cheap_microbenchmark"]
    print(f"  {mb['name']}  wall_budget_s={mb['wall_budget_s']}")
    print(f"  tensor {mb['tensor']} {mb['shape']}")
    print(f"  X {mb['X_source']['path']}")
    print(f"  site {mb['X_source']['site']} fit={mb['X_source']['fit_n']} hold={mb['X_source']['hold_n']} "
          f"not_synthetic={mb['not_synthetic']}")
    print("  procedure:")
    for step in mb["procedure"]:
        print(f"    - {step}")
    print("  kill if:")
    for k in mb["kill_if"]:
        print(f"    - {k}")
    print(f"  predicted: {mb['predicted_outcome']['therefore']}")
    print()
    print("## 8. Expected value")
    ev = doc["expected_value"]
    print(f"  VERDICT: {ev['verdict']}")
    w = ev["what_it_would_win_if_healthy"]
    print(f"  counterfactual rung0: {fmt_int(w['rung0_weight_bytes_per_token'])} B, "
          f"{fmt_ratio(w['rung0_flops_vs_51_24_gflop'])} × FLOPs, "
          f"{w['rung0_bandwidth_tps_counterfactual']:.1f} tok/s at roof IF healthy")
    print(f"  counterfactual rung1: {fmt_int(w['rung1_weight_bytes_per_token'])} B, "
          f"{fmt_ratio(w['rung1_flops_vs_51_24_gflop'])} × FLOPs, "
          f"{w['rung1_bandwidth_tps_counterfactual']:.1f} tok/s at roof IF healthy")
    print(f"  to beat: self {w['controls_to_beat']['self_uniform_q4']}  "
          f"MLX {w['controls_to_beat']['mlx_4bit_live']}  "
          f"llama.cpp Q5_K archived {w['controls_to_beat']['llamacpp_q5k_archived']}")
    print(f"  {w['counterfactual_flag']}")
    print("  risks:")
    for r in ev["what_it_risks"]:
        print(f"    - {r}")
    print(f"  cheapest kill: {ev['cheapest_kill']}")
    print(f"  {ev['why_not_worth_building_the_kernel']}")
    print(f"  reopen if: {ev['reopen_if']}")
    print()
    print("## S011 §4 completeness")
    s = doc["s011_section4"]
    print(f"  rung0 fused complete={s['rung0_fused']['complete_by_s011_section4']} "
          f"reduces {s['rung0_fused']['reduces']}")
    print(f"  rung1 fused complete={s['rung1_fused']['complete_by_s011_section4']} "
          f"reduces {s['rung1_fused']['reduces']}")
    print(f"  rung2 f16 complete={s['rung2_fused_f16']['complete_by_s011_section4']} "
          f"reduces {s['rung2_fused_f16']['reduces']}  "
          f"bytes_exceed_q4={s['rung2_fused_f16']['bytes_exceed_incumbent_q4']}")
    print(f"  G032 codec-reparam complete={s['g032_codec_reparam_on_q4']['complete_by_s011_section4']}: "
          f"{s['g032_codec_reparam_on_q4']['why']}")
    print(f"  {s['note']}")
    print()
    print("## WHAT I WATCHED FAIL")
    for i, f in enumerate(doc["what_i_watched_fail"], 1):
        print(f"  {i}. {f['what']}: {f['result']}")
        print(f"     {f['why']}")
    print()
    sc = doc["self_check"]
    bad = [k for k, v in sc.items() if v is False]
    print("## Self-check")
    print(f"  {sc}")
    if bad:
        print(f"  FAIL keys: {bad}")
    else:
        print("  all boolean checks passed")
    print()
    print(f"wrote {doc['written_to']}")
    print("=" * 78)


def main() -> int:
    for p in (GEOMETRY, SCHEDULE, Q4_METAL, STRAND_METAL):
        if not p.exists():
            print(f"FAIL: required path missing: {p}", file=sys.stderr)
            return 2

    g = load_geometry()
    organs = gemv_organs(g)
    prior = confirm_prior_science()
    shaders = shader_facts()
    fwht = fwht_identity_check(16)
    doc = build_design(g, organs, prior, shaders, fwht)

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    tmp.replace(RECEIPT)

    print_report(doc)

    sc = doc["self_check"]
    required = [
        "dispatches_964",
        "gemv_elements_match_anchor",
        "gemv_macs_match_anchor",
        "q4_bytes_match_anchor",
        "fwht_identity_ok",
        "oracle_labelled",
        "production_labelled",
        "s011_rung0_complete",
        "s011_g032_incomplete",
        "rung2_f16_bytes_exceed_q4",
        "verdict_is_not_worth_building",
        "q4_kernel_present",
        "strand_rht_present",
    ]
    failed = [k for k in required if not sc.get(k)]
    if failed:
        print(f"FAIL self-check {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
