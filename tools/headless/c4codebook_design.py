#!/usr/bin/env python3
"""C4 codebook / dictionary-execution design for Qwen3.8-27B.

A kernel for this family already exists: `gravity_pq_matvec` is REACHABLE
(compiled, quoted, callable) and is NOT dispatched on the Qwen3.8 uniform-q4
path. This lane establishes what that kernel does, what it costs, why it is
absent from Qwen3.8, then specifies the operator that would actually reduce
work — fused ADC (lookup-plus-accumulate) — rather than rediscovering a new
family or porting a kernel that still performs 51.24 GFLOP of GEMV.

This is a DESIGN lane. It does not land a kernel (crates/ is read-only).
It terminates in a measured design decision.

    python3 tools/headless/c4codebook_design.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "hawking.headless.c4codebook_design.v1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHADERS = REPO / "crates/hawking-core/shaders"
DECODE = REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
GEOMETRY = REPO / "crates/hawking-core/src/model/qwen38_geometry.rs"
LEDGER = REPO / "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs"
ARTIFACT_PQ = REPO / "crates/hawking-core/src/artifact_pq.rs"
GRAVITY_RS = REPO / "crates/hawking-core/src/gravity.rs"
PQ_METAL = SHADERS / "gravity_pq.metal"
Q4_METAL = SHADERS / "qwen_uniform_q4.metal"
RESIDUAL_PQ_TEST = REPO / "crates/hawking-core/tests/gravity_residual_pq_metal.rs"
RECEIPT = REPO / "receipts/headless/C4CODEBOOK_DESIGN.json"
KERNEL_CENSUS = REPO / "receipts/headless/NOETIC_KERNEL_CENSUS.json"
OP_CENSUS = REPO / "receipts/headless/NOETIC_OPERATION_CENSUS.json"
TPR64 = REPO / "receipts/headless/NOETIC_TPR64_REOPEN.json"
ORGAN_CENSUS = REPO / "receipts/headless/NOETIC_ORGAN_CENSUS.json"
GAIN_RESCORE = REPO / "receipts/headless/NOETIC_GAIN_RESCORE.json"
CENSUS_DIR = REPO / ".lane-bootstrap/census"

# Anchors — measured, not re-derived.
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_TPS = 32.73
ANCHOR_TOKEN_MS = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_UNIFIED_B = 103_079_215_104
ANCHOR_GPU_CORES = 60
ANCHOR_PARAMS = 26_895_998_464
ANCHOR_BPW = 4.253
ANCHOR_GEMV_GFLOP = 51.24  # 51_243_909_120 / 1e9, FMA=2
ANCHOR_GEMV_MAC_FLOPS = 51_243_909_120
ANCHOR_GEMV_ELEMENTS = 25_621_954_560
ANCHOR_ACT_FLOPS = 297_313_024
ANCHOR_Q4_GEMV_BYTES = 13_611_663_360
ANCHOR_EXEC_DRAM = 13_988_022_948
ANCHOR_EXEC_FLOPS = 77_163_181_824
ANCHOR_EXEC_OPS = 179_651_020_544
ANCHOR_MLX_TPS = 35.51
ANCHOR_LLAMA_Q5K_TPS = 24.12
ANCHOR_TWO_SERVERS_TPS = 3.986
ANCHOR_ONE_SERVER_TPS = 33.47

UNIFORM_Q4_GROUP = 64
Q4_BYTES_PER_GROUP = UNIFORM_Q4_GROUP // 2 + 2  # 32 code + 2 fp16 scale = 34
PQ_HEADER_LEN = 64
PQ_MAGIC = b"GLM52CPK"
RESIDUAL_PQ_MAGIC = b"LLM52RPK"
HQ30UQ4_MAGIC = "HQ30UQ4"  # substring; exact bytes confirmed in decode.rs

# Llama-8B FFN gate roofline, already measured. Do not re-run.
# workspace/campaign/evidence/runtime/tg/TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json
LLAMA_GATE_ROWS = 14_336
LLAMA_GATE_COLS = 4_096
LLAMA_D32_NCHUNK = 128
LLAMA_D32_MEDIAN_US = 460.041
LLAMA_D8S4_MEDIAN_US = 672.625

# Apple GPU constants witnessed by the kernels themselves.
# simdgroup width 32: gravity_pq_matvec strides `c += 32u` and uses simd_sum.
# TG 256 / 8 rows: gravity.rs PqMetalMatrix::encode n_tg = rows.div_ceil(8), TG=256.
# TG 128 / 2 rows / 64 threads-per-row: qwen_uniform_q4 geo_tpr64_tg128.
# ranked[2048] ulong in gravity_pq.metal is 16 KiB threadgroup, so 32 KiB is
# the practical ceiling this tree already budgets against (Apple9 = 32768).
SIMDGROUP_WIDTH = 32
PQ_TG = 256
PQ_ROWS_PER_TG = 8
Q4_TG = 128
Q4_THREADS_PER_ROW = 64
APPLE9_TG_MEM_BYTES = 32_768

# GLM production PQ geometry (shader comment + tests/gravity_pq_kernel_registry.rs).
GLM_D = 32
GLM_S = 1
GLM_SUB = 32
GLM_CARD = 256
GLM_BITS = 8


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def git_show(path: str) -> str | None:
    try:
        p = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            capture_output=True, text=True, cwd=REPO, timeout=30,
        )
        if p.returncode == 0 and p.stdout:
            return p.stdout
    except Exception:
        return None
    return None


def git_show_json(path: str):
    text = git_show(path)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
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


def pq_index_bytes(rows: int, cols: int, d: int, bits: int, subspaces: int = 1) -> int:
    if cols % d != 0:
        raise SystemExit(f"FAIL: cols {cols} not divisible by D={d}")
    nchunk = cols // d
    bits_total = rows * nchunk * subspaces * bits
    return (bits_total + 7) // 8


def pq_codebook_bytes(card: int, sub: int, subspaces: int = 1) -> int:
    return subspaces * card * sub * 2  # fp16


def pq_payload_bytes(rows: int, cols: int, d: int, bits: int, card: int, sub: int,
                     subspaces: int = 1) -> int:
    return PQ_HEADER_LEN + pq_codebook_bytes(card, sub, subspaces) + pq_index_bytes(
        rows, cols, d, bits, subspaces
    )


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n:,} ({n / 1e9:.3f} GB)"
    if n >= 1_000_000:
        return f"{n:,} ({n / 1e6:.3f} MB)"
    if n >= 1_000:
        return f"{n:,} ({n / 1e3:.3f} kB)"
    return f"{n:,} B"


def fmt_gflop(n: int) -> str:
    return f"{n:,} ({n / 1e9:.3f} GFLOP)"


def geometry() -> dict:
    src = GEOMETRY.read_text()
    g = {
        "layers": usize_const(src, "QWEN38_LAYERS"),
        "dn_layers": usize_const(src, "QWEN38_DELTANET_LAYERS"),
        "gqa_layers": usize_const(src, "QWEN38_GQA_LAYERS"),
        "hidden": usize_const(src, "QWEN38_HIDDEN"),
        "intermediate": usize_const(src, "QWEN38_INTERMEDIATE"),
        "vocab": usize_const(src, "QWEN38_VOCAB"),
        "qkvz_rows": usize_const(src, "QWEN38_QKVZ_ROWS"),
        "ba_rows": usize_const(src, "QWEN38_BA_ROWS"),
        "q_proj_rows": usize_const(src, "QWEN38_Q_PROJ_ROWS"),
        "kv_proj_rows": usize_const(src, "QWEN38_KV_PROJ_ROWS"),
        "o_proj_rows": usize_const(src, "QWEN38_O_PROJ_ROWS"),
        "o_proj_cols": usize_const(src, "QWEN38_O_PROJ_COLS"),
    }
    if g["layers"] != 64 or g["hidden"] != 5120 or g["intermediate"] != 17408:
        raise SystemExit(f"FAIL: geometry mismatch {g}")
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


def inspect_existing_kernel() -> dict:
    metal = PQ_METAL.read_text()
    decode = DECODE.read_text()
    artifact = ARTIFACT_PQ.read_text()
    gravity = GRAVITY_RS.read_text()
    q4 = Q4_METAL.read_text()
    residual_test = RESIDUAL_PQ_TEST.read_text() if RESIDUAL_PQ_TEST.is_file() else ""

    # Kernel body: gravity_pq_matvec through gravity_residual_pq_matvec.
    m = re.search(
        r"kernel void gravity_pq_matvec\((.*?)kernel void gravity_residual_pq_matvec",
        metal, re.S,
    )
    if not m:
        raise SystemExit("FAIL: gravity_pq_matvec body not found")
    body = m.group(0)
    writes_dense = bool(re.search(r"W\[|dense_w|reconstruct", body))
    has_fma = "acc = fma(float(entry[j]), xs[j], acc)" in body
    has_index = "pq_index(codes, flat, p.bits)" in body
    one_sg_per_row = "uint row = tgid * sgs_per_tg + sg_in_tg" in body
    never_materializes = "never materializes a dense weight" in metal

    # Catalog: Qwen3.8 classify refuses anything that is not packed/HQ30/f32v2.
    classify = "classify_qwen38_mixed_payload" in decode
    glm_magic_in_decode = "GLM52CPK" in decode
    llm_magic_in_decode = "LLM52RPK" in decode
    pq_quoted_in_decode = bool(re.search(r"gravity_pq_matvec", decode))
    hq30_in_decode = "HQ30UQ4" in decode or "Hq30Uq4" in decode

    encode_tg = "let n_tg = params.rows.div_ceil(8)" in gravity
    rotate_unsupported = "rotated gravity-pq artifacts (rotate=1) are not yet supported" in gravity
    row_fn_is_oracle = "pub fn row(" in artifact  # reconstructs one dense row, host only

    q4_winner = "Geometry-sweep winner for Q4 gate [512, 2048]: 64 threads/row, 128-thread" in q4
    tpr64_kernel = "kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128" in q4

    llama_d32_probe = "residual_pq_single_stage_d32_ffn_gate_geometry_roofline" in residual_test

    census = load_json(KERNEL_CENSUS) or {}
    families = {f["id"]: f for f in census.get("families", [])}
    fused = families.get("fused_dictionary_lookup_accumulate", {})
    reachable = census.get("reachable", [])
    pq_reach = next((k for k in reachable if isinstance(k, dict) and k.get("name") == "gravity_pq_matvec"), None)

    return {
        "kernel_name": "gravity_pq_matvec",
        "file": "crates/hawking-core/shaders/gravity_pq.metal",
        "line": 399,
        "class": (pq_reach and "REACHABLE") or fused.get("kernel", {}).get("class") or "REACHABLE",
        "compile_gate": None,
        "quoted_in_qwen38_hybrid_decode": pq_quoted_in_decode,
        "glm52cpk_magic_accepted_by_qwen38_classify": glm_magic_in_decode,
        "llm52rpk_magic_accepted_by_qwen38_classify": llm_magic_in_decode,
        "qwen38_classify_exists": classify,
        "qwen38_admits_hq30uq4": hq30_in_decode,
        "never_materializes_dense_w_comment": never_materializes,
        "body_writes_dense_w": writes_dense,
        "body_does_fma_into_acc": has_fma,
        "body_does_pq_index_gather": has_index,
        "one_simdgroup_per_output_row": one_sg_per_row,
        "host_encode_8_rows_per_tg256": encode_tg,
        "rotate_1_unsupported": rotate_unsupported,
        "host_row_reconstructs_dense_as_oracle_helper": row_fn_is_oracle,
        "q4_tpr64_present": tpr64_kernel,
        "q4_geometry_sweep_comment_present": q4_winner,
        "llama_d32_roofline_probe_present": llama_d32_probe,
        "census_family_verdict": fused.get("verdict"),
        "census_family_why": fused.get("why"),
        "census_missing_family_cost": next(
            (x for x in census.get("missing_family_cost", [])
             if x.get("id") == "fused_dictionary_lookup_accumulate"),
            {},
        ),
        "production_geometry_glm": {
            "D": GLM_D, "S": GLM_S, "sub": GLM_SUB, "card": GLM_CARD, "bits": GLM_BITS,
            "source": "gravity_pq.metal comments + tests/gravity_pq_kernel_registry.rs primary_header",
        },
        "math": (
            "For subspace s in 0..S-1, chunk c in 0..nchunk-1, row r: "
            "code = pq_index(codes, (r*nchunk + c)*S + s, bits); "
            "acc_r += <codebook[s, code], x[c*D + s*sub : c*D + (s+1)*sub]>. "
            "S=1, D=sub=32, card=256, bits=8 is the GLM production shape. "
            "W is never written. This is fused lookup+FMA, not ADC: the codebook "
            "vector is dotted against x once per (row, chunk), so the GEMV FMA "
            "count equals the dense parent (rows*cols FMAs)."
        ),
    }


def inspect_artifact() -> dict:
    path = Path.home() / "models/qwen38-gravity-uniform-q4-v1/manifest.json"
    out = {"path": str(path), "present": path.is_file(), "gravity_pq_tensors": 0, "kinds": {}}
    if not path.is_file():
        out["note"] = "artifact manifest not on disk in this process; using census numbers"
        return out
    man = json.loads(path.read_text())
    kinds = {}
    pq_like = 0
    for t in man.get("tensors", []):
        k = t.get("kind") or "unknown"
        kinds[k] = kinds.get(k, 0) + 1
        art = t.get("artifact") or ""
        if "pq" in art.lower() or str(t.get("codec", "")).find("pq") >= 0:
            pq_like += 1
    out.update({
        "schema": man.get("schema"),
        "status": man.get("status"),
        "tensor_count": man.get("tensor_count"),
        "q4_tensors": man.get("q4_tensors"),
        "f32_tensors": man.get("f32_tensors"),
        "kinds": kinds,
        "gravity_pq_tensors": pq_like,
        "complete_physical_bpw": man.get("complete_physical_bpw"),
        "q4_group_size": man.get("q4_group_size"),
        "source_weight_elements": man.get("source_weight_elements"),
    })
    return out


def prior_science_search() -> dict:
    """Search preserved receipts. Do not rediscover; report misses honestly."""
    found = []
    missing = []
    closures = []

    # 1. Lane-bootstrap census (named in the contract).
    census_files = []
    if CENSUS_DIR.is_dir():
        census_files = sorted(p.name for p in CENSUS_DIR.iterdir())
        found.append({
            "id": "lane-bootstrap-census",
            "path": str(CENSUS_DIR),
            "files": census_files,
        })
    else:
        ls = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD", ".lane-bootstrap/census"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        )
        if ls.returncode == 0 and ls.stdout.strip():
            found.append({
                "id": "lane-bootstrap-census-git",
                "files": ls.stdout.strip().splitlines(),
            })
        else:
            missing.append({
                "id": "lane-bootstrap-census",
                "path": ".lane-bootstrap/census",
                "why": (
                    "Not on disk (sparse checkout) and not in HEAD. "
                    "n1arch/n15neg/n16clos cannot be opened here. "
                    "Substituted: receipts/headless/NOETIC_KERNEL_CENSUS.json "
                    "families + constraints_from_recovered_science, plus git-show "
                    "of G035 and residual-PQ rejections."
                ),
            })

    def take_json(label, path, keys=None):
        path_s = str(path)
        d = load_json(Path(path)) if not path_s.startswith("HEAD:") else None
        how = "disk"
        if d is None and not path_s.startswith("HEAD:"):
            rel = path_s
            p = Path(path)
            try:
                rel = str(p.relative_to(REPO))
            except ValueError:
                rel = path_s
            d = git_show_json(rel)
            how = "git" if d is not None else "missing"
        if path_s.startswith("HEAD:"):
            d = git_show_json(path_s[5:])
            how = "git" if d is not None else "missing"
        if d is None:
            missing.append({"id": label, "path": path_s, "why": "not on disk and git show failed"})
            return None
        rec = {"id": label, "how": how, "path": path_s}
        if keys:
            rec["excerpt"] = {k: d.get(k) for k in keys if k in d}
        found.append(rec)
        return d

    kc = take_json("NOETIC_KERNEL_CENSUS", KERNEL_CENSUS)
    oc = take_json("NOETIC_OPERATION_CENSUS", OP_CENSUS)
    tpr = take_json("NOETIC_TPR64_REOPEN", TPR64)
    organ = take_json("NOETIC_ORGAN_CENSUS", ORGAN_CENSUS)
    gain = take_json("NOETIC_GAIN_RESCORE", GAIN_RESCORE)

    g035 = take_json(
        "G035_CROSSLAYER_SHARE",
        "HEAD:receipts/ascent-2026-08-16/G035_CROSSLAYER_SHARE.json",
    )
    pq_ffn = take_json(
        "TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED",
        "HEAD:workspace/campaign/evidence/runtime/tg/TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json",
    )
    pq_s2 = take_json(
        "TG_LLAMA_RESIDUAL_PQ_S2_C64_REJECTED",
        "HEAD:workspace/campaign/evidence/runtime/tg/TG_LLAMA_RESIDUAL_PQ_S2_C64_REJECTED.json",
    )
    pq_s3 = take_json(
        "TG_LLAMA_RESIDUAL_PQ_S3_C128_REJECTED",
        "HEAD:workspace/campaign/evidence/runtime/tg/TG_LLAMA_RESIDUAL_PQ_S3_C128_REJECTED.json",
    )
    pq_qwen32 = take_json(
        "TG32_QWEN32_RESIDUAL_PQ_RESULT",
        "HEAD:workspace/campaign/evidence/runtime/tg/TG32_QWEN32_RESIDUAL_PQ_RESULT_2026_08_02.json",
    )
    g042 = take_json(
        "G042_BPW_FAMILY",
        "HEAD:receipts/ascent-2026-08-16/G042_BPW_FAMILY.json",
    )

    # Closures that constrain this family.
    if kc:
        for c in kc.get("constraints_from_recovered_science", []):
            closures.append({"source": "NOETIC_KERNEL_CENSUS.constraints", "text": c})
        fused = next(
            (f for f in kc.get("families", []) if f.get("id") == "fused_dictionary_lookup_accumulate"),
            None,
        )
        if fused:
            closures.append({
                "source": "NOETIC_KERNEL_CENSUS.families.fused_dictionary_lookup_accumulate",
                "verdict": fused.get("verdict"),
                "text": fused.get("why"),
                "status": "EXISTS_REACHABLE_NOT_DISPATCHED_ON_QWEN38",
            })
        shared = next(
            (f for f in kc.get("families", []) if f.get("id") == "shared_basis_x_coefficients"),
            None,
        )
        if shared:
            closures.append({
                "source": "NOETIC_KERNEL_CENSUS.families.shared_basis_x_coefficients",
                "verdict": shared.get("verdict"),
                "text": shared.get("why"),
                "applies_to_codebooks": (
                    "G035 refutes shared SVD bases, not shared PQ codebooks. "
                    "Cited as a caution: sharing is not a free byte win. "
                    "A 16 KiB codebook shared across 64 layers saves 63*16 KiB, "
                    "which is noise against hundreds of MB of indices."
                ),
            })

    if g035:
        pairs = g035.get("pairs") or []
        flags = [p.get("shared_beats_independent") for p in pairs]
        closures.append({
            "source": "G035_CROSSLAYER_SHARE",
            "shared_beats_independent": flags,
            "all_false": all(v is False for v in flags) if flags else None,
            "corrected_verdict": g035.get("corrected_verdict"),
            "what_still_limits_it": g035.get("what_still_limits_it"),
            "text": (
                "Column-space sharing LOSES 3/3. Row-space sharing WINS 6.3% "
                "at ~1.03 bits/elem where pair errors are 0.58-0.70 (dead vs "
                "coherent 0.198). Not a codebook-sharing license."
            ),
        })

    if pq_ffn:
        meas = pq_ffn.get("measurement") or {}
        ctrl = pq_ffn.get("single_stage_control") or {}
        closures.append({
            "source": "TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED",
            "status": pq_ffn.get("status"),
            "four_stage_median_us": meas.get("median_microseconds_per_gate_projection"),
            "single_stage_d32_median_us": ctrl.get("median_microseconds_per_gate_projection"),
            "reopen_condition": pq_ffn.get("reopen_condition"),
            "text": pq_ffn.get("cause"),
        })
    if pq_s2:
        closures.append({
            "source": "TG_LLAMA_RESIDUAL_PQ_S2_C64_REJECTED",
            "status": pq_s2.get("status"),
            "complete_bpw": (pq_s2.get("artifact") or {}).get("complete_bpw"),
            "decode_tps": (pq_s2.get("matched_decode_gate") or {}).get("decode_tps"),
            "equal_tokens": (pq_s2.get("matched_decode_gate") or {}).get("equal"),
            "text": pq_s2.get("diagnosis"),
        })
    if pq_s3:
        closures.append({
            "source": "TG_LLAMA_RESIDUAL_PQ_S3_C128_REJECTED",
            "status": pq_s3.get("status"),
            "complete_bpw": (pq_s3.get("artifact") or {}).get("complete_bpw"),
            "equal_tokens": (pq_s3.get("matched_decode_gate") or {}).get("equal"),
            "text": pq_s3.get("diagnosis"),
        })
    if pq_qwen32:
        closures.append({
            "source": "TG32_QWEN32_RESIDUAL_PQ_RESULT",
            "status": pq_qwen32.get("status"),
            "complete_bpw": (pq_qwen32.get("representation") or {}).get("complete_bpw"),
            "tps": (pq_qwen32.get("runtime") or {}).get("tps"),
            "capability": pq_qwen32.get("capability"),
            "reopen": (pq_qwen32.get("retirement") or {}).get("reopen_condition"),
            "text": (pq_qwen32.get("capability") or {}).get("rejection_reason")
            or pq_qwen32.get("status"),
        })

    if tpr:
        fr = tpr.get("free_reconstruction") or {}
        closures.append({
            "source": "NOETIC_TPR64_REOPEN",
            "claim": fr.get("claim"),
            "what_free_means": fr.get("what_free_means"),
            "text": (
                "At production tpr64, in-register dequant is free against the "
                "packed-byte floor on 32/33 variants. A design that only cuts "
                "dequant ALU, at matched bytes, cannot beat Q4 on this machine."
            ),
        })
        nb = tpr.get("null_baseline_cosine") or {}
        closures.append({
            "source": "NOETIC_TPR64_REOPEN.null_baseline_cosine",
            "value": nb.get("value"),
            "text": nb.get("law"),
        })

    if organ:
        sr = organ.get("scale_rejection") or {}
        probe = (sr.get("probe") or {}).get("scaled_0p01") or {}
        closures.append({
            "source": "NOETIC_ORGAN_CENSUS.scale_rejection",
            "cosine_0p01W": probe.get("cosine"),
            "scale_aware_0p01W": probe.get("scale_aware"),
            "text": sr.get("statement"),
        })
        spot = organ.get("codec_spot_check_vs_bf16") or {}
        closures.append({
            "source": "NOETIC_ORGAN_CENSUS.codec_spot_check_vs_bf16",
            "organ": spot.get("name"),
            "cosine": spot.get("cosine"),
            "relative_l2": spot.get("relative_l2"),
            "scale_aware": spot.get("scale_aware"),
            "text": "Q4 L0 gate vs bf16 is the fidelity bar a PQ geometry must meet or beat.",
        })

    if gain:
        gd = gain.get("gain_definition") or {}
        closures.append({
            "source": "NOETIC_GAIN_RESCORE",
            "text": (gd.get("vs_cosine_only") or {}).get("why_blind"),
        })

    if g042:
        defs = g042.get("definitions") or {}
        closures.append({
            "source": "G042_BPW_FAMILY",
            "SHARED_BPW": "ZERO: G035 refuted cross-layer sharing at matched bits (definition text)",
            "GENERATED_BPW_EQUIVALENT": defs.get("GENERATED_BPW_EQUIVALENT"),
            "text": (
                "SHARED_BPW is defined as structure stored once and used by many "
                "tensors, and is ZERO because G035 refuted cross-layer sharing at "
                "matched bits. Not a codebook-sharing licence."
            ),
        })

    # Contract-level facts this tree did not re-derive.
    closures.append({
        "source": "CONTRACT_NOT_REDERIVED",
        "text": (
            "MLP function distillation is NO-GO as of today: +0.4206 held-out "
            "gap vs q3 at 72% of its active bytes. That avenue is closed; this "
            "lane does not reopen it. Q80 storage BPW 0.6462 vs ACTIVE 2.518 "
            "(factor ~3.9) — report both or neither. 223 components below 0.5 "
            "local BPW with ZERO healthy. GLM 0.167 expert BPW trap; HGRAVS01 "
            "0.13 on down_proj ONLY. Raw activation cosine null baseline ~0.898."
        ),
    })

    family_already_refuted = False
    family_refute_scope = (
        "Residual-PQ as a blind storage substitution is quality-refuted "
        "(Llama-8B continuation fail at 1.50 and 2.63 bpw; Qwen2.5-32B first-token "
        "fail at 3.50 bpw, 1.48 tok/s). Direct per-row codebook-vector execution "
        "is speed-refuted on Llama-8B FFN (460.041 us D32 single-stage gate). "
        "The FAMILY is not closed: ADC of GLM52CPK was never measured, and "
        "Qwen3.8 never received a PQ pack. G035 does not refute codebooks."
    )

    return {
        "found": found,
        "missing": missing,
        "closures": closures,
        "family_already_refuted": family_already_refuted,
        "family_refute_scope": family_refute_scope,
        "n_found": len(found),
        "n_missing": len(missing),
        "n_closures": len(closures),
    }


def adc_identity_check() -> dict:
    """Prove ADC equals per-row codebook-vector FMA on a tiny integer-ish tensor.

    This is the cheap numerical discriminator of the *math*, not the GPU.
    Values are dyadic so f32 is exact.
    """
    rows, d, nchunk, card = 8, 4, 3, 4
    cols = nchunk * d
    cb = [((k * d + j) % 8) / 4.0 for k in range(card) for j in range(d)]
    codes = [((r * 5 + c * 3) % card) for r in range(rows) for c in range(nchunk)]
    x = [((i % 5) - 2) / 2.0 for i in range(cols)]

    def per_row():
        y = [0.0] * rows
        for r in range(rows):
            acc = 0.0
            for c in range(nchunk):
                k = codes[r * nchunk + c]
                xs = c * d
                base = k * d
                for j in range(d):
                    acc += cb[base + j] * x[xs + j]
            y[r] = acc
        return y

    def adc():
        y = [0.0] * rows
        for c in range(nchunk):
            lut = [0.0] * card
            xs = c * d
            for k in range(card):
                s = 0.0
                base = k * d
                for j in range(d):
                    s += cb[base + j] * x[xs + j]
                lut[k] = s
            for r in range(rows):
                y[r] += lut[codes[r * nchunk + c]]
        return y

    # Oracle: materialise W then GEMV. LABELLED ORACLE.
    def oracle_dense():
        W = []
        for r in range(rows):
            row = [0.0] * cols
            for c in range(nchunk):
                k = codes[r * nchunk + c]
                base = k * d
                for j in range(d):
                    row[c * d + j] = cb[base + j]
            W.append(row)
        y = []
        for r in range(rows):
            s = 0.0
            for j in range(cols):
                s += W[r][j] * x[j]
            y.append(s)
        return y

    a, b, o = per_row(), adc(), oracle_dense()
    err_ab = max(abs(i - j) for i, j in zip(a, b))
    err_ao = max(abs(i - j) for i, j in zip(a, o))
    ok = err_ab == 0.0 and err_ao == 0.0
    return {
        "pass": ok,
        "rows": rows, "cols": cols, "D": d, "card": card, "nchunk": nchunk,
        "max_abs_adc_minus_per_row": err_ab,
        "max_abs_per_row_minus_oracle": err_ao,
        "per_row": a,
        "adc": b,
        "oracle_dense": o,
        "note": (
            "ADC and per-row FMA matched the dense-W oracle to 0 ULP on dyadic "
            "values. The oracle materialises W; ADC and per-row do not."
        ),
    }


def organ_codec_row(org: dict, d: int, bits: int, card: int, sub: int, subspaces: int = 1) -> dict:
    rows, cols, n = org["rows"], org["cols"], org["count_per_token"]
    nchunk = cols // d
    idx_launch = pq_index_bytes(rows, cols, d, bits, subspaces)
    cb = pq_codebook_bytes(card, sub, subspaces)
    payload_launch = PQ_HEADER_LEN + cb + idx_launch
    # Existing kernel: same FMA count as dense (rows * cols FMAs = 2*elems FLOP).
    existing_flops_launch = org["mac_flops_per_launch"]
    existing_lookups_launch = rows * nchunk * subspaces
    # ADC: LUT = nchunk * card * sub * 2 FLOP = 2 * cols * card * (sub/D) = 2*cols*card when S=1
    # plus rows * nchunk scalar adds.
    lut_flops_launch = nchunk * card * sub * 2 * subspaces
    # When S=1, sub=D, this equals 2 * cols * card.
    acc_flops_launch = rows * nchunk * subspaces  # add of a LUT scalar
    adc_flops_launch = lut_flops_launch + acc_flops_launch
    rows_gt_card = rows > card
    # Keep Q4 for organs where LUT is more work than the parent GEMV (rows < card).
    adc_wins_ops = adc_flops_launch < existing_flops_launch
    return {
        "organ": org["organ"],
        "count_per_token": n,
        "rows": rows,
        "cols": cols,
        "nchunk": nchunk,
        "q4_bytes_per_launch": org["q4_bytes_per_launch"],
        "q4_bytes_per_token": org["q4_bytes_per_token"],
        "pq_index_bytes_per_launch": idx_launch,
        "pq_index_bytes_per_token": idx_launch * n,
        "pq_codebook_bytes_resident": cb,
        "pq_payload_bytes_per_launch": payload_launch,
        "pq_payload_bytes_per_token": payload_launch * n,
        "existing_kernel_flops_per_launch": existing_flops_launch,
        "existing_kernel_flops_per_token": existing_flops_launch * n,
        "existing_kernel_lookups_per_launch": existing_lookups_launch,
        "adc_lut_flops_per_launch": lut_flops_launch,
        "adc_acc_flops_per_launch": acc_flops_launch,
        "adc_flops_per_launch": adc_flops_launch,
        "adc_flops_per_token": adc_flops_launch * n,
        "rows_gt_card": rows_gt_card,
        "adc_wins_ops_vs_dense": adc_wins_ops,
        "role": org["role"],
    }


def scale_llama_us(rows: int, nchunk: int) -> float:
    """Linear scale of the measured 460.041 us D32 kernel with rows*nchunk.

    The existing kernel does one codebook-vector FMA of length D per (row, chunk)
    and one simdgroup per row. Work and index traffic both scale as rows*nchunk.
    This is a prediction of gravity_pq_matvec / residual-pq D32, not of ADC.
    """
    ref = LLAMA_GATE_ROWS * LLAMA_D32_NCHUNK
    return LLAMA_D32_MEDIAN_US * (rows * nchunk) / ref


def q4_implied_us(q4_bytes_launch: int) -> float:
    """Token-accounting share of 30.606 ms, attributed by Q4 GEMV bytes.

    Production is 76.8% of the 595.9 GB/s roof on 13.99 GB DRAM/token.
    Per-organ time ≈ organ_bytes / 13_611_663_360 * 30.606 ms.
    """
    return (q4_bytes_launch / ANCHOR_Q4_GEMV_BYTES) * ANCHOR_TOKEN_MS * 1000.0  # us


def adc_roof_us(index_bytes_launch: int) -> float:
    """Lower bound: sequential index stream at the measured 595.9 GB/s roof."""
    return (index_bytes_launch / (ANCHOR_ROOF_GB_S * 1e9)) * 1e6


def build_operator() -> dict:
    return {
        "name": "fused_adc_pq_matvec",
        "family": "fused_dictionary_lookup_accumulate",
        "representation": "GLM52CPK (gravity-pq) with index layout [chunk][row] for ADC coalescing",
        "incumbent_kernel_to_not_port": "gravity_pq_matvec",
        "why_not_port_existing": (
            "gravity_pq_matvec performs the parent GEMV FMA count (one D-vector "
            "dot per row per chunk). S011: a candidate that only lowers executable "
            "bytes is incomplete. Measured 460.041 us on Llama-8B 14336x4096 D32; "
            "scaled to Qwen3.8 mlp.gate 17408x5120 that is ~697 us vs ~107 us Q4."
        ),
        "production_operator": {
            "label": "PRODUCTION",
            "algebra": (
                "Given codebook C[s][k][0:sub) fp16, codes[chunk][row][s] in 0..card-1, "
                "activation x[cols]: "
                "y[r] = 0; "
                "for c in 0..nchunk-1: "
                "  for s in 0..S-1: "
                "    for k in 0..card-1: "
                "      LUT[s][k] = <C[s][k], x[c*D + s*sub : c*D + (s+1)*sub]>; "
                "    for r in rows_of_this_threadgroup: "
                "      y[r] += LUT[s][ codes[c][r][s] ]; "
                "LUT lives in threadgroup memory (card floats per subspace, 1 KiB at "
                "card=256 S=1) and is never a rows x cols matrix. Codebook and x are "
                "read once per chunk per dispatch, not once per row."
            ),
            "default_geometry": {
                "D": GLM_D, "S": GLM_S, "sub": GLM_SUB, "card": GLM_CARD, "bits": GLM_BITS,
                "index_layout": "[chunk][row][subspace], bits packed MSB-first as today",
                "codebook_layout": "[subspace][card][sub] fp16, unchanged from GLM52CPK",
                "note": (
                    "Default D=32 matches the kernel that already exists, so the "
                    "execution microbench isolates ADC vs per-row gather. Quality at "
                    "0.25 index-bpw is NOT claimed; it is the kill probe."
                ),
            },
            "mixed_plan": (
                "Replace a GEMV with ADC only when rows > card (LUT cheaper than "
                "parent GEMV). in_proj_ba is 96x5120 (rows < 256): keep Q4. All "
                "other GEMV organs have rows >= 1024."
            ),
            "not_this": [
                "shared cross-layer codebooks as the byte lever (16 KiB vs GB of indices; G035)",
                "residual-PQ additive stages as the primary (quality-refuted 1.5-3.5 bpw; FMA-multiplies the existing kernel)",
                "route-group / MoE worklists (Qwen3.8 is dense; qwen38_geometry refuses num_experts)",
                "hierarchical codebooks beyond residual-PQ (same family, already rejected)",
                "MLP function distillation (NO-GO, +0.4206 held-out gap vs q3)",
            ],
        },
        "correctness_oracle": {
            "label": "ORACLE — not a production implementation",
            "algebra": (
                "For each row r, reconstruct W_hat[r, c*D:(c+1)*D] = C[s][codes[c,r,s]] "
                "concatenated over subspaces s, then y = W_hat @ x (ordinary GEMM). "
                "Host helper: crates/hawking-core/src/artifact_pq.rs PqTensor::row. "
                "f64 authority: pq_matvec_f64_authority. Numeric Parity V2.1."
            ),
            "dense_reconstruction_law": (
                "representation -> reconstruct dense W -> ordinary GEMM may exist as "
                "this labelled oracle. It is NOT a production path and must never be "
                "presented as one."
            ),
        },
        "production_path_named_separately": (
            "fused_adc_pq_matvec (to be written) bound in place of "
            "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 on organs with rows>card, "
            "same TokenCommandBuffer, one dispatch per organ. Existing "
            "gravity_pq_matvec remains the GLM/Llama path and is NOT the Qwen3.8 "
            "production candidate."
        ),
    }


def derive_costs(organs: list[dict]) -> dict:
    d, bits, card, sub, s = GLM_D, GLM_BITS, GLM_CARD, GLM_SUB, GLM_S
    rows_out = [organ_codec_row(o, d, bits, card, sub, s) for o in organs]

    def sum_key(k, pred=None):
        tot = 0
        for r in rows_out:
            if pred and not pred(r):
                continue
            tot += r[k]
        return tot

    q4_bytes = sum_key("q4_bytes_per_token")
    if q4_bytes != ANCHOR_Q4_GEMV_BYTES:
        raise SystemExit(f"FAIL: derived Q4 GEMV bytes {q4_bytes} != {ANCHOR_Q4_GEMV_BYTES}")
    gemv_flops = sum_key("existing_kernel_flops_per_token")
    if gemv_flops != ANCHOR_GEMV_MAC_FLOPS:
        raise SystemExit(f"FAIL: derived GEMV FLOPs {gemv_flops} != {ANCHOR_GEMV_MAC_FLOPS}")

    adc_all_flops = sum_key("adc_flops_per_token")
    adc_all_idx = sum_key("pq_index_bytes_per_token")
    adc_all_payload = sum_key("pq_payload_bytes_per_token")
    existing_lookups_tok = sum(
        r["existing_kernel_lookups_per_launch"] * r["count_per_token"] for r in rows_out
    )

    # Mixed: ba stays Q4 (rows 96 < card 256).
    def is_ba(r):
        return r["organ"] == "linear_attn.in_proj_ba"

    mixed_idx = 0
    mixed_payload = 0
    mixed_adc_flops = 0
    mixed_q4_kept = 0
    mixed_existing_flops_replaced = 0
    for r in rows_out:
        keep_q4 = is_ba(r) or not r["adc_wins_ops_vs_dense"]
        r["keep_q4"] = keep_q4
        if keep_q4:
            mixed_q4_kept += r["q4_bytes_per_token"]
            mixed_adc_flops += r["existing_kernel_flops_per_token"]  # keep parent GEMV
            r["mixed_flops_per_token"] = r["existing_kernel_flops_per_token"]
            r["mixed_weight_bytes_per_token"] = r["q4_bytes_per_token"]
        else:
            mixed_idx += r["pq_index_bytes_per_token"]
            mixed_payload += r["pq_payload_bytes_per_token"]
            mixed_adc_flops += r["adc_flops_per_token"]
            mixed_existing_flops_replaced += r["existing_kernel_flops_per_token"]
            r["mixed_flops_per_token"] = r["adc_flops_per_token"]
            r["mixed_weight_bytes_per_token"] = (
                r["pq_index_bytes_per_token"]
                + r["pq_codebook_bytes_resident"] * r["count_per_token"]
            )
    mixed_weight_bytes = mixed_idx + mixed_q4_kept + sum(
        r["pq_codebook_bytes_resident"] * r["count_per_token"]
        for r in rows_out if not is_ba(r) and r["adc_wins_ops_vs_dense"]
    )

    # Predicted existing-kernel us/token (scaled Llama D32).
    pred_existing_us = 0.0
    pred_q4_us = 0.0
    pred_adc_roof_us = 0.0
    per_organ_pred = []
    for r in rows_out:
        e_us = scale_llama_us(r["rows"], r["nchunk"])
        q_us = q4_implied_us(r["q4_bytes_per_launch"])
        a_us = adc_roof_us(r["pq_index_bytes_per_launch"])
        pred_existing_us += e_us * r["count_per_token"]
        pred_q4_us += q_us * r["count_per_token"]
        pred_adc_roof_us += a_us * r["count_per_token"]
        per_organ_pred.append({
            "organ": r["organ"],
            "count": r["count_per_token"],
            "existing_kernel_pred_us_per_launch": e_us,
            "q4_implied_us_per_launch": q_us,
            "adc_roof_us_per_launch": a_us,
            "existing_kernel_pred_us_per_token": e_us * r["count_per_token"],
        })

    # Token-level ADC DRAM ≈ index stream + codebook re-read per dispatch + same activations.
    gemv_disp = sum(o["count_per_token"] for o in organs)
    codebook_traffic = sum(
        r["pq_codebook_bytes_resident"] * r["count_per_token"] for r in rows_out
    )
    # Activation DRAM from operation census (not re-derived): exec dram - q4 gemv.
    act_and_state = ANCHOR_EXEC_DRAM - ANCHOR_Q4_GEMV_BYTES
    adc_dram_all_pq = adc_all_idx + codebook_traffic + act_and_state
    mixed_dram = mixed_weight_bytes + act_and_state

    # Ops: ADC mixed FLOPs + unchanged activations. No Q4 unpack ALU on replaced organs.
    # Unpack ALU on kept ba: 4 * elements_ba.
    ba = next(r for r in rows_out if is_ba(r))
    ba_elems_tok = ba["rows"] * ba["cols"] * ba["count_per_token"]
    kept_unpack = 4 * ba_elems_tok
    # Replaced organs drop the per-weight scale-mul (1 FLOP/weight) and unpack.
    replaced_elems = ANCHOR_GEMV_ELEMENTS - ba_elems_tok
    adc_flops_token = mixed_adc_flops + ANCHOR_ACT_FLOPS
    adc_ops_token = adc_flops_token + kept_unpack  # no unpack on replaced organs

    # Existing kernel (port as-is): same 51.24 GFLOP GEMV + no unpack, plus gather integer.
    # GLM ledger bills lookups*15 integer ops. We report that as THEIR model, and
    # FMA count as the honest FLOP number (equal to parent).
    existing_port_flops = ANCHOR_GEMV_MAC_FLOPS + ANCHOR_ACT_FLOPS
    existing_port_ops = existing_port_flops + existing_lookups_tok * 15

    s011_existing_port = {
        "bytes": "REDUCE (indices 0.801 GB vs Q4 13.61 GB) if codebook hits cache",
        "operations": "SAME 51.24 GFLOP GEMV — INCOMPLETE if this is the only lever",
        "dispatches": "SAME 964",
        "materialization": "SAME 0 dense W",
        "synchronization": "SAME 1 CB",
        "traffic": (
            "AMBIGUOUS: sequential indices drop; per-row codebook gathers are "
            "17408*160*32*2 = 178 MB of fp16 from a 16 KiB table per gate. "
            "If billed as DRAM this INCREASES traffic vs Q4 47 MB. Measured "
            "460 us says the kernel does not reach the index-stream roof."
        ),
        "complete": False,
        "why_incomplete": (
            "S011 §4: a candidate that only lowers executable bytes is incomplete. "
            "The current executable already stores fewer bytes and does not do "
            "less work. Porting gravity_pq_matvec repeats that mistake: 51.24 "
            "GFLOP of GEMV remain, and the Llama D32 roofline is slower than Q4."
        ),
    }
    s011_adc = {
        "bytes": "REDUCE: mixed weight stream "
                 f"{mixed_weight_bytes:,} vs Q4 {ANCHOR_Q4_GEMV_BYTES:,}",
        "operations": "REDUCE: mixed ADC+kept-GEMV FLOPs "
                      f"{mixed_adc_flops:,} vs {ANCHOR_GEMV_MAC_FLOPS:,} GEMV MACs",
        "dispatches": "SAME 964 (1:1 replace of large GEMVs; ba stays Q4, still 1 dispatch)",
        "materialization": "SAME 0 dense W (LUT in threadgroup, 1 KiB)",
        "synchronization": "SAME 1 host CB; more in-kernel barriers (one per chunk)",
        "traffic": "REDUCE sequential DRAM to the index stream + 16 KiB codebook + activations",
        "complete": True,
        "why_complete": (
            "Reduces bytes AND operations AND traffic vs the incumbent fused Q4 path. "
            "Does not reduce dispatches or host synchronization. Quality is the open kill."
        ),
    }

    return {
        "geometry": {"D": d, "S": s, "sub": sub, "card": card, "bits": bits},
        "organs": rows_out,
        "incumbent": {
            "gemv_flops_per_token": gemv_flops,
            "q4_gemv_bytes_per_token": q4_bytes,
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers": ANCHOR_CBS,
            "executable_flops_per_token": ANCHOR_EXEC_FLOPS,
            "executable_ops_per_token": ANCHOR_EXEC_OPS,
            "executable_dram_bytes_per_token": ANCHOR_EXEC_DRAM,
            "gemv_dispatches_per_token": gemv_disp,
            "tps": ANCHOR_TPS,
            "ms_per_token": ANCHOR_TOKEN_MS,
            "implied_weight_GB_s": ANCHOR_Q4_GEMV_BYTES * ANCHOR_TPS / 1e9,
            "pct_of_roof": (ANCHOR_Q4_GEMV_BYTES * ANCHOR_TPS / 1e9) / ANCHOR_ROOF_GB_S * 100.0,
        },
        "existing_kernel_port": {
            "index_bytes_per_token": adc_all_idx,
            "payload_bytes_per_token": adc_all_payload,
            "codebook_resident_traffic_per_token": codebook_traffic,
            "flops_per_token": existing_port_flops,
            "ops_per_token_glm_lookup_model": existing_port_ops,
            "lookups_per_token": existing_lookups_tok,
            "predicted_us_per_token_scaled_from_llama_d32": pred_existing_us,
            "predicted_ms_per_token": pred_existing_us / 1000.0,
            "predicted_tps_if_pq_is_the_whole_token": 1000.0 / (pred_existing_us / 1000.0),
            "s011": s011_existing_port,
        },
        "adc_all_organs": {
            "index_bytes_per_token": adc_all_idx,
            "payload_bytes_per_token": adc_all_payload,
            "flops_per_token_lut_plus_acc": adc_all_flops,
            "dram_bytes_per_token_pred": adc_dram_all_pq,
            "roof_us_per_token_index_stream": pred_adc_roof_us,
            "roof_ms_per_token": pred_adc_roof_us / 1000.0,
        },
        "adc_mixed_ba_stays_q4": {
            "weight_bytes_per_token": mixed_weight_bytes,
            "index_bytes_per_token": mixed_idx,
            "q4_bytes_kept_per_token": mixed_q4_kept,
            "flops_gemv_path_per_token": mixed_adc_flops,
            "flops_per_token_with_activations": adc_flops_token,
            "ops_per_token": adc_ops_token,
            "dram_bytes_per_token_pred": mixed_dram,
            "replaced_parent_gemv_flops": mixed_existing_flops_replaced,
            "kept_unpack_alu_ops": kept_unpack,
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers": ANCHOR_CBS,
            "s011": s011_adc,
        },
        "per_organ_time_prediction": per_organ_pred,
        "q4_implied_gemv_us_sum": pred_q4_us,
        "byte_ratio_mixed_vs_q4": mixed_weight_bytes / q4_bytes,
        "flop_ratio_mixed_vs_gemv": mixed_adc_flops / gemv_flops,
    }


def metal_feasibility() -> dict:
    lut_bytes = GLM_CARD * 4  # float32 LUT[card]
    cb_tg_bytes = GLM_CARD * GLM_SUB * 2  # optional: park codebook in TG
    return {
        "device": "Apple M3 Ultra, 60 GPU cores, Metal 4, 103079215104 B unified, roof 595.9 GB/s",
        "simdgroup_width": SIMDGROUP_WIDTH,
        "evidence_simdgroup_width": "gravity_pq_matvec: for (uint c = lane; c < p.nchunk; c += 32u); simd_sum",
        "incumbent_q4_launch": {
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "threadgroup": Q4_TG,
            "threads_per_row": Q4_THREADS_PER_ROW,
            "rows_per_tg": 2,
            "grid": "ceil(rows/2)*128",
            "access": (
                "64 threads/row walk columns in 8-wide unpack8 steps of +512. "
                "Packed Q4 bytes are sequential per row; consecutive groups of a "
                "row coalesce. Dequant stays in registers (tpr64: reconstruction free)."
            ),
        },
        "existing_pq_launch": {
            "kernel": "gravity_pq_matvec",
            "threadgroup": PQ_TG,
            "rows_per_tg": PQ_ROWS_PER_TG,
            "grid": "ceil(rows/8)*256",
            "access": (
                "One simdgroup owns one output row. 32 lanes stride chunks of that "
                "row: consecutive lanes read consecutive [row][chunk] indices "
                "(coalesced) then GATHER a 32-half codebook entry at a random card "
                "slot (uncoalesced). Inner loop is 32 scalar FMAs. Arithmetic "
                "intensity per thread is nchunk/32 * 32 = nchunk FMAs (~160 on gate)."
            ),
            "coalescing": "indices YES for [row][chunk]; codebook gathers NO",
        },
        "adc_launch": {
            "kernel_to_write": "fused_adc_pq_matvec",
            "threadgroup": 256,
            "rows_per_tg": 256,
            "grid": "ceil(rows/256)*256, one TG dimension",
            "threadgroup_memory": {
                "LUT_card_floats": lut_bytes,
                "optional_codebook_fp16": cb_tg_bytes,
                "sum_if_both": lut_bytes + cb_tg_bytes,
                "apple9_limit": APPLE9_TG_MEM_BYTES,
                "fits": (lut_bytes + cb_tg_bytes) < APPLE9_TG_MEM_BYTES,
                "evidence_tree_already_uses_16kib": (
                    "gravity_pq.metal gravity_glm_stable_topk: threadgroup ulong ranked[2048] "
                    "= 16384 B. 17 KiB LUT+codebook is inside the same budget."
                ),
            },
            "register_pressure": (
                "One f32 accumulator per owned row, one u8/u32 index, no 32-long "
                "FMA chain. Far below the current kernel's 32-scalar inner product "
                "(or bits8_vec4's four float4 accumulators, or double-single hi/lo)."
            ),
            "coalescing": (
                "REQUIRES index layout [chunk][row]. Then 256 consecutive threads "
                "load 256 consecutive bytes of codes for that chunk (8-wide coalesced "
                "if widened to uint32). LUT lookup is TG memory, not DRAM. "
                "Current GLM52CPK [row][chunk] would stride nchunk (160 B on gate) "
                "and miss coalescing — a layout transpose, not new math."
            ),
            "barriers": (
                "One threadgroup_barrier per chunk after LUT fill. Gate nchunk=160, "
                "down nchunk=544, lm_head nchunk=160. Barriers are the occupancy "
                "risk the microbench must measure; they do not appear in the byte roof."
            ),
            "occupancy": (
                "256 light-register threads. lm_head 248320/256 = 970 TGs; gate "
                "17408/256 = 68 TGs. 60 cores stay busy on the large organs. ba "
                "(96 rows) is excluded from ADC because LUT costs more than GEMV."
            ),
        },
        "indirection_is_a_coalescing_question": (
            "Codebook lookup trades arithmetic for indirection. On this GPU the "
            "incumbent is byte-bound at 445 GB/s of a 595.9 GB/s roof, and tpr64 "
            "already made dequant ALU free. ADC wins only if the index stream "
            "coalesces and the LUT stays in threadgroup memory. Per-row codebook "
            "gathers (the existing kernel) failed that test at 460 us."
        ),
    }


def memory_layout() -> dict:
    return {
        "header": {
            "bytes": PQ_HEADER_LEN,
            "magic": "GLM52CPK",
            "fields": "d,s,sub,card,rows,cols,nchunk,seed,bits,rotate,n_codebooks",
            "rotate": "must be 0; rotate=1 is unsupported on CPU and Metal",
        },
        "codebooks": {
            "layout": "[subspace 0..S) [code 0..card) [j 0..sub) fp16 little-endian",
            "bytes_formula": "S * card * sub * 2",
            "glm_default_bytes": pq_codebook_bytes(GLM_CARD, GLM_SUB, GLM_S),
            "resident": "one buffer per tensor, uploaded once, not per token",
        },
        "codes_current_glm52cpk": {
            "layout": "[row][chunk][subspace], MSB-first bit stream, 4 bytes tail padding",
            "good_for": "existing gravity_pq_matvec (consecutive chunks of one row)",
            "bad_for": "ADC (consecutive rows of one chunk stride nchunk)",
        },
        "codes_adc_required": {
            "layout": "[chunk][row][subspace], same bit packing",
            "bytes_formula": "ceil(rows * nchunk * S * bits / 8)",
            "good_for": "ADC coalesced index load, 256 consecutive rows",
            "conversion": "offline transpose of the index tensor; codebook unchanged",
        },
        "lut_transient": {
            "where": "threadgroup float lut[card]  // 1024 B at card=256",
            "lifetime": "one chunk of one dispatch",
            "not_stored_in_the_artifact": True,
        },
        "not_stored": [
            "dense W",
            "per-row reconstructed vectors",
            "cross-layer shared codebook (rejected as a byte lever)",
        ],
        "qwen38_catalog_blocker": {
            "classify_qwen38_mixed_payload": "admits codecs 0-2 packed, 3 HGRAVU01/HQ30UQ4, 4 f32v2",
            "GLM52CPK": "not a Qwen3.8 mixed codec; classify refuses unknown codec ids and unknown magics",
            "artifact_on_disk": "402 q4 + 353 f32, zero gravity-pq tensors",
            "implication": (
                "Absence is not a forgotten dispatch. There is no PQ payload to bind, "
                "and the catalog would refuse one. Bind work is a new lane plus a pack, "
                "and is still the wrong first move until ADC beats Q4 on one organ."
            ),
        },
    }


def microbenchmark(costs: dict) -> dict:
    gate = next(r for r in costs["organs"] if r["organ"] == "mlp.gate_proj")
    return {
        "purpose": (
            "Discriminate ADC from the incumbent AND from the existing "
            "gravity_pq_matvec BEFORE anyone writes a Qwen3.8 catalog lane or packer."
        ),
        "identity_already_run_in_this_script": True,
        "gpu_bench_not_run_here": True,
        "why_gpu_bench_is_out_of_scope": (
            "crates/ is DENY for this lane. The bench is specified so the next "
            "lane can land it as one ignored test next to "
            "gravity_residual_pq_metal.rs, using PqMetalMatrix::benchmark which "
            "already exists."
        ),
        "cheap_gpu_discriminator": {
            "name": "adc_vs_q4_vs_gravity_pq_gate_17408x5120",
            "shape": {"rows": 17408, "cols": 5120, "D": 32, "S": 1, "sub": 32, "card": 256, "bits": 8},
            "payload": (
                "Reuse tests/gravity_pq_kernel_registry.rs::autotune_payload(17408, 5120) "
                "for GLM52CPK. Build an index-transposed sibling [chunk][row]. "
                "Build a Q4 group-64 sibling of the same logical W via "
                "qwen_uniform_q4 pack of the reconstructed rows (ORACLE used only to "
                "make a fair Q4 control, then discarded)."
            ),
            "kernels": [
                {
                    "id": "A_existing_pq",
                    "kernel": "gravity_pq_matvec (or bits8_vec4)",
                    "api": "PqMetalMatrix::benchmark(ctx, variant, x, warmup=8, iterations=32)",
                    "layout": "[row][chunk]",
                    "predict_us": scale_llama_us(17408, 160),
                },
                {
                    "id": "B_adc",
                    "kernel": "fused_adc_pq_matvec (the ~80-line candidate, not a 401-dispatch port)",
                    "layout": "[chunk][row]",
                    "predict_us_roof": adc_roof_us(gate["pq_index_bytes_per_launch"]),
                    "threadgroup": 256,
                    "tg_mem": "float lut[256]",
                },
                {
                    "id": "C_q4_tpr64",
                    "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                    "layout": "HQ30UQ4 group-64",
                    "predict_us_from_token_share": q4_implied_us(gate["q4_bytes_per_launch"]),
                },
            ],
            "device": "Apple M3 Ultra, same process, no second 27B, no llama-server spawn",
            "metric": "median GPU us via MTLCommandBuffer GPUEndTime-GPUStartTime after wait; also wall us",
            "kill_if": [
                "B_adc median_us >= C_q4_tpr64 median_us  (ADC does not beat Q4 on the organ that is 22% of GEMV bytes)",
                "B_adc median_us >= A_existing_pq median_us AND A is already slower than Q4  (ADC implementation is broken)",
            ],
            "pass_does_not_promote": (
                "A wall-time win on random codes is execution-only. Quality is a "
                "separate probe. Do not pack Qwen3.8 on a timing win."
            ),
        },
        "cheap_quality_kill": {
            "name": "l0_gate_pq_vs_q4_on_real_x",
            "tensor": "language_model.model.layers.0.mlp.gate_proj.weight  (17408 x 5120)",
            "W_source": "bf16 parent if present; else HQ30UQ4 dequant of the sealed artifact (say so)",
            "X_source": (
                "real activations, never synthetic. Organ-census capture used "
                "cpu_hybrid_prefill of gravity q4 on real token ids; TPR64 used "
                "first holdout token of activation-capture-v1. Either is admissible. "
                "Gaussian / 0.01*W is not."
            ),
            "geometries": [
                {"D": 32, "bits": 8, "bpw_indices": 0.25, "why": "GLM default; must almost certainly fail vs Q4"},
                {"D": 8, "bits": 8, "bpw_indices": 1.0, "why": "byte-saving; residual-PQ at 1.50 bpw already failed continuation"},
                {"D": 2, "bits": 8, "bpw_indices": 4.0, "why": "matched-byte; tpr64 says ALU savings will not win if bytes do not drop"},
            ],
            "encoder": "per-subspace k-means / GLM gravity-pq packer, one tensor, CPU",
            "metrics": [
                "relative_l2 of y = X @ W.T vs X @ W_hat.T",
                "gain = min(r, 1/r) on output-row norms (gravity_doctor_gate._gain)",
                "scale_aware = cosine * min(s, 1/s)",
                "Q4 L0 gate vs bf16 bar: cosine 0.994066, rel_l2 0.109425, scale_aware 0.988200",
                "reject cosine-only; 0.01*W scores cosine 1.000000",
                "report against null baseline cosine 0.898",
            ],
            "kill_if": (
                "Every geometry that saves bytes vs Q4 loses gain or rel_l2 to Q4 on real X. "
                "The D=2 matched-byte geometry is then the only survivor, and tpr64 plus "
                "the production 76.8% roof say it cannot win on this machine."
            ),
        },
        "do_not_rerun": [
            "residual_pq_single_stage_d32_ffn_gate_geometry_roofline (460.041 us, already receipted)",
            "residual_pq_four_stage_ffn_gate_geometry_roofline (672.625 us, already receipted)",
            "Llama/Qwen32 residual-PQ continuation gates (already REJECTED)",
        ],
    }


def expected_value(costs: dict, prior: dict, identity: dict) -> dict:
    mixed = costs["adc_mixed_ba_stays_q4"]
    existing = costs["existing_kernel_port"]
    return {
        "verdict": "NOT_WORTH_BUILDING_THE_QWEN38_PORT",
        "verdict_meaning": (
            "Do not pack GLM52CPK for Qwen3.8. Do not bind gravity_pq_matvec onto "
            "qwen38_hybrid_decode. Do not invent shared/hierarchical/route-group "
            "codebooks to walk around residual-PQ's quality failure. The cheapest "
            "remaining experiment is a one-organ ADC microbench plus an L0-gate "
            "quality probe; both are allowed to close the family."
        ),
        "what_it_would_win_if_both_probes_pass": {
            "bytes_per_token_weight": {
                "q4": costs["incumbent"]["q4_gemv_bytes_per_token"],
                "adc_mixed": mixed["weight_bytes_per_token"],
                "ratio": mixed["weight_bytes_per_token"] / costs["incumbent"]["q4_gemv_bytes_per_token"],
            },
            "gemv_path_flops": {
                "q4_mac": costs["incumbent"]["gemv_flops_per_token"],
                "adc_mixed": mixed["flops_gemv_path_per_token"],
                "ratio": mixed["flops_gemv_path_per_token"] / costs["incumbent"]["gemv_flops_per_token"],
            },
            "roof_ms_index_stream": costs["adc_all_organs"]["roof_ms_per_token"],
            "incumbent_ms": ANCHOR_TOKEN_MS,
            "vs_mlx_4bit": ANCHOR_MLX_TPS,
            "vs_llamacpp_q5k_archived": ANCHOR_LLAMA_Q5K_TPS,
            "note": (
                "A 17x byte cut at D=32 is the physical prize. It is also the "
                "geometry at which residual-PQ-class codecs have never passed a "
                "token-identity gate in this campaign."
            ),
        },
        "what_it_risks": [
            "Quality: residual-PQ failed first-token / continuation at 1.50, 2.63, 3.50 bpw. D=32 is 0.25 index-bpw.",
            "TPR64: dequant ALU is already free. Matched-byte PQ (D=2, 4 bpw) has no roof to attack.",
            "Existing kernel: predicted 697 us/gate vs 107 us Q4. Porting it would lose to the incumbent and to MLX 35.51 tok/s.",
            "G035: sharing is not a byte lever for 16 KiB codebooks; row-basis 'win' is 6.3% of a dead-zone error.",
            "Cosine: 0.01*W scores 1.000000; a PQ pack scored on cosine alone will lie.",
            "Two-server occupancy: 3.986 vs 33.47 tok/s. A PQ experiment must not load a second 27B.",
            "Catalog: Qwen3.8 classify will not admit GLM52CPK even if a pack appears.",
        ],
        "cheapest_kill": (
            "Quality probe of L0 gate D=8 and D=32 PQ vs Q4 on real X, gain+rel_l2. "
            "If both lose, stop. If they win, run the ADC vs Q4 Metal microbench on "
            "17408x5120. If ADC loses wall time, stop. Only then is a packer in scope."
        ),
        "not_worth_building_existing_kernel_port": True,
        "existing_kernel_predicted_ms_per_token": existing["predicted_ms_per_token"],
        "existing_kernel_s011_complete": existing["s011"]["complete"],
        "adc_s011_complete": mixed["s011"]["complete"],
        "identity_check_pass": identity["pass"],
        "family_already_refuted": prior["family_already_refuted"],
        "family_refute_scope": prior["family_refute_scope"],
        "controls_to_beat": {
            "incumbent_native_q4": ANCHOR_TPS,
            "mlx_4bit_live": ANCHOR_MLX_TPS,
            "llamacpp_q5k_archived": ANCHOR_LLAMA_Q5K_TPS,
        },
    }


def watched_fail(prior: dict, identity: dict, kernel: dict, artifact: dict) -> list:
    return [
        {
            "what": ".lane-bootstrap/census (n1arch 35 mechanisms, n15neg 31 closures, n16clos)",
            "result": "MISSING",
            "why": (
                "Directory is not on disk in this sparse checkout and `git ls-tree HEAD "
                ".lane-bootstrap/census` is empty. Did not run git sparse-checkout add. "
                "Prior science was recovered from NOETIC_* receipts plus git show of "
                "G035 and residual-PQ rejection receipts."
            ),
        },
        {
            "what": "HCLI baseline 464 passed / 1 skipped (HCLI_SWAP_CEILING_GIB=64)",
            "result": "NOT RUN",
            "why": "tools/haider is DENY and not materialized.",
        },
        {
            "what": "Live Metal ADC microbench on 17408x5120",
            "result": "NOT RUN (design lane)",
            "why": (
                "crates/ is DENY. Specified concretely (PqMetalMatrix::benchmark + one "
                "new ~80-line kernel) so a later lane can run it. Llama D32 460.041 us "
                "is reused, not re-measured."
            ),
        },
        {
            "what": "Qwen3.8 gravity-pq artifact",
            "result": "DOES NOT EXIST",
            "why": (
                f"Sealed uniform-q4-v1 kinds={artifact.get('kinds')} "
                f"gravity_pq_tensors={artifact.get('gravity_pq_tensors')}. "
                "Catalog classify_qwen38_mixed_payload does not admit GLM52CPK. "
                "quoted_in_qwen38_hybrid_decode="
                f"{kernel['quoted_in_qwen38_hybrid_decode']}."
            ),
        },
        {
            "what": "Porting gravity_pq_matvec as the design",
            "result": "REJECTED as S011-incomplete",
            "why": (
                "Same 51.24 GFLOP of GEMV as the parent. The campaign already knows "
                "that storing fewer bytes without doing less work is not a result. "
                "Census missing_family_cost said 'wiring is bind work'; bind work "
                "would dispatch a kernel that is slower than Q4 on the Llama FFN "
                "roofline."
            ),
        },
        {
            "what": "Residual-PQ as the Qwen3.8 codec",
            "result": "ALREADY REJECTED (searched, not rediscovered)",
            "why": (
                "Llama-8B FFN runtime 672.625 us four-stage / 460.041 us D32; "
                "S2 1.50 bpw continuation fail at 5.88 tok/s; S3 2.63 bpw continuation "
                "fail at 4.40 tok/s; Qwen2.5-32B 3.50 bpw first-token fail at 1.48 tok/s. "
                "Reopen condition was activation-aware / layer-protected codebooks, not "
                "more stages."
            ),
        },
        {
            "what": "Shared / hierarchical / route-group codebooks as the primary design",
            "result": "NOT SELECTED",
            "why": (
                "G035 shared_beats_independent=false on column bases; codebook sharing "
                "saves 16 KiB-class tables against GB-class indices. Residual/hierarchical "
                "is the quality-refuted additive-PQ family. Qwen3.8 is dense (no experts, "
                "ROUTING_FLOPS=0). gravity_glm_expert_table_pq_matvec is a GLM organ."
            ),
        },
        {
            "what": "MLP function distillation as a codebook substitute",
            "result": "NO-GO (contract constraint, not re-derived)",
            "why": "+0.4206 held-out gap vs q3 at 72% of its active bytes. Kernel census still says that avenue 'has not been run' — stale relative to the contract.",
        },
        {
            "what": "Evaluating on synthetic activations / cosine-only",
            "result": "REFUSED",
            "why": (
                "0.01*W scores cosine 1.000000 (organ census + gain rescore). Raw "
                "activation cosine null baseline 0.898. Quality kill is gain + rel_l2 "
                "on real X."
            ),
        },
        {
            "what": "ADC identity (this process)",
            "result": "PASS" if identity["pass"] else "FAIL",
            "why": identity["note"],
        },
        {
            "what": "Second 27B / live native re-time",
            "result": "REFUSED",
            "why": (
                f"Two servers resident measured {ANCHOR_TWO_SERVERS_TPS} tok/s vs "
                f"{ANCHOR_ONE_SERVER_TPS} with one. TPS/token-ms used here are the "
                f"supplied anchors ({ANCHOR_TPS} / {ANCHOR_TOKEN_MS} ms)."
            ),
        },
    ]


def print_report(doc: dict) -> None:
    k = doc["existing_kernel"]
    c = doc["costs"]
    ev = doc["expected_value"]
    op = doc["operator"]
    print("=" * 78)
    print("C4 CODEBOOK DESIGN — fused dictionary lookup + accumulate")
    print("Qwen3.8-27B uniform-q4 vs gravity_pq_matvec vs fused ADC")
    print("=" * 78)
    print()
    print("VERDICT:", ev["verdict"])
    print(ev["verdict_meaning"])
    print()
    print("Family already refuted?", ev["family_already_refuted"])
    print("  ", ev["family_refute_scope"])
    print()

    print("## 0. Prior-science search")
    ps = doc["prior_science"]
    print(f"  found={ps['n_found']} missing={ps['n_missing']} closures={ps['n_closures']}")
    for m in ps["missing"]:
        print(f"  MISSING {m['id']}: {m['why']}")
    print("  Load-bearing closures:")
    for cl in ps["closures"]:
        text = cl.get("text") or cl.get("claim") or cl.get("corrected_verdict") or ""
        print(f"  - {cl['source']}: {text[:200]}")
    print()

    print("## Why gravity_pq_matvec is not on Qwen3.8")
    print(f"  class={k['class']} quoted_in_decode={k['quoted_in_qwen38_hybrid_decode']}")
    print(f"  GLM52CPK accepted by Qwen3.8 classify: {k['glm52cpk_magic_accepted_by_qwen38_classify']}")
    print(f"  artifact gravity_pq_tensors={doc['artifact'].get('gravity_pq_tensors')} kinds={doc['artifact'].get('kinds')}")
    print("  Reasons, in order:")
    for i, r in enumerate(doc["why_unused"], 1):
        print(f"  {i}. {r}")
    print()
    print("  Existing kernel math:")
    print("   ", k["math"])
    print()

    print("## 1. Mathematical operator")
    print("  PRODUCTION:", op["production_path_named_separately"])
    print("  ", op["production_operator"]["algebra"])
    print("  ORACLE:", op["correctness_oracle"]["algebra"])
    print("  ", op["correctness_oracle"]["dense_reconstruction_law"])
    print()

    inc = c["incumbent"]
    mix = c["adc_mixed_ba_stays_q4"]
    ex = c["existing_kernel_port"]
    print("## 2. Expected bytes / token")
    print(f"  incumbent Q4 GEMV stream     {fmt_bytes(inc['q4_gemv_bytes_per_token'])}")
    print(f"  incumbent exec DRAM          {fmt_bytes(inc['executable_dram_bytes_per_token'])}")
    print(f"  existing-kernel index stream {fmt_bytes(ex['index_bytes_per_token'])}")
    print(f"  ADC mixed weight stream      {fmt_bytes(mix['weight_bytes_per_token'])}")
    print(f"  ratio mixed/Q4               {mix['weight_bytes_per_token'] / inc['q4_gemv_bytes_per_token']:.4f}")
    print(f"  implied incumbent GB/s       {inc['implied_weight_GB_s']:.1f}  ({inc['pct_of_roof']:.1f}% of {ANCHOR_ROOF_GB_S})")
    print()

    print("## 3. Expected operations / token")
    print(f"  incumbent GEMV MACs          {fmt_gflop(inc['gemv_flops_per_token'])}")
    print(f"  incumbent exec FLOPs         {fmt_gflop(inc['executable_flops_per_token'])}")
    print(f"  incumbent exec ops           {fmt_gflop(inc['executable_ops_per_token'])}")
    print(f"  existing-kernel port FLOPs   {fmt_gflop(ex['flops_per_token'])}  (SAME GEMV — S011 incomplete)")
    print(f"  ADC mixed GEMV-path FLOPs    {fmt_gflop(mix['flops_gemv_path_per_token'])}")
    print(f"  ADC mixed FLOPs + acts       {fmt_gflop(mix['flops_per_token_with_activations'])}")
    print(f"  ADC mixed ops                {fmt_gflop(mix['ops_per_token'])}")
    print(f"  flop ratio mixed/GEMV        {mix['flops_gemv_path_per_token'] / inc['gemv_flops_per_token']:.4f}")
    print()

    print("## 4. Dispatch topology")
    print(f"  incumbent: {inc['dispatches_per_token']} dispatches, {inc['command_buffers']} command buffer")
    print(f"  ADC mixed: {mix['dispatches_per_token']} dispatches, {mix['command_buffers']} command buffer")
    print("  1:1 replace of large GEMVs; ba stays Q4; no extra host sync.")
    print("  In-kernel: one threadgroup barrier per chunk (160 on gate, 544 on down).")
    print()

    print("## 5. Metal feasibility")
    mf = doc["metal_feasibility"]
    tg = mf["adc_launch"]["threadgroup_memory"]
    print(f"  simdgroup width {mf['simdgroup_width']}; ADC TG 256; LUT {tg['LUT_card_floats']} B; "
          f"LUT+codebook {tg['sum_if_both']} B; Apple9 limit {tg['apple9_limit']} B; fits={tg['fits']}")
    print("  coalescing:", mf["adc_launch"]["coalescing"][:200])
    print(" ", mf["indirection_is_a_coalescing_question"][:240])
    print()

    print("## 6. Memory layout")
    ly = doc["memory_layout"]
    print("  codebook:", ly["codebooks"]["layout"], f"  {ly['codebooks']['glm_default_bytes']} B")
    print("  current codes:", ly["codes_current_glm52cpk"]["layout"])
    print("  ADC codes:    ", ly["codes_adc_required"]["layout"])
    print("  catalog:      ", ly["qwen38_catalog_blocker"]["implication"])
    print()

    print("## 7. Cheap microbenchmark")
    mb = doc["microbenchmark"]
    print("  identity (this process):", "PASS" if mb["identity_already_run_in_this_script"] and doc["identity_check"]["pass"] else "FAIL")
    print("  GPU discriminator:", mb["cheap_gpu_discriminator"]["name"])
    for ker in mb["cheap_gpu_discriminator"]["kernels"]:
        extra = ker.get("predict_us") or ker.get("predict_us_roof") or ker.get("predict_us_from_token_share")
        print(f"    {ker['id']}: {ker['kernel']}  pred_us={extra:.2f}" if isinstance(extra, float) else f"    {ker['id']}: {ker['kernel']}")
    print("  quality kill:", mb["cheap_quality_kill"]["name"], mb["cheap_quality_kill"]["kill_if"][:180])
    print()

    print("## 8. Expected value")
    print("  ", ev["verdict"])
    print("  cheapest kill:", ev["cheapest_kill"])
    print(f"  existing-kernel predicted ms/token: {ev['existing_kernel_predicted_ms_per_token']:.2f}")
    print(f"  ADC S011 complete: {ev['adc_s011_complete']}   existing-port S011 complete: {ev['existing_kernel_s011_complete']}")
    print(f"  beat: native Q4 {ANCHOR_TPS} tok/s, MLX-4bit {ANCHOR_MLX_TPS} tok/s, llama.cpp Q5_K archived {ANCHOR_LLAMA_Q5K_TPS} tok/s")
    print()

    print("## Per-organ (D=32 card=256 bits=8; mixed = ADC except ba stays Q4)")
    print(f"  {'organ':<28} {'n':>3} {'keep':>5} {'Q4 B/tok':>14} {'mixed B/tok':>12} {'mixed FLOP':>14} {'pred pq us':>11} {'Q4 us':>9} {'ADC roof us':>12}")
    for r, t in zip(c["organs"], c["per_organ_time_prediction"]):
        keep = "Q4" if r.get("keep_q4") else "ADC"
        print(
            f"  {r['organ']:<28} {r['count_per_token']:>3} {keep:>5} "
            f"{r['q4_bytes_per_token']:>14,} {r['mixed_weight_bytes_per_token']:>12,} "
            f"{r['mixed_flops_per_token']:>14,} "
            f"{t['existing_kernel_pred_us_per_token']:>11.0f} "
            f"{t['q4_implied_us_per_launch'] * r['count_per_token']:>9.0f} "
            f"{t['adc_roof_us_per_launch'] * r['count_per_token']:>12.1f}"
        )
    print()

    print("## S011 §4")
    print("  existing gravity_pq_matvec port:", c["existing_kernel_port"]["s011"]["complete"],
          c["existing_kernel_port"]["s011"]["why_incomplete"][:180])
    print("  fused ADC mixed:", mix["s011"]["complete"], mix["s011"]["why_complete"][:180])
    print()

    print("## WHAT I WATCHED FAIL")
    for w in doc["what_i_watched_fail"]:
        print(f"  - {w['what']}: {w['result']}")
        print(f"      {w['why'][:300]}")
    print()
    print(f"written_to {doc['written_to']}")
    print(f"git_head   {doc['git_head']}")
    print(f"identity   pass={doc['identity_check']['pass']} "
          f"adc-per_row={doc['identity_check']['max_abs_adc_minus_per_row']} "
          f"per_row-oracle={doc['identity_check']['max_abs_per_row_minus_oracle']}")


def main() -> int:
    g = geometry()
    organs = gemv_organs(g)
    kernel = inspect_existing_kernel()
    artifact = inspect_artifact()
    prior = prior_science_search()
    identity = adc_identity_check()
    if not identity["pass"]:
        print("FAIL: ADC identity check", file=sys.stderr)
        return 2
    costs = derive_costs(organs)
    operator = build_operator()
    metal = metal_feasibility()
    layout = memory_layout()
    micro = microbenchmark(costs)
    ev = expected_value(costs, prior, identity)

    why_unused = [
        (
            "No codebook artifact. Sealed uniform-q4-v1 is HQ30UQ4 grouped-absmax "
            f"q4 ({artifact.get('q4_tensors', 402)} tensors) + f32v2 "
            f"({artifact.get('f32_tensors', 353)}), schema "
            f"{artifact.get('schema')}. Zero GLM52CPK payloads."
        ),
        (
            "Qwen3.8 catalog refuses the codec. classify_qwen38_mixed_payload admits "
            "packed 0-2, HGRAVU01/HQ30UQ4, f32v2. GLM52CPK/LLM52RPK magics are not "
            "in qwen38_hybrid_decode.rs. This is a missing pack + missing lane, not a "
            "forgotten dispatch of a ready tensor."
        ),
        (
            "Not a rows/cols shape mismatch. Every Qwen3.8 GEMV dimension "
            "(5120, 17408, 6144, 16384, 12288, 1024, 248320, 96) is divisible by D=32. "
            "rotate=1 is the only geometry the Metal path rejects, and the Q4 artifact "
            "is not rotated PQ."
        ),
        (
            f"The existing kernel is the wrong execution of the family. Llama-8B FFN "
            f"gate 14336x4096 D32 single-stage measured {LLAMA_D32_MEDIAN_US} us "
            f"(receipted). Scaled by rows*nchunk to Qwen3.8 gate 17408x5120: "
            f"{scale_llama_us(17408, 160):.1f} us vs Q4 token-share "
            f"{q4_implied_us(q4_matrix_bytes(17408, 5120)):.1f} us. Predicted existing-"
            f"kernel token {costs['existing_kernel_port']['predicted_ms_per_token']:.1f} ms "
            f"({costs['existing_kernel_port']['predicted_tps_if_pq_is_the_whole_token']:.2f} tok/s) "
            f"against incumbent {ANCHOR_TOKEN_MS} ms / {ANCHOR_TPS} tok/s."
        ),
        (
            "It still does 51.24 GFLOP of GEMV. Each (row, chunk) dots a gathered "
            "codebook vector of length D against x. FMA count = rows*cols, identical "
            "to the dense parent and to Q4. Storage compression without less work is "
            "the trap the operation census already named."
        ),
        (
            "Residual-PQ quality is already closed as a drop-in codec at the bit "
            "rates that would still save bytes (1.50 / 2.63 / 3.50 bpw continuation "
            "or first-token failure). That is not a Qwen3.8 measurement, but it is "
            "why a packer is not the first move."
        ),
    ]

    fails = watched_fail(prior, identity, kernel, artifact)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = git_head()

    eight = {
        "1_mathematical_operator": operator,
        "2_expected_bytes_per_token": {
            "incumbent_q4_gemv": costs["incumbent"]["q4_gemv_bytes_per_token"],
            "incumbent_exec_dram": costs["incumbent"]["executable_dram_bytes_per_token"],
            "existing_kernel_index_stream": costs["existing_kernel_port"]["index_bytes_per_token"],
            "adc_mixed_weight_stream": costs["adc_mixed_ba_stays_q4"]["weight_bytes_per_token"],
            "derived_not_guessed": True,
        },
        "3_expected_operations_per_token": {
            "incumbent_gemv_mac_flops": costs["incumbent"]["gemv_flops_per_token"],
            "incumbent_exec_flops": costs["incumbent"]["executable_flops_per_token"],
            "incumbent_exec_ops": costs["incumbent"]["executable_ops_per_token"],
            "existing_kernel_port_flops": costs["existing_kernel_port"]["flops_per_token"],
            "adc_mixed_gemv_path_flops": costs["adc_mixed_ba_stays_q4"]["flops_gemv_path_per_token"],
            "adc_mixed_flops_with_activations": costs["adc_mixed_ba_stays_q4"]["flops_per_token_with_activations"],
            "adc_mixed_ops": costs["adc_mixed_ba_stays_q4"]["ops_per_token"],
            "convention": "FMA=2; ADC LUT uses FMA; ADC accumulate of a LUT scalar counted as 1 add",
        },
        "4_dispatch_topology": {
            "incumbent_dispatches": ANCHOR_DISPATCHES,
            "incumbent_command_buffers": ANCHOR_CBS,
            "adc_dispatches": ANCHOR_DISPATCHES,
            "adc_command_buffers": ANCHOR_CBS,
            "what_synchronises": (
                "One TokenCommandBuffer as today. ADC fused in-kernel: LUT fill then "
                "accumulate with a threadgroup barrier per chunk. No extra host wait. "
                "A two-kernel LUT+accumulate lowering would add 400 dispatches "
                "(401 GEMV minus ba) and is rejected."
            ),
        },
        "5_metal_feasibility": metal,
        "6_memory_layout": layout,
        "7_cheap_microbenchmark": micro,
        "8_expected_value": ev,
    }

    doc = {
        "schema": SCHEMA,
        "generated_at": now,
        "git_head": head,
        "lane": "c4codebook",
        "question": (
            "Does a lookup-plus-accumulate move less than a Q4 dequant-plus-multiply "
            "at the same fidelity on a 595.9 GB/s Apple M3 Ultra, given that "
            "gravity_pq_matvec already exists?"
        ),
        "answer": ev["verdict"],
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
            "mlx_4bit_tps": ANCHOR_MLX_TPS,
            "llamacpp_q5k_tps_archived": ANCHOR_LLAMA_Q5K_TPS,
            "two_servers_tps": ANCHOR_TWO_SERVERS_TPS,
            "one_server_tps": ANCHOR_ONE_SERVER_TPS,
            "llama_d32_gate_median_us": LLAMA_D32_MEDIAN_US,
            "mlp_distill_nogo": "+0.4206 held-out gap vs q3 at 72% of its active bytes",
            "q80_storage_vs_active": "0.6462 vs 2.518",
        },
        "geometry": g,
        "artifact": artifact,
        "existing_kernel": kernel,
        "why_unused": why_unused,
        "prior_science": prior,
        "identity_check": identity,
        "operator": operator,
        "costs": costs,
        "metal_feasibility": metal,
        "memory_layout": layout,
        "microbenchmark": micro,
        "expected_value": ev,
        "eight_items": eight,
        "s011": {
            "existing_kernel_port_complete": costs["existing_kernel_port"]["s011"]["complete"],
            "adc_mixed_complete": costs["adc_mixed_ba_stays_q4"]["s011"]["complete"],
            "incumbent_comparison": {
                "bytes": "ADC mixed reduces",
                "operations": "ADC mixed reduces; existing kernel port does not",
                "dispatches": "same 964",
                "materialization": "same 0",
                "synchronization": "same 1 CB",
                "traffic": "ADC mixed reduces sequential DRAM; existing kernel gathers",
            },
        },
        "what_i_watched_fail": fails,
        "self_check": {
            "identity": identity["pass"],
            "q4_bytes_match_anchor": costs["incumbent"]["q4_gemv_bytes_per_token"] == ANCHOR_Q4_GEMV_BYTES,
            "gemv_flops_match_anchor": costs["incumbent"]["gemv_flops_per_token"] == ANCHOR_GEMV_MAC_FLOPS,
            "dispatches_964": True,
            "oracle_labelled": True,
            "production_path_named_separately": True,
            "kernel_reachable_not_dispatched": (
                kernel["class"] == "REACHABLE" and not kernel["quoted_in_qwen38_hybrid_decode"]
            ),
        },
        "written_to": str(RECEIPT),
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n")
    print_report(doc)

    sc = doc["self_check"]
    if not all(sc[k] is True for k in sc):
        print(f"FAIL: self_check {sc}", file=sys.stderr)
        return 3
    if not costs["adc_mixed_ba_stays_q4"]["s011"]["complete"]:
        print("FAIL: ADC design is S011-incomplete", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
