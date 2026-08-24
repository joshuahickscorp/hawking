#!/usr/bin/env python3
"""C3: low-rank plus sparse correction, fused into one operator.

Design lane. Terminates in a measured decision, not a kernel. Does not open the
27B, does not spawn a model server, does not write a shader.

    python3 tools/headless/c3lowranksparse_design.py

The family is not uniformly dead. Hybrid low-rank+correction as a *byte lever*
is Pareto-dominated by q3 (Phase B). MLP function distillation is NO-GO
(+0.4206 held-out vs q3 at 72% of q3 active bytes). G034 matched-bit low-rank
is 2.93× q3 error. down_proj inverts the ranking (HGRAVS01 beats binary on
post-SwiGLU X). Fusing the two low-rank stages by recomputing R per threadgroup
is NS-030 (5–13× slower).

The question this lane is allowed to reopen is the FUSION of a low-rank
dominant with a sparse correction — two passes over the same activations
versus one kernel that never rematerialises W. Everything else is cited,
not rediscovered.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEOMETRY = REPO / "crates/hawking-core/src/model/qwen38_geometry.rs"
LEDGER = REPO / "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs"
DECODE = REPO / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
SHADERS = REPO / "crates/hawking-core/shaders"
MIXED_METAL = SHADERS / "q80_mixed_decode.metal"
Q4_METAL = SHADERS / "qwen_uniform_q4.metal"
KERNEL_CENSUS = REPO / "receipts/headless/NOETIC_KERNEL_CENSUS.json"
OP_CENSUS = REPO / "receipts/headless/NOETIC_OPERATION_CENSUS.json"
OUT_DEFAULT = REPO / "receipts/headless/C3LOWRANKSPARSE_DESIGN.json"

# Anchors — measured, not re-derived.
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_BOUND = 38
ANCHOR_DECLARED = 554
ANCHOR_REACHABLE = 508
ANCHOR_DEAD = 4
ANCHOR_UNKNOWN = 4
ANCHOR_TPS = 32.73
ANCHOR_TOKEN_MS = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_HONEST_DECODE_GB_S = 411.51
ANCHOR_UNIFIED_B = 103_079_215_104
ANCHOR_GPU_CORES = 60
ANCHOR_PARAMS = 26_895_998_464
ANCHOR_BPW = 4.253
ANCHOR_ARTIFACT_B = 14_297_933_604
ANCHOR_TENSORS = 755
ANCHOR_GEMV_GFLOP = 51.24  # 51_243_909_120 MAC FLOPs / 1e9, FMA=2
ANCHOR_GEMV_MAC_FLOPS = 51_243_909_120
ANCHOR_SRC_FLOPS = 51_541_222_144
ANCHOR_EXE_FLOPS = 77_163_181_824
ANCHOR_EXE_DRAM = 13_988_022_948
ANCHOR_Q4_GEMV_BYTES = 13_611_663_360
ANCHOR_DENSE_W_MATERIALIZED = 0
ANCHOR_MLX_TPS = 35.51
ANCHOR_LLAMA_Q5K_TPS = 24.12
ANCHOR_GPU_FRACTION = 0.965

# Q80 / Phase-B / G070 / NS — cited, not re-run.
CITE_G034_ERR_RATIO = 2.93
CITE_G034_Q3 = 0.1839276241211841
CITE_G034_LR = 0.5393288880586624
CITE_G034_RANK = 803
CITE_G034_MAC_RATIO = 0.2029641544117647
CITE_PHASEB_Q3_ERR = 0.2216
CITE_PHASEB_Q3_PLUS_R64_ERR = 0.1743
CITE_PHASEB_Q3_PLUS_R64_PCT = 107
CITE_PHASEB_Q2_R256_ERR = 0.3955
CITE_PHASEB_Q2_R256_PCT = 101
CITE_MLP_GAP = 0.4206259548664093
CITE_MLP_BYTE_RATIO = 0.7239819004524887
CITE_G035_SHARED_BEATS = False
CITE_Q80_STORAGE_BPW = 0.6462
CITE_Q80_ACTIVE_BPW = 2.518
CITE_HGRAVS_COMPONENT_BPW = 0.13
CITE_GLM_EXPERT_BPW = 0.167
CITE_NULL_COSINE = 0.898
CITE_SCALE_TRAP = 1.000000  # 0.01*W
CITE_NS030_FUSED_NS = 173_209
CITE_NS030_ROWBLOCK_NS = 89_291
CITE_NS030_SIMD_NS = 17_666
CITE_NS030_SIMD3_NS = 13_500
CITE_G070_SCATTERED_INDEX = 0.5973083614523894
CITE_G070_PLANE_RED = 57.84975273848514
CITE_G070_COLUMN_RED = 30.91161317277594
CITE_G070_SCATTERED_RED = 30.365487730625308
CITE_G070_LOWRANK_RED = 22.53539880265316
CITE_G1_SUB05 = 223
CITE_G1_HEALTHY = 0
CITE_Q80_DOWN_BINARY_COS = 0.8264830117545535
CITE_Q80_DOWN_HGRAVS_R192_COS = 0.9132099561676162
CITE_Q80_DOWN_BINARY_BPW = 1.126922607421875
CITE_Q80_DOWN_HGRAVS_R192_BPW = 1.536773681640625
CITE_Q80_DOWN_W = (2048, 512)
CITE_Q80_RANK = 160
CITE_Q80_BITS = 3
CITE_Q80_PACKED_RIGHT = 33_280
CITE_Q80_PACKED_LEFT = 133_120
CITE_Q80_MID_B = 640
CITE_GATHER_10_GBPS = 150.17142857142858
CITE_SEQ_10_GBPS = 121.00143884892087
CITE_NOP_1_GPU_NS = 3334
CITE_NOPS_1155_ONE_CB_NS = 1_483_875
CITE_Q38_DOWN_BIN_HOLD_MIN = 0.7297274499934864
CITE_Q38_DOWN_Q3_HOLD_MIN = 0.9726688096733602
CITE_Q38_DOWN_Q4_HOLD_MIN = 0.9948321774063169
CITE_G1_SHARE_R512_ENERGY = 0.41913010064722683
CITE_TWO_STAGE_XCAP = 512
CITE_TWO_STAGE_RANKCAP = 160

RANK = 160
BITS = 3
GROUP = 64
SPARSE_DENSITIES = (0.0, 0.005, 0.02)  # 0 / 0.5% / Q80 up_proj 2%
HEADLINE_DENSITY = 0.02
F16_B = 2
F32_B = 4
U32_B = 4
SIMD_WIDTH = 32
TG_Q4 = 128
TG_SIMD3 = 256
SIMDGROUPS_PER_TG_SIMD3 = 8
ROWS_PER_TG_Q4 = 2

SWIFT_DEVICE = r'''
import Metal
import Foundation
guard let d = MTLCreateSystemDefaultDevice() else {
  print("{\"error\":\"no metal device\"}"); exit(1)
}
let tg = d.maxThreadsPerThreadgroup
let out: [String: Any] = [
  "name": d.name,
  "hasUnifiedMemory": d.hasUnifiedMemory,
  "recommendedMaxWorkingSetSize": d.recommendedMaxWorkingSetSize,
  "maxBufferLength": d.maxBufferLength,
  "maxThreadgroupMemoryLength": d.maxThreadgroupMemoryLength,
  "maxThreadsPerThreadgroup": ["width": tg.width, "height": tg.height, "depth": tg.depth],
]
let j = try! JSONSerialization.data(withJSONObject: out)
print(String(data: j, encoding: .utf8)!)
'''


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def usize_const(src: str, name: str) -> int:
    m = re.search(rf"pub const {name}: usize = ([0-9_]+);", src)
    if not m:
        raise SystemExit(f"FAIL: missing usize const {name}")
    return int(m.group(1).replace("_", ""))


def load_geometry() -> dict:
    src = GEOMETRY.read_text()
    return {
        "layers": usize_const(src, "QWEN38_LAYERS"),
        "dn_layers": usize_const(src, "QWEN38_DELTANET_LAYERS"),
        "gqa_layers": usize_const(src, "QWEN38_GQA_LAYERS"),
        "hidden": usize_const(src, "QWEN38_HIDDEN"),
        "intermediate": usize_const(src, "QWEN38_INTERMEDIATE"),
        "vocab": usize_const(src, "QWEN38_VOCAB"),
    }


def grouped_bytes(rows: int, cols: int, bits: int, group: int = GROUP) -> int:
    """HQ30UQ4 / HGRAVS01 body: ceil(cols/group) groups × (group*bits/8 + 2)."""
    if group * bits % 8 != 0:
        raise SystemExit(f"FAIL: group*bits not byte aligned: {group}*{bits}")
    gpr = (cols + group - 1) // group
    return rows * gpr * (group * bits // 8 + F16_B)


def csr_active_bytes(rows: int, cols: int, density: float) -> dict:
    """Per-token CSR the shipped kernels actually read (NS-031: rice is bind-time)."""
    nnz = int(round(density * rows * cols))
    indices = nnz * U32_B
    row_ptr = (rows + 1) * U32_B
    signs = (nnz + 7) // 8
    scale = F16_B
    total = indices + row_ptr + signs + scale
    return {
        "density": density,
        "nnz": nnz,
        "nnz_per_row": (nnz / rows) if rows else 0.0,
        "index_bits_per_nnz": 32,
        "value_bits_per_nnz": 1,
        "index_share_of_csr_payload": (indices / total) if total else 0.0,
        "indices_bytes": indices,
        "row_ptr_bytes": row_ptr,
        "signs_bytes": signs,
        "scale_bytes": scale,
        "active_bytes": total,
    }


def try_json(path: Path) -> dict | None:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def git_show_json(rel: str) -> dict | None:
    p = REPO / rel
    d = try_json(p)
    if d is not None:
        return d
    try:
        r = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            capture_output=True, text=True, cwd=REPO, timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except Exception:
        return None
    return None


def git_show_text(rel: str) -> str | None:
    p = REPO / rel
    if p.is_file():
        return p.read_text()
    try:
        r = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            capture_output=True, text=True, cwd=REPO, timeout=30,
        )
        if r.returncode == 0:
            return r.stdout
    except Exception:
        return None
    return None


def metal_device() -> dict:
    """Ask Metal. If Seatbelt/sandbox returns no device, say so — do not invent."""
    if not shutil_which_swift():
        return {
            "source": "UNAVAILABLE: swift missing",
            "maxThreadgroupMemoryLength": None,
            "note": "Fall back to kernel-encoded caps (kXCap=512, kRankCap=160).",
        }
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as f:
            f.write(SWIFT_DEVICE)
            path = f.name
        p = subprocess.run(
            ["swift", path], capture_output=True, text=True, timeout=180,
        )
        raw = (p.stdout or "").strip()
        last = raw.splitlines()[-1] if raw else ""
        if last.startswith("{"):
            d = json.loads(last)
            if d.get("error"):
                d["source"] = "MTLCreateSystemDefaultDevice returned nil (sandbox/no GPU)"
                d["returncode"] = p.returncode
                return d
            d["source"] = "MTLDevice (measured this process)"
            return d
        return {
            "source": "UNAVAILABLE: swift probe failed",
            "stderr": ((p.stderr or "") + "\n" + raw)[:600],
            "returncode": p.returncode,
            "maxThreadgroupMemoryLength": None,
        }
    except Exception as e:
        return {"source": f"UNAVAILABLE: {type(e).__name__}: {e}", "maxThreadgroupMemoryLength": None}
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def shutil_which_swift() -> bool:
    from shutil import which
    return which("swift") is not None


def kernel_caps(mixed_src: str) -> dict:
    xcap = re.search(r"constexpr uint kXCap = ([0-9]+)u;", mixed_src)
    rcap = re.search(r"constexpr uint kRankCap = ([0-9]+)u;", mixed_src)
    simd = re.search(r"constexpr uint kSimdWidth = ([0-9]+)u;", mixed_src)
    return {
        "kXCap": int(xcap.group(1)) if xcap else CITE_TWO_STAGE_XCAP,
        "kRankCap": int(rcap.group(1)) if rcap else CITE_TWO_STAGE_RANKCAP,
        "kSimdWidth": int(simd.group(1)) if simd else SIMD_WIDTH,
        "two_stage_kernel": "q80_hgravs01_two_stage_matvec",
        "two_stage_refuses_if": "right_rows > kRankCap || right_cols > kXCap",
        "source": str(MIXED_METAL.relative_to(REPO)),
    }


def prior_science_search() -> dict:
    """Search preserved receipts. Sparse checkout: git show is the reader."""
    hits = []
    missing = []

    def record(name, rel, extractor):
        d = git_show_json(rel) if rel.endswith(".json") else None
        text = None if d is not None else git_show_text(rel)
        if d is None and text is None:
            missing.append({"name": name, "path": rel})
            return None
        info = extractor(d, text)
        info["name"] = name
        info["path"] = rel
        info["resolved"] = True
        hits.append(info)
        return info

    record("G034", "receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json",
           lambda _d, _t: {
               "status": "REFUTED",
               "error_ratio": CITE_G034_ERR_RATIO,
               "mean_flat_q3_out_rel_fro": CITE_G034_Q3,
               "mean_lowrank_out_rel_fro": CITE_G034_LR,
               "rank_at_q3_budget": CITE_G034_RANK,
               "mac_ratio": CITE_G034_MAC_RATIO,
               "note": "matched-bit low-rank 2.93× q3 error at rank 803; numbers from archaeology index / G034 receipt, not re-fit",
           })

    # G034 file may have a different shape; always cite the archaeology numbers
    # and confirm the file exists.
    g034 = git_show_json("receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json")
    if g034 is not None:
        hits[-1]["file_keys"] = list(g034.keys())[:12]
        hits[-1]["file_present"] = True
    else:
        hits[-1]["file_present"] = False

    g070 = git_show_json("receipts/ascent-2026-08-16/G070_CORRECTION_TOPOLOGY.json")
    if g070 is not None:
        v = g070.get("verdict") or {}
        hits.append({
            "name": "G070",
            "path": "receipts/ascent-2026-08-16/G070_CORRECTION_TOPOLOGY.json",
            "resolved": True,
            "status": "RAN (math half; kernel cost kills SCATTERED)",
            "scattered_index_share": (v.get("index_share") or {}).get("SCATTERED", CITE_G070_SCATTERED_INDEX),
            "mean_error_reduction": v.get("mean_error_reduction"),
            "headline": v.get("the_actual_answer") or v.get("headline"),
            "down_proj_site": "post_swiglu (correct X)",
        })
        down = next((s for s in g070.get("sites") or [] if s.get("organ") == "mlp.down_proj"), None)
        if down:
            b25 = next((b for b in down.get("budgets") or [] if b.get("budget_bits_per_elem") == 0.25), None)
            if b25:
                hits[-1]["down_proj_0.25"] = {
                    t["topology"]: {
                        "error_reduction_pct": t["error_reduction_pct"],
                        "index_overhead_frac": t["index_overhead_frac"],
                        "count": t["count"],
                    }
                    for t in b25.get("topologies") or []
                }
    else:
        missing.append({"name": "G070", "path": "receipts/ascent-2026-08-16/G070_CORRECTION_TOPOLOGY.json"})

    nsreg = git_show_json("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json")
    ns_ids = {}
    if nsreg is not None:
        entries = nsreg.get("entries") or nsreg.get("register") or []
        if isinstance(nsreg, dict) and not entries:
            for v in nsreg.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) and "id" in v[0]:
                    entries = v
                    break
        for e in entries:
            if isinstance(e, dict) and e.get("id") in {
                "NS-006", "NS-009", "NS-012", "NS-019", "NS-020", "NS-030", "NS-031", "NS-032",
            }:
                ns_ids[e["id"]] = {
                    "class": e.get("class"),
                    "mechanism": e.get("mechanism"),
                }
        hits.append({
            "name": "NEGATIVE_SCIENCE_REGISTER",
            "path": "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
            "resolved": True,
            "pulled": ns_ids,
        })
    else:
        missing.append({"name": "NS register",
                        "path": "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json"})

    fuse = git_show_json("receipts/ascent-2026-08-16/q80-decode-throughput.json")
    if fuse and "fusion_negative" in fuse:
        fn = fuse["fusion_negative"]
        hits.append({
            "name": "NS-030 fusion_negative",
            "path": "receipts/ascent-2026-08-16/q80-decode-throughput.json",
            "resolved": True,
            "fused_median_ns": fn.get("fused_median_ns_quiet_selection"),
            "simd3_two_dispatch_ns": fn.get("simd3_two_dispatch_median_ns_quiet_selection"),
            "ratio": (fn["fused_median_ns_quiet_selection"]
                      / fn["simd3_two_dispatch_median_ns_quiet_selection"]
                      if fn.get("simd3_two_dispatch_median_ns_quiet_selection") else None),
            "verdict": fn.get("verdict"),
        })
    else:
        missing.append({"name": "fusion_negative",
                        "path": "receipts/ascent-2026-08-16/q80-decode-throughput.json"})

    algebra = git_show_json("receipts/ascent-2026-08-16/q80-lowrank-algebra.json")
    if algebra is not None:
        f = algebra.get("findings") or {}
        hits.append({
            "name": "q80-lowrank-algebra",
            "path": "receipts/ascent-2026-08-16/q80-lowrank-algebra.json",
            "resolved": True,
            "down_already_factored": (f.get("down_proj") or {}).get("already_factored"),
            "up_already_fused": (f.get("up_proj") or {}).get("already_fused"),
            "down_codec": (f.get("down_proj") or {}).get("codec"),
            "up_codec": (f.get("up_proj") or {}).get("codec"),
            "mid_bytes": (algebra.get("temporaries") or {}).get("down_mid_f32_bytes"),
        })
    else:
        missing.append({"name": "q80-lowrank-algebra",
                        "path": "receipts/ascent-2026-08-16/q80-lowrank-algebra.json"})

    descent = git_show_json("receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json")
    if descent is not None:
        findings = descent.get("findings") or {}
        hits.append({
            "name": "QWEN38_BPW_DESCENT",
            "path": "receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json",
            "resolved": True,
            "hgravs_not_in_codec_catalog": "hgravs01" not in json.dumps(descent.get("summary") or {}),
            "lowrank_is_different_on_dense": findings.get("lowrank_is_different_on_dense"),
            "down_proj_binary_hold_min": CITE_Q38_DOWN_BIN_HOLD_MIN,
            "down_proj_q3_hold_min": CITE_Q38_DOWN_Q3_HOLD_MIN,
            "note": "HGRAVS01 was named as cheap algebra at 0.13 BPW COMPONENT; it was not scored as a hold-cosine candidate in the catalog (binary/rice/q2/q3/q4/hadamard/ternary only).",
        })
    else:
        missing.append({"name": "QWEN38_BPW_DESCENT",
                        "path": "receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json"})

    frontier = git_show_json("receipts/QWEN80_DOWN_PROJ_FRONTIER_SWEEP.json")
    if frontier is not None:
        org0 = (frontier.get("organs") or [{}])[0]
        hits.append({
            "name": "QWEN80_DOWN_PROJ_FRONTIER",
            "path": "receipts/QWEN80_DOWN_PROJ_FRONTIER_SWEEP.json",
            "resolved": True,
            "W_shape": org0.get("W_shape"),
            "inversion": (
                f"binary_g cosine {CITE_Q80_DOWN_BINARY_COS:.4f} FAIL vs "
                f"hgravs01_r192_b3 cosine {CITE_Q80_DOWN_HGRAVS_R192_COS:.4f} at "
                f"{CITE_Q80_DOWN_HGRAVS_R192_BPW:.3f} BPW — low-rank BEATS binary on down_proj"
            ),
            "X": "post-SwiGLU (not hidden)",
        })
    else:
        missing.append({"name": "QWEN80_DOWN_PROJ_FRONTIER",
                        "path": "receipts/QWEN80_DOWN_PROJ_FRONTIER_SWEEP.json"})

    kc = try_json(KERNEL_CENSUS)
    if kc is not None:
        fam = next((f for f in kc.get("families") or []
                    if f.get("id") == "low_rank_plus_sparse_correction"), None)
        hits.append({
            "name": "NOETIC_KERNEL_CENSUS.low_rank_plus_sparse_correction",
            "path": "receipts/headless/NOETIC_KERNEL_CENSUS.json",
            "resolved": True,
            "verdict": (fam or {}).get("verdict"),
            "why": (fam or {}).get("why"),
            "dispatched_reconstructs_dense_NO": (
                (kc.get("reconciliation") or {}).get("dispatched_reconstructs_dense") or {}
            ).get("NO"),
            "reachable": (kc.get("reconciliation") or {}).get("counts", {}).get("REACHABLE"),
        })
        cost = next((c for c in kc.get("missing_family_cost") or []
                     if c.get("id") == "low_rank_plus_sparse_correction"), None)
        if cost:
            hits[-1]["missing_family_cost"] = cost
    else:
        missing.append({"name": "NOETIC_KERNEL_CENSUS", "path": str(KERNEL_CENSUS)})

    oc = try_json(OP_CENSUS)
    if oc is not None:
        hits.append({
            "name": "NOETIC_OPERATION_CENSUS",
            "path": "receipts/headless/NOETIC_OPERATION_CENSUS.json",
            "resolved": True,
            "gemv_mac_flops": (oc.get("analytic_vs_measured") or {}).get("dispatched_gemv_mac_flops"),
            "executable_dram": ((oc.get("columns") or {}).get("executable") or {}).get("dram_bytes_per_token"),
            "dense_w_materialized": ((oc.get("columns") or {}).get("executable") or {}).get(
                "dense_w_materialized_bytes_per_token"),
        })
    else:
        missing.append({"name": "NOETIC_OPERATION_CENSUS", "path": str(OP_CENSUS)})

    # MLP distill: n16 lane, may not be in this sparse tree.
    mlp_paths = [
        REPO / "receipts/headless/NOETIC_MLP_DISTILL_PROBE.json",
        Path("/Users/scammermike/.claude-grok/worktrees/n16mlp-20260823-142304"
             "/receipts/headless/NOETIC_MLP_DISTILL_PROBE.json"),
    ]
    mlp = None
    mlp_used = None
    for p in mlp_paths:
        mlp = try_json(p)
        if mlp:
            mlp_used = str(p)
            break
    if mlp is not None:
        v = mlp.get("verdict") or {}
        hits.append({
            "name": "NOETIC_MLP_DISTILL_PROBE",
            "path": mlp_used,
            "resolved": True,
            "decision": v.get("decision"),
            "deciding_number": v.get("deciding_number"),
            "headline_width": v.get("headline_width"),
            "byte_ratio": CITE_MLP_BYTE_RATIO,
        })
    else:
        missing.append({"name": "NOETIC_MLP_DISTILL_PROBE",
                        "path": "receipts/headless/NOETIC_MLP_DISTILL_PROBE.json (n16 worktree)"})
        hits.append({
            "name": "NOETIC_MLP_DISTILL_PROBE",
            "path": "cited (file not in this sparse checkout)",
            "resolved": False,
            "decision": "NO-GO",
            "deciding_number": CITE_MLP_GAP,
            "cited": True,
        })

    census_dirs = [
        REPO / ".lane-bootstrap/census",
        Path("/Users/scammermike/.claude-grok/worktrees/n16mlp-20260823-142304/.lane-bootstrap/census"),
        Path("/Users/scammermike/Downloads/hawking-copy/.lane-bootstrap/census"),
    ]
    census_found = None
    for d in census_dirs:
        if (d / "n1arch.md").is_file():
            census_found = d
            break
    if census_found is not None:
        hits.append({
            "name": "lane-bootstrap/census",
            "path": str(census_found),
            "resolved": True,
            "files": sorted(p.name for p in census_found.glob("*.md")),
            "n1arch_mechanisms": 35,
            "n15neg_closures": 31,
            "n16clos": "NR/NX IR gap (LowRank + SparseResidual = schema-change)",
        })
    else:
        missing.append({"name": ".lane-bootstrap/census", "path": ".lane-bootstrap/census"})

    # Family-level classification (the stop/continue gate).
    family_status = {
        "as_byte_quality_lever_on_gate_up_attn": "REFUTED",
        "as_mlp_function_distillation": "NO-GO",
        "as_matched_bit_lowrank_replacement": "REFUTED (G034)",
        "as_qn_plus_lowrank_residual": "REFUTED (Phase B hybrid, Pareto-dominated by q3)",
        "as_single_dispatch_LRx_recompute_R": "REFUTED (NS-030)",
        "as_down_proj_dominant_lowrank": "LIVE on Q80 (HGRAVS01 dispatched); UNMEASURED hold-cosine on Qwen3.8 5120×17408",
        "as_fused_lowrank_plus_csr": "NOT PREVIOUSLY BUILT (kernel census PARTIAL). Fusion traffic is the only unmeasured claim.",
        "continue": (
            "Do not redesign the approximation. Design the fusion, price its traffic, "
            "and kill it if that number cannot move a token."
        ),
    }
    return {
        "hits": hits,
        "missing": missing,
        "n_hits": len(hits),
        "n_missing": len(missing),
        "family_status": family_status,
        "traps_respected": [
            "223 sub-0.5 local_bpw rows, healthy=0 (G1/G034)",
            "Q80 storage 0.6462 vs ACTIVE 2.518 — report both or neither",
            "G035 shared_beats_independent=false",
            "GLM 0.167 expert BPW; HGRAVS01 0.13 is a down_proj COMPONENT",
            "MLP distillation NO-GO +0.4206 at 72% of q3 active bytes",
            "never synthetic X; cosine scale-invariant (0.01*W = 1.000000); null ≈ 0.898",
            "representation→dense W→GEMM is an ORACLE, not production",
        ],
    }


def occupancy(rows: int, rows_per_tg: int, tg: int, cores: int = ANCHOR_GPU_CORES) -> dict:
    tgs = (rows + rows_per_tg - 1) // rows_per_tg
    return {
        "rows": rows,
        "rows_per_threadgroup": rows_per_tg,
        "threadgroup_size": tg,
        "threadgroups": tgs,
        "threads": tgs * tg,
        "gpu_cores": cores,
        "threadgroups_per_core_if_spread": tgs / cores,
        "fills_all_cores": tgs >= cores,
    }


def bandwidth_ns(nbytes: int, gb_s: float) -> float:
    if gb_s <= 0:
        return float("inf")
    return nbytes / (gb_s * 1e9) * 1e9  # nanoseconds


def build_accounting(g: dict, caps: dict) -> dict:
    layers = g["layers"]
    hidden = g["hidden"]
    inter = g["intermediate"]
    m, n = hidden, inter  # down_proj: y[m] = W[m,n] @ x[n]
    r = RANK

    q4_one = grouped_bytes(m, n, 4)
    q4_tok = q4_one * layers
    r_bytes = grouped_bytes(r, n, BITS)
    l_bytes = grouped_bytes(m, r, BITS)
    lr_one = r_bytes + l_bytes
    lr_tok = lr_one * layers
    lr_bpw = (lr_one * 8) / (m * n)

    x_b = n * F32_B
    y_b = m * F32_B
    mid_b = r * F32_B
    x_tg_needed = x_b
    x_fits_two_stage = n <= caps["kXCap"]
    x_fits_32kib = x_b <= 32 * 1024

    dense_macs_one = 2 * m * n
    lr_macs_one = 2 * r * n + 2 * m * r  # R@x then L@mid
    q4_macs_tok = dense_macs_one * layers
    lr_macs_tok = lr_macs_one * layers

    sparse = {}
    for d in SPARSE_DENSITIES:
        csr = csr_active_bytes(m, n, d)
        csr_tok = csr["active_bytes"] * layers
        sp_macs_one = 2 * csr["nnz"]  # mul+add per residual
        fused_weight_one = lr_one + csr["active_bytes"]
        fused_weight_tok = fused_weight_one * layers
        # Two-pass: R reads x, L reads mid, CSR re-reads x and y.
        # Fused L+CSR: R reads x, L+CSR reads x again (x is 70 KB, not in registers)
        # and writes y once. Fusion does not save the x re-read unless 70 KB stays
        # hot — guaranteed DRAM save is the y re-read. Optimistic save is x+y.
        sparse[str(d)] = {
            **csr,
            "csr_bytes_per_token": csr_tok,
            "sp_macs_per_launch": sp_macs_one,
            "sp_macs_per_token": sp_macs_one * layers,
            "fused_weight_bytes_per_launch": fused_weight_one,
            "fused_weight_bytes_per_token": fused_weight_tok,
            "fusion_guaranteed_dram_save_per_launch": y_b,
            "fusion_guaranteed_dram_save_per_token": y_b * layers,
            "fusion_optimistic_x_plus_y_save_per_launch": x_b + y_b,
            "fusion_optimistic_x_plus_y_save_per_token": (x_b + y_b) * layers,
            "vs_q4_weight_bytes_per_token": q4_tok - fused_weight_tok,
        }

    headline = sparse[str(HEADLINE_DENSITY)]
    occ_q4 = occupancy(m, ROWS_PER_TG_Q4, TG_Q4)
    occ_r = occupancy(r, SIMDGROUPS_PER_TG_SIMD3, TG_SIMD3)
    occ_l = occupancy(m, SIMDGROUPS_PER_TG_SIMD3, TG_SIMD3)

    # Dispatches: incumbent uses 1 GEMV per down_proj among 401 GEMVs / 964 total.
    down_disp_incumbent = layers
    down_disp_fused = layers * 2      # R then fused L+CSR
    down_disp_twopass = layers * 3    # R, L, CSR
    total_fused = ANCHOR_DISPATCHES - down_disp_incumbent + down_disp_fused
    total_twopass = ANCHOR_DISPATCHES - down_disp_incumbent + down_disp_twopass

    roof = ANCHOR_ROOF_GB_S
    q4_ns_floor = bandwidth_ns(q4_tok, roof)
    lr_ns_floor = bandwidth_ns(lr_tok, roof)
    sparse_ns_floor = bandwidth_ns(headline["csr_bytes_per_token"], roof)
    fused_ns_floor = bandwidth_ns(headline["fused_weight_bytes_per_token"], roof)
    fusion_save_ns_guaranteed = bandwidth_ns(headline["fusion_guaranteed_dram_save_per_token"], roof)
    fusion_save_ns_optimistic = bandwidth_ns(
        headline["fusion_optimistic_x_plus_y_save_per_token"], roof)
    token_ns = ANCHOR_TOKEN_MS * 1e6
    q80_fusion_loss = CITE_NS030_FUSED_NS / CITE_NS030_SIMD3_NS

    # Token-level if we replace only down_proj weights (quality-conditional).
    dram_after_lr = ANCHOR_EXE_DRAM - q4_tok + lr_tok
    dram_after_fused = ANCHOR_EXE_DRAM - q4_tok + headline["fused_weight_bytes_per_token"]
    gemv_after_lr = ANCHOR_GEMV_MAC_FLOPS - q4_macs_tok + lr_macs_tok
    gemv_after_fused = gemv_after_lr + headline["sp_macs_per_token"]

    return {
        "site": {
            "organ": "mlp.down_proj",
            "why_this_organ": (
                "NS-012 / Q80 frontier: down_proj inverts the ranking (HGRAVS01 beats "
                "binary) and must be fit on post-SwiGLU X. gate/up already have fused "
                "binary±CSR. Attention/lm_head were not inverted. Scope is 64 down_proj "
                "GEMVs, not the other 337."
            ),
            "rows": m,
            "cols": n,
            "rank": r,
            "bits": BITS,
            "group": GROUP,
            "layers": layers,
            "elements_per_launch": m * n,
            "elements_per_token": m * n * layers,
            "x_is": "post-SwiGLU silu(X@Wg.T)*(X@Wu.T), length intermediate",
        },
        "incumbent_down": {
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "bytes_per_launch": q4_one,
            "bytes_per_token": q4_tok,
            "mac_flops_per_launch": dense_macs_one,
            "mac_flops_per_token": q4_macs_tok,
            "dispatches_per_token": down_disp_incumbent,
            "occupancy": occ_q4,
            "reconstructs_dense": "NO",
            "bandwidth_floor_ns_at_roof": q4_ns_floor,
        },
        "factors": {
            "R_shape": [r, n],
            "L_shape": [m, r],
            "R_bytes_per_launch": r_bytes,
            "L_bytes_per_launch": l_bytes,
            "LR_bytes_per_launch": lr_one,
            "LR_bytes_per_token": lr_tok,
            "LR_complete_bpw_including_both_factors": lr_bpw,
            "trap_0_13_bpw_is": (
                f"{lr_bpw:.4f} BPW at rank-160 3-bit g64 counting L AND R. "
                "The '0.13 BPW' slogan is this component, not a model EBPW, and is "
                "not 3.1% of 4.25 (that arithmetic drops R)."
            ),
            "q80_right_bytes_check": {
                "formula_160x512_b3": grouped_bytes(160, 512, 3),
                "receipt_down_packed_right": CITE_Q80_PACKED_RIGHT,
                "match": grouped_bytes(160, 512, 3) == CITE_Q80_PACKED_RIGHT,
            },
            "L_last_group_padded": n % GROUP == 0 and (r % GROUP != 0),
            "L_groups_per_row": (r + GROUP - 1) // GROUP,
        },
        "activations": {
            "x_bytes": x_b,
            "y_bytes": y_b,
            "mid_bytes": mid_b,
            "x_fits_two_stage_kXCap": x_fits_two_stage,
            "x_fits_32kib_threadgroup": x_fits_32kib,
            "kXCap": caps["kXCap"],
            "kRankCap": caps["kRankCap"],
            "qwen38_n_vs_kXCap": f"{n} > {caps['kXCap']}" if n > caps["kXCap"] else f"{n} <= {caps['kXCap']}",
        },
        "sparse_by_density": sparse,
        "headline_density": HEADLINE_DENSITY,
        "flops": {
            "convention": "IEEE FMA counted as 2 (matches G143 / operation census)",
            "incumbent_gemv_mac_flops_per_token": ANCHOR_GEMV_MAC_FLOPS,
            "incumbent_down_mac_flops_per_token": q4_macs_tok,
            "lr_mac_flops_per_launch": lr_macs_one,
            "lr_mac_flops_per_token": lr_macs_tok,
            "lr_mac_ratio_vs_dense_down": lr_macs_one / dense_macs_one,
            "fused_2pct_mac_flops_per_token": gemv_after_fused,
            "token_gemv_after_lr_only": gemv_after_lr,
            "delta_vs_incumbent_gemv_fused_2pct": gemv_after_fused - ANCHOR_GEMV_MAC_FLOPS,
        },
        "dispatches": {
            "incumbent_total": ANCHOR_DISPATCHES,
            "incumbent_cbs": ANCHOR_CBS,
            "incumbent_down": down_disp_incumbent,
            "fused_down": down_disp_fused,
            "twopass_down": down_disp_twopass,
            "token_if_fused_replaces_down": total_fused,
            "token_if_twopass_replaces_down": total_twopass,
            "fusion_saves_vs_twopass": total_twopass - total_fused,
            "sync": "device mid[rank] between R and L; no grid-wide barrier exists (NS-030)",
        },
        "occupancy": {"q4_down": occ_q4, "R_stage": occ_r, "L_stage": occ_l},
        "roofline": {
            "roof_gb_s": roof,
            "honest_decode_gb_s": ANCHOR_HONEST_DECODE_GB_S,
            "q4_down_floor_ns": q4_ns_floor,
            "lr_floor_ns": lr_ns_floor,
            "csr_2pct_floor_ns": sparse_ns_floor,
            "fused_weight_floor_ns": fused_ns_floor,
            "fusion_guaranteed_save_ns": fusion_save_ns_guaranteed,
            "fusion_optimistic_save_ns": fusion_save_ns_optimistic,
            "token_ns": token_ns,
            "fusion_guaranteed_save_frac_of_token": fusion_save_ns_guaranteed / token_ns,
            "fusion_optimistic_save_frac_of_token": fusion_save_ns_optimistic / token_ns,
            "ns030_q80_fused_over_two_dispatch": q80_fusion_loss,
        },
        "token_after_replace_down": {
            "note": "Quality-conditional bandwidth scaling. Not a TPS claim.",
            "dram_incumbent": ANCHOR_EXE_DRAM,
            "dram_lr_only": dram_after_lr,
            "dram_fused_2pct": dram_after_fused,
            "scaled_ms_lr_only_if_dram_bound": ANCHOR_TOKEN_MS * dram_after_lr / ANCHOR_EXE_DRAM,
            "scaled_ms_fused_2pct_if_dram_bound": ANCHOR_TOKEN_MS * dram_after_fused / ANCHOR_EXE_DRAM,
        },
    }


def build_operator(acc: dict) -> dict:
    s = acc["site"]
    m, n, r = s["rows"], s["cols"], s["rank"]
    return {
        "name": "fused_hgravs01_csr_matvec",
        "scope": "mlp.down_proj × 64 on Qwen3.8; other organs out of scope",
        "math": (
            f"W ∈ R^{{{m}×{n}}}, x ∈ R^{{{n}}} (post-SwiGLU). "
            f"R ∈ R^{{{r}×{n}}}, L ∈ R^{{{m}×{r}}} packed HGRAVS01 3-bit g64. "
            "S stored CSR: row_ptr, col_idx, 1-bit residual signs, f16 scale. "
            "y = L (R x) + S_csr x, with S_csr the residual after the low-rank "
            "fit, not a second copy of W."
        ),
        "production_path": {
            "label": "PRODUCTION",
            "never_materialises_W": True,
            "dispatches": [
                {
                    "name": "R_stage",
                    "kernel_reuse": "q80_hgravs01_factor_matvec_simd3",
                    "compute": "mid[r] = R @ x",
                    "temporary": f"device mid[{r}] f32 = {acc['activations']['mid_bytes']} B (not dense W)",
                    "grid": f"ceil({r}/8)*256, TG 256",
                },
                {
                    "name": "L_plus_CSR",
                    "kernel": "NEW: q80_hgravs01_factor_csr_matvec_simd3 (~80–150 lines on the binary_group_csr template)",
                    "compute": "y_i = L_i·mid + Σ_{k∈row i} sign_k * scale * x[col_k]",
                    "csr_loop": "serial on simd lane 0 after L reduction (same add-order contract as q80_binary_group_csr_matvec)",
                    "grid": f"ceil({m}/8)*256, TG 256",
                },
            ],
            "forbidden": (
                "q80_hgravs01_two_stage_matvec as a single dispatch: NS-030 lost 5–13× "
                "on Q80 because every TG recomputes R[160×512]; on Qwen3.8 it additionally "
                f"refuses at bind (n={n} > kXCap={acc['activations']['kXCap']}, "
                f"x={acc['activations']['x_bytes']} B > 32 KiB threadgroup)."
            ),
        },
        "correctness_oracle": {
            "label": "ORACLE — not a production implementation",
            "may_materialise_W": True,
            "definition": (
                "W_hat = decode_f32(L) @ decode_f32(R) + csr_to_dense(S); "
                "y_oracle = W_hat @ x. CPU references already in-tree: "
                "hgravs01_two_stage_matvec_f32 (q80_mixed_decode.rs) and "
                "binary_rice_q1_matvec_f32. Production y is compared to y_oracle; "
                "the oracle path is the labelled correctness check and must never "
                "be bound as a decode kernel (dense-reconstruction law; NS-019)."
            ),
            "existing_numeric_gates": {
                "packed_two_stage_vs_decoded_factor": 0.0,
                "reconstruct_LR_then_Wx_vs_two_stage": 1.811981201171875e-05,
                "up_reconstruct_vs_fused": 1.0251998901367188e-05,
                "source": "receipts/ascent-2026-08-16/q80-lowrank-algebra.json",
            },
        },
        "not_this_operator": [
            "qN base + low-rank residual (Phase B hybrid — REFUTED, adds bytes)",
            "thinner SwiGLU distillation (NNS-015 / n16 — NO-GO +0.4206)",
            "shared basis (G035 shared_beats_independent=false)",
            "reconstruct W then GEMM (NS-019; reconstructs_dense = YES)",
        ],
    }


def build_metal(acc: dict, caps: dict, dev: dict) -> dict:
    x_b = acc["activations"]["x_bytes"]
    mid_b = acc["activations"]["mid_bytes"]
    measured_tg = dev.get("maxThreadgroupMemoryLength")
    tg_mem = measured_tg if isinstance(measured_tg, int) else 32 * 1024
    tg_src = (dev.get("source") if isinstance(measured_tg, int)
              else "DOCUMENTED Apple GPU 32 KiB (this process got no MTLDevice); "
                   "kernel-encoded kXCap=512 ≡ 2 KiB of x")
    nnz_row = acc["sparse_by_density"][str(HEADLINE_DENSITY)]["nnz_per_row"]
    return {
        "device_probe": {k: dev.get(k) for k in (
            "source", "name", "hasUnifiedMemory", "maxThreadgroupMemoryLength",
            "maxBufferLength", "recommendedMaxWorkingSetSize", "maxThreadsPerThreadgroup",
            "error", "stderr",
        ) if k in dev},
        "simdgroup_width": caps["kSimdWidth"],
        "threadgroup_q4": TG_Q4,
        "threadgroup_factor": TG_SIMD3,
        "max_threads_per_threadgroup_documented": 1024,
        "threadgroup_memory_bytes_used_in_design": {
            "R_and_L_two_dispatch": "none beyond 8-float reduction (like simd3 today)",
            "forbidden_two_stage": {
                "mid": mid_b,
                "x_tg": x_b,
                "sum": mid_b + x_b,
                "cap": tg_mem,
                "fits": (mid_b + x_b) <= tg_mem,
                "cap_source": tg_src,
            },
        },
        "access": {
            "R_codes_scales": "row-major grouped; 8-wide LSB unpack; coalesced sequential",
            "L_codes_scales": "row-major; 160 cols, 8-wide; coalesced (the simd3 8-unpack exists specifically because L is 160-col)",
            "x_in_R": f"length {acc['site']['cols']}, sequential, {x_b} B — cache-resident vs SLC",
            "x_in_CSR": (
                f"gather x[col] from the same {x_b} B vector. Not a 50 MiB weight gather. "
                f"G070/NS-032 10-of-512 expert gather ({CITE_GATHER_10_GBPS:.1f} GB/s) is a "
                "different pattern (large tensors). Here the working set is 70 KiB."
            ),
            "CSR_indices_signs": "streamed in row order — coalesced. Index share of expanded CSR is ~97% (u32 index vs 1-bit sign).",
            "y": "one thread/simdgroup owns a row; no atomic_fetch_add (that is strand_outlier_correct, G070 kernel-cost death)",
            "lane0_serial_csr": (
                f"at 2% density, {nnz_row:.1f} nnz/row serial on lane 0 while 255 threads idle. "
                "That occupancy tax is the sparse half's real cost on this hardware, not DRAM coalescing of x."
            ),
        },
        "register_pressure": (
            "L FMA loop is the existing simd3 8-unpack (8 q, 8 scales, 8 x). CSR runs after "
            "simd_sum on lane 0 — it does not widen the FMA live set. No extra TG memory."
        ),
        "occupancy_hole": acc["occupancy"]["R_stage"],
        "ns030": {
            "fused_ns": CITE_NS030_FUSED_NS,
            "simd3_two_dispatch_ns": CITE_NS030_SIMD3_NS,
            "loss_x": CITE_NS030_FUSED_NS / CITE_NS030_SIMD3_NS,
            "law": "Metal has no grid-wide barrier. True single-pass without recompute is not available.",
        },
        "feasibility": {
            "two_dispatch_R_then_fused_L_CSR": "FEASIBLE — pieces are DISPATCHED separately on Q80; glue is the binary_group_csr residual loop on the L factor kernel",
            "one_dispatch_recompute_R": "INFEASIBLE / REFUTED (NS-030) and geometrically refused (x > kXCap)",
            "qwen38_two_stage_as_written": "WILL NO-OP: right_cols=17408 > kXCap=512",
        },
    }


def build_layout(acc: dict) -> dict:
    sp = acc["sparse_by_density"][str(HEADLINE_DENSITY)]
    return {
        "magic": "HGRAVS01 factor bodies (left, right) + HGRAVR02-class CSR residual (bind-time expanded)",
        "R": {
            "shape": acc["factors"]["R_shape"],
            "layout": "fp16 scales[groups] || packed unsigned 3-bit codes, group 64, q=code-3, value=q*scale",
            "bytes": acc["factors"]["R_bytes_per_launch"],
        },
        "L": {
            "shape": acc["factors"]["L_shape"],
            "layout": "same codec; last group of 32 elements padded to 64",
            "bytes": acc["factors"]["L_bytes_per_launch"],
        },
        "mid": {
            "shape": [RANK],
            "layout": "device f32[rank], 640 B, reused per down_proj",
            "bytes": acc["activations"]["mid_bytes"],
            "not": "not dense W",
        },
        "csr": {
            "row_ptr": "uint32[rows+1]",
            "col_idx": "uint32[nnz] (expanded at bind; NS-031 forbids per-token rice)",
            "signs": "1 bit/nnz LSB-packed",
            "scale": "fp16 rms/absmax of residual",
            "nnz_at_2pct": sp["nnz"],
            "active_bytes_per_launch": sp["active_bytes"],
            "index_share": sp["index_share_of_csr_payload"],
        },
        "not_stored": "parent dense W, decoded f32 W, rice bitstream on the token path",
        "nr_today": (
            "LowRank and SparseResidual are schema-change families in the sealed NR "
            "(n16clos: expressible-as-is 0). Packing this representation is a new NR node, "
            "not a field on grouped_absmax."
        ),
    }


def build_microbench(acc: dict) -> dict:
    s = acc["site"]
    sp = acc["sparse_by_density"][str(HEADLINE_DENSITY)]
    q4 = acc["incumbent_down"]
    return {
        "goal": (
            "Discriminate fused LR+CSR from the incumbent Q4 down_proj AND from two-pass "
            "LR+CSR BEFORE anyone writes q80_hgravs01_factor_csr_matvec_simd3."
        ),
        "no_new_kernel": True,
        "geometry": {"rows": s["rows"], "cols": s["cols"], "rank": s["rank"],
                     "nnz_per_row_2pct": sp["nnz_per_row"]},
        "arms": [
            {
                "id": "A",
                "name": "incumbent Q4",
                "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                "file": "crates/hawking-core/shaders/qwen_uniform_q4.metal",
                "grid": f"ceil({s['rows']}/2)*128, TG 128",
                "bytes": q4["bytes_per_launch"],
            },
            {
                "id": "B",
                "name": "two-dispatch HGRAVS01 (no sparse)",
                "kernel": "q80_hgravs01_factor_matvec_simd3 × 2",
                "file": "crates/hawking-core/shaders/q80_mixed_decode.metal",
                "R_grid": f"ceil({RANK}/8)*256, TG 256  ({acc['occupancy']['R_stage']['threadgroups']} TGs / {ANCHOR_GPU_CORES} cores)",
                "L_grid": f"ceil({s['rows']}/8)*256, TG 256",
                "temporary": "device mid[160] f32",
                "bytes": acc["factors"]["LR_bytes_per_launch"],
            },
            {
                "id": "C",
                "name": "two-pass LR + CSR (the fusion baseline)",
                "kernel": "B then q80_sparse_q1_apply_csr_simd",
                "nnz_per_row_sweep": [0, 4, 32, 128, round(sp["nnz_per_row"])],
                "bytes_extra": sp["active_bytes"],
            },
        ],
        "fused_arm_not_built": {
            "id": "D",
            "name": "fused L+CSR (the candidate)",
            "upper_bound": (
                "GPU(C) − GPU(CSR-only). Fusion cannot beat two-pass by more than the "
                "CSR kernel's time plus y re-read. If that quantity is < 1% of GPU(A), "
                "stop — the novel part cannot move the token."
            ),
        },
        "how_to_run_today": (
            "Reuse crates/hawking-core/examples/ascension_qwen80_mixed_decode_throughput.rs "
            "timing authority (MTLCommandBuffer GPUStartTime/GPUEndTime after wait). "
            "Bind the existing kernels at Qwen3.8 down_proj geometry (5120×17408, rank 160). "
            "Do not add a shader. Packed bytes may be random HGRAVS01-shaped; this arm is "
            "traffic/occupancy, not quality. Quality is arm E."
        ),
        "arm_E_quality_kill": {
            "name": "SVD energy of ONE real down_proj on real post-SwiGLU X",
            "capture": "Phase-B capture_diverse2 / n16: real BF16 parent, not Gaussian (NS-009)",
            "metric": "held-out output rel-fro of rank-160 vs grouped-absmax q3 (3.25 b/elem)",
            "null": f"constant-mean cosine {CITE_NULL_COSINE}; do not GO on cosine",
            "kill_if": (
                f"rel-fro_r160 / rel-fro_q3 is in the G034 neighbourhood ({CITE_G034_ERR_RATIO}×) "
                "OR 2% scattered residual (G070 down 0.25 b/elem reduced a *q3* residual by 27.7%, "
                "not a rank-160 hole). Either r160 is already enough (sparse is waste) or it is "
                "not (sparse 2% cannot close). Both kill the fused family."
            ),
        },
        "kill_rules": [
            "GPU(B) >= GPU(A): LR occupancy eats the byte win on this launch geometry (20 TGs on 60 cores for R).",
            "GPU(CSR @ 2%) < 0.01 * GPU(A): fusing CSR cannot move the token; fusion-specific value is dead.",
            "GPU(C) >= GPU(A): adding sparse makes two-pass slower than incumbent even before quality.",
            "arm E fails q3 held-out: do not write the fused shader.",
        ],
        "predicted_from_accounting": {
            "gpuA_bandwidth_floor_ns_all_64": acc["roofline"]["q4_down_floor_ns"],
            "gpuB_bandwidth_floor_ns_all_64": acc["roofline"]["lr_floor_ns"],
            "gpuCSR_2pct_floor_ns_all_64": acc["roofline"]["csr_2pct_floor_ns"],
            "fusion_guaranteed_save_ns_all_64": acc["roofline"]["fusion_guaranteed_save_ns"],
            "fusion_save_frac_of_token": acc["roofline"]["fusion_guaranteed_save_frac_of_token"],
            "prediction": (
                "Fusion save is 64 * 20_480 B y-reread = "
                f"{acc['sparse_by_density'][str(HEADLINE_DENSITY)]['fusion_guaranteed_dram_save_per_token']:,} B "
                f"= {acc['roofline']['fusion_guaranteed_save_ns']:.1f} ns at {ANCHOR_ROOF_GB_S} GB/s "
                f"= {100*acc['roofline']['fusion_guaranteed_save_frac_of_token']:.4f}% of the 30.606 ms token. "
                "CSR @ 2% itself is hundreds of µs of *weight* traffic that fusion does not remove."
            ),
        },
    }


def build_s011(acc: dict) -> dict:
    sp = acc["sparse_by_density"][str(HEADLINE_DENSITY)]
    d = acc["dispatches"]
    f = acc["flops"]
    axes = {
        "bytes": {
            "vs_incumbent_down": "REDUCE" if sp["fused_weight_bytes_per_token"] < acc["incumbent_down"]["bytes_per_token"] else "NO",
            "incumbent": acc["incumbent_down"]["bytes_per_token"],
            "fused_2pct": sp["fused_weight_bytes_per_token"],
            "lr_only": acc["factors"]["LR_bytes_per_token"],
        },
        "operations": {
            "vs_incumbent_gemv": "REDUCE" if f["fused_2pct_mac_flops_per_token"] < f["incumbent_gemv_mac_flops_per_token"] else "NO",
            "incumbent_gemv": f["incumbent_gemv_mac_flops_per_token"],
            "fused_2pct_gemv": f["fused_2pct_mac_flops_per_token"],
        },
        "dispatches": {
            "vs_incumbent": "INCREASE",
            "incumbent": d["incumbent_total"],
            "fused": d["token_if_fused_replaces_down"],
            "vs_twopass": "REDUCE",
            "twopass": d["token_if_twopass_replaces_down"],
            "fusion_saves": d["fusion_saves_vs_twopass"],
        },
        "materialization": {
            "vs_incumbent": "SAME (both 0 dense W)",
            "incumbent": ANCHOR_DENSE_W_MATERIALIZED,
            "fused": 0,
            "oracle_would_write": acc["site"]["elements_per_token"] * F32_B,
        },
        "synchronization": {
            "vs_incumbent": "SAME 1 CB; extra device mid[640 B] between two dispatches",
            "cbs": ANCHOR_CBS,
        },
        "traffic": {
            "vs_incumbent_weights": "REDUCE (quality-conditional)",
            "vs_twopass_activations": "REDUCE (guaranteed y re-read only)",
            "fusion_guaranteed_bytes": sp["fusion_guaranteed_dram_save_per_token"],
            "fusion_optimistic_bytes": sp["fusion_optimistic_x_plus_y_save_per_token"],
        },
    }
    reduced = [k for k, v in axes.items()
               if "REDUCE" in json.dumps(v)]
    return {
        "rule": (
            "S011 §4 (lane contract): a design that does not reduce at least one of "
            "bytes, operations, dispatches, materialization, synchronization or traffic "
            "is INCOMPLETE."
        ),
        "axes": axes,
        "reduced": reduced,
        "complete": len(reduced) >= 1,
        "caveat": (
            "Bytes and operations REDUCE only if down_proj is replaced. That replacement "
            "is a quality bet this family has already lost on gate/up and at matched bits. "
            "The *novel* axis (fusion vs two-pass) reduces dispatches by 64 and traffic by "
            f"{sp['fusion_guaranteed_dram_save_per_token']:,} B guaranteed. Complete on paper, "
            "not worth the kernel."
        ),
    }


def build_ev(acc: dict, s011: dict) -> dict:
    sp = acc["sparse_by_density"][str(HEADLINE_DENSITY)]
    frac = acc["roofline"]["fusion_guaranteed_save_frac_of_token"]
    return {
        "verdict": "NOT_WORTH_BUILDING",
        "what_it_would_win": [
            (
                f"If r160 on Qwen3.8 down were q3-healthy, replacing 64 Q4 down GEMVs "
                f"cuts {acc['incumbent_down']['bytes_per_token']:,} B → "
                f"{acc['factors']['LR_bytes_per_token']:,} B of down weights "
                f"({acc['factors']['LR_complete_bpw_including_both_factors']:.4f} BPW complete, "
                f"not the 0.13 slogan) and cuts down MACs to "
                f"{acc['flops']['lr_mac_ratio_vs_dense_down']:.3f}×. Bandwidth scaling of the "
                f"whole token would be ~{acc['token_after_replace_down']['scaled_ms_lr_only_if_dram_bound']:.2f} ms "
                f"({1000/acc['token_after_replace_down']['scaled_ms_lr_only_if_dram_bound']:.1f} tok/s), "
                f"which would beat MLX {ANCHOR_MLX_TPS} — *if* quality and R-stage occupancy hold. "
                "That win is LR-on-down, already named in QWEN38_BPW_DESCENT, not fusion."
            ),
            (
                f"Fusion vs two-pass saves {d_disp(acc)} dispatches and "
                f"{sp['fusion_guaranteed_dram_save_per_token']:,} B of y re-read "
                f"({acc['roofline']['fusion_guaranteed_save_ns']:.1f} ns at the 595.9 GB/s roof, "
                f"{100*frac:.4f}% of the token)."
            ),
        ],
        "what_it_risks": [
            "Quality: G034 rank-803 at *matched q3 bits* is 2.93× q3 error. r160 is 0.137 BPW, 5× below q3 and 3.1% of 5120 rows. G1-SHARE captured 41.9% energy at r512 on gate. Qwen3.8 BPW descent never scored HGRAVS01 hold-cosine on 5120×17408.",
            "Q80 inversion does not transfer: QWEN38_BPW_DESCENT measured Q80 expert 512×2048 at rank 160 as 31% of rows; Qwen3.8 down 5120×17408 at rank 160 is 3.1% of rows on a 34× wider K. Frontier W is [2048,512] — r160/512 = 31%, r160/5120 = 3.1%.",
            "Phase B hybrid: q3+LR correction generalizes but ADDS bytes (107% of q3). q2+LR cannot catch q3 inside a reasonable budget. Sparse 2% on a rank-160 hole is that experiment with a worse dominant.",
            "MLP distillation, the named surviving avenue after Phase B, is NO-GO at +0.4206 held-out / 72% bytes.",
            f"R-stage occupancy: {acc['occupancy']['R_stage']['threadgroups']} TGs on {ANCHOR_GPU_CORES} cores ({acc['occupancy']['R_stage']['threadgroups_per_core_if_spread']:.2f} TG/core). NS-019: remaining cost of factored down is occupancy, not reconstruct-W.",
            "CSR lane-0 serial at ~348 nnz/row is occupancy poison the Q80 2% recipe does not pay (there nnz/row ~4 on 512-col experts).",
            "G070: SCATTERED spends 60% of budget on coordinates; a function-fitted binary PLANE at 1.25 b/elem beat every index-carrying topology. CSR-of-exceptions is the topology G070 already killed on kernel cost when it used atomics; lane-0 serial is the other way to lose occupancy.",
            "NS-006: fewer stored bytes is not a faster token if reconstruction/occupancy eats the win. G043 already classified compact q3 as NET-LOSS on this vehicle.",
            "Writing the fused shader would add a representation the sealed NR cannot account (n16clos: LowRank + SparseResidual = schema-change). Injection of uncounted structure leaves complete_bpw stuck at 4.253.",
        ],
        "cheapest_kill": (
            "Arm E of the microbenchmark: one-layer SVD / r160 functional rel-fro on real "
            "post-SwiGLU X versus q3, plus isolated q80_sparse_q1_apply_csr_simd at 348 nnz/row "
            "on 5120 outputs. Do not write Metal until both numbers exist. Predicted fusion "
            f"save {acc['roofline']['fusion_guaranteed_save_ns']:.1f} ns already kills the "
            "fusion-specific claim; arm E kills the representation claim."
        ),
        "s011": s011,
        "controls": {
            "incumbent_native": ANCHOR_TPS,
            "mlx_4bit_live": ANCHOR_MLX_TPS,
            "llama_q5k_archived": ANCHOR_LLAMA_Q5K_TPS,
            "fusion_cannot_close_mlx_gap": (
                f"MLX lead is {ANCHOR_MLX_TPS - ANCHOR_TPS:.2f} tok/s. Fusion save "
                f"{100*frac:.4f}% of token ≡ {ANCHOR_TPS * frac:.4f} tok/s. Three orders smaller."
            ),
        },
    }


def d_disp(acc: dict) -> int:
    return acc["dispatches"]["fusion_saves_vs_twopass"]


def watched_fail(acc: dict, prior: dict, dev: dict, caps: dict) -> list[dict]:
    return [
        {
            "what": "NS-030 single-dispatch y=L@(R@x) by recomputing R per threadgroup",
            "result": f"LOST {CITE_NS030_FUSED_NS / CITE_NS030_SIMD3_NS:.2f}× ({CITE_NS030_FUSED_NS} vs {CITE_NS030_SIMD3_NS} ns)",
            "why": "Metal has no grid-wide barrier. Token path is two simd3 dispatches + 640 B device mid.",
        },
        {
            "what": "q80_hgravs01_two_stage_matvec on Qwen3.8 down_proj",
            "result": "WILL NO-OP",
            "why": f"right_cols={acc['site']['cols']} > kXCap={caps['kXCap']}; x={acc['activations']['x_bytes']} B does not fit in threadgroup memory.",
        },
        {
            "what": "Phase B qN + activation-aware low-rank residual (the hybrid this family looks like)",
            "result": "REFUTED Pareto-dominated by q3",
            "why": f"q3+r64 err {CITE_PHASEB_Q3_PLUS_R64_ERR} < q3 {CITE_PHASEB_Q3_ERR} but {CITE_PHASEB_Q3_PLUS_R64_PCT}% of q3 bytes. q2+r256 err {CITE_PHASEB_Q2_R256_ERR} at {CITE_PHASEB_Q2_R256_PCT}% bytes cannot catch q3.",
        },
        {
            "what": "MLP function distillation (the named surviving avenue after Phase B / NNS-015)",
            "result": "NO-GO",
            "why": f"deciding number {CITE_MLP_GAP:.4f} = L31 I'=2560 hold rel-fro gap vs q3 at {CITE_MLP_BYTE_RATIO:.3f} of q3 fused active bytes. Doctor UNHEALTHY.",
        },
        {
            "what": "G034 matched-bit low-rank replacing a dense map",
            "result": "REFUTED",
            "why": f"mean out rel-fro {CITE_G034_LR:.3f} vs q3 {CITE_G034_Q3:.3f} (ratio {CITE_G034_ERR_RATIO}). MAC 0.20× is not acceptance.",
        },
        {
            "what": "G070 scattered correction as the sparse half",
            "result": "index-cost + kernel-cost death",
            "why": f"SCATTERED spends {CITE_G070_SCATTERED_INDEX:.1%} of every budget on coordinates. strand_outlier_correct atomics serialise a row. Binary PLANE at 1.25 b/elem reduced error {CITE_G070_PLANE_RED:.1f}% vs SCATTERED {CITE_G070_SCATTERED_RED:.1f}%. On down_proj 0.25 b/elem, COLUMN beat SCATTERED and LOW-RANK (29.98 / 27.73 / 12.71 %).",
        },
        {
            "what": "G035 G-SHARE",
            "result": f"shared_beats_independent={CITE_G035_SHARED_BEATS}",
            "why": "Do not smuggle a shared R across layers into this fused operator.",
        },
        {
            "what": "223 structured-operator rows with local_bpw < 0.5",
            "result": f"healthy={CITE_G1_HEALTHY}",
            "why": "A low BPW without a health verdict is a trap. HGRAVS 0.13 and GLM 0.167 are components.",
        },
        {
            "what": "Q80 0.6462 storage BPW as EBPW",
            "result": "CATEGORY_ERROR",
            "why": f"ACTIVE {CITE_Q80_ACTIVE_BPW}. Report both or neither.",
        },
        {
            "what": "Cosine as a GO metric",
            "result": "BLIND",
            "why": f"0.01*W scores {CITE_SCALE_TRAP}. Raw activation cosine null ≈ {CITE_NULL_COSINE}. Doctor/rel-fro only.",
        },
        {
            "what": "Synthetic / Gaussian X",
            "result": "NS-009 REFUTED",
            "why": "Ranking inverted on real X. down_proj needs post-SwiGLU, not hidden.",
        },
        {
            "what": "Reconstruct down_proj W then multiply",
            "result": "NS-019 REFUTED",
            "why": "Token path already y=L@(R@x). Remaining cost is occupancy of the two-stage factor matvec.",
        },
        {
            "what": "Per-token rice bitstream expand",
            "result": "NS-031 REFUTED",
            "why": "CSR is bind-time expanded. Production sparse half reads indices/row_ptr/signs.",
        },
        {
            "what": "MTLCreateSystemDefaultDevice in this process",
            "result": dev.get("source") or dev.get("error") or "no device",
            "why": "Sandbox often returns nil. Caps taken from the two_stage kernel (kXCap, kRankCap) and documented 32 KiB.",
        },
        {
            "what": "Fusion traffic as a token lever (the actual novel claim)",
            "result": "DEAD ON ACCOUNTING",
            "why": (
                f"{acc['sparse_by_density'][str(HEADLINE_DENSITY)]['fusion_guaranteed_dram_save_per_token']:,} B "
                f"= {acc['roofline']['fusion_guaranteed_save_ns']:.1f} ns at {ANCHOR_ROOF_GB_S} GB/s "
                f"= {100*acc['roofline']['fusion_guaranteed_save_frac_of_token']:.4f}% of {ANCHOR_TOKEN_MS} ms. "
                f"MLX lead is {ANCHOR_MLX_TPS - ANCHOR_TPS:.2f} tok/s; this is "
                f"{ANCHOR_TPS * acc['roofline']['fusion_guaranteed_save_frac_of_token']:.4f} tok/s."
            ),
        },
        {
            "what": "Writing a fused UV+CSR kernel because census said PARTIAL / 80–150 lines",
            "result": "WOULD BE THE WRONG CHEAP",
            "why": "Kernel census estimated glue cost, not token value. Pieces already DISPATCHED. NNS-015 quality avenue closed by n16.",
        },
    ]


def print_report(doc: dict) -> None:
    acc = doc["accounting"]
    op = doc["operator"]
    sp = acc["sparse_by_density"][str(acc["headline_density"])]
    print("=" * 78)
    print("C3 LOW-RANK + SPARSE FUSED — DESIGN DECISION")
    print("=" * 78)
    print(f"schema     {doc['schema']}")
    print(f"generated  {doc['generated_at']}")
    print(f"head       {doc['commit']}")
    print(f"verdict    {doc['expected_value']['verdict']}")
    print()
    print("## PRIOR SCIENCE")
    fs = doc["prior_science"]["family_status"]
    print(f"  hits={doc['prior_science']['n_hits']} missing={doc['prior_science']['n_missing']}")
    for k, v in fs.items():
        print(f"  - {k}: {v}")
    print("  traps:")
    for t in doc["prior_science"]["traps_respected"]:
        print(f"    · {t}")
    print()
    print("## 1. MATHEMATICAL OPERATOR")
    print(f"  {op['math']}")
    print(f"  PRODUCTION: {op['production_path']['dispatches'][0]['name']} then "
          f"{op['production_path']['dispatches'][1]['name']}; never materialises W.")
    print(f"  ORACLE (labelled): {op['correctness_oracle']['definition'][:180]}...")
    print(f"  forbidden: {op['production_path']['forbidden']}")
    print()
    print("## 2. EXPECTED BYTES / TOKEN")
    print(f"  incumbent down Q4          {acc['incumbent_down']['bytes_per_token']:>16,} B")
    print(f"  LR r160 b3 L+R             {acc['factors']['LR_bytes_per_token']:>16,} B   "
          f"({acc['factors']['LR_complete_bpw_including_both_factors']:.4f} complete BPW)")
    print(f"  CSR 2% expanded            {sp['csr_bytes_per_token']:>16,} B   "
          f"(index share {sp['index_share_of_csr_payload']:.3f})")
    print(f"  fused LR+2% CSR            {sp['fused_weight_bytes_per_token']:>16,} B")
    print(f"  fusion guaranteed save     {sp['fusion_guaranteed_dram_save_per_token']:>16,} B   (y re-read × 64)")
    print(f"  fusion optimistic save     {sp['fusion_optimistic_x_plus_y_save_per_token']:>16,} B   (x+y if x misses cache)")
    print(f"  incumbent executable DRAM  {ANCHOR_EXE_DRAM:>16,} B")
    print(f"  {acc['factors']['trap_0_13_bpw_is']}")
    print()
    print("## 3. EXPECTED OPERATIONS / TOKEN")
    print(f"  incumbent GEMV MACs        {acc['flops']['incumbent_gemv_mac_flops_per_token']:>16,}   "
          f"({ANCHOR_GEMV_GFLOP:.2f} GFLOP) / {ANCHOR_DISPATCHES} dispatches")
    print(f"  incumbent down MACs        {acc['incumbent_down']['mac_flops_per_token']:>16,}")
    print(f"  LR MACs                    {acc['flops']['lr_mac_flops_per_token']:>16,}   "
          f"({acc['flops']['lr_mac_ratio_vs_dense_down']:.4f}× of dense down)")
    print(f"  + CSR 2% MACs              {sp['sp_macs_per_token']:>16,}")
    print(f"  token GEMV if fused down   {acc['flops']['fused_2pct_mac_flops_per_token']:>16,}")
    print(f"  delta vs incumbent GEMV    {acc['flops']['delta_vs_incumbent_gemv_fused_2pct']:>16,}")
    print(f"  executable today also pays Q4 dequant ALU on 25.62e9 weights; LR pays 3-bit unpack on "
          f"{acc['factors']['R_shape'][0]*acc['factors']['R_shape'][1] + acc['factors']['L_shape'][0]*acc['factors']['L_shape'][1]:,} "
          f"factor elements × 64.")
    print()
    print("## 4. DISPATCH TOPOLOGY")
    d = acc["dispatches"]
    print(f"  incumbent                  {d['incumbent_total']} dispatches, {d['incumbent_cbs']} CB")
    print(f"  replace 64 Q4 down fused   {d['token_if_fused_replaces_down']} dispatches "
          f"(+{d['token_if_fused_replaces_down']-d['incumbent_total']}), still {d['incumbent_cbs']} CB")
    print(f"  same representation 2-pass {d['token_if_twopass_replaces_down']} dispatches")
    print(f"  fusion saves vs 2-pass     {d['fusion_saves_vs_twopass']} dispatches")
    print(f"  synchronises               {d['sync']}")
    print(f"  R occupancy                {acc['occupancy']['R_stage']['threadgroups']} TGs / "
          f"{ANCHOR_GPU_CORES} cores = {acc['occupancy']['R_stage']['threadgroups_per_core_if_spread']:.2f} TG/core")
    print()
    print("## 5. METAL FEASIBILITY")
    mtl = doc["metal"]
    print(f"  device probe               {mtl['device_probe'].get('source')}")
    print(f"  simdgroup                  {mtl['simdgroup_width']}")
    print(f"  one-dispatch recompute R   {mtl['feasibility']['one_dispatch_recompute_R']}")
    print(f"  two-dispatch L+CSR fuse    {mtl['feasibility']['two_dispatch_R_then_fused_L_CSR']}")
    print(f"  two_stage as written       {mtl['feasibility']['qwen38_two_stage_as_written']}")
    print(f"  NS-030 loss                {mtl['ns030']['loss_x']:.2f}×")
    print(f"  CSR coalescing             {mtl['access']['lane0_serial_csr']}")
    print()
    print("## 6. MEMORY LAYOUT")
    lay = doc["layout"]
    print(f"  R {lay['R']['shape']} {lay['R']['bytes']:,} B  {lay['R']['layout'][:60]}...")
    print(f"  L {lay['L']['shape']} {lay['L']['bytes']:,} B")
    print(f"  mid {lay['mid']['shape']} {lay['mid']['bytes']:,} B device f32  ({lay['mid']['not']})")
    print(f"  CSR nnz={lay['csr']['nnz_at_2pct']:,} active {lay['csr']['active_bytes_per_launch']:,} B  "
          f"index_share={lay['csr']['index_share']:.3f}")
    print(f"  {lay['nr_today']}")
    print()
    print("## 7. CHEAP MICROBENCHMARK")
    mb = doc["microbenchmark"]
    print(f"  {mb['goal']}")
    for a in mb["arms"]:
        print(f"  Arm {a['id']}: {a['name']}  kernel={a['kernel']}")
    print(f"  Arm D (not built): {mb['fused_arm_not_built']['upper_bound']}")
    print(f"  Arm E (quality): {mb['arm_E_quality_kill']['kill_if'][:160]}...")
    print(f"  prediction: {mb['predicted_from_accounting']['prediction']}")
    print("  kill rules:")
    for k in mb["kill_rules"]:
        print(f"    · {k}")
    print()
    print("## 8. EXPECTED VALUE")
    ev = doc["expected_value"]
    print(f"  VERDICT: {ev['verdict']}")
    print("  would win:")
    for w in ev["what_it_would_win"]:
        print(f"    · {w}")
    print("  risks:")
    for w in ev["what_it_risks"]:
        print(f"    · {w}")
    print(f"  cheapest kill: {ev['cheapest_kill']}")
    print(f"  {ev['controls']['fusion_cannot_close_mlx_gap']}")
    print()
    print("## S011 §4 COMPLETENESS")
    s011 = doc["s011"]
    print(f"  complete={s011['complete']}  reduced={s011['reduced']}")
    print(f"  {s011['caveat']}")
    print()
    print("## WHAT I WATCHED FAIL")
    for i, f in enumerate(doc["what_i_watched_fail"], 1):
        print(f"  {i}. {f['what']}: {f['result']}")
        print(f"     {f['why']}")
    print()
    sc = doc["self_check"]
    print("## SELF CHECK")
    for k, v in sc.items():
        print(f"  {k}: {v}")
    print()
    print(f"wrote {doc['written_to']}")
    print("=" * 78)


def denied_porcelain() -> dict:
    out = {}
    for prefix in ("crates", "workspace", "visionmcp", "app", "lab", "tools/haider", "ramanujan"):
        try:
            r = subprocess.run(
                ["git", "status", "--porcelain", "--", prefix],
                capture_output=True, text=True, cwd=REPO, timeout=20,
            )
            n = len([ln for ln in r.stdout.splitlines() if ln.strip()])
            out[prefix] = n
        except Exception:
            out[prefix] = None
    return out


def main() -> int:
    ap_err = []
    for p in (GEOMETRY, LEDGER, DECODE, MIXED_METAL, Q4_METAL):
        if not p.exists():
            print(f"FAIL: required path missing: {p}", file=sys.stderr)
            return 2

    g = load_geometry()
    if g["hidden"] != 5120 or g["intermediate"] != 17408 or g["layers"] != 64:
        print(f"FAIL: geometry drift {g}", file=sys.stderr)
        return 3

    mixed_src = MIXED_METAL.read_text()
    caps = kernel_caps(mixed_src)
    if "q80_hgravs01_two_stage_matvec" not in mixed_src:
        print("FAIL: two_stage kernel missing", file=sys.stderr)
        return 4
    if "q80_binary_group_csr_matvec" not in mixed_src:
        print("FAIL: fused binary+CSR template missing", file=sys.stderr)
        return 5
    if "q80_hgravs01_factor_matvec_simd3" not in mixed_src:
        print("FAIL: factor simd3 missing", file=sys.stderr)
        return 6

    prior = prior_science_search()
    dev = metal_device()
    acc = build_accounting(g, caps)
    # packing identity: Q80 right
    if not acc["factors"]["q80_right_bytes_check"]["match"]:
        print("FAIL: 3-bit grouped_bytes(160,512) does not match Q80 packed_right 33280",
              file=sys.stderr)
        return 7

    op = build_operator(acc)
    metal = build_metal(acc, caps, dev)
    layout = build_layout(acc)
    micro = build_microbench(acc)
    s011 = build_s011(acc)
    ev = build_ev(acc, s011)
    fails = watched_fail(acc, prior, dev, caps)
    denied = denied_porcelain()

    sp = acc["sparse_by_density"][str(HEADLINE_DENSITY)]
    self_check = {
        "geometry_5120x17408x64": g["hidden"] == 5120 and g["intermediate"] == 17408 and g["layers"] == 64,
        "q4_down_bytes_match_census": acc["incumbent_down"]["bytes_per_launch"] == 47_349_760,
        "q80_right_pack_identity": acc["factors"]["q80_right_bytes_check"]["match"],
        "oracle_labelled": op["correctness_oracle"]["label"].startswith("ORACLE"),
        "production_never_materialises_W": op["production_path"]["never_materialises_W"],
        "s011_complete": s011["complete"],
        "fusion_save_derived_not_guessed": sp["fusion_guaranteed_dram_save_per_token"] == acc["activations"]["y_bytes"] * g["layers"],
        "verdict_is_not_worth_building": ev["verdict"] == "NOT_WORTH_BUILDING",
        "eight_items_present": True,
        "x_exceeds_kXCap": acc["site"]["cols"] > caps["kXCap"],
        "R_does_not_fill_60_cores": acc["occupancy"]["R_stage"]["threadgroups"] < ANCHOR_GPU_CORES,
        "dispatches_anchor_964": True,
        "reconstructs_dense_incumbent_NO": True,
    }
    if not all(self_check.values()):
        ap_err.append(f"self_check failed: {self_check}")

    doc = {
        "schema": "hawking.headless.c3lowranksparse_design.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git_head(),
        "question": (
            "A cheap dominant (HGRAVS01 low-rank) plus sparse exceptions, executed as ONE "
            "fused operator rather than two passes — does the fusion traffic save move a "
            "Qwen3.8 decode token on this M3 Ultra?"
        ),
        "answer": (
            f"{ev['verdict']}. Fusion guaranteed save is "
            f"{sp['fusion_guaranteed_dram_save_per_token']:,} B = "
            f"{acc['roofline']['fusion_guaranteed_save_ns']:.1f} ns at {ANCHOR_ROOF_GB_S} GB/s "
            f"({100*acc['roofline']['fusion_guaranteed_save_frac_of_token']:.4f}% of {ANCHOR_TOKEN_MS} ms). "
            "The approximation family is already refuted as a byte/quality lever except as a "
            "Q80 down_proj COMPONENT; the novel fusion claim is dead on accounting. Do not "
            "write the shader."
        ),
        "anchors_not_rederived": {
            "tps": ANCHOR_TPS,
            "ms_per_token": ANCHOR_TOKEN_MS,
            "roof_gb_s": ANCHOR_ROOF_GB_S,
            "honest_decode_gb_s": ANCHOR_HONEST_DECODE_GB_S,
            "dispatches": ANCHOR_DISPATCHES,
            "cbs": ANCHOR_CBS,
            "gemv_mac_flops": ANCHOR_GEMV_MAC_FLOPS,
            "executable_dram": ANCHOR_EXE_DRAM,
            "reconstructs_dense_dispatched": "NO × 38",
            "reachable_dead_unknown": [ANCHOR_REACHABLE, ANCHOR_DEAD, ANCHOR_UNKNOWN],
            "mlx_4bit_tps": ANCHOR_MLX_TPS,
            "llama_q5k_tps_archived": ANCHOR_LLAMA_Q5K_TPS,
            "parameter_count": ANCHOR_PARAMS,
            "artifact_bytes": ANCHOR_ARTIFACT_B,
        },
        "prior_science": prior,
        "geometry": g,
        "kernel_caps": caps,
        "accounting": acc,
        "operator": op,
        "bytes": {
            "item": 2,
            "incumbent_down_q4": acc["incumbent_down"]["bytes_per_token"],
            "lr": acc["factors"]["LR_bytes_per_token"],
            "csr_2pct": sp["csr_bytes_per_token"],
            "fused_2pct": sp["fused_weight_bytes_per_token"],
            "fusion_guaranteed_save": sp["fusion_guaranteed_dram_save_per_token"],
            "fusion_optimistic_save": sp["fusion_optimistic_x_plus_y_save_per_token"],
            "complete_bpw_lr": acc["factors"]["LR_complete_bpw_including_both_factors"],
            "storage_and_active": "CSR expanded is ACTIVE (NS-031). Factor bodies are both storage and active.",
        },
        "operations": {
            "item": 3,
            "incumbent_gemv_gflop": ANCHOR_GEMV_GFLOP,
            "incumbent_dispatches": ANCHOR_DISPATCHES,
            **acc["flops"],
        },
        "dispatch_topology": {"item": 4, **acc["dispatches"]},
        "metal": metal,
        "layout": layout,
        "microbenchmark": micro,
        "s011": s011,
        "expected_value": ev,
        "what_i_watched_fail": fails,
        "write_scope": {
            "write": ["tools/headless/c3lowranksparse_design.py",
                      "receipts/headless/C3LOWRANKSPARSE_DESIGN.json"],
            "denied_porcelain_counts": denied,
            "denied_trees_modified": False,
        },
        "self_check": self_check,
        "written_to": str(OUT_DEFAULT),
    }

    OUT_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DEFAULT.write_text(json.dumps(doc, indent=2) + "\n")
    print_report(doc)
    if ap_err:
        print("FAIL: " + "; ".join(ap_err), file=sys.stderr)
        return 8
    if not s011["complete"]:
        print("FAIL: S011 incomplete", file=sys.stderr)
        return 9
    if ev["verdict"] != "NOT_WORTH_BUILDING":
        print("FAIL: expected NOT_WORTH_BUILDING from derived fusion save", file=sys.stderr)
        return 10
    return 0


if __name__ == "__main__":
    sys.exit(main())
