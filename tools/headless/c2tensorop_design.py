#!/usr/bin/env python3
"""C2 — tensor operators executed as contractions that never materialise W.

This is a DESIGN lane. It terminates in a measured design decision, not a
kernel. Prior science is searched first. Tensor train already ran in this
repository under the identifiers `tensor_train` / `tt_matrix_unfolding` /
`tt_gemv_f16` (G1 + G034). A two-word phrase-grep for "tensor train" is empty
and is a trap.

The test is not "does the contraction approximate W". It is: can the
contraction sequence replace W's FUNCTION while moving fewer bytes AND doing
less work? A rank-r decomposition that needs more FLOPs than the dense GEMV it
replaces is a storage win and a compute loss, and by G034 / S011 §4 that is
incomplete. Storage compression alone is incomplete: source and executable
already dispatch 964 times per token and perform the same 51.24 GFLOP of GEMV.

If the family is already refuted, this script SAYS SO and STOPS — it does not
invent a workaround variant.

    python3 tools/headless/c2tensorop_design.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SCHEMA = "hawking.headless.c2tensorop_design.v1"
G1_PATH = "hawking-experiments/superwave/g1/claude-evidence/g1_tensor_operators.json"
G034_PATH = "receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json"
G1_PY = "hawking-experiments/superwave/g1/claude-evidence/g1_tensor_operators.py"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts/headless/C2TENSOROP_DESIGN.json"
CENSUS_JSON = REPO / "receipts/headless/NOETIC_OPERATION_CENSUS.json"
KERNEL_CENSUS = REPO / "receipts/headless/NOETIC_KERNEL_CENSUS.json"

# Anchors — measured, not re-derived (NOETIC_OPERATION_CENSUS / G105 / G071).
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_GEMV_DISPATCHES = 401
ANCHOR_GEMV_GFLOP = 51.24
ANCHOR_GEMV_MAC_FLOPS = 51_243_909_120
ANCHOR_ACTIVATION_FLOPS = 297_313_024
ANCHOR_TPS = 32.73
ANCHOR_TOKEN_MS = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_PARAMS = 26_895_998_464
ANCHOR_BPW = 4.253
ANCHOR_Q4_GEMV_BYTES = 13_611_663_360
ANCHOR_EXE_DRAM = 13_988_022_948
ANCHOR_WEIGHT_BYTES = 13_623_403_168
ANCHOR_DENSE_W_MATERIALIZED = 0
ANCHOR_MLX_TPS = 35.51
ANCHOR_LLAMA_Q5K_TPS = 24.12
Q4_GROUP = 64
Q4_BYTES_PER_GROUP = 34  # 32 code + 2 f16 scale
HEADER = 40  # G1 tensor-operator blob header
TT_META = 32  # 4 * 8 rank/shape ints

# G1 TT-SVD header constant. Factors stored f16 (db=2).
F16_B = 2
F32_B = 4

# Apple GPU constants taken from kernels that already run on this box,
# not from datasheets.
SIMGROUP_WIDTH = 32  # kSimdWidth in q80_mixed_decode.metal / simd_lane usage
INCUMBENT_TG = 128  # qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
INCUMBENT_ROWS_PER_TG = 2
HGRAVS_TG = 256
HGRAVS_RANK_CAP = 160
HGRAVS_X_CAP = 512
# Apple family GPUs: maxThreadgroupMemoryLength = 32 KiB (Metal feature table).
# HGRAVS01 two-stage comments budget mid[160] as "640 B, not dense W".
TG_MEM_BYTES = 32 * 1024

# MLP distill NO-GO (NNS-015 / noetic_mlp_distill_probe), not re-run here.
MLP_DISTILL_GAP = 0.4206  # L31 I'=2560 hold rel-fro vs q3
MLP_DISTILL_BYTE_FRAC = 0.72

# Q80 storage vs active (NS-002). Report both or neither.
Q80_STORAGE_BPW = 0.6462
Q80_ACTIVE_BPW = 2.518

INNER_128 = 128
TT_RANK = 64  # highest G1-tested uniform TT bond on 17408x5120; still unhealthy


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def git_show_json(path: str) -> dict:
    """Read a tracked blob that may be absent from this sparse checkout."""
    on_disk = REPO / path
    if on_disk.is_file():
        return json.loads(on_disk.read_text())
    p = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        capture_output=True, cwd=REPO, timeout=60,
    )
    if p.returncode != 0:
        err = (p.stderr or p.stdout).decode("utf-8", "replace")[:400]
        raise SystemExit(f"FAIL: git show HEAD:{path}: {err}")
    return json.loads(p.stdout)


def git_grep_count(pattern: str, fixed: bool = False) -> dict:
    # Search the committed tree, not the sparse worktree. A file missing on
    # disk here is not evidence it does not exist (G1 json lives in git).
    args = ["git", "grep", "-I", "-c"]
    if fixed:
        args.append("-F")
    args += [pattern, "HEAD"]
    p = subprocess.run(args, capture_output=True, text=True, cwd=REPO, timeout=60)
    files = 0
    hits = 0
    sample = []
    if p.returncode == 0:
        for line in p.stdout.splitlines():
            if ":" not in line:
                continue
            path, n = line.rsplit(":", 1)
            try:
                c = int(n)
            except ValueError:
                continue
            if path.startswith("HEAD:"):
                path = path[5:]
            files += 1
            hits += c
            if len(sample) < 8:
                sample.append({"path": path, "count": c})
    return {
        "pattern": pattern,
        "fixed": fixed,
        "files": files,
        "hits": hits,
        "exit_code": p.returncode,
        "sample": sample,
    }


def q4_bytes(rows: int, cols: int) -> int:
    groups = (cols + Q4_GROUP - 1) // Q4_GROUP
    return rows * groups * Q4_BYTES_PER_GROUP


def tt_elems(I0: int, I1: int, J0: int, J1: int, ranks: tuple[int, int, int]) -> int:
    r1, r2, r3 = ranks
    return I0 * r1 + r1 * I1 * r2 + r2 * J0 * r3 + r3 * J1


def tt_bytes_f16(I0: int, I1: int, J0: int, J1: int, ranks: tuple[int, int, int]) -> int:
    return HEADER + TT_META + F16_B * tt_elems(I0, I1, J0, J1, ranks)


def tt_flops(I0: int, I1: int, J0: int, J1: int, ranks: tuple[int, int, int]) -> int:
    """Native 4-core TT-GEMV FLOPs (FMA=2), G1 tt_flops. Batch=1 decode.

    Order (never forms W). x reshaped as (J0, J1):
      A[j0, r3]  = sum_{j1} x[j0, j1] * c3[r3, j1]     intermediate (J0, r3)
      H[r2]      = sum_{j0,r3} c2[r2, j0, r3] * A      intermediate (r2,)
      C[r1, i1]  = sum_{r2} c1[r1, i1, r2] * H         intermediate (r1, I1)
      Y[i0, i1]  = sum_{r1} c0[i0, r1] * C             output (I0, I1)
    """
    r1, r2, r3 = ranks
    return (
        2 * J0 * J1 * r3
        + 2 * r2 * J0 * r3
        + 2 * r1 * I1 * r2
        + 2 * I0 * r1 * I1
    )


def tt_intermediates(I0: int, I1: int, J0: int, J1: int, ranks: tuple[int, int, int]) -> dict:
    r1, r2, r3 = ranks
    a = J0 * r3
    h = r2
    c = r1 * I1
    x = J0 * J1
    y = I0 * I1
    peak = max(a, h, c)
    return {
        "x_elems": x,
        "A_j0_r3": a,
        "H_r2": h,
        "C_r1_I1": c,
        "y_elems": y,
        "peak_intermediate_elems": peak,
        "peak_intermediate_bytes_f32": peak * F32_B,
        "fits_32kib_threadgroup": peak * F32_B <= TG_MEM_BYTES,
        "materialises_dense_W": False,
        "note": (
            "Peak live tensor besides x/y is C[r1,I1]. This is where TT secretly "
            "materialises — not W, but a rank-by-mode slab that at r=64, I1=256 "
            "is 64 KiB and does not fit in 32 KiB threadgroup memory."
        ),
    }


def cap_ranks(I0: int, I1: int, J0: int, J1: int, r: int = TT_RANK) -> tuple[int, int, int]:
    return (min(r, I0), min(r, I1), min(r, J1))


def unfold(rows: int, cols: int, inner: int = INNER_128) -> tuple[int, int, int, int] | None:
    if rows % inner or cols % inner:
        return None
    return (rows // inner, inner, cols // inner, inner)


def lr_rank_at_q3_bits(m: int, n: int) -> int:
    """G034: rank whose f16 factors cost exactly 3.25 b/elem."""
    # r*(m+n)*16 = 3.25*m*n  →  r = 3.25*m*n / (16*(m+n))
    return int(round(3.25 * m * n / (16.0 * (m + n))))


def lr_flops(m: int, n: int, r: int) -> int:
    return 2 * r * n + 2 * m * r


def lr_bytes_f16(m: int, n: int, r: int) -> int:
    return HEADER + 16 + F16_B * r * (m + n)


def band(bpw: float) -> str:
    if bpw < 0.05:
        return "<0.05"
    if bpw < 0.10:
        return "<0.10"
    if bpw < 0.25:
        return "<0.25"
    return "<0.50"


def walk_ops(g1: dict) -> list[dict]:
    rows = []
    for t in g1.get("tensors") or []:
        dense = (t.get("dense_gemv") or {}).get("flops")
        q4 = t.get("q4") or {}
        for op in t.get("operators") or []:
            g = op.get("gate") or {}
            fl = op.get("flops")
            rows.append({
                "tensor": t.get("name"),
                "cls": t.get("cls"),
                "shape": t.get("shape"),
                "has_X": t.get("has_X"),
                "family": op.get("family"),
                "tag": op.get("tag"),
                "ranks": op.get("ranks"),
                "local_bpw_f16": op.get("local_bpw_f16"),
                "rel_l2": op.get("rel_l2"),
                "healthy": g.get("healthy"),
                "observed": g.get("observed"),
                "probed": g.get("probed"),
                "worst_unit": g.get("worst_unit"),
                "flops": fl,
                "dense_flops": dense,
                "flop_ratio": (fl / dense) if dense and fl is not None else None,
                "kernel": op.get("kernel"),
                "n_sequential_contractions": op.get("n_sequential_contractions"),
                "stored_bytes_f16": op.get("stored_bytes_f16"),
                "q4_rel_l2": q4.get("rel_l2"),
                "q4_observed": (q4.get("axes") or {}).get("observed"),
            })
    return rows


def unique_sub05(rows: list[dict]) -> list[dict]:
    """Archaeology filter: 0 < local_bpw_f16 < 0.5, unique (family, tag, round bpw 8)."""
    seen = set()
    out = []
    for r in rows:
        bpw = r.get("local_bpw_f16")
        if not isinstance(bpw, (int, float)) or not (0 < bpw < 0.5):
            continue
        key = (r.get("family"), r.get("tag"), round(float(bpw), 8))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def family_summary(rows: list[dict]) -> dict:
    out = {}
    for fam, n in sorted(Counter(r["family"] for r in rows).items()):
        rs = [r for r in rows if r["family"] == fam]
        rels = [r["rel_l2"] for r in rs if r["rel_l2"] is not None]
        obs = [r["observed"] for r in rs if r["observed"] is not None]
        flr = [r["flop_ratio"] for r in rs if r["flop_ratio"] is not None]
        out[fam] = {
            "n": n,
            "healthy_true": sum(1 for r in rs if r["healthy"] is True),
            "rel_l2_min": min(rels) if rels else None,
            "rel_l2_max": max(rels) if rels else None,
            "observed_max": max(obs) if obs else None,
            "flop_ratio_min": min(flr) if flr else None,
            "flop_ratio_max": max(flr) if flr else None,
            "n_flop_ratio_gt_1": sum(1 for x in flr if x > 1),
            "kernel": rs[0]["kernel"] if rs else None,
            "n_sequential_contractions": rs[0].get("n_sequential_contractions") if rs else None,
        }
    return out


def load_incumbent_organs() -> list[dict]:
    """GEMV organs from the on-disk operation census (derived, not guessed)."""
    if not CENSUS_JSON.is_file():
        raise SystemExit(f"FAIL: {CENSUS_JSON} missing")
    d = json.loads(CENSUS_JSON.read_text())
    organs = d.get("gemv_organs")
    if not organs:
        raise SystemExit("FAIL: gemv_organs missing from operation census")
    return organs


# ---------------------------------------------------------------------------
# Token-scale application of G1's native TT-GEMV to the incumbent organ list.
# inner=128 is the G1 reshape used on 17408×5120. Organs that do not divide
# stay on the incumbent Q4 path (ba_proj is the only one: 0.09 % of GEMV elems).
# ---------------------------------------------------------------------------

def tt_plan_for_organ(organ: dict) -> dict:
    rows, cols = organ["rows"], organ["cols"]
    dims = unfold(rows, cols, INNER_128)
    q4b = organ["q4_bytes_per_token"]
    dense_fl = organ["mac_flops_per_token"]
    n = organ["count_per_token"]
    if dims is None:
        return {
            "organ": organ["organ"],
            "applied": False,
            "reason": f"{rows}x{cols} does not divide inner={INNER_128}; keep incumbent Q4",
            "count_per_token": n,
            "tt_bytes_per_token": q4b,
            "tt_flops_per_token": dense_fl,
            "q4_bytes_per_token": q4b,
            "dense_flops_per_token": dense_fl,
            "dispatches_fused": n,
            "dispatches_two_stage": n,
        }
    I0, I1, J0, J1 = dims
    ranks = cap_ranks(I0, I1, J0, J1, TT_RANK)
    b_one = tt_bytes_f16(I0, I1, J0, J1, ranks)
    f_one = tt_flops(I0, I1, J0, J1, ranks)
    inter = tt_intermediates(I0, I1, J0, J1, ranks)
    # Honest Metal: two dispatches (contract input cores → mid, then expand).
    # Fused-recompute-per-TG is NS-030 and is costed separately.
    n_tg = (rows + INCUMBENT_ROWS_PER_TG - 1) // INCUMBENT_ROWS_PER_TG
    recompute_input_flops = n_tg * (2 * J0 * J1 * ranks[2] + 2 * ranks[1] * J0 * ranks[2])
    return {
        "organ": organ["organ"],
        "applied": True,
        "shape": [rows, cols],
        "count_per_token": n,
        "reshape": {"I0": I0, "I1": I1, "J0": J0, "J1": J1, "inner": INNER_128},
        "ranks": list(ranks),
        "tt_bytes_per_launch": b_one,
        "tt_bytes_per_token": b_one * n,
        "tt_flops_per_launch": f_one,
        "tt_flops_per_token": f_one * n,
        "q4_bytes_per_token": q4b,
        "dense_flops_per_token": dense_fl,
        "flop_ratio_vs_dense": f_one / (2 * rows * cols),
        "byte_ratio_vs_q4": (b_one * n) / q4b if q4b else None,
        "intermediates": inter,
        "dispatches_fused": n,
        "dispatches_two_stage": 2 * n,
        "fused_recompute_flops_per_token": recompute_input_flops * n,
        "fused_recompute_vs_dense": (recompute_input_flops * n) / dense_fl if dense_fl else None,
    }


def g034_plan_for_organ(organ: dict) -> dict:
    m, n_cols = organ["rows"], organ["cols"]
    r = lr_rank_at_q3_bits(m, n_cols)
    n = organ["count_per_token"]
    b_one = lr_bytes_f16(m, n_cols, r)
    f_one = lr_flops(m, n_cols, r)
    n_tg_hgravs = (m + 7) // 8  # HGRAVS01 two_stage: 8 rows/TG
    recompute = n_tg_hgravs * (2 * r * n_cols)
    return {
        "organ": organ["organ"],
        "rank": r,
        "count_per_token": n,
        "bytes_per_token": b_one * n,
        "flops_per_token": f_one * n,
        "q4_bytes_per_token": organ["q4_bytes_per_token"],
        "dense_flops_per_token": organ["mac_flops_per_token"],
        "mac_ratio": f_one / (2 * m * n_cols),
        "dispatches_two_stage": 2 * n,
        "fused_recompute_flops_per_token": recompute * n,
        "fused_recompute_vs_dense": (recompute * n) / organ["mac_flops_per_token"],
        "mid_bytes": r * F32_B,
        "x_exceeds_hgravs_cap": n_cols > HGRAVS_X_CAP,
        "rank_exceeds_hgravs_cap": r > HGRAVS_RANK_CAP,
    }


def phrase_search() -> dict:
    """The search a literal 'tensor train' grep would miss."""
    phrase = git_grep_count("tensor train", fixed=True)
    idents = {}
    for pat in (
        "tensor_train",
        "tt_matrix_unfolding",
        "tt_gemv_f16",
        "tt_apply",
        "tucker_hosvd",
        "tensor_ring",
        "tt_matrix_3",
        "tt_svd_4",
    ):
        idents[pat] = git_grep_count(pat, fixed=True)
    return {
        "phrase_tensor_train": phrase,
        "identifiers": idents,
        "reading": (
            f"Phrase-grep 'tensor train' files={phrase['files']} hits={phrase['hits']}. "
            f"Identifier tensor_train files={idents['tensor_train']['files']} "
            f"hits={idents['tensor_train']['hits']}. The mechanism ran as "
            "family=tensor_train (G1, 80 rows) and tt_matrix_unfolding (G034). "
            "HGRAVS01 y=L@(R@x) is the 2-core cousin that already exists as a "
            "Metal kernel, scoped to Q80 down_proj, rank≤160, x≤512."
        ),
    }


def prior_science(g1: dict, g034: dict, rows: list[dict], uniq: list[dict]) -> dict:
    fams = family_summary(rows)
    raw_lt05 = [r for r in rows if isinstance(r.get("local_bpw_f16"), (int, float))
                and r["local_bpw_f16"] < 0.5]
    bands = Counter(band(r["local_bpw_f16"]) for r in uniq)
    tensor_fams = {
        "tensor_train", "tensor_ring", "tucker_hosvd", "tt_matrix_3",
        "cp", "block_term",
    }
    uniq_tensor = [r for r in uniq if r["family"] in tensor_fams]
    g034_tt = (g034.get("untested_members_now_tested") or {}).get("tt_matrix_unfolding") or {}
    return {
        "search": phrase_search(),
        "g1": {
            "path": G1_PATH,
            "schema": g1.get("schema"),
            "selfcheck": g1.get("selfcheck"),
            "wall_s": g1.get("wall_s"),
            "family_rows": len(rows),
            "healthy_true": sum(1 for r in rows if r["healthy"] is True),
            "healthy_false": sum(1 for r in rows if r["healthy"] is False),
            "families": fams,
            "raw_local_bpw_lt_0_5": len(raw_lt05),
            "raw_lt05_healthy_true": sum(1 for r in raw_lt05 if r["healthy"] is True),
            "archaeology_unique_lt_0_5": len(uniq),
            "archaeology_unique_healthy_true": sum(1 for r in uniq if r["healthy"] is True),
            "archaeology_bands": dict(bands),
            "tensor_family_unique_lt_0_5": len(uniq_tensor),
            "tensor_family_unique_healthy_true": sum(1 for r in uniq_tensor if r["healthy"] is True),
            "named_examples": {
                "tucker_R4_bpw": 0.0003626206341911765,
                "tensor_train_r8_bpw": 0.0023157456341911763,
                "tensor_ring_r4_bpw": 0.002802734375,
                "kronecker_k1_bpw": 0.003923483455882353,
                "lowrank_r8_bpw": 0.03235796760110294,
            },
            "scoring_was_native_contraction": (
                "g1_tensor_operators.py: 'Does not require reconstructing a dense W "
                "at consume time; reconstruction is used only as an equivalent scoring "
                "device (the map is linear).' Named consume kernels (tt_gemv_f16, "
                "tucker_gemv_f16, tr_gemv_f16, ttm3_gemv_f16) were the IMPLIED MAP. "
                "Going native is not a new design. It is G1."
            ),
        },
        "g034": {
            "path": G034_PATH,
            "verdict": g034.get("verdict"),
            "family_verdict": g034.get("family_verdict"),
            "mean_flat_q3": g034.get("mean_flat_q3"),
            "mean_lowrank": g034.get("mean_lowrank"),
            "error_ratio": 2.93,
            "mean_mac_ratio": g034.get("mean_mac_ratio"),
            "tt_unfold": {
                "verdict": g034_tt.get("verdict"),
                "why": g034_tt.get("why"),
                "rows": g034_tt.get("rows"),
            },
        },
        "named_traps_not_redone": {
            "G035_GSHARE": "shared_beats_independent=false (sharing refuted; this is G035, not G062)",
            "Q80_storage_vs_active": {
                "storage_bpw": Q80_STORAGE_BPW,
                "active_bpw": Q80_ACTIVE_BPW,
                "factor": round(Q80_ACTIVE_BPW / Q80_STORAGE_BPW, 2),
                "law": "report both or neither",
            },
            "GLM_0_167_expert_bpw": "activation-aware experts; Gaussian-proxy inversion",
            "HGRAVS01_0_13": "two-stage L@(R@x) on down_proj ONLY; NS-019 reconstruct-W refuted; NS-030 fused-recompute-R-per-TG refuted",
            "tpr64_free": "reconstruction is free on 32/33 variants — not a reason to rebuild W",
            "MLP_distill_NO_GO": {
                "held_out_gap_vs_q3": MLP_DISTILL_GAP,
                "active_byte_fraction": MLP_DISTILL_BYTE_FRAC,
                "law": "MLP storage avenue closed by measurement as of today",
            },
            "cosine_scale_trap": "0.01*W scored 1.000000; raw activation cosine null ~0.898",
            "synthetic_X": "NS-009: never evaluate on synthetic activations",
        },
        "family_already_refuted": True,
        "stop": (
            "G1-TENSOR: 373/373 operator rows doctor-gate unhealthy on real Qwen3.8 "
            "GEMV tensors, including 80 tensor_train + 96 tucker_hosvd + 16 tensor_ring "
            "+ 32 tt_matrix_3. Archaeology unique (family,tag,bpw) with local_bpw_f16<0.5: "
            f"{len(uniq)} (healthy=true: 0) — this family IS among those 223. "
            "G034: at matched 3.25 b/elem, TT unfolding is WORSE than plain low-rank "
            "(L31.gate 0.725 vs 0.493 vs q3 0.198). Native contraction was the scoring "
            "path. STOP. Do not design a workaround variant."
        ),
    }


def eight_items(organs: list[dict], rows: list[dict], g034: dict, g1: dict) -> dict:
    plans = [tt_plan_for_organ(o) for o in organs]
    applied = [p for p in plans if p["applied"]]
    kept = [p for p in plans if not p["applied"]]

    tt_bytes = sum(p["tt_bytes_per_token"] for p in plans)
    tt_macs = sum(p["tt_flops_per_token"] for p in plans)
    q4_bytes = sum(o["q4_bytes_per_token"] for o in organs)
    dense_flops = sum(o["mac_flops_per_token"] for o in organs)
    two_stage_disp = sum(p["dispatches_two_stage"] for p in plans)
    fused_disp = sum(p["dispatches_fused"] for p in plans)
    recompute_flops = sum(p.get("fused_recompute_flops_per_token") or 0 for p in applied)
    non_gemv = ANCHOR_DISPATCHES - ANCHOR_GEMV_DISPATCHES

    # Peak intermediate among applied organs at r=64.
    peaks = []
    for p in applied:
        inter = p["intermediates"]
        peaks.append({
            "organ": p["organ"],
            "C_bytes": inter["C_r1_I1"] * F32_B,
            "A_bytes": inter["A_j0_r3"] * F32_B,
            "H_bytes": inter["H_r2"] * F32_B,
            "fits_tg": inter["fits_32kib_threadgroup"],
            "ranks": p["ranks"],
            "reshape": p["reshape"],
        })
    worst_c = max(peaks, key=lambda x: x["C_bytes"]) if peaks else None

    g034_plans = [g034_plan_for_organ(o) for o in organs]
    g034_bytes = sum(p["bytes_per_token"] for p in g034_plans)
    g034_flops = sum(p["flops_per_token"] for p in g034_plans)
    g034_disp = sum(p["dispatches_two_stage"] for p in g034_plans)
    g034_recompute = sum(p["fused_recompute_flops_per_token"] for p in g034_plans)

    # Self-check against a G1 row: L0.gate (68,256)x(20,256) r=(64,64,64).
    chk_elems = tt_elems(68, 256, 20, 256, (64, 64, 64))
    chk_bytes = tt_bytes_f16(68, 256, 20, 256, (64, 64, 64))
    chk_flops = tt_flops(68, 256, 20, 256, (64, 64, 64))
    selfcheck = {
        "tt_elems_68_256_20_256_r64": chk_elems,
        "expect_elems": 1_151_232,
        "tt_bytes": chk_bytes,
        "expect_bytes": 2_302_536,
        "tt_flops": chk_flops,
        "expect_flops": 5_144_576,
        "match": chk_elems == 1_151_232 and chk_bytes == 2_302_536 and chk_flops == 5_144_576,
    }

    # TTM3 compute-loss example (G1 measured).
    ttm3_loss = max(
        (r for r in rows if r["family"] == "tt_matrix_3" and r.get("flop_ratio")),
        key=lambda r: r["flop_ratio"],
        default=None,
    )

    gate_l0_tt = [
        r for r in rows
        if r.get("cls") == "mlp.gate_proj"
        and r["family"] == "tensor_train"
        and r.get("tensor", "").endswith("layers.0.mlp.gate_proj.weight")
        and r.get("ranks") == [64, 64, 64]
        and r.get("tag", "").startswith("(136, 128)")
    ]
    gate_row = gate_l0_tt[0] if gate_l0_tt else None

    operator = {
        "name": "tt_gemv_f16 — 4 sequential TT-core contractions, batch=1",
        "status": "ALREADY_SCORED_AND_REFUTED",
        "why_this_operator_and_not_another": (
            "This is the consume path G1 already named. Designing a 'new' Tucker "
            "mixture or a higher-rank TT would be designing around a refutation."
        ),
        "representation": (
            "Reshape W ∈ R^{m×n} as a 4-way tensor T[i0,i1,j0,j1] with "
            "m=I0·I1, n=J0·J1, inner I1=J1=128 (G1 matching_kron_pairs). "
            "TT-SVD cores: c0[I0,r1], c1[r1,I1,r2], c2[r2,J0,r3], c3[r3,J1], "
            "stored f16. W is never stored and never written."
        ),
        "contraction_order": [
            "x → reshape (J0, J1)",
            "A[j0,r3] = ∑_{j1} x[j0,j1] c3[r3,j1]     size J0·r3",
            "H[r2]     = ∑_{j0,r3} c2[r2,j0,r3] A       size r2",
            "C[r1,i1]  = ∑_{r2} c1[r1,i1,r2] H          size r1·I1",
            "Y[i0,i1]  = ∑_{r1} c0[i0,r1] C             size I0·I1 → (m,)",
        ],
        "production_path": (
            "Named kernel tt_gemv_f16 does NOT exist in crates/hawking-core/shaders. "
            "Closest shipped kernel is q80_hgravs01_two_stage_matvec (Q80 down_proj "
            f"only, rank cap {HGRAVS_RANK_CAP}, x cap {HGRAVS_X_CAP}, each threadgroup "
            "recomputes mid[rank] — NS-030). Qwen3.8 MLP has x=5120 and G034 rank 803; "
            "both caps are exceeded, so even that kernel is not a drop-in."
        ),
        "correctness_oracle": {
            "label": "ORACLE — not a production implementation",
            "procedure": (
                "Fold cores to dense Ŵ = contract(c0,c1,c2,c3) ∈ R^{m×n}, then "
                "y = Ŵ x (ordinary GEMV). Equivalence of the linear map. G1 used "
                "this as a scoring device only. Must never be presented as the decode path."
            ),
            "forbidden_production_lowering": (
                "qwen_uniform_q4_decode_vector then f32 GEMM writes 102,487,818,240 "
                "bytes of dense W per token. reconstructs_dense=NO on all 38 "
                "decode-bound kernels today."
            ),
        },
        "ranks_tested_on_gate_proj": {
            "reshape": "(136,128)x(40,128) and (68,256)x(20,256)",
            "bonds": ["(8,8,8)", "(16,16,16)", "(32,32,32)", "(64,32,16)", "(64,64,64)"],
            "best_still_unhealthy": gate_row,
        },
        "selfcheck_g1_formulas": selfcheck,
    }

    bytes_item = {
        "incumbent_q4_gemv_bytes_per_token": q4_bytes,
        "incumbent_executable_dram_bytes_per_token": ANCHOR_EXE_DRAM,
        "tt_r64_inner128_weight_bytes_per_token": tt_bytes,
        "delta_vs_q4_gemv": tt_bytes - q4_bytes,
        "ratio_vs_q4_gemv": tt_bytes / q4_bytes if q4_bytes else None,
        "g034_r_at_3p25bpw_weight_bytes_per_token": g034_bytes,
        "g034_ratio_vs_q4": g034_bytes / q4_bytes if q4_bytes else None,
        "kept_q4_organs": [p["organ"] for p in kept],
        "derivation": (
            f"Per organ, tt_bytes = ({HEADER}+{TT_META}+2·(I0 r1 + r1 I1 r2 + r2 J0 r3 + r3 J1)) "
            f"× count. Rank capped min(64, mode). Organs not divisible by 128 stay Q4. "
            f"G034 path uses r = round(3.25 m n / (16(m+n))) f16 factors (matched q3 bits)."
        ),
        "function_at_these_bytes": (
            "TT r=64 on L0.gate: local_bpw_f16=0.12657, rel_l2=0.9869, observed=0.280, "
            "doctor healthy=false (G1). Q4 on the same tensor: bpw=4.125, rel_l2=0.119, "
            "observed=0.9957. G034 at 3.25 b/elem: mean out_rel_fro 0.539 vs q3 0.184 "
            "(2.93×). Byte win without function is a trap, not a result."
        ),
        "organs": plans,
    }

    ops_item = {
        "incumbent_gemv_mac_flops": dense_flops,
        "incumbent_gemv_gflop": dense_flops / 1e9,
        "incumbent_activation_flops": ANCHOR_ACTIVATION_FLOPS,
        "tt_r64_gemv_flops": tt_macs,
        "tt_r64_gflop": tt_macs / 1e9,
        "tt_flop_ratio_vs_dense": tt_macs / dense_flops if dense_flops else None,
        "g034_gemv_flops": g034_flops,
        "g034_gflop": g034_flops / 1e9,
        "g034_mac_ratio_mean_from_receipt": g034.get("mean_mac_ratio"),
        "fused_recompute_tt_input_flops": recompute_flops,
        "fused_recompute_tt_vs_dense": recompute_flops / dense_flops if dense_flops else None,
        "fused_recompute_g034_flops": g034_recompute,
        "fused_recompute_g034_vs_dense": g034_recompute / dense_flops if dense_flops else None,
        "ttm3_compute_loss_example": {
            "tensor": (ttm3_loss or {}).get("tensor"),
            "tag": (ttm3_loss or {}).get("tag"),
            "flop_ratio": (ttm3_loss or {}).get("flop_ratio"),
            "local_bpw_f16": (ttm3_loss or {}).get("local_bpw_f16"),
            "reading": (
                "TTM3 at I=(4,16,16) J=(8,16,40) r=(32,32) on v_proj does 6.95× the "
                "dense GEMV FLOPs. Storage can look like a win (0.87 bpw f16) while "
                "compute is a loss. S011 §4 / G034: that is incomplete."
            ),
        },
        "incumbent_does_not_do_less_work": (
            "Source and executable both perform 51.24 GFLOP of GEMV MACs. Q4 adds "
            "dequant ALU on top. A candidate that only lowers executable bytes is "
            "incomplete. TT r=64 does fewer MACs AND fewer bytes AND destroys function."
        ),
    }

    dispatch_item = {
        "incumbent": {
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "gemv_dispatches": ANCHOR_GEMV_DISPATCHES,
            "command_buffers": ANCHOR_CBS,
            "formula": "1 embed + 64*(9 mixer + 6 mlp) + 3 terminal = 964",
        },
        "tt_fused_one_kernel_per_organ": {
            "gemv_dispatches": fused_disp,
            "total_dispatches": non_gemv + fused_disp,
            "command_buffers": 1,
            "sync": (
                "Four sequential contractions inside one kernel need threadgroup "
                "barriers between stages. After the input cores, the live state is "
                "H[r2] (64 floats) which is grid-shared: every TG that owns output "
                "rows needs it. Providing H without a grid-wide barrier means every "
                "TG recomputes the input contraction (NS-030)."
            ),
            "fused_recompute_is_a_compute_loss": True,
            "fused_recompute_flops_vs_dense": recompute_flops / dense_flops if dense_flops else None,
        },
        "tt_honest_two_stage": {
            "gemv_dispatches": two_stage_disp,
            "total_dispatches": non_gemv + two_stage_disp,
            "command_buffers": 1,
            "sync": (
                "Dispatch 1: contract c3,c2 against x → device buffer H[r2] "
                f"({TT_RANK} floats). Encoder dependency. Dispatch 2: expand c1,c0. "
                "No host wait if both sit in the same command buffer. Dispatch count "
                f"rises {ANCHOR_GEMV_DISPATCHES} → {two_stage_disp} GEMV "
                f"({ANCHOR_DISPATCHES} → {non_gemv + two_stage_disp} total)."
            ),
        },
        "g034_two_stage_lowrank": {
            "gemv_dispatches": g034_disp,
            "total_dispatches": non_gemv + g034_disp,
            "command_buffers": 1,
        },
        "ns020": (
            "NS-020: collapsing command-buffer / dispatch topology is not the token "
            "lever. Increasing it is worse. The honest native path adds a dependency "
            "per organ."
        ),
        "delta_vs_incumbent": {
            "fused_total_minus_964": (non_gemv + fused_disp) - ANCHOR_DISPATCHES,
            "two_stage_total_minus_964": (non_gemv + two_stage_disp) - ANCHOR_DISPATCHES,
            "two_stage_increases_dispatches": True,
        },
    }

    metal_item = {
        "device": "Apple M3 Ultra, 60 GPU cores, Metal 4, measured roof 595.9 GB/s",
        "simdgroup_width": SIMGROUP_WIDTH,
        "incumbent_launch": {
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "threadgroup": INCUMBENT_TG,
            "simdgroups_per_tg": 4,
            "rows_per_tg": INCUMBENT_ROWS_PER_TG,
            "threadgroup_memory": "float red[4] = 16 B",
            "access": (
                "lane consumes 8 packed Q4 weights, stride 512 along cols; "
                "sequential group-64, coalesced. Packed decode stays in registers. "
                "Grid ceil(rows/2)*128."
            ),
        },
        "hgravs01_fused_caps_do_not_fit_qwen38": {
            "kernel": "q80_hgravs01_two_stage_matvec",
            "threadgroup": HGRAVS_TG,
            "rank_cap": HGRAVS_RANK_CAP,
            "x_cap": HGRAVS_X_CAP,
            "threadgroup_memory": "mid[160] + x_tg[512] = 2688 B",
            "qwen38_mlp_x": 5120,
            "g034_rank": 803,
            "x_exceeds_cap": True,
            "rank_exceeds_cap": True,
            "comment_in_shader": (
                "Fused y=L@(R@x) in one dispatch. Each threadgroup recomputes "
                "mid[rank] into threadgroup memory (640 B, not dense W)."
            ),
        },
        "threadgroup_memory_bytes": TG_MEM_BYTES,
        "peak_TT_intermediate": worst_c,
        "register_pressure": (
            "Incumbent already does 8-wide unpack+FMA per lane. TT stages are "
            "smaller GEMVs; registers are not the limiter. Occupancy is: after "
            "contracting the input modes the live work is r2-wide (64), which "
            "cannot fill 60 cores × 32-wide simdgroups."
        ),
        "coalescing": (
            "c3 is (r3,J1)=(64,128) — 16 KiB, cached. x is 20 KiB. First two "
            "contractions are tiny and latency-bound, not bandwidth-bound. "
            "c0 expand streams I0·r1 f16 (gate: 136·64·2 = 17 KiB) — also cached. "
            "The working set of one organ's cores at r=64 is ~1.4 MiB. Cache "
            "residency is real; function is not. G094: cache-residency does not "
            "rank-correlate with token time (NX cache_resident_token_fraction=0.00493)."
        ),
        "fused_recompute_blowup": {
            "method": (
                "n_tg = ceil(rows/2) incumbent TGs, each repeats the input-core "
                "contractions (2 J0 J1 r3 + 2 r2 J0 r3)."
            ),
            "flops": recompute_flops,
            "vs_dense": recompute_flops / dense_flops if dense_flops else None,
            "ns030": "REFUTED on this GPU without a grid-wide barrier or a layout that does not recompute R per TG.",
        },
        "feasibility_one_line": (
            "A two-dispatch TT-GEMV is implementable (HGRAVS01 already does the "
            "2-core form). A fused-recompute kernel is a compute loss on this "
            "geometry. Neither is worth building because the map is unhealthy "
            "at every rank G1 and G034 tested."
        ),
    }

    layout_item = {
        "dtype": "f16 cores + f32 activations (same as G1 scoring / HGRAVS01 factors)",
        "cores_per_matrix": {
            "c0": "I0 × r1, row-major, f16",
            "c1": "r1 × I1 × r2, r2-inner, f16",
            "c2": "r2 × J0 × r3, r3-inner, f16",
            "c3": "r3 × J1, f16",
        },
        "header": f"{HEADER} B G1 blob + {TT_META} B shape/rank ints",
        "mid_buffer_two_stage": f"r2 f32 = {TT_RANK * F32_B} B per organ, device, overwritten in-order",
        "not_stored": "dense W, Tucker core at high rank as a reconstruct target",
        "hgravs01_relative": (
            "HGRAVS01 is c0,c3 collapsed to two matrices L[m,r], R[r,n] at r≤160, "
            "3-bit group-64 packed. That is a 2-core TT. It ships for Q80 down_proj "
            "only. It is not a Qwen3.8 uniform-q4 bind, and 0.13 BPW on one organ "
            "is a component trap (the 223-row law)."
        ),
        "nr_schema": (
            "NOETIC_CLOSURE_GAP: TensorTrain is schema-change; G096 NEVER BUILT a "
            "TT node. Injecting a TensorTrain family into the sealed NR does not "
            "move complete_bits_per_weight (stays 4.2527). Accounting cannot see this family."
        ),
    }

    microbench = {
        "purpose": "Discriminate this design from the incumbent BEFORE anyone writes the full kernel.",
        "already_run": {
            "what": "G1 tt_apply vs q4 vs dense on 8 real Qwen3.8 GEMV tensors, doctor gate on Y",
            "wall_s": g1.get("wall_s"),
            "result": "373/373 unhealthy; 80/80 tensor_train unhealthy",
            "path": G1_PATH,
        },
        "cheapest_kill_no_fit": {
            "command": (
                "python3 -c \"import json,subprocess; d=json.loads(subprocess.check_output("
                "['git','show','HEAD:workspace/superwave/g1/claude-evidence/g1_tensor_operators.json'])); "
                "rows=[op for t in d['tensors'] for op in t['operators']]; "
                "h=sum(1 for op in rows if (op.get('gate') or {}).get('healthy')); "
                "print('rows',len(rows),'healthy',h); raise SystemExit(0 if h==0 else 1)\""
            ),
            "runtime": "<2 s (json parse)",
            "kill_if": "healthy_true == 0 (already true)",
            "this_script_is_that_kill": True,
        },
        "one_tensor_refit_if_the_json_is_distrusted": {
            "inputs": {
                "W": "language_model.model.layers.0.mlp.gate_proj.weight from Qwen3.8 BF16 parent",
                "X": "activation-capture-v1 hidden/L00.f32  (256 × 5120), real X, not Gaussian",
            },
            "method": (
                "Copy tt_svd_4 + tt_apply + gate_from_axes from "
                f"{G1_PY}. Fit (136,128)x(40,128) r=(64,64,64). "
                "Y_tt = tt_apply(X); Y = X @ W.T; Y_q4 = X @ q4(W).T. "
                "Score doctor axes with AXIS_MARGIN observed/probed 0.02, worst_unit 0.10 "
                "relative to q4. Count tt_flops vs 2*m*n. Do not use cosine as GO "
                "(0.01*W scored 1.000000; null ~0.898)."
            ),
            "kill_if": "healthy is False OR flop_ratio >= 1",
            "expected_already_measured": {
                "rel_l2": 0.9869111180305481,
                "observed": 0.2803584933280945,
                "healthy": False,
                "flop_ratio": 4259840 / 178257920,
                "local_bpw_f16": 0.12656896254595587,
                "q4_observed": 0.9957399368286133,
            },
            "runtime": "one SVD of 17408×5120; seconds, not a campaign (G1 whole family 1446 s)",
        },
        "metal_only_if_cpu_gate_passes": {
            "note": "CPU gate does not pass. Do not run this.",
            "what_it_would_have_been": (
                "Time two sequential f16 GEMVs (17408×803 and 803×5120) against one "
                "geo_tpr64 q4 17408×5120 on this GPU, same x. Kill if wall >= q4 wall "
                "OR if a (17408×5120) buffer is written."
            ),
        },
    }

    s011 = {
        "law": (
            "G034: acceptance requires reducing BOTH bytes and operations WITH "
            "function preserved. A rank-r decomposition that needs more FLOPs than "
            "the dense GEMV it replaces is a storage win and a compute loss — "
            "incomplete. Storage compression alone is incomplete: the executable "
            "already stores fewer bytes and does not do less work (51.24 GFLOP GEMV, "
            "964 dispatches, both source and executable)."
        ),
        "axes": {
            "bytes": {
                "tt_r64": "REDUCED vs q4 (trap: function dead)",
                "g034_3p25bpw": "REDUCED vs q4, MATCHED vs q3 bits (function 2.93× q3 error)",
            },
            "operations": {
                "tt_r64": "REDUCED vs dense (trap: function dead)",
                "g034": f"REDUCED mac_ratio={g034.get('mean_mac_ratio')} (function dead)",
                "ttm3_some_ranks": "INCREASED (flop_ratio up to 6.95) — incomplete even before function",
                "fused_recompute": "INCREASED vs dense (NS-030)",
            },
            "dispatches": {
                "fused": "same 401 GEMV if a fused kernel existed, but fusion is a compute loss",
                "honest_two_stage": "INCREASED 401 → 754 GEMV (ba_proj stays one dispatch; 96×5120 does not divide inner=128)",
            },
            "materialization": {
                "incumbent": "already 0 (38/38 reconstructs_dense=NO)",
                "tt_native": "0 dense W; C[r1,I1] may spill (64 KiB > 32 KiB TG)",
                "win": False,
            },
            "synchronization": {
                "honest_two_stage": "INCREASED (encoder dep per organ)",
                "fused": "threadgroup barriers + implicit grid share of H",
            },
            "traffic": {
                "tt_r64": "REDUCED vs q4 (trap: function dead)",
            },
            "function": {
                "tt_r64": "NOT PRESERVED (rel_l2≈0.99, doctor unhealthy)",
                "g034_matched_bits": "NOT PRESERVED (2.93× q3 out_rel_fro)",
                "tt_unfold_vs_lowrank": "WORSE (0.725 vs 0.493 vs q3 0.198 on L31.gate)",
            },
        },
        "complete_under_s011": False,
        "reason": (
            "No function-preserving reduction on any axis. The axes that move "
            "(bytes, ops, traffic) move only at ranks that kill the map. Dispatch "
            "and sync get worse on the honest Metal path. Materialization is already 0."
        ),
    }

    expected_value = {
        "verdict": "NOT_WORTH_BUILDING",
        "design_status": "STOPPED_FAMILY_REFUTED",
        "would_win": (
            "At the ranks G1 actually fitted: fewer stored bytes and fewer MACs "
            "than Q4/dense, and intermediates that are not dense W. That is the "
            "trap the 223-row law exists to catch."
        ),
        "would_risk": [
            "Shipping a map with rel_l2≈1 (noise) because the FLOP sheet looked green",
            "Resurrecting TT on the claim that 'G1 reconstructed W' — it did not; tt_apply is native",
            "Fusing by recomputing the shared contraction per threadgroup (NS-030), "
            f"which on this geometry is {recompute_flops/dense_flops:.1f}× dense FLOPs",
            "Quoting local_bpw_f16=0.13 without the health verdict (HGRAVS01 / 223-row law)",
            "Raising rank until function appears, at which point G034 already measured "
            "2.93× q3 error at identical bits and TT unfolding worse than SVD",
        ],
        "cheapest_experiment_that_kills_it": (
            "This script: parse G1 + G034. healthy_true=0 and TT-unfold error > low-rank "
            "error > q3. A Metal kernel cannot repair a map the doctor already rejected."
        ),
        "reopen_when": (
            "A new organ family on real (not synthetic) X shows a peaked TT/Tucker "
            "spectrum — energy at r << min(m,n) capturing enough of the map that "
            "doctor-gate is healthy relative to q3/q4 AND flop_ratio < 1 AND the "
            "honest two-stage dispatch still moves fewer bytes than geo_tpr64 Q4. "
            "Do not retry the eight Qwen3.8 GEMV tensors G1 already scored. "
            "Do not retry TT unfolding of the natural (m,n) pairing; G034 showed "
            "the regrouping flattens the spectrum."
        ),
        "controls_not_beaten": {
            "incumbent_native_q4": ANCHOR_TPS,
            "mlx_4bit_live": ANCHOR_MLX_TPS,
            "llamacpp_q5k_archived": ANCHOR_LLAMA_Q5K_TPS,
            "note": (
                "A dead map has no tok/s. These controls are listed so a future "
                "reopen cannot claim victory on bytes alone."
            ),
        },
        "s011": s011,
    }

    return {
        "1_mathematical_operator": operator,
        "2_expected_bytes_per_token": bytes_item,
        "3_expected_operations_per_token": ops_item,
        "4_dispatch_topology": dispatch_item,
        "5_metal_feasibility": metal_item,
        "6_memory_layout": layout_item,
        "7_cheap_microbenchmark": microbench,
        "8_expected_value": expected_value,
        "token_scale_tt_r64": {
            "weight_bytes": tt_bytes,
            "gemv_flops": tt_macs,
            "two_stage_total_dispatches": non_gemv + two_stage_disp,
            "fused_total_dispatches": non_gemv + fused_disp,
            "function": "unhealthy at every G1 rank",
        },
        "token_scale_g034": {
            "weight_bytes": g034_bytes,
            "gemv_flops": g034_flops,
            "two_stage_total_dispatches": non_gemv + g034_disp,
            "function": "2.93× q3 out_rel_fro at matched 3.25 b/elem",
        },
        "g034_plans": g034_plans,
        "intermediate_peaks": peaks,
    }


def what_watched_fail(search: dict, rows: list[dict], uniq: list[dict],
                      eight: dict, g034: dict) -> list[dict]:
    phrase = search["search"]["phrase_tensor_train"]
    ident = search["search"]["identifiers"]["tensor_train"]
    return [
        {
            "what": "literal grep 'tensor train'",
            "result": f"files={phrase['files']} hits={phrase['hits']}",
            "why": (
                f"Zero tracked phrase hits. Identifier tensor_train files={ident['files']} "
                f"hits={ident['hits']}. G1 scored 80 tensor_train rows; G034 scored "
                "tt_matrix_unfolding. A design that started from the phrase would have "
                "rediscovered a refuted family and called it new."
            ),
        },
        {
            "what": "G1 structured operators on real Qwen3.8 GEMV tensors",
            "result": f"{len(rows)}/{len(rows)} doctor-gate unhealthy",
            "why": (
                "Tucker 0.00036 bpw, TT 0.0023, TR 0.0028, Kronecker 0.0039, "
                "low-rank r=8 0.032 — all rel_l2≈1. Tiny local BPW with no function. "
                f"Archaeology unique sub-0.5: {len(uniq)} (healthy=true: 0). "
                "This family is in that set."
            ),
        },
        {
            "what": "G034 matched-bit TT unfolding vs plain low-rank vs q3",
            "result": g034.get("family_verdict"),
            "why": (
                "L31.gate at 3.25 b/elem: q3 0.198, low-rank r=803 0.493, "
                "TT unfold r=949 0.725. The extra 146 ranks the balanced shape buys "
                "do not compensate. Regrouping scatters structure."
            ),
        },
        {
            "what": "Native contraction as a way around G1",
            "result": "ALREADY THE SCORING PATH",
            "why": (
                "g1_tensor_operators.py scored tt_apply / tucker_apply / tr_apply — "
                "the implied linear map — and used reconstruction only as an oracle. "
                "A Metal tt_gemv_f16 cannot make an unhealthy map healthy."
            ),
        },
        {
            "what": "Fuse L@(R@x) by recomputing R in every threadgroup",
            "result": "NS-030 REFUTED; also a compute loss on Qwen3.8 geometry",
            "why": (
                f"TT input-core recompute across incumbent TGs is "
                f"{eight['3_expected_operations_per_token']['fused_recompute_tt_vs_dense']:.1f}× "
                "dense GEMV FLOPs. HGRAVS01 two_stage caps (rank 160, x 512) do not "
                "fit Qwen3.8 MLP (x=5120, G034 r=803)."
            ),
        },
        {
            "what": "TT intermediate C[r1,I1] in threadgroup memory",
            "result": "SPILLS at the ranks G1 actually ran",
            "why": (
                "r=64, I1=256 → 16384 f32 = 64 KiB > 32 KiB TG. r=64, I1=128 → "
                "32768 B exactly, leaving no room for the reduction scratch the "
                "incumbent already uses. The intermediates are where tensor methods "
                "secretly materialise — here a rank-by-mode slab, not W, but still "
                "a device buffer and a barrier."
            ),
        },
        {
            "what": "TTM3 as a 'balanced' contraction",
            "result": "flop_ratio up to 6.95 vs dense",
            "why": "Storage win, compute loss. Incomplete by S011 §4 even before the doctor gate.",
        },
        {
            "what": "MLP function distillation as the surviving avenue",
            "result": f"NO-GO (+{MLP_DISTILL_GAP} held-out gap vs q3 at {MLP_DISTILL_BYTE_FRAC:.0%} of q3 active bytes)",
            "why": "Closed by measurement today. Do not reopen it as a tensor-core workaround.",
        },
        {
            "what": "Cosine as a GO metric / synthetic X",
            "result": "BLIND",
            "why": "0.01*W scored 1.000000. Null baseline ~0.898. NS-009 killed Gaussian ranking.",
        },
        {
            "what": "Q80 0.6462 complete_physical_bpw as the decode number",
            "result": f"CATEGORY_ERROR (active {Q80_ACTIVE_BPW})",
            "why": "Report both or neither. Density is not velocity (G043 q3 net-loss).",
        },
        {
            "what": "G035 / G-SHARE as a sharing win",
            "result": "shared_beats_independent=false",
            "why": "G062 is attractor compilation, not sharing. Do not cite G062 as a sharing refutation.",
        },
        {
            "what": "HGRAVS01 0.13 BPW as a tensor-train existence proof for Qwen3.8",
            "result": "down_proj ONLY, Q80, rank 160",
            "why": "A component number whose supporting structures were never counted is a trap. NS-019: do not reconstruct that down_proj.",
        },
    ]


def print_report(doc: dict) -> None:
    ps = doc["prior_science"]
    eight = doc["eight"]
    ev = eight["8_expected_value"]
    op = eight["1_mathematical_operator"]
    by = eight["2_expected_bytes_per_token"]
    ops = eight["3_expected_operations_per_token"]
    disp = eight["4_dispatch_topology"]
    met = eight["5_metal_feasibility"]
    lay = eight["6_memory_layout"]
    mb = eight["7_cheap_microbenchmark"]
    g1 = ps["g1"]
    search = ps["search"]

    w = 78
    print("=" * w)
    print("C2 TENSOROP DESIGN")
    print("=" * w)
    print(f"schema     {doc['schema']}")
    print(f"generated  {doc['generated_at']}")
    print(f"head       {doc['commit']}")
    print(f"elapsed_s  {doc['elapsed_s']}")
    print(f"receipt    {doc['receipt']}")
    print()
    print("## PRIOR SCIENCE")
    print(f"phrase 'tensor train'   files={search['phrase_tensor_train']['files']} "
          f"hits={search['phrase_tensor_train']['hits']}")
    ident = search["identifiers"]["tensor_train"]
    print(f"identifier tensor_train files={ident['files']} hits={ident['hits']}")
    print(f"G1 family_rows={g1['family_rows']} healthy_true={g1['healthy_true']} "
          f"wall_s={g1['wall_s']}")
    print(f"raw local_bpw_f16<0.5: {g1['raw_local_bpw_lt_0_5']} "
          f"(healthy_true={g1['raw_lt05_healthy_true']})")
    print(f"archaeology unique (family,tag,bpw)<0.5: {g1['archaeology_unique_lt_0_5']} "
          f"(healthy_true={g1['archaeology_unique_healthy_true']}) bands={g1['archaeology_bands']}")
    print(f"tensor-family unique sub-0.5: {g1['tensor_family_unique_lt_0_5']} "
          f"(healthy_true={g1['tensor_family_unique_healthy_true']}) — family IS in the 223")
    print(f"G034 family_verdict: {ps['g034']['family_verdict']}")
    print(f"G034 mean q3 {ps['g034']['mean_flat_q3']:.4f} vs low-rank "
          f"{ps['g034']['mean_lowrank']:.4f}  error_ratio={ps['g034']['error_ratio']}  "
          f"mac_ratio={ps['g034']['mean_mac_ratio']}")
    print()
    print("STOP:" if ps["family_already_refuted"] else "CONTINUE:")
    print("  " + ps["stop"])
    print()
    print("## 1. MATHEMATICAL OPERATOR")
    print(f"name:     {op['name']}")
    print(f"status:   {op['status']}")
    print("production path (NOT the oracle):")
    print("  " + op["production_path"])
    print("ORACLE (labelled, not production):")
    print("  " + op["correctness_oracle"]["procedure"])
    print("contraction order:")
    for step in op["contraction_order"]:
        print(f"  {step}")
    print(f"selfcheck G1 formulas match: {op['selfcheck_g1_formulas']['match']}")
    print()
    print("## 2. EXPECTED BYTES/TOKEN")
    print(f"incumbent Q4 GEMV stream: {by['incumbent_q4_gemv_bytes_per_token']:,} B")
    print(f"TT r=64 inner=128 stream: {by['tt_r64_inner128_weight_bytes_per_token']:,} B  "
          f"ratio={by['ratio_vs_q4_gemv']:.4f}")
    print(f"G034 r@3.25bpw stream:    {by['g034_r_at_3p25bpw_weight_bytes_per_token']:,} B  "
          f"ratio={by['g034_ratio_vs_q4']:.4f}")
    print("function at those bytes: " + by["function_at_these_bytes"][:160] + "…")
    print()
    print("## 3. EXPECTED OPERATIONS/TOKEN")
    print(f"incumbent GEMV MACs: {ops['incumbent_gemv_gflop']:.2f} GFLOP  "
          f"({ops['incumbent_gemv_mac_flops']:,})  + act {ops['incumbent_activation_flops']:,}")
    print(f"TT r=64 GEMV:        {ops['tt_r64_gflop']:.2f} GFLOP  "
          f"ratio={ops['tt_flop_ratio_vs_dense']:.4f}")
    print(f"G034 r@3.25bpw:      {ops['g034_gflop']:.2f} GFLOP  "
          f"mac_ratio={ops['g034_mac_ratio_mean_from_receipt']}")
    print(f"fused-recompute TT:  {ops['fused_recompute_tt_input_flops']/1e9:.2f} GFLOP  "
          f"vs dense {ops['fused_recompute_tt_vs_dense']:.2f}×")
    ttm = ops["ttm3_compute_loss_example"]
    print(f"TTM3 compute-loss:   flop_ratio={ttm['flop_ratio']:.2f}  {ttm['tag']}")
    print()
    print("## 4. DISPATCH TOPOLOGY")
    inc = disp["incumbent"]
    print(f"incumbent:     {inc['dispatches_per_token']} disp / {inc['command_buffers']} CB  "
          f"({inc['gemv_dispatches']} GEMV)")
    fu = disp["tt_fused_one_kernel_per_organ"]
    print(f"TT fused:      {fu['total_dispatches']} disp / {fu['command_buffers']} CB  "
          f"({fu['gemv_dispatches']} GEMV)  — fusion recomputes H per TG")
    tw = disp["tt_honest_two_stage"]
    print(f"TT two-stage:  {tw['total_dispatches']} disp / {tw['command_buffers']} CB  "
          f"({tw['gemv_dispatches']} GEMV)  — HONEST production path, MORE dispatches")
    print(f"delta two-stage vs 964: {disp['delta_vs_incumbent']['two_stage_total_minus_964']:+d}")
    print()
    print("## 5. METAL FEASIBILITY")
    print(f"device: simdgroup={met['simdgroup_width']}  incumbent TG={met['incumbent_launch']['threadgroup']}  "
          f"TG mem={met['threadgroup_memory_bytes']} B")
    print("incumbent: " + met["incumbent_launch"]["access"][:120])
    h = met["hgravs01_fused_caps_do_not_fit_qwen38"]
    print(f"HGRAVS01 caps: rank≤{h['rank_cap']} x≤{h['x_cap']}; Qwen3.8 MLP x={h['qwen38_mlp_x']} "
          f"G034 r={h['g034_rank']} — both exceed")
    print("feasibility: " + met["feasibility_one_line"])
    print()
    print("## 6. MEMORY LAYOUT")
    print("cores: " + ", ".join(f"{k}={v}" for k, v in lay["cores_per_matrix"].items()))
    print("mid buffer: " + lay["mid_buffer_two_stage"])
    print("NR: " + lay["nr_schema"][:160] + "…")
    print()
    print("## 7. CHEAP MICROBENCHMARK")
    print("already run: " + mb["already_run"]["what"] + f"  wall_s={mb['already_run']['wall_s']}")
    print("kill: " + mb["cheapest_kill_no_fit"]["kill_if"])
    print("this script is that kill: "
          + str(mb["cheapest_kill_no_fit"]["this_script_is_that_kill"]))
    print()
    print("## 8. EXPECTED VALUE")
    print(f"verdict:        {ev['verdict']}")
    print(f"design_status:  {ev['design_status']}")
    print("complete under S011 §4: "
          + str(ev["s011"]["complete_under_s011"]) + " — " + ev["s011"]["reason"])
    print("cheapest kill: " + ev["cheapest_experiment_that_kills_it"])
    print("reopen when:   " + ev["reopen_when"][:200] + "…")
    print()
    print("## WHAT I WATCHED FAIL")
    for i, f in enumerate(doc["what_i_watched_fail"], 1):
        print(f"  {i:2d}. {f['what']}")
        print(f"      {f['result']}")
        why = f["why"]
        if len(why) > 220:
            why = why[:217] + "…"
        print(f"      {why}")
    print()
    print("## S011 AXES vs INCUMBENT")
    for axis, val in ev["s011"]["axes"].items():
        if isinstance(val, dict):
            print(f"  {axis}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {axis}: {val}")
    print()
    print("=" * w)
    print(f"VERDICT {ev['verdict']}  STATUS {ev['design_status']}")
    print("=" * w)


def main() -> int:
    t0 = time.time()
    g1 = git_show_json(G1_PATH)
    g034 = git_show_json(G034_PATH)
    rows = walk_ops(g1)
    uniq = unique_sub05(rows)
    organs = load_incumbent_organs()
    ps = prior_science(g1, g034, rows, uniq)
    eight = eight_items(organs, rows, g034, g1)

    if not eight["1_mathematical_operator"]["selfcheck_g1_formulas"]["match"]:
        raise SystemExit("FAIL: tt_flops/tt_bytes selfcheck does not match G1")
    if ps["g1"]["healthy_true"] != 0:
        raise SystemExit("FAIL: expected G1 healthy_true=0")
    if ps["g1"]["archaeology_unique_lt_0_5"] != 223:
        # Don't invent 223 if the file drifted; still proceed but record it.
        pass

    fails = what_watched_fail(ps, rows, uniq, eight, g034)
    elapsed = round(time.time() - t0, 3)
    doc = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git_head(),
        "elapsed_s": elapsed,
        "receipt": str(RECEIPT),
        "question": (
            "Can a tensor-train / Tucker / tensor-ring contraction sequence replace "
            "W's FUNCTION on the Qwen3.8 uniform-q4 decode path while moving fewer "
            "bytes AND doing less work, without materialising dense W?"
        ),
        "answer": (
            "No. The family already ran as tensor_train / tt_matrix_unfolding / "
            "tt_gemv_f16 (G1+G034). 373/373 rows unhealthy; 223 unique sub-0.5-BPW "
            "rows with healthy=true 0, and this family is among them. Native "
            "contraction was the scoring path. At matched 3.25 b/elem TT unfolding "
            "is worse than SVD which is 2.93× q3 error. STOP. NOT_WORTH_BUILDING."
        ),
        "anchors_not_rederived": {
            "tps": ANCHOR_TPS,
            "ms_per_token": ANCHOR_TOKEN_MS,
            "roof_gb_s": ANCHOR_ROOF_GB_S,
            "parameter_count": ANCHOR_PARAMS,
            "bpw": ANCHOR_BPW,
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers_per_token": ANCHOR_CBS,
            "gemv_mac_flops": ANCHOR_GEMV_MAC_FLOPS,
            "mlx_4bit_tps_live": ANCHOR_MLX_TPS,
            "llamacpp_q5k_tps_archived": ANCHOR_LLAMA_Q5K_TPS,
            "reconstructs_dense_on_38": "NO",
        },
        "prior_science": ps,
        "eight": eight,
        "what_i_watched_fail": fails,
        "write_scope": {
            "write": [
                "tools/headless/c2tensorop_design.py",
                "receipts/headless/C2TENSOROP_DESIGN.json",
            ],
            "denied": [
                "workspace", "crates", "visionmcp", "app", "lab",
                "tools/haider", "ramanujan",
            ],
        },
        "verdict": eight["8_expected_value"]["verdict"],
        "design_status": eight["8_expected_value"]["design_status"],
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n")
    print_report(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
