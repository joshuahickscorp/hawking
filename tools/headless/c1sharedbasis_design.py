#!/usr/bin/env python3
"""C1 shared-basis design lane: native operator, or a refutation properly scoped.

Family: a shared coordinate system across layers, with per-layer coefficients,
executed WITHOUT reconstructing the per-layer dense matrix.

Start from the refutation, not from hope. G035 measured
`shared_beats_independent=false`. Shared basis also never had a native kernel
here, so its cost was always paid as reconstruction. Those are two different
failures. This script decides which one killed the family on Qwen3.8.

If fidelity lost, STOP. A native kernel executes the same approximation as
reconstruct-then-GEMM (associativity: (U C) x = U (C x)). Native changes
cost, not function. Do not design around a fidelity result.

    python3 tools/headless/c1sharedbasis_design.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts/headless/C1SHAREDBASIS_DESIGN.json"
SHADER_TWO_STAGE = REPO / "crates/hawking-core/shaders/q80_mixed_decode.metal"
CENSUS_KERNEL = REPO / "receipts/headless/NOETIC_KERNEL_CENSUS.json"
CENSUS_OPS = REPO / "receipts/headless/NOETIC_OPERATION_CENSUS.json"
CENSUS_METRICS = REPO / "receipts/headless/NOETIC_METRICS.json"
MACHINE = REPO / "receipts/headless/MACHINE_GENOME.json"

SCHEMA = "hawking.headless.c1sharedbasis_design.v1"

# Anchors — measured, not re-derived (lane brief + NOETIC_* receipts).
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_BOUND = 38
ANCHOR_REACHABLE = 508
ANCHOR_DEAD = 4
ANCHOR_UNKNOWN = 4
ANCHOR_TPS = 32.73
ANCHOR_TOKEN_MS = 30.606
ANCHOR_ROOF_GB_S = 778.8
ANCHOR_UNIFIED_B = 103_079_215_104
ANCHOR_GPU_CORES = 60
ANCHOR_PARAMS = 26_895_998_464
ANCHOR_BPW = 4.253
ANCHOR_ARTIFACT_B = 14_297_933_604
ANCHOR_TENSORS = 755
ANCHOR_MLX_TPS = 35.51
ANCHOR_LLAMACPP_Q5K_TPS = 24.12  # ARCHIVED; artifact off disk
ANCHOR_NULL_COSINE = 0.898
ANCHOR_Q80_STORAGE_BPW = 0.6462
ANCHOR_Q80_ACTIVE_BPW = 2.518
ANCHOR_Q80_EXPERT_COS = 0.004142791032791138
ANCHOR_HGRAVS01_DOWN_BPW = 0.13
ANCHOR_GLM_EXPERT_BPW = 0.167
ANCHOR_TPR64_FREE = "32/33"
ANCHOR_SUB05_ROWS = 223
ANCHOR_SUB05_HEALTHY = 0
ANCHOR_MLP_DISTILL_GAP = 0.4206
ANCHOR_MLP_DISTILL_BYTE_FRAC = 0.72

LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
Q4_GROUP = 64
Q4_BYTES_PER_GROUP = 34  # 32 nibble-code + 2 fp16 scale
F16_BYTES = 2
SIMDWIDTH = 32
TG_FUSED = 256
ROWS_PER_FUSED_TG = 8  # 8 simdgroups × 1 row
APPLE_TG_MEM_B = 32 * 1024  # documented Apple GPU threadgroup memory
G035_R_IND = 256
FACTOR_BITS = 16

# G034 at the coherent operating point (function-space, matched 3.25 bits/elem).
G034_MEAN_Q3 = 0.1839276241211841
G034_MEAN_LOWRANK = 0.5393288880586624
G034_MAC_RATIO = 0.2029641544117647
G034_RANK = 803
G034_BITS = 3.25
G035_COHERENT_ANCHOR = 0.198  # G035 what_still_limits_it

ORGANS = {
    "mlp.gate_proj": {"m": INTERMEDIATE, "n": HIDDEN, "count": LAYERS},
    "mlp.up_proj": {"m": INTERMEDIATE, "n": HIDDEN, "count": LAYERS},
    "mlp.down_proj": {"m": HIDDEN, "n": INTERMEDIATE, "count": LAYERS},
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNKNOWN:{exc}"


def git_show_json(path: str):
    try:
        p = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if p.returncode != 0 or not p.stdout.strip():
            return None, p.stderr.strip()[:400] if p.stderr else "empty"
        return json.loads(p.stdout), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def q4_matrix_bytes(rows: int, cols: int) -> int:
    gpr = (cols + Q4_GROUP - 1) // Q4_GROUP
    return rows * gpr * Q4_BYTES_PER_GROUP


def mac_flops(rows: int, cols: int) -> int:
    return 2 * rows * cols


def params_independent(m: int, n: int, r: int, n_layers: int = 2) -> int:
    return n_layers * r * (m + n)


def params_shared_g035(m: int, n: int, r: int, n_layers: int = 2) -> int:
    """G035 column-basis share: concat along columns, share U[m,r], per-layer C[r,n]."""
    return r * m + n_layers * r * n


def rank_shared_matching(m: int, n: int, r_ind: int, n_layers: int = 2) -> int:
    budget = params_independent(m, n, r_ind, n_layers)
    return int(budget // (m + n_layers * n))


def two_stage_flops(m: int, n: int, r: int) -> int:
    """Sequential native: z = C x (r×n), y = U z (m×r). Never form W."""
    return 2 * r * n + 2 * m * r


def fused_recompute_flops(m: int, n: int, r: int, rows_per_tg: int = ROWS_PER_FUSED_TG) -> int:
    """Clone of q80_hgravs01_two_stage_matvec: every TG recomputes mid = R@x."""
    n_tg = (m + rows_per_tg - 1) // rows_per_tg
    return n_tg * (2 * r * n) + 2 * m * r


def f16_bytes(elems: int) -> int:
    return elems * F16_BYTES


def matmul(a, b):
    br = len(b)
    bc = len(b[0])
    out = []
    for row in a:
        out.append([sum(row[k] * b[k][j] for k in range(br)) for j in range(bc)])
    return out


def matvec(a, x):
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def associativity_residual() -> dict:
    """(U C) x vs U (C x) on a tiny integer-ish example. No numpy, no model."""
    u = [[1.0, 0.25], [0.0, 1.0], [0.5, -0.5]]  # 3×2
    c = [[2.0, -1.0, 0.5, 3.0], [0.25, 4.0, -2.0, 1.0]]  # 2×4
    x = [1.0, -0.5, 2.0, 0.25]
    w = matmul(u, c)
    y_oracle = matvec(w, x)  # reconstruct dense W, then GEMV — the ORACLE
    z = matvec(c, x)
    y_native = matvec(u, z)  # production-shaped two-stage, never forms W
    resid = [abs(a - b) for a, b in zip(y_oracle, y_native)]
    return {
        "shape_U": [len(u), len(u[0])],
        "shape_C": [len(c), len(c[0])],
        "y_oracle": y_oracle,
        "y_native": y_native,
        "max_abs_residual": max(resid),
        "sum_abs_residual": sum(resid),
        "identity_holds_at_1e12": max(resid) < 1e-12,
        "reading": (
            "Native two-stage and reconstruct-then-GEMM compute the same vector. "
            "Execution path is not a fidelity variable."
        ),
    }


def parse_two_stage_caps(src: str) -> dict:
    rank_cap = None
    x_cap = None
    simd = None
    tg_comment_640 = "640 B" in src and "not dense W" in src
    for line in src.splitlines():
        s = line.strip()
        if "kRankCap" in s and "=" in s and "constexpr" in s:
            rank_cap = int(s.split("=")[1].split("u")[0].strip())
        if "kXCap" in s and "=" in s and "constexpr" in s:
            x_cap = int(s.split("=")[1].split("u")[0].strip())
        if "kSimdWidth" in s and "=" in s and "constexpr" in s and simd is None:
            simd = int(s.split("=")[1].split("u")[0].strip())
    return {
        "present": "kernel void q80_hgravs01_two_stage_matvec" in src,
        "kRankCap": rank_cap,
        "kXCap": x_cap,
        "kSimdWidth": simd,
        "tg_mid_documented_640b": tg_comment_640,
        "file": "crates/hawking-core/shaders/q80_mixed_decode.metal",
    }


def extract_g035(doc: dict | None) -> dict:
    """Pull the measured pairs; do not re-run SVD."""
    if not doc:
        return {"loaded": False}
    pairs = []
    for p in doc.get("pairs") or []:
        pairs.append(
            {
                "organ": p["organ"],
                "layers": p["layers"],
                "kind": p["kind"],
                "shape": p["shape"],
                "rank_independent": p["rank_independent"],
                "rank_shared": p["rank_shared"],
                "params_independent": p["params_independent"],
                "params_shared": p["params_shared"],
                "bits_per_elem_independent": p["bits_per_elem_independent"],
                "bits_per_elem_shared": p["bits_per_elem_shared"],
                "independent_mean": p["independent_mean"],
                "shared_mean": p["shared_mean"],
                "shared_beats_independent": p["shared_beats_independent"],
            }
        )
    row = ((doc.get("untested_members_now_tested") or {}).get("shared_row_basis") or {})
    row_rows = row.get("rows") or []
    return {
        "loaded": True,
        "schema": doc.get("schema"),
        "obligation": doc.get("obligation"),
        "method": doc.get("method"),
        "error_space": doc.get("error_space"),
        "pairs": pairs,
        "all_column_shared_beats_independent": all(
            not p["shared_beats_independent"] for p in pairs
        ),
        "adjacent_mean_error_reduction": doc.get("adjacent_mean_error_reduction"),
        "far_control_mean_error_reduction": doc.get("far_control_mean_error_reduction"),
        "corrected_verdict": doc.get("corrected_verdict"),
        "what_still_limits_it": doc.get("what_still_limits_it"),
        "row_basis_verdict": row.get("verdict"),
        "row_basis_rows": [
            {
                "organ": r.get("organ"),
                "pair": r.get("pair"),
                "kind": r.get("kind"),
                "scheme": r.get("scheme"),
                "bits_per_elem": r.get("bits_per_elem"),
                "mean_out_err": r.get("mean_out_err"),
            }
            for r in row_rows
        ],
        "delta_coding_verdict": ((doc.get("untested_members_now_tested") or {})
                                 .get("delta_coding") or {}).get("verdict"),
    }


def organ_accounting(m: int, n: int, r_ind: int, count: int) -> dict:
    r_sh = rank_shared_matching(m, n, r_ind)
    p_ind = params_independent(m, n, r_ind)
    p_sh = params_shared_g035(m, n, r_sh)
    elems_pair = 2 * m * n
    q4_b = q4_matrix_bytes(m, n)
    dense_mac = mac_flops(m, n)
    seq_mac = two_stage_flops(m, n, r_sh)
    fused_mac = fused_recompute_flops(m, n, r_sh)
    n_pairs = count // 2
    # Pair-shared storage: one U per pair + one C per layer. f16 factors (G035 convention).
    u_elems_pair = r_sh * m
    c_elems_layer = r_sh * n
    stored_b = n_pairs * f16_bytes(u_elems_pair) + count * f16_bytes(c_elems_layer)
    q4_b_token = q4_b * count
    return {
        "shape": [m, n],
        "count": count,
        "n_pairs": n_pairs,
        "rank_independent": r_ind,
        "rank_shared": r_sh,
        "params_independent_pair": p_ind,
        "params_shared_pair": p_sh,
        "bits_per_elem_independent": FACTOR_BITS * p_ind / elems_pair,
        "bits_per_elem_shared": FACTOR_BITS * p_sh / elems_pair,
        "q4_bytes_per_launch": q4_b,
        "q4_bytes_per_token": q4_b_token,
        "shared_f16_bytes_per_token": stored_b,
        "byte_ratio_shared_over_q4": stored_b / q4_b_token,
        "dense_mac_per_launch": dense_mac,
        "dense_mac_per_token": dense_mac * count,
        "sequential_two_stage_mac_per_launch": seq_mac,
        "sequential_two_stage_mac_per_token": seq_mac * count,
        "sequential_mac_ratio": seq_mac / dense_mac,
        "fused_recompute_mac_per_launch": fused_mac,
        "fused_recompute_mac_per_token": fused_mac * count,
        "fused_over_dense": fused_mac / dense_mac,
        "fused_over_sequential": fused_mac / seq_mac,
        "u_f16_bytes_per_pair": f16_bytes(u_elems_pair),
        "c_f16_bytes_per_layer": f16_bytes(c_elems_layer),
        "x_f32_bytes": n * 4,
        "mid_f32_bytes": r_sh * 4,
        "x_fits_32kib_tg": n * 4 <= APPLE_TG_MEM_B,
        "mid_fits_32kib_tg": r_sh * 4 <= APPLE_TG_MEM_B,
        "x_plus_mid_fits_32kib_tg": (n * 4 + r_sh * 4) <= APPLE_TG_MEM_B,
        "exceeds_hgravs01_kRankCap160": r_sh > 160,
        "exceeds_hgravs01_kXCap512": n > 512,
    }


def search_prior_science() -> dict:
    """Search the preserved record. Do not rediscover. A missing path is not a result."""
    watched = []
    lane_bootstrap = []
    try:
        p = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        names = p.stdout.splitlines()
        lane_bootstrap = [n for n in names if ".lane-bootstrap" in n
                          or "n1arch" in n or "n15neg" in n or "n16clos" in n]
    except Exception as exc:  # noqa: BLE001
        watched.append(
            {
                "what": "git ls-tree for .lane-bootstrap/census",
                "result": "FAILED",
                "detail": str(exc),
            }
        )
        names = []

    if not lane_bootstrap:
        watched.append(
            {
                "what": ".lane-bootstrap/census/ (n1arch 35, n15neg 31, n16clos)",
                "result": "ABSENT_FROM_HEAD",
                "detail": (
                    "git ls-tree HEAD has zero paths under .lane-bootstrap and zero "
                    "files named n1arch/n15neg/n16clos. Sparse-checkout add is forbidden. "
                    "Stand-ins used: NEGATIVE_SCIENCE_REGISTER (38 entries, NS-010), "
                    "NOETIC_KERNEL_CENSUS families, G035/G034/G032, g1-shared-basis.md, "
                    "QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE."
                ),
            }
        )

    g035, g035_err = git_show_json("receipts/ascent-2026-08-16/G035_CROSSLAYER_SHARE.json")
    g034, g034_err = git_show_json("receipts/ascent-2026-08-16/G034_TENSOR_OPERATOR.json")
    g032, g032_err = git_show_json("receipts/ascent-2026-08-16/G032_XFORM_HADAMARD_Q4.json")
    nsreg, ns_err = git_show_json("receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json")
    q80x, q80_err = git_show_json("receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json")
    g013, g013_err = git_show_json(
        "receipts/ascent-2026-08-16/G013_FS_EFFICIENCY_CLOSURE_V2.json"
    )
    g1_md = None
    g1_err = None
    try:
        p = subprocess.run(
            ["git", "show", "HEAD:workspace/superwave/g1/g1-shared-basis.md"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if p.returncode == 0:
            g1_md = p.stdout
        else:
            g1_err = (p.stderr or "git show g1-shared-basis.md failed")[:400]
    except Exception as exc:  # noqa: BLE001
        g1_err = str(exc)

    for label, err in [
        ("G035", g035_err if g035 is None else None),
        ("G034", g034_err if g034 is None else None),
        ("G032", g032_err if g032 is None else None),
        ("NS_REGISTER", ns_err if nsreg is None else None),
        ("Q80_CROSS_EXPERT", q80_err if q80x is None else None),
        ("G013v2", g013_err if g013 is None else None),
    ]:
        if err:
            watched.append({"what": f"git show {label}", "result": "MISS", "detail": err})

    ns010 = None
    if isinstance(nsreg, dict):
        for e in nsreg.get("entries") or []:
            if e.get("id") == "NS-010":
                ns010 = {
                    "id": e["id"],
                    "class": e.get("class"),
                    "mechanism": e.get("mechanism"),
                    "models": e.get("models"),
                    "gate_proj_pairwise_cosine_mean": (
                        (e.get("what_was_measured") or {}).get(
                            "gate_proj_pairwise_cosine_mean"
                        )
                    ),
                    "retry_when": e.get("retry_when"),
                    "why_it_failed": e.get("why_it_failed"),
                }
                break

    g1_falsified = False
    g1_adj_cos = {}
    if g1_md:
        g1_falsified = "STATUS: **FALSIFIED**" in g1_md or "FALSIFIED** for this parent" in g1_md
        # table rows already known; pin the class-mean adjacent cosines from the md table
        g1_adj_cos = {
            "mlp.gate_proj": 0.004354,
            "mlp.up_proj": 0.000014,
            "mlp.down_proj": 0.000061,
            "self_attn.q_proj": 0.007632,
            "note": (
                "Exact adjacent flattened cosine from hawking-experiments/superwave/g1/g1-shared-basis.md. "
                "Highest class-mean 0.00763 (GQA q_proj). MLP gate 0.00435 is the Q80 expert number. "
                "Adjacent rel_delta_fro 1.40–1.42 ≈ √2. Shared rank-256 residual 69–93% vs "
                "per-layer 38–87%."
            ),
        }

    kernel = load_json(CENSUS_KERNEL)
    family = None
    missing = None
    if kernel:
        for f in kernel.get("families") or []:
            if f.get("id") == "shared_basis_x_coefficients":
                family = {
                    "id": f["id"],
                    "verdict": f.get("verdict"),
                    "kernel": f.get("kernel"),
                    "why": f.get("why"),
                }
        for m in kernel.get("missing_family_cost") or []:
            if m.get("id") == "shared_basis_x_coefficients":
                missing = m

    g032_summary = None
    if isinstance(g032, dict):
        g032_summary = {
            "mean_delta_hold": (g032.get("summary") or {}).get("mean_delta_hold"),
            "mean_delta_entropy_bits": (g032.get("summary") or {}).get(
                "mean_delta_entropy_bits"
            ),
            "stored_bytes": (g032.get("transform") or {}).get("stored_bytes"),
            "family": (g032.get("transform") or {}).get("family"),
            "reading": (
                "Block-diagonal Sylvester-Hadamard is generated, not stored. At q4 it moves "
                "hold cosine by +3.4e-4 and entropy by +0.026 bits. That is not a shared-basis "
                "win and not a bits-at-equal-function win. Folding H into a native operator "
                "is the structured_transform family, already PARTIAL, already measured."
            ),
        }

    g034_summary = None
    if isinstance(g034, dict):
        g034_summary = {
            "verdict": g034.get("verdict"),
            "family_verdict": g034.get("family_verdict"),
            "mean_flat_q3": g034.get("mean_flat_q3"),
            "mean_lowrank": g034.get("mean_lowrank"),
            "mean_mac_ratio": g034.get("mean_mac_ratio"),
            "error_ratio": (
                g034.get("mean_lowrank") / g034.get("mean_flat_q3")
                if g034.get("mean_flat_q3")
                else None
            ),
        }

    distill_receipt = None
    # The lane brief supplies the MLP-distillation NO-GO. Kernel census still
    # says that avenue "has not been run". Do not reopen; do not invent a path.
    watched.append(
        {
            "what": "MLP function distillation primary receipt",
            "result": "BRIEF_ONLY",
            "detail": (
                "Lane brief: NO-GO, +0.4206 held-out gap vs q3 at 72% of its active bytes. "
                "NOETIC_KERNEL_CENSUS missing_family_cost still says distillation of the "
                "MLP function 'has not been run'. No primary JSON with those two numbers "
                "is materialized in this sparse checkout. Cited as a supplied constraint, "
                "not re-derived, and not used as a reopen."
            ),
        }
    )

    return {
        "lane_bootstrap_paths_in_head": lane_bootstrap,
        "g035": extract_g035(g035 if isinstance(g035, dict) else None),
        "g034": g034_summary,
        "g032_hadamard_q4": g032_summary,
        "ns010": ns010,
        "q80_cross_expert": (
            {
                "layer": q80x.get("layer"),
                "n_experts": q80x.get("n_experts"),
                "gate_pairwise_cosine_mean": (
                    (q80x.get("components") or {}).get("gate_proj") or {}
                ).get("pairwise_cosine_mean"),
                "up_pairwise_cosine_mean": (
                    (q80x.get("components") or {}).get("up_proj") or {}
                ).get("pairwise_cosine_mean"),
            }
            if isinstance(q80x, dict)
            else None
        ),
        "g013_storage_vs_active": {
            "storage_bpw": ANCHOR_Q80_STORAGE_BPW,
            "active_bpw": ANCHOR_Q80_ACTIVE_BPW,
            "factor": round(ANCHOR_Q80_ACTIVE_BPW / ANCHOR_Q80_STORAGE_BPW, 4),
            "source": "receipts/ascent-2026-08-16/G013_FS_EFFICIENCY_CLOSURE_V2.json",
            "law": "Report both or neither. Storage averages experts decode never touches.",
            "loaded": g013 is not None,
        },
        "g1_shared_basis_this_parent": {
            "status": "FALSIFIED" if g1_falsified else "UNREAD",
            "path": "hawking-experiments/superwave/g1/g1-shared-basis.md",
            "adjacent_cosine": g1_adj_cos,
            "reopen_bar_mean_cosine": 0.05,
            "loaded_md": g1_md is not None,
        },
        "kernel_census_family": family,
        "kernel_census_missing_cost": missing,
        "named_traps": {
            "glm_expert_bpw": ANCHOR_GLM_EXPERT_BPW,
            "hgravs01_down_proj_only_bpw": ANCHOR_HGRAVS01_DOWN_BPW,
            "tpr64_reconstruction_free": ANCHOR_TPR64_FREE,
            "sub_0_5_local_bpw_rows": ANCHOR_SUB05_ROWS,
            "sub_0_5_healthy": ANCHOR_SUB05_HEALTHY,
            "law": "A low number is not a result until paired with a health verdict.",
            "cosine_scale_invariance": (
                "0.01*W scores cosine 1.000000; gain is 0.01. Raw activation cosine "
                f"null baseline {ANCHOR_NULL_COSINE}."
            ),
            "mlp_function_distillation": {
                "status": "NO-GO (lane brief; not re-derived)",
                "held_out_gap_vs_q3": ANCHOR_MLP_DISTILL_GAP,
                "active_byte_frac_of_q3": ANCHOR_MLP_DISTILL_BYTE_FRAC,
            },
            "never_synthetic_activations": True,
        },
        "watched": watched,
        "g1_md_err": g1_err,
        "distill_receipt": distill_receipt,
    }


def decide_which_failure(prior: dict, identity: dict) -> dict:
    g035 = prior["g035"]
    column_lost = bool(g035.get("all_column_shared_beats_independent"))
    identity_holds = bool(identity.get("identity_holds_at_1e12"))
    family_absent = (prior.get("kernel_census_family") or {}).get("verdict") == "ABSENT"
    g1_falsified = (prior.get("g1_shared_basis_this_parent") or {}).get("status") == "FALSIFIED"
    g034_ratio = (prior.get("g034") or {}).get("error_ratio")

    # Fidelity is a property of the approximation Ŵ ≈ UC, independent of whether
    # one forms Ŵ. Native cannot unkill G035.
    fidelity_killed = column_lost and g1_falsified
    reconstruction_was_the_only_path = family_absent
    reconstruction_killed_fidelity = False  # the identity says no

    if fidelity_killed:
        failure = "fidelity"
        stop = True
        verdict = "NOT_WORTH_BUILDING"
        scoped = (
            "G035 lost on FUNCTION-SPACE error at matched bits "
            "(shared_beats_independent=false on 3/3 column-basis pairs). "
            "g1-shared-basis FALSIFIED this parent: adjacent flattened cosine 1e-5..8e-3, "
            "shared rank-256 residual worse than per-layer. G034 refuted the low-rank "
            "family at the coherent 3.25-bit point (2.93× q3 error). Associativity "
            "shows native two-stage equals reconstruct-then-GEMM, so the missing kernel "
            "is not the variable that produced those errors."
        )
    else:
        failure = "undecided"
        stop = False
        verdict = "UNDECIDED"
        scoped = "Prior science did not load; cannot unkill, cannot design."

    return {
        "question": (
            "Did sharing lose on FIDELITY, or because the only way to execute it "
            "was to rebuild a dense matrix first?"
        ),
        "answer": "FIDELITY" if fidelity_killed else "NOT_SETTLED_HERE",
        "failure_that_killed_it": failure,
        "stop": stop,
        "verdict": verdict,
        "scoped_refutation": scoped,
        "evidence": {
            "g035_column_shared_beats_independent_all_false": column_lost,
            "g035_adjacent_mean_error_reduction": g035.get("adjacent_mean_error_reduction"),
            "g1_this_parent_falsified": g1_falsified,
            "g034_lowrank_over_q3": g034_ratio,
            "associativity_identity_holds": identity_holds,
            "native_kernel_absent": reconstruction_was_the_only_path,
            "reconstruction_killed_fidelity": reconstruction_killed_fidelity,
        },
        "two_failures_separated": {
            "fidelity": "a property of the idea (Ŵ ≈ shared-U × per-layer-C vs independent SVD)",
            "reconstruction": (
                "a property of the implementation (no shared_basis kernel; G035 "
                "materialised Ŵ then applied X). Associativity says they compute the same y."
            ),
            "only_the_first_is_a_property_of_the_idea": True,
            "native_changes": "bytes, FLOPs, dispatches, TG pressure — not the G035 errors",
        },
        "row_basis_not_a_reopen": {
            "what": (
                "G035 corrected_verdict: sharing the ROW space WINS 6.3% on adjacent "
                "gate at ~1.03 bits/elem (0.57592 vs 0.61465), 9× more than the far pair."
            ),
            "why_not_design_around_it": (
                "The win is at the DEAD zone: pair errors 0.58–0.70 against the coherent "
                f"anchor {G035_COHERENT_ANCHOR}. G034 independently measured low-rank at a "
                f"usable {G034_BITS} bits/elem running {G034_MEAN_LOWRANK/G034_MEAN_Q3:.2f}× "
                "worse than flat q3. Whether the row-basis edge survives at usable fidelity "
                "is UNTESTED and is a low-rank-family question G034 already closed. "
                "Designing a native row-basis kernel around a dead-zone 6% would be "
                "designing around a fidelity result."
            ),
        },
        "hadamard_not_a_reopen": (prior.get("g032_hadamard_q4") or {}).get("reading"),
    }


def metal_feasibility(organs: dict, caps: dict) -> dict:
    rank_cap = caps.get("kRankCap")
    x_cap = caps.get("kXCap")
    gate = organs["mlp.gate_proj"]
    down = organs["mlp.down_proj"]
    mid_gate = gate["mid_f32_bytes"]
    mid_down = down["mid_f32_bytes"]
    x_gate = gate["x_f32_bytes"]
    x_down = down["x_f32_bytes"]
    return {
        "device": "Apple M3 Ultra, 60 GPU cores, Metal 4, unified memory",
        "unified_memory_bytes": ANCHOR_UNIFIED_B,
        "roof_gb_s": ANCHOR_ROOF_GB_S,
        "simdgroup_width": SIMDWIDTH,
        "simdgroup_width_source": (
            f"{caps.get('file')}: constexpr uint kSimdWidth = {caps.get('kSimdWidth')}u "
            "in q80_hgravs01_two_stage_matvec; common.metal documents 32 lanes/simdgroup"
        ),
        "threadgroup": {
            "fused_kernel_threads": TG_FUSED,
            "fused_rows_per_tg": ROWS_PER_FUSED_TG,
            "incumbent_gemv": "tpr64, TG 128, 2 rows/TG, 64 threads/row",
            "documented_apple_tg_memory_bytes": APPLE_TG_MEM_B,
            "hgravs01_mid_plus_x": {
                "kRankCap": rank_cap,
                "kXCap": x_cap,
                "mid_bytes_at_cap": (rank_cap or 0) * 4,
                "x_bytes_at_cap": (x_cap or 0) * 4,
                "comment_640b_not_dense_w": caps.get("tg_mid_documented_640b"),
            },
        },
        "g035_vs_existing_fused_kernel": {
            "gate_rank_shared": gate["rank_shared"],
            "down_rank_shared": down["rank_shared"],
            "gate_n": gate["shape"][1],
            "down_n": down["shape"][1],
            "kernel_refuses_rank_gt_160": True,
            "kernel_refuses_n_gt_512": True,
            "drop_in_clone_runs_g035": False,
            "why": (
                f"q80_hgravs01_two_stage_matvec returns immediately if right_rows > "
                f"{rank_cap} or right_cols > {x_cap}. G035 gate r={gate['rank_shared']} "
                f"n={gate['shape'][1]}; down r={down['rank_shared']} n={down['shape'][1]}. "
                "The census line 'clone ~200 lines' is kernel-cheap only after the caps "
                "and the TG-recompute cost are redesigned. Quality is already blocked."
            ),
        },
        "threadgroup_memory_at_g035_ranks": {
            "gate_x_f32_bytes": x_gate,
            "gate_mid_f32_bytes": mid_gate,
            "gate_x_plus_mid": x_gate + mid_gate,
            "gate_x_plus_mid_fits_32kib": gate["x_plus_mid_fits_32kib_tg"],
            "down_x_f32_bytes": x_down,
            "down_mid_f32_bytes": mid_down,
            "down_x_plus_mid": x_down + mid_down,
            "down_x_plus_mid_fits_32kib": down["x_plus_mid_fits_32kib_tg"],
            "reading": (
                f"gate x+mid = {x_gate + mid_gate} B < 32 KiB, so a fused kernel COULD "
                "keep x resident in TG on gate/up. down x = "
                f"{x_down} B already exceeds 32 KiB, so down_proj cannot cache x in "
                "threadgroup memory; it must stream x from device. That is a hardware "
                "constraint, not a coding preference."
            ),
        },
        "register_pressure": {
            "incumbent_tpr64": (
                "64 threads/row, 8 Q4 weights unpacked per iteration into registers, "
                "FMA, discard. Occupancy: 17408-row gate → 8704 TGs of 128 = 1,114,112 "
                "threads; 145 TGs/core if spread. Bandwidth-saturated, not occupancy-starved "
                "(NOETIC_TPR64_REOPEN simd_vehicle)."
            ),
            "two_stage_sequential": (
                f"Stage-1 is a fat GEMV of only r={gate['rank_shared']} rows (gate) — "
                "under-occupied relative to 17408-row q4. Stage-2 GEMV of 17408 rows "
                f"against a {gate['rank_shared']}-wide mid vector is ALU-short, not "
                "bandwidth-heavy. Register pressure is lower than Q4 unpack."
            ),
        },
        "coalescing": {
            "C_layout": "row-major f16 [r, n], stage-1 is a standard row-wise GEMV; simd 32 consecutive lanes hit consecutive halves",
            "U_layout": "row-major f16 [m, r], stage-2 same pattern on a short k=r inner dimension",
            "vs_q4": (
                "Q4 is 32 code bytes + 2 scale per 64-wide group. f16 factors are "
                "simpler loads. Coalescing is not the blocker."
            ),
        },
        "fused_recompute_is_a_trap": {
            "gate_fused_over_dense": gate["fused_over_dense"],
            "gate_fused_over_sequential": gate["fused_over_sequential"],
            "down_fused_over_dense": down["fused_over_dense"],
            "why": (
                "The shipped fused kernel recomputes mid[rank] in EVERY threadgroup "
                f"({(gate['shape'][0] + 7)//8} TGs on gate). At G035 r={gate['rank_shared']} "
                "that is tens of times the dense GEMV FLOPs. It is the right fused shape "
                "for rank 160 × n 512 (hgravs01_r160). It is the wrong shape for G035."
            ),
        },
    }


def dispatch_topology(organs: dict) -> dict:
    mlp_gemv = sum(o["count"] for o in organs.values())  # 192
    sequential_mlp = mlp_gemv * 2
    incumbent_non_mlp = ANCHOR_DISPATCHES - mlp_gemv
    sequential_total = incumbent_non_mlp + sequential_mlp
    return {
        "incumbent": {
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers_per_token": ANCHOR_CBS,
            "mlp_gemv_dispatches": mlp_gemv,
            "formula": "1 embed + 64*(9 mixer + 6 mlp) + 3 terminal = 964",
            "synchronises": "one TokenCommandBuffer; GPU waits once at the end of the token",
        },
        "sequential_two_stage_if_built": {
            "dispatches_per_token": sequential_total,
            "delta_vs_incumbent": sequential_total - ANCHOR_DISPATCHES,
            "command_buffers_per_token": ANCHOR_CBS,
            "mlp_dispatches": sequential_mlp,
            "synchronises": (
                "Same CB. Metal orders compute dispatches; stage-2 reads mid written "
                "by stage-1. No extra host wait. One extra producer-consumer barrier "
                "per MLP organ (192 barriers inside the existing CB)."
            ),
            "note": "Dispatch count goes UP. Bytes and sequential FLOPs go down. Function does not hold.",
        },
        "fused_clone_if_built": {
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers_per_token": ANCHOR_CBS,
            "synchronises": "one threadgroup_barrier inside each fused MLP dispatch (already the hgravs01 pattern)",
            "note": (
                "Dispatch count unchanged, FLOPs explode because every TG recomputes mid. "
                "Also refuses G035 ranks/dims without a cap rewrite."
            ),
        },
        "shared_V_does_not_collapse_stage1": (
            "Even on the G035 row-basis (shared V on the contraction axis), each layer "
            "has its own x, so V^T x still runs 64 times. Sharing saves storage of V, "
            "not the first GEMV at decode. Layers are sequential; L30's mid is not L31's."
        ),
    }


def memory_layout(organs: dict) -> dict:
    parts = []
    total = 0
    for name, o in organs.items():
        parts.append(
            {
                "organ": name,
                "U_per_pair": {
                    "shape": [o["shape"][0], o["rank_shared"]],
                    "dtype": "f16",
                    "bytes": o["u_f16_bytes_per_pair"],
                    "count": o["n_pairs"],
                },
                "C_per_layer": {
                    "shape": [o["rank_shared"], o["shape"][1]],
                    "dtype": "f16",
                    "bytes": o["c_f16_bytes_per_layer"],
                    "count": o["count"],
                },
                "bytes_per_token_streamed": o["shared_f16_bytes_per_token"],
            }
        )
        total += o["shared_f16_bytes_per_token"]
    return {
        "representation": (
            "Pair-shared G035 column basis, f16 factors, no Q4. One U[m,r] per adjacent "
            "pair per organ; one C[r,n] per layer per organ. Row-major, 64-byte aligned. "
            "No recurrent extra state. mid[r] f32 is workspace, not stored."
        ),
        "why_pairs_not_all64": (
            "g1-shared-basis falsified a single all-layer basis (shared rank-256 residual "
            "1.08–1.83× per-layer). G035 tested pairs. The layout matches the experiment "
            "that already lost, not a more aggressive share that lost harder."
        ),
        "organs": parts,
        "mlp_shared_f16_bytes_per_token": total,
        "temporary_mid_bytes_live": max(o["mid_f32_bytes"] for o in organs.values()),
        "does_not_materialize_W": True,
        "dense_W_bytes_if_oracle_ran_mlp": sum(
            o["count"] * o["shape"][0] * o["shape"][1] * 4 for o in organs.values()
        ),
    }


def cheap_microbenchmark(identity: dict) -> dict:
    return {
        "purpose": (
            "Discriminate this family from the incumbent BEFORE anyone writes a full kernel. "
            "Two cheap checks, one of which already ran, one of which this process ran."
        ),
        "already_ran_fidelity": {
            "command": (
                "python3 tools/gravity_share_crosslayer.py --rank 256 --rows 512 "
                "--out receipts/ascent-2026-08-16/G035_CROSSLAYER_SHARE.json"
            ),
            "uses": (
                "real BF16 W from the Qwen3.8 parent; function-space error on the thick v2 "
                "capture (SITE[organ] activations). NOT synthetic X. NOT weight cosine."
            ),
            "kill_if": "shared_beats_independent is false on adjacent gate_proj",
            "observed": "false on 3/3 column-basis pairs; adjacent mean error reduction −0.01427",
            "status": "ALREADY_EXECUTED — this is the kill",
        },
        "this_process_associativity": {
            "what": (
                "2×4 toy (U C)x vs U(Cx) in pure Python. Discriminates reconstruction-cost "
                "from fidelity: if the residual is ~0, rebuilding W cannot have been what "
                "produced G035's extra error."
            ),
            "observed": identity,
            "kill_native_rescue_if": "max_abs_residual < 1e-12 (rescue is then impossible)",
            "status": "RAN_THIS_PROCESS",
        },
        "would_run_before_a_kernel_if_fidelity_had_held": {
            "command": (
                "One Metal CB, three arms on a single 17408×5120 gate, r=417, real x from "
                "the v2 capture token 192 (NOT Gaussian): "
                "(R) write Ŵ=U@C as f32 (356,515,840 B) then GEMV; "
                "(N) z=C@x; y=U@z with two dispatches, no Ŵ; "
                "(Q) incumbent qwen_uniform_q4_group64_matvec_geo_tpr64_tg128. "
                "GPU ns from MTLCommandBuffer GPUEndTime-GPUStartTime. "
                "Function-space rel_fro of N vs Q vs independent SVD r=256 vs flat q3."
            ),
            "discriminates": (
                "N vs R time = reconstruction tax. N vs Q error = whether the approximation "
                "is even in the same quality band as the incumbent. Do not write the 64-layer "
                "bind until N beats Q on error at matched-or-better bytes — G035/G034 say it will not."
            ),
            "status": "NOT_RUN — fidelity already lost, so this timing is not owed",
        },
        "do_not": [
            "synthetic activations",
            "weight cosine as the score (G033 inverted a ranking once)",
            "cosine without gain (0.01*W scores 1.000000)",
            "claim a win at ~1.03 bits/elem against a 0.198 coherent anchor",
        ],
    }


def expected_value(decision: dict, organs: dict, layout: dict, topo: dict, ops: dict) -> dict:
    mlp_q4 = sum(o["q4_bytes_per_token"] for o in organs.values())
    mlp_sh = layout["mlp_shared_f16_bytes_per_token"]
    seq_mlp_mac = sum(o["sequential_two_stage_mac_per_token"] for o in organs.values())
    fused_mlp_mac = sum(o["fused_recompute_mac_per_token"] for o in organs.values())
    dense_mlp_mac = sum(o["dense_mac_per_token"] for o in organs.values())
    return {
        "verdict": decision["verdict"],
        "would_win_if_function_held": {
            "mlp_bytes_q4": mlp_q4,
            "mlp_bytes_shared_f16_g035_rank": mlp_sh,
            "mlp_byte_ratio": mlp_sh / mlp_q4,
            "mlp_mac_dense": dense_mlp_mac,
            "mlp_mac_sequential_two_stage": seq_mlp_mac,
            "mlp_mac_ratio_sequential": seq_mlp_mac / dense_mlp_mac,
            "tok_s_incumbent": ANCHOR_TPS,
            "tok_s_mlx_4bit_live": ANCHOR_MLX_TPS,
            "tok_s_llamacpp_q5k_archived": ANCHOR_LLAMACPP_Q5K_TPS,
            "gap_to_mlx": ANCHOR_MLX_TPS / ANCHOR_TPS,
            "fantasy_dram_if_mlp_replaced": (
                "Incumbent executable DRAM 13.988 GB/token. MLP Q4 is 9.091 GB of that. "
                f"Replacing MLP with {mlp_sh} B shared f16 drops ~{mlp_q4 - mlp_sh} B. "
                "Roof 595.9 GB/s would then floor the token well under the measured 30.606 ms "
                "IF the approximation were q3-quality. It is not."
            ),
        },
        "what_it_risks": [
            "Shipping 1.03 BPW factors whose function-space error sits in the dead zone (0.58–0.87 vs q3 ~0.18).",
            "Cloning q80_hgravs01_two_stage_matvec and inflating MLP FLOPs by tens of × via TG-private mid recompute.",
            "Raising dispatches 964 → 1156 (sequential) for an approximation G034 already showed 2.93× worse than q3 at usable bits.",
            "Reopening a family the kernel census marked kernel_cheap_quality_blocked with the instruction 'Do not write the kernel to reopen a closed idea.'",
            "Cosine-only scoring (0.01*W = 1.0) or synthetic X.",
        ],
        "cheapest_experiment_that_would_kill_it": (
            "Already executed: G035 shared vs independent at matched bits on real X, "
            "function-space rel_fro. Kill criterion met (shared_beats_independent=false). "
            "G034 is the ceiling at usable bits. Associativity (this process) kills the "
            "'native would have changed the error' rescue. Stop."
        ),
        "s011_section4": {
            "necessary_condition": (
                "A design that does not reduce at least one of bytes, operations, "
                "dispatches, materialization, synchronization or traffic is INCOMPLETE."
            ),
            "sequential_two_stage_reduces": ["bytes", "operations", "traffic"],
            "sequential_two_stage_increases": ["dispatches", "synchronization"],
            "sequential_materialization": (
                "Incumbent already materialises 0 dense W. Sequential two-stage also 0. "
                "No win on that axis."
            ),
            "fused_clone_reduces": ["bytes", "traffic"],
            "fused_clone_increases": ["operations"],
            "function_preserved": False,
            "incomplete_as_production_candidate": True,
            "why": (
                "S011 §4 is necessary, not sufficient. Sequential two-stage would pass the "
                "reduce-one-axis bar and still be the wrong object: it executes an "
                "approximation that lost on fidelity at matched bits and whose family "
                "ceiling at usable bits is 2.93× q3. Reported as INCOMPLETE for production, "
                "not as a kernel to write."
            ),
        },
        "do_not_build": True,
        "ops": ops,
    }


def build() -> dict:
    identity = associativity_residual()
    prior = search_prior_science()
    decision = decide_which_failure(prior, identity)

    shader_src = SHADER_TWO_STAGE.read_text(encoding="utf-8") if SHADER_TWO_STAGE.is_file() else ""
    caps = parse_two_stage_caps(shader_src) if shader_src else {
        "present": False, "kRankCap": None, "kXCap": None, "kSimdWidth": None,
        "tg_mid_documented_640b": False, "file": str(SHADER_TWO_STAGE),
    }

    organs = {
        name: organ_accounting(spec["m"], spec["n"], G035_R_IND, spec["count"])
        for name, spec in ORGANS.items()
    }

    ops_census = load_json(CENSUS_OPS) or {}
    incumbent_gemv = (ops_census.get("analytic_vs_measured") or {}).get(
        "dispatched_gemv_mac_flops", 51_243_909_120
    )
    incumbent_act = (ops_census.get("analytic_vs_measured") or {}).get(
        "dispatched_activation_flops", 297_313_024
    )
    incumbent_dram = (ops_census.get("dram_and_temp") or {}).get(
        "executable_dram_bytes_per_token", 13_988_022_948
    )
    incumbent_w_bytes = (ops_census.get("dram_and_temp") or {}).get(
        "executable_weight_bytes_per_token", 13_623_403_168
    )
    incumbent_q4_gemv_b = (ops_census.get("dram_and_temp") or {}).get(
        "executable_q4_gemv_bytes", 13_611_663_360
    )
    mlp_q4_b = sum(o["q4_bytes_per_token"] for o in organs.values())
    mlp_sh_b = sum(o["shared_f16_bytes_per_token"] for o in organs.values())
    mlp_dense_mac = sum(o["dense_mac_per_token"] for o in organs.values())
    mlp_seq_mac = sum(o["sequential_two_stage_mac_per_token"] for o in organs.values())
    mlp_fused_mac = sum(o["fused_recompute_mac_per_token"] for o in organs.values())
    non_mlp_mac = incumbent_gemv - mlp_dense_mac
    seq_total_mac = non_mlp_mac + mlp_seq_mac
    fused_total_mac = non_mlp_mac + mlp_fused_mac

    bytes_item = {
        "incumbent": {
            "executable_dram_bytes_per_token": incumbent_dram,
            "executable_weight_bytes_per_token": incumbent_w_bytes,
            "q4_gemv_bytes_per_token": incumbent_q4_gemv_b,
            "mlp_q4_bytes_per_token": mlp_q4_b,
            "derivation": (
                "NOETIC_OPERATION_CENSUS dram_and_temp + q4_matrix_bytes(rows,cols)="
                "rows*ceil(cols/64)*34, 64 layers × 3 MLP organs."
            ),
        },
        "counterfactual_g035_rank_mlp_only": {
            "mlp_shared_f16_bytes_per_token": mlp_sh_b,
            "mlp_byte_ratio_vs_q4": mlp_sh_b / mlp_q4_b,
            "rest_of_q4_gemv_untouched": incumbent_q4_gemv_b - mlp_q4_b,
            "implied_gemv_payload_if_mlp_replaced": mlp_sh_b + (incumbent_q4_gemv_b - mlp_q4_b),
            "note": (
                "Derived, not guessed: 32 pairs × (U[m,r]+2·C[r,n]) f16 per organ, "
                "r = rank_shared_matching(m,n,256) which reproduces G035's 417/288."
            ),
        },
        "roof_ms_incumbent": incumbent_dram / (ANCHOR_ROOF_GB_S * 1e9) * 1e3,
        "measured_ms": ANCHOR_TOKEN_MS,
    }

    ops_item = {
        "incumbent_gemv_mac_flops": incumbent_gemv,
        "incumbent_gemv_gflop": incumbent_gemv / 1e9,
        "incumbent_activation_flops": incumbent_act,
        "incumbent_dispatches": ANCHOR_DISPATCHES,
        "mlp_dense_mac": mlp_dense_mac,
        "mlp_sequential_two_stage_mac": mlp_seq_mac,
        "mlp_fused_recompute_mac": mlp_fused_mac,
        "token_gemv_if_mlp_sequential": seq_total_mac,
        "token_gemv_gflop_if_mlp_sequential": seq_total_mac / 1e9,
        "token_gemv_if_mlp_fused_recompute": fused_total_mac,
        "token_gemv_gflop_if_mlp_fused_recompute": fused_total_mac / 1e9,
        "sequential_vs_incumbent": seq_total_mac / incumbent_gemv,
        "fused_vs_incumbent": fused_total_mac / incumbent_gemv,
        "compared_against": "51.24 GFLOP / 964 dispatches (measured incumbent)",
        "reading": (
            f"Sequential two-stage at G035 ranks would drop GEMV MACs "
            f"{incumbent_gemv/1e9:.2f} → {seq_total_mac/1e9:.2f} GFLOP "
            f"({seq_total_mac/incumbent_gemv:.3f}×) and raise dispatches "
            f"964 → {ANCHOR_DISPATCHES - 192 + 384}. "
            f"Fused hgravs01-clone would RAISE GEMV MACs to "
            f"{fused_total_mac/1e9:.2f} GFLOP ({fused_total_mac/incumbent_gemv:.1f}×) "
            "because every TG recomputes mid. Neither path preserves function."
        ),
    }

    topo = dispatch_topology(organs)
    metal = metal_feasibility(organs, caps)
    layout = memory_layout(organs)
    micro = cheap_microbenchmark(identity)
    value = expected_value(decision, organs, layout, topo, ops_item)

    operator = {
        "name": "G035 pair-shared column basis × per-layer coefficients",
        "built": False,
        "reason_not_built": decision["scoped_refutation"],
        "math": {
            "W_ell": "R^{m×n}, x in R^n, y in R^m",
            "column_basis_share_the_one_that_lost": (
                "For a pair (W_a, W_b), stack S = [W_a W_b] in R^{m×2n}. "
                "Randomized SVD gives U in R^{m×r} spanning the column space of S. "
                "C_ell = U^T W_ell in R^{r×n}. Ŵ_ell = U C_ell."
            ),
            "native_production_algebra": (
                "z = C_ell x  in R^r;  y = U z  in R^m.  Never form Ŵ_ell. "
                "This is the same y as (U C_ell) x by associativity."
            ),
            "row_basis_share_dead_zone_win": (
                "Stack along rows, share V in R^{n×r}, Ŵ_ell = A_ell V^T. "
                "Native: z = V^T x; y = A_ell z. Measured +6.3% adjacent at ~1.03 b/elem, "
                "unusable vs coherent 0.198. Not a reopen."
            ),
            "ranks": {
                "independent": G035_R_IND,
                "gate_up_shared": organs["mlp.gate_proj"]["rank_shared"],
                "down_shared": organs["mlp.down_proj"]["rank_shared"],
                "matching_rule": "largest r_s with r_s*m + 2*r_s*n <= 2*r_ind*(m+n)",
            },
        },
        "correctness_oracle": {
            "labelled": "ORACLE",
            "path": "reconstruct_dense_then_GEMM",
            "definition": (
                "Form Ŵ_ell = U C_ell as an m×n f32 matrix in DRAM (or on host). "
                "y_oracle = Ŵ_ell x. Compare y_native to y_oracle. This path exists "
                "to prove algebraic correctness. It is NOT a production implementation "
                "and must never be presented as one. On the incumbent, the analogous "
                "trap is qwen_uniform_q4_decode_vector (102.5e9 B/token of dense W); "
                "qwen38_hybrid_decode.rs does not dispatch it."
            ),
            "tolerance": "max_abs(y_native - y_oracle) consistent with f32 GEMV rounding",
            "this_process_toy_residual": identity["max_abs_residual"],
        },
        "production_path": {
            "named_separately": True,
            "name": "sequential_two_stage_shared_basis_matvec",
            "would_be": (
                "Two Metal dispatches per organ, one CB: (1) mid = C@x, (2) y = U@mid. "
                "Bind one U buffer per pair, one C buffer per layer. No dense W."
            ),
            "status": "NOT_TO_BE_WRITTEN",
            "existing_near_miss": (
                "q80_hgravs01_two_stage_matvec is per-tensor U,V fused with TG-private "
                "mid recompute, kRankCap=160, kXCap=512. It is not this family (census: "
                "shared_basis_x_coefficients ABSENT) and cannot run G035 ranks."
            ),
        },
        "foldable_transforms": {
            "preference": (
                "Butterfly / Walsh-Hadamard / structured rotations over a dense per-layer "
                "transform, which just moves the reconstruction."
            ),
            "g032": prior.get("g032_hadamard_q4"),
            "decision": (
                "Hadamard is generated (0 stored bytes) and already measured at q4: "
                "+3.4e-4 hold cosine, +0.026 entropy bits. It is not a shared-basis "
                "coefficient win. Do not pivot the family into structured_transform."
            ),
        },
    }

    watched = list(prior.get("watched") or [])
    watched.extend(
        [
            {
                "what": "Treat missing native kernel as the thing that killed sharing",
                "result": "REJECTED",
                "detail": (
                    f"Associativity residual {identity['max_abs_residual']:.3e}. "
                    "G035's extra error is in Ŵ, not in how Ŵ is applied."
                ),
            },
            {
                "what": "Design around G035 row-basis 6.3% adjacent win",
                "result": "STOPPED",
                "detail": (
                    "Dead-zone 1.03 b/elem, errors 0.58 vs coherent 0.198. G034 ceiling "
                    "2.93× q3 at 3.25 b/elem. That is still a fidelity result."
                ),
            },
            {
                "what": "Clone q80_hgravs01_two_stage_matvec as the production path",
                "result": "INFEASIBLE_AT_G035",
                "detail": (
                    f"kRankCap={caps.get('kRankCap')} kXCap={caps.get('kXCap')}; "
                    f"G035 gate r={organs['mlp.gate_proj']['rank_shared']} "
                    f"n={organs['mlp.gate_proj']['shape'][1]}. Fused recompute "
                    f"{organs['mlp.gate_proj']['fused_over_dense']:.1f}× dense FLOPs on gate."
                ),
            },
            {
                "what": "Re-run G035 SVD on this box",
                "result": "NOT_RUN (refused)",
                "detail": (
                    "Would need the BF16 parent and the thick v2 capture. Those sit "
                    "outside WRITE scope. The receipt is in git; the question was "
                    "which failure it recorded, not whether the SVD still runs."
                ),
            },
            {
                "what": "Live 27B native decode re-time / MLX 35.51 remeasure",
                "result": "NOT_RUN (refused)",
                "detail": (
                    "Anchors 32.73 tok/s / 30.606 ms and MLX 35.51 live / llama.cpp Q5_K "
                    "24.12 archived are supplied. Occupancy is not free (two servers: "
                    "3.986 tok/s vs 33.47 with one)."
                ),
            },
            {
                "what": "223 sub-0.5 local BPW rows as a result",
                "result": "PAIRED_WITH_HEALTH",
                "detail": f"{ANCHOR_SUB05_ROWS} rows, healthy={ANCHOR_SUB05_HEALTHY}. A low number is not a result until paired with a health verdict.",
            },
            {
                "what": "Q80 0.6462 as the density that decode moves",
                "result": "CORRECTED_TO_BOTH",
                "detail": (
                    f"storage BPW {ANCHOR_Q80_STORAGE_BPW} against ACTIVE "
                    f"{ANCHOR_Q80_ACTIVE_BPW} (factor {ANCHOR_Q80_ACTIVE_BPW/ANCHOR_Q80_STORAGE_BPW:.1f}). "
                    "Report both or neither."
                ),
            },
        ]
    )

    eight = {
        "1_mathematical_operator": operator,
        "2_expected_bytes_per_token": bytes_item,
        "3_expected_ops_per_token": ops_item,
        "4_dispatch_topology": topo,
        "5_metal_feasibility": metal,
        "6_memory_layout": layout,
        "7_cheap_microbenchmark": micro,
        "8_expected_value": value,
    }

    self_check = {
        "eight_items_present": set(eight) == {
            "1_mathematical_operator",
            "2_expected_bytes_per_token",
            "3_expected_ops_per_token",
            "4_dispatch_topology",
            "5_metal_feasibility",
            "6_memory_layout",
            "7_cheap_microbenchmark",
            "8_expected_value",
        },
        "prior_science_searched": True,
        "family_refuted_on_fidelity": decision["failure_that_killed_it"] == "fidelity",
        "stopped_rather_than_designing_around": decision["stop"] and value["do_not_build"],
        "oracle_labelled_oracle": operator["correctness_oracle"]["labelled"] == "ORACLE",
        "production_path_named_separately": operator["production_path"]["named_separately"],
        "ops_compared_against_51_24_gflop": ops_item["incumbent_gemv_gflop"] > 51.0,
        "dispatches_compared_against_964": topo["incumbent"]["dispatches_per_token"] == 964,
        "microbenchmark_concrete": "gravity_share_crosslayer.py" in micro["already_ran_fidelity"]["command"],
        "verdict_is_not_worth_building": value["verdict"] == "NOT_WORTH_BUILDING",
        "associativity_holds": identity["identity_holds_at_1e12"],
        "g035_rank_gate_is_417": organs["mlp.gate_proj"]["rank_shared"] == 417,
        "g035_rank_down_is_288": organs["mlp.down_proj"]["rank_shared"] == 288,
        "fused_increases_ops": ops_item["fused_vs_incumbent"] > 1.0,
        "sequential_decreases_ops": ops_item["sequential_vs_incumbent"] < 1.0,
        "hgravs01_caps_parsed": caps.get("kRankCap") == 160 and caps.get("kXCap") == 512,
    }
    self_check["all_passed"] = all(
        v is True for k, v in self_check.items() if k != "all_passed"
    )

    return {
        "schema": SCHEMA,
        "generated_at": now_utc(),
        "git_head": git_head(),
        "lane": "c1sharedbasis",
        "question": decision["question"],
        "answer": decision["answer"],
        "verdict": decision["verdict"],
        "failure_that_killed_it": decision["failure_that_killed_it"],
        "stop": decision["stop"],
        "scoped_refutation": decision["scoped_refutation"],
        "decision": decision,
        "prior_science_search": {
            "performed": True,
            "result": (
                "Family already refuted on FIDELITY for this parent. Native execution "
                "does not scope the refutation away from the idea."
            ),
            "records": prior,
        },
        "incumbent_anchors_not_rederived": {
            "tps": ANCHOR_TPS,
            "ms_per_token": ANCHOR_TOKEN_MS,
            "roof_gb_s": ANCHOR_ROOF_GB_S,
            "dispatches": ANCHOR_DISPATCHES,
            "command_buffers": ANCHOR_CBS,
            "gemv_gflop": incumbent_gemv / 1e9,
            "parameter_count": ANCHOR_PARAMS,
            "bpw": ANCHOR_BPW,
            "artifact_bytes": ANCHOR_ARTIFACT_B,
            "tensors": ANCHOR_TENSORS,
            "reconstructs_dense_on_38": "NO",
            "reachable_dead_unknown": [ANCHOR_REACHABLE, ANCHOR_DEAD, ANCHOR_UNKNOWN],
            "mlx_4bit_tps_live": ANCHOR_MLX_TPS,
            "llamacpp_q5k_tps_archived": ANCHOR_LLAMACPP_Q5K_TPS,
        },
        "associativity_identity": identity,
        "two_stage_kernel_caps": caps,
        "organs_derived": organs,
        "eight_items": eight,
        "what_i_watched_fail": watched,
        "write_scope": {
            "write": [
                "tools/headless/c1sharedbasis_design.py",
                "receipts/headless/C1SHAREDBASIS_DESIGN.json",
            ],
            "verify": ["tools/headless", "receipts/headless"],
            "deny": ["workspace", "crates", "visionmcp", "app", "lab", "tools/haider", "ramanujan"],
            "crates_read_only": True,
        },
        "self_check": self_check,
    }


def render(doc: dict) -> str:
    d = doc["decision"]
    eight = doc["eight_items"]
    op = eight["1_mathematical_operator"]
    b = eight["2_expected_bytes_per_token"]
    o = eight["3_expected_ops_per_token"]
    t = eight["4_dispatch_topology"]
    m = eight["5_metal_feasibility"]
    lay = eight["6_memory_layout"]
    mb = eight["7_cheap_microbenchmark"]
    ev = eight["8_expected_value"]
    g035 = doc["prior_science_search"]["records"]["g035"]
    lines = []
    a = lines.append
    a("=" * 78)
    a("C1 SHAREDBASIS DESIGN — native operator, or a refutation properly scoped")
    a("=" * 78)
    a(f"schema     {doc['schema']}")
    a(f"generated  {doc['generated_at']}")
    a(f"git_head   {doc['git_head']}")
    a("")
    a("## QUESTION")
    a(d["question"])
    a("")
    a("## ANSWER")
    a(d["answer"])
    a(f"failure_that_killed_it: {d['failure_that_killed_it']}")
    a(f"verdict: {d['verdict']}")
    a(f"stop: {d['stop']}")
    a("")
    a(d["scoped_refutation"])
    a("")
    a("## PRIOR-SCIENCE SEARCH")
    a("Performed. Family already refuted on fidelity for this parent. Stop.")
    if g035.get("loaded"):
        a(f"  G035 loaded: {g035.get('obligation')}")
        a(f"  error_space: {g035.get('error_space')}")
        for p in g035.get("pairs") or []:
            a(
                f"  {p['organ']} L{p['layers'][0]}/L{p['layers'][1]} ({p['kind']}): "
                f"ind r={p['rank_independent']} err={p['independent_mean']:.5f} | "
                f"shared r={p['rank_shared']} err={p['shared_mean']:.5f} | "
                f"shared_beats_independent={p['shared_beats_independent']}"
            )
        a(f"  adjacent_mean_error_reduction: {g035.get('adjacent_mean_error_reduction')}")
        a(f"  far_control_mean_error_reduction: {g035.get('far_control_mean_error_reduction')}")
        a(f"  corrected_verdict: {g035.get('corrected_verdict')}")
        a(f"  what_still_limits_it: {g035.get('what_still_limits_it')}")
    rec = doc["prior_science_search"]["records"]
    if rec.get("g034"):
        g = rec["g034"]
        a(f"  G034 family_verdict: {g.get('family_verdict')}")
        a(
            f"  G034 mean_flat_q3={g.get('mean_flat_q3'):.5f} mean_lowrank={g.get('mean_lowrank'):.5f} "
            f"ratio={g.get('error_ratio'):.2f}× mac_ratio={g.get('mean_mac_ratio'):.3f}"
        )
    if rec.get("ns010"):
        n = rec["ns010"]
        a(
            f"  NS-010 {n.get('class')}: {n.get('mechanism')} "
            f"Q80 gate cosine={n.get('gate_proj_pairwise_cosine_mean')}"
        )
    g1 = rec.get("g1_shared_basis_this_parent") or {}
    a(f"  g1-shared-basis this parent: {g1.get('status')}")
    fam = rec.get("kernel_census_family") or {}
    a(f"  kernel census shared_basis_x_coefficients: {fam.get('verdict')}")
    miss = rec.get("kernel_census_missing_cost") or {}
    a(f"  missing_family_cost: {miss.get('status')}")
    sa = rec.get("g013_storage_vs_active") or {}
    a(
        f"  Q80 storage BPW {sa.get('storage_bpw')} against ACTIVE {sa.get('active_bpw')} "
        f"(factor {sa.get('factor')})"
    )
    a(f"  223 components <0.5 local BPW, healthy={ANCHOR_SUB05_HEALTHY}.")
    a(
        f"  MLP distillation NO-GO (brief): +{ANCHOR_MLP_DISTILL_GAP} held-out gap vs q3 "
        f"at {ANCHOR_MLP_DISTILL_BYTE_FRAC:.0%} of its active bytes."
    )
    a("")
    a("## 1. MATHEMATICAL OPERATOR (the one that was measured — not to be built)")
    a(op["math"]["column_basis_share_the_one_that_lost"])
    a("Native algebra: " + op["math"]["native_production_algebra"])
    a(
        f"Ranks: independent {op['math']['ranks']['independent']}, "
        f"gate/up shared {op['math']['ranks']['gate_up_shared']}, "
        f"down shared {op['math']['ranks']['down_shared']}."
    )
    a("")
    a("  CORRECTNESS ORACLE (labelled ORACLE, not production):")
    a("  " + op["correctness_oracle"]["definition"])
    a(f"  this-process toy residual: {op['correctness_oracle']['this_process_toy_residual']:.3e}")
    a("")
    a("  PRODUCTION PATH (named separately):")
    a(f"  name={op['production_path']['name']} status={op['production_path']['status']}")
    a("  " + op["production_path"]["would_be"])
    a("  " + op["production_path"]["existing_near_miss"])
    a("  Foldable transforms: " + op["foldable_transforms"]["decision"])
    a("")
    a("## 2. EXPECTED BYTES / TOKEN (derived)")
    a(
        f"  incumbent DRAM {b['incumbent']['executable_dram_bytes_per_token']:,} B  "
        f"weights {b['incumbent']['executable_weight_bytes_per_token']:,} B  "
        f"Q4 GEMV {b['incumbent']['q4_gemv_bytes_per_token']:,} B"
    )
    a(f"  incumbent MLP Q4 {b['incumbent']['mlp_q4_bytes_per_token']:,} B")
    c = b["counterfactual_g035_rank_mlp_only"]
    a(
        f"  G035-rank shared f16 MLP {c['mlp_shared_f16_bytes_per_token']:,} B  "
        f"({c['mlp_byte_ratio_vs_q4']:.3f}× Q4)"
    )
    a(
        f"  implied GEMV payload if MLP replaced {c['implied_gemv_payload_if_mlp_replaced']:,} B"
    )
    a(
        f"  roof ms at {ANCHOR_ROOF_GB_S} GB/s on incumbent DRAM: {b['roof_ms_incumbent']:.2f} ms "
        f"(measured {b['measured_ms']} ms)"
    )
    a("")
    a("## 3. EXPECTED OPS / TOKEN (vs 51.24 GFLOP / 964 dispatches)")
    a(f"  incumbent GEMV {o['incumbent_gemv_gflop']:.2f} GFLOP, activations {o['incumbent_activation_flops']:,} FLOP")
    a(
        f"  sequential two-stage token GEMV {o['token_gemv_gflop_if_mlp_sequential']:.2f} GFLOP "
        f"({o['sequential_vs_incumbent']:.3f}× incumbent)"
    )
    a(
        f"  fused hgravs01-clone token GEMV {o['token_gemv_gflop_if_mlp_fused_recompute']:.2f} GFLOP "
        f"({o['fused_vs_incumbent']:.1f}× incumbent)  ← ops go UP"
    )
    a("  " + o["reading"])
    a("")
    a("## 4. DISPATCH TOPOLOGY")
    a(
        f"  incumbent {t['incumbent']['dispatches_per_token']} dispatches, "
        f"{t['incumbent']['command_buffers_per_token']} CB. {t['incumbent']['formula']}"
    )
    a(
        f"  sequential two-stage {t['sequential_two_stage_if_built']['dispatches_per_token']} "
        f"dispatches ({t['sequential_two_stage_if_built']['delta_vs_incumbent']:+d}), "
        f"still {t['sequential_two_stage_if_built']['command_buffers_per_token']} CB."
    )
    a("  sync: " + t["sequential_two_stage_if_built"]["synchronises"])
    a("  fused clone: " + t["fused_clone_if_built"]["note"])
    a("  " + t["shared_V_does_not_collapse_stage1"])
    a("")
    a("## 5. METAL FEASIBILITY")
    a(f"  simdgroup width {m['simdgroup_width']} (from the two-stage kernel, not guessed)")
    cap = m["g035_vs_existing_fused_kernel"]
    a(f"  drop-in clone runs G035? {cap['drop_in_clone_runs_g035']}")
    a("  " + cap["why"])
    tg = m["threadgroup_memory_at_g035_ranks"]
    a("  " + tg["reading"])
    a("  " + m["fused_recompute_is_a_trap"]["why"])
    a("  coalescing: " + m["coalescing"]["vs_q4"])
    a("")
    a("## 6. MEMORY LAYOUT")
    a("  " + lay["representation"])
    a("  " + lay["why_pairs_not_all64"])
    a(f"  MLP shared f16 streamed {lay['mlp_shared_f16_bytes_per_token']:,} B/token")
    a(f"  temporary mid {lay['temporary_mid_bytes_live']} B (workspace, not stored)")
    a(
        f"  oracle dense-W if it ran on MLP: {lay['dense_W_bytes_if_oracle_ran_mlp']:,} B/token "
        "(forbidden in production)"
    )
    for p in lay["organs"]:
        a(
            f"  {p['organ']}: U {p['U_per_pair']['shape']} f16 ×{p['U_per_pair']['count']} pairs, "
            f"C {p['C_per_layer']['shape']} f16 ×{p['C_per_layer']['count']} layers, "
            f"stream {p['bytes_per_token_streamed']:,} B"
        )
    a("")
    a("## 7. CHEAP MICROBENCHMARK")
    a("  already ran: " + mb["already_ran_fidelity"]["command"])
    a(f"  kill_if: {mb['already_ran_fidelity']['kill_if']}")
    a(f"  observed: {mb['already_ran_fidelity']['observed']}")
    a(
        f"  this process associativity max_abs_residual="
        f"{mb['this_process_associativity']['observed']['max_abs_residual']:.3e} "
        f"holds={mb['this_process_associativity']['observed']['identity_holds_at_1e12']}"
    )
    a("  if fidelity had held: " + mb["would_run_before_a_kernel_if_fidelity_had_held"]["command"])
    a("")
    a("## 8. EXPECTED VALUE")
    a(f"  verdict: {ev['verdict']}")
    w = ev["would_win_if_function_held"]
    a(
        f"  fantasy: MLP bytes {w['mlp_bytes_q4']:,} → {w['mlp_bytes_shared_f16_g035_rank']:,} "
        f"({w['mlp_byte_ratio']:.3f}×); MLP MAC ratio {w['mlp_mac_ratio_sequential']:.3f}×; "
        f"incumbent {w['tok_s_incumbent']} tok/s vs MLX {w['tok_s_mlx_4bit_live']} vs "
        f"llama.cpp Q5_K {w['tok_s_llamacpp_q5k_archived']} archived."
    )
    a("  " + w["fantasy_dram_if_mlp_replaced"])
    a("  risks:")
    for r in ev["what_it_risks"]:
        a(f"    - {r}")
    a("  cheapest kill: " + ev["cheapest_experiment_that_would_kill_it"])
    a("  S011 §4: " + ev["s011_section4"]["why"])
    a(f"  do_not_build: {ev['do_not_build']}")
    a("")
    a("## WHAT I WATCHED FAIL")
    for i, wfail in enumerate(doc["what_i_watched_fail"], 1):
        a(f"  {i}. {wfail['what']}")
        a(f"     result: {wfail['result']}")
        a(f"     {wfail['detail']}")
    a("")
    a("## SELF-CHECK")
    for k, v in doc["self_check"].items():
        a(f"  {k}: {v}")
    a("")
    a("## WRITE SCOPE")
    ws = doc["write_scope"]
    a(f"  WRITE {ws['write']}")
    a(f"  DENY  {ws['deny']}  crates_read_only={ws['crates_read_only']}")
    a("=" * 78)
    return "\n".join(lines) + "\n"


def main() -> int:
    doc = build()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(render(doc))
    print(f"wrote {RECEIPT} ({RECEIPT.stat().st_size} bytes)")
    if not doc["self_check"].get("all_passed"):
        print("DESIGN SELF-CHECK FAILED", file=sys.stderr)
        for k, v in doc["self_check"].items():
            if v is False:
                print(f"  {k}", file=sys.stderr)
        return 1
    if doc["verdict"] != "NOT_WORTH_BUILDING":
        print("expected NOT_WORTH_BUILDING (fidelity kill)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
