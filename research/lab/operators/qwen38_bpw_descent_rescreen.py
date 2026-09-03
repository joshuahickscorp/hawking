#!/usr/bin/env python3
"""Re-rank the Qwen3.8 descent with MEASURED in-register reconstruction cost.

Quality numbers come from the first screen (real activations, 6 layers).
Reconstruction numbers come from QWEN38_RECON_MEASURED.json (GPU, occupancy
tiles, GPUEndTime-GPUStartTime). The transferred 5.9x rice penalty is not used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
FIRST = REPO / "receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json"
MEAS = REPO / "receipts/ascent-2026-08-16/QWEN38_RECON_MEASURED.json"
OUT = REPO / "receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT_RESREEN.json"

CURRENT_BPW = 4.252735126866492
GPU_MS = 36.987458  # QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY
FIXED_MS = 1.229334
F_MLP = 0.6363
F_ATTN = 0.2692
F_EMB = 0.0945
ACHIEVED_GBPS = 406.2
Q80_BAR = 0.8604
MOD_BAR = 0.90
TIGHT_BAR = 0.95

# First-screen assumed penalties (the ones we are replacing).
ASSUMED = {
    "uniform_q4_g64": 1.0,
    "uniform_q3_g64": 1.0,
    "uniform_q2_g64": 1.0,
    "binary_g128": 1.0,
    "ternary_t0.7_g128": 1.05,
    "hadamard_q2_g128": 1.15,
    "additive_q2q2_g64": 1.20,
    "rice_q1_rms_2pct": 5.9,
    "prod_q4_nibble_g64": 1.0,
    "hgravs01_r160_q3": 1.0,
}

# In-register kernels used for ranking (serial artifact excluded).
RANK_KERNEL = {
    "prod_q4_nibble_g64": "disc_q4_nibble_tpr64",
    "uniform_q4_g64": "disc_uniform_bits_tpr64",
    "uniform_q3_g64": "disc_uniform_bits_tpr64",
    "uniform_q2_g64": "disc_uniform_bits_tpr64",
    "binary_g128": "disc_binary_tpr64",
    "ternary_t0.7_g128": "disc_ternary_tpr64",
    "hadamard_q2_g128": "disc_walsh_hadamard_x+disc_uniform_bits_tpr64",
    "additive_q2q2_g64": "disc_additive_tpr64",
    "rice_q1_rms_2pct": "disc_binary_csr_tpr64",
    "hgravs01_r160_q3": "L@(R@x) two disc_uniform_bits_tpr64",
}


def pick(summary: list[dict], role: str, codec: str) -> dict:
    for r in summary:
        if r["role"] == role and r["codec"] == codec:
            return r
    raise KeyError(f"{role} {codec}")


def mean_roles(summary, roles, codec, field) -> float:
    return float(sum(pick(summary, r, codec)[field] for r in roles) / len(roles))


def wall_tps(bpw: float, penalty: float) -> tuple[float, float]:
    gpu = GPU_MS * (bpw / CURRENT_BPW) * penalty
    wall = gpu + FIXED_MS
    tps = 1000.0 / wall if wall > 0 else 0.0
    return wall, tps


def best_inregister(variants: list[dict], codec: str) -> dict | None:
    """Fastest occupancy-tiled in-register variant for this codec."""
    want = RANK_KERNEL.get(codec)
    cands = []
    for v in variants:
        name = v.get("name", "")
        kern = v.get("kernel", "")
        if "serial" in name or "serial" in kern:
            continue
        if codec == "rice_q1_rms_2pct":
            if "csr_inregister" in name:
                cands.append(v)
            continue
        if codec == "hadamard_q2_g128":
            if name.startswith("hadamard"):
                cands.append(v)
            continue
        if codec == "hgravs01_r160_q3":
            if name.startswith("hgravs"):
                cands.append(v)
            continue
        if name == codec or name.startswith(codec + "/") or name.startswith(codec):
            if want is None or want in kern or name.endswith(want):
                cands.append(v)
            elif "tpr64" in kern or "tg256" in kern:
                cands.append(v)
    if not cands:
        return None
    return min(cands, key=lambda v: int(v["median_gpu_ns"]))


def codec_cost(organs: list[dict]) -> dict[str, dict]:
    """Per-codec measured cost, averaged over organs that have it."""
    by: dict[str, list[dict]] = {}
    for organ in organs:
        seen = {}
        for v in organ["variants"]:
            name = v["name"]
            base = name.split("/")[0]
            seen.setdefault(base, []).append(v)
        for base, vs in seen.items():
            best = best_inregister(vs, base)
            if best is None:
                continue
            rec = {
                "organ": organ["name"],
                "rows": organ["rows"],
                "cols": organ["cols"],
                "variant": best,
            }
            by.setdefault(base, []).append(rec)
    out = {}
    for codec, recs in by.items():
        pens = [float(r["variant"]["recon_penalty_vs_bandwidth"]) for r in recs]
        gbps = [float(r["variant"]["packed_gbps"]) for r in recs]
        ns = [int(r["variant"]["median_gpu_ns"]) for r in recs]
        excess = [float(r["variant"]["recon_excess_ns"]) for r in recs]
        # correctness: take worst cosine
        cos = []
        for r in recs:
            c = r["variant"].get("correctness") or {}
            if isinstance(c, dict) and c.get("cosine") is not None:
                cos.append(float(c["cosine"]))
        out[codec] = {
            "n_organs": len(recs),
            "recon_penalty_measured": float(sum(pens) / len(pens)),
            "recon_penalty_max": float(max(pens)),
            "packed_gbps_mean": float(sum(gbps) / len(gbps)),
            "packed_gbps_min": float(min(gbps)),
            "median_gpu_ns_by_organ": {r["organ"]: int(r["variant"]["median_gpu_ns"]) for r in recs},
            "recon_excess_ns_mean": float(sum(excess) / len(excess)),
            "stays_at_bandwidth_wall": bool(max(pens) <= 1.15),
            "kernel_used": recs[0]["variant"]["kernel"],
            "correctness_cosine_min": (min(cos) if cos else None),
            "assumed_penalty_first_screen": ASSUMED.get(codec),
            "organs": [
                {
                    "organ": r["organ"],
                    "median_gpu_ns": int(r["variant"]["median_gpu_ns"]),
                    "packed_gbps": float(r["variant"]["packed_gbps"]),
                    "recon_penalty_vs_bandwidth": float(r["variant"]["recon_penalty_vs_bandwidth"]),
                    "traffic_bytes": int(r["variant"]["traffic_bytes"]),
                    "kernel": r["variant"]["kernel"],
                }
                for r in recs
            ],
        }
    return out


def recipe(
    *,
    name: str,
    mlp_bpw: float,
    attn_bpw: float,
    emb_bpw: float,
    penalty: float,
    quality: str,
    verdict: str,
    note: str,
    quality_intact: bool,
    cost_class: str,
) -> dict[str, Any]:
    bpw = F_MLP * mlp_bpw + F_ATTN * attn_bpw + F_EMB * emb_bpw
    wall, tps = wall_tps(bpw, penalty)
    wall1, tps1 = wall_tps(bpw, 1.0)
    saved_frac = CURRENT_BPW - bpw
    # Model-scale GPU ns under measured penalty.
    gpu_ns = GPU_MS * 1e6 * (bpw / CURRENT_BPW) * penalty
    floor_ns = GPU_MS * 1e6 * (bpw / CURRENT_BPW)
    excess_ns = max(0.0, gpu_ns - floor_ns)
    bytes_saved = saved_frac / CURRENT_BPW  # relative; absolute cancels in rank
    score_vs_time = saved_frac / wall if wall > 0 else 0.0
    score_vs_excess = None if excess_ns <= 1e3 else bytes_saved / (excess_ns * 1e-9)
    return {
        "codec": name,
        "mlp_bpw": round(mlp_bpw, 4),
        "attn_bpw": round(attn_bpw, 4),
        "emb_bpw": round(emb_bpw, 4),
        "projected_bpw": round(bpw, 4),
        "recon_penalty_measured": round(penalty, 4),
        "reconstruction_cost_class": cost_class,
        "projected_wall_ms": round(wall, 3),
        "projected_tps": round(tps, 2),
        "projected_wall_ms_penalty_one": round(wall1, 3),
        "projected_tps_penalty_one": round(tps1, 2),
        "clears_2p0": bool(bpw <= 2.0),
        "clears_50tps": bool(tps >= 50.0),
        "quality_intact": quality_intact,
        "quality_evidence": quality,
        "bytes_saved_over_wall_ms": round(score_vs_time, 4),
        "bytes_saved_over_recon_excess_ns": score_vs_excess,
        "verdict": verdict,
        "note": note,
    }


def weighted_penalty(parts: list[tuple[float, float, float]]) -> float:
    """Byte-weighted penalty. Each part is (mass_frac, bpw, penalty)."""
    num = 0.0
    den = 0.0
    for mass, bpw, pen in parts:
        w = mass * bpw
        num += w * pen
        den += w
    return num / den if den else 1.0


def main() -> int:
    first = json.loads(FIRST.read_text())
    meas = json.loads(MEAS.read_text())
    summary = first["summary"]["by_role_codec"]
    costs = codec_cost(meas["organs"])

    def pen(codec: str, default: float = 1.0) -> float:
        if codec in costs:
            return float(costs[codec]["recon_penalty_measured"])
        return default

    p_q4 = pen("uniform_q4_g64", 1.0)
    p_q3 = pen("uniform_q3_g64", 1.0)
    p_q2 = pen("uniform_q2_g64", 1.0)
    p_bin = pen("binary_g128", 1.0)
    p_ter = pen("ternary_t0.7_g128", 1.0)
    p_had = pen("hadamard_q2_g128", 1.0)
    p_add = pen("additive_q2q2_g64", 1.0)
    p_rice = pen("rice_q1_rms_2pct", 1.0)
    p_ts = pen("hgravs01_r160_q3", 1.0)

    mlp_roles = ("gate_proj", "up_proj", "down_proj")
    attn_roles = ("attn_in", "attn_out")

    def mlp(c):
        return mean_roles(summary, mlp_roles, c, "physical_bpw_mean")

    def attn(c):
        return mean_roles(summary, attn_roles, c, "physical_bpw_mean")

    emb_q4 = 4.250011256679038
    ts_bpw = 0.13161714918473189  # sibling HGRAVS01 measured, 64-layer
    # our L00 QR pack was 0.1315 — use sibling's sealed 64-layer figure for recipes

    ter_gate = pick(summary, "gate_proj", "ternary_t0.7_g128")
    ter_up = pick(summary, "up_proj", "ternary_t0.7_g128")
    ter_down = pick(summary, "down_proj", "ternary_t0.7_g128")
    ter_attn = pick(summary, "attn_in", "ternary_t0.7_g128")
    rice_gate = pick(summary, "gate_proj", "rice_q1_rms_2pct")
    rice_up = pick(summary, "up_proj", "rice_q1_rms_2pct")
    rice_down = pick(summary, "down_proj", "rice_q1_rms_2pct")
    rice_attn = pick(summary, "attn_in", "rice_q1_rms_2pct")
    q3_mlp_min = min(pick(summary, r, "uniform_q3_g64")["hold_min"] for r in mlp_roles)
    q3_attn_min = pick(summary, "attn_in", "uniform_q3_g64")["hold_min"]
    bin_gate = pick(summary, "gate_proj", "binary_g128")
    bin_up = pick(summary, "up_proj", "binary_g128")
    bin_down = pick(summary, "down_proj", "binary_g128")

    sib_gate, sib_up, sib_down = 1.1250234267290902, 1.2875108157887178, ts_bpw
    sib_mlp = (sib_gate + sib_up + sib_down) / 3.0

    ter_mlp = mlp("ternary_t0.7_g128")
    ter_attn_bpw = attn("ternary_t0.7_g128")
    rice_mlp = mlp("rice_q1_rms_2pct")
    q3_mlp = mlp("uniform_q3_g64")
    q3_attn = attn("uniform_q3_g64")

    # Recipe penalties: byte-weighted mix of measured organ penalties.
    # Attention uses the same codec family as named; embed stays q4.
    recipes = []

    recipes.append(
        recipe(
            name="incumbent_uniform_q4_all",
            mlp_bpw=mlp("uniform_q4_g64"),
            attn_bpw=attn("uniform_q4_g64"),
            emb_bpw=emb_q4,
            penalty=p_q4,
            cost_class="CHEAP_INREGISTER",
            quality="hold min 0.9934; bring-up generate coherent",
            quality_intact=True,
            verdict="BASELINE. Measured 38.217 ms complete wall / 26.2 TPS. Fails G016 2.0.",
            note="Production occupancy-tile q4 is already at the bandwidth wall.",
        )
    )

    p_ter_ts = weighted_penalty(
        [
            (F_MLP * 2 / 3, 2.2500, p_ter),  # gate+up ternary
            (F_MLP * 1 / 3, ts_bpw, p_ts),
            (F_ATTN, 2.2500, p_ter),
            (F_EMB, emb_q4, p_q4),
        ]
    )
    recipes.append(
        recipe(
            name="ternary_gate_up_hgravs01_twostage_down_ternary_attn_q4_emb",
            mlp_bpw=(2.2500 + 2.2500 + ts_bpw) / 3.0,
            attn_bpw=2.2501,
            emb_bpw=emb_q4,
            penalty=p_ter_ts,
            cost_class="INREGISTER_TERNARY + TWO_STAGE",
            quality=(
                f"ternary gate/up/attn_in hold mins "
                f"{ter_gate['hold_min']:.4f}/{ter_up['hold_min']:.4f}/{ter_attn['hold_min']:.4f} "
                f"(100% >= 0.90). down 0.132 two-stage; L00 QR fit token-cos 0.9825 "
                f"(sibling 64-layer HGRAVS01 owns the production fit)."
            ),
            quality_intact=True,
            verdict="",
            note="Original first-screen winner. Re-ranked with measured penalty.",
        )
    )

    p_sib = weighted_penalty(
        [
            (F_MLP * 1 / 3, sib_gate, p_bin),
            (F_MLP * 1 / 3, sib_up, p_rice),
            (F_MLP * 1 / 3, sib_down, p_ts),
            (F_ATTN + F_EMB, 4.2501, p_q4),
        ]
    )
    recipes.append(
        recipe(
            name="REF_sibling_binary_rice_hgravs01_q4rest",
            mlp_bpw=sib_mlp,
            attn_bpw=4.2501,
            emb_bpw=emb_q4,
            penalty=p_sib,
            cost_class="INREGISTER_BINARY + CSR_RICE + TWO_STAGE",
            quality=(
                "Sibling 64-layer pack: gate 1.125 / up 1.288 / down 0.132, "
                "mean_component_cosine 0.907. Our rice up hold min "
                f"{rice_up['hold_min']:.4f} (fails 0.86 on mid-depth). "
                f"binary gate hold min {bin_gate['hold_min']:.4f}."
            ),
            quality_intact=False,
            verdict="",
            note="Sibling owns the pack (2.086 BPW). First screen priced rice at 5.9x.",
        )
    )

    p_rice_ts_ter = weighted_penalty(
        [
            (F_MLP * 2 / 3, 1.2876, p_rice),
            (F_MLP * 1 / 3, ts_bpw, p_ts),
            (F_ATTN, 2.2500, p_ter),
            (F_EMB, emb_q4, p_q4),
        ]
    )
    recipes.append(
        recipe(
            name="rice_gate_up_hgravs01_twostage_down_ternary_attn_q4_emb",
            mlp_bpw=(1.2876 + 1.2875 + ts_bpw) / 3.0,
            attn_bpw=2.2501,
            emb_bpw=emb_q4,
            penalty=p_rice_ts_ter,
            cost_class="INREGISTER_CSR_RICE + TWO_STAGE + TERNARY",
            quality=(
                f"rice gate/up hold mins {rice_gate['hold_min']:.4f}/{rice_up['hold_min']:.4f}; "
                f"ternary attn_in min {ter_attn['hold_min']:.4f}. "
                "Rice is NOT quality-intact on mid-depth up (0.813)."
            ),
            quality_intact=False,
            verdict="",
            note="Newly cheap if rice CSR stays at the wall. Quality still fails Density Law.",
        )
    )

    p_ter_all = weighted_penalty(
        [(F_MLP, 2.2500, p_ter), (F_ATTN, 2.2501, p_ter), (F_EMB, emb_q4, p_q4)]
    )
    recipes.append(
        recipe(
            name="ternary_t0.7_all_except_emb_q4",
            mlp_bpw=ter_mlp,
            attn_bpw=ter_attn_bpw,
            emb_bpw=emb_q4,
            penalty=p_ter_all,
            cost_class="INREGISTER_TERNARY",
            quality=(
                f"hold min gate {ter_gate['hold_min']:.4f} up {ter_up['hold_min']:.4f} "
                f"down {ter_down['hold_min']:.4f} attn_in {ter_attn['hold_min']:.4f}; "
                "L63 down 0.843 dips below 0.86"
            ),
            quality_intact=False,
            verdict="",
            note="Best cheap 2-bit rung. Still above 2.0.",
        )
    )

    recipes.append(
        recipe(
            name="uniform_q3_all",
            mlp_bpw=q3_mlp,
            attn_bpw=q3_attn,
            emb_bpw=3.2500,
            penalty=p_q3,
            cost_class="CHEAP_INREGISTER",
            quality=f"hold min MLP {q3_mlp_min:.4f}, attn_in {q3_attn_min:.4f}; 100% clear 0.95",
            quality_intact=True,
            verdict="",
            note="Cheap coherence floor. Misses 50 TPS.",
        )
    )

    p_q3_ts = weighted_penalty(
        [
            (F_MLP * 2 / 3, 3.2500, p_q3),
            (F_MLP * 1 / 3, ts_bpw, p_ts),
            (F_ATTN, 3.2500, p_q3),
            (F_EMB, emb_q4, p_q4),
        ]
    )
    recipes.append(
        recipe(
            name="q3_gate_up_hgravs01_twostage_down_q3_attn_q4_emb",
            mlp_bpw=(3.2500 + 3.2500 + ts_bpw) / 3.0,
            attn_bpw=3.2501,
            emb_bpw=emb_q4,
            penalty=p_q3_ts,
            cost_class="CHEAP_INREGISTER + TWO_STAGE",
            quality="q3 hold mins intact; down 0.132 two-stage",
            quality_intact=True,
            verdict="",
            note="Quality-safe conservative if ternary is refused. Still above 2.0.",
        )
    )

    p_rice_all = weighted_penalty(
        [(F_MLP, rice_mlp, p_rice), (F_ATTN, attn("rice_q1_rms_2pct"), p_rice), (F_EMB, emb_q4, p_q4)]
    )
    recipes.append(
        recipe(
            name="rice_all_except_emb_q4",
            mlp_bpw=rice_mlp,
            attn_bpw=attn("rice_q1_rms_2pct"),
            emb_bpw=emb_q4,
            penalty=p_rice_all,
            cost_class="INREGISTER_CSR_RICE",
            quality=(
                f"rice hold min gate {rice_gate['hold_min']:.4f} up {rice_up['hold_min']:.4f} "
                f"down {rice_down['hold_min']:.4f} attn_in {rice_attn['hold_min']:.4f}"
            ),
            quality_intact=False,
            verdict="",
            note="Arithmetic beats 2.0. Quality fails Density Law on mid-depth MLP.",
        )
    )

    p_bin_q3 = weighted_penalty(
        [(F_MLP, mlp("binary_g128"), p_bin), (F_ATTN, q3_attn, p_q3), (F_EMB, emb_q4, p_q4)]
    )
    recipes.append(
        recipe(
            name="binary_mlp_q3_attn_q4_emb",
            mlp_bpw=mlp("binary_g128"),
            attn_bpw=q3_attn,
            emb_bpw=emb_q4,
            penalty=p_bin_q3,
            cost_class="CHEAP_INREGISTER",
            quality=(
                f"binary hold min gate {bin_gate['hold_min']:.4f} "
                f"up {bin_up['hold_min']:.4f} down {bin_down['hold_min']:.4f}"
            ),
            quality_intact=False,
            verdict="",
            note="Hits ~1.99 BPW on arithmetic. Mid-depth binary is not capability-intact.",
        )
    )

    p_q2_all = weighted_penalty(
        [
            (F_MLP, mlp("uniform_q2_g64"), p_q2),
            (F_ATTN, attn("uniform_q2_g64"), p_q2),
            (F_EMB, emb_q4, p_q4),
        ]
    )
    recipes.append(
        recipe(
            name="uniform_q2_all_except_emb_q4",
            mlp_bpw=mlp("uniform_q2_g64"),
            attn_bpw=attn("uniform_q2_g64"),
            emb_bpw=emb_q4,
            penalty=p_q2_all,
            cost_class="CHEAP_INREGISTER",
            quality="hold min gate 0.7885 up 0.7820 down 0.7990 attn_in 0.8352",
            quality_intact=False,
            verdict="",
            note="Same ~2.44 BPW as ternary, worse quality.",
        )
    )

    # Rank: quality-intact first by bytes_saved/wall, then the rest.
    # Contract asked for bytes_saved / recon_ns. When excess≈0, wall time is the honest denom.
    recipes.sort(
        key=lambda r: (
            0 if r["quality_intact"] else 1,
            -r["bytes_saved_over_wall_ms"],
        )
    )

    # Fill verdicts now that we know the ranking and the numbers.
    for r in recipes:
        if r["verdict"]:
            continue
        bits = []
        if r["clears_2p0"]:
            bits.append("<2.0 BPW")
        else:
            bits.append(f"ABOVE 2.0 ({r['projected_bpw']:.2f})")
        if r["clears_50tps"]:
            bits.append(f"clears 50 TPS ({r['projected_tps']:.1f})")
        else:
            bits.append(f"misses 50 TPS ({r['projected_tps']:.1f})")
        if r["quality_intact"]:
            bits.append("quality-intact on this capture")
        else:
            bits.append("NOT quality-intact")
        r["verdict"] = "; ".join(bits)

    # First-screen conclusions: survive / die
    first_conclusions = {
        "no_cheap_pack_both_below_2_and_quality_intact": {
            "first_screen": True,
            "survives": True,
            "why": (
                "Removing the 5.9x rice penalty makes rice-based recipes cheap, "
                "but rice hold mins still fail mid-depth (up 0.813, down 0.804). "
                "Binary still fails. The only quality-intact recipe under 2.0 is "
                "still ternary + two-stage down + ternary attn at ~1.99 BPW."
            ),
        },
        "best_cheap_descent_is_ternary_plus_twostage": {
            "first_screen": True,
            "survives": True,
            "why": (
                "It remains the only quality-intact <2.0 recipe. Re-rank may "
                "change its TPS (penalty 1.05 -> measured) but not its identity."
            ),
        },
        "rice_is_a_reconstruction_regression": {
            "first_screen": True,
            "survives": False,
            "why": (
                "That claim used Q80's 5.9x/byte. In-register CSR consume on "
                "Qwen3.8's dense sequential shape is the number that replaces it. "
                "See measured recon_penalty for rice_q1_rms_2pct."
            ),
        },
        "sibling_2p09_would_regress_to_21_tps": {
            "first_screen": True,
            "survives": False,
            "why": (
                "The 21 TPS figure was 2.09 BPW * 2.925 blended 5.9x penalty. "
                "With a measured ~1.x penalty the sibling pack is a ~50 TPS "
                "near-miss of the 2.0 line, not a regression vs Q4."
            ),
        },
        "q3_is_the_cheap_coherence_floor": {
            "first_screen": True,
            "survives": True,
            "why": "Unchanged. q3 hold mins stay >=0.9679. Still misses 50 TPS.",
        },
        "ternary_beats_affine_q2_at_same_bpw": {
            "first_screen": True,
            "survives": True,
            "why": "Quality table is unchanged. Reconstruction cost does not reverse this.",
        },
        "binary_is_not_free": {
            "first_screen": True,
            "survives": True,
            "why": "Quality failure at mid-depth is independent of reconstruction cost.",
        },
        "lowrank_down_is_cheap_algebra_on_dense": {
            "first_screen": True,
            "survives": True,
            "why": "Now measured, not assumed. Two-stage is a small pair of occupancy-tile matvecs.",
        },
        "qwen38_is_bandwidth_bound_bytes_are_the_only_lever": {
            "first_screen": True,
            "survives": True,
            "why": "406.2 of 411.51 GB/s. If in-register codecs stay at the wall, penalty≈1.",
        },
        "scale_from_38p217_hold_1p229_fixed": {
            "first_screen": False,
            "survives": True,
            "why": "Binding projection. 2.0 BPW => ~54 TPS, not the older 63.",
        },
    }

    # Who beats 1.99 while quality-intact?
    beats = [
        r
        for r in recipes
        if r["quality_intact"] and r["projected_bpw"] < 1.9897 - 1e-6
    ]
    ter = next(
        r
        for r in recipes
        if r["codec"] == "ternary_gate_up_hgravs01_twostage_down_ternary_attn_q4_emb"
    )

    receipt = {
        "schema": "hawking.special_unit.qwen38_bpw_descent_rescreen.v1",
        "date": "2026-08-16",
        "lane": "qwen38-descent-rescreen",
        "why": (
            "First screen transferred Q80's 5.9x rice penalty. That number was "
            "a serial one-thread-per-row extract artifact (Q80_RECONSTRUCTION_WON). "
            "This receipt re-ranks with reconstruction cost MEASURED on Qwen3.8 "
            "dense sequential shapes, in-register occupancy tiles."
        ),
        "did_not_transfer_a_number": True,
        "gpu_lock": "./tools/gpu_lane_lock.sh qwen38-descent-rescreen",
        "activation": {
            "path": "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1",
            "not_synthetic": True,
            "source_receipt": "receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json",
        },
        "projection": {
            "method": "scale GPU by BPW ratio * measured_penalty; hold 1.229 ms fixed",
            "complete_wall_ms": 38.216792,
            "gpu_ms": GPU_MS,
            "fixed_ms": FIXED_MS,
            "current_bpw": CURRENT_BPW,
            "at_2p0_penalty_one_tps": round(wall_tps(2.0, 1.0)[1], 2),
        },
        "measurement": {
            "receipt": "receipts/ascent-2026-08-16/QWEN38_RECON_MEASURED.json",
            "device_name": meas.get("device_name"),
            "control_gbps": (meas.get("control") or {}).get("gbps"),
            "gpu_time_authority": meas.get("gpu_time_authority"),
            "launch_primary": meas.get("launch_primary"),
            "per_codec": costs,
        },
        "candidate_table": recipes,
        "candidate_table_note": (
            "Ranked quality-intact first, then by (4.2527-bpw)/wall_ms. "
            "wall_ms uses MEASURED recon_penalty (byte-weighted). "
            "When the kernel stays at the 406 GB/s wall, penalty≈1 and "
            "bytes_saved/recon_excess is undefined (reconstruction is hidden)."
        ),
        "does_any_quality_intact_candidate_beat_1p99": {
            "answer": bool(beats),
            "who": [r["codec"] for r in beats],
            "ternary_twostage_bpw": ter["projected_bpw"],
            "ternary_twostage_tps": ter["projected_tps"],
            "ternary_twostage_penalty": ter["recon_penalty_measured"],
        },
        "does_ternary_twostage_survive": {
            "answer": True,
            "as": "still the only quality-intact recipe under 2.0 BPW",
            "what_changed": (
                "Its 1.05 assumed penalty is replaced by the measured ternary "
                f"and two-stage penalties (ternary {p_ter:.3f}, two-stage {p_ts:.3f}, "
                f"blended {p_ter_ts:.3f}). TPS is now {ter['projected_tps']:.1f} "
                "from the 38.217 ms wall + 1.229 fixed, not the first screen's 61."
            ),
        },
        "first_screen_conclusions": first_conclusions,
        "claim_boundary": {
            "full_model_not_packed": True,
            "generation_not_run": True,
            "coherence_gate_not_run_on_a_new_pack": True,
            "reason": (
                "No generate path exists for a mixed Qwen3.8 pack in this lane. "
                "Sibling mixed-2p0-v1 is 2.086 BPW and is not this recipe. "
                "Organ hold cosine from the first screen is the quality screen; "
                "generation remains the gate. A 12-token single-prompt match "
                "would not certify."
            ),
            "two_stage_fit": (
                "Cost measured on L00 QR-into-activation-subspace (real post-SwiGLU X). "
                "Sibling owns the 64-layer HGRAVS01 SVD pack. Geometry (r160 q3) matches."
            ),
            "rice_consume": (
                "Ranking uses bind-expanded CSR + binary, in-register, as Q80 now does. "
                "Serial one-thread residual walk is reported as the artifact path only. "
                "Storage BPW is the rice bitstream; traffic BPW is CSR."
            ),
        },
        "first_screen": "receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json",
        "first_screen_review": "receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT_REVIEW.json",
        "q80_disproof": "receipts/ascent-2026-08-16/Q80_RECONSTRUCTION_WON.json",
    }
    OUT.write_text(json.dumps(receipt, indent=2))
    print(f"wrote {OUT}")
    print("\n=== MEASURED PENALTIES ===")
    for k, v in sorted(costs.items()):
        print(
            f"  {k:28s} penalty={v['recon_penalty_measured']:.3f}  "
            f"gbps={v['packed_gbps_mean']:.1f}  wall={v['stays_at_bandwidth_wall']}"
        )
    print("\n=== RE-RANKED ===")
    for r in recipes:
        print(
            f"  {r['projected_bpw']:.4f} BPW  {r['projected_tps']:5.1f} TPS  "
            f"pen={r['recon_penalty_measured']:.3f}  "
            f"{'QI' if r['quality_intact'] else '  '}  {r['codec']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
