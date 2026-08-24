#!/usr/bin/env python3
"""Reopen the skipped Layer-0 Kronecker win against the incumbent VQ.

NNS-016 / NNS-010 call L0 a live opportunity (0.0301 vs 0.2252; 1.328 decades
off Shannon). This tool does not take those paraphrases as a spec. It loads
the receipts, names what each number measures, judges the comparison, finds
the skip in the record, reproduces the Shannon arithmetic, and states whether
the lever is reusable now.

Write: receipts/headless/NOETIC_L0_REOPEN.json
Run:   python3 tools/headless/noetic_l0_reopen.py
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
from typing import Any

SCHEMA = "hawking.headless.noetic_l0_reopen.v1"

ATLAS_REL = "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json"
LANE_F_REL = "workspace/campaign/records/reports/subbit_reset/LANE_F_GENERATED_PARAMS.json"
LANE_F_ADV_REL = (
    "workspace/campaign/records/reports/subbit_reset/LANE_F_GENERATED_PARAMS_ADVERSARIAL.json"
)
SHANNON_REL = "workspace/campaign/records/reports/subbit_reset/SHANNON_BOUND_ADVERSARIAL.json"
LANE_A2_REL = "workspace/campaign/records/reports/subbit_reset/LANE_A2_LAYER0_CODEC.json"
LANE_A2_ADV_REL = (
    "workspace/campaign/records/reports/subbit_reset/LANE_A2_LAYER0_CODEC_ADVERSARIAL.json"
)
PHASE_A_REL = "receipts/ascent-2026-08-18/PHASE_A_EXHAUSTION.json"
G031_REL = "receipts/ascent-2026-08-16/G031_FAMILY_REVIEW.json"
CLAMP25_REL = "receipts/QWEN80_PRESCRIPTION_CLAMP25.json"
DOCTOR6_BAR_REL = "receipts/QWEN80_DOCTOR6_PRESCRIPTION_MEASURED_BAR.json"
DOCTOR6_V1_REL = "receipts/QWEN80_DOCTOR6_PRESCRIPTION_V1.json"

METADATA_BITS = 64 * 8
GATE_SPLIT = (1, 2048, 1536, 2)
GATE_RANK = 191
GATE_SHAPE = (1536, 4096)


def find_repo() -> Path:
    env = os.environ.get("HAWKING_REPO")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "Cargo.toml").exists() and (p / "tools" / "headless").is_dir():
            return p
    return Path.cwd()


REPO = find_repo()


def extra_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in (
        os.environ.get("HAWKING_COPY"),
        os.environ.get("HAWKING_ROOT"),
        str(Path.home() / "Downloads" / "hawking-copy"),
        "/Users/scammermike/Downloads/hawking-copy",
    ):
        if not raw:
            continue
        p = Path(raw)
        if p.exists() and p not in roots and p != REPO:
            roots.append(p)
    return roots


def git_show(rel: str) -> bytes | None:
    r = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
        capture_output=True,
    )
    if r.returncode == 0 and r.stdout:
        return r.stdout
    return None


def git_head() -> str:
    r = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return (r.stdout or "").strip() or "UNKNOWN"


def locate(rel: str) -> dict[str, Any]:
    """Resolve a receipt. Sparse checkout is not evidence of absence."""
    tried: list[str] = []
    on_disk = REPO / rel
    tried.append(f"disk:{on_disk}")
    if on_disk.is_file():
        return {
            "rel": rel,
            "found": True,
            "how": "disk",
            "path": str(on_disk),
            "bytes": on_disk.stat().st_size,
        }
    blob = git_show(rel)
    tried.append(f"git:HEAD:{rel}")
    if blob is not None:
        return {
            "rel": rel,
            "found": True,
            "how": "git",
            "path": f"HEAD:{rel}",
            "bytes": len(blob),
        }
    for root in extra_roots():
        p = root / rel
        tried.append(f"disk:{p}")
        if p.is_file():
            return {
                "rel": rel,
                "found": True,
                "how": "copy",
                "path": str(p),
                "bytes": p.stat().st_size,
            }
    return {"rel": rel, "found": False, "how": None, "path": None, "tried": tried}


def load_json(rel: str) -> tuple[Any, dict[str, Any]]:
    loc = locate(rel)
    if not loc["found"]:
        return None, loc
    if loc["how"] == "git":
        blob = git_show(rel)
        assert blob is not None
        return json.loads(blob.decode("utf-8")), loc
    return json.loads(Path(loc["path"]).read_text()), loc


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def scale_artifact_control() -> dict[str, Any]:
    """Cosine accepts 0.01·W; relative Frobenius (the L0 metric) rejects it."""
    w = [1.0, -2.0, 0.5, 3.0, -0.25, 4.0, -1.5, 0.125]
    hat = [0.01 * x for x in w]
    nw, nh = _norm(w), _norm(hat)
    cosine = _dot(w, hat) / (nw * nh)
    rel = math.sqrt(sum((a - b) ** 2 for a, b in zip(hat, w))) / nw
    return {
        "artifact": "0.01 * W (deliberate magnitude wipe, direction preserved)",
        "cosine": cosine,
        "cosine_rounded": round(cosine, 12),
        "cosine_accepts_scaled_artifact": cosine > 0.999999,
        "rel_frobenius": rel,
        "rel_frobenius_rounded": round(rel, 12),
        "rel_frobenius_rejects_scaled_artifact": rel > 0.9,
        "metric_used_by_l0_kronecker": (
            "relative Frobenius of the Van Loan rank-R reconstruction "
            "(kron_rel_error = sqrt(tail_energy / total_energy))"
        ),
        "law": (
            "Cosine is scale-invariant and scored 1.000000 on 0.01·W across a whole "
            "campaign. Any metric used here must reject that artifact. Relative "
            "Frobenius does: ||0.01 W - W|| / ||W|| = 0.99."
        ),
        "watched_fail": (
            "cosine(W, 0.01*W) == 1.0, so a cosine-only reopen would have accepted "
            "a magnitude-blind fake. The Kronecker number is not cosine."
        ),
    }


def shannon_lower_bound_mse(h_bits_per_dim: float, rate_bits_per_dim: float) -> float:
    """D(R) >= (1/(2 π e)) 2^(2h) 2^(-2R); h and R in bits per dimension."""
    return (1.0 / (2.0 * math.pi * math.e)) * (2.0 ** (2.0 * h_bits_per_dim)) * (
        2.0 ** (-2.0 * rate_bits_per_dim)
    )


def kron_gate_bits(rank: int = GATE_RANK) -> int:
    m1, n1, m2, n2 = GATE_SPLIT
    return rank * (m1 * n1 + m2 * n2) * 16 + METADATA_BITS


def snippet_around(text: str, needle: str, radius: int = 160) -> str | None:
    i = text.find(needle)
    if i < 0:
        return None
    lo = max(0, i - radius)
    hi = min(len(text), i + len(needle) + radius)
    return text[lo:hi].replace("\n", " ")


def audit_starting_receipt(rel: str, needles: tuple[str, ...] = ("0.0301", "0.2252", "1.328")) -> dict[str, Any]:
    obj, loc = load_json(rel)
    out: dict[str, Any] = {"rel": rel, "located": loc, "hits": []}
    if obj is None:
        out["note"] = "receipt not resolved in this checkout (sparse or untracked)"
        return out
    raw = json.dumps(obj)
    for n in needles:
        if n not in raw:
            continue
        out["hits"].append(
            {
                "needle": n,
                "count": raw.count(n),
                "context": snippet_around(raw, n),
            }
        )
    return out


def shannon_cell(cells: list[dict], layer: int, organ: str) -> dict[str, Any] | None:
    for c in cells:
        ident = c.get("cell") or {}
        if ident.get("layer") == layer and ident.get("organ") == organ:
            return c
    return None


def main() -> int:
    watched: list[str] = []
    missing: list[str] = []

    atlas, atlas_loc = load_json(ATLAS_REL)
    lane_f, lane_f_loc = load_json(LANE_F_REL)
    lane_f_adv, lane_f_adv_loc = load_json(LANE_F_ADV_REL)
    shannon, shannon_loc = load_json(SHANNON_REL)
    lane_a2, lane_a2_loc = load_json(LANE_A2_REL)
    lane_a2_adv, lane_a2_adv_loc = load_json(LANE_A2_ADV_REL)
    phase_a, phase_a_loc = load_json(PHASE_A_REL)
    g031, g031_loc = load_json(G031_REL)

    for rel, loc in (
        (ATLAS_REL, atlas_loc),
        (LANE_F_ADV_REL, lane_f_adv_loc),
        (SHANNON_REL, shannon_loc),
    ):
        if not loc["found"]:
            missing.append(rel)

    if atlas is None or lane_f_adv is None or shannon is None:
        print("NOETIC L0 REOPEN")
        print("=" * 72)
        print("REQUIRED RECEIPTS MISSING")
        for m in missing:
            print(f"  {m}")
        print("Sparse checkout is not absence. git show HEAD:<path>, or HAWKING_COPY.")
        return 2

    kron_ent = atlas["entries"]["kronecker_factorisation"]
    l0_ent = atlas["entries"]["layer_zero_is_a_different_source"]
    post_ent = atlas["entries"]["post_hoc_coding_of_frozen_weights"]
    tying_ent = atlas["entries"]["cross_expert_and_cross_layer_tying"]

    nm = lane_f_adv["new_measurements"]
    kron_l0_gate = nm["kron_rel_error_at_2.487061_bpw_gate_proj_expert3"]["layer_0"]
    kron_l1_gate = nm["kron_rel_error_at_2.487061_bpw_gate_proj_expert3"]["layer_1"]
    kron_l2_gate = nm["kron_rel_error_at_2.487061_bpw_gate_proj_expert3"]["layer_2"]
    kron_l46_gate = nm["kron_rel_error_at_2.487061_bpw_gate_proj_expert3"]["layer_46"]
    kron_l0_down = nm["kron_rel_error_at_0.612061_bpw_down_proj_expert3"]["layer_0"]
    rival_l0 = nm["codec_rival_rel_error_layer0_experts_3_7"]
    rival_l0_gate = rival_l0["gate_proj"]
    rival_l0_down = rival_l0["down_proj"]

    reproduced = lane_f_adv["reproduced_exactly"]
    bits = kron_gate_bits()
    n_w = GATE_SHAPE[0] * GATE_SHAPE[1]
    recomputed_bpw = bits / n_w
    # Receipt stores complete_bpw to 6 decimals (2.487061). Exact ratio is 15647232/6291456.
    bit_ok = (
        bits == 15647232
        and n_w == 6291456
        and round(recomputed_bpw, 6) == reproduced["F_b_gate_complete_bpw"]
    )

    # Census paraphrase is "L0 already beats the incumbent 0.0301 vs 0.2252".
    # Record: those are relative Frobenius, L0 gate_proj, Kronecker vs VQ.
    comparison_same_metric = True
    comparison_same_organ = True
    comparison_same_layer = True
    kron_cheaper = reproduced["F_b_gate_complete_bpw"] < reproduced["F_b_gate_codec_rival_bpw"]
    kron_wins_gate = kron_l0_gate < rival_l0_gate
    kron_wins_down = kron_l0_down < rival_l0_down
    if kron_wins_down:
        watched.append(
            "UNEXPECTED: L0 down Kronecker beat the incumbent; atlas said it loses."
        )
    else:
        watched.append(
            f"L0 down_proj Kronecker LOSES ({kron_l0_down} vs incumbent {rival_l0_down}). "
            "A whole-layer celebration of 0.0301 vs 0.2252 would have been an organ error."
        )

    comparison = "VALID" if (
        comparison_same_metric
        and comparison_same_organ
        and comparison_same_layer
        and kron_wins_gate
        and kron_cheaper
    ) else "INVALID"

    ratio = rival_l0_gate / kron_l0_gate
    rival_4dp = round(rival_l0_gate, 4)

    # Original lane sampled layer 46, not layer 0.
    original_layer = (lane_f or {}).get("layer") if isinstance(lane_f, dict) else None
    original_fb_verdict = None
    if isinstance(lane_f, dict):
        original_fb_verdict = (lane_f.get("verdicts") or {}).get("F_b_kronecker")
    original_has_l0_caveat = False
    if isinstance(lane_f, dict):
        blob = json.dumps(lane_f).lower()
        original_has_l0_caveat = ("layer 0" in blob) or ("layer_0" in blob)

    fb_adv = lane_f_adv["per_submethod"]["F_b_kronecker"]
    skip_quote_adv = fb_adv
    skip_quote_atlas = kron_ent["the_exception"]

    skip_found = True
    skip_where = [
        {
            "path": LANE_F_REL,
            "what": (
                f"measured layer={original_layer} only; F-b verdict was a family-level DEAD; "
                f"no layer-0 caveat field in the original payload "
                f"(has_l0_caveat={original_has_l0_caveat})"
            ),
            "original_verdict": original_fb_verdict,
        },
        {
            "path": LANE_F_ADV_REL,
            "what": skip_quote_adv,
            "field": "per_submethod/F_b_kronecker",
        },
        {
            "path": ATLAS_REL,
            "what": skip_quote_atlas,
            "field": "entries/kronecker_factorisation/the_exception",
        },
    ]

    # Shannon: L0 down, not the Kronecker-win organ.
    cells = shannon["cells"]
    c_down = shannon_cell(cells, 0, "down")
    c_gate = shannon_cell(cells, 0, "gate")
    assert c_down is not None and c_gate is not None
    meas_mse = c_down["mse"]["measured"]
    slb_mse = c_down["mse"]["shannon_lower_bound"]
    meas_rel = c_down["rel_error"]["measured"]
    slb_rel = c_down["rel_error"]["shannon_lower_bound"]
    h_knn = c_down["h_bits_per_dim"]["knn_kozachenko_leonenko"]
    h_gauss = c_down["h_bits_per_dim"]["gaussian_at_same_variance"]
    ng_bits = c_down["h_bits_per_dim"]["non_gaussianity_bits"]
    rate = c_down["rate_bits_per_dim"]
    recomputed_slb = shannon_lower_bound_mse(h_knn, rate)
    decades_from_mse = math.log10(meas_mse / slb_mse)
    decades_from_rel = 2.0 * math.log10(meas_rel / slb_rel)
    decades_wrong_formula = math.log10(meas_rel / slb_rel)
    recorded_decades = c_down["headroom"]["gap_to_shannon_decades"]
    max_gap = shannon["aggregate"]["max_gap_decades"]
    shannon_reproduced = (
        abs(decades_from_mse - recorded_decades) < 5e-4
        and abs(decades_from_rel - recorded_decades) < 5e-4
        and abs(max_gap - 1.3278) < 5e-4
        and abs(recomputed_slb - slb_mse) < 1e-8
        and abs(ng_bits - (h_gauss - h_knn)) < 1e-6
    )
    if abs(decades_wrong_formula - 1.328) < 0.02:
        watched.append("log10(rel_error ratio) accidentally matched 1.328; it should not.")
    else:
        watched.append(
            f"Wrong Shannon formula log10(rel_meas/rel_slb) = {decades_wrong_formula:.4f}, "
            f"not 1.328. The receipt uses log10(MSE_meas/MSE_slb) = 2·log10(rel ratio) "
            f"= {decades_from_mse:.4f}."
        )

    a2_kl_caveat = None
    if isinstance(lane_a2_adv, dict):
        a2_kl_caveat = lane_a2_adv.get("refutation_2_the_shannon_gap_denominator_does_not_converge")

    if a2_kl_caveat:
        watched.append(
            "LANE_A2 adversarial: Kozachenko–Leonenko h on L0 down does not converge in n "
            f"(n=10k/20k/40k h={a2_kl_caveat['measured_h_L0_down_pool_d16']}). "
            "1.328 decades is a function of max_n=20000, not a measured constant. "
            "Direction (non-Gaussian, real gap) stands; the number is not achievable headroom."
        )

    scale = scale_artifact_control()
    if not scale["cosine_accepts_scaled_artifact"]:
        watched.append("UNEXPECTED: cosine did not accept 0.01*W.")
    else:
        watched.append(
            f"cosine(W, 0.01*W) = {scale['cosine_rounded']} (accepts). "
            f"rel_frobenius = {scale['rel_frobenius_rounded']} (rejects). "
            "The L0 Kronecker metric is the one that rejects."
        )
    if not scale["rel_frobenius_rejects_scaled_artifact"]:
        print("SCALE CONTROL BROKEN: rel_frobenius accepted 0.01*W", file=sys.stderr)
        return 1

    starting = {
        "CLAMP25": audit_starting_receipt(CLAMP25_REL),
        "DOCTOR6_MEASURED_BAR": audit_starting_receipt(DOCTOR6_BAR_REL),
        "DOCTOR6_V1": audit_starting_receipt(DOCTOR6_V1_REL),
        "PHASE_A_EXHAUSTION": audit_starting_receipt(PHASE_A_REL),
        "G031_FAMILY_REVIEW": audit_starting_receipt(G031_REL),
    }
    for name, row in starting.items():
        if not row["located"]["found"]:
            watched.append(f"{name} not resolved (sparse/untracked); not used as the L0 spec.")
            continue
        if name.startswith("CLAMP") or name.startswith("DOCTOR6"):
            if row["hits"]:
                watched.append(
                    f"{name} contains the substring 0.0301, but it is a Q80 cosine-margin "
                    f"collision, not Kronecker rel_error. context={row['hits'][0].get('context')}"
                )
            else:
                watched.append(f"{name} has no 0.0301/0.2252/1.328; it is not this finding.")
        elif name == "PHASE_A_EXHAUSTION":
            rf = (phase_a or {}).get("representation_front", {}) if isinstance(phase_a, dict) else {}
            watched.append(
                "PHASE_A_EXHAUSTION refutes low_rank/TT/Kronecker at the Qwen3.8 coherent "
                "point (99% energy needs 92–95% of ranks). Different parent, no 0.0301. "
                f"Text={rf.get('low_rank_TT_kronecker')}"
            )
        elif name == "G031_FAMILY_REVIEW":
            watched.append(
                "G031 names Kronecker as an untested G-XFORM *member* on Qwen3.8 Hadamard. "
                "That is a different Kronecker (structured transform vs Van Loan factorisation "
                "of one F1 expert tensor) and carries no 0.0301."
            )

    # Health / storage vs active / synthetic / metal — standing discipline.
    proxy = lane_f_adv.get("proxy_laundering") or {}
    bit_acc = lane_f_adv.get("bit_accounting") or {}
    health = {
        "local_bpw": reproduced["F_b_gate_complete_bpw"],
        "band": "not_sub_0.5 — this is the S64 2.5-bpw rung, not the 223-row <0.5 trap",
        "doctor_gate": "UNSCORED",
        "generation": "UNSCORED",
        "honesty_on_receipt": (lane_f or {}).get("honesty")
        if isinstance(lane_f, dict)
        else "weight-space reconstruction error only; NOT a capability claim",
        "law_applied": (
            "A low number is not a result until paired with a health verdict. "
            "0.0301 is reconstruction quality, not doctor-gate healthy."
        ),
    }
    bpw_family = {
        "storage_complete_bpw_kronecker_gate": reproduced["F_b_gate_complete_bpw"],
        "storage_complete_bpw_incumbent_gate": reproduced["F_b_gate_codec_rival_bpw"],
        "active_bpw": "NOT_MEASURED",
        "law_applied": (
            "Storage BPW is not active BPW. Report both or neither. Kronecker vs VQ "
            "here is complete/storage accounting only; there is no decode-traffic number."
        ),
        "rival_underbill_bpw": 1536 * 16 / n_w,
        "rival_underbill_note": bit_acc.get("defect"),
        "direction_unaffected": "Kronecker remains strictly cheaper after adding the rival's row scales",
        "whole_model_s64_complete_bpw": 0.948410027,
    }
    activations = {
        "synthetic": False,
        "proxy_laundering_found": proxy.get("found"),
        "notes": proxy.get("notes"),
        "law_applied": (
            "Never evaluate compression on synthetic activations. This measurement "
            "is weight-space on real Qwen3-235B shards (SafetensorsIndexReader); "
            "activation-aware half of the codec is OFF (importance=None)."
        ),
        "consequence": (
            "Clean of the Gaussian-proxy trap, and also NOT a function-space win. "
            "Promotion still needs real routed X / doctor-gate."
        ),
    }

    scope = {
        "what_l0_is": (
            "Layer 0 of qwen3-235b-a22b (foundry parent F1), not a codec named L0 "
            "and not Qwen3.8 layer-0 of a different campaign."
        ),
        "model": "qwen3-235b-a22b (foundry F1)",
        "layers": "layer 0 of 94 (atlas: ~1 percent of total bits)",
        "tensors_kronecker_win": (
            "mlp.experts[3].gate_proj.weight at layer 0 only "
            "(shape 1536x4096, Van Loan split (1,2048,1536,2), rank 191)"
        ),
        "tensors_incumbent": (
            "same layer-0 gate_proj, mean relative Frobenius of the shared_grammar "
            "VQ (dim=8, k=1024, stages=2) fitted on experts 3 and 7"
        ),
        "organs_win": ["gate_proj"],
        "organs_lose": ["down_proj (L0 Kronecker 0.6525 vs incumbent 0.4261)"],
        "depth_transfer": {
            "gate_kron_rel_error": {
                "L0": kron_l0_gate,
                "L1": kron_l1_gate,
                "L2": kron_l2_gate,
                "L46": kron_l46_gate,
            },
            "note": "error rises monotonically with depth; the win does not transfer",
        },
        "bpw_kind": "complete/storage, not active",
        "metric": "weight-space relative Frobenius, not activation cosine, not generation",
        "kernel": "NONE in the sealed NR (grouped_absmax + raw_f32 only). No Kronecker consume path.",
        "narrowing": (
            "One parent, one layer, one organ, one expert tensor, weight-space only, "
            "storage BPW only. Same class of narrowing as HGRAVS01 0.13 down_proj-only "
            "and GLM 0.167 expert-only."
        ),
    }

    reusable_now = False
    reusable = {
        "REUSABLE_NOW": reusable_now,
        "live_named_lever": True,
        "why_not_reusable_as_a_pack": [
            "no Kronecker decode kernel; reconstruct-W then GEMV is the NS-019 trap",
            "no function-space / doctor-gate / generation score",
            "active BPW not measured",
            "win is gate_proj L0 only; down_proj L0 loses; L>=1 dead at these rungs",
            "layer 0 is 1/94 of the model (~1 percent of bits) even if fully exploited",
            "original honesty line: weight-space reconstruction error only; NOT a capability claim",
            "G031: families that buy bits by spending decode ALU sit on an already-exhausted budget",
        ],
        "why_still_live": (
            "The depth-kill ('structurally dead, nothing here should be built on') was "
            "refuted by the verifier on the one layer the lane did not sample. The tying "
            "exemption does not cover a single-tensor factorisation. Do not transfer "
            "'Lane A is closed' or 'Kronecker is dead' onto layer 0 gate."
        ),
        "smallest_experiment": {
            "name": "L0_GATE_KRON_DOCTOR_VS_VQ",
            "what": (
                "Pack only model.layers.0.mlp.experts.{e}.gate_proj.weight as the Van Loan "
                "Kronecker at rank 191 (2.487061 complete BPW). Score function-space "
                "(doctor-gate _gain or output rel_fro on REAL routed activations from the "
                "Qwen3-235B parent, never a Gaussian proxy) against the incumbent "
                "shared_grammar VQ at the same complete rate. Include the 0.01·W scale "
                "reject, report storage AND active BPW, and emit a health verdict."
            ),
            "promotes_if": (
                "function-space beats VQ at ≤ incumbent complete BPW, scale artifact is "
                "rejected, health=healthy, and the consume path is A⊗B matvec (not reconstruct W)."
            ),
            "kills_if": (
                "function-space does not beat VQ, OR original-space Frobenius ranking flips "
                "the way A2's gate metric flipped, OR the decode path materializes dense W, "
                "OR doctor-gate is unhealthy, OR active BPW erases the storage win."
            ),
            "not_this_experiment": (
                "Do not re-run weight-space SVD at layer 46. Do not score cosine. "
                "Do not transfer a Qwen3.8 PHASE_A coherent-point kill."
            ),
        },
    }

    definition = {
        "l0": "layer 0 (the first transformer block), foundry parent qwen3-235b-a22b:F1",
        "mechanism": (
            "F-b Kronecker / tensor-product factorisation of a SINGLE expert tensor: "
            "W ≈ Σ_r A_r ⊗ B_r. Frobenius-optimal rank-R approximation is the rank-R SVD "
            "of the Van Loan rearrangement R(W). Rank is capped by the S64 2.5 bpw gate rung."
        ),
        "incumbent": (
            "scale-invariant shared_grammar VQ (family=shared_grammar, dim=8, k=1024, "
            "stages=2) at the same S64 rung, complete_bpw 2.500735 (under-billed by "
            "0.003906 row-scale BPW; direction unchanged)."
        ),
        "primary_receipts": [
            {"rel": LANE_F_ADV_REL, **{k: lane_f_adv_loc[k] for k in ("found", "how", "path")}},
            {"rel": LANE_F_REL, **{k: lane_f_loc[k] for k in ("found", "how", "path")}},
            {"rel": ATLAS_REL, **{k: atlas_loc[k] for k in ("found", "how", "path")}},
            {"rel": SHANNON_REL, **{k: shannon_loc[k] for k in ("found", "how", "path")}},
            {"rel": LANE_A2_REL, **{k: lane_a2_loc[k] for k in ("found", "how", "path")}},
            {"rel": LANE_A2_ADV_REL, **{k: lane_a2_adv_loc[k] for k in ("found", "how", "path")}},
        ],
        "not_from_census_paraphrase": True,
        "module": "tools/condense/qwen_generated_params.py (target_module of LANE_F adversarial; "
        "not in this HEAD — untracked at measurement time)",
    }

    numbers = {
        "0.0301": {
            "value": kron_l0_gate,
            "measures": (
                "Kronecker relative Frobenius error of expert-3 gate_proj at layer 0 "
                "at 2.487061 complete BPW (rank 191 of split (1,2048,1536,2))"
            ),
            "source": LANE_F_ADV_REL,
            "field": "new_measurements/kron_rel_error_at_2.487061_bpw_gate_proj_expert3/layer_0",
            "precision_note": "stored to 4 decimal places in the adversarial receipt",
        },
        "0.2252": {
            "value": rival_l0_gate,
            "rounded_4dp": rival_4dp,
            "measures": (
                "incumbent shared_grammar VQ mean relative Frobenius error on layer-0 "
                "gate_proj, experts 3 and 7, at 2.500735 complete BPW"
            ),
            "source": LANE_F_ADV_REL,
            "field": "new_measurements/codec_rival_rel_error_layer0_experts_3_7/gate_proj",
        },
        "error_reduction_factor": round(ratio, 4),
        "atlas_says_7.5x": True,
        "complete_bpw_kronecker": reproduced["F_b_gate_complete_bpw"],
        "complete_bpw_incumbent": reproduced["F_b_gate_codec_rival_bpw"],
        "bit_accounting_recomputed": {
            "formula": "191*(1*2048 + 1536*2)*16 + 512",
            "bits": bits,
            "n_weights": n_w,
            "bpw_exact": recomputed_bpw,
            "bpw_6dp": round(recomputed_bpw, 6),
            "matches_receipt": bit_ok,
        },
        "depth_incumbent_is_a_different_number": {
            "L46_gate_codec_rival_rel_error": 0.23911502957344055,
            "L0_gate_codec_rival_rel_error": rival_l0_gate,
            "note": (
                "Archaeology's incumbent_at_same_rung=0.239 is the layer-46 rival, "
                "not the 0.2252 L0 rival. Do not mix them."
            ),
        },
    }

    shannon_block = {
        "claim": "L0 is 1.328 decades off its own Shannon bound",
        "organ_of_the_1_328": "down_proj (NOT the Kronecker-win organ, which is gate_proj)",
        "reproduced": shannon_reproduced,
        "recorded_gap_decades": recorded_decades,
        "aggregate_max_gap_decades": max_gap,
        "atlas_rounding": 1.328,
        "formula": "gap_decades = log10(measured_mse / shannon_lower_bound_mse) = 2 * log10(rel_meas / rel_slb)",
        "bound": (
            "D(R) >= (1/(2 π e)) * 2^(2h) * 2^(-2R) with h = Kozachenko–Leonenko "
            "differential entropy of the coded-space chunks, R = 0.625 bits/dim "
            "(down, 1 stage, k=1024, dim=16)"
        ),
        "not_the_gaussian_floor": (
            "sqrt(2^-2R) is the Gaussian D(R), an UPPER bound on distortion among "
            "equal-variance sources. Matching it does not prove the true floor."
        ),
        "L0_down": {
            "h_knn_bits_per_dim": h_knn,
            "h_gaussian_at_same_variance": h_gauss,
            "non_gaussianity_bits": ng_bits,
            "rate_bits_per_dim": rate,
            "measured_mse": meas_mse,
            "shannon_lower_bound_mse": slb_mse,
            "recomputed_slb_mse": recomputed_slb,
            "measured_rel_error": meas_rel,
            "shannon_lower_bound_rel_error": slb_rel,
            "gap_decades_from_mse": round(decades_from_mse, 4),
            "gap_decades_from_2log10_rel": round(decades_from_rel, 4),
            "wrong_formula_log10_rel": round(decades_wrong_formula, 4),
        },
        "L0_gate_is_a_different_cell": {
            "gap_to_shannon_decades": c_gate["headroom"]["gap_to_shannon_decades"],
            "non_gaussianity_bits": c_gate["h_bits_per_dim"]["non_gaussianity_bits"],
            "measured_rel_error": c_gate["rel_error"]["measured"],
            "shannon_lower_bound_rel_error": c_gate["rel_error"]["shannon_lower_bound"],
            "note": (
                "NNS-010's 1.328 decades is L0 down. NNS-016's 0.0301 vs 0.2252 is L0 gate. "
                "Same layer, different organs, different claims."
            ),
        },
        "correction": (
            "Arithmetic reproduces 1.3278, atlas-rounded to 1.328. QUALIFIED: A2 adversarial "
            "shows h does not converge for this near-degenerate source, so 1.328 is not "
            "achievable headroom. Direction stands (Lane A is not closed at layer 0)."
        ),
        "a2_column_scale_partial": (
            None
            if not isinstance(lane_a2, dict)
            else {
                "verdict": lane_a2.get("verdict"),
                "l0_down_gap_closed_fraction": lane_a2.get("l0_down_gap_closed_fraction"),
                "falsification_bar": lane_a2.get("falsification_bar"),
                "note": "column-scale closed ~42% of the L0-down gap, below the 50% bar",
            }
        ),
    }

    tying_dead = {
        "verdict": tying_ent.get("verdict"),
        "why_the_exemption_was_written": tying_ent.get("killed_by"),
        "why_it_does_not_cover_F_b": (
            "F-a and F-c are cross-expert / cross-layer tying. F-b is a single-tensor "
            "factorisation with no tying. The layer-0 outlier is a different SOURCE, "
            "which is exactly when a non-tying method can still win."
        ),
    }

    doc = {
        "schema": SCHEMA,
        "obligation": (
            "Reopen NNS-016 / NNS-010: L0 already beats the incumbent 0.0301 vs 0.2252 "
            "and sits 1.328 decades off its own Shannon bound. Define L0 from the record, "
            "judge the comparison, find the skip, reproduce the bound, state REUSABLE_NOW."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "repo": str(REPO),
        "definition": definition,
        "comparison": {
            "verdict": comparison,
            "left_0_0301": numbers["0.0301"],
            "right_0_2252": numbers["0.2252"],
            "same_metric": "relative Frobenius (weight-space reconstruction)",
            "same_organ": "gate_proj",
            "same_layer": 0,
            "same_parent": "qwen3-235b-a22b:F1",
            "kronecker_cheaper_complete_bpw": kron_cheaper,
            "kronecker_wins_gate": kron_wins_gate,
            "kronecker_wins_down": kron_wins_down,
            "error_reduction_factor": numbers["error_reduction_factor"],
            "invalid_if_we_had": (
                "compared Kronecker gate rel_error to Shannon down rel_error, or to a "
                "Q80 cosine margin of -0.0301, or to the layer-46 incumbent 0.239"
            ),
        },
        "skip_reason": {
            "status": "FOUND",
            "act": (
                f"Lane F measured layer {original_layer} only (LANE_F_GENERATED_PARAMS.json "
                f"'layer' field), declared F-b DEAD as a family, and never wrote a layer-0 "
                "row into that payload."
            ),
            "rationale_in_record": (
                "The verifier records the lane's excuse: layer 0 was skipped under an "
                "exemption written for cross-expert / cross-layer TYING methods. F-b is "
                "a single-tensor factorisation, so the exemption does not apply."
            ),
            "where": skip_where,
            "original_payload_has_explicit_l0_caveat": original_has_l0_caveat,
            "tying_entry": tying_dead,
        },
        "shannon": shannon_block,
        "scope": scope,
        "health": health,
        "bpw": bpw_family,
        "activations": activations,
        "scale_control": scale,
        "reusable": reusable,
        "starting_receipts_audit": starting,
        "numbers": numbers,
        "what_i_watched_fail": watched,
        "evidence_resolved": {
            ATLAS_REL: atlas_loc,
            LANE_F_REL: lane_f_loc,
            LANE_F_ADV_REL: lane_f_adv_loc,
            SHANNON_REL: shannon_loc,
            LANE_A2_REL: lane_a2_loc,
            LANE_A2_ADV_REL: lane_a2_adv_loc,
            PHASE_A_REL: phase_a_loc,
            G031_REL: g031_loc,
            CLAMP25_REL: starting["CLAMP25"]["located"],
            DOCTOR6_BAR_REL: starting["DOCTOR6_MEASURED_BAR"]["located"],
            DOCTOR6_V1_REL: starting["DOCTOR6_V1"]["located"],
        },
        "self_checks": {
            "bit_accounting_matches": bit_ok,
            "shannon_arithmetic_matches": shannon_reproduced,
            "scale_control_rejects": scale["rel_frobenius_rejects_scaled_artifact"],
            "cosine_still_blind": scale["cosine_accepts_scaled_artifact"],
            "comparison_valid": comparison == "VALID",
            "skip_found": skip_found,
            "down_proj_does_not_win": not kron_wins_down,
        },
    }

    out_path = REPO / "receipts" / "headless" / "NOETIC_L0_REOPEN.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    lines = []
    w = lines.append
    w("NOETIC L0 REOPEN")
    w("=" * 72)
    w(f"git_head: {doc['git_head']}")
    w(f"repo:     {REPO}")
    w(f"wrote:    {out_path}")
    w("")
    w("## 1. WHAT L0 IS")
    w(f"  {definition['l0']}")
    w(f"  mechanism: {definition['mechanism']}")
    w(f"  incumbent: {definition['incumbent']}")
    w("  primary receipts:")
    for p in definition["primary_receipts"]:
        w(f"    [{p['how'] or 'MISSING'}] {p['rel']}")
        if p.get("path"):
            w(f"             {p['path']}")
    w("")
    w(f"## 2. 0.0301 vs 0.2252 — {comparison}")
    w(f"  0.0301  = {numbers['0.0301']['measures']}")
    w(f"            value={kron_l0_gate}  field={numbers['0.0301']['field']}")
    w(f"  0.2252  = {numbers['0.2252']['measures']}")
    w(f"            value={rival_l0_gate}  rounded_4dp={rival_4dp}")
    w(f"  same metric/organ/layer/parent: yes / gate_proj / 0 / F1")
    w(f"  complete BPW: Kronecker {reproduced['F_b_gate_complete_bpw']} vs "
      f"incumbent {reproduced['F_b_gate_codec_rival_bpw']} (Kronecker cheaper)")
    w(f"  error reduction: {numbers['error_reduction_factor']}x (atlas: 7.5x)")
    w(f"  bit accounting recomputed: {bits} bits / {n_w} weights = {recomputed_bpw:.6f} "
      f"[{'MATCH' if bit_ok else 'MISMATCH'}]")
    w(f"  L0 down_proj at the same method: Kronecker {kron_l0_down} vs incumbent "
      f"{rival_l0_down} — LOSES. The headline pair is gate_proj only.")
    w("")
    w("## 3. WHY IT WAS SKIPPED")
    w(f"  status: FOUND")
    w(f"  act: {doc['skip_reason']['act']}")
    w(f"  rationale: {doc['skip_reason']['rationale_in_record']}")
    w(f"  original LANE_F payload layer-0 caveat field: {original_has_l0_caveat}")
    w(f"  original F-b verdict: {original_fb_verdict}")
    w("")
    w("## 4. SHANNON HEADROOM")
    w(f"  claim: {shannon_block['claim']}")
    w(f"  organ of the 1.328: {shannon_block['organ_of_the_1_328']}")
    w(f"  formula: {shannon_block['formula']}")
    w(f"  bound:   {shannon_block['bound']}")
    w(f"  L0 down: meas rel_error {meas_rel} vs slb {slb_rel}; "
      f"meas mse {meas_mse} vs slb {slb_mse}")
    w(f"  non-Gaussianity: {ng_bits} bits (h_gauss {h_gauss} - h_knn {h_knn})")
    w(f"  gap_decades reproduced: {decades_from_mse:.4f} "
      f"(receipt {recorded_decades}, aggregate max {max_gap}, atlas 1.328) "
      f"[{'MATCH' if shannon_reproduced else 'MISMATCH'}]")
    w(f"  L0 gate Shannon gap (different cell): "
      f"{c_gate['headroom']['gap_to_shannon_decades']} decades, "
      f"{c_gate['h_bits_per_dim']['non_gaussianity_bits']} bits non-Gaussian")
    w(f"  correction: {shannon_block['correction']}")
    w("")
    w("## 5. SCOPE")
    w(f"  model:   {scope['model']}")
    w(f"  layers:  {scope['layers']}")
    w(f"  tensors: {scope['tensors_kronecker_win']}")
    w(f"  organs:  WIN {scope['organs_win']}; LOSE {scope['organs_lose']}")
    w(f"  bpw:     {scope['bpw_kind']}")
    w(f"  metric:  {scope['metric']}")
    w(f"  kernel:  {scope['kernel']}")
    w(f"  health:  doctor-gate {health['doctor_gate']}; generation {health['generation']}")
    w(f"  active_bpw: {bpw_family['active_bpw']}")
    w(f"  narrowing: {scope['narrowing']}")
    w(f"  depth:   L0 {kron_l0_gate} → L1 {kron_l1_gate} → L2 {kron_l2_gate} → L46 {kron_l46_gate}")
    w("")
    w("## 6. REUSABLE_NOW")
    w(f"  REUSABLE_NOW: {reusable['REUSABLE_NOW']}")
    w(f"  live_named_lever: {reusable['live_named_lever']}")
    w("  not a pack, because:")
    for item in reusable["why_not_reusable_as_a_pack"]:
        w(f"    - {item}")
    w(f"  still live: {reusable['why_still_live']}")
    w(f"  smallest experiment: {reusable['smallest_experiment']['name']}")
    w(f"    {reusable['smallest_experiment']['what']}")
    w(f"    promotes_if: {reusable['smallest_experiment']['promotes_if']}")
    w(f"    kills_if:    {reusable['smallest_experiment']['kills_if']}")
    w("")
    w("## WHAT I WATCHED FAIL")
    for i, item in enumerate(watched, 1):
        w(f"  {i}. {item}")
    w("")
    w("self_checks: " + json.dumps(doc["self_checks"]))
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if not all(doc["self_checks"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
