#!/usr/bin/env python3.12
"""S3D: per-layer codec PORTFOLIO with a validated predictor and a hard Kronecker admission rule.

THE CLAIM UNDER TEST. The campaign has two sealed observations that together say "one codec for
all 94 layers is the wrong shape":

  * post-hoc coding is nearly exhausted at depth (mid/late layers sit 0.06-0.28 decades from their
    true Shannon lower bound and are essentially Gaussian in the coded space), but LAYER 0 is
    wildly non-Gaussian and Kronecker beats the incumbent VQ there at a CHEAPER rate;
  * gamma^2 anisotropy (data-free, from post_attention_layernorm.weight) is 1.843 decades at L0,
    0.904 at L1, 0.203 at L46, 0.062 at L93 - i.e. it ranks layers by how much an output-aware
    codec should be able to buy.

This module turns that into a selector: four codecs, one mode field per tensor, an exact ledger
for the mode bits, and a falsification test at equal complete bits.

HONESTY CONTRACT (binding, read before quoting any number out of the report):

  * WEIGHT-SPACE ERROR IS NEVER A CAPABILITY CLAIM. Nothing here is capability. Only a real
    parent-vs-packed 94-layer forward on the scored holdout may adjudicate capability, and the
    sealed S1 result is that the incumbent full stack COLLAPSES 12/12 (symKL 7.7-8.6 against a
    gate of 0.10). Assume nothing in this file closes that gap.

  * THE "OUTPUT SPACE" HERE IS A SURROGATE INPUT MODEL, NOT MEASURED ACTIVATIONS. Capturing real
    organ inputs at layer L needs a forward through L layers of a 235B model; a heavy campaign
    owns that hardware. So for gate/up we score under the input model x = gamma (*) z, z ~ N(0,I),
    whose exact second moment is diag(gamma^2) - gamma IS the post-attention-LayerNorm gain and
    the LayerNorm output is unit-RMS-normalized before it, so this is the structurally correct
    zeroth-order model and nothing more. Output error is then ||(W-W')diag(gamma)||_F over
    ||W diag(gamma)||_F. It is a PROXY of a PROXY. It is not capability and it is not even a
    measured activation proxy.

  * CIRCULARITY, DECLARED. For gate/up the predictor (gamma^2 anisotropy) and the surrogate output
    metric (gamma-weighted Frobenius) share gamma. A correlation between them is therefore
    PARTLY definitional. The non-definitional content is only the MAGNITUDE of the achievable
    gain and its ordering across layers, plus the control measurement of what gamma-weighting
    COSTS in the isotropic metric. Both are reported. Do not read the correlation as an
    independent validation of the predictor; read the magnitudes.

  * down_proj gets NO gamma-weighted arm. Its input is the SwiGLU intermediate; gamma describes
    the attention-residual stream, not that. For down_proj the isotropic Frobenius error IS the
    output error under an isotropic input model, and no better model is available without a
    forward. Stated, not hidden.

  * The codebook is fit on the same small expert cluster that is then scored. That is the
    incumbent's own protocol (qwen_generated_params.codec_baseline) so the comparison is fair,
    but it is a fit/score overlap and every codec in the portfolio enjoys it equally.

  * Only 9 of 94 layers are MEASURED. The whole-model selection for the other 85 is a PREDICTION
    from the data-free score, not a measurement, and is labelled as such in the report.

THE FOUR MODES (2 mode bits per tensor, billed):
  vq    incumbent scale-invariant VQ at the sealed S64 rungs (gate/up 2.5 bpw, down 0.625 bpw)
  gvq   the same codec with importance = gamma^2 (gate/up ONLY; zero extra artifact bytes,
        gamma already ships native as a pass-through tensor)
  kron  Van Loan rearrangement + truncated SVD, rank capped so complete bits <= incumbent bits
  rvq   an extra residual stage at the IDENTICAL rate (gate 2.5 bpw as dim16/k1024/stages4,
        down 0.625 bpw as dim32/k1024/stages2) - deeper residual, coarser per-stage vectors

KRONECKER ADMISSION RULE, enforced in `select_mode` and asserted in the self-check:
  (i)   a structural score predicts it (spectral concentration of the Van Loan rearrangement),
  (ii)  it BEATS the incumbent in weight space at equal-or-fewer complete bits, and
  (iii) it also beats the incumbent in the surrogate output space.
  All three, per cell. It won on layer 0 and lost badly at depth; it is not retained on reputation.

FALSIFICATION: if the portfolio does not beat the best SINGLE codec applied everywhere, at equal
complete bits, the selector is not worth its mode bits and the report says so in `verdict`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "foundry")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qwen_function_aware_codec as C           # noqa: E402  scale-invariant VQ (incumbent)
import qwen_generated_params as G               # noqa: E402  kron_splits / kron_bits / kron_svals
import qwen_subhalfbit_search as SHB            # noqa: E402  exact per-tensor bit charge
import qwen_structural_plan as SP               # noqa: E402  omission-aware whole-model ledger
import qwen3_moe_adapter as A                   # noqa: E402  tensor inventory
from one_bit_ceiling import (                   # noqa: E402  THE ceiling
    CompleteByteLedger, assert_complete_bpw_le_one,
)

SCHEMA = "hawking.gravity.codec_portfolio.v1"
SOURCE_DIR = "models/qwen3-235b-a22b"
ROUTING = Path("reports/subbit_reset/QWEN3_235B_ROUTING_CALIBRATION_1200.json")
REPORT = Path("reports/subbit_reset/S3D_CODEC_PORTFOLIO.json")

MODES: tuple[str, ...] = ("vq", "gvq", "kron", "rvq")
KEEP = 64                       # S64 survivor inventory (sealed legal arm)
N_LAYERS = 94
N_EXPERTS_FIT = 3               # tensors resident at once; the box is shared, keep it tiny
KRON_SPLITS = 3                 # cheapest Van Loan splits to try per tensor
KRON_CONCENTRATION_MIN = 0.50   # admission (i): energy share of the budgeted rank in R(W)

# Sealed S64 rungs, plus an equal-RATE residual variant of each.
RUNGS: dict[str, dict[str, Any]] = {
    "gate_proj": {"vq": {"dim": 8, "k": 1024, "stages": 2},      # 2.5 bpw
                  "rvq": {"dim": 16, "k": 1024, "stages": 4}},   # 2.5 bpw, deeper residual
    "down_proj": {"vq": {"dim": 16, "k": 1024, "stages": 1},     # 0.625 bpw
                  "rvq": {"dim": 32, "k": 1024, "stages": 2}},   # 0.625 bpw, deeper residual
}
# Layers measured on real weights. Spans the anisotropy range; includes every layer the stage
# brief names (0, 1, 2, 70, 46, 93, 90) plus two mid-band points.
SAMPLE_LAYERS = (0, 1, 2, 5, 18, 46, 70, 90, 93)

_EPS = 1e-12


# ── the predictor (data-free, seconds) ────────────────────────────────────────────────────────
def anisotropy(gamma: np.ndarray) -> float:
    """gamma^2 anisotropy in decades, log10(p99/p50). The campaign's existing definition.

    Reproduces reports/subbit_reset/GAMMA_ANISOTROPY_ALL_LAYERS.json exactly (L0 = 1.8428).
    """
    g2 = np.asarray(gamma, np.float64) ** 2
    return float(np.log10(max(np.percentile(g2, 99), _EPS) / max(np.percentile(g2, 50), _EPS)))


# ── error metrics ─────────────────────────────────────────────────────────────────────────────
def rel_error(w: np.ndarray, rec: np.ndarray, col_w: np.ndarray | None = None) -> float:
    """Relative Frobenius error, optionally with a per-COLUMN (input-dimension) weight.

    col_w = gamma gives the surrogate output error for an organ whose input is gamma (*) z:
    E||Wx - W'x||^2 = ||(W-W')diag(gamma)||_F^2. col_w = None is the isotropic/weight-space error.
    """
    a = np.asarray(w, np.float32)
    d = a - np.asarray(rec, np.float32)
    if col_w is not None:
        cw = np.asarray(col_w, np.float32)[None, :]
        a, d = a * cw, d * cw
    return float(np.linalg.norm(d) / max(np.linalg.norm(a), _EPS))


# ── Kronecker (Van Loan), reusing qwen_generated_params for splits and the exact bit charge ───
def _bf16(x: np.ndarray) -> np.ndarray:
    """Truncate to the bf16 grid - the factors are SHIPPED as bf16, so score what ships."""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32) >> np.uint32(16)
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _rearrange(w: np.ndarray, split: tuple[int, int, int, int]) -> np.ndarray:
    m1, n1, m2, n2 = split
    return np.ascontiguousarray(
        np.asarray(w, np.float32).reshape(m1, m2, n1, n2).transpose(0, 2, 1, 3)
    ).reshape(m1 * n1, m2 * n2)


def _unrearrange(r: np.ndarray, split: tuple[int, int, int, int]) -> np.ndarray:
    m1, n1, m2, n2 = split
    return np.ascontiguousarray(
        r.reshape(m1, n1, m2, n2).transpose(0, 2, 1, 3)
    ).reshape(m1 * m2, n1 * n2)


def kron_fit(w: np.ndarray, split: tuple[int, int, int, int], rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank-R Kronecker reconstruction (Frobenius-OPTIMAL for the family) + the R(W) spectrum.

    Factors are rounded to bf16 because kron_bits charges them as bf16. No tuning knobs exist
    here: the rank-R SVD of the rearrangement IS the optimum, so a loss is a loss for the family.
    """
    r = _rearrange(w, split)
    u, s, vt = np.linalg.svd(r, full_matrices=False)
    rank = int(max(0, min(rank, len(s))))
    if rank == 0:
        return np.zeros_like(np.asarray(w, np.float32)), s
    approx = _bf16(u[:, :rank] * s[:rank]) @ _bf16(vt[:rank])
    return _unrearrange(approx, split), s


def kron_concentration(svals: np.ndarray, rank: int) -> float:
    """Admission (i): energy share of R(W) captured by the budgeted rank. Structural, data-free."""
    s2 = np.asarray(svals, np.float64) ** 2
    return float(s2[:max(0, rank)].sum() / max(s2.sum(), _EPS))


# ── one measured cell ─────────────────────────────────────────────────────────────────────────
def measure_cell(mats: list[np.ndarray], organ: str, gamma: np.ndarray | None) -> dict[str, Any]:
    """Fit every admissible codec on one (layer, organ) cell and score all of them.

    <= N_EXPERTS_FIT expert tensors resident. Every arm is charged its own exact complete bits.
    """
    w = mats[0]
    n = w.shape[0] * w.shape[1]
    spec = dict(RUNGS[organ]["vq"], family="shared_grammar")
    base_bits = SHB.expert_bits(w.shape, spec, KEEP)
    col_w = gamma if (organ != "down_proj" and gamma is not None) else None
    out: dict[str, Any] = {"organ": organ, "shape": list(w.shape), "n_fit_tensors": len(mats),
                           "incumbent_complete_bits_per_tensor": int(base_bits),
                           "output_metric": ("gamma_weighted_surrogate" if col_w is not None
                                             else "isotropic_input_model")}
    arms: dict[str, dict[str, Any]] = {}

    def _vq(tag: str, sp: dict[str, Any], imp: np.ndarray | None) -> None:
        books = C.fit(mats, dim=sp["dim"], k=sp["k"], stages=sp["stages"],
                      importance=imp, iters=4)
        rec = C.apply_refit(books, w, dim=sp["dim"], importance=imp)
        bits = SHB.expert_bits(w.shape, dict(sp, family="shared_grammar"), KEEP)
        arms[tag] = {"spec": dict(sp), "complete_bits_per_tensor": int(bits),
                     "complete_bpw_tensor": round(bits / n, 6),
                     "weight_rel_error": round(rel_error(w, rec), 6),
                     "output_rel_error": round(rel_error(w, rec, col_w), 6)}
        del books, rec

    _vq("vq", RUNGS[organ]["vq"], None)
    _vq("rvq", RUNGS[organ]["rvq"], None)
    if col_w is not None:
        _vq("gvq", RUNGS[organ]["vq"], C.importance_from_activations(gamma[None, :]))

    best_k: dict[str, Any] | None = None
    for split in G.kron_splits(w.shape, top=KRON_SPLITS):
        rank = (base_bits - G.METADATA_BITS) // ((split[0] * split[1] + split[2] * split[3]) * 16)
        rec, sv = kron_fit(w, split, rank)
        rank = int(min(rank, len(sv)))
        cand = {"split": list(split), "rank": rank,
                "complete_bits_per_tensor": int(G.kron_bits(split, rank)),
                "complete_bpw_tensor": round(G.kron_bits(split, rank) / n, 6),
                "concentration": round(kron_concentration(sv, rank), 6),
                "weight_rel_error": round(rel_error(w, rec), 6),
                "output_rel_error": round(rel_error(w, rec, col_w), 6)}
        del rec, sv
        if best_k is None or cand["output_rel_error"] < best_k["output_rel_error"]:
            best_k = cand
    arms["kron"] = best_k or {}
    out["arms"] = arms
    out["selected"], out["selection_reason"] = select_mode(arms)
    return out


def kron_admitted(inc: dict[str, Any], k: dict[str, Any]) -> dict[str, bool]:
    """THE ADMISSION RULE, itemized so each condition can be tested and audited separately.

    (i) structural score predicts it, (ii) it beats the incumbent in WEIGHT space at
    equal-or-fewer complete bits, (iii) it beats the incumbent in OUTPUT space too. All three.
    """
    if not k:
        return {"structural": False, "cheaper_or_equal": False, "beats_weight": False,
                "beats_output": False, "admitted": False}
    c = {"structural": k["concentration"] >= KRON_CONCENTRATION_MIN,
         "cheaper_or_equal": k["complete_bits_per_tensor"] <= inc["complete_bits_per_tensor"],
         "beats_weight": k["weight_rel_error"] < inc["weight_rel_error"],
         "beats_output": k["output_rel_error"] < inc["output_rel_error"]}
    return c | {"admitted": all(c.values())}


def select_mode(arms: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Pick the mode. Kronecker must clear all three admission conditions; ties go to incumbent."""
    inc = arms["vq"]
    k = arms.get("kron") or {}
    admitted = {"vq": inc}
    for tag, a in arms.items():
        if tag in ("vq", "kron") or not a:
            continue
        if a["complete_bits_per_tensor"] <= inc["complete_bits_per_tensor"]:
            admitted[tag] = a
    if kron_admitted(inc, k)["admitted"]:
        admitted["kron"] = k
    win = min(admitted, key=lambda t: (admitted[t]["output_rel_error"], t != "vq"))
    if win == "vq":
        return "vq", "incumbent not beaten by any admitted alternative"
    return win, (f"{win} output_rel_error {admitted[win]['output_rel_error']} < incumbent "
                 f"{inc['output_rel_error']} at {admitted[win]['complete_bits_per_tensor']} "
                 f"<= {inc['complete_bits_per_tensor']} complete bits")


# ── predictor validation ──────────────────────────────────────────────────────────────────────
def _corr(x: list[float], y: list[float]) -> dict[str, float]:
    """Pearson and Spearman. Two-line, no scipy."""
    if len(x) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": len(x)}
    a, b = np.asarray(x, np.float64), np.asarray(y, np.float64)
    rk = lambda v: np.argsort(np.argsort(v)).astype(np.float64)  # noqa: E731
    p = lambda u, v: float(np.corrcoef(u, v)[0, 1]) if u.std() > 0 and v.std() > 0 else float("nan")  # noqa: E731
    return {"pearson": round(p(a, b), 6), "spearman": round(p(rk(a), rk(b)), 6), "n": len(x)}


# ── the ledger ────────────────────────────────────────────────────────────────────────────────
def mode_bits_per_tensor(n_modes: int = len(MODES)) -> int:
    """ceil(log2 n_modes). One field per coded expert tensor. Undeclared is not zero."""
    return int(math.ceil(math.log2(max(1, int(n_modes)))))


def portfolio_ledger(inv, routing: dict[str, Any] | None, n_modes: int = len(MODES)) -> dict[str, Any]:
    """Complete portfolio ledger = the sealed S64 arm + one mode field per coded expert tensor.

    CONSERVATIVE BY CONSTRUCTION: every non-incumbent arm is admitted only at equal-or-FEWER
    complete bits than the incumbent, so charging every tensor at the incumbent rate is an exact
    UPPER bound on the portfolio's payload. Kronecker's cheaper cells are given no credit. The
    Kronecker factors, column scales and rotation seeds live inside kron_bits, which is what the
    admission test compares against, so no selected codec's side information is uncharged.
    """
    base = SP.ledger(inv, KEEP, dict(RUNGS["gate_proj"]["vq"], family="shared_grammar"),
                     dict(RUNGS["down_proj"]["vq"], family="shared_grammar"), routing)
    mb = mode_bits_per_tensor(n_modes)
    comp = dict(base["components"])
    mode_total = mb * int(base["coded_expert_tensors"])
    comp["metadata"] += mode_total
    led = CompleteByteLedger(metadata_alignment_reserve_bits=0, **comp,
                             note="S3D per-layer codec portfolio, incumbent-rate upper bound")
    receipt = assert_complete_bpw_le_one(led, base["original_weight_count"])
    inc_bpw = Fraction(base["complete_bits"], base["original_weight_count"])
    pf_bpw = led.complete_bpw(base["original_weight_count"])
    return {"mode_bits_per_tensor": mb, "coded_expert_tensors": int(base["coded_expert_tensors"]),
            "mode_bits_total": int(mode_total),
            "incumbent_everywhere_complete_bpw": round(float(inc_bpw), 9),
            "portfolio_complete_bpw": round(float(pf_bpw), 9),
            "portfolio_complete_bpw_exact": f"{pf_bpw.numerator}/{pf_bpw.denominator}",
            "selector_surcharge_bpw": float(pf_bpw - inc_bpw),
            "legal_under_one_bit_ceiling": True, "ceiling_receipt": receipt,
            "components": {k: int(v) for k, v in comp.items()}}


# ── build ─────────────────────────────────────────────────────────────────────────────────────
def build(layers: tuple[int, ...] = SAMPLE_LAYERS) -> dict[str, Any]:
    from qwen_real_forward import SafetensorsIndexReader
    r = SafetensorsIndexReader(SOURCE_DIR)
    t0 = time.time()

    gammas = {L: r.bf16(f"model.layers.{L}.post_attention_layernorm.weight").astype(np.float32)
              for L in range(N_LAYERS)}
    aniso = {L: round(anisotropy(gammas[L]), 6) for L in range(N_LAYERS)}

    cells: list[dict[str, Any]] = []
    for L in layers:
        for organ in ("gate_proj", "down_proj"):
            mats = [r.bf16(f"model.layers.{L}.mlp.experts.{e}.{organ}.weight").astype(np.float32)
                    for e in range(N_EXPERTS_FIT)]
            cell = measure_cell(mats, organ, gammas[L])
            cell |= {"layer": L, "anisotropy_decades": aniso[L]}
            cells.append(cell)
            del mats
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001 - cache clearing is best effort
                pass

    # predictor validation: does anisotropy rank the MEASURED gvq benefit on gate/up?
    gate = [c for c in cells if c["organ"] == "gate_proj" and "gvq" in c["arms"]]
    ax = [c["anisotropy_decades"] for c in gate]
    gain_out = [math.log10(max(c["arms"]["vq"]["output_rel_error"], _EPS)
                           / max(c["arms"]["gvq"]["output_rel_error"], _EPS)) for c in gate]
    gain_wt = [math.log10(max(c["arms"]["vq"]["weight_rel_error"], _EPS)
                          / max(c["arms"]["gvq"]["weight_rel_error"], _EPS)) for c in gate]
    kron_gain = [math.log10(max(c["arms"]["vq"]["output_rel_error"], _EPS)
                            / max(c["arms"]["kron"]["output_rel_error"], _EPS)) for c in cells]
    predictor = {
        "definition": "log10(p99/p50) of post_attention_layernorm.weight ** 2, data-free",
        "all_94_layers": aniso,
        "gvq_gain_vs_anisotropy_gate": _corr(ax, gain_out),
        "gvq_isotropic_cost_vs_anisotropy_gate": _corr(ax, gain_wt),
        "kron_gain_vs_anisotropy_all_cells": _corr([c["anisotropy_decades"] for c in cells], kron_gain),
        "per_layer_gate": [{"layer": c["layer"], "anisotropy_decades": c["anisotropy_decades"],
                            "gvq_output_gain_decades": round(g, 6),
                            "gvq_weight_space_cost_decades": round(w, 6)}
                           for c, g, w in zip(gate, gain_out, gain_wt)],
        "circularity_caveat": ("gate/up predictor and gate/up output metric share gamma, so this "
                               "correlation is PARTLY definitional; the load-bearing numbers are "
                               "the gain MAGNITUDES and the isotropic-space cost, not the r value"),
    }

    # portfolio vs the best SINGLE codec applied everywhere, on the measured cells
    per_mode = {m: float(np.mean([c["arms"][m]["output_rel_error"]
                                  for c in cells if m in c["arms"] and c["arms"][m]]))
                for m in MODES if any(m in c["arms"] and c["arms"][m] for c in cells)}
    # a single codec must be applicable to EVERY cell to count as "one codec everywhere"
    universal = {m: v for m, v in per_mode.items()
                 if all(m in c["arms"] and c["arms"][m] for c in cells)}
    best_single = min(universal, key=universal.get)
    pf_mean = float(np.mean([c["arms"][c["selected"]]["output_rel_error"] for c in cells]))
    beats = pf_mean < universal[best_single] - 1e-9

    inv = A.build_inventory(A.load_config(), A.load_index())
    routing = json.loads(ROUTING.read_text()) if ROUTING.exists() else None
    led = portfolio_ledger(inv, routing)

    sel = {c["selected"] for c in cells}
    kron_cells = [f"L{c['layer']}.{c['organ']}" for c in cells if c["selected"] == "kron"]
    # whole-model assignment for the 85 UNMEASURED layers: prediction, never measurement
    thr = min([c["anisotropy_decades"] for c in cells
               if c["organ"] == "gate_proj" and c["selected"] == "gvq"], default=None)
    predicted = ({L: ("gvq" if aniso[L] >= thr else "vq") for L in range(N_LAYERS)}
                 if thr is not None else {L: "vq" for L in range(N_LAYERS)})
    for c in cells:                                            # measured layers override
        if c["organ"] == "gate_proj":
            predicted[c["layer"]] = c["selected"]

    verdict = (
        f"PORTFOLIO WINS: per-(layer,organ) selection beats the best single codec "
        f"({best_single}) at equal complete bits, mean surrogate output rel_error "
        f"{pf_mean:.4f} vs {universal[best_single]:.4f}, selector surcharge "
        f"{led['selector_surcharge_bpw']:.3e} BPW."
        if beats else
        f"SELECTOR NOT WORTH ITS MODE BITS: the best single codec ({best_single}, mean "
        f"{universal[best_single]:.4f}) matches or beats the per-cell portfolio "
        f"({pf_mean:.4f}); the mode field buys nothing and must not be shipped."
    ) + (" NOT A CAPABILITY CLAIM: surrogate output space on a synthetic gamma input model, "
         "no forward was run.")

    return {
        "schema": SCHEMA, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "modes": list(MODES), "keep_experts": KEEP, "layers_measured": list(layers),
        "n_layers": N_LAYERS, "n_fit_tensors_per_cell": N_EXPERTS_FIT,
        "predictor": predictor, "cells": cells,
        "comparison_at_equal_complete_bits": {
            "mean_output_rel_error_per_single_codec": {k: round(v, 6) for k, v in per_mode.items()},
            "universal_codecs": sorted(universal),
            "best_single_codec": best_single,
            "best_single_mean_output_rel_error": round(universal[best_single], 6),
            "portfolio_mean_output_rel_error": round(pf_mean, 6),
            "portfolio_beats_best_single": bool(beats),
            "distinct_modes_selected": sorted(sel),
            "kron_admitted_cells": kron_cells,
        },
        "ledger": led,
        "whole_model_mode_assignment_gate_up": {
            "status": "PREDICTED for unmeasured layers, MEASURED for "
                      + ",".join(f"L{L}" for L in layers),
            "gvq_anisotropy_threshold_decades": thr,
            "assignment": predicted,
            "n_gvq": sum(1 for v in predicted.values() if v == "gvq"),
        },
        "verdict": verdict,
        "honesty": {
            "capability": "NONE. Weight-space and surrogate-output-space only; no forward ran.",
            "output_space": "synthetic input model x = gamma (*) z for gate/up; isotropic for "
                            "down_proj. NOT measured activations.",
            "circularity": predictor["circularity_caveat"],
            "coverage": f"{len(layers)}/{N_LAYERS} layers measured; the rest are predicted.",
            "fit_score_overlap": "codebooks are fit on the same expert cluster they are scored "
                                 "on, identically for every arm.",
            "bits": "every arm charged with SHB.expert_bits / G.kron_bits; portfolio ledger is an "
                    "exact UPPER bound (cheaper Kronecker cells get no credit).",
        },
    }


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def demo() -> None:
    """Asserts on the core logic. Fails loudly if any of it breaks. No weights needed."""
    rng = np.random.default_rng(7)

    # 1. predictor: flat gamma is isotropic, spiky gamma is not, and it is monotone in spikiness.
    flat = np.ones(4096, np.float32)
    assert abs(anisotropy(flat)) < 1e-6, anisotropy(flat)
    spiky = np.ones(4096, np.float32)
    spiky[:60] = 30.0
    assert anisotropy(spiky) > 1.0, anisotropy(spiky)
    mild = np.ones(4096, np.float32)
    mild[:60] = 3.0
    assert anisotropy(spiky) > anisotropy(mild) >= 0.0

    # 2. Kronecker: an exact Kronecker product must be recovered at rank 1 (bf16 factors), a
    #    generic matrix must not, and the rearrange/unrearrange pair must round-trip exactly.
    a4, b4 = rng.standard_normal((8, 4)).astype(np.float32), rng.standard_normal((4, 8)).astype(np.float32)
    W = np.kron(a4, b4).astype(np.float32)
    split = (8, 4, 4, 8)
    assert np.array_equal(_unrearrange(_rearrange(W, split), split), W)
    rec, sv = kron_fit(W, split, 1)
    assert rel_error(W, rec) < 1.5e-2, rel_error(W, rec)   # bf16 TRUNCATION floor, ~0.7 pct
    assert kron_concentration(sv, 1) > 0.999, kron_concentration(sv, 1)
    gen = rng.standard_normal((32, 32)).astype(np.float32)
    assert rel_error(gen, kron_fit(gen, (8, 4, 4, 8), 1)[0]) > 0.5
    # rank monotonicity: more rank never hurts
    e1 = rel_error(gen, kron_fit(gen, (8, 4, 4, 8), 1)[0])
    e4 = rel_error(gen, kron_fit(gen, (8, 4, 4, 8), 4)[0])
    assert e4 < e1, (e1, e4)

    # 3. the output metric really is column-weighted, and weighting changes the answer.
    w = rng.standard_normal((16, 8)).astype(np.float32)
    rec = w.copy()
    rec[:, 0] += 1.0
    g = np.ones(8, np.float32)
    g[0] = 10.0
    assert rel_error(w, rec, g) > rel_error(w, rec), (rel_error(w, rec, g), rel_error(w, rec))
    assert rel_error(w, w) == 0.0

    # 4. mode bits: exactly ceil(log2 n_modes), and 4 modes cost 2 bits.
    assert mode_bits_per_tensor(4) == 2 and mode_bits_per_tensor(1) == 0
    assert mode_bits_per_tensor(3) == 2 and mode_bits_per_tensor(5) == 3
    assert mode_bits_per_tensor(len(MODES)) == 2

    # 5. the Kronecker ADMISSION RULE. All three conditions, each independently blocking.
    inc = {"complete_bits_per_tensor": 1000, "weight_rel_error": 0.30, "output_rel_error": 0.30}
    good = {"complete_bits_per_tensor": 900, "concentration": 0.90,
            "weight_rel_error": 0.05, "output_rel_error": 0.05}
    assert kron_admitted(inc, good)["admitted"]
    assert select_mode({"vq": inc, "kron": good})[0] == "kron"
    for broken, cond in ((dict(good, concentration=0.10), "structural"),                  # (i)
                         (dict(good, weight_rel_error=0.40), "beats_weight"),             # (ii)
                         (dict(good, output_rel_error=0.40), "beats_output"),             # (iii)
                         (dict(good, complete_bits_per_tensor=1001), "cheaper_or_equal")):
        c = kron_admitted(inc, broken)
        assert not c[cond] and not c["admitted"], (cond, c)
        assert select_mode({"vq": inc, "kron": broken})[0] == "vq"
    # each condition is INDEPENDENTLY blocking: a Kronecker arm that wins in weight space at a
    # cheaper rate but LOSES in output space must not be admitted, even though it is cheaper.
    weight_only = dict(good, weight_rel_error=0.01, output_rel_error=0.31)
    assert not kron_admitted(inc, weight_only)["admitted"], weight_only
    assert not kron_admitted(inc, {})["admitted"]
    # ties go to the incumbent, and an over-budget non-Kronecker arm is refused too
    assert select_mode({"vq": inc, "rvq": dict(inc)})[0] == "vq"
    assert select_mode({"vq": inc, "rvq": {"complete_bits_per_tensor": 2000,
                                           "output_rel_error": 0.01}})[0] == "vq"
    assert select_mode({"vq": inc, "gvq": {"complete_bits_per_tensor": 1000,
                                           "output_rel_error": 0.10}})[0] == "gvq"

    # 6. gamma-weighted VQ must actually beat plain VQ in the gamma output space on a planted
    #    anisotropic cell - if it does not, the gvq arm is broken, not merely unlucky.
    cols, rows = 64, 256
    gv = np.ones(cols, np.float32)
    gv[:4] = 12.0
    base = rng.standard_normal((rows, cols)).astype(np.float32)
    imp = C.importance_from_activations(gv[None, :])
    books_p = C.fit([base], dim=4, k=16, stages=1, iters=8)
    books_g = C.fit([base], dim=4, k=16, stages=1, importance=imp, iters=8)
    e_p = rel_error(base, C.apply_refit(books_p, base, dim=4), gv)
    e_g = rel_error(base, C.apply_refit(books_g, base, dim=4, importance=imp), gv)
    assert e_g < e_p, ("gamma-weighted VQ lost in gamma output space", e_p, e_g)
    # ...and the importance must actually reach the FIT, not only the assignment.
    assert not np.allclose(books_p[0].detach().cpu().numpy(),
                           books_g[0].detach().cpu().numpy()), "importance never reached the fit"
    # gamma^2 importance is the mean-1 normalized square of gamma, exactly.
    assert np.allclose(imp, gv ** 2 / float((gv ** 2).mean()), rtol=1e-5), imp[:4]

    # 7. the ledger: mode bits are really charged, and the ceiling really bites.
    comp = {c: 0 for c in ("indices", "codebooks", "scales", "metadata", "alignment",
                           "protected_islands", "doctor", "pass_through_tensors",
                           "packaging", "runtime_tables")}
    comp["indices"] = 900
    led = CompleteByteLedger(metadata_alignment_reserve_bits=0, **comp)
    assert assert_complete_bpw_le_one(led, 1000)["legal"]
    bad = CompleteByteLedger(metadata_alignment_reserve_bits=0, **(comp | {"indices": 1001}))
    try:
        assert_complete_bpw_le_one(bad, 1000)
        raise AssertionError("ceiling did not fire above 1 BPW")
    except Exception as exc:  # noqa: BLE001
        assert "ceiling violated" in str(exc), exc

    print("qwen_codec_portfolio self-check OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S3D per-layer codec portfolio.")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--layers", default=",".join(str(L) for L in SAMPLE_LAYERS))
    a = ap.parse_args(argv)
    if a.self_check:
        demo()
        return 0
    if not a.run:
        ap.error("pass --self-check or --run")
    out = build(tuple(int(x) for x in a.layers.split(",")))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out["comparison_at_equal_complete_bits"], indent=2))
    print(json.dumps(out["ledger"] | {"ceiling_receipt": "..."}, indent=2))
    print(out["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
