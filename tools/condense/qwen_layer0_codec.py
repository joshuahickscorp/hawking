#!/usr/bin/env python3.12
"""Layer-0 specific codec: per-column scale modulation of the shared VQ grammar.

WHY. qwen_shannon_bound measured that layer 0 is the one place where this campaign's codec is
far from its own Shannon lower bound: L0 down_proj sits 1.328 decades above the bound with
3.014 bits of non-Gaussianity, L0 gate 0.791 decades / 0.669 bits. Layers 46 and 93 show
0.06-0.28 decades and essentially zero non-Gaussianity. So layer 0 has exploitable structure
that the current codec ignores.

WHAT THE STRUCTURE ACTUALLY IS (measured here, `characterise`, not assumed):
  * L0 down_proj: after per-row scale normalisation the PER-COLUMN rms still spans 2.55-2.94
    decades (L46: 0.10). Kurtosis of the row-normalised entries is ~53 (L46: 3.0, exactly
    Gaussian) and 93.6 percent of entries are below 0.1 of unit rms. The "near-sparsity" and
    the huge kurtosis ARE the column-scale structure: energy lives in a small set of input
    channels, the first block's known massive-activation channels.
  * L0 gate/up: the span is in the ROWS (4.9-7.6 decades) which the existing per-row scale
    already removes; its residual column span is only 1.2 decades.
  * Both L0 organs are also low effective rank (~65/1536 vs ~1210 at L46). Not exploited here:
    a per-expert rank-64 basis costs 64*1536*16 bits, a quarter of the whole 1-bit budget for
    that tensor, and the shared-basis-across-experts form is already in the dead-lever atlas.

THE FIX, cheapest form. The codec chunks each row into `dim` consecutive columns, so chunk
position j always covers the same columns of every row. Divide the tensor by a per-column
bf16 scale c, code the preconditioned tensor, and fit/assign with the diagonal importance
h = c^2 so the objective stays the ORIGINAL-space MSE. Effect: one shared codebook of k
centroids becomes k centroids modulated by (cols/dim) per-position scale patterns, at the
SAME index rate. Extra artifact cost: 16 bits per column plus one mode bit per tensor, both
billed by `col_bits`. Nothing else changes: same dim, same k, same stages, same layout.

MEASURED RESULT (see reports/subbit_reset/LANE_A2_LAYER0_CODEC.json). It is a real but
PARTIAL win, and it is not uniform:
  * L0 down 0.3616 -> 0.1910 rel_error, 0.552 of a 1.308-decade gap closed (42.2 percent).
    BELOW the falsification bar of half the gap. Residual gap 0.756 decades, reported not hidden.
  * The diagnosis is nonetheless exactly right: L0 down non-Gaussianity falls 3.0143 -> -0.0035
    bits. The per-column scale structure WAS the entire non-Gaussianity. The reason only 42
    percent of the DISTORTION gap follows is that the structure is bought as side information
    (16 bits/column) instead of being coded, so the codec now faces a Gaussian source at the
    same rate rather than the original low-entropy one. Whatever closes the remaining 0.756
    decades has to code that structure, not declare it.
  * L0 gate gets WORSE under modulation (0.2728 -> 0.4761), so the encoder must SELECT per
    tensor group by measured error; that selection is the one billed mode bit.
  * L46 gate/down improve by 0.7-1.6 percent, i.e. this is a layer-0 lever, not a general
    codec improvement.
  * Closed-form refit of the column scales after coding makes it worse (0.19 -> 0.39): the
    codebook goes stale against the new preconditioner. Sealed negative, do not retry.

HONESTY. Every number here is WEIGHT-SPACE reconstruction error. Weight-space error is NEVER
a capability claim; only a real parent-vs-packed forward can select a frontier. The rel_error
reported is the row-normalised quantity qwen_shannon_bound uses, so it is comparable with the
sealed cells; note it weights every row equally regardless of row norm, which is why a codec
optimising true output MSE can look worse on it for a tensor with a 7-decade row-norm span.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import qwen_function_aware_codec as FAC  # noqa: E402
import qwen_shannon_bound as QSB  # noqa: E402
from qwen_real_forward import SafetensorsIndexReader  # noqa: E402

SCHEMA = "hawking.gravity.layer0_codec.v1"

# same cells the sealed Shannon run used, so the before-numbers are directly comparable
SPECS = {"gate": (8, 1024, 2, "gate_proj"),
         "up": (8, 1024, 2, "up_proj"),
         "down": (16, 1024, 1, "down_proj")}
_EPS = 1e-20


def _bf16(x: np.ndarray) -> np.ndarray:
    """Round to the bf16 grid that is actually shipped (truncation, as elsewhere in the campaign)."""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32) >> np.uint32(16)
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def col_scales(w: np.ndarray) -> np.ndarray:
    """Per-column rms, on the bf16 grid. This is the whole preconditioner."""
    return np.maximum(_bf16(np.sqrt((np.asarray(w, np.float32) ** 2).mean(0))), _EPS)


def col_bits(shape: tuple[int, int]) -> int:
    """Exact artifact cost: one bf16 scale per column + one mode bit per tensor. Never free."""
    return int(shape[1]) * 16 + 1


# ── characterisation ──────────────────────────────────────────────────────────────────────────
def characterise(w: np.ndarray) -> dict[str, float]:
    """What makes this tensor non-Gaussian: row span, column span, kurtosis, rank, sparsity."""
    a = np.asarray(w, np.float32)
    rn = np.linalg.norm(a, axis=1)
    u, _ = FAC.normalize_rows(a)
    ucr = np.sqrt((u * u).mean(0))
    sv = np.linalg.svd(a, compute_uv=False) ** 2
    p = sv / sv.sum()
    return {
        "row_norm_span_decades": round(math.log10(float(rn.max()) / max(float(rn.min()), _EPS)), 3),
        "col_rms_span_decades_after_row_norm":
            round(math.log10(float(ucr.max()) / max(float(ucr.min()), _EPS)), 3),
        "kurtosis_row_normalised": round(float((u ** 4).mean() / (u ** 2).mean() ** 2), 2),
        "effective_rank": round(float(np.exp(-(p * np.log(p + 1e-30)).sum())), 1),
        "min_dim": int(min(a.shape)),
        "near_zero_frac_below_0p1_rms": round(float((np.abs(u) < 0.1).mean()), 4),
        "top8_col_energy_share": round(float(np.sort(np.linalg.norm(a, axis=0) ** 2)[-8:].sum()
                                             / float((np.linalg.norm(a, axis=0) ** 2).sum())), 4),
    }


# ── the codec ─────────────────────────────────────────────────────────────────────────────────
def fit_modulated(mats: list[np.ndarray], *, dim: int, k: int, stages: int, seed: int = 0,
                  iters: int = 16):
    """Fit the shared codebook in the column-preconditioned space. Returns (books, scales, hs)."""
    cs = [col_scales(m) for m in mats]
    hs = [((c ** 2) / float((c ** 2).mean())).astype(np.float32) for c in cs]
    pre = [m / c[None, :] for m, c in zip(mats, cs)]
    hm = np.mean(np.stack(hs), 0).astype(np.float32)     # one importance vector for the pool
    books = FAC.fit(pre, dim=dim, k=k, stages=stages, seed=seed, row_scale=True,
                    importance=hm, iters=iters)
    return books, cs, hs, pre


def apply_modulated(books, pre_m: np.ndarray, c: np.ndarray, h: np.ndarray, *, dim: int):
    """Decode a preconditioned tensor and undo the preconditioner."""
    return FAC.apply_refit(books, pre_m, dim=dim, importance=h) * c[None, :]


def rel_error_coded(w: np.ndarray, rec: np.ndarray) -> tuple[float, float]:
    """(mse, var) in the row-normalised space qwen_shannon_bound measures in."""
    u, s = FAC.normalize_rows(w)
    ur, _ = FAC.normalize_rows(rec, s)
    return float(np.mean((u - ur) ** 2)), float(np.var(u))


def cell(mats: list[np.ndarray], *, dim: int, k: int, stages: int, seed: int = 0,
         iters: int = 16, entropy: bool = True) -> dict[str, Any]:
    """Baseline vs column-modulated on the same tensors, both against the Shannon lower bound."""
    rows, cols = mats[0].shape
    base_rate = stages * math.log2(k) / dim
    add_rate = col_bits((rows, cols)) / float(rows * cols)

    books = FAC.fit(mats, dim=dim, k=k, stages=stages, seed=seed, row_scale=True, iters=iters)
    bm = [rel_error_coded(m, FAC.apply_refit(books, m, dim=dim)) for m in mats]
    del books

    books2, cs, hs, pre = fit_modulated(mats, dim=dim, k=k, stages=stages, seed=seed, iters=iters)
    nm = [rel_error_coded(m, apply_modulated(books2, p, c, h, dim=dim))
          for m, p, c, h in zip(mats, pre, cs, hs)]
    del books2

    sigma2 = float(np.mean([v for _, v in bm]))
    base_mse = float(np.mean([e for e, _ in bm]))
    new_mse = float(np.mean([e for e, _ in nm]))

    # h of the ORIGINAL coded source fixes the bound; the modulated arm pays a slightly higher
    # rate so its bound is slightly lower (i.e. this cannot flatter the modulated arm).
    pool = np.concatenate([FAC.normalize_rows(m)[0].reshape(-1, dim) for m in mats], 0)
    h_knn = QSB.differential_entropy_knn(pool, seed=seed) if entropy else float("nan")
    h_gauss = 0.5 * math.log2(2.0 * math.pi * math.e * sigma2)
    if entropy:
        pool_pre = np.concatenate([FAC.normalize_rows(p)[0].reshape(-1, dim) for p in pre], 0)
        h_pre = QSB.differential_entropy_knn(pool_pre, seed=seed)
        s2_pre = float(np.mean([np.var(FAC.normalize_rows(p)[0]) for p in pre]))
        ng_pre = 0.5 * math.log2(2.0 * math.pi * math.e * s2_pre) - h_pre
        del pool_pre
    else:
        h_pre, ng_pre = float("nan"), float("nan")
    del pool, pre

    slb_base = QSB.shannon_lower_bound_mse(h_knn, base_rate)
    slb_new = QSB.shannon_lower_bound_mse(h_knn, base_rate + add_rate)
    gap = lambda m, s: round(math.log10(max(m, 1e-30) / max(s, 1e-30)), 4)  # noqa: E731
    g0, g1 = gap(base_mse, slb_base), gap(new_mse, slb_new)
    return {
        "rate_bits_per_dim": {"baseline": round(base_rate, 6),
                              "modulated_total": round(base_rate + add_rate, 6),
                              "column_scale_surcharge": round(add_rate, 8)},
        "non_gaussianity_bits": {"original_coded_space": round(h_gauss - h_knn, 4),
                                 "after_column_modulation": round(ng_pre, 4)},
        "rel_error": {"baseline": round(math.sqrt(base_mse / sigma2), 4),
                      "modulated": round(math.sqrt(new_mse / sigma2), 4),
                      "shannon_lower_bound": round(math.sqrt(slb_base / sigma2), 4)},
        "gap_to_shannon_decades": {"baseline": g0, "modulated": g1,
                                   "closed": round(g0 - g1, 4),
                                   "closed_fraction": round((g0 - g1) / g0, 4) if g0 > 0 else 0.0},
        "bits_charged_per_tensor": col_bits((rows, cols)),
        "shape": [rows, cols],
    }


def layer0_bpw_surcharge(*, keep_experts: int = 64, n_layers: int = 94,
                         original_weight_count: int = 235093634560) -> dict[str, Any]:
    """Exact complete-BPW cost of turning this on for LAYER 0 ONLY, as a Fraction.

    Charged: 16 bits/column for every kept layer-0 down tensor (the only organ where the
    modulation wins) plus one mode bit for every kept layer-0 tensor of all three organs.
    """
    down_cols, gate_cols = 1536, 4096
    bits = keep_experts * (down_cols * 16 + 1) + keep_experts * 2  # down scales+bit, gate/up bits
    f = Fraction(bits, original_weight_count)
    return {"layers_enabled": 1, "of_layers": n_layers, "keep_experts": keep_experts,
            "extra_bits": bits, "extra_bpw_exact": str(f), "extra_bpw": float(f),
            "sealed_s64_with_doctor_bpw": 0.999769787,
            "still_legal_under_one_bit_ceiling": 0.999769787 + float(f) <= 1.0,
            "gate_cols_unused_note": f"gate/up ({gate_cols} cols) carry only the mode bit, "
                                     "modulation loses there"}


# ── measurement run ───────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Layer-0 column-modulated codec vs baseline.")
    ap.add_argument("--source", default="models/qwen3-235b-a22b")
    ap.add_argument("--layers", default="0,46")
    ap.add_argument("--organs", default="gate,down")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--out", default="reports/subbit_reset/LANE_A2_LAYER0_CODEC.json")
    args = ap.parse_args(argv)

    r = SafetensorsIndexReader(args.source)
    if not r.source_present():
        raise SystemExit("source shards absent")
    cells = []
    for L in (int(x) for x in args.layers.split(",")):
        for organ in args.organs.split(","):
            dim, k, st, suf = SPECS[organ]
            mats = [r.bf16(f"model.layers.{L}.mlp.experts.{e}.{suf}.weight").astype(np.float32)
                    for e in range(args.experts)]     # <= 4 tensors resident, freed below
            c = cell(mats, dim=dim, k=k, stages=st, iters=args.iters)
            c["cell"] = {"layer": L, "organ": organ, "n_experts": args.experts,
                         "dim": dim, "k": k, "stages": st}
            c["characterisation"] = characterise(mats[0])
            cells.append(c)
            print(json.dumps({"layer": L, "organ": organ,
                              "rel_error": c["rel_error"],
                              "gap": c["gap_to_shannon_decades"]}, sort_keys=True), flush=True)
            del mats

    l0d = next((c for c in cells if c["cell"]["layer"] == 0 and c["cell"]["organ"] == "down"), None)
    closed = l0d["gap_to_shannon_decades"]["closed_fraction"] if l0d else 0.0
    out = {
        "schema": SCHEMA,
        "lever": "per-column bf16 scale preconditioner + c^2 diagonal importance, same index rate",
        "falsification_bar": "close >= half of the 1.328-decade L0 down gap at equal-or-lower bits",
        "verdict": ("PARTIAL_WIN_BELOW_BAR" if closed < 0.5 else "WIN"),
        "l0_down_gap_closed_fraction": closed,
        "bits_charged": layer0_bpw_surcharge(),
        "honesty": ("weight-space reconstruction error only; NOT a capability claim. rel_error is "
                    "the row-normalised quantity qwen_shannon_bound uses, which weights every row "
                    "equally regardless of row norm."),
        "dead_sub_levers": ["closed-form refit of the column scales after coding (0.19 -> 0.39 on "
                            "L0 down, codebook goes stale)",
                            "column preconditioning WITHOUT the c^2 importance (0.37 -> 1.15 on "
                            "L0 down: it changes the objective, not just the geometry)",
                            "partial exponents c^a for a in {0.25,0.5,0.75}: monotone, no interior "
                            "optimum on either organ"],
        "cells": cells,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("verdict", "l0_down_gap_closed_fraction", "bits_charged")}, indent=2,
                     sort_keys=True))
    return 0


def demo() -> None:
    """Self-check on synthetic sources with KNOWN structure. Fails if the logic breaks."""
    rng = np.random.default_rng(0)
    dim, k, st = 16, 256, 1

    # 1. exact bit accounting
    assert col_bits((4096, 1536)) == 1536 * 16 + 1
    b = layer0_bpw_surcharge()
    assert Fraction(b["extra_bpw_exact"]) == Fraction(64 * (1536 * 16 + 1) + 128, 235093634560)
    assert b["still_legal_under_one_bit_ceiling"], b

    # 2. a source WITH per-column scale structure: modulation must win
    base = rng.standard_normal((512, 256)).astype(np.float32)
    c = (10.0 ** rng.uniform(-1.5, 1.5, 256)).astype(np.float32)
    structured = [base * c[None, :],
                  (rng.standard_normal((512, 256)).astype(np.float32)) * c[None, :]]
    s = cell(structured, dim=dim, k=k, stages=st, iters=8)
    assert s["rel_error"]["modulated"] < 0.8 * s["rel_error"]["baseline"], s["rel_error"]
    assert s["gap_to_shannon_decades"]["closed"] > 0.0, s

    # 3. an iid source with NO column structure: modulation must not help materially
    flat = [rng.standard_normal((512, 256)).astype(np.float32) for _ in range(2)]
    f = cell(flat, dim=dim, k=k, stages=st, iters=8)
    assert f["rel_error"]["modulated"] > 0.9 * f["rel_error"]["baseline"], f["rel_error"]

    # 4. characterise separates the two
    assert characterise(structured[0])["col_rms_span_decades_after_row_norm"] > 2.0
    assert characterise(flat[0])["col_rms_span_decades_after_row_norm"] < 0.5

    # 5. the surcharge is really paid: modulated rate strictly exceeds baseline rate
    assert s["rate_bits_per_dim"]["modulated_total"] > s["rate_bits_per_dim"]["baseline"]
    print(json.dumps({"ok": True, "structured": s["rel_error"], "flat": f["rel_error"]}, indent=2))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        raise SystemExit(main())
