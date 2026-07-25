#!/usr/bin/env python3.12
"""Lane F: generated / tied parameters. Bounded falsification of three structure hypotheses.

HONESTY. Everything here is WEIGHT-SPACE reconstruction error. Weight-space error is NEVER a
capability claim. Only a real parent-vs-packed forward can select a frontier. This module exists
to CLOSE cheap hypotheses so the next parent does not spend a week on them.

The negative-transfer atlas already killed raw inter-expert cosine redundancy (mean pairwise
1e-4) and shared low-rank bases across experts. Not yet tested, and tested here:

  F-a  templates + deltas in the ROW-NORMALIZED (scale-invariant) space rather than raw space.
       The raw-space test could in principle have been swamped by the ~15-decade row-norm span:
       one huge row dominates the flattened inner product. Test: stack E experts of one
       (layer, organ) as unit-norm vectors, take the Gram eigenspectrum. The energy fraction a
       single best shared template can explain is lambda_max / E. Mutually orthogonal experts
       give exactly 1/E. Measured raw AND row-normalized, plus the row-index-aligned mean cosine
       (which is already scale invariant per row, so it is the control on the "swamped" story).

  F-b  Kronecker / tensor-product factorisation of ONE expert tensor, W ~ sum_r A_r (x) B_r.
       Van Loan: the Frobenius-optimal rank-R Kronecker approximation is the rank-R SVD of the
       rearranged matrix R(W), so the achievable rel_error at every rank is read straight off the
       singular values - no fitting, no tuning knobs, and the answer is OPTIMAL for the family.
       Rank is capped by the bit budget at the S64 rungs (gate/up 2.5 bpw, down 0.625 bpw).

  F-c  recurrent parameter reuse ACROSS LAYERS at the same expert index: is expert e of layer L
       related to expert e of layer L+1? Measured against a control that pairs DIFFERENT expert
       indices across the same two layers, so any generic layer-to-layer similarity is subtracted
       off instead of being mistaken for index-specific tying.

BIT ACCOUNTING. F-b is charged exactly (bf16 factors + flat per-tensor metadata) and is given the
benefit of the doubt twice: the factors are billed at bf16 but the error is computed in fp32 (no
factor-rounding penalty), and no index/permutation side information is billed. If it still loses,
it loses. F-a and F-c are measurements of whether a representation EXISTS at all; they charge
nothing because there is nothing to charge until the residual energy is small, and it is not.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SCHEMA = "hawking.gravity.generated_params.v1"
SOURCE_DIR = Path("models/qwen3-235b-a22b")
REPORT = Path("reports/subbit_reset/LANE_F_GENERATED_PARAMS.json")

# S64 survivor rungs (qwen_structural_plan.GATE_RUNGS / DOWN_RUNGS). S64 g2.5 d0.625 is the
# budget-neutral legal arm at 0.948410027 complete BPW. Nothing here may exceed these rates.
GATE_SPEC = {"family": "shared_grammar", "dim": 8, "k": 1024, "stages": 2}    # 2.5 bpw
DOWN_SPEC = {"family": "shared_grammar", "dim": 16, "k": 1024, "stages": 1}   # 0.625 bpw
GATE_BPW, DOWN_BPW = 2.5, 0.625
CLUSTER = 64                     # survivor experts sharing one codebook under S64
METADATA_BITS = 64 * 8           # same flat per-tensor charge gravity_forge bills

_EPS = 1e-12


# ── F-b: Kronecker (Van Loan rearrangement) ───────────────────────────────────────────────────
def kron_splits(shape: tuple[int, int], top: int = 8) -> list[tuple[int, int, int, int]]:
    """The `top` cheapest (m1,n1,m2,n2) with m1*m2=rows, n1*n2=cols, by m1*n1 + m2*n2.

    That sum is the per-rank bit cost, so cheapest = most rank per bit. Several splits tie on
    cost while inducing genuinely different Kronecker families (a [1,2048] x [1536,2] split is
    not the same hypothesis as [48,64] x [32,64]), so we score all of them and keep the best -
    a falsification must beat the method's best case, not an arbitrary tie-break.
    """
    rows, cols = shape
    div = lambda n: [d for d in range(1, n + 1) if n % d == 0]
    cand = [(m1 * n1 + (rows // m1) * (cols // n1), m1, n1, rows // m1, cols // n1)
            for m1 in div(rows) for n1 in div(cols)]
    return [(m1, n1, m2, n2) for _, m1, n1, m2, n2 in sorted(cand)[:top]]


def kron_split(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    return kron_splits(shape, 1)[0]


def kron_svals(w: np.ndarray, split: tuple[int, int, int, int]) -> np.ndarray:
    """Singular values of the Van Loan rearrangement R(W)[(i1,j1),(i2,j2)] = W[i1*m2+i2, j1*n2+j2].

    Rank-R SVD of R(W) IS the Frobenius-optimal sum of R Kronecker products (Van Loan & Pitsianis).
    """
    m1, n1, m2, n2 = split
    a = np.asarray(w, np.float32).reshape(m1, m2, n1, n2)
    r = np.ascontiguousarray(a.transpose(0, 2, 1, 3)).reshape(m1 * n1, m2 * n2)
    return np.linalg.svd(r, compute_uv=False)


def kron_rel_error(svals: np.ndarray, rank: int) -> float:
    """Exact relative Frobenius error of the best rank-R Kronecker approximation."""
    s2 = np.asarray(svals, np.float64) ** 2
    tot = s2.sum()
    return float(math.sqrt(max(0.0, s2[rank:].sum()) / max(tot, _EPS)))


def kron_bits(split: tuple[int, int, int, int], rank: int) -> int:
    """Exact artifact cost: R bf16 A-factors + R bf16 B-factors + flat per-tensor metadata."""
    m1, n1, m2, n2 = split
    return rank * (m1 * n1 + m2 * n2) * 16 + METADATA_BITS


def kron_rank_for_budget(split: tuple[int, int, int, int], n_weights: int, bpw: float) -> int:
    """Largest rank whose COMPLETE bits fit `bpw` over the original weight count."""
    m1, n1, m2, n2 = split
    budget = int(math.floor(bpw * n_weights))
    return max(0, (budget - METADATA_BITS) // ((m1 * n1 + m2 * n2) * 16))


# ── F-a / F-c: similarity in raw vs row-normalized space ──────────────────────────────────────
def _rownorm(w: np.ndarray) -> np.ndarray:
    a = np.asarray(w, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64).ravel()
    return v / max(np.linalg.norm(v), _EPS)


def template_share(mats: list[np.ndarray], *, row_normalized: bool) -> dict[str, float]:
    """Energy fraction ONE shared template can explain across E experts.

    Stack the experts as unit vectors, form the E x E Gram. The best rank-1 (single template with
    a per-expert coefficient) explains lambda_max / E of the total energy. Mutually orthogonal
    experts give exactly 1/E, which is the null. `excess_over_orthogonal` is the whole claim.
    """
    e = len(mats)
    v = np.stack([_unit(_rownorm(m) if row_normalized else m) for m in mats])
    g = v @ v.T
    ev = np.linalg.eigvalsh(g)
    off = g[~np.eye(e, dtype=bool)]
    return {
        "n_experts": e,
        "mean_abs_offdiag_cosine": float(np.abs(off).mean()),
        "max_abs_offdiag_cosine": float(np.abs(off).max()),
        "lambda_max": float(ev[-1]),
        "rank1_energy_share": float(ev[-1] / e),
        "orthogonal_null": 1.0 / e,
        "excess_over_orthogonal": float(ev[-1] / e - 1.0 / e),
    }


def row_aligned_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over rows of cos(a_i, b_i). Already scale-invariant per row - the 'swamped' control."""
    x, y = _rownorm(a), _rownorm(b)
    return float((x * y).sum(1).mean())


def delta_residual(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Code b as c*a + delta with the optimal scalar c. residual_frac = 1 - cos^2(a,b).

    residual_frac ~ 1 means the reference carries no information about the target and delta
    coding buys exactly nothing.
    """
    x, y = _unit(a), _unit(b)
    c = float(x @ y)
    return {"flat_cosine": c, "residual_energy_frac": float(1.0 - c * c)}


# ── codec baseline at the identical rate ──────────────────────────────────────────────────────
def codec_baseline(mats: list[np.ndarray], spec: dict[str, Any]) -> dict[str, Any]:
    """The incumbent scale-invariant VQ at the same rung, so Kronecker is scored against a rival."""
    import qwen_function_aware_codec as C
    import qwen_subhalfbit_search as SHB
    books = C.fit(mats, dim=spec["dim"], k=spec["k"], stages=spec["stages"], iters=6)
    errs = [C.rel_error(m, C.apply_refit(books, m, dim=spec["dim"])) for m in mats]
    bits = SHB.expert_bits(mats[0].shape, spec, CLUSTER)
    return {"spec": dict(spec), "rel_error_mean": float(np.mean(errs)),
            "complete_bits": int(bits),
            "complete_bpw": round(bits / (mats[0].shape[0] * mats[0].shape[1]), 6)}


# ── real measurement ──────────────────────────────────────────────────────────────────────────
def _reader():
    from qwen_real_forward import SafetensorsIndexReader
    return SafetensorsIndexReader(str(SOURCE_DIR))


def _w(r, layer: int, expert: int, organ: str) -> np.ndarray:
    return r.bf16(f"model.layers.{layer}.mlp.experts.{expert}.{organ}.weight").astype(np.float32)


def measure(layer: int = 46, experts: tuple[int, ...] = (3, 7, 19, 71),
            next_layer: int = 47) -> dict[str, Any]:
    r = _reader()
    if not r.source_present():
        raise SystemExit("parent weights absent; Lane F refuses to run on a synthetic twin")
    out: dict[str, Any] = {"schema": SCHEMA, "source": str(SOURCE_DIR),
                           "honesty": "weight-space reconstruction error only; NOT a capability claim",
                           "rungs": {"gate_up_bpw": GATE_BPW, "down_bpw": DOWN_BPW,
                                     "gate_spec": GATE_SPEC, "down_spec": DOWN_SPEC,
                                     "cluster": CLUSTER},
                           "layer": layer, "experts": list(experts), "next_layer": next_layer,
                           "methods": {}}
    t0 = time.time()

    # ── F-a: templates in normalized space (<=4 tensors resident) ────────────────────────────
    fa: dict[str, Any] = {}
    for organ in ("gate_proj", "down_proj"):
        mats = [_w(r, layer, e, organ) for e in experts]
        fa[organ] = {
            "raw": template_share(mats, row_normalized=False),
            "row_normalized": template_share(mats, row_normalized=True),
            "row_aligned_cosine_mean": float(np.mean(
                [row_aligned_cosine(mats[i], mats[j])
                 for i in range(len(mats)) for j in range(i + 1, len(mats))])),
        }
        del mats
    out["methods"]["F_a_templates_normalized_space"] = fa

    # ── F-b: Kronecker at the S64 rungs (1 tensor resident) ──────────────────────────────────
    fb: dict[str, Any] = {}
    for organ, bpw, spec in (("gate_proj", GATE_BPW, GATE_SPEC), ("down_proj", DOWN_BPW, DOWN_SPEC)):
        w = _w(r, layer, experts[0], organ)
        n = w.shape[0] * w.shape[1]
        tried = []
        for split in kron_splits(w.shape):
            sv = kron_svals(w, split)
            rank = min(kron_rank_for_budget(split, n, bpw), len(sv))
            bits = kron_bits(split, rank)
            tried.append({
                "split": list(split), "max_rank": int(len(sv)), "rank_at_budget": int(rank),
                "complete_bits": int(bits), "complete_bpw": round(bits / n, 6),
                "rel_error_at_budget": kron_rel_error(sv, rank),
                "rel_error_at_full_rank": round(kron_rel_error(sv, len(sv)), 9),
                "top1_energy_share": round(float(sv[0] ** 2 / max(float((sv ** 2).sum()), _EPS)), 6),
            })
            del sv
        best = min(tried, key=lambda x: x["rel_error_at_budget"])
        fb[organ] = {"shape": list(w.shape), "splits_tried": tried, **best,
                     "energy_share_at_budget": round(1.0 - best["rel_error_at_budget"] ** 2, 6)}
        # rival at the identical rung, same tensor plus one sibling (2 tensors resident max)
        sib = _w(r, layer, experts[1], organ)
        fb[organ]["codec_rival"] = codec_baseline([w, sib], spec)
        fb[organ]["kron_beats_codec"] = bool(
            fb[organ]["rel_error_at_budget"] < fb[organ]["codec_rival"]["rel_error_mean"])
        del w, sib
    out["methods"]["F_b_kronecker"] = fb

    # ── F-c: cross-layer same-expert-index tying (<=3 tensors resident) ──────────────────────
    fc: dict[str, Any] = {}
    for organ in ("gate_proj", "down_proj"):
        rows = []
        for e in experts[:3]:
            a = _w(r, layer, e, organ)
            b = _w(r, next_layer, e, organ)                       # same index, next layer
            ctl = _w(r, next_layer, (e + 37) % 128, organ)         # different index, same layer pair
            rows.append({
                "expert": int(e),
                "same_index": {**delta_residual(a, b), "row_aligned_cosine": row_aligned_cosine(a, b)},
                "control_diff_index": {**delta_residual(a, ctl),
                                       "row_aligned_cosine": row_aligned_cosine(a, ctl)},
            })
            del a, b, ctl
        same = float(np.mean([x["same_index"]["residual_energy_frac"] for x in rows]))
        ctlm = float(np.mean([x["control_diff_index"]["residual_energy_frac"] for x in rows]))
        fc[organ] = {"pairs": rows, "same_index_residual_frac_mean": same,
                     "control_residual_frac_mean": ctlm,
                     "index_specific_gain": round(ctlm - same, 8)}
    out["methods"]["F_c_cross_layer_tying"] = fc

    out["verdicts"] = _verdicts(out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def _verdicts(o: dict[str, Any]) -> dict[str, str]:
    fa = o["methods"]["F_a_templates_normalized_space"]
    fb = o["methods"]["F_b_kronecker"]
    fc = o["methods"]["F_c_cross_layer_tying"]
    v = {}
    # F-a lives only if row-normalization buys real shared energy over the orthogonal null.
    exc = max(fa[k]["row_normalized"]["excess_over_orthogonal"] for k in fa)
    v["F_a_templates_normalized_space"] = (
        "DEAD: row-normalized experts are still mutually orthogonal; no shared template exists "
        f"(max excess over the 1/E null {exc:.2e})" if exc < 0.05 else
        f"LIVE: shared template explains {exc:.3f} above the orthogonal null - re-test with a "
        "delta codec at the S64 rungs")
    # F-b lives only if it beats the incumbent codec at the same complete rate.
    wins = [k for k in fb if fb[k]["kron_beats_codec"]]
    v["F_b_kronecker"] = (
        "DEAD: at the S64 rungs the Frobenius-OPTIMAL Kronecker factorisation loses to the "
        "incumbent scale-invariant VQ on every organ tested" if not wins else
        f"LIVE on {wins}: optimal Kronecker beats the codec at the identical complete rate")
    g = max(fc[k]["index_specific_gain"] for k in fc)
    v["F_c_cross_layer_tying"] = (
        f"DEAD: same-index cross-layer pairs are no more predictive than different-index controls "
        f"(max index-specific residual gain {g:.2e})" if g < 0.05 else
        f"LIVE: same-index tying removes {g:.3f} more residual energy than the control")
    return v


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def demo() -> None:
    rng = np.random.default_rng(0)

    # kron_split is balanced and exactly factorises the shape.
    m1, n1, m2, n2 = kron_split((1536, 4096))
    assert m1 * m2 == 1536 and n1 * n2 == 4096
    assert m1 * n1 + m2 * n2 <= 6000, (m1, n1, m2, n2)

    # An exact Kronecker product is rank 1 in the rearranged space -> ~0 error at R=1.
    A, B = rng.standard_normal((8, 4)).astype(np.float32), rng.standard_normal((4, 8)).astype(np.float32)
    W = np.kron(A, B)
    sv = kron_svals(W, (8, 4, 4, 8))
    assert kron_rel_error(sv, 1) < 1e-5, kron_rel_error(sv, 1)
    assert kron_rel_error(sv, 0) > 0.99
    # ... and a generic matrix is NOT: rank 1 must leave most of the energy behind.
    assert kron_rel_error(kron_svals(rng.standard_normal((8, 8)).astype(np.float32), (2, 2, 4, 4)), 1) > 0.5

    # bit accounting: the budgeted rank fits, one more rank does not.
    split = kron_split((1536, 4096))
    rk = kron_rank_for_budget(split, 1536 * 4096, 2.5)
    assert kron_bits(split, rk) <= 2.5 * 1536 * 4096 < kron_bits(split, rk + 1)

    # template_share: orthogonal experts hit the 1/E null, identical experts hit 1.0.
    orth = [rng.standard_normal((64, 128)).astype(np.float32) for _ in range(4)]
    s = template_share(orth, row_normalized=True)
    assert abs(s["rank1_energy_share"] - 0.25) < 0.05, s
    same = [x.copy() for x in [orth[0]] * 4]
    assert template_share(same, row_normalized=True)["rank1_energy_share"] > 0.99

    # row-norm span really can swamp a flat cosine: two experts sharing ONE huge row look
    # correlated raw and orthogonal normalized. This is the failure mode F-a had to rule out.
    big = rng.standard_normal((1, 128)).astype(np.float32) * 1e4
    p = np.vstack([big, rng.standard_normal((63, 128)).astype(np.float32)])
    q = np.vstack([big, rng.standard_normal((63, 128)).astype(np.float32)])
    assert template_share([p, q], row_normalized=False)["mean_abs_offdiag_cosine"] > 0.9
    assert template_share([p, q], row_normalized=True)["mean_abs_offdiag_cosine"] < 0.3

    # delta_residual: a scaled copy is free, an independent matrix is not.
    a = rng.standard_normal((32, 64)).astype(np.float32)
    assert delta_residual(a, a * -3.0)["residual_energy_frac"] < 1e-6
    assert delta_residual(a, rng.standard_normal((32, 64)).astype(np.float32))["residual_energy_frac"] > 0.9

    # verdict wiring: a dead-looking payload must read DEAD on all three.
    dead = {"methods": {
        "F_a_templates_normalized_space": {"gate_proj": {"row_normalized": {"excess_over_orthogonal": 1e-4}}},
        "F_b_kronecker": {"gate_proj": {"kron_beats_codec": False}},
        "F_c_cross_layer_tying": {"gate_proj": {"index_specific_gain": 1e-4}}}}
    assert all(v.startswith("DEAD") for v in _verdicts(dead).values())
    dead["methods"]["F_b_kronecker"]["gate_proj"]["kron_beats_codec"] = True
    assert _verdicts(dead)["F_b_kronecker"].startswith("LIVE")
    print("qwen_generated_params selftest OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--layer", type=int, default=46)
    ap.add_argument("--next-layer", type=int, default=47)
    ap.add_argument("--out", default=str(REPORT))
    a = ap.parse_args(argv)
    if a.selftest:
        demo()
    if a.measure:
        rep = measure(layer=a.layer, next_layer=a.next_layer)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2) + "\n")
        print(json.dumps(rep["verdicts"], indent=2))
        print(f"wrote {a.out}")
    if not (a.selftest or a.measure):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
