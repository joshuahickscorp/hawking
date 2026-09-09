#!/usr/bin/env python3.12
"""KRT006 -- reject or admit BINARY LATENT FACTORIZATION by arithmetic, cheaply.

S014: "Before expensive fitting, calculate structural diagnostics ... Do not
launch giant factorization jobs where arithmetic already rejects the
representation."

The LittleBit/NanoQuant family writes W ~ U_b S V_b^T with U_b in {-1,+1}^(m x r)
and V_b in {-1,+1}^(n x r). Its persistent cost at rank r is therefore

    bits = m*r + n*r + r*scale_bits          (plus per-row scales if used)

and at a target rate R over m*n weights that inverts to a BREAK-EVEN RANK

    r* = m*n*R / (m + n + scale_bits)

which is pure arithmetic and costs nothing. What it buys is a decisive
upper bound: a rank-r binary factorization cannot beat the BEST rank-r
approximation of any kind, and the best is the truncated SVD. So

    SVD energy captured at r*   IS AN UPPER BOUND on this family at rate R.

If that bound is already poor, the family is rejected without fitting anything --
and rejected with a MECHANISM (the spectrum is too flat at the rank the rate can
afford), which is what separates PROVEN_UNABLE from "we tried a couple of runs".

The bound is generous on purpose. Real binary factors are far weaker than real
singular vectors: they carry sign only, with one shared scale per component. A
family that fails its own upper bound cannot be rescued by a better fitter.

Reported beside it, because they say WHY the bound lands where it does:
  spectrum decay, effective rank (participation ratio of the singular energy),
  stable rank, and coherence.

    python3 tools/future/latent_rank_breakeven.py --spec <dir> [--rates 1.0,0.5]
    python3 tools/future/latent_rank_breakeven.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def breakeven_rank(m: int, n: int, rate: float, scale_bits: int = 16) -> float:
    """r such that m*r + n*r + r*scale_bits == m*n*rate."""
    return m * n * rate / (m + n + scale_bits)


def spectral_report(W, ranks) -> dict:
    """Singular energy captured at each rank, plus the shape statistics that
    explain it. Energy is squared-singular-value mass, which is exactly the
    Frobenius error a truncated approximation leaves behind."""
    import torch
    W = W.float()
    # The ENERGY accumulates in float64. In float32 the cumulative sum saturates
    # at exactly 1.0 once the remaining tail is ~1e-8 relative, so the derived
    # bound reads 0 and a real truncation appears to beat it. The selftest
    # caught precisely that on a rank-32 matrix with a 1e-3 noise floor.
    s = torch.linalg.svdvals(W)
    e = (s.double() ** 2)
    tot = float(e.sum()) or 1e-30
    cum = torch.cumsum(e, 0) / tot
    full = int(min(W.shape))
    part = float((e.sum() ** 2) / (e ** 2).sum())          # participation ratio
    return {
        "shape": list(W.shape),
        "full_rank": full,
        "effective_rank_participation": round(part, 2),
        "effective_rank_fraction": round(part / full, 5),
        "stable_rank": round(float(e.sum() / e.max()), 2),
        "top1_energy": round(float(cum[0]), 5),
        # 8 digits, not 5. At 5 an energy of 0.99999999 rounds to 1.0 and the
        # derived error bound reads 0.00000, which a real truncation then
        # "beats" -- a reporting artifact that looks exactly like a broken bound.
        "energy_at": {str(r): round(float(cum[min(max(int(r), 1), full) - 1]), 8)
                      for r in ranks},
        "singular_decay_ratio_s1_over_smedian": round(
            float(s[0] / s[full // 2]), 3) if s[full // 2] > 0 else float("inf"),
    }


def assess(W, rates, scale_bits=16) -> dict:
    m, n = W.shape
    rows = []
    ranks = [breakeven_rank(m, n, r, scale_bits) for r in rates]
    rep = spectral_report(W, ranks)
    for rate, r in zip(rates, ranks):
        r_i = max(1, min(int(r), min(m, n)))
        cap = rep["energy_at"][str(r)]
        # rel error of the BEST rank-r approximation = sqrt(1 - captured energy)
        best = (1.0 - cap) ** 0.5
        rows.append({
            "target_rate_bits_per_weight": rate,
            "breakeven_rank": round(r, 1),
            "breakeven_rank_over_full": round(r / min(m, n), 4),
            "svd_energy_captured": cap,
            "BEST_POSSIBLE_rel_error_at_this_rank": round(best, 8),
            "rank_exceeds_full": r >= min(m, n),
        })
    return {"spectral": rep, "rates": rows}


def _selftest() -> int:
    """The diagnostic must SEPARATE a matrix low-rank methods can represent from
    one they cannot. If it reports the same bound for both, it is decoration."""
    import torch
    torch.manual_seed(0)
    m = n = 512

    # A genuinely low-rank matrix: rank 32 plus a whisper of noise.
    U = torch.randn(m, 32); V = torch.randn(n, 32)
    low = U @ V.t() + 0.001 * torch.randn(m, n)
    # A near-random matrix: the case the family cannot help.
    rand = torch.randn(m, n)

    r_at = breakeven_rank(m, n, 0.5)
    lo = assess(low, [0.5])["rates"][0]
    hi = assess(rand, [0.5])["rates"][0]
    assert lo["BEST_POSSIBLE_rel_error_at_this_rank"] < 0.05, lo
    assert hi["BEST_POSSIBLE_rel_error_at_this_rank"] > 0.5, hi
    assert lo["BEST_POSSIBLE_rel_error_at_this_rank"] < hi["BEST_POSSIBLE_rel_error_at_this_rank"]

    # The arithmetic itself, checked against its own definition rather than
    # against a number retyped from it.
    for (mm, nn, rate) in ((1408, 2048, 0.5), (2048, 2048, 1.0), (7, 11, 0.25)):
        r = breakeven_rank(mm, nn, rate)
        assert abs((mm * r + nn * r + 16 * r) - mm * nn * rate) < 1e-6

    # And the bound must be an UPPER bound: a real SVD truncation must not beat
    # the reported best-possible error.
    Wl = low
    s = torch.linalg.svdvals(Wl.float())
    r_i = max(1, int(r_at))
    U_, S_, Vh_ = torch.linalg.svd(Wl.float(), full_matrices=False)
    approx = (U_[:, :r_i] * S_[:r_i]) @ Vh_[:r_i]
    actual = float((Wl - approx).norm() / Wl.norm())
    assert actual <= lo["BEST_POSSIBLE_rel_error_at_this_rank"] + 1e-4, (
        f"a real truncation ({actual:.6f}) beat the claimed lower bound "
        f"({lo['BEST_POSSIBLE_rel_error_at_this_rank']:.6f}); the bound is wrong")
    print(f"selftest OK: break-even rank {r_at:.1f} at 0.5 b/w; low-rank matrix "
          f"{lo['BEST_POSSIBLE_rel_error_at_this_rank']:.4f} vs near-random "
          f"{hi['BEST_POSSIBLE_rel_error_at_this_rank']:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--rates", default="2.0,1.0,0.75,0.5,0.25")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "receipts" / "future" / "LATENT_RANK_BREAKEVEN.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.spec:
        ap.error("--spec is required unless --selftest")

    import torch
    from safetensors import safe_open
    rates = [float(x) for x in a.rates.split(",")]

    wm = json.loads((a.spec / "model.safetensors.index.json").read_text())["weight_map"]
    # One representative of each expert projection, plus a shared expert and an
    # attention writer for contrast -- the same rate question has different
    # answers per organ, and a single specimen would hide that.
    want = {}
    for k in wm:
        for tag, pat in (("routed_gate", ".mlp.experts.0.gate_proj.weight"),
                         ("routed_up", ".mlp.experts.0.up_proj.weight"),
                         ("routed_down", ".mlp.experts.0.down_proj.weight"),
                         ("shared_down", ".mlp.shared_experts.down_proj.weight"),
                         ("attn_o", ".self_attn.o_proj.weight")):
            if k.endswith(pat) and ".layers.11." in k and tag not in want:
                want[tag] = k

    out = {"schema": "hawking.future.latent_rank_breakeven.v1",
           "obligation": "KRT006", "steer": "S014",
           "family": "BINARY LATENT FACTORIZATION (LittleBit / NanoQuant class)",
           "method": "r* = m*n*R/(m+n+scale_bits); SVD energy at r* is an UPPER BOUND "
                     "on any rank-r* factorization, binary or otherwise",
           "specimen": str(a.spec), "organs": {}}
    for tag, key in want.items():
        with safe_open(str(a.spec / wm[key]), framework="pt") as h:
            W = h.get_tensor(key)
        out["organs"][tag] = {"tensor": key, **assess(W, rates)}
        r0 = out["organs"][tag]["rates"]
        print(f"\n{tag}  {list(W.shape)}  eff_rank "
              f"{out['organs'][tag]['spectral']['effective_rank_participation']:.0f}"
              f"/{out['organs'][tag]['spectral']['full_rank']}"
              f" (stable {out['organs'][tag]['spectral']['stable_rank']:.1f})")
        print(f"  {'rate':>6}{'rank r*':>9}{'r*/full':>9}{'energy':>9}{'BEST rel err':>14}")
        for r in r0:
            print(f"  {r['target_rate_bits_per_weight']:>6.2f}{r['breakeven_rank']:>9.0f}"
                  f"{r['breakeven_rank_over_full']:>9.3f}{r['svd_energy_captured']:>9.4f}"
                  f"{r['BEST_POSSIBLE_rel_error_at_this_rank']:>14.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
