#!/usr/bin/env python3
"""G066: does entanglement entropy predict where a family collapses, better than
singular values alone? If it does not, the obligation says KILL it rather than
keeping it as flavour, so this is written to be able to return that.

The point of a predictor is to skip an expensive search. Locating a site's cliff
by measurement costs a full codec sweep against real activations; a spectral
statistic costs one eigendecomposition. So the test is whether the cheap number
ranks sites the same way the expensive one does.

WHAT IS BEING PREDICTED, defined so it is not circular. G067 measured, per site,
the flat family's output error against bits. The target here is the bits/elem at
which that curve crosses a FIXED functional error level, interpolated -- "how many
bits does this tensor need". It varies per site, it is not a family member, and it
is exactly the quantity a bond-dimension predictor claims to anticipate.

  E_ref is the MEDIAN flat-q3 error across sites, so the level sits inside the
  measured range at every site rather than being a round number chosen to flatter
  a correlation.

THE CANDIDATE, and its rivals:

  S_vn        von Neumann entropy of the normalized singular spectrum across the
              row|col cut -- the entanglement entropy of the matrix bipartition
  stable_rank ||W||_F^2 / ||W||_2^2                      singular values alone
  eff_rank_90 count of singular values reaching 90% energy   singular values alone

S_vn is a function of the same spectrum, so this is not entropy-versus-something-
else; it is whether the entropy FUNCTIONAL of the spectrum beats simpler
functionals of it. Stated plainly because the framing invites the other reading.

Spearman rank correlation is the scoring rule, chosen before looking: the claim is
that the predictor RANKS sites, and rank correlation does not reward a lucky
linear fit on ten points.

  ./tools/gravity_entanglement_predictor.py --out receipts/.../G066_PREDICTOR.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PTP = ROOT / "receipts/ascent-2026-08-16/G067_PHASE_TRANSITION.json"

SUFFIX = {"mlp.gate_proj": "mlp.gate_proj.weight", "mlp.down_proj": "mlp.down_proj.weight",
          "linear_attn.out_proj": "linear_attn.out_proj.weight",
          "self_attn.o_proj": "self_attn.o_proj.weight",
          "self_attn.q_proj": "self_attn.q_proj.weight"}


def spectrum(w):
    """Singular values via the smaller Gram matrix -- a full SVD of a 17408x5120
    is not needed and not affordable at ten sites."""
    g = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
    ev = np.linalg.eigvalsh(g.astype(np.float64))
    return np.sqrt(np.clip(ev, 0, None))[::-1]


def stats(s):
    e = s ** 2
    p = e / e.sum()
    p = p[p > 0]
    energy = np.cumsum(e) / e.sum()
    return {"S_vn_bits": float(-(p * np.log2(p)).sum()),
            "stable_rank": float(e.sum() / e.max()),
            "eff_rank_90": int(np.searchsorted(energy, 0.90) + 1),
            "n_singular_values": int(s.size)}


def spearman(a, b):
    def rank(v):
        o = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return o.astype(float)
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def crossing(curve, level):
    pts = sorted((c["bits"], c["err"]) for c in curve)
    for i in range(len(pts) - 1):
        (b0, e0), (b1, e1) = pts[i], pts[i + 1]
        if (e0 - level) * (e1 - level) <= 0 and e0 != e1:
            return float(b0 + (b1 - b0) * (level - e0) / (e1 - e0))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    ptp = json.loads(PTP.read_text())

    flat = {(r["organ"], r["layer"]): r["curve"] for r in ptp["families"]["FLAT"]}
    q3 = [next(c["err"] for c in cv if abs(c["bits"] - 3.25) < 1e-9) for cv in flat.values()]
    E_ref = float(np.median(q3))
    print(f"E_ref = median flat-q3 output error across sites = {E_ref:.6f}")

    rows = []
    for (organ, L), cv in flat.items():
        target = crossing(cv, E_ref)
        if target is None:
            print(f"  {organ} L{L}: no crossing of E_ref -- excluded")
            continue
        w = load_tensor(f"language_model.model.layers.{L}.{SUFFIX[organ]}").astype(np.float32)
        st = stats(spectrum(w))
        del w
        st.update({"organ": organ, "layer": L, "bits_needed_for_E_ref": target})
        rows.append(st)
        print(f"  {organ:<22} L{L:<3} bits_needed {target:6.4f}   S_vn {st['S_vn_bits']:7.4f}   "
              f"stable_rank {st['stable_rank']:9.2f}   eff_rank_90 {st['eff_rank_90']:5d}")

    y = [r["bits_needed_for_E_ref"] for r in rows]
    scores = {k: abs(spearman([r[k] for r in rows], y))
              for k in ("S_vn_bits", "stable_rank", "eff_rank_90")}
    best_rival = max(("stable_rank", "eff_rank_90"), key=lambda k: scores[k])
    wins = scores["S_vn_bits"] > scores[best_rival]

    print(f"\n|Spearman| against bits needed, n={len(rows)} sites")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        tag = "  <- candidate" if k == "S_vn_bits" else "  (singular values alone)"
        print(f"  {k:<14}{v:.4f}{tag}")
    print(f"\nVERDICT: entanglement entropy {'BEATS' if wins else 'DOES NOT BEAT'} "
          f"the best singular-value rival ({best_rival} at {scores[best_rival]:.4f}). "
          f"{'Kept.' if wins else 'KILLED, as the obligation requires.'}")

    doc = {
        "schema": "hawking.nos.entanglement_predictor.v1",
        "obligation": "G066 -- entanglement entropy as a predictor of the collapse point",
        "target": {"definition": "bits/elem at which the flat family's measured output error "
                                 "crosses a fixed level, interpolated -- 'how many bits does this "
                                 "tensor need'",
                   "E_ref": E_ref,
                   "E_ref_choice": "median flat-q3 output error across sites, so the level sits "
                                   "inside the measured range at every site rather than being a "
                                   "round number chosen to flatter a correlation",
                   "source": str(PTP.relative_to(ROOT))},
        "framing_note": ("S_vn is a functional of the same singular spectrum as its rivals, so this "
                         "measures whether the ENTROPY functional beats simpler functionals of that "
                         "spectrum -- not entropy against something unrelated."),
        "scoring_rule": "Spearman rank correlation, chosen before looking, because the claim is "
                        "that the predictor RANKS sites and rank correlation does not reward a "
                        "lucky linear fit on ten points",
        "sites": rows, "abs_spearman": scores,
        "best_singular_value_rival": best_rival,
        "entanglement_entropy_wins": wins,
        "verdict": ("KEPT" if wins else "KILLED -- the obligation requires killing a predictor that "
                    "does not beat singular values alone, rather than keeping it as flavour"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
