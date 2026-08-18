#!/usr/bin/env python3
"""G035: does cross-layer sharing beat independent compression at matched bits?

The obligation is already OPEN on a spectral observation -- real layer pairs are
closer to each other than to random, most strongly between adjacent layers. That
is necessary, not sufficient: two matrices can have near-identical spectra and
completely unaligned subspaces, and only the subspaces can be shared.

So this asks the acceptance's question directly. At MATCHED total parameters:

  INDEPENDENT   each layer gets its own rank-r factorization
                params = 2 * r * (m + n)
  SHARED        one column basis for both layers plus per-layer coefficients
                params = r_s * m + 2 * r_s * n

Sharing saves r*m on the basis, so at matched parameters it can afford a LARGER
rank. That is exactly the trade the obligation wants priced, and there is no
separate alignment matrix to forget to count: the shared basis IS the alignment,
and it is inside the parameter budget.

Error is measured in FUNCTION space on the thick v2 capture, not on weight
cosine, because weight space has already inverted a ranking once in this campaign
(G033: 3 planes beat flat q3 on hold and lost to it on real activations).

A far-apart pair is included as a control. If sharing helps adjacent layers it
should help distant ones measurably less, and if it helps both equally the effect
is not layer affinity but the factorization itself.

  ./tools/gravity_share_crosslayer.py --out receipts/ascent-2026-08-16/G035_CROSSLAYER_SHARE.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor  # noqa: E402
from gravity_function_space_rank import load_activations, out_error, SITE  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FACTOR_BITS = 16  # f16 factors, the same convention the campaign's scales use


def randomized_svd(a, rank, oversample=16, power=1, seed=0):
    """Range-finder SVD. Exact SVD of a 17408x10240 matrix is not affordable and
    is not needed -- only the leading subspace is being compared."""
    rng = np.random.default_rng(seed)
    om = rng.standard_normal((a.shape[1], rank + oversample)).astype(np.float32)
    y = a @ om
    for _ in range(power):
        y = a @ (a.T @ y)
    q, _ = np.linalg.qr(y)
    b = q.T @ a
    ub, s, vt = np.linalg.svd(b, full_matrices=False)
    return (q @ ub)[:, :rank], s[:rank], vt[:rank]


def independent(w, rank):
    u, s, vt = randomized_svd(w, rank)
    return u @ (s[:, None] * vt)


def shared_basis(w1, w2, rank):
    """One column basis for both layers, per-layer coefficients."""
    stacked = np.concatenate([w1, w2], axis=1)
    u, _, _ = randomized_svd(stacked, rank)
    # Least squares in the shared basis; u is orthonormal so it is a projection.
    return u @ (u.T @ w1), u @ (u.T @ w2), u


def params_independent(m, n, r):
    return 2 * r * (m + n)


def params_shared(m, n, r):
    return r * m + 2 * r * n


def rank_shared_matching(m, n, r_ind):
    """Largest shared rank whose parameters fit the independent budget."""
    budget = params_independent(m, n, r_ind)
    return int(budget // (m + 2 * n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    pairs = [("gate_proj", 30, 31, "adjacent"),
             ("down_proj", 30, 31, "adjacent"),
             ("gate_proj", 15, 47, "far control")]

    rows_out = []
    for organ, l1, l2, kind in pairs:
        w1 = load_tensor(f"language_model.model.layers.{l1}.mlp.{organ}.weight").astype(np.float32)
        w2 = load_tensor(f"language_model.model.layers.{l2}.mlp.{organ}.weight").astype(np.float32)
        m, n = w1.shape
        r_ind = a.rank
        r_sh = rank_shared_matching(m, n, r_ind)

        x1, _ = load_activations(SITE[organ], l1, a.rows, n)
        x2, _ = load_activations(SITE[organ], l2, a.rows, n)

        i1 = independent(w1, r_ind)
        i2 = independent(w2, r_ind)
        e_i1, c_i1 = out_error(x1, w1, i1)
        e_i2, c_i2 = out_error(x2, w2, i2)
        del i1, i2

        s1, s2, _ = shared_basis(w1, w2, r_sh)
        e_s1, c_s1 = out_error(x1, w1, s1)
        e_s2, c_s2 = out_error(x2, w2, s2)
        del s1, s2

        p_ind = params_independent(m, n, r_ind)
        p_sh = params_shared(m, n, r_sh)
        elems = 2 * m * n
        row = {
            "organ": organ, "layers": [l1, l2], "kind": kind, "shape": [m, n],
            "rank_independent": r_ind, "rank_shared": r_sh,
            "params_independent": p_ind, "params_shared": p_sh,
            "bits_per_elem_independent": FACTOR_BITS * p_ind / elems,
            "bits_per_elem_shared": FACTOR_BITS * p_sh / elems,
            "independent_out_rel_fro": [e_i1, e_i2],
            "shared_out_rel_fro": [e_s1, e_s2],
            "independent_mean": (e_i1 + e_i2) / 2,
            "shared_mean": (e_s1 + e_s2) / 2,
            "shared_beats_independent": (e_s1 + e_s2) / 2 < (e_i1 + e_i2) / 2,
        }
        rows_out.append(row)
        print(f"{organ} L{l1}/L{l2} ({kind}): "
              f"independent r={r_ind} {row['bits_per_elem_independent']:.4f} b/elem "
              f"err {row['independent_mean']:.5f} | "
              f"shared r={r_sh} {row['bits_per_elem_shared']:.4f} b/elem "
              f"err {row['shared_mean']:.5f} -> "
              f"{'SHARED WINS' if row['shared_beats_independent'] else 'independent wins'}")
        del w1, w2, x1, x2

    adj = [r for r in rows_out if r["kind"] == "adjacent"]
    far = [r for r in rows_out if r["kind"] == "far control"]
    adj_gain = sum(r["independent_mean"] - r["shared_mean"] for r in adj) / max(len(adj), 1)
    far_gain = sum(r["independent_mean"] - r["shared_mean"] for r in far) / max(len(far), 1)
    print(f"\nmean error reduction from sharing: adjacent {adj_gain:+.5f}, far control {far_gain:+.5f}")

    doc = {
        "schema": "hawking.nos.crosslayer_share.v1",
        "obligation": "G035 -- shared basis plus per-layer coefficients vs independent, matched bits",
        "method": "Randomized-SVD range finder; only the leading subspace is compared, an exact SVD "
                  "of a 17408x10240 matrix is neither affordable nor needed. Independent gives each "
                  "layer its own rank-r factorization; shared computes one column basis from the "
                  "two layers concatenated and projects both onto it. The shared basis IS the "
                  "alignment and it is inside the parameter budget, so there is no separate "
                  "alignment cost to omit.",
        "matched_budget": "Shared rank is the largest whose parameters fit the independent budget, "
                          "so sharing spends its saving on rank rather than on a smaller artifact.",
        "error_space": "FUNCTION space on the thick v2 capture, not weight cosine -- weight space "
                       "already inverted a ranking once in this campaign (G033).",
        "control": "A far-apart layer pair. If sharing helps adjacent layers it should help distant "
                   "ones measurably less; equal help would mean the effect is the factorization, "
                   "not layer affinity.",
        "pairs": rows_out,
        "adjacent_mean_error_reduction": adj_gain,
        "far_control_mean_error_reduction": far_gain,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
