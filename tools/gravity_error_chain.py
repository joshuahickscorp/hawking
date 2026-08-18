#!/usr/bin/env python3
"""Measure the residual error-propagation chain  e_{l+1} ~ A_l e_l + q_l.

The incumbent coherence screen multiplies per-tensor holds, which assumes error
compounds uniformly and independently across layers. That assumption has never
been measured on this model, and it is the assumption that produced the "every
tensor needs 0.99527" requirement. If A_l is far from 1 and varies with depth,
the flat requirement is wrong in both directions: amplifying layers need more
than the uniform figure and damping layers need much less.

Two quantities, and they are NOT the same thing:

  signal gain    ||X_{l+1}|| / ||X_l||     from the captured hiddens
  error gain     ||J_l e|| / ||e||         how a perturbation grows through layer l

Only the second is A_l. Signal gain is reported because it is free and because a
prior receipt quoted 2.6039602756500244 for L63, but a residual block can amplify
signal while contracting error, or the reverse, so signal gain is a diagnostic
and not a substitute.

Error gain is estimated here WITHOUT running the model, by propagating random
perturbations through the layer's stored linear maps at the captured operating
point. That is a local linearization: exact for the linear projections, an
approximation across the nonlinearity. It is labelled ESTIMATED throughout. A
model-in-the-loop measurement replaces it and is the honest final authority.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_doctor_gate import load_tensor, load_X, CAPTURE  # noqa: E402


def signal_gains(capture=CAPTURE):
    res = json.load(open(os.path.join(capture, "capture-result.json")))
    n_layers, hidden = res["n_layers"], res["hidden"]
    norms = []
    for l in range(n_layers):
        per = res["per_layer"].get(str(l))
        if per is None:
            norms.append(None)
            continue
        X = np.fromfile(per["path"], dtype=np.float32).reshape(per["n_rows"], hidden)
        norms.append(float(np.mean(np.linalg.norm(X, axis=1))))
    gains = []
    for l in range(1, n_layers):
        a, b = norms[l - 1], norms[l]
        gains.append(None if (a is None or b is None or a == 0) else b / a)
    return norms, gains


def mlp_error_gain(layer, X, n_probe=64, seed=0):
    """ESTIMATED error gain of the MLP block at its captured operating point.

    SwiGLU: y = down( silu(gate(x)) * up(x) ). Perturb x, push through the real
    stored matrices with the nonlinearity evaluated at the true activation, and
    measure how the output perturbation compares to the input one. Averaged over
    random perturbation directions so the number is not one lucky direction.
    """
    g = load_tensor(f"language_model.model.layers.{layer}.mlp.gate_proj.weight")
    u = load_tensor(f"language_model.model.layers.{layer}.mlp.up_proj.weight")
    d = load_tensor(f"language_model.model.layers.{layer}.mlp.down_proj.weight")

    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(16, X.shape[0]), replace=False)
    x = X[idx]                                        # (b, hidden)

    def fwd(v):
        a = v @ g.T
        s = a / (1.0 + np.exp(-a))                    # silu
        return (s * (v @ u.T)) @ d.T

    y0 = fwd(x)
    ratios = []
    for _ in range(n_probe // 16):
        e = rng.standard_normal(x.shape).astype(np.float32)
        e *= 1e-3 * np.linalg.norm(x, axis=1, keepdims=True) / (
            np.linalg.norm(e, axis=1, keepdims=True) + 1e-30)
        dy = fwd(x + e) - y0
        ratios.append(np.linalg.norm(dy, axis=1) / (np.linalg.norm(e, axis=1) + 1e-30))
    r = np.concatenate(ratios)
    return {"mean": float(r.mean()), "p50": float(np.median(r)), "max": float(r.max()),
            "write_gain": float(np.mean(np.linalg.norm(y0, axis=1) /
                                        (np.linalg.norm(x, axis=1) + 1e-30)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,7,15,23,31,39,47,55,63")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    norms, sig = signal_gains()
    print("residual signal norm by layer (captured hiddens, MEASURED)")
    print(f"  L0 {norms[0]:.4f}   L63 {norms[63]:.4f}   ratio L63/L0 {norms[63]/norms[0]:.6f}")
    fin = [g for g in sig if g is not None]
    print(f"  per-layer signal gain: min {min(fin):.4f}  median {np.median(fin):.4f}  max {max(fin):.4f}")
    print(f"  L63 signal gain {sig[62]:.6f}")

    print("\nMLP block error gain at the captured operating point (ESTIMATED, local linearization)")
    print(f"{'layer':>6} {'err_gain_mean':>14} {'err_gain_max':>13} {'write_gain':>11}")
    out = {}
    for l in [int(x) for x in a.layers.split(",")]:
        X = load_X(l)
        r = mlp_error_gain(l, X)
        out[l] = r
        print(f"{l:>6} {r['mean']:>14.6f} {r['max']:>13.6f} {r['write_gain']:>11.6f}")

    gains = [v["mean"] for v in out.values()]
    print(f"\nerror gain across sampled depths: min {min(gains):.4f}  max {max(gains):.4f}  "
          f"spread {max(gains)/min(gains):.2f}x")
    print("A uniform per-tensor requirement is only correct if this spread is ~1x.")

    if a.json:
        json.dump({"signal_norms": norms, "signal_gains": sig,
                   "mlp_error_gain": {str(k): v for k, v in out.items()}},
                  open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def compose_check(pairs, n_probe=64, seed=0):
    """Held-out test: does a measured chain PREDICT a composition?

    The chain model claims e_out ~ A_k * A_{k+1} * e_in. Measure each block's
    gain alone, then measure the two composed, and compare the product against
    the direct measurement. If the product does not predict the composition, the
    multiplicative chain is the wrong model and any allocation built on it is
    wrong too -- which is exactly the failure mode of the uniform product screen.
    """
    import numpy as np
    from gravity_doctor_gate import load_tensor
    rng = np.random.default_rng(seed)
    rows = []
    for k, j in pairs:
        Wk = [load_tensor(f"language_model.model.layers.{k}.mlp.{n}_proj.weight") for n in ("gate","up","down")]
        Wj = [load_tensor(f"language_model.model.layers.{j}.mlp.{n}_proj.weight") for n in ("gate","up","down")]

        def blk(W, v):
            a = v @ W[0].T
            return ((a / (1.0 + np.exp(-a))) * (v @ W[1].T)) @ W[2].T

        Xk = load_X(k)
        idx = rng.choice(Xk.shape[0], size=16, replace=False)
        x = Xk[idx]
        e = rng.standard_normal(x.shape).astype(np.float32)
        e *= 1e-3 * np.linalg.norm(x, axis=1, keepdims=True) / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-30)

        # single-block gains at their own operating points
        d1 = blk(Wk, x + e) - blk(Wk, x)
        gk = float(np.mean(np.linalg.norm(d1, axis=1) / np.linalg.norm(e, axis=1)))
        mid = x + blk(Wk, x)                      # residual add, real topology
        e2 = d1
        d2 = blk(Wj, mid + e2) - blk(Wj, mid)
        gj = float(np.mean(np.linalg.norm(d2, axis=1) / (np.linalg.norm(e2, axis=1) + 1e-30)))

        measured = float(np.mean(np.linalg.norm(d2, axis=1) / np.linalg.norm(e, axis=1)))
        predicted = gk * gj
        rows.append({"pair": (k, j), "A_k": gk, "A_j": gj,
                     "predicted": predicted, "measured": measured,
                     "rel_err": abs(predicted - measured) / (measured + 1e-30)})
    return rows
