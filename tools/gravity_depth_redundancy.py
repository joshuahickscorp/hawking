#!/usr/bin/env python3
"""G064: is there any depth redundancy for a conditional-depth controller to exploit?

Mixture-of-recursions proposes a controller that decides continue | exit per token.
Before building a controller, two things have to be true and neither has been
measured on this patient:

  1. the residual stream must SATURATE before the last layer, or there is nothing
     to skip
  2. the saturation layer must VARY across tokens, or there is nothing for a
     controller to condition on -- a constant exit depth is a smaller model, not
     a mixture of recursions

Both are measurable on the thick v2 capture without building anything.

The residual stream direction is recovered exactly rather than approximated. The
capture stores post_input_norm[L] = (h_L / rms(h_L)) * g_L, so dividing out the
layer's own input_layernorm gain returns h_L up to a positive scale, which is all
a cosine needs. The same undoing on final_norm gives the stream's terminal
direction. Without this the gains of two different layers would be measured as
if they were stream rotation.

This is an UPPER BOUND on conditional-depth headroom, and it is stated as one:
the stream pointing where it will finally point does not prove the argmax cannot
still flip. A ceiling below the cost of the controller kills the mechanism; a
ceiling above it only earns an end-to-end test.

The null control is not optional here. This repository has already recorded raw
activation cosine with a null of 0.898 -- residual streams share an enormous
common component, so 0.95 can mean "identical" or "unrelated" depending on the
null, and a saturation curve read without one is unreadable.

  ./tools/gravity_depth_redundancy.py --rows 2048 --out receipts/.../G064_DEPTH.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = ROOT / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
HIDDEN = 5120
N_LAYERS = 64


def site(name, layer, rows):
    p = CAP / name / f"L{layer:02d}.f16"
    avail = p.stat().st_size // (2 * HIDDEN)
    take = min(rows, avail)
    a = np.fromfile(p, dtype=np.float16, count=take * HIDDEN).reshape(take, HIDDEN)
    return a.astype(np.float32), avail


def unit(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--taus", default="0.99,0.999")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]

    gF = load_tensor("language_model.model.norm.weight").astype(np.float32)
    fin, avail = site("final_norm", 0, a.rows)
    uF = unit(fin / gF)
    n = uF.shape[0]
    print(f"{n} tokens (of {avail} captured), hidden {HIDDEN}, {N_LAYERS} layers")

    # NULL: the same terminal directions against a derangement of themselves.
    # If unrelated tokens already sit at 0.9, no saturation number is readable
    # without this line beside it.
    perm = np.roll(np.arange(n), 1)
    null = float(np.einsum("ij,ij->i", uF, uF[perm]).mean())

    C = np.empty((N_LAYERS, n), dtype=np.float32)
    for L in range(N_LAYERS):
        g = load_tensor(f"language_model.model.layers.{L}.input_layernorm.weight").astype(np.float32)
        x, _ = site("post_input_norm", L, a.rows)
        C[L] = np.einsum("ij,ij->i", unit(x[:n] / g), uF)
        del x

    print(f"\nNULL (terminal direction vs a DIFFERENT token's): {null:.6f}")
    print(f"{'layer':>6}{'mean cos to final':>20}{'p10':>10}{'p90':>10}")
    for L in list(range(0, N_LAYERS, 8)) + [N_LAYERS - 1]:
        print(f"{L:>6}{C[L].mean():>20.6f}{np.percentile(C[L],10):>10.6f}"
              f"{np.percentile(C[L],90):>10.6f}")

    rows = []
    for tau in taus:
        # First layer from which the stream stays within tau of its terminal
        # direction for the REST of the stack. "First time it touches tau" would
        # count a transient crossing as convergence.
        ok = C >= tau
        suffix = np.ones(n, dtype=bool)
        star = np.full(n, N_LAYERS, dtype=np.int32)
        for L in range(N_LAYERS - 1, -1, -1):
            suffix &= ok[L]
            star[suffix] = L
        saved = N_LAYERS - star
        rows.append({
            "tau": tau,
            "exit_layer_mean": float(star.mean()), "exit_layer_std": float(star.std()),
            "exit_layer_min": int(star.min()), "exit_layer_max": int(star.max()),
            "exit_layer_p50": float(np.percentile(star, 50)),
            "exit_layer_p90": float(np.percentile(star, 90)),
            "exit_layer_p99": float(np.percentile(star, 99)),
            "layers_saved_mean": float(saved.mean()),
            "layers_saved_frac_mean": float(saved.mean()) / N_LAYERS,
            "tokens_that_never_saturate": int((star == N_LAYERS).sum()),
            "distinct_exit_layers": int(len(np.unique(star))),
        })
        r = rows[-1]
        print(f"\ntau {tau}: exit layer mean {r['exit_layer_mean']:.2f} "
              f"std {r['exit_layer_std']:.2f} range [{r['exit_layer_min']},{r['exit_layer_max']}] "
              f"p99 {r['exit_layer_p99']:.0f}")
        print(f"  layers saved {r['layers_saved_mean']:.2f}/{N_LAYERS} "
              f"= {r['layers_saved_frac_mean']*100:.1f}%   "
              f"never saturate: {r['tokens_that_never_saturate']}/{n}   "
              f"distinct exit depths: {r['distinct_exit_layers']}")

    doc = {
        "schema": "hawking.nos.depth_redundancy.v1",
        "obligation": "G064 -- conditional depth: is there redundancy, and does it vary per token",
        "method": ("post_input_norm[L] = (h_L/rms(h_L)) * g_L, so dividing out the layer's own "
                   "input_layernorm gain recovers the residual stream direction exactly. Same "
                   "undoing on final_norm gives the terminal direction. cos(u_L, u_final) per "
                   "token is the saturation curve; the exit layer is the first L from which the "
                   "stream stays within tau for the REST of the stack, not the first touch."),
        "capture": {"root": str(CAP.relative_to(ROOT)), "tokens_used": n, "tokens_available": avail,
                    "note": "real BF16 parent activations, adequacy-gated, NOT synthetic"},
        "null_control": {
            "terminal_vs_other_token_cosine": null,
            "why": ("residual streams share a large common component. This repository already "
                    "recorded raw activation cosine with a null of 0.898, so a saturation number "
                    "without its null is unreadable."),
        },
        "upper_bound_disclaimer": (
            "This is a CEILING on conditional-depth headroom, not a result. The stream already "
            "pointing where it will finally point does not prove the argmax cannot still flip in "
            "the remaining layers. A ceiling below the controller's own cost kills the mechanism; "
            "a ceiling above it earns an end-to-end test and nothing more."),
        "per_tau": rows,
        "layer_curve_mean": [float(C[L].mean()) for L in range(N_LAYERS)],
        "layer_curve_p10": [float(np.percentile(C[L], 10)) for L in range(N_LAYERS)],
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
