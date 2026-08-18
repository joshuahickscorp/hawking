#!/usr/bin/env python3
"""G032: the G-XFORM members Hadamard's refutation did not cover.

Hadamard was killed for raising code entropy while buying 0.6-4% of a bit-step of
hold. The obligation names four more members, and two of them attack the actual
failure mode rather than the distribution's shape:

  SIGN          W diag(+-1), x -> diag(+-1) x. Predicted null: absmax is a
                magnitude statistic and sign flips do not move it. Included
                BECAUSE it is predicted null -- a member that shows a gain here
                would mean the harness is measuring something other than the
                transform.
  PERMUTATION   reorder the contraction axis so that weights sharing a group
                share a magnitude scale. Group quantization's whole cost is that
                one outlier sets the scale for 63 neighbours, so this is aimed
                directly at that.
  CHANNEL SCALE W diag(s), x -> diag(1/s) x, with s activation-aware from the
                thick capture. The AWQ/SmoothQuant idea: move dynamic range out
                of the weights and into the activations, where it is free.

All are EXACTLY function-preserving in real arithmetic, so any change in output
error is a change in what the codec had to represent, not in what the layer
computes.

Description cost is reported, not waved away. A channel scale is n floats. A
permutation is n*log2(n) bits unless it can be folded into the surrounding
layout, and whether it folds is a claim about the runtime that this file does not
get to make.

  ./tools/gravity_xform_members.py --out receipts/ascent-2026-08-16/G032_XFORM_MEMBERS.json
"""
from __future__ import annotations
import argparse, json, math, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group, code_entropy  # noqa: E402
from gravity_function_space_rank import load_activations, out_error, SITE  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def apply_and_score(w, x, bits, group, s=None, perm=None):
    """Transform, quantize, invert the transform on the ACTIVATION side, score.

    y = W x = (W T)(T^-1 x) exactly, so the codec sees W T while the layer still
    computes y. Nothing about the function changes; only what must be stored.
    """
    wt = w
    xt = x
    if s is not None:
        wt = wt * s[None, :]
        xt = xt / s[None, :]
    if perm is not None:
        wt = wt[:, perm]
        xt = xt[:, perm]
    hat, codes = quantize_group(wt, bits, group)
    rel, cos = out_error(xt, wt, hat)
    return rel, cos, code_entropy(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,31,63")
    ap.add_argument("--organs", default="gate_proj,down_proj")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    rows_out, agg = [], {}
    for layer in [int(x) for x in a.layers.split(",")]:
        for organ in a.organs.split(","):
            name = f"language_model.model.layers.{layer}.mlp.{organ}.weight"
            w = load_tensor(name).astype(np.float32)
            x, _ = load_activations(SITE[organ], layer, a.rows, w.shape[1])
            n = w.shape[1]
            members = {}

            members["baseline"] = apply_and_score(w, x, a.bits, a.group)

            rng = np.random.default_rng(0)
            sign = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
            members["sign"] = apply_and_score(w, x, a.bits, a.group, s=sign)

            # Order the contraction axis by column magnitude so a group's members
            # have similar scale. One outlier then sets the scale for 63
            # comparable neighbours instead of 63 small ones.
            colmag = np.abs(w).max(axis=0)
            perm = np.argsort(colmag).astype(np.int64)
            members["permutation_magnitude_sorted"] = apply_and_score(
                w, x, a.bits, a.group, perm=perm)

            # Activation-aware channel scale. Dynamic range the weights are
            # paying for moves to the activation side, where nothing quantizes it.
            act = np.abs(x).mean(axis=0) + 1e-8
            for alpha in (0.25, 0.5):
                s = (act ** alpha).astype(np.float32)
                s = (s / float(np.exp(np.log(s).mean()))).astype(np.float32)  # unit geometric mean
                members[f"channel_scale_a{alpha}"] = apply_and_score(
                    w, x, a.bits, a.group, s=s)

            # Both, since they attack different halves of the same problem.
            s = (act ** 0.5).astype(np.float32)
            s = (s / float(np.exp(np.log(s).mean()))).astype(np.float32)
            ws = w * s[None, :]
            colmag2 = np.abs(ws).max(axis=0)
            members["channel_scale_a0.5_then_permutation"] = apply_and_score(
                w, x, a.bits, a.group, s=s, perm=np.argsort(colmag2).astype(np.int64))

            row = {"tensor": name, "shape": list(w.shape),
                   "members": {k: {"out_rel_fro": v[0], "out_cosine": v[1],
                                   "code_entropy_bits": v[2]} for k, v in members.items()}}
            rows_out.append(row)
            for k, v in members.items():
                agg.setdefault(k, []).append(v)
            base = members["baseline"][0]
            print(f"  {layer:>2} {organ:<10} " + "  ".join(
                f"{k.split('_')[0][:6]}:{v[0]/base:.4f}" for k, v in members.items()))
            del w, x

    print(f"\n{'member':<38}{'out rel_fro':>13}{'vs baseline':>13}{'entropy':>10}{'d entropy':>11}")
    b_err = sum(v[0] for v in agg["baseline"]) / len(agg["baseline"])
    b_ent = sum(v[2] for v in agg["baseline"]) / len(agg["baseline"])
    table = []
    for k, v in agg.items():
        e = sum(t[0] for t in v) / len(v)
        h = sum(t[2] for t in v) / len(v)
        table.append({"member": k, "mean_out_rel_fro": e, "ratio_vs_baseline": e / b_err,
                      "mean_code_entropy_bits": h, "delta_entropy_bits": h - b_ent,
                      "wins": e < b_err})
        print(f"{k:<38}{e:>13.5f}{e/b_err:>13.4f}{h:>10.4f}{h-b_ent:>+11.4f}")

    best = min((t for t in table if t["member"] != "baseline"), key=lambda t: t["mean_out_rel_fro"])
    doc = {
        "schema": "hawking.nos.gxform_members.v1",
        "obligation": "G032 -- the G-XFORM members Hadamard's refutation did not cover",
        "fixed": {"bits": a.bits, "group": a.group, "quantizer": "symmetric absmax group"},
        "error_space": "FUNCTION space on the thick v2 capture. All transforms are exactly "
                       "function-preserving, so any change is a change in what the codec must "
                       "represent, not in what the layer computes.",
        "sign_is_a_control": "Sign flips cannot move an absmax magnitude statistic, so sign is "
                             "predicted null. A gain there would indict the harness, not the member.",
        "description_cost": {
            "sign": "n bits, and foldable into any adjacent diagonal at zero runtime cost.",
            "channel_scale": "n floats per tensor (5120 or 17408). Foldable into the preceding "
                             "RMSNorm ONLY where the consumers of that norm share one scale; "
                             "gate and up share post_input_norm, down_proj consumes post-SwiGLU "
                             "and has no norm to fold into. That is a runtime claim this file "
                             "does not get to make.",
            "permutation": "n*log2(n) bits explicitly, or free if it can be folded into the "
                           "surrounding layout. Also a runtime claim.",
        },
        "table": table,
        "best_non_baseline": best,
        "per_tensor": rows_out,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
