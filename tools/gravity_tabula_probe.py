#!/usr/bin/env python3
"""Tabula: does compression quietly restore the suppression abliteration removed?

The patient is ABLITERATED. abliteration-manifest.json: refusal-direction orthogonal weight
projection, direction taken from residual_post at layer 53, projected out of 80 tensors on
layers 24-63 -- 10 full_attention_out, 30 linear_attention_out, 40 mlp_down -- norm_preserve
true. Those are exactly the tensors that WRITE to the residual stream.

A behavioural refusal count is the obvious test and it is weak: it needs prompts that provoke
refusal, a judge for what counts as one, and it cannot distinguish "compression restored the
refusal direction" from "sampling wandered". There is a sharper measurement available.

If W_abl = (I - d d^T) W, then d^T W_abl = 0 EXACTLY. The refusal direction lies in the LEFT
NULL SPACE of every abliterated tensor. For a 5120 x 17408 matrix that null space is
generically empty, so the smallest left singular vector IS the removed direction -- and the
same direction must appear in all 80 tensors independently. That agreement is the check that
the recovery is real rather than an arbitrary small singular direction.

Then the Tabula question becomes a number: how much energy along d does a quantized artifact
put BACK? Quantization error is not orthogonal to d, so a lossy codec necessarily reintroduces
some. The obligation is that it stays negligible relative to the direction the abliteration
removed, and that it does not grow as density falls.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_doctor_gate import load_tensor, c_uniform  # noqa: E402

MANIFEST = "workspace/campaign/records/runs/qwen38-27b/bf16/abliteration-manifest.json"


def smallest_left_dir(W):
    """Left singular vector of least singular value, via the 5120x5120 Gram."""
    G = (W @ W.T).astype(np.float64)
    w, V = np.linalg.eigh(G)
    return V[:, 0].astype(np.float32), float(w[0]), float(w[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="30,40,50,60")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    man = json.load(open(MANIFEST))
    dest = man["projection"]["destination_layers"]
    print(f"abliteration: {man['method']}, source_layer {man['direction']['source_layer']}, "
          f"{man['projection']['modified_tensor_count']} tensors on layers "
          f"{dest[0]}-{dest[-1]}, norm_preserve {man['projection']['norm_preserve']}")
    print(f"target kinds: {man['projection']['target_kinds']}\n")

    layers = [int(x) for x in a.layers.split(",")]
    dirs, info = {}, []
    print(f"{'tensor':<34}{'lambda_min':>13}{'lambda_max':>13}{'ratio':>12}")
    for l in layers:
        name = f"language_model.model.layers.{l}.mlp.down_proj.weight"
        W = load_tensor(name).astype(np.float32)
        d, lo, hi = smallest_left_dir(W)
        dirs[l] = d
        info.append({"layer": l, "lambda_min": lo, "lambda_max": hi})
        print(f"{'mlp.down_proj@L'+str(l):<34}{lo:>13.4e}{hi:>13.4e}{lo/hi:>12.3e}")

    print("\nagreement between the recovered directions (|cos|). A shared direction across")
    print("independent matrices is the evidence that this is the ABLITERATED direction and")
    print("not just each matrix's own smallest singular vector:")
    ks = sorted(dirs)
    print("      " + "".join(f"{l:>10}" for l in ks))
    agree = []
    for i in ks:
        row = "".join(f"{abs(float(dirs[i] @ dirs[j])):>10.5f}" for j in ks)
        print(f"  L{i:<4}{row}")
        for j in ks:
            if i < j:
                agree.append(abs(float(dirs[i] @ dirs[j])))
    rng = np.random.default_rng(0)
    r1, r2 = rng.standard_normal(5120), rng.standard_normal(5120)
    null = abs(float(r1 @ r2 / (np.linalg.norm(r1) * np.linalg.norm(r2))))
    print(f"\n  random-pair null for 5120 dims: {null:.5f}")
    print(f"  recovered-pair agreement: {min(agree):.5f} - {max(agree):.5f}")
    recovered = max(agree) > 0.5
    if not recovered:
        print("\n  VERDICT: the smallest left directions DO NOT agree across layers, so this")
        print("  recovery does not identify a shared abliterated direction. The projection is")
        print("  per-tensor, or norm_preserve re-mixed it, or it is not rank-1 in this basis.")
        print("  Report this as a FAILED recovery. Do not proceed to a drift number that")
        print("  would be measured along an arbitrary direction.")
    else:
        print("\n  VERDICT: shared direction recovered. Measuring reintroduced energy.")
        print(f"\n{'tensor':<26}{'codec':<14}{'||dW||/||W||':>14}{'vs bf16':>12}")
        for l in ks:
            name = f"language_model.model.layers.{l}.mlp.down_proj.weight"
            W = load_tensor(name).astype(np.float32)
            d = dirs[l]
            base = float(np.linalg.norm(d @ W) / np.linalg.norm(W))
            for tag, bits in (("bf16 parent", None), ("q4 g64", 4), ("q3 g64", 3), ("q2 g64", 2)):
                Wx = W if bits is None else c_uniform(W, bits, 64)
                e = float(np.linalg.norm(d @ Wx) / np.linalg.norm(Wx))
                print(f"{('mlp.down@L'+str(l)) if bits is None else '':<26}{tag:<14}"
                      f"{e:>14.6e}{('-' if bits is None else f'{e/max(base,1e-30):.1f}x'):>12}")
    if a.json:
        json.dump({"info": info, "agreement": agree, "recovered": bool(recovered)},
                  open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
