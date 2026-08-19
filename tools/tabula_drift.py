#!/usr/bin/env python3
"""G123: recover the abliterated direction from the artifact and measure drift.

The abliteration manifest states the method exactly: refusal-direction ORTHOGONAL
WEIGHT PROJECTION, applied to full_attention_out, linear_attention_out and mlp_down
across layers 24-63, norm-preserving, scale 1.0. Those tensors write into the
residual stream, so the removed direction v lives in their OUTPUT space and

    W' = (I - v v^T) W    =>    v^T W' = 0

which makes v a LEFT NULL VECTOR of every abliterated tensor. That is what makes it
recoverable from the artifact alone with no access to the base model: take the
smallest left singular direction of W'.

Two controls, because a direction recovered from a null space is exactly the kind of
thing that looks convincing and means nothing:

  AGREEMENT  v recovered independently from widely separated layers must agree. If
             each layer yields its own arbitrary near-null direction, there is no
             shared removed direction and the recovery is noise.
  NULL       a random unit direction scored the same way. Residual-stream vectors
             share a large common component, and this repository has already
             recorded raw activation cosine with a null of 0.898, so a similarity
             without its null is unreadable.

DRIFT is then how much a quantized candidate puts back:

    drift(W_hat) = ||v^T W_hat|| / ||v^T W'||

with the parent at 1.0 by construction.

  ./tools/tabula_drift.py --layers 24,43,63 --out receipts/.../G123_TABULA.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BF16 = ROOT / "workspace/campaign/records/runs/qwen38-27b/bf16"
# The manifest's three target kinds, by their tensor suffixes.
KIND = {"full_attention_out": "self_attn.o_proj.weight",
        "linear_attention_out": "linear_attn.out_proj.weight",
        "mlp_down": "mlp.down_proj.weight"}


def recover_direction(w):
    """Smallest left singular direction of W. W is [out=5120, in]; the projection
    removed v from the OUTPUT space, so the Gram is over rows."""
    g = (w @ w.T).astype(np.float64)
    ev, U = np.linalg.eigh(g)
    return U[:, 0].astype(np.float32), float(ev[0]), float(ev[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="24,43,63")
    ap.add_argument("--kind", default="full_attention_out", choices=sorted(KIND))
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    man = json.loads((BF16 / "abliteration-manifest.json").read_text())
    dest = set(man["projection"]["destination_layers"])
    print(f"manifest: {man['method']}, layers {min(dest)}-{max(dest)}, "
          f"kinds {man['projection']['target_kinds']}, norm_preserve "
          f"{man['projection']['norm_preserve']}")
    # full_attention_out only exists every fullAttentionInterval layers; picking a
    # linear-attention layer for it fails with a bare KeyError that reads like a
    # missing tensor rather than a wrong question.
    iv = man["architecture"]["fullAttentionInterval"]
    for L in layers:
        if L not in dest:
            raise SystemExit(f"layer {L} is NOT abliterated -- pick from {sorted(dest)[:6]}...")
        if a.kind == "full_attention_out" and (L + 1) % iv != 0:
            raise SystemExit(f"layer {L} has no full attention (interval {iv}); valid abliterated "
                             f"full-attn layers: {[x for x in sorted(dest) if (x+1)%iv==0][:8]}")
        if a.kind == "linear_attention_out" and (L + 1) % iv == 0:
            raise SystemExit(f"layer {L} IS a full-attention layer; it has no linear_attn")

    dirs, meta = {}, []
    for L in layers:
        w = load_tensor(f"language_model.model.layers.{L}.{KIND[a.kind]}").astype(np.float32)
        v, lo, hi = recover_direction(w)
        resid = float(np.linalg.norm(v @ w))
        dirs[L] = v
        meta.append({"layer": L, "shape": list(w.shape), "smallest_eig": lo, "largest_eig": hi,
                     "eig_ratio": lo / hi, "residual_v_dot_W": resid,
                     "frobenius": float(np.linalg.norm(w))})
        print(f"  L{L}: smallest/largest eigenvalue {lo/hi:.3e}   ||v^T W'|| {resid:.6f}")
        del w

    rng = np.random.default_rng(123)
    nullv = rng.standard_normal(5120).astype(np.float32); nullv /= np.linalg.norm(nullv)
    print(f"\nAGREEMENT of the recovered direction across layers (null control beside it):")
    agree = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            c = abs(float(dirs[layers[i]] @ dirs[layers[j]]))
            n = abs(float(dirs[layers[i]] @ nullv))
            agree.append({"a": layers[i], "b": layers[j], "abs_cos": c, "null_abs_cos": n})
            print(f"  L{layers[i]} vs L{layers[j]}:  |cos| {c:.6f}      random null |cos| {n:.6f}")

    # DRIFT ladder on the deepest layer.
    L = layers[-1]
    w = load_tensor(f"language_model.model.layers.{L}.{KIND[a.kind]}").astype(np.float32)
    v = dirs[L]
    base = float(np.linalg.norm(v @ w))
    ladder = []
    for bits in (4, 3, 2):
        wq, _ = quantize_group(w, bits, 64)
        d = float(np.linalg.norm(v @ wq)) / base
        ladder.append({"codec": f"q{bits}_group64", "drift_x": d})
        print(f"  q{bits}: drift {d:.2f}x")
        del wq
    print(f"\nDRIFT LADDER on L{L} {a.kind} (parent = 1.00x by construction)")
    RECORDED = {"q4_group64": (11, 12), "q3_group64": (25, 27), "q2_group64": (64, 67)}
    for r in ladder:
        lo, hi = RECORDED[r["codec"]]
        r["recorded_range"] = [lo, hi]
        r["reproduces_recorded"] = lo <= r["drift_x"] <= hi
        print(f"  {r['codec']:<14} measured {r['drift_x']:7.2f}x   recorded {lo}-{hi}x   "
              f"{'REPRODUCES' if r['reproduces_recorded'] else 'DOES NOT REPRODUCE'}")

    doc = {"schema": "hawking.nos.tabula_drift.v1",
           "obligation": "G123 -- Tabula drift instrument, reproducible for any candidate",
           "manifest": {k: man["projection"][k] for k in ("target_kinds", "scale", "norm_preserve",
                                                          "modified_tensor_count")},
           "method": ("W' = (I - v v^T) W means v^T W' = 0, so the removed direction is a LEFT NULL "
                      "VECTOR of every abliterated tensor and is recoverable from the artifact "
                      "alone, with no access to the base model."),
           "per_layer": meta, "agreement": agree,
           "null_control": {"kind": "random unit direction", "seed": 123},
           "drift_ladder": ladder, "drift_layer": L, "drift_kind": a.kind,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=ROOT).stdout.strip()}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
