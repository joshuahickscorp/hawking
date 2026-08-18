#!/usr/bin/env python3
"""G075: which physical information actually causes capability, by more than one signal.

The verify is the whole design: a single signal is a hypothesis, and only agreement
between INDEPENDENT signals is evidence. So the two cheap signals here are chosen
to share no inputs at all, and a third, genuinely causal one is computed to judge
both.

Everything is measured on the final residual stream, because that is the one place
in this model where the path to a decision is short enough to compute exactly:
final_norm = (h/rms(h)) * g_F, and z = lm_head @ final_norm. No approximation, no
backward pass, no surrogate loss.

  S1  DATA SIDE     E_tokens[final_norm_i^2]
      how much this channel actually carries, from real captured activations.
      Knows nothing about lm_head.

  S2  WEIGHT SIDE   ||lm_head[:, i]||_2
      how far this channel can reach into the logits. Knows nothing about the data.

  S3  INTERVENTION  fraction of tokens whose ARGMAX changes when channel i is
      zeroed. z' = z - u_i*g_F[i]*lm_head[:, i], exact rather than linearized, and
      nonlinear because argmax is. This is the ground truth the other two are
      trying to predict.

S1 and S2 are computed from disjoint tensors, so their agreement cannot be an
artefact of a shared input. S3 is expensive per channel, so it is measured on a
STRATIFIED sample spanning the S1 x S2 range rather than on the head of either
ranking -- sampling by one signal's top-N would hand that signal the correlation.

  ./tools/gravity_causal_map.py --out receipts/.../G075_CAUSAL_MAP.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = ROOT / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
HIDDEN = 5120


def spearman(a, b):
    def rk(v):
        v = np.asarray(v, float)
        o = np.empty(len(v)); o[np.argsort(v)] = np.arange(len(v))
        return o
    ra, rb = rk(a), rk(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    rng = np.random.default_rng(75)

    fn = np.fromfile(CAP / "final_norm/L00.f16", dtype=np.float16,
                     count=a.tokens * HIDDEN).reshape(a.tokens, HIDDEN).astype(np.float32)
    gF = load_tensor("language_model.model.norm.weight").astype(np.float32)
    head = load_tensor("language_model.lm_head.weight").astype(np.float32)
    print(f"final_norm {fn.shape}, lm_head {head.shape}")

    S1 = (fn.astype(np.float64) ** 2).mean(0)
    S2 = np.linalg.norm(head.astype(np.float64), axis=0)
    print(f"S1 data-side   range {S1.min():.3e} .. {S1.max():.3e}")
    print(f"S2 weight-side range {S2.min():.4f} .. {S2.max():.4f}")

    z = fn @ head.T
    top = z.argmax(1)
    print(f"logits {z.shape}, {len(set(top.tolist()))} distinct argmax tokens over {a.tokens}")

    # Stratified sample over the S1 x S2 plane: 4x4 quantile cells, equal draw from
    # each. Sampling by one signal's top-N would hand that signal the correlation.
    q1 = np.clip(np.searchsorted(np.quantile(S1, [.25, .5, .75]), S1), 0, 3)
    q2 = np.clip(np.searchsorted(np.quantile(S2, [.25, .5, .75]), S2), 0, 3)
    per = max(1, a.channels // 16)
    sel = []
    for i in range(4):
        for j in range(4):
            pool = np.flatnonzero((q1 == i) & (q2 == j))
            if pool.size:
                sel.append(rng.choice(pool, size=min(per, pool.size), replace=False))
    sel = np.unique(np.concatenate(sel))
    print(f"stratified sample: {sel.size} channels over 16 quantile cells")

    u = fn / gF          # normalized stream direction, exactly what the norm produced
    flips = np.empty(sel.size)
    for n, i in enumerate(sel):
        c = (u[:, i] * gF[i])[:, None]        # this channel's exact logit contribution
        flips[n] = float((np.argmax(z - c * head[:, i][None, :], 1) != top).mean())
    S3 = flips

    r12 = spearman(S1, S2)
    r13 = spearman(S1[sel], S3)
    r23 = spearman(S2[sel], S3)
    prod = spearman((S1[sel] * S2[sel] ** 2), S3)
    print(f"\nSpearman")
    print(f"  S1 data   vs S2 weight    {r12:+.4f}   (all {HIDDEN} channels)")
    print(f"  S1 data   vs S3 ablation  {r13:+.4f}   (sampled {sel.size})")
    print(f"  S2 weight vs S3 ablation  {r23:+.4f}")
    print(f"  S1*S2^2   vs S3 ablation  {prod:+.4f}   (the product the algebra predicts)")
    print(f"\nablation flip rate: mean {S3.mean():.4f}, max {S3.max():.4f}, "
          f"{int((S3 == 0).sum())}/{sel.size} channels flip NOTHING")

    agree = (r13 > 0.3) and (r23 > 0.3)
    doc = {
        "schema": "hawking.nos.causal_information_map.v1",
        "obligation": "G075 -- which bits cause protected capability, by independent signals",
        "why_the_final_stream": ("the one place in this model where the path to a decision is short "
                                 "enough to compute EXACTLY: final_norm = (h/rms(h))*g_F and "
                                 "z = lm_head @ final_norm. No backward pass, no surrogate loss."),
        "signals": {
            "S1_data_side": "E_tokens[final_norm_i^2] -- knows nothing about lm_head",
            "S2_weight_side": "||lm_head[:,i]||_2 -- knows nothing about the data",
            "S3_intervention": "fraction of tokens whose ARGMAX changes when channel i is zeroed; "
                               "exact rather than linearized, and nonlinear because argmax is",
        },
        "independence": "S1 and S2 are computed from DISJOINT tensors, so their agreement cannot be "
                        "an artefact of a shared input",
        "sampling": f"S3 measured on a stratified sample of {int(sel.size)} channels over 16 "
                    "quantile cells of the S1 x S2 plane. Sampling by one signal's top-N would hand "
                    "that signal the correlation.",
        "tokens": a.tokens, "channels_total": HIDDEN, "channels_sampled": int(sel.size),
        "spearman": {"S1_vs_S2": r12, "S1_vs_S3": r13, "S2_vs_S3": r23,
                     "S1_times_S2_squared_vs_S3": prod},
        "ablation": {"mean_flip_rate": float(S3.mean()), "max_flip_rate": float(S3.max()),
                     "channels_flipping_nothing": int((S3 == 0).sum())},
        "signals_agree": bool(agree),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
