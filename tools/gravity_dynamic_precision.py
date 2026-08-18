#!/usr/bin/env python3
"""G058: is a per-token precision policy real, and what does its TAIL cost?

The verify is unusually specific and it is the whole design here: report MEAN and
TAIL active BPW and TOKEN_NS, not the mean alone, because a mechanism that is
cheap on average and catastrophic at p99 has not helped an agent workload.

The policy under test is the cheapest one the measured hardware actually supports:
one binary plane as the cheap prefix, a second plane added when a NAMED signal
says the state is uncertain. Both tiers are already measured on device in G072
(0.6059 and 0.6676 ps/element at 1.25 and 2.50 bits), so nothing about the cost
side is projected.

The signal is LOGIT MARGIN, chosen because it is the only candidate from the
acceptance list that is free at decode time: the previous token's logits are
already computed, so reading their top-two gap costs nothing and needs no extra
pass.

A policy is only as good as its signal, so the signal is tested before the policy:
does low margin actually predict that this token's answer is FRAGILE? Fragility is
measured by injecting perturbations into the final residual stream and checking
whether the argmax moves. If margin does not separate fragile tokens from robust
ones, the policy is dead regardless of how cheap its tiers are.

PERTURBATION SCALE, labelled because it is the weakest link: each tier's
perturbation is applied at the RELATIVE output error that tier was measured to
produce in G069. That overstates the effect if this layer contributes only part of
the stream, and understates it if error accumulates across layers. It is an
order-of-magnitude stand-in for a real assembled artifact, not a substitute.

  ./tools/gravity_dynamic_precision.py --out receipts/.../G058_DYNAMIC.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = ROOT / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
HIDDEN = 5120
# measured on device, G072
PS = {1: 0.6059, 2: 0.6676}
BITS = {1: 1.25, 2: 2.50}
# measured relative output error per tier, G069 function-fitted means
EPS = {1: 0.4274, 2: 0.2447}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--dirs", type=int, default=4)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    rng = np.random.default_rng(58)

    fn = np.fromfile(CAP / "final_norm/L00.f16", dtype=np.float16,
                     count=a.tokens * HIDDEN).reshape(a.tokens, HIDDEN).astype(np.float32)
    gF = load_tensor("language_model.model.norm.weight").astype(np.float32)
    head = load_tensor("language_model.lm_head.weight").astype(np.float32)
    z = fn @ head.T
    part = np.argpartition(z, -2, axis=1)[:, -2:]
    top2 = np.take_along_axis(z, part, 1)
    top2.sort(1)
    margin = (top2[:, 1] - top2[:, 0])
    base_top = z.argmax(1)
    print(f"{a.tokens} tokens, logit margin: min {margin.min():.3f} p10 "
          f"{np.percentile(margin,10):.3f} median {np.median(margin):.3f} "
          f"p90 {np.percentile(margin,90):.3f} max {margin.max():.3f}")

    # Fragility: does the argmax move when the stream is perturbed at each tier's
    # measured relative error? Random directions, so this asks about the tier's
    # MAGNITUDE rather than about one particular error pattern.
    frag = {}
    nrm = np.linalg.norm(fn, axis=1, keepdims=True)
    for tier, eps in EPS.items():
        flips = np.zeros(a.tokens)
        for _ in range(a.dirs):
            d = rng.standard_normal(fn.shape).astype(np.float32)
            d *= eps * nrm / np.linalg.norm(d, axis=1, keepdims=True)
            flips += ((fn + d) @ head.T).argmax(1) != base_top
        frag[tier] = flips / a.dirs
        print(f"tier k{tier} (eps {eps:.4f}): argmax flip rate mean {frag[tier].mean():.4f}, "
              f"tokens ever flipping {int((frag[tier]>0).sum())}/{a.tokens}")

    # Is the signal predictive? Split by margin quartile and compare fragility.
    q = np.percentile(margin, [25, 50, 75])
    bucket = np.searchsorted(q, margin)
    pred = []
    print(f"\n{'margin quartile':<18}{'n':>5}{'k1 flip rate':>14}{'k2 flip rate':>14}")
    for b in range(4):
        m = bucket == b
        pred.append({"quartile": b, "n": int(m.sum()),
                     "k1_flip": float(frag[1][m].mean()), "k2_flip": float(frag[2][m].mean())})
        print(f"{'Q'+str(b+1)+(' (lowest)' if b==0 else ''):<18}{int(m.sum()):>5}"
              f"{frag[1][m].mean():>14.4f}{frag[2][m].mean():>14.4f}")
    lo, hi = pred[0]["k1_flip"], pred[3]["k1_flip"]
    separates = lo > hi * 2 and lo > 0.01
    print(f"\nSIGNAL: lowest-margin quartile flips {lo:.4f} vs highest {hi:.4f} -> "
          f"{'PREDICTIVE' if separates else 'NOT PREDICTIVE -- the policy has nothing to act on'}")

    # The policy, and its tail. Promote the fraction of tokens below a margin
    # threshold to the expensive tier.
    rows = []
    for frac in (0.05, 0.10, 0.25, 0.50):
        tau = np.percentile(margin, frac * 100)
        up = margin <= tau
        bpw = np.where(up, BITS[2], BITS[1])
        ps = np.where(up, PS[2], PS[1])
        resid = np.where(up, frag[2], frag[1])
        rows.append({
            "promoted_fraction": frac, "margin_threshold": float(tau),
            "mean_active_bpw": float(bpw.mean()), "p99_active_bpw": float(np.percentile(bpw, 99)),
            "max_active_bpw": float(bpw.max()),
            "mean_ps_per_element": float(ps.mean()),
            "p99_ps_per_element": float(np.percentile(ps, 99)),
            "max_ps_per_element": float(ps.max()),
            "mean_flip_rate_under_policy": float(resid.mean()),
            "p99_flip_rate_under_policy": float(np.percentile(resid, 99)),
            "vs_uniform_k2_bpw": float(bpw.mean() / BITS[2]),
            "vs_uniform_k2_ps": float(ps.mean() / PS[2]),
            "vs_uniform_k2_flip": float(resid.mean() / frag[2].mean()) if frag[2].mean() else None})
        r = rows[-1]
        print(f"\npromote {frac*100:.0f}% (margin <= {tau:.3f})")
        print(f"  active BPW   mean {r['mean_active_bpw']:.4f}  p99 {r['p99_active_bpw']:.4f}  "
              f"max {r['max_active_bpw']:.4f}")
        print(f"  ps/element   mean {r['mean_ps_per_element']:.4f}  "
              f"p99 {r['p99_ps_per_element']:.4f}  max {r['max_ps_per_element']:.4f}")
        print(f"  flip rate    mean {r['mean_flip_rate_under_policy']:.4f}  "
              f"p99 {r['p99_flip_rate_under_policy']:.4f}")
        print(f"  vs uniform k2: {r['vs_uniform_k2_bpw']*100:.1f}% of the bits, "
              f"{r['vs_uniform_k2_ps']*100:.1f}% of the ALU, "
              f"{r['vs_uniform_k2_flip']:.2f}x the flip rate")

    doc = {
        "schema": "hawking.nos.dynamic_precision.v1",
        "obligation": "G058 -- active bits chosen per token, reported at mean AND tail",
        "policy": "one binary plane as the cheap prefix, a second added when the signal says the "
                  "state is uncertain; both tiers measured on device in G072",
        "signal": "LOGIT MARGIN of the previous token -- the only candidate from the acceptance "
                  "list that is FREE at decode time, since those logits are already computed",
        "tier_costs_measured": {"k1": {"bits": BITS[1], "ps_per_element": PS[1]},
                                "k2": {"bits": BITS[2], "ps_per_element": PS[2]}},
        "perturbation_scale_caveat": (
            "each tier's perturbation is applied at the RELATIVE output error that tier was "
            "measured to produce in G069. That OVERSTATES the effect if this layer contributes only "
            "part of the stream and UNDERSTATES it if error accumulates across layers. It is an "
            "order-of-magnitude stand-in for an assembled artifact, not a substitute for one."),
        "tokens": a.tokens, "random_directions_per_tier": a.dirs,
        "margin_distribution": {"min": float(margin.min()), "p10": float(np.percentile(margin, 10)),
                                "median": float(np.median(margin)),
                                "p90": float(np.percentile(margin, 90)), "max": float(margin.max())},
        "fragility_by_tier": {f"k{t}": {"mean_flip_rate": float(v.mean()),
                                        "tokens_ever_flipping": int((v > 0).sum())}
                              for t, v in frag.items()},
        "signal_predictiveness": {"by_margin_quartile": pred,
                                  "lowest_over_highest_flip_ratio":
                                      float(lo / hi) if hi else None,
                                  "predictive": bool(separates)},
        "policy_sweep": rows,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
