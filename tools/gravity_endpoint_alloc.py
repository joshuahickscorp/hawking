#!/usr/bin/env python3
"""The endpoint tables set the sub-1.0 floor. Can they be allocated per ROW?

embed_tokens and lm_head are 9.454% of N, one site each, and are NOT tied in this checkpoint
(distinct SHA-256, distinct payloads, tie_word_embeddings false). With both held at G0's Q4
the zero-body floor is 0.401802783208 complete BPW, so 40% of the entire sub-1.0 budget is
spent before a single body weight is stored. Uniform Q3 on the tables already fails Doctor on
EOS, so the uniform ladder is finished at Q4.

But the table is not one object. It is 248,320 independent output units, and a decode reads
all of them while Doctor only protects a few classes strongly. So the question the uniform
sweep cannot ask: can bits be allocated PER VOCABULARY ROW?

Rows are ranked by measured logit influence on the real final_norm activations, not by weight
magnitude -- magnitude selection is already dead here by 10-143x. Then a fraction f of rows
is kept rich and the rest is pushed down, and the whole table is scored on the four-axis gate
INCLUDING gain, plus separately on the row classes Doctor protects.

Scored on the thick capture's held-out half. The fit half is used only to rank rows, so the
ranking cannot be a memorisation of the evaluation set.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_doctor_gate import load_tensor, axes, c_uniform  # noqa: E402

SOURCE_PARAM_COUNT = 26_895_998_464
CAP = "workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"


def final_norm_split():
    res = json.load(open(f"{CAP}/capture-result.json"))
    st = res["sites"]["final_norm"]
    p = st["per_layer"]["0"]
    X = np.fromfile(p["path"], dtype=np.float16).reshape(-1, st["width"]).astype(np.float32)
    return X[:st["n_fit"]], X[st["n_fit"]:st["n_fit"] + st["n_hold"]]


def bits_per_elem(frac_rich, rich_bits, poor_bits, group=128, scale_bytes=2):
    """Complete bits per element for a two-tier per-row allocation.

    The per-row tier assignment is itself information: one bit per row, which over a
    248320 x 5120 table is 1/5120 bits per element. Small, but counted -- an allocation
    whose own index is not counted is the oldest way to fake a density number.
    """
    meta = 8 * scale_bytes / group
    return frac_rich * (rich_bits + meta) + (1 - frac_rich) * (poor_bits + meta) + 1.0 / 5120


def two_tier(W, rank, frac_rich, rich_bits, poor_bits, group=128):
    n_rich = int(frac_rich * W.shape[0])
    rich = rank[:n_rich]
    out = c_uniform(W, poor_bits, group)
    out[rich] = c_uniform(W[rich], rich_bits, group)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="language_model.lm_head.weight")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    W = load_tensor(a.tensor).astype(np.float32)
    A, B = final_norm_split()
    V, H = W.shape
    print(f"{a.tensor}: {V} x {H} = {V*H:,} elements = {100*V*H/SOURCE_PARAM_COUNT:.6f}% of N")
    print(f"final_norm capture: fit {A.shape[0]} rows, HELD {B.shape[0]} rows, prompt-level split\n")

    # rank rows by measured logit influence on the FIT half only
    infl = np.abs(A @ W.T).mean(0)
    rank = np.argsort(-infl)
    # the classes Doctor protects, identified by measured behaviour rather than by id lists:
    # the rows that actually win, and the rows that almost never do
    top1 = np.bincount((A @ W.T).argmax(1), minlength=V)
    winners = np.flatnonzero(top1 > 0)
    tail = rank[int(0.90 * V):]
    print(f"rows that ever win top-1 on the fit half: {len(winners)} of {V}")

    def score(Wh, label, bpe):
        ax = axes(W, Wh, B, seed=None)
        Y, Yh = B @ W.T, B @ Wh.T
        # protected-class agreement, measured per row-set across held-out tokens
        def cls(idx):
            num = (Y[:, idx] * Yh[:, idx]).sum(0)
            den = np.linalg.norm(Y[:, idx], axis=0) * np.linalg.norm(Yh[:, idx], axis=0) + 1e-30
            return float((num / den).min())
        # does the argmax decision survive?
        keep = float((Y.argmax(1) == Yh.argmax(1)).mean())
        print(f"  {label:<34}{bpe:>8.4f}{ax['observed']:>10.6f}{ax['probed']:>10.6f}"
              f"{ax['worst_unit']:>10.6f}{ax['gain']:>10.6f}{cls(winners):>10.6f}"
              f"{cls(tail):>9.6f}{keep:>9.4f}")
        return {"label": label, "bits_per_elem": bpe, **ax,
                "winners_worst": cls(winners), "tail_worst": cls(tail), "argmax_keep": keep}

    print(f"  {'scheme':<34}{'b/elem':>8}{'observed':>10}{'probed':>10}{'worst_u':>10}"
          f"{'gain':>10}{'winners':>10}{'tail':>9}{'argmax':>9}")
    out = []
    out.append(score(c_uniform(W, 4, 128), "uniform q4 g128 (reference)", 4 + 8 * 2 / 128))
    for b in (3, 2):
        out.append(score(c_uniform(W, b, 128), f"uniform q{b} g128", b + 8 * 2 / 128))
    for frac in (0.02, 0.05, 0.10, 0.25):
        for rb, pb in ((4, 2), (4, 3), (6, 2)):
            bpe = bits_per_elem(frac, rb, pb)
            out.append(score(two_tier(W, rank, frac, rb, pb),
                             f"top {100*frac:g}% q{rb}, rest q{pb}", bpe))

    ref = out[0]
    print("\n  a scheme is adequate only if EVERY axis is within margin of the q4 reference")
    print(f"  {'scheme':<34}{'b/elem':>8}  verdict")
    M = {"observed": 0.02, "probed": 0.02, "worst_unit": 0.10, "gain": 0.02}
    best = None
    for r in out[1:]:
        bad = [k for k, m in M.items() if r[k] < ref[k] - m]
        if r["winners_worst"] < ref["winners_worst"] - 0.02:
            bad.append("winners")
        ok = not bad
        if ok and (best is None or r["bits_per_elem"] < best["bits_per_elem"]):
            best = r
        print(f"  {r['label']:<34}{r['bits_per_elem']:>8.4f}  "
              f"{'ADEQUATE' if ok else 'reject: ' + ','.join(bad)}")

    tables_frac = 2 * V * H / SOURCE_PARAM_COUNT
    print(f"\n  both tables are {100*tables_frac:.6f}% of N")
    print(f"  q4 g64  (G0)                 -> {(4+8*2/64)*tables_frac:.12f} complete BPW  [measured floor 0.401802783208]")
    print(f"  q4 g128                      -> {ref['bits_per_elem']*tables_frac:.12f} complete BPW")
    if best:
        b = best["bits_per_elem"] * tables_frac
        print(f"  CHEAPEST ADEQUATE            -> {b:.12f} complete BPW  ({best['label']})")
        left = 1.0 - b
        body = left * SOURCE_PARAM_COUNT / (SOURCE_PARAM_COUNT - 2 * V * H)
        print(f"  => body may average {body:.6f} bits/element for sub-1.0")
    else:
        print("  NO per-row allocation cheaper than uniform q4 is adequate on this tensor.")
        left = 1.0 - 0.401802783208
        body = left * SOURCE_PARAM_COUNT / (SOURCE_PARAM_COUNT - 2 * V * H)
        print(f"  => body must average {body:.6f} bits/element for sub-1.0, tables at G0 q4 g64")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
