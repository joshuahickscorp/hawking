#!/usr/bin/env python3
"""G061: does choosing the factorization for the STATE beat choosing it for the weights?

The acceptance describes W = U V^T with state maintained in V^T x rather than
repeatedly expanded. This architecture ALREADY does that and it is worth saying
before measuring anything: k_proj is [1024, 5120], so K = W_k x is a 1024-dim
projection of a 5120-dim hidden, and the KV cache stores exactly that. The cache
IS V^T x. Nothing is repeatedly expanded.

So the real question is not whether to factorize but at what INNER rank r < 1024,
and whether the r-dim subspace should be chosen to minimize weight error or state
error. Three candidates at matched rank:

  (a) WEIGHT SVD   best rank-r approximation of the W_k head block. Minimizes
                   ||W_k - W_hat||_F and sees no activations at all. This is what
                   "optimizing weights independently" means.
  (b) STATE PCA    best rank-r subspace of the K vectors actually produced on real
                   activations. Minimizes ||K - K_hat||_F on the data.
  (c) QUERY-WEIGHTED  best rank-r in the metric induced by the empirical query
                   second moment. K is only ever consumed through q^T k, so this
                   is the fully joint criterion -- it knows both what the state
                   contains and how the state is read.

Everything is scored on the ERROR IN THE ATTENTION SCORES q^T k, not on
reconstruction of K, because scores are what the mechanism actually produces. GQA
is respected: 48 query heads over 4 kv heads, 12 queries per kv head.

The subspaces are fitted on one half of the captured tokens and scored on the
other, so an in-sample PCA cannot win by memorising.

If (b) and (c) tie with (a), the joint framing bought nothing and this says so.

  ./tools/gravity_joint_state_weight.py --out receipts/.../G061_JOINT.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP = ROOT / "workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
HIDDEN, HEAD_DIM, KV_HEADS, Q_HEADS = 5120, 256, 4, 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=63)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--ranks", default="16,32,64,128,192")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    ranks = [int(r) for r in a.ranks.split(",")]

    Wq = load_tensor(f"language_model.model.layers.{a.layer}.self_attn.q_proj.weight").astype(np.float32)
    Wk = load_tensor(f"language_model.model.layers.{a.layer}.self_attn.k_proj.weight").astype(np.float32)
    x = np.fromfile(CAP / f"post_input_norm/L{a.layer:02d}.f16", dtype=np.float16,
                    count=a.tokens * HIDDEN).reshape(a.tokens, HIDDEN).astype(np.float32)
    half = x.shape[0] // 2
    xf, xe = x[:half], x[half:]
    print(f"L{a.layer}: W_q {Wq.shape}, W_k {Wk.shape}, tokens fit {half} / eval {xe.shape[0]}")
    print(f"KV cache stores W_k x directly: {Wk.shape[0]} dims per position per layer, "
          f"so the cache IS V^T x at rank {Wk.shape[0]}")

    per_head, eff = [], []
    for h in range(KV_HEADS):
        Wk_h = Wk[h * HEAD_DIM:(h + 1) * HEAD_DIM]                     # [256, 5120]
        Wq_h = Wq[h * 12 * HEAD_DIM:(h + 1) * 12 * HEAD_DIM]           # [3072, 5120]
        Kf, Ke = xf @ Wk_h.T, xe @ Wk_h.T
        Qf, Qe = xf @ Wq_h.T, xe @ Wq_h.T
        Qf = Qf.reshape(-1, 12, HEAD_DIM).reshape(-1, HEAD_DIM)
        Qe = Qe.reshape(-1, 12, HEAD_DIM).reshape(-1, HEAD_DIM)
        S = Qe @ Ke.T
        sn = np.linalg.norm(S)

        # Effective rank of the state itself, on held-out tokens.
        sv = np.linalg.svd(Ke, compute_uv=False)
        e = np.cumsum(sv ** 2) / (sv ** 2).sum()
        eff.append({"head": h, "rank_for_90pct": int(np.searchsorted(e, .90) + 1),
                    "rank_for_99pct": int(np.searchsorted(e, .99) + 1),
                    "rank_for_999pct": int(np.searchsorted(e, .999) + 1)})

        # (a) weight SVD -- no activations
        _, _, Vt_w = np.linalg.svd(Wk_h, full_matrices=False)
        # (b) state PCA on fitted tokens
        _, _, Vt_s = np.linalg.svd(Kf, full_matrices=False)
        # (c) query-weighted: whiten K by the empirical query second moment, since
        #     K is only consumed through q^T k
        G = (Qf.T @ Qf) / Qf.shape[0]
        w, U = np.linalg.eigh(G)
        Gh = U @ np.diag(np.sqrt(np.clip(w, 0, None))) @ U.T
        _, _, Vt_q = np.linalg.svd(Kf @ Gh, full_matrices=False)

        rows = []
        for r in ranks:
            out = {"rank": r}
            # weight-SVD basis lives in the 5120 input space; project W then re-derive K
            Wr = (Wk_h @ Vt_w[:r].T) @ Vt_w[:r]
            out["weight_svd"] = float(np.linalg.norm(Qe @ (xe @ Wr.T).T - S) / sn)
            for tag, Vt in (("state_pca", Vt_s), ("query_weighted", Vt_q)):
                P = Vt[:r].T @ Vt[:r]
                out[tag] = float(np.linalg.norm(Qe @ (Ke @ P).T - S) / sn)
            rows.append(out)
        per_head.append({"head": h, "ranks": rows})
        print(f"  head {h}: K effective rank 90/99/99.9% = "
              f"{eff[-1]['rank_for_90pct']}/{eff[-1]['rank_for_99pct']}/{eff[-1]['rank_for_999pct']}"
              f" of {HEAD_DIM}")

    print(f"\n{'rank':>6}{'weight SVD':>14}{'state PCA':>13}{'query-weighted':>17}"
          f"{'best/weight':>13}")
    agg = []
    for i, r in enumerate(ranks):
        w_ = float(np.mean([h["ranks"][i]["weight_svd"] for h in per_head]))
        s_ = float(np.mean([h["ranks"][i]["state_pca"] for h in per_head]))
        q_ = float(np.mean([h["ranks"][i]["query_weighted"] for h in per_head]))
        agg.append({"rank": r, "weight_svd": w_, "state_pca": s_, "query_weighted": q_,
                    "best_over_weight": w_ / min(s_, q_)})
        print(f"{r:>6}{w_:>14.5f}{s_:>13.5f}{q_:>17.5f}{w_/min(s_,q_):>12.2f}x")

    ties = all(a_["best_over_weight"] < 1.05 for a_ in agg)
    print(f"\nJOINT FRAMING: {'BOUGHT NOTHING -- ties with independent weight optimization' if ties else 'WINS'}"
          f" (best ratio {max(a_['best_over_weight'] for a_ in agg):.2f}x)")

    doc = {
        "schema": "hawking.nos.joint_state_weight.v1",
        "obligation": "G061 -- a factorization chosen because it induces a cheap state",
        "architecture_already_does_the_easy_half": (
            f"k_proj is {list(Wk.shape)}, so K = W_k x is a {Wk.shape[0]}-dim projection of a "
            f"{HIDDEN}-dim hidden and the KV cache stores exactly that. The cache IS V^T x; nothing "
            "is repeatedly expanded. The open question is the INNER rank r < "
            f"{Wk.shape[0]} and how its subspace is chosen."),
        "candidates": {
            "weight_svd": "best rank-r of the W_k head block; sees no activations. This is what "
                          "optimizing weights independently means.",
            "state_pca": "best rank-r subspace of the K vectors actually produced on real tokens",
            "query_weighted": "best rank-r in the metric of the empirical query second moment -- K "
                              "is only consumed through q^T k, so this knows both what the state "
                              "contains and how it is read",
        },
        "scored_on": "relative error in the attention scores q^T k, not reconstruction of K, "
                     "because scores are what the mechanism produces. GQA respected: 48 query "
                     "heads over 4 kv heads, 12 per kv head.",
        "holdout": "subspaces fitted on the first half of captured tokens, scored on the second, "
                   "so an in-sample PCA cannot win by memorising",
        "layer": a.layer, "tokens": a.tokens,
        "kv_effective_rank": eff, "per_head": per_head, "aggregate": agg,
        "joint_ties_with_independent": ties,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
