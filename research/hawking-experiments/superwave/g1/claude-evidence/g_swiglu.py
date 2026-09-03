"""The MLP interior: is the 17408-wide intermediate actually 17408-dimensional?

The three MLP tensors are 63.6% of N. They exist only to produce and consume one vector:
    h = SwiGLU(gate(x), up(x))  in R^17408, then down(h) in R^5120.
Nothing outside the block ever sees h. So h's width is a REPRESENTATION CHOICE, not a
semantic requirement -- the source parameterization is an IR, not a law.

If h occupies rank q << 17408 on real text, the whole block factors through R^q and 63.6%
of N shrinks by 17408/q with no per-element code anywhere in it.

Measured on the thick capture's post_swiglu site with the capture's own prompt-level
holdout, so rank is read on text the basis was NOT fitted to. Fit energy saturates by
construction and would report a rank that does not exist; held-out energy is the real number.

Randomized range finder at k=3072 rather than a 17408^3 eigendecomposition. The reported
rank is therefore a LOWER BOUND on the true 0.999 rank whenever it saturates at k.
"""
import sys, json, numpy as np, time
sys.path.insert(0,'tools')
CAP="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
res=json.load(open(f"{CAP}/capture-result.json"))
K=3072

def held_rank_curve(path, width, nf, nh, k=K, seed=0):
    X=np.fromfile(path,dtype=np.float16).reshape(-1,width).astype(np.float32)
    A,B=X[:nf],X[nf:nf+nh]
    rng=np.random.default_rng(seed)
    Om=rng.standard_normal((A.shape[0], k+64)).astype(np.float32)
    Q,_=np.linalg.qr(A.T@Om)                 # width x (k+64), spans A's column space
    _,s,Vt=np.linalg.svd(A@Q, full_matrices=False)
    V=(Q@Vt.T).astype(np.float32)            # width x (k+64), fit principal directions
    P=B@V
    e=np.cumsum((P*P).sum(0)); tot=float((B*B).sum())
    return e/tot, V.shape[1]

for name,site_key in (("post_swiglu","post_swiglu"),):
    site=res["sites"][site_key]; W=site["width"]; NF,NH=site["n_fit"],site["n_hold"]
    print(f"{site_key}: width {W}, {site['store_n']} rows, fit {NF} hold {NH}, range finder k={K}")
    print(f"\n{'layer':>6}{'r@0.99':>9}{'r@0.999':>10}{'r@0.9999':>10}{'held E@k':>10}"
          f"{'r999/w':>9}{'MLP b/el':>10}{'x smaller':>10}")
    for l in (0,7,15,23,31,39,47,55,63):
        p=site["per_layer"][str(l)]
        t=time.time(); e,kk=held_rank_curve(p["path"], W, NF, NH); dt=time.time()-t
        def rk(th):
            i=np.searchsorted(e,th)
            return int(i+1) if i < len(e) else -1
        r99,r999,r9999=rk(0.99),rk(0.999),rk(0.9999)
        rr = r999 if r999>0 else kk
        # block refactored through R^rr: gate,up : 5120->rr ; down : rr->5120
        # source elements of the block = 3 * 17408 * 5120
        vals = 2*5120*rr + rr*5120
        bel = vals*4 / (3*17408*5120)
        print(f"{l:>6}{r99:>9}{r999:>10}{r9999:>10}{e[-1]:>10.6f}{rr/W:>9.4f}"
              f"{bel:>10.4f}{17408/max(rr,1):>10.2f}   ({dt:.0f}s)")
