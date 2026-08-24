"""Does the MLP interior fail to be narrow because it is HIGH RANK, or because its narrow
subspace is PROMPT-DEPENDENT? The two imply different mechanisms and must not be conflated.

  high rank              -> no narrow interior exists; static AND conditional both dead
  prompt-dependent       -> no SHARED narrow interior; a CONDITIONAL interior stays open

Distinguished by reading fit energy and held energy on the same fitted basis. Fit energy
near 1.0 with held energy far below it is prompt dependence. Both low is high rank.

Structural reason to expect high rank, stated up front so the result can falsify it:
gate(x) and up(x) are linear in a 5120-dim input, so each has rank <= 5120. But
h = silu(gate(x)) * up(x) is an ELEMENTWISE PRODUCT of two such matrices, and that product
is not rank-bounded by 5120. The 17408 width may therefore be doing real work.
"""
import sys, json, numpy as np, time
sys.path.insert(0,'tools')
CAP="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
res=json.load(open(f"{CAP}/capture-result.json"))
K=3072
site=res["sites"]["post_swiglu"]; W=site["width"]; NF,NH=site["n_fit"],site["n_hold"]
print(f"post_swiglu width {W}, fit {NF} hold {NH}, range finder k={K+64}")
print(f"\n{'layer':>6}{'fitE@256':>10}{'fitE@1024':>10}{'fitE@k':>9}"
      f"{'heldE@256':>11}{'heldE@1024':>11}{'heldE@k':>9}   verdict")
for l in (0,31,63):
    p=site["per_layer"][str(l)]
    X=np.fromfile(p["path"],dtype=np.float16).reshape(-1,W).astype(np.float32)
    A,B=X[:NF],X[NF:NF+NH]
    rng=np.random.default_rng(0)
    Q,_=np.linalg.qr(A.T@rng.standard_normal((A.shape[0],K+64)).astype(np.float32))
    _,s,Vt=np.linalg.svd(A@Q, full_matrices=False)
    V=(Q@Vt.T).astype(np.float32)
    fa=np.cumsum(((A@V)**2).sum(0))/float((A*A).sum())
    fb=np.cumsum(((B@V)**2).sum(0))/float((B*B).sum())
    v = "HIGH RANK" if fa[-1] < 0.9 else ("prompt-dependent" if fb[-1] < 0.9 else "narrow")
    print(f"{l:>6}{fa[255]:>10.6f}{fa[1023]:>10.6f}{fa[-1]:>9.6f}"
          f"{fb[255]:>11.6f}{fb[1023]:>11.6f}{fb[-1]:>9.6f}   {v}")
