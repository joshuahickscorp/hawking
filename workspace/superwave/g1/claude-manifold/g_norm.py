"""Guard against the outlier artifact before any low-rank claim is recorded.

"rank 256 holds 99.2% of fit energy" is an ENERGY-WEIGHTED statement. If a handful of rows
carry most of the squared norm, the top directions align with those rows and the number says
"the outliers are low rank", not "the activations are low rank". This campaign has already
been burned by a global-scale artifact, so the check runs before the claim.

Two readings side by side:
  energy-weighted   cumulative squared energy in the top-r fit directions
  per-row mean      mean over rows of the fraction of THAT row's own energy in the subspace
They agree when the distribution is even and diverge exactly when outliers dominate.
"""
import sys, json, numpy as np
sys.path.insert(0,'tools')
CAP="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
res=json.load(open(f"{CAP}/capture-result.json"))
for key in ("post_swiglu","post_input_norm"):
    site=res["sites"][key]; W=site["width"]; NF,NH=site["n_fit"],site["n_hold"]
    print(f"\n=== {key} (width {W}) ===")
    print(f"{'layer':>6}{'norm max/med':>13}{'top1% rows E':>13}"
          f"{'fit E@256':>11}{'fit row@256':>12}{'held E@256':>12}{'held row@256':>13}")
    for l in (0,31,63):
        p=site["per_layer"][str(l)]
        X=np.fromfile(p["path"],dtype=np.float16).reshape(-1,W).astype(np.float32)
        A,B=X[:NF],X[NF:NF+NH]
        na=np.linalg.norm(A,axis=1); e=na*na
        srt=-np.sort(-e); top1=srt[:max(1,len(e)//100)].sum()/e.sum()
        rng=np.random.default_rng(0)
        Q,_=np.linalg.qr(A.T@rng.standard_normal((A.shape[0],320)).astype(np.float32))
        _,_,Vt=np.linalg.svd(A@Q, full_matrices=False)
        V=(Q@Vt.T)[:,:256].astype(np.float32)
        def ew(M): return float(((M@V)**2).sum())/float((M*M).sum())
        def rw(M):
            num=((M@V)**2).sum(1); den=(M*M).sum(1)+1e-30
            return float(np.mean(num/den))
        print(f"{l:>6}{na.max()/np.median(na):>13.2f}{top1:>13.4f}"
              f"{ew(A):>11.6f}{rw(A):>12.6f}{ew(B):>12.6f}{rw(B):>13.6f}")
