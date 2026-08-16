"""Is there cross-expert structure in Q80's 512 routed experts worth exploiting?

Gravity's premise is that a model has structure a per-tensor quantizer cannot see. Q80 has
512 experts x 48 layers = 24,576 expert triples. If experts within a layer share a subspace,
a shared basis + small per-expert delta beats quantizing each expert alone.

This measures whether that structure EXISTS before anyone builds a codec for it.
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,"/Users/scammermike/Downloads/hawking")
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

MD=Path(os.environ["MODEL_DIR"]); rows=json.load(open(os.environ["ROWS"]))
LAYER=10
experts=sorted(int(k.split("_")[1]) for k in rows if k.startswith(f"{LAYER}_"))[:96]
wmap=load_weight_map(MD)
print(f"[cross] layer {LAYER}, {len(experts)} experts", flush=True)

out={"layer":LAYER,"n_experts":len(experts),"components":{}}
for comp in ("gate_proj","up_proj"):
    W=np.stack([np.asarray(load_tensor(MD,wmap,
        f"model.layers.{LAYER}.mlp.experts.{e}.{comp}.weight"),dtype=np.float32) for e in experts])
    n,r,c = W.shape
    print(f"\n=== {comp}  stack {W.shape}  ({n*r*c*4/1e9:.2f} GB fp32)", flush=True)

    # 1. Do experts occupy a shared subspace? SVD of the flattened stack.
    F = W.reshape(n, r*c)
    Fc = F - F.mean(0, keepdims=True)
    # economical: singular values of the n x n Gram
    G = Fc @ Fc.T
    ev = np.linalg.eigvalsh(G)[::-1].clip(0)
    tot = ev.sum()
    cum = np.cumsum(ev)/max(tot,1e-30)
    k50 = int(np.searchsorted(cum,0.50))+1
    k90 = int(np.searchsorted(cum,0.90))+1
    k99 = int(np.searchsorted(cum,0.99))+1
    print(f"  expert-space energy: {k50}/{n} comps for 50%, {k90} for 90%, {k99} for 99%")

    # 2. How similar are experts pairwise, and how much does the MEAN explain?
    mean_w = F.mean(0)
    mn = np.linalg.norm(mean_w); fn = np.linalg.norm(F,axis=1)
    mean_frac = float((F@mean_w/ (fn*mn+1e-30)).mean())
    # pairwise cosine on a subsample
    idx = np.random.default_rng(0).choice(n, size=min(24,n), replace=False)
    S = F[idx]/ (np.linalg.norm(F[idx],axis=1,keepdims=True)+1e-30)
    pc = S@S.T; off = pc[~np.eye(len(idx),dtype=bool)]
    print(f"  mean-direction cosine: {mean_frac:.4f} | pairwise expert cosine: "
          f"mean {off.mean():.4f} p95 {np.percentile(off,95):.4f} max {off.max():.4f}")

    # 3. Column (input-channel) subspace sharing: do experts share row-space structure?
    #    Compare per-expert top-32 right singular subspace overlap between two experts.
    def topV(M,k=32):
        _,_,Vt = np.linalg.svd(M, full_matrices=False)
        return Vt[:k]
    a,b = topV(W[0]), topV(W[1])
    overlap = float(np.linalg.norm(a@b.T,'fro')**2/32.0)
    print(f"  top-32 right-subspace overlap, expert[0] vs expert[1]: {overlap:.4f} (1.0 = identical)")

    out["components"][comp]={
        "k50":k50,"k90":k90,"k99":k99,"n":n,
        "mean_direction_cosine":mean_frac,
        "pairwise_cosine_mean":float(off.mean()),
        "pairwise_cosine_p95":float(np.percentile(off,95)),
        "subspace_overlap_top32":overlap,
    }
    del W,F,Fc,G
Path(os.environ["OUT"]).write_text(json.dumps(out,indent=2))
print(f"\nwrote {os.environ['OUT']}", flush=True)
