"""CONTEXTUAL SPARSITY: the dense MLP may already be conditional.

Preceding result: post_swiglu rank 256 holds 99.2% of FIT energy at L0 but only 28.4% of
HELD energy. Narrow per prompt, rotating across prompts. The mechanical explanation is that
SwiGLU is a gate: h = silu(gate(x)) * up(x), so silu(gate(x)) soft-masks which of the 17408
channels carry the token. If so the dense operator is ALREADY conditional and the width is
a union over inputs, not a per-token requirement.

That changes which cost axis is attackable. It is not B: every channel still has to be
STORED because some token uses it. It is M and F: bytes MOVED and computation DONE per
token, which is what the 19.78 ms remainder and the 100 TPS target actually care about.

Measured per token on held-out rows:
    k99      channels holding 99% of that token's squared activation
    reuse    Jaccard overlap of active sets, within a prompt vs across prompts
The gap between those two overlaps is the whole mechanism: high within-prompt reuse with low
across-prompt reuse is a predictable working set. Equal overlaps mean there is nothing to
predict and the finding is a curiosity.
"""
import sys, json, numpy as np
sys.path.insert(0,'tools')
CAP="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
res=json.load(open(f"{CAP}/capture-result.json"))
site=res["sites"]["post_swiglu"]; W=site["width"]; NF,NH=site["n_fit"],site["n_hold"]
rng=np.random.default_rng(0)
print(f"{'layer':>6}{'k99 med':>9}{'k99 p95':>9}{'k99/W':>8}{'k999 med':>10}"
      f"{'J adjacent':>12}{'J far-same':>12}{'J random':>10}{'union/W':>9}")
for l in (0,15,31,47,63):
    p=site["per_layer"][str(l)]
    X=np.fromfile(p["path"],dtype=np.float16).reshape(-1,W).astype(np.float32)
    B=X[NF:NF+NH]
    S=B*B
    srt=-np.sort(-S,axis=1)
    cs=np.cumsum(srt,axis=1)/np.maximum(srt.sum(1,keepdims=True),1e-30)
    k99=(cs<0.99).sum(1)+1
    k999=(cs<0.999).sum(1)+1
    K=int(np.median(k99))
    top=np.argpartition(-S,K,axis=1)[:,:K]
    sets=[set(r.tolist()) for r in top[:400]]
    def J(i,j):
        a,b=sets[i],sets[j]; return len(a&b)/len(a|b)
    adj=np.mean([J(i,i+1) for i in range(0,len(sets)-1)])
    far=np.mean([J(i,i+16) for i in range(0,len(sets)-16)])
    idx=rng.integers(0,len(sets),(200,2))
    rnd=np.mean([J(int(a),int(b)) for a,b in idx if a!=b])
    union=len(set().union(*sets))/W
    print(f"{l:>6}{K:>9}{int(np.percentile(k99,95)):>9}{K/W:>8.4f}{int(np.median(k999)):>10}"
          f"{adj:>12.4f}{far:>12.4f}{rnd:>10.4f}{union:>9.4f}")
