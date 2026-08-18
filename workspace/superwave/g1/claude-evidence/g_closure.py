"""THE DECIDING TEST for the manifold operator.

The manifold operator W V_r V_r^T is EXACT (observed 1.000000) on the activations it was
fitted to, at rank 256 = 0.200 bits/element, which would clear sub-1.0 by a wide margin.
Its probed score is 0.32 -- statistically the same as a RANDOM subspace of equal rank
(0.3157), so the probe axis says the activation basis buys nothing off-span.

That leaves exactly one question worth asking, and probes cannot answer it:

    DOES REAL HELD-OUT TEXT VISIT DIRECTIONS OUTSIDE THE FITTED SPAN?

If not, the operator is faithful on everything reachable and the probes are testing
directions that do not occur. If so, the family dies on real data and no guard argument is
needed. Isotropic probes are a proxy for reachability; held-out prompts ARE reachability.

So: fit V_r on prompts [0..k), score the SAME operator on prompts [k..end). The train score
is the ceiling; the held-out score is the answer. A random basis of equal rank is carried
through as the null so a held-out number can be read as good or bad rather than merely large.
"""
import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X, _rowcos

CAP="workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
res=json.load(open(f"{CAP}/capture-result.json"))
lens=[p["n_tokens"] for p in res["prompts"]]
names=[p["prompt"][:34] for p in res["prompts"]]
bounds=np.cumsum([0]+lens)
print(f"capture: {len(lens)} prompts, {bounds[-1]} rows")
for i,(nm,L) in enumerate(zip(names,lens)): print(f"  [{i}] {L:>4} rows  {nm}")
k=len(lens)//2
tr=slice(0,bounds[k]); te=slice(bounds[k],bounds[-1])
print(f"\nfit on prompts [0..{k}) = {bounds[k]} rows; held out [{k}..{len(lens)}) = {bounds[-1]-bounds[k]} rows\n")

print(f"{'tensor':<24}{'r':>6}{'b/el':>7}{'trainE':>9}{'heldE':>9}"
      f"{'obs train':>11}{'obs HELD':>11}{'null HELD':>11}")
for cls,l in (("mlp.gate_proj",0),("mlp.gate_proj",31),("mlp.gate_proj",63),
              ("linear_attn.in_proj_qkv",30)):
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    n=W.shape[1]; X=load_X(l)
    if X is None or X.shape[1]!=n or X.shape[0]!=bounds[-1]: 
        print(f"{cls}@L{l}: shape mismatch {None if X is None else X.shape}"); continue
    A,B=X[tr],X[te]
    w,V=np.linalg.eigh((A.T@A).astype(np.float64)); V=V[:,::-1].astype(np.float32)
    rng=np.random.default_rng(0)
    for r in (256,512,1024):
        Vr=V[:,:r]
        Wh=(W@Vr)@Vr.T
        eA=float(((A@Vr)**2).sum())/float((A*A).sum())
        eB=float(((B@Vr)**2).sum())/float((B*B).sum())
        ot=_rowcos(A@W.T, A@Wh.T); oh=_rowcos(B@W.T, B@Wh.T)
        Q,_=np.linalg.qr(rng.standard_normal((n,r)).astype(np.float32))
        Wn=(W@Q)@Q.T; on=_rowcos(B@W.T, B@Wn.T)
        print(f"{(cls+'@L'+str(l)) if r==256 else '':<24}{r:>6}{r*4/n:>7.3f}"
              f"{eA:>9.6f}{eB:>9.6f}{ot:>11.6f}{oh:>11.6f}{on:>11.6f}")
