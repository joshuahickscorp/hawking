"""Is there ONE input basis for many layers?

Every layer reads the SAME residual stream. If their input covariances are aligned, a single
shared basis V serves a whole band and its n x r cost amortizes across sites -- which is the
only way the shared-basis family gets under 1 bit per element. If they are not aligned, V is
per-site, costs r*n*2 bytes at one site, and the family dies on arithmetic alone.

Measured three ways, because subspace overlap has a misleading scalar form:
  chordal    mean squared cosine between the two top-r subspaces, 1.0 = identical
  energy     how much of layer B's activation energy the basis fitted at layer A holds
  null       the same for a random r-dim subspace, so overlap can be read as good or bad

The decisive column is ENERGY HELD BY A'S BASIS ON B'S ACTIVATIONS. Subspace angles are
geometry; energy is what the operator actually loses.
"""
import sys, json, numpy as np
sys.path.insert(0,'tools')
CAP="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
res=json.load(open(f"{CAP}/capture-result.json")); site=res["sites"]["post_input_norm"]
NF=site["n_fit"]
def X_of(l):
    p=site["per_layer"][str(l)]
    return np.fromfile(p["path"],dtype=np.float16).reshape(p["n_rows"],p["width"]).astype(np.float32)
def basis(l,r):
    A=X_of(l)[:NF]
    w,V=np.linalg.eigh((A.T@A).astype(np.float64))
    return V[:,::-1][:,:r].astype(np.float32)
R=256
LS=[0,7,15,23,31,39,47,55,63]
B={l:basis(l,R) for l in LS}
rng=np.random.default_rng(0)
Q,_=np.linalg.qr(rng.standard_normal((5120,R)).astype(np.float32))
print(f"basis rank r={R} of 5120, fitted on {NF} fit rows per layer\n")
print(f"{'fit@':>5}{'eval@':>6}{'chordal':>9}{'energy held':>13}{'random null':>13}")
for la in LS:
    for lb in LS:
        Xb=X_of(lb)[:NF]; e0=float((Xb*Xb).sum())
        eh=float(((Xb@B[la])**2).sum())/e0
        en=float(((Xb@Q)**2).sum())/e0
        ch=float((B[la].T@B[lb])**2 .sum())/R if False else float(((B[la].T@B[lb])**2).sum())/R
        if la==lb or abs(la-lb)<=8 or la==0 or lb==63:
            print(f"{la:>5}{lb:>6}{ch:>9.5f}{eh:>13.6f}{en:>13.6f}")
