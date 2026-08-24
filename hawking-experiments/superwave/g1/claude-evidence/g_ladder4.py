"""Re-score the 1-bit progressive ladder on FOUR axes.

The ladder is the strongest surviving density mechanism (G015: down_proj 1x1bit at 1.1250
b/w scored 0.613624 against flat q2 at 2.1250 b/w scoring 0.464836). That verdict predates
the gain axis, and a sign plane is exactly the construction most at risk from it: it keeps
direction by definition and gets its magnitude entirely from a fitted per-group scale.

So the question is whether the ladder's advantage survives being scored on magnitude.
Scored on HELD-OUT activations from the thick capture, against the same-tensor honest Q4
reference the gate judges by.
"""
import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, axes, c_uniform

CAP="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
res=json.load(open(f"{CAP}/capture-result.json")); site=res["sites"]["post_input_norm"]
NF,NH=site["n_fit"],site["n_hold"]
def Xh(l):
    p=site["per_layer"][str(l)]
    X=np.fromfile(p["path"],dtype=np.float16).reshape(-1,p["width"]).astype(np.float32)
    return X[NF:NF+NH]

def ladder(W, planes, group=128):
    """p sign planes, each with a least-squares per-group scale on the running residual."""
    m,n=W.shape; R=W.copy().astype(np.float32); out=np.zeros_like(R)
    g=group
    for _ in range(planes):
        S=np.sign(R); S[S==0]=1.0
        Rg=R.reshape(m,-1,g); Sg=S.reshape(m,-1,g)
        a=(Rg*Sg).sum(2)/g                      # LS scale per group
        out+=(Sg*a[:,:,None]).reshape(m,n); R=W-out
    return out

print(f"{'tensor':<22}{'scheme':<16}{'bits/w':>8}{'observed':>10}{'probed':>10}"
      f"{'worst_u':>10}{'GAIN':>10}")
for cls,l in (("mlp.down_proj",31),("mlp.gate_proj",31),("self_attn.q_proj",31)):
    try: W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    except Exception as e: print(cls,l,"skip",e); continue
    X = Xh(l) if W.shape[1]==5120 else None
    if X is None:
        rng=np.random.default_rng(1); X=rng.standard_normal((256,W.shape[1])).astype(np.float32)
    rows=[("flat q4 g128 (reference)", c_uniform(W,4,128), 4.125),
          ("flat q2 g128", c_uniform(W,2,128), 2.125),
          ("ladder 1x1bit", ladder(W,1), 1.125),
          ("ladder 2x1bit", ladder(W,2), 2.250),
          ("ladder 3x1bit", ladder(W,3), 3.375)]
    for name,Wh,bw in rows:
        a=axes(W,Wh,X,seed=None)
        print(f"{(cls+'@L'+str(l)) if name.startswith('flat q4') else '':<22}{name:<16}{bw:>8.3f}"
              f"{a['observed']:>10.6f}{a['probed']:>10.6f}{a['worst_unit']:>10.6f}{a['gain']:>10.6f}")
