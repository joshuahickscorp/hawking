"""Repair the ladder's magnitude for 0.003 bits per element.

The 1-bit progressive ladder keeps DIRECTION and loses MAGNITUDE: on L31 it beats flat q2 on
observed and probed at half the bits, but its gain is 0.143-0.655 against an honest Q4
reference of 0.774-0.953. The cause is structural, not incidental: each plane's per-group
scale is fitted by least squares on the residual, which minimises L2 error and is under no
obligation to preserve the norm of the output.

The fix should be almost free, because gain error that is CONSISTENT per output unit needs
only one number per output unit to undo. One f16 per row of W costs 2*8/n bits per element:
0.003125 at n=5120, 0.000919 at n=17408.

Fitted on the FIT half of the thick capture and scored on the HELD half, so the correction
cannot be a memorisation of the evaluation set.
"""
import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, axes, c_uniform

CAP="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
res=json.load(open(f"{CAP}/capture-result.json")); site=res["sites"]["post_input_norm"]
NF,NH=site["n_fit"],site["n_hold"]
def X_of(l):
    p=site["per_layer"][str(l)]
    X=np.fromfile(p["path"],dtype=np.float16).reshape(-1,p["width"]).astype(np.float32)
    return X[:NF], X[NF:NF+NH]

def ladder(W, planes, group=128):
    m,n=W.shape; R=W.astype(np.float32).copy(); out=np.zeros_like(R); g=group
    for _ in range(planes):
        S=np.sign(R); S[S==0]=1.0
        a=(R.reshape(m,-1,g)*S.reshape(m,-1,g)).sum(2)/g
        out+=(S.reshape(m,-1,g)*a[:,:,None]).reshape(m,n); R=W-out
    return out

def gain_fix(W, Wh, Xfit):
    """One f16 per output unit, fitted on the FIT half only."""
    Y, Yh = Xfit@W.T, Xfit@Wh.T
    c = np.linalg.norm(Y,axis=0)/(np.linalg.norm(Yh,axis=0)+1e-30)
    c = c.astype(np.float16).astype(np.float32)      # stored as f16, so scored as f16
    return Wh * c[:,None]

print(f"{'tensor':<22}{'scheme':<22}{'bits/w':>8}{'observed':>10}{'probed':>10}"
      f"{'worst_u':>10}{'GAIN':>10}")
for cls,l in (("mlp.down_proj",31),("mlp.gate_proj",31),("self_attn.q_proj",31)):
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    m,n=W.shape
    if n==5120: A,B=X_of(l)
    else:
        rng=np.random.default_rng(1); A=rng.standard_normal((512,n)).astype(np.float32); B=A
    fix_bits = 2*8/n
    ref=axes(W,c_uniform(W,4,128),B,seed=None)
    print(f"{cls+'@L'+str(l):<22}{'flat q4 g128 (ref)':<22}{4.125:>8.3f}"
          f"{ref['observed']:>10.6f}{ref['probed']:>10.6f}{ref['worst_unit']:>10.6f}{ref['gain']:>10.6f}")
    for p in (1,2,3):
        L=ladder(W,p)
        for tag,Wh,extra in ((f"ladder {p}x1bit", L, 0.0),
                             (f"ladder {p}x1bit +gainfix", gain_fix(W,L,A), fix_bits)):
            a=axes(W,Wh,B,seed=None)
            print(f"{'':<22}{tag:<22}{p*1.125+extra:>8.6f}"
                  f"{a['observed']:>10.6f}{a['probed']:>10.6f}{a['worst_unit']:>10.6f}{a['gain']:>10.6f}")
