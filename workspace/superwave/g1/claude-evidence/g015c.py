import sys, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X, _probe
G=128
def unit_cos(A,B):
    num=(A*B).sum(0); na=np.linalg.norm(A,axis=0); nb=np.linalg.norm(B,axis=0)
    live=na>1e-20; c=np.zeros_like(num); c[live]=num[live]/(na[live]*nb[live]+1e-30)
    return c[live]
for cls,l in (("self_attn.q_proj",31),("mlp.gate_proj",31)):
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    X=load_X(l); hold=X[X.shape[0]//2:]; P=_probe(W.shape[1],seed=0)
    acc=np.zeros_like(W)
    print(f"\n{cls}@L{l}   units={W.shape[0]}")
    print(f"{'rung':>5}{'min':>10}{'q0.1%':>10}{'q1%':>10}{'q5%':>10}")
    for i in range(4):
        R=W-acc; Pl=np.zeros_like(R)
        for s in range(0,R.shape[1]-R.shape[1]%G,G):
            blk=R[:,s:s+G]; Pl[:,s:s+G]=np.sign(blk)*np.abs(blk).mean(axis=1,keepdims=True)
        acc=acc+Pl
        c=np.concatenate([unit_cos(hold@W.T,hold@acc.T), unit_cos(P@W.T,P@acc.T)])
        print(f"{i+1:>5}{c.min():>10.6f}{np.quantile(c,0.001):>10.6f}{np.quantile(c,0.01):>10.6f}{np.quantile(c,0.05):>10.6f}")
