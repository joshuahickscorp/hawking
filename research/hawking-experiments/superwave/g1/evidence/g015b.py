import sys, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X, axes
G=128
for cls,l in (("self_attn.q_proj",31),("mlp.gate_proj",31)):
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    X=load_X(l); hold=X[X.shape[0]//2:]
    acc=np.zeros_like(W)
    print(f"\n{cls}@L{l}")
    print(f"{'rung':>5}{'||W-acc||':>13}{'rel_resid':>11}{'observed':>10}{'probed':>9}{'worst_u':>9}{'gate':>10}")
    nW=np.linalg.norm(W)
    for i in range(4):
        R=W-acc
        P=np.zeros_like(R)
        for s in range(0,R.shape[1]-R.shape[1]%G,G):
            blk=R[:,s:s+G]
            P[:,s:s+G]=np.sign(blk)*np.abs(blk).mean(axis=1,keepdims=True)
        acc=acc+P
        a=axes(W,acc,hold); g=min(a["observed"],a["probed"],a["worst_unit"])
        r=np.linalg.norm(W-acc)
        print(f"{i+1:>5}{r:>13.2f}{r/nW:>11.6f}{a['observed']:>10.6f}{a['probed']:>9.6f}{a['worst_unit']:>9.6f}{g:>10.6f}")
