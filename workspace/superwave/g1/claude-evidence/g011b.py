import sys, numpy as np, json
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor

def flatcos(A,B):
    a,b=A.ravel(),B.ravel(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))

def sign_align(A,B):
    s=np.sign((A*B).sum(0)+1e-30); return A*s

def scale_align(A,B):
    # per-input-channel diagonal: best d minimising ||A diag(d) - B||
    num=(A*B).sum(0); den=(A*A).sum(0)+1e-30
    return A*(num/den)

def perm_align(A,B,cap=1024):
    # greedy matching on column correlation. full Hungarian at n=5120 is O(n^3); greedy
    # on the top-`cap` columns by norm is the affordable proxy and is what a real packer
    # could also afford.
    An=A/(np.linalg.norm(A,axis=0,keepdims=True)+1e-30)
    Bn=B/(np.linalg.norm(B,axis=0,keepdims=True)+1e-30)
    idx=np.argsort(-np.linalg.norm(A,axis=0))[:cap]
    C=An[:,idx].T@Bn[:,idx]
    out=np.arange(A.shape[1]); used=set(); order=np.argsort(-np.abs(C).max(1))
    for i in order:
        j=int(np.argmax(np.abs(C[i])*np.array([0 if k in used else 1 for k in range(C.shape[1])])))
        if j in used: continue
        used.add(j); out[idx[i]]=idx[j]
    return A[:,out]

def procrustes(A,B):
    # UNAFFORDABLE (0.998050 BPW over 64 sites) -- measured only as the CEILING.
    M=A.T@B
    U,_,Vt=np.linalg.svd(M,full_matrices=False)
    return A@(U@Vt)

pairs=[("mlp.gate_proj",30,31),("mlp.down_proj",30,31),("self_attn.q_proj",31,35),
       ("mlp.gate_proj",15,47)]
print(f"{'class':<17}{'pair':>9}{'raw':>8}{'sign':>8}{'scale':>8}{'perm':>8}{'procrustes':>11}")
rows=[]
for cls,la,lb in pairs:
    A=load_tensor(f"language_model.model.layers.{la}.{cls}.weight").astype(np.float32)
    B=load_tensor(f"language_model.model.layers.{lb}.{cls}.weight").astype(np.float32)
    if A.shape!=B.shape: continue
    r=flatcos(A,B)
    sg=flatcos(sign_align(A,B),B)
    sc=flatcos(scale_align(A,B),B)
    pm=flatcos(perm_align(A,B),B)
    pr=flatcos(procrustes(A,B),B)
    rows.append(dict(cls=cls,pair=(la,lb),raw=r,sign=sg,scale=sc,perm=pm,procrustes=pr))
    print(f"{cls:<17}{str((la,lb)):>9}{r:>8.4f}{sg:>8.4f}{sc:>8.4f}{pm:>8.4f}{pr:>11.4f}")
print()
aff=[x['procrustes'] for x in rows]; cheap=[max(x['sign'],x['scale'],x['perm']) for x in rows]
print(f"affordable-transform best: {min(cheap):.4f}-{max(cheap):.4f}")
print(f"procrustes CEILING       : {min(aff):.4f}-{max(aff):.4f}  (costs 0.998050 BPW/64 sites)")
json.dump(rows,open('/tmp/g011b.json','w'),indent=2,default=str)
