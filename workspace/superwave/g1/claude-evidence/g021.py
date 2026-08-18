import sys, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, c_uniform
C="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
def site(s,l,w,n):
    return np.fromfile(f"{C}/{s}/L{l:02d}.f16",dtype=np.float16).astype(np.float32).reshape(-1,w)[:n]
def mlp(W,x):
    a=x@W[0].T; return ((a/(1.0+np.exp(-a)))*(x@W[1].T))@W[2].T
def blk(l):
    return [load_tensor(f"language_model.model.layers.{l}.mlp.{n}_proj.weight").astype(np.float32) for n in ("gate","up","down")]

print(f"{'pair':>10}{'bits':>5}{'err_uncomp':>12}{'err_comp':>10}{'cancelled':>11}")
for l in (15,31,47):
    j=l+1
    Wl,Wj=blk(l),blk(j)
    X=site("post_input_norm",l,5120,6827); fit,hold=X[:len(X)//2],X[len(X)//2:]
    for bits in (2,3):
        Wlq=[c_uniform(w,bits,128) for w in Wl]
        # teacher: exact l then exact j.  student: quantized l then j (refit or not)
        def chain(Wa,Wb,x):
            m=x+mlp(Wa,x); return m+mlp(Wb,m)
        T=chain(Wl,Wj,hold)
        S_un=chain(Wlq,Wj,hold)
        # compensate: refit j's down against the perturbed mid-state on the FIT half
        mf_t=fit+mlp(Wl,fit); mf_s=fit+mlp(Wlq,fit)
        Ht=(lambda a:(a/(1.0+np.exp(-a))))(mf_t@Wj[0].T)*(mf_t@Wj[1].T)
        Hs=(lambda a:(a/(1.0+np.exp(-a))))(mf_s@Wj[0].T)*(mf_s@Wj[1].T)
        tgt=Ht@Wj[2].T + (mf_t-mf_s)          # ask j to also absorb the mid-state offset
        dq=np.linalg.lstsq(Hs,tgt,rcond=1e-3)[0].T.astype(np.float32)
        Wjc=[Wj[0],Wj[1],c_uniform(dq,bits,128)]
        S_c=chain(Wlq,Wjc,hold)
        e_un=float(np.linalg.norm(S_un-T)/np.linalg.norm(T))
        e_c =float(np.linalg.norm(S_c -T)/np.linalg.norm(T))
        print(f"{str((l,j)):>10}{bits:>5}{e_un:>12.6f}{e_c:>10.6f}{1-e_c/max(e_un,1e-30):>+11.4f}")
