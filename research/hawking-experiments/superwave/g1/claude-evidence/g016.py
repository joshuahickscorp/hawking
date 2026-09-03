import sys, json, math, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X, axes, c_uniform
G=128
def gate(W,Wh,X):
    a=axes(W,Wh,X); return min(a["observed"],a["probed"],a["worst_unit"]), a

for cls,l in (("mlp.gate_proj",31),("self_attn.q_proj",31)):
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    X=load_X(l); fit,hold=X[:X.shape[0]//2],X[X.shape[0]//2:]
    base=c_uniform(W,2,G)                      # cheap base the exceptions must rescue
    g0,_=gate(W,base,hold)
    E=W.size; rows,cols=W.shape
    xe=np.sqrt((fit*fit).mean(0))+1e-30        # per-input-channel activation energy
    err=W-base
    print(f"\n{cls}@L31  base q2 gate {g0:.6f}   shape {W.shape}")
    print(f"{'selector':<22}{'topology':<10}{'n':>8}{'+BPW':>9}{'gate':>10}{'delta':>9}")
    trials=[]
    # scattered exceptions: index cost log2(E) bits each + 16 bit value
    for name,score in (("|W| magnitude", np.abs(err)),
                       ("functional |X.err|", np.abs(err)*xe[None,:])):
        for frac in (1e-4, 1e-3):
            k=int(E*frac)
            idx=np.argpartition(score.ravel(), -k)[-k:]
            Wh=base.copy().ravel(); Wh[idx]=W.ravel()[idx]; Wh=Wh.reshape(W.shape)
            bits=k*(math.log2(E)+16)
            g,_=gate(W,Wh,hold); trials.append((name,"scattered",k,bits/E,g,g-g0))
    # ROW exceptions: index cost log2(rows) bits, covers `cols` values each
    rs_w=np.linalg.norm(err,axis=1)
    rs_f=np.linalg.norm(err*xe[None,:],axis=1)
    for name,score in (("|W| magnitude",rs_w),("functional |X.err|",rs_f)):
        for nr in (16,128):
            idx=np.argpartition(score,-nr)[-nr:]
            Wh=base.copy(); Wh[idx]=W[idx]
            bits=nr*(math.log2(rows)+16*cols)
            g,_=gate(W,Wh,hold); trials.append((name,"rows",nr,bits/E,g,g-g0))
    for n,t,k,b,g,d in trials:
        print(f"{n:<22}{t:<10}{k:>8}{b:>9.4f}{g:>10.6f}{d:>+9.6f}")
    # efficiency: gate gain per BPW spent
    print("  gain per BPW:")
    for n,t,k,b,g,d in sorted(trials,key=lambda r:-(r[5]/max(r[3],1e-9)))[:4]:
        print(f"    {n:<22}{t:<10} {d/max(b,1e-9):>10.3f} per BPW  (+{d:.4f} at {b:.4f} BPW)")
