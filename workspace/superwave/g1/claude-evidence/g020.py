import sys, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, c_uniform
C="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
H=np.fromfile(f"{C}/final_norm/L00.f16",dtype=np.float16).astype(np.float32).reshape(-1,5120)
rng=np.random.default_rng(0); idx=rng.choice(len(H),512,replace=False); X=H[idx]
W=load_tensor("language_model.lm_head.weight").astype(np.float32) if True else None
print("lm_head",W.shape,"tokens",X.shape)
L=X@W.T
part=np.partition(L,-2,axis=1)
top1,top2=part[:,-1],part[:,-2]
m=top1-top2
sd=L.std(1)+1e-30
mn=m/sd                                   # margin in units of logit spread
print(f"margin abs   p1 {np.quantile(m,.01):.4f}  p10 {np.quantile(m,.10):.4f}  median {np.median(m):.4f}  p90 {np.quantile(m,.90):.4f}")
print(f"margin /sd   p1 {np.quantile(mn,.01):.4f}  p10 {np.quantile(mn,.10):.4f}  median {np.median(mn):.4f}")
# how much lm_head error flips the argmax?
a0=L.argmax(1)
print(f"\n{'bits':>5}{'flip rate':>11}{'flips@low-margin decile':>25}")
lo=mn<=np.quantile(mn,.10)
for bits in (2,3,4,6):
    Wq=c_uniform(W,bits,128)
    a=(X@Wq.T).argmax(1)
    fl=(a!=a0)
    print(f"{bits:>5}{fl.mean():>11.4f}{fl[lo].mean():>25.4f}")
print(f"\nfragile = bottom decile by margin/sd; they flip at {fl[lo].mean()/max(fl.mean(),1e-9):.1f}x the base rate at 6 bits")
