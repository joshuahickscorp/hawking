import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, c_uniform
C="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
H=np.fromfile(f"{C}/final_norm/L00.f16",dtype=np.float16).astype(np.float32).reshape(-1,5120)
rng=np.random.default_rng(0); X=H[rng.choice(len(H),768,replace=False)]
W=load_tensor("language_model.lm_head.weight").astype(np.float32)
L=X@W.T; a0=L.argmax(1)
part=np.partition(L,-2,axis=1); m=(part[:,-1]-part[:,-2])/(L.std(1)+1e-30)
SPECIAL=set(range(248044,248077))          # added/control tokens (verified earlier)
cls=np.array(["control" if int(t) in SPECIAL else "ordinary" for t in a0])
print(f"{'class':>10}{'n':>6}{'median m/sd':>13}{'p10':>8}{'frac m/sd<0.5':>15}")
for c in ("ordinary","control"):
    sel=cls==c
    if sel.sum()==0: print(f"{c:>10}{0:>6}   (none predicted in sample)"); continue
    print(f"{c:>10}{sel.sum():>6}{np.median(m[sel]):>13.4f}{np.quantile(m[sel],.10):>8.4f}{(m[sel]<0.5).mean():>15.4f}")
print(f"\n{'bits':>5}{'flip ordinary':>15}{'flip control':>14}")
for bits in (2,3,4,6):
    a=(X@c_uniform(W,bits,128).T).argmax(1); fl=a!=a0
    o=fl[cls=="ordinary"].mean() if (cls=="ordinary").any() else float('nan')
    k=fl[cls=="control"].mean() if (cls=="control").any() else float('nan')
    print(f"{bits:>5}{o:>15.4f}{k:>14.4f}")
# does quantization ever CREATE a control token where there was none, or destroy one?
for bits in (2,4):
    a=(X@c_uniform(W,bits,128).T).argmax(1)
    was=np.array([int(t) in SPECIAL for t in a0]); now=np.array([int(t) in SPECIAL for t in a])
    print(f"bits={bits}: control->ordinary {int((was&~now).sum())}  ordinary->control {int((~was&now).sum())}")
