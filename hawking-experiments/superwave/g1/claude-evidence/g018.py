import sys, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, c_uniform, axes
C="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"

def fwht(a):
    """in-place fast Walsh-Hadamard along last axis, length must be a power of 2"""
    a=a.copy(); n=a.shape[-1]; h=1
    while h<n:
        for i in range(0,n,h*2):
            x=a[...,i:i+h].copy(); y=a[...,i+h:i+2*h].copy()
            a[...,i:i+h]=x+y; a[...,i+h:i+2*h]=x-y
        h*=2
    return a/np.sqrt(n)

def load_site(site,l,w,n):
    X=np.fromfile(f"{C}/{site}/L{l:02d}.f16",dtype=np.float16).astype(np.float32)
    return X.reshape(-1,w)[:n]

# 5120 = 4096 + 1024, not a power of 2 -> use BLOCK Hadamard, which is also what a
# kernel would actually do (per-group, matching the quantization group)
def block_h(M, blk):
    d=M.shape[-1]; out=M.copy()
    for s in range(0, d-d%blk, blk):
        out[..., s:s+blk]=fwht(M[..., s:s+blk])
    return out

for l in (0,31,63):
    W=load_tensor(f"language_model.model.layers.{l}.mlp.gate_proj.weight").astype(np.float32)
    X=load_site("post_input_norm",l,5120,6827); hold=X[len(X)//2:]
    print(f"\nL{l} gate_proj   (block Hadamard, zero storage)")
    print(f"{'blk':>5}{'bits':>5}{'kurtosis raw':>14}{'kurt after':>12}{'gate raw':>10}{'gate H':>9}{'delta':>9}")
    for blk in (128, 1024):
        Wh_=block_h(W,blk); Xh=block_h(hold,blk)
        k0=float(((W-W.mean())**4).mean()/((W.var()+1e-30)**2))
        k1=float(((Wh_-Wh_.mean())**4).mean()/((Wh_.var()+1e-30)**2))
        for bits in (2,3):
            a0=axes(W, c_uniform(W,bits,128), hold); g0=min(a0.values())
            # transformed domain: quantize W_H, score against W_H on transformed activations
            a1=axes(Wh_, c_uniform(Wh_,bits,128), Xh); g1=min(a1.values())
            print(f"{blk:>5}{bits:>5}{k0:>14.3f}{k1:>12.3f}{g0:>10.6f}{g1:>9.6f}{g1-g0:>+9.6f}")
