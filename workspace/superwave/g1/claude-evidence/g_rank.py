"""Is the affordable rank enough? The envelope says the body must average 0.871
bits/element, which a shared-input-basis buys at rank <= 20.7% of n. So: score real
rank-r truncations with the probe-inclusive adequacy gate against the honest-Q4 reference.

Randomized range finder rather than full SVD: 17408x5120 exact SVD is not worth the wall
clock when the question is a score at a few ranks.
"""
import sys, time, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X, axes, c_uniform

def lowrank(W, r, over=32, seed=0):
    rng=np.random.default_rng(seed)
    n=W.shape[1]
    Om=rng.standard_normal((n, r+over)).astype(np.float32)
    Y=W@Om
    Q,_=np.linalg.qr(Y)
    B=Q.T@W
    U,s,Vt=np.linalg.svd(B, full_matrices=False)
    return (Q@U[:,:r])@(np.diag(s[:r])@Vt[:r])

def bits_shared(r, n, bits=4, group=128):
    return r*bits/n + r*8*2/(group*n)
def bits_persite(r, m, n, bits=8):
    return (m+n)*r*bits/(m*n)

print(f"{'tensor':<22}{'r':>6}{'r/n':>7}{'b/elem sh':>10}{'observed':>10}{'probed':>10}{'worst_u':>10}{'s':>6}")
for cls, l in (("mlp.gate_proj",0),("mlp.gate_proj",31),("mlp.gate_proj",63),
               ("mlp.down_proj",31),("linear_attn.in_proj_qkv",30)):
    try:
        W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    except Exception as e:
        print(f"{cls}@L{l}: SKIP {e}"); continue
    m,n=W.shape
    X=load_X(l) if n==5120 else None
    if X is None:
        X=np.random.default_rng(1).standard_normal((64,n)).astype(np.float32)
    ref=axes(W, c_uniform(W,4,128), X, seed=0)
    print(f"{cls+'@L'+str(l):<22}{'Q4ref':>6}{'':>7}{4.125:>10.4f}"
          f"{ref['observed']:>10.6f}{ref['probed']:>10.6f}{ref['worst_unit']:>10.6f}{'':>6}")
    for frac in (0.05,0.10,0.20):
        r=int(frac*n)
        t=time.time(); Wh=lowrank(W,r); dt=time.time()-t
        a=axes(W,Wh,X,seed=0)
        print(f"{'':<22}{r:>6}{frac:>7.2f}{bits_shared(r,n):>10.4f}"
              f"{a['observed']:>10.6f}{a['probed']:>10.6f}{a['worst_unit']:>10.6f}{dt:>6.1f}")
