import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X, axes, c_uniform

def q_absmax(W, bits, group):
    """weight-space objective: minimise ||W - What||"""
    return c_uniform(W, bits, group)

def q_funcfit(W, X, bits, group):
    """function-space objective: per-group scale minimising ||X (W - What)|| on a FIT split."""
    lim=(1<<(bits-1))-1
    Wh=W.astype(np.float32).copy(); d=W.shape[1]
    # per-input-channel activation energy from the fit split only
    e=np.sqrt((X*X).mean(0))+1e-30
    for s in range(0, d-d%group, group):
        blk=W[:, s:s+group]; w=e[s:s+group]
        amax=np.abs(blk).max(axis=1,keepdims=True)+1e-30
        best=None
        for f in np.linspace(0.55,1.0,10):          # clip fraction search
            step=(amax*f)/lim
            q=np.clip(np.round(blk/step),-lim,lim)*step
            err=(((blk-q)*w)**2).sum()              # activation-weighted, not raw
            if best is None or err<best[0]: best=(err,q)
        Wh[:, s:s+group]=best[1]
    return Wh

rows=[]
cases=[("mlp.gate_proj",0),("mlp.gate_proj",31),("mlp.up_proj",47),
       ("self_attn.q_proj",31),("self_attn.v_proj",63)]
print(f"{'tensor':<26}{'bits':>5}{'objective':>11}{'gate_heldout':>14}{'observed':>10}{'probed':>9}{'worst_u':>9}")
for cls,l in cases:
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight")
    X=load_X(l)
    n=X.shape[0]; fit,hold = X[:n//2], X[n//2:]     # held-out split, never scored on fit
    for bits in (3,4):
        for name,Wh in (("weight-space", q_absmax(W,bits,128)),
                        ("function",     q_funcfit(W,fit,bits,128))):
            a=axes(W,Wh,hold)
            g=min(a["observed"],a["probed"],a["worst_unit"])
            rows.append(dict(cls=cls,layer=l,bits=bits,obj=name,gate=g,**a))
            print(f"{cls+'@L'+str(l):<26}{bits:>5}{name:>11}{g:>14.6f}{a['observed']:>10.6f}{a['probed']:>9.6f}{a['worst_unit']:>9.6f}")
print()
import collections
d=collections.defaultdict(dict)
for r in rows: d[(r['cls'],r['layer'],r['bits'])][r['obj']]=r['gate']
wins=sum(1 for v in d.values() if v['function']>v['weight-space'])
print(f"function objective wins {wins}/{len(d)} cells at identical bit width")
for k,v in d.items():
    print(f"  {k[0]}@L{k[1]} {k[2]}b: weight {v['weight-space']:.6f} -> function {v['function']:.6f}  delta {v['function']-v['weight-space']:+.6f}")
json.dump(rows,open('/tmp/g010.json','w'),indent=2)
