import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X, axes, c_uniform
from gravity_ir import quant_tensor, SOURCE_PARAM_COUNT as N

def plane(R, X, bits, group):
    """One structured plane fitted to the ACTIVATION-WEIGHTED residual (G010 says the
    objective matters most exactly here, at low bits)."""
    lim=(1<<(bits-1))-1
    P=np.zeros_like(R); d=R.shape[1]
    e=np.sqrt((X*X).mean(0))+1e-30
    for s in range(0,d-d%group,group):
        blk=R[:,s:s+group]; w=e[s:s+group]
        amax=np.abs(blk).max(axis=1,keepdims=True)+1e-30
        best=None
        for f in (0.6,0.75,0.9,1.0):
            step=(amax*f)/lim
            q=np.clip(np.round(blk/step),-lim,lim)*step
            err=(((blk-q)*w)**2).sum()
            if best is None or err<best[0]: best=(err,q)
        P[:,s:s+group]=best[1]
    return P

def bpw(elements, terms):
    """terms = [(bits, group)] ; all scales and headers counted."""
    return 8*sum(quant_tensor(elements,b,g,"x").stored_bytes for b,g in terms)/elements

cases=[("mlp.gate_proj",31),("mlp.down_proj",31),("self_attn.q_proj",31)]
G=128
print(f"{'tensor':<24}{'scheme':<22}{'bits/w':>8}{'gate_heldout':>14}")
rows=[]
for cls,l in cases:
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    Xa=load_X(l); n=Xa.shape[0]
    if Xa.shape[1]!=W.shape[1]:
        rng=np.random.default_rng(0); Xa=rng.standard_normal((256,W.shape[1])).astype(np.float32)
        n=256
    fit,hold=Xa[:n//2],Xa[n//2:]
    E=W.size
    def score(Wh): 
        a=axes(W,Wh,hold); return min(a["observed"],a["probed"],a["worst_unit"])
    # flat baselines
    for b in (2,3,4):
        Wh=c_uniform(W,b,G)
        rows.append((cls,l,f"flat q{b}",bpw(E,[(b,G)]),score(Wh)))
    # ladder: 1-bit base + successive residual planes
    acc=np.zeros_like(W); terms=[]
    for i,b in enumerate((1,1,1)):
        P=plane(W-acc, fit, max(b,2) if b==1 else b, G)   # 1-bit uses 2-level symmetric
        if b==1:
            R=W-acc
            amax=np.abs(R).reshape(R.shape[0],-1)
            P=np.sign(R)*np.abs(R).mean()
            # per-group scale for the sign plane
            P=np.zeros_like(R)
            for s in range(0,R.shape[1]-R.shape[1]%G,G):
                blk=R[:,s:s+G]
                P[:,s:s+G]=np.sign(blk)*np.abs(blk).mean(axis=1,keepdims=True)
        acc=acc+P; terms.append((1,G))
        rows.append((cls,l,f"ladder {i+1}x1bit",bpw(E,terms),score(acc)))
for cls,l,name,b,g in rows:
    print(f"{cls+'@L'+str(l):<24}{name:<22}{b:>8.4f}{g:>14.6f}")
print()
# head-to-head at matched bits
import collections
byt=collections.defaultdict(list)
for cls,l,name,b,g in rows: byt[(cls,l)].append((name,b,g))
for k,v in byt.items():
    flats=[(n,b,g) for n,b,g in v if n.startswith("flat")]
    lads =[(n,b,g) for n,b,g in v if n.startswith("ladder")]
    print(f"{k[0]}@L{k[1]}")
    for ln,lb,lg in lads:
        near=min(flats,key=lambda f:abs(f[1]-lb))
        verdict = "LADDER WINS" if lg>near[2] else "flat wins"
        print(f"  {ln:<16} {lb:.4f}b {lg:.6f}   vs {near[0]:<8} {near[1]:.4f}b {near[2]:.6f}   {verdict}")
json.dump([{"cls":c,"layer":l,"scheme":n,"bpw":b,"gate":g} for c,l,n,b,g in rows],open('/tmp/g015.json','w'),indent=2)
