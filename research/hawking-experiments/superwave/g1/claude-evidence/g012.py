import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor

def sketch_spectrum(W, k=256, p=768, seed=0):
    """Randomised sketch spectrum. Rotation- and permutation-invariant, so it is a
    lower bound on post-alignment distance without paying for the alignment."""
    rng=np.random.default_rng(seed)
    G=rng.standard_normal((W.shape[1], p)).astype(np.float32)
    Y=W@G
    s=np.linalg.svd(Y, compute_uv=False)[:k]
    return s/(s[0]+1e-30)

def relL2(a,b):
    n=min(len(a),len(b)); a,b=a[:n],b[:n]
    return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30))

CLS = {"mlp.gate_proj":"linear", "mlp.down_proj":"linear", "self_attn.q_proj":"gqa"}
out={}
for cls in CLS:
    layers=[]
    for l in range(64):
        try:
            W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight")
        except KeyError:
            continue
        layers.append((l, sketch_spectrum(W)))
    if len(layers)<4: continue
    n=len(layers)
    D=np.zeros((n,n),dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i!=j: D[i,j]=relL2(layers[i][1], layers[j][1])
    ls=[l for l,_ in layers]
    # null: same-shape random matrices
    rng=np.random.default_rng(0)
    W0=load_tensor(f"language_model.model.layers.{ls[0]}.{cls}.weight")
    r1=sketch_spectrum(rng.standard_normal(W0.shape).astype(np.float32))
    r2=sketch_spectrum(rng.standard_normal(W0.shape).astype(np.float32))
    null=relL2(r1,r2)
    real_vs_rand=np.mean([relL2(s, r1) for _,s in layers])
    off=D[np.triu_indices(n,1)]
    # nearest-neighbour distance per layer, and whether it is an adjacent layer
    nn=[]
    for i in range(n):
        d=D[i].copy(); d[i]=1e9
        j=int(np.argmin(d)); nn.append((ls[i], ls[j], float(D[i,j]), abs(ls[i]-ls[j])))
    adj=sum(1 for _,_,_,g in nn if g<=2)
    out[cls]=dict(n=n, mean=float(off.mean()), min=float(off.min()), max=float(off.max()),
                  null_rand_rand=null, mean_real_vs_rand=float(real_vs_rand),
                  nn_adjacent_frac=adj/n)
    print(f"{cls:<20} n={n:>2}  pairwise relL2 mean {off.mean():.4f} [{off.min():.4f},{off.max():.4f}]")
    print(f"{'':<20} real-vs-random {real_vs_rand:.4f}   random-vs-random {null:.4f}")
    print(f"{'':<20} nearest neighbour is within 2 layers for {adj}/{n} layers")
print()
for cls,v in out.items():
    sep = v["mean_real_vs_rand"]/max(v["mean"],1e-9)
    print(f"{cls:<20} real pairs are {sep:.1f}x closer to each other than to random")
json.dump(out,open('/tmp/g012.json','w'),indent=2)
