import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X

def block(l):
    return [load_tensor(f"language_model.model.layers.{l}.mlp.{n}_proj.weight") for n in ("gate","up","down")]

def f(W, v):
    a = v @ W[0].T
    return ((a/(1.0+np.exp(-a))) * (v @ W[1].T)) @ W[2].T

rng = np.random.default_rng(0)
layers = [0,7,15,23,31,39,47,55,63]
rows = {}
print(f"{'layer':>6} {'residual_gain':>14} {'block_gain':>11} {'write_gain':>11} {'q_inject':>11}")
for l in layers:
    W = block(l); X = load_X(l); x = X[rng.choice(X.shape[0],16,replace=False)]
    e = rng.standard_normal(x.shape).astype(np.float32)
    e *= 1e-3*np.linalg.norm(x,axis=1,keepdims=True)/np.linalg.norm(e,axis=1,keepdims=True)
    d = f(W,x+e) - f(W,x)
    y = f(W,x)
    rg = float(np.mean(np.linalg.norm(e+d,axis=1)/np.linalg.norm(e,axis=1)))
    bg = float(np.mean(np.linalg.norm(d,axis=1)/np.linalg.norm(e,axis=1)))
    wg = float(np.mean(np.linalg.norm(y,axis=1)/np.linalg.norm(x,axis=1)))
    # q_l: perturbation this block INJECTS into the stream under a fixed relative
    # weight error, normalised by the stream magnitude -- this is what a bit removed
    # from THIS layer actually costs the residual stream
    Wq = [w + (1e-3*np.linalg.norm(w)/np.sqrt(w.size))*rng.standard_normal(w.shape).astype(np.float32) for w in W]
    q = float(np.mean(np.linalg.norm(f(Wq,x)-y,axis=1)/np.linalg.norm(x,axis=1)))
    rows[l] = {"residual_gain":rg,"block_gain":bg,"write_gain":wg,"q_inject":q}
    print(f"{l:>6} {rg:>14.6f} {bg:>11.6f} {wg:>11.6f} {q:>11.3e}")

# held-out: propagate WITH the residual carry across a real span and compare
# against the product of measured residual gains
def span_test(k, j):
    Ws = {l: block(l) for l in range(k, j+1)}
    X = load_X(k); x = X[rng.choice(X.shape[0],16,replace=False)]
    e = rng.standard_normal(x.shape).astype(np.float32)
    e *= 1e-3*np.linalg.norm(x,axis=1,keepdims=True)/np.linalg.norm(e,axis=1,keepdims=True)
    e0 = e.copy(); xc, ec, pred = x, e, 1.0
    for l in range(k, j+1):
        d = f(Ws[l], xc+ec) - f(Ws[l], xc)
        g = float(np.mean(np.linalg.norm(ec+d,axis=1)/np.linalg.norm(ec,axis=1)))
        pred *= g
        xc = xc + f(Ws[l], xc); ec = ec + d
    meas = float(np.mean(np.linalg.norm(ec,axis=1)/np.linalg.norm(e0,axis=1)))
    return pred, meas

print()
print(f"{'span':>10} {'chain_pred':>11} {'measured':>10} {'rel_err':>9} {'uniform':>8} {'unif_err':>9}")
res=[]
for k,j in [(0,7),(15,22),(31,38),(47,54),(56,63)]:
    p_,m_ = span_test(k,j)
    ce=abs(p_-m_)/m_; ue=abs(1.0-m_)/m_
    res.append((ce,ue))
    print(f"{str((k,j)):>10} {p_:>11.6f} {m_:>10.6f} {ce:>9.4f} {1.0:>8.1f} {ue:>9.4f}")
print(f"\nchain worst {max(c for c,_ in res):.4f}  uniform worst {max(u for _,u in res):.4f}")
json.dump({"per_layer":rows}, open('/tmp/g004b.json','w'), indent=2)
