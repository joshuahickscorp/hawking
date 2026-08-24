import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor
from gravity_ir import SOURCE_PARAM_COUNT as N
C="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"

def load_site(site,l,w,n):
    X=np.fromfile(f"{C}/{site}/L{l:02d}.f16",dtype=np.float16).astype(np.float32)
    return X.reshape(-1,w)[:n]

def rowcos(A,B):
    n=(A*B).sum(1); return float(np.mean(n/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-30)))

for l,cls in ((0,"mlp.gate_proj"),(31,"mlp.gate_proj"),(63,"mlp.gate_proj")):
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    X=load_site("post_input_norm",l,5120,6827)
    fit,hold=X[:len(X)//2],X[len(X)//2:]          # disjoint halves
    U,S,Vt=np.linalg.svd(fit,full_matrices=False) # basis from FIT half only
    Y=hold@W.T
    r0=int(np.linalg.matrix_rank(fit,tol=1e-3*np.linalg.norm(fit,2)))
    print(f"\nL{l} {cls}  fit-half rank {r0}")
    print(f"{'r':>6}{'margin':>8}{'held cos':>10}{'coef+basis B':>14}{'BPW@4b':>10}{'vs dense':>9}")
    dense=W.size
    for mult in (1,2,4,8,16):
        r=min(int(r0*mult),5120)
        B=Vt[:r]
        Wr=(W@B.T)@B                              # restrict W to the activation subspace
        c=rowcos(Y, hold@Wr.T)
        stored=W.shape[0]*r + r*W.shape[1]        # coefficients + basis
        print(f"{r:>6}{mult:>7}x{c:>10.6f}{stored:>14,}{8*(stored//2)/N:>10.6f}{dense/stored:>8.1f}x")
