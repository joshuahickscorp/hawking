import sys, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X, c_uniform, _probe
G=128
def mlp(g,u,d,x):
    a=x@g.T; return ((a/(1.0+np.exp(-a)))*(x@u.T))@d.T
def rowcos(A,B):
    n=(A*B).sum(1); return float(np.mean(n/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-30)))

for l in (31,47):
    g=load_tensor(f"language_model.model.layers.{l}.mlp.gate_proj.weight").astype(np.float32)
    u=load_tensor(f"language_model.model.layers.{l}.mlp.up_proj.weight").astype(np.float32)
    d=load_tensor(f"language_model.model.layers.{l}.mlp.down_proj.weight").astype(np.float32)
    X=load_X(l); fit,hold=X[:X.shape[0]//2],X[X.shape[0]//2:]
    Y=mlp(g,u,d,hold)
    print(f"\nL{l} MLP block, bits=3 group={G}")
    for bits in (2,3):
        gq,uq,dq=c_uniform(g,bits,G),c_uniform(u,bits,G),c_uniform(d,bits,G)
        ind=rowcos(Y, mlp(gq,uq,dq,hold))
        # JOINT: refit down against the ALREADY-QUANTIZED gate/up hidden, so down absorbs
        # their error instead of approximating its own source matrix
        H  = (lambda a: (a/(1.0+np.exp(-a))))(fit@g.T)*(fit@u.T)     # teacher hidden
        Hq = (lambda a: (a/(1.0+np.exp(-a))))(fit@gq.T)*(fit@uq.T)   # student hidden
        Yt = H@d.T
        # least squares d' minimising ||Hq d'^T - Yt|| then quantize d'
        A = Hq; B = Yt
        dprime = np.linalg.lstsq(A, B, rcond=1e-3)[0].T.astype(np.float32)
        dpq = c_uniform(dprime, bits, G)
        jnt = rowcos(Y, mlp(gq,uq,dpq,hold))
        print(f"  q{bits}: independent {ind:.6f}   joint(down refit) {jnt:.6f}   delta {jnt-ind:+.6f}")
