"""G014 procedural blocks: is a weight PREDICTABLE from a compact rule?

Two independent ceilings, both measured on the real parent tensors:

  1. STRUCTURAL. If a rule generates the matrix, neighbouring entries must be
     statistically dependent. Correlate adjacent columns, adjacent rows and lag-16,
     against a Gaussian null of the same shape. Zero dependence = no rule to find.

  2. INFORMATION. Even a perfect rule cannot beat the empirical entropy of the
     quantized stream. Measured PER GROUP: a global absmax over 4M elements sends
     almost every value to zero and reports a fake ~0 entropy. That artifact is the
     reason this is measured the way it is, and the earlier global-scale figure is
     discarded, not averaged in.
"""
import sys, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor
rng=np.random.default_rng(0)
print(f"{'tensor':<26}{'r|neigh-col':>12}{'r|neigh-row':>12}{'r|lag16':>10}{'gauss null':>12}{'kurt':>7}")
for cls,l in (("mlp.gate_proj",0),("mlp.gate_proj",31),("mlp.down_proj",31),("self_attn.q_proj",31)):
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    S=W[:2048,:2048]
    def corr(a,b):
        a=a.ravel()-a.mean(); b=b.ravel()-b.mean()
        return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
    c_col=corr(S[:,:-1],S[:,1:]); c_row=corr(S[:-1,:],S[1:,:]); c_l16=corr(S[:,:-16],S[:,16:])
    R=rng.standard_normal(S.shape).astype(np.float32)
    null=corr(R[:,:-1],R[:,1:])
    k=float(((S-S.mean())**4).mean()/((S.var()+1e-30)**2))
    print(f"{cls+'@L'+str(l):<26}{c_col:>12.5f}{c_row:>12.5f}{c_l16:>10.5f}{null:>12.5f}{k:>7.3f}")
print()
W=load_tensor("language_model.model.layers.31.mlp.gate_proj.weight").astype(np.float32).ravel()[:4_000_000]
G=128; Wg=W.reshape(-1,G)
for bits in (2,3,4):
    lim=(1<<(bits-1))-1
    s=np.abs(Wg).max(axis=1,keepdims=True)/lim
    q=np.clip(np.round(Wg/np.maximum(s,1e-30)),-lim,lim).astype(np.int32).ravel()
    _,cnt=np.unique(q,return_counts=True); p=cnt/cnt.sum()
    H=float(-(p*np.log2(p)).sum())
    nz=float((q!=0).mean())
    print(f"  q{bits} per-group(g=128): entropy {H:.4f} b vs nominal {bits} -> ideal saves {bits-H:.4f} b/w "
          f"(nonzero {100*nz:.1f}%)")
