"""MANIFOLD OPERATOR: the shared-input-basis form the envelope actually affords.

Plain SVD of W is the wrong object. The envelope needs a SHARED INPUT BASIS, because that
is the only factorization whose per-site cost is sublinear: the site stores W V_r (m x r),
the basis V_r (n x r) is stored once for the band, so bits/element = r*b/n.

Choose V_r as the top-r principal directions of the REAL captured activations. Then
W_hat x = (W V_r)(V_r^T x) is EXACT for any x inside the captured span and loses only the
off-span component. That is a manifold operator by construction, and the probe axis
measures exactly what it loses off-manifold -- the size of the guard it still owes.

Compared against three things, because a number alone decides nothing:
  Q4 reference   the honest same-tensor cost the gate judges against
  random basis   the null: an r-dim subspace carrying no activation information
  span energy    how much captured energy the basis actually holds
"""
import sys, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, load_X, axes, c_uniform

print(f"{'tensor':<24}{'basis':<12}{'r':>6}{'r/n':>6}{'b/el':>7}{'span E':>9}"
      f"{'observed':>10}{'probed':>10}{'worst_u':>10}", flush=True)
for cls, l in (("mlp.gate_proj",0),("mlp.gate_proj",31),("mlp.gate_proj",63),
               ("linear_attn.in_proj_qkv",30)):
    W = load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    m, n = W.shape
    X = load_X(l)
    if X is None or X.shape[1] != n:
        print(f"{cls}@L{l}: no captured X, skipped", flush=True); continue
    ref = axes(W, c_uniform(W,4,128), X, seed=0)
    print(f"{cls+'@L'+str(l):<24}{'Q4 g128':<12}{'':>6}{'':>6}{4.125:>7.3f}{'':>9}"
          f"{ref['observed']:>10.6f}{ref['probed']:>10.6f}{ref['worst_unit']:>10.6f}", flush=True)
    w, Vfull = np.linalg.eigh((X.T @ X).astype(np.float64))   # ONE eigh per tensor
    Vfull = Vfull[:, ::-1].astype(np.float32)
    Xe = float((X*X).sum())
    for frac in (0.02, 0.05, 0.10, 0.20):
        r = int(frac*n); V = Vfull[:, :r]
        Wh = (W @ V) @ V.T
        a = axes(W, Wh, X, seed=0)
        P = X @ V
        print(f"{'':<24}{'activation':<12}{r:>6}{frac:>6.2f}{r*4/n:>7.3f}"
              f"{float((P*P).sum())/Xe:>9.5f}{a['observed']:>10.6f}{a['probed']:>10.6f}"
              f"{a['worst_unit']:>10.6f}", flush=True)
    r = int(0.10*n)
    rng = np.random.default_rng(0)
    Vr,_ = np.linalg.qr(rng.standard_normal((n, r)).astype(np.float32))
    Wh = (W @ Vr) @ Vr.T
    a = axes(W, Wh, X, seed=0); P = X @ Vr
    print(f"{'':<24}{'random NULL':<12}{r:>6}{0.10:>6.2f}{r*4/n:>7.3f}"
          f"{float((P*P).sum())/Xe:>9.5f}{a['observed']:>10.6f}{a['probed']:>10.6f}"
          f"{a['worst_unit']:>10.6f}", flush=True)
