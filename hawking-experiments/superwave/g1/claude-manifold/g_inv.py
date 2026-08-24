"""Is bit allocation on an OUTPUT table inverted relative to a body tensor?

Measured in the forward direction: making the top 2/5/10/25% of lm_head rows by logit
influence rich leaves worst_unit, gain and the protected-class scores BIT-IDENTICAL
(-0.029176 / 0.472461 / 0.574988 / 0.127215 at every fraction). The damage is not where the
influence is. High-influence rows win by margin and are robust; the fragile rows are the ones
influence ranking demotes.

For a body tensor "spend bits where the activation is large" is right. For an output table it
may be exactly backwards, because the table's job is not to produce a large logit, it is to
produce a correctly ORDERED one, and the ordering is decided at the margin.

So: rank by influence and make the BOTTOM fraction rich instead. Same budget accounting,
same held-out split, same four-axis gate plus argmax survival.
"""
import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, axes, c_uniform
from gravity_endpoint_alloc import final_norm_split, bits_per_elem

W=load_tensor("language_model.lm_head.weight").astype(np.float32)
A,B=final_norm_split(); V,H=W.shape
infl=np.abs(A@W.T).mean(0); rank=np.argsort(-infl)
Y=B@W.T
def sc(Wh,label,bpe):
    ax=axes(W,Wh,B,seed=None); Yh=B@Wh.T
    keep=float((Y.argmax(1)==Yh.argmax(1)).mean())
    t5=float(np.mean([len(set(a)&set(b))/5 for a,b in
        zip(np.argpartition(-Y,5,1)[:,:5], np.argpartition(-Yh,5,1)[:,:5])]))
    print(f"  {label:<32}{bpe:>8.4f}{ax['observed']:>10.6f}{ax['probed']:>10.6f}"
          f"{ax['worst_unit']:>10.6f}{ax['gain']:>10.6f}{keep:>9.4f}{t5:>9.4f}")
def two(frac_rich, rb, pb, invert):
    n=int(frac_rich*V); sel = rank[-n:] if invert else rank[:n]
    out=c_uniform(W,pb,128); out[sel]=c_uniform(W[sel],rb,128); return out
print(f"  {'scheme':<32}{'b/elem':>8}{'observed':>10}{'probed':>10}{'worst_u':>10}"
      f"{'gain':>10}{'argmax':>9}{'top5':>9}")
sc(c_uniform(W,4,128),"uniform q4 g128 (reference)",4.125)
sc(c_uniform(W,3,128),"uniform q3 g128",3.125)
for f in (0.10,0.25,0.50,0.75):
    sc(two(f,4,2,True),  f"BOTTOM {100*f:g}% q4, top rest q2", bits_per_elem(f,4,2))
for f in (0.25,0.50):
    sc(two(f,4,3,True),  f"BOTTOM {100*f:g}% q4, top rest q3", bits_per_elem(f,4,3))
