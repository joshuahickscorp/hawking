"""MANIFOLD OPERATOR vs REAL HELD-OUT TEXT -- thick capture, prompt-level holdout.

Setup that makes the answer mean something: activation-capture-v2, 23,216 tokens across 69
prompts in 7 classes (prose, code, math, instruction, multilingual, adversarial, long),
6,827 rows stored per layer, split BY PROMPT into 5,120 fit and 1,707 hold. Row shuffling is
refused by the capture itself, so the holdout is genuinely unseen text and not a reshuffle of
seen text.

The operator W V_r V_r^T with V_r the top-r activation directions is exact on what it was
fitted to. The only question that decides the family:

    does real held-out text visit directions outside the fitted span?

Reported per rank: fit-span energy, held-span energy, observed on fit, observed on HELD, and
the same-rank random-basis null on held. Q4's own held-out score is the bar, because the gate
judges against a same-tensor honest Q4 and not against 1.0.
"""
import sys, json, numpy as np
sys.path.insert(0,'tools')
from gravity_doctor_gate import load_tensor, _rowcos, c_uniform

CAP="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
res=json.load(open(f"{CAP}/capture-result.json"))
site=res["sites"]["post_input_norm"]
NF, NH = site["n_fit"], site["n_hold"]
print(f"thick capture: {res['n_tokens']} tokens, {len(res['prompts'])} prompts, "
      f"{site['store_n']} rows/layer, fit {NF} / hold {NH}, holdout_by_prompt="
      f"{site['holdout_by_prompt']}")

def X_of(layer):
    p=site["per_layer"][str(layer)]
    return np.fromfile(p["path"],dtype=np.float16).reshape(p["n_rows"],p["width"]).astype(np.float32)

print(f"\n{'tensor':<24}{'r':>6}{'b/el':>7}{'fitE':>9}{'heldE':>9}"
      f"{'obs fit':>10}{'obs HELD':>10}{'null HELD':>10}{'Q4 HELD':>9}")
for cls,l in (("mlp.gate_proj",0),("mlp.gate_proj",15),("mlp.gate_proj",31),
              ("mlp.gate_proj",63),("linear_attn.in_proj_qkv",30),("mlp.up_proj",47)):
    W=load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    n=W.shape[1]
    X=X_of(l)
    if X.shape[1]!=n:
        print(f"{cls}@L{l}: width {X.shape[1]} != n {n}, skipped"); continue
    A,B=X[:NF],X[NF:NF+NH]
    q4=_rowcos(B@W.T, B@c_uniform(W,4,128).T)
    w,V=np.linalg.eigh((A.T@A).astype(np.float64)); V=V[:,::-1].astype(np.float32)
    rank=int((w[::-1] > w.max()*1e-10).sum())
    rng=np.random.default_rng(0)
    eA0=float((A*A).sum()); eB0=float((B*B).sum())
    first=True
    for r in (64,128,256,512,1024,2048):
        Vr=V[:,:r]; Wh=(W@Vr)@Vr.T
        Q,_=np.linalg.qr(rng.standard_normal((n,r)).astype(np.float32))
        Wn=(W@Q)@Q.T
        print(f"{(cls+'@L'+str(l)+f' rk{rank}') if first else '':<24}{r:>6}{r*4/n:>7.3f}"
              f"{float(((A@Vr)**2).sum())/eA0:>9.6f}{float(((B@Vr)**2).sum())/eB0:>9.6f}"
              f"{_rowcos(A@W.T,A@Wh.T):>10.6f}{_rowcos(B@W.T,B@Wh.T):>10.6f}"
              f"{_rowcos(B@W.T,B@Wn.T):>10.6f}{q4 if first else 0:>9.6f}")
        first=False
