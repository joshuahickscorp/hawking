#!/usr/bin/env python3
# G11/G12 make-or-break: swap EVERY layer's MLP for the trained shared operator in the real MLX
# forward, generate on Doctor probes, compare to the true patient. Tests assembled coherence
# WITHOUT the Metal kernel. Coherence here -> invest in Metal for speed; incoherence -> fix operator first.
import sys,json,numpy as np,torch
import mlx.core as mx
from mlx_lm import load
BF16="workspace/campaign/records/runs/qwen38-27b/bf16"
CKPT=sys.argv[1] if len(sys.argv)>1 else "workspace/campaign/phaseB/ckpt/shared_m6144_K1.pt"
PROBES=[("The capital of France is","Paris"),("17 times 19 equals","323"),
        ("The opposite of hot is","cold"),("Water is made of hydrogen and","oxygen")]

def to_mx(t): return mx.array(t.detach().cpu().numpy().astype(np.float32))
ck=torch.load(CKPT,map_location="cpu"); st=ck["state"]; m=ck["m"]; K=ck.get("K",1)
print(f"loaded {CKPT}: m={m} K={K} held={ck.get('held')} q3={ck.get('q3')}",flush=True)
# shared operator weights (K groups). FiLM per-layer gamma/beta. Rebuild as mx arrays.
def W(name): return to_mx(st[name])
# checkpoints store ModuleList keys g.0/u.0/d.0 for all K (incl K==1)
gW=[W(f"g.{k}.weight") for k in range(K)];uW=[W(f"u.{k}.weight") for k in range(K)];dW=[W(f"d.{k}.weight") for k in range(K)]
gamma=W("gamma.weight");beta=W("beta.weight")  # [nL,m]
grp=[min(l*K//64,K-1) for l in range(64)]
def shared_mlp(x,L):
 k=grp[L]
 inter=(mx.sigmoid(x@gW[k].T)*(x@gW[k].T))*(x@uW[k].T)  # silu(g)*u
 inter=inter*gamma[L]+beta[L]
 return inter@dW[k].T

print("loading MLX bf16 model...",flush=True)
model,tok=load(BF16)
# locate the decoder layers
lm=model
for attr in ("model","language_model"):
 if hasattr(lm,attr): lm=getattr(lm,attr)
layers=lm.layers if hasattr(lm,"layers") else lm.model.layers
print(f"model has {len(layers)} layers; swapping MLPs...",flush=True)

# Robust swap: replace layer.mlp with a class whose CLASS-level __call__ is the shared op
# (Python's mlp(x) dispatches to type(mlp).__call__, not an instance attribute).
class SharedMLP:
 def __init__(s,L): s.L=L
 def __call__(s,x,*a,**k): return shared_mlp(x,s.L)
from mlx_lm import generate as gen
def run():
 out={}
 for p,want in PROBES:
  txt=gen(model,tok,prompt=p,max_tokens=12,verbose=False)
  out[p]={"want":want,"got":txt[:60],"hit":want.lower() in txt.lower()}
 return out
true=run()
# BINDING SANITY: swap ONE layer with a zero op and confirm output changes, else the swap no-ops.
_orig=layers[30].mlp
class _Zero:
 def __call__(s,x,*a,**k): return x*0.0
layers[30].mlp=_Zero()
probe0=gen(model,tok,prompt=PROBES[0][0],max_tokens=6,verbose=False)
layers[30].mlp=_orig
if probe0==true[PROBES[0][0]]["got"][:len(probe0)]:
 print("!! WARNING: zero-op swap did NOT change output -- module replacement is NOT binding. Results invalid.",flush=True)
else:
 print("binding OK: zero-op swap changed output as expected",flush=True)
for i,layer in enumerate(layers):
 layer.mlp=SharedMLP(i)
mx.eval(gamma,beta,*gW,*uW,*dW)
swap=run()
if all(true[p]["got"]==swap[p]["got"] for p,_ in PROBES):
 print("!! WARNING: SWAP identical to TRUE -- swap may not have bound; treat COHERENT with suspicion",flush=True)
print("\n=== ASSEMBLED DOCTOR ===")
hits=0
for p,want in PROBES:
 t=true[p];s=swap[p];hits+=s["hit"]
 print(f"[{p[:28]:28}] want {want:7} | TRUE {'OK' if t['hit'] else 'XX'} {t['got'][:30]!r} | SWAP {'OK' if s['hit'] else 'XX'} {s['got'][:30]!r}")
print(f"\nSWAP coherence: {hits}/{len(PROBES)} probes hit. Assembled patient {'COHERENT' if hits>=3 else 'DEGRADED'}.")
json.dump({"ckpt":CKPT,"true":true,"swap":swap,"swap_hits":hits},open("receipts/ascent-2026-08-18/MLX_ASSEMBLED_DOCTOR.json","w"),indent=1)
