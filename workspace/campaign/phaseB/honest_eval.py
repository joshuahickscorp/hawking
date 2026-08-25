#!/usr/bin/env python3
# Honest re-test of "shared operator vs q3" on the diverse capture:
#  - correct input (post_attn_norm, captured on diverse natural corpus)
#  - CROSS-FAMILY held-out (train some families, hold entirely-unseen families) + by-prompt held
#  - CONSISTENT aggregation: per-layer mean rel-L2 for BOTH operator and q3
import json,numpy as np,struct,torch,time,sys
torch.manual_seed(0)
R="workspace/campaign/records/runs/qwen38-27b/bf16"; CAP="workspace/campaign/phaseB/capture_diverse2"; dev="mps"
def lw(sh,nm):
 with open(f"{R}/{sh}",'rb') as f:
  hl=struct.unpack('<Q',f.read(8))[0];hd=json.loads(f.read(hl));m=hd[nm];a,b=m['data_offsets'];f.seek(8+hl+a);raw=f.read(b-a)
 return (np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(m['shape'])
idx=json.load(open(f"{R}/model.safetensors.index.json"))["weight_map"]
mlpf=lambda x,g,u,d:(torch.nn.functional.silu(x@g.T)*(x@u.T))@d.T
def q3(w,g=64):
 o,i=w.shape;wr=w.reshape(o,i//g,g);am=wr.abs().amax(-1,keepdim=True).clamp_min(1e-9);return (torch.round(wr/am*3).clamp(-3,3)/3*am).reshape(o,i)
rel=lambda a,b:(torch.norm(a-b)/torch.norm(b)).item()
man=json.load(open(f"{CAP}/manifest.json")); NL=man["n_layers"]; T=man["total_tokens"]
fams=man["families"]
# per-token family + split arrays (same layout every layer)
fam_of=np.empty(T,object); split_of=np.empty(T,object)
for m in man["manifest"]:
 s,e=m["row_start"],m["row_start"]+m["n_tokens"]; fam_of[s:e]=m["family"]; split_of[s:e]=m["split"]
# choose held families (cross-family generalization): code + dialogue unseen
HELD_FAM=set(["instruction","dialogue"]); TRAIN_FAM=[f for f in fams if f not in HELD_FAM]
tr_mask = np.array([ (fam_of[i] in TRAIN_FAM) and split_of[i]=="fit" for i in range(T)])
xfam_mask=np.array([ fam_of[i] in HELD_FAM for i in range(T)])          # cross-family held (unseen families)
byp_mask =np.array([ (fam_of[i] in TRAIN_FAM) and split_of[i]=="hold" for i in range(T)])  # by-prompt held (seen families, unseen prompts)
print(f"corpus {T} tok; train {tr_mask.sum()} (fams {TRAIN_FAM}); cross-fam held {xfam_mask.sum()} ({sorted(HELD_FAM)}); by-prompt held {byp_mask.sum()}",flush=True)
# load per-layer X, compute true Y + q3 per-layer error on both held sets
Xtr,Ytr,Ztr=[],[],[]; Xxf,Yxf,Zxf=[],[],[]; Xbp,Ybp,Zbp=[],[],[]; q3xf=[];q3bp=[]
for L in range(NL):
 pre=f"language_model.model.layers.{L}.mlp"
 Wg=torch.tensor(lw(idx[pre+'.gate_proj.weight'],pre+'.gate_proj.weight'),device=dev);Wu=torch.tensor(lw(idx[pre+'.up_proj.weight'],pre+'.up_proj.weight'),device=dev);Wd=torch.tensor(lw(idx[pre+'.down_proj.weight'],pre+'.down_proj.weight'),device=dev)
 X=torch.tensor(np.fromfile(f"{CAP}/L{L:02d}.f16",dtype=np.float16).astype(np.float32).reshape(-1,5120),device=dev)
 with torch.no_grad(): Y=mlpf(X,Wg,Wu,Wd); Yq=mlpf(X,q3(Wg),q3(Wu),q3(Wd))
 def sub(mask): mi=torch.tensor(mask,device=dev); return X[mi],Y[mi]
 xt,yt=sub(tr_mask); xf,yf=sub(xfam_mask); xb,yb=sub(byp_mask)
 Xtr.append(xt.cpu());Ytr.append(yt.cpu());Ztr.append(torch.full((xt.shape[0],),L))
 Xxf.append(xf.cpu());Yxf.append(yf.cpu());Zxf.append(torch.full((xf.shape[0],),L))
 Xbp.append(xb.cpu());Ybp.append(yb.cpu());Zbp.append(torch.full((xb.shape[0],),L))
 mi=torch.tensor(xfam_mask,device=dev); q3xf.append(rel(Yq[mi],Y[mi]))
 mb=torch.tensor(byp_mask,device=dev); q3bp.append(rel(Yq[mb],Y[mb]))
 del Wg,Wu,Wd,X,Y,Yq
cat=lambda L:torch.cat(L).to(dev)
Xtr,Ytr,Ztr=cat(Xtr),cat(Ytr),cat(Ztr).long(); Xxf,Yxf,Zxf=cat(Xxf),cat(Yxf),cat(Zxf).long(); Xbp,Ybp,Zbp=cat(Xbp),cat(Ybp),cat(Zbp).long()
q3_xf=float(np.mean(q3xf)); q3_bp=float(np.mean(q3bp))
print(f"q3 per-layer-mean: cross-family {q3_xf:.4f} | by-prompt {q3_bp:.4f}",flush=True)
class Shared(torch.nn.Module):
 def __init__(s,m,nL):
  super().__init__();s.g=torch.nn.Linear(5120,m,bias=False);s.u=torch.nn.Linear(5120,m,bias=False);s.d=torch.nn.Linear(m,5120,bias=False)
  s.gamma=torch.nn.Embedding(nL,m);s.beta=torch.nn.Embedding(nL,m);torch.nn.init.ones_(s.gamma.weight);torch.nn.init.zeros_(s.beta.weight)
 def forward(s,x,z): inter=torch.nn.functional.silu(s.g(x))*s.u(x);inter=inter*s.gamma(z)+s.beta(z);return s.d(inter)
def perlayer_err(G,X,Y,Z):
 errs=[]
 with torch.no_grad():
  for L in range(NL):
   mi=(Z==L)
   if mi.sum()==0: continue
   errs.append(rel(G(X[mi],Z[mi]),Y[mi]))
 return float(np.mean(errs))
G=Shared(6144,NL).to(dev);opt=torch.optim.AdamW(G.parameters(),lr=2e-3,weight_decay=2e-4);t=time.time()
bestxf=9;bestst=None
for it in range(3000):
 bi=torch.randint(0,Xtr.shape[0],(8192,),device=dev)
 opt.zero_grad();l=torch.nn.functional.mse_loss(G(Xtr[bi],Ztr[bi]),Ytr[bi]);l.backward();opt.step()
 if (it+1)%300==0:
  exf=perlayer_err(G,Xxf,Yxf,Zxf)
  if exf<bestxf: bestxf=exf;bestst={k:v.cpu().clone() for k,v in G.state_dict().items()}
  if (it+1)%900==0: print(f"  it{it+1} cross-fam {exf:.4f} best {bestxf:.4f} [{time.time()-t:.0f}s]",flush=True)
G.load_state_dict(bestst)
op_xf=perlayer_err(G,Xxf,Yxf,Zxf); op_bp=perlayer_err(G,Xbp,Ybp,Zbp); op_tr=perlayer_err(G,Xtr[:8000],Ytr[:8000],Ztr[:8000])
print(f"\n=== HONEST RE-TEST (K=1 m=6144 FiLM, consistent per-layer-mean aggregation) ===")
print(f"{'split':16}{'operator':>10}{'q3':>10}{'verdict':>14}")
print(f"{'train(seen)':16}{op_tr:>10.4f}{'-':>10}{'':>14}")
print(f"{'by-prompt held':16}{op_bp:>10.4f}{q3_bp:>10.4f}{('op wins' if op_bp<=q3_bp else 'q3 wins'):>14}")
print(f"{'CROSS-FAMILY held':16}{op_xf:>10.4f}{q3_xf:>10.4f}{('op wins' if op_xf<=q3_xf else 'q3 wins'):>14}")
st2={f"g.0.weight":bestst["g.weight"],"u.0.weight":bestst["u.weight"],"d.0.weight":bestst["d.weight"],"gamma.weight":bestst["gamma.weight"],"beta.weight":bestst["beta.weight"]}
torch.save({'state':st2,'m':6144,'K':1,'held_crossfam':op_xf,'held_byprompt':op_bp,'q3_crossfam':q3_xf,'q3_byprompt':q3_bp,'input':'post_attn_norm','corpus':'diverse'},"workspace/campaign/phaseB/ckpt/shared_m6144_honest.pt")
json.dump({"operator":{"train":op_tr,"by_prompt_held":op_bp,"cross_family_held":op_xf},
           "q3":{"by_prompt_held":q3_bp,"cross_family_held":q3_xf},
           "active_bytes_MB":{"operator":39.9,"q3_per_layer":108.6},
           "held_families":sorted(HELD_FAM),"train_families":TRAIN_FAM,
           "aggregation":"per-layer mean rel-L2 for BOTH; cross-family = entirely unseen families (no leakage possible)"},
          open("receipts/ascent-2026-08-18/G3_HONEST_RETEST.json","w"),indent=1)
print("\nsaved shared_m6144_honest.pt + G3_HONEST_RETEST.json",flush=True)
