#!/usr/bin/env python3
# Decisive FAIR density test: max-capacity grouped-8 + rank-64 LoRA, BALANCED family sampling
# (kills code-dominance confound), report by-prompt held (all families) + cross-family held.
import json,numpy as np,struct,torch,time
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
man=json.load(open(f"{CAP}/manifest.json"));T=man["total_tokens"];fams=man["families"]
fam_of=np.empty(T,object);split_of=np.empty(T,object)
for m in man["manifest"]: fam_of[m["row_start"]:m["row_start"]+m["n_tokens"]]=m["family"];split_of[m["row_start"]:m["row_start"]+m["n_tokens"]]=m["split"]
HELD_FAM=set(["instruction","dialogue"]);TRAIN_FAM=[f for f in fams if f not in HELD_FAM]
trm=np.array([(fam_of[i] in TRAIN_FAM) and split_of[i]=="fit" for i in range(T)])
bpm=np.array([(fam_of[i] in TRAIN_FAM) and split_of[i]=="hold" for i in range(T)])
xfm=np.array([fam_of[i] in HELD_FAM for i in range(T)])
trfam=fam_of[trm]  # family per train row (same every layer)
NL=64
Xtr,Ytr,Ztr,Xbp,Ybp,Zbp,Xxf,Yxf,Zxf,q3bp,q3xf=[],[],[],[],[],[],[],[],[],[],[]
for L in range(NL):
 pre=f"language_model.model.layers.{L}.mlp"
 Wg=torch.tensor(lw(idx[pre+'.gate_proj.weight'],pre+'.gate_proj.weight'),device=dev);Wu=torch.tensor(lw(idx[pre+'.up_proj.weight'],pre+'.up_proj.weight'),device=dev);Wd=torch.tensor(lw(idx[pre+'.down_proj.weight'],pre+'.down_proj.weight'),device=dev)
 X=torch.tensor(np.fromfile(f"{CAP}/L{L:02d}.f16",dtype=np.float16).astype(np.float32).reshape(-1,5120),device=dev)
 with torch.no_grad(): Y=mlpf(X,Wg,Wu,Wd);Yq=mlpf(X,q3(Wg),q3(Wu),q3(Wd))
 mt=torch.tensor(trm,device=dev);mb=torch.tensor(bpm,device=dev);mx_=torch.tensor(xfm,device=dev)
 Xtr.append(X[mt].cpu());Ytr.append(Y[mt].cpu());Ztr.append(torch.full((int(mt.sum()),),L))
 Xbp.append(X[mb].cpu());Ybp.append(Y[mb].cpu());Zbp.append(torch.full((int(mb.sum()),),L))
 Xxf.append(X[mx_].cpu());Yxf.append(Y[mx_].cpu());Zxf.append(torch.full((int(mx_.sum()),),L))
 q3bp.append(rel(Yq[mb],Y[mb]));q3xf.append(rel(Yq[mx_],Y[mx_]));del Wg,Wu,Wd,X,Y,Yq
cat=lambda L:torch.cat(L).to(dev)
Xtr,Ytr,Ztr=cat(Xtr),cat(Ytr),cat(Ztr).long();Xbp,Ybp,Zbp=cat(Xbp),cat(Ybp),cat(Zbp).long();Xxf,Yxf,Zxf=cat(Xxf),cat(Yxf),cat(Zxf).long()
q3_bp=float(np.mean(q3bp));q3_xf=float(np.mean(q3xf))
# balanced sampling: per-(layer,family) index groups. train rows are ordered [layer0 all trfam, layer1 ...]
ntr=trm.sum()  # per-layer train count
fam_idx={}  # family -> tensor of global train-row indices (across layers)
for f in TRAIN_FAM:
 base=np.where(trfam==f)[0]  # positions within a layer's train block
 allidx=np.concatenate([base+L*ntr for L in range(NL)])
 fam_idx[f]=torch.tensor(allidx,device=dev)
print(f"train {Xtr.shape[0]} ({ntr}/layer) fams {TRAIN_FAM}; by-prompt held {Xbp.shape[0]}; cross-fam held {Xxf.shape[0]}",flush=True)
print(f"q3 bars: by-prompt {q3_bp:.4f} | cross-family {q3_xf:.4f}",flush=True)
class Op(torch.nn.Module):
 def __init__(s,m,K,r,nL=64):
  super().__init__();s.K=K;s.r=r;s.m=m;s.nL=nL;s.grp=[min(l*K//nL,K-1) for l in range(nL)]
  s.g=torch.nn.ModuleList([torch.nn.Linear(5120,m,bias=False) for _ in range(K)]);s.u=torch.nn.ModuleList([torch.nn.Linear(5120,m,bias=False) for _ in range(K)]);s.d=torch.nn.ModuleList([torch.nn.Linear(m,5120,bias=False) for _ in range(K)])
  s.gm=torch.nn.Embedding(nL,m);s.bt=torch.nn.Embedding(nL,m);torch.nn.init.ones_(s.gm.weight);torch.nn.init.zeros_(s.bt.weight)
  P=torch.nn.Parameter
  s.gB=P(torch.randn(nL,5120,r)*0.02);s.gA=P(torch.zeros(nL,r,m));s.uB=P(torch.randn(nL,5120,r)*0.02);s.uA=P(torch.zeros(nL,r,m));s.dB=P(torch.randn(nL,m,r)*0.02);s.dA=P(torch.zeros(nL,r,5120))
 def forward(s,x,z):
  out=torch.zeros(x.shape[0],5120,device=x.device)
  for L in range(s.nL):
   mi=(z==L)
   if mi.sum()==0: continue
   xk=x[mi];k=s.grp[L]
   g=s.g[k](xk)+(xk@s.gB[L])@s.gA[L];u=s.u[k](xk)+(xk@s.uB[L])@s.uA[L]
   inter=torch.nn.functional.silu(g)*u; inter=inter*s.gm.weight[L]+s.bt.weight[L]
   out[mi]=s.d[k](inter)+(inter@s.dB[L])@s.dA[L]
  return out
def perlayer(G,X,Y,Z):
 e=[]
 with torch.no_grad():
  for L in range(NL):
   mi=(Z==L)
   if mi.sum()>0: e.append(rel(G(X[mi],Z[mi]),Y[mi]))
 return float(np.mean(e))
m,K,r=6144,8,64
G=Op(m,K,r).to(dev);opt=torch.optim.AdamW(G.parameters(),lr=5e-4,weight_decay=1e-4);t=time.time()
per=1024
for it in range(1800):
 # balanced batch: equal rows per family
 bi=torch.cat([fi[torch.randint(0,len(fi),(per,),device=dev)] for fi in fam_idx.values()])
 opt.zero_grad();l=torch.nn.functional.mse_loss(G(Xtr[bi],Ztr[bi]),Ytr[bi]);l.backward()
 torch.nn.utils.clip_grad_norm_(G.parameters(),1.0);opt.step()
 if (it+1)%600==0:
  bp=perlayer(G,Xbp,Ybp,Zbp);xf=perlayer(G,Xxf,Yxf,Zxf)
  print(f"  it{it+1} by-prompt {bp:.4f} cross-fam {xf:.4f} [{time.time()-t:.0f}s]",flush=True)
bp=perlayer(G,Xbp,Ybp,Zbp);xf=perlayer(G,Xxf,Yxf,Zxf)
shared=K*3*5120*m*0.40625/1e6;lora=NL*r*(2*5120+2*m+m+5120)*0.40625/1e6;film=NL*2*m*2/1e6;act=shared+lora+film
print(f"\n=== DECISIVE DENSITY (grouped K=8 rank-64 LoRA, balanced) ===")
print(f"by-prompt held : op {bp:.4f}  q3 {q3_bp:.4f}  {'OP OK' if bp<=q3_bp else 'q3 wins'}")
print(f"cross-family   : op {xf:.4f}  q3 {q3_xf:.4f}  {'OP OK' if xf<=q3_xf else 'q3 wins'}")
print(f"active {act:.0f}MB (shared {shared:.0f} resident + LoRA {lora:.0f} + film {film:.1f}) vs q3 full-stack 6950MB = {100*act/6950:.1f}%")
json.dump({"config":"grouped K=8 r=64 balanced","by_prompt":{"op":bp,"q3":q3_bp},"cross_family":{"op":xf,"q3":q3_xf},"active_MB":act,"q3_fullstack_MB":6950},open("receipts/ascent-2026-08-18/DECISIVE_DENSITY.json","w"),indent=1)
print("saved DECISIVE_DENSITY.json",flush=True)
