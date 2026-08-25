#!/usr/bin/env python3
# Honest cross-family test of GROUPED operators + per-layer LoRA (survey-predicted winner).
# Same honest pipeline as honest_eval.py (diverse capture, cross-family held, per-layer-mean agg).
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
man=json.load(open(f"{CAP}/manifest.json")); NL=man["n_layers"]; T=man["total_tokens"]; fams=man["families"]
fam_of=np.empty(T,object);split_of=np.empty(T,object)
for m in man["manifest"]:
 s,e=m["row_start"],m["row_start"]+m["n_tokens"];fam_of[s:e]=m["family"];split_of[s:e]=m["split"]
HELD_FAM=set(["instruction","dialogue"]);TRAIN_FAM=[f for f in fams if f not in HELD_FAM]
tr_mask=np.array([(fam_of[i] in TRAIN_FAM) and split_of[i]=="fit" for i in range(T)])
xf_mask=np.array([fam_of[i] in HELD_FAM for i in range(T)])
Xtr,Ytr,Ztr,Xxf,Yxf,Zxf,q3xf=[],[],[],[],[],[],[]
for L in range(NL):
 pre=f"language_model.model.layers.{L}.mlp"
 Wg=torch.tensor(lw(idx[pre+'.gate_proj.weight'],pre+'.gate_proj.weight'),device=dev);Wu=torch.tensor(lw(idx[pre+'.up_proj.weight'],pre+'.up_proj.weight'),device=dev);Wd=torch.tensor(lw(idx[pre+'.down_proj.weight'],pre+'.down_proj.weight'),device=dev)
 X=torch.tensor(np.fromfile(f"{CAP}/L{L:02d}.f16",dtype=np.float16).astype(np.float32).reshape(-1,5120),device=dev)
 with torch.no_grad(): Y=mlpf(X,Wg,Wu,Wd);Yq=mlpf(X,q3(Wg),q3(Wu),q3(Wd))
 mt=torch.tensor(tr_mask,device=dev);mx_=torch.tensor(xf_mask,device=dev)
 Xtr.append(X[mt].cpu());Ytr.append(Y[mt].cpu());Ztr.append(torch.full((int(mt.sum()),),L))
 Xxf.append(X[mx_].cpu());Yxf.append(Y[mx_].cpu());Zxf.append(torch.full((int(mx_.sum()),),L))
 q3xf.append(rel(Yq[mx_],Y[mx_]));del Wg,Wu,Wd,X,Y,Yq
cat=lambda L:torch.cat(L).to(dev)
Xtr,Ytr,Ztr=cat(Xtr),cat(Ytr),cat(Ztr).long();Xxf,Yxf,Zxf=cat(Xxf),cat(Yxf),cat(Zxf).long()
q3_xf=float(np.mean(q3xf));print(f"train {Xtr.shape[0]} cross-fam held {Xxf.shape[0]} ({sorted(HELD_FAM)}); q3 cross-fam {q3_xf:.4f}",flush=True)
class GroupedLoRA(torch.nn.Module):
 def __init__(s,m,K,r,nL):
  super().__init__();s.K=K;s.r=r;s.m=m;s.nL=nL;s.grp=[min(l*K//nL,K-1) for l in range(nL)]
  s.g=torch.nn.ModuleList([torch.nn.Linear(5120,m,bias=False) for _ in range(K)]);s.u=torch.nn.ModuleList([torch.nn.Linear(5120,m,bias=False) for _ in range(K)]);s.d=torch.nn.ModuleList([torch.nn.Linear(m,5120,bias=False) for _ in range(K)])
  P=torch.nn.Parameter
  s.gB=P(torch.randn(nL,5120,r)*0.02);s.gA=P(torch.zeros(nL,r,m));s.uB=P(torch.randn(nL,5120,r)*0.02);s.uA=P(torch.zeros(nL,r,m));s.dB=P(torch.randn(nL,m,r)*0.02);s.dA=P(torch.zeros(nL,r,5120))
  s.gm=torch.nn.Embedding(nL,m);s.bt=torch.nn.Embedding(nL,m);torch.nn.init.ones_(s.gm.weight);torch.nn.init.zeros_(s.bt.weight)
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
def run(K,r):
 G=GroupedLoRA(6144,K,r,NL).to(dev);opt=torch.optim.AdamW(G.parameters(),lr=1.5e-3,weight_decay=2e-4);best=9;bst=None;t=time.time()
 for it in range(3000):
  bi=torch.randint(0,Xtr.shape[0],(8192,),device=dev)
  opt.zero_grad();l=torch.nn.functional.mse_loss(G(Xtr[bi],Ztr[bi]),Ytr[bi]);l.backward();opt.step()
  if (it+1)%300==0:
   e=perlayer(G,Xxf,Yxf,Zxf)
   if e<best: best=e;bst={k:v.cpu().clone() for k,v in G.state_dict().items()}
 shared=K*3*5120*6144*0.40625/1e6; lora=NL*r*(2*5120+2*6144+6144+5120)*0.40625/1e6; film=NL*2*6144*2/1e6
 print(f"K={K} r={r}: CROSS-FAM held {best:.4f} vs q3 {q3_xf:.4f} {'*** op wins' if best<=q3_xf else 'q3 wins'} | active {shared+lora+film:.0f}MB (shared {shared:.0f} resident + LoRA {lora:.0f} + film {film:.1f}) vs q3 6950 [{time.time()-t:.0f}s]",flush=True)
 return best
run(4,32)   # grouped-4 + rank-32 LoRA
run(2,48)   # grouped-2 + rank-48
print("done",flush=True)
