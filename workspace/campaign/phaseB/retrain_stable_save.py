#!/usr/bin/env python3
# Stable single-op retrain on diverse data (all families fit), vectorized FiLM (fast), lr 5e-4+clip.
# Save checkpoint for the DEFINITIVE MLX assembled-Doctor test. Best honest operator gets its shot.
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
man=json.load(open(f"{CAP}/manifest.json"));T=man["total_tokens"];split_of=np.empty(T,object)
for m in man["manifest"]: split_of[m["row_start"]:m["row_start"]+m["n_tokens"]]=m["split"]
fitm=np.array([split_of[i]=="fit" for i in range(T)]);holdm=~fitm
X,Y,Z,Xh,Yh,Zh,q3h=[],[],[],[],[],[],[]
for L in range(64):
 pre=f"language_model.model.layers.{L}.mlp"
 Wg=torch.tensor(lw(idx[pre+'.gate_proj.weight'],pre+'.gate_proj.weight'),device=dev);Wu=torch.tensor(lw(idx[pre+'.up_proj.weight'],pre+'.up_proj.weight'),device=dev);Wd=torch.tensor(lw(idx[pre+'.down_proj.weight'],pre+'.down_proj.weight'),device=dev)
 xx=torch.tensor(np.fromfile(f"{CAP}/L{L:02d}.f16",dtype=np.float16).astype(np.float32).reshape(-1,5120),device=dev)
 with torch.no_grad(): yy=mlpf(xx,Wg,Wu,Wd);yq=mlpf(xx,q3(Wg),q3(Wu),q3(Wd))
 mf=torch.tensor(fitm,device=dev);mh=torch.tensor(holdm,device=dev)
 X.append(xx[mf].cpu());Y.append(yy[mf].cpu());Z.append(torch.full((int(mf.sum()),),L))
 Xh.append(xx[mh].cpu());Yh.append(yy[mh].cpu());Zh.append(torch.full((int(mh.sum()),),L));q3h.append(rel(yq[mh],yy[mh]))
 del Wg,Wu,Wd,xx,yy,yq
cat=lambda L:torch.cat(L).to(dev)
X,Y,Z=cat(X),cat(Y),cat(Z).long();Xh,Yh,Zh=cat(Xh),cat(Yh),cat(Zh).long();q3m=float(np.mean(q3h))
print(f"train {X.shape[0]} held(by-prompt,all-fam) {Xh.shape[0]}; q3 held {q3m:.4f}",flush=True)
class Shared(torch.nn.Module):
 def __init__(s,m,nL):
  super().__init__();s.g=torch.nn.Linear(5120,m,bias=False);s.u=torch.nn.Linear(5120,m,bias=False);s.d=torch.nn.Linear(m,5120,bias=False)
  s.gamma=torch.nn.Embedding(nL,m);s.beta=torch.nn.Embedding(nL,m);torch.nn.init.ones_(s.gamma.weight);torch.nn.init.zeros_(s.beta.weight)
 def forward(s,x,z): inter=torch.nn.functional.silu(s.g(x))*s.u(x);inter=inter*s.gamma(z)+s.beta(z);return s.d(inter)
G=Shared(6144,64).to(dev);opt=torch.optim.AdamW(G.parameters(),lr=5e-4,weight_decay=1e-4);t=time.time();best=9;bst=None
for it in range(1800):
 bi=torch.randint(0,X.shape[0],(8192,),device=dev)
 opt.zero_grad();l=torch.nn.functional.mse_loss(G(X[bi],Z[bi]),Y[bi]);l.backward();torch.nn.utils.clip_grad_norm_(G.parameters(),1.0);opt.step()
 if (it+1)%300==0:
  with torch.no_grad(): e=rel(G(Xh,Zh),Yh); tr=rel(G(X[:8000],Z[:8000]),Y[:8000])
  if e<best: best=e;bst={k:v.cpu().clone() for k,v in G.state_dict().items()}
  print(f"  it{it+1} train {tr:.4f} held {e:.4f} best {best:.4f} [{time.time()-t:.0f}s]",flush=True)
st2={"g.0.weight":bst["g.weight"],"u.0.weight":bst["u.weight"],"d.0.weight":bst["d.weight"],"gamma.weight":bst["gamma.weight"],"beta.weight":bst["beta.weight"]}
torch.save({'state':st2,'m':6144,'K':1,'held':best,'q3':q3m,'input':'post_attn_norm','corpus':'diverse2'},"workspace/campaign/phaseB/ckpt/shared_m6144_stable.pt")
print(f"\nSTABLE single-op: held {best:.4f} vs q3 {q3m:.4f} | saved shared_m6144_stable.pt",flush=True)
