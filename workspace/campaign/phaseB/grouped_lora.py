# G6: grouped shared operators + per-layer low-rank LoRA relaxation (literature-backed gap closer).
# K contiguous groups (early/mid/late phases), one SwiGLU op/group, per-layer rank-r LoRA on g/u/d.
import json,numpy as np,struct,torch,time
torch.manual_seed(0)
R="workspace/campaign/records/runs/qwen38-27b/bf16"; AC="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"; dev="mps"
def lw(sh,nm):
 with open(f"{R}/{sh}",'rb') as f:
  hl=struct.unpack('<Q',f.read(8))[0];hd=json.loads(f.read(hl));m=hd[nm];a,b=m['data_offsets'];f.seek(8+hl+a);raw=f.read(b-a)
 return (np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(m['shape'])
idx=json.load(open(f"{R}/model.safetensors.index.json"))["weight_map"]
mlp=lambda x,g,u,d:(torch.nn.functional.silu(x@g.T)*(x@u.T))@d.T
def q3(w,g=64):
 o,i=w.shape;wr=w.reshape(o,i//g,g);am=wr.abs().amax(-1,keepdim=True).clamp_min(1e-9);return (torch.round(wr/am*3).clamp(-3,3)/3*am).reshape(o,i)
rel=lambda a,b:(torch.norm(a-b)/torch.norm(b)).item()
NTOK=2000
Xtr,Ytr,Ztr,Xte,Yte,Zte,eq3s=[],[],[],[],[],[],[]
for L in range(64):
 pre=f"language_model.model.layers.{L}.mlp"
 Wg=torch.tensor(lw(idx[pre+'.gate_proj.weight'],pre+'.gate_proj.weight'),device=dev);Wu=torch.tensor(lw(idx[pre+'.up_proj.weight'],pre+'.up_proj.weight'),device=dev);Wd=torch.tensor(lw(idx[pre+'.down_proj.weight'],pre+'.down_proj.weight'),device=dev)
 raw=open(f"{AC}/post_attn_norm/L{L:02d}.f16","rb").read()
 h=torch.tensor(np.frombuffer(raw,dtype=np.float16).astype(np.float32).reshape(-1,5120)[:NTOK],device=dev)
 with torch.no_grad(): Y=mlp(h,Wg,Wu,Wd); eq3s.append(rel(mlp(h,q3(Wg),q3(Wu),q3(Wd))[h.shape[0]*4//5:],Y[h.shape[0]*4//5:]))
 n=h.shape[0];tr=slice(0,n*4//5);te=slice(n*4//5,n)
 Xtr.append(h[tr].cpu());Ytr.append(Y[tr].cpu());Ztr.append(torch.full((h[tr].shape[0],),L));Xte.append(h[te].cpu());Yte.append(Y[te].cpu());Zte.append(torch.full((h[te].shape[0],),L))
 del Wg,Wu,Wd,h,Y
Xtr=torch.cat(Xtr).to(dev);Ytr=torch.cat(Ytr).to(dev);Ztr=torch.cat(Ztr).long().to(dev);Xte=torch.cat(Xte).to(dev);Yte=torch.cat(Yte).to(dev);Zte=torch.cat(Zte).long().to(dev)
q3m=float(np.mean(eq3s));print(f"train {Xtr.shape[0]} held {Xte.shape[0]}; q3 held {q3m:.4f}",flush=True)
class GroupedLoRA(torch.nn.Module):
 # LoRA is per-LAYER (shared across a layer's tokens), so loop per layer -> one matmul each, no per-sample expansion.
 def __init__(s,m,K,r,nL=64):
  super().__init__();s.K=K;s.r=r;s.m=m;s.nL=nL;s.grp=[min(l*K//nL,K-1) for l in range(nL)]
  s.g=torch.nn.ModuleList([torch.nn.Linear(5120,m,bias=False) for _ in range(K)]);s.u=torch.nn.ModuleList([torch.nn.Linear(5120,m,bias=False) for _ in range(K)]);s.d=torch.nn.ModuleList([torch.nn.Linear(m,5120,bias=False) for _ in range(K)])
  P=torch.nn.Parameter
  # per-layer LoRA factors: gate/up input-side (5120->r->m), down output-side (m->r->5120). B random, A zero (starts pure shared).
  s.gB=P(torch.randn(nL,5120,r)*0.02);s.gA=P(torch.zeros(nL,r,m));s.uB=P(torch.randn(nL,5120,r)*0.02);s.uA=P(torch.zeros(nL,r,m));s.dB=P(torch.randn(nL,m,r)*0.02);s.dA=P(torch.zeros(nL,r,5120))
 def forward(s,x,z):
  out=torch.zeros(x.shape[0],5120,device=x.device)
  zc=z.cpu()
  for L in range(s.nL):
   mask=(z==L)
   if mask.sum()==0: continue
   xk=x[mask];k=s.grp[L]
   g=s.g[k](xk)+(xk@s.gB[L])@s.gA[L];u=s.u[k](xk)+(xk@s.uB[L])@s.uA[L]
   inter=torch.nn.functional.silu(g)*u
   out[mask]=s.d[k](inter)+(inter@s.dB[L])@s.dA[L]
  return out
def train(m,K,r,iters=2500):
 G=GroupedLoRA(m,K,r).to(dev);opt=torch.optim.AdamW(G.parameters(),lr=1.5e-3,weight_decay=2e-4);best=9;t=time.time()
 for it in range(iters):
  bi=torch.randint(0,Xtr.shape[0],(8192,),device=dev)
  opt.zero_grad();l=torch.nn.functional.mse_loss(G(Xtr[bi],Ztr[bi]),Ytr[bi]);l.backward();opt.step()
  if (it+1)%250==0:
   with torch.no_grad(): e=rel(G(Xte,Zte),Yte)
   if e<best: best=e
 with torch.no_grad(): etr=rel(G(Xtr[:9000],Ztr[:9000]),Ytr[:9000])
 shared=K*3*5120*m*0.40625/1e6; lora=64*r*(2*5120+2*m+m+5120)*0.40625/1e6
 print(f"K={K} m={m} r={r}: held {best:.4f} train {etr:.4f} gap {best-etr:.3f} | shared {shared:.1f}MB(resident) + LoRA {lora:.1f}MB ({lora/64:.2f}/layer) | active/token ~{shared+lora:.0f}MB vs q3 6950 | {'BEATS q3' if best<=q3m else f'vs q3 {q3m:.3f}'} [{time.time()-t:.0f}s]",flush=True)
 return best
train(6144,4,32)   # grouped + light LoRA -- predicted winner
train(6144,1,64)   # LoRA only, no grouping -- isolates richer-code effect
train(6144,4,0) if False else None
print("done",flush=True)
