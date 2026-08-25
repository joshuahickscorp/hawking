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
LAYERS=list(range(4,64,4))  # 15 layers spread
NTOK=3000
Xtr,Ytr,Ztr,Xte,Yte,Zte=[],[],[],[],[],[]
eq3s=[]
for zi,L in enumerate(LAYERS):
 pre=f"language_model.model.layers.{L}.mlp"
 Wg=torch.tensor(lw(idx[pre+'.gate_proj.weight'],pre+'.gate_proj.weight'),device=dev);Wu=torch.tensor(lw(idx[pre+'.up_proj.weight'],pre+'.up_proj.weight'),device=dev);Wd=torch.tensor(lw(idx[pre+'.down_proj.weight'],pre+'.down_proj.weight'),device=dev)
 h=torch.tensor(np.frombuffer(open(f"{AC}/post_input_norm/L{L:02d}.f16","rb").read(),dtype=np.float16).astype(np.float32).reshape(-1,5120)[:NTOK],device=dev)
 with torch.no_grad():
  Y=mlp(h,Wg,Wu,Wd); eq3s.append(rel(mlp(h,q3(Wg),q3(Wu),q3(Wd))[NTOK//2:],Y[NTOK//2:]))
 n=h.shape[0];tr=slice(0,n*4//5);te=slice(n*4//5,n)
 Xtr.append(h[tr]);Ytr.append(Y[tr]);Ztr.append(torch.full((h[tr].shape[0],),zi,device=dev))
 Xte.append(h[te]);Yte.append(Y[te]);Zte.append(torch.full((h[te].shape[0],),zi,device=dev))
 del Wg,Wu,Wd
Xtr=torch.cat(Xtr);Ytr=torch.cat(Ytr);Ztr=torch.cat(Ztr).long();Xte=torch.cat(Xte);Yte=torch.cat(Yte);Zte=torch.cat(Zte).long()
print(f"pooled {len(LAYERS)} layers, train {Xtr.shape[0]} held-out {Xte.shape[0]}; mean per-layer q3 held-out {np.mean(eq3s):.4f}",flush=True)
# shared SwiGLU-op + per-layer FiLM (gamma,beta on intermediate)
class Shared(torch.nn.Module):
 def __init__(s,m,nL):
  super().__init__();s.g=torch.nn.Linear(5120,m,bias=False);s.u=torch.nn.Linear(5120,m,bias=False);s.d=torch.nn.Linear(m,5120,bias=False)
  s.gamma=torch.nn.Embedding(nL,m);s.beta=torch.nn.Embedding(nL,m);torch.nn.init.ones_(s.gamma.weight);torch.nn.init.zeros_(s.beta.weight)
 def forward(s,x,z):
  inter=torch.nn.functional.silu(s.g(x))*s.u(x); inter=inter*s.gamma(z)+s.beta(z); return s.d(inter)
for m in (4096,):
 G=Shared(m,len(LAYERS)).to(dev);opt=torch.optim.Adam(G.parameters(),lr=2e-3);t=time.time()
 for it in range(1500):
  opt.zero_grad();l=torch.nn.functional.mse_loss(G(Xtr,Ztr),Ytr);l.backward();opt.step()
 with torch.no_grad():e=rel(G(Xte,Zte),Yte);etr=rel(G(Xtr,Ztr),Ytr)
 shared_mb=3*5120*m*0.40625/1e6; code_mb=len(LAYERS)*2*m*2/1e6
 print(f"SHARED m={m}: held {e:.4f} train {etr:.4f} gap {e-etr:.3f} | shared {shared_mb:.1f}MB + codes {code_mb:.2f}MB, REUSED across {len(LAYERS)} layers = active/layer ~{shared_mb:.1f}MB vs q3 108.6MB ({100*shared_mb/108.6:.0f}%) | {'*** BEATS q3 (generalizes + cache-resident)' if e<=np.mean(eq3s) else 'held-out '+('OK' if e-etr<0.15 else 'GAP')} [{time.time()-t:.0f}s]",flush=True)
