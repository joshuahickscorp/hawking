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
LAYERS=list(range(64)); NTOK=2000
Xtr,Ytr,Ztr,Xte,Yte,Zte,eq3s=[],[],[],[],[],[],[]
for zi,L in enumerate(LAYERS):
 pre=f"language_model.model.layers.{L}.mlp"
 try:
  Wg=torch.tensor(lw(idx[pre+'.gate_proj.weight'],pre+'.gate_proj.weight'),device=dev);Wu=torch.tensor(lw(idx[pre+'.up_proj.weight'],pre+'.up_proj.weight'),device=dev);Wd=torch.tensor(lw(idx[pre+'.down_proj.weight'],pre+'.down_proj.weight'),device=dev)
 except KeyError: continue
 raw=open(f"{AC}/post_input_norm/L{L:02d}.f16","rb").read()
 h=torch.tensor(np.frombuffer(raw,dtype=np.float16).astype(np.float32).reshape(-1,5120)[:NTOK],device=dev)
 with torch.no_grad(): Y=mlp(h,Wg,Wu,Wd); eq3s.append(rel(mlp(h,q3(Wg),q3(Wu),q3(Wd))[h.shape[0]*4//5:],Y[h.shape[0]*4//5:]))
 n=h.shape[0];tr=slice(0,n*4//5);te=slice(n*4//5,n)
 Xtr.append(h[tr].cpu());Ytr.append(Y[tr].cpu());Ztr.append(torch.full((h[tr].shape[0],),zi));Xte.append(h[te].cpu());Yte.append(Y[te].cpu());Zte.append(torch.full((h[te].shape[0],),zi))
 del Wg,Wu,Wd,h,Y
nL=len(Xtr)
Xtr=torch.cat(Xtr).to(dev);Ytr=torch.cat(Ytr).to(dev);Ztr=torch.cat(Ztr).long().to(dev);Xte=torch.cat(Xte).to(dev);Yte=torch.cat(Yte).to(dev);Zte=torch.cat(Zte).long().to(dev)
print(f"pooled {nL} layers, train {Xtr.shape[0]} held-out {Xte.shape[0]}; mean q3 held-out {np.mean(eq3s):.4f}",flush=True)
class Shared(torch.nn.Module):
 def __init__(s,m,nL,rank=0):
  super().__init__();s.g=torch.nn.Linear(5120,m,bias=False);s.u=torch.nn.Linear(5120,m,bias=False);s.d=torch.nn.Linear(m,5120,bias=False)
  s.gamma=torch.nn.Embedding(nL,m);s.beta=torch.nn.Embedding(nL,m);torch.nn.init.ones_(s.gamma.weight);torch.nn.init.zeros_(s.beta.weight)
 def forward(s,x,z):
  inter=torch.nn.functional.silu(s.g(x))*s.u(x);inter=inter*s.gamma(z)+s.beta(z);return s.d(inter)
for m in (4096,6144):
 G=Shared(m,nL).to(dev);opt=torch.optim.Adam(G.parameters(),lr=2e-3);t=time.time()
 for it in range(2000):
  bi=torch.randint(0,Xtr.shape[0],(16384,),device=dev)
  opt.zero_grad();l=torch.nn.functional.mse_loss(G(Xtr[bi],Ztr[bi]),Ytr[bi]);l.backward();opt.step()
 with torch.no_grad():e=rel(G(Xte,Zte),Yte);etr=rel(G(Xtr[:9000],Ztr[:9000]),Ytr[:9000])
 smb=3*5120*m*0.40625/1e6;cmb=nL*2*m*2/1e6
 print(f"SHARED-64 m={m}: held {e:.4f} train {etr:.4f} gap {e-etr:.3f} | shared {smb:.1f}MB REUSED across {nL} layers + codes {cmb:.2f}MB | active/layer {smb:.1f}MB vs q3 108.6 ({100*smb/108.6:.0f}%) | {'*** BEATS q3' if e<=np.mean(eq3s) else 'vs q3 '+str(round(np.mean(eq3s),3))} [{time.time()-t:.0f}s]",flush=True)
