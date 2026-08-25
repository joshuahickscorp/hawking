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
 raw=open(f"{AC}/post_attn_norm/L{L:02d}.f16","rb").read()  # CORRECT MLP input
 h=torch.tensor(np.frombuffer(raw,dtype=np.float16).astype(np.float32).reshape(-1,5120)[:NTOK],device=dev)
 with torch.no_grad(): Y=mlp(h,Wg,Wu,Wd); eq3s.append(rel(mlp(h,q3(Wg),q3(Wu),q3(Wd))[h.shape[0]*4//5:],Y[h.shape[0]*4//5:]))
 n=h.shape[0];tr=slice(0,n*4//5);te=slice(n*4//5,n)
 Xtr.append(h[tr].cpu());Ytr.append(Y[tr].cpu());Ztr.append(torch.full((h[tr].shape[0],),L));Xte.append(h[te].cpu());Yte.append(Y[te].cpu());Zte.append(torch.full((h[te].shape[0],),L))
 del Wg,Wu,Wd,h,Y
Xtr=torch.cat(Xtr).to(dev);Ytr=torch.cat(Ytr).to(dev);Ztr=torch.cat(Ztr).long().to(dev);Xte=torch.cat(Xte).to(dev);Yte=torch.cat(Yte).to(dev);Zte=torch.cat(Zte).long().to(dev)
q3m=float(np.mean(eq3s));print(f"[CORRECT INPUT post_attn_norm] train {Xtr.shape[0]} held {Xte.shape[0]}; q3 held {q3m:.4f}",flush=True)
class Shared(torch.nn.Module):
 def __init__(s,m,nL):
  super().__init__();s.g=torch.nn.Linear(5120,m,bias=False);s.u=torch.nn.Linear(5120,m,bias=False);s.d=torch.nn.Linear(m,5120,bias=False)
  s.gamma=torch.nn.Embedding(nL,m);s.beta=torch.nn.Embedding(nL,m);torch.nn.init.ones_(s.gamma.weight);torch.nn.init.zeros_(s.beta.weight)
 def forward(s,x,z):
  inter=torch.nn.functional.silu(s.g(x))*s.u(x);inter=inter*s.gamma(z)+s.beta(z);return s.d(inter)
G=Shared(6144,64).to(dev);opt=torch.optim.AdamW(G.parameters(),lr=2e-3,weight_decay=2e-4);best=9;bestst=None;t=time.time()
for it in range(2500):
 bi=torch.randint(0,Xtr.shape[0],(16384,),device=dev)
 opt.zero_grad();l=torch.nn.functional.mse_loss(G(Xtr[bi],Ztr[bi]),Ytr[bi]);l.backward();opt.step()
 if (it+1)%150==0:
  with torch.no_grad(): e=rel(G(Xte,Zte),Yte)
  if e<best: best=e;bestst={k:v.cpu().clone() for k,v in G.state_dict().items()}
  if (it+1)%600==0: print(f"  it{it+1} held {e:.4f} best {best:.4f} [{time.time()-t:.0f}s]",flush=True)
with torch.no_grad(): etr=rel(G(Xtr[:9000],Ztr[:9000]),Ytr[:9000])
# store keys in the SAME g.0/u.0/d.0 layout as the K=1 tuned checkpoint (via ModuleList naming) for loader parity
st2={"g.0.weight":bestst["g.weight"],"u.0.weight":bestst["u.weight"],"d.0.weight":bestst["d.weight"],"gamma.weight":bestst["gamma.weight"],"beta.weight":bestst["beta.weight"]}
torch.save({'state':st2,'m':6144,'K':1,'held':best,'q3':q3m,'input':'post_attn_norm'},"workspace/campaign/phaseB/ckpt/shared_m6144_K1_correct.pt")
tag='BEATS q3' if best<=q3m else f'vs q3 {q3m:.3f}'
print(f"\nCORRECTED K=1 m=6144: BEST held {best:.4f} train {etr:.4f} gap {best-etr:.3f} | active 39.9MB vs q3 108.6 (37%) | {tag} [{time.time()-t:.0f}s]",flush=True)
print("SAVED shared_m6144_K1_correct.pt",flush=True)
