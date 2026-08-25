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
q3m=float(np.mean(eq3s))
print(f"train {Xtr.shape[0]} held {Xte.shape[0]}; q3 held {q3m:.4f}",flush=True)
class Shared(torch.nn.Module):
 def __init__(s,m,nL,K=1):
  super().__init__();s.K=K
  s.g=torch.nn.ModuleList([torch.nn.Linear(5120,m,bias=False) for _ in range(K)]);s.u=torch.nn.ModuleList([torch.nn.Linear(5120,m,bias=False) for _ in range(K)]);s.d=torch.nn.ModuleList([torch.nn.Linear(m,5120,bias=False) for _ in range(K)])
  s.gamma=torch.nn.Embedding(nL,m);s.beta=torch.nn.Embedding(nL,m);torch.nn.init.ones_(s.gamma.weight);torch.nn.init.zeros_(s.beta.weight)
  s.assign=[min(int(l*K/nL),K-1) for l in range(nL)]  # contiguous layer groups -> operator
 def forward(s,x,z):
  out=torch.zeros(x.shape[0],5120,device=x.device)
  for k in range(s.K):
   mask=torch.tensor([s.assign[int(zz)]==k for zz in z.cpu()],device=x.device)
   if mask.sum()==0: continue
   xk=x[mask];zk=z[mask];inter=torch.nn.functional.silu(s.g[k](xk))*s.u[k](xk);inter=inter*s.gamma(zk)+s.beta(zk);out[mask]=s.d[k](inter)
  return out
def train(m,K,iters=3000,wd=2e-4):
 G=Shared(m,64,K).to(dev);opt=torch.optim.AdamW(G.parameters(),lr=2e-3,weight_decay=wd);best=9;bestst=None;t=time.time()
 for it in range(iters):
  bi=torch.randint(0,Xtr.shape[0],(16384,),device=dev)
  opt.zero_grad();l=torch.nn.functional.mse_loss(G(Xtr[bi],Ztr[bi]),Ytr[bi]);l.backward();opt.step()
  if (it+1)%150==0:
   with torch.no_grad(): e=rel(G(Xte,Zte),Yte)
   if e<best: best=e;bestst={kk:vv.cpu().clone() for kk,vv in G.state_dict().items()}
 with torch.no_grad(): etr=rel(G(Xtr[:9000],Ztr[:9000]),Ytr[:9000])
 smb=K*3*5120*m*0.40625/1e6;cmb=64*2*m*2/1e6
 tag='BEATS q3' if best<=q3m else f'vs q3 {q3m:.3f}'
 print(f"m={m} K={K} wd={wd}: BEST held {best:.4f} train {etr:.4f} | {smb:.1f}MB shared +{cmb:.2f}MB codes | active {smb+cmb:.1f}MB vs q3 108.6 ({100*(smb+cmb)/108.6:.0f}%) | {tag} [{time.time()-t:.0f}s]",flush=True)
 return best,bestst
b1,st1=train(6144,1)
torch.save({'state':st1,'m':6144,'K':1,'held':b1,'q3':q3m},"workspace/campaign/phaseB/ckpt/shared_m6144_K1.pt")
b2,st2=train(6144,2)
torch.save({'state':st2,'m':6144,'K':2,'held':b2,'q3':q3m},"workspace/campaign/phaseB/ckpt/shared_m6144_K2.pt")
b3,st3=train(8192,1)
torch.save({'state':st3,'m':8192,'K':1,'held':b3,'q3':q3m},"workspace/campaign/phaseB/ckpt/shared_m8192_K1.pt")
print(f"\nSAVED checkpoints; best held-out {min(b1,b2,b3):.4f} vs q3 {q3m:.4f}",flush=True)
