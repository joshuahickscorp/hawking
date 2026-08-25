import json, numpy as np, struct, torch, time, sys
torch.manual_seed(0)
R="workspace/campaign/records/runs/qwen38-27b/bf16"
AC="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
dev="mps"
def load_w(shard,name):
    with open(f"{R}/{shard}",'rb') as f:
        hlen=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(hlen))
        m=hdr[name]; a,b=m['data_offsets']; f.seek(8+hlen+a); raw=f.read(b-a)
    return (np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(m['shape'])
idx=json.load(open(f"{R}/model.safetensors.index.json"))["weight_map"]
L=31; pre=f"language_model.model.layers.{L}.mlp"
Wg=torch.tensor(load_w(idx[pre+'.gate_proj.weight'],pre+'.gate_proj.weight'),device=dev)
Wu=torch.tensor(load_w(idx[pre+'.up_proj.weight'],pre+'.up_proj.weight'),device=dev)
Wd=torch.tensor(load_w(idx[pre+'.down_proj.weight'],pre+'.down_proj.weight'),device=dev)
raw=open(f"{AC}/post_input_norm/L{L:02d}.f16","rb").read()
h=torch.tensor(np.frombuffer(raw,dtype=np.float16).astype(np.float32).reshape(-1,5120),device=dev)
N=h.shape[0]
def mlp(hh,g,u,d): return (torch.nn.functional.silu(hh@g.T)*(hh@u.T))@d.T
with torch.no_grad(): Y=mlp(h,Wg,Wu,Wd)
tr=slice(0,N//2); te=slice(N//2,N); Ytr,htr=Y[tr],h[tr]
def rel(a,b): return (torch.norm(a-b)/torch.norm(b)).item()
def q3(w,g=64):
    o,i=w.shape; wr=w.reshape(o,i//g,g); am=wr.abs().amax(-1,keepdim=True).clamp_min(1e-9)
    return (torch.round(wr/am*3).clamp(-3,3)/3*am).reshape(o,i)
with torch.no_grad(): eq3=rel(mlp(h,q3(Wg),q3(Wu),q3(Wd))[te],Y[te])
print(f"q3 MLP held-out {eq3:.4f} (teacher intermediate 17408, active 108.6 MB)",flush=True)
class SwiGLUOp(torch.nn.Module):
    def __init__(s,m):
        super().__init__(); s.g=torch.nn.Linear(5120,m,bias=False); s.u=torch.nn.Linear(5120,m,bias=False); s.d=torch.nn.Linear(m,5120,bias=False)
    def forward(s,x): return s.d(torch.nn.functional.silu(s.g(x))*s.u(x))
for m in (4096,8704):
    G=SwiGLUOp(m).to(dev)
    # init from teacher's top-m (warm start): use random for now
    opt=torch.optim.Adam(G.parameters(),lr=2e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,3000)
    t=time.time()
    for it in range(3000):
        opt.zero_grad(); l=torch.nn.functional.mse_loss(G(htr),Ytr); l.backward(); opt.step(); sch.step()
    with torch.no_grad(): e=rel(G(h[te]),Y[te]); etr=rel(G(htr),Ytr)
    params=2*(5120*m)+m*5120; mb_q3=params*0.40625/1e6
    print(f"  SwiGLU-op m={m}: held-out {e:.4f} (train {etr:.4f})  active@q3 {mb_q3:.1f}MB ({100*mb_q3/108.6:.0f}%)  {'*** BEATS q3' if e<=eq3 and mb_q3<108.6 else ''}  gap={e-etr:.3f}  [{time.time()-t:.0f}s]",flush=True)
