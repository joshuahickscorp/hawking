import sys, json, glob, os, numpy as np
C="workspace/campaign/records/runs/qwen38-27b/activation-capture-v2/parent_bf16"
j=json.load(open(f"{C}/capture-result.json"))
site="post_input_norm"; w=j["sites"][site]["width"]; n=j["sites"][site]["store_n"]
f=sorted(glob.glob(f"{C}/{site}/*0*"))[:1] or sorted(glob.glob(f"{C}/{site}/*"))[:1]
print("file:", f[0], os.path.getsize(f[0]))
X=np.fromfile(f[0],dtype=np.float16).astype(np.float32)
X=X.reshape(-1,w)[:n]
print("X",X.shape)
# prompt boundaries unknown; split by CONTIGUOUS BLOCKS, which follow prompt order,
# so block A and block B are largely disjoint prompt families
k=4; blocks=np.array_split(X,k)
def span(A,r):
    _,_,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:r]
def energy_in(B,V):
    P=B@V.T; return float((P**2).sum()/ (B**2).sum()+1e-30)
r=int(np.linalg.matrix_rank(X,tol=1e-3*np.linalg.norm(X,2)))
print(f"full-set numerical rank {r} / {w}")
print(f"{'train block':>12}{'held block':>11}{'energy of held in train span':>30}")
for i in range(k):
    V=span(blocks[i], r)
    for jj in range(k):
        if jj==i: continue
        print(f"{i:>12}{jj:>11}{energy_in(blocks[jj],V):>30.6f}")
        break
# union rank: does adding blocks grow the span?
for m in (1,2,3,4):
    sub=np.concatenate(blocks[:m])
    rm=int(np.linalg.matrix_rank(sub,tol=1e-3*np.linalg.norm(sub,2)))
    print(f"rank of first {m}/4 blocks: {rm}")
