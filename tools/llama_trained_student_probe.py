#!/usr/bin/env python3
"""Bounded fit-only trained SiLU student probe for a sealed Llama dataset."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

SCHEMA = "hawking.tg.llama_trained_student_probe.v1"

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True); p.add_argument("--capture-receipt",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--width",type=int,default=128)
    p.add_argument("--epochs",type=int,default=4); p.add_argument("--batch",type=int,default=256); p.add_argument("--lr",type=float,default=1e-3)
    a=p.parse_args()
    receipt=json.loads(a.capture_receipt.read_text())
    if receipt.get("dataset",{}).get("sha256") != sha256(a.dataset): raise SystemExit("dataset hash does not match capture receipt")
    d=np.load(a.dataset); x=np.asarray(d["inputs"],dtype=np.float32); y=np.asarray(d["targets"],dtype=np.float32); hold=np.asarray(d["heldout"],dtype=bool); d.close()
    if (~hold).sum()<8192 or hold.sum()<2048: raise SystemExit("sealed dataset below capability minimum")
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(17); np.random.seed(17)
    x_mean=x[~hold].mean(axis=0,dtype=np.float64).astype(np.float32); y_mean=y[~hold].mean(axis=0,dtype=np.float64).astype(np.float32)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],a.width),torch.nn.SiLU(),torch.nn.Linear(a.width,y.shape[1])).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    order=np.arange((~hold).sum()); xf=x[~hold]; yf=y[~hold]
    for epoch in range(a.epochs):
        rng=np.random.default_rng(17+epoch); rng.shuffle(order)
        for start in range(0,len(order),a.batch):
            idx=order[start:start+a.batch]
            xb=torch.from_numpy(xf[idx]-x_mean).to(device); yb=torch.from_numpy(yf[idx]-y_mean).to(device)
            opt.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(model(xb),yb); loss.backward(); opt.step()
    def error(mask: np.ndarray) -> float:
        total=base=0.0
        with torch.no_grad():
            for start in range(0,int(mask.sum()),a.batch):
                rows=np.flatnonzero(mask)[start:start+a.batch]; pred=model(torch.from_numpy(x[rows]-x_mean).to(device)).float().cpu().numpy()+y_mean
                total+=float(np.square(pred-y[rows],dtype=np.float64).sum()); base+=float(np.square(y[rows]-y_mean,dtype=np.float64).sum())
        return float(np.sqrt(total/max(base,np.finfo(np.float64).tiny)))
    fit,held=error(~hold),error(hold)
    params=sum(v.numel() for v in model.parameters())+x_mean.size+y_mean.size
    out={"schema":SCHEMA,"status":"OFFLINE_SURFACE_GATE_PASS_RUNTIME_REQUIRED" if held<=.1 else "OFFLINE_SURFACE_GATE_FAILED","dataset_sha256":sha256(a.dataset),"device":str(device),"architecture":{"width":a.width,"activation":"silu","epochs":a.epochs,"batch":a.batch,"lr":a.lr,"seed":17},"score":{"fit_normalized_rmse":fit,"heldout_normalized_rmse":held},"physical":{"stored_parameters":params,"prospective_fp16_artifact_bytes":params*2,"executed_macs_per_token":2*x.shape[1]*a.width,"sequential_matvecs_per_token":2},"runtime_eligibility":"NO: requires generated-token capability, artifact/runtime parity, and matched decode evidence","tps_claim":None}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"status":out["status"],"out":str(a.out)},indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
