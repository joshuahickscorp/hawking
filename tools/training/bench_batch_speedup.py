#!/usr/bin/env python3
"""Quick MPS throughput + RAM A/B: current (batch=1, per-example empty_cache,
grad-checkpoint on) vs optimized (batched, no empty_cache, grad-checkpoint off).

Synthetic fixed token budget so the comparison is apples-to-apples. NOTE: if a
training sweep is using the GPU concurrently, absolute tok/s is depressed by
contention but the A/B *ratio* stays representative.
"""
import sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rwkv7_torch_model import RWKV7Config, RWKV7Model
from rwkv7_custom_configs import CUSTOM_VARIANTS
from dataclasses import replace

DEV = "mps"
SEQ = 256
N_SEQS = 64                  # same token budget for both arms
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "draft_50m_probe"


def peak_mb():
    f = getattr(torch.mps, "driver_allocated_memory", None) or getattr(torch.mps, "current_allocated_memory", None)
    return (f() / 1024 / 1024) if f else float("nan")


def run(batch_size, empty_cache_each, grad_ckpt):
    torch.manual_seed(0)
    cfg = replace(CUSTOM_VARIANTS[VARIANT], use_chunked=True, chunk_size=32)
    model = RWKV7Model(cfg).to(DEV, torch.float32)
    model.grad_checkpoint = grad_ckpt
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ids = torch.randint(1, cfg.vocab_size, (N_SEQS, SEQ), dtype=torch.long)
    # warmup (kernel/PSO compile) not counted
    xb = ids[:batch_size].to(DEV)
    model(xb, return_final_hidden=True).sum().backward(); opt.zero_grad(set_to_none=True)
    if hasattr(torch.mps, "synchronize"): torch.mps.synchronize()
    t0 = time.time()
    for k in range(0, N_SEQS, batch_size):
        xb = ids[k:k + batch_size].to(DEV)
        hidden = model(xb, return_final_hidden=True)
        loss = model.lm_head(hidden.reshape(-1, hidden.size(-1))).float().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if empty_cache_each:
            torch.mps.empty_cache()
    if hasattr(torch.mps, "synchronize"): torch.mps.synchronize()
    dt = time.time() - t0
    toks = N_SEQS * SEQ
    return toks / dt, peak_mb()


print(f"variant={VARIANT}  tokens={N_SEQS*SEQ}  seq={SEQ}")
cur_tps, cur_mb = run(batch_size=1, empty_cache_each=True, grad_ckpt=True)
print(f"  current  (bs=1, empty_cache/step, ckpt=on):  {cur_tps:7.0f} tok/s   ~{cur_mb:6.0f} MB")
opt_tps, opt_mb = run(batch_size=16, empty_cache_each=False, grad_ckpt=False)
print(f"  optimized(bs=16, no empty_cache, ckpt=off):  {opt_tps:7.0f} tok/s   ~{opt_mb:6.0f} MB")
print(f"  SPEEDUP: {opt_tps/cur_tps:.2f}x   (RAM {opt_mb/max(cur_mb,1):.1f}x)")
