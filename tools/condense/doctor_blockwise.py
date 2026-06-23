#!/usr/bin/env python3.12
"""Block/layer-wise QAT (BRECQ-lite) — the full-rank ceiling-breaker, done STABLY.

The global STRAND-QAT diverged (one stale anchor over the whole model = drift). This does
it PER LINEAR, locally: optimize each layer's weights so their FAKE-QUANTIZED form matches
the f16 layer's OUTPUT on calib activations (local MSE + STE). Local scope = stable. The
result is full-rank quant-robust weights (no LoRA ceiling, no bpw overhead) -> STRAND-bake.

Usage: doctor_blockwise.py <hf-model-dir> <out_raw.safetensors> [bits] [steps]
Then STRAND-bake <out_raw> and measure. Env: DOCTOR_CALIB.
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

MODEL = sys.argv[1]
OUT = sys.argv[2]
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 80
dev = "mps" if torch.backends.mps.is_available() else "cpu"
CALIB = open(os.environ.get("DOCTOR_CALIB", "scratch/calib_corpus.txt"), errors="ignore").read()[:8000]


def fq(w, bits):  # per-output-channel symmetric uniform + STE
    qmax = 2 ** (bits - 1) - 1
    s = (w.abs().amax(1, keepdim=True) / qmax).clamp(min=1e-8)
    return w + (torch.clamp((w / s).round(), -qmax - 1, qmax) * s - w).detach()


m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok(CALIB, return_tensors="pt").input_ids[:, :1024].to(dev)

# capture each linear's INPUT activations (f16 run)
inps, hooks = {}, []
def mk(n):
    def h(mod, i, o): inps[n] = i[0].detach().reshape(-1, i[0].shape[-1])
    return h
for n, mod in m.named_modules():
    if isinstance(mod, nn.Linear) and mod.weight.shape[1] >= 256:
        hooks.append(mod.register_forward_hook(mk(n)))
with torch.no_grad():
    m(ids)
for h in hooks:
    h.remove()
print(f"# block-wise QAT: {len(inps)} linears, {BITS}-bit, {STEPS} steps/layer", file=sys.stderr)

sd = load_file(os.path.join(MODEL, "model.safetensors"))
out_sd = dict(sd)
for n, mod in m.named_modules():
    k = n + ".weight"
    if isinstance(mod, nn.Linear) and n in inps and k in sd:
        X = inps[n]                                   # [N, in]
        W0 = mod.weight.detach().clone()             # [out, in]
        with torch.no_grad():
            Y = X @ W0.T                             # f16 output target
        W = W0.clone().requires_grad_(True)
        opt = torch.optim.Adam([W], lr=1e-3)
        for _ in range(STEPS):
            opt.zero_grad()
            loss = F.mse_loss(X @ fq(W, BITS).T, Y)  # match f16 output UNDER quant
            loss.backward()
            opt.step()
        out_sd[k] = W.detach().cpu().to(torch.float16)   # raw quant-robust weights (STRAND-bake next)
        del X, Y, W

save_file(out_sd, OUT)
print(f"# saved quant-robust weights -> {OUT} (STRAND-bake this, then measure)")
