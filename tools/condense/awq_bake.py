#!/usr/bin/env python3.12
"""AWQ-bake: activation-aware STRAND condense. Scale each weight's input columns by the
activation magnitude^alpha BEFORE quant (protects high-activation channels), unscale after
— folded into the weight so serving is unchanged. Proven in the output-space harness
(3-bit 1.96x -> 1.28x). Training-free => the doctor then has a far smaller gap to close.

Usage: awq_bake.py <hf-model-dir> <out.safetensors> [bits] [alpha]
"""
import sys, os, subprocess, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

MODEL = sys.argv[1]
OUT = sys.argv[2]
TAG = os.path.basename(OUT).replace(".safetensors", "").replace("/", "_")  # unique temps per run
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
BAKER = "vendor/strand-quant/target/release/quantize-model"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
CALIB = open(os.environ.get("DOCTOR_CALIB", "scratch/calib_corpus.txt"), errors="ignore").read()[:20000]

m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
tok = AutoTokenizer.from_pretrained(MODEL)

# capture per-input-channel mean|x| per linear (the AWQ importance)
sig, hooks = {}, []
def mk(name):
    def h(mod, inp, out):
        x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).mean(0)
        sig[name] = sig.get(name, torch.zeros_like(x)) + x
    return h
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear) and mod.weight.shape[1] >= 256:
        hooks.append(mod.register_forward_hook(mk(name)))
ids = tok(CALIB, return_tensors="pt").input_ids[:, :2048].to(dev)
with torch.no_grad():
    m(ids)
for h in hooks:
    h.remove()
print(f"# captured activation scale for {len(sig)} linears", file=sys.stderr)

sd = load_file(os.path.join(MODEL, "model.safetensors"))
scaled, scales = {}, {}
for name, mod in m.named_modules():
    k = name + ".weight"
    if isinstance(mod, nn.Linear) and k in sd and name in sig:
        s = (sig[name].cpu().float() + 1e-6) ** ALPHA            # [in_features]
        scaled[k] = (sd[k].float() * s).to(torch.float16)        # scale columns
        scales[k] = s
    elif k in sd:
        scaled[k] = sd[k]
save_file(scaled, f"/tmp/awq_scaled_{TAG}.safetensors")

subprocess.run([BAKER, "--in", f"/tmp/awq_scaled_{TAG}.safetensors", "--out", f"/tmp/awq_baked_{TAG}.safetensors",
                "--bits", str(BITS), "--quality", "--rht-cols", "--outlier-channel", "1",
                "--outlier-bits", "8", "--threads", "10"], check=True, capture_output=True)

baked = load_file(f"/tmp/awq_baked_{TAG}.safetensors")
awq = {}
for k, v in baked.items():
    awq[k] = (v.float() / scales[k]).to(torch.float16) if k in scales else v   # unscale -> fold
save_file(awq, OUT)
print(f"AWQ base ({BITS}-bit, alpha={ALPHA}) saved -> {OUT}")
