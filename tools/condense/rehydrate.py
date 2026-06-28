#!/usr/bin/env python3.12
"""Rehydrate a condensed artifact into a servable f16 GGUF — closes condense->run in
Hawking WITHOUT runtime surgery. Merges condensed linear weights into the original HF
model, then convert_hf_to_gguf -> f16 GGUF that Hawking's EXISTING loader serves.

This proves the condensed model RUNS end-to-end in Hawking (quality = the condensed
quality). NOTE: tps here is f16-class; the low-bit RAM-cliff tps win needs native
bitslice .tq serving (a separate build). This is the safe, surgery-free path.

Usage: rehydrate.py <condensed.safetensors> <out.gguf> [hf-model-dir]
"""
import sys, os, shutil, subprocess
from safetensors.torch import load_file, save_file

COND = sys.argv[1]
OUT = sys.argv[2]
HF = sys.argv[3] if len(sys.argv) > 3 else "scratch/qwen-05b"
TMP = "/tmp/_rehydrate_hf"

orig = load_file(os.path.join(HF, "model.safetensors"))
cond = load_file(COND)
merged = dict(orig)
n = 0
for k, v in cond.items():
    if k in merged:
        merged[k] = v.to(merged[k].dtype)
        n += 1
print(f"# merged {n} condensed linears into {len(merged)} tensors", file=sys.stderr)

os.makedirs(TMP, exist_ok=True)
for f in os.listdir(HF):
    src = os.path.join(HF, f)
    if os.path.isfile(src) and not f.endswith(".safetensors"):
        shutil.copy(src, os.path.join(TMP, f))
save_file(merged, os.path.join(TMP, "model.safetensors"))

subprocess.run(["python3.12", "tools/strand/tools/gguf/convert_hf_to_gguf.py",
                TMP, "--outfile", OUT, "--outtype", "f16"], check=True)
print(f"rehydrated -> {OUT}")
