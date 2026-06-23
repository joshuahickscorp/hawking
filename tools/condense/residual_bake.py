#!/usr/bin/env python3.12
"""Residual STRAND quantization — the FULL-RANK, codec-native ceiling-breaker.

The LoRA doctor is low-rank (caps); uniform-proxy QAT (global/block-wise) optimizes weights
for the WRONG grid (catastrophic on STRAND's trellis). This instead quantizes the ACTUAL
residual error with the SAME codec:  W ≈ STRAND_b1(W) + STRAND_b2(W − STRAND_b1(W)).
The residual term is FULL-RANK (captures the high-rank quant error LoRA can't) and uses STRAND
(no transfer gap). Cost: +b2 bpw. The principled "instill quality" that isn't rank-limited.

Usage: residual_bake.py <hf-dir|base.safetensors> <out.safetensors> [b1] [b2]
"""
import sys, os, subprocess, torch
from safetensors.torch import load_file, save_file

SRC = sys.argv[1]
OUT = sys.argv[2]
B1 = int(sys.argv[3]) if len(sys.argv) > 3 else 3
B2 = int(sys.argv[4]) if len(sys.argv) > 4 else 2
BAKER = "vendor/strand-quant/target/release/quantize-model"
TAG = os.path.basename(OUT).replace(".safetensors", "").replace("/", "_")
src = os.path.join(SRC, "model.safetensors") if os.path.isdir(SRC) else SRC


def bake(inp, outp, bits):
    subprocess.run([BAKER, "--in", inp, "--out", outp, "--bits", str(bits), "--quality",
                    "--rht-cols", "--outlier-channel", "1", "--outlier-bits", "8",
                    "--threads", "10"], check=True, capture_output=True)


W = load_file(src)
# stage 1: bake the base, get decoded W_hat1
bake(src, f"/tmp/res_b1_{TAG}.safetensors", B1)
Wh1 = load_file(f"/tmp/res_b1_{TAG}.safetensors")
# which tensors the baker actually quantized (linears differ; embeddings/norms copied)
qkeys = {k for k, v in W.items()
         if k in Wh1 and v.dim() == 2 and Wh1[k].shape == v.shape and not torch.equal(Wh1[k], v)}
print(f"# residual: {len(qkeys)} quantized tensors, base {B1}-bit + residual {B2}-bit", file=sys.stderr)

# stage 2: build residual model (residual on qkeys, originals elsewhere so the baker has full ctx)
Rin = {}
for k, v in W.items():
    Rin[k] = (v.float() - Wh1[k].float()).to(torch.float16) if k in qkeys else v
save_file(Rin, f"/tmp/res_R_{TAG}.safetensors")
bake(f"/tmp/res_R_{TAG}.safetensors", f"/tmp/res_b2_{TAG}.safetensors", B2)
Rh = load_file(f"/tmp/res_b2_{TAG}.safetensors")

# sum ONLY the quantized tensors (base + residual); keep originals (embeddings/norms) once
out = {}
for k, v in W.items():
    out[k] = (Wh1[k].float() + Rh[k].float()).to(torch.float16) if k in qkeys else v
save_file(out, OUT)
print(f"residual {B1}+{B2}-bit saved -> {OUT}  (~{B1+B2} bpw, full-rank correction)")
