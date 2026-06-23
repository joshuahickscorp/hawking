#!/usr/bin/env python3.12
"""Emit a TWO-PART COMPRESSED residual STRAND artifact for SERVING.

Companion to residual_bake.py, but instead of materialising the decoded SUM as an
f16 safetensors (quality-measurement only), this keeps BOTH STRAND passes
COMPRESSED on disk so the runtime can sum them at GEMV time — the whole density
point of residual quant:

    W  ≈  STRAND_b1(W)  +  STRAND_b2(W − decode(STRAND_b1(W)))
          └── base ──┘     └──────── residual ────────┘

Output: two STR2 (`.tq`) archives the existing loader (`crate::tq::read_strand`)
already parses, summed at serve time by the residual GEMV path
(HAWKING_TQ_RESIDUAL: base `strand_bitslice_gemv_tcb` then residual
`strand_bitslice_gemv_tcb_accum`):

    <out>.tq        # base pass   (b1-bit STRAND of W)
    <out>.res.tq    # residual    (b2-bit STRAND of W − decode(base))

The serving GPU bitslice kernel decodes RAW Q12 and dots directly — it does NOT
apply the RHT-cols activation transform or OUTL outlier overwrites. So this tool
bakes BOTH passes with `--no-rht` and NO outlier channel by default, which is the
exact contract that path reproduces bit-faithfully (see
kernels::residual_serve_tests in hawking-core). `--rht-cols` / `--outlier-channel`
are exposed for experimentation but will NOT be served correctly until those
serving steps are wired — a warning is printed if you pass them.

Usage:
    residual_tq.py <hf-dir|base.safetensors> <out-prefix> [b1] [b2] [extra baker args...]

    <out-prefix> may end in `.tq` (stripped) or not; the two files are written as
    <out-prefix>.tq and <out-prefix>.res.tq.

Example:
    python3.12 tools/condense/residual_tq.py models/qwen7b out/qwen7b_res 3 2
    # -> out/qwen7b_res.tq (base 3-bit) + out/qwen7b_res.res.tq (residual 2-bit)
    # serve: HAWKING_RWKV7_TQ=1 HAWKING_RWKV7_TQ_PATH=out/qwen7b_res.tq \
    #        HAWKING_TQ_RESIDUAL=1   (residual auto-discovered at <path>.res.tq)
"""
import os
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file

if len(sys.argv) < 3:
    print(__doc__, file=sys.stderr)
    sys.exit(2)

SRC = sys.argv[1]
OUT_PREFIX = sys.argv[2]
if OUT_PREFIX.endswith(".tq"):
    OUT_PREFIX = OUT_PREFIX[:-3]
B1 = int(sys.argv[3]) if len(sys.argv) > 3 else 3
B2 = int(sys.argv[4]) if len(sys.argv) > 4 else 2
EXTRA = sys.argv[5:]  # passed through to every baker invocation

BAKER = "vendor/strand-quant/target/release/quantize-model"
TAG = os.path.basename(OUT_PREFIX).replace("/", "_")
src = os.path.join(SRC, "model.safetensors") if os.path.isdir(SRC) else SRC

BASE_TQ = f"{OUT_PREFIX}.tq"
RES_TQ = f"{OUT_PREFIX}.res.tq"

# Serving-faithful defaults: the bitslice GEMV serves RAW q12 (no RHT, no OUTL).
# --no-rht keeps the decode un-rotated; we add NO --outlier-channel. Warn if the
# caller forces serving-incompatible options through EXTRA.
if any(a in ("--rht-cols",) for a in EXTRA) or any("--outlier-channel" in a for a in EXTRA):
    print(
        "WARNING: --rht-cols / --outlier-channel are NOT applied by the residual GPU "
        "serve path yet; the artifact will not serve bit-faithfully.",
        file=sys.stderr,
    )
    BASE_FLAGS = []  # caller takes responsibility
else:
    BASE_FLAGS = ["--no-rht"]


def run_baker(extra):
    cmd = [BAKER, "--threads", "10", *BASE_FLAGS, *EXTRA, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit(f"baker failed ({r.returncode}): {' '.join(cmd)}")
    return r.stdout + r.stderr


def grep_bpw(log, label):
    for line in log.splitlines():
        if "AGGREGATE effective bpw" in line:
            print(f"# {label}: {line.strip()}", file=sys.stderr)
            return


W = load_file(src)

# ── Stage 1: base pass ──────────────────────────────────────────────────────
# (a) recon-bake to get the decoded base weights W_hat1 (needed to form residual).
recon_base = f"/tmp/restq_b1_recon_{TAG}.safetensors"
log = run_baker(["--in", src, "--out", recon_base, "--bits", str(B1)])
grep_bpw(log, f"base recon {B1}-bit")
Wh1 = load_file(recon_base)

# Which tensors the baker actually quantized (2-D linears; embeddings/norms copied).
qkeys = {
    k
    for k, v in W.items()
    if k in Wh1 and v.dim() == 2 and Wh1[k].shape == v.shape and not torch.equal(Wh1[k], v)
}
print(
    f"# residual: {len(qkeys)} quantized tensors, base {B1}-bit + residual {B2}-bit",
    file=sys.stderr,
)

# (b) pack the base pass to a compressed STR2 archive (.tq) and stop.
log = run_baker(["--in", src, "--bits", str(B1), "--packed-v2-out", BASE_TQ])
grep_bpw(log, f"base packed {B1}-bit")

# ── Stage 2: residual pass ──────────────────────────────────────────────────
# Build a residual model: residual on qkeys, ORIGINALS elsewhere so the baker has
# full context (matches residual_bake.py). Only the qkeys' compressed residual is
# summed at serve time; non-quantized tensors are served from their base archive.
Rin = {}
for k, v in W.items():
    Rin[k] = (v.float() - Wh1[k].float()).to(torch.float16) if k in qkeys else v
res_in = f"/tmp/restq_R_{TAG}.safetensors"
save_file(Rin, res_in)

# Pack the residual pass to a compressed STR2 archive (.res.tq) and stop.
log = run_baker(["--in", res_in, "--bits", str(B2), "--packed-v2-out", RES_TQ])
grep_bpw(log, f"residual packed {B2}-bit")


def fsize(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


base_b = fsize(BASE_TQ)
res_b = fsize(RES_TQ)
print(
    f"residual two-part artifact written:\n"
    f"  base     -> {BASE_TQ}  ({base_b/1e6:.1f} MB, {B1}-bit)\n"
    f"  residual -> {RES_TQ}  ({res_b/1e6:.1f} MB, {B2}-bit)\n"
    f"  combined ~{(base_b+res_b)/1e6:.1f} MB (~{B1}+{B2} bpw, full-rank correction)\n"
    f"serve: HAWKING_RWKV7_TQ=1 HAWKING_RWKV7_TQ_PATH={BASE_TQ} HAWKING_TQ_RESIDUAL=1"
)
