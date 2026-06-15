#!/usr/bin/env bash
# =============================================================================
# runpod-bootstrap.sh — one-shot STRAND 7B run on a fresh RunPod (or any
# Linux + NVIDIA CUDA) box. Deploy a pod, run this once, results land on the
# persistent volume. This is the whole pipeline:
#   rust toolchain -> clone -> build (quantizer + decode kernel) -> download 7B
#   -> calibrate Hessian -> the moat sweep (same configs as the master notebook)
#   -> decode-kernel bench -> a RESULTS.txt summary.
#
# Idempotent + resumable: re-run after ANY interruption or pod restart and it
# skips finished work and continues. Toolchain, model, build, and HF cache all
# live on the volume, so a restart is fast and nothing re-downloads.
#
# Usage (interactive, survives SSH drop):
#     nohup bash scripts/runpod-bootstrap.sh > /workspace/run.log 2>&1 &
#     tail -f /workspace/run.log
#
# Fully hands-off — paste as the Pod's "Container Start Command":
#     bash -lc 'cd /workspace && (git -C strand pull -q 2>/dev/null || \
#       git clone -q -b sub4bit-innovation https://github.com/joshuahickscorp/strand.git strand) && \
#       bash strand/scripts/runpod-bootstrap.sh'
#
# Tunables (env): WORK (volume mount; default /workspace), MODEL_ID, BRANCH,
#                 N_SEQS (calibration sequences), DEVICE (eval device).
# Quant runs on CPU across all vCPUs; only the PPL eval uses the GPU.
# =============================================================================
set -uo pipefail   # deliberately NOT -e: one failing config must not abort the sweep

WORK="${WORK:-/workspace}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
BRANCH="${BRANCH:-sub4bit-innovation}"
N_SEQS="${N_SEQS:-128}"
DEVICE="${DEVICE:-cuda}"
BF16_PPL=7.7362

REPO="$WORK/strand"
MODEL_DIR="$REPO/scratch/qwen"
RESULTS="$WORK/strand-results"

# Keep the toolchain + caches on the persistent volume so restarts are instant.
export CARGO_HOME="$WORK/.cargo" RUSTUP_HOME="$WORK/.rustup" HF_HOME="$WORK/.hf"
export PATH="$CARGO_HOME/bin:$PATH"
export STRAND_PYTHON="${STRAND_PYTHON:-python3}"

mkdir -p "$WORK" "$RESULTS"
log(){ echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

# ---- 1. Rust toolchain -------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
    log "installing Rust toolchain -> $CARGO_HOME"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path >/dev/null 2>&1
fi

# ---- 2. Repo -----------------------------------------------------------------
if [ -d "$REPO/.git" ]; then
    log "updating repo"; git -C "$REPO" pull -q 2>/dev/null || true
else
    log "cloning $BRANCH"; git clone -q -b "$BRANCH" https://github.com/joshuahickscorp/strand.git "$REPO"
fi
cd "$REPO" || { log "FATAL: repo missing"; exit 1; }

# ---- 3. Build quantizer + decode kernel -------------------------------------
log "building strand-quant + strand-decode-kernel (release)"
cargo build --release -p strand-quant        2>&1 | tail -2
cargo build --release -p strand-decode-kernel 2>&1 | tail -2
[ -x target/release/quantize-model ] || { log "FATAL: quantize-model did not build"; exit 1; }

# ---- 4. Python deps (torch is preinstalled on RunPod PyTorch images) --------
log "installing python deps"
$STRAND_PYTHON -m pip install -q transformers datasets safetensors accelerate huggingface_hub 2>/dev/null \
  || pip install -q transformers datasets safetensors accelerate huggingface_hub 2>/dev/null || true

# ---- 5. Model ----------------------------------------------------------------
if [ ! -f "$MODEL_DIR/config.json" ]; then
    log "downloading $MODEL_ID -> $MODEL_DIR"
    $STRAND_PYTHON - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
    allow_patterns=['*.safetensors','*.json','tokenizer*','vocab*','merges*'])
print("model ready")
PY
else
    log "model already present"
fi

# ---- 6. Calibrate the diagonal Hessian (resumable) --------------------------
HESS="$MODEL_DIR/hessian.hsdi"
if [ ! -f "$HESS" ]; then
    log "calibrating diagonal Hessian (resumable checkpoint on volume)"
    $STRAND_PYTHON scripts/calibrate-hsdi.py --model "$MODEL_DIR" --out "$HESS" \
        --device "$DEVICE" --dtype bfloat16 --n-seqs "$N_SEQS" \
        --checkpoint "$RESULTS/hessian.partial" --checkpoint-every 8
fi
[ -f "$HESS" ] && log "Hessian ready" || log "WARN: no Hessian — sweep will run uncalibrated"

# ---- 7. The moat sweep (mirrors the master notebook), resumable per-config ---
run_cfg(){  # name  bits  [extra flags...]
    local name="$1" bits="$2"; shift 2
    if [ -f "$RESULTS/$name.ppl.json" ]; then log "skip $name (already on volume)"; return; fi
    local out="$MODEL_DIR/sweep/$name"
    log "=== $name (bits=$bits) $* ==="
    bash scripts/strand-7b-ppl.sh "$MODEL_DIR" --bits "$bits" "$@" \
        --label "$name" --device "$DEVICE" --limit-chunks "${RUNPOD_CHUNKS:-64}" --resume --out-dir "$out" \
        2>&1 | tee "$RESULTS/$name.log"
    local pj; pj="$(ls "$out"/ppl_*.json 2>/dev/null | grep -v fp16 | head -1)"
    if [ -n "$pj" ]; then cp "$pj" "$RESULTS/$name.ppl.json"; log "SAVED $name"; else log "WARN: no PPL json for $name"; fi
    rm -rf "$out/recon"   # reclaim ~15 GB before the next config
}

# The PTQ sweep that used to live here is done — its verdicts are canon
# (will.md §3/§4: salient dead, vec floors recorded, 2-bit reopened via l=12+outlier).
# Act-3 (cloud 7B QAT + quant) commands go here once the winning 0.5B recipe is
# chosen; run_cfg above stays as the quantize→eval harness.

# ---- 8. Decode kernel: correctness ------------------------------------------
log "decode kernel tests"
cargo test -q -p strand-decode-kernel 2>&1 | tail -3

# ---- 9. Summary table --------------------------------------------------------
log "building RESULTS.txt"
$STRAND_PYTHON - "$RESULTS" "$BF16_PPL" <<'PY'
import json, re, os, sys, glob
res, bf16 = sys.argv[1], float(sys.argv[2])
def relrms_bpw(logp):
    try:
        m = re.search(r'effective bpw = ([0-9.]+).*?weighted rel-RMS = ([0-9.]+)', open(logp).read(), re.S)
        if m: return float(m.group(2)), float(m.group(1))
    except Exception:
        pass
    return None, None
rows = ['%-18s %6s %9s %8s %9s' % ('config', 'bpw', 'PPL', 'dPPL%', 'size_GB')]
for pj in sorted(glob.glob(os.path.join(res, '*.ppl.json'))):
    name = os.path.basename(pj)[:-9]
    d = json.load(open(pj)); ppl = d.get('ppl', float('nan'))
    rr, bpw = relrms_bpw(os.path.join(res, name + '.log')); bpw = bpw or 0
    g = bpw * 7.0e9 / 8 / 1e9
    rows.append('%-18s %6.2f %9.4f %+7.1f %8.2f' % (name, bpw, ppl, 100 * (ppl - bf16) / bf16, g))
open(os.path.join(res, 'RESULTS.txt'), 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
print('\nbf16 baseline PPL = %.4f' % bf16)
PY

log "DONE. Everything is in $RESULTS (persists on the volume):"
ls -1 "$RESULTS"
