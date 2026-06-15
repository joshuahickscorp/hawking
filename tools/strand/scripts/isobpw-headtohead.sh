#!/usr/bin/env bash
# isobpw-headtohead.sh — the ONE iso-bpw STRAND-vs-GGUF harness.
#
# Settles "have we beaten GGUF" by scoring BOTH formats on the SAME perplexity
# harness (ops/eval-ppl.py, the STRAND canon) at the SAME (model, ctx, chunks,
# dtype, device). GGUF quants are dequantized back to HF safetensors so they go
# through the identical eval — a TRUE iso-harness, not cross-harness.
#
# Pipeline:
#   GGUF side : convert HF -> f16 GGUF -> {Q2_K,Q3_K_M,IQ3_S,Q4_K_M}
#               -> dequant each to an HF dir -> eval-ppl.py -> PPL
#               -> gguf_bpw.py for proj_bpw / file_bpw
#   STRAND side: quantize-model recon for {q2_l12_out1, q3_l12_out1, mp_light}
#               -> eval-ppl.py -> PPL ; bpw from the quant sidecar aggregate
#
# CPU-only, gated: refuses to launch heavy work while another quantize-model /
# eval-ppl owns the box (poll 120s). Resumable: every step skips if its output
# json already exists.
#
# Machine stamp: Apple M3 Pro, 18 GB, 12 logical cores (will.md §1).
set -uo pipefail

ROOT="/Users/scammermike/Downloads/strand"
cd "$ROOT"
PY=/usr/local/bin/python3
MODEL=scratch/qwen-05b
RES=research/isobpw
GGUF=$RES/gguf
DEQ=$RES/dequant
RECON=$RES/strand-recon
PPL=$RES/ppl
CTX=2048
CHUNKS=64
DTYPE=bfloat16
DEV=cpu
THREADS=12
mkdir -p "$GGUF" "$DEQ" "$RECON" "$PPL"

log() { echo "[isobpw $(date +%H:%M:%S)] $*"; }

# --- contention gate: wait until no OTHER quantize-model / eval-ppl is running.
wait_for_box() {
  while pgrep -f 'quantize-model' >/dev/null || pgrep -f 'eval-ppl' >/dev/null; do
    # ignore our own children by checking we are not the only matcher is hard;
    # simplest honest rule: if anything else is quanting/eval'ing, wait.
    log "box busy (quantize-model/eval-ppl live) — wait 120s"
    sleep 120
  done
}

evalppl() { # <hf_dir> <tag> <out_json>
  local dir="$1" tag="$2" out="$3"
  [ -f "$out" ] && { log "skip eval (have $out)"; return 0; }
  log "eval $tag  ($dir)"
  OMP_NUM_THREADS=$THREADS $PY ops/eval-ppl.py "$dir" $CTX $CHUNKS $DEV $DTYPE "$tag" "$out" 2>&1 | tail -2
}

# ============================ GGUF side ============================
[ -f "$GGUF/qwen05b-f16.gguf" ] || {
  log "convert HF -> f16 GGUF"
  $PY tools/gguf/convert_hf_to_gguf.py "$MODEL" --outtype f16 \
      --outfile "$GGUF/qwen05b-f16.gguf" 2>&1 | tail -2
}
for q in Q2_K Q3_K_M IQ3_S Q4_K_M; do
  [ -f "$GGUF/qwen05b-$q.gguf" ] || {
    log "quantize -> $q"
    /opt/homebrew/bin/llama-quantize "$GGUF/qwen05b-f16.gguf" "$GGUF/qwen05b-$q.gguf" "$q" 2>&1 | tail -2
  }
done
# bpw report
$PY - <<'EOF' > "$RES/gguf-bpw.json"
import json, subprocess, glob, os
rows=[]
for f in sorted(glob.glob("research/isobpw/gguf/qwen05b-*.gguf")):
    if "f16" in f: continue
    rows.append(json.loads(subprocess.check_output(["/usr/local/bin/python3","tools/gguf/gguf_bpw.py",f])))
print(json.dumps(rows, indent=2))
EOF
log "wrote $RES/gguf-bpw.json"

# dequant + eval each GGUF + bf16 anchor (via f16 gguf dequant == lossless-ish to bf16? no:
# the bf16 anchor must be the ORIGINAL HF weights, scored on the same harness)
evalppl "$MODEL" bf16_anchor "$PPL/ppl_bf16_anchor.json"
for q in Q2_K Q3_K_M IQ3_S Q4_K_M; do
  d="$DEQ/qwen05b-$q"
  [ -f "$d/model.safetensors" ] || {
    log "dequant $q -> HF"
    $PY tools/gguf/gguf_to_hf.py "$GGUF/qwen05b-$q.gguf" "$MODEL" "$d" 2>&1 | tail -1
  }
  evalppl "$d" "gguf_$q" "$PPL/ppl_gguf_$q.json"
done

# ============================ STRAND side ============================
# (a) q2_l12_out1  (b) q3_l12_out1  (c) mp_light = rung-attn4-ffn3
strand_recon() { # <tag> <extra-flags...>
  local tag="$1"; shift
  local out="$RECON/$tag"
  mkdir -p "$out"
  if [ ! -f "$out/model.safetensors" ]; then
    wait_for_box
    log "STRAND recon $tag : $*"
    STRAND_NO_GPU=1 target/release/quantize-model --in "$MODEL/model.safetensors" \
      --out "$out/model.safetensors" "$@" --threads $THREADS 2>&1 | tail -3
    for f in config.json tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json; do
      [ -f "$MODEL/$f" ] && cp "$MODEL/$f" "$out/"; done
  else
    log "skip recon (have $tag)"
  fi
  evalppl "$out" "strand_$tag" "$PPL/ppl_strand_$tag.json"
}

strand_recon q2_l12_out1 --bits 2 --l 12 --outlier-channel 1
strand_recon q3_l12_out1 --bits 3 --l 12 --outlier-channel 1
strand_recon mp_light    --bits 3 --rung-config configs/rung-attn4-ffn3.json --l 12 --outlier-channel 1

log "ALL DONE. Build the table with: $PY tools/gguf/isobpw_table.py"
