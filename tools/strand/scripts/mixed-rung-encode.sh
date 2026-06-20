#!/usr/bin/env bash
# mixed-rung-encode.sh — encode a model with per-pattern rung overrides and
# report the achieved weighted-average bpw.  Optionally measures PPL if the
# eval pipeline is available.
#
# Usage:
#   ./scripts/mixed-rung-encode.sh <MODEL_SAFETENSORS> <RUNG_CONFIG_JSON> [options]
#
# Required:
#   <MODEL_SAFETENSORS>   path to the input model (e.g. scratch/qwen-05b/model.safetensors)
#   <RUNG_CONFIG_JSON>    flat key→bits JSON (e.g. configs/rung-attn4-ffn3.json)
#
# Options:
#   --bits N              global fallback bits when no rung-config rule matches (default: 4)
#   --l N                 explicit trellis register width (default: k+4 per spec)
#   --out PATH            output recon safetensors (default: scratch/mixed-rung/<label>.safetensors)
#   --measure-only        skip writing output; report bpw only (no PPL)
#   --threads N           quantizer threads (default: 8)
#   --eval                run PPL eval on the reconstructed model (requires HF model dir)
#   --hf-dir PATH         HF model directory with config.json+tokenizer (for --eval)
#   --ctx N               PPL context length (default: 2048)
#   --chunks N            number of eval windows (default: 64)
#   --label NAME          label for output files (default: derived from rung-config filename)
#   -h, --help            show this message
#
# Output:
#   The achieved weighted-average bpw across all quantized tensors.
#   If --eval is passed and --hf-dir is set: the WikiText-2 PPL.
#
# HONEST LABELING:
#   PPL numbers are labeled "MEASURED" or "ESTIMATED (literature-prior)".
#   bpw is always computed from the encoder's sidecar JSON (measured).
#
# WHY PPL SWEEP IS FEASIBLE ONLY ON THE CLOUD:
#   A full mixed-precision encode of Qwen2.5-0.5B at --bits 4 (168 tensors) takes
#   ~7500s on M3 Pro (CPU Viterbi, --threads 8).  At --bits 3: ~4250s.  A mixed
#   attn@4/FFN@3 encode is ~4250-7500s depending on the fraction at 4-bit.
#   PPL eval adds ~80s.  Total per run: ~1-2 hours locally, not minutes.
#   See research/mixed-rung-routing.md for the literature-prior routing table and
#   the estimated PPL benefit (unverified locally).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QM="$ROOT/target/release/quantize-model"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <MODEL_SAFETENSORS> <RUNG_CONFIG_JSON> [options]" >&2
    exit 1
fi

MODEL="$1"; shift
RUNG_CONFIG="$1"; shift

BITS=4
L_FLAG=""
OUT=""
MEASURE_ONLY=0
THREADS=8
DO_EVAL=0
HF_DIR=""
CTX=2048
CHUNKS=64
LABEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bits)    BITS="$2"; shift 2 ;;
        --l)       L_FLAG="--l $2"; shift 2 ;;
        --out)     OUT="$2"; shift 2 ;;
        --measure-only) MEASURE_ONLY=1; shift ;;
        --threads) THREADS="$2"; shift 2 ;;
        --eval)    DO_EVAL=1; shift ;;
        --hf-dir)  HF_DIR="$2"; shift 2 ;;
        --ctx)     CTX="$2"; shift 2 ;;
        --chunks)  CHUNKS="$2"; shift 2 ;;
        --label)   LABEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# *//'
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -f "$QM" ]] || {
    echo "FATAL: quantize-model not found at $QM" >&2
    echo "  Run: cargo build --release -p strand-quant --bin quantize-model" >&2
    exit 1
}
[[ -f "$MODEL" ]] || { echo "FATAL: model not found: $MODEL" >&2; exit 1; }
[[ -f "$RUNG_CONFIG" ]] || { echo "FATAL: rung-config not found: $RUNG_CONFIG" >&2; exit 1; }

# Derive label from rung-config filename if not provided
if [[ -z "$LABEL" ]]; then
    LABEL="$(basename "$RUNG_CONFIG" .json)"
fi

OUT_DIR="$ROOT/scratch/mixed-rung"
mkdir -p "$OUT_DIR"

if [[ -z "$OUT" && $MEASURE_ONLY -eq 0 ]]; then
    OUT="$OUT_DIR/$LABEL.safetensors"
fi

LOG="$OUT_DIR/$LABEL.log"
SIDECAR="${OUT%.safetensors}.safetensors.json"

echo "[mixed-rung] model       : $MODEL"
echo "[mixed-rung] rung-config : $RUNG_CONFIG"
echo "[mixed-rung] bits (fallback): $BITS"
echo "[mixed-rung] label       : $LABEL"
echo "[mixed-rung] log         : $LOG"
if [[ $MEASURE_ONLY -eq 1 ]]; then
    echo "[mixed-rung] mode        : measure-only (no output written)"
else
    echo "[mixed-rung] output      : $OUT"
fi

# ── 1. Encode / measure ──────────────────────────────────────────────────────
ENCODE_FLAGS="--in $MODEL --bits $BITS ${L_FLAG} --rung-config $RUNG_CONFIG --threads $THREADS"
if [[ $MEASURE_ONLY -eq 1 ]]; then
    ENCODE_FLAGS="$ENCODE_FLAGS --measure-only"
else
    ENCODE_FLAGS="$ENCODE_FLAGS --out $OUT"
fi

STRAND_NO_GPU=1 "$QM" $ENCODE_FLAGS 2>&1 | tee "$LOG"
echo ""
echo "[mixed-rung] encode done."

# ── 2. Extract bpw from sidecar or log ───────────────────────────────────────
if [[ $MEASURE_ONLY -eq 1 ]]; then
    # Parse bpw from log AGGREGATE line
    BPW=$(grep -oP 'AGGREGATE effective bpw = \K[\d.]+' "$LOG" | tail -1 || echo "N/A")
    echo "[mixed-rung] achieved bpw : $BPW (measure-only, from log)"
elif [[ -f "$SIDECAR" ]]; then
    BPW=$(python3 -c "
import json, sys
with open('$SIDECAR') as f:
    d = json.load(f)
agg = d.get('aggregate', {})
print(f\"{agg.get('effective_bpw', 'N/A'):.4f}\")
" 2>/dev/null || grep -oP 'effective_bpw[\":\s]+\K[\d.]+' "$SIDECAR" | head -1 || echo "N/A")
    echo "[mixed-rung] achieved bpw : $BPW (from sidecar: $SIDECAR)"
else
    BPW=$(grep -oP 'AGGREGATE effective bpw = \K[\d.]+' "$LOG" | tail -1 || echo "N/A")
    echo "[mixed-rung] achieved bpw : $BPW (from log; sidecar not found)"
fi

# ── 3. Compare with all-same-rung baseline at same bpw ───────────────────────
echo ""
echo "[mixed-rung] bpw comparison:"
echo "  all-4-bit baseline : ~4.5000 bpw (measured, q4_l12 full model)"
echo "  all-3-bit baseline : ~3.3399 bpw (measured, q3_l7 full model)"
echo "  this mixed config  : ~$BPW bpw"
echo "  routing            : $(basename $RUNG_CONFIG)"

# ── 4. PPL eval (optional) ────────────────────────────────────────────────────
if [[ $DO_EVAL -eq 1 ]]; then
    if [[ $MEASURE_ONLY -eq 1 ]]; then
        echo "[mixed-rung] WARNING: --eval ignored with --measure-only (no recon written)"
    elif [[ -z "$HF_DIR" ]]; then
        echo "[mixed-rung] WARNING: --eval requires --hf-dir pointing to the base HF model dir"
        echo "[mixed-rung]          (config.json + tokenizer files next to the recon safetensors)"
        echo "[mixed-rung] PPL: SKIPPED (no --hf-dir)"
    elif [[ ! -f "$OUT" ]]; then
        echo "[mixed-rung] WARNING: output recon not found at $OUT — PPL skipped"
    else
        # Create a temp HF model dir: copy HF metadata, symlink/copy the recon safetensors
        EVAL_DIR="$OUT_DIR/${LABEL}-eval-hf"
        mkdir -p "$EVAL_DIR"
        for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
            [[ -f "$HF_DIR/$f" ]] && cp "$HF_DIR/$f" "$EVAL_DIR/"
        done
        cp "$OUT" "$EVAL_DIR/model.safetensors"
        PPL_JSON="$OUT_DIR/$LABEL-ppl.json"
        echo "[mixed-rung] running PPL eval (ctx=$CTX, chunks=$CHUNKS)..."
        python3 "$ROOT/ops/eval-ppl.py" \
            "$EVAL_DIR" "$CTX" "$CHUNKS" "auto" "bfloat16" "$LABEL" "$PPL_JSON" 2>&1 | tee "$OUT_DIR/$LABEL-eval.log"
        if [[ -f "$PPL_JSON" ]]; then
            PPL=$(python3 -c "import json; d=json.load(open('$PPL_JSON')); print(f\"{d.get('ppl', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
            echo "[mixed-rung] PPL (MEASURED): $PPL  [bpw=$BPW, tag=$LABEL, ctx=$CTX, chunks=$CHUNKS]"
        else
            echo "[mixed-rung] PPL: eval completed but JSON not found at $PPL_JSON"
        fi
    fi
fi

echo ""
echo "[mixed-rung] done. rung-config=$RUNG_CONFIG bpw=$BPW"
