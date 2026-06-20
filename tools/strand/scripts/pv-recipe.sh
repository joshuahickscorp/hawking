#!/usr/bin/env bash
# pv-recipe.sh — the canonical STRAND-PV 2-bit deep run (REMEDY #3).
#
# This is THE recipe the 7B Act-3 run will inherit: a strand-trellis-in-the-loop
# PV (PV = "post-training, in-the-loop re-quant" a la AQLM-PV) arm with ALL the
# levers that landed by 2026-06-11 stacked:
#   --quant strand          forward = the REAL Rust encoder's recon (train-through-
#                           what-you-ship; proxy-transfer is DEAD, will.md §4)
#   --bits 2 --l 12         the 2-bit operating point (state count saturates at l=12)
#   --outlier-channel 1     pre-RHT top-|w| side-channel (+1%, the only live PTQ lever)
#   --kd                    KL-distill from the frozen bf16 teacher (chunked-KD proven
#                           through 4 requant segments, drv <= 12.2GB on the 18GB box)
#   --warmup-frac 0.05      Apple WSD warmup (intel scorecard 2026-06-11)
#   --cooldown-frac 0.2     Apple WSD final linear decay-to-zero — THE NEW VARIABLE
#                           this run isolates vs the prior plain-cosine PV floor 26.77
#
# Prior PV floor to beat (0.5B, 2-bit, plain cosine): 26.77 PPL (will.md 2026-06-11).
# bf16 anchor 12.55. PTQ floor (no training) 80.7. The cooldown is the only schedule
# change vs the 26.77 run; everything else held => a clean A/B on the WSD cooldown.
#
# 18GB-box hardening (will.md §7 freeze trap) is BAKED IN:
#   - non-proj params frozen by default (the harness does this)
#   - --grad-checkpoint (cuts ~2GB transient)
#   - watermark env caps (0.92/0.7 worked; prompt asked 1.0/0.85 -> we use the
#     PROVEN 0.92/0.7 because 1.0 lets MPS map the whole pool and froze the box twice;
#     override via WATERMARK_HI/LO env if you really want the looser caps)
#   - RUN ALONE: this script REFUSES to start while quantize-model or eval-ppl is live.
#
# Segmentation: 300 steps split into 4 segments of 75 (= --requant-every 75), each a
# FRESH process on a pristine MPS pool (--skip-after on all but the last; --chunk-offset
# walks the train data forward; --init-state warm-restarts from the prior segment's save).
# This is the segmented-arm isolation that held through the 27.02->26.77 asymptote.
#
# Machine-stamp: built on Apple M3 Pro, 18GB unified, macOS Darwin 25.5.0, 2026-06-11.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"

MODEL="${MODEL:-scratch/qwen-05b}"
PY="${PY:-/usr/local/bin/python3}"
OUT="${OUT:-research/pv-deep}"
STEPS="${STEPS:-300}"
SEG="${SEG:-75}"               # steps per segment == requant cadence
LR="${LR:-1e-4}"              # the rung-2 PV LR (sub-2-bit re-learn regime; will.md)
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
COOLDOWN_FRAC="${COOLDOWN_FRAC:-0.2}"
WATERMARK_HI="${WATERMARK_HI:-0.92}"   # 0.92 PROVEN; prompt's 1.0 froze the box twice
WATERMARK_LO="${WATERMARK_LO:-0.7}"
STRAND_FLAGS="${STRAND_FLAGS:---bits 2 --l 12 --outlier-channel 1 --threads 8}"

mkdir -p "$OUT"
LOG="$OUT/pv-deep-$(date +%Y%m%d-%H%M%S).log"

# --- box-free guard: MPS training MUST wait for the cohabiting jobs to clear ---
if pgrep -f quantize-model >/dev/null || pgrep -f eval-ppl >/dev/null; then
  echo "REFUSING: quantize-model or eval-ppl is live (will.md §7 freeze trap)." | tee -a "$LOG"
  echo "Poll: until ! pgrep -f quantize-model && ! pgrep -f eval-ppl; do sleep 120; done" | tee -a "$LOG"
  exit 3
fi

echo "[pv-recipe] $(sysctl -n machdep.cpu.brand_string) 18GB | $(date)" | tee -a "$LOG"
echo "[pv-recipe] model=$MODEL steps=$STEPS seg=$SEG lr=$LR warmup=$WARMUP_FRAC cooldown=$COOLDOWN_FRAC" | tee -a "$LOG"
echo "[pv-recipe] floor-to-beat=26.77 (plain-cosine PV) | bf16=12.55 | PTQ=80.7" | tee -a "$LOG"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO="$WATERMARK_HI"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="$WATERMARK_LO"

NSEG=$(( (STEPS + SEG - 1) / SEG ))
STATE="$OUT/pv-deep.state.pt"
ARM="pv-deep-wsd"

for ((s=0; s<NSEG; s++)); do
  OFFSET=$(( s * SEG ))
  LAST=$(( s == NSEG-1 ? 1 : 0 ))
  echo "[pv-recipe] === segment $((s+1))/$NSEG  chunk-offset=$OFFSET  last=$LAST ===" | tee -a "$LOG"

  ARGS=( "$MODEL"
    --quant strand --bits 2 --l 12
    --steps "$SEG" --lr "$LR"
    --requant-every "$SEG"
    --kd --grad-checkpoint
    --warmup-frac "$WARMUP_FRAC" --cooldown-frac "$COOLDOWN_FRAC"
    --chunk-offset "$OFFSET"
    --strand-flags "$STRAND_FLAGS"
    --arm-name "$ARM" --lineage-label science
    --out "$OUT/pv-deep-seg$s.json" )

  # warm-restart from the prior segment
  if [ "$s" -gt 0 ]; then ARGS+=( --init-state "$STATE" ); fi
  # all but the last segment exit before the final eval/requant (fresh pool next)
  if [ "$LAST" -eq 0 ]; then ARGS+=( --skip-after ); fi
  # always persist the state for the next segment / final save
  ARGS+=( --save "$STATE" )
  if [ "$LAST" -eq 1 ]; then ARGS+=( --save-hf "$OUT/pv-deep-hf" ); fi

  "$PY" --version >/dev/null 2>&1 || { echo "no python at $PY" | tee -a "$LOG"; exit 1; }
  caffeinate -dimsu "$PY" scripts/strand-qat.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
done

echo "[pv-recipe] DONE. trajectory:" | tee -a "$LOG"
grep -hoE 'ppl[^ ]* [0-9.]+' "$OUT"/pv-deep-seg*.json 2>/dev/null | tee -a "$LOG" || true
echo "[pv-recipe] final HF dir: $OUT/pv-deep-hf (feed to strand-7b-ppl.sh for canon PPL)" | tee -a "$LOG"
echo "[pv-recipe] verdict: compare final PPL to the 26.77 plain-cosine floor." | tee -a "$LOG"
