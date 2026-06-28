#!/usr/bin/env bash
# ── hawking doctor ── quality-infusion / recovery for condensed models ──────────
#
# Condensation loses quality at low bits; the doctor HEALS it. It re-fits the
# model's weights with the low-bit quantizer in the loop (QAT) and, with --kd,
# distills the f16 teacher's logits into the student (KD). The output is a set of
# recovered "shadow" fp32 weights (.pt) that the condense ENCODE stage bakes — so
# the shipped low-bit artifact reads ~1:1 with the f16 parent but runs faster
# (fewer bytes => RAM cliff / less bandwidth). This is the lever the output-space
# data proved is REQUIRED: data-free post-hoc patches (low-rank residual) are a
# measured NO-GO at 2-bit — only gradient recovery moves the quantized values.
#
# Wraps tools/strand/scripts/strand-qat.py (generic AutoModelForCausalLM).
#
# Usage:
#   tools/condense/doctor.sh <hf-model-dir> [opts]
#     --bits N       target bits to recover for (default 2 — the Hawking lead)
#     --steps S      QAT steps (default 400; --smoke forces 2)
#     --kd           add knowledge-distillation loss vs the f16 teacher
#     --lr LR        learning rate (default 2e-5)
#     --ctx N        sequence length (default 1024)
#     --save PATH    shadow-weights out (default <model>/qat-<bits>bit.pt)
#     --device DEV   mps|cpu (default mps)
#     --smoke        2-step load+step validation (battery-safe; NOT a real heal)
#     --dry-run      print the command, run nothing
#
# HEAVY: a real heal is a training run — run it plugged in / via the cron, not on
# battery. --smoke and --dry-run are the battery-safe validations.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QAT="$HERE/tools/strand/scripts/strand-qat.py"

[ $# -ge 1 ] || { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
MODEL="$1"; shift

BITS=2; STEPS=400; LR=2e-5; CTX=1024; DEV=mps; KD=""; SAVE=""; SMOKE=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bits)   BITS="$2"; shift 2;;
    --steps)  STEPS="$2"; shift 2;;
    --lr)     LR="$2"; shift 2;;
    --ctx)    CTX="$2"; shift 2;;
    --save)   SAVE="$2"; shift 2;;
    --device) DEV="$2"; shift 2;;
    --kd)     KD="--kd"; shift;;
    --smoke)  SMOKE=1; shift;;
    --dry-run) DRY=1; shift;;
    *) echo "doctor: unknown arg '$1'" >&2; exit 2;;
  esac
done

[ -d "$MODEL" ] || { echo "doctor: model dir not found: $MODEL" >&2; exit 2; }
[ -f "$QAT" ]   || { echo "doctor: missing $QAT" >&2; exit 2; }
[ -n "$SAVE" ]  || SAVE="$MODEL/qat-${BITS}bit.pt"

if [ "$SMOKE" = 1 ]; then STEPS=2; CTX=256; TC="--train-chunks 8 --eval-chunks 4"; else TC="--train-chunks 256 --eval-chunks 32"; fi

CMD=(python3 "$QAT" --model "$MODEL" --quant uniform --bits "$BITS"
     --steps "$STEPS" --lr "$LR" --ctx "$CTX" $TC --device "$DEV" --save "$SAVE" $KD)

echo "── hawking doctor (recover @ ${BITS}-bit${KD:+ +KD}) ──"
echo "  model : $MODEL"
echo "  save  : $SAVE"
echo "  cmd   : ${CMD[*]}"
[ "$DRY" = 1 ] && { echo "  (dry-run — not executed)"; exit 0; }
[ "$SMOKE" = 1 ] && echo "  (smoke — 2 steps, validates the heal loop runs; not a real recovery)"
exec "${CMD[@]}"
