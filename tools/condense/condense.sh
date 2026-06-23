#!/usr/bin/env bash
# ── hawking condense ── the condensation pipeline ───────────────────────────────
#
#   take the biggest possible model -> make it the smallest highest-performing form.
#
# Unlike llama.cpp (pick a fixed Q-format and accept the loss), Hawking condenses:
# lowest-bit by default, dynamic per-tensor, and with an optional RECOVERY pass
# (the doctor) that heals the quality back toward 1:1. The product is one verb:
#
#     hawking condense <model>   ->   hawking run <model.tq>
#
# Stages:  PLAN -> [RECOVER (doctor)] -> ENCODE+VERIFY (tq_bake) -> quality CARD
#
# Usage:
#   tools/condense/condense.sh <source.gguf|hf-dir> [opts]
#     --bits N        target bits (default 3; 2 = the lead — pair with --recover)
#     --out PATH      output .tq (default: <source>.b<bits>.tq alongside source)
#     --recover[=S]   run the QAT/KD doctor first (heals quality; HEAVY/deferred)
#     --kd            doctor uses knowledge distillation vs the f16 teacher
#     --match PAT     only condense tensors matching PAT (default: all)
#     --hf DIR        HF twin dir for --recover when source is a GGUF
#     --dry-run       print the plan, run nothing (battery-safe)
#     --allow-quant-source   permit a Q* GGUF input — PLUMBING ONLY: this is
#                            quant-of-quant, NOT a quality artifact (rule: bake
#                            quality from f16/f32 sources only)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"

[ $# -ge 1 ] || { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
SRC="$1"; shift

BITS=3; OUT=""; RECOVER=0; RSTEPS=400; KD=""; MATCH=""; HF=""; DRY=0; ALLOWQ=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bits)  BITS="$2"; shift 2;;
    --out)   OUT="$2"; shift 2;;
    --recover)        RECOVER=1; shift;;
    --recover=*)      RECOVER=1; RSTEPS="${1#*=}"; shift;;
    --kd)    KD="--kd"; shift;;
    --match) MATCH="$2"; shift 2;;
    --hf)    HF="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --allow-quant-source) ALLOWQ=1; shift;;
    *) echo "condense: unknown arg '$1'" >&2; exit 2;;
  esac
done

[ -e "$SRC" ] || { echo "condense: source not found: $SRC" >&2; exit 2; }

# bits -> bpw payload (matches TrellisConfig::for_bpw_quality targets)
case "$BITS" in
  1) BPW=1.34;; 2) BPW=2.34;; 3) BPW=3.34;; 4) BPW=4.50;;
  *) echo "condense: --bits must be 1|2|3|4 (got $BITS)" >&2; exit 2;;
esac

# f16-source rule: refuse quant-of-quant unless explicitly overridden (plumbing).
case "$SRC" in
  *.gguf)
    if echo "$SRC" | grep -qiE 'Q[0-9]+_[0-9KMS]'; then
      if [ "$ALLOWQ" = 1 ]; then
        echo "condense: WARNING quant source ($SRC) — PLUMBING ONLY, not a quality artifact"
      else
        echo "condense: REFUSING quant source ($SRC) — bake quality from f16/f32 only." >&2
        echo "          convert to f16 GGUF first, or pass --allow-quant-source for plumbing tests." >&2
        exit 3
      fi
    fi ;;
esac

[ -n "$OUT" ] || OUT="${SRC%.gguf}.b${BITS}.tq"
SRCBYTES=$(stat -f%z "$SRC" 2>/dev/null || du -sk "$SRC" | awk '{print $1*1024}')

echo "════════════ hawking condense ════════════"
echo "  source : $SRC ($(awk -v b="$SRCBYTES" 'BEGIN{printf "%.2f GB", b/1e9}'))"
echo "  target : ${BITS}-bit (~${BPW} bpw)   out: $OUT"
echo "  recover: $([ "$RECOVER" = 1 ] && echo "yes (doctor, ${RSTEPS} steps${KD:+ +KD})" || echo "no (PTQ only)")"
echo "──────────────────────────────────────────"

# 1) PLAN — predicted footprint (lever: density vs Q4_K's ~4.5 bpw)
echo "[1/4] PLAN  : ~${BPW} bpw vs Q4_K 4.50 bpw  =>  ~$(awk -v b="$BPW" 'BEGIN{printf "%.0f%%", (1-b/4.50)*100}') denser"

# 2) RECOVER — the doctor (heals quality; HEAVY — deferred unless invoked plugged in)
if [ "$RECOVER" = 1 ]; then
  RDIR="$HF"; [ -n "$RDIR" ] || RDIR="$(echo "${SRC%.gguf}" | sed -E 's/-(Q[0-9].*)?$//')-hf"
  echo "[2/4] HEAL  : doctor on $RDIR @ ${BITS}-bit"
  if [ "$DRY" = 1 ]; then
    bash tools/condense/doctor.sh "$RDIR" --bits "$BITS" --steps "$RSTEPS" $KD --dry-run
  else
    bash tools/condense/doctor.sh "$RDIR" --bits "$BITS" --steps "$RSTEPS" $KD \
      || { echo "condense: doctor failed (recovery skipped)" >&2; }
  fi
else
  echo "[2/4] HEAL  : skipped (PTQ only — quality caps ~1.28x Q4_K at 3-bit; 2-bit needs --recover)"
fi

# 3) ENCODE + VERIFY — tq_bake produces the servable .tq and self-checks (round-trip)
BAKE=(cargo run --release -q -p tq_bake_tool -- "$SRC" "$OUT" --bpw "$BPW")
[ -n "$MATCH" ] && BAKE+=(--match "$MATCH")
echo "[3/4] ENCODE: ${BAKE[*]}"
if [ "$DRY" = 1 ]; then
  echo "        (dry-run — not executed)"
else
  "${BAKE[@]}" || { echo "condense: encode failed" >&2; exit 4; }
fi

# 4) CARD — the artifact's quality/size summary
echo "[4/4] CARD"
if [ "$DRY" = 0 ] && [ -f "$OUT" ]; then
  OB=$(stat -f%z "$OUT" 2>/dev/null || echo 0)
  echo "  $OUT  $(awk -v b="$OB" 'BEGIN{printf "%.2f GB", b/1e9}')  (source $(awk -v s="$SRCBYTES" -v o="$OB" 'BEGIN{printf "%.2fx smaller", s/o}'))"
  echo "  run:  hawking generate --weights $OUT   # (lower bits => higher tps)"
else
  echo "  (planned — run without --dry-run to produce + size the artifact)"
fi
echo "══════════════════════════════════════════"
