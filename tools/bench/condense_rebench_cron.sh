#!/usr/bin/env bash
# One-shot cron runner: the deferred CLEAN rigorous rebench (Hawking vs llama.cpp
# vs MLX) — scheduled for ~3h out so it runs when the Mac is recharged, NOT on
# battery. Self-removes its own crontab entry so it fires exactly once.
#
# Produces the clean-absolute fair-comparison + the condensation density ladder
# (the dirty in-session run inflated absolutes; this is the trustworthy one).
#
# Caveat: macOS cron does NOT wake a sleeping Mac — leave it awake + plugged.
set -uo pipefail
cd "$HOME/Downloads/hawking" || exit 2

# one-shot: strip self from crontab before running
( crontab -l 2>/dev/null | grep -v 'condense_rebench_cron' | crontab - ) 2>/dev/null || true

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p reports/cron
LOG="reports/cron/rebench_${STAMP}.log"
{
  echo "[cron] start ${STAMP}"
  echo "[cron] battery: $(pmset -g batt 2>/dev/null | tr '\n' ' ')"
  cargo build --release -p hawking 2>&1 | tail -2

  # ── 1. condensation QUALITY: real ppl + the DOCTOR (QAT) recovery — the thesis ──
  # DIVERSE calib corpus (else the doctor overfits one passage -> loses; see verdict v1).
  CORPUS=scratch/calib_corpus.txt
  if [ ! -s "$CORPUS" ]; then
    python3.12 - <<'PY' 2>/dev/null || echo "[cron] corpus gen failed"
from datasets import load_dataset
ds = load_dataset("wikitext","wikitext-2-raw-v1",split="train")
open("scratch/calib_corpus.txt","w").write("\n".join(t for t in ds["text"] if len(t.strip())>80)[:400000])
PY
  fi
  export DOCTOR_CALIB="$CORPUS"
  echo "[cron] === ppl sweep (f16 vs TQ3 vs TQ2) ==="
  bash tools/condense/quality_sweep.sh scratch/qwen-05b 3,2 2>&1 | tail -20
  echo "[cron] === DOCTOR (self-CE, diverse calib): 2-bit, 300 steps (the money shot) ==="
  python3.12 tools/condense/doctor_qat.py 2 300 3e-5 scratch/qwen-05b-healed2.safetensors 2>&1 | tail -30 || echo "[cron] doctor-2 failed"
  echo "[cron] === PRODUCT bridge: STRAND-bake the healed shadow -> real TQ2 ppl ==="
  vendor/strand-quant/target/release/quantize-model --in scratch/qwen-05b-healed2.raw.safetensors \
    --out scratch/qwen-05b-healed2-strand.safetensors --bits 2 --quality --rht-cols \
    --outlier-channel 1 --outlier-bits 8 2>&1 | tail -2 || echo "[cron] strand re-bake failed"
  python3.12 tools/condense/ppl_bench.py scratch/qwen-05b scratch/qwen-05b-healed2-strand.safetensors "tq2+doctor+STRAND" 2>/dev/null || echo "[cron] strand ppl failed"
  echo "[cron] === 3-WAY QUALITY: healed TQ2 vs llama Q4_K (the win check) ==="
  bash tools/condense/quality_3way.sh scratch/qwen-05b-healed2-strand.safetensors "tq2+doctor" 2>&1 | tail -10 || echo "[cron] 3way failed"
  echo "[cron] === DOCTOR (KD): distillation 2-bit, 300 steps ==="
  KD=1 python3.12 tools/condense/doctor_qat.py 2 300 2e-5 scratch/qwen-05b-healed2kd.safetensors 2>&1 | tail -12 || echo "[cron] doctor-2-kd failed"
  echo "[cron] === DOCTOR: 3-bit, 200 steps ==="
  python3.12 tools/condense/doctor_qat.py 3 200 2e-5 scratch/qwen-05b-healed3.safetensors 2>&1 | tail -8 || echo "[cron] doctor-3 failed"

  # ── 2. tps/footprint: full rigorous bench, CLEAN (cron has no agent inflating tps) ──
  echo "[cron] === SOTA bench (Hawking vs llama vs MLX) ==="
  TRIALS=5 TOK=256 BIT_TARGETS=8,6,5,4,3,2,1 STRICT_CLEAN=0 \
    OUT="reports/sota-compare/cron-${STAMP}" bash tools/bench/compare_sota.sh
  echo "[cron] done ${STAMP} — reports/sota-compare/cron-${STAMP}/report.md + reports/condense/"
} >>"$LOG" 2>&1

# clean morning summary (reads the log's JSON lines + the ppl sweep)
python3.12 tools/condense/overnight_summary.py "$LOG" reports/condense/ppl_sweep.jsonl >> "$LOG" 2>&1 || true
echo "[cron] summary -> reports/condense/OVERNIGHT_RESULTS.md" >> "$LOG" 2>&1
