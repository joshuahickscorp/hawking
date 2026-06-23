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
  echo "[cron] === ppl sweep (f16 vs TQ3 vs TQ2) ==="
  bash tools/condense/quality_sweep.sh scratch/qwen-05b 3,2 2>&1 | tail -20
  echo "[cron] === DOCTOR: QAT recovery 2-bit, 300 steps (the money shot) ==="
  python3.12 tools/condense/doctor_qat.py 2 300 2e-5 scratch/qwen-05b-healed2.safetensors 2>&1 | tail -30
  echo "[cron] === healed-2bit held-out ppl via ppl_bench ==="
  python3.12 tools/condense/ppl_bench.py scratch/qwen-05b scratch/qwen-05b-healed2.safetensors "tq2+doctor" 2>/dev/null
  echo "[cron] === DOCTOR: 3-bit, 200 steps ==="
  python3.12 tools/condense/doctor_qat.py 3 200 2e-5 scratch/qwen-05b-healed3.safetensors 2>&1 | tail -8

  # ── 2. tps/footprint: full rigorous bench, CLEAN (cron has no Claude inflating tps) ──
  echo "[cron] === SOTA bench (Hawking vs llama vs MLX) ==="
  TRIALS=5 TOK=256 BIT_TARGETS=8,6,5,4,3,2,1 STRICT_CLEAN=0 \
    OUT="reports/sota-compare/cron-${STAMP}" bash tools/bench/compare_sota.sh
  echo "[cron] done ${STAMP} — reports/sota-compare/cron-${STAMP}/report.md + reports/condense/"
} >>"$LOG" 2>&1
