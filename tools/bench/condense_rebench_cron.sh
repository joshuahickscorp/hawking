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
  echo "[cron] clean rebench start ${STAMP}"
  echo "[cron] battery: $(pmset -g batt 2>/dev/null | tr '\n' ' ')"
  cargo build --release -p hawking 2>&1 | tail -2
  # Full rigorous bench, CLEAN (cron has no Claude session inflating tps).
  TRIALS=5 TOK=256 BIT_TARGETS=8,6,5,4,3,2,1 STRICT_CLEAN=0 \
    OUT="reports/sota-compare/cron-${STAMP}" bash tools/bench/compare_sota.sh
  echo "[cron] done ${STAMP} — report: reports/sota-compare/cron-${STAMP}/report.md"
} >>"$LOG" 2>&1
