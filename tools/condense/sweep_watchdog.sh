#!/usr/bin/env bash
# Hawking condense parameter-sweep — detached Studio launcher (the long-test + watchdog
# pattern). Runs the bit-floor search across the ladder in PHASE/PRIORITY order so the
# clean Qwen2.5 curve lands first, then cross-family, then the 100B+/MoE frontier. Each
# phase logs independently (a slow later phase never loses an earlier result); sweep.py is
# idempotent so a kill+relaunch resumes from the JSONL.
#
# Mac Studio (96 GB):   bash tools/condense/sweep_watchdog.sh studio   &   # detached
# This 19 GB box (slow, ≤7B only):  bash tools/condense/sweep_watchdog.sh here
#
# Scaffold note: NOT auto-launched. Phase-2 (>34B) condense is recorded pending until the
# block-wise condenser is built; Stream-B projection (size/fit/cliff) runs for every size.
set -uo pipefail
cd "$HOME/Downloads/hawking" || exit 2
PROFILE="${1:-studio}"
LOG=reports/cron/sweep.md; mkdir -p reports/cron
PY=python3.12
S=(tools/condense/sweep.py --profile "$PROFILE" --go)

echo "# Hawking condense sweep ($PROFILE)" > "$LOG"; date >> "$LOG"
{
  echo ""; echo "## Phase 1 — Qwen2.5 spine (P0, the clean scaling curve)"; date
  $PY "${S[@]}" --max-prio 0 --max-params 34 2>&1 | tail -40

  echo ""; echo "## Phase 1 — cross-family ≤34B (P1, generality)"; date
  $PY "${S[@]}" --max-prio 1 --max-params 34 2>&1 | tail -60

  echo ""; echo "## Phase 1 — serve projection + 100B+/MoE (P2; condense pending phase-2)"; date
  $PY "${S[@]}" --max-prio 2 2>&1 | tail -60

  echo ""; echo "## Matrix"; date
  $PY tools/condense/sweep_render.py 2>&1 | tail -3
  echo "→ reports/condense/MATRIX.md"
  echo ""; echo "## DONE"; date
} >> "$LOG" 2>&1
