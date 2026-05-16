#!/usr/bin/env bash
# tools/training/chain_capture.sh
#
# Autonomous chain runner for path-to-90 C2 capture.
#
# Watches the in-flight `dismantle capture-hidden` run for its completion
# sentinel (the .meta.json file is only written when the loop exits cleanly),
# then:
#   1. Auto-converts the binary shard to parquet (judgment-free, ~5 sec).
#   2. Prints a one-line summary so the notification message is informative.
#   3. EXITS, which fires a completion notification back to Claude.
#
# Claude (active again post-notification) then does the analysis,
# decides whether to kick off the next-tier capture (50K extension), and
# manually launches it.
#
# If Claude does NOT activate within $FALLBACK_DELAY seconds after capture
# completion, this script falls back to auto-launching the 50K capture so
# the long weekend's compute doesn't sit idle. The fallback writes a marker
# file `chain.auto_fallback_fired` so Claude knows on wake-up that the 50K
# was started automatically rather than from his decision.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

SHARD_BIN="training_data/c2_hidden/eagle3_v0/shard_000.bin"
SHARD_META="training_data/c2_hidden/eagle3_v0/shard_000.meta.json"
SHARD_PARQUET="training_data/c2_hidden/eagle3_v0/shard_000.parquet"
SHARD_LOG="training_data/c2_hidden/eagle3_v0/shard_000.log"
CHAIN_LOG="training_data/c2_hidden/eagle3_v0/chain.log"
CHAIN_SUMMARY="training_data/c2_hidden/eagle3_v0/chain.summary.txt"
CHAIN_FALLBACK_MARKER="training_data/c2_hidden/eagle3_v0/chain.auto_fallback_fired"

PY="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
DISMANTLE="./target/release/dismantle"
WEIGHTS="models/deepseek-v2-lite-q4.gguf"
PROFILE="profiles/deepseek-v2-lite-q4.m3pro18.json"
SAMPLES_5K="tests/data/ultrachat_5k.jsonl"
SAMPLES_50K="tests/data/ultrachat_50k.jsonl"

# Seconds to wait for Claude to interpret + kick off 50K manually before
# auto-firing as fallback. 3600 = 1 hour. Set to 0 to disable fallback.
FALLBACK_DELAY="${FALLBACK_DELAY:-3600}"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$CHAIN_LOG"; }

log "chain runner up; waiting for $SHARD_META"

# ---- Phase 1: wait for capture completion -------------------------------
# meta.json is only written when capture_hidden_main exits cleanly past the
# main loop, so its mere existence is the completion sentinel.
PHASE1_START=$(date +%s)
while [ ! -f "$SHARD_META" ]; do
  sleep 30
done
PHASE1_END=$(date +%s)
log "capture completed (waited $((PHASE1_END - PHASE1_START))s); meta.json present"

# Verify the capture actually processed all 5000 samples (defensive — meta
# is also written by short re-runs that finish quickly).
SAMPLES_DONE=$("$PY" -c "import json,sys; print(json.load(open('$SHARD_META'))['samples_processed'])")
RECORDS=$("$PY" -c "import json,sys; print(json.load(open('$SHARD_META'))['records'])")
log "samples_processed=$SAMPLES_DONE records=$RECORDS"

if [ "$SAMPLES_DONE" -lt 5000 ]; then
  log "WARN: only $SAMPLES_DONE samples; expected 5000. Exiting without parquet conversion or fallback."
  echo "INCOMPLETE: $SAMPLES_DONE / 5000 samples" > "$CHAIN_SUMMARY"
  exit 2
fi

# ---- Phase 2: parquet conversion ----------------------------------------
log "converting $SHARD_BIN -> $SHARD_PARQUET"
if "$PY" tools/training/capture_hidden.py to-parquet \
    --src "$SHARD_BIN" --dst "$SHARD_PARQUET" --compression zstd \
    >> "$CHAIN_LOG" 2>&1; then
  PARQUET_SIZE=$(stat -f %z "$SHARD_PARQUET")
  log "parquet OK: $PARQUET_SIZE bytes"
else
  log "parquet conversion FAILED — see $CHAIN_LOG"
fi

# ---- Phase 3: summary line (becomes the notification body) --------------
{
  echo "===== C2 chain phase 1 complete ====="
  echo "  samples_processed: $SAMPLES_DONE"
  echo "  records          : $RECORDS"
  echo "  shard_bin        : $SHARD_BIN ($(du -h "$SHARD_BIN" | cut -f1))"
  if [ -f "$SHARD_PARQUET" ]; then
    echo "  shard_parquet    : $SHARD_PARQUET ($(du -h "$SHARD_PARQUET" | cut -f1))"
  fi
  echo "  capture wall     : see $SHARD_META 'elapsed_s_last_run'"
  echo
  echo "Next-step expectation (per stage3_c2/training_brief.md):"
  echo "  - Claude interprets results (vocab coverage, hidden distribution, etc.)"
  echo "  - Claude kicks off 50K extension capture with --resume"
  echo "  - Fallback: if no manual kickoff within ${FALLBACK_DELAY}s, this script"
  echo "    auto-launches the 50K extension."
} | tee "$CHAIN_SUMMARY"
log "phase 1 summary written; notifying claude"

# Emit the line that the foreground Bash run_in_background watcher matches.
echo "CHAIN_PHASE_1_COMPLETE samples=$SAMPLES_DONE records=$RECORDS"

# ---- Phase 4: fallback timer --------------------------------------------
if [ "$FALLBACK_DELAY" -le 0 ]; then
  log "fallback disabled; exiting"
  exit 0
fi

log "fallback timer armed: $FALLBACK_DELAY s. Exit conditions:"
log "  - $CHAIN_FALLBACK_MARKER created by Claude (means he kicked off 50K manually)"
log "  - timer expires (auto-fire)"

WAITED=0
while [ "$WAITED" -lt "$FALLBACK_DELAY" ]; do
  if [ -f "$CHAIN_FALLBACK_MARKER" ]; then
    log "claude kicked off 50K manually (marker found); exiting fallback"
    exit 0
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

log "FALLBACK FIRING: auto-launching 50K extension"
touch "$CHAIN_FALLBACK_MARKER"
echo "fallback-auto-fired" > "$CHAIN_FALLBACK_MARKER"

# Prep 50K samples if not already.
if [ ! -f "$SAMPLES_50K" ]; then
  log "prep 50K samples → $SAMPLES_50K"
  "$PY" tools/training/capture_hidden.py prep \
    --out "$SAMPLES_50K" \
    --dataset HuggingFaceH4/ultrachat_200k \
    --split train_sft \
    --streaming \
    --n 50000 \
    --min-chars 200 --max-chars 2000 \
    --id-prefix ultrachat \
    --force >> "$CHAIN_LOG" 2>&1 \
    || { log "prep FAILED"; exit 3; }
fi

# Launch 50K extension (resume on the same shard).
log "launching 50K extension capture"
nohup nice -n 19 taskpolicy -b "$DISMANTLE" capture-hidden \
  --weights "$WEIGHTS" \
  --samples "$SAMPLES_50K" \
  --out "$SHARD_BIN" \
  --max-tokens 128 \
  --no-lm-head \
  --resume \
  --kernel-profile "$PROFILE" \
  >> "$SHARD_LOG" 2>&1 < /dev/null &
NEW_PID=$!
disown
log "50K capture launched as PID $NEW_PID; chain runner exiting"
echo "CHAIN_PHASE_2_FALLBACK_FIRED pid=$NEW_PID"
exit 0
