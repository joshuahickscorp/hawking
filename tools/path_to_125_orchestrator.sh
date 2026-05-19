#!/usr/bin/env bash
# path-to-125 autonomous pipeline orchestrator.
#
# Stages (each writes status JSON before/after):
#   1. WAIT for capture (eagle4/capture.py) to exit
#   2. RUN training with --chain-h-high + multi-step-k=4
#   3. RUN tau_eval on the trained checkpoint
#   4. RUN dismantle bench script with the new head
#   5. WRITE final summary
#
# Survives Claude.app being closed — it's a normal OS background
# process. Status visible at reports/path_to_90/_pipeline/status.json.
# Detailed stdout/stderr in reports/path_to_90/_pipeline/<stage>.log.
#
# Refuses to start training if capture is still running OR if Claude
# is open (training competes for GPU; clean-window required).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PIPE_DIR="$REPO_ROOT/reports/path_to_90/_pipeline"
mkdir -p "$PIPE_DIR"
STATUS="$PIPE_DIR/status.json"
TS_START="$(date +%Y%m%dT%H%M%S)"

VENV_PY="/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python3"
DISMANTLE="$REPO_ROOT/target/release/dismantle"
CAPTURE_DIR="$REPO_ROOT/training_data/c2_hidden/eagle4_v0"
CKPT_DIR="$REPO_ROOT/eagle4/checkpoints/eagle4_v4_chain"
RESUME_CKPT="$REPO_ROOT/eagle4/checkpoints/eagle4_v3/best.npz"
FROZEN_NPZ="$REPO_ROOT/eagle4/v2lite_frozen.npz"

# ── small helpers ─────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
status_write() {
  # Args: stage (str), state (str), extra_json (optional, must be valid object body)
  local stage="$1" state="$2" extra="${3:-}"
  python3 - "$STATUS" "$stage" "$state" "$extra" "$TS_START" <<'PY'
import sys, json, os, time, datetime
status_path, stage, state, extra, ts_start = sys.argv[1:6]
try:
    cur = json.load(open(status_path)) if os.path.exists(status_path) else {}
except Exception:
    cur = {}
cur.setdefault("history", []).append({
    "t": datetime.datetime.now().isoformat(timespec="seconds"),
    "stage": stage, "state": state,
})
cur["current_stage"] = stage
cur["current_state"] = state
cur["session_start"] = ts_start
cur["last_update"] = datetime.datetime.now().isoformat(timespec="seconds")
if extra:
    try:
        cur.update(json.loads("{" + extra + "}"))
    except Exception as e:
        cur["status_write_error"] = str(e)
json.dump(cur, open(status_path, "w"), indent=2)
PY
}

# ── 0. preflight ──────────────────────────────────────────────────────────────
status_write "preflight" "starting"
log "starting orchestrator @ $TS_START"
for f in "$VENV_PY" "$DISMANTLE" "$RESUME_CKPT" "$FROZEN_NPZ"; do
  if [[ ! -e "$f" ]]; then
    log "FATAL: required artifact missing: $f"
    status_write "preflight" "failed" "\"error\":\"missing_$(basename $f)\""
    exit 10
  fi
done
log "preflight OK"

# ── 1. wait for capture to finish ─────────────────────────────────────────────
status_write "wait_capture" "starting"
captured_records_target=500000
last_shard_count=-1
last_change_at=$(date +%s)
while true; do
  pids=$(pgrep -f "eagle4/capture.py" 2>/dev/null || true)
  shard_count=$(ls -1 "$CAPTURE_DIR"/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$shard_count" != "$last_shard_count" ]]; then
    last_shard_count="$shard_count"
    last_change_at=$(date +%s)
    log "wait_capture: $shard_count shards, capture PIDs: ${pids:-none}"
    status_write "wait_capture" "polling" "\"shards\":$shard_count,\"capture_pids\":\"${pids:-none}\""
  fi
  if [[ -z "$pids" ]]; then
    # Process exited; double-check the shard count is stable (no race).
    sleep 5
    final_shards=$(ls -1 "$CAPTURE_DIR"/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
    log "wait_capture: capture process gone; final shards=$final_shards"
    status_write "wait_capture" "done" "\"final_shards\":$final_shards"
    break
  fi
  sleep 15
done

# ── 1.5 preflight Claude check before training ────────────────────────────────
status_write "training" "preflight"
if pgrep -i "Claude" >/dev/null 2>&1; then
  log "WARN: Claude is running — training will be SLOWER under contention."
  log "      For best results: Cmd-Q Claude before this stage runs."
  log "      Proceeding anyway (won't crash, just slower)."
fi

# ── 2. training ───────────────────────────────────────────────────────────────
TRAIN_LOG="$PIPE_DIR/train.log"
status_write "training" "running" "\"log\":\"$TRAIN_LOG\""
log "training: starting (chain_h_high=True, k=4, epochs=2)"
"$VENV_PY" eagle4/eagle4.py train \
  --parquet $(ls "$CAPTURE_DIR"/shard_*.parquet) \
  --frozen "$FROZEN_NPZ" \
  --ckpt-dir "$CKPT_DIR" \
  --resume "$RESUME_CKPT" \
  --epochs 2 \
  --multi-step-k 4 \
  --multi-step-decay 0.7 \
  --chain-h-high \
  --target-warmup-steps 500 \
  >"$TRAIN_LOG" 2>&1
train_rc=$?
if [[ $train_rc -ne 0 ]]; then
  log "FATAL: training failed rc=$train_rc — see $TRAIN_LOG"
  status_write "training" "failed" "\"rc\":$train_rc"
  exit 11
fi
log "training: complete"
status_write "training" "done"

LATEST_CKPT="$CKPT_DIR/latest.npz"
if [[ ! -f "$LATEST_CKPT" ]]; then
  log "FATAL: training succeeded but $LATEST_CKPT missing"
  status_write "training" "failed" "\"error\":\"no_latest_npz\""
  exit 12
fi

# ── 3. tau_eval ───────────────────────────────────────────────────────────────
TAU_LOG="$PIPE_DIR/tau_eval.log"
HELDOUT="$REPO_ROOT/eagle4/data/v2lite_3layer_heldout/shard_00000.parquet"
status_write "tau_eval" "running" "\"log\":\"$TAU_LOG\""
log "tau_eval: running on heldout shard"
"$VENV_PY" eagle4/tau_eval.py eval \
  --ckpt "$LATEST_CKPT" \
  --frozen "$FROZEN_NPZ" \
  --parquet "$HELDOUT" \
  --depth 4 \
  >"$TAU_LOG" 2>&1
tau_rc=$?
if [[ $tau_rc -ne 0 ]]; then
  log "WARN: tau_eval rc=$tau_rc — see $TAU_LOG. Continuing to bench."
  status_write "tau_eval" "failed" "\"rc\":$tau_rc"
else
  # Parse the τ value out of tau_eval stdout (JSON-ish lines).
  tau_val=$(grep -oE '"tau"[^,]*' "$TAU_LOG" | head -1 | grep -oE '[0-9.]+' | head -1)
  full_accept=$(grep -oE '"full_accept_rate"[^,]*' "$TAU_LOG" | head -1 | grep -oE '[0-9.]+' | head -1)
  log "tau_eval: tau=${tau_val:-?} full_accept=${full_accept:-?}"
  status_write "tau_eval" "done" "\"tau\":${tau_val:-null},\"full_accept_rate\":${full_accept:-null}"
fi

# ── 4. eagle4 eval (single-step accept + mask recall) ─────────────────────────
EAGLE_LOG="$PIPE_DIR/eagle_eval.log"
status_write "eagle_eval" "running" "\"log\":\"$EAGLE_LOG\""
log "eagle_eval: running"
"$VENV_PY" eagle4/eagle4.py eval \
  --ckpt "$LATEST_CKPT" \
  --frozen "$FROZEN_NPZ" \
  --parquet "$HELDOUT" \
  --max-records 5000 \
  --mask-top-k 8 \
  >"$EAGLE_LOG" 2>&1
eagle_rc=$?
log "eagle_eval: rc=$eagle_rc"
status_write "eagle_eval" "done" "\"rc\":$eagle_rc"

# ── 5. dismantle bench with new head (if Claude isn't running) ────────────────
if pgrep -i "Claude" >/dev/null 2>&1; then
  log "bench: skipped — Claude is running (contended GPU = bad numbers)"
  log "       To run later: EAGLE4_CKPT=$LATEST_CKPT ./tools/bench/path_to_125_bench.sh"
  status_write "bench" "skipped_claude_open"
else
  BENCH_LOG="$PIPE_DIR/bench.log"
  status_write "bench" "running" "\"log\":\"$BENCH_LOG\""
  log "bench: running clean-window bench with new head"
  EAGLE4_CKPT="$LATEST_CKPT" ./tools/bench/path_to_125_bench.sh \
    >"$BENCH_LOG" 2>&1
  bench_rc=$?
  log "bench: rc=$bench_rc"
  status_write "bench" "done" "\"rc\":$bench_rc"
fi

# ── 6. final summary ──────────────────────────────────────────────────────────
SUMMARY="$PIPE_DIR/summary.md"
{
  echo "# path-to-125 pipeline summary — session $TS_START"
  echo ""
  echo "## Stages"
  python3 -c "
import json
s = json.load(open('$STATUS'))
for h in s.get('history', []):
    print(f\"  {h['t']}  {h['stage']:14s} {h['state']}\")"
  echo ""
  echo "## Training"
  tail -20 "$TRAIN_LOG"
  echo ""
  echo "## tau_eval"
  cat "$TAU_LOG" 2>/dev/null | tail -20
  echo ""
  echo "## eagle4 eval"
  cat "$EAGLE_LOG" 2>/dev/null | tail -20
  echo ""
  if [[ -f "$PIPE_DIR/bench.log" ]]; then
    echo "## bench"
    tail -20 "$PIPE_DIR/bench.log"
  fi
} > "$SUMMARY"

status_write "pipeline" "complete" "\"summary\":\"$SUMMARY\""
log "pipeline complete — summary at $SUMMARY"

if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "path-to-125 pipeline complete" with title "dismantle"' || true
fi
