#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_DIR="artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8"
WATCH_LOG="$RUN_DIR/autoinject_watch.log"

mkdir -p "$RUN_DIR"
exec >>"$WATCH_LOG" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

latest_checkpoint_dir() {
  find "$RUN_DIR" -maxdepth 1 -type d -name 'step_[0-9][0-9][0-9][0-9][0-9][0-9]' -print | sort | tail -n 1
}

checkpoint_step() {
  basename "$1" | sed -E 's/^step_0*([0-9]+)$/\1/'
}

find_current_pid() {
  ps ax -o pid= -o command= | awk -v out="$RUN_DIR" '
    /[r]wkv7_qat.py/ && index($0, "--out " out) { print $1; exit }
  '
}

file_size() {
  stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null
}

wait_for_stable_file() {
  local file="$1"
  local last_size=""
  local stable_count=0
  local size=""

  while true; do
    if [[ -s "$file" ]]; then
      size="$(file_size "$file")"
      if [[ "$size" == "$last_size" ]]; then
        stable_count=$((stable_count + 1))
      else
        stable_count=0
        last_size="$size"
      fi
      if (( stable_count >= 2 )); then
        return 0
      fi
    fi
    sleep 10
  done
}

START_CKPT="$(latest_checkpoint_dir || true)"
if [[ -z "$START_CKPT" ]]; then
  log "no starting checkpoint found in $RUN_DIR"
  exit 1
fi

START_STEP="$(checkpoint_step "$START_CKPT")"
OLD_PID="${1:-$(find_current_pid || true)}"
log "watching from step $START_STEP; current pid=${OLD_PID:-none}"

NEXT_CKPT=""
while true; do
  CANDIDATE="$(latest_checkpoint_dir || true)"
  if [[ -n "$CANDIDATE" ]]; then
    CANDIDATE_STEP="$(checkpoint_step "$CANDIDATE")"
    if (( CANDIDATE_STEP > START_STEP )); then
      NEXT_CKPT="$CANDIDATE"
      log "found next checkpoint: $NEXT_CKPT"
      break
    fi
  fi

  if [[ -n "$OLD_PID" ]] && ! kill -0 "$OLD_PID" 2>/dev/null; then
    log "old process exited before a newer checkpoint appeared"
    exit 1
  fi

  sleep 30
done

wait_for_stable_file "$NEXT_CKPT/state_dict.pt"
log "checkpoint is stable: $NEXT_CKPT/state_dict.pt"

if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
  log "stopping old process $OLD_PID after checkpoint"
  kill -TERM "$OLD_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    if ! kill -0 "$OLD_PID" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  if kill -0 "$OLD_PID" 2>/dev/null; then
    log "old process still alive; forcing stop"
    kill -KILL "$OLD_PID" 2>/dev/null || true
  fi
fi

RELAUNCH_LOG="$RUN_DIR/autoinject_relaunch_$(date '+%Y%m%d_%H%M%S').log"
log "relaunching from $NEXT_CKPT; training log: $RELAUNCH_LOG"

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 nohup .venv-rwkv/bin/python tools/training/rwkv7_qat.py \
  --model models/rwkv7-g1-04-hf/model.safetensors \
  --hf-dir models/rwkv7-g1-04-hf \
  --data artifacts/rwkv7_posttrain/sft.jsonl \
  --out "$RUN_DIR" \
  --stage ffn --quant ternary --last-n-layers 8 \
  --max-length 1024 --grad-accum 16 --lr 5e-6 \
  --epochs 1 --save-every 25 --eval-every 0 --eval-tokens 4096 \
  --device mps --run-id g1a \
  --resume-from "$NEXT_CKPT" \
  >"$RELAUNCH_LOG" 2>&1 &

NEW_PID="$!"
printf '%s\n' "$NEW_PID" > "$RUN_DIR/autoinject_relaunch.pid"
log "started new process pid=$NEW_PID"
