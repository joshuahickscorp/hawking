#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_DIR="artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8"
LOG="$RUN_DIR/autocycle_step50_ozempic.log"
PID_FILE="$RUN_DIR/autocycle_step50_ozempic.pid"
RELAUNCH_PID_FILE="$RUN_DIR/autocycle_relaunch.pid"
POLICY_LOG="$RUN_DIR/autocycle_decisions.jsonl"
TARGET_STEPS="${OZEMPIC_TARGET_STEPS:-50 75 100 125}"
COOLDOWN_SECONDS="${OZEMPIC_COOLDOWN_SECONDS:-300}"
PRUNE_POSTTRAIN="${OZEMPIC_PRUNE_POSTTRAIN:-1}"
DETERMINISTIC_SEED="${OZEMPIC_SEED:-1337}"
DETERMINISTIC="${OZEMPIC_DETERMINISTIC:-1}"
PRETOKENIZE_WORKERS="${OZEMPIC_PRETOKENIZE_WORKERS:-2}"
USE_CHUNKED="${OZEMPIC_USE_CHUNKED:-1}"
CHUNK_SIZE="${OZEMPIC_CHUNK_SIZE:-32}"
MPS_EMPTY_CACHE_EVERY="${OZEMPIC_MPS_EMPTY_CACHE_EVERY:-1}"
MPS_HIGH_WATERMARK_RATIO="${OZEMPIC_MPS_HIGH_WATERMARK_RATIO:-0.0}"
QAT_MAX_LENGTH="${OZEMPIC_MAX_LENGTH:-1024}"
QAT_GRAD_ACCUM="${OZEMPIC_GRAD_ACCUM:-16}"
QAT_LR="${OZEMPIC_LR:-5e-6}"
QAT_RUN_ID="${OZEMPIC_RUN_ID:-g1a}"
SAVE_EVERY="${OZEMPIC_SAVE_EVERY:-25}"
KEEP_LAST_N_CHECKPOINTS="${OZEMPIC_KEEP_LAST_N_CHECKPOINTS:-2}"
TARGET_INTERVAL="${OZEMPIC_TARGET_INTERVAL:-25}"
MAX_AUTO_STEP="${OZEMPIC_MAX_AUTO_STEP:-200}"
MIN_EMA_DROP="${OZEMPIC_MIN_EMA_DROP:-0.015}"
HOT_LOSS_UNDER="${OZEMPIC_HOT_LOSS_UNDER:-6.0}"
EMA_CEILING="${OZEMPIC_EMA_CEILING:-6.7}"
MAX_EMA_RISE="${OZEMPIC_MAX_EMA_RISE:-0.12}"
HANDOFF_EMA_TARGET="${OZEMPIC_HANDOFF_EMA_TARGET:-}"
HANDOFF_AT_STEP="${OZEMPIC_HANDOFF_AT_STEP:-0}"
HANDOFF_MIN_STEP="${OZEMPIC_HANDOFF_MIN_STEP:-0}"
HANDOFF_COMMAND="${OZEMPIC_HANDOFF_COMMAND:-}"
HANDOFF_STOP_AFTER="${OZEMPIC_HANDOFF_STOP_AFTER:-1}"

mkdir -p "$RUN_DIR"
exec >>"$LOG" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

step_dir() {
  printf '%s/step_%06d' "$RUN_DIR" "$1"
}

checkpoint_step() {
  basename "$1" | sed -E 's/^step_0*([0-9]+)$/\1/'
}

latest_checkpoint_before() {
  local limit_step="$1"
  local dir=""
  local step=""
  local best=0

  for dir in "$RUN_DIR"/step_[0-9][0-9][0-9][0-9][0-9][0-9]; do
    [[ -d "$dir" ]] || continue
    step="$(checkpoint_step "$dir")"
    if (( step < limit_step && step > best )); then
      best="$step"
    fi
  done
  printf '%s\n' "$best"
}

file_size() {
  stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null
}

human_size() {
  du -sh "$1" 2>/dev/null | awk '{print $1}' || printf '?'
}

python_cmd() {
  if [[ -x ".venv-rwkv/bin/python" ]]; then
    printf '%s\n' ".venv-rwkv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    command -v python
  fi
}

float_lte() {
  "$(python_cmd)" - "$1" "$2" <<'PY'
import math
import sys

try:
    left = float(sys.argv[1])
    right = float(sys.argv[2])
except (TypeError, ValueError):
    sys.exit(1)
sys.exit(0 if math.isfinite(left) and math.isfinite(right) and left <= right else 1)
PY
}

wait_for_stable_file() {
  local file="$1"
  local min_bytes="$2"
  local last_size=""
  local stable_count=0
  local size=""

  while true; do
    if [[ -s "$file" ]]; then
      size="$(file_size "$file")"
      if (( size >= min_bytes )); then
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
    fi
    sleep 10
  done
}

handoff_reason() {
  local target_step="$1"
  local current_ema="$2"

  if [[ -z "$HANDOFF_COMMAND" ]]; then
    return 1
  fi
  if (( target_step < HANDOFF_MIN_STEP )); then
    return 1
  fi
  if (( HANDOFF_AT_STEP > 0 && target_step >= HANDOFF_AT_STEP )); then
    printf 'target step reached: %s >= %s\n' "$target_step" "$HANDOFF_AT_STEP"
    return 0
  fi
  if [[ -n "$HANDOFF_EMA_TARGET" ]] && float_lte "$current_ema" "$HANDOFF_EMA_TARGET"; then
    printf 'ema target reached: %s <= %s\n' "$current_ema" "$HANDOFF_EMA_TARGET"
    return 0
  fi
  return 1
}

maybe_run_handoff() {
  local target_step="$1"
  local current_ema="$2"
  local checkpoint="$3"
  local reason=""
  local marker="$RUN_DIR/.handoff_step_${target_step}.done"
  local hook_log="$RUN_DIR/handoff_step_${target_step}_$(date '+%Y%m%d_%H%M%S').log"
  local rc=0

  reason="$(handoff_reason "$target_step" "$current_ema")" || return 1
  if [[ -e "$marker" ]]; then
    log "handoff already completed for step $target_step; marker=$marker"
    return 0
  fi

  log "handoff trigger at step $target_step: $reason; checkpoint=$checkpoint; log=$hook_log"
  set +e
  (
    export HAWKING_TRIGGER_STEP="$target_step"
    export HAWKING_TRIGGER_EMA="$current_ema"
    export HAWKING_TRIGGER_CHECKPOINT="$checkpoint"
    bash -lc "$HANDOFF_COMMAND"
  ) >>"$hook_log" 2>&1
  rc=$?
  set -e

  if (( rc == 0 )); then
    date '+%Y-%m-%d %H:%M:%S %Z' > "$marker"
    log "handoff completed at step $target_step"
    return 0
  fi

  log "handoff failed at step $target_step rc=$rc; continuing autocycle"
  return 1
}

evaluate_checkpoint_policy() {
  local prev_step="$1"
  local target_step="$2"
  "$(python_cmd)" - \
    "$RUN_DIR/events.jsonl" \
    "$POLICY_LOG" \
    "$prev_step" \
    "$target_step" \
    "$MIN_EMA_DROP" \
    "$HOT_LOSS_UNDER" \
    "$EMA_CEILING" \
    "$MAX_EMA_RISE" <<'PY'
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

events_path = Path(sys.argv[1])
policy_log = Path(sys.argv[2])
prev_step = int(sys.argv[3])
target_step = int(sys.argv[4])
min_ema_drop = float(sys.argv[5])
hot_loss_under = float(sys.argv[6])
ema_ceiling = float(sys.argv[7])
max_ema_rise = float(sys.argv[8])

events = []
if events_path.exists():
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            event["step"] = int(event.get("step") or 0)
        except (TypeError, ValueError):
            continue
        events.append(event)

by_step = {event["step"]: event for event in events}
current = by_step.get(target_step)
previous = by_step.get(prev_step) if prev_step > 0 else None

decision = "hold"
maxx = False
reason = "missing target metrics"
current_ema = None
previous_ema = None
recent_loss_mean = None

if current is not None:
    current_ema = float(current.get("loss_ema") or "nan")
    current_loss = float(current.get("loss") or "nan")
    previous_ema = (
        float(previous.get("loss_ema") or "nan")
        if previous is not None else None
    )
    segment = [
        event for event in events
        if event["step"] <= target_step and (prev_step <= 0 or event["step"] > prev_step)
    ]
    recent = segment[-5:]
    losses = [float(event.get("loss") or "nan") for event in recent]
    losses = [value for value in losses if math.isfinite(value)]
    recent_loss_mean = sum(losses) / len(losses) if losses else float("nan")

    improved = (
        previous_ema is None
        or (math.isfinite(current_ema) and math.isfinite(previous_ema)
            and current_ema <= previous_ema - min_ema_drop)
    )
    hot = (
        (math.isfinite(current_loss) and current_loss <= hot_loss_under)
        or (math.isfinite(recent_loss_mean) and recent_loss_mean <= hot_loss_under)
    )
    stable = (
        math.isfinite(current_ema)
        and current_ema <= ema_ceiling
        and (previous_ema is None
             or not math.isfinite(previous_ema)
             or current_ema <= previous_ema + max_ema_rise)
    )

    if stable:
        decision = "continue"
        reason = "stable"
        if improved:
            reason = "ema improved"
        if hot:
            reason = f"{reason}; hot loss"
        maxx = improved or hot
    else:
        reason = (
            f"unstable ema={current_ema:.6f}"
            + (f" prev={previous_ema:.6f}" if previous_ema is not None else "")
        )

record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "prev_step": prev_step,
    "target_step": target_step,
    "decision": decision,
    "maxx": maxx,
    "reason": reason,
    "metrics": {
        "loss_ema": current_ema,
        "prev_loss_ema": previous_ema,
        "recent_loss_mean": recent_loss_mean,
    },
    "thresholds": {
        "min_ema_drop": min_ema_drop,
        "hot_loss_under": hot_loss_under,
        "ema_ceiling": ema_ceiling,
        "max_ema_rise": max_ema_rise,
    },
}
policy_log.parent.mkdir(parents=True, exist_ok=True)
with policy_log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")

def fmt(value):
    if value is None:
        return "none"
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"

print("\t".join([
    decision,
    "1" if maxx else "0",
    reason,
    fmt(current_ema),
    fmt(previous_ema),
    fmt(recent_loss_mean),
]))
PY
}

find_current_pid() {
  ps ax -o pid= -o command= | awk -v out="$RUN_DIR" '
    /[r]wkv7_qat.py/ && index($0, "--out " out) { print $1; exit }
  '
}

stop_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    log "process $pid is already stopped"
    return 0
  fi

  log "stopping training process pid=$pid"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 90); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "process $pid stopped"
      return 0
    fi
    sleep 2
  done

  log "process $pid still alive; forcing stop"
  kill -KILL "$pid" 2>/dev/null || true
}

publish_latest() {
  local checkpoint="$1"
  local source_state="$checkpoint/state_dict.pt"
  local latest_dir="$RUN_DIR/latest"
  local tmp="$latest_dir/.state_dict.pt.tmp"

  mkdir -p "$latest_dir"
  rm -f "$tmp"
  if ln "$source_state" "$tmp" 2>/dev/null; then
    :
  else
    cp -p "$source_state" "$tmp"
  fi
  mv -f "$tmp" "$latest_dir/state_dict.pt"
  log "latest now points at $checkpoint"
}

wait_for_train_start() {
  local pid="$1"
  local relaunch_log="$2"

  for _ in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "relaunch pid=$pid exited before train start"
      return 1
    fi
    if grep -q "\\[train\\] starting" "$relaunch_log" 2>/dev/null; then
      log "relaunch pid=$pid reached train loop"
      return 0
    fi
    sleep 5
  done

  log "timed out waiting for relaunch pid=$pid to reach train loop"
  return 1
}

relaunch_from_checkpoint() {
  local checkpoint="$1"
  local step="$2"
  local relaunch_log="$RUN_DIR/autocycle_relaunch_step_${step}_$(date '+%Y%m%d_%H%M%S').log"
  local runner=()
  local qat_args=()

  log "cooling down for ${COOLDOWN_SECONDS}s before relaunch"
  sleep "$COOLDOWN_SECONDS"

  if command -v caffeinate >/dev/null 2>&1; then
    runner=(caffeinate -dimsu .venv-rwkv/bin/python)
    log "using caffeinate to keep the machine awake during training"
  else
    runner=(.venv-rwkv/bin/python)
  fi

  qat_args=(
    tools/training/rwkv7_qat.py
    --model models/rwkv7-g1-04-hf/model.safetensors
    --hf-dir models/rwkv7-g1-04-hf
    --data artifacts/rwkv7_posttrain/sft.jsonl
    --out "$RUN_DIR"
    --stage ffn --quant ternary --last-n-layers 8
    --max-length "$QAT_MAX_LENGTH" --grad-accum "$QAT_GRAD_ACCUM" --lr "$QAT_LR"
    --epochs 1 --save-every "$SAVE_EVERY" --eval-every 0 --eval-tokens 4096
    --device mps --run-id "$QAT_RUN_ID"
    --seed "$DETERMINISTIC_SEED"
    --pretokenize-workers "$PRETOKENIZE_WORKERS"
    --mps-empty-cache-every "$MPS_EMPTY_CACHE_EVERY"
    --resume-from "$checkpoint"
  )
  if [[ "$DETERMINISTIC" == "1" ]]; then
    qat_args+=(--deterministic)
  fi
  if [[ "$USE_CHUNKED" == "1" ]]; then
    qat_args+=(--use-chunked --chunk-size "$CHUNK_SIZE")
  fi

  log "relaunching from $checkpoint; log=$relaunch_log"
  log "relaunch config: max_length=${QAT_MAX_LENGTH}; grad_accum=${QAT_GRAD_ACCUM}; lr=${QAT_LR}; deterministic=${DETERMINISTIC}; seed=${DETERMINISTIC_SEED}; save_every=${SAVE_EVERY}; pretokenize_workers=${PRETOKENIZE_WORKERS}; mps_empty_cache_every=${MPS_EMPTY_CACHE_EVERY}; mps_high_watermark_ratio=${MPS_HIGH_WATERMARK_RATIO}; use_chunked=${USE_CHUNKED}; chunk_size=${CHUNK_SIZE}"
  PYTHONHASHSEED="$DETERMINISTIC_SEED" \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO="$MPS_HIGH_WATERMARK_RATIO" nohup "${runner[@]}" "${qat_args[@]}" \
    >"$relaunch_log" 2>&1 &

  local new_pid="$!"
  printf '%s\n' "$new_pid" > "$RELAUNCH_PID_FILE"
  log "started relaunch pid=$new_pid"
  wait_for_train_start "$new_pid" "$relaunch_log"
}

prune_old_checkpoints() {
  local keep_step="$1"
  local dir=""
  local step=""
  local keep_floor=$((keep_step - (KEEP_LAST_N_CHECKPOINTS - 1) * TARGET_INTERVAL))

  for dir in "$RUN_DIR"/step_[0-9][0-9][0-9][0-9][0-9][0-9]; do
    [[ -d "$dir" ]] || continue
    step="$(checkpoint_step "$dir")"
    if (( step < keep_floor )); then
      log "pruning old checkpoint $dir ($(human_size "$dir"))"
      rm -rf -- "$dir"
    fi
  done
}

prune_posttrain_artifacts_once() {
  local marker="$RUN_DIR/.posttrain_ozempic_pruned"
  local path=""

  if [[ "$PRUNE_POSTTRAIN" != "1" ]]; then
    log "posttrain pruning disabled"
    return 0
  fi
  if [[ -e "$marker" ]]; then
    log "posttrain pruning already done"
    return 0
  fi

  log "pruning derived posttrain model/output folders; keeping JSONL corpus and logs"
  for path in \
    artifacts/rwkv7_posttrain/sft_out \
    artifacts/rwkv7_posttrain/dpo_out \
    artifacts/rwkv7_posttrain/sft_hf \
    artifacts/rwkv7_posttrain/dpo_hf
  do
    if [[ -e "$path" ]]; then
      log "pruning $path ($(human_size "$path"))"
      rm -rf -- "$path"
    fi
  done
  date '+%Y-%m-%d %H:%M:%S %Z' > "$marker"
}

main() {
  printf '%s\n' "$$" > "$PID_FILE"
  log "autocycle started; targets: $TARGET_STEPS; cooldown=${COOLDOWN_SECONDS}s"
  log "policy: seed=${DETERMINISTIC_SEED}; deterministic=${DETERMINISTIC}; max_length=${QAT_MAX_LENGTH}; grad_accum=${QAT_GRAD_ACCUM}; lr=${QAT_LR}; save_every=${SAVE_EVERY}; pretokenize_workers=${PRETOKENIZE_WORKERS}; mps_empty_cache_every=${MPS_EMPTY_CACHE_EVERY}; mps_high_watermark_ratio=${MPS_HIGH_WATERMARK_RATIO}; keep_last=${KEEP_LAST_N_CHECKPOINTS}; max_auto_step=${MAX_AUTO_STEP}; min_ema_drop=${MIN_EMA_DROP}; hot_loss_under=${HOT_LOSS_UNDER}; ema_ceiling=${EMA_CEILING}; max_ema_rise=${MAX_EMA_RISE}"
  if [[ -n "$HANDOFF_COMMAND" ]]; then
    log "handoff: ema_target=${HANDOFF_EMA_TARGET:-none}; at_step=${HANDOFF_AT_STEP}; min_step=${HANDOFF_MIN_STEP}; stop_after=${HANDOFF_STOP_AFTER}; command=$HANDOFF_COMMAND"
  fi

  local current_pid="${1:-$(cat "$RELAUNCH_PID_FILE" 2>/dev/null || true)}"
  if [[ -z "$current_pid" ]] || ! kill -0 "$current_pid" 2>/dev/null; then
    current_pid="$(find_current_pid || true)"
  fi
  log "initial training pid=${current_pid:-none}"

  local targets=($TARGET_STEPS)
  local target_index=0
  local target=""
  local checkpoint=""
  local previous_target
  local policy_line=""
  local decision=""
  local maxx=""
  local reason=""
  local current_ema=""
  local previous_ema=""
  local recent_loss_mean=""
  local next_target=""

  previous_target="$(latest_checkpoint_before "${targets[0]}")"
  log "policy baseline before step ${targets[0]}: step $previous_target"

  while (( target_index < ${#targets[@]} )); do
    target="${targets[$target_index]}"
    checkpoint="$(step_dir "$target")"

    if [[ ! -s "$checkpoint/state_dict.pt" ]]; then
      log "waiting for checkpoint $checkpoint"
      wait_for_stable_file "$checkpoint/state_dict.pt" 1000000000
    else
      log "checkpoint already exists: $checkpoint"
      wait_for_stable_file "$checkpoint/state_dict.pt" 1000000000
    fi
    log "checkpoint stable: $checkpoint/state_dict.pt"
    policy_line="$(evaluate_checkpoint_policy "$previous_target" "$target")"
    IFS=$'\t' read -r decision maxx reason current_ema previous_ema recent_loss_mean <<< "$policy_line"
    log "policy decision at step $target: decision=$decision maxx=$maxx reason=$reason loss_ema=$current_ema prev_ema=$previous_ema recent_loss_mean=$recent_loss_mean"

    current_pid="$(find_current_pid || true)"
    stop_pid "$current_pid"
    publish_latest "$checkpoint"

    if maybe_run_handoff "$target" "$current_ema" "$checkpoint"; then
      if [[ "$HANDOFF_STOP_AFTER" == "1" ]]; then
        log "handoff stop_after=1; autocycle exiting after step $target"
        exit 0
      fi
    fi

    if [[ "$decision" != "continue" ]]; then
      log "holding at $checkpoint due to policy decision; not relaunching"
      exit 0
    fi

    if relaunch_from_checkpoint "$checkpoint" "$target"; then
      prune_old_checkpoints "$target"
      if (( target == 50 )); then
        prune_posttrain_artifacts_once
      fi
      if [[ "$maxx" == "1" ]] && (( target_index == ${#targets[@]} - 1 )); then
        next_target=$((target + TARGET_INTERVAL))
        if (( next_target <= MAX_AUTO_STEP )); then
          targets+=("$next_target")
          log "maxx condition true at step $target; appended target step $next_target"
        else
          log "maxx condition true at step $target, but max_auto_step=$MAX_AUTO_STEP blocks extension"
        fi
      fi
    else
      log "relaunch failed after $checkpoint; leaving old checkpoints intact"
      exit 1
    fi
    previous_target="$target"
    target_index=$((target_index + 1))
  done

  log "autocycle targets complete; final training process continues from last checkpoint"
}

main "$@"
