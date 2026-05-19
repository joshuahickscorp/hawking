#!/bin/bash
# path-to-125 L8 — auto-iter sequencer.
#
# Watches the current training PID, runs the chain-decode smoke at
# the final checkpoint, records the result, and (optionally) launches
# the next iter from a pre-configured queue. Lets the user queue a
# series of training experiments and walk away.
#
# Usage:
#   tools/l8_autoiter.sh watch <pid> [chain_k]
#     Waits for PID to die, then runs smoke + records.
#
#   tools/l8_autoiter.sh queue <config_script> [config_script ...]
#     Runs each config script in sequence. After each completes,
#     records its result. If accept_rate ≥ progression_threshold,
#     launches the next; else halts with summary.
#
#   tools/l8_autoiter.sh status
#     Prints a markdown table of all completed iters.
#
# Config script convention:
#   - Sets ITER_NAME (used for ckpt dir + log archive)
#   - Sets CHAIN_K (the K-value to test chain smoke with)
#   - Calls the python eagle4.py train command directly
#   - Writes pid to reports/path_to_90/_levers/l8_status.json on launch
#
# Progression threshold (success = launch next): chain_accept ≥ 25%
# Termination threshold (fail = halt queue): chain_accept < 10%

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="reports/path_to_90/_levers"
RESULTS_LOG="$LOG_DIR/l8_iter_results.jsonl"
mkdir -p "$LOG_DIR"

PROGRESSION_THRESHOLD=0.25   # ≥ 25% chain accept → advance to next iter
TERMINATION_THRESHOLD=0.10   # < 10% chain accept → halt the queue

VENV_PYTHON="/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python"
DISMANTLE="./target/release/dismantle"
WEIGHTS="models/deepseek-v2-lite-q4.gguf"
PROFILE="profiles/deepseek-v2-lite-q4.m3pro18.json"
FROZEN="eagle4/v2lite_frozen.npz"

# -- helpers --------------------------------------------------------

iso_now() { date -u +%FT%TZ; }

current_pid() {
  python3 -c "import json,sys; print(json.load(open('$LOG_DIR/l8_status.json'))['pid'])" 2>/dev/null \
    || echo ""
}

current_ckpt_dir() {
  python3 -c "import json,sys; print(json.load(open('$LOG_DIR/l8_status.json'))['ckpt_dir'])" 2>/dev/null \
    || echo "eagle4/checkpoints/eagle4_v4_fromscratch"
}

# Block until PID dies. Prints a dot every minute while waiting.
wait_for_pid_death() {
  local pid="$1"
  local mins=0
  while ps -p "$pid" >/dev/null 2>&1; do
    sleep 60
    mins=$((mins + 1))
    echo "[autoiter] pid $pid still alive at +${mins} min" >&2
  done
  echo "[autoiter] pid $pid ended after ${mins} min" >&2
}

# Run chain smoke ×2 against the given ckpt at the given K. Returns
# accept_rate to stdout (rounded to 4 decimals).
run_chain_smoke() {
  local ckpt="$1"; local chain_k="${2:-2}"
  local total_acc=0 total_rej=0
  for i in 1 2; do
    local out
    out=$(EAGLE4_CHAIN_K="$chain_k" "$DISMANTLE" generate \
            --weights "$WEIGHTS" \
            --kernel-profile "$PROFILE" \
            --speculate eagle4 \
            --draft-head "$ckpt" \
            --eagle4-frozen "$FROZEN" \
            --prompt "The capital of France is" \
            --max-new-tokens 48 2>&1 | grep -E "^\[stats\]" | head -1 || true)
    local acc rej
    acc=$(echo "$out" | grep -oE "draft_accepted=[0-9]+" | cut -d= -f2 || echo "0")
    rej=$(echo "$out" | grep -oE "draft_rejected=[0-9]+" | cut -d= -f2 || echo "0")
    total_acc=$((total_acc + acc))
    total_rej=$((total_rej + rej))
  done
  python3 -c "print(round($total_acc / max($total_acc + $total_rej, 1), 4))"
}

# Record one iter's result to the JSONL log.
record_result() {
  local iter_name="$1"; local chain_k="$2"; local accept_rate="$3"
  local ckpt_dir="$4"; local notes="${5:-}"
  python3 - <<PY >>"$RESULTS_LOG"
import json
print(json.dumps({
  "iter_name": "$iter_name",
  "finished_at": "$(iso_now)",
  "chain_k_tested": $chain_k,
  "accept_rate": $accept_rate,
  "ckpt_dir": "$ckpt_dir",
  "verdict": "ADVANCE" if $accept_rate >= $PROGRESSION_THRESHOLD else ("HALT" if $accept_rate < $TERMINATION_THRESHOLD else "WATCH"),
  "notes": "$notes"
}))
PY
}

# -- subcommands ----------------------------------------------------

cmd_watch() {
  local pid="$1"; local chain_k="${2:-2}"; local iter_name="${3:-iter_unnamed}"
  local ckpt_dir
  ckpt_dir=$(current_ckpt_dir)
  echo "[autoiter] watching pid=$pid chain_k=$chain_k ckpt_dir=$ckpt_dir" >&2
  wait_for_pid_death "$pid"

  if [ ! -f "$ckpt_dir/latest.npz" ]; then
    echo "[autoiter] no latest.npz at $ckpt_dir — training crashed early?" >&2
    record_result "$iter_name" "$chain_k" 0.0 "$ckpt_dir" "no checkpoint produced"
    return 1
  fi

  local accept_rate
  accept_rate=$(run_chain_smoke "$ckpt_dir/latest.npz" "$chain_k")
  echo "[autoiter] $iter_name: chain_accept @ K=$chain_k = $accept_rate" >&2
  record_result "$iter_name" "$chain_k" "$accept_rate" "$ckpt_dir"

  echo "$accept_rate"
}

cmd_queue() {
  local configs=("$@")
  echo "[autoiter] queue: ${#configs[@]} configs" >&2
  for cfg in "${configs[@]}"; do
    if [ ! -x "$cfg" ]; then
      echo "[autoiter] config script not executable: $cfg" >&2
      exit 1
    fi
    echo "[autoiter] launching: $cfg" >&2
    "$cfg" --nohup
    local pid
    pid=$(current_pid)
    local iter_name
    iter_name=$(basename "$cfg" .sh)
    local chain_k
    # Each config script writes its CHAIN_K_FOR_SMOKE into l8_status.json (or default 2)
    chain_k=$(python3 -c "import json; print(json.load(open('$LOG_DIR/l8_status.json')).get('chain_k_for_smoke', 2))" 2>/dev/null || echo "2")

    echo "[autoiter] $iter_name launched as pid $pid; chain_k_for_smoke=$chain_k" >&2
    local accept_rate
    accept_rate=$(cmd_watch "$pid" "$chain_k" "$iter_name")

    local rc=0
    python3 -c "import sys; sys.exit(0 if $accept_rate >= $PROGRESSION_THRESHOLD else 1)" || rc=1
    if [ $rc -ne 0 ]; then
      local kill_rc=0
      python3 -c "import sys; sys.exit(0 if $accept_rate < $TERMINATION_THRESHOLD else 1)" || kill_rc=1
      if [ $kill_rc -eq 0 ]; then
        echo "[autoiter] HALT — $iter_name accept_rate=$accept_rate < $TERMINATION_THRESHOLD" >&2
        return 2
      else
        echo "[autoiter] WATCH — $iter_name accept_rate=$accept_rate marginal; stopping queue" >&2
        return 0
      fi
    fi
    echo "[autoiter] ADVANCE — $iter_name accept_rate=$accept_rate ≥ $PROGRESSION_THRESHOLD" >&2
  done
  echo "[autoiter] queue complete" >&2
}

cmd_status() {
  if [ ! -f "$RESULTS_LOG" ]; then
    echo "no iters recorded yet"
    return 0
  fi
  echo "| iter | K | accept | verdict | finished |"
  echo "|------|---|--------|---------|----------|"
  python3 - <<PY
import json
for line in open("$RESULTS_LOG"):
  r = json.loads(line)
  pct = f"{r['accept_rate']*100:.1f}%"
  print(f"| {r['iter_name']} | {r['chain_k_tested']} | {pct} | {r['verdict']} | {r['finished_at']} |")
PY
}

# watch_chain — wait for the current training PID to die, run its smoke,
# record the result, and if accept_rate ≥ progression threshold launch
# the next config in the queue. Recursively chains until queue empty or
# a config fails. Designed to run in the background (nohup).
#
# Usage:
#   tools/l8_autoiter.sh watch_chain <pid> <chain_k> <iter_name> [next_config.sh ...]
cmd_watch_chain() {
  local pid="$1"; local chain_k="$2"; local iter_name="$3"; shift 3
  local rest=("$@")

  echo "[autoiter] watch_chain pid=$pid iter_name=$iter_name; queue tail: ${#rest[@]}" >&2
  local accept_rate
  accept_rate=$(cmd_watch "$pid" "$chain_k" "$iter_name")
  echo "[autoiter] $iter_name done: accept_rate=$accept_rate" >&2

  local advance=1
  python3 -c "import sys; sys.exit(0 if $accept_rate >= $PROGRESSION_THRESHOLD else 1)" || advance=0
  if [ $advance -eq 0 ]; then
    echo "[autoiter] queue halted: $iter_name accept_rate=$accept_rate < progression $PROGRESSION_THRESHOLD" >&2
    return 0
  fi

  if [ ${#rest[@]} -eq 0 ]; then
    echo "[autoiter] queue complete after $iter_name" >&2
    return 0
  fi

  local next="${rest[0]}"
  local tail=("${rest[@]:1}")
  echo "[autoiter] advancing to $next" >&2
  "$next" --nohup
  local next_pid next_iter next_chain_k
  next_pid=$(current_pid)
  next_iter=$(python3 -c "import json; print(json.load(open('$LOG_DIR/l8_status.json')).get('iter_name','unknown'))" 2>/dev/null || echo "unknown")
  next_chain_k=$(python3 -c "import json; print(json.load(open('$LOG_DIR/l8_status.json')).get('chain_k_for_smoke',2))" 2>/dev/null || echo "2")

  cmd_watch_chain "$next_pid" "$next_chain_k" "$next_iter" "${tail[@]:-}"
}

# -- main -----------------------------------------------------------

case "${1:-}" in
  watch)
    shift; cmd_watch "$@"
    ;;
  queue)
    shift; cmd_queue "$@"
    ;;
  watch_chain)
    shift; cmd_watch_chain "$@"
    ;;
  status)
    cmd_status
    ;;
  *)
    cat <<USAGE
Usage:
  tools/l8_autoiter.sh watch <pid> [chain_k] [iter_name]
      Block until pid dies, run smoke, record result.
  tools/l8_autoiter.sh queue <config_script> [config_script ...]
      Launch each config sequentially; advance only if accept ≥ 25%.
  tools/l8_autoiter.sh watch_chain <pid> <chain_k> <iter_name> [next_config.sh ...]
      Watch current pid, then auto-launch the queue. Run in nohup.
  tools/l8_autoiter.sh status
      Markdown table of all completed iters.
USAGE
    exit 1
    ;;
esac
