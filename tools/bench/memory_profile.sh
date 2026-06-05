#!/usr/bin/env bash
# =============================================================================
# tools/bench/memory_profile.sh — Track 7.4 cold/warm RSS profile.
#
# Runs the model THREE times and reports:
#   Run 1 (cold)  — OS pages not yet resident; model bytes faulted in during
#                   the load pass.  High RSS indicates the mmap is being read.
#   Run 2 (warm)  — Pages still in the kernel page cache from run 1.
#                   RSS should be similar but load latency drops significantly.
#   Run 3 (hot)   — Confirms steady warm state; variance check.
#
# Delta (cold – warm) shows the "mmap overhead" — how much of the cold-start
# RSS is just page-faulting the model into cache.  A small delta is good:
# it means the zero-copy loader is working and startup cost amortises quickly.
#
# OUTPUT:
#   run       | peak_rss_mb | load_s | decode_tps | note
#   cold(1)   | 2240        | 5.2    | 28.4       |
#   warm(2)   | 2180        | 1.8    | 29.1       |
#   hot(3)    | 2175        | 1.7    | 29.3       |
#   delta_cold_warm: 60 MB  (2.7%)
#
# USAGE:
#   tools/bench/memory_profile.sh
#   tools/bench/memory_profile.sh --tokens 128 --prompt "Once upon a time"
#   WEIGHTS=models/other.gguf tools/bench/memory_profile.sh
#
# ENVIRONMENT:
#   BIN       dismantle binary (default: ./target/release/dismantle)
#   WEIGHTS   GGUF path        (default: models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   PROFILE   kernel profile JSON (default: profiles/qwen3b-instruct-q4k.m3pro18.json)
#   TOKENS    decode tokens per run (default: 128)
#   PROMPT    prompt text (default: fibonacci)
#   POLL_MS   RSS poll interval in ms (default: 200)
#
# NOTE: There is no purge/vmstat cache-drop between runs on macOS without sudo.
#   The delta between run 1 and run 2 is a LOWER BOUND on mmap effectiveness
#   (the OS may have already warmed some pages before run 1 if the model was
#   recently used).  For a true cold measurement, reboot or use
#   `sudo purge` before this script.
#
# COEXISTENCE:
#   All dismantle runs use `nice -n 19 taskpolicy -b` (background QoS).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/dismantle}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
TOKENS="${TOKENS:-128}"
PROMPT="${PROMPT:-fn fibonacci(n: u64) -> u64 {}"
POLL_MS="${POLL_MS:-200}"

BASE_ENV="DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
DISMANTLE_QWEN_Q4K_PREDEC=1"

RSS_SENTINEL_MB=5000

die()  { printf 'error: %s\n' "$*" >&2; exit 64; }
warn() { printf 'warn: %s\n'  "$*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens)  TOKENS="$2";  shift 2 ;;
    --prompt)  PROMPT="$2";  shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --poll-ms) POLL_MS="$2"; shift 2 ;;
    -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[[ -x "$BIN" ]]    || die "binary not found/executable: $BIN (cargo build --release?)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"

POLL_S=$(awk -v ms="$POLL_MS" 'BEGIN{printf "%.3f", ms/1000}')
TMPD="$(mktemp -d /tmp/memory_profile.XXXXXX)"
cleanup() {
  [[ -n "${PROC_PID:-}" ]] && kill "$PROC_PID" 2>/dev/null || true
  rm -rf "$TMPD"
}
trap cleanup EXIT

printf '=== memory_profile — Track 7.4 cold/warm RSS profile ===\n'
printf 'binary  : %s\n' "$BIN"
printf 'weights : %s\n' "$WEIGHTS"
printf 'tokens  : %s\n' "$TOKENS"
printf 'poll_ms : %s\n' "$POLL_MS"
printf '\nNOTE: no sudo purge between runs — delta is lower-bound on cold/warm gap.\n'
printf '      For true cold: sudo purge before run 1.\n\n'

# ── Single-run function ───────────────────────────────────────────────────────
# Outputs: peak_rss_mb load_s decode_tps
run_once() {
  local run_label="$1"
  local log_f="$TMPD/run_${run_label}.log"
  local rss_f="$TMPD/rss_${run_label}.dat"
  : > "$rss_f"

  T_START=$(date +%s.%N)

  env $BASE_ENV nice -n 19 taskpolicy -b "$BIN" generate \
    --weights "$WEIGHTS" \
    --kernel-profile "$PROFILE" \
    --prompt "$PROMPT" \
    --max-new-tokens "$TOKENS" \
    --temperature 0 --seed 0 \
    > "$log_f" 2>&1 &
  PROC_PID=$!

  # Poll RSS until process exits
  PEAK_RSS=0
  while kill -0 "$PROC_PID" 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p "$PROC_PID" 2>/dev/null | tr -d ' ' || echo 0)
    RSS_MB=$(awk -v k="${RSS_KB:-0}" 'BEGIN{printf "%d", k/1024}')
    printf '%s\n' "$RSS_MB" >> "$rss_f"
    if [[ "${RSS_MB:-0}" -gt "$PEAK_RSS" ]]; then
      PEAK_RSS="${RSS_MB:-0}"
    fi
    sleep "$POLL_S"
  done
  wait "$PROC_PID" 2>/dev/null || true
  PROC_PID=""

  T_END=$(date +%s.%N)
  WALL_S=$(awk -v s="$T_START" -v e="$T_END" 'BEGIN{printf "%.2f", e-s}')

  # Parse load time and decode tps from [stats] line
  STATLINE=$(grep -E '\[stats\]' "$log_f" | tail -1 || true)
  LOAD_S="?"
  DEC_TPS="?"
  if [[ -n "$STATLINE" ]]; then
    prefill_ms=$(printf '%s' "$STATLINE" | grep -oE 'prefill_ms=[0-9.]+' | grep -oE '[0-9.]+' || true)
    dec_tps_val=$(printf '%s' "$STATLINE" | grep -oE 'dec_tps=[0-9.]+' | grep -oE '[0-9.]+' || true)
    [[ -n "$prefill_ms" ]] && LOAD_S=$(awk -v m="$prefill_ms" 'BEGIN{printf "%.2f", m/1000}')
    [[ -n "$dec_tps_val" ]] && DEC_TPS="$dec_tps_val"
  fi

  # Return: peak_rss wall_s load_s dec_tps
  printf '%s %s %s %s' "$PEAK_RSS" "$WALL_S" "$LOAD_S" "$DEC_TPS"
}

# ── Run three times ───────────────────────────────────────────────────────────
printf '%-12s  %-12s  %-8s  %-10s  %-10s  %s\n' \
  "run" "peak_rss_mb" "wall_s" "load_s" "decode_tps" "note"
printf '%.0s-' {1..70}; printf '\n'

RESULTS=()
for run_num in 1 2 3; do
  case "$run_num" in
    1) label="cold(1)" ;;
    2) label="warm(2)" ;;
    3) label="hot(3)"  ;;
  esac

  printf 'running %s...\r' "$label"
  read -r peak_rss wall_s load_s dec_tps < <(run_once "$run_num")
  RESULTS+=("$peak_rss $wall_s $load_s $dec_tps")

  note=""
  [[ "$run_num" -eq 1 ]] && note="cold-start (mmap page-fault)"
  [[ "$run_num" -eq 2 ]] && note="warm (pages resident)"
  [[ "$run_num" -eq 3 ]] && note="hot"

  printf '%-12s  %-12s  %-8s  %-10s  %-10s  %s\n' \
    "$label" "$peak_rss" "$wall_s" "$load_s" "${dec_tps}" "$note"

  # Brief pause between runs (let GPU drain)
  [[ "$run_num" -lt 3 ]] && sleep 1
done

# ── Delta analysis ────────────────────────────────────────────────────────────
COLD_RSS=$(awk '{print $1}' <<< "${RESULTS[0]}")
WARM_RSS=$(awk '{print $1}' <<< "${RESULTS[1]}")
HOT_RSS=$(awk  '{print $1}' <<< "${RESULTS[2]}")

DELTA_MB=$(( ${COLD_RSS:-0} - ${WARM_RSS:-0} ))
DELTA_PCT=$(awk -v d="$DELTA_MB" -v c="${COLD_RSS:-1}" 'BEGIN{
  if(c>0) printf "%.1f", d*100/c; else print "?"
}')

COLD_WALL=$(awk '{print $2}' <<< "${RESULTS[0]}")
WARM_WALL=$(awk '{print $2}' <<< "${RESULTS[1]}")
WALL_DELTA=$(awk -v c="$COLD_WALL" -v w="$WARM_WALL" 'BEGIN{
  if(c~/^[0-9]/ && w~/^[0-9]/) printf "%.2f", c-w; else print "?"
}')

COLD_TPS=$(awk '{print $4}' <<< "${RESULTS[0]}")
WARM_TPS=$(awk '{print $4}' <<< "${RESULTS[1]}")

printf '\n'
printf '%.0s=' {1..70}; printf '\n'
printf 'MEMORY PROFILE SUMMARY\n'
printf '%.0s=' {1..70}; printf '\n'
printf 'cold_rss_mb    : %s\n' "$COLD_RSS"
printf 'warm_rss_mb    : %s\n' "$WARM_RSS"
printf 'hot_rss_mb     : %s\n' "$HOT_RSS"
printf 'delta_cold_warm: %s MB  (%s%%)\n' "$DELTA_MB" "$DELTA_PCT"
printf '  -> mmap effectiveness: small delta = good (pages reused)\n'
printf 'wall_s_delta   : %s s  (cold load vs warm load)\n' "$WALL_DELTA"
printf 'decode_tps     : cold=%s  warm=%s\n' "$COLD_TPS" "$WARM_TPS"
printf '\n'

# Sentinel check across all runs
MAX_RSS=$(printf '%s\n' "$COLD_RSS" "$WARM_RSS" "$HOT_RSS" | sort -n | tail -1)
if [[ "${MAX_RSS:-0}" -gt "$RSS_SENTINEL_MB" ]]; then
  printf 'SENTINEL: FAIL — peak RSS %s MB > %s MB limit! Investigate for regression.\n' \
    "$MAX_RSS" "$RSS_SENTINEL_MB"
  exit 1
else
  printf 'SENTINEL: PASS — peak RSS %s MB is within the %s MB limit.\n' \
    "$MAX_RSS" "$RSS_SENTINEL_MB"
fi
