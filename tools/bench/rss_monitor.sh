#!/usr/bin/env bash
# =============================================================================
# tools/bench/rss_monitor.sh — Track 7.4 memory residency harness.
#
# Monitors RSS (resident set size) of a dismantle process over its lifetime,
# detecting phase transitions (loading, warming, decoding) from stdout log
# lines and producing a phase-annotated residency table.
#
# WHAT IT MEASURES:
#   loading   — from process start until "loaded" or "model loaded" appears
#   warming   — between "loaded" and first "prefill" / "decode" marker
#   decode    — from first decode marker onward (steady-state)
#
# OUTPUT TABLE:
#   phase       | elapsed_s | rss_mb | note
#   loading     | 0.2       | 1840   |
#   warming     | 1.1       | 2240   |
#   decode      | 3.4       | 2180   | steady
#   peak        | -         | 2240   | WITHIN 5GB limit
#
# 5 GB SENTINEL (from CLAUDE.md):
#   Peak RSS > 5000 MB prints a warning and exits non-zero.
#   Qwen-3B steady-state is ~0.8–2 GB; the zero-copy loader keeps it near
#   model size.  Any reading far above that signals a regression.
#
# USAGE (generate mode):
#   tools/bench/rss_monitor.sh --mode generate \
#       --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
#       --prompt "Tell me about recursive descent parsers" \
#       --tokens 200
#
# USAGE (serve mode):
#   tools/bench/rss_monitor.sh --mode serve \
#       --weights models/qwen2.5-3b-instruct-q4_k_m.gguf
#
# ENVIRONMENT (all optional):
#   BIN       dismantle binary (default: ./target/release/dismantle)
#   WEIGHTS   GGUF model path  (default: models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   PROFILE   kernel profile JSON (default: profiles/qwen3b-instruct-q4k.m3pro18.json)
#   POLL_MS   RSS poll interval in ms (default: 500)
#   TOKENS    decode tokens in generate mode (default: 200)
#   PROMPT    prompt in generate mode (default: fibonacci)
#
# COEXISTENCE:
#   dismantle runs under `nice -n 19 taskpolicy -b` (background QoS).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/dismantle}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
POLL_MS="${POLL_MS:-500}"
TOKENS="${TOKENS:-200}"
PROMPT="${PROMPT:-fn fibonacci(n: u64) -> u64 {}"
MODE="generate"

BASE_ENV="DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
DISMANTLE_QWEN_Q4K_PREDEC=1"

RSS_SENTINEL_MB=5000

die()  { printf 'error: %s\n' "$*" >&2; exit 64; }
warn() { printf 'warn: %s\n'  "$*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)    MODE="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --tokens)  TOKENS="$2"; shift 2 ;;
    --prompt)  PROMPT="$2"; shift 2 ;;
    --poll-ms) POLL_MS="$2"; shift 2 ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[[ -x "$BIN" ]] || die "binary not found/executable: $BIN (cargo build --release?)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"

POLL_S=$(awk -v ms="$POLL_MS" 'BEGIN{printf "%.3f", ms/1000}')

# ── Tmpfiles ─────────────────────────────────────────────────────────────────
TMPD="$(mktemp -d /tmp/rss_monitor.XXXXXX)"
LOG_F="$TMPD/proc.log"
cleanup() {
  [[ -n "${PROC_PID:-}" ]] && kill "$PROC_PID" 2>/dev/null || true
  rm -rf "$TMPD"
}
trap cleanup EXIT

# ── Launch dismantle process ──────────────────────────────────────────────────
if [[ "$MODE" == "generate" ]]; then
  env $BASE_ENV nice -n 19 taskpolicy -b "$BIN" generate \
    --weights "$WEIGHTS" \
    --kernel-profile "$PROFILE" \
    --prompt "$PROMPT" \
    --max-new-tokens "$TOKENS" \
    --temperature 0 --seed 0 \
    > "$LOG_F" 2>&1 &
  PROC_PID=$!
elif [[ "$MODE" == "serve" ]]; then
  env $BASE_ENV nice -n 19 taskpolicy -b "$BIN" serve \
    --weights "$WEIGHTS" \
    --kernel-profile "$PROFILE" \
    > "$LOG_F" 2>&1 &
  PROC_PID=$!
else
  die "unknown --mode: $MODE (choose generate or serve)"
fi

# ── RSS polling loop ──────────────────────────────────────────────────────────
# Phase detection via log keywords (case-insensitive grep):
#   loading  : process started but no marker yet
#   warming  : "loaded" / "model loaded" seen but no prefill/decode marker yet
#   decode   : "prefill" or "decode" marker seen

printf '=== rss_monitor — Track 7.4 memory residency ===\n'
printf 'mode    : %s\n' "$MODE"
printf 'binary  : %s\n' "$BIN"
printf 'weights : %s\n' "$WEIGHTS"
printf 'poll_ms : %s\n' "$POLL_MS"
printf '\n'
printf '%-12s  %-10s  %-8s  %s\n' "phase" "elapsed_s" "rss_mb" "note"
printf '%-12s  %-10s  %-8s  %s\n' "------------" "----------" "--------" "----"

T_START=$(date +%s.%N)
PHASE="loading"
PREV_PHASE=""

SAMPLES=()    # (elapsed_s, rss_mb, phase)
PEAK_RSS=0

while kill -0 "$PROC_PID" 2>/dev/null; do
  NOW=$(date +%s.%N)
  ELAPSED=$(awk -v s="$T_START" -v n="$NOW" 'BEGIN{printf "%.2f", n-s}')

  # RSS in KB from ps, convert to MB
  RSS_KB=$(ps -o rss= -p "$PROC_PID" 2>/dev/null | tr -d ' ' || echo 0)
  RSS_MB=$(awk -v k="${RSS_KB:-0}" 'BEGIN{printf "%.0f", k/1024}')

  # Phase detection: read new lines from log
  if [[ -f "$LOG_F" ]]; then
    if [[ "$PHASE" == "loading" ]] && \
       grep -qi -E 'loaded|model loaded|weights loaded|load complete' "$LOG_F" 2>/dev/null; then
      PHASE="warming"
    fi
    if [[ "$PHASE" == "warming" ]] && \
       grep -qi -E 'prefill|decode|first token|generating' "$LOG_F" 2>/dev/null; then
      PHASE="decode"
    fi
  fi

  # Print a line when phase changes or every ~5 seconds
  SHOULD_PRINT=0
  [[ "$PHASE" != "$PREV_PHASE" ]] && SHOULD_PRINT=1
  # Print every 5s (10 samples at 500ms)
  MOD=$(awk -v e="$ELAPSED" -v p="$POLL_S" 'BEGIN{printf "%d", int(e/p) % 10}')
  [[ "$MOD" -eq 0 ]] && SHOULD_PRINT=1

  if [[ "$SHOULD_PRINT" -eq 1 ]]; then
    NOTE=""
    [[ "$PHASE" == "decode" ]] && NOTE="steady"
    [[ "$PHASE" != "$PREV_PHASE" ]] && NOTE="phase-start"
    printf '%-12s  %-10s  %-8s  %s\n' "$PHASE" "$ELAPSED" "$RSS_MB" "$NOTE"
    PREV_PHASE="$PHASE"
  fi

  # Track peak
  if [[ "${RSS_MB:-0}" -gt "$PEAK_RSS" ]]; then
    PEAK_RSS="${RSS_MB:-0}"
  fi

  SAMPLES+=("$ELAPSED $RSS_MB $PHASE")
  sleep "$POLL_S"
done

wait "$PROC_PID" 2>/dev/null || true

# ── Compute steady-state RSS (median of decode-phase samples) ─────────────────
DECODE_RSS_LIST=()
for sample in "${SAMPLES[@]}"; do
  phase=$(awk '{print $3}' <<< "$sample")
  rss=$(awk '{print $2}' <<< "$sample")
  [[ "$phase" == "decode" ]] && DECODE_RSS_LIST+=("$rss")
done

STEADY_RSS=0
if [[ "${#DECODE_RSS_LIST[@]}" -gt 0 ]]; then
  # Median via sort + middle element
  SORTED_DECODE=$(printf '%s\n' "${DECODE_RSS_LIST[@]}" | sort -n)
  N_D="${#DECODE_RSS_LIST[@]}"
  MID=$(( N_D / 2 ))
  STEADY_RSS=$(printf '%s\n' "$SORTED_DECODE" | sed -n "${MID}p")
fi

# ── Summary table ─────────────────────────────────────────────────────────────
printf '\n'
printf '%-12s  %-10s  %-8s  %s\n' "peak" "-" "$PEAK_RSS" ""
printf '\n'
printf '=== summary ===\n'
printf 'peak_rss_mb    : %s\n' "$PEAK_RSS"
printf 'steady_rss_mb  : %s  (median of decode-phase samples)\n' "$STEADY_RSS"

# 5 GB sentinel check
SENTINEL_OK=1
if [[ "$PEAK_RSS" -gt "$RSS_SENTINEL_MB" ]]; then
  SENTINEL_OK=0
fi

if [[ "$SENTINEL_OK" -eq 1 ]]; then
  printf 'sentinel check : PASS (peak %s MB <= %s MB limit)\n' "$PEAK_RSS" "$RSS_SENTINEL_MB"
else
  printf 'sentinel check : FAIL — peak RSS %s MB EXCEEDS %s MB limit!\n' "$PEAK_RSS" "$RSS_SENTINEL_MB"
  printf '                 Qwen-3B steady-state is ~0.8–2 GB. Investigate for regression.\n'
fi

# macmon note for RSS context
printf '\nNOTE: RSS includes all mapped pages (model mmap + heap + stack).\n'
printf '      Qwen-3B-Q4_K_M model file is ~2.0 GB; RSS ≈ model + runtime overhead.\n'
printf '      Run twice (cold then warm) to distinguish mmap page-fault phase.\n'
printf '      For that, use: tools/bench/memory_profile.sh\n'

if [[ "$SENTINEL_OK" -eq 0 ]]; then
  exit 1
fi
