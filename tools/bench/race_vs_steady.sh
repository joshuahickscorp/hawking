#!/usr/bin/env bash
# =============================================================================
# tools/bench/race_vs_steady.sh — Track 7.3 Race-to-idle vs Steady-state bench.
#
# Directly validates the Track 7.3 plan claim that a gather window reduces
# J/task (total joules per completed 8-request batch) at the cost of slightly
# higher TTFT, while aggregate tps stays flat or improves via better batching.
#
# STRATEGIES:
#   race    — no gather window (--energy-mode off):
#             each request dispatches immediately; TTFT is minimal; GPU works
#             at B=1 for most of each request's lifetime.
#
#   steady  — 8ms gather window (--energy-mode efficient):
#             server waits up to 8ms after the first request in a batch window
#             arrives, collecting concurrent requests before dispatching prefill.
#             Aims for larger B, more tokens/dispatch, lower J/tok.
#
# METRICS (per strategy):
#   total_wall_s  — wall time from first request to last DONE
#   agg_tps       — total tokens / total_wall_s
#   J/tok_GPU     — GPU joules per token (macmon, if available)
#   J/task        — total GPU joules for the full 8-request batch
#   TTFT_p50_ms   — median time-to-first-token across 8 streams
#   TTFT_p95_ms   — 95th-percentile TTFT
#
# OUTPUT TABLE:
#   strategy | wall_s | agg_tps | J/tok_gpu | J/task_gpu | TTFT_p50_ms | TTFT_p95_ms
#   race     | 12.4   | 22.1    | 0.196     | 4.88       | 42          | 55
#   steady   | 13.1   | 20.9    | 0.168     | 4.40       | 51          | 72
#   delta    | +5.6%  | -5.4%   | -14.3%    | -9.8%      | +21%        | +31%
#
# POWER:
#   GPU power measured via macmon (sudo-free). If macmon is absent, J columns
#   print "N/A" but all timing/tps numbers are still reported.
#
# USAGE (auto-start two servers on different ports):
#   tools/bench/race_vs_steady.sh
#
# USAGE (pre-started servers):
#   RACE_URL=http://127.0.0.1:8080 STEADY_URL=http://127.0.0.1:8081 \
#   tools/bench/race_vs_steady.sh --no-start
#
# ENVIRONMENT:
#   BIN             dismantle binary (default: ./target/release/dismantle)
#   WEIGHTS         GGUF model path  (default: models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   PROFILE         kernel profile JSON (default: profiles/qwen3b-instruct-q4k.m3pro18.json)
#   BATCH           concurrent requests per strategy (default: 8)
#   TOKENS          decode tokens per request (default: 128)
#   PROMPT          prompt for all requests (default: fibonacci)
#   SAMPLE_MS       macmon poll interval ms (default: 200)
#   WARMUP          warmup requests before measurement (default: 1)
#   REQUEST_TIMEOUT per-request timeout seconds (default: 300)
#   RACE_PORT       port for the race (no-gather) server (default: 8280)
#   STEADY_PORT     port for the steady (gather) server (default: 8281)
#   RACE_URL        override the race server URL
#   STEADY_URL      override the steady server URL
#
# CONTAMINATION NOTE:
#   Absolute tps and J/tok are inflated ~4-5x with a live Claude session.
#   The J/task delta and TTFT delta are contamination-robust (relative, inflation
#   cancels). For publishable absolute numbers: quit Claude and re-run.
#
# COEXISTENCE:
#   dismantle serve processes run under `nice -n 19 taskpolicy -b`.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/dismantle}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
BATCH="${BATCH:-8}"
TOKENS="${TOKENS:-128}"
PROMPT="${PROMPT:-Write the first 20 Fibonacci numbers and explain the recurrence.}"
SAMPLE_MS="${SAMPLE_MS:-200}"
WARMUP="${WARMUP:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
RACE_PORT="${RACE_PORT:-8280}"
STEADY_PORT="${STEADY_PORT:-8281}"
RACE_URL="${RACE_URL:-http://127.0.0.1:${RACE_PORT}}"
STEADY_URL="${STEADY_URL:-http://127.0.0.1:${STEADY_PORT}}"
NO_START=0

BASE_ENV="DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
DISMANTLE_QWEN_Q4K_PREDEC=1"

SAMPLE_S=$(awk -v ms="$SAMPLE_MS" 'BEGIN{printf "%.3f", ms/1000}')

die()  { printf 'error: %s\n' "$*" >&2; exit 64; }
warn() { printf 'warn: %s\n'  "$*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-start)       NO_START=1; shift ;;
    --batch)          BATCH="$2"; shift 2 ;;
    --tokens)         TOKENS="$2"; shift 2 ;;
    --prompt)         PROMPT="$2"; shift 2 ;;
    --weights)        WEIGHTS="$2"; shift 2 ;;
    --profile)        PROFILE="$2"; shift 2 ;;
    --sample-ms)      SAMPLE_MS="$2"; shift 2 ;;
    --warmup)         WARMUP="$2"; shift 2 ;;
    --race-port)      RACE_PORT="$2"; RACE_URL="http://127.0.0.1:${RACE_PORT}"; shift 2 ;;
    --steady-port)    STEADY_PORT="$2"; STEADY_URL="http://127.0.0.1:${STEADY_PORT}"; shift 2 ;;
    --race-url)       RACE_URL="$2"; shift 2 ;;
    --steady-url)     STEADY_URL="$2"; shift 2 ;;
    -h|--help) sed -n '2,65p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[[ -x "$BIN" ]]     || die "binary not found/executable: $BIN (cargo build --release?)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"
command -v curl >/dev/null 2>&1 || die "curl not found"

# ── macmon detection ──────────────────────────────────────────────────────────
HAS_MACMON=0
command -v macmon >/dev/null 2>&1 && HAS_MACMON=1

# ── Tmpfiles ──────────────────────────────────────────────────────────────────
TMPD="$(mktemp -d /tmp/race_vs_steady.XXXXXX)"
RACE_LOG="$TMPD/race_serve.log"
STEADY_LOG="$TMPD/steady_serve.log"
RACE_PID=""
STEADY_PID=""

cleanup() {
  [[ -n "$RACE_PID"   ]] && { kill "$RACE_PID"   2>/dev/null || true; wait "$RACE_PID"   2>/dev/null || true; }
  [[ -n "$STEADY_PID" ]] && { kill "$STEADY_PID" 2>/dev/null || true; wait "$STEADY_PID" 2>/dev/null || true; }
  rm -rf "$TMPD"
}
trap cleanup EXIT

# ── Server startup ─────────────────────────────────────────────────────────────
wait_for_server() {
  local url="$1"
  local label="$2"
  local pid="$3"
  local waited=0
  printf 'waiting for %s server...' "$label"
  until curl -sf "${url}/healthz" >/dev/null 2>&1; do
    sleep 1; waited=$(( waited + 1 ))
    kill -0 "$pid" 2>/dev/null || { printf ' DIED (check logs)\n'; return 1; }
    [[ "$waited" -gt 120 ]] && { printf ' TIMEOUT\n'; return 1; }
  done
  printf ' ready (%ds)\n' "$waited"
}

if [[ "$NO_START" -eq 0 ]]; then
  # Start race server (no gather window)
  printf 'starting race server on port %s (--energy-mode off)...\n' "$RACE_PORT"
  CMD_RACE=("$BIN" serve
    --weights "$WEIGHTS"
    --max-batch-size "$BATCH"
    --port "$RACE_PORT"
    --energy-mode off)
  [[ -n "$PROFILE" ]] && CMD_RACE+=(--kernel-profile "$PROFILE")
  env $BASE_ENV nice -n 19 taskpolicy -b "${CMD_RACE[@]}" \
    > "$RACE_LOG" 2>&1 &
  RACE_PID=$!
  wait_for_server "$RACE_URL" "race" "$RACE_PID" || die "race server failed to start"

  # Start steady server (8ms gather window via --energy-mode efficient)
  printf 'starting steady server on port %s (--energy-mode efficient, 8ms gather)...\n' "$STEADY_PORT"
  CMD_STEADY=("$BIN" serve
    --weights "$WEIGHTS"
    --max-batch-size "$BATCH"
    --port "$STEADY_PORT"
    --energy-mode efficient)
  [[ -n "$PROFILE" ]] && CMD_STEADY+=(--kernel-profile "$PROFILE")
  env $BASE_ENV nice -n 19 taskpolicy -b "${CMD_STEADY[@]}" \
    > "$STEADY_LOG" 2>&1 &
  STEADY_PID=$!
  wait_for_server "$STEADY_URL" "steady" "$STEADY_PID" || die "steady server failed to start"
else
  # Validate pre-started servers
  curl -sf "${RACE_URL}/healthz"   >/dev/null 2>&1 || die "race server not reachable at ${RACE_URL}/healthz"
  curl -sf "${STEADY_URL}/healthz" >/dev/null 2>&1 || die "steady server not reachable at ${STEADY_URL}/healthz"
  printf 'using pre-started servers: race=%s  steady=%s\n' "$RACE_URL" "$STEADY_URL"
fi

# ── Request helpers ────────────────────────────────────────────────────────────
PROMPT_JSON=$(printf '%s' "$PROMPT" | \
  /usr/bin/python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" 2>/dev/null \
  || printf '"%s"' "$PROMPT")

# fire $logf $url — fires one streaming request, records TTFT
fire_req() {
  local logf="$1"
  local url="$2"
  : > "$logf"
  local t0
  t0=$(date +%s.%N)
  printf '%s\n' "$t0" > "${logf}.t0"

  curl --no-buffer --silent -X POST "${url}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":${PROMPT_JSON},\"max_tokens\":${TOKENS},\"stream\":true,\"temperature\":0,\"seed\":0}" \
    > "$logf" 2>&1 &
  local curl_pid=$!

  # Watch for first token (first non-metadata SSE data line)
  local ttft_ms="?"
  local seen=0
  local start_s
  start_s=$(date +%s)
  while kill -0 "$curl_pid" 2>/dev/null; do
    if [[ "$ttft_ms" == "?" ]]; then
      # First token line?
      first_tok=$(grep -m1 '^data: {"' "$logf" 2>/dev/null || true)
      if [[ -n "$first_tok" ]]; then
        t1=$(date +%s.%N)
        ttft_ms=$(awk -v s="$t0" -v e="$t1" 'BEGIN{printf "%.1f", (e-s)*1000}')
        printf '%s\n' "$ttft_ms" > "${logf}.ttft"
      fi
    fi
    grep -q '^data: \[DONE\]' "$logf" 2>/dev/null && break
    local now_s
    now_s=$(date +%s)
    [[ $(( now_s - start_s )) -ge "$REQUEST_TIMEOUT" ]] && break
    sleep 0.02
  done

  if kill -0 "$curl_pid" 2>/dev/null; then
    kill "$curl_pid" 2>/dev/null || true
  fi
  wait "$curl_pid" 2>/dev/null || true

  # Ensure TTFT file exists
  [[ -f "${logf}.ttft" ]] || printf '?\n' > "${logf}.ttft"
}

count_toks() {
  awk '/^data: / && $0 != "data: [DONE]" { n++ } END { print n + 0 }' "$1" 2>/dev/null || echo 0
}

# ── Warmup ─────────────────────────────────────────────────────────────────────
do_warmup() {
  local url="$1"
  local label="$2"
  [[ "$WARMUP" -le 0 ]] && return
  printf '  warmup (%s req) on %s...' "$WARMUP" "$label"
  for j in $(seq 1 "$WARMUP"); do
    fire_req "$TMPD/warmup_${label}_${j}.log" "$url"
  done
  printf ' done\n'
}

# ── macmon power sampling ──────────────────────────────────────────────────────
start_macmon() {
  local gpu_file="$1"
  [[ "$HAS_MACMON" -eq 0 ]] && return
  macmon pipe -i "$SAMPLE_MS" 2>/dev/null | while IFS= read -r line; do
    /usr/bin/python3 -c 'import sys,json
try:
  d=json.load(sys.stdin)
except Exception:
  sys.exit(0)
gpu=d.get("gpu_power",0) or 0
print(f"{gpu:.4f}")' <<< "$line" >> "$gpu_file"
  done &
  printf '%s\n' "$!"
}

stop_macmon() {
  local pid="$1"
  [[ -z "$pid" ]] && return
  pkill -P "$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

mean_watts() {
  awk '{s+=$1; n++} END{ if(n>0) printf "%.4f", s/n; else print "0" }' "$1" 2>/dev/null || echo 0
}

# ── Run one strategy — fires BATCH concurrent requests ────────────────────────
run_strategy() {
  local label="$1"
  local url="$2"

  printf '\n[%s] firing %d concurrent requests...\n' "$label" "$BATCH"

  LOGS=()
  for slot in $(seq 0 $(( BATCH - 1 ))); do
    LOGS+=("$TMPD/${label}_req_${slot}.log")
  done

  # Start power sampling
  GPU_F="$TMPD/${label}_gpu.dat"
  : > "$GPU_F"
  MACMON_PID=""
  [[ "$HAS_MACMON" -eq 1 ]] && MACMON_PID=$(start_macmon "$GPU_F")
  sleep 0.2  # let macmon warm up

  T_BATCH_START=$(date +%s.%N)
  PIDS=()
  for slot in $(seq 0 $(( BATCH - 1 ))); do
    fire_req "${LOGS[$slot]}" "$url" &
    PIDS+=("$!")
    # Stagger slightly so server sees requests arrive over a short window
    # (otherwise they all land in the same microsecond; real traffic is spread)
    sleep 0.003
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" || true
  done
  T_BATCH_END=$(date +%s.%N)

  stop_macmon "${MACMON_PID:-}"

  WALL_S=$(awk -v s="$T_BATCH_START" -v e="$T_BATCH_END" 'BEGIN{printf "%.3f", e-s}')

  # Count total tokens
  TOTAL_TOKS=0
  for slot in $(seq 0 $(( BATCH - 1 ))); do
    t=$(count_toks "${LOGS[$slot]}")
    TOTAL_TOKS=$(( TOTAL_TOKS + t ))
  done

  # Aggregate tps
  AGG_TPS=$(awk -v tok="$TOTAL_TOKS" -v w="$WALL_S" \
    'BEGIN{if(w>0) printf "%.2f", tok/w; else print "?"}')

  # TTFT stats
  TTFT_VALUES=()
  for slot in $(seq 0 $(( BATCH - 1 ))); do
    ttft=$(cat "${LOGS[$slot]}.ttft" 2>/dev/null || echo "?")
    [[ "$ttft" =~ ^[0-9] ]] && TTFT_VALUES+=("$ttft")
  done

  TTFT_P50="?"
  TTFT_P95="?"
  if [[ "${#TTFT_VALUES[@]}" -gt 0 ]]; then
    SORTED=$(printf '%s\n' "${TTFT_VALUES[@]}" | sort -n)
    N_T="${#TTFT_VALUES[@]}"
    IDX_P50=$(awk -v n="$N_T" 'BEGIN{printf "%d", int(n*0.50)}')
    IDX_P95=$(awk -v n="$N_T" 'BEGIN{printf "%d", int(n*0.95+0.5)}')
    [[ "$IDX_P50" -ge "$N_T" ]] && IDX_P50=$(( N_T - 1 ))
    [[ "$IDX_P95" -ge "$N_T" ]] && IDX_P95=$(( N_T - 1 ))
    TTFT_P50=$(printf '%s\n' "$SORTED" | sed -n "$(( IDX_P50 + 1 ))p")
    TTFT_P95=$(printf '%s\n' "$SORTED" | sed -n "$(( IDX_P95 + 1 ))p")
  fi

  # Energy
  JTOK_GPU="N/A"
  JTASK_GPU="N/A"
  if [[ "$HAS_MACMON" -eq 1 ]]; then
    AVG_GPU_W=$(mean_watts "$GPU_F")
    JTOK_GPU=$(awk -v w="$AVG_GPU_W" -v s="$WALL_S" -v tok="$TOTAL_TOKS" \
      'BEGIN{if(tok>0) printf "%.4f", w*s/tok; else print "?"}')
    JTASK_GPU=$(awk -v w="$AVG_GPU_W" -v s="$WALL_S" \
      'BEGIN{printf "%.3f", w*s}')
  fi

  # Store results in named vars (caller reads them)
  printf '%s %s %s %s %s %s %s\n' \
    "$WALL_S" "$TOTAL_TOKS" "$AGG_TPS" "$JTOK_GPU" "$JTASK_GPU" "$TTFT_P50" "$TTFT_P95"
}

# ── Main ───────────────────────────────────────────────────────────────────────
printf '=== race_vs_steady — Track 7.3 Race-to-idle vs Steady-state ===\n'
printf 'binary  : %s\n' "$BIN"
printf 'weights : %s\n' "$WEIGHTS"
printf 'batch   : %d concurrent requests\n' "$BATCH"
printf 'tokens  : %d per request\n' "$TOKENS"
printf 'macmon  : %s\n' "$([[ "$HAS_MACMON" -eq 1 ]] && echo yes || echo no)"
printf '\nCONTAMINATION: absolute tps/J inflated ~4-5x with a live Claude session.\n'
printf 'J/task delta and TTFT delta are contamination-robust (relative).\n'

do_warmup "$RACE_URL"   "race"
do_warmup "$STEADY_URL" "steady"

read -r RACE_WALL RACE_TOK RACE_TPS RACE_JTOK RACE_JTASK RACE_P50 RACE_P95 \
  < <(run_strategy "race"   "$RACE_URL")
read -r STDY_WALL STDY_TOK STDY_TPS STDY_JTOK STDY_JTASK STDY_P50 STDY_P95 \
  < <(run_strategy "steady" "$STEADY_URL")

# ── Compute deltas ────────────────────────────────────────────────────────────
compute_pct_delta() {
  local a="$1" b="$2"
  [[ "$a" =~ ^[0-9.]+$ && "$b" =~ ^[0-9.]+$ ]] || { printf 'N/A'; return; }
  awk -v aa="$a" -v bb="$b" 'BEGIN{
    if(aa==0){print "N/A"; exit}
    d=(bb-aa)/aa*100
    if(d>=0) printf "+%.1f%%", d
    else     printf "%.1f%%", d
  }'
}

D_WALL=$(compute_pct_delta "$RACE_WALL" "$STDY_WALL")
D_TPS=$(compute_pct_delta  "$RACE_TPS"  "$STDY_TPS")
D_JTOK="N/A"
D_JTASK="N/A"
if [[ "$HAS_MACMON" -eq 1 ]]; then
  [[ "$RACE_JTOK"  =~ ^[0-9.]+$ ]] && [[ "$STDY_JTOK"  =~ ^[0-9.]+$ ]] && \
    D_JTOK=$(compute_pct_delta  "$RACE_JTOK"  "$STDY_JTOK")
  [[ "$RACE_JTASK" =~ ^[0-9.]+$ ]] && [[ "$STDY_JTASK" =~ ^[0-9.]+$ ]] && \
    D_JTASK=$(compute_pct_delta "$RACE_JTASK" "$STDY_JTASK")
fi
D_P50=$(compute_pct_delta "$RACE_P50" "$STDY_P50")
D_P95=$(compute_pct_delta "$RACE_P95" "$STDY_P95")

# ── Results table ─────────────────────────────────────────────────────────────
printf '\n'
printf '%.0s=' {1..78}; printf '\n'
printf 'RACE-TO-IDLE VS STEADY-STATE — B=%d requests, %d tokens/req\n' "$BATCH" "$TOKENS"
printf '%.0s=' {1..78}; printf '\n'
printf '%-10s  %-8s  %-8s  %-10s  %-11s  %-12s  %-12s\n' \
  strategy wall_s agg_tps J/tok_gpu J/task_gpu TTFT_p50_ms TTFT_p95_ms
printf '%.0s-' {1..78}; printf '\n'
printf '%-10s  %-8s  %-8s  %-10s  %-11s  %-12s  %-12s\n' \
  "race"   "$RACE_WALL" "$RACE_TPS" "$RACE_JTOK" "$RACE_JTASK" "$RACE_P50" "$RACE_P95"
printf '%-10s  %-8s  %-8s  %-10s  %-11s  %-12s  %-12s\n' \
  "steady" "$STDY_WALL" "$STDY_TPS" "$STDY_JTOK" "$STDY_JTASK" "$STDY_P50" "$STDY_P95"
printf '%.0s-' {1..78}; printf '\n'
printf '%-10s  %-8s  %-8s  %-10s  %-11s  %-12s  %-12s\n' \
  "delta"  "$D_WALL" "$D_TPS" "$D_JTOK" "$D_JTASK" "$D_P50" "$D_P95"
printf '%.0s=' {1..78}; printf '\n'
printf '\n'
printf 'J/task  = GPU joules for the full B-request batch window\n'
printf 'J/tok   = J/task / total_tokens  (normalized efficiency)\n'
printf 'TTFT    = time-to-first-token (latency impact of gather window)\n'
printf '\n'
printf 'Theory: steady J/task < race J/task  (more tokens/dispatch -> less idle GPU)\n'
printf '        steady TTFT_p50 > race TTFT_p50  (gather-window latency cost)\n'
printf '        steady agg_tps ~= race agg_tps or slightly higher (better batching)\n'

if [[ "$HAS_MACMON" -eq 0 ]]; then
  printf '\nNOTE: macmon not found; J columns not available (brew install macmon).\n'
fi
