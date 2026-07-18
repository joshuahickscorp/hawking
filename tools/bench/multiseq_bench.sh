#!/usr/bin/env bash
# =============================================================================
# tools/bench/multiseq_bench.sh — concurrent-throughput bench for dismantle serve.
#
# Measures AGGREGATE decode throughput (total tokens/sec across all concurrent
# slots) at B=1, 2, 4, 8 by firing concurrent SSE streaming requests against
# a running `dismantle serve` instance.
#
# This is the "serving capacity" number: how many total tokens/sec the engine
# produces at capacity, vs llama.cpp single-stream baseline.
#
# USAGE (server pre-started):
#   ./target/release/hawking serve \
#       --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
#       --max-batch-size 8 &
#   tools/bench/multiseq_bench.sh
#
# USAGE (auto-start):
#   tools/bench/multiseq_bench.sh --start-server
#
# ENVIRONMENT:
#   SERVE_URL           default: http://127.0.0.1:8080
#   TOKENS              decode tokens per slot (default: 128)
#   PROMPT              prompt text (default: Fibonacci prompt)
#   WARMUP              warmup requests before each B (default: 1)
#   BATCH_SIZES         space-separated (default: "1 2 4 8")
#   WEIGHTS             gguf path, used with --start-server
#   PROFILE             kernel profile json, used with --start-server (optional)
#   LLAMA_BASELINE_TPS  reference single-stream tps for ratio column (default: 55)
#   REQUEST_TIMEOUT_SEC per-stream wall-time cap (default: 600)
#
# CONTAMINATION NOTE:
#   B>1/B=1 scaling ratio and vs-llama ratio are contamination-robust (relative;
#   inflation cancels). Absolute agg_tps at B=1 is inflated ~4-5x with a live
#   agent session — tag accordingly.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

SERVE_URL="${SERVE_URL:-http://127.0.0.1:8080}"
TOKENS="${TOKENS:-128}"
PROMPT="${PROMPT:-Write the first 20 Fibonacci numbers and explain how the recurrence works.}"
WARMUP="${WARMUP:-1}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-}"
LLAMA_BASELINE_TPS="${LLAMA_BASELINE_TPS:-55}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-600}"
DBIN="${DBIN:-./target/release/hawking}"
START_SERVER=0
SERVER_PID=""
TMPDIR_BENCH="/tmp/multiseq_bench_$$"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-server)   START_SERVER=1; shift ;;
        --weights)        WEIGHTS="$2"; shift 2 ;;
        --profile)        PROFILE="$2"; shift 2 ;;
        --serve-url)      SERVE_URL="$2"; shift 2 ;;
        --tokens)         TOKENS="$2"; shift 2 ;;
        --prompt)         PROMPT="$2"; shift 2 ;;
        --warmup)         WARMUP="$2"; shift 2 ;;
        --batch-sizes)    BATCH_SIZES="$2"; shift 2 ;;
        --llama-baseline) LLAMA_BASELINE_TPS="$2"; shift 2 ;;
        -h|--help) sed -n '/^#/!q; s/^# \{0,2\}//; /^!/d; p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

command -v curl >/dev/null 2>&1 || { echo "FAIL: curl not found"; exit 3; }
mkdir -p "$TMPDIR_BENCH"

cleanup() {
    [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
    rm -rf "$TMPDIR_BENCH"
}
trap cleanup EXIT

# ── Optional server start ─────────────────────────────────────────────────────
if [[ "$START_SERVER" -eq 1 ]]; then
    [[ -x "$DBIN" ]] || { echo "FAIL: $DBIN not built — cargo build --release -p hawking"; exit 3; }
    [[ -f "$WEIGHTS" ]] || { echo "FAIL: weights not found: $WEIGHTS"; exit 3; }
    MAX_B=1
    for b in $BATCH_SIZES; do [[ "$b" -gt "$MAX_B" ]] && MAX_B="$b"; done
    CMD=("$DBIN" serve --weights "$WEIGHTS" --max-batch-size "$MAX_B")
    [[ -n "$PROFILE" ]] && CMD+=(--kernel-profile "$PROFILE")
    echo "starting: ${CMD[*]}"
    "${CMD[@]}" > "$TMPDIR_BENCH/serve.log" 2>&1 &
    SERVER_PID=$!
    echo -n "waiting for server..."
    WAIT=0
    until curl -sf "${SERVE_URL}/healthz" >/dev/null 2>&1; do
        sleep 1; WAIT=$(( WAIT + 1 ))
        kill -0 "$SERVER_PID" 2>/dev/null || { echo " DIED"; tail -10 "$TMPDIR_BENCH/serve.log" >&2; exit 3; }
        [[ "$WAIT" -gt 120 ]] && { echo " TIMEOUT"; exit 3; }
    done
    echo " ready (${WAIT}s)"
else
    curl -sf "${SERVE_URL}/healthz" >/dev/null 2>&1 || {
        echo "FAIL: server not reachable at ${SERVE_URL}/healthz"
        echo "      Either pre-start: ./target/release/hawking serve --weights <gguf> --max-batch-size 8"
        echo "      Or pass:          --start-server"
        exit 3
    }
fi

# ── Fire one SSE request into $1 ─────────────────────────────────────────────
fire() {
    local logf="$1"
    local prompt_json curl_pid start_s now_s elapsed seen stat
    prompt_json=$(printf '%s' "$PROMPT" | \
        python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" 2>/dev/null \
        || printf '"%s"' "$PROMPT")
    : > "$logf"
    curl --no-buffer --silent -X POST "${SERVE_URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":${prompt_json},\"max_tokens\":${TOKENS},\"stream\":true,\"temperature\":0,\"seed\":0}" \
        > "$logf" 2>&1 &
    curl_pid=$!
    start_s=$(date +%s)

    while :; do
        grep -q '^data: \[DONE\]' "$logf" 2>/dev/null && break

        seen=$(count_toks "$logf")
        [[ "$seen" -ge "$TOKENS" ]] && break

        stat=$(ps -o stat= -p "$curl_pid" 2>/dev/null || true)
        [[ -z "$stat" || "$stat" == *Z* ]] && break

        now_s=$(date +%s)
        elapsed=$(( now_s - start_s ))
        if [[ "$REQUEST_TIMEOUT_SEC" -gt 0 && "$elapsed" -ge "$REQUEST_TIMEOUT_SEC" ]]; then
            printf '\nbench: request timeout after %ss (saw %s/%s token events)\n' \
                "$REQUEST_TIMEOUT_SEC" "$seen" "$TOKENS" >> "$logf"
            break
        fi

        sleep 0.05
    done

    if ps -p "$curl_pid" >/dev/null 2>&1; then
        kill "$curl_pid" 2>/dev/null || true
    fi
    wait "$curl_pid" 2>/dev/null || true
}

# ── Count tokens from an SSE log ─────────────────────────────────────────────
count_toks() {
    awk '/^data: / && $0 != "data: [DONE]" { n++ } END { print n + 0 }' "$1" 2>/dev/null || echo 0
}

# ── Warmup ────────────────────────────────────────────────────────────────────
do_warmup() {
    local n="$1"
    [[ "$n" -le 0 ]] && return
    printf '  warmup (%s request(s))...' "$n"
    for i in $(seq 1 "$n"); do fire "$TMPDIR_BENCH/warmup_$i.log"; done
    echo " done"
}

RESULTS=()
printf '\n'
printf '%-4s  %12s  %8s  %10s  %13s  %12s\n' B total_toks wall_s agg_tps per_slot_tps vs_llama
printf '%.0s-' {1..70}; echo

for B in $BATCH_SIZES; do
    printf '\n[B=%s]\n' "$B"
    do_warmup "$WARMUP"

    LOGS=()
    for slot in $(seq 0 $(( B - 1 ))); do
        logf="$TMPDIR_BENCH/req_B${B}_S${slot}.log"
        LOGS+=("$logf")
        : > "$logf"
    done

    PIDS=()
    T0=$(date +%s.%N)
    for slot in $(seq 0 $(( B - 1 ))); do
        fire "${LOGS[$slot]}" &
        PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" || true
    done
    T1=$(date +%s.%N)

    TOTAL=0
    for slot in $(seq 0 $(( B - 1 ))); do
        t=$(count_toks "${LOGS[$slot]}")
        TOTAL=$(( TOTAL + t ))
    done

    WALL=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.3f", b-a}')
    AGG=$(awk -v tok="$TOTAL" -v w="$WALL" 'BEGIN{if(w>0) printf "%.2f", tok/w; else print "?"}')
    PER=$(awk -v a="$AGG" -v b="$B" 'BEGIN{if(b>0) printf "%.2f", a/b; else print "?"}')
    RATIO=$(awk -v a="$AGG" -v r="$LLAMA_BASELINE_TPS" 'BEGIN{if(r>0&&a>0) printf "%.2fx", a/r; else print "?"}')

    RESULTS+=("$B $TOTAL $WALL $AGG $PER $RATIO")
    printf '%-4s  %12s  %8s  %10s  %13s  %12s\n' "$B" "$TOTAL" "$WALL" "$AGG" "$PER" "$RATIO"
done

printf '\n'
printf '%.0s=' {1..70}; echo
printf 'dismantle serve — aggregate decode throughput\n'
printf 'prompt: %.60s...\n' "$PROMPT"
printf 'tokens/slot: %s   warmup: %s/B   llama_baseline: ~%s tps   request_timeout: %ss\n\n' \
    "$TOKENS" "$WARMUP" "$LLAMA_BASELINE_TPS" "$REQUEST_TIMEOUT_SEC"
printf '%-4s  %12s  %8s  %10s  %13s  %12s\n' B total_toks wall_s agg_tps per_slot_tps vs_llama
printf '%.0s-' {1..70}; echo
for row in "${RESULTS[@]}"; do
    read -r rb rt rw ra rs rr <<< "$row"
    printf '%-4s  %12s  %8s  %10s  %13s  %12s\n' "$rb" "$rt" "$rw" "$ra" "$rs" "$rr"
done
printf '\n'
printf 'agg_tps      = total tokens/s across all B slots\n'
printf 'per_slot_tps = agg_tps / B  (latency proxy; ideal: constant as B grows)\n'
printf 'vs_llama     = agg_tps / llama_baseline  (>B×1.0 means linear scaling)\n'
printf '\nContamination: B>1/B=1 scaling ratio is robust. B=1 absolute tps is\n'
printf 'inflated ~4-5x with a live agent session. Run clean for absolute numbers.\n'
printf '%.0s=' {1..70}; echo
