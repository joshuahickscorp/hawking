#!/usr/bin/env bash
#
# tools/bench/bench_server_driver.sh — multi-request bench via persistent model.
#
# Spawns `dismantle bench-server` once, feeds it N requests, collects
# timing, and outputs the same summary format as coexist_bench.sh.
# Eliminates the 5-15s model-load cost per smoke iteration — 30-60%
# faster than launching a new process per trial.
#
# Usage:
#   bash tools/bench/bench_server_driver.sh \
#       --weights models/deepseek-v2-lite-q4.gguf \
#       --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
#       --trials 8 --tokens 24 --prompt "Once upon a time"
#
#   # Or using environment variables:
#   WEIGHTS=... PROFILE=... TRIALS=6 TOKENS=24 \
#       bash tools/bench/bench_server_driver.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

BIN="./target/release/dismantle"
WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
TRIALS="${TRIALS:-6}"
TOKENS="${TOKENS:-32}"
PROMPT="${PROMPT:-Once upon a time}"
QUIET=0

# ─── ARGUMENT PARSING ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights)         WEIGHTS="$2"; shift 2 ;;
        --kernel-profile)  PROFILE="$2"; shift 2 ;;
        --trials)          TRIALS="$2"; shift 2 ;;
        --tokens)          TOKENS="$2"; shift 2 ;;
        --prompt)          PROMPT="$2"; shift 2 ;;
        --quiet|-q)        QUIET=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ts()  { date -u +%FT%TZ; }
log() { [[ "$QUIET" -eq 0 ]] && printf '%s %s\n' "$(ts)" "$*" || true; }

OUT_DIR="bench_results/bench_server_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

if ! "$BIN" --version >/dev/null 2>&1; then
    log "binary missing — running cargo build --release --workspace..."
    cargo build --release --workspace >/dev/null 2>&1
fi

log "=== BENCH SERVER MODE ==="
log "  trials=${TRIALS}  tokens=${TOKENS}  weights=$(basename "$WEIGHTS")"
log "  output: $OUT_DIR"

# ─── SERVER SETUP ─────────────────────────────────────────────────────────────
SERVER_STDIN=$(mktemp -u)
SERVER_STDOUT="$OUT_DIR/server.stdout"
SERVER_STDERR="$OUT_DIR/server.stderr"

mkfifo "$SERVER_STDIN"

# Spawn server in background, feeding from the named pipe.
"$BIN" bench-server \
    --weights "$WEIGHTS" \
    --kernel-profile "$PROFILE" \
    --trace-dispatch \
    --stdin \
    < "$SERVER_STDIN" \
    > "$SERVER_STDOUT" \
    2> "$SERVER_STDERR" &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    rm -f "$SERVER_STDIN"
}
trap cleanup EXIT

# Open the write end of the pipe so it stays open while we feed requests.
exec 3>"$SERVER_STDIN"

# Give server time to load model.
log "waiting for model load..."
WAIT=0
until grep -q "ready for requests" "$SERVER_STDERR" 2>/dev/null; do
    sleep 1
    WAIT=$(( WAIT + 1 ))
    if [[ $WAIT -gt 120 ]]; then
        log "ERROR: server did not start within 120s"
        cat "$SERVER_STDERR" >&2
        exit 1
    fi
done
log "server ready (${WAIT}s load time)"

# ─── REQUEST LOOP ─────────────────────────────────────────────────────────────
TRIALS_TPS=()
for i in $(seq 1 "$TRIALS"); do
    log "=== trial $i / $TRIALS ==="

    # JSON-escape the prompt (handle quotes/backslashes).
    PROMPT_ESC=$(printf '%s' "$PROMPT" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" 2>/dev/null \
        || printf '"%s"' "$PROMPT")
    REQ="{\"id\":\"req_${i}\",\"prompt\":${PROMPT_ESC},\"max_tokens\":${TOKENS},\"seed\":42}"

    # Send request.
    printf '%s\n' "$REQ" >&3

    # Read exactly one response line from server stdout.
    # Wait up to 120s per request (model load already done).
    RESP=""
    WAITED=0
    while true; do
        LINE=$(tail -n +$i "$SERVER_STDOUT" | head -1 2>/dev/null || echo "")
        if [[ -n "$LINE" ]]; then
            RESP="$LINE"
            break
        fi
        sleep 0.2
        WAITED=$(( WAITED + 1 ))
        if [[ $WAITED -gt 600 ]]; then
            log "DISCARD trial $i: timed out waiting for response"
            TRIALS_TPS+=("0")
            break
        fi
    done

    if [[ -z "$RESP" ]]; then
        continue
    fi

    ERR=$(printf '%s' "$RESP" | jq -r '.error // empty' 2>/dev/null || true)
    if [[ -n "$ERR" ]]; then
        log "DISCARD trial $i: server error: $ERR"
        TRIALS_TPS+=("0")
        continue
    fi

    DEC_TPS=$(printf '%s' "$RESP" | jq -r '.dec_tps // 0' 2>/dev/null || echo "0")
    COMP_TOK=$(printf '%s' "$RESP" | jq -r '.completion_tokens // 0' 2>/dev/null || echo "0")
    printf '%s' "$RESP" > "$OUT_DIR/trial_${i}.json"
    log "trial $i: ${DEC_TPS} dec_tps (${COMP_TOK} tokens)"
    TRIALS_TPS+=("$DEC_TPS")
done

# Close write end of pipe → server gets EOF → server exits.
exec 3>&-

# ─── STATISTICS ───────────────────────────────────────────────────────────────
SORTED=$(printf '%s\n' "${TRIALS_TPS[@]}" | awk '$1 > 0' | sort -n)
COUNT=$(printf '%s\n' "$SORTED" | wc -l | tr -d ' ')
if [[ "$COUNT" -lt 2 ]]; then
    log "FAIL: only $COUNT valid trials. See $OUT_DIR/server.stderr"
    exit 1
fi

MEDIAN_IDX=$(( (COUNT + 1) / 2 ))
MEDIAN=$(printf '%s\n' "$SORTED" | sed -n "${MEDIAN_IDX}p")
MIN=$(printf '%s\n' "$SORTED" | head -1)
MAX=$(printf '%s\n' "$SORTED" | tail -1)

if [[ "$COUNT" -ge 4 ]]; then
    TRIM_N=$(( COUNT / 4 ))
    TRIMMED_VALS=$(printf '%s\n' "$SORTED" | tail -n "+$((TRIM_N + 1))" | head -n "$(( COUNT - 2 * TRIM_N ))")
    TRIMMED_MEAN=$(printf '%s\n' "$TRIMMED_VALS" | awk '{ s+=$1; n++ } END { printf "%.3f", s/n }')
    ESTIMATOR="trimmed_mean_25pct"
else
    TRIMMED_MEAN="$MEDIAN"
    ESTIMATOR="median_fallback"
fi

MEAN=$(printf '%s\n' "$SORTED" | awk '{ s+=$1; n++ } END { printf "%.6f", s/n }')
STDDEV=$(printf '%s\n' "$SORTED" | awk -v mean="$MEAN" \
    '{ s+=($1-mean)^2; n++ } END { if (n>1) printf "%.6f", sqrt(s/(n-1)); else print "0" }')
CI_HALF=$(awk -v sd="$STDDEV" -v n="$COUNT" 'BEGIN { printf "%.3f", 1.96*sd/sqrt(n) }')
CI_LO=$(awk -v m="$MEDIAN" -v h="$CI_HALF" 'BEGIN { printf "%.3f", m-h }')
CI_HI=$(awk -v m="$MEDIAN" -v h="$CI_HALF" 'BEGIN { printf "%.3f", m+h }')

Q1_IDX=$(( (COUNT + 3) / 4 ))
Q3_IDX=$(( (3 * COUNT + 3) / 4 ))
Q1=$(printf '%s\n' "$SORTED" | sed -n "${Q1_IDX}p")
Q3=$(printf '%s\n' "$SORTED" | sed -n "${Q3_IDX}p")
IQR=$(awk -v q1="$Q1" -v q3="$Q3" 'BEGIN { printf "%.3f", q3-q1 }')

log ""
log "=== SUMMARY (BENCH-SERVER MODE) ==="
printf 'all trials:   %s\n' "${TRIALS_TPS[*]}"
printf 'valid trials: %d / %d\n' "$COUNT" "$TRIALS"
printf 'median:       %s dec_tps (95%% CI: [%s, %s], IQR: %s)\n' "$MEDIAN" "$CI_LO" "$CI_HI" "$IQR"
printf 'trimmed_mean: %s dec_tps (%s)\n' "$TRIMMED_MEAN" "$ESTIMATOR"
printf 'min..max:     %s .. %s\n' "$MIN" "$MAX"

# Structural from last successful trial.
STRUCT_JSON=""
for try_trial in $(seq "$TRIALS" -1 1); do
    if [[ -f "$OUT_DIR/trial_${try_trial}.json" ]]; then
        STRUCT_JSON=$(jq '{
            commits_per_token: .structural.commits_per_token,
            buffers_per_token: .structural.buffers_per_token,
            alloc_bytes_per_token: .structural.alloc_bytes_per_token
        }' "$OUT_DIR/trial_${try_trial}.json" 2>/dev/null || true)
        [[ -n "$STRUCT_JSON" ]] && break
    fi
done
[[ -n "$STRUCT_JSON" ]] && printf '\nStructural metrics:\n%s\n' "$STRUCT_JSON"

# Append to bench_history.jsonl.
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
MODEL_TAG=$(basename "$WEIGHTS" .gguf 2>/dev/null || echo "unknown")
VALID_TRIALS_JSON=$(printf '%s\n' "${TRIALS_TPS[@]}" | \
    awk '$1>0 { printf "%s,", $1 }' | sed 's/,$//')

mkdir -p bench_results
cat >> bench_results/bench_history.jsonl <<EOF
{"timestamp":"$(ts)","commit":"${COMMIT}","branch":"${BRANCH}","tool":"bench_server_driver","config":{"trials":${TRIALS},"tokens":${TOKENS},"model":"${MODEL_TAG}"},"results":{"median":${MEDIAN},"trimmed_mean":${TRIMMED_MEAN},"ci_95_lo":${CI_LO},"ci_95_hi":${CI_HI},"iqr":${IQR},"estimator":"${ESTIMATOR}","trials":[${VALID_TRIALS_JSON}]},"structural":null}
EOF

{
    printf '# bench-server driver — %s\n\n' "$(ts)"
    printf '**Mode:** BENCH-SERVER (one model load, %d inference requests)\n' "$TRIALS"
    printf '**Speedup:** ~30-60%% vs separate process per trial (no reload)\n\n'
    printf '## Stats\n'
    printf '%s\n' "- valid trials: $COUNT / $TRIALS"
    printf '%s\n' "- median: **$MEDIAN dec_tps** (95% CI: [$CI_LO, $CI_HI], IQR: $IQR)"
    printf '%s\n' "- trimmed_mean: $TRIMMED_MEAN dec_tps ($ESTIMATOR)"
    printf '%s\n' "- min..max: $MIN .. $MAX"
} > "$OUT_DIR/summary.md"

log "results: $OUT_DIR/summary.md"
log "exit 0"
