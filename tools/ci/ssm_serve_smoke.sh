#!/usr/bin/env bash
# tools/ci/ssm_serve_smoke.sh -- non-destructive serve-path smoke for SSM models.
#
# Starts `hawking serve`, verifies the HTTP surface, sends one native streaming
# generation request, records metrics, and shuts the server down. This proves an
# SSM model family works through the server front door, not just `generate`.
set -u

REPO="${REPO:-$HOME/Downloads/hawking}"
cd "$REPO" || exit 2

MODEL="${1:-${MODEL:-models/rwkv7-g1-04-sft-Q4_K_M.gguf}}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-reports/serve-smoke/$STAMP}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-$((18080 + ($$ % 1000)))}"
ADDR="${HOST}:${PORT}"
BASE_URL="http://${ADDR}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
START_TIMEOUT_SECS="${START_TIMEOUT_SECS:-420}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-180}"
MAX_TOKENS="${MAX_TOKENS:-16}"
PROMPT="${PROMPT:-Summarize the key product risk in one concise sentence: long-context systems must be fast, correct, and isolated between requests.}"

mkdir -p "$OUT"
COMMAND_LOG="$OUT/commands.log"
SUMMARY="$OUT/summary.md"
SERVER_LOG="$OUT/server.log"
SERVER_PID=""
FAIL=0

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$COMMAND_LOG"
}

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

fail() {
  log "FAIL $*"
  FAIL=1
}

run_capture() {
  local name="$1"; shift
  local logfile="$OUT/${name}.log"
  log "BEGIN $name"
  log "+ $*"
  if "$@" >"$logfile" 2>&1; then
    log "PASS  $name -> $logfile"
  else
    local rc=$?
    fail "$name rc=$rc -> $logfile"
  fi
}

{
  echo "# SSM Serve Smoke - $STAMP"
  echo
  echo "- Repo: \`$REPO\`"
  echo "- Model: \`$MODEL\`"
  echo "- Address: \`$ADDR\`"
  echo "- Output: \`$OUT\`"
  echo
  echo "## Initial Git State"
  echo '```'
  git status --short
  echo '```'
} >"$SUMMARY"

if [ ! -x ./target/release/hawking ]; then
  fail "missing ./target/release/hawking; run cargo build --release first"
fi

if [ ! -f "$MODEL" ]; then
  fail "missing model file: $MODEL"
fi

if [ "$FAIL" = "0" ]; then
  log "starting hawking serve model=$MODEL addr=$ADDR"
  ./target/release/hawking serve \
    --weights "$MODEL" \
    --addr "$ADDR" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --explain-performance >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  log "server pid=$SERVER_PID log=$SERVER_LOG"

  ready=0
  waited=0
  while [ "$waited" -lt "$START_TIMEOUT_SECS" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      fail "server exited before healthz became ready; see $SERVER_LOG"
      break
    fi
    if curl -fsS "$BASE_URL/healthz" >"$OUT/healthz.log" 2>"$OUT/healthz.err"; then
      ready=1
      log "healthz ready after ${waited}s"
      break
    fi
    sleep 2
    waited=$((waited + 2))
  done

  if [ "$ready" != "1" ]; then
    fail "server did not become ready within ${START_TIMEOUT_SECS}s"
  fi
fi

if [ "$FAIL" = "0" ]; then
  run_capture "models" curl -fsS "$BASE_URL/v1/models"
  run_capture "metrics_before" curl -fsS "$BASE_URL/metrics"

  GEN_BODY="$OUT/generate_request.json"
  if ! python3 - "$GEN_BODY" "$PROMPT" "$MAX_TOKENS" <<'PY'
import json
import sys

path, prompt, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path, "w") as f:
    json.dump(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 5,
        },
        f,
    )
PY
  then
    fail "failed to write native generate request JSON"
  fi

  if [ "$FAIL" = "0" ]; then
    log "BEGIN native_generate_sse"
    if curl -N -sS --max-time "$REQUEST_TIMEOUT_SECS" \
      -H 'content-type: application/json' \
      --data-binary "@$GEN_BODY" \
      "$BASE_URL/v1/hawking/generate" >"$OUT/generate.sse" 2>"$OUT/generate.err"; then
      if grep -Fq 'data: [DONE]' "$OUT/generate.sse" &&
        grep -q '"stats"' "$OUT/generate.sse"; then
        log "PASS  native_generate_sse -> $OUT/generate.sse"
      else
        fail "native_generate_sse missing [DONE] or stats; see $OUT/generate.sse"
      fi
    else
      rc=$?
      fail "native_generate_sse rc=$rc -> $OUT/generate.err"
    fi
  fi

  run_capture "metrics_after" curl -fsS "$BASE_URL/metrics"
fi

{
  echo
  echo "## Step Results"
  echo '```'
  cat "$COMMAND_LOG"
  echo '```'
  echo
  echo "## Server Log Tail"
  echo '```'
  tail -80 "$SERVER_LOG" 2>/dev/null || true
  echo '```'
  echo
  echo "## Native Generate SSE Tail"
  echo '```'
  tail -40 "$OUT/generate.sse" 2>/dev/null || true
  echo '```'
  echo
  if [ "$FAIL" = "0" ]; then
    echo "**Result:** PASS (serve path loaded model and completed native streaming generation)."
  else
    echo "**Result:** FAIL/ATTENTION (inspect logs above)."
  fi
} >>"$SUMMARY"

log "SSM serve smoke complete fail=$FAIL summary=$SUMMARY"
exit "$FAIL"
