#!/usr/bin/env bash
# =============================================================================
# tools/bench/report_card.sh — single-command full-comparison bench.
#
# Runs every relevant lane (dismantle-default, dismantle-fast, two serve lanes,
# llama-cli, llama-server-b8) and prints a comparison table:
#
#   lane | dec_tps | J/tok_GPU | J/tok_pkg | wall_s | readback_bytes/tok | lane_type | feature_flags | git_sha
#
# CONTAMINATION NOTE (printed in the output header):
#   Absolute numbers require a clean room (agent closed). Relative lane ratios
#   are contamination-robust and valid with the agent open.
#
# USAGE:
#   tools/bench/report_card.sh
#   ONLY=dismantle-fast,llama-cli tools/bench/report_card.sh
#   TOKENS=200 PROMPT="Tell me about the history of AI." tools/bench/report_card.sh
#   GGUF=models/other.gguf tools/bench/report_card.sh
#
# ENVIRONMENT:
#   TOKENS           decode tokens per lane (default: 200)
#   PROMPT           prompt text (default: "Tell me about the history of machine learning.")
#   GGUF             path to Q4_K_M GGUF (default: models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   DBIN             dismantle binary (default: ./target/release/hawking)
#   PROFILE          kernel profile JSON (default: profiles/qwen3b-instruct-q4k.m3pro18.json)
#   LLAMA_BIN        llama-cli/llama-completion binary; auto-detected if unset
#   LLAMA_SERVER_BIN llama-server binary; auto-detected if unset
#   SERVE_PORT       base port for dismantle serve lanes (default: 8282)
#   LLAMA_SERVE_PORT port for llama-server lane (default: 8283)
#   SAMPLE_MS        macmon sampling interval ms (default: 200)
#   RUN_TIMEOUT_SEC  per-run wall-time cap (default: 600)
#   ONLY             comma-separated subset of lane names to run, e.g.
#                    "dismantle-default,dismantle-fast,llama-cli"
#                    Valid names: dismantle-default  dismantle-fast
#                                 hawking-serve-full-logits
#                                 hawking-serve-greedy-b1
#                                 hawking-serve-greedy-b8
#                                 llama-cli  llama-server-b8
#
# NOTES:
#   - "--kernel-profile" (hardware autotune JSON) is NOT "--profile fast"
#     (the runtime lever bundle). The table's feature_flags column makes the
#     distinction explicit.
#   - dismantle serve lanes start their own server, wait for /healthz, run
#     one warm-up + the timed measurement, then shut down.
#   - For dismantle serve lanes, /metrics is polled after the run to determine
#     readback_bytes/tok and whether the greedy or full-logits path was used.
#   - macmon is required for J/tok; if missing, J/tok columns show "N/A".
#   - HAWKING_SERVE_FORCE_LOGITS=1 overrides greedy routing for the
#     "hawking-serve-full-logits" lane so you can measure the old path.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

# ── Config ────────────────────────────────────────────────────────────────────
TOKENS="${TOKENS:-200}"
PROMPT="${PROMPT:-Tell me about the history of machine learning.}"
GGUF="${GGUF:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
DBIN="${DBIN:-./target/release/hawking}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
SAMPLE_MS="${SAMPLE_MS:-200}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-600}"
SERVE_PORT="${SERVE_PORT:-8282}"
LLAMA_SERVE_PORT="${LLAMA_SERVE_PORT:-8283}"
ONLY="${ONLY:-}"
NICE=(nice -n 19 taskpolicy -b)
TMPD="/tmp/report_card_$$"
mkdir -p "$TMPD"
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# ── State for active server PIDs (cleaned on exit) ───────────────────────────
SERVE_PID=""
LLAMA_SERVE_PID=""
SAMP_PIDS=()

cleanup() {
    [[ -n "$SERVE_PID" ]] && { kill "$SERVE_PID" 2>/dev/null || true; wait "$SERVE_PID" 2>/dev/null || true; }
    [[ -n "$LLAMA_SERVE_PID" ]] && { kill "$LLAMA_SERVE_PID" 2>/dev/null || true; wait "$LLAMA_SERVE_PID" 2>/dev/null || true; }
    for _sp in "${SAMP_PIDS[@]:-}"; do
        [[ -n "$_sp" ]] || continue
        pkill -P "$_sp" 2>/dev/null || true
        kill "$_sp" 2>/dev/null || true
        wait "$_sp" 2>/dev/null || true
    done
    rm -rf "$TMPD"
}
trap cleanup EXIT

# ── Helpers ───────────────────────────────────────────────────────────────────
want() { [[ -z "$ONLY" || ",$ONLY," == *",$1,"* ]]; }
have() { [[ -x "$1" ]] 2>/dev/null || command -v "$1" >/dev/null 2>&1; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
avg_file() { awk '{s+=$1;n++} END{if(n>0)printf "%.4f",s/n; else print "0"}' "$1" 2>/dev/null; }

# per-engine dec_tps parsers
tps_dismantle() { grep -oE 'dec_tps=[0-9.]+' "$1" | grep -oE '[0-9.]+' | tail -1; }
tps_llama() {
    local t
    # prefer "--perf" / "eval time" line
    t=$(tail -200 "$1" | grep -iE 'eval time.*tokens per second' | grep -vi 'prompt eval' \
        | sed -nE 's/.*[[:space:],(]([0-9]+(\.[0-9]+)?|inf)[[:space:]]+tokens per second.*/\1/p' | tail -1)
    [[ -n "$t" ]] && { echo "$t"; return; }
    # fallback: "Generation: X t/s"
    tail -200 "$1" | grep -oiE 'Generation:[^0-9]*[0-9.]+ *t/s' | grep -oE '[0-9.]+' | tail -1
}

# SSE token counter
count_sse_toks() {
    awk '/^data: / && $0 != "data: [DONE]"{n++} END{print n+0}' "$1" 2>/dev/null || echo 0
}

# run_with_timeout — portable (coreutils timeout, gtimeout, or perl fallback)
run_with_timeout() {
    local secs="$1"; shift
    if command -v timeout >/dev/null 2>&1; then timeout "$secs" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then gtimeout "$secs" "$@"
    else perl -e 'alarm shift; exec { $ARGV[0] } @ARGV or die "exec: $!"' "$secs" "$@"; fi
}

# ── macmon sampler ────────────────────────────────────────────────────────────
HAS_MACMON=0
command -v macmon >/dev/null 2>&1 && HAS_MACMON=1

sample_macmon() {  # $1=pkg_file  $2=gpu_file
    if [[ "$HAS_MACMON" -eq 0 ]]; then return; fi
    macmon pipe -i "$SAMPLE_MS" 2>/dev/null | while IFS= read -r line; do
        /usr/bin/python3 -c 'import sys,json
try:
  d=json.load(sys.stdin)
except Exception:
  sys.exit(0)
allp=d.get("all_power")
if allp is None:
  allp=sum(d.get(k,0) or 0 for k in ("cpu_power","gpu_power","ane_power"))
gpu=d.get("gpu_power",0) or 0
print(f"{allp:.4f}\t{gpu:.4f}")' <<< "$line" | {
            IFS=$'\t' read -r p g
            [[ -n "${p:-}" ]] && printf '%s\n' "$p" >> "$1"
            [[ -n "${g:-}" ]] && printf '%s\n' "$g" >> "$2"
        }
    done
}

stop_sampler() {  # $1 sampler_pid
    pkill -P "$1" 2>/dev/null || true
    kill "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
}

jtok_or_na() {  # $1=avg_power_W  $2=tps  → J/tok or N/A
    if [[ "$HAS_MACMON" -eq 0 ]]; then echo "N/A"; return; fi
    awk -v p="$1" -v t="$2" 'BEGIN{if(t+0>0&&p+0>0)printf "%.4f",p/t; else print "N/A"}'
}

# ── Results accumulator ───────────────────────────────────────────────────────
# Each lane appends one pipe-delimited record to $TMPD/results:
#   name|dec_tps|J/tok_GPU|J/tok_pkg|wall_s|readback_bytes_per_tok|lane_type|feature_flags|git_sha
: > "$TMPD/results"

record_lane() {
    local name="$1" tps="$2" jg="$3" jp="$4" wall="$5" rb="$6" ltype="$7" flags="$8"
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "$name" "${tps:-?}" "${jg:-N/A}" "${jp:-N/A}" "${wall:-?}" \
        "${rb:-N/A}" "${ltype:-?}" "${flags:-}" "$GIT_SHA" \
        >> "$TMPD/results"
}

# ── Lane 1 & 2: dismantle generate (default / --profile fast) ─────────────────
run_dismantle_generate() {
    local lane="$1"   # "dismantle-default" or "dismantle-fast"
    local fast="$2"   # "0" or "1"

    [[ -x "$DBIN" ]] || { warn "$DBIN not found — skipping $lane"; return; }
    [[ -f "$GGUF"  ]] || { warn "GGUF $GGUF not found — skipping $lane"; return; }

    local pkgf="$TMPD/${lane}_pkg" gpuf="$TMPD/${lane}_gpu" out="$TMPD/${lane}.log"
    : > "$pkgf"; : > "$gpuf"

    local profile_arg=()
    local kprofile_arg=()
    [[ "$fast" == "1" ]] && profile_arg=(--profile fast)
    [[ -f "$PROFILE" ]] && kprofile_arg=(--kernel-profile "$PROFILE")

    printf '\n[%s]\n' "$lane"
    sample_macmon "$pkgf" "$gpuf" & local smp=$!; SAMP_PIDS+=("$smp")
    local t0 t1
    t0=$(date +%s.%N)
    run_with_timeout "$RUN_TIMEOUT_SEC" \
        "${NICE[@]}" "$DBIN" generate \
            ${profile_arg[@]+"${profile_arg[@]}"} \
            --weights "$GGUF" ${kprofile_arg[@]+"${kprofile_arg[@]}"} \
            --prompt "$PROMPT" --max-new-tokens "$TOKENS" \
            --temperature 0 --seed 0 --max-stall-ms 30000 \
        </dev/null > "$out" 2>&1
    local rc=$?
    t1=$(date +%s.%N)
    stop_sampler "$smp"

    local wall tps apkg agpu jg jp
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",b-a}')
    tps=$(tps_dismantle "$out"); [[ -z "$tps" ]] && tps="?"
    apkg=$(avg_file "$pkgf"); agpu=$(avg_file "$gpuf")
    jg=$(jtok_or_na "$agpu" "$tps")
    jp=$(jtok_or_na "$apkg" "$tps")

    if [[ $rc -eq 124 || $rc -eq 142 ]]; then
        warn "$lane timed out after ${RUN_TIMEOUT_SEC}s"
    elif [[ $rc -ne 0 || "$tps" == "?" ]]; then
        warn "$lane rc=$rc tps='$tps' — see $out"; tail -6 "$out" >&2
    fi
    printf '  dec_tps=%s  J/tok_GPU=%s  J/tok_pkg=%s  wall=%ss\n' "$tps" "$jg" "$jp" "$wall"

    local flags="kernel-profile=$(basename "${PROFILE:-none}")"
    [[ "$fast" == "1" ]] && flags="--profile fast + $flags"
    local ltype="generate"
    [[ "$fast" == "1" ]] && ltype="generate(fast)"
    record_lane "$lane" "$tps" "$jg" "$jp" "$wall" "N/A" "$ltype" "$flags"
}

# ── Server lifecycle ──────────────────────────────────────────────────────────
start_dismantle_serve() {  # $1=port  $2=max_batch  $3=extra_envvar (or "")  → sets SERVE_PID
    local port="$1" max_b="$2" extra_env="${3:-}"
    local url="http://127.0.0.1:${port}"
    local slog="$TMPD/serve_${port}.log"

    local cmd=("${NICE[@]}" "$DBIN" serve
        --weights "$GGUF"
        --addr "127.0.0.1:${port}"
        --max-batch-size "$max_b")
    [[ -f "$PROFILE" ]] && cmd+=(--kernel-profile "$PROFILE")

    # Build the env block: base serve defaults + optional override
    local env_prefix=""
    [[ -n "$extra_env" ]] && env_prefix="$extra_env "

    printf '  starting dismantle serve (port=%s, max_batch=%s%s)...' \
        "$port" "$max_b" "${extra_env:+, $extra_env}"
    eval "env ${env_prefix}${cmd[*]} > '$slog' 2>&1 &"
    SERVE_PID=$!

    local wait=0
    until curl -sf "${url}/healthz" >/dev/null 2>&1; do
        sleep 1; wait=$(( wait + 1 ))
        kill -0 "$SERVE_PID" 2>/dev/null || { printf ' DIED\n'; tail -8 "$slog" >&2; SERVE_PID=""; return 1; }
        [[ "$wait" -gt 120 ]] && { printf ' TIMEOUT\n'; kill "$SERVE_PID" 2>/dev/null; SERVE_PID=""; return 1; }
    done
    printf ' ready (%ss)\n' "$wait"
    return 0
}

stop_dismantle_serve() {
    [[ -n "$SERVE_PID" ]] || return
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
    SERVE_PID=""
    sleep 1  # brief GPU drain before next server
}

# ── Fire one SSE request and wait for completion ──────────────────────────────
fire_sse() {  # $1=url  $2=logfile
    local url="$1" logf="$2"
    local pj
    pj=$(printf '%s' "$PROMPT" | python3 -c "import sys,json;print(json.dumps(sys.stdin.read()))" 2>/dev/null \
         || printf '"%s"' "$PROMPT")
    : > "$logf"
    curl --no-buffer --silent --max-time "$RUN_TIMEOUT_SEC" \
        -X POST "${url}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":${pj},\"max_tokens\":${TOKENS},\"stream\":true,\"temperature\":0,\"seed\":0}" \
        > "$logf" 2>&1 &
    local cpid=$!
    local ss; ss=$(date +%s)
    while :; do
        grep -q '^data: \[DONE\]' "$logf" 2>/dev/null && break
        local n; n=$(count_sse_toks "$logf")
        [[ "$n" -ge "$TOKENS" ]] && break
        local st; st=$(ps -o stat= -p "$cpid" 2>/dev/null || true)
        [[ -z "$st" || "$st" == *Z* ]] && break
        [[ $(( $(date +%s) - ss )) -ge "$RUN_TIMEOUT_SEC" ]] && break
        sleep 0.05
    done
    kill "$cpid" 2>/dev/null || true
    wait "$cpid" 2>/dev/null || true
}

# ── Read /metrics from a running server ──────────────────────────────────────
read_metrics() {  # $1=url  → sets METRIC_GREEDY METRIC_LOGITS METRIC_READBACK_BYTES
    METRIC_GREEDY=0; METRIC_LOGITS=0; METRIC_READBACK_BYTES=0
    local raw
    raw=$(curl -sf "${1}/metrics" 2>/dev/null) || return
    METRIC_GREEDY=$(printf '%s\n' "$raw" | grep '^hawking_greedy_decode_steps_total' | awk '{print $2}')
    METRIC_LOGITS=$(printf '%s\n' "$raw" | grep '^hawking_logits_decode_steps_total' | awk '{print $2}')
    METRIC_READBACK_BYTES=$(printf '%s\n' "$raw" | grep '^hawking_gpu_readback_bytes_total' | awk '{print $2}')
    METRIC_GREEDY="${METRIC_GREEDY:-0}"
    METRIC_LOGITS="${METRIC_LOGITS:-0}"
    METRIC_READBACK_BYTES="${METRIC_READBACK_BYTES:-0}"
}

# Compute readback_bytes/tok from metrics delta vs token count
readback_per_tok() {  # $1=readback_bytes  $2=token_count
    awk -v b="$1" -v t="$2" 'BEGIN{if(t+0>0&&b+0>0)printf "%.0f",b/t; else print "N/A"}'
}

# ── Lane 3: hawking-serve-full-logits (B=1, temperature=0, force full logits)
run_serve_full_logits() {
    local lane="hawking-serve-full-logits"
    [[ -x "$DBIN" ]] || { warn "$DBIN not found — skipping $lane"; return; }
    [[ -f "$GGUF"  ]] || { warn "GGUF $GGUF not found — skipping $lane"; return; }

    printf '\n[%s]\n' "$lane"
    # HAWKING_SERVE_FORCE_LOGITS=1 disables the greedy-lane routing so even
    # temperature=0 requests materialise full logits. If the env var is not
    # wired in the binary it's a no-op and the server may still take the greedy
    # lane; the /metrics readback column will reveal the actual path taken.
    start_dismantle_serve "$SERVE_PORT" 1 "HAWKING_SERVE_FORCE_LOGITS=1" || {
        record_lane "$lane" "?" "N/A" "N/A" "?" "N/A" "serve/full-logits" "HAWKING_SERVE_FORCE_LOGITS=1"
        return
    }
    local url="http://127.0.0.1:${SERVE_PORT}"

    # Warm-up
    printf '  warmup...'
    fire_sse "$url" "$TMPD/${lane}_warmup.log"
    printf ' done\n'

    # Reset metrics baseline (read before timed run)
    read_metrics "$url"
    local greedy0=$METRIC_GREEDY logits0=$METRIC_LOGITS rb0=$METRIC_READBACK_BYTES

    local pkgf="$TMPD/${lane}_pkg" gpuf="$TMPD/${lane}_gpu"
    : > "$pkgf"; : > "$gpuf"
    sample_macmon "$pkgf" "$gpuf" & local smp=$!; SAMP_PIDS+=("$smp")

    local t0 t1
    t0=$(date +%s.%N)
    fire_sse "$url" "$TMPD/${lane}_run.log"
    t1=$(date +%s.%N)
    stop_sampler "$smp"

    local wall toks apkg agpu jg jp
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",b-a}')
    toks=$(count_sse_toks "$TMPD/${lane}_run.log")
    apkg=$(avg_file "$pkgf"); agpu=$(avg_file "$gpuf")
    local tps="?"
    [[ "$wall" != "0" && "$toks" -gt 0 ]] && tps=$(awk -v t="$toks" -v w="$wall" 'BEGIN{printf "%.2f",t/w}')
    jg=$(jtok_or_na "$agpu" "$tps")
    jp=$(jtok_or_na "$apkg" "$tps")

    # Post-run metrics
    read_metrics "$url"
    local greedy_delta logits_delta rb_delta
    greedy_delta=$(( METRIC_GREEDY - greedy0 ))
    logits_delta=$(( METRIC_LOGITS - logits0 ))
    rb_delta=$(( METRIC_READBACK_BYTES - rb0 ))
    local rb_per_tok; rb_per_tok=$(readback_per_tok "$rb_delta" "$toks")
    local ltype="serve/full-logits(g=${greedy_delta},l=${logits_delta})"

    stop_dismantle_serve

    printf '  tps=%s  J/tok_GPU=%s  J/tok_pkg=%s  wall=%ss  toks=%s  readback=%s bytes/tok\n' \
        "$tps" "$jg" "$jp" "$wall" "$toks" "$rb_per_tok"
    printf '  metrics: greedy_steps=%s  logits_steps=%s  readback_bytes=%s\n' \
        "$greedy_delta" "$logits_delta" "$rb_delta"

    local flags="HAWKING_SERVE_FORCE_LOGITS=1 kernel-profile=$(basename "${PROFILE:-none}")"
    record_lane "$lane" "$tps" "$jg" "$jp" "$wall" "$rb_per_tok" "$ltype" "$flags"
}

# ── Lane 4 & 5: hawking-serve-greedy-b1 / b8 ───────────────────────────────
run_serve_greedy() {
    local lane="$1"   # "hawking-serve-greedy-b1" or "hawking-serve-greedy-b8"
    local bsize="$2"  # 1 or 8

    [[ -x "$DBIN" ]] || { warn "$DBIN not found — skipping $lane"; return; }
    [[ -f "$GGUF"  ]] || { warn "GGUF $GGUF not found — skipping $lane"; return; }

    printf '\n[%s]\n' "$lane"
    start_dismantle_serve "$SERVE_PORT" "$bsize" "" || {
        record_lane "$lane" "?" "N/A" "N/A" "?" "N/A" "serve/greedy(b${bsize})" ""
        return
    }
    local url="http://127.0.0.1:${SERVE_PORT}"

    # Warm-up
    printf '  warmup...'
    fire_sse "$url" "$TMPD/${lane}_warmup.log"
    printf ' done\n'

    # Baseline metrics snapshot
    read_metrics "$url"
    local greedy0=$METRIC_GREEDY logits0=$METRIC_LOGITS rb0=$METRIC_READBACK_BYTES

    local pkgf="$TMPD/${lane}_pkg" gpuf="$TMPD/${lane}_gpu"
    : > "$pkgf"; : > "$gpuf"
    sample_macmon "$pkgf" "$gpuf" & local smp=$!; SAMP_PIDS+=("$smp")

    # Fire B concurrent requests for B=8; single for B=1
    local t0 t1 total_toks=0
    local logs=() pids=()
    for slot in $(seq 0 $(( bsize - 1 ))); do
        local _lf="$TMPD/${lane}_run_s${slot}.log"
        logs+=("$_lf"); : > "$_lf"
    done

    t0=$(date +%s.%N)
    for slot in $(seq 0 $(( bsize - 1 ))); do
        fire_sse "$url" "${logs[$slot]}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    t1=$(date +%s.%N)
    stop_sampler "$smp"

    for slot in $(seq 0 $(( bsize - 1 ))); do
        local t; t=$(count_sse_toks "${logs[$slot]}")
        total_toks=$(( total_toks + t ))
    done

    local wall tps apkg agpu jg jp
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",b-a}')
    tps="?"
    [[ "$wall" != "0" && "$total_toks" -gt 0 ]] && tps=$(awk -v t="$total_toks" -v w="$wall" 'BEGIN{printf "%.2f",t/w}')
    # J/tok from aggregate energy divided by aggregate tps (wall-dominated by GPU work)
    apkg=$(avg_file "$pkgf"); agpu=$(avg_file "$gpuf")
    jg=$(jtok_or_na "$agpu" "$tps")
    jp=$(jtok_or_na "$apkg" "$tps")

    # Post-run metrics
    read_metrics "$url"
    local greedy_delta logits_delta rb_delta
    greedy_delta=$(( METRIC_GREEDY - greedy0 ))
    logits_delta=$(( METRIC_LOGITS - logits0 ))
    rb_delta=$(( METRIC_READBACK_BYTES - rb0 ))
    local rb_per_tok; rb_per_tok=$(readback_per_tok "$rb_delta" "$total_toks")
    local ltype="serve/greedy(b${bsize},g=${greedy_delta},l=${logits_delta})"

    stop_dismantle_serve

    printf '  agg_tps=%s  J/tok_GPU=%s  J/tok_pkg=%s  wall=%ss  total_toks=%s  readback=%s bytes/tok\n' \
        "$tps" "$jg" "$jp" "$wall" "$total_toks" "$rb_per_tok"
    printf '  metrics: greedy_steps=%s  logits_steps=%s  readback_bytes=%s\n' \
        "$greedy_delta" "$logits_delta" "$rb_delta"

    local flags="kernel-profile=$(basename "${PROFILE:-none}")"
    record_lane "$lane" "$tps" "$jg" "$jp" "$wall" "$rb_per_tok" "$ltype" "$flags"
}

# ── Lane 6: llama-cli ─────────────────────────────────────────────────────────
resolve_llama_bin() {
    if [[ -n "${LLAMA_BIN:-}" && -x "${LLAMA_BIN:-}" ]]; then echo "$LLAMA_BIN"; return; fi
    for cand in llama-completion llama-cli llama \
                /usr/local/bin/llama-completion /usr/local/bin/llama-cli \
                /opt/homebrew/bin/llama-completion /opt/homebrew/bin/llama-cli \
                /opt/homebrew/bin/llama; do
        local r; r=$(command -v "$cand" 2>/dev/null || true)
        [[ -x "${r:-}" ]] && { echo "$r"; return; }
    done
}

resolve_llama_server_bin() {
    if [[ -n "${LLAMA_SERVER_BIN:-}" && -x "${LLAMA_SERVER_BIN:-}" ]]; then echo "$LLAMA_SERVER_BIN"; return; fi
    for cand in llama-server \
                /usr/local/bin/llama-server \
                /opt/homebrew/bin/llama-server; do
        local r; r=$(command -v "$cand" 2>/dev/null || true)
        [[ -x "${r:-}" ]] && { echo "$r"; return; }
    done
}

run_llama_cli() {
    local lane="llama-cli"
    local bin; bin=$(resolve_llama_bin)
    if [[ -z "${bin:-}" ]]; then
        warn "llama CLI not found (set LLAMA_BIN= or brew install llama.cpp) — skipping $lane"
        return
    fi
    [[ -f "$GGUF" ]] || { warn "GGUF $GGUF not found — skipping $lane"; return; }

    printf '\n[%s]  binary=%s\n' "$lane" "$(basename "$bin")"

    # Mode args: llama-cli needs --single-turn; older llama-completion uses -no-cnv
    local mode_args=()
    case "$(basename "$bin")" in
        llama-cli|llama) mode_args=(--single-turn) ;;
        *)               mode_args=(-no-cnv) ;;
    esac

    local pkgf="$TMPD/${lane}_pkg" gpuf="$TMPD/${lane}_gpu" out="$TMPD/${lane}.log"
    : > "$pkgf"; : > "$gpuf"
    sample_macmon "$pkgf" "$gpuf" & local smp=$!; SAMP_PIDS+=("$smp")
    local t0 t1
    t0=$(date +%s.%N)
    run_with_timeout "$RUN_TIMEOUT_SEC" \
        "${NICE[@]}" "$bin" \
            -m "$GGUF" -p "$PROMPT" -n "$TOKENS" \
            --temp 0 --seed 0 -ngl 99 ${mode_args[@]+"${mode_args[@]}"} \
            --no-display-prompt --no-warmup --perf \
        </dev/null > "$out" 2>&1
    local rc=$?
    t1=$(date +%s.%N)
    stop_sampler "$smp"

    local wall tps apkg agpu jg jp
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",b-a}')
    tps=$(tps_llama "$out"); [[ -z "$tps" ]] && tps="?"
    apkg=$(avg_file "$pkgf"); agpu=$(avg_file "$gpuf")
    jg=$(jtok_or_na "$agpu" "$tps")
    jp=$(jtok_or_na "$apkg" "$tps")

    if [[ $rc -eq 124 || $rc -eq 142 ]]; then
        warn "$lane timed out"
    elif [[ $rc -ne 0 || "$tps" == "?" ]]; then
        warn "$lane rc=$rc tps='$tps' — see $out"; tail -6 "$out" >&2
    fi
    printf '  dec_tps=%s  J/tok_GPU=%s  J/tok_pkg=%s  wall=%ss\n' "$tps" "$jg" "$jp" "$wall"

    local llver; llver=$("$bin" --version 2>&1 | head -1 | grep -oE '[Vv]?[0-9]+[^ ]*' | head -1 || true)
    record_lane "$lane" "$tps" "$jg" "$jp" "$wall" "N/A" "llama-cli" "ver=${llver:-?}"
}

# ── Lane 7: llama-server-b8 ───────────────────────────────────────────────────
run_llama_server_b8() {
    local lane="llama-server-b8"
    local bin; bin=$(resolve_llama_server_bin)
    if [[ -z "${bin:-}" ]]; then
        warn "llama-server not found (set LLAMA_SERVER_BIN= or brew install llama.cpp) — skipping $lane"
        return
    fi
    [[ -f "$GGUF" ]] || { warn "GGUF $GGUF not found — skipping $lane"; return; }

    printf '\n[%s]  binary=%s\n' "$lane" "$(basename "$bin")"
    local url="http://127.0.0.1:${LLAMA_SERVE_PORT}"
    local slog="$TMPD/llama_serve.log"

    printf '  starting llama-server (port=%s, --parallel 8)...' "$LLAMA_SERVE_PORT"
    "${NICE[@]}" "$bin" \
        -m "$GGUF" -ngl 99 --parallel 8 \
        --host 127.0.0.1 --port "$LLAMA_SERVE_PORT" \
        --log-disable \
        > "$slog" 2>&1 &
    LLAMA_SERVE_PID=$!

    local wait=0
    until curl -sf "${url}/health" >/dev/null 2>&1 || curl -sf "${url}/v1/models" >/dev/null 2>&1; do
        sleep 1; wait=$(( wait + 1 ))
        kill -0 "$LLAMA_SERVE_PID" 2>/dev/null || { printf ' DIED\n'; tail -8 "$slog" >&2; LLAMA_SERVE_PID=""; break; }
        [[ "$wait" -gt 120 ]] && { printf ' TIMEOUT\n'; kill "$LLAMA_SERVE_PID" 2>/dev/null; LLAMA_SERVE_PID=""; break; }
    done
    if [[ -z "$LLAMA_SERVE_PID" ]]; then
        record_lane "$lane" "?" "N/A" "N/A" "?" "N/A" "llama-server/b8" ""
        return
    fi
    printf ' ready (%ss)\n' "$wait"

    # Warm-up: single request
    printf '  warmup...'
    fire_sse "$url" "$TMPD/${lane}_warmup.log"
    printf ' done\n'

    local pkgf="$TMPD/${lane}_pkg" gpuf="$TMPD/${lane}_gpu"
    : > "$pkgf"; : > "$gpuf"
    local logs=() pids=()
    for slot in $(seq 0 7); do
        local _lf="$TMPD/${lane}_run_s${slot}.log"
        logs+=("$_lf"); : > "$_lf"
    done

    sample_macmon "$pkgf" "$gpuf" & local smp=$!; SAMP_PIDS+=("$smp")
    local t0 t1 total_toks=0
    t0=$(date +%s.%N)
    for slot in $(seq 0 7); do fire_sse "$url" "${logs[$slot]}" & pids+=("$!"); done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    t1=$(date +%s.%N)
    stop_sampler "$smp"

    for slot in $(seq 0 7); do
        local t; t=$(count_sse_toks "${logs[$slot]}"); total_toks=$(( total_toks + t ))
    done

    local wall tps apkg agpu jg jp
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",b-a}')
    tps="?"
    [[ "$wall" != "0" && "$total_toks" -gt 0 ]] && tps=$(awk -v t="$total_toks" -v w="$wall" 'BEGIN{printf "%.2f",t/w}')
    apkg=$(avg_file "$pkgf"); agpu=$(avg_file "$gpuf")
    jg=$(jtok_or_na "$agpu" "$tps")
    jp=$(jtok_or_na "$apkg" "$tps")

    kill "$LLAMA_SERVE_PID" 2>/dev/null || true
    wait "$LLAMA_SERVE_PID" 2>/dev/null || true
    LLAMA_SERVE_PID=""

    printf '  agg_tps=%s  J/tok_GPU=%s  J/tok_pkg=%s  wall=%ss  total_toks=%s\n' \
        "$tps" "$jg" "$jp" "$wall" "$total_toks"

    local llver; llver=$("$bin" --version 2>&1 | head -1 | grep -oE '[Vv]?[0-9]+[^ ]*' | head -1 || true)
    record_lane "$lane" "$tps" "$jg" "$jp" "$wall" "N/A" "llama-server/b8" "ver=${llver:-?}"
}

# =============================================================================
# MAIN — preflight, runs, table
# =============================================================================

# ── Diagnostics ───────────────────────────────────────────────────────────────
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || echo "unknown")
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
RAM_GB=$(awk -v b="$RAM_BYTES" 'BEGIN{printf "%d",b/1073741824}')
MACOS=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
BUILD_MTIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$DBIN" 2>/dev/null || echo "unknown")
GGUF_HASH="?"
if command -v b3sum >/dev/null 2>&1 && [[ -f "$GGUF" ]]; then
    GGUF_HASH=$(b3sum --no-names "$GGUF" 2>/dev/null | cut -c1-12 || echo "?")
elif command -v md5 >/dev/null 2>&1 && [[ -f "$GGUF" ]]; then
    GGUF_HASH=$(md5 -q "$GGUF" 2>/dev/null | cut -c1-12 || echo "?")
fi
LOCAL_MODS=$(git status --short 2>/dev/null | wc -l | tr -d ' ')

echo "============================================================="
echo " dismantle report_card — $(date '+%Y-%m-%d %H:%M')"
echo "============================================================="
printf ' Device  : %s  %s GB RAM  macOS %s\n' "$CHIP" "$RAM_GB" "$MACOS"
printf ' Model   : %s  (b3sum:%s)\n' "$(basename "${GGUF}")" "$GGUF_HASH"
printf ' Profile : %s\n' "$(basename "${PROFILE:-none}")"
printf ' GitSHA  : %s  local_mods: %s\n' "$GIT_SHA" "$LOCAL_MODS"
printf ' Binary  : %s  built: %s\n' "$DBIN" "$BUILD_MTIME"
printf ' Tokens  : %s   macmon: %s\n' "$TOKENS" "$(macmon --version 2>/dev/null || echo 'NOT FOUND — J/tok = N/A')"
echo
echo " !! CONTAMINATION NOTE: absolute numbers require clean room (agent closed)."
echo "    Relative lane-vs-lane ratios are contamination-robust."
echo "============================================================="

[[ -z "$ONLY" ]] || printf ' ONLY=%s\n\n' "$ONLY"

# ── Run lanes ─────────────────────────────────────────────────────────────────
want "dismantle-default"          && run_dismantle_generate "dismantle-default" "0"
want "dismantle-fast"             && run_dismantle_generate "dismantle-fast" "1"
want "hawking-serve-full-logits" && run_serve_full_logits
want "hawking-serve-greedy-b1"  && run_serve_greedy "hawking-serve-greedy-b1" 1
want "hawking-serve-greedy-b8"  && run_serve_greedy "hawking-serve-greedy-b8" 8
want "llama-cli"                  && run_llama_cli
want "llama-server-b8"            && run_llama_server_b8

# ── Print table ───────────────────────────────────────────────────────────────
echo
echo "============================================================="
echo " REPORT CARD — Qwen2.5-3B Q4_K_M, N=${TOKENS}, greedy temp=0"
echo "============================================================="
printf '%-32s %9s %11s %11s %8s %22s %30s %28s %10s\n' \
    lane dec_tps J/tok_GPU J/tok_pkg wall_s readback_bytes/tok lane_type feature_flags git_sha
printf '%s\n' "$(printf '─%.0s' {1..178})"

while IFS='|' read -r name tps jg jp wall rb ltype flags sha; do
    printf '%-32s %9s %11s %11s %8s %22s %30s %28s %10s\n' \
        "$name" "$tps" "$jg" "$jp" "$wall" "$rb" "$ltype" "$flags" "$sha"
done < "$TMPD/results"

echo
echo " Columns:"
echo "   dec_tps              = decode tokens/sec (B=1: per-stream; B=8 lanes: aggregate tps)"
echo "   J/tok_GPU            = avg GPU power W / dec_tps  (N/A if macmon missing)"
echo "   J/tok_pkg            = avg package power W / dec_tps  (CPU+GPU+ANE; N/A if macmon missing)"
echo "   wall_s               = total wall seconds for the run"
echo "   readback_bytes/tok   = GPU→CPU bytes per token (serve lanes: from /metrics; generate/llama: N/A)"
echo "                          B=1 greedy: 4 bytes/tok  |  B=1 full-logits: ~128K bytes/tok (32K vocab×f32)"
echo "   lane_type            = routing path taken (from /metrics greedy_steps vs logits_steps)"
echo "   feature_flags        = active HAWKING_* lever bundle and kernel profile"
echo "   git_sha              = HEAD commit (first 7 chars)"
echo
echo " --kernel-profile (hardware autotune JSON) is distinct from --profile fast (lever bundle)."
echo " The feature_flags column shows both explicitly."
echo
echo " Model: $(basename "${GGUF}")   Prompt: $(printf '%.70s' "$PROMPT")..."
echo " Anchors: dismantle clean single-stream ~31 dec_tps | llama.cpp ~49 dec_tps"
echo "============================================================="
echo " done."
